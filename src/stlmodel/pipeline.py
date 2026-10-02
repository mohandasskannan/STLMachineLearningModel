"""Cache training representations and run the same inference path from CLI and UI."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (
    FORMAT_VERSION,
    device_for,
    file_hash,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from .data import SPLITS, VoxelDataset
from .mesh import export_stl
from .models import Denoiser, Diffusion
from .text import TextEncoder
from .training import load_vae


@torch.inference_mode()
def cache_embeddings(
    manifest: Path, vae_path: Path, output: Path, device_name="auto", encoder=None
) -> dict:
    if Path(output).exists():
        raise ValueError(f"Cache already exists: {output}; choose a new output path")
    device = device_for(device_name)
    vae, state = load_vae(vae_path, device)
    if state["source_sha256"] != file_hash(manifest):
        raise ValueError("Autoencoder was trained on a different manifest")
    encoder = encoder or TextEncoder()
    shape_splits = {}
    for split in SPLITS:
        dataset = VoxelDataset(manifest, split)
        latents = []
        for voxels, _ in DataLoader(dataset, batch_size=8):
            mu, _ = vae.encode(voxels.to(device))
            latents.append(mu.cpu())
        shape_splits[split] = (dataset, torch.cat(latents))
    train_latents = shape_splits["train"][1]
    mean = train_latents.mean(0)
    std = train_latents.std(0, unbiased=False).clamp_min(0.05)
    cache = {
        "format_version": FORMAT_VERSION,
        "kind": "embeddings",
        "manifest_sha256": file_hash(manifest),
        "vae_sha256": file_hash(vae_path),
        "text_model": encoder.model_name,
        "text_revision": getattr(encoder, "revision", None),
        "latent_mean": mean,
        "latent_std": std,
    }
    for split, (dataset, latents) in shape_splits.items():
        captions, indices, ids = [], [], []
        for index, row in enumerate(dataset.records):
            for caption in row["captions"]:
                captions.append(caption)
                indices.append(index)
                ids.append(row["id"])
        text = torch.from_numpy(encoder.encode(captions))
        if text.shape != (len(captions), 384) or not torch.isfinite(text).all():
            raise ValueError("Text encoder returned invalid embeddings")
        cache[split] = {
            "latents": ((latents - mean) / std)[indices],
            "text": text,
            "ids": ids,
            "captions": captions,
        }
    save_checkpoint(Path(output), cache)
    return {
        "cache": str(Path(output).resolve()),
        "examples": {split: len(cache[split]["ids"]) for split in SPLITS},
    }


class Generator:
    def __init__(self, vae_path: Path, diffusion_path: Path, device_name="auto", encoder=None):
        self.device = device_for(device_name)
        self.vae, vae_state = load_vae(vae_path, self.device)
        state = load_checkpoint(diffusion_path, "diffusion")
        if state["vae_sha256"] != file_hash(vae_path):
            raise ValueError(
                "Diffusion checkpoint requires the exact autoencoder used to cache embeddings"
            )
        if state["architecture"]["latent_dim"] != vae_state["architecture"]["latent_dim"]:
            raise ValueError("Model latent dimensions do not match")
        self.denoiser = Denoiser(**state["architecture"]).to(self.device).eval()
        self.denoiser.load_state_dict(state["model"])
        self.schedule = Diffusion(state["diffusion_steps"], self.device)
        self.mean = state["latent_mean"].to(self.device)
        self.std = state["latent_std"].to(self.device)
        self.encoder = encoder or TextEncoder(state["text_model"], state["text_revision"])
        if self.encoder.model_name != state["text_model"]:
            raise ValueError("Text encoder does not match the trained model")
        self.metadata = {
            "vae_sha256": file_hash(vae_path),
            "diffusion_sha256": file_hash(diffusion_path),
            "text_model": state["text_model"],
            "text_revision": state["text_revision"],
        }

    @torch.inference_mode()
    def occupancy(self, prompt: str, seed: int = 42, sampling_steps: int = 50) -> np.ndarray:
        if not prompt.strip():
            raise ValueError("Enter a text prompt")
        if not 0 <= seed < 2**32:
            raise ValueError("Seed must be between 0 and 4294967295")
        text = torch.from_numpy(self.encoder.encode([prompt.strip()])).to(self.device)
        z = self.schedule.sample(self.denoiser, text, seed, sampling_steps)
        return self.vae.decode(z * self.std + self.mean).sigmoid()[0, 0].cpu().numpy()

    def generate(
        self, prompt: str, output: Path, seed=42, size_mm=100, sampling_steps=50, threshold=0.5
    ) -> dict:
        if not np.isfinite(size_mm) or size_mm <= 0:
            raise ValueError("Longest dimension must be positive and finite")
        volume = self.occupancy(prompt, seed, sampling_steps)
        report = export_stl(volume, output, size_mm, threshold)
        write_json(
            Path(output).with_suffix(".generation.json"),
            {
                **self.metadata,
                "prompt": prompt.strip(),
                "seed": seed,
                "size_mm": size_mm,
                "sampling_steps": sampling_steps,
                "threshold": threshold,
                "mesh": report,
            },
        )
        return report


@torch.inference_mode()
def reconstruct(
    manifest: Path, vae_path: Path, out: Path, split="val", count=4, device_name="auto"
) -> list[dict]:
    device = device_for(device_name)
    vae, _ = load_vae(vae_path, device)
    dataset = VoxelDataset(manifest, split, count)
    reports = []
    for i in range(len(dataset)):
        voxels, object_id = dataset[i]
        logits, _, _ = vae(voxels[None].to(device))
        prefix = Path(out) / f"{i:04d}"
        export_stl(voxels[0].numpy(), prefix.with_name(prefix.name + "-target.stl"))
        try:
            mesh = export_stl(
                logits.sigmoid()[0, 0].cpu().numpy(),
                prefix.with_name(prefix.name + "-reconstruction.stl"),
            )
            reports.append({"id": object_id, "mesh": mesh})
        except ValueError as exc:
            reports.append({"id": object_id, "error": str(exc)})
    write_json(Path(out) / "reconstructions.json", reports)
    return reports


@torch.inference_mode()
def evaluate_generation(
    generator: Generator,
    manifest: Path,
    prompts: list[str],
    out: Path,
    seeds=(42, 43, 44),
    sampling_steps=50,
) -> dict:
    """Save prompt/seed comparisons, diversity and nearest-training voxel overlap.

    These are diagnostics, not evidence by themselves of generalization or semantics.
    """
    if not prompts or not seeds:
        raise ValueError("Provide at least one prompt and seed")
    dataset = VoxelDataset(manifest, "train")
    training = torch.stack([dataset[i][0][0].flatten().bool() for i in range(len(dataset))])
    rows, samples = [], []
    for p, prompt in enumerate(prompts):
        for seed in seeds:
            volume = generator.occupancy(prompt, seed, sampling_steps)
            occupied = torch.from_numpy(volume > 0.5).flatten()
            best_iou, nearest = -1.0, None
            for start in range(0, len(training), 128):
                chunk = training[start : start + 128]
                intersection = (chunk & occupied).sum(1).float()
                union = (chunk | occupied).sum(1).clamp_min(1)
                values = intersection / union
                value, index = values.max(0)
                if float(value) > best_iou:
                    best_iou = float(value)
                    nearest = dataset.records[start + int(index)]["id"]
            row = {
                "prompt": prompt,
                "seed": seed,
                "nearest_train_id": nearest,
                "nearest_train_iou": best_iou,
            }
            try:
                row["mesh"] = export_stl(volume, Path(out) / f"prompt-{p}-seed-{seed}.stl")
            except ValueError as exc:
                row["error"] = str(exc)
            rows.append(row)
            samples.append(occupied)
    comparisons = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if rows[i]["prompt"] == rows[j]["prompt"] or rows[i]["seed"] == rows[j]["seed"]:
                intersection = (samples[i] & samples[j]).sum().item()
                union = (samples[i] | samples[j]).sum().item()
                comparisons.append({"a": i, "b": j, "voxel_iou": intersection / max(1, union)})
    report = {
        "samples": rows,
        "comparisons": comparisons,
        "note": "High overlap can indicate collapse or memorization; low overlap does not prove quality.",
    }
    write_json(Path(out) / "evaluation.json", report)
    return report
