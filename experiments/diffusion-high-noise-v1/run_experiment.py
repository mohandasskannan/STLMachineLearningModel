"""Train fresh, bounded candidates and compare them with the protected baseline."""

import argparse
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from experiment import Config, evaluate, load_candidate, train_candidate
from scipy.ndimage import generate_binary_structure, label

from stlmodel.common import file_hash, load_checkpoint, write_json
from stlmodel.data import VoxelDataset
from stlmodel.mesh import export_stl
from stlmodel.models import Denoiser, Diffusion
from stlmodel.text import TextEncoder
from stlmodel.training import load_vae

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("E:/STLModelData/diffusion-high-noise-v1")
SOURCE = Path("E:/STLModelData/text2shape-v1")
CACHE = SOURCE / "embeddings.pt"
MANIFEST = SOURCE / "prepared/manifest.json"
VAE = ROOT / "runs/text2shape-pilot-vae-v1/best.pt"
BASELINE = ROOT / "runs/text2shape-pilot-diffusion-v1/best.pt"
TRACKED = Path(__file__).resolve().parent
RUNS = {
    name: ROOT / ("runs/text2shape-diffusion-high-noise-v1-" + name)
    for name in ("velocity", "conditioned")
}
PROMPTS = [
    "A chair with a tall back and armrests",
    "a chair with armrests",
    "a chair without armrests",
    "a table with four legs",
]


def source_hashes():
    return {str(path): file_hash(path) for path in (CACHE, MANIFEST, VAE, BASELINE)}


def verify_sources(cache, hashes):
    baseline = load_checkpoint(BASELINE, "diffusion")
    if (
        cache["vae_sha256"] != hashes[str(VAE)]
        or baseline["vae_sha256"] != hashes[str(VAE)]
        or baseline["source_sha256"] != hashes[str(CACHE)]
        or cache["manifest_sha256"] != hashes[str(MANIFEST)]
    ):
        raise ValueError("The baseline/cache/decoder fingerprints do not match")


@torch.inference_mode()
def legacy_sample(model, schedule, text, seed, start=199):
    z = torch.randn(
        (len(text), model.latent_dim),
        device=text.device,
        generator=torch.Generator(device=text.device).manual_seed(seed),
    )
    times = torch.linspace(start, 0, 50).round().long().tolist()
    for i, time_index in enumerate(times):
        t = torch.full((len(z),), time_index, device=text.device, dtype=torch.long)
        noise = model(z, t, text)
        alpha = schedule.alpha_bar[time_index]
        previous = schedule.alpha_bar[times[i + 1]] if i + 1 < len(times) else z.new_tensor(1.0)
        clean = (z - (1 - alpha).sqrt() * noise) / alpha.sqrt()
        z = previous.sqrt() * clean + (1 - previous).sqrt() * noise
    return z


def unique_indices(cache, split, metadata):
    selected = {}
    for index, identifier in enumerate(cache[split]["ids"]):
        category = metadata[identifier]["category"]
        pattern = r"\b" + category + r"\b"
        if identifier not in selected or (
            not re.search(pattern, cache[split]["captions"][selected[identifier]], re.IGNORECASE)
            and re.search(pattern, cache[split]["captions"][index], re.IGNORECASE)
        ):
            selected[identifier] = index
    return list(selected.values())


@torch.inference_mode()
def review(
    cache,
    source_hash,
    device,
    *,
    candidate_runs=None,
    candidate_sources=None,
    artifact_dir=ARTIFACTS,
    include_legacy=True,
    guidance_scales=(1.0, 2.0, 3.0),
    evaluate_test=True,
):
    vae, _ = load_vae(VAE, device)
    baseline_state = load_checkpoint(BASELINE, "diffusion")
    baseline = Denoiser(**baseline_state["architecture"]).to(device).eval()
    baseline.load_state_dict(baseline_state["model"])
    legacy_schedule = Diffusion(baseline_state["diffusion_steps"], device)
    metadata = {row["id"]: row for row in json.loads(MANIFEST.read_text())["records"]}
    indices = {split: unique_indices(cache, split, metadata) for split in ("train", "val")}
    train = cache["train"]["latents"][indices["train"]].to(device)
    val = cache["val"]["latents"][indices["val"]].to(device)
    val_text = cache["val"]["text"][indices["val"]].to(device)
    ids = {split: [cache[split]["ids"][i] for i in indices[split]] for split in indices}
    categories = {
        split: torch.tensor(
            [int(metadata[i]["category"] == "table") for i in ids[split]], device=device
        )
        for split in indices
    }
    chairs, tables = (torch.where(categories["val"] == c)[0] for c in (0, 1))
    selected = torch.cat((chairs[:20], tables[:20]))
    opposite = torch.empty_like(categories["val"])
    opposite[chairs], opposite[tables] = tables, chairs
    text = val_text[selected]
    expected = categories["val"][selected].repeat(2)
    centroid = train.mean(0)
    covariance = (train - centroid).T @ (train - centroid) / len(train)
    eigen, basis = torch.linalg.eigh(covariance)
    eigen, basis = eigen.flip(0), basis.flip(1)
    mean, std = cache["latent_mean"].to(device), cache["latent_std"].to(device)
    datasets = {split: VoxelDataset(MANIFEST, split) for split in indices}
    voxels = {}
    for split, dataset in datasets.items():
        lookup = {row["id"]: i for i, row in enumerate(dataset.records)}
        voxels[split] = torch.stack([dataset[lookup[i]][0][0] for i in ids[split]]).to(device)
    training_flat = voxels["train"].flatten(1)
    training_sums = training_flat.sum(1)

    def decode(z):
        return torch.cat([vae.decode(chunk * std + mean).sigmoid()[:, 0] for chunk in z.split(16)])

    def category(volume):
        flat = (volume > 0.5).flatten(1).float()
        intersection = flat @ training_flat.T
        overlaps = intersection / (
            flat.sum(1, keepdim=True) + training_sums[None] - intersection
        ).clamp_min(1)
        return categories["train"][overlaps.argmax(1)], overlaps.max(1).values

    def overlap(a, b):
        a, b = a > 0.5, b > 0.5
        return float(
            (
                (a.flatten(1) & b.flatten(1)).sum(1)
                / (a.flatten(1) | b.flatten(1)).sum(1).clamp_min(1)
            ).mean()
        )

    def quality(z, volume):
        pcs = (z - centroid) @ basis
        inferred, nearest_iou = category(volume)
        nearest = torch.cdist(z, train)
        counts, largest = [], []
        for v in volume.cpu().numpy():
            regions, count = label(v > 0.5, generate_binary_structure(3, 3))
            sizes = np.bincount(regions.ravel())[1:]
            counts.append(count)
            largest.append(float(sizes.max() / sizes.sum()) if len(sizes) else 0.0)
        return {
            "latent_rms": float(z.square().mean().sqrt()),
            "nearest_train_latent_rmse": float(nearest.min(1).values.mean() / 16),
            "bottom64_energy_fraction": float(pcs[:, 192:].square().sum() / pcs.square().sum()),
            "regularized_whitened_rms": float((pcs.square() / (eigen + 0.001)).mean().sqrt()),
            "category_agreement": float((inferred == expected).float().mean()),
            "chair_category_agreement": float((inferred[expected == 0] == 0).float().mean()),
            "table_category_agreement": float((inferred[expected == 1] == 1).float().mean()),
            "nearest_training_voxel_iou_mean": float(nearest_iou.mean()),
            "nearest_training_voxel_iou_max": float(nearest_iou.max()),
            "voxel_components_26_mean": float(np.mean(counts)),
            "largest_voxel_component_fraction_mean": float(np.mean(largest)),
            "empty_count": int(sum(c == 0 for c in counts)),
        }

    reconstructed = decode(val)
    inferred, _ = category(reconstructed)
    result = {
        "seeds": [42, 43],
        "validation_caption_objects": 40,
        "selected_ids": [ids["val"][i] for i in selected.cpu().tolist()],
        "selected_captions": [
            cache["val"]["captions"][indices["val"][i]] for i in selected.cpu().tolist()
        ],
        "category_diagnostic_reconstruction_calibration": float(
            (inferred == categories["val"]).float().mean()
        ),
        "decoder_validation_iou": overlap(reconstructed, voxels["val"]),
        "generation": {},
        "high_noise": {},
        "candidate_evaluations": {},
    }
    candidates = {}
    for name, path in (RUNS if candidate_runs is None else candidate_runs).items():
        paired_cache, paired_hash = (candidate_sources or {}).get(name, (cache, source_hash))
        model, schedule, state = load_candidate(path / "best.pt", paired_cache, paired_hash, device)
        candidates[name] = (model, schedule)
        result["candidate_evaluations"][name] = {
            "selected_epoch": state["epoch"],
            "selected_total_epoch": state.get("epoch_offset", 0) + state["epoch"],
            "selected_step": state["step"],
            "val": evaluate(model, cache["val"], schedule, device),
        }
        if evaluate_test:
            result["candidate_evaluations"][name]["test"] = evaluate(
                model, cache["test"], schedule, device
            )
    samplers = {}
    if include_legacy:
        samplers.update(
            {
                "legacy": lambda txt, seed: legacy_sample(baseline, legacy_schedule, txt, seed),
                "legacy_start190": lambda txt, seed: legacy_sample(
                    baseline, legacy_schedule, txt, seed, 190
                ),
            }
        )
    for name, (model, schedule) in candidates.items():
        scales = (1.0,) if name == "velocity" else guidance_scales
        for guidance in scales:
            key = name if name == "velocity" else f"{name}_g{int(guidance)}"
            samplers[key] = lambda txt, seed, m=model, s=schedule, g=guidance: s.sample(
                m, txt, seed, guidance=g
            )
    for name, sampler in samplers.items():
        print({"review": name}, flush=True)
        z = torch.cat([sampler(text, seed) for seed in (42, 43)])
        volume = decode(z)
        wrong_z = torch.cat([sampler(val_text[opposite[selected]], seed) for seed in (42, 43)])
        wrong_volume = decode(wrong_z)
        measurements = quality(z, volume)
        measurements.update(
            {
                "opposite_caption_latent_rmse": float((z - wrong_z).square().mean().sqrt()),
                "opposite_caption_voxel_iou": overlap(volume, wrong_volume),
                "seed_change_latent_rmse": float((z[:40] - z[40:]).square().mean().sqrt()),
                "seed_change_voxel_iou": overlap(volume[:40], volume[40:]),
            }
        )
        result["generation"][name] = measurements
        np.savez_compressed(
            artifact_dir / (name + "-validation.npz"),
            latents=z.cpu().numpy(),
            occupancy=volume.cpu().numpy(),
        )
        print({"review": name, **measurements}, flush=True)
    noise = torch.randn(
        val.shape, device=device, generator=torch.Generator(device=device).manual_seed(1234)
    )
    legacy_candidates = [("legacy", baseline, legacy_schedule)] if include_legacy else []
    for name, model, schedule in legacy_candidates + [
        (name, *pair) for name, pair in candidates.items()
    ]:
        records = []
        for time_index in (100, 190, 199):
            t = torch.full((len(val),), time_index, device=device, dtype=torch.long)
            noisy = schedule.noisy(val, t, noise)
            record = {"timestep": time_index}
            for condition, captions in [("correct", val_text), ("opposite", val_text[opposite])]:
                prediction = model(noisy, t, captions)
                if name == "legacy":
                    a = schedule.alpha_bar[t, None]
                    clean = (noisy - (1 - a).sqrt() * prediction) / a.sqrt()
                else:
                    clean, _ = schedule.parts(noisy, prediction, t)
                record[condition + "_clean_latent_rmse"] = float(
                    (clean - val).square().mean().sqrt()
                )
            records.append(record)
        result["high_noise"][name] = records
    encoder = TextEncoder(cache["text_model"], cache["text_revision"])
    prompt_text = torch.from_numpy(encoder.encode(PROMPTS)).to(device)
    preview_paths, preview_labels = [], []
    result["fixed_prompt_exports"] = {}
    for name, sampler in samplers.items():
        rows = []
        for prompt_index, prompt in enumerate(PROMPTS):
            for seed in (42, 43, 44):
                volume = decode(sampler(prompt_text[prompt_index : prompt_index + 1], seed))[0]
                path = artifact_dir / name / f"prompt-{prompt_index}-seed-{seed}.stl"
                row = {"prompt": prompt, "seed": seed}
                try:
                    row["mesh"] = export_stl(volume.cpu().numpy(), path)
                    if prompt_index in (0, 3):
                        preview_paths.append(path)
                        preview_labels.append(
                            f"{name} / {'chair' if prompt_index == 0 else 'table'} / {seed}"
                        )
                except ValueError as exc:
                    row["error"] = str(exc)
                rows.append(row)
        result["fixed_prompt_exports"][name] = rows
    spec = importlib.util.spec_from_file_location(
        "experiment_renderer", ROOT / "experiments/procedural-v1/render_review.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    result["reopened_mesh_checks"] = renderer.render(
        preview_paths, preview_labels, artifact_dir / "comparison.png", columns=6, front=True
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "train", "review"), default="all", nargs="?")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=600, help="Per candidate session cap")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    protected = source_hashes()
    cache = load_checkpoint(CACHE, "embeddings")
    verify_sources(cache, protected)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS / "protected-sources.json", protected)
    if args.stage in ("all", "train"):
        for name, path in RUNS.items():
            config = Config(
                variant=name,
                epochs=args.epochs,
                max_seconds=args.max_seconds,
                text_dropout=0.15 if name == "conditioned" else 0.0,
            )
            summary = train_candidate(
                cache, protected[str(CACHE)], path, config, args.device, resume=args.resume
            )
            print(summary, flush=True)
            if source_hashes() != protected:
                raise ValueError("Protected artifacts changed during candidate training")
            if summary["stop_reason"] == "interrupted":
                return
    if args.stage in ("all", "review"):
        result = review(cache, protected[str(CACHE)], args.device)
        result["protected_source_hashes"] = protected
        result["protected_artifacts_unchanged"] = source_hashes() == protected
        if not result["protected_artifacts_unchanged"]:
            raise ValueError("Protected artifacts changed during review")
        result["production_model_promoted"] = False
        write_json(ARTIFACTS / "results.json", result)
        write_json(TRACKED / "results.json", result)
        shutil.copy2(ARTIFACTS / "comparison.png", TRACKED / "comparison.png")
        for name, path in RUNS.items():
            destination = TRACKED / "training" / name
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("config.json", "summary.json", "metrics.jsonl"):
                shutil.copy2(path / filename, destination / filename)
        print({"results": str(TRACKED / "results.json")}, flush=True)


if __name__ == "__main__":
    main()
