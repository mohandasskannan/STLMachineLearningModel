"""Resumable, time-bounded training with object-disjoint validation."""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    FORMAT_VERSION,
    device_for,
    file_hash,
    load_checkpoint,
    save_checkpoint,
    seed_all,
    write_json,
)
from .data import VoxelDataset
from .models import Denoiser, Diffusion, ShapeVAE, vae_loss, voxel_iou


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 0.0002
    epochs: int = 100
    max_hours: float = 3.0
    max_steps: int = 0  # Total optimizer steps, including previous sessions.
    checkpoint_every: int = 100
    latent_dim: int = 256
    diffusion_steps: int = 200
    kl_weight: float = 0.001
    kl_warmup_steps: int = 1000
    limit: int = 0

    def validate(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Training settings must be finite")
        if (
            self.batch_size < 1
            or self.epochs < 1
            or self.max_hours <= 0
            or self.learning_rate <= 0
            or self.checkpoint_every < 1
            or self.latent_dim < 1
            or self.diffusion_steps < 2
            or self.kl_weight < 0
            or self.kl_warmup_steps < 1
            or self.max_steps < 0
            or self.limit < 0
        ):
            raise ValueError("Invalid training configuration; counts/rates must be positive")


def load_vae(path: Path, device="cpu"):
    state = load_checkpoint(path, "vae")
    model = ShapeVAE(state["architecture"]["latent_dim"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.inference_mode()
def evaluate_vae(model, dataset, device, batch_size=8):
    model.eval()
    totals = {"iou": 0.0, "bce": 0.0}
    loader = DataLoader(dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(0))
    for voxels, _ in loader:
        voxels = voxels.to(device)
        logits, _, _ = model(voxels)
        totals["iou"] += float(voxel_iou(logits, voxels)) * len(voxels)
        totals["bce"] += float(F.binary_cross_entropy_with_logits(logits, voxels)) * len(voxels)
    return {key: value / len(dataset) for key, value in totals.items()}


@torch.inference_mode()
def evaluate_diffusion(model, dataset, schedule, device, batch_size=64):
    """Matched noise/times for true versus globally shuffled caption embeddings."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(1234)
    z_all, text_all = dataset.tensors
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(1234))
    if len(dataset) > 1 and torch.equal(permutation, torch.arange(len(dataset))):
        permutation = permutation.roll(1)
    totals = {"noise_mse": 0.0, "shuffled_text_mse": 0.0}
    for start in range(0, len(dataset), batch_size):
        z = z_all[start : start + batch_size].to(device)
        text = text_all[start : start + batch_size].to(device)
        shuffled = text_all[permutation[start : start + batch_size]].to(device)
        t = torch.randint(schedule.steps, (len(z),), generator=generator, device=device)
        noise = torch.randn(z.shape, generator=generator, device=device)
        noisy = schedule.noisy(z, t, noise)
        totals["noise_mse"] += float(F.mse_loss(model(noisy, t, text), noise)) * len(z)
        totals["shuffled_text_mse"] += float(F.mse_loss(model(noisy, t, shuffled), noise)) * len(z)
    result = {key: value / len(dataset) for key, value in totals.items()}
    result["conditioning_gap"] = result["shuffled_text_mse"] - result["noise_mse"]
    return result


def train(
    kind: str,
    source: Path,
    out: Path,
    config: TrainConfig,
    device_name="auto",
    resume: Path | None = None,
) -> dict:
    config.validate()
    device = device_for(device_name)
    seed_all(config.seed)
    out = Path(out)
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError(f"Run directory is not empty: {out}; use --resume or a new --out")
    source_hash = file_hash(source)
    extra = {}
    if kind == "vae":
        training = VoxelDataset(source, "train", config.limit)
        validation = VoxelDataset(source, "val")
        model = ShapeVAE(config.latent_dim).to(device)
        architecture = {"latent_dim": config.latent_dim}
    elif kind == "diffusion":
        cache = load_checkpoint(source, "embeddings")
        training = TensorDataset(cache["train"]["latents"], cache["train"]["text"])
        validation = TensorDataset(cache["val"]["latents"], cache["val"]["text"])
        if len(training) == 0 or len(validation) == 0:
            raise ValueError("Embedding cache must contain train and validation examples")
        if training.tensors[0].shape[1] != config.latent_dim:
            raise ValueError("latent_dim does not match cached autoencoder")
        model = Denoiser(config.latent_dim).to(device)
        architecture = {"latent_dim": config.latent_dim, "text_dim": 384, "width": 512}
        schedule = Diffusion(config.diffusion_steps, device)
        extra = {
            key: cache[key]
            for key in (
                "vae_sha256",
                "text_model",
                "text_revision",
                "latent_mean",
                "latent_std",
                "manifest_sha256",
            )
        }
        extra["diffusion_steps"] = config.diffusion_steps
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    step, epoch, batch_cursor, elapsed_before, best = 0, 0, 0, 0.0, float("inf")
    if resume:
        state = load_checkpoint(resume, kind)
        if state["source_sha256"] != source_hash or state["architecture"] != architecture:
            raise ValueError("Resume source or architecture differs from the saved run")
        for key in (
            "seed",
            "batch_size",
            "learning_rate",
            "latent_dim",
            "diffusion_steps",
            "kl_weight",
            "kl_warmup_steps",
            "limit",
        ):
            if state["config"][key] != getattr(config, key):
                raise ValueError(f"Cannot change {key} when resuming")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, batch_cursor = state["step"], state["epoch"], state["batch_cursor"]
        elapsed_before, best = state["elapsed_seconds"], state["best_score"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json",
        {
            **asdict(config),
            "kind": kind,
            "device": str(device),
            "source": str(Path(source).resolve()),
            "source_sha256": source_hash,
        },
    )
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def snapshot():
        return {
            "format_version": FORMAT_VERSION,
            "kind": kind,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": architecture,
            "config": asdict(config),
            "source_sha256": source_hash,
            "step": step,
            "epoch": epoch,
            "batch_cursor": batch_cursor,
            "best_score": best,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            **extra,
        }

    def validate_and_save():
        nonlocal best
        metrics = (
            evaluate_vae(model, validation, device, config.batch_size)
            if kind == "vae"
            else evaluate_diffusion(model, validation, schedule, device)
        )
        score = -metrics["iou"] if kind == "vae" else metrics["noise_mse"]
        improved = score < best
        if improved:
            best = score
        event = {
            "step": step,
            "epoch": epoch,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "validation": metrics,
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        save_checkpoint(out / "last.pt", snapshot())
        if improved:
            save_checkpoint(out / "best.pt", snapshot())
        print(json.dumps(event), flush=True)
        return metrics

    def exhausted():
        return (
            config.max_steps > 0 and step >= config.max_steps
        ) or time.monotonic() - started >= config.max_hours * 3600

    interrupted = False
    try:
        while epoch < config.epochs and not exhausted():
            loader = DataLoader(
                training,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(config.seed + epoch),
            )
            for batch_index, batch in enumerate(loader):
                if batch_index < batch_cursor:
                    continue
                if exhausted():
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if kind == "vae":
                    voxels = batch[0].to(device)
                    logits, mu, logvar = model(voxels)
                    beta = config.kl_weight * min(1, step / config.kl_warmup_steps)
                    loss, _ = vae_loss(logits, voxels, mu, logvar, beta)
                else:
                    z, text = [part.to(device) for part in batch]
                    t = torch.randint(schedule.steps, (len(z),), device=device)
                    noise = torch.randn_like(z)
                    loss = F.mse_loss(model(schedule.noisy(z, t, noise), t, text), noise)
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Non-finite training loss; last saved checkpoint remains available"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                batch_cursor = batch_index + 1
                if step % config.checkpoint_every == 0:
                    save_checkpoint(out / "last.pt", snapshot())
                    print(f"step={step} train_loss={float(loss.detach()):.5f}", flush=True)
            if batch_cursor >= len(loader):
                epoch += 1
                batch_cursor = 0
                validate_and_save()
            elif exhausted():
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving current training state.", flush=True)
    metrics = validate_and_save()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    result = {
        "step": step,
        "epoch": epoch,
        "interrupted": interrupted,
        "session_seconds": time.monotonic() - started,
        "peak_allocated_gib": peak,
        "validation": metrics,
        "checkpoint": str((out / "last.pt").resolve()),
    }
    write_json(out / "summary.json", result)
    return result


def benchmark(device_name="auto", batch_size=8, steps=10) -> dict:
    if batch_size < 1 or steps < 1:
        raise ValueError("Batch size and benchmark steps must be positive")
    device = device_for(device_name)
    seed_all(42)
    model = ShapeVAE().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    target = torch.zeros(batch_size, 1, 32, 32, 32, device=device)
    target[:, :, 8:24, 8:24, 8:24] = 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def iteration():
        optimizer.zero_grad(set_to_none=True)
        logits, mu, logvar = model(target)
        loss, _ = vae_loss(logits, target, mu, logvar, 0.001)
        loss.backward()
        optimizer.step()

    for _ in range(2):
        iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "batch_size": batch_size,
        "steps": steps,
        "seconds_per_step": duration / steps,
        "examples_per_second": batch_size * steps / duration,
        "peak_allocated_gib": peak,
        "under_5_gib": peak < 5 if peak is not None else None,
        "note": "Synthetic VAE compute benchmark; excludes data loading and validation time.",
    }
