"""Isolated velocity diffusion experiment; incompatible with legacy epsilon weights.

References: https://arxiv.org/abs/2305.08891 and https://arxiv.org/abs/2207.12598.
The production generator and all existing checkpoints remain unchanged.
"""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from stlmodel.common import (
    FORMAT_VERSION,
    file_hash,
    load_checkpoint,
    save_checkpoint,
    seed_all,
    write_json,
)
from stlmodel.models import Denoiser, Diffusion

KIND = "diffusion_velocity_experiment"
SCHEDULE = "cosine_zero_terminal_v1"


class VelocitySchedule(Diffusion):
    def __init__(self, steps=200, device="cpu"):
        super().__init__(steps, device)
        roots = self.alpha_bar.sqrt()
        first, last = roots[0].clone(), roots[-1].clone()
        self.alpha_bar = ((roots - last) * first / (first - last)).square()

    def target(self, clean, times, noise):
        alpha = self.alpha_bar[times, None]
        return alpha.sqrt() * noise - (1 - alpha).sqrt() * clean

    def parts(self, noisy, velocity, times):
        alpha = self.alpha_bar[times, None]
        clean = alpha.sqrt() * noisy - (1 - alpha).sqrt() * velocity
        noise = (1 - alpha).sqrt() * noisy + alpha.sqrt() * velocity
        return clean, noise

    @torch.inference_mode()
    def sample(self, model, text, seed=42, sampling_steps=50, guidance=1.0):
        if not 2 <= sampling_steps <= self.steps:
            raise ValueError("Invalid sampling step count")
        if not math.isfinite(guidance) or not 0 <= guidance <= 5:
            raise ValueError("Guidance must be between zero and five")
        z = torch.randn(
            (len(text), model.latent_dim),
            generator=torch.Generator(device=text.device).manual_seed(seed),
            device=text.device,
        )
        times = torch.linspace(self.steps - 1, 0, sampling_steps).round().long().tolist()
        for index, time_index in enumerate(times):
            t = torch.full((len(z),), time_index, device=text.device, dtype=torch.long)
            if guidance == 1:
                velocity = model(z, t, text)
            else:
                unconditional, conditional = model(
                    torch.cat((z, z)),
                    torch.cat((t, t)),
                    torch.cat((torch.zeros_like(text), text)),
                ).chunk(2)
                velocity = unconditional + guidance * (conditional - unconditional)
            clean, noise = self.parts(z, velocity, t)
            previous = (
                self.alpha_bar[times[index + 1]] if index + 1 < len(times) else z.new_tensor(1.0)
            )
            z = previous.sqrt() * clean + (1 - previous).sqrt() * noise
            if not torch.isfinite(z).all():
                raise ValueError("Non-finite sampling trajectory")
        return z


class ConditionedBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)
        self.transform = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden, condition):
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        return hidden + self.transform(self.norm(hidden) * (1 + scale) + shift)


class AdaptiveDenoiser(nn.Module):
    def __init__(self, latent_dim=256, text_dim=384, width=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.input = nn.Linear(latent_dim, width)
        self.condition = nn.Sequential(
            nn.Linear(text_dim + 64, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([ConditionedBlock(width) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, latent_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, z, times, text):
        frequencies = torch.exp(torch.arange(32, device=z.device) * (-math.log(10000) / 31))
        angles = times.float()[:, None] * frequencies[None]
        time_features = torch.cat((angles.sin(), angles.cos()), dim=1)
        condition = self.condition(torch.cat((text, time_features), dim=1))
        hidden = self.input(z)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


@dataclass
class Config:
    variant: str = "velocity"
    seed: int = 42
    batch_size: int = 64
    width: int = 512
    learning_rate: float = 0.0002
    epochs: int = 200
    max_seconds: float = 600
    patience: int = 20
    max_steps: int = 0
    text_dropout: float = 0.0
    diffusion_steps: int = 200

    def validate(self):
        if self.variant not in ("velocity", "conditioned"):
            raise ValueError("Unknown experimental model variant")
        if not all(math.isfinite(v) for k, v in asdict(self).items() if k != "variant"):
            raise ValueError("Training settings must be finite")
        if (
            min(self.batch_size, self.width, self.epochs, self.patience) < 1
            or self.learning_rate <= 0
            or self.max_seconds <= 0
            or self.diffusion_steps < 2
            or self.max_steps < 0
            or not 0 <= self.text_dropout < 1
        ):
            raise ValueError("Invalid experimental training settings")
        if self.variant == "velocity" and self.text_dropout:
            raise ValueError("The velocity-only ablation does not train null conditioning")


def build_model(config, latent_dim, device):
    cls = Denoiser if config.variant == "velocity" else AdaptiveDenoiser
    return cls(latent_dim=latent_dim, width=config.width).to(device)


@torch.inference_mode()
def evaluate(model, cache_split, schedule, device, batch_size=64):
    model.eval()
    random = torch.Generator(device=device).manual_seed(1234)
    permutation = torch.randperm(
        len(cache_split["latents"]), generator=torch.Generator().manual_seed(1234)
    )
    totals = {"velocity_mse": 0.0, "shuffled_velocity_mse": 0.0, "clean_latent_mse": 0.0}
    for start in range(0, len(permutation), batch_size):
        z = cache_split["latents"][start : start + batch_size].to(device)
        text = cache_split["text"][start : start + batch_size].to(device)
        shuffled = cache_split["text"][permutation[start : start + batch_size]].to(device)
        times = torch.randint(schedule.steps, (len(z),), generator=random, device=device)
        noise = torch.randn(z.shape, generator=random, device=device)
        noisy = schedule.noisy(z, times, noise)
        target = schedule.target(z, times, noise)
        predicted = model(noisy, times, text)
        clean, _ = schedule.parts(noisy, predicted, times)
        totals["velocity_mse"] += float(F.mse_loss(predicted, target)) * len(z)
        totals["shuffled_velocity_mse"] += float(
            F.mse_loss(model(noisy, times, shuffled), target)
        ) * len(z)
        totals["clean_latent_mse"] += float(F.mse_loss(clean, z)) * len(z)
    result = {key: value / len(permutation) for key, value in totals.items()}
    result["conditioning_gap"] = result["shuffled_velocity_mse"] - result["velocity_mse"]
    return result


def validate_pairing(state, cache, source_hash):
    """Reject weights paired with different data, normalization, or prediction math."""
    if (
        state["source_sha256"] != source_hash
        or state["schedule"] != SCHEDULE
        or state["prediction_type"] != "v_prediction"
        or state["latent_dim"] != cache["latent_mean"].numel()
        or any(
            state[key] != cache[key]
            for key in ("vae_sha256", "manifest_sha256", "text_model", "text_revision")
        )
        or any(
            not torch.equal(state[key].cpu(), cache[key].cpu())
            for key in ("latent_mean", "latent_std")
        )
    ):
        raise ValueError("Experimental checkpoint pairing or prediction type mismatch")


def train_candidate(
    cache, source_hash, out, config, device="cuda", resume=False, initialize_from=None
):
    """Save model/optimizer/RNG/cursor state without touching the source artifacts."""
    config.validate()
    if resume and initialize_from is not None:
        raise ValueError("Resume and checkpoint fork are mutually exclusive")
    out = Path(out)
    last = out / "last.pt"
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError(f"Run directory is not empty: {out}")
    if resume and not last.exists():
        raise ValueError(f"No resumable checkpoint in {out}")
    seed_all(config.seed)
    latent_dim = cache["latent_mean"].numel()
    model = build_model(config, latent_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    schedule = VelocitySchedule(config.diffusion_steps, device)
    epoch, cursor, step, best, stale, elapsed_before = 0, 0, 0, float("inf"), 0, 0.0
    provenance, epoch_offset, step_offset = None, 0, 0
    if resume or initialize_from is not None:
        source = last if resume else Path(initialize_from).resolve()
        state = load_checkpoint(source, KIND)
        validate_pairing(state, cache, source_hash)
        old = state["config"]
        mutable = {"epochs", "max_seconds", "max_steps"}
        if not resume:
            mutable |= {"learning_rate", "patience"}
        for key in asdict(config):
            if key not in mutable and old[key] != getattr(config, key):
                raise ValueError(f"Cannot change {key} on {'resume' if resume else 'fork'}")
        if not resume and state["batch_cursor"]:
            raise ValueError("Checkpoint fork requires a completed parent epoch")
        if str(device).startswith("cuda") and not state["cuda_rng"]:
            raise ValueError("Checkpoint has no CUDA random state for this device")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if resume:
            epoch, cursor, step = state["epoch"], state["batch_cursor"], state["step"]
            best, stale, elapsed_before = (
                state["best_score"],
                state["stale_epochs"],
                state["elapsed_seconds"],
            )
            provenance = state.get("parent")
            epoch_offset, step_offset = state.get("epoch_offset", 0), state.get("step_offset", 0)
        else:
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
            epoch_offset = state.get("epoch_offset", 0) + state["epoch"]
            step_offset = state.get("step_offset", 0) + state["step"]
            provenance = {
                "path": str(source),
                "sha256": file_hash(source),
                "epoch": epoch_offset,
                "step": step_offset,
            }
        torch.set_rng_state(state["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json",
        {**asdict(config), "source_sha256": source_hash, "parent": provenance},
    )
    train_z, train_text = cache["train"]["latents"].to(device), cache["train"]["text"].to(device)
    started = time.monotonic()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def snapshot():
        return {
            "format_version": FORMAT_VERSION,
            "kind": KIND,
            "prediction_type": "v_prediction",
            "schedule": SCHEDULE,
            "config": asdict(config),
            "latent_dim": latent_dim,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "source_sha256": source_hash,
            "epoch": epoch,
            "epoch_offset": epoch_offset,
            "step_offset": step_offset,
            "parent": provenance,
            "batch_cursor": cursor,
            "step": step,
            "best_score": best,
            "stale_epochs": stale,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            **{
                key: cache[key]
                for key in (
                    "vae_sha256",
                    "manifest_sha256",
                    "text_model",
                    "text_revision",
                    "latent_mean",
                    "latent_std",
                )
            },
        }

    def exhausted():
        return time.monotonic() - started >= config.max_seconds or (
            config.max_steps > 0 and step >= config.max_steps
        )

    def validate_save(completed_epoch):
        nonlocal best, stale
        metrics = evaluate(model, cache["val"], schedule, device)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Non-finite validation metrics")
        improved = metrics["velocity_mse"] < best
        if improved:
            best, stale = metrics["velocity_mse"], 0
        elif completed_epoch:
            stale += 1
        event = {"epoch": epoch, "step": step, "stale_epochs": stale, "validation": metrics}
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            import json

            stream.write(json.dumps(event) + "\n")
        save_checkpoint(last, snapshot())
        if improved:
            save_checkpoint(out / "best.pt", snapshot())
        if epoch % 10 == 0 or not completed_epoch:
            print({"variant": config.variant, **event}, flush=True)
        return metrics

    stop_reason = "epoch_limit"
    metrics = None
    if initialize_from is not None:
        # Anchor checkpoint selection to a freshly measured parent, not its historical score.
        metrics = validate_save(False)
    try:
        while epoch < config.epochs and not exhausted() and stale < config.patience:
            order = torch.randperm(
                len(train_z),
                generator=torch.Generator().manual_seed(config.seed + epoch_offset + epoch),
            )
            batches = order.split(config.batch_size)
            model.train()
            for index, indices in enumerate(batches):
                if index < cursor:
                    continue
                if exhausted():
                    break
                indices = indices.to(device)
                z, text = train_z[indices], train_text[indices]
                if config.text_dropout:
                    keep = torch.rand((len(z), 1), device=device) >= config.text_dropout
                    text = text * keep
                times = torch.randint(schedule.steps, (len(z),), device=device)
                noise = torch.randn_like(z)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(
                    model(schedule.noisy(z, times, noise), times, text),
                    schedule.target(z, times, noise),
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite experimental training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                cursor = index + 1
            if cursor == len(batches):
                epoch += 1
                cursor = 0
                metrics = validate_save(True)
            elif exhausted():
                break
        if stale >= config.patience:
            stop_reason = "validation_plateau"
        elif exhausted():
            stop_reason = (
                "step_limit" if config.max_steps and step >= config.max_steps else "time_budget"
            )
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    if cursor or metrics is None:
        metrics = validate_save(False)
    result = {
        "variant": config.variant,
        "step": step,
        "epoch": epoch,
        "total_epoch": epoch_offset + epoch,
        "parent": provenance,
        "stop_reason": stop_reason,
        "session_seconds": time.monotonic() - started,
        "validation": metrics,
        "best_velocity_mse": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if str(device).startswith("cuda")
            else None
        ),
    }
    write_json(out / "summary.json", result)
    return result


def load_candidate(path, cache, source_hash, device="cuda"):
    state = load_checkpoint(Path(path), KIND)
    validate_pairing(state, cache, source_hash)
    config = Config(**state["config"])
    config.validate()
    model = build_model(config, state["latent_dim"], device).eval()
    model.load_state_dict(state["model"])
    return model, VelocitySchedule(config.diffusion_steps, device), state
