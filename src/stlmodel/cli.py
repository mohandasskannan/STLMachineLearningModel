"""Command-line entry points; run `python -m stlmodel --help`."""

import argparse
import json
from pathlib import Path

from .common import device_for, file_hash, load_checkpoint, write_json
from .data import VoxelDataset, make_fixture, prepare, read_voxels
from .mesh import export_stl
from .pipeline import Generator, cache_embeddings, evaluate_generation, reconstruct
from .training import TrainConfig, benchmark, evaluate_diffusion, evaluate_vae, load_vae, train


def parser():
    root = argparse.ArgumentParser(description="Local text-to-STL research pipeline")
    sub = root.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="Create procedural data for engineering checks")
    fixture.add_argument("--out", type=Path, default=Path("data/fixture"))
    fixture.add_argument("--count", type=int, default=12, help="Objects per category")
    fixture.add_argument("--seed", type=int, default=42)
    prep = sub.add_parser(
        "prepare", help="Import caption CSV and extracted solid 32-resolution voxels"
    )
    prep.add_argument("--captions", type=Path, required=True)
    prep.add_argument("--voxels", type=Path, required=True)
    prep.add_argument("--out", type=Path, default=Path("data/prepared"))
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--limit-per-category", type=int, default=0)
    export = sub.add_parser("export", help="Export known occupancy data to STL")
    export.add_argument("--voxel", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--size-mm", type=float, default=100)
    bench = sub.add_parser("benchmark", help="Measure synthetic VAE training speed and GPU memory")
    bench.add_argument("--batch-size", type=int, default=8)
    bench.add_argument("--steps", type=int, default=10)
    bench.add_argument("--out", type=Path, default=Path("runs/benchmark.json"))
    for kind in ("vae", "diffusion"):
        training = sub.add_parser("train-" + kind, help=f"Train or resume {kind}")
        training.add_argument(
            "--manifest" if kind == "vae" else "--cache", type=Path, required=True
        )
        training.add_argument("--out", type=Path, required=True)
        training.add_argument("--resume", type=Path)
        training.add_argument("--batch-size", type=int, default=None)
        training.add_argument("--epochs", type=int, default=None)
        training.add_argument("--max-hours", type=float, default=None)
        training.add_argument(
            "--max-steps", type=int, default=None, help="Total steps, including prior sessions"
        )
        training.add_argument("--seed", type=int, default=None)
        training.add_argument("--learning-rate", type=float, default=None)
        training.add_argument(
            "--limit", type=int, default=None, help="VAE training objects for overfit checks"
        )
    cache = sub.add_parser(
        "cache", help="Cache frozen text embeddings and trained shape representations"
    )
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--vae", type=Path, required=True)
    cache.add_argument("--out", type=Path, default=Path("data/embeddings.pt"))
    recon = sub.add_parser("reconstruct", help="Export targets and reconstructed validation shapes")
    recon.add_argument("--manifest", type=Path, required=True)
    recon.add_argument("--vae", type=Path, required=True)
    recon.add_argument("--out", type=Path, default=Path("outputs/reconstructions"))
    recon.add_argument("--split", choices=("train", "val", "test"), default="val")
    recon.add_argument("--count", type=int, default=4)
    evaluate = sub.add_parser(
        "evaluate", help="Evaluate a VAE or diffusion checkpoint on held-out data"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--source", type=Path, required=True, help="VAE manifest or diffusion embedding cache"
    )
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--out", type=Path, default=Path("outputs/evaluation.json"))
    generate = sub.add_parser("generate", help="Generate one STL with trained models")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--size-mm", type=float, default=100)
    generate.add_argument("--steps", type=int, default=50)
    generate.add_argument("--threshold", type=float, default=0.5)
    generate.add_argument("--out", type=Path, default=Path("outputs/generated.stl"))
    compare = sub.add_parser(
        "compare", help="Export prompt/seed comparisons and memorization diagnostics"
    )
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--prompt", action="append", required=True)
    compare.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    compare.add_argument("--steps", type=int, default=50)
    compare.add_argument("--out", type=Path, default=Path("outputs/comparison"))
    serve = sub.add_parser("serve", help="Launch the local browser interface")
    serve.add_argument("--out", type=Path, default=Path("outputs/ui"))
    serve.add_argument("--port", type=int, default=7860)
    for command in (generate, compare, serve):
        command.add_argument("--vae", type=Path, required=True)
        command.add_argument("--diffusion", type=Path, required=True)
    for command in (
        bench,
        cache,
        recon,
        evaluate,
        generate,
        compare,
        serve,
        sub.choices["train-vae"],
        sub.choices["train-diffusion"],
    ):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return root


def dispatch(args):
    command = args.command
    if command == "fixture":
        return {"manifest": str(make_fixture(args.out, args.count, args.seed))}
    if command == "prepare":
        if args.limit_per_category < 0:
            raise ValueError("limit-per-category cannot be negative")
        return {
            "manifest": str(
                prepare(args.captions, args.voxels, args.out, args.seed, args.limit_per_category)
            )
        }
    if command == "export":
        return export_stl(read_voxels(args.voxel), args.out, args.size_mm)
    if command == "benchmark":
        result = benchmark(args.device, args.batch_size, args.steps)
        write_json(args.out, result)
        return result
    if command.startswith("train-"):
        kind = command.removeprefix("train-")
        config = (
            TrainConfig(**load_checkpoint(args.resume, kind)["config"])
            if args.resume
            else TrainConfig()
        )
        if kind == "diffusion" and not args.resume:
            config.batch_size = 64
            config.latent_dim = load_checkpoint(args.cache, "embeddings")["latent_mean"].numel()
        for name in (
            "batch_size",
            "epochs",
            "max_hours",
            "max_steps",
            "seed",
            "learning_rate",
            "limit",
        ):
            value = getattr(args, name)
            if value is not None:
                setattr(config, name, value)
        if kind == "diffusion" and config.limit:
            raise ValueError("--limit applies only to VAE training")
        source = args.manifest if kind == "vae" else args.cache
        return train(kind, source, args.out, config, args.device, args.resume)
    if command == "cache":
        return cache_embeddings(args.manifest, args.vae, args.out, args.device)
    if command == "reconstruct":
        if args.count < 1:
            raise ValueError("count must be positive")
        return reconstruct(args.manifest, args.vae, args.out, args.split, args.count, args.device)
    if command == "evaluate":
        from torch.utils.data import TensorDataset

        from .models import Denoiser, Diffusion

        state = load_checkpoint(args.checkpoint)
        if state["source_sha256"] != file_hash(args.source):
            raise ValueError("Evaluation source does not match checkpoint")
        device = device_for(args.device)
        if state["kind"] == "vae":
            model, _ = load_vae(args.checkpoint, device)
            result = evaluate_vae(model, VoxelDataset(args.source, args.split), device)
        elif state["kind"] == "diffusion":
            cache = load_checkpoint(args.source, "embeddings")[args.split]
            model = Denoiser(**state["architecture"]).to(device)
            model.load_state_dict(state["model"])
            result = evaluate_diffusion(
                model,
                TensorDataset(cache["latents"], cache["text"]),
                Diffusion(state["diffusion_steps"], device),
                device,
            )
        else:
            raise ValueError("Only VAE and diffusion checkpoints can be evaluated")
        result = {"split": args.split, "checkpoint": str(args.checkpoint), "metrics": result}
        write_json(args.out, result)
        return result
    if command == "generate":
        return Generator(args.vae, args.diffusion, args.device).generate(
            args.prompt, args.out, args.seed, args.size_mm, args.steps, args.threshold
        )
    if command == "compare":
        return evaluate_generation(
            Generator(args.vae, args.diffusion, args.device),
            args.manifest,
            args.prompt,
            args.out,
            args.seeds,
            args.steps,
        )
    if command == "serve":
        from .ui import serve

        serve(args.vae, args.diffusion, args.out, args.device, args.port)
        return None
    raise ValueError(f"Unknown command: {command}")


def main():
    root = parser()
    args = root.parse_args()
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError) as exc:
        root.exit(2, f"Error: {exc}\n")
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
