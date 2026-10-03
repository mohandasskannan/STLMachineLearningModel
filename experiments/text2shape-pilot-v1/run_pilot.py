"""Download and audit a real-furniture pilot, then train with bounded budgets.

Run from the repository root with the project Python. Data stays on E:.
Commands: prepare, vae, diffusion. Review reconstructions before diffusion.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import nrrd
import numpy as np
import torch

from stlmodel.common import file_hash, load_checkpoint, write_json
from stlmodel.data import VoxelDataset, prepare, read_voxels
from stlmodel.mesh import export_stl
from stlmodel.training import evaluate_vae, load_vae

ROOT = Path("E:/STLModelData/text2shape-v1")
RAW = ROOT / "raw"
OUT = ROOT / "review"
MANIFEST = ROOT / "prepared/manifest.json"
CACHE = ROOT / "embeddings.pt"
VAE = Path("runs/text2shape-pilot-vae-v1")
DIFFUSION = Path("runs/text2shape-pilot-diffusion-v1")
BASE = "https://svl.stanford.edu/projects/text2shape/dataset/"
PROMPTS = [
    "A chair with a tall back and armrests",
    "a chair with armrests",
    "a chair without armrests",
    "a table with four legs",
]


def status(stage, **extra):
    value = {"stage": stage, "utc": datetime.now(UTC).isoformat(), **extra}
    write_json(OUT / "status.json", value)
    print(json.dumps(value), flush=True)


def renderer():
    path = Path("experiments/procedural-v1/render_review.py")
    spec = importlib.util.spec_from_file_location("review_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render


def canonical_voxels(path):
    """Use NRRD space directions to reorder storage axes into world XYZ (Y up)."""
    grid = read_voxels(path)
    header = nrrd.read_header(str(path))
    directions = np.asarray(header["space directions"])
    directions = directions[np.isfinite(directions).all(axis=1)]
    if directions.shape != (3, 3):
        raise ValueError("Expected three spatial direction vectors")
    world_axes = np.argmax(np.abs(directions), axis=1)
    if sorted(world_axes.tolist()) != [0, 1, 2]:
        raise ValueError("Spatial axes are not a signed permutation of XYZ")
    for source_axis, world_axis in enumerate(world_axes):
        off_axis = np.delete(directions[source_axis], world_axis)
        if not np.allclose(off_axis, 0):
            raise ValueError("Oblique grids require unsupported resampling")
    permutation = np.argsort(world_axes)
    grid = grid.transpose(tuple(permutation))
    for world_axis, source_axis in enumerate(permutation):
        if directions[source_axis, world_axis] < 0:
            grid = np.flip(grid, axis=world_axis)
    return np.ascontiguousarray(grid), permutation.tolist()


def download(url, target):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                length = response.headers.get("Content-Length")
                total = offset + int(length) if length else None
                received = offset
                last_print = time.monotonic()
                with partial.open("ab" if offset else "wb") as sink:
                    while chunk := response.read(1024 * 1024):
                        sink.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_print > 20:
                            status("download", file=target.name, bytes=received, total_bytes=total)
                            last_print = time.monotonic()
                if total is not None and received != total:
                    raise OSError(f"Incomplete download: {received}/{total}")
            partial.replace(target)
            return
        except OSError as exc:
            print(f"Download attempt {attempt + 1}: {exc}", flush=True)
            if attempt == 3:
                raise


def prepare_pilot():
    if MANIFEST.exists():
        raise RuntimeError("Prepared pilot already exists; do not replace its splits")
    status("download", drive=str(ROOT), seed=42, objects_per_category=1000)
    captions = RAW / "captions.tablechair.csv"
    archive = RAW / "nrrd_256_filter_div_32_solid.zip"
    sources = [(BASE + captions.name, captions), (BASE + "shapenet/" + archive.name, archive)]
    for url, path in sources:
        download(url, path)
    status("archive_validation")
    extraction = RAW / "voxels"
    with zipfile.ZipFile(archive) as z:
        corrupt = z.testzip()
        if corrupt:
            raise RuntimeError(f"ZIP integrity failure: {corrupt}")
        destination = extraction.resolve()
        for entry in z.infolist():
            resolved = (destination / entry.filename).resolve()
            if not resolved.is_relative_to(destination):
                raise RuntimeError(f"Unsafe archive member: {entry.filename}")
        z.extractall(extraction)
    audit = {
        "source_urls": [url for url, _ in sources],
        "download_sha256": {path.name: file_hash(path) for _, path in sources},
        "archive_crc_verified": True,
        "seed": 42,
        "excluded": [],
        "duplicates": [],
    }
    objects = {}
    with captions.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 2):
            identifier = row["modelId"].strip()
            category = row["category"].strip().lower()
            caption = row["description"].strip()
            if not identifier or not caption or category not in {"chair", "table"}:
                audit["excluded"].append(
                    {
                        "csv_row": index,
                        "id": identifier,
                        "reason": "Missing ID/caption or unsupported category",
                    }
                )
                continue
            obj = objects.setdefault(
                identifier, {"id": identifier, "category": category, "captions": []}
            )
            if obj["category"] != category:
                raise RuntimeError(f"Conflicting categories: {identifier}")
            if caption not in obj["captions"]:
                obj["captions"].append(caption)
    audit["captioned_objects_by_category"] = dict(Counter(o["category"] for o in objects.values()))
    index = defaultdict(list)
    for path in extraction.rglob("*.nrrd"):
        if "__MACOSX" not in path.parts and not path.name.startswith("._"):
            index[path.stem].append(path)
    seen, valid = {}, defaultdict(list)
    axis_permutations = Counter()
    status("voxel_audit", objects=len(objects))
    for n, obj in enumerate(sorted(objects.values(), key=lambda o: o["id"]), 1):
        identifier = obj["id"]
        paths = index.get(identifier, [])
        try:
            if len(paths) != 1:
                raise ValueError(f"Expected one voxel file, found {len(paths)}")
            grid, permutation = canonical_voxels(paths[0])
        except (ValueError, OSError, nrrd.NRRDError) as exc:
            audit["excluded"].append({"id": identifier, "reason": str(exc)})
            continue
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        axis_permutations[str(permutation)] += 1
        if digest in seen:
            audit["duplicates"].append(
                {"id": identifier, "retained_id": seen[digest], "occupancy_sha256": digest}
            )
            continue
        seen[digest] = identifier
        obj.update(voxel_path=str(paths[0].resolve()), occupancy_sha256=digest)
        valid[obj["category"]].append(obj)
        if n % 1000 == 0:
            status(
                "voxel_audit",
                checked=n,
                total=len(objects),
                excluded=len(audit["excluded"]),
                duplicates=len(audit["duplicates"]),
            )
    audit["unique_valid_objects_by_category"] = {c: len(v) for c, v in valid.items()}
    audit["orientation"] = {
        "method": "NRRD space directions: reorder storage axes into world XYZ, Y up",
        "permutation_counts": dict(axis_permutations),
        "application_source_changed": False,
    }
    rng = random.Random(42)
    selected = []
    for category in ("chair", "table"):
        if len(valid[category]) < 1000:
            write_json(OUT / "data-audit.json", audit)
            raise RuntimeError(f"Insufficient valid {category} objects: {len(valid[category])}")
        selected.extend(rng.sample(valid[category], 1000))
    selection = ROOT / "selected-captions.csv"
    canonical = ROOT / "canonical-voxels"
    canonical.mkdir(parents=True, exist_ok=True)
    with selection.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["modelId", "category", "description", "voxel_path"]
        )
        writer.writeheader()
        for obj in sorted(selected, key=lambda o: o["id"]):
            grid, _ = canonical_voxels(Path(obj["voxel_path"]))
            normalized_path = canonical / (obj["id"] + ".npz")
            np.savez_compressed(normalized_path, occupancy=grid)
            for caption in obj["captions"]:
                writer.writerow(
                    {
                        "modelId": obj["id"],
                        "category": obj["category"],
                        "description": caption,
                        "voxel_path": str(normalized_path.resolve()),
                    }
                )
    prepare(selection, extraction, MANIFEST.parent, seed=42)
    document = json.loads(MANIFEST.read_text())
    records = document["records"]
    split_counts = Counter(r["split"] for r in records)
    assert dict(split_counts) == {"train": 1600, "val": 200, "test": 200}
    assert len({r["id"] for r in records}) == 2000
    hashes = set()
    for record in records:
        digest = hashlib.sha256(
            read_voxels(MANIFEST.parent / record["voxel"]).tobytes()
        ).hexdigest()
        assert digest not in hashes, record["id"]
        hashes.add(digest)
    audit.update(
        selected_objects=2000,
        selected_captions=sum(len(r["captions"]) for r in records),
        split_objects=dict(split_counts),
        split_categories={
            s: dict(Counter(r["category"] for r in records if r["split"] == s))
            for s in ("train", "val", "test")
        },
        split_captions={
            s: sum(len(r["captions"]) for r in records if r["split"] == s)
            for s in ("train", "val", "test")
        },
        manifest_sha256=file_hash(MANIFEST),
        selected_caption_sha256=file_hash(selection),
        duplicate_geometry_across_splits=False,
        selected_objects_provenance=[
            {k: o[k] for k in ("id", "category", "occupancy_sha256")}
            for o in sorted(selected, key=lambda o: o["id"])
        ],
    )
    write_json(OUT / "data-audit.json", audit)
    # Export known grids before spending GPU time; confirm source axes visually.
    targets = OUT / "orientation"
    targets.mkdir(parents=True, exist_ok=True)
    chosen = []
    for category in ("chair", "table"):
        chosen.extend([r for r in records if r["category"] == category and r["split"] == "val"][:4])
    for i, r in enumerate(chosen):
        export_stl(
            read_voxels(MANIFEST.parent / r["voxel"]),
            targets / f"{i:02d}-{r['category']}-target.stl",
        )
    paths = sorted(targets.glob("*.stl"))
    checks = renderer()(paths, [p.stem for p in paths], targets / "preview.png", 4, front=True)
    write_json(targets / "reopened-mesh-checks.json", checks)
    write_json(
        OUT / "review-objects.json",
        [{"id": r["id"], "category": r["category"], "captions": r["captions"]} for r in chosen],
    )
    status(
        "prepared",
        manifest=str(MANIFEST),
        split_objects=dict(split_counts),
        captions=audit["selected_captions"],
    )


def train_stage(kind):
    run = VAE if kind == "vae" else DIFFUSION
    source = MANIFEST if kind == "vae" else CACHE
    record_path = OUT / f"{kind}-sessions.json"
    if run.exists() and not record_path.exists():
        raise RuntimeError(f"Existing run {run} has no budget ledger; inspect before resuming")
    ledger = (
        json.loads(record_path.read_text())
        if record_path.exists()
        else {"sessions": [], "training_budget_seconds": 1800}
    )
    elapsed = sum(s["session_seconds"] for s in ledger["sessions"])
    initial_remaining = 1800 - elapsed
    if initial_remaining <= 1:
        raise RuntimeError("Stage budget has already been consumed")
    status(f"training_{kind}", remaining_training_seconds=initial_remaining)
    stopping = "time_budget"
    while elapsed < 1799:
        last = run / "last.pt"
        epoch = load_checkpoint(last, kind)["epoch"] if last.exists() else 0
        command = [
            sys.executable,
            "-m",
            "stlmodel",
            "train-" + kind,
            "--manifest" if kind == "vae" else "--cache",
            str(source),
            "--out",
            str(run),
            "--device",
            "cuda",
            "--epochs",
            str(epoch + 5),
            "--max-hours",
            str((1800 - elapsed) / 3600),
        ]
        if last.exists():
            command.extend(["--resume", str(last)])
        else:
            command.extend(
                [
                    "--batch-size",
                    "8" if kind == "vae" else "64",
                    "--seed",
                    "42",
                    "--learning-rate",
                    "0.0002",
                ]
            )
        log = OUT / f"{kind}-console.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command}) + "\n")
            stream.flush()
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((run / "summary.json").read_text())
        ledger["sessions"].append(summary)
        elapsed += summary["session_seconds"]
        write_json(record_path, ledger)
        state = load_checkpoint(last, kind)
        best = load_checkpoint(run / "best.pt", kind)
        status(
            f"training_{kind}",
            training_seconds=elapsed,
            epoch=state["epoch"],
            step=state["step"],
            best_epoch=best["epoch"],
            best_step=best["step"],
        )
        if summary["interrupted"]:
            stopping = "interrupted"
            break
        if state["epoch"] - best["epoch"] >= 10:
            stopping = "validation_plateau_10_epochs"
            break
    ledger.update(
        training_seconds=elapsed,
        stopping_reason=stopping,
        selected_checkpoint=str(run / "best.pt"),
        selected_sha256=file_hash(run / "best.pt"),
    )
    write_json(record_path, ledger)
    status(f"trained_{kind}", training_seconds=elapsed, stopping_reason=stopping)
    return ledger


def review_vae():
    model, state = load_vae(VAE / "best.pt", "cuda")
    metrics = {
        s: evaluate_vae(model, VoxelDataset(MANIFEST, s), torch.device("cuda"))
        for s in ("val", "test")
    }
    reviews = json.loads((OUT / "review-objects.json").read_text())
    records = {r["id"]: r for r in json.loads(MANIFEST.read_text())["records"]}
    folder = OUT / "reconstructions"
    folder.mkdir(parents=True, exist_ok=True)
    reports = []
    with torch.inference_mode():
        for i, review in enumerate(reviews):
            record = records[review["id"]]
            grid = read_voxels(MANIFEST.parent / record["voxel"])
            tensor = torch.from_numpy(grid.astype(np.float32))[None, None].to("cuda")
            logits, _, _ = model(tensor)
            item = {**review}
            prefix = f"{i:02d}-{record['category']}"
            item["target"] = export_stl(grid, folder / f"{prefix}-0-target.stl")
            try:
                item["reconstruction"] = export_stl(
                    logits.sigmoid()[0, 0].cpu().numpy(), folder / f"{prefix}-1-reconstruction.stl"
                )
            except ValueError as exc:
                item["error"] = str(exc)
            reports.append(item)
    paths = sorted(folder.glob("*.stl"))
    checks = renderer()(paths, [p.stem for p in paths], folder / "preview.png", 4, front=True)
    write_json(folder / "reopened-mesh-checks.json", checks)
    result = {
        "checkpoint": str(VAE / "best.pt"),
        "sha256": file_hash(VAE / "best.pt"),
        "step": state["step"],
        "epoch": state["epoch"],
        "metrics": metrics,
        "iou_gate_passed": metrics["val"]["iou"] >= 0.50,
        "visual_gate": "pending",
        "objects": reports,
    }
    write_json(OUT / "vae-evaluation.json", result)
    status(
        "reconstruction_review_required", metrics=metrics, iou_gate_passed=result["iou_gate_passed"]
    )


def diffusion_stage():
    review = json.loads((OUT / "vae-evaluation.json").read_text())
    if not review["iou_gate_passed"] or review["visual_gate"] != "passed":
        raise RuntimeError(
            "VAE reconstruction gate has not passed; diffusion budget remains unused"
        )
    from torch.utils.data import TensorDataset

    from stlmodel.models import Denoiser, Diffusion
    from stlmodel.pipeline import Generator, cache_embeddings, evaluate_generation
    from stlmodel.training import evaluate_diffusion

    if not CACHE.exists():
        status("caching_embeddings")
        write_json(
            OUT / "cache-summary.json", cache_embeddings(MANIFEST, VAE / "best.pt", CACHE, "cuda")
        )
    assert file_hash(VAE / "best.pt") == review["sha256"]
    cache_hash = file_hash(CACHE)
    train_stage("diffusion")
    cache = load_checkpoint(CACHE, "embeddings")
    evaluations = {}
    for name in ("best", "last"):
        path = DIFFUSION / f"{name}.pt"
        state = load_checkpoint(path, "diffusion")
        model = Denoiser(**state["architecture"]).to("cuda")
        model.load_state_dict(state["model"])
        schedule = Diffusion(state["diffusion_steps"], "cuda")
        metrics = {
            s: evaluate_diffusion(
                model,
                TensorDataset(cache[s]["latents"], cache[s]["text"]),
                schedule,
                torch.device("cuda"),
            )
            for s in ("val", "test")
        }
        folder = OUT / f"generated-{name}"
        generator = Generator(VAE / "best.pt", path, "cuda")
        report = evaluate_generation(generator, MANIFEST, PROMPTS, folder, seeds=[42, 43, 44])
        for row in report["samples"]:
            if "mesh" in row:
                meshpath = Path(row["mesh"]["path"])
                write_json(
                    meshpath.with_suffix(".generation.json"),
                    {
                        **generator.metadata,
                        "prompt": row["prompt"],
                        "seed": row["seed"],
                        "size_mm": 100,
                        "sampling_steps": 50,
                        "threshold": 0.5,
                    },
                )
        paths = sorted(folder.glob("*.stl"))
        checks = (
            renderer()(paths, [p.stem for p in paths], folder / "preview.png", 3, front=True)
            if paths
            else []
        )
        write_json(folder / "reopened-mesh-checks.json", checks)
        evaluations[name] = {
            "checkpoint": str(path),
            "sha256": file_hash(path),
            "step": state["step"],
            "epoch": state["epoch"],
            "metrics": metrics,
            "samples": report["samples"],
        }
    assert file_hash(VAE / "best.pt") == review["sha256"]
    assert file_hash(CACHE) == cache_hash
    write_json(
        OUT / "diffusion-evaluation.json",
        {
            "checkpoints": evaluations,
            "vae_sha256": review["sha256"],
            "cache_sha256": cache_hash,
            "prompts": PROMPTS,
            "seeds": [42, 43, 44],
        },
    )
    status("complete", evaluations={k: v["metrics"] for k, v in evaluations.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "vae", "diffusion"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare_pilot()
    elif args.stage == "vae":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        train_stage("vae")
        review_vae()
    else:
        diffusion_stage()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
