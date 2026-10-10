"""Frozen-decoder, weights-only adaptation to the screened expanded furniture data."""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments/diffusion-high-noise-v1"
sys.path.insert(0, str(PREVIOUS))

import numpy as np
import torch
from experiment import Config, evaluate, load_candidate, train_candidate
from torch.utils.data import DataLoader
from verify_results import verify

from stlmodel.common import file_hash, load_checkpoint, save_checkpoint, write_json
from stlmodel.data import VoxelDataset
from stlmodel.text import TextEncoder
from stlmodel.training import load_vae

spec = importlib.util.spec_from_file_location(
    "matched_furniture_review", PREVIOUS / "run_experiment.py"
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

TRACKED = Path(__file__).resolve().parent
ARTIFACTS = Path("E:/STLModelData/furniture-diffusion-v2")
MANIFEST = Path("E:/STLModelData/furniture-data-v2/prepared/manifest.json")
CACHE = ARTIFACTS / "embeddings.pt"
RUN = ROOT / "runs/text2shape-furniture-diffusion-v2-conditioned"
PARENT = previous.RUNS["conditioned"] / "best.pt"
MANIFEST_HASH = "46c8a1140d454d13d05dd2545b6b0486edad5bd3bc1453afe4498735147e00a5"


def protected_hashes():
    paths = [previous.CACHE, previous.MANIFEST, previous.VAE, MANIFEST]
    for directory in (
        *previous.RUNS.values(),
        ROOT / "runs/text2shape-diffusion-low-lr-v1-conditioned",
        ROOT / "runs/text2shape-pilot-diffusion-v1",
    ):
        paths.extend(path for path in directory.iterdir() if path.is_file())
    return {str(path): file_hash(path) for path in paths}


def pilot_lookups(old):
    latents, texts = {}, {}
    for split in ("train", "val"):
        for index, identifier in enumerate(old[split]["ids"]):
            latents.setdefault(identifier, old[split]["latents"][index])
            texts.setdefault(old[split]["captions"][index], old[split]["text"][index])
    return latents, texts


def coverage_summary(rows):
    groups = {}
    for split in ("train", "val"):
        for category in ("chair", "table", "all"):
            values = [
                row["iou"]
                for row in rows
                if row["split"] == split
                and row["added"]
                and (category == "all" or row["category"] == category)
            ]
            if values:
                groups[f"added_{split}_{category}"] = {
                    "objects": len(values),
                    "mean_iou": float(np.mean(values)),
                    "p10_iou": float(np.quantile(values, 0.1)),
                    "fraction_iou_at_least_0.4": float(np.mean(np.asarray(values) >= 0.4)),
                }
    checks = {
        "added_validation_mean_iou_at_least_0.65": groups["added_val_all"]["mean_iou"] >= 0.65,
        "added_validation_90_percent_iou_at_least_0.4": groups["added_val_all"][
            "fraction_iou_at_least_0.4"
        ]
        >= 0.9,
        **{
            f"added_validation_{category}_mean_iou_at_least_0.60": groups[f"added_val_{category}"][
                "mean_iou"
            ]
            >= 0.6
            for category in ("chair", "table")
        },
    }
    return {
        "groups": groups,
        "checks": checks,
        "passed": all(checks.values()),
        "test_evaluated": False,
        "reconstruction": "deterministic posterior mean",
    }


@torch.inference_mode()
def build_cache(old, device):
    if CACHE.exists():
        raise ValueError("Expanded cache already exists; use train or review")
    vae, _ = load_vae(previous.VAE, device)
    old_latents, old_texts = pilot_lookups(old)
    old_records = {row["id"]: row for row in json.loads(previous.MANIFEST.read_text())["records"]}
    encoder = TextEncoder(old["text_model"], old["text_revision"])
    if encoder.revision != old["text_revision"]:
        raise ValueError("Text encoder revision changed")
    cache = {
        key: old[key]
        for key in (
            "format_version",
            "kind",
            "vae_sha256",
            "text_model",
            "text_revision",
            "latent_mean",
            "latent_std",
        )
    }
    cache.update(
        manifest_sha256=file_hash(MANIFEST),
        normalization_policy="frozen_epoch80_pilot_training_statistics",
        normalization_source_sha256=file_hash(previous.CACHE),
        test_policy="not_encoded_or_evaluated",
    )
    rows = []
    for split in ("train", "val"):
        dataset = VoxelDataset(MANIFEST, split)
        for row in dataset.records:
            if row["id"] in old_records:
                original = old_records[row["id"]]
                if original["split"] != split or original["sha256"] != row["sha256"]:
                    raise ValueError("Pilot object changed geometry or split")
        shape_latents = []
        cursor = 0
        for voxels, identifiers in DataLoader(dataset, batch_size=8, num_workers=0):
            voxels = voxels.to(device)
            mu, _ = vae.encode(voxels)
            normalized = (mu.cpu() - old["latent_mean"]) / old["latent_std"]
            # Original objects retain their exact cached latent values, including the fixed holdout.
            for i, identifier in enumerate(identifiers):
                if identifier in old_latents:
                    normalized[i] = old_latents[identifier]
            logits = vae.decode(
                normalized.to(device) * old["latent_std"].to(device) + old["latent_mean"].to(device)
            )
            predicted, truth = logits > 0, voxels > 0.5
            ious = (
                (
                    (predicted & truth).flatten(1).sum(1)
                    / (predicted | truth).flatten(1).sum(1).clamp_min(1)
                )
                .cpu()
                .tolist()
            )
            for i, identifier in enumerate(identifiers):
                record = dataset.records[cursor + i]
                rows.append(
                    {
                        "id": identifier,
                        "split": split,
                        "category": record["category"],
                        "added": identifier not in old_records,
                        "iou": ious[i],
                    }
                )
            cursor += len(identifiers)
            shape_latents.append(normalized)
            if cursor % 512 == 0 or cursor == len(dataset):
                print(
                    {
                        "stage": "decoder_coverage",
                        "split": split,
                        "objects": cursor,
                        "total": len(dataset),
                    },
                    flush=True,
                )
        shape_latents = torch.cat(shape_latents)
        captions, ids, indices = [], [], []
        for index, record in enumerate(dataset.records):
            for caption in record["captions"]:
                captions.append(caption)
                ids.append(record["id"])
                indices.append(index)
        unseen = list(dict.fromkeys(caption for caption in captions if caption not in old_texts))
        for start in range(0, len(unseen), 256):
            batch = unseen[start : start + 256]
            encoded = torch.from_numpy(encoder.encode(batch))
            if encoded.shape != (len(batch), 384) or not torch.isfinite(encoded).all():
                raise ValueError("Invalid text embeddings")
            old_texts.update(zip(batch, encoded, strict=True))
            if start % 2048 == 0:
                print(
                    {
                        "stage": "text_embeddings",
                        "split": split,
                        "captions": min(start + 256, len(unseen)),
                        "total": len(unseen),
                    },
                    flush=True,
                )
        cache[split] = {
            "latents": shape_latents[indices],
            "text": torch.stack([old_texts[caption] for caption in captions]),
            "ids": ids,
            "captions": captions,
        }
        if not torch.isfinite(cache[split]["latents"]).all():
            raise ValueError("Non-finite expanded latents")
    coverage = coverage_summary(rows)
    write_json(ARTIFACTS / "decoder-coverage-objects.json", rows)
    write_json(ARTIFACTS / "decoder-coverage.json", coverage)
    write_json(TRACKED / "decoder-coverage.json", coverage)
    save_checkpoint(CACHE, cache)
    write_json(
        ARTIFACTS / "cache-summary.json",
        {
            "cache_sha256": file_hash(CACHE),
            "manifest_sha256": file_hash(MANIFEST),
            "normalization_policy": cache["normalization_policy"],
            "examples": {split: len(cache[split]["ids"]) for split in ("train", "val")},
            "unique_objects": {split: len(set(cache[split]["ids"])) for split in ("train", "val")},
            "test_encoded": False,
        },
    )
    print({"decoder_coverage": coverage}, flush=True)


def load_expanded(old):
    cache = load_checkpoint(CACHE, "embeddings")
    summary = json.loads((ARTIFACTS / "cache-summary.json").read_text())
    if summary["cache_sha256"] != file_hash(CACHE) or cache["manifest_sha256"] != file_hash(
        MANIFEST
    ):
        raise ValueError("Expanded cache or manifest changed")
    for key in ("vae_sha256", "text_model", "text_revision"):
        if cache[key] != old[key]:
            raise ValueError(f"Expanded cache changes {key}")
    for key in ("latent_mean", "latent_std"):
        if not torch.equal(cache[key], old[key]):
            raise ValueError("Expanded cache changes normalization")
    if not json.loads((ARTIFACTS / "decoder-coverage.json").read_text())["passed"]:
        raise ValueError("Decoder coverage failed; investigate before training")
    return cache, summary["cache_sha256"]


def review(old, cache, cache_hash, protected, device):
    result = previous.review(
        old,
        file_hash(previous.CACHE),
        device,
        candidate_runs={"parent": PARENT.parent, "expanded": RUN},
        candidate_sources={"expanded": (cache, cache_hash)},
        artifact_dir=ARTIFACTS,
        include_legacy=False,
        guidance_scales=(2.0,),
        evaluate_test=False,
    )
    for name, path, paired, paired_hash in (
        ("parent", PARENT, old, file_hash(previous.CACHE)),
        ("expanded", RUN / "best.pt", cache, cache_hash),
    ):
        model, schedule, _ = load_candidate(path, paired, paired_hash, device)
        result["candidate_evaluations"][name]["expanded_val"] = evaluate(
            model, cache["val"], schedule, device
        )
    result.update(
        protected_source_hashes=protected,
        production_model_promoted=False,
        test_evaluation="not_encoded_or_evaluated",
        selected_checkpoint_hashes={
            "parent": file_hash(PARENT),
            "expanded": file_hash(RUN / "best.pt"),
        },
    )
    verification = verify(result, TRACKED, ARTIFACTS, ("parent_g2", "expanded_g2"))
    p, c = (result["candidate_evaluations"][name] for name in ("parent", "expanded"))
    pg, cg = (result["generation"][name] for name in ("parent_g2", "expanded_g2"))
    checks = {
        "expanded_validation_velocity_improves_at_least_0.5_percent": c["expanded_val"][
            "velocity_mse"
        ]
        <= p["expanded_val"]["velocity_mse"] * 0.995,
        "pilot_validation_velocity_regression_at_most_1_percent": c["val"]["velocity_mse"]
        <= p["val"]["velocity_mse"] * 1.01,
        "pilot_category_agreement_no_worse": cg["category_agreement"] >= pg["category_agreement"],
        "pilot_largest_voxel_fraction_no_worse": cg["largest_voxel_component_fraction_mean"]
        >= pg["largest_voxel_component_fraction_mean"] - 1e-7,
        "fixed_prompt_mesh_components_no_worse": verification["fixed_prompt_mesh_summary"][
            "expanded_g2"
        ]["mesh_components_mean"]
        <= verification["fixed_prompt_mesh_summary"]["parent_g2"]["mesh_components_mean"],
    }
    result["acceptance"] = {
        "checks": checks,
        "numerical_gates_passed": all(checks.values()),
        "visual_review": "pending",
        "accepted": False,
        "retained_checkpoint": str(PARENT),
    }
    for directory in (TRACKED, ARTIFACTS):
        write_json(directory / "results.json", result)
    shutil.copy2(ARTIFACTS / "comparison.png", TRACKED / "comparison.png")
    for filename in ("config.json", "summary.json", "metrics.jsonl"):
        shutil.copy2(RUN / filename, TRACKED / filename)
    print({"acceptance": result["acceptance"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("all", "cache", "train", "review"), nargs="?", default="all"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if file_hash(MANIFEST) != MANIFEST_HASH:
        raise ValueError("Expected the finalized screened furniture manifest")
    protected = protected_hashes()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    recorded = ARTIFACTS / "protected-sources.json"
    if recorded.exists() and json.loads(recorded.read_text()) != protected:
        raise ValueError("Protected artifacts changed since initialization")
    write_json(recorded, protected)
    old = load_checkpoint(previous.CACHE, "embeddings")
    previous.verify_sources(old, protected)
    _, _, state = load_candidate(PARENT, old, file_hash(previous.CACHE), args.device)
    if state["epoch"] != 80 or state["config"]["variant"] != "conditioned":
        raise ValueError("Expected the selected epoch-80 parent")
    if args.stage in ("all", "cache") and not (args.resume and CACHE.exists()):
        build_cache(old, args.device)
    if args.stage in ("all", "train", "review"):
        cache, cache_hash = load_expanded(old)
        if args.stage in ("all", "train"):
            elapsed = load_checkpoint(RUN / "last.pt")["elapsed_seconds"] if args.resume else 0
            if elapsed < 600:
                config = Config(
                    **{
                        **state["config"],
                        "learning_rate": 0.00005,
                        "epochs": 30,
                        "max_seconds": 600 - elapsed,
                        "patience": 10,
                        "max_steps": 0,
                    }
                )
                summary = train_candidate(
                    cache,
                    cache_hash,
                    RUN,
                    config,
                    args.device,
                    resume=args.resume,
                    warm_start_from=None if args.resume else PARENT,
                    warm_start_source=None if args.resume else (old, file_hash(previous.CACHE)),
                )
                print(summary, flush=True)
                if summary["stop_reason"] == "interrupted":
                    return
        if args.stage in ("all", "review"):
            review(old, cache, cache_hash, protected, args.device)
    if protected_hashes() != protected:
        raise ValueError("A protected artifact changed")
    print({"completed": args.stage, "protected_artifacts_unchanged": True}, flush=True)


if __name__ == "__main__":
    main()
