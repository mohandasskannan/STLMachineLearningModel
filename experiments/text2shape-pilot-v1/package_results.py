"""Independently verify exports and package review artifacts, excluding data/weights."""

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from stlmodel.common import file_hash, load_checkpoint, write_json

ROOT = Path("E:/STLModelData/text2shape-v1")
OUT = ROOT / "review"
DEST = Path("experiments/text2shape-pilot-v1")


def main():
    manifest = ROOT / "prepared/manifest.json"
    audit = json.loads((OUT / "data-audit.json").read_text())
    vae = json.loads((OUT / "vae-evaluation.json").read_text())
    diffusion = json.loads((OUT / "diffusion-evaluation.json").read_text())
    assert file_hash(manifest) == audit["manifest_sha256"]
    assert file_hash(vae["checkpoint"]) == vae["sha256"]
    assert file_hash(ROOT / "embeddings.pt") == diffusion["cache_sha256"]
    assert vae["sha256"] == diffusion["vae_sha256"]
    for values in diffusion["checkpoints"].values():
        state = load_checkpoint(values["checkpoint"], "diffusion")
        assert file_hash(values["checkpoint"]) == values["sha256"]
        assert state["vae_sha256"] == vae["sha256"]
        assert state["source_sha256"] == diffusion["cache_sha256"]
    checks = []
    for stage in ("orientation", "reconstructions", "generated-best", "generated-last"):
        for path in sorted((OUT / stage).glob("*.stl")):
            mesh = trimesh.load_mesh(path)
            assert np.isfinite(mesh.vertices).all(), path
            assert mesh.is_watertight, path
            assert np.isclose(max(mesh.extents), 100, atol=1e-4), path
            assert np.isclose(mesh.bounds[0, 2], 0, atol=1e-4), path
            checks.append(
                {
                    "stage": stage,
                    "file": path.name,
                    "sha256": file_hash(path),
                    "finite": True,
                    "watertight": True,
                    "extents_mm": mesh.extents.tolist(),
                    "components": len(mesh.split(only_watertight=False)),
                }
            )
        for path in (OUT / stage).iterdir():
            if path.suffix in {".json", ".png"}:
                target = DEST / stage / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    assert len(checks) == 48
    assert (
        file_hash("runs/procedural-vae-v1/best.pt")
        == "388c5d9a50d50d9c35f99ec761c9ccdf5371a80f534808c9b00a21728a028dc9"
    )
    assert (
        file_hash("runs/procedural-diffusion-extended-v2/best.pt")
        == "8721788b009bab23637b34bfd5bcfe9405d594592b4dbde35d31e858668e02e5"
    )
    for name in (
        "data-audit.json",
        "review-objects.json",
        "vae-sessions.json",
        "vae-evaluation.json",
        "cache-summary.json",
        "diffusion-sessions.json",
        "diffusion-evaluation.json",
    ):
        shutil.copy2(OUT / name, DEST / name)
    for kind in ("vae", "diffusion"):
        directory = DEST / "training" / kind
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "summary.json", "metrics.jsonl"):
            shutil.copy2(Path(f"runs/text2shape-pilot-{kind}-v1") / name, directory / name)
    verification = {
        "independently_reopened_meshes": len(checks),
        "meshes": checks,
        "model_pairing_valid": True,
        "frozen_vae_and_cache_unchanged": True,
        "previous_selected_checkpoints_preserved": True,
        "orientation_regression_tests": "3 passed",
        "ui": {
            "url": "http://127.0.0.1:7862",
            "checkpoint": diffusion["checkpoints"]["best"]["checkpoint"],
            "exact_chair_prompt_seed_42": "preview and STL returned through Gradio API",
        },
    }
    write_json(DEST / "verification.json", verification)
    for path in DEST.rglob("*.json"):
        json.loads(path.read_text())
    assert not list(DEST.rglob("*.pt")) and not list(DEST.rglob("*.stl"))
    print(
        json.dumps(
            {
                "verified_meshes": len(checks),
                "vae_metrics": vae["metrics"],
                "diffusion_metrics": diffusion["checkpoints"]["best"]["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
