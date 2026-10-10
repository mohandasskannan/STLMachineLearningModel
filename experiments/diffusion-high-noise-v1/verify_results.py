"""Independently reopen every exported STL and render matched armrest controls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import trimesh

from stlmodel.common import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
TRACKED = Path(__file__).resolve().parent
ARTIFACTS = Path("E:/STLModelData/diffusion-high-noise-v1")


def verify(result, tracked=TRACKED, artifact_dir=ARTIFACTS, selected_variants=("conditioned_g2",)):
    checks = []
    aggregates = {}
    for variant, rows in result["fixed_prompt_exports"].items():
        counts = []
        for row in rows:
            if "error" in row:
                raise ValueError(f"Failed export in {variant}: {row['error']}")
            path = Path(row["mesh"]["path"])
            mesh = trimesh.load_mesh(path)
            check = {
                "variant": variant,
                "path": str(path),
                "finite": bool(np.isfinite(mesh.vertices).all()),
                "scale_correct": bool(np.isclose(mesh.extents.max(), 100, atol=0.001)),
                "on_build_plane": bool(np.isclose(mesh.bounds[0, 2], 0, atol=0.001)),
                "watertight": bool(mesh.is_watertight),
                "components": len(mesh.split(only_watertight=False)),
            }
            if not all(check[key] for key in ("finite", "scale_correct", "on_build_plane")):
                raise ValueError(f"Invalid exported STL: {check}")
            counts.append(check["components"])
            checks.append(check)
        aggregates[variant] = {
            "samples": len(rows),
            "mesh_components_mean": float(np.mean(counts)),
            "mesh_components_min": int(min(counts)),
            "mesh_components_max": int(max(counts)),
        }
    protected = {
        path: file_hash(Path(path)) == expected
        for path, expected in result["protected_source_hashes"].items()
    }
    if not all(protected.values()):
        raise ValueError("A protected source artifact changed")
    selected_paths = [
        Path(row["mesh"]["path"])
        for variant in selected_variants
        for row in result["fixed_prompt_exports"][variant]
    ]
    labels = [
        f"{variant} / {row['prompt']} / seed {row['seed']}"
        for variant in selected_variants
        for row in result["fixed_prompt_exports"][variant]
    ]
    spec = importlib.util.spec_from_file_location(
        "independent_renderer", ROOT / "experiments/procedural-v1/render_review.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.render(
        selected_paths, labels, tracked / "conditioned-g2-prompts.png", columns=3, front=True
    )
    verification = {
        "reopened_count": len(checks),
        "all_finite": all(row["finite"] for row in checks),
        "all_scale_correct": all(row["scale_correct"] for row in checks),
        "all_on_build_plane": all(row["on_build_plane"] for row in checks),
        "watertight_count": sum(row["watertight"] for row in checks),
        "protected_sources_unchanged": all(protected.values()),
        "fixed_prompt_mesh_summary": aggregates,
        "meshes": checks,
    }
    write_json(tracked / "verification.json", verification)
    write_json(artifact_dir / "verification.json", verification)
    print(
        json.dumps({key: value for key, value in verification.items() if key != "meshes"}, indent=2)
    )
    return verification


def main():
    result = json.loads((TRACKED / "results.json").read_text())
    verify(result)


if __name__ == "__main__":
    main()
