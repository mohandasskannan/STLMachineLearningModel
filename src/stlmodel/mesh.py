"""Conservative surface extraction: never discard disconnected components."""

from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from .common import write_json


def export_stl(
    volume: np.ndarray, output: Path, size_mm: float = 100, threshold: float = 0.5
) -> dict:
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or min(volume.shape) < 2 or not np.isfinite(volume).all():
        raise ValueError("Expected a finite 3D occupancy probability grid")
    if volume.min() < 0 or volume.max() > 1:
        raise ValueError("Occupancy probabilities must be between zero and one")
    if not np.isfinite(size_mm) or size_mm <= 0:
        raise ValueError("Longest dimension must be a positive number of millimeters")
    if not 0 < threshold < 1:
        raise ValueError("Surface threshold must be between zero and one")
    if not np.any(volume > threshold):
        raise ValueError("The model produced empty geometry at the selected surface threshold")
    warnings = []
    boundary = np.concatenate(
        [
            volume[0].ravel(),
            volume[-1].ravel(),
            volume[:, 0].ravel(),
            volume[:, -1].ravel(),
            volume[:, :, 0].ravel(),
            volume[:, :, -1].ravel(),
        ]
    )
    if np.any(boundary > threshold):
        warnings.append("Geometry touches the grid boundary; the exterior is capped for export.")
    vertices, faces, _, _ = marching_cubes(
        np.pad(volume, 1), level=threshold, allow_degenerate=False
    )
    vertices -= 1
    # Source data uses Y up. Rotate to Z up for conventional slicer orientation.
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] *= -1
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals(multibody=True)
    mesh.apply_scale(size_mm / float(mesh.extents.max()))
    mesh.apply_translation(
        [-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), -mesh.bounds[0, 2]]
    )
    components = len(mesh.split(only_watertight=False))
    if components > 1:
        warnings.append(f"{components} disconnected components; all have been preserved.")
    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight; repair it before slicing.")
    warnings.append(
        "Check wall thickness, supports, and scale in your slicer. Printability is unverified."
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output), file_type="stl")
    report = {
        "path": str(output.resolve()),
        "units": "mm (interpret STL coordinates as millimeters)",
        "extents_mm": mesh.extents.tolist(),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "components": components,
        "watertight": bool(mesh.is_watertight),
        "warnings": warnings,
    }
    write_json(output.with_suffix(".mesh.json"), report)
    return report
