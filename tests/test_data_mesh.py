import csv
import json

import nrrd
import numpy as np
import pytest
import trimesh

from stlmodel.data import VoxelDataset, make_fixture, prepare, read_voxels
from stlmodel.mesh import export_stl


def test_object_splits_are_disjoint_and_repeatable(tmp_path):
    a = make_fixture(tmp_path / "a", 6)
    b = make_fixture(tmp_path / "b", 6)
    first = json.loads(a.read_text())["records"]
    second = json.loads(b.read_text())["records"]
    assert {r["id"]: r["split"] for r in first} == {r["id"]: r["split"] for r in second}
    splits = [{r["id"] for r in first if r["split"] == split} for split in ("train", "val", "test")]
    assert all(splits)
    assert not (splits[0] & splits[1] or splits[1] & splits[2] or splits[0] & splits[2])
    assert len(VoxelDataset(a, "train")) == 8
    with pytest.raises(ValueError, match="empty"):
        make_fixture(tmp_path / "a")


def test_nrrd_uses_alpha_and_preserves_axes(tmp_path):
    rgba = np.zeros((4, 32, 32, 32), dtype=np.uint8)
    rgba[3, 2:5, 10:17, 20:22] = 255  # Opaque black is geometry.
    rgba[:3, 0, 0, 0] = 255  # Transparent white is not geometry.
    path = tmp_path / "shape.nrrd"
    nrrd.write(str(path), rgba, index_order="F")
    grid = read_voxels(path)
    assert grid.sum() == 3 * 7 * 2
    assert grid[2, 10, 20] == 1
    assert grid[0, 0, 0] == 0


def test_caption_import_and_tamper_detection(tmp_path):
    captions = tmp_path / "captions.csv"
    with captions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["modelId", "description", "category"])
        writer.writeheader()
        for i in range(5):
            grid = np.zeros((32, 32, 32), dtype=np.uint8)
            grid[3 : 10 + i, 4:12, 4:15] = 1
            np.save(tmp_path / f"shape{i}.npy", grid)
            for caption in ("a chair", "a chair with legs"):
                writer.writerow(
                    {"modelId": f"shape{i}", "description": caption, "category": "chair"}
                )
    manifest = prepare(captions, tmp_path, tmp_path / "prepared")
    rows = json.loads(manifest.read_text())["records"]
    assert len(rows) == 5
    assert all(len(r["captions"]) == 2 for r in rows)
    row = next(r for r in rows if r["split"] == "train")
    (manifest.parent / row["voxel"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        VoxelDataset(manifest, "train")


def test_export_round_trip_size_components_and_orientation(tmp_path):
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume[2:8, 3:23, 2:8] = 1
    volume[24:28, 3:10, 24:28] = 1
    output = tmp_path / "mesh.stl"
    report = export_stl(volume, output, size_mm=123)
    loaded = trimesh.load_mesh(output)
    assert np.isfinite(loaded.vertices).all()
    assert loaded.extents.max() == pytest.approx(123, rel=1e-5)
    assert loaded.bounds[0, 2] == pytest.approx(0, abs=1e-5)
    assert report["components"] == 2
    assert loaded.is_watertight
    assert len(loaded.split()) == 2
    assert output.with_suffix(".mesh.json").exists()


def test_empty_invalid_and_boundary_meshes(tmp_path):
    volume = np.zeros((32, 32, 32))
    with pytest.raises(ValueError, match="empty geometry"):
        export_stl(volume, tmp_path / "empty.stl")
    volume[0:5, 0:5, 0:5] = 1
    with pytest.raises(ValueError, match="positive"):
        export_stl(volume, tmp_path / "negative.stl", size_mm=-1)
    report = export_stl(volume, tmp_path / "boundary.stl")
    assert any("boundary" in w for w in report["warnings"])
    assert report["watertight"]
