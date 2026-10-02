"""Object-level dataset preparation. Coordinates remain in source XYZ order."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import file_hash, write_json

RESOLUTION = 32
SPLITS = ("train", "val", "test")


def read_voxels(path: Path) -> np.ndarray:
    """Read binary NPY/NPZ or Text2Shape RGBA NRRD; alpha defines occupancy."""
    path = Path(path)
    if path.suffix.lower() == ".nrrd":
        import nrrd

        array, _ = nrrd.read(str(path), index_order="F")
        if array.shape == (4, 32, 32, 32):
            array = array[3]  # Black but opaque voxels are still occupied.
        elif array.shape == (32, 32, 32, 4):
            array = array[..., 3]
        elif array.ndim != 3:
            raise ValueError(f"{path}: expected scalar occupancy or four-channel RGBA NRRD")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: NRRD occupancy contains non-finite values")
        array = array > 0
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            array = archive["occupancy"]
    elif path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported voxel format: {path.suffix}")
    if array.shape != (32, 32, 32):
        raise ValueError(
            f"{path}: expected (32, 32, 32), got {array.shape}; no implicit resampling"
        )
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{path}: occupancy must contain only finite zeros and ones")
    array = array.astype(np.uint8)
    if not array.any() or array.all():
        raise ValueError(f"{path}: empty or fully occupied grids are not useful training shapes")
    return array


def assign_splits(records: list[dict], seed: int) -> None:
    """Stratify by category, assigning whole objects, independent of input row order."""
    categories = defaultdict(list)
    for row in records:
        categories[row["category"]].append(row)
    for category, rows in categories.items():
        if len(rows) < 3:
            raise ValueError(f"Need at least 3 unique objects in category {category!r}")
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        holdout = max(1, int(len(rows) * 0.1))
        for index, row in enumerate(rows):
            row["split"] = "val" if index < holdout else "test" if index < 2 * holdout else "train"


def write_dataset(objects: list[dict], out: Path, seed: int, source: str) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Dataset output must be empty: {out}")
    records = [{k: v for k, v in row.items() if k != "occupancy"} for row in objects]
    assign_splits(records, seed)
    (out / "voxels").mkdir(parents=True, exist_ok=True)
    for record, original in zip(records, objects, strict=True):
        filename = hashlib.sha256(record["id"].encode()).hexdigest()[:24] + ".npz"
        record["voxel"] = "voxels/" + filename
        np.savez_compressed(out / record["voxel"], occupancy=original["occupancy"])
        record["sha256"] = file_hash(out / record["voxel"])
    manifest = out / "manifest.json"
    write_json(
        manifest,
        {
            "version": 1,
            "resolution": 32,
            "seed": seed,
            "source": source,
            "records": sorted(records, key=lambda r: r["id"]),
        },
    )
    return manifest


def prepare(
    captions: Path, voxels: Path, out: Path, seed: int = 42, limit_per_category: int = 0
) -> Path:
    """Import Text2Shape CSV (modelId, description, category) and extracted NRRDs.

    Also accepts model_id/object_id, caption/text, and an optional voxel_path
    relative to the CSV for locally supplied NPY/NPZ data.
    """
    objects = {}
    with Path(captions).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            identifier = row.get("modelId") or row.get("model_id") or row.get("object_id")
            caption = row.get("description") or row.get("caption") or row.get("text")
            category = (row.get("category") or "").strip().lower()
            if not identifier or not caption or not caption.strip() or not category:
                raise ValueError("CSV requires modelId, description, and category on every row")
            identifier = identifier.strip()
            obj = objects.setdefault(
                identifier,
                {
                    "id": identifier,
                    "category": category,
                    "captions": [],
                    "source_path": row.get("voxel_path"),
                },
            )
            if obj["category"] != category or obj["source_path"] != row.get("voxel_path"):
                raise ValueError(f"Conflicting metadata for object {identifier}")
            if caption.strip() not in obj["captions"]:
                obj["captions"].append(caption.strip())
    index = defaultdict(list)
    for path in Path(voxels).rglob("*"):
        if path.suffix.lower() in {".nrrd", ".npy", ".npz"}:
            index[path.stem].append(path)
            if path.parent.name != path.stem:
                index[path.parent.name].append(path)
    selected, counts = [], defaultdict(int)
    for obj in sorted(objects.values(), key=lambda r: r["id"]):
        if limit_per_category and counts[obj["category"]] >= limit_per_category:
            continue
        explicit = obj.pop("source_path")
        matches = [Path(captions).parent / explicit] if explicit else index.get(obj["id"], [])
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise ValueError(
                f"Object {obj['id']}: expected one voxel file, found {len(matches)}. "
                "Use only the solid 32-resolution archive, or specify voxel_path in CSV."
            )
        obj["occupancy"] = read_voxels(matches[0])
        selected.append(obj)
        counts[obj["category"]] += 1
    if not selected:
        raise ValueError("No captioned objects found")
    return write_dataset(selected, out, seed, "User-supplied Text2Shape-compatible data")


def make_fixture(out: Path, count: int = 12, seed: int = 42) -> Path:
    """Small procedural chairs/tables for engineering checks, not a research dataset."""
    if count < 3:
        raise ValueError("Use at least 3 fixtures per category")
    rng = np.random.default_rng(seed)
    objects = []
    for category in ("chair", "table"):
        for i in range(count):
            grid = np.zeros((32, 32, 32), dtype=np.uint8)
            left = int(rng.integers(4, 9))
            right = 32 - left
            height = int(rng.integers(12, 18))
            grid[left:right, height : height + 3, 6:26] = 1
            for x in (left, right - 3):
                for z in (6, 23):
                    grid[x : x + 3, 3:height, z : z + 3] = 1
            caption = f"a {category} with four legs"
            if category == "chair":
                grid[left:right, height + 3 : 28, 23:26] = 1
                arms = i % 2 == 0
                if arms:
                    grid[left : left + 2, height + 3 : height + 7, 9:24] = 1
                    grid[right - 2 : right, height + 3 : height + 7, 9:24] = 1
                caption += " with armrests" if arms else " without armrests"
            objects.append(
                {
                    "id": f"fixture-{category}-{i:04d}",
                    "category": category,
                    "captions": [caption],
                    "occupancy": grid,
                }
            )
    return write_dataset(objects, out, seed, "Procedural engineering fixture; not Text2Shape")


class VoxelDataset(Dataset):
    def __init__(self, manifest: Path, split: str, limit: int = 0):
        self.manifest = Path(manifest)
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if document.get("version") != 1 or document.get("resolution") != 32:
            raise ValueError("Expected a version 1, 32-resolution dataset")
        seen = set()
        for row in document["records"]:
            if row["id"] in seen or row["split"] not in SPLITS or not row["captions"]:
                raise ValueError(
                    "Dataset has duplicate object IDs, invalid splits, or missing captions"
                )
            seen.add(row["id"])
        self.records = [r for r in document["records"] if r["split"] == split]
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"Dataset has no objects in split {split}")
        for row in self.records:
            path = self.manifest.parent / row["voxel"]
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"Voxel data changed for {row['id']}; prepare a new dataset")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        grid = read_voxels(self.manifest.parent / row["voxel"])
        return torch.from_numpy(grid.astype(np.float32))[None], row["id"]
