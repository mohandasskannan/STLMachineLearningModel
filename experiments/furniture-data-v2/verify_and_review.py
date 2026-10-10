"""Verify every prepared object and render targeted human-caption spot checks."""

import argparse
import csv
import hashlib
import importlib.util
import json
import textwrap
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from stlmodel.common import file_hash, write_json
from stlmodel.data import VoxelDataset, read_voxels
from stlmodel.mesh import export_stl

ROOT = Path(__file__).resolve().parents[2]
TRACKED = Path(__file__).resolve().parent
DATA = Path("E:/STLModelData/furniture-data-v2")
MANIFEST = DATA / "prepared/manifest.json"
PILOT_MANIFEST = Path("E:/STLModelData/text2shape-v1/prepared/manifest.json")


def verify_caption_provenance(records, quality, source_csv):
    with source_csv.open(encoding="utf-8-sig", newline="") as stream:
        source = {
            row["id"]: {**row, "csv_row": number}
            for number, row in enumerate(csv.DictReader(stream), 2)
        }
    count = 0
    for record in records:
        row = quality[record["id"]]
        if row["protected_holdout"]:
            continue  # Their complete original records are verified separately.
        assert record["captions"] == [p["text"] for p in row["caption_provenance"]]
        for caption in row["caption_provenance"]:
            original = source[caption["source_caption_id"]]
            assert (
                original["modelId"] == record["id"]
                and original["category"].lower() == record["category"]
            )
            assert original["csv_row"] == caption["source_csv_row"]
            assert (
                " ".join(unicodedata.normalize("NFC", original["description"]).split())
                == caption["text"]
            )
            count += 1
    return count


def choose_reviews(index):
    pool = [r for r in index if r["split"] == "train" and r["legacy_split"] is None]
    pool.sort(key=lambda r: hashlib.sha256(f"review:42:{r['id']}".encode()).hexdigest())
    selected, seen = [], set()
    groups = [
        ("chair_armrests", "chair", 6, lambda p: p["armrests"] is True),
        ("chair_armless", "chair", 6, lambda p: p["armrests"] is False),
        ("chair_other", "chair", 4, lambda p: True),
        ("table_fourlegs", "table", 6, lambda p: p["leg_counts"] == [4]),
        (
            "table_round_base",
            "table",
            5,
            lambda p: any(w in p["text"].lower() for w in ("round", "pedestal", "oval")),
        ),
        (
            "table_storage",
            "table",
            5,
            lambda p: any(
                w in p["text"].lower() for w in ("drawer", "shelf", "shelves", "storage")
            ),
        ),
    ]
    for name, category, count, predicate in groups:
        added = 0
        for row in pool:
            captions = [p for p in row["caption_provenance"] if predicate(p)]
            if row["category"] != category or row["id"] in seen or not captions:
                continue
            caption = min(captions, key=lambda p: len(p["text"]))
            seen.add(row["id"])
            selected.append(
                {
                    "id": row["id"],
                    "category": category,
                    "group": name,
                    "caption": caption["text"],
                    "source_caption_id": caption["source_caption_id"],
                    "voxel": row["voxel"],
                    "all_retained_captions": row["captions"],
                    "visual_verdict": "pending",
                    "review_notes": "",
                    "rejected_attribute": "",
                }
            )
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError(f"Insufficient objects for targeted review group {name}")
    return selected


def annotate_sheet(source, rows, destination):
    original = Image.open(source).convert("RGB")
    canvas = Image.new("RGB", (1760, 1040), "white")
    for i, row in enumerate(rows):
        x, y = i % 4 * 440, i // 4 * 520
        tile = original.crop((x, i // 4 * 380, x + 440, i // 4 * 380 + 380))
        canvas.paste(tile, (x, y))
        draw = ImageDraw.Draw(canvas)
        lines = textwrap.wrap(row["caption"], width=64)
        for line_number, line in enumerate(lines[:8]):
            draw.text((x + 10, y + 385 + 15 * line_number), line, fill="black")
        draw.text(
            (x + 10, y + 507),
            f"ID {row['id'][:12]} / caption {row['source_caption_id']}",
            fill="black",
        )
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Preserve existing recorded review decisions and images",
    )
    args = parser.parse_args()
    document = json.loads(MANIFEST.read_text())
    records = document["records"]
    index = [json.loads(line) for line in (DATA / "quality-index.jsonl").read_text().splitlines()]
    quality = {r["id"]: r for r in index}
    verified_captions = verify_caption_provenance(
        records, quality, Path("E:/STLModelData/text2shape-v1/raw/captions.tablechair.csv")
    )
    assert len(quality) == len(records) == len({r["id"] for r in records})
    geometry = {}
    split_counts = {}
    for split in ("train", "val", "test"):
        dataset = VoxelDataset(MANIFEST, split)
        split_counts[split] = len(dataset)
        for record in dataset.records:
            grid = read_voxels(MANIFEST.parent / record["voxel"])
            digest = hashlib.sha256(grid.tobytes()).hexdigest()
            assert digest == quality[record["id"]]["occupancy_sha256"]
            assert digest not in geometry, "Exact duplicate geometry in prepared dataset"
            geometry[digest] = record["id"]
    old = json.loads(PILOT_MANIFEST.read_text())["records"]
    new = {r["id"]: r for r in records}
    holdouts = [r for r in old if r["split"] != "train"]
    assert len(holdouts) == 400
    assert all(new[r["id"]] == r for r in holdouts), "Original evaluation objects changed"
    pairs = json.loads((DATA / "near-duplicate-pairs.json").read_text())
    retained_pairs = [p for p in pairs if p["a"] in new and p["b"] in new]
    introduced_leaks = [
        p
        for p in retained_pairs
        if new[p["a"]]["split"] != new[p["b"]]["split"]
        and not (quality[p["a"]]["protected_holdout"] and quality[p["b"]]["protected_holdout"])
    ]
    assert not introduced_leaks
    protected = json.loads((DATA / "protected-sources.json").read_text())
    assert all(file_hash(Path(path)) == expected for path, expected in protected.items())
    selected = (
        json.loads((TRACKED / "review-objects.json").read_text())
        if args.verify_only
        else choose_reviews(index)
    )
    render_spec = importlib.util.spec_from_file_location(
        "furniture_review_renderer", ROOT / "experiments/procedural-v1/render_review.py"
    )
    renderer = importlib.util.module_from_spec(render_spec)
    render_spec.loader.exec_module(renderer)
    mesh_checks = []
    for number, row in enumerate(selected):
        if args.verify_only:
            row.setdefault("captions_before_review", row["all_retained_captions"])
            row["all_retained_captions"] = quality.get(row["id"], {}).get("captions", [])
            row["removed_from_dataset"] = row["id"] not in new
        row["review_number"] = number + 1
        path = DATA / "review" / f"{number + 1:02d}-{row['category']}-{row['id']}.stl"
        report = (
            row["mesh"]
            if args.verify_only
            else export_stl(read_voxels(MANIFEST.parent / row["voxel"]), path)
        )
        mesh = trimesh.load_mesh(path)
        assert np.isfinite(mesh.vertices).all() and np.isclose(mesh.extents.max(), 100, atol=0.001)
        assert np.isclose(mesh.bounds[0, 2], 0, atol=0.001)
        row["mesh"] = report
        mesh_checks.append(
            {
                "id": row["id"],
                "watertight": bool(mesh.is_watertight),
                "components": len(mesh.split(only_watertight=False)),
            }
        )
    for start in () if args.verify_only else range(0, len(selected), 8):
        page = start // 8 + 1
        rows = selected[start : start + 8]
        paths = [Path(row["mesh"]["path"]) for row in rows]
        labels = [f"{row['review_number']:02d} / {row['group']}" for row in rows]
        for view, front in (("front", True), ("back", False)):
            raw_image = DATA / "review" / f"page-{page}-{view}.png"
            renderer.render(paths, labels, raw_image, 4, front=front)
            annotate_sheet(raw_image, rows, TRACKED / f"review-{page}-{view}.png")
    write_json(DATA / "review/objects.json", selected)
    write_json(TRACKED / "review-objects.json", selected)
    if not args.verify_only:
        with (TRACKED / "caption-review.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "review_number",
                    "id",
                    "category",
                    "source_caption_id",
                    "caption",
                    "visual_verdict",
                    "review_notes",
                    "rejected_attribute",
                ],
            )
            writer.writeheader()
            for row in selected:
                writer.writerow({key: row[key] for key in writer.fieldnames})
    verification = {
        "verified_objects": len(records),
        "split_objects": split_counts,
        "all_voxel_file_hashes_verified": True,
        "all_occupancy_hashes_verified": True,
        "source_caption_object_pairings_verified": verified_captions,
        "exact_duplicate_geometry": False,
        "preserved_original_holdouts": 400,
        "original_holdout_records_identical": True,
        "introduced_cross_split_near_duplicates_in_checked_pairs": 0,
        "remaining_original_holdout_near_duplicate_pairs": len(retained_pairs),
        "near_duplicate_search_exhaustive": False,
        "protected_artifacts_unchanged": True,
        "visual_review_objects": len(selected),
        "visual_review_groups": dict(Counter(r["group"] for r in selected)),
        "visual_review_verdicts": dict(Counter(r["visual_verdict"] for r in selected)),
        "review_meshes": mesh_checks,
    }
    write_json(DATA / "verification.json", verification)
    write_json(TRACKED / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
