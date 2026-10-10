"""Expand local Text2Shape furniture with explicit quality and leakage checks."""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from stlmodel.common import file_hash, write_json
from stlmodel.data import read_voxels

PROJECT = Path(__file__).resolve().parents[2]
PILOT = Path("E:/STLModelData/text2shape-v1")
OUTPUT = Path("E:/STLModelData/furniture-data-v2")
TRACKED = Path(__file__).resolve().parent
CSV = PILOT / "raw/captions.tablechair.csv"
RAW = PILOT / "raw/voxels"
MANIFEST = PILOT / "prepared/manifest.json"
GEOMETRY_WORDS = re.compile(
    r"\b(back\w*|arm\w*|armless|seat\w*|leg\w*|base|pedestal|support\w*|cushion\w*|"
    r"round\w*|rectang\w*|square|oval|circul\w*|triang\w*|curv\w*|straight|flat|"
    r"tall|short|high|low|thin|thick|wide|narrow|long|small|large|slender|"
    r"drawer\w*|shel\w*|storage|fold\w*|swivel\w*|rock\w*|bench|stool|sofa|"
    r"slat\w*|bar\w*|rail\w*|panel\w*|hole\w*|opening\w*|box\w*|cross\w*|"
    r"wheel\w*|caster\w*|taper\w*|tubular|tripod|hexagon\w*|trapez\w*)\b",
    re.IGNORECASE,
)
NEGATIVE_ARMS = re.compile(
    r"\b(armless|(?:no|without|with\s+out|lacks|lack of|does not have|doesn't have|hasn't got)\s+"
    r"(?:(?:any|side|supporting|both|two)\s+){0,3}arm(?:[\s-]*rest)?s?)\b",
    re.IGNORECASE,
)
ARMS = re.compile(r"\b(armrests?|arm rests?|arms|armchair)\b", re.IGNORECASE)
LEG_COUNTS = re.compile(
    r"\b(one|two|three|four|five|six|[1-6])(?:\s*-\s*|\s+)(?:(?:short|long|thin|thick|tall|"
    r"wooden|metal|black|white|brown|curved|straight)\s+){0,2}leg(?:s|ged)?\b",
    re.IGNORECASE,
)
NUMBERS = dict(zip(("one", "two", "three", "four", "five", "six"), range(1, 7)))

spec = importlib.util.spec_from_file_location(
    "pilot_orientation", PROJECT / "experiments/text2shape-pilot-v1/run_pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def caption_attributes(text):
    arms = False if NEGATIVE_ARMS.search(text) else True if ARMS.search(text) else None
    counts = {
        NUMBERS.get(match.lower(), int(match) if match.isdigit() else 0)
        for match in LEG_COUNTS.findall(text)
    }
    return {"armrests": arms, "leg_counts": sorted(counts)}


def caption_reason(text, category):
    words = re.findall(r"[a-z]+", text.lower())
    if not 4 <= len(words) <= 120:
        return "caption_length"
    if len(set(words)) / len(words) < 0.35 or re.search(r"(.)\1{5,}", text):
        return "repetitive_caption"
    own = (
        r"\b(chairs?|armchairs?|stools?|benches?|sofas?|seats?)\b"
        if category == "chair"
        else (r"\b(tables?|desks?|stands?|counters?|consoles?|nightstands?)\b")
    )
    other = r"\b(tables?|desks?)\b" if category == "chair" else r"\b(chairs?|armchairs?|stools?)\b"
    if re.search(other, text, re.IGNORECASE) and not re.search(own, text, re.IGNORECASE):
        return "opposite_category_caption"
    if not GEOMETRY_WORDS.search(text):
        return "no_geometry_description"
    return None


def screen_captions(rows, category):
    accepted, rejected, seen = [], [], set()
    for row in rows:
        text = " ".join(unicodedata.normalize("NFC", row["description"]).split())
        key = text.casefold()
        reason = "duplicate_caption" if key in seen else caption_reason(text, category)
        seen.add(key)
        item = {"text": text, "source_caption_id": row["id"], "source_csv_row": row["csv_row"]}
        if reason:
            rejected.append({**item, "reason": reason})
        else:
            accepted.append({**item, **caption_attributes(text)})
    arms = {r["armrests"] for r in accepted if r["armrests"] is not None}
    legs = {value for r in accepted for value in r["leg_counts"]}
    kept = []
    for item in accepted:
        conflict = (len(arms) > 1 and item["armrests"] is not None) or (
            len(legs) > 1 and item["leg_counts"]
        )
        if conflict:
            rejected.append({**item, "reason": "conflicting_attribute_captions"})
        else:
            kept.append(item)
    return kept, rejected


def geometry_quality(grid):
    points = np.argwhere(grid)
    extent = points.max(0) - points.min(0) + 1
    regions, count = label(grid, generate_binary_structure(3, 3))
    sizes = np.bincount(regions.ravel())[1:]
    total = int(grid.sum())
    largest = float(sizes.max() / total)
    metrics = {
        "occupied_voxels": total,
        "bbox_xyz": extent.tolist(),
        "components_26": count,
        "largest_component_fraction": largest,
    }
    reason = None
    if total < 64 or total > grid.size * 0.45:
        reason = "degenerate_occupancy"
    elif min(extent) < 3 or extent[1] < 6:
        reason = "collapsed_or_very_flat_geometry"
    elif largest < 0.98:
        reason = "disconnected_geometry"
    return metrics, reason


def protect_sources():
    paths = [CSV, MANIFEST, PILOT / "embeddings.pt", PILOT / "review/data-audit.json"]
    for name in (
        "text2shape-pilot-vae-v1",
        "text2shape-pilot-diffusion-v1",
        "text2shape-diffusion-high-noise-v1-conditioned",
        "text2shape-diffusion-high-noise-v1-velocity",
        "text2shape-diffusion-low-lr-v1-conditioned",
    ):
        paths.extend(sorted((PROJECT / "runs" / name).glob("*.pt")))
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths.extend(MANIFEST.parent / r["voxel"] for r in document["records"] if r["split"] != "train")
    return {str(path): file_hash(path) for path in paths}


def choose_representatives(records, near_pairs):
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for a, b, _ in near_pairs:
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    retained, excluded = [], []
    for group in groups.values():
        locked = [i for i in group if records[i].get("legacy_split") in ("val", "test")]
        winners = locked or [
            min(group, key=lambda i: (records[i].get("legacy_split") != "train", records[i]["id"]))
        ]
        retained.extend(winners)
        for i in group:
            if i not in winners:
                excluded.append(
                    {
                        "id": records[i]["id"],
                        "reason": "near_duplicate_geometry",
                        "retained_ids": [records[j]["id"] for j in winners],
                    }
                )
    return sorted(retained), excluded


def near_duplicates(grids, device):
    """Shortlist by pooled occupancy cosine, confirm using full-resolution IoU."""
    packed = np.packbits(grids.reshape(len(grids), -1), axis=1)
    features = grids.reshape(-1, 8, 4, 8, 4, 8, 4).mean(axis=(2, 4, 6)).reshape(-1, 512)
    vectors = torch.from_numpy(features.astype(np.float32)).to(device)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    bitcounts = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    pairs, checked = [], set()
    with torch.inference_mode():
        for start in range(0, len(grids), 256):
            similarities = vectors[start : start + 256] @ vectors.T
            candidates = similarities.topk(min(17, len(grids)), dim=1).indices.cpu().numpy()
            for offset, neighbors in enumerate(candidates):
                a = start + offset
                for b in neighbors:
                    b = int(b)
                    if a == b:
                        continue
                    pair = min(a, b), max(a, b)
                    if pair in checked:
                        continue
                    checked.add(pair)
                    intersection = int(bitcounts[np.bitwise_and(packed[a], packed[b])].sum())
                    union = int(bitcounts[np.bitwise_or(packed[a], packed[b])].sum())
                    iou = intersection / union
                    if iou >= 0.98:
                        pairs.append((*pair, iou))
            if start % 2048 == 0:
                print(
                    {"stage": "near_duplicates", "checked_objects": start, "total": len(grids)},
                    flush=True,
                )
    return pairs, len(checked)


def assign_expanded_splits(records, seed=42):
    for category in ("chair", "table"):
        added = [r for r in records if r["category"] == category and not r.get("legacy_split")]
        added.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        holdout = int(len(added) * 0.1)
        for index, row in enumerate(added):
            row["split"] = "val" if index < holdout else "test" if index < holdout * 2 else "train"
    for row in records:
        if row.get("legacy_split"):
            row["split"] = row["legacy_split"]


def prepare(out, device):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output must be empty; existing datasets are preserved: {out}")
    protected = protect_sources()
    audit_before = json.loads((PILOT / "review/data-audit.json").read_text())
    if protected[str(CSV)] != audit_before["download_sha256"][CSV.name]:
        raise ValueError("Source caption CSV differs from the audited original")
    old = {r["id"]: r for r in json.loads(MANIFEST.read_text())["records"]}
    objects = {}
    with CSV.open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            row["csv_row"] = number
            identifier, category = row["modelId"].strip(), row["category"].strip().lower()
            if not identifier or category not in ("chair", "table"):
                raise ValueError(f"Unsupported source category or ID at row {number}")
            obj = objects.setdefault(
                identifier, {"id": identifier, "category": category, "rows": []}
            )
            if obj["category"] != category:
                raise ValueError(f"Conflicting source category for {identifier}")
            obj["rows"].append(row)
    paths = defaultdict(list)
    for path in RAW.rglob("*.nrrd"):
        if "__MACOSX" not in path.parts and not path.name.startswith("._"):
            paths[path.stem].append(path)
    out.mkdir(parents=True)
    write_json(out / "protected-sources.json", protected)
    ordered = sorted(
        objects.values(),
        key=lambda r: (
            0
            if r["id"] in old and old[r["id"]]["split"] != "train"
            else 1
            if r["id"] in old
            else 2,
            r["id"],
        ),
    )
    records, grids, excluded, rejected, seen = [], [], [], [], {}
    last_print = time.monotonic()
    for index, obj in enumerate(ordered, 1):
        identifier, category = obj["id"], obj["category"]
        legacy = old.get(identifier)
        locked = legacy is not None and legacy["split"] != "train"
        captions, bad = screen_captions(obj["rows"], category)
        rejected.extend({"object_id": identifier, **r} for r in bad)
        try:
            if len(paths[identifier]) != 1:
                raise ValueError(f"Expected one source grid, found {len(paths[identifier])}")
            grid, _ = pilot.canonical_voxels(paths[identifier][0])
            metrics, reason = geometry_quality(grid)
        except (ValueError, OSError, pilot.nrrd.NRRDError) as exc:
            if locked:
                raise ValueError(f"Protected holdout cannot be read: {identifier}") from exc
            excluded.append({"id": identifier, "reason": "unreadable_geometry", "detail": str(exc)})
            continue
        if locked:
            original = read_voxels(MANIFEST.parent / legacy["voxel"])
            if not np.array_equal(grid, original):
                raise ValueError(f"Source geometry changed for protected holdout {identifier}")
        elif reason or not captions:
            excluded.append(
                {
                    "id": identifier,
                    "category": category,
                    "reason": reason or "no_usable_geometry_caption",
                    "quality": metrics,
                }
            )
            continue
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        if digest in seen:
            if locked:
                raise ValueError("Original protected holdouts contain exact duplicate geometry")
            excluded.append(
                {
                    "id": identifier,
                    "reason": "exact_duplicate_geometry",
                    "retained_id": seen[digest],
                }
            )
            continue
        seen[digest] = identifier
        records.append(
            {
                "id": identifier,
                "category": category,
                "captions": legacy["captions"] if locked else [r["text"] for r in captions],
                "caption_provenance": captions,
                "subtypes": sorted({r["subSynsetId"] for r in obj["rows"]}),
                "legacy_split": legacy["split"] if legacy else None,
                "protected_holdout": locked,
                "quality": metrics,
                "occupancy_sha256": digest,
                "source_voxel": str(paths[identifier][0]),
            }
        )
        grids.append(grid)
        if time.monotonic() - last_print > 20:
            print(
                {
                    "stage": "audit",
                    "checked": index,
                    "total": len(ordered),
                    "accepted": len(records),
                    "excluded": len(excluded),
                },
                flush=True,
            )
            last_print = time.monotonic()
    all_grids = np.stack(grids)
    del grids
    pairs, pair_count = near_duplicates(all_grids, device)
    retained, near_excluded = choose_representatives(records, pairs)
    excluded.extend(near_excluded)
    selected = [records[i] for i in retained]
    assign_expanded_splits(selected)
    prepared = out / "prepared"
    (prepared / "voxels").mkdir(parents=True)
    manifest_records = []
    quality_records = []
    for i in retained:
        row = records[i]
        if row["protected_holdout"]:
            record = dict(old[row["id"]])
            shutil.copy2(MANIFEST.parent / record["voxel"], prepared / record["voxel"])
        else:
            voxel = "voxels/" + hashlib.sha256(row["id"].encode()).hexdigest()[:24] + ".npz"
            np.savez_compressed(prepared / voxel, occupancy=all_grids[i])
            record = {key: row[key] for key in ("id", "category", "captions", "split")}
            record.update(voxel=voxel, sha256=file_hash(prepared / voxel))
        manifest_records.append(record)
        quality_records.append({**row, "voxel": record["voxel"], "sha256": record["sha256"]})
    manifest_records.sort(key=lambda r: r["id"])
    write_json(
        prepared / "manifest.json",
        {
            "version": 1,
            "resolution": 32,
            "seed": 42,
            "source": "Text2Shape human captions and solid voxels; screened expansion; preserved pilot holdouts",
            "records": manifest_records,
        },
    )
    with (out / "quality-index.jsonl").open("w", encoding="utf-8") as stream:
        for row in quality_records:
            stream.write(json.dumps(row) + "\n")
    with (out / "rejected-captions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rejected:
            stream.write(json.dumps(row) + "\n")
    write_json(out / "excluded-objects.json", excluded)
    write_json(
        out / "near-duplicate-pairs.json",
        [{"a": records[a]["id"], "b": records[b]["id"], "voxel_iou": iou} for a, b, iou in pairs],
    )
    training = [r for r in selected if r["split"] == "train"]
    summary = {
        "source": "https://svl.stanford.edu/projects/text2shape/",
        "source_caption_sha256": protected[str(CSV)],
        "seed": 42,
        "source_objects": len(objects),
        "source_caption_rows": sum(len(r["rows"]) for r in objects.values()),
        "selected_objects": len(selected),
        "selected_captions": sum(len(r["captions"]) for r in selected),
        "split_objects": dict(Counter(r["split"] for r in selected)),
        "split_captions": {
            s: sum(len(r["captions"]) for r in selected if r["split"] == s)
            for s in ("train", "val", "test")
        },
        "split_categories": {
            s: dict(Counter(r["category"] for r in selected if r["split"] == s))
            for s in ("train", "val", "test")
        },
        "new_training_objects": sum(r["legacy_split"] is None for r in training),
        "retained_pilot_training_objects": sum(r["legacy_split"] == "train" for r in training),
        "preserved_baseline_holdouts": sum(r["protected_holdout"] for r in selected),
        "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
        "caption_rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "training_subtype_counts": dict(Counter(t for r in training for t in r["subtypes"])),
        "training_attribute_caption_counts": dict(
            Counter(
                "with_armrests"
                if p["armrests"] is True
                else "without_armrests"
                if p["armrests"] is False
                else "unspecified_armrests"
                for r in training
                for p in r["caption_provenance"]
            )
        ),
        "near_duplicate_method": "16 nearest neighbors by pooled 8-cubed occupancy cosine; confirm full-resolution IoU >= 0.98; connected-group deduplication",
        "near_duplicate_checked_pairs": pair_count,
        "near_duplicate_confirmed_pairs": len(pairs),
        "near_duplicate_search_is_exhaustive": False,
        "geometry_filters": {
            "minimum_occupied_voxels": 64,
            "maximum_occupied_fraction": 0.45,
            "minimum_bbox_extent": 3,
            "minimum_vertical_bbox_extent": 6,
            "minimum_largest_component_fraction": 0.98,
            "connectivity": 26,
        },
        "protected_holdout_filter_exemption": True,
        "caption_accuracy": "Source-human captions screened for quality and cross-caption conflicts; not every retained caption is visually verified",
        "manifest_sha256": file_hash(prepared / "manifest.json"),
        "model_training_performed": False,
        "visual_review": "pending",
    }
    if protect_sources() != protected:
        raise ValueError("A protected artifact changed during preparation")
    summary["protected_artifacts_unchanged"] = True
    write_json(out / "summary.json", summary)
    write_json(TRACKED / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.out, args.device)


if __name__ == "__main__":
    main()
