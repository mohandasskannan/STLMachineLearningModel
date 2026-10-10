"""Record geometry-only visual review and quarantine unsupported source captions."""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import caption_attributes, screen_captions

from stlmodel.common import file_hash, write_json

DATA = Path("E:/STLModelData/furniture-data-v2")
TRACKED = Path(__file__).resolve().parent


def repeats_unsupported_claim(caption, selected, attribute):
    return (
        caption["source_caption_id"] == selected["source_caption_id"]
        or attribute == "armrests"
        and selected["armrests"] is not None
        and caption["armrests"] == selected["armrests"]
        or attribute == "leg_counts"
        and bool(set(selected["leg_counts"]) & set(caption["leg_counts"]))
        or attribute == "drawers"
        and bool(re.search(r"\bdrawers?\b", caption["text"], re.IGNORECASE))
    )


def main():
    if (DATA / "review/curation.json").exists():
        raise ValueError("Reviews are already finalized; preserve the recorded curation ledger")
    review_path = TRACKED / "caption-review.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    allowed = {"geometry_consistent", "caption_mismatch", "unverifiable"}
    if len(reviews) != 32 or any(
        r["visual_verdict"] not in allowed or not r["review_notes"] for r in reviews
    ):
        raise ValueError("All 32 review entries need an explicit verdict and observation")
    if any(
        r["visual_verdict"] != "geometry_consistent"
        and r.get("rejected_attribute") not in ("armrests", "leg_counts", "drawers", "caption")
        for r in reviews
    ):
        raise ValueError("Flagged captions need the specific unsupported attribute")
    manifest = DATA / "prepared/manifest.json"
    document = json.loads(manifest.read_text())
    quality_path = DATA / "quality-index.jsonl"
    index = [json.loads(line) for line in quality_path.read_text().splitlines()]
    quality = {r["id"]: r for r in index}
    original_hash = file_hash(manifest)
    protected = json.loads((DATA / "protected-sources.json").read_text())
    if not all(file_hash(Path(path)) == expected for path, expected in protected.items()):
        raise ValueError("Protected source fingerprints changed before curation")
    original_selected = {
        review["id"]: next(
            p
            for p in quality[review["id"]]["caption_provenance"]
            if p["source_caption_id"] == review["source_caption_id"]
        )
        for review in reviews
    }
    raw = defaultdict(list)
    source = Path("E:/STLModelData/text2shape-v1/raw/captions.tablechair.csv")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            row["csv_row"] = number
            raw[row["modelId"]].append(row)
    rejected = []
    refreshed = {}
    for identifier, rows in raw.items():
        accepted, bad = screen_captions(rows, rows[0]["category"].lower())
        refreshed[identifier] = accepted
        rejected.extend({"object_id": identifier, **r} for r in bad)
    for row in index:
        if not row["protected_holdout"]:
            row["caption_provenance"] = refreshed[row["id"]]
            row["captions"] = [p["text"] for p in row["caption_provenance"]]
    visually_rejected = []
    resolved_conflicts = 0
    for review in reviews:
        row = quality[review["id"]]
        if row["protected_holdout"] or row["split"] != "train":
            raise ValueError("Visual caption curation is limited to added training objects")
        if review["visual_verdict"] == "geometry_consistent":
            accepted = next(
                (
                    p
                    for p in row["caption_provenance"]
                    if p["source_caption_id"] == review["source_caption_id"]
                ),
                None,
            )
            if accepted is None:
                original = original_selected[review["id"]]
                # A recorded geometry review can resolve a source-caption conflict
                # for this specific caption; other conflicting claims stay excluded.
                accepted = {**original, **caption_attributes(original["text"])}
                resolved_conflicts += 1
                row["caption_provenance"].append(accepted)
                row["captions"].append(accepted["text"])
                rejected = [
                    r for r in rejected if r["source_caption_id"] != accepted["source_caption_id"]
                ]
            accepted["visual_geometry_reviewed"] = True
            continue
        selected = original_selected[review["id"]]

        # When an explicit armrest or leg-count claim fails visual review, also
        # quarantine other captions repeating that same unsupported claim.
        def unsupported(caption, selected=selected, attribute=review["rejected_attribute"]):
            return repeats_unsupported_claim(caption, selected, attribute)

        removed = [p for p in row["caption_provenance"] if unsupported(p)]
        if not any(p["source_caption_id"] == selected["source_caption_id"] for p in removed):
            removed.append(selected)
        row["caption_provenance"] = [p for p in row["caption_provenance"] if not unsupported(p)]
        row["captions"] = [p["text"] for p in row["caption_provenance"]]
        visually_rejected.extend(
            {
                "object_id": row["id"],
                **p,
                "reason": "visual_" + review["visual_verdict"],
                "notes": review["review_notes"],
            }
            for p in removed
        )
    removed_objects = []
    kept = []
    for record in document["records"]:
        row = quality[record["id"]]
        if not row["captions"]:
            removed_objects.append(record["id"])
        else:
            kept.append({**record, "captions": row["captions"]})
    if rejected or visually_rejected:
        write_json(DATA / "manifest-before-visual-review.json", document)
        document["records"] = kept
        write_json(manifest, document)
        index = [row for row in index if row["id"] not in removed_objects]
        with quality_path.open("w", encoding="utf-8") as stream:
            for row in index:
                stream.write(json.dumps(row) + "\n")
        visual_ids = {r["source_caption_id"] for r in visually_rejected}
        rejected = [
            r for r in rejected if r["source_caption_id"] not in visual_ids
        ] + visually_rejected
        with (DATA / "rejected-captions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rejected:
                stream.write(json.dumps(row) + "\n")
    review_objects = json.loads((TRACKED / "review-objects.json").read_text())
    for row, reviewed in zip(review_objects, reviews, strict=True):
        assert row["id"] == reviewed["id"]
        row["visual_verdict"] = reviewed["visual_verdict"]
        row["review_notes"] = reviewed["review_notes"]
        row["rejected_attribute"] = reviewed["rejected_attribute"]
        row["captions_before_review"] = row["all_retained_captions"]
        row["all_retained_captions"] = quality[row["id"]]["captions"]
        row["removed_from_dataset"] = row["id"] in removed_objects
        if row["group"] == "chair_armrests" and "with out any side arms" in row["caption"].lower():
            row["group"] = "chair_armless"
        row["quarantined_from_training"] = reviewed["visual_verdict"] != "geometry_consistent"
    write_json(TRACKED / "review-objects.json", review_objects)
    write_json(DATA / "review/objects.json", review_objects)
    shutil_path = DATA / "review/caption-review.csv"
    shutil_path.write_bytes(review_path.read_bytes())
    training = [row for row in index if row["split"] == "train"]
    summary = json.loads((DATA / "summary.json").read_text())
    summary.update(
        selected_objects=len(index),
        selected_captions=sum(len(r["captions"]) for r in index),
        split_objects=dict(Counter(r["split"] for r in index)),
        split_captions={
            s: sum(len(r["captions"]) for r in index if r["split"] == s)
            for s in ("train", "val", "test")
        },
        split_categories={
            s: dict(Counter(r["category"] for r in index if r["split"] == s))
            for s in ("train", "val", "test")
        },
        new_training_objects=sum(r["legacy_split"] is None for r in training),
        retained_pilot_training_objects=sum(r["legacy_split"] == "train" for r in training),
        training_subtype_counts=dict(Counter(t for r in training for t in r["subtypes"])),
        training_attribute_caption_counts=dict(
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
        manifest_sha256=file_hash(manifest),
        visual_review="completed_with_flagged_captions_quarantined",
        visual_review_counts=dict(Counter(r["visual_verdict"] for r in reviews)),
        visually_quarantined_captions=len(visually_rejected),
        visually_removed_objects=len(removed_objects),
        caption_accuracy="Human source captions with automated screening and 32 geometry-only visual spot checks; flagged claims quarantined; full corpus not exhaustively verified",
    )
    summary["caption_rejection_reasons"] = dict(Counter(r["reason"] for r in rejected))
    summary["caption_screening_refreshed_after_visual_review"] = True
    summary["visually_resolved_source_conflicts"] = resolved_conflicts
    summary["visually_checked_retained_captions"] = sum(
        bool(p.get("visual_geometry_reviewed")) for r in index for p in r["caption_provenance"]
    )
    if removed_objects:
        exclusions = json.loads((DATA / "excluded-objects.json").read_text())
        exclusions.extend(
            {"id": identifier, "reason": "no_captions_after_visual_review"}
            for identifier in removed_objects
        )
        write_json(DATA / "excluded-objects.json", exclusions)
        counts = Counter(summary["exclusion_reasons"])
        counts["no_captions_after_visual_review"] += len(removed_objects)
        summary["exclusion_reasons"] = dict(counts)
    write_json(DATA / "summary.json", summary)
    write_json(TRACKED / "summary.json", summary)
    write_json(
        DATA / "review/curation.json",
        {
            "original_manifest_sha256": original_hash,
            "curated_manifest_sha256": summary["manifest_sha256"],
            "review_counts": summary["visual_review_counts"],
            "removed_captions": visually_rejected,
            "removed_objects": removed_objects,
        },
    )
    print(
        json.dumps(
            {
                "review_counts": summary["visual_review_counts"],
                "quarantined_captions": len(visually_rejected),
                "removed_objects": len(removed_objects),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
