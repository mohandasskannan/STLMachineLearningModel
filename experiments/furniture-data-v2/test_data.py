"""Quality screening and holdout-preservation regressions for the expansion."""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "furniture_data_v2", Path(__file__).with_name("prepare_data.py")
)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


def rows(*texts):
    return [{"id": str(i), "csv_row": i + 2, "description": text} for i, text in enumerate(texts)]


def test_screen_preserves_source_words_and_rejects_conflicts():
    accepted, rejected = data.screen_captions(
        rows(
            "A chair with armrests and four legs.",
            "A chair without arm rests and four legs.",
            "  A tall chair  with a curved back.  ",
            "a tall chair with a curved back.",
            "A brown wooden chair.",
        ),
        "chair",
    )
    assert [r["text"] for r in accepted] == ["A tall chair with a curved back."]
    assert {r["reason"] for r in rejected} == {
        "duplicate_caption",
        "conflicting_attribute_captions",
        "no_geometry_description",
    }


def test_leg_conflicts_do_not_select_unverified_majority():
    accepted, rejected = data.screen_captions(
        rows(
            "A table with four legs.", "A table with three legs.", "A round table with a flat top."
        ),
        "table",
    )
    assert [r["text"] for r in accepted] == ["A round table with a flat top."]
    assert len(rejected) == 2


@pytest.mark.parametrize(
    "text,category,reason",
    (
        ("A rectangular desk with drawers.", "chair", "opposite_category_caption"),
        ("A tall chair with four legs.", "table", "opposite_category_caption"),
        ("chair", "chair", "caption_length"),
        ("chair chair chair chair chair chair chair chair", "chair", "repetitive_caption"),
        ("A grey wooden table.", "table", "no_geometry_description"),
    ),
)
def test_caption_quality_rules(text, category, reason):
    assert data.caption_reason(text, category) == reason


def test_geometry_rejects_fragments_without_modifying_shape():
    grid = np.zeros((32, 32, 32), dtype=np.uint8)
    grid[2:10, 2:10, 2:10] = 1
    original = grid.copy()
    _, reason = data.geometry_quality(grid)
    assert reason is None
    assert np.array_equal(grid, original)
    grid[20:24, 20:24, 20:24] = 1
    metrics, reason = data.geometry_quality(grid)
    assert reason == "disconnected_geometry"
    assert metrics["components_26"] == 2
    assert grid[20:24, 20:24, 20:24].all()


def test_near_duplicate_group_preserves_both_original_holdouts():
    records = [
        {"id": "val", "legacy_split": "val"},
        {"id": "test", "legacy_split": "test"},
        {"id": "train", "legacy_split": "train"},
        {"id": "new", "legacy_split": None},
    ]
    retained, excluded = data.choose_representatives(
        records, [(0, 2, 0.99), (2, 3, 0.99), (3, 1, 0.99)]
    )
    assert retained == [0, 1]
    assert {r["id"] for r in excluded} == {"train", "new"}


def test_split_expansion_does_not_reshuffle_old_objects():
    records = [{"id": str(i), "category": "chair", "legacy_split": None} for i in range(30)]
    records += [
        {"id": "old-" + split, "category": "chair", "legacy_split": split}
        for split in ("train", "val", "test")
    ]
    data.assign_expanded_splits(records)
    old = {r["id"]: r["split"] for r in records if r["legacy_split"]}
    assert old == {"old-train": "train", "old-val": "val", "old-test": "test"}
    expected = {r["id"]: r["split"] for r in records}
    records.reverse()
    data.assign_expanded_splits(records)
    assert expected == {r["id"]: r["split"] for r in records}


def test_shortlist_confirms_full_resolution_duplicates():
    grid = np.zeros((32, 32, 32), dtype=np.uint8)
    grid[2:20, 3:24, 4:25] = 1
    other = np.zeros_like(grid)
    other[10:24, 10:26, 10:26] = 1
    pairs, checked = data.near_duplicates(np.stack((grid, grid, other)), "cpu")
    assert pairs == [(0, 1, 1.0)]
    assert checked == 3


def test_visual_quarantine_only_repeats_the_failed_attribute():
    review_spec = importlib.util.spec_from_file_location(
        "furniture_finalizer", Path(__file__).with_name("finalize_reviews.py")
    )
    finalizer = importlib.util.module_from_spec(review_spec)
    review_spec.loader.exec_module(finalizer)
    selected = {"source_caption_id": "a", "armrests": True, "leg_counts": [4]}
    good_four_legs = {"source_caption_id": "b", "armrests": None, "leg_counts": [4]}
    repeated_arms = {"source_caption_id": "c", "armrests": True, "leg_counts": []}
    assert finalizer.repeats_unsupported_claim(selected, selected, "armrests")
    assert not finalizer.repeats_unsupported_claim(good_four_legs, selected, "armrests")
    assert finalizer.repeats_unsupported_claim(repeated_arms, selected, "armrests")
    assert finalizer.repeats_unsupported_claim(good_four_legs, selected, "leg_counts")
    assert not finalizer.repeats_unsupported_claim(repeated_arms, selected, "leg_counts")


@pytest.mark.parametrize(
    "text,arms,legs",
    (
        ("Curve chair sofa with out any side arms.", False, []),
        ("A chair without any arm rests.", False, []),
        ("A chair with four-legged support.", None, [4]),
        ("A contemporary 3 legged round table.", None, [3]),
    ),
)
def test_source_wording_variants(text, arms, legs):
    attributes = data.caption_attributes(text)
    assert attributes == {"armrests": arms, "leg_counts": legs}


def test_caption_provenance_detects_wrong_object_pairing(tmp_path):
    review_spec = importlib.util.spec_from_file_location(
        "furniture_verifier", Path(__file__).with_name("verify_and_review.py")
    )
    verifier = importlib.util.module_from_spec(review_spec)
    review_spec.loader.exec_module(verifier)
    source = tmp_path / "captions.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "modelId", "category", "description"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "c",
                "modelId": "shape",
                "category": "Chair",
                "description": "A  chair with armrests.",
            }
        )
    record = {"id": "shape", "category": "chair", "captions": ["A chair with armrests."]}
    provenance = {"text": record["captions"][0], "source_caption_id": "c", "source_csv_row": 2}
    quality = {"shape": {"protected_holdout": False, "caption_provenance": [provenance]}}
    assert verifier.verify_caption_provenance([record], quality, source) == 1
    with pytest.raises(AssertionError):
        verifier.verify_caption_provenance(
            [{**record, "id": "wrong"}], {"wrong": quality["shape"]}, source
        )
