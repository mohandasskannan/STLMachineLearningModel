# Screened furniture expansion

This dataset expands the existing Text2Shape chair/table pilot using the complete
source already downloaded on E:. It adds real source shapes and human-written
descriptions; it does not invent captions or train a model. The upstream
[Text2Shape project](https://svl.stanford.edu/projects/text2shape/) supplies the
caption CSV and solid voxelizations. Source attribution and applicable dataset
terms continue to apply to the expanded copy.

## Preparation

```powershell
.venv/Scripts/python.exe experiments/furniture-data-v2/prepare_data.py
.venv/Scripts/python.exe experiments/furniture-data-v2/verify_and_review.py
```

After recording all 32 visual decisions in `caption-review.csv`, run
`finalize_reviews.py`, then `verify_and_review.py --verify-only`. Finalization
refreshes source-caption screening, applies the recorded claim-specific
quarantines, and writes a curation ledger. It refuses repeated finalization.
The verifier's `--verify-only` mode preserves existing decisions and images.

Prepared data is written to `E:/STLModelData/furniture-data-v2/prepared/manifest.json`.
The preparer refuses a nonempty output directory. Full quality/provenance indexes,
caption rejections, object exclusions, confirmed near-duplicate pairs, protected
hashes, and mesh reviews remain on E:. This folder packages summaries and reviews.

Source NRRD coordinates are normalized into world XYZ, with Y up, using the
existing audited signed-axis permutation. RGBA alpha defines occupied voxels.
The 32-cubed resolution matches the existing decoder. All shape components are
preserved: the pipeline excludes severely fragmented objects rather than
silently removing inconvenient parts.

New training and evaluation objects must contain at least 64 occupied voxels,
occupy no more than 45% of the grid, span at least three voxels in each axis and
six vertically, and place at least 98% of occupied voxels in their largest
26-connected component. These thresholds screen gross defects; they do not prove
manifold mesh topology or printability and may exclude some legitimate thin parts.

Caption screening normalizes Unicode and whitespace without rewriting source
claims. It removes duplicate, very short/long, repetitive, category-mismatched,
and material/color-only descriptions with no recognized geometry information.
If source descriptions disagree about armrests or leg counts, all captions
asserting the conflicting attribute are quarantined. The pipeline does not
choose a majority claim as ground truth. Source caption IDs and CSV row numbers
are recorded. Mixed shape/color/material captions remain; monochrome occupancy
cannot verify color or material claims.

## Split and duplication policy

The original 200 validation and 200 test objects retain their exact record,
caption, split, and compressed voxel-file contents. They are explicitly exempt
from new filters to preserve the baseline benchmark. Screened original training
objects stay in training. Only added objects receive new deterministic,
category-stratified splits: 80% train, 10% validation, 10% test.

Exact canonical occupancy duplicates are excluded. A second search shortlists
16 neighbors per object by cosine similarity of pooled 8-cubed occupancy, then
confirms near duplicates with full-resolution voxel IoU of at least 0.98. Groups
keep an original holdout when present, otherwise prefer surviving pilot training
objects; new aliases are removed before split assignment. This is a practical
shortlist search, not an exhaustive guarantee against all near duplicates.
Any existing near-duplicate relation between protected holdouts is documented
rather than altering their established benchmark.

## Caption review and next training

Thirty-two added training objects are sampled deterministically for geometry
review: chairs with/without armrests and other chairs, plus four-leg tables,
round/base variants, and storage furniture. Each has front/back target renders,
its source caption, and a review entry. The review addresses represented shape
attributes; text screening alone cannot establish factual caption accuracy.
The broader corpus is source-human and screened, not exhaustively verified.

The existing VAE, embedding cache, model weights, and local playground remain
preserved. This task does not produce replacement weights or overwrite the
original cache. A later diffusion experiment must create a separate embedding
cache and explicitly choose its normalization before any checkpoint continuation.
Evaluate the fixed pilot holdout subset as well as the expanded evaluation set
to keep the comparison interpretable.

## Checks

```powershell
.venv/Scripts/python.exe -m pytest experiments/furniture-data-v2/test_data.py -q -p no:cacheprovider --basetemp E:/STLModelData/furniture-data-v2-test
.venv/Scripts/ruff.exe check experiments/furniture-data-v2
.venv/Scripts/ruff.exe format --check experiments/furniture-data-v2
```

The verifier checks every prepared voxel file and occupancy hash, exact geometry
uniqueness, preserved holdout records, checked near-duplicate relations, source
fingerprints, and reopened target meshes. See [summary](summary.json),
[verification](verification.json), [caption spot checks](caption-review.csv),
and [report](report.md).
