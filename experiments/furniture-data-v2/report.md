# Furniture data expansion — October 10, 2026

Prepared **11,214 training objects and 47,662 screened source captions**, up from
the pilot's 1,600 training objects and 8,015 captions: approximately **7.0× more
objects and 5.95× more captions**. The full prepared dataset has 14,028 objects
and 59,974 captions. Existing checkpoints, the original embedding cache, and the
running epoch-80 playground remain preserved. No model training was performed.

## Source and resulting splits

The full [Text2Shape source](https://svl.stanford.edu/projects/text2shape/), already
downloaded locally, contains 15,032 captioned chair/table objects and 75,360
description rows. It provides a substantially broader sample than the earlier
1,000-chair/1,000-table pilot without mixing in a new category or synthetic label
source. Caption IDs, CSV row numbers, canonical occupancy hashes, subtype IDs,
and source paths are retained in the quality index.

| Split | Objects | Chairs | Tables | Captions |
| --- | ---: | ---: | ---: | ---: |
| Training | 11,214 | 4,792 | 6,422 | 47,662 |
| Validation | 1,407 | 603 | 804 | 6,100 |
| Test | 1,407 | 603 | 804 | 6,212 |
| Total | 14,028 | 5,998 | 8,030 | 59,974 |

The training split contains **9,671 added objects** and 1,543 screened objects
from the original training split. It covers 42 source subtype IDs and includes
chairs with different back/arm structures, office and lounge forms, and tables
with round/rectangular tops, pedestals, curved bases, and storage structures.
The source's chair/table imbalance is retained rather than discarding valid
tables to achieve equal counts.

The original **200 validation and 200 test objects** retain their complete
original manifest records, captions, split assignments, and compressed voxel
file contents. They are exempt from new filters solely to preserve the baseline
benchmark; their captions are not newly certified as accurate. Only added objects
receive new category-stratified 80/10/10 assignments. The existing holdout IDs
must be evaluated separately during a future model comparison.

## Geometry and caption screening

New shapes use the audited NRRD signed-axis normalization into XYZ, Y up, and
RGBA alpha occupancy at 32-cubed resolution. Gross quality checks require useful
occupancy volume and bounding extents and at least 98% of occupied voxels in
one 26-connected region. Source parts are never cropped away to make a shape
appear cleaner.

| Object exclusion | Count |
| --- | ---: |
| Degenerate occupancy | 275 |
| Disconnected geometry | 118 |
| Collapsed or extremely flat geometry | 53 |
| Unreadable geometry | 9 |
| No usable geometry caption | 32 |
| Exact occupancy duplicates | 222 |
| Confirmed near-duplicate aliases | 293 |
| No captions after visual review | 2 |
| Total | 1,004 |

Near-duplicate screening checked 177,123 shortlisted pairs and confirmed 344
pairs with full-resolution voxel IoU at least 0.98. Connected duplicate groups
were reduced before assigning new splits, favoring protected holdouts and then
surviving pilot training objects. No cross-split near-duplicate relation remains
among retained objects in the checked pairs. The 16-neighbor pooled-occupancy
shortlist is **not an exhaustive near-duplicate search**.

Captions remain source descriptions, with only Unicode and whitespace
normalization. Screening excludes 9,089 rows without recognized geometry
information, 730 category-mismatched rows, 619 unsuitable-length rows, nine
duplicates, and 1,149 captions with conflicting armrest or leg-count claims.
These are source-wide counts and include captions of excluded objects.
No majority attribute claim is treated as factual ground truth.

Visual review exposed the negation variant “with out any side arms” and leg-count
variants such as “3 legged” and “four-legged.” The parser was corrected and
caption screening refreshed across the selected corpus. Previously excluded
objects were not reintroduced by this conservative refresh. Mixed shape/color
and material descriptions remain; monochrome occupancy cannot verify their
color or material claims.

## Visual caption review

Thirty-two added training objects were reviewed from front and back, targeting
armrests, armless chairs, varied back structures, four-leg tables, rounded/base
forms, and storage furniture. **25** sampled descriptions were consistent with
their represented geometry; **seven** had claims that could not be established
from the voxel target. These included a drawer count, a pull drawer, small back
openings, a back curvature claim, a vague shape-only caption, and a folding
mechanism. They were marked unverifiable rather than declared false descriptions
of the original higher-detail object.

Nine captions were quarantined, including other captions repeating the same
unsupported drawer claim. Two objects had no usable captions left and were
excluded. Other captions repeating a different, supported attribute were kept.
The review log records each observation and the specific claim being quarantined.

This targeted review is **not a corpus-wide caption-accuracy estimate**. The
remaining descriptions have source provenance and automated screening, with
limited geometry spot checks; they are not exhaustively verified. Color,
materials, comfort, and mechanical functionality were not validated.

- [Chair controls, front](review-1-front.png) · [back](review-1-back.png)
- [Other chair structures, front](review-2-front.png) · [back](review-2-back.png)
- [Table supports, front](review-3-front.png) · [back](review-3-back.png)
- [Table bases/storage, front](review-4-front.png) · [back](review-4-back.png)
- [All review decisions](caption-review.csv)

## Verification and next use

- All **14,028** prepared voxel files pass file-hash and canonical occupancy-hash
  verification and contain unique object IDs and exact geometry.
- All **57,982** non-protected caption/object pairings were independently checked
  against source caption ID, CSV row, object ID, category, and normalized text.
  Protected holdouts were verified through equality with their original records.
- All **400** original holdouts are preserved, and protected source/model hashes
  match their pre-preparation fingerprints.
- All **32** target STL reviews reopened with finite coordinates, 100 mm maximum
  extent, minimum Z=0, and watertight surfaces. They are visual QA artifacts,
  not printability certifications.
- **17** data regression tests, lint, formatting, and whitespace checks pass.

The prepared manifest is
`E:/STLModelData/furniture-data-v2/prepared/manifest.json`. Larger audit ledgers,
caption provenance, exclusion records, and target meshes remain alongside it on
E:. This repository packages the preparation/verification tools, summary,
decisions, and preview sheets.

Before a new diffusion training experiment, build a separate embedding cache
with the frozen decoder and explicitly choose normalization. The existing
epoch-80 checkpoint is tied to the original cache and normalization; a changed
cache must not be passed to an ordinary resume. Also check decoder coverage of
the added shapes, since added data alone does not raise the 32-cubed resolution
or prove that the current VAE represents every new subtype well.

[Methods and commands](README.md) · [Summary](summary.json) ·
[Independent verification](verification.json).
