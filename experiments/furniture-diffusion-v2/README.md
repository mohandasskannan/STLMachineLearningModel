# Expanded furniture diffusion experiment

This run adapts the selected epoch-80 conditioned velocity model to the screened
11,214-object training split. The VAE, MiniLM revision, 256-dimensional latent
coordinate system, zero-terminal schedule, architecture, and 15% text dropout
are frozen. A separate cache uses the original pilot training mean and standard
deviation; it never refits normalization on the changed dataset. Original
objects reuse their exact cached shape latents and available text embeddings.

Initialization transfers weights only and resets AdamW moments and random states.
It records both source and target cache fingerprints, and validates the parent
against its original cache before transfer. Ordinary resume still requires an
exact cache match. New-data epoch/step counters start at zero; they are not
directly equivalent to the smaller pilot's epochs.

## Run

```powershell
.venv/Scripts/python.exe experiments/furniture-diffusion-v2/run_experiment.py all
```

Stages `cache`, `train`, and `review` can run separately. `train --resume` preserves
optimizer, cursor, and random state within the candidate run and enforces the
total ten-minute training budget. Existing cache/run files are not overwritten
by a fresh run. Large artifacts stay at `E:/STLModelData/furniture-diffusion-v2`;
weights stay in `runs/text2shape-furniture-diffusion-v2-conditioned`.

The first experiment uses batch 64, learning rate 0.00005, at most 30 additional
epochs, ten minutes, and ten stale validation epochs. Best checkpoint selection
uses expanded validation velocity MSE, anchored to the unmodified parent's score
on that same set. Decoder coverage is checked on all training/validation shapes
first. Added validation objects must have mean reconstruction IoU >=0.65,
category means >=0.60, and at least 90% must have IoU >=0.40. These are practical
screening thresholds, not claims of printability or complete subtype coverage.

No new test objects are encoded or evaluated. Review compares the parent and
candidate on both expanded validation and the unchanged original 200-object
validation subset. The 80-caption sampling comparison and 12 fixed prompts per
model retain the previous seeds, guidance 2, frozen decoder, nearest-pilot-shape
category diagnostic, and independently reopened STL checks.

Numerical gates require >=0.5% expanded-validation improvement, <=1% original
validation regression, and no reduction in category agreement, dominant voxel
fraction, or fixed-prompt mesh connectivity. Visual review is required in addition
to those diagnostics. The current epoch-80 playground stays selected until the
candidate is reviewed; this runner never promotes a model automatically.

## Checks

```powershell
.venv/Scripts/python.exe -m pytest experiments/furniture-diffusion-v2/test_transfer.py experiments/diffusion-high-noise-v1/test_experiment.py -q
.venv/Scripts/ruff.exe check experiments/furniture-diffusion-v2 experiments/diffusion-high-noise-v1
```
