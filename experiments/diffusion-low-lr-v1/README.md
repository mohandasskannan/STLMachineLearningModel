# Epoch-80 low-rate fine-tuning

This isolated experiment tests smaller updates after the conditioned diffusion
pilot plateaued. It forks the selected epoch-80 checkpoint, preserving the
decoder, embedding cache, data splits, normalization, architecture, velocity
objective, optimizer moments, and random states. Only the learning rate changes
from 0.0002 to 0.00005. Batch size remains 64 and text dropout remains 15%.

## Training and checkpoint handling

```powershell
.venv/Scripts/python.exe experiments/diffusion-low-lr-v1/run_experiment.py all
```

The candidate uses `runs/text2shape-diffusion-low-lr-v1-conditioned`; its outputs
use `E:/STLModelData/diffusion-low-lr-v1`. Existing run files are fingerprinted
and never overwritten. A nonempty candidate run requires explicit `--resume`.
Use `review` to repeat comparisons without training.

Training stops at 30 additional epochs, three minutes of accumulated training
time, or ten consecutive epochs without validation velocity MSE improvement.
Validation runs after each completed epoch. Time is checked between batches;
checkpoint writing and final validation can slightly exceed the wall-clock cap.
The new run resets its stopping counters while recording parent path, hash,
epoch, and step. Its shuffle sequence continues after parent epoch 80.
Optimizer learning rates are explicitly overridden after loading their state.

The shared trainer exposes `initialize_from` as a checkpoint fork, mutually
exclusive with ordinary `resume`. Forking requires a completed parent epoch.
Architecture, diffusion schedule, data fingerprints, normalization, and text
encoder identity must match. Resume still rejects learning-rate changes.
The freshly evaluated parent initializes the candidate's best checkpoint, so
no improvement cannot select a worse checkpoint. Additional epochs are counted
locally; `epoch_offset + epoch` gives the total epoch.

## Comparison and acceptance

Guidance is fixed at 2 with 50 deterministic sampling steps. Parent and selected
candidate are reevaluated with identical validation noise and the previous
40-object, two-seed caption comparison (80 generations per model). The same four
fixed prompts and three seeds produce 12 STL files per model. All 24 exports are
independently reopened and the full prompt controls are rendered.

Acceptance requires at least 0.5% lower validation velocity MSE, no increase in
validation clean-latent MSE or correct-caption high-noise RMSE at timesteps 190
and 199, and no regression in category agreement, largest connected voxel
fraction, or mean fixed-prompt mesh components. Floating-point error comparisons
allow only 1e-7 rounding tolerance. These are pilot diagnostics, not proof of
semantic accuracy, generalization, or printability.

Visual review checks armrests, backs, table supports, and newly introduced
defects. The parent stays selected unless numerical and visual checks pass.
Test metrics are withheld until acceptance; the `test` stage requires passing
numerical gates and a recorded passing visual review. Record the inspection with
`finalize --visual-verdict passed|failed --visual-note "observation"`; repeat the
note option for additional observations. The finalize and test stages verify that
the selected checkpoint hashes still match the comparison. No checkpoint is
automatically promoted to the production generator, which uses a different
prediction interface.

## Checks

```powershell
.venv/Scripts/python.exe -m pytest experiments/diffusion-high-noise-v1/test_experiment.py experiments/diffusion-low-lr-v1/test_acceptance.py -q -p no:cacheprovider --basetemp E:/STLModelData/diffusion-low-lr-v1-pytest
.venv/Scripts/ruff.exe check experiments/diffusion-high-noise-v1 experiments/diffusion-low-lr-v1
.venv/Scripts/ruff.exe format --check experiments/diffusion-high-noise-v1 experiments/diffusion-low-lr-v1
```

Tests cover preservation of weights, AdamW moments, random states and source
hashes; continuation of the parent's shuffle sequence; exact CPU resumption;
incompatible forks; and each acceptance gate. See [results](results.json),
[mesh verification](verification.json), and [report](report.md).
