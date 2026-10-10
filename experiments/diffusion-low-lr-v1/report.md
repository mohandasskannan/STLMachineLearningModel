# Low-rate fine-tuning result — October 10, 2026

**Retain the epoch-80 conditioned checkpoint.** Fine-tuning produced a small
validation improvement and better aggregate generation diagnostics, but missed
the predefined improvement threshold and introduced visible regressions.
No production model or existing checkpoint was replaced.

## Training outcome

The run forked `runs/text2shape-diffusion-high-noise-v1-conditioned/best.pt`
(epoch 80, step 10,080), carrying forward its weights, AdamW moments and random
states. Its learning rate was reduced fourfold, from 0.0002 to 0.00005. Decoder,
data, architecture, objective, text dropout and batch size stayed fixed.

Training took **20.375 seconds**, completing 12 additional epochs and 1,512
updates before stopping on ten consecutive epochs without validation improvement.
The selected candidate is **total epoch 82**, two additional epochs and 252
updates after the parent. The final epoch-92 weights were worse than the selected
candidate and are retained only as `last.pt` for reproducibility.

| Validation measure | Parent, epoch 80 | Candidate, epoch 82 |
| --- | ---: | ---: |
| Velocity MSE | 0.57083060 | 0.56814197 |
| Clean-latent MSE | 0.26753064 | 0.26516004 |
| Correct-versus-shuffled caption error gap | 0.05765669 | 0.05821680 |
| High-noise clean-latent RMSE, timestep 190 | 0.92034686 | 0.91923308 |
| High-noise clean-latent RMSE, timestep 199 | 0.93985981 | 0.93771553 |

Velocity error improved by **0.4710%**, below the agreed **0.5%** minimum.
Passing that gate required MSE at most 0.56797644. The threshold was set before
training and was not changed after seeing this result. The final epoch-92
velocity MSE was 0.57362107, also worse than the original parent.

## Matched generations

Both models used guidance 2 and 50 sampling steps. The caption diagnostic used
the same 40 validation objects and seeds 42/43, giving 80 samples per model.
Category agreement is a nearest-training-shape diagnostic; its increase here
corresponds to just one additional sample and is not evidence of a statistically
established improvement in generalization.

| Generation measure | Parent | Candidate |
| --- | ---: | ---: |
| Category agreement | 97.5% | 98.75% |
| Latent RMS | 0.99047 | 0.93223 |
| Regularized whitened RMS | 1.19347 | 1.09135 |
| Mean connected voxel regions, 26-neighbor | 4.0875 | 3.6375 |
| Mean largest voxel-region fraction | 0.96661 | 0.96781 |
| Mean nearest-training voxel IoU | 0.58012 | 0.56585 |

The separate four-prompt, three-seed mesh comparison yielded mean mesh components
**4.33 → 3.92**. Component ranges changed from 1–15 to 1–8. Lower aggregate counts
do not mean every matched shape improved.

Visual inspection of all 24 prompt exports found:

- Tall-back chairs still generally lack the requested armrests. Seeds 42 and 44
  have more mesh components than their parent counterparts; seed 44 still has
  a damaged back and supports.
- The armrest prompt at seed 44 loses the parent's enclosing armchair-like
  sides and produces an armless form with fragmented lower supports. This is
  a visible regression in the requested attribute and geometry.
- The no-armrest prompt at seed 42 improves its requested attribute, but
  its supports remain rough. Other seeds still have holes or detached pieces.
- Table seed 42 has a cleaner top, while seed 43 still has broken support
  structure despite fewer detached mesh components. Seed 44 has visible gaps
  and roughness in its top.

See the [matched chair/table comparison](comparison.png) and
[all armrest and table controls](conditioned-g2-prompts.png). These are fixed-view
renders; no slicer or physical-print validation was performed.

## Acceptance and verification

Seven of eight numerical gates pass. The 0.5% validation-improvement gate fails,
and visual review fails because of the matched armrest/support regression.
**The parent remains selected.** Test-split metrics were deliberately not run
for this rejected candidate and did not influence checkpoint selection.

- All **19 regression tests** pass, including preservation of optimizer moments,
  parent hashes and random states, continued shuffle sequence, exact CPU fork
  resumption, incompatible fork rejection and acceptance-gate behavior.
- Experiment lint and formatting checks pass.
- All **24 STL exports** were independently reopened: finite vertices, 100 mm
  maximum extent, minimum Z=0, and watertight surfaces. Disconnected watertight
  pieces remain defective geometry.
- The original decoder, cache, manifest, legacy diffusion weights, and both
  previous candidate run directories retain their recorded hashes.

Candidate weights are isolated in
`runs/text2shape-diffusion-low-lr-v1-conditioned/{best.pt,last.pt}`. Large mesh and
latent artifacts are under `E:/STLModelData/diffusion-low-lr-v1`. Protected-source
fingerprints and the selected parent/candidate checkpoint hashes are recorded.
The experimental artifact kind continues to prevent use by the production
epsilon-only generator.

[Reproduction and methods](README.md) · [Full results](results.json) ·
[Independent mesh verification](verification.json).
