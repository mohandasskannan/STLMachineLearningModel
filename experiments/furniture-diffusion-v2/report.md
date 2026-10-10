# Expanded furniture training — October 10, 2026

Completed **30 new-data epochs / 22,350 optimizer steps** on 11,214 training
objects and 47,662 screened source captions. The run materially improves both
validation benchmarks and average generated connectivity, but **epoch 80 remains
selected**: the tall-chair prompt at seed 42 develops a visibly collapsed support
and detached geometry. All numerical gates pass; visual acceptance fails.

The candidate is preserved for inspection at
`runs/text2shape-furniture-diffusion-v2-conditioned/best.pt`. The existing local
playground at **http://127.0.0.1:7863** still serves epoch 80. No existing model,
cache, manifest, or protected checkpoint was overwritten.

## Training and decoder coverage

Weights initialize from the selected conditioned epoch-80 velocity model.
AdamW moments, random states, epoch counters, and step counters reset for the new
distribution. Decoder weights, text encoder revision, latent normalization,
architecture, zero-terminal schedule, and 15% text dropout stay fixed. The new
cache explicitly uses the original pilot training mean and standard deviation.
Original train/validation objects reuse their exact shape embeddings. Test
objects were not encoded or evaluated.

The expanded validation set contains 1,407 objects / 6,100 captions, including
the unchanged original 200 validation objects. Every training/validation object
was reconstructed with the frozen decoder before training. Added validation
mean IoU is **0.7720**, with chair mean **0.7329**, table mean **0.8000**, and
**99.75%** of added validation objects at IoU >=0.40. All prespecified coverage
gates pass. Added training mean IoU is 0.7773. These checks establish useful
decoder coverage, not accurate reproduction of every thin part or subtype.

Batch size is 64 and learning rate is 0.00005. Training stopped at its 30-epoch
ceiling after **287.17 seconds**, before the ten-minute or ten-stale-epoch limits.
The best expanded-validation checkpoint is the final new-data epoch 30. One new
epoch has 745 batches, compared with 126 in the pilot; these counters must not
be interpreted as a directly comparable "epoch 110."

## Matched validation results

Both models are evaluated with identical noise, timesteps, embeddings, and
normalization. The original subset uses its complete unchanged cache rather
than refitted representations. Lower MSE is better.

| Measurement | Epoch-80 parent | Expanded candidate |
| --- | ---: | ---: |
| Expanded validation velocity MSE | 0.577702 | **0.485898** |
| Original validation velocity MSE | 0.570831 | **0.477138** |
| Expanded clean-latent MSE | 0.283032 | **0.249954** |
| Original clean-latent MSE | 0.267531 | **0.234036** |
| Expanded shuffled-minus-matched conditioning gap | 0.056193 | 0.058199 |
| Original shuffled-minus-matched conditioning gap | 0.057657 | 0.057355 |

Velocity error improves **15.89%** on expanded validation and **16.41%** on the
original subset. Correct captions retain an advantage over shuffled captions;
the original-subset conditioning gap does not improve. High-noise recovery gains
are modest: original-object clean-latent RMSE at timestep 190 changes from 0.92035
to 0.91246 and at the zero-signal terminal timestep from 0.93986 to 0.93264.
These diagnostics do not certify individual furniture attributes.

## Generated shapes and visual decision

The same 40 original validation objects, two seeds, guidance 2, and 50 sampling
steps produce 80 samples per model. Category agreement remains the existing
nearest-pilot-training-shape diagnostic, not an independent semantic benchmark.

| Measurement | Epoch-80 parent | Expanded candidate |
| --- | ---: | ---: |
| Category agreement | 97.5% | **98.75%** |
| Mean 26-connected voxel regions | 4.0875 | **3.1125** |
| Mean largest occupied region fraction | 0.96661 | **0.97238** |
| Mean mesh components across 12 fixed prompts/seeds | 4.3333 | **2.3333** |
| Empty validation generations | 0 | 0 |

The four fixed prompts retain seeds 42, 43, and 44. All **24** parent/candidate
STLs reopen with finite coordinates, 100 mm longest dimension, minimum Z=0, and
watertight surfaces. Watertightness does not establish complete supports,
structural strength, or printability.

Front and back review shows substantial table cleanup, preservation of the
seed-44 armchair enclosure, and improved adherence to the armless request at
seed 42. However, **"A chair with a tall back and armrests," seed 42** now produces
a collapsed seat/support with a substantial detached fragment; the parent's
support structure is more coherent. Armrests, back openings, thin supports,
and rough surfaces remain unreliable. The candidate is **not promoted** because
of that visible regression despite passing all aggregate numerical gates.

- [Chair/table comparison](comparison.png)
- [All fixed prompts, front](conditioned-g2-prompts.png)
- [All fixed prompts, back](controls-back.png)
- [Recorded decision and measurements](results.json)
- [Reopened STL verification](verification.json)

## Reproducibility and limitations

The source data remains human-captioned and screened with limited visual spot
checks; it is not exhaustively caption-verified. Geometry remains 32-cubed and
the decoder is unchanged. Neither dataset breadth nor lower validation loss
guarantees a usable result for a particular prompt and seed. The original 400
holdouts and all protected artifact fingerprints remain unchanged, and no test
set was used for selection.

The new cache, per-object decoder coverage, protected-source ledger, STL meshes,
and console logs are stored under `E:/STLModelData/furniture-diffusion-v2`.
Only code, summaries, decisions, and preview images are tracked in Git.

Candidate SHA256:
`8909159cf765dcd568fd051566350ee97c23fc58525f7714abe5654304c475f7`.
Parent SHA256:
`8cead1fab3127f35797e3fffaf3d3c7cf9b9512f7b0ead8a5aa93546f8935de3`.

Verification includes **six** transfer/finalization regressions, **19** earlier
diffusion/fine-tuning regressions, **17** data checks, and **10** core tests, plus
lint, formatting, and Git whitespace checks. Transfer checks cover optimizer
reset, unchanged source weights, exact CPU resumption, normalization/frozen-model
incompatibility, and preventing visual approval from overriding failed metrics
or changed checkpoints.

[Protocol and commands](README.md) · [Training summary](summary.json) ·
[Decoder coverage](decoder-coverage.json) · [Cache summary](cache-summary.json).
