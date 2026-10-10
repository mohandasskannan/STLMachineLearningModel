# Diffusion high-noise and conditioning follow-up — October 10, 2026

The frozen decoder can produce substantially better furniture when diffusion's
high-noise behavior is corrected. Fresh velocity-prediction models generate
recognizable chairs and tables. Text modulation plus classifier-free guidance
improves category agreement further. Geometry and attribute control remain
imperfect, so the production model was not replaced.

The recommended candidate for further validation is **conditioned, guidance 2**.
It uses the selected checkpoint at epoch 80 / step 10,080, with the existing VAE
and latent cache. Guidance 3 offers the same category diagnostic score while
moving the latent distribution farther from real shapes. This recommendation
uses validation generation and visual review, not test-data checkpoint selection.

## What changed

Both candidates were trained from scratch using the same 8,015 training caption
pairs and 1,600 training objects as the Text2Shape pilot. The existing selected
VAE, encoder, cache, normalization, manifest, and legacy diffusion weights were
protected by hashes. They remain byte-for-byte unchanged.

The velocity candidate retains the original concatenation MLP but predicts
`v = sqrt(alpha) * noise - sqrt(1-alpha) * clean_latent`. Its cosine schedule is
rescaled to exactly zero terminal signal. Sampling reconstructs clean latents as
`sqrt(alpha) * noisy_latent - sqrt(1-alpha) * predicted_v`, avoiding division by a
vanishing signal scale. Both candidates sample from the final timestep with
50 deterministic DDIM steps; neither uses clipping, PCA projection, nearest-shape
replacement, or category labels during generation.

The conditioned candidate injects text/time features through learned scale/shift
modulation in every residual block and trains null text using 15% text dropout.
Guidance compares scales 1, 2, and 3. This candidate combines architecture and
dropout changes; this experiment cannot separate their individual contributions.

Method sources: [zero terminal signal and velocity prediction](https://arxiv.org/abs/2305.08891),
[classifier-free guidance](https://arxiv.org/abs/2207.12598).

## Training and high-noise results

Defaults were width 512, batch 64, AdamW learning rate 0.0002, seed 42, a
200-epoch ceiling, a ten-minute cap per candidate, and a stop after 20 consecutive
validation epochs without improvement. Both stopped on validation plateaus.

| Candidate | Training seconds | Final epoch / steps | Selected epoch / steps | Parameters | Peak allocated GPU GiB |
| --- | ---: | --- | --- | ---: | ---: |
| Velocity | 151.641 | 180 / 22,680 | 160 / 20,160 | 1,547,008 | 0.064 |
| Conditioned | 160.688 | 100 / 12,600 | 80 / 10,080 | 4,958,976 | 0.146 |

Total measured training time was **312.329 seconds (5.21 minutes)** on the GTX
1660 Ti. It includes trainer validation/checkpoint work and excludes interpreter
startup and generation review. Peak allocation excludes CUDA context memory.

Checkpoint selection used validation velocity MSE only. Test metrics were
computed after selecting weights. Velocity MSE cannot be directly compared with
the legacy epsilon/noise MSE because their targets differ.

| Selected candidate | Validation velocity MSE | Test velocity MSE | Validation conditioning gap | Test conditioning gap |
| --- | ---: | ---: | ---: | ---: |
| Velocity | 0.572066 | 0.559960 | 0.068282 | 0.062679 |
| Conditioned | 0.570831 | 0.557993 | 0.057657 | 0.051270 |

The gap is shuffled-caption target MSE minus correct-caption target MSE. The
conditioned model's larger generation response under guidance does not imply a
larger unguided denoising gap.

On all 200 validation objects, with matched noise and supplied captions, the
clean-latent reconstruction RMSE at the final timestep is **817.192** for legacy
epsilon prediction, **0.956** for velocity, and **0.940** for conditioned velocity.
The velocity schedule's terminal signal is exactly zero; legacy terminal signal
is nearly zero. These results demonstrate stable reconstruction at extreme
noise, rather than claiming recovery of a unique object from pure noise.

## Generation comparison

The main diagnostic uses one supplied caption from each of 40 validation objects
(20 chairs and 20 tables), with seeds 42 and 43: 80 samples per sampler.
Category agreement uses nearest-training voxel overlap. This diagnostic achieves
98% on all 200 held-out VAE reconstructions; it is not a human semantic benchmark.

| Sampler | Latent RMS | Category agreement | Connected voxel regions, mean | Largest voxel region, mean fraction | Regularized whitened RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy, original sampler | 583.556 | 70.0% | 22.10 | 0.791 | 2066.985 |
| Legacy, start timestep 190 | 1.331 | 73.8% | 9.70 | 0.856 | 2.430 |
| Velocity | 0.905 | 92.5% | 4.88 | 0.905 | 1.169 |
| Conditioned, guidance 1 | 0.912 | 91.3% | 4.10 | 0.965 | 1.111 |
| Conditioned, guidance 2 | 0.990 | 97.5% | 4.09 | 0.967 | 1.193 |
| Conditioned, guidance 3 | 1.068 | 97.5% | 3.93 | 0.958 | 1.303 |

Voxel regions use 26-neighbor connectivity and differ from mesh components.
Training latent RMS is 1.00; the previous diagnostic's real-validation
regularized whitened RMS was 0.943 using the same train-derived covariance and
0.001 diagonal regularization. Candidate latents remain somewhat outside the
real distribution, but the discrepancy is substantially reduced.

For the conditioned model at guidance 2, replacing captions with opposite-category
captions changes latent values by RMSE **1.123**, compared with **1.283** for changing
the seed. The stabilized legacy model's corresponding values were **0.416** and
**1.790**. Matched opposite-category voxel IoU drops from **0.577** to **0.105**,
consistent with stronger category control. This does not establish attribute
control within a category.

The fixed prompts from the original pilot were also generated with seeds
42, 43, and 44, producing 12 STL files per sampler:

| Sampler | Mean mesh components | Component range |
| --- | ---: | --- |
| Legacy, original sampler | 20.50 | 8–31 |
| Legacy, start timestep 190 | 7.92 | 5–11 |
| Velocity | 3.67 | 1–13 |
| Conditioned, guidance 1 | 3.58 | 1–9 |
| Conditioned, guidance 2 | 4.33 | 1–15 |
| Conditioned, guidance 3 | 3.83 | 1–13 |

These samples are a separate set from the 80-caption generation diagnostic.

[All-sampler comparison](comparison.png) · [Conditioned guidance-2 prompt controls](conditioned-g2-prompts.png).

Visual review finds recognizable chairs and tables in the new candidates.
Guidance-2 tall-back chairs have complete seat/back structures, but fail to
consistently include requested armrests. The simpler armrest prompt produces
armchair-like forms in two seeds; the no-armrest control still retains an
armchair-like form in one seed. Reliable armrest control has not been achieved.
Table seed 43 has heavily broken supports. Detached pieces, holes, distorted
supports, and missing detail remain. No slicer or physical-print validation was
performed.

The guidance-2 samples' nearest-training voxel overlap averages 0.580 and reaches
0.937 for one sample. Similarity and nonzero seed diversity do not exclude
memorization; this is a pilot rather than evidence of broad generalization.

## Artifacts and verification

- Five experiment tests pass: zero-signal velocity inverse, known-answer sampling,
  determinism, conditioning gradients, guidance composition, exact CPU resume,
  source mismatch, and incompatible checkpoint rejection are covered.
- Experiment lint and formatting checks pass.
- All **72 STL exports** were independently reopened: finite vertices, a 100 mm
  longest dimension, minimum Z=0, and watertight surfaces. Disconnected watertight
  pieces remain defective geometry.
- Protected model, cache, and dataset hashes remain unchanged. No application
  code, production default, or legacy checkpoint was replaced.

The selected new weights are in
`runs/text2shape-diffusion-high-noise-v1-conditioned/best.pt`. Their explicit
experimental artifact kind prevents the production epsilon-only generator from
misinterpreting them. Use this folder's experiment runner for reproduction;
production integration remains a separate change.

[Reproduction instructions](README.md) · [Full results](results.json) · [Independent mesh verification](verification.json).
