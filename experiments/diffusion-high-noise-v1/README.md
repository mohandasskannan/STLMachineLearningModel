# Diffusion high-noise and conditioning experiment

This experiment keeps the selected Text2Shape VAE, embedding cache, object splits,
text encoder, and legacy diffusion checkpoint fixed. Two fresh diffusion models
test changes suggested by the latent-distribution investigation:

- **Velocity:** existing concatenation MLP, velocity prediction, and a cosine
  schedule rescaled to exactly zero terminal signal.
- **Conditioned:** the same velocity objective and schedule, with text/time
  modulation at every residual block and 15% text dropout. Inference compares
  classifier-free guidance scales 1, 2, and 3.

The second candidate combines architecture and dropout changes; it does not
isolate their individual effects. Velocity losses cannot be directly compared
with the legacy noise-prediction loss. Common clean-latent diagnostics and actual
decoded shapes provide the cross-model comparison.

The method references are [noise schedules and velocity prediction](https://arxiv.org/abs/2305.08891)
and [classifier-free guidance](https://arxiv.org/abs/2207.12598).

## Run

To try the selected epoch-80 conditioned model interactively, without training:

```powershell
.venv/Scripts/python.exe experiments/diffusion-high-noise-v1/try_model.py
```

Open **http://127.0.0.1:7863**. Enter a chair/table prompt, vary the seed, rotate
the 3D preview, and download the STL. This playground fixes guidance at 2 and
uses the selected experimental checkpoint; the older production generator
cannot load these weights. Generated files and metadata are stored under
`E:/STLModelData/diffusion-high-noise-v1/playground`. Stop with Ctrl+C when running
the command in a terminal.

For a single export instead of the browser:

```powershell
.venv/Scripts/python.exe experiments/diffusion-high-noise-v1/try_model.py generate --prompt "a table with four legs" --seed 43
```

From the repository root, with the existing local dataset and checkpoints:

```powershell
.venv/Scripts/python.exe experiments/diffusion-high-noise-v1/run_experiment.py all
```

Defaults: seed 42, batch 64, learning rate 0.0002, width 512, 200 epochs maximum,
ten minutes per candidate, and a validation plateau stop after 20 consecutive
epochs without improvement. Validation velocity MSE selects the checkpoint;
test metrics are computed afterward and do not select weights or guidance.
Partial-epoch checkpoints include optimizer, random states, and batch cursor.
Use `train --resume` to continue only these experimental run directories, or
`review` to repeat evaluation without training.

Fresh checkpoints use the explicit `diffusion_velocity_experiment` artifact kind
under `runs/text2shape-diffusion-high-noise-v1-{velocity,conditioned}`. They cannot
be passed to the production generator, which expects legacy epsilon checkpoints.
This is intentional: changing a checkpoint's prediction-type label does not
convert its weights. No production model is automatically promoted.

## Review

The comparison includes the untouched legacy sampler, its timestep-190 diagnostic
variant, velocity prediction, and three guidance scales for the conditioned model.
Forty validation objects (20 chairs and 20 tables), with one supplied caption each,
use seeds 42 and 43. Category measurements use nearest training voxel overlap;
their calibration on all 200 validation reconstructions is reported. They are
diagnostics, not a human semantic benchmark or proof of attribute control.

The fixed prompts from the earlier pilot use seeds 42, 43, and 44. STL exports,
latent/occupancy arrays, and source fingerprints stay on E: under
`STLModelData/diffusion-high-noise-v1`. Summary JSON, previews, and training metadata
are packaged here. All protected artifact hashes must remain unchanged.

Training uses captions from the training split only. PCA/distribution statistics
and nearest-shape references also use only training objects. No sampling-time
clipping, PCA projection, nearest-shape replacement, or category label is used to
generate either candidate's geometry.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest experiments/diffusion-high-noise-v1/test_experiment.py -q -p no:cacheprovider --basetemp E:/STLModelData/diffusion-high-noise-v1-pytest
.venv/Scripts/ruff.exe check experiments/diffusion-high-noise-v1
.venv/Scripts/ruff.exe format --check experiments/diffusion-high-noise-v1
```

Tests cover the velocity/noise inverse at zero terminal signal, known-answer
sampling, seeded determinism, conditioning gradients, exact CPU resumption, and
rejection of incompatible checkpoints or changed training settings. Different
devices and library versions can still introduce numerical differences.
