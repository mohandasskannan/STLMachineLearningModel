# Full procedural fixture experiment

The fresh VAE learned useful held-out furniture structure. Diffusion reduced noise prediction error but did not produce useful geometry or reliable prompt control within this run.

This folder is a tracked snapshot of the report, metrics, mesh diagnostics, previews, and review scripts. Checkpoints, cached embeddings, console logs, and STL binaries remain in the ignored local `runs/`, `data/`, and `outputs/procedural-v1/` folders. Recorded absolute paths refer to the original experiment machine. The review scripts operate on those local artifacts. Training configuration, summaries, and complete validation histories are archived under `training/`.

## Runs and checkpoint selection

Both stages used the existing architecture, default learning rate 0.0002, CUDA on the GTX 1660 Ti, and a 15-minute session cap. Both finished their requested epochs early. The fixture and object-level splits were unchanged: 20 training objects, two validation objects, two test objects. Existing smoke runs were preserved. No dependencies were downloaded and no application source or CLI was changed.

| Stage | Batch | Epochs completed | Steps | Actual training session | Selected best step |
| --- | ---: | ---: | ---: | ---: | ---: |
| VAE | 8 | 1000 | 3000 | 119.094 s | 2718 (epoch 906) |
| Diffusion | 64 | 3000 | 3000 | 166.204 s | 2870 |

Combined recorded training time was 285.298 seconds (4 minutes 45 seconds). These session measurements include validation and checkpoint writes, and exclude Python startup, caching, generation, and visual review. Peak PyTorch allocated GPU memory was 0.083 GiB for VAE and 0.046 GiB for diffusion; these measurements exclude CUDA context and other processes.

The selected checkpoints are `runs/procedural-vae-v1/best.pt` and `runs/procedural-diffusion-v1/best.pt`. Each run also has `last.pt`, configuration, validation history, summary, and a console log. Selection used validation only. Test evaluation occurred after each checkpoint was selected.

## Reconstruction review

| Selected VAE | IoU | BCE |
| --- | ---: | ---: |
| Validation (2 objects) | 0.891406 | 0.036802 |
| Test (2 objects) | 0.948203 | 0.017414 |

Validation IoU exceeded the previous 0.300. Both held-out reconstructions retain recognizable chair/table structure and reopened as finite, watertight meshes with one component and a 100 mm longest dimension. The chair loses its armrests. The table has distorted edges and a partial raised back-like ridge; its proportions are imperfect. The numeric average therefore overstates attribute fidelity. These reconstructions met the plan's recognizable-structure gate for diffusion.

See [front preview](reconstructions/preview-front.png) and [rear preview](reconstructions/preview.png); each compares reconstruction (left) with target (right), chair above table. Mesh reports and reopened checks are archived in the same folder; STL files remain in the original local output folder.

## Cache integrity

`data/procedural-embeddings-v1.pt` contains 20/2/2 train/validation/test examples. Cached MiniLM weights were loaded with offline mode enabled. The text revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

The selected VAE was kept fixed after caching. Its SHA256 is `388c5d9a50d50d9c35f99ec761c9ccdf5371a80f534808c9b00a21728a028dc9`; both cache and diffusion checkpoint reference that exact hash. Selected diffusion SHA256 is `1458bf6694d206c6dcc96d21c72d02c9b2a3dc5a94631ef34409189a237f1726`.

## Text conditioning and generated geometry

| Selected diffusion | Correct text noise MSE | Shuffled text MSE | Gap (shuffled minus correct) |
| --- | ---: | ---: | ---: |
| Validation (2 examples) | 0.225077 | 0.228952 | 0.003874 |
| Test (2 examples) | 0.133528 | 0.139198 | 0.005670 |

The gap is positive but small. Each held-out split contains only two examples, and this diagnostic uses one fixed sampled noise/timestep set. It does not establish semantic control.

All combinations of seeds 42, 43, and 44 with these prompts were exported:

- Prompt 0: `a chair with armrests`
- Prompt 1: `a chair without armrests`
- Prompt 2: `a table with four legs`

`comparisons/preview-front.png` shows prompts as rows and seeds as columns. Prompt changes visibly perturb surfaces, but the matched-seed outputs remain similar and do not show reliable changes in armrests or chair versus table identity. Seed 42 retains some table-like legs/platform; seed 43 is highly fragmented; seed 44 has a fragmented platform and floating pieces. None is a useful complete furniture model.

| Matched-seed prompt pair | Seed 42 IoU | Seed 43 IoU | Seed 44 IoU | Mean |
| --- | ---: | ---: | ---: | ---: |
| With versus without armrests | 0.896 | 0.813 | 0.737 | 0.815 |
| Chair with armrests versus table | 0.852 | 0.823 | 0.754 | 0.810 |
| Chair without armrests versus table | 0.851 | 0.779 | 0.670 | 0.767 |

Within each prompt, mean pairwise IoU between seeds was 0.084, 0.073, and 0.082 respectively. Seeds dominate the variation. Low overlap here reflects unstable geometry rather than useful diversity. Nearest training-shape IoU ranged from 0.151 to 0.593; the reviewed samples do not look like exact training-shape copies, but this small check cannot exclude memorization.

All nine STL files exported and reopened with finite coordinates, watertight surfaces, and a 100 mm longest dimension. Component counts in prompt/seed order were **37, 134, 35; 36, 123, 38; 56, 124, 36**. All nine touched the voxel grid boundary and were capped by the existing exporter. Disconnected pieces were preserved. Watertightness does not compensate for the severe fragmentation, and no slicer or physical-print validation was performed. Full warnings are retained in each `.mesh.json` and `comparisons/evaluation.json`.

PowerShell logged MiniLM's stderr progress display as `NativeCommandError` during comparison, causing the shell session to report exit 1. The comparison itself completed, wrote its final evaluation JSON, and produced all nine STL files; those artifacts were independently reopened and checked successfully.

## Result

Reconstruction improved substantially with the full fixture. Prompt conditioning produces measurable perturbations but does not reliably control furniture category or armrests, and generated geometry remains severely defective. No further training or architecture changes were performed in this experiment. The results concern this tiny procedural fixture and do not establish performance on real objects.

`results.json` records selected checkpoint metrics, timings, hashes, prompt/seed overlap, nearest training overlap, and reopened mesh checks. Reusable review scripts are stored alongside the report; they do not modify the application.
