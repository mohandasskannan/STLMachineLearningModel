# One-hour procedural diffusion extension — October 2, 2026

The requested prompt was **`A chair with a tall back and armrests`**, corresponding to the first-run screenshot added in GitHub commit `ea35eb6`. One additional hour of GPU training improved the best noise-prediction metrics early, then substantially overfit the small fixture. Generated furniture remains severely fragmented. Later weights show clearer category differences in some seeds, but neither checkpoint produces useful complete furniture.

## Training and selection

Diffusion resumed from the previous `last.pt` at step 3,000 in a separate directory, `runs/procedural-diffusion-extended-v1`. The GTX 1660 Ti ran for **3,600.219 seconds** (60 minutes), adding **86,876 steps** for **89,876 total**. The run reached its time cap normally, without interruption. Peak PyTorch allocated memory was 0.0448 GiB; monitored total GPU memory was about 2.3–2.7 GB, and monitored GPU temperatures stayed around 44–45°C. Allocated-memory measurements exclude CUDA context and other applications.

Architecture, batch size 64, learning rate 0.0002, data, splits, and sampling defaults were unchanged. The VAE remained frozen at `runs/procedural-vae-v1/best.pt`; its validation/test IoU remains **0.891406/0.948203**. The embedding cache remained `data/procedural-embeddings-v1.pt`. Both hashes were verified unchanged after training. Original smoke and procedural run directories were preserved.

There are still only **20 training shapes, two validation shapes, and two test shapes**. Captions distinguish chairs, tables, and armrests. They do not distinguish tall versus short backs, so this experiment cannot establish control over that attribute.

`best.pt` was selected by validation noise MSE only, including the original best as a candidate. The new selected checkpoint is step **5,542**, reached about **115 seconds** into this extension. No later step improved its validation score. Test results were evaluated after selection; the final `last.pt` was evaluated separately as a diagnostic and did not replace the validation-selected checkpoint.

## Held-out metrics

| Checkpoint | Validation noise MSE | Validation shuffled MSE | Validation gap | Test noise MSE | Test shuffled MSE | Test gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous best, step 2,870 | 0.225077 | 0.228952 | 0.003874 | 0.133528 | 0.139198 | 0.005670 |
| Extended best, step 5,542 | 0.203372 | 0.212614 | 0.009242 | 0.102985 | 0.115613 | 0.012628 |
| Final state, step 89,876 | 1.245852 | 1.574163 | 0.328311 | 0.540591 | 0.749303 | 0.208712 |

Gap means shuffled-text MSE minus correct-text MSE, with matched noise and timesteps. The early checkpoint reduced validation MSE by about 9.6% and test MSE by about 22.9%. Large late conditioning gaps coexist with substantially worse correct-text error; they do not establish better generation or attribute control. These diagnostics use one fixed sampled noise/timestep set on each two-example split, so they provide limited evidence.

Later training losses remained low while held-out MSE increased, consistent with overfitting. Extending training to a second hour was not justified by this trend, so the run stopped at the requested range's one-hour endpoint.

## Prompt and geometry review

Each of these prompts was tested with seeds **42, 43, and 44**, using 50 DDIM sampling steps, threshold 0.5, and a 100 mm longest dimension:

1. `A chair with a tall back and armrests`
2. `a chair with armrests`
3. `a chair without armrests`
4. `a table with four legs`

The grids below use prompts as rows and seeds as columns:

- [Previous best preview](before/preview-front.png)
- [Selected extended best preview](after/preview-front.png)
- [Final full-hour state preview](last/preview-front.png)

| Geometry diagnostic | Previous best | Extended best | Final state |
| --- | ---: | ---: | ---: |
| Successfully exported samples | 12/12 | 12/12 | 12/12 |
| Watertight meshes | 12/12 | 12/12 | 12/12 |
| Disconnected component range | 31–141 | 37–174 | 9–123 |
| Exact requested prompt, components for seeds 42/43/44 | 32 / 141 / 31 | 68 / 158 / 37 | 37 / 66 / 24 |
| Boundary-capped samples | 12/12 | 12/12 | 9/12 |

The selected extended checkpoint has more visible backs in seeds 42 and 44, but broken seats/legs and many floating fragments. Its improved loss did not improve mesh integrity, and matched-seed prompt changes still do not provide reliable furniture or armrest control.

The final full-hour state shows clearer table-versus-chair differences in some seeds and more evident armrest-like structures in some chair outputs. The table at seed 43 is more recognizable and has the fewest components (nine), but it still includes disconnected pieces. The exact requested chair prompt remains damaged in all three seeds; seed 43 loses most chair structure. This is partial visual improvement in some samples, not dependable prompt following. Tall-back control is not supported by the training captions.

All **36** reviewed STL files were independently reopened and checked for finite coordinates, watertightness, and a 100 mm longest dimension. All passed these structural checks. Severe disconnection and boundary caps remain visible defects; watertightness does not establish useful or printable geometry. No slicer or physical-print validation was performed.

Nearest training-shape IoU ranged from 0.151–0.593 before, 0.216–0.526 for the selected extended best, and 0.251–0.752 for the final state. The final table at seed 43 was nearest a training table with IoU 0.752. These limited comparisons cannot exclude memorization. Pairwise seed/prompt overlaps and nearest object IDs are retained in each stage's `evaluation.json`.

## Try the trained checkpoints

For the validation-selected checkpoint, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m stlmodel serve --vae runs/procedural-vae-v1/best.pt --diffusion runs/procedural-diffusion-extended-v1/best.pt --device cuda
```

Open `http://127.0.0.1:7860`. To try the final full-hour state and its partially improved visual category separation, replace the diffusion path with `runs/procedural-diffusion-extended-v1/last.pt`. That state has much worse held-out metrics and remains severely defective. Both checkpoints require the exact frozen VAE above.

## Artifacts and reproducibility

This tracked folder contains results, prompts/seeds, model hashes, mesh and generation metadata, reopened checks, previews, training configuration/summary, and a validation history sampled every 100 steps plus the baseline/selected/final steps. Full history and console logs remain in the local run/output folders. Checkpoints, caches, and STL binaries remain local under the repository's existing ignore policy; a GitHub clone alone cannot run these weights. Recorded absolute paths identify original local artifacts.

`run_extended.py` is a copy of the local driver. It runs from the repository root and uses the earlier review renderer at `outputs/procedural-v1/render_review.py`; the tracked copy of that renderer is in `experiments/procedural-v1/`. The driver rejects an existing extended run directory to prevent accidental restart. Resume from `last.pt` only when intentionally budgeting another session.

`results.json` records selected-checkpoint results and actual training time; `last-results.json` records the final state's diagnostics. The unchanged VAE SHA256 is `388c5d9a50d50d9c35f99ec761c9ccdf5371a80f534808c9b00a21728a028dc9`. Selected extended diffusion SHA256 is `8721788b009bab23637b34bfd5bcfe9405d594592b4dbde35d31e858668e02e5`; final diffusion SHA256 is `e9839c5d7f04ade8199d55b86d58762b5b4d97297f87a7aeb6441404d9eddb10`.

These are procedural-fixture results, not Text2Shape or real-object performance. The main unresolved problems are fragmentary generation and unreliable semantic control; additional time alone did not solve them in this experiment.
