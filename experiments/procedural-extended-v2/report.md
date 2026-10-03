# Second one-hour diffusion extension - October 3, 2026

The requested additional hour completed normally on the NVIDIA GTX 1660 Ti. It did not improve the validation-selected checkpoint, and final held-out error and visible geometry worsened. The earlier best remains recommended; more training time alone has not solved generation on this small fixture.

## Training and checkpoint selection

Resumed `runs/procedural-diffusion-extended-v1/last.pt` at step **89,876**, preserving optimizer/RNG state, in a new directory, `runs/procedural-diffusion-extended-v2`. Training ran for **3,600.063 seconds**, adding **74,202 steps** to reach **164,078 total**. Both one-hour extensions plus the original 166.204-second diffusion session total approximately **7,366.486 seconds (122.8 minutes)**. Peak PyTorch allocated memory was **0.0448 GiB**, excluding CUDA context and other applications; monitored GPU temperature was 42-43 degrees C with roughly 2.1-2.3 GB total memory use.

The architecture, batch size 64, learning rate 0.0002, 200-step diffusion schedule, dataset, and splits remained fixed. No downloads or application source changes were needed. The frozen VAE and embedding-cache hashes were verified unchanged. VAE reconstruction IoU remains **0.891406 validation / 0.948203 test**.

There are only 20 training shapes, two validation shapes, and two test shapes. The fixture does not distinguish tall versus short backs. These results describe procedural furniture, not real-object or Text2Shape performance.

The previous best was copied into the new run and remained selected by validation noise MSE: step **5,542**, SHA256 `8721788b009bab23637b34bfd5bcfe9405d594592b4dbde35d31e858668e02e5`. Its file hash is identical before and after this session. Test data did not select any checkpoint. Final-state test evaluation followed training and checkpoint selection.

## Held-out metrics

| Checkpoint | Validation noise MSE | Validation shuffled MSE | Validation gap | Test noise MSE | Test shuffled MSE | Test gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected best, step 5,542 | 0.203372 | 0.212614 | 0.009242 | 0.102985 | 0.115613 | 0.012628 |
| Previous final, step 89,876 | 1.245852 | 1.574163 | 0.328311 | 0.540591 | 0.749303 | 0.208712 |
| Second-hour final, step 164,078 | 1.831380 | 2.281292 | 0.449912 | 1.011875 | 0.944590 | -0.067285 |

Compared with the previous final state, the second-hour final validation noise MSE rose **47.0%**, and test noise MSE rose **87.2%**. Its negative test conditioning gap means correct captions performed worse than shuffled captions for the fixed sampled noise/timesteps. This tiny diagnostic does not establish general semantic behavior. The high held-out errors alongside low training loss are consistent with overfitting.

## Matched-seed geometry review

Prompts, in preview row order:

1. `A chair with a tall back and armrests`
2. `a chair with armrests`
3. `a chair without armrests`
4. `a table with four legs`

Columns use seeds **42, 43, 44**, with 50 DDIM sampling steps, threshold 0.5, and a 100 mm longest dimension. See [selected-best baseline](before/preview-front.png), [unchanged selected best after training](after/preview-front.png), [new final state](last/preview-front.png), and [previous final state](../procedural-extended-v1/last/preview-front.png).

All **36** new before/selected/final exports reopened with finite coordinates, watertight surfaces, and the requested 100 mm longest dimension. The 12 final samples contain **16-101 disconnected components**, and all 12 touch the grid boundary and require export caps. Your exact chair prompt has **82 / 57 / 84 components** for seeds 42 / 43 / 44, compared with **37 / 66 / 24** in the previous final state.

The final chairs show missing seats, interrupted legs/backs, and floating fragments. Armrest and no-armrest prompts visibly change some structure, but all outputs remain damaged and attribute control is unreliable. Table prompts produce partial platforms with legs, plus unwanted upright chair-like features, especially seeds 43 and 44. Seed variation is visible; it does not establish useful diversity. This review does not show a consistent visual gain over the previous hour.

Final nearest-training-shape IoU ranges from **0.140-0.488**. Pairwise voxel comparisons and nearest object IDs are retained in `last/evaluation.json`. These comparisons cannot exclude memorization. Watertightness of disconnected pieces does not establish printability; no slicer or physical-print check was performed.

## Try the newest final state

Stop any existing Gradio server with Ctrl+C, then run:

```powershell
.\.venv\Scripts\python.exe -m stlmodel serve --vae runs/procedural-vae-v1/best.pt --diffusion runs/procedural-diffusion-extended-v2/last.pt --device cuda
```

Open **http://127.0.0.1:7860**. For the recommended validation-selected model, replace `last.pt` with `best.pt`. That best file is identical to the previously selected model. An already-running server retains its loaded weights; restart it to use a different checkpoint. If port 7860 is already occupied, stop that server or add `--port 7861`.

## Reproducibility and artifacts

`results.json` includes timing, baseline/best/final metrics, prompts, seeds, hashes, and geometry summaries. `verification.json` records independent reopened-mesh checks; `training/` contains configuration, summary, and validation history sampled every 100 steps plus selected/start/final steps. Original runs remain preserved. Checkpoints, STL binaries, full logs, and data remain local under the existing ignore policy.

The driver `run_extended.py` uses the renderer at `outputs/procedural-v1/render_review.py` (tracked copy in `experiments/procedural-v1/`) and rejects an existing destination run. `package_results.py` verifies and packages completed artifacts. Use the CLI with `--resume .../last.pt` for an intentional later continuation.

Final diffusion SHA256: `245961fa89264ab6ff4d1ad321a1ee2c81552b182b56215f18e3900eb29d98f2`.
Frozen VAE SHA256: `388c5d9a50d50d9c35f99ec761c9ccdf5371a80f534808c9b00a21728a028dc9`.
Embedding cache SHA256: `50875efe17f4b313678886b581edb286482ffa58582684ec290193d6cb438fc9`.
