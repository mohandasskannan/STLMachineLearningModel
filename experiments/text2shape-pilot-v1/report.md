# Text2Shape furniture pilot - October 3, 2026

The project now has a real-furniture pilot containing **2,000 objects and 10,007 captions**, stored on **E:**. A fresh VAE reconstructs held-out chairs and tables with **0.784570 validation IoU / 0.758062 test IoU**. Both training stages stopped early on validation plateaus. Text-conditioned generation remains severely defective; the larger dataset alone did not produce useful furniture in this pilot.

## Data preparation and orientation

Downloaded the caption CSV and **solid 32-resolution** voxel archive from the [official Text2Shape project](https://svl.stanford.edu/projects/text2shape/). Downloaded and prepared data, cached embeddings, logs, and STL exports are under `E:\STLModelData\text2shape-v1`. The archive is 1,066,672,179 bytes compressed; its entries total 1,286,260,073 uncompressed bytes. ZIP CRCs were verified. The existing repository, virtual environment, and checkpoint directories remain on C:.

The source CSV contains 6,589 chair IDs and 8,443 table IDs. An audit excluded **nine empty or fully occupied objects** and **227 exact duplicate occupancy grids**, keeping the lowest object ID for each identical normalized grid. No objects were excluded silently; reasons and retained duplicate IDs appear in `data-audit.json`. There are 6,501 unique usable chairs and 8,295 unique usable tables after these exclusions.

The orientation preview exposed a concrete format mismatch: Text2Shape NRRD storage axes are **Y, Z, X**, as specified by their `space directions` headers. Direct import as XYZ exported furniture sideways. The experiment driver transposes storage axes **[2, 0, 1]** into world XYZ, with Y up, before using the existing importer. All **15,023 nonempty source grids** use this permutation. It supports signed axis permutations and rejects oblique grids rather than silently resampling. Corrected targets were visually checked upright. Application source and CLI behavior were not changed.

Objects were sampled deterministically with Python random seed **42**, separately by category, retaining all distinct supplied descriptions for each selected object. The existing category-stratified, object-level splitter keeps all captions for each object together:

| Split | Chairs | Tables | Total objects | Caption pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 800 | 800 | 1,600 | 8,015 |
| Validation | 100 | 100 | 200 | 999 |
| Test | 100 | 100 | 200 | 993 |

All prepared grids were checked for finite binary occupancy and 32-cubed dimensions. IDs and exact normalized geometry do not cross splits. These are custom pilot splits, not the published Text2Shape benchmark splits. Seed/sample selection, excluded IDs, source URLs, hashes, and selected occupancy fingerprints are recorded in `data-audit.json`.

## Training, budgets, and selection

Budget: up to **30 minutes per stage**, with five-epoch resumable chunks and a stop at the first chunk boundary after ten consecutive validation epochs without improvement. There was no obligation to exhaust the budget. Timing below sums the trainer's session measurements, including its validation/checkpoint work; it excludes interpreter startup, preparation, embedding caching, and final review. Chunk startup overhead means wall time is longer than measured training time.

| Stage | Actual training seconds | Final epochs / steps | Selected epoch / step | Stop reason |
| --- | ---: | ---: | ---: | --- |
| VAE | 207.390 | 50 / 10,000 | 36 / 7,200 | Validation plateau |
| Diffusion | 131.799 | 125 / 15,750 | 115 / 14,490 | Validation plateau |

Combined measured training time was **339.189 seconds (5.65 minutes)**, within the one-hour total allowance. Training stages ran on the **NVIDIA GTX 1660 Ti**. Peak PyTorch allocations were approximately **0.083 GiB VAE / 0.046 GiB diffusion**, excluding CUDA context and other processes. Download start through final generation review took approximately **24.6 minutes**, including preparation, the orientation correction, caching, process startup, and evaluation.

Architecture and learning rate remain unchanged: 32-cubed occupancy, latent size 256, VAE batch 8, diffusion batch 64, AdamW learning rate 0.0002, and the existing 200-step diffusion schedule. MiniLM reused cached pretrained weights offline. Fresh model checkpoints are in `runs/text2shape-pilot-vae-v1` and `runs/text2shape-pilot-diffusion-v1`; no procedural weights were resumed or replaced. Optimizer and RNG states were preserved between chunks.

VAE selection used validation IoU only. Its selected checkpoint passed the **0.50 validation IoU gate**. Four chairs and four tables from validation were reviewed as target/reconstruction pairs. All eight retain recognizable category structure, exceeding the requirement of three per category. Chair back openings and details are smoothed or lost; some side supports, feet, and under-seat braces break apart. The round pedestal table loses connected feet. These reconstruction defects remain material.

The exact selected VAE was then frozen and used to cache embeddings. Diffusion selection used validation noise MSE only. Test metrics were evaluated after checkpoint selection and did not choose either model. Model fingerprints and pairing were verified after training.

## Held-out results

| VAE selected checkpoint | IoU | Binary cross entropy |
| --- | ---: | ---: |
| Validation, 200 objects | 0.784570 | 0.086402 |
| Test, 200 objects | 0.758062 | 0.082905 |

| Diffusion checkpoint | Validation noise MSE | Validation shuffled MSE | Validation gap | Test noise MSE | Test shuffled MSE | Test gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected best, step 14,490 | 0.318789 | 0.325893 | 0.007103 | 0.319103 | 0.326071 | 0.006968 |
| Final state, step 15,750 | 0.319304 | 0.326593 | 0.007289 | 0.319624 | 0.326868 | 0.007245 |

Conditioning gap is shuffled-text MSE minus correct-text MSE, evaluated with matched sampled noise/timesteps. Both selected gaps are positive but small; these diagnostics do not establish faithful prompt control. VAE results and diffusion losses on this dataset are **not directly comparable** with the old 20-shape procedural fixture's metrics.

## Visual generation and mesh verification

[Known target orientation](orientation/preview.png), [held-out reconstructions](reconstructions/preview.png), [selected generation preview](generated-best/preview.png), and [final-state generation preview](generated-last/preview.png).

Generation preview rows use:

1. `A chair with a tall back and armrests`
2. `a chair with armrests`
3. `a chair without armrests`
4. `a table with four legs`

Columns use seeds **42, 43, and 44**, with 50 DDIM sampling steps, threshold 0.5, and a 100 mm longest dimension.

| Geometry diagnostic | Selected best | Final state |
| --- | ---: | ---: |
| Exported samples | 12/12 | 12/12 |
| Watertight samples | 12/12 | 12/12 |
| Disconnected component range | 8-31 | 13-34 |
| Exact requested chair prompt, components for seeds 42/43/44 | 19 / 15 / 28 | 17 / 14 / 30 |
| Boundary-capped samples | 12/12 | 12/12 |
| Nearest training-shape IoU range | 0.209-0.323 | 0.198-0.522 |

The selected outputs have partial seats, legs, and disconnected upper pieces, with substantial missing structure. The final state has more continuous chair-like backs in some seeds but remains damaged, and its table outputs resemble malformed chairs. Neither state produces dependable furniture or armrest control. Differences between prompts affect voxels, but are not reliable changes to the requested attributes. Tall-back control was not established.

For the selected model, matched-seed armrest-versus-no-armrest voxel IoU is **0.749-0.772**, and chair-versus-table IoU is **0.410-0.504**. Seed changes also visibly affect geometry. Pairwise and nearest-training-object comparisons are retained in each `evaluation.json`; they do not exclude memorization.

All **48** orientation, target/reconstruction, and generated STL exports were independently reopened. All have finite coordinates, watertight surfaces, a 100 mm longest dimension, and minimum Z=0. Reconstruction component counts range from 1 to 5; generated geometry remains much more disconnected. Watertightness of disconnected pieces does not establish printability. No slicer or physical-print review was performed.

## Local interface and reproduction

The selected model is running on **http://127.0.0.1:7862**, leaving the older servers on 7860 and 7861 unchanged. Its Gradio API was tested with the exact chair prompt, seed 42, and size 100: both preview and downloadable STL were returned, with **19 components** reported.

To launch it later from the repository root:

```powershell
.\.venv\Scripts\python.exe -m stlmodel serve --vae runs/text2shape-pilot-vae-v1/best.pt --diffusion runs/text2shape-pilot-diffusion-v1/best.pt --out E:/STLModelData/text2shape-v1/review/ui --device cuda --port 7862
```

If this server is already running, open its URL. Stop it with Ctrl+C before launching another server on the same port. To inspect the final diffusion state, use `runs/text2shape-pilot-diffusion-v1/last.pt` with the same selected VAE.

The experiment driver provides `prepare`, `vae`, and `diffusion` stages. On a fresh machine with E: and the documented project environment:

```powershell
.\.venv\Scripts\python.exe experiments/text2shape-pilot-v1/run_pilot.py prepare
.\.venv\Scripts\python.exe experiments/text2shape-pilot-v1/run_pilot.py vae
```

Review the generated reconstructions before proceeding. The `diffusion` stage requires a recorded passing visual review in the local `review/vae-evaluation.json` (`visual_gate` set to `passed`) as well as the numerical gate. This record was completed after visual inspection in this run. Existing prepared data is rejected to preserve its splits; interrupted training can resume through its saved budget ledger and `last.pt`.

Regression checks cover channel-first RGBA with YZX storage, reversed spatial axes, and rejection of oblique grids. **Three tests passed**. Tests used an isolated temporary directory on E:. Experiment scripts pass lint and formatting checks. Full application tests were not rerun because application code and interfaces are unchanged.

`package_results.py` verifies model pairing and meshes, then copies review JSON, previews, and training metadata into this tracked folder. Data, complete captions, prepared voxel arrays, embeddings, weights, logs, and STL binaries remain local. A GitHub clone alone does not contain trained checkpoints. Dataset use remains subject to the applicable [ShapeNet terms](https://huggingface.co/datasets/ShapeNet/ShapeNetCore).

Selected VAE SHA256: `3ca72f7237a975bd77df6ae9faa2857600c8797eb7a4e9dcf2a157c46f49c72d`.
Selected diffusion SHA256: `338a5e5bb34cf15b22fb7c34e4dd487fc39e5a8815be096494df2fbdae091732`.
Embedding cache SHA256: `099380612172cbcbf069c8bd13ea740b606d480b3d6356772fc7e0c896695dff`.

The larger, varied dataset establishes a more useful reconstruction baseline, but generation remains the main unresolved problem. Inspecting latent distribution and sampling behavior is a reasonable next investigation; this run does not identify or fix a specific sampling defect.
