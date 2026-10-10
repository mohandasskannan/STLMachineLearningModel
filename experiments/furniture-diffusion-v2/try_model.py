"""Local preview of the expanded-data candidate, alongside the selected epoch-80 model."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments/diffusion-high-noise-v1"
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location(
    "furniture_candidate_playground", PREVIOUS / "try_model.py"
)
playground = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playground)

import torch

from stlmodel.common import file_hash

TRACKED = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "runs/text2shape-furniture-diffusion-v2-conditioned/best.pt"
CACHE = Path("E:/STLModelData/furniture-diffusion-v2/embeddings.pt")
OUTPUT = CACHE.parent / "playground"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = json.loads((TRACKED / "results.json").read_text())
    cache_summary = json.loads((TRACKED / "cache-summary.json").read_text())
    if file_hash(CHECKPOINT) != results["selected_checkpoint_hashes"]["expanded"]:
        raise ValueError("Candidate weights changed after the recorded comparison")
    if file_hash(CACHE) != cache_summary["cache_sha256"]:
        raise ValueError("Candidate embedding cache changed")
    generator = playground.ExperimentalGenerator(
        args.device, checkpoint=CHECKPOINT, cache_path=CACHE, expected_epoch=30
    )
    generator.metadata.update(
        experiment="furniture-diffusion-v2",
        selection_status="candidate_not_promoted",
        visual_review=results["acceptance"]["visual_review"],
    )
    app = playground.build_app(
        generator,
        OUTPUT,
        title="STLModel — expanded furniture candidate",
        introduction=(
            "# STLModel — expanded furniture candidate\n"
            "This is the newly trained model (30 epochs on the expanded dataset). "
            "Rotate the preview or download the STL. Some chairs still have broken supports, "
            "including the tall-chair example with seed 42. "
            "[Compare with epoch 80](http://127.0.0.1:7863)."
        ),
    )
    app.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        allowed_paths=[str(OUTPUT.resolve())],
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
