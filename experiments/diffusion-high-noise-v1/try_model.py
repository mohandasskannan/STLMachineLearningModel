"""Interactive inference for the selected velocity checkpoint, without retraining."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from experiment import load_candidate

from stlmodel.common import device_for, file_hash, load_checkpoint
from stlmodel.pipeline import Generator
from stlmodel.text import TextEncoder
from stlmodel.training import load_vae

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "runs/text2shape-diffusion-high-noise-v1-conditioned/best.pt"
VAE = ROOT / "runs/text2shape-pilot-vae-v1/best.pt"
CACHE = Path("E:/STLModelData/text2shape-v1/embeddings.pt")
OUTPUT = Path("E:/STLModelData/diffusion-high-noise-v1/playground")
PROMPTS = [
    "A chair with a tall back and armrests",
    "a chair with armrests",
    "a chair without armrests",
    "a table with four legs",
]


class ExperimentalGenerator(Generator):
    def __init__(self, device_name="auto"):
        self.device = device_for(device_name)
        cache = load_checkpoint(CACHE, "embeddings")
        self.denoiser, self.schedule, state = load_candidate(
            CHECKPOINT, cache, file_hash(CACHE), self.device
        )
        if state["config"]["variant"] != "conditioned" or state["epoch"] != 80:
            raise ValueError("The playground requires the selected epoch-80 conditioned model")
        if state["vae_sha256"] != file_hash(VAE):
            raise ValueError("The selected decoder fingerprint does not match")
        self.vae, vae_state = load_vae(VAE, self.device)
        if vae_state["architecture"]["latent_dim"] != state["latent_dim"]:
            raise ValueError("Decoder and diffusion latent dimensions differ")
        self.mean, self.std = (
            state["latent_mean"].to(self.device),
            state["latent_std"].to(self.device),
        )
        self.encoder = TextEncoder(state["text_model"], state["text_revision"])
        if state["text_revision"] and self.encoder.revision != state["text_revision"]:
            raise ValueError("The cached text encoder revision does not match")
        self.metadata = {
            "vae_sha256": state["vae_sha256"],
            "diffusion_sha256": file_hash(CHECKPOINT),
            "checkpoint": str(CHECKPOINT),
            "epoch": 80,
            "text_model": state["text_model"],
            "text_revision": state["text_revision"],
            "prediction_type": state["prediction_type"],
            "schedule": state["schedule"],
            "guidance": 2.0,
        }

    @torch.inference_mode()
    def occupancy(self, prompt, seed=42, sampling_steps=50):
        if not prompt.strip():
            raise ValueError("Enter a text prompt")
        if not 0 <= seed < 2**32:
            raise ValueError("Seed must be between 0 and 4294967295")
        text = torch.from_numpy(self.encoder.encode([prompt.strip()])).to(self.device)
        latent = self.schedule.sample(self.denoiser, text, seed, sampling_steps, guidance=2.0)
        return self.vae.decode(latent * self.std + self.mean).sigmoid()[0, 0].cpu().numpy()


def build_app(generator, output_dir=OUTPUT):
    import gradio as gr

    def generate(prompt, seed, size_mm):
        try:
            output = Path(output_dir).resolve() / f"shape-{uuid4().hex}.stl"
            report = generator.generate(prompt, output, seed=int(seed), size_mm=float(size_mm))
            dimensions = " × ".join(f"{value:.1f}" for value in report["extents_mm"])
            status = f"Dimensions: {dimensions} mm\n\n" + "\n\n".join(report["warnings"])
            return str(output), str(output), status
        except (ValueError, OSError, RuntimeError) as exc:
            return None, None, f"Generation failed: {exc}"

    with gr.Blocks(title="STLModel — epoch 80", analytics_enabled=False) as app:
        gr.Markdown(
            "# STLModel — current model (epoch 80)\n"
            "Describe a chair or table, then rotate the preview or download the STL. "
            "Armrests, thin parts, and supports can still be inconsistent."
        )
        prompt = gr.Textbox(label="Describe your object", value=PROMPTS[0])
        with gr.Row():
            seed = gr.Number(value=42, minimum=0, maximum=4294967295, precision=0, label="Seed")
            size = gr.Number(value=100, minimum=1, label="Longest dimension (mm)")
        gr.Examples([[text] for text in PROMPTS], inputs=[prompt])
        button = gr.Button("Generate", variant="primary")
        preview = gr.Model3D(label="Preview")
        download = gr.File(label="Download STL", interactive=False)
        status = gr.Textbox(label="Result", interactive=False)
        button.click(
            generate,
            [prompt, seed, size],
            [preview, download, status],
            concurrency_limit=1,
            api_name="generate",
        )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("serve", "generate"), nargs="?", default="serve")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--prompt", default=PROMPTS[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-mm", type=float, default=100)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    generator = ExperimentalGenerator(args.device)
    if args.stage == "generate":
        output = args.out / f"shape-{uuid4().hex}.stl"
        print(generator.generate(args.prompt, output, args.seed, args.size_mm), flush=True)
        return
    app = build_app(generator, args.out)
    app.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        allowed_paths=[str(args.out.resolve())],
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
