"""Interactive inference for the selected velocity checkpoint, without retraining."""

import argparse
import os
import threading
from pathlib import Path
from uuid import uuid4

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import trimesh
from experiment import load_candidate
from PIL import Image, ImageDraw

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
    def __init__(
        self, device_name="auto", *, checkpoint=CHECKPOINT, cache_path=CACHE, expected_epoch=80
    ):
        self.device = device_for(device_name)
        cache = load_checkpoint(cache_path, "embeddings")
        self.denoiser, self.schedule, state = load_candidate(
            checkpoint, cache, file_hash(cache_path), self.device
        )
        if state["config"]["variant"] != "conditioned" or state["epoch"] != expected_epoch:
            raise ValueError(f"The playground requires the expected epoch-{expected_epoch} model")
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
            "diffusion_sha256": file_hash(checkpoint),
            "checkpoint": str(checkpoint),
            "epoch": expected_epoch,
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


def make_previews(stl_path):
    """Render a browser-independent image and a centered, unit-sized Y-up GLB."""
    mesh = trimesh.load_mesh(stl_path)
    vertices = mesh.vertices - mesh.bounds.mean(axis=0)
    canvas = Image.new("RGB", (840, 440), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    for column, direction in enumerate((1.8, -1.8)):
        view = np.array([1.3, direction, 1.1])
        view /= np.linalg.norm(view)
        right = np.cross([0, 0, 1], view)
        right /= np.linalg.norm(right)
        up = np.cross(view, right)
        projected = np.stack((vertices @ right, -(vertices @ up)), axis=1)
        span = np.ptp(projected, axis=0).max()
        projected *= 350 / max(span, 1e-6)
        projected -= (projected.max(axis=0) + projected.min(axis=0)) / 2
        projected += [column * 420 + 210, 225]
        shades = 0.45 + 0.55 * np.maximum(0, mesh.face_normals @ light)
        for face in np.argsort(mesh.triangles_center @ view):
            color = tuple(int(value * shades[face]) for value in (65, 145, 210))
            draw.polygon([tuple(point) for point in projected[mesh.faces[face]]], fill=color)
        draw.text((column * 420 + 16, 14), "Front" if column == 0 else "Back", fill="#26374a")
    image_path = Path(stl_path).with_suffix(".png")
    canvas.save(image_path)
    preview_mesh = mesh.copy()
    preview_mesh.vertices = vertices / mesh.extents.max() * 2
    # STL export uses Z up; glTF viewers use Y up.
    preview_mesh.vertices = preview_mesh.vertices[:, [0, 2, 1]] * [1, 1, -1]
    preview_mesh.visual.vertex_colors = [65, 145, 210, 255]
    glb_path = Path(stl_path).with_suffix(".glb")
    preview_mesh.export(glb_path)
    return str(image_path), str(glb_path)


def build_app(generator, output_dir=OUTPUT, *, title="STLModel — epoch 80", introduction=None):
    import gradio as gr

    generation_lock = threading.Lock()

    def generate(prompt, seed, size_mm):
        try:
            output = Path(output_dir).resolve() / f"shape-{uuid4().hex}.stl"
            with generation_lock:
                report = generator.generate(prompt, output, seed=int(seed), size_mm=float(size_mm))
            image_path, glb_path = make_previews(output)
            dimensions = " × ".join(f"{value:.1f}" for value in report["extents_mm"])
            status = f"Generated successfully. Dimensions: {dimensions} mm\n\n" + "\n\n".join(
                report["warnings"]
            )
            return image_path, glb_path, str(output), status
        except (ValueError, OSError, RuntimeError) as exc:
            return None, None, None, f"Generation failed: {exc}"

    with gr.Blocks(title=title, analytics_enabled=False) as app:
        gr.Markdown(
            introduction
            or "# STLModel — current model (epoch 80)\n"
            "Describe a chair or table, then rotate the preview or download the STL. "
            "Armrests, thin parts, and supports can still be inconsistent."
        )
        prompt = gr.Textbox(label="Describe your object", value=PROMPTS[0])
        with gr.Row():
            seed = gr.Number(value=42, minimum=0, maximum=4294967295, precision=0, label="Seed")
            size = gr.Number(value=100, minimum=1, label="Longest dimension (mm)")
        gr.Examples([[text] for text in PROMPTS], inputs=[prompt])
        button = gr.Button("Generate", variant="primary")
        image_preview = gr.Image(
            label="Generated shape — front and back", interactive=False, height=440
        )
        preview = gr.Model3D(
            label="Rotate the 3D preview",
            height=420,
            interactive=False,
            display_mode="solid",
            clear_color=(0.96, 0.97, 0.99, 1),
            camera_position=(45, 65, 4),
        )
        download = gr.File(label="Download STL", interactive=False)
        status = gr.Textbox(label="Result", interactive=False)
        button.click(
            generate,
            [prompt, seed, size],
            [image_preview, preview, download, status],
            concurrency_limit=1,
            api_name="generate",
            queue=False,
            scroll_to_output=True,
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
