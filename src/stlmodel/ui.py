"""Local browser interface. Training stays in the CLI."""

from pathlib import Path
from uuid import uuid4

from .pipeline import Generator


def build_app(vae_path: Path, diffusion_path: Path, output_dir: Path, device_name="auto"):
    import gradio as gr

    generator = None

    def generate(prompt, seed, size_mm):
        nonlocal generator
        try:
            if not Path(vae_path).is_file() or not Path(diffusion_path).is_file():
                return (
                    None,
                    None,
                    "Train both model stages first. The configured checkpoint files are missing.",
                )
            if generator is None:
                generator = Generator(vae_path, diffusion_path, device_name)
            output = Path(output_dir).resolve() / f"shape-{uuid4().hex}.stl"
            report = generator.generate(prompt, output, seed=int(seed), size_mm=float(size_mm))
            dimensions = " × ".join(f"{x:.1f}" for x in report["extents_mm"])
            status = f"Dimensions: {dimensions} mm\n\n" + "\n\n".join(report["warnings"])
            return str(output), str(output), status
        except (ValueError, OSError, RuntimeError) as exc:
            # Clear previous preview/download so a failed request cannot expose an old result as new.
            return None, None, f"Generation failed: {exc}"

    with gr.Blocks(title="STLModel", analytics_enabled=False) as app:
        gr.Markdown(
            "# STLModel\nDescribe a chair or table to generate an editable starting mesh. "
            "Fine details and objects outside the training data may not be represented."
        )
        prompt = gr.Textbox(
            label="Describe your object", placeholder="A chair with a tall back and armrests"
        )
        with gr.Row():
            seed = gr.Number(value=42, minimum=0, maximum=4294967295, precision=0, label="Seed")
            size = gr.Number(value=100, minimum=1, label="Longest dimension (mm)")
        button = gr.Button("Generate", variant="primary")
        preview = gr.Model3D(label="Preview")
        download = gr.File(label="Download STL", interactive=False)
        status = gr.Textbox(label="Result", interactive=False)
        button.click(
            generate, [prompt, seed, size], [preview, download, status], concurrency_limit=1
        )
    return app


def serve(vae_path, diffusion_path, output_dir, device_name="auto", port=7860):
    app = build_app(vae_path, diffusion_path, output_dir, device_name)
    app.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        allowed_paths=[str(Path(output_dir).resolve())],
        inbrowser=False,
    )
