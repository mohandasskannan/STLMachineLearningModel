import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stlmodel import ui
from stlmodel.mesh import export_stl


def test_ui_routes_and_missing_checkpoint_clear_old_outputs(tmp_path):
    blocks = ui.build_app(tmp_path / "missing-vae.pt", tmp_path / "missing-diffusion.pt", tmp_path)
    app = gr.mount_gradio_app(FastAPI(), blocks, path="/")
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        config = client.get("/config").json()
        assert any(item["type"] == "model3d" for item in config["components"])
    preview, download, message = blocks.fns[0].fn("a chair", 42, 100)
    assert preview is None and download is None
    assert "checkpoint files are missing" in message


def test_ui_download_and_failure_cleanup(tmp_path, monkeypatch):
    class StubGenerator:
        def __init__(self, *args):
            pass

        def generate(self, prompt, output, seed, size_mm):
            if not prompt.strip():
                raise ValueError("Enter a text prompt")
            volume = np.zeros((32, 32, 32))
            volume[8:24, 8:24, 8:24] = 1
            return export_stl(volume, output, size_mm)

    monkeypatch.setattr(ui, "Generator", StubGenerator)
    vae, diffusion = tmp_path / "vae.pt", tmp_path / "diffusion.pt"
    vae.touch()
    diffusion.touch()
    blocks = ui.build_app(vae, diffusion, tmp_path / "outputs")
    callback = blocks.fns[0].fn
    preview, download, message = callback("a chair", 42, 125)
    assert preview == download
    assert preview.endswith(".stl")
    assert "125.0" in message
    assert "Printability is unverified" in message
    preview, download, message = callback(" ", 42, 125)
    assert preview is None and download is None
    assert "Enter a text prompt" in message
