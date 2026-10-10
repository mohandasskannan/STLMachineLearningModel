"""Preview artifacts and direct browser responses must remain usable without WebGL/SSE."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from stlmodel.mesh import export_stl

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("preview_playground", DIRECTORY / "try_model.py")
playground = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playground)


def test_image_glb_and_download_survive_without_streaming_or_webgl(tmp_path):
    class StubGenerator:
        def generate(self, prompt, output, seed, size_mm):
            if not prompt.strip():
                raise ValueError("Enter a text prompt")
            voxels = np.zeros((32, 32, 32))
            voxels[8:24, 9:22, 6:26] = 1
            return export_stl(voxels, output, size_mm)

    app = playground.build_app(StubGenerator(), tmp_path)
    generation = next(fn for fn in app.fns.values() if fn.api_name == "generate")
    assert generation.queue is False
    image, model, download, message = generation.fn("a chair", 42, 125)
    assert "Generated successfully" in message
    picture = np.asarray(Image.open(image)).astype(np.int16)
    assert np.count_nonzero(picture[:, :, 2] > picture[:, :, 0] + 20) > 1000
    preview = trimesh.load(model, force="mesh")
    original = trimesh.load_mesh(download)
    assert np.isfinite(preview.vertices).all()
    assert np.isclose(preview.extents.max(), 2)
    assert np.allclose(preview.bounds.mean(axis=0), 0, atol=1e-6)
    assert np.isclose(original.extents.max(), 125, atol=0.001)
    image, model, download, message = generation.fn(" ", 42, 125)
    assert image is model is download is None
    assert "Enter a text prompt" in message
