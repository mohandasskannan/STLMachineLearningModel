from dataclasses import replace

import numpy as np
import pytest
import torch

from stlmodel.common import load_checkpoint
from stlmodel.data import make_fixture
from stlmodel.models import Denoiser, Diffusion
from stlmodel.pipeline import Generator, cache_embeddings
from stlmodel.training import TrainConfig, train


class FakeTextEncoder:
    """Offline test double only; production never substitutes fake embeddings."""

    model_name = "test-only"
    revision = None

    def encode(self, captions):
        result = np.zeros((len(captions), 384), dtype=np.float32)
        for i, caption in enumerate(captions):
            result[i, 0 if "chair" in caption else 1] = 1
        return result


def test_diffusion_determinism_and_conditioning():
    torch.manual_seed(4)
    model = Denoiser(latent_dim=8, width=32).eval()
    schedule = Diffusion(20)
    a = torch.zeros(1, 384)
    b = torch.ones(1, 384)
    result = schedule.sample(model, a, 42, 10)
    assert torch.equal(result, schedule.sample(model, a, 42, 10))
    assert not torch.equal(result, schedule.sample(model, a, 43, 10))
    assert not torch.equal(result, schedule.sample(model, b, 42, 10))
    assert schedule.alpha_bar[-1] < 0.001
    with pytest.raises(ValueError, match="Sampling steps"):
        schedule.sample(model, a, 42, 1)


def test_training_resume_matches_uninterrupted_run(tmp_path):
    manifest = make_fixture(tmp_path / "data", 5)
    config = TrainConfig(batch_size=2, max_steps=4, latent_dim=8, checkpoint_every=2)
    train("vae", manifest, tmp_path / "full", config, "cpu")
    train("vae", manifest, tmp_path / "resumed", replace(config, max_steps=2), "cpu")
    checkpoint = tmp_path / "resumed/last.pt"
    train("vae", manifest, tmp_path / "resumed", config, "cpu", checkpoint)
    full = load_checkpoint(tmp_path / "full/last.pt", "vae")
    resumed = load_checkpoint(checkpoint, "vae")
    assert resumed["step"] == full["step"] == 4
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="batch_size"):
        train(
            "vae", manifest, tmp_path / "resumed", replace(config, batch_size=4), "cpu", checkpoint
        )


def test_two_stage_pipeline_and_checkpoint_pairing(tmp_path):
    manifest = make_fixture(tmp_path / "data", 5)
    config = TrainConfig(batch_size=2, max_steps=2, latent_dim=8, diffusion_steps=20)
    train("vae", manifest, tmp_path / "vae", config, "cpu")
    vae = tmp_path / "vae/last.pt"
    cache_path = tmp_path / "cache.pt"
    encoder = FakeTextEncoder()
    cache_embeddings(manifest, vae, cache_path, "cpu", encoder)
    cache = load_checkpoint(cache_path, "embeddings")
    torch.testing.assert_close(cache["train"]["latents"].mean(0), torch.zeros(8), atol=1e-5, rtol=0)
    assert not set(cache["train"]["ids"]) & set(cache["val"]["ids"])
    train("diffusion", cache_path, tmp_path / "diffusion", config, "cpu")
    generator = Generator(vae, tmp_path / "diffusion/last.pt", "cpu", encoder)
    volume = generator.occupancy("a chair", sampling_steps=10)
    assert volume.shape == (32, 32, 32)
    assert np.isfinite(volume).all()
    assert np.array_equal(volume, generator.occupancy("a chair", sampling_steps=10))
    with pytest.raises(ValueError, match="text prompt"):
        generator.occupancy(" ")
    # Check provenance rather than silently pairing two incompatible checkpoints.
    with vae.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="exact autoencoder"):
        Generator(vae, tmp_path / "diffusion/last.pt", "cpu", encoder)
