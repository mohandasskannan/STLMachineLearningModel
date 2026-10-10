"""Mathematical and checkpoint regressions for the isolated experiment."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from stlmodel.common import file_hash, load_checkpoint, save_checkpoint

spec = importlib.util.spec_from_file_location(
    "velocity_experiment", Path(__file__).with_name("experiment.py")
)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def test_velocity_inverse_at_zero_terminal_signal():
    schedule = experiment.VelocitySchedule(20)
    clean = torch.randn(3, 8)
    noise = torch.randn_like(clean)
    times = torch.tensor([0, 10, 19])
    noisy = schedule.noisy(clean, times, noise)
    recovered, recovered_noise = schedule.parts(noisy, schedule.target(clean, times, noise), times)
    torch.testing.assert_close(recovered, clean, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(recovered_noise, noise, atol=1e-6, rtol=1e-6)
    assert schedule.alpha_bar[-1] == 0
    assert torch.all(schedule.alpha_bar[1:] <= schedule.alpha_bar[:-1])


def test_oracle_sampling_and_seed_determinism():
    schedule = experiment.VelocitySchedule(20)
    clean = torch.randn(1, 8)

    class Oracle:
        latent_dim = 8

        def __call__(self, z, t, text):
            a = schedule.alpha_bar[t, None]
            return (a.sqrt() * z - clean) / (1 - a).sqrt()

    recovered = schedule.sample(Oracle(), torch.zeros(1, 384), sampling_steps=10)
    torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)
    model = experiment.Denoiser(latent_dim=8, width=16).eval()
    text = torch.randn(1, 384)
    a = schedule.sample(model, text, 42, 10)
    assert torch.equal(a, schedule.sample(model, text, 42, 10))
    assert not torch.equal(a, schedule.sample(model, text, 43, 10))
    with pytest.raises(ValueError, match="Guidance"):
        schedule.sample(model, text, sampling_steps=10, guidance=float("nan"))


def test_adaptive_conditioning_receives_gradients():
    model = experiment.AdaptiveDenoiser(latent_dim=8, width=16)
    torch.nn.init.normal_(model.output[-1].weight, std=0.1)
    model(torch.randn(2, 8), torch.tensor([1, 19]), torch.randn(2, 384)).square().mean().backward()
    assert model.blocks[0].modulation.weight.grad.abs().sum() > 0


def test_guidance_combines_conditional_and_null_predictions():
    schedule = experiment.VelocitySchedule(20)

    class CaptionOracle:
        latent_dim = 8

        def __call__(self, z, t, text):
            a = schedule.alpha_bar[t, None]
            desired = text[:, :1].expand_as(z)
            return (a.sqrt() * z - desired) / (1 - a).sqrt()

    text = torch.ones(1, 384)
    for guidance in (0.0, 1.0, 2.0, 3.0):
        result = schedule.sample(CaptionOracle(), text, sampling_steps=10, guidance=guidance)
        torch.testing.assert_close(result, torch.full_like(result, guidance), atol=1e-5, rtol=1e-5)


def fake_cache():
    torch.manual_seed(12)
    return {
        "train": {"latents": torch.randn(6, 8), "text": torch.randn(6, 384)},
        "val": {"latents": torch.randn(4, 8), "text": torch.randn(4, 384)},
        "latent_mean": torch.zeros(8),
        "latent_std": torch.ones(8),
        "vae_sha256": "test-vae",
        "manifest_sha256": "test-manifest",
        "text_model": "test-only",
        "text_revision": None,
    }


def test_resume_exactly_matches_and_rejects_incompatible_checkpoint(tmp_path):
    torch.set_num_threads(2)
    cache = fake_cache()
    config = experiment.Config(
        variant="conditioned", width=16, batch_size=2, epochs=5, max_steps=4, text_dropout=0.15
    )
    experiment.train_candidate(cache, "test-source", tmp_path / "full", config, "cpu")
    experiment.train_candidate(
        cache, "test-source", tmp_path / "split", replace(config, max_steps=2), "cpu"
    )
    experiment.train_candidate(cache, "test-source", tmp_path / "split", config, "cpu", resume=True)
    full = load_checkpoint(tmp_path / "full/last.pt", experiment.KIND)
    resumed = load_checkpoint(tmp_path / "split/last.pt", experiment.KIND)
    assert full["step"] == resumed["step"] == 4
    for key, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Cannot change"):
        experiment.train_candidate(
            cache,
            "test-source",
            tmp_path / "split",
            replace(config, text_dropout=0.2),
            "cpu",
            resume=True,
        )
    with pytest.raises(ValueError, match="not empty"):
        experiment.train_candidate(cache, "test-source", tmp_path / "split", config, "cpu")
    with pytest.raises(ValueError, match="pairing"):
        experiment.load_candidate(tmp_path / "split/best.pt", cache, "wrong-source", "cpu")
    save_checkpoint(tmp_path / "epsilon.pt", {"format_version": 1, "kind": "diffusion"})
    with pytest.raises(ValueError, match="Expected"):
        experiment.load_candidate(tmp_path / "epsilon.pt", cache, "test-source", "cpu")


def test_fork_preserves_parent_optimizer_and_can_resume_exactly(tmp_path):
    torch.set_num_threads(2)
    cache = fake_cache()
    parent_config = experiment.Config(
        variant="conditioned", width=16, batch_size=2, epochs=1, text_dropout=0.15
    )
    experiment.train_candidate(cache, "test-source", tmp_path / "parent", parent_config, "cpu")
    parent_path = tmp_path / "parent/best.pt"
    before = file_hash(parent_path)
    parent = load_checkpoint(parent_path, experiment.KIND)
    config = replace(parent_config, epochs=3, learning_rate=0.00005, patience=10)
    experiment.train_candidate(
        cache,
        "test-source",
        tmp_path / "anchor",
        replace(config, max_seconds=1e-9),
        "cpu",
        initialize_from=parent_path,
    )
    anchor = load_checkpoint(tmp_path / "anchor/best.pt", experiment.KIND)
    assert anchor["epoch"] == anchor["step"] == anchor["stale_epochs"] == 0
    assert anchor["epoch_offset"] == parent["epoch"] == 1
    assert anchor["parent"]["sha256"] == before
    assert anchor["optimizer"]["param_groups"][0]["lr"] == 0.00005
    for key, value in parent["model"].items():
        torch.testing.assert_close(value, anchor["model"][key], atol=0, rtol=0)
    for parameter, values in parent["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                value, anchor["optimizer"]["state"][parameter][key], atol=0, rtol=0
            )
    assert torch.equal(parent["torch_rng"], anchor["torch_rng"])
    for name, steps in (("full-fork", 4), ("split-fork", 2)):
        experiment.train_candidate(
            cache,
            "test-source",
            tmp_path / name,
            replace(config, max_steps=steps),
            "cpu",
            initialize_from=parent_path,
        )
    experiment.train_candidate(
        cache,
        "test-source",
        tmp_path / "split-fork",
        replace(config, max_steps=4),
        "cpu",
        resume=True,
    )
    full = load_checkpoint(tmp_path / "full-fork/last.pt", experiment.KIND)
    split = load_checkpoint(tmp_path / "split-fork/last.pt", experiment.KIND)
    for key, value in full["model"].items():
        torch.testing.assert_close(value, split["model"][key], atol=0, rtol=0)
    assert full["parent"] == split["parent"]
    assert file_hash(parent_path) == before
    with pytest.raises(ValueError, match="Cannot change learning_rate on resume"):
        experiment.train_candidate(
            cache,
            "test-source",
            tmp_path / "split-fork",
            replace(config, learning_rate=0.0001),
            "cpu",
            resume=True,
        )


def test_fork_rejects_incompatible_data_and_settings(tmp_path):
    cache = fake_cache()
    config = experiment.Config(width=16, batch_size=2, epochs=1)
    experiment.train_candidate(cache, "test-source", tmp_path / "parent", config, "cpu")
    parent = tmp_path / "parent/best.pt"
    for changed in (
        replace(config, width=32),
        replace(config, diffusion_steps=100),
        replace(config, variant="conditioned"),
    ):
        with pytest.raises(ValueError, match="Cannot change"):
            experiment.train_candidate(
                cache, "test-source", tmp_path / "bad", changed, "cpu", initialize_from=parent
            )
    for key, value in (
        ("latent_mean", torch.ones(8)),
        ("manifest_sha256", "other-manifest"),
        ("vae_sha256", "other-vae"),
        ("text_revision", "other-text"),
    ):
        with pytest.raises(ValueError, match="pairing"):
            experiment.train_candidate(
                {**cache, key: value},
                "test-source",
                tmp_path / "bad",
                config,
                "cpu",
                initialize_from=parent,
            )
    with pytest.raises(ValueError, match="pairing"):
        experiment.train_candidate(
            cache, "other-cache", tmp_path / "bad", config, "cpu", initialize_from=parent
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        experiment.train_candidate(
            cache, "test-source", tmp_path / "bad", config, "cpu", True, parent
        )
    state = load_checkpoint(parent, experiment.KIND)
    for key, value in (("schedule", "wrong"), ("prediction_type", "epsilon")):
        altered = tmp_path / "altered.pt"
        save_checkpoint(altered, {**state, key: value})
        with pytest.raises(ValueError, match="pairing"):
            experiment.train_candidate(
                cache, "test-source", tmp_path / "bad", config, "cpu", initialize_from=altered
            )
    save_checkpoint(tmp_path / "partial.pt", {**state, "batch_cursor": 1})
    with pytest.raises(ValueError, match="completed parent epoch"):
        experiment.train_candidate(
            cache,
            "test-source",
            tmp_path / "bad",
            config,
            "cpu",
            initialize_from=tmp_path / "partial.pt",
        )


def test_fork_continues_parent_shuffle_and_random_sequence(tmp_path):
    cache = fake_cache()
    config = experiment.Config(width=16, batch_size=2, epochs=2)
    experiment.train_candidate(cache, "test-source", tmp_path / "full", config, "cpu")
    experiment.train_candidate(
        cache, "test-source", tmp_path / "parent", replace(config, epochs=1), "cpu"
    )
    experiment.train_candidate(
        cache,
        "test-source",
        tmp_path / "fork",
        replace(config, epochs=1),
        "cpu",
        initialize_from=tmp_path / "parent/best.pt",
    )
    full = load_checkpoint(tmp_path / "full/last.pt", experiment.KIND)
    fork = load_checkpoint(tmp_path / "fork/last.pt", experiment.KIND)
    for key, value in full["model"].items():
        torch.testing.assert_close(value, fork["model"][key], atol=0, rtol=0)
    assert torch.equal(full["torch_rng"], fork["torch_rng"])
