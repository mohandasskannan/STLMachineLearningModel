"""Guard changed-data initialization, optimizer reset, and exact resumption."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from stlmodel.common import file_hash, load_checkpoint

DIRECTORY = Path(__file__).resolve().parents[1] / "diffusion-high-noise-v1"
spec = importlib.util.spec_from_file_location(
    "data_transfer_experiment", DIRECTORY / "experiment.py"
)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def cache():
    torch.manual_seed(33)
    return {
        "train": {"latents": torch.randn(6, 8), "text": torch.randn(6, 384)},
        "val": {"latents": torch.randn(4, 8), "text": torch.randn(4, 384)},
        "latent_mean": torch.zeros(8),
        "latent_std": torch.ones(8),
        "vae_sha256": "frozen-vae",
        "manifest_sha256": "old-data",
        "text_model": "frozen-text",
        "text_revision": "frozen-revision",
    }


def test_changed_data_transfer_resets_optimizer_and_resumes_exactly(tmp_path):
    torch.set_num_threads(2)
    original = cache()
    config = experiment.Config(
        variant="conditioned", width=16, batch_size=2, text_dropout=0.15, epochs=1
    )
    experiment.train_candidate(original, "old-source", tmp_path / "parent", config, "cpu")
    parent_path = tmp_path / "parent/best.pt"
    protected = file_hash(parent_path)
    parent = load_checkpoint(parent_path)
    expanded = {
        **original,
        "manifest_sha256": "new-data",
        "train": {"latents": original["train"]["latents"] + 0.1, "text": original["train"]["text"]},
    }
    settings = replace(config, epochs=3, learning_rate=0.00005)
    experiment.train_candidate(
        expanded,
        "new-source",
        tmp_path / "anchor",
        replace(settings, max_seconds=1e-9),
        "cpu",
        warm_start_from=parent_path,
        warm_start_source=(original, "old-source"),
    )
    anchor = load_checkpoint(tmp_path / "anchor/best.pt")
    assert anchor["optimizer"]["state"] == {}
    assert anchor["source_sha256"] == "new-source"
    assert anchor["manifest_sha256"] == "new-data"
    assert anchor["parent"]["mode"] == "new_data_weights_only"
    assert anchor["step"] == anchor["epoch"] == 0
    for key, value in parent["model"].items():
        torch.testing.assert_close(anchor["model"][key], value, rtol=0, atol=0)
    for name, steps in (("full", 4), ("split", 2)):
        experiment.train_candidate(
            expanded,
            "new-source",
            tmp_path / name,
            replace(settings, max_steps=steps),
            "cpu",
            warm_start_from=parent_path,
            warm_start_source=(original, "old-source"),
        )
    experiment.train_candidate(
        expanded,
        "new-source",
        tmp_path / "split",
        replace(settings, max_steps=4),
        "cpu",
        resume=True,
    )
    full, resumed = (load_checkpoint(tmp_path / name / "last.pt") for name in ("full", "split"))
    for key, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    assert file_hash(parent_path) == protected
    with pytest.raises(ValueError, match="pairing"):
        experiment.train_candidate(
            expanded,
            "new-source",
            tmp_path / "invalid",
            settings,
            "cpu",
            initialize_from=parent_path,
        )
    with pytest.raises(ValueError, match="pairing"):
        experiment.train_candidate(
            expanded, "wrong-source", tmp_path / "split", settings, "cpu", resume=True
        )


@pytest.mark.parametrize("key", ["latent_mean", "latent_std", "vae_sha256", "text_revision"])
def test_transfer_rejects_changed_latent_coordinates(tmp_path, key):
    original = cache()
    config = experiment.Config(variant="conditioned", width=16, batch_size=2, epochs=1)
    experiment.train_candidate(original, "old-source", tmp_path / "parent", config, "cpu")
    changed = {**original, "manifest_sha256": "new-data"}
    changed[key] = original[key] + 1 if torch.is_tensor(original[key]) else "different"
    with pytest.raises(ValueError, match="Warm start changes"):
        experiment.train_candidate(
            changed,
            "new-source",
            tmp_path / "candidate",
            config,
            "cpu",
            warm_start_from=tmp_path / "parent/best.pt",
            warm_start_source=(original, "old-source"),
        )
