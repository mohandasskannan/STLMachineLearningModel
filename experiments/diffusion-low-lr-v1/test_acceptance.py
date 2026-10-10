"""Acceptance gates cannot approve a candidate with a measured regression."""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "low_lr_runner", Path(__file__).with_name("run_experiment.py")
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def example():
    result = {
        "candidate_evaluations": {
            "parent": {"val": {"velocity_mse": 0.57, "clean_latent_mse": 0.27}},
            "finetuned": {"val": {"velocity_mse": 0.56, "clean_latent_mse": 0.26}},
        },
        "generation": {
            name: {"category_agreement": 0.975, "largest_voxel_component_fraction_mean": 0.966}
            for name in ("parent_g2", "finetuned_g2")
        },
        "high_noise": {
            name: [{"timestep": t, "correct_clean_latent_rmse": 0.93} for t in (100, 190, 199)]
            for name in ("parent", "finetuned")
        },
    }
    verification = {
        "fixed_prompt_mesh_summary": {
            name: {"mesh_components_mean": 4.3} for name in ("parent_g2", "finetuned_g2")
        },
        "reopened_count": 24,
        "all_finite": True,
        "all_scale_correct": True,
        "all_on_build_plane": True,
    }
    return result, verification


def test_accepts_improvement_but_not_equal_parent():
    result, verification = example()
    assert all(runner.acceptance_checks(result, verification).values())
    result["candidate_evaluations"]["finetuned"] = copy.deepcopy(
        result["candidate_evaluations"]["parent"]
    )
    assert not runner.acceptance_checks(result, verification)[
        "validation_velocity_improves_at_least_0.5_percent"
    ]


@pytest.mark.parametrize(
    "failure", ("clean", "high_noise", "category", "fraction", "mesh", "export")
)
def test_rejects_each_regression(failure):
    result, verification = example()
    if failure == "clean":
        result["candidate_evaluations"]["finetuned"]["val"]["clean_latent_mse"] = 0.28
    elif failure == "high_noise":
        result["high_noise"]["finetuned"][1]["correct_clean_latent_rmse"] = 0.94
    elif failure == "category":
        result["generation"]["finetuned_g2"]["category_agreement"] = 0.95
    elif failure == "fraction":
        result["generation"]["finetuned_g2"]["largest_voxel_component_fraction_mean"] = 0.95
    elif failure == "mesh":
        verification["fixed_prompt_mesh_summary"]["finetuned_g2"]["mesh_components_mean"] = 5
    else:
        verification["reopened_count"] = 23
    assert not all(runner.acceptance_checks(result, verification).values())


@pytest.mark.parametrize(
    "numerical,visual,accepted",
    ((False, "passed", False), (True, "failed", False), (True, "passed", True)),
)
def test_visual_review_cannot_bypass_numerical_gates(numerical, visual, accepted):
    result = {
        "acceptance": {
            "checks": {"validation_gate": numerical},
            "numerical_gates_passed": numerical,
        }
    }
    runner.record_visual_review(result, visual, ["Recorded matched-prompt inspection"])
    assert result["acceptance"]["accepted"] is accepted
    assert result["acceptance"]["retained_checkpoint"] == str(
        runner.RUN / "best.pt" if accepted else runner.PARENT
    )


def test_runner_can_be_imported_by_its_filename():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); import run_experiment; "
                "assert hasattr(run_experiment, 'acceptance_checks')"
            ),
            str(Path(__file__).resolve().parent),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
