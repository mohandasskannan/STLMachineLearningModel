"""Fork epoch 80 for bounded low-rate fine-tuning; never promote production weights."""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments/diffusion-high-noise-v1"
sys.path.insert(0, str(PREVIOUS))

import torch
from experiment import Config, evaluate, load_candidate, train_candidate
from verify_results import verify

from stlmodel.common import file_hash, load_checkpoint, write_json

previous_spec = importlib.util.spec_from_file_location(
    "high_noise_review", PREVIOUS / "run_experiment.py"
)
previous = importlib.util.module_from_spec(previous_spec)
previous_spec.loader.exec_module(previous)
CACHE, RUNS = previous.CACHE, previous.RUNS
source_hashes, verify_sources, compare = (
    previous.source_hashes,
    previous.verify_sources,
    previous.review,
)

TRACKED = Path(__file__).resolve().parent
PARENT = RUNS["conditioned"] / "best.pt"
RUN = ROOT / "runs/text2shape-diffusion-low-lr-v1-conditioned"
ARTIFACTS = Path("E:/STLModelData/diffusion-low-lr-v1")


def protected_hashes():
    hashes = source_hashes()
    for directory in RUNS.values():
        for path in directory.iterdir():
            if path.is_file():
                hashes[str(path)] = file_hash(path)
    return hashes


def acceptance_checks(result, verification):
    """Fixed validation-only criteria agreed before the experiment."""
    parent = result["candidate_evaluations"]["parent"]["val"]
    candidate = result["candidate_evaluations"]["finetuned"]["val"]
    pg = result["generation"]["parent_g2"]
    cg = result["generation"]["finetuned_g2"]
    meshes = verification["fixed_prompt_mesh_summary"]
    checks = {
        "validation_velocity_improves_at_least_0.5_percent": candidate["velocity_mse"]
        <= parent["velocity_mse"] * 0.995,
        "validation_clean_latent_error_no_worse": candidate["clean_latent_mse"]
        <= parent["clean_latent_mse"] + 1e-7,
        "category_agreement_no_worse": cg["category_agreement"] >= pg["category_agreement"],
        "largest_voxel_fraction_no_worse": cg["largest_voxel_component_fraction_mean"]
        >= pg["largest_voxel_component_fraction_mean"] - 1e-7,
        "mesh_components_no_worse": meshes["finetuned_g2"]["mesh_components_mean"]
        <= meshes["parent_g2"]["mesh_components_mean"],
        "all_exports_verified": verification["reopened_count"] == 24
        and verification["all_finite"]
        and verification["all_scale_correct"]
        and verification["all_on_build_plane"],
    }
    for timestep in (190, 199):
        errors = {
            name: next(row for row in result["high_noise"][name] if row["timestep"] == timestep)[
                "correct_clean_latent_rmse"
            ]
            for name in ("parent", "finetuned")
        }
        checks[f"high_noise_{timestep}_error_no_worse"] = (
            errors["finetuned"] <= errors["parent"] + 1e-7
        )
    return checks


def write_results(result):
    write_json(TRACKED / "results.json", result)
    write_json(ARTIFACTS / "results.json", result)


def verify_selected_checkpoints(result):
    current = {"parent": file_hash(PARENT), "finetuned": file_hash(RUN / "best.pt")}
    if current != result["selected_checkpoint_hashes"]:
        raise ValueError("Selected checkpoints changed after the matched comparison")


def record_visual_review(result, verdict, notes):
    acceptance = result["acceptance"]
    acceptance["visual_review"] = verdict
    acceptance["visual_observations"] = notes
    acceptance["accepted"] = acceptance["numerical_gates_passed"] and verdict == "passed"
    acceptance["retained_checkpoint"] = str(RUN / "best.pt" if acceptance["accepted"] else PARENT)
    acceptance["rejection_reasons"] = [
        name for name, passed in acceptance["checks"].items() if not passed
    ] + (["visual_regression"] if verdict == "failed" else [])
    if not acceptance["accepted"]:
        result["test_evaluation"] = "not_run_candidate_failed_acceptance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("all", "train", "review", "finalize", "test"), nargs="?", default="all"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Additional epochs, at most 30")
    parser.add_argument(
        "--max-seconds", type=float, default=180, help="Total training budget, at most 180"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--visual-verdict", choices=("passed", "failed"))
    parser.add_argument("--visual-note", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or not 0 < args.max_seconds <= 180:
        parser.error("Epoch and time budgets must be positive and within the experiment limits")
    if args.stage == "finalize" and (args.visual_verdict is None or not args.visual_note):
        parser.error("Finalizing requires a visual verdict and at least one observation")
    torch.set_num_threads(4)
    protected = protected_hashes()
    recorded = ARTIFACTS / "protected-sources.json"
    if recorded.exists() and json.loads(recorded.read_text()) != protected:
        raise ValueError("Protected artifacts differ from the experiment's initial fingerprints")
    cache = load_checkpoint(CACHE, "embeddings")
    verify_sources(cache, protected)
    _, _, parent_state = load_candidate(PARENT, cache, protected[str(CACHE)], args.device)
    if parent_state["epoch"] != 80 or parent_state["config"]["variant"] != "conditioned":
        raise ValueError("Expected the selected epoch-80 conditioned parent")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not recorded.exists():
        write_json(recorded, protected)
    if args.stage in ("all", "train"):
        elapsed = 0.0
        if args.resume:
            elapsed = load_checkpoint(RUN / "last.pt", "diffusion_velocity_experiment")[
                "elapsed_seconds"
            ]
        remaining = args.max_seconds - elapsed
        if remaining > 0:
            config = Config(
                **{
                    **parent_state["config"],
                    "learning_rate": 0.00005,
                    "epochs": args.epochs,
                    "max_seconds": remaining,
                    "patience": 10,
                    "max_steps": 0,
                }
            )
            summary = train_candidate(
                cache,
                protected[str(CACHE)],
                RUN,
                config,
                args.device,
                resume=args.resume,
                initialize_from=None if args.resume else PARENT,
            )
            print(summary, flush=True)
            if protected_hashes() != protected:
                raise ValueError("Protected artifacts changed during training")
            if summary["stop_reason"] == "interrupted":
                return
        else:
            print({"training_skipped": "total_time_budget_exhausted"}, flush=True)
    if args.stage in ("all", "review"):
        result = compare(
            cache,
            protected[str(CACHE)],
            args.device,
            candidate_runs={"parent": PARENT.parent, "finetuned": RUN},
            artifact_dir=ARTIFACTS,
            include_legacy=False,
            guidance_scales=(2.0,),
            evaluate_test=False,
        )
        result["protected_source_hashes"] = protected
        result["protected_artifacts_unchanged"] = protected_hashes() == protected
        if not result["protected_artifacts_unchanged"]:
            raise ValueError("Protected artifacts changed during review")
        result["production_model_promoted"] = False
        result["selected_checkpoint_hashes"] = {
            "parent": file_hash(PARENT),
            "finetuned": file_hash(RUN / "best.pt"),
        }
        verification = verify(result, TRACKED, ARTIFACTS, ("parent_g2", "finetuned_g2"))
        checks = acceptance_checks(result, verification)
        result["acceptance"] = {
            "checks": checks,
            "numerical_gates_passed": all(checks.values()),
            "visual_review": "pending",
            "accepted": False,
            "retained_checkpoint": str(PARENT),
        }
        result["test_evaluation"] = "not_run_pending_acceptance"
        write_results(result)
        shutil.copy2(ARTIFACTS / "comparison.png", TRACKED / "comparison.png")
        for filename in ("config.json", "summary.json", "metrics.jsonl"):
            shutil.copy2(RUN / filename, TRACKED / filename)
        print({"acceptance": result["acceptance"]}, flush=True)
    if args.stage == "finalize":
        result = json.loads((TRACKED / "results.json").read_text())
        verify_selected_checkpoints(result)
        record_visual_review(result, args.visual_verdict, args.visual_note)
        write_results(result)
        print({"acceptance": result["acceptance"]}, flush=True)
    if args.stage == "test":
        result = json.loads((TRACKED / "results.json").read_text())
        verify_selected_checkpoints(result)
        if (
            not result["acceptance"]["numerical_gates_passed"]
            or result["acceptance"]["visual_review"] != "passed"
        ):
            raise ValueError("Test evaluation requires passed numerical and visual acceptance")
        for name, path in (("parent", PARENT), ("finetuned", RUN / "best.pt")):
            model, schedule, _ = load_candidate(path, cache, protected[str(CACHE)], args.device)
            result["candidate_evaluations"][name]["test"] = evaluate(
                model, cache["test"], schedule, args.device
            )
        result["test_evaluation"] = "reported_after_validation_selection"
        write_results(result)


if __name__ == "__main__":
    main()
