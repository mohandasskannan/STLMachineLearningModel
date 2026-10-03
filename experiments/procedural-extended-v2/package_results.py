"""Package experiment metadata and verify all generated meshes after training."""
import json
import shutil
from pathlib import Path
import numpy as np
from stlmodel.common import write_json

root = Path("outputs/procedural-extended-v2")
run = Path("runs/procedural-diffusion-extended-v2")
dest = Path("experiments/procedural-extended-v2")
r = json.loads((root / "results.json").read_text())
checks = []
for stage in ("before", "after", "last"):
    stage_checks = json.loads((root / stage / "reopened-mesh-checks.json").read_text())
    assert len(stage_checks) == 12
    for c in stage_checks:
        assert c["finite"] and c["watertight"] and np.isclose(max(c["extents_mm"]), 100, atol=1e-4), c
    checks.extend(stage_checks)
    for path in (root / stage).iterdir():
        if path.suffix in (".json", ".png"):
            target = dest / stage / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
for name in ("results.json", "command.json", "run_extended.py", "package_results.py"):
    shutil.copy2(root / name, dest / name)
(dest / "training").mkdir(exist_ok=True)
for name in ("config.json", "summary.json"):
    shutil.copy2(run / name, dest / "training" / name)
selected = r["extended"]["step"]
final_step = r["final"]["step"]
with (run / "metrics.jsonl").open() as source, (dest / "training" / "validation-sampled.jsonl").open("w") as target:
    for line in source:
        event = json.loads(line)
        step = event["step"]
        if step % 100 == 0 or step in (selected, r["initial_step"], final_step):
            target.write(line)
write_json(dest / "verification.json", {"reopened_mesh_count":len(checks), "finite":True, "watertight":True, "longest_dimension_mm":100, "selected_checkpoint_unchanged":r["baseline"]["sha256"] == r["extended"]["sha256"]})
print(json.dumps({"training_seconds":r["training_summary"]["session_seconds"], "additional_steps":final_step-r["initial_step"], "total_steps":final_step, "selected_step":selected, "final_metrics":r["final"]["metrics"], "final_geometry":r["final_geometry"], "verified_meshes":len(checks)}, indent=2))
