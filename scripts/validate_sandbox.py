"""Run the acceptance matrix and preserve every result, including failed trials."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--trials", type=int, default=10)
p.add_argument(
    "--task",
    choices=[
        "all",
        "petri",
        "tube_insertion",
        "reactor_assembly",
        "fridge_retrieval",
        "pipetting",
    ],
    default="all",
)
p.add_argument("--scope", choices=["assets", "robot"], default="assets")
p.add_argument("--skip-scripted", action="store_true")
a = p.parse_args()
OUT = ROOT / "outputs/sandbox"
OUT.mkdir(parents=True, exist_ok=True)
tasks = (
    ["petri", "tube_insertion", "reactor_assembly", "fridge_retrieval", "pipetting"]
    if a.task == "all"
    else [a.task]
)
results = []
env = os.environ.copy()
env["OMNI_KIT_ACCEPT_EULA"] = "YES"
env["PYTHONPATH"] = str(ROOT / "source") + os.pathsep + env.get("PYTHONPATH", "")


def run(label, argv, timeout=1800):
    path = OUT / (label + ".log")
    started = time.time()
    with path.open("w") as out:
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_sandbox.py"), *argv],
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            entry = {
                "check": label,
                "passed": False,
                "failure_reason": "process_timeout",
                "log": str(path),
            }
        else:
            task = argv[argv.index("--task") + 1]
            mode = argv[argv.index("--mode") + 1]
            petri = (
                argv[argv.index("--petri-mode") + 1]
                if "--petri-mode" in argv
                else "open"
            )
            report = OUT / (task + "_" + petri + "_" + mode + ".json")
            valid = (
                report.exists()
                and report.stat().st_mtime >= started
                and json.loads(report.read_text()).get("overall_passed", False)
            )
            entry = {
                "check": label,
                "passed": bool(result.returncode == 0 and valid),
                "exit_code": result.returncode,
                "log": str(path),
                "report": str(report),
            }
    results.append(entry)
    print(json.dumps(entry), flush=True)
    (OUT / "acceptance.json").write_text(
        json.dumps({"passed": False, "checks": results}, indent=2)
    )


for task in tasks:
    run(
        task + "_stability",
        [
            "--headless",
            "--task",
            task,
            "--mode",
            "validate",
            "--steps",
            "2400",
            "--reset-cycles",
            "20",
        ],
    )
    if a.scope == "robot" and not a.skip_scripted:
        for mode in ["open", "close"] if task == "petri" else ["open"]:
            run(
                task + "_" + mode + "_trials",
                [
                    "--headless",
                    "--task",
                    task,
                    "--mode",
                    "scripted",
                    "--petri-mode",
                    mode,
                    "--trials",
                    str(a.trials),
                    "--jitter-m",
                    ".001",
                    "--figures",
                ],
                timeout=7200,
            )
for label, module, report_name in [
    ("fixture_fits", "fixtures", "fixture_fits.json"),
    ("export", "export", "export_validation.json"),
]:
    started = time.time()
    log = OUT / (label + ".log")
    with log.open("w") as output:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "wetlab_isaaclab.sandbox.checks." + module],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
            path = OUT / report_name
            passed = (
                completed.returncode == 0
                and path.exists()
                and path.stat().st_mtime >= started
                and json.loads(path.read_text()).get("passed", False)
            )
            results.append(
                {
                    "check": label,
                    "passed": bool(passed),
                    "log": str(log),
                    "report": str(path),
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {"check": label, "passed": False, "failure_reason": "process_timeout"}
            )
checks_passed = bool(results) and all(r["passed"] for r in results)
complete_matrix = a.task == "all" and (
    a.scope == "assets" or (not a.skip_scripted and a.trials >= 10)
)
export_path = OUT / "export_validation.json"
export_passed = export_path.exists() and json.loads(export_path.read_text()).get(
    "passed", False
)
report = {
    "passed": bool(checks_passed and complete_matrix and export_passed),
    "scope": a.scope,
    "robot_execution_required": a.scope == "robot",
    "selected_checks_passed": checks_passed,
    "complete_task_matrix": complete_matrix,
    "export_smoke_passed": bool(export_passed),
    "checks": results,
    "requested_trials": a.trials,
}
(OUT / "acceptance.json").write_text(json.dumps(report, indent=2))
sys.exit(0 if report["passed"] else 2)
