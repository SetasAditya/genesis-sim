"""Single launcher for wet-lab scene inspection, validation and robot trials."""

import argparse
import json
import sys
import traceback
import hashlib
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
p = argparse.ArgumentParser()
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
    default="petri",
)
p.add_argument(
    "--mode",
    choices=["interactive", "scripted", "inspect", "validate"],
    default="inspect",
)
p.add_argument("--petri-mode", choices=["open", "close"], default="open")
p.add_argument("--headless", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=2400)
p.add_argument("--trials", type=int, default=1)
p.add_argument("--jitter-m", type=float, default=0.0)
p.add_argument("--reset-cycles", type=int, default=20)
p.add_argument("--figures", action="store_true")
p.add_argument("--build-only", action="store_true")
p.add_argument("--volume-ul", type=float, default=100.0)
p.add_argument(
    "--insert-collision",
    choices=["nominal", "clearance"],
    default="clearance",
    help="Clearance retains CAD visuals with an explicitly estimated 19.55 mm collider",
)
args = p.parse_args()
if (
    args.trials < 1
    or args.reset_cycles < 1
    or args.steps < 0
    or not 0.0 <= args.jitter_m <= 0.005
):
    p.error(
        "trials/reset-cycles must be positive; steps nonnegative; jitter-m in [0,.005]"
    )
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1600,
        "height": 1000,
        "multi_gpu": False,
        "active_gpu": 0,
        "physics_gpu": 0,
    }
)
from wetlab_isaaclab.sandbox.environment import WetLabSandbox
from wetlab_isaaclab.sandbox.scripted import ScriptedTrial
from wetlab_isaaclab.sandbox.config import (
    Action,
    PHYSICS_DT,
    TASK_IDS,
    STATIONS,
    BENCH_Z,
)
import numpy as np

env = None
exit_code = 0


def capture(env, label, position, look_at, focal_length=18.0):
    import cv2
    import omni.replicator.core as rep

    cam = rep.create.camera(
        position=position, look_at=look_at, focal_length=focal_length
    )
    product = rep.create.render_product(cam, (1600, 1000))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    for _ in range(8):
        env.world.render()
    rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
    data = rgb.get_data()
    if data is None or not data.size:
        raise RuntimeError("No rendered image")
    path = env.output / (label + ".png")
    cv2.imwrite(str(path), cv2.cvtColor(np.array(data)[:, :, :3], cv2.COLOR_RGB2BGR))
    rgb.detach()
    product.destroy()
    return str(path)


try:
    initial_task = "petri" if args.task == "all" else args.task
    env = WetLabSandbox(
        ROOT,
        initial_task,
        args.petri_mode,
        args.seed,
        render=not args.headless,
        insert_collision=args.insert_collision,
    )
    source_files = sorted((ROOT / "source/wetlab_isaaclab/sandbox").glob("*.py"))
    digest = hashlib.sha256()
    for file in source_files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    run_metadata = {
        "arguments": vars(args),
        "source_sha256": digest.hexdigest(),
        "versions": {
            n: importlib.metadata.version(n) for n in ("isaacsim", "numpy", "scipy")
        },
        "physics_dt_s": PHYSICS_DT,
        "control_dt_s": 1 / 30,
        "physics_device": "cpu",
    }
    (env.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    if not 0.0 < args.volume_ul <= 200.0:
        raise ValueError("Requested pipette volume must be in (0, 200] microlitres")
    env.volume.requested = args.volume_ul
    print("ENVIRONMENT_READY", args.task, flush=True)
    env.save()
    reports = []
    if args.mode == "scripted":
        for task in TASK_IDS if args.task == "all" else [args.task]:
            for mode in (
                ["open", "close"]
                if args.task == "all" and task == "petri"
                else [args.petri_mode]
            ):
                for i in range(args.trials):
                    env.reset(args.seed + i, task, mode, jitter_m=args.jitter_m)
                    trial = ScriptedTrial(env)
                    report = trial.run()
                    reports.append(report)
                    print("TRIAL", json.dumps(report), flush=True)
                    (env.output / "trial_progress.json").write_text(
                        json.dumps({"reports": reports}, indent=2)
                    )
    elif args.mode == "validate":
        for task in TASK_IDS if args.task == "all" else [args.task]:
            reset_errors = []
            for i in range(args.reset_cycles):
                obs, _ = env.reset(args.seed + i, task, args.petri_mode)
                if (
                    obs["tools"]["attached"]
                    or obs["pipette_volume"]["target_ul"] != 0.0
                ):
                    reset_errors.append(i)
                if not all(
                    np.isfinite(o["position"]).all() for o in obs["objects"].values()
                ):
                    reset_errors.append(i)
            start = env.object_states()
            base = env.robot.get_world_pose()[0].copy()
            for _ in range(args.steps):
                env.world.step(render=False)
            end = env.object_states()
            finish = env.robot.get_world_pose()[0]
            finite = (
                all(np.isfinite(o.position).all() for o in end.values())
                and np.isfinite(finish).all()
            )
            drift = float(np.linalg.norm((finish - base)[:2]))
            openings = env.check_openings()
            escaped = [
                n
                for n, o in end.items()
                if n not in ["fridge_door", "plunger"] and o.position[2] < 0.05
            ]
            report = {
                "task": task,
                "seconds": args.steps * PHYSICS_DT,
                "reset_cycles": args.reset_cycles,
                "reset_errors": reset_errors,
                "finite": bool(finite),
                "base_horizontal_drift_m": drift,
                "escaped_objects": escaped,
                "rack_openings": openings,
                "passed": bool(
                    finite
                    and not reset_errors
                    and drift < 0.05
                    and not escaped
                    and all(c["passed"] for c in openings)
                ),
            }
            reports.append(report)
            print("VALIDATION", json.dumps(report), flush=True)
            (env.output / "stability_progress.json").write_text(
                json.dumps({"reports": reports}, indent=2)
            )
    elif args.mode == "interactive":
        if args.headless:
            raise ValueError(
                "Interactive mode needs a display; use scripted/inspect for headless runs"
            )
        import omni.ui as ui

        window = ui.Window("Wet-lab sandbox", width=430, height=820)
        paused = [True]
        run_trial = [False]
        selected = [args.task]
        base_velocity = [(0.0, 0.0, 0.0)]
        env.world.pause()
        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Galbot wet-lab sandbox")
                task_model = ui.ComboBox(TASK_IDS.index(initial_task), *TASK_IDS)
                status = ui.Label("Paused")

                def reset():
                    selected[0] = TASK_IDS[
                        task_model.model.get_item_value_model().as_int
                    ]
                    env.world.play()
                    env.reset(args.seed, selected[0], args.petri_mode)
                    if paused[0]:
                        env.world.pause()

                def toggle_pause():
                    paused[0] = not paused[0]
                    if paused[0]:
                        env.world.pause()
                    else:
                        env.world.play()

                def single_step():
                    env.world.play()
                    env.step(Action())
                    env.world.pause()
                    paused[0] = True

                def command(action):
                    env.controller.command(action, 1 / 30)

                ui.Button("Reset selected task", clicked_fn=reset)
                ui.Button("Pause / run", clicked_fn=toggle_pause)
                ui.Button("Single step", clicked_fn=single_step)
                ui.Button(
                    "Run scripted trial",
                    clicked_fn=lambda: run_trial.__setitem__(0, True),
                )
                arm = ui.ComboBox(0, "right", "left")
                xyz = [ui.FloatField() for _ in range(3)]

                def read_wrist():
                    for field, value in zip(
                        xyz,
                        env.controller.tcp(
                            "right"
                            if arm.model.get_item_value_model().as_int == 0
                            else "left"
                        )[:3, 3],
                    ):
                        field.model.set_value(float(value))

                ui.Button("Read wrist XYZ (metres)", clicked_fn=read_wrist)
                read_wrist()

                def send_wrist():
                    side = (
                        "right"
                        if arm.model.get_item_value_model().as_int == 0
                        else "left"
                    )
                    command(
                        Action(
                            wrist_targets={
                                side: {"position": [f.model.as_float for f in xyz]}
                            }
                        )
                    )

                ui.Button("Command wrist target", clicked_fn=send_wrist)
                ui.Button(
                    "Open grippers",
                    clicked_fn=lambda: command(
                        Action(grippers={"left": 0.0, "right": 0.0})
                    ),
                )
                ui.Button(
                    "Close grippers",
                    clicked_fn=lambda: command(
                        Action(grippers={"left": 1.6, "right": 1.6})
                    ),
                )
                ui.Button(
                    "Toggle magnet",
                    clicked_fn=lambda: setattr(
                        env, "magnetic_enabled", not env.magnetic_enabled
                    ),
                )
                ui.Label("Base body-frame vx, vy (m/s), yaw (rad/s)")
                velocity_fields = [ui.FloatField() for _ in range(3)]

                def apply_velocity():
                    v = tuple(f.model.as_float for f in velocity_fields)
                    if max(abs(v[0]), abs(v[1])) > 0.15 or abs(v[2]) > 0.3:
                        status.text = "Velocity limit: 0.15 m/s, 0.3 rad/s"
                        return
                    base_velocity[0] = v

                ui.Button("Apply base velocity", clicked_fn=apply_velocity)
                ui.Button(
                    "Stop base",
                    clicked_fn=lambda: base_velocity.__setitem__(0, (0.0, 0.0, 0.0)),
                )
        while app.is_running():
            if run_trial[0]:
                run_trial[0] = False
                env.world.play()
                reports.append(ScriptedTrial(env).run())
                if paused[0]:
                    env.world.pause()
            if not paused[0]:
                env.step(Action(base_velocity=base_velocity[0]))
            else:
                env.world.render()
            status.text = (
                json.dumps(env.evaluator.result())
                + "\nTool: "
                + str(env.attachments)
                + "\nVolume: "
                + str(env.volume.observation())
            )
    else:
        if not args.build_only:
            for _ in range(args.steps):
                env.world.step(render=False)
    observation = env.observe()
    if args.figures:
        overview = capture(env, "overview", (-3.8, 3.90, 2.50), (-3.85, 2.8, 0.70), 4.8)
        for task in STATIONS if args.task == "all" else [args.task]:
            x, y = STATIONS[task]
            if task == "fridge_retrieval":
                position = (x + 1.05, 1.10, 1.50)
                look = (x, 2.45, 0.70)
                focal = 22.0
            else:
                extent = 1.0 if task == "reactor_assembly" else 0.60
                position = (x + extent, y - 0.60, 1.45)
                look = (x, y, BENCH_Z + 0.05)
                focal = 24.0
            figure = capture(env, task + "_" + args.petri_mode, position, look, focal)
            print("FIGURES", overview, figure, flush=True)
    groups = {}
    for report in reports:
        key = report["task"] + ":" + report.get("mode", "default")
        groups.setdefault(key, []).append(bool(report["passed"]))
    qualified = {
        key: sum(values) >= int(np.ceil(0.9 * len(values)))
        if args.mode == "scripted"
        else all(values)
        for key, values in groups.items()
    }
    overall = bool(qualified) and all(qualified.values())
    (
        env.output / (args.task + "_" + args.petri_mode + "_" + args.mode + ".json")
    ).write_text(
        json.dumps(
            {
                "reports": reports,
                "debug_used": env.debug_used,
                "qualified_scenarios": qualified,
                "overall_passed": overall,
            },
            indent=2,
            allow_nan=False,
        )
    )
    (env.output / (args.task + "_observation.json")).write_text(
        json.dumps(observation, indent=2, allow_nan=False)
    )
    if reports:
        if not overall:
            exit_code = 2
except Exception as error:
    traceback.print_exc()
    out = ROOT / "outputs/sandbox"
    out.mkdir(parents=True, exist_ok=True)
    (out / (args.task + "_runtime_error.json")).write_text(
        json.dumps(
            {"passed": False, "error": str(error), "traceback": traceback.format_exc()},
            indent=2,
        )
    )
    exit_code = 1
finally:
    if env:
        env.close()
    app.close()
raise SystemExit(exit_code)
