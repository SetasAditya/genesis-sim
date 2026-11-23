"""Evaluate a controller once per explicit rack layout."""

import argparse
import importlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def run(args):
    import gymnasium as gym
    import wetlab_isaaclab.tasks
    from wetlab_isaaclab.tasks.pipette_evaluation import score_rack_layouts
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout

    layouts = json.loads(args.layouts.read_text())
    if not layouts:
        raise ValueError("The evaluation needs at least one layout")
    module_name, factory_name = args.controller.split(":", 1)
    module = importlib.import_module(module_name)
    env = gym.make(f"Wetlab-Pipette-Rack-{args.robot.title()}-v0",
                   assets=args.assets, device=args.device, num_envs=len(layouts),
                   layout=PipetteLayout(randomize=False), seed=args.seed)
    try:
        controller_spec = {
            "joint_names": env.joint_names,
            "observation_slices": env.observation_slices.copy(),
            "action_low": env.single_action_space.low.copy(),
            "action_high": env.single_action_space.high.copy(),
            "control_dt": env.step_dt,
        }
        rows = score_rack_layouts(env, layouts, seed=args.seed,
                                  controller_factory=lambda: getattr(module, factory_name)(controller_spec))
    finally:
        env.close()
    metrics = ("lift", "valid", "fixture_stable", "timeout", "goal_reached", "aligned", "sustained_press", "success")
    counts = {name: sum(row[name] for row in rows) for name in metrics}
    rates = {name: count / len(rows) for name, count in counts.items()}
    summary = dict(episodes=len(rows), robot=args.robot, seed=args.seed,
                   counts=counts, rates=rates, layouts=rows)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--controller", required=True, metavar="MODULE:FACTORY")
    parser.add_argument("--robot", choices=("mounted", "floating"), default="mounted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        run(args)
    finally:
        launcher.app.close()
