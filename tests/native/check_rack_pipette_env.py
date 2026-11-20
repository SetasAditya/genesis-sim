"""Exercise registered rack environments and measure control throughput."""

import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher


def check(args):
    import gymnasium as gym
    import torch
    import wetlab_isaaclab.tasks
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout

    env = gym.make(f"Wetlab-Pipette-Rack-{args.robot.title()}-v0",
                   assets=args.assets, device=args.device, num_envs=args.num_envs,
                   layout=PipetteLayout(randomize=False))
    try:
        obs, _ = env.reset(seed=41)
        joint = env.joint_names.index("R_arm_j7" if args.robot == "mounted" else "root_x_axis_joint")
        pose = env.observation_slices["pipette_pose"]
        positions = env.observation_slices["robot_joint_pos"]
        home = obs["policy"][:, positions].clone()
        target = home.clone()
        target[:, joint] += 0.02
        assert env.action_space.shape == target.shape
        assert env.observation_space.shape == obs["policy"].shape
        start = obs["policy"].clone()
        clock = env.sim.get_physics_step_count()
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(target)
        assert env.sim.get_physics_step_count() - clock == 50
        assert not (terminated | truncated).any()
        assert info["events"]["valid"].all()
        assert not reward.any()

        moved = obs["policy"][:, positions][:, joint] - home[:, joint]
        assert (moved.abs() > 1e-3).all()
        displacement = (obs["policy"][:, pose][:, :3] - start[:, pose][:, :3]).norm(dim=1)
        before = obs["policy"].clone()
        clock = env.sim.get_physics_step_count()
        reset = env.reset_indices([0])["policy"]
        assert env.sim.get_physics_step_count() == clock
        torch.testing.assert_close(reset[1:], before[1:], atol=0, rtol=0)
        torch.testing.assert_close(reset[0, positions], home[0], atol=1e-6, rtol=0)

        if env.num_envs == 2:
            pair = [{"rack": env.layout.rack, "tube": env.layout.tube},
                    {"rack": env.layout.rack, "tube": (*env.layout.tube[:1],
                                                       env.layout.tube[1] + 0.01,
                                                       *env.layout.tube[2:])}]
            obs, _ = env.reset(options={"layouts": pair})
            tube = obs["policy"][:, env.observation_slices["tube_pose"]]
            torch.testing.assert_close(tube[1, 1] - tube[0, 1], target.new_tensor(0.01))

        env.reset()
        if "cuda" in env.device:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(target)
        if "cuda" in env.device:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        assert torch.isfinite(obs["policy"]).all()
        assert not (terminated | truncated).any()
        result = dict(robot=args.robot, worlds=args.num_envs, control_steps=args.steps,
                      seconds=seconds, batch_control_steps_s=args.steps / seconds,
                      world_steps_s=args.num_envs * args.steps / seconds,
                      joint_delta=moved.tolist(), pipette_displacement=displacement.tolist())
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.check_evaluator:
            from wetlab_isaaclab.tasks.pipette_evaluation import score_rack_layouts

            layouts = [dict(id=f"nominal-{index}", rack=env.layout.rack, tube=env.layout.tube)
                       for index in range(env.num_envs)]
            rows = score_rack_layouts(env, layouts, controller_factory=lambda: (lambda obs, mask: home),
                                      planning="development-only", seed=41)
            assert all(row["valid"] and row["fixture_stable"] and not row["success"] for row in rows)
            assert all(row["timeout"] and row["control_steps"] == 500 for row in rows)
            result["nominal_no_motion"] = rows
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("RACK_ENV_PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("mounted", "floating"), default="mounted")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--check-evaluator", action="store_true")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        check(args)
    finally:
        launcher.app.close()
