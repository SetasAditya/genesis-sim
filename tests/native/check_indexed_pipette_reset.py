"""Reset eight seated pipettes repeatedly while two other mounted robots keep moving."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


def angle_deg(a, b):
    """Shortest orientation distance for either sign of the native xyzw quaternion."""
    import torch

    a = torch.nn.functional.normalize(a.double(), dim=-1)
    b = torch.nn.functional.normalize(b.double(), dim=-1)
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    return (4 * torch.atan2((a - b).norm(dim=-1), (a + b).norm(dim=-1))).rad2deg()


def check(assets, holder, output):
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from wetlab_isaaclab.robots.dg5f import dg5f_robot_cfg
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout, default_robot_base_pose
    from wetlab_isaaclab.tasks.pipette_placement import screen_pipette_layouts
    from wetlab_isaaclab.tasks.pipette_scene import cabinet_pipette_scene_cfg, reset_cabinet_pipette
    from wetlab_isaaclab.tasks.pipette_task import CabinetPipetteTask

    def usd(name):
        return next((assets / name).glob("*/*.usda"))

    seed = 41
    nominal = PipetteLayout(holder=holder)
    base = default_robot_base_pose(robot="mounted", holder=holder)
    screened = screen_pipette_layouts(nominal, base_position=base[:3], base_rotation=base[3:], seed=seed)
    sim = SimulationContext(SimulationCfg(
        device="cuda:0", dt=.01, physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=2.),
    ))
    cfg = cabinet_pipette_scene_cfg(
        robot=dg5f_robot_cfg(usd("mounted"), position=base[:3], orientation=base[3:]),
        pipette_usd=usd("pipette"), tube_usd=usd("tube"), layout=nominal, num_envs=10, env_spacing=3.,
    )
    fixture = "Rack" if holder == "rack" else "RailFrame"
    cfg.seat_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Pipette/Geometry/pipette_body",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + fixture],
        track_friction_forces=True, max_contact_data_count_per_prim=128,
    )
    scene = InteractiveScene(cfg)
    sim.reset()
    task = CabinetPipetteTask(scene, control_dt=.05)
    tool, robot = scene["pipette"], scene["robot"]
    assert tool.root_view.max_shapes == 11
    device = scene.device
    all_ids = torch.arange(10, device=device)
    reset_ids = all_ids[[0, 1, 3, 4, 5, 6, 8, 9]]
    continuing = all_ids[[2, 7]]
    rng = torch.Generator(device=device).manual_seed(seed)
    reference_rng = torch.Generator(device=device).manual_seed(seed)
    joint = robot.joint_names.index("R_arm_j7")
    target = robot.data.default_joint_pos.torch.clone()
    signed_offset = target.new_tensor([.02, -.02])
    # Clean releases in these ten world origins reach 0.10 degrees on the rack and 0.26 on the rail.
    angle_limit = .12 if holder == "rack" else .27
    report = dict(holder=holder, seed=seed, reset_ids=reset_ids.tolist(), continuing_ids=continuing.tolist(),
                  max_drift_limit_mm=.4, final_angle_limit_deg=angle_limit, rounds=[])

    def ongoing_state():
        state = {}
        for name, asset in {**scene.articulations, **scene.rigid_objects}.items():
            for field in ("root_link_pose_w", "root_link_vel_w"):
                state[name, field] = getattr(asset.data, field).torch[continuing].clone()
        for name, asset in scene.articulations.items():
            for field in ("joint_pos", "joint_vel"):
                state[name, field] = getattr(asset.data, field).torch[continuing].clone()
        return state

    def assert_untouched(before):
        after = ongoing_state()
        for key in before:
            assert torch.equal(before[key], after[key]), f"Indexed reset overwrote {key} in continuing worlds"

    def step():
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(.01)

    def reset(ids, layout, *, check_continuing=True, settle_seat=True):
        requested = layout.sample(len(ids), device=device, generator=reference_rng)
        before = ongoing_state() if check_continuing else None
        clock = sim.get_physics_step_count()
        reset_cabinet_pipette(scene, ids, layout, generator=rng, settle_seat=settle_seat)
        assert sim.get_physics_step_count() == clock, "Reset advanced the physics clock"
        if before is not None:
            assert_untouched(before)
        task.reset(ids)
        obs = task.observe()
        for name, pose in requested.items():
            native = scene[name].data.root_link_pose_w.torch if name == "rack" else obs[name + "_pose"]
            expected = pose[:, (0, 1, 2, 4, 5, 6, 3)].clone()
            expected[:, :3] += scene.env_origins[ids]
            torch.testing.assert_close(native[ids, :3], expected[:, :3], atol=5e-7, rtol=0)
            assert (angle_deg(native[ids, 3:], expected[:, 3:]) < 5e-5).all(), name
        return requested

    for cycle in range(4):
        ids = reset_ids if cycle else all_ids
        layout = screened if cycle else replace(nominal, randomize=False)
        requested = reset(ids, layout, check_continuing=bool(cycle))
        start = tool.data.root_link_pose_w.torch[ids].clone()
        q_before = robot.data.joint_pos.torch[continuing, joint].clone()
        states, loads, valid = [], [], []
        for tick in range(650):
            # Reverse the continuing wrists just before the next reset. Their command
            # must survive the reset and keep moving both wrists in opposite directions.
            if tick == 649:
                target[continuing, joint] = robot.data.default_joint_pos.torch[continuing, joint] + signed_offset * (-1)**cycle
                robot.actuators.target_command.set_position_index(value=target)
            step()
            states.append(tool.data.root_link_pose_w.torch[ids].clone())
            contact = scene["seat_contact"].data
            loads.append((contact.normal_force_matrix_w.torch + contact.friction_force_matrix_w.torch)[:, 0, 0, 2].clone())
            if (tick + 1) % 5 == 0:
                valid.append(task.update()["valid"].clone())
            if tick == 49:
                progress = (robot.data.joint_pos.torch[continuing, joint] - q_before) * signed_offset.sign() * (-1)**(cycle - 1)

        poses, force = torch.stack(states), torch.stack(loads)
        drift = (1000 * (poses[..., :3] - start[:, :3]).norm(dim=-1)).amax(0)
        first_step_drift = 1000 * (poses[0, :, :3] - start[:, :3]).norm(dim=-1)
        angle = angle_deg(poses[-1, :, 3:], start[:, 3:])
        tail_load = (force[-100:] > .1).sum(0)
        row = dict(cycle=cycle, world_ids=ids.tolist(), requested_local_wxyz={k: v.tolist() for k, v in requested.items()},
                   first_step_drift_mm=first_step_drift.tolist(), first_step_support_n=force[0, ids].tolist(),
                   max_drift_mm=drift.tolist(), final_angle_deg=angle.tolist(), final100_load=tail_load.tolist(),
                   finite=bool(torch.isfinite(poses).all() & torch.isfinite(force).all()),
                   valid_endpoint=valid[-1].tolist(), valid_samples=torch.stack(valid).sum(0).tolist(),
                   continuing_directed_motion_rad=progress.tolist() if cycle else None)
        row["passed"] = bool(row["finite"] and all(row["valid_endpoint"]) and (tail_load == 100).all()
                             and (first_step_drift <= .001).all() and (force[0, ids].abs() < .001).all()
                             and (drift <= .4).all() and (angle <= angle_limit).all()
                             and (not cycle or (progress > .005).all()))
        report["rounds"].append(row)
        print(f"{holder} cycle {cycle}: max drift {float(drift.max()):.4f} mm, final rotation "
              f"{float(angle.max()):.4f} deg, passed={row['passed']}", flush=True)
        if output:
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failed = [row["cycle"] for row in report["rounds"] if not row["passed"]]
    assert not failed, f"{holder} retention or continuing motion failed in cycles {failed}"

    # One double reset before the same physics tick. World 1 first requests
    # an airborne pose, then a screened seat. World 3 retains its airborne request.
    raised = list(nominal.pipette)
    raised[2] += .35
    reset(all_ids[[0, 1, 3]], replace(nominal, randomize=False, pipette=tuple(raised)))
    reset(all_ids[[1, 4, 5]], screened)
    # Cancel a pending hold on the same airborne pose. Its first tick must be in free fall.
    reset(all_ids[[0]], replace(nominal, randomize=False, pipette=tuple(raised)), settle_seat=False)
    seat = tool.data.root_link_pose_w.torch[1].clone()

    # World 4 overrides its seat and velocity. World 3 overrides only its
    # velocity. Both tools have room to move horizontally under gravity.
    custom = tool.data.root_link_pose_w.torch[[4]].clone()
    custom[:, 2] += .35
    tool.write_root_pose_to_sim_index(root_pose=custom, env_ids=all_ids[[4]])
    free_ids = all_ids[[4, 3]]
    velocity = target.new_zeros(2, 6)
    velocity[:, 0] = target.new_tensor([.25, -.25])
    tool.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=free_ids)
    free_before = tool.data.root_link_pose_w.torch[free_ids, :3].clone()
    wrist_before = robot.data.joint_pos.torch[continuing, joint].clone()
    first_tick = sim.get_physics_step_count()
    step()
    assert sim.get_physics_step_count() == first_tick + 1
    after = tool.data.root_link_pose_w.torch
    assert tool.data.root_link_vel_w.torch[0, 2] < -.05, "Free release was overridden by an earlier hold"
    overlap_mm = 1000 * (after[1, :3] - seat[:3]).norm()
    assert overlap_mm < .4 and angle_deg(after[1, 3:], seat[3:]) < angle_limit, "Overlap lost its latest seat"
    delta = after[free_ids, :3] - free_before
    expected_delta = velocity[:, :3] * .01
    expected_delta[:, 2] -= 9.81 * .01**2
    torch.testing.assert_close(delta, expected_delta, rtol=0, atol=.001)
    vx = tool.data.root_link_vel_w.torch[free_ids, 0]
    torch.testing.assert_close(vx, velocity[:, 0], rtol=0, atol=.02)
    wrist_motion = (wrist_before - robot.data.joint_pos.torch[continuing, joint]) * signed_offset.sign()
    assert (wrist_motion > 0).all(), "Continuing wrists stopped at the overlapping reset"
    report["overlap_and_overrides"] = dict(
        physics_step_before=first_tick, physics_step_after=sim.get_physics_step_count(),
        overlap_world=1, overlap_drift_mm=float(overlap_mm), override_worlds=free_ids.tolist(),
        free_delta_m=delta.tolist(), free_vx_m_s=vx.tolist(), continuing_directed_motion_rad=wrist_motion.tolist(),
    )
    if output:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PASS: repeated indexed resets, overlapping requests and caller overrides", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--holder", choices=("rack", "frame-edge"), required=True)
    parser.add_argument("--output", type=Path, help="Optional per-world measurements (JSON)")
    args = parser.parse_args()
    app = AppLauncher(headless=True, device="cuda:0", kit_args="--/renderer/enabled=false").app
    try:
        check(args.assets.resolve(), args.holder, args.output)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
