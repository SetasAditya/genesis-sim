"""Exercise the floating hand, native plunger contact sensor and task success in four worlds.

Pass the asset directory created by run_cabinet_pipette.py --prepare.
The pipette body is clamped over the opening to isolate the press from pickup and transport.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


def check(assets, output):
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.utils.math import quat_apply
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from wetlab_isaaclab.robots.dg5f import dg5f_robot_cfg
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout
    from wetlab_isaaclab.tasks.pipette_scene import cabinet_pipette_scene_cfg, reset_cabinet_pipette
    from wetlab_isaaclab.tasks.pipette_task import CabinetPipetteTask

    names = ("aligned", "off_axis", "short_hold", "shallow")
    aligned, off_axis, short_hold, shallow = range(len(names))
    sim = SimulationContext(SimulationCfg(
        device="cuda:0", dt=0.01, physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=2.0),
    ))
    layout = PipetteLayout(randomize=False)
    source = assets / "source/gilson_pipette/pipette_gilson_articulated.urdf"
    clamped = UrdfConverter(UrdfConverterCfg(
        asset_path=str(source), usd_dir=str(assets / "press_check_clamped"),
        fix_base=True, merge_fixed_joints=False, joint_drive=None,
    ))
    # Tilt the body so the plunger slide points down. Position the nozzle
    # 25 mm above the vertical tube, and the middle fingertip above the cap.
    body_wxyz = (0.9984931, 0.0135952, -0.05316695, 0.)
    nozzle_offset = quat_apply(torch.tensor([[*body_wxyz[1:], body_wxyz[0]]]),
                               torch.tensor([[0.0016495, 0.0000139, -0.14165]]))[0]
    body_xyz = torch.tensor(layout.tube[:3]) + torch.tensor([0., 0., 0.137]) - nozzle_offset
    layout = replace(layout, pipette=(*body_xyz.tolist(), *body_wxyz))
    robot_cfg = dg5f_robot_cfg(
        next((assets / "floating").glob("*/*.usda")), floating=True,
        position=(body_xyz + torch.tensor([0.076695, -0.153623, 0.416080])).tolist(),
        orientation=(-0.1517696, 0.7859673, -0.5171752, -0.3029047),
    )
    cfg = cabinet_pipette_scene_cfg(robot=robot_cfg, pipette_usd=clamped.usd_path,
                                    tube_usd=next((assets / "tube").glob("*/*.usda")),
                                    num_envs=len(names), env_spacing=3., layout=layout)
    scene = InteractiveScene(cfg)
    sim.reset()
    ids = torch.arange(len(names), device=scene.device)
    reset_cabinet_pipette(scene, ids, layout)
    tube = scene["tube"]
    poses = tube.data.root_link_pose_w.torch.clone()
    poses[off_axis, 0] += 0.014
    tube.write_root_pose_to_sim_index(root_pose=poses)
    task = CabinetPipetteTask(scene, control_dt=0.05)
    task.reset(ids)
    robot = scene["robot"]
    home = robot.data.default_joint_pos.torch.clone()
    prepress = home.clone()
    for digit in (2, 4, 5):
        for joint in (3, 4):
            prepress[:, robot.joint_names.index(f"rj_dg_{digit}_{joint}")] = 0.48
    # A short translation and yaw in the floating mount's local frame drives
    # the fingertip into the cap. The shallow world stops ~2.5 mm earlier.
    loaded = prepress.clone()
    for joint, deep, partial in (
        ("root_x_axis_joint", .0289, .02685), ("root_y_axis_joint", -.00409, -.00379),
        ("root_z_axis_joint", .03481, .03234), ("root_z_rot_joint", .06265, .05793),
    ):
        index = robot.joint_names.index(joint)
        loaded[:, index] = deep
        loaded[shallow, index] = partial

    data = []
    for tick in range(90):
        # Approach in one second, then hold. One hand releases after 0.4 s.
        progress = home.new_full((len(names), 1), max(0., min(1., (tick - 9) / 20)))
        progress[short_hold] *= max(0., min(1., (43 - tick) / 6))
        progress *= max(0., min(1., (81 - tick) / 10))
        target = torch.lerp(prepress, loaded, progress.square() * (3 - 2 * progress))
        robot.actuators.target_command.set_position_index(value=target)
        for _ in range(5):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(.01)
        flags = task.update()
        obs = task.observe()
        delta = obs["nozzle"] - obs["tube_opening"]
        height = (delta * obs["tube_axis"]).sum(-1, keepdim=True)
        lateral_mm = 1000 * (delta - height * obs["tube_axis"]).norm(dim=-1)
        data.append(dict(t=(tick + 1) / 20, travel_mm=(1000 * obs["plunger_travel"]).tolist(),
                         force_N=obs["axial_force"].tolist(), lateral_mm=lateral_mm.tolist(),
                         **{name: value.tolist() for name, value in flags.items()}))
    if output:
        output.write_text(json.dumps(dict(worlds=names, samples=data), allow_nan=False) + "\n")
    for index, name in enumerate(names):
        print(name, "peak travel (mm)", max(row["travel_mm"][index] for row in data),
              "successful samples", sum(row["success"][index] for row in data), flush=True)
    assert all(all(row["valid"]) for row in data)
    assert [any(row["success"][i] for row in data) for i in range(4)] == [True, False, False, False]
    assert all(row["aligned"][aligned] and not row["aligned"][off_axis] for row in data)
    assert any(row["sustained_press"][off_axis] for row in data)
    assert any(row["travel_mm"][short_hold] >= 15 and row["force_N"][short_hold] > .1 for row in data)
    assert not any(row["sustained_press"][short_hold] for row in data)
    assert max(row["travel_mm"][shallow] for row in data) < 15
    assert any(row["force_N"][shallow] > .1 for row in data)
    print("PASS: aligned held press and three negative controls", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional per-control measurements (JSON)")
    args = parser.parse_args()
    app = AppLauncher(headless=True, device="cuda:0", kit_args="--/renderer/enabled=false").app
    try:
        check(args.assets.resolve(), args.output)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
