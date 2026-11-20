"""Launch the standalone open-frame pipette scene with joint-position controls."""

import argparse
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", choices=("mounted", "floating", "vega"), default="mounted")
parser.add_argument("--holder", choices=("frame-edge", "rack"), default="rack")
parser.add_argument("--pipette-pose", nargs=7, type=float, metavar=("X", "Y", "Z", "W", "QX", "QY", "QZ"),
                    help="Set the nominal pipette pose in metres and wxyz")
parser.add_argument("--robot-base-pose", nargs=7, type=float, metavar=("X", "Y", "Z", "W", "QX", "QY", "QZ"),
                    help="Override the robot base pose relative to the bench-top frame")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--fixed-layout", action="store_true", help="Disable both object randomizers")
parser.add_argument("--fixed-pipette", action="store_true")
parser.add_argument("--fixed-tube", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=50, help="Number of 100 Hz simulation steps")
parser.add_argument("--assets", type=Path, required=True, help="Converted asset cache directory")
parser.add_argument("--prepare", action="store_true", help="Convert the sources into a new asset cache")
parser.add_argument("--state-output", type=Path, help="Write the final native link poses for inspection")
parser.add_argument("--rgb", type=Path, help="Save a native cabinet camera PNG")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
launcher = AppLauncher(args, enable_cameras=args.rgb is not None)

try:
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

    from wetlab_isaaclab.equipment.prepare_pipette_assets import prepare_pipette_assets
    from wetlab_isaaclab.robots.dexmate import vega_1_robot_cfg
    from wetlab_isaaclab.robots.dg5f import dg5f_robot_cfg
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout, default_robot_base_pose
    from wetlab_isaaclab.tasks.pipette_placement import screen_pipette_layouts
    from wetlab_isaaclab.tasks.pipette_task import CabinetPipetteTask
    from wetlab_isaaclab.tasks.pipette_scene import cabinet_pipette_scene_cfg, reset_cabinet_pipette

    if args.prepare:
        paths = prepare_pipette_assets(args.assets)
    else:
        paths = {}
        for name in (args.robot, "pipette", "tube"):
            path = next((args.assets / name).glob("*/*.usda"), None)
            if path is None:
                raise FileNotFoundError(
                    f"Missing converted {name} in {args.assets}. Use --prepare with a new cache directory."
                )
            paths[name] = path

    sim = SimulationContext(SimulationCfg(
        device=args.device, dt=0.01, physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=2.0),
    ))
    base_pose = args.robot_base_pose or default_robot_base_pose(robot=args.robot, holder=args.holder)
    pose = dict(position=base_pose[:3], orientation=base_pose[3:])
    configure = vega_1_robot_cfg if args.robot == "vega" else dg5f_robot_cfg
    if args.robot == "floating":
        pose["floating"] = True
    robot = configure(paths[args.robot], **pose)
    layout = PipetteLayout(
        holder=args.holder,
        pipette=tuple(args.pipette_pose) if args.pipette_pose is not None else None,
        randomize=not args.fixed_layout,
        randomize_pipette=not args.fixed_pipette, randomize_tube=not args.fixed_tube,
    )
    if not args.fixed_layout:
        layout = screen_pipette_layouts(
            layout, base_position=pose["position"],
            base_rotation=pose.get("orientation", (1, 0, 0, 0)),
            floating=args.robot == "floating", seed=args.seed,
        )
        print(f"placement_attempts={layout.attempted} pipette={len(layout.pipette_poses)} "
              f"tube={len(layout.tube_poses)}", flush=True)

    scene_cfg = cabinet_pipette_scene_cfg(
        robot=robot, pipette_usd=paths["pipette"], tube_usd=paths["tube"], num_envs=args.num_envs,
        env_spacing=3.0, layout=layout,
    )
    if args.rgb:
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import CuboidCfg, DomeLightCfg, PinholeCameraCfg, PreviewSurfaceCfg

        scene_cfg.inspection_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/InspectionCamera", width=640, height=480,
            data_types=["rgb"], spawn=PinholeCameraCfg(focal_length=24, horizontal_aperture=25),
        )
        scene_cfg.inspection_light = AssetBaseCfg(
            prim_path="/World/InspectionLight", spawn=DomeLightCfg(intensity=1600),
        )
        # A visual background separates the white frame from the camera's clear color.
        scene_cfg.inspection_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/InspectionBackdrop",
            spawn=CuboidCfg(
                size=(2.60, .02, 2.1),
                visual_material=PreviewSurfaceCfg(diffuse_color=(.29, .20, .17)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(.1, .37, .14)),
        )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    ids = torch.arange(args.num_envs, device=scene.device)
    rng = torch.Generator(device=scene.device).manual_seed(args.seed)
    reset_cabinet_pipette(scene, ids, layout, generator=rng)
    task = CabinetPipetteTask(scene, control_dt=0.05)
    task.reset(ids)
    initial = scene["pipette"].data.root_link_pose_w.torch.clone()

    target = scene["robot"].data.default_joint_pos.torch.clone()
    names = scene["robot"].joint_names
    joint = "root_x_axis_joint" if args.robot == "floating" else "R_arm_j7"
    joint_id = names.index(joint)
    target[:, joint_id] += 0.02
    scene["robot"].actuators.target_command.set_position_index(value=target)
    for step in range(args.steps):
        scene.write_data_to_sim()
        sim.step(render=args.rgb is None)
        scene.update(0.01)
        if (step + 1) % 5 == 0:
            outcomes = task.update()
    outcomes = task.update() if args.steps < 5 else outcomes
    print(f"event_outcomes={ {k: v.tolist() for k, v in outcomes.items()} }", flush=True)
    final = scene["pipette"].data.root_link_pose_w.torch.clone()
    displacement = (final[:, :3] - initial[:, :3]).norm(dim=1)
    actual = scene["robot"].data.joint_pos.torch
    if not torch.isfinite(final).all() or not torch.isfinite(actual).all():
        raise RuntimeError("Nonfinite scene state")
    motion = actual[:, joint_id] - scene["robot"].data.default_joint_pos.torch[:, joint_id]
    print(f"robot={args.robot} worlds={args.num_envs} pipette_displacement_m={displacement.tolist()} joint_delta={motion.tolist()}", flush=True)
    if args.rgb:
        from PIL import Image

        camera = scene["inspection_camera"]
        eye, target_point = (1.53, -2.11, 1.12), (0.08, -0.24, 0.24)
        origin = scene.env_origins
        camera.set_world_poses_from_view(origin + origin.new_tensor(eye), origin + origin.new_tensor(target_point))
        for _ in range(6):
            sim.render()
            camera.update(0.01, force_recompute=True)
        rgb = camera.data.output["rgb"].torch[..., :3].cpu().numpy()
        print(f"CAMERA_SHAPE {rgb.shape}", flush=True)
        if rgb.shape != (args.num_envs, 480, 640, 3) or any(float(frame.std()) < 2 for frame in rgb):
            raise RuntimeError("Cabinet RGB camera did not produce a usable image for every world")
        args.rgb.parent.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(rgb):
            path = args.rgb if index == 0 else args.rgb.with_name(f"{args.rgb.stem}_env{index}{args.rgb.suffix}")
            Image.fromarray(frame).save(path)
        print("CABINET_RGB_PASS", flush=True)
    if args.state_output:
        import json

        def links(asset):
            return dict(zip(asset.body_names, asset.data.body_link_pose_w.torch[0].tolist(), strict=True))

        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        args.state_output.write_text(json.dumps({
            "env_origin": scene.env_origins[0].tolist(),
            "robot": links(scene["robot"]), "pipette": links(scene["pipette"]),
            "cabinet": scene["cabinet"].data.root_link_pose_w.torch[0].tolist(),
            "tube": scene["tube"].data.root_link_pose_w.torch[0].tolist(),
            "nozzle": task.observe()["nozzle"][0].tolist(),
            "tube_opening": task.observe()["tube_opening"][0].tolist(),
        }, indent=2) + "\n")
    if args.num_envs > 1:
        frame_before = scene["cabinet"].data.root_link_pose_w.torch.clone()
        object_before = {name: scene[name].data.root_link_pose_w.torch[1:].clone()
                         for name in ("pipette", "tube", "rack") if name in scene.rigid_objects or name == "pipette"}
        reset_cabinet_pipette(scene, ids[:1], layout, generator=rng)
        task.reset(ids[:1])
        torch.testing.assert_close(scene["cabinet"].data.root_link_pose_w.torch, frame_before)
        for name, before in object_before.items():
            torch.testing.assert_close(scene[name].data.root_link_pose_w.torch[1:], before)
    print("CABINET_SCENE_STEP_PASS", flush=True)
except Exception:
    traceback.print_exc()
    launcher.app.close(exit_code=1)
    raise
else:
    launcher.app.close()
