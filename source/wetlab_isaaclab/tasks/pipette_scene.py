"""Isaac Lab scene configuration for the cabinet pipette task."""

from pathlib import Path

from .pipette_layout import PipetteLayout, Pose


def cabinet_pipette_scene_cfg(
    *,
    robot,
    pipette_usd: str | Path,
    cabinet_usd: str | Path | None = None,
    tube_usd: str | Path,
    work_surface=None,
    num_envs: int,
    env_spacing: float,
    layout: PipetteLayout = PipetteLayout(),
):
    """Compose a cloned scene from converted equipment and a deployment surface.

    The open frame and bench use configurable box geometry. A supplied fixture
    USD or work surface can replace them for another deployment.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass

    from wetlab_isaaclab.equipment.gilson_pipette import gilson_pipette_cfg

    from isaaclab.sim.utils.prims import clone, create_prim

    @clone
    def spawn_boxes(prim_path, cfg, translation=None, orientation=None):
        root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
        for index, ((center, size), color) in enumerate(zip(cfg.boxes, cfg.colors, strict=True)):
            box = sim_utils.CuboidCfg(
                size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            box.func(f"{prim_path}/Part{index}", box, translation=center)
        if cfg.kinematic:
            sim_utils.define_rigid_body_properties(
                prim_path, sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
        return root

    @configclass
    class BoxAssemblyCfg(sim_utils.SpawnerCfg):
        func = spawn_boxes
        boxes: tuple = ()
        colors: tuple = ()
        kinematic: bool = False

    white = (0.84, 0.86, 0.87)
    frame_boxes = tuple(layout.frame.boxes())
    bench_boxes = tuple(layout.bench.boxes())
    if work_surface is None:
        work_surface = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/WorkSurface",
            spawn=BoxAssemblyCfg(
                boxes=bench_boxes, colors=((0.055, 0.06, 0.065),) + (white,) * (len(bench_boxes)-1),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(*layout.bench.center_xy, 0.)),
        )

    def fixture(name: str, usd: str | Path | None, pose: Pose):
        spawn = (
            BoxAssemblyCfg(boxes=frame_boxes, colors=(white,) * len(frame_boxes), kinematic=True)
            if usd is None else sim_utils.UsdFileCfg(
                usd_path=str(Path(usd).expanduser().resolve(strict=True)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
        )
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pose[:3], rot=(*pose[4:], pose[3]),
            ),
        )

    def body_paths(usd, instance, accepts):
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(usd))
        root = stage.GetDefaultPrim()
        prefix = str(root.GetPath())
        return [
            f"{{ENV_REGEX_NS}}/{instance}" + str(prim.GetPath())[len(prefix):]
            for prim in Usd.PrimRange(root)
            if accepts(prim.GetName()) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]

    hand_paths = body_paths(robot.spawn.usd_path, "Robot", lambda name: name.startswith("rl_dg_"))
    plunger_path, = body_paths(pipette_usd, "Pipette", lambda name: name == "pipette_plunger")

    @configclass
    class CabinetPipetteSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg | None = None
        work_surface: AssetBaseCfg | None = None
        pipette: ArticulationCfg | None = None
        cabinet: RigidObjectCfg | None = None
        tube: RigidObjectCfg | None = None
        press_contact: ContactSensorCfg | None = None

    return CabinetPipetteSceneCfg(
        robot=robot,
        work_surface=work_surface,
        pipette=gilson_pipette_cfg(
            pipette_usd, position=layout.pipette[:3], orientation=layout.pipette[3:],
        ),
        cabinet=fixture("ShelfFrame" if cabinet_usd is None else "Cabinet", cabinet_usd, layout.cabinet),
        tube=fixture("Tube", tube_usd, layout.tube),
        press_contact=ContactSensorCfg(
            prim_path=plunger_path, filter_prim_paths_expr=hand_paths,
            track_friction_forces=True, max_contact_data_count_per_prim=256,
        ) if hand_paths else None,
        num_envs=num_envs,
        env_spacing=env_spacing,
    )


def reset_cabinet_pipette(scene, env_ids, layout: PipetteLayout, *, generator=None):
    """Reset selected worlds to the robot home and the sampled equipment poses."""
    from wetlab_isaaclab.equipment.gilson_pipette import set_spring_preload

    poses = layout.sample(len(env_ids), device=scene.device, generator=generator)
    relative_poses = {
        name: pose[:, (0, 1, 2, 4, 5, 6, 3)] for name, pose in poses.items()
    }

    state = {"articulation": {}, "rigid_object": {}}
    for name, asset in scene.articulations.items():
        data = asset.data
        state["articulation"][name] = {
            "root_pose": (
                relative_poses["pipette"] if name == "pipette"
                else data.default_root_pose.torch[env_ids]
            ),
            "root_velocity": data.default_root_vel.torch[env_ids],
            "joint_position": data.default_joint_pos.torch[env_ids],
            "joint_velocity": data.default_joint_vel.torch[env_ids],
        }

    for name, asset in scene.rigid_objects.items():
        data = asset.data
        root_pose = relative_poses.get(name)
        state["rigid_object"][name] = {
            "root_pose": data.default_root_pose.torch[env_ids] if root_pose is None else root_pose,
            "root_velocity": data.default_root_vel.torch[env_ids],
        }

    scene.reset(env_ids)
    scene.reset_to(state, env_ids=env_ids, is_relative=True)
    set_spring_preload(scene["pipette"], env_ids)
    scene.write_data_to_sim()
