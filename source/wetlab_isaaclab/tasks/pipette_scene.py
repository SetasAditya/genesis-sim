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

    from wetlab_isaaclab.equipment.crossbar_rack import crossbar_rack_boxes
    from wetlab_isaaclab.equipment.gilson_pipette import gilson_pipette_cfg

    from isaaclab.sim.utils.prims import clone, create_prim

    @clone
    def spawn_boxes(prim_path, cfg, translation=None, orientation=None):
        root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
        for index, ((center, size), color) in enumerate(zip(cfg.boxes, cfg.colors, strict=True)):
            physical_material = None
            if index == cfg.support_index:
                physical_material = sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5, dynamic_friction=1.5, friction_combine_mode="max",
                )
            box = sim_utils.CuboidCfg(
                size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=physical_material,
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
        support_index: int = -1

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

    def boxes_asset(boxes, color, support_index):
        return BoxAssemblyCfg(
            boxes=boxes, colors=(color,) * len(boxes), kinematic=True, support_index=support_index,
        )

    def fixture(name: str, pose: Pose, *, usd=None, spawn=None):
        if spawn is None:
            spawn = sim_utils.UsdFileCfg(
                usd_path=str(Path(usd).expanduser().resolve(strict=True)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pose[:3], rot=(*pose[4:], pose[3]),
            ),
        )

    frame = (fixture("RailFrame", layout.cabinet, spawn=boxes_asset(frame_boxes, white, len(frame_boxes) - 1))
             if cabinet_usd is None else fixture("Cabinet", layout.cabinet, usd=cabinet_usd))
    rack = None
    if layout.holder == "rack":
        rack_boxes = crossbar_rack_boxes()
        rack = fixture("Rack", layout.rack, spawn=boxes_asset(
            tuple(rack_boxes.values()), (.20, .33, .55), tuple(rack_boxes).index("crossbar"),
        ))

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

    hand_paths = (
        body_paths(robot.spawn.usd_path, "Robot", lambda name: name.startswith("rl_dg_"))
        if robot is not None else []
    )
    plunger_path, = body_paths(pipette_usd, "Pipette", lambda name: name == "pipette_plunger")

    @configclass
    class CabinetPipetteSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg | None = None
        work_surface: AssetBaseCfg | None = None
        pipette: ArticulationCfg | None = None
        cabinet: RigidObjectCfg | None = None
        tube: RigidObjectCfg | None = None
        rack: RigidObjectCfg | None = None
        press_contact: ContactSensorCfg | None = None

    pipette = gilson_pipette_cfg(
        pipette_usd, position=layout.pipette[:3], orientation=layout.pipette[3:],
    )
    if cabinet_usd is None:
        from isaaclab_physx.sim.schemas.schemas_cfg import PhysxRigidBodyPropertiesCfg

        # Bound the depenetration impulse when a narrow edge seats in the hook.
        pipette.spawn.rigid_props = PhysxRigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=0.5,
        )

    return CabinetPipetteSceneCfg(
        robot=robot,
        work_surface=work_surface,
        pipette=pipette,
        cabinet=frame,
        tube=fixture("Tube", layout.tube, usd=tube_usd),
        rack=rack,
        press_contact=ContactSensorCfg(
            prim_path=plunger_path, filter_prim_paths_expr=hand_paths,
            track_friction_forces=True, max_contact_data_count_per_prim=256,
        ) if hand_paths else None,
        num_envs=num_envs,
        env_spacing=env_spacing,
    )


class _SeatRelease:
    """Break previous support contacts for one physics tick, then release the pipette at rest."""

    def __init__(self, scene):
        import weakref

        import torch
        from isaaclab_physx.physics.physx_manager import IsaacEvents

        self.pipette = scene["pipette"]
        self.pending = None
        self.seats = torch.empty_like(self.pipette.data.root_link_pose_w.torch)
        self.release = None
        manager = scene.sim.physics_manager
        for fn, event in ((self.before_step, IsaacEvents.PRE_PHYSICS_STEP),
                          (self.after_step, IsaacEvents.POST_PHYSICS_STEP)):
            handle = manager.register_callback(fn, event)
            weakref.finalize(self, handle.deregister)

    def queue(self, env_ids, settle):
        import torch

        ids = torch.as_tensor(env_ids, device=self.seats.device, dtype=torch.long)
        if settle:
            self.seats[ids] = self.pipette.data.root_link_pose_w.torch[ids]
            self.pending = ids.clone() if self.pending is None else torch.cat((self.pending, ids)).unique()
        elif self.pending is not None:
            remaining = self.pending[~torch.isin(self.pending, ids)]
            self.pending = remaining if remaining.numel() else None

    def before_step(self, dt):
        if self.pending is None:
            return
        ids, self.pending = self.pending, None
        data = self.pipette.data
        pose = data.root_link_pose_w.torch[ids]
        # Explicit state writes by the caller after reset take precedence.
        at_seat = (pose - self.seats[ids]).abs().amax(dim=1) < 1e-6
        at_seat &= (data.root_link_vel_w.torch[ids] == 0).all(dim=1)
        at_seat &= (data.joint_pos.torch[ids] == data.default_joint_pos.torch[ids]).all(dim=1)
        at_seat &= (data.joint_vel.torch[ids] == 0).all(dim=1)
        ids, seat = ids[at_seat], pose[at_seat]
        if ids.numel():
            # Park above the apparatus only between the native pre/post-step events.
            # Reset observations and the first completed step both expose the seat.
            clear = seat.clone()
            clear[:, 2] += 2.0
            self.pipette.write_root_pose_to_sim_index(root_pose=clear, env_ids=ids)
            self.release = ids, seat

    def after_step(self, dt):
        if self.release is None:
            return
        ids, pose = self.release
        self.release = None
        pipe = self.pipette
        pipe.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        pipe.write_root_velocity_to_sim_index(root_velocity=pipe.data.default_root_vel.torch[ids], env_ids=ids)
        pipe.write_joint_state_to_sim_index(
            position=pipe.data.default_joint_pos.torch[ids], velocity=pipe.data.default_joint_vel.torch[ids],
            env_ids=ids,
        )


def reset_cabinet_pipette(scene, env_ids, layout: PipetteLayout, *, generator=None, settle_seat=True, poses=None):
    """Reset selected worlds immediately, then release their seated pipettes on the next physics tick.

    An optional poses mapping supplies the layout sample for the selected worlds.
    Pass settle_seat=False for a free start where a controller should act on the pipette
    during that first tick. Post-reset pose, velocity or joint overrides supersede the hold.
    """
    from wetlab_isaaclab.equipment.gilson_pipette import set_spring_preload

    if len(env_ids) == 0:
        return
    if poses is None:
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

    release = getattr(scene, "_pipette_seat_release", None)
    if release is None and settle_seat and not scene["pipette"].is_fixed_base:
        release = scene._pipette_seat_release = _SeatRelease(scene)
    if release is not None:
        release.queue(env_ids, settle_seat)
