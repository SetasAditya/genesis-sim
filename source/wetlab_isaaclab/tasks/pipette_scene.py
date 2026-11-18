"""Isaac Lab scene configuration for the cabinet pipette task."""

from pathlib import Path

from .pipette_layout import PipetteLayout, Pose


def cabinet_pipette_scene_cfg(
    *,
    robot,
    pipette_usd: str | Path,
    cabinet_usd: str | Path,
    tube_usd: str | Path,
    work_surface,
    num_envs: int,
    env_spacing: float,
    layout: PipetteLayout = PipetteLayout(),
):
    """Compose a cloned scene from converted equipment and a deployment surface.

    The cabinet and tube USDs each supply one rigid body with colliders. Their
    kinematic roots allow the environment to reposition them on indexed resets.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    from wetlab_isaaclab.equipment.gilson_pipette import gilson_pipette_cfg

    def fixture(name: str, usd: str | Path, pose: Pose):
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(usd).expanduser().resolve(strict=True)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pose[:3], rot=(*pose[4:], pose[3]),
            ),
        )

    @configclass
    class CabinetPipetteSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg | None = None
        work_surface: AssetBaseCfg | None = None
        pipette: ArticulationCfg | None = None
        cabinet: RigidObjectCfg | None = None
        tube: RigidObjectCfg | None = None

    return CabinetPipetteSceneCfg(
        robot=robot,
        work_surface=work_surface,
        pipette=gilson_pipette_cfg(
            pipette_usd, position=layout.pipette[:3], orientation=layout.pipette[3:],
        ),
        cabinet=fixture("Cabinet", cabinet_usd, layout.cabinet),
        tube=fixture("Tube", tube_usd, layout.tube),
        num_envs=num_envs,
        env_spacing=env_spacing,
    )
