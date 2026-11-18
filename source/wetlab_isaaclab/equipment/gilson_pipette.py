"""Articulation configuration for the Gilson spring-return pipette prototype."""

from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import Articulation

BODY_LINK = "pipette_body"
PLUNGER_LINK = "pipette_plunger"
PLUNGER_JOINT = "pipette_plunger_slide"
SPRING_REST_TARGET_M = -0.01095647488920874


def gilson_pipette_cfg(
    usd_path: str | Path,
    *,
    prim_path: str = "{ENV_REGEX_NS}/Pipette",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ArticulationCfg:
    """Configure a free-root USD with the two prototype links and slide joint.

    The USD supplies the link frames, masses, collision shapes and joint limits.
    The scene owns the initial placement and contact material. The input orientation is WXYZ.
    """
    usd = Path(usd_path).expanduser().resolve(strict=True)
    if not usd.is_file() or usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"Expected a local pipette USD file: {usd}")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=(*orientation[1:], orientation[0]),
            joint_pos={PLUNGER_JOINT: 0.0},
            joint_vel={PLUNGER_JOINT: 0.0},
        ),
        actuators={
            "plunger_spring": ImplicitActuatorCfg(
                joint_names_expr=[PLUNGER_JOINT],
                stiffness=20.0,
                damping=0.2,
                effort_limit_sim=10.0,
                velocity_limit_sim=0.2,
            ),
        },
    )


def set_spring_preload(
    pipette: "Articulation", env_ids: "torch.Tensor | None" = None
) -> None:
    """Set drive targets after reset. The physical slide starts at zero travel."""
    joint_ids, names = pipette.find_joints(PLUNGER_JOINT)
    if names != [PLUNGER_JOINT]:
        raise ValueError(f"Expected one pipette plunger joint, found {names}")
    rows = slice(None) if env_ids is None else env_ids
    positions = pipette.data.default_joint_pos.torch[rows][:, joint_ids]
    command = pipette.actuators.target_command
    command.set_position_index(
        value=positions.new_full(positions.shape, SPRING_REST_TARGET_M),
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
    command.set_velocity_index(
        value=positions.new_zeros(positions.shape),
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
