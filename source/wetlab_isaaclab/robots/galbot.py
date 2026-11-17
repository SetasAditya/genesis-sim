"""Galbot G1 Isaac Lab articulation configuration."""

import json
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from wetlab_isaaclab.assets import GALBOT_USD, GENERATED_ASSET_DIR

STANDING_POSE = json.loads(
    (GENERATED_ASSET_DIR / "standing_joint_positions.json").read_text()
)

GALBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Galbot",
    articulation_root_prim_path="/OmniChassis/base_link",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GALBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-4.8, 2.66, 0.075),
        rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        joint_pos=STANDING_POSE,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=["leg_joint.*", ".*_arm_joint.*", "head_joint.*"],
            stiffness=60000.0,
            damping=600.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_joint", "right_gripper_joint"],
            stiffness=200.0,
            damping=20.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel[1-4]_joint"],
            stiffness=0.0,
            damping=20.0,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_passive_.*", "gripper_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
