"""Native Isaac Lab action configs for joint and Cartesian robot control."""


def joint_position_action(joint_names, *, asset_name="robot"):
    """Set absolute joint positions in the given order, in metres or radians."""
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

    return JointPositionActionCfg(
        asset_name=asset_name, joint_names=list(joint_names),
        preserve_order=True, use_default_offset=False,
    )


def body_pose_action(joint_names, body_name, *, asset_name="robot"):
    """Set an absolute base-frame body pose with an XYZW quaternion."""
    from isaaclab.controllers import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

    return DifferentialInverseKinematicsActionCfg(
        asset_name=asset_name, joint_names=list(joint_names), body_name=body_name,
        controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls"),
    )
