"""Gymnasium registrations for wet-lab tasks."""

import gymnasium as gym

for robot, suffix in (("mounted", "Mounted"), ("floating", "Floating")):
    gym.register(
        id=f"Wetlab-Pipette-Rack-{suffix}-v0",
        entry_point=f"{__name__}.pipette_env:RackPipetteEnv",
        kwargs={"robot_kind": robot},
        disable_env_checker=True,
        order_enforce=False,
    )
