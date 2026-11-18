"""Isaac Lab configurations for the mounted and floating DG-5F robots."""

import json
import re
from pathlib import Path

from typing import TYPE_CHECKING

from wetlab_isaaclab.assets import ASSET_ROOT

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

_CONTRACT = ASSET_ROOT / "robots" / "dexmate_dg5f" / "simulation_contract.json"
MOUNTED_PALM_LINK = "rl_dg_mount"
FLOATING_PALM_LINK = "base_link"


def _actuator(rows):
    from isaaclab.actuators import ImplicitActuatorCfg

    def values(key):
        return {
            re.escape(row["name"]): row[key]
            for row in rows
            if row[key] is not None
        }

    return ImplicitActuatorCfg(
        joint_names_expr=[re.escape(row["name"]) for row in rows],
        stiffness=values("stiffness"),
        damping=values("damping"),
        effort_limit_sim=values("force_limit"),
        velocity_limit_sim=values("velocity_limit") or None,
    )


def dg5f_robot_cfg(
    usd_path: str | Path,
    *,
    floating: bool = False,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> "ArticulationCfg":
    """Configure a fixed-root DG-5F USD. The input orientation is WXYZ."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg

    contract = json.loads(_CONTRACT.read_text())
    mounted = contract["mounted_joints"]
    fingers = [row for row in mounted if row["name"].startswith("rj_dg_")]
    if floating:
        root = contract["floating_root"]
        root_rows = [
            dict(name=name, reset_qpos=0.0, velocity_limit=None, **{
                key: root[key] for key in ("stiffness", "damping", "force_limit")
            })
            for name in root["joints"]
        ]
        groups = {"palm": root_rows, "fingers": fingers}
    else:
        groups = {"body": [row for row in mounted if row not in fingers], "fingers": fingers}

    spawn = sim_utils.UsdFileCfg(
        usd_path=str(Path(usd_path).expanduser().resolve(strict=True)),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=20,
            solver_velocity_iteration_count=1,
        ),
    )
    home = {
        row["name"]: row["reset_qpos"] for rows in groups.values() for row in rows
    }
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position, rot=(*orientation[1:], orientation[0]), joint_pos=home, joint_vel={".*": 0.0}
        ),
        actuators={name: _actuator(rows) for name, rows in groups.items()},
    )


def author_dg5f_collision_filters(stage, root_path: str, *, floating: bool = False) -> int:
    """Author pair exclusions on the converted source USD before cloning.

    This requires an editable stage and separate, named link prims. The floating
    palm is named base_link rather than rl_dg_mount in the mounted model.
    """
    from pxr import Usd, UsdPhysics

    pairs = json.loads(_CONTRACT.read_text())["excluded_pairs"]
    if floating:
        pairs = [
            [("base_link" if name == "rl_dg_mount" else name) for name in pair]
            for pair in pairs
            if all(name.startswith("rl_dg_") for name in pair)
        ]
    expected = {name for pair in pairs for name in pair}
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f"Unknown robot prim {root_path}")
    links = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name not in expected or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if name in links:
            raise ValueError(f"Multiple rigid bodies named {name} under {root_path}")
        links[name] = prim
    if missing := expected - links.keys():
        raise ValueError(f"Missing rigid links under {root_path}: {sorted(missing)}")
    for first, second in pairs:
        api = UsdPhysics.FilteredPairsAPI.Apply(links[first])
        api.CreateFilteredPairsRel().AddTarget(links[second].GetPath())
    return len(pairs)
