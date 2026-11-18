"""Isaac Lab configurations for the mounted and floating DG-5F robots."""

import json
from pathlib import Path

from typing import TYPE_CHECKING

from .dexmate import ROBOT_CONTRACT, _articulation_cfg, _author_collision_filters

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

MOUNTED_PALM_LINK = "rl_dg_mount"
FLOATING_PALM_LINK = "base_link"


def dg5f_robot_cfg(
    usd_path: str | Path,
    *,
    floating: bool = False,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> "ArticulationCfg":
    """Configure a fixed-root DG-5F USD. The input orientation is WXYZ."""
    contract = json.loads(ROBOT_CONTRACT.read_text())
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

    return _articulation_cfg(usd_path, prim_path, position, orientation, groups)


def author_dg5f_collision_filters(stage, root_path: str, *, floating: bool = False) -> int:
    """Author pair exclusions on the converted source USD before cloning.

    This requires an editable stage and separate, named link prims. The floating
    palm is named base_link rather than rl_dg_mount in the mounted model.
    """
    pairs = json.loads(ROBOT_CONTRACT.read_text())["excluded_pairs"]
    if floating:
        pairs = [
            [("base_link" if name == "rl_dg_mount" else name) for name in pair]
            for pair in pairs
            if all(name.startswith("rl_dg_") for name in pair)
        ]
    return _author_collision_filters(stage, root_path, pairs)
