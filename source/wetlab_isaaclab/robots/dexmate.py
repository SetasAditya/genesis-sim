"""DexMate Vega-1 and shared articulation configuration."""

import json
import re
from pathlib import Path

from wetlab_isaaclab.assets import ASSET_ROOT

ROBOT_CONTRACT = ASSET_ROOT / "robots" / "dexmate_dg5f" / "simulation_contract.json"
VEGA_1_URDF = ASSET_ROOT / "robots" / "vega_1" / "vega_1.urdf"


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



def _articulation_cfg(usd_path, prim_path, position, orientation, groups):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg

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



def vega_1_robot_cfg(
    usd_path: str | Path,
    *,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Configure the fixed-base Vega-1 with no end effector. Orientation is WXYZ."""
    contract = json.loads(ROBOT_CONTRACT.read_text())
    body = [row for row in contract["mounted_joints"] if not row["name"].startswith("rj_dg_")]
    return _articulation_cfg(usd_path, prim_path, position, orientation, {"body": body})


def author_vega_1_collision_filters(stage, root_path: str) -> int:
    """Keep collision exclusions between links of the shared DexMate body."""
    pairs = json.loads(ROBOT_CONTRACT.read_text())["excluded_pairs"]
    body_pairs = [pair for pair in pairs if not any(name.startswith("rl_dg_") for name in pair)]
    return _author_collision_filters(stage, root_path, body_pairs)


def _author_collision_filters(stage, root_path, pairs):
    from pxr import Usd, UsdPhysics

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
