"""Original LabUtopia bench, dynamic HTR parts, and standing Galbot."""

import json
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from wetlab_isaaclab.assets import GENERATED_ASSET_DIR, HTR_USD_DIR, LAB_SCENE_USD
from wetlab_isaaclab.robots import GALBOT_CFG

LAYOUT = json.loads((GENERATED_ASSET_DIR / "bench_layout.json").read_text())


def usd_asset(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(HTR_USD_DIR) / f"{name}.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(LAYOUT["asset_positions"][name])
        ),
    )


@configclass
class WetLabSceneCfg(InteractiveSceneCfg):
    """Single complete laboratory; use num_envs=1 until room cloning is implemented."""

    lab = AssetBaseCfg(
        prim_path="/World/LabUtopia",
        spawn=sim_utils.UsdFileCfg(usd_path=str(LAB_SCENE_USD)),
    )
    dome = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=700.0),
    )
    robot = GALBOT_CFG
    basket = usd_asset("HTR_Bottom_Basket")
    well = usd_asset("HTR_Well")
    insert = usd_asset("HTR_Insert")
    plate = usd_asset("HTR_Disassembly_Plate")
    cover = usd_asset("HTR_Cover")
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        height=1000,
        width=1600,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.0, horizontal_aperture=24.0, clipping_range=(0.05, 100.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0, 0, 1), rot=(1, 0, 0, 0), convention="opengl"
        ),
    )


@configclass
class WetLabVisualSceneCfg(WetLabSceneCfg):
    """Compatibility name; the canonical scene now keeps robot physics enabled."""
