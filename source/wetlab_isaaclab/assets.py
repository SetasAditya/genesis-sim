"""Portable asset paths and provenance-aware remote asset resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets"
HTR_USD_DIR = ASSET_ROOT / "htr"
RAW_LAB_SCENE_USD = ASSET_ROOT / "labutopia" / "navigation_lab_01" / "lab.usd"
GENERATED_ASSET_DIR = ASSET_ROOT / "generated"
LAB_SCENE_USD = GENERATED_ASSET_DIR / "lab_collision.usda"
COBOT_ASSET_DIR = ASSET_ROOT / "robots" / "kinova_mobile_cobot"
COBOT_USD = COBOT_ASSET_DIR / "kinova_mobile_cobot.usda"
COBOT_PLANNER_URDF = COBOT_ASSET_DIR / "curobo" / "kinova_gen3_robolab.urdf"

# NVIDIA's 6.1 catalog page shows an obsolete relative path. The public 6.1
# asset bucket currently publishes the entry point below as galbot_g1.usda.
GALBOT_USD_6_1 = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.1/Isaac/Robots/Galbot/galbot_g1/galbot_g1.usda"
)
GALBOT_USD = os.environ.get(
    "GALBOT_USD_PATH", str(GENERATED_ASSET_DIR / "galbot_standing.usda")
)


def validate_local_assets() -> None:
    required = [
        LAB_SCENE_USD,
        HTR_USD_DIR / "HTR_Bottom_Basket.usda",
        HTR_USD_DIR / "HTR_Cover.usda",
        HTR_USD_DIR / "HTR_Disassembly_Plate.usda",
        HTR_USD_DIR / "HTR_Insert.usda",
        HTR_USD_DIR / "HTR_Well.usda",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing project assets:\n" + "\n".join(missing))
