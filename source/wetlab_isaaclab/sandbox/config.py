"""Explicit representative dimensions and deterministic lab layout (SI units)."""

from dataclasses import dataclass, field
from typing import Any

BENCH_Z = 0.762715557
PHYSICS_DT = 1 / 240
CONTROL_DT = 1 / 30
STATIONS = {
    "petri": (-5.15, 3.55),
    "tube_insertion": (-5.15, 3.55),
    "reactor_assembly": (-3.55, 3.55),
    "pipetting": (-1.65, 3.55),
    "fridge_retrieval": (-6.65, 1.85),
}
TASK_IDS = (
    "petri",
    "tube_insertion",
    "reactor_assembly",
    "fridge_retrieval",
    "pipetting",
)


@dataclass
class Action:
    """All commands address the robot; object pose changes belong to reset/debug."""

    base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wrist_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    grippers: dict[str, float] = field(default_factory=dict)
    magnetic: bool | None = None
    fixture_clamp: bool | None = None


@dataclass
class ObjectState:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    up_dot: float = 1.0


def point(x, y, z=0.0):
    return (x, y, BENCH_Z + z)


def layout():
    """Asset root frames are bottom-centered, except the native CAD HTR roots."""
    return {
        "dish": point(-5.35, 3.57),
        "lid": point(-5.35, 3.57, 0.005),
        "lid_park": point(-5.55, 3.57),
        "rack": point(-4.96, 3.55),
        "tube": point(-4.79, 3.55, 0.052),
        "jig": point(-3.55, 3.61),
        "basket": point(-3.96, 3.58),
        "well": point(-3.74, 3.52),
        "insert": point(-3.35, 3.44),
        "cover": point(-3.92, 3.80),
        "disassembly_plate": point(-3.15, 3.9),
        "pipette": point(-1.64, 3.58, 0.085),
        "tip": point(-1.88, 3.56, 0.006),
        "source_vessel": point(-1.82, 3.60),
        "target_vessel": point(-1.57, 3.64),
        "tip_bin": point(-1.42, 3.48),
        "fridge": (-6.65, 2.48, 0.071),
        "sample": (-6.65, 2.43, 0.48),
        "sample_target": point(-6.20, 3.57),
    }


PARAMETERS = {
    "dish": {"diameter_m": 0.09, "height_m": 0.016, "wall_m": 0.001, "mass_kg": 0.018},
    "lid": {
        "inner_diameter_m": 0.0912,
        "height_m": 0.012,
        "wall_m": 0.0008,
        "mass_kg": 0.014,
    },
    "tube": {"diameter_m": 0.016, "height_m": 0.100, "wall_m": 0.001, "mass_kg": 0.008},
    "rack": {"holes": 6, "hole_diameter_m": 0.0166, "pitch_m": 0.03, "height_m": 0.05},
    "pipette": {
        "capacity_ul": 200.0,
        "plunger_stroke_m": 0.008,
        "return_stiffness_n_m": 160.0,
        "mass_kg": 0.085,
    },
    "magnet": {
        "capture_gap_m": 0.002,
        "alignment_deg": 10.0,
        "break_force_n": 150.0,
        "break_torque_nm": 10.0,
        "extension_m": 0.066,
        "mass_kg": 0.05,
    },
    "representative_bolt": {
        "thread_diameter_m": 0.010,
        "shaft_length_m": 0.035,
        "mass_kg": 0.035,
    },
}
