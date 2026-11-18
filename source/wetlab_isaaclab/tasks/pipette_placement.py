"""Generate and screen cabinet layouts before starting vectorized resets."""

from dataclasses import dataclass
import json
from xml.etree import ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from wetlab_isaaclab.assets import ASSET_ROOT
from wetlab_isaaclab.robots.dexmate import ROBOT_CONTRACT, VEGA_1_URDF

from .pipette_layout import PipetteLayout

POSE_NAMES = ("cabinet", "pipette", "tube")


def _xyz(value):
    return np.fromstring(value, sep=" ")


def _frame(xyz, rotation):
    frame = np.eye(4)
    frame[:3, :3] = rotation.as_matrix()
    frame[:3, 3] = xyz
    return frame


class _WristReach:
    """Position IK with actual URDF joints and fixed robot home torso."""

    def __init__(self, base_position, base_rotation):
        by_child = {j.find("child").get("link"): j for j in ET.parse(VEGA_1_URDF).findall("joint")}
        chain = []
        link = "R_arm_l7"
        while link in by_child:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        home = {row["name"]: row["reset_qpos"] for row in json.loads(ROBOT_CONTRACT.read_text())["mounted_joints"]}
        prefix = _frame(base_position, Rotation.from_quat((*base_rotation[1:], base_rotation[0])))
        self.chain = []
        limits = []
        seed = []
        for joint in reversed(chain):
            name = joint.get("name")
            origin = joint.find("origin")
            static = np.eye(4) if origin is None else _frame(
                _xyz(origin.get("xyz", "0 0 0")), Rotation.from_euler("xyz", _xyz(origin.get("rpy", "0 0 0")))
            )
            axis_tag = joint.find("axis")
            axis = _xyz(axis_tag.get("xyz")) if axis_tag is not None else None
            if name.startswith("R_arm_j"):
                self.chain.append((static, axis))
                limit = joint.find("limit")
                limits.append((float(limit.get("lower")), float(limit.get("upper"))))
                seed.append(home[name])
            else:
                prefix = prefix @ static
                if axis is not None and joint.get("type") == "revolute":
                    prefix = prefix @ _frame((0, 0, 0), Rotation.from_rotvec(axis * home[name]))
        self.base = prefix
        self.bounds = np.asarray(limits).T
        self.home = np.clip(seed, *self.bounds)

    def _position(self, joints):
        hand = self.base
        for (offset, axis), angle in zip(self.chain, joints, strict=True):
            hand = hand @ offset @ _frame((0, 0, 0), Rotation.from_rotvec(axis * angle))
        return hand[:3, 3]

    def reaches(self, point):
        def residual(joints):
            return self._position(joints) - point

        result = least_squares(residual, self.home, bounds=self.bounds, max_nfev=45, gtol=1e-5)
        return np.linalg.norm(result.fun) <= 0.005


@dataclass(frozen=True)
class PlacementVariation:
    cabinet_xy_m: tuple[float, float] = (0.07, 0.05)
    cabinet_yaw_rad: float = 0.35
    tube_xy_m: tuple[float, float] = (0.04, 0.04)


@dataclass(frozen=True)
class ScreenedPipetteLayouts:
    nominal: PipetteLayout
    poses: torch.Tensor
    attempted: int

    @property
    def cabinet(self):
        return self.nominal.cabinet

    @property
    def tube(self):
        return self.nominal.tube

    @property
    def pipette(self):
        return self.nominal.pipette

    def sample(self, count, *, device, generator=None):
        indices = torch.randint(len(self.poses), (count,), device=device, generator=generator)
        return dict(zip(POSE_NAMES, self.poses.to(device)[indices].unbind(dim=1), strict=True))


def screen_pipette_layouts(
    nominal: PipetteLayout,
    *,
    base_position,
    base_rotation,
    floating: bool = False,
    variation: PlacementVariation = PlacementVariation(),
    seed: int = 0,
    count: int = 32,
) -> ScreenedPipetteLayouts:
    """Check fixture clearance and robot wrist reach at two approach waypoints.

    This is a reset-time workspace screen. It checks two positions with
    joint-limited IK for the mounted arm or translation bounds for the floating hand. It does not plan a collision-free trajectory.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    base = np.asarray([getattr(nominal, key) for key in POSE_NAMES])
    if floating:
        rotation = Rotation.from_quat((*base_rotation[1:], base_rotation[0]))
        stage_urdf = ASSET_ROOT / "robots/dexmate_dg5f/simulation.urdf"
        translations = {j.get("name"): j.find("limit") for j in ET.parse(stage_urdf).findall("joint")}
        bounds = np.asarray([
            [float(translations[f"root_{axis}_axis_joint"].get(key)) for axis in "xyz"]
            for key in ("lower", "upper")
        ])

        def reaches(point):
            displacement = rotation.inv().apply(point - np.asarray(base_position))
            return np.all((displacement >= bounds[0]) & (displacement <= bounds[1]))
    else:
        reaches = _WristReach(base_position, base_rotation).reaches
    boxes = []
    cabinet_urdf = ASSET_ROOT / "equipment/cabinet/cabinet.urdf"
    for collision in ET.parse(cabinet_urdf).findall("link/collision"):
        center = _xyz(collision.find("origin").get("xyz"))
        radius = _xyz(collision.find("geometry/box").get("size")) / 2
        boxes.append((center, radius))
    layouts = []
    for attempted in range(1, count * 8 + 1):
        values = base.copy()
        offsets = rng.uniform(-1, 1, size=5)
        rotation = Rotation.from_rotvec((0, 0, offsets[2] * variation.cabinet_yaw_rad))
        values[:2, :3] = base[0, :3] + rotation.apply(base[:2, :3] - base[0, :3])
        values[:2, :2] += offsets[:2] * variation.cabinet_xy_m
        values[:2, 3:] = (rotation * Rotation.from_quat(base[:2, (4, 5, 6, 3)])).as_quat()[:, (3, 0, 1, 2)]
        values[2, :2] += offsets[3:] * variation.tube_xy_m
        cabinet, pipette, tube = values
        frame = Rotation.from_quat(cabinet[[4, 5, 6, 3]])
        tube_approach = tube[:3] + (0, -0.08, 0.27)
        cabinet_approach = pipette[:3] + frame.apply((0.13, 0, 0.035))
        tube_column = tube[:3] + np.array([[0, 0, z] for z in (0.018, 0.06, 0.112)])
        # The outer tube base has a 30 mm half-width. Use a 45 mm
        # horizontal envelope to also bound its corners under relative yaw.
        tested = np.vstack((tube_column, cabinet_approach, tube_approach))
        pad = np.array([[.045, .045, .015]] * 3 + [[.025] * 3] * 2)
        local = frame.inv().apply(tested - cabinet[:3])
        if any(np.any(np.all(np.abs(local - center) < half + pad, axis=-1)) for center, half in boxes):
            continue
        if not all(reaches(point) for point in (cabinet_approach, tube_approach)):
            continue
        layouts.append(values)
        if len(layouts) == count:
            break
    if not layouts:
        raise ValueError("No placement passed the configured workspace screen")
    return ScreenedPipetteLayouts(nominal, torch.tensor(np.asarray(layouts), dtype=torch.float32), attempted)
