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
class ScreenedPipetteLayouts:
    nominal: PipetteLayout
    pipette_poses: torch.Tensor
    tube_poses: torch.Tensor
    attempted: tuple[int, int]

    @property
    def cabinet(self):
        return self.nominal.cabinet

    @property
    def tube(self):
        return self.nominal.tube

    @property
    def pipette(self):
        return self.nominal.pipette

    @property
    def frame(self):
        return self.nominal.frame

    @property
    def bench(self):
        return self.nominal.bench

    def sample(self, count, *, device, generator=None):
        def draw(bank):
            # Draw even for singleton banks to preserve the other object's stream.
            indices = torch.randint(len(bank), (count,), device=device, generator=generator)
            return bank.to(device)[indices]
        return {
            "cabinet": torch.tensor(self.cabinet, device=device).expand(count, -1),
            "pipette": draw(self.pipette_poses),
            "tube": draw(self.tube_poses),
        }


def screen_pipette_layouts(
    nominal: PipetteLayout,
    *,
    base_position,
    base_rotation,
    floating: bool = False,
    seed: int = 0,
    count: int = 32,
) -> ScreenedPipetteLayouts:
    """Independently sample shelf and bench placements with waypoint reach checks."""

    if count < 1:
        raise ValueError("count must be positive")
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
    frame_pose = np.asarray(nominal.cabinet)
    frame = Rotation.from_quat(frame_pose[[4, 5, 6, 3]])
    boxes = [(np.asarray(center), np.asarray(size) / 2) for center, size in nominal.frame.boxes()]

    def clear(points, pads):
        local = frame.inv().apply(points - frame_pose[:3])
        return not any(np.any(np.all(np.abs(local - center) < half + pads, axis=-1))
                       for center, half in boxes)

    def valid_pipette(pose):
        center = frame.inv().apply(pose[:3] - frame_pose[:3])
        # Conservative horizontal tool footprint in the shelf plane.
        orientation = frame.inv() * Rotation.from_quat(pose[[4, 5, 6, 3]])
        half = np.abs(orientation.as_matrix()) @ np.array((.04, .04, .15))
        shelf_half = np.array((nominal.frame.width / 2 - nominal.frame.post_width, nominal.frame.depth / 2))
        if np.any(np.abs(center[:2]) + half[:2] > shelf_half):
            return False
        approach = pose[:3] + frame.apply((0, -.13, .035))
        return clear(approach[None, :], .025) and reaches(approach)

    def valid_tube(pose):
        bench_half = np.array((nominal.bench.width, nominal.bench.depth)) / 2
        if np.any(np.abs(pose[:2] - nominal.bench.center_xy) + .045 > bench_half):
            return False
        approach = pose[:3] + (0, -.08, .27)
        tube_column = pose[:3] + np.array([[0, 0, z] for z in (.018, .06, .112)])
        tested = np.vstack((tube_column, approach))
        pads = np.array([[.045, .045, .015]] * 3 + [[.025] * 3])
        return clear(tested, pads) and reaches(approach)

    def screen(pose, enabled, half_range, accept, rng, name):
        if not (nominal.randomize and enabled):
            return torch.tensor([pose], dtype=torch.float32), 0
        bank = []
        for attempted in range(1, count * 8 + 1):
            candidate = np.array(pose)
            candidate[:2] += rng.uniform(-1, 1, 2) * half_range
            if accept(candidate):
                bank.append(candidate)
                if len(bank) == count:
                    break
        if not bank:
            raise ValueError(f"No {name} placement passed the configured workspace screen")
        return torch.tensor(np.asarray(bank), dtype=torch.float32), attempted

    pipette_seed, tube_seed = np.random.SeedSequence(seed).spawn(2)
    pipettes, pipette_attempts = screen(
        nominal.pipette, nominal.randomize_pipette, nominal.pipette_xy_half_range_m,
        valid_pipette, np.random.default_rng(pipette_seed), "pipette",
    )
    tubes, tube_attempts = screen(
        nominal.tube, nominal.randomize_tube, nominal.tube_xy_half_range_m,
        valid_tube, np.random.default_rng(tube_seed), "tube",
    )
    return ScreenedPipetteLayouts(nominal, pipettes, tubes, (pipette_attempts, tube_attempts))
