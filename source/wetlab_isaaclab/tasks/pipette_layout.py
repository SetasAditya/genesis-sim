"""Nominal pipette and tube placements in the environment frame."""

from dataclasses import dataclass

import torch

from wetlab_isaaclab.equipment.shelf_frame import LabBench, ShelfFrame


Pose = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class PipetteLayout:
    """Poses in metres and wxyz. The frame is fixed across resets."""

    cabinet: Pose = (0.10, -0.18, 0.0, 1.0, 0.0, 0.0, 0.0)
    pipette: Pose | None = None
    tube: Pose = (0.15535563, -0.39385803, 0.0, 0.98822186, 0.0, 0.0, -0.15302796)
    randomize: bool = True
    randomize_pipette: bool = True
    randomize_tube: bool = True
    pipette_xy_half_range_m: tuple[float, float] = (0.14, 0.035)
    tube_xy_half_range_m: tuple[float, float] = (0.06, 0.035)
    frame: ShelfFrame = ShelfFrame()
    bench: LabBench = LabBench()

    def __post_init__(self):
        if self.pipette is None:
            # With this orientation, the pipette rests on its horizontal body.
            offset = (0.0, -0.105, self.frame.shelf_top + 0.021335)
            w, x, y, z = self.cabinet[3:]
            # Apply the frame yaw to the center and to the local pipette orientation.
            if x or y:
                raise ValueError("An upright frame or an explicit pipette pose is required")
            ox, oy, oz = offset
            px, py, pz = self.cabinet[:3]
            cos_yaw, sin_yaw = 1 - 2*z*z, 2*w*z
            object.__setattr__(self, "pipette", (
                px + cos_yaw * ox - sin_yaw * oy, py + sin_yaw * ox + cos_yaw * oy, pz + oz,
                (w + z)/2, (w + z)/2, (z - w)/2, (z - w)/2,
            ))

    def sample(self, count: int, *, device, generator: torch.Generator | None = None):
        """Sample each movable object independently in world XY."""
        poses = torch.tensor((self.cabinet, self.pipette, self.tube), device=device)
        poses = poses.expand(count, -1, -1).clone()
        if self.randomize:
            for index, enabled, half_range in (
                (1, self.randomize_pipette, self.pipette_xy_half_range_m),
                (2, self.randomize_tube, self.tube_xy_half_range_m),
            ):
                # Always draw from both streams so a toggle leaves the other
                # object's placements reproducible under the same seed.
                offset = torch.rand((count, 2), device=device, generator=generator)
                if enabled:
                    poses[:, index, :2] += (2 * offset - 1) * poses.new_tensor(half_range)
        return dict(zip(("cabinet", "pipette", "tube"), poses.unbind(dim=1), strict=True))
