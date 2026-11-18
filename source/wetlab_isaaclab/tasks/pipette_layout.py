"""Configurable cabinet pipette poses in the environment frame."""

from dataclasses import dataclass

import torch


Pose = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class PipetteLayout:
    """Positions in metres followed by a wxyz quaternion.

    Shift the cabinet, pipette and tube together to preserve their relative
    geometry. The robot and work surface stay in their configured locations.
    """

    cabinet: Pose = (0.05770144, -0.36297586, 0.0, 0.85552198, 0.0, 0.0, -0.51776650)
    # The nominal robot-home reset leaves the pipette resting on the shelf.
    pipette: Pose = (0.00899869, -0.26995413, 0.296335, 0.60494539, 0.60494539, -0.36611620, -0.36611620)
    tube: Pose = (0.15535563, -0.39385803, 0.0, 0.98822186, 0.0, 0.0, -0.15302796)
    randomize: bool = True
    xy_half_range_m: tuple[float, float] = (0.01, 0.01)

    def sample(self, count: int, *, device, generator: torch.Generator | None = None):
        """Return named pose batches suitable for indexed root-state reset."""
        poses = torch.tensor((self.cabinet, self.pipette, self.tube), device=device)
        poses = poses.expand(count, -1, -1).clone()
        if self.randomize:
            bounds = poses.new_tensor(self.xy_half_range_m)
            offset = torch.rand((count, 2), device=device, generator=generator)
            poses[:, :, :2] += ((2 * offset - 1) * bounds)[:, None, :]
        return dict(zip(("cabinet", "pipette", "tube"), poses.unbind(dim=1), strict=True))
