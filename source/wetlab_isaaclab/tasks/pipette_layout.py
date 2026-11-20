"""Nominal pipette and tube placements in the environment frame."""

from dataclasses import dataclass
from typing import Literal

import torch

from wetlab_isaaclab.equipment.crossbar_rack import crossbar_rack_boxes
from wetlab_isaaclab.equipment.rail_frame import LabBench, RailFrame


Pose = tuple[float, float, float, float, float, float, float]


def default_robot_base_pose(
    *, robot: Literal["mounted", "floating", "vega"], holder: Literal["frame-edge", "rack"]
) -> Pose:
    """Bench-relative base pose in metres and wxyz for the two holder locations."""
    if robot == "floating":
        return (-.18, -.64, .32, 1., 0., 0., 0.)
    # Turn the right shoulder toward the back rail from the same bench-side base.
    orientation = (0., 0., 0., 1.) if holder == "frame-edge" else (.70710678, 0., 0., .70710678)
    return (-.20, -.75, -.8647001386, *orientation)


@dataclass(frozen=True)
class PipetteLayout:
    """Poses in metres and wxyz. The frame is fixed across resets."""

    cabinet: Pose | None = None
    pipette: Pose | None = None
    tube: Pose | None = None
    holder: Literal["frame-edge", "rack"] = "rack"
    rack: Pose = (-0.11, -0.40, 0., 2**-0.5, 0., 0., 2**-0.5)
    randomize: bool = True
    randomize_pipette: bool = True
    randomize_tube: bool = True
    pipette_slide_half_range_m: float = 0.10
    rack_xy_half_range_m: tuple[float, float] = (0.055, 0.02)
    tube_xy_half_range_m: tuple[float, float] = (0.06, 0.035)
    frame: RailFrame = RailFrame()
    bench: LabBench = LabBench()

    @property
    def rail_axis_xy(self):
        w, _, _, z = self.cabinet[3:]
        return (1 - 2 * z * z, 2 * w * z)

    def __post_init__(self):
        if self.holder not in ("frame-edge", "rack"):
            raise ValueError(f"Unknown pipette holder: {self.holder}")
        if self.cabinet is None:
            x, y = self.bench.center_xy
            back = y + (self.bench.depth - self.frame.depth) / 2
            object.__setattr__(self, "cabinet", (x, back, 0., 1., 0., 0., 0.))
        if self.tube is None:
            # Provide space for the rack and its placement range on the left.
            x = 0.28 if self.holder == "rack" else 0.15535563
            object.__setattr__(self, "tube", (x, -0.39385803, 0., 0.98822186, 0., 0., -0.15302796))
        if self.pipette is not None:
            return
        holder_pose = self.cabinet if self.holder == "frame-edge" else self.rack
        w, x, y, z = holder_pose[3:]
        if x or y:
            raise ValueError("The holder must be upright or supplied with an explicit pipette pose")
        c, s = 1 - 2 * z * z, 2 * w * z
        if self.holder == "frame-edge":
            # Balance the finger hook over the lower rail at a slight lateral tilt.
            ex, ey, ez = self.frame.edge_center
            ox, oy, oz = ex, ey + .000640878747049, ez - .105510401581100
            qw, qx, qy, qz = .565250980206578, .095548225237135, .098567850095761, -.813416403176771
        else:
            # Seat the hook against the top-only crossbar in its local frame.
            bx, by, bz = crossbar_rack_boxes()["crossbar"][0]
            ox, oy, oz = bx - .01432224037070, by - .00156960062965, bz - .09979704340709
            qw, qx, qy, qz = .956332694386065, .046761838883144, .195681459931020, .212013854056333
        px, py, pz = holder_pose[:3]
        object.__setattr__(self, "pipette", (
            px + c*ox - s*oy, py + s*ox + c*oy, pz + oz,
            w*qw - z*qz, w*qx - z*qy, w*qy + z*qx, w*qz + z*qw,
        ))

    def rack_for_pipette(self, pipettes):
        """Keep the rack and its seated pipette together across each indexed reset."""
        rack = pipettes.new_tensor(self.rack).expand_as(pipettes).clone()
        rack[:, :2] += pipettes[:, :2] - pipettes.new_tensor(self.pipette[:2])
        return rack

    def sample(self, count, *, device, generator=None):
        """Draw holder and tube placements from independent random streams."""
        poses = torch.tensor((self.cabinet, self.pipette, self.tube), device=device)
        poses = poses.expand(count, -1, -1).clone()
        if self.randomize:
            holder_xy = 2 * torch.rand((count, 2), device=device, generator=generator) - 1
            tube_xy = 2 * torch.rand((count, 2), device=device, generator=generator) - 1
            if self.randomize_pipette:
                if self.holder == "rack":
                    shift = holder_xy * poses.new_tensor(self.rack_xy_half_range_m)
                else:
                    shift = holder_xy[:, :1] * self.pipette_slide_half_range_m * poses.new_tensor(self.rail_axis_xy)
                poses[:, 1, :2] += shift
            if self.randomize_tube:
                poses[:, 2, :2] += tube_xy * poses.new_tensor(self.tube_xy_half_range_m)
        result = dict(zip(("cabinet", "pipette", "tube"), poses.unbind(dim=1), strict=True))
        if self.holder == "rack":
            result["rack"] = self.rack_for_pipette(result["pipette"])
        return result
