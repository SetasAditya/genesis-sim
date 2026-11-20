"""Configurable rail frame and lab bench, with dimensions in metres."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RailFrame:
    """One upright frame, with both uprights and rails in a single plane."""

    width: float = 1.20
    height: float = 1.00
    rail_top: float = 0.45
    post_width: float = 0.050
    tube_width: float = 0.100
    tube_height: float = 0.050
    post_depth: float = 0.100

    @property
    def depth(self):
        return max(self.post_depth, self.tube_width)

    @property
    def edge_center(self):
        """The front top edge of the lower member."""
        return (0., -self.tube_width / 2, self.rail_top)

    def boxes(self):
        """Configurable rectangular members in local frame coordinates."""
        post = self.post_width
        x = (self.width - post) / 2
        span = self.width - 2 * post
        for side_x in (-x, x):
            yield (side_x, 0., self.height / 2), (post, self.post_depth, self.height)
        for top in (self.height, self.rail_top):
            yield (0., 0., top - self.tube_height / 2), (
                span, self.tube_width, self.tube_height,
            )


@dataclass(frozen=True)
class LabBench:
    """The bench top is the environment's z=0 work surface."""

    center_xy: tuple[float, float] = (0.16, -0.10)
    width: float = 1.55
    depth: float = 0.86
    height: float = 0.865
    top_thickness: float = 0.039
    leg_width: float = 0.042

    def boxes(self):
        yield (0., 0., -self.top_thickness / 2), (self.width, self.depth, self.top_thickness)
        x = self.width / 2 - 1.5 * self.leg_width
        y = self.depth / 2 - 1.5 * self.leg_width
        length = self.height - self.top_thickness
        for side_x in (-x, x):
            for side_y in (-y, y):
                yield (side_x, side_y, -self.top_thickness - length / 2), (
                    self.leg_width, self.leg_width, length,
                )
