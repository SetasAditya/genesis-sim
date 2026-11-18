"""Configurable open shelf and lab bench, with dimensions in metres."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShelfFrame:
    """The frame origin is at the center of its footprint on the bench top."""

    width: float = 1.20
    depth: float = 0.36
    height: float = 0.78
    shelf_top: float = 0.275
    shelf_thickness: float = 0.024
    post_width: float = 0.027

    def boxes(self):
        """Local centers and sizes for both the collision and visual geometry."""
        post = self.post_width
        x = (self.width - post) / 2
        y = (self.depth - post) / 2
        for side_x in (-x, x):
            for side_y in (-y, y):
                yield (side_x, side_y, self.height / 2), (post, post, self.height)
        yield (0., 0., self.shelf_top - self.shelf_thickness / 2), (
            self.width - 2 * post, self.depth, self.shelf_thickness,
        )
        top = self.height - post / 2
        for side_y in (-y, y):
            yield (0., side_y, top), (self.width - 2 * post, post, post)
        for side_x in (-x, x):
            yield (side_x, 0., top), (post, self.depth - 2 * post, post)


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
