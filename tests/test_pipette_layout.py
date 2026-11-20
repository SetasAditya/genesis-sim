"""The deployment rails stay fixed and each movable object has its own switch."""

from dataclasses import replace

import pytest


@pytest.mark.parametrize("holder", ["rack", "frame-edge"])
def test_placement_toggles_preserve_independent_draws(holder):
    torch = pytest.importorskip("torch")

    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout
    from wetlab_isaaclab.tasks.pipette_placement import screen_pipette_layouts

    nominal = PipetteLayout(holder=holder)

    def sample(layout, screened):
        if screened:
            layout = screen_pipette_layouts(
                layout, base_position=(-0.18, -0.64, 0.32), base_rotation=(1., 0., 0., 0.),
                floating=True, seed=41,
            )
        return layout.sample(16, device="cpu", generator=torch.Generator().manual_seed(12))

    for screened in (False, True):
        both = sample(nominal, screened)
        assert torch.unique(both["pipette"], dim=0).shape[0] > 1
        assert torch.unique(both["tube"], dim=0).shape[0] > 1
        for name, other in (("tube", "pipette"), ("pipette", "tube")):
            result = sample(replace(nominal, **{f"randomize_{name}": False}), screened)
            torch.testing.assert_close(result["cabinet"], torch.tensor(nominal.cabinet).expand(16, -1))
            torch.testing.assert_close(result[name], torch.tensor(getattr(nominal, name)).expand(16, -1))
            torch.testing.assert_close(result[other], both[other])
        fixed = sample(replace(nominal, randomize=False), screened)
        for name, poses in fixed.items():
            torch.testing.assert_close(poses, torch.tensor(getattr(nominal, name)).expand(16, -1))


def test_default_frame_sits_at_the_back_edge_of_the_bench():
    pytest.importorskip("torch")

    from wetlab_isaaclab.equipment.rail_frame import RailFrame
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout

    layout = PipetteLayout(frame=RailFrame(rail_top=0.4))
    assert layout.frame.edge_center[2] == pytest.approx(0.4)
    assert (layout.frame.tube_width, layout.frame.tube_height) == (.100, .050)
    uprights = list(layout.frame.boxes())[:2]
    assert all(size == (.050, .100, layout.frame.height) for _, size in uprights)
    assert layout.cabinet[0] == layout.bench.center_xy[0]
    assert layout.cabinet[1] + layout.frame.depth / 2 == pytest.approx(
        layout.bench.center_xy[1] + layout.bench.depth / 2
    )
    deep_posts = PipetteLayout(frame=RailFrame(post_width=.035, post_depth=.120))
    assert deep_posts.cabinet[1] + .120 / 2 == pytest.approx(
        deep_posts.bench.center_xy[1] + deep_posts.bench.depth / 2
    )


def test_edge_pose_follows_the_front_top_of_the_configured_frame():
    pytest.importorskip("torch")
    from wetlab_isaaclab.equipment.rail_frame import RailFrame
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout

    default = PipetteLayout(holder="frame-edge")
    assert default.pipette == pytest.approx((0.160000000, 0.230640879, 0.344489598, 0.565250980,
                                             0.095548225, 0.098567850, -0.813416403))
    yaw = 2**-.5
    rotated = PipetteLayout(holder="frame-edge", cabinet=(.2, .3, .1, yaw, 0., 0., yaw),
                            frame=RailFrame(rail_top=.6, tube_width=.08))
    assert rotated.pipette[:3] == pytest.approx((0.239359121, 0.300000000, 0.594489598))
    assert rotated.pipette[3:] == pytest.approx((0.974865056, -0.002135197, 0.137260793, -0.175479453), abs=1e-7)


def test_pipette_moves_only_along_the_rail():
    torch = pytest.importorskip("torch")

    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout

    # A 90-degree yaw puts the long dimension of the frame along world Y.
    half_turn = 2**-0.5
    layout = PipetteLayout(
        holder="frame-edge", cabinet=(.10, -.18, 0., half_turn, 0., 0., half_turn),
        pipette=(.14, -.18, .36, 1., 0., 0., 0.),
    )
    pipette = layout.sample(16, device="cpu", generator=torch.Generator().manual_seed(41))["pipette"]
    torch.testing.assert_close(pipette[:, 0], torch.full((16,), layout.pipette[0]))
    assert torch.unique(pipette[:, 1]).numel() > 1
    assert all(size[1] < .15 for _, size in layout.frame.boxes())
    posts = [(center, size) for center, size in layout.frame.boxes() if size[2] == layout.frame.height]
    assert len(posts) == 2
    assert all(center[1] == 0 for center, _ in layout.frame.boxes())


def test_rack_moves_with_pipette_and_tube_stays_independent():
    torch = pytest.importorskip("torch")

    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout
    from wetlab_isaaclab.tasks.pipette_placement import screen_pipette_layouts

    nominal = PipetteLayout(holder="rack")
    assert nominal.pipette == pytest.approx((-.108430399, -.434322240, .172202957, .526312899,
                                              -.105302074, .171433301, .826145767))
    unrotated = PipetteLayout(holder="rack", rack=(0., 0., 0., 1., 0., 0., 0.))
    dx, dy, dz = (v - t for v, t in zip(nominal.pipette[:3], nominal.rack[:3]))
    assert unrotated.pipette[:3] == pytest.approx((dy, -dx, dz))
    w, x, y, z = nominal.pipette[3:]
    assert unrotated.pipette[3:] == pytest.approx(tuple(
        k * 2**-.5 for k in (w + z, x + y, y - x, z - w)
    ))

    def sample(layout, screened):
        if screened:
            layout = screen_pipette_layouts(
                layout, base_position=(-.18, -.64, .32), base_rotation=(1., 0., 0., 0.),
                floating=True,
            )
        return layout.sample(16, device="cpu", generator=torch.Generator().manual_seed(12))

    for screened in (False, True):
        result = sample(nominal, screened)
        fixed = sample(replace(nominal, randomize_pipette=False), screened)
        offset = torch.tensor(nominal.pipette[:3]) - torch.tensor(nominal.rack[:3])
        torch.testing.assert_close(result["pipette"][:, :3] - result["rack"][:, :3], offset.expand(16, -1))
        torch.testing.assert_close(result["tube"], fixed["tube"])
        torch.testing.assert_close(fixed["rack"], torch.tensor(nominal.rack).expand(16, -1))
        assert torch.unique(result["rack"], dim=0).shape[0] > 1


def test_mounted_frame_layout_screen_reaches_both_objects():
    pytest.importorskip("torch")
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout, default_robot_base_pose
    from wetlab_isaaclab.tasks.pipette_placement import screen_pipette_layouts

    layout = PipetteLayout(holder="frame-edge")
    base = default_robot_base_pose(robot="mounted", holder=layout.holder)
    assert base == default_robot_base_pose(robot="vega", holder=layout.holder)
    bank = screen_pipette_layouts(layout, base_position=base[:3], base_rotation=base[3:],
                                  floating=False, seed=41, count=4)
    assert len(bank.pipette_poses) == len(bank.tube_poses) == 4
