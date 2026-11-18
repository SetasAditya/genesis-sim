"""The deployment shelves stay fixed and each movable object has its own switch."""

from dataclasses import replace

import pytest


def test_placement_toggles_preserve_independent_draws():
    torch = pytest.importorskip("torch")

    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout
    from wetlab_isaaclab.tasks.pipette_placement import screen_pipette_layouts

    nominal = PipetteLayout()

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


def test_default_pipette_pose_follows_shelf_height():
    pytest.importorskip("torch")

    from wetlab_isaaclab.equipment.shelf_frame import ShelfFrame
    from wetlab_isaaclab.tasks.pipette_layout import PipetteLayout

    layout = PipetteLayout(frame=ShelfFrame(shelf_top=0.4))
    assert layout.pipette[2] == pytest.approx(0.421335)
