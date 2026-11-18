import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from wetlab_isaaclab.tasks.pipette_events import SustainedPress, nozzle_aligned


@unittest.skipIf(torch is None, "PyTorch is required for batched task events")
class PipetteEventsTests(unittest.TestCase):
    def test_interruption_and_subset_reset(self):
        press = SustainedPress(3, 0.05, device="cpu")
        travel = torch.tensor([0.016, 0.016, 0.019])
        force = torch.tensor([0.11, 0.11, 0.11])
        retained = torch.ones(3, dtype=torch.bool)
        for _ in range(19):
            self.assertFalse(press.update(travel, force, retained).any())
        force[1] = 0.1
        self.assertEqual(press.update(travel, force, retained).tolist(), [True, False, True])
        press.reset(torch.tensor([2]))
        force[1] = 0.11
        self.assertEqual(press.update(travel, force, retained).tolist(), [True, False, False])

    def test_alignment_uses_tube_axis(self):
        nozzle = torch.tensor([[0.01, 0.006, 0.0], [0.01, 0.008, 0.0], [-0.01, 0.0, 0.0]])
        opening = torch.zeros(3)
        axis = torch.tensor([1.0, 0.0, 0.0])
        self.assertEqual(nozzle_aligned(nozzle, opening, axis).tolist(), [True, False, False])
