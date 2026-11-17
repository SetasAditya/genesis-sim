import math
import random
import unittest
from wetlab_isaaclab.sandbox.config import ObjectState
from wetlab_isaaclab.sandbox.tasks import TaskEvaluator, VolumeModel


def state(p, **kw):
    return ObjectState(tuple(p), **kw)


def settle(e, s, **kw):
    for _ in range(20):
        e.update(s, 0.05, **kw)
    return e.success


class TaskTests(unittest.TestCase):
    def test_petri_open_requires_release_and_clearance(self):
        t = {"lid_park": (0.2, 0, 1.0), "lid_seated": (0, 0, 1.0)}
        e = TaskEvaluator("petri", t)
        s = {"dish": state((0, 0, 1)), "lid": state((0.2, 0, 1))}
        self.assertFalse(settle(e, s, attached=["lid"]))
        self.assertTrue(settle(e, s))

    def test_petri_close_rejects_misalignment(self):
        e = TaskEvaluator("petri", {"lid_seated": (0, 0, 1)}, mode="close")
        self.assertFalse(settle(e, {"lid": state((0.002, 0, 1))}))
        self.assertTrue(settle(e, {"lid": state((0, 0, 1))}))

    def test_tube_wrong_hole_and_tilt(self):
        e = TaskEvaluator("tube_insertion", {"tube_seated": (0, 0, 1)})
        self.assertFalse(settle(e, {"tube": state((0.03, 0, 1))}))
        self.assertFalse(settle(e, {"tube": state((0, 0, 1), up_dot=0.98)}))
        self.assertTrue(settle(e, {"tube": state((0, 0, 1))}))

    def test_reactor_sequence_and_unseated_part(self):
        names = [
            "basket",
            "well",
            "insert",
            "cover",
            "bolt0",
            "bolt1",
            "bolt2",
            "bolt3",
        ]
        targets = {n: (i * 0.02, 0, 1) for i, n in enumerate(names)}
        s = {n: state((i * 0.02, 0.1, 1)) for i, n in enumerate(names)}
        e = TaskEvaluator("reactor_assembly", targets)
        s["cover"] = state(targets["cover"])
        self.assertFalse(settle(e, s))
        self.assertEqual(e.stage, 0)
        for n in names:
            s[n] = state(targets[n])
            settle(e, s)
        self.assertTrue(e.success)
        e.reset()
        s["well"] = state((0.02, 0.01, 1))
        settle(e, s)
        self.assertFalse(e.success)
        self.assertEqual(e.stage, 1)

    def test_fridge_closed_door_retrieval_rejected(self):
        e = TaskEvaluator("fridge_retrieval", {"sample_target": (0, 0, 1)})
        s = {"sample": state((0, 0, 1))}
        self.assertFalse(settle(e, s, joints={"fridge_door": 0.0}))
        self.assertTrue(settle(e, s, joints={"fridge_door": math.radians(80)}))

    def test_pipette_no_tip_or_wrong_region(self):
        v = VolumeModel()
        v.update(0.5, False, None)
        v.update(0, False, "source")
        self.assertEqual(v.tip, 0.0)
        v.update(0.5, True, None)
        v.update(0, True, None)
        self.assertEqual(v.tip, 0.0)

    def test_pipette_transfer_and_ejection(self):
        v = VolumeModel()
        v.update(0.5, True, None)
        v.update(0, True, "source")
        self.assertEqual(v.tip, 100.0)
        v.update(0.5, True, "target")
        self.assertEqual(v.target, 100.0)
        v.eject(True)
        e = TaskEvaluator("pipetting", {})
        e.update({}, 0.05, attached=["tip"], volume=v.observation())
        self.assertTrue(settle(e, {}, volume=v.observation()))

    def test_outside_vessel_spills_cannot_succeed(self):
        v = VolumeModel()
        v.update(0.5, True, None)
        v.update(0, True, "source")
        v.update(0.5, True, None)
        self.assertEqual(v.target, 0.0)
        self.assertEqual(v.spilled, 100.0)
        self.assertEqual(v.source + v.tip + v.target + v.waste, 1000.0)
        e = TaskEvaluator("pipetting", {})
        e.update({}, 0.05, attached=["tip"], volume=v.observation())
        v.eject(True)
        self.assertFalse(settle(e, {}, volume=v.observation()))

    def test_volume_conservation_and_capacity(self):
        v = VolumeModel()
        r = random.Random(42)
        for _ in range(2000):
            v.update(r.random(), r.random() > 0.1, r.choice(["source", "target", None]))
            self.assertAlmostEqual(v.source + v.tip + v.target + v.waste, 1000.0)
            self.assertGreaterEqual(v.source, 0.0)
            self.assertGreaterEqual(v.tip, 0.0)
            self.assertLessEqual(v.tip, 200.0)

    def test_reset_clears_stages_and_volume(self):
        for _ in range(20):
            e = TaskEvaluator("tube_insertion", {"tube_seated": (0, 0, 1)})
            settle(e, {"tube": state((0, 0, 1))})
            e.reset()
            self.assertFalse(e.success)
            self.assertEqual(e.stage, 0)
            v = VolumeModel()
            v.update(1.0, True, None)
            v.update(0, True, "source")
            v.eject(True)
            v.reset()
            self.assertEqual(v.tip, 0.0)
            self.assertEqual(v.source, 1000.0)
            self.assertFalse(v.ejected)

    def test_invalid_state_fails(self):
        e = TaskEvaluator("tube_insertion", {"tube_seated": (0, 0, 1)})
        e.update({"tube": state((float("nan"), 0, 1))}, 0.05)
        self.assertEqual(e.failure_reason, "non_finite_state")


if __name__ == "__main__":
    unittest.main()
