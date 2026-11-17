"""State-based success checks; no simulator imports or object-motion shortcuts."""

import math
from .config import TASK_IDS


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class VolumeModel:
    """Volume transfer is driven by measured plunger motion, never a transfer command."""

    def __init__(self, capacity_ul=200.0, requested_ul=100.0):
        self.capacity = capacity_ul
        self.requested = requested_ul
        self.reset()

    def reset(self):
        self.source = 1000.0
        self.tip = 0.0
        self.target = 0.0
        self.waste = 0.0
        self.spilled = 0.0
        self.previous = 0.0
        self.aspirated = 0.0
        self.dispensed = 0.0
        self.ejected = False

    def update(self, stroke_fraction, tip_attached, region):
        stroke = max(0.0, min(1.0, stroke_fraction))
        delta = (stroke - self.previous) * self.capacity
        self.previous = stroke
        if not tip_attached:
            return
        if delta < 0 and region == "source":
            v = min(-delta, self.source, self.capacity - self.tip)
            self.source -= v
            self.tip += v
            self.aspirated += v
        elif delta > 0 and region == "target":
            v = min(delta, self.tip, 2000.0 - self.target)
            self.tip -= v
            self.target += v
            self.dispensed += v
        elif delta > 0 and region == "source":
            v = min(delta, self.tip, 2000.0 - self.source)
            self.tip -= v
            self.source += v
        elif delta > 0:
            v = min(delta, self.tip)
            self.tip -= v
            self.waste += v
            self.spilled += v

    def eject(self, over_bin):
        if over_bin:
            self.waste += self.tip
            self.tip = 0.0
            self.ejected = True

    def observation(self):
        return {
            "source_ul": self.source,
            "tip_ul": self.tip,
            "target_ul": self.target,
            "waste_ul": self.waste,
            "spilled_ul": self.spilled,
            "requested_ul": self.requested,
            "ejected": self.ejected,
        }


class TaskEvaluator:
    def __init__(self, task, targets, mode="open", stable_seconds=0.5):
        if task not in TASK_IDS:
            raise ValueError(f"Unknown task: {task}")
        if mode not in ("open", "close"):
            raise ValueError("Petri mode must be open or close")
        self.task = task
        self.targets = targets
        self.mode = mode
        self.stable_seconds = stable_seconds
        self.reset()

    def reset(self):
        self.stage = 0
        self.stable = 0.0
        self.elapsed = 0.0
        self.success = False
        self.failure_reason = None
        self.history = []
        self.door_opened = False
        self.tip_seen = False

    def _placed(self, state, name, target, attached, xy=0.004, z=0.003, upright=0.985):
        o = state.get(name)
        return bool(
            o
            and name not in attached
            and distance(o.position[:2], target[:2]) <= xy
            and abs(o.position[2] - target[2]) <= z
            and o.up_dot >= upright
            and o.linear_speed < 0.02
            and o.angular_speed < 0.15
        )

    def update(self, state, dt, attached=(), joints=None, volume=None):
        self.elapsed += dt
        joints = joints or {}
        attached = set(attached)
        if self.success:
            return self.result()
        for name, o in state.items():
            if (
                not all(
                    math.isfinite(x)
                    for x in (
                        *o.position,
                        *o.orientation,
                        o.up_dot,
                        o.linear_speed,
                        o.angular_speed,
                    )
                )
                or abs(sum(q * q for q in o.orientation) - 1.0) > 0.001
            ):
                self.failure_reason = "non_finite_state"
                return self.result()
        good = False
        if self.task == "petri":
            target = self.targets["lid_park" if self.mode == "open" else "lid_seated"]
            good = self._placed(
                state,
                "lid",
                target,
                attached,
                xy=0.0008 if self.mode == "close" else 0.008,
                z=0.002,
            )
            if self.mode == "open" and good:
                good = (
                    distance(state["lid"].position[:2], state["dish"].position[:2])
                    > 0.10
                )
        elif self.task == "tube_insertion":
            good = self._placed(
                state,
                "tube",
                self.targets["tube_seated"],
                attached,
                xy=0.0003,
                z=0.001,
                upright=0.999,
            )
        elif self.task == "reactor_assembly":
            ordered = [
                "basket",
                "well",
                "insert",
                "cover",
                "bolt0",
                "bolt1",
                "bolt2",
                "bolt3",
            ]
            if self.stage < len(ordered):
                n = ordered[self.stage]
                good = self._placed(
                    state,
                    n,
                    self.targets[n],
                    attached,
                    xy=0.001 if n.startswith("bolt") else 0.002,
                    z=0.0015,
                )
                # Later pieces cannot conceal an earlier part becoming unseated.
                good = good and all(
                    self._placed(
                        state, k, self.targets[k], attached, xy=0.002, z=0.0025
                    )
                    for k in ordered[: self.stage]
                )
        elif self.task == "fridge_retrieval":
            self.door_opened |= joints.get("fridge_door", 0.0) >= math.radians(65)
            good = self.door_opened and self._placed(
                state, "sample", self.targets["sample_target"], attached, xy=0.01
            )
        elif self.task == "pipetting":
            volume = volume or {}
            self.tip_seen |= "tip" in attached
            good = (
                self.tip_seen
                and volume.get("ejected", False)
                and abs(
                    volume.get("target_ul", 0.0) - volume.get("requested_ul", 100.0)
                )
                <= 2.0
                and volume.get("tip_ul", 0.0) <= 2.0
                and volume.get("spilled_ul", 0.0) <= 0.5
                and "tip" not in attached
            )
        self.stable = self.stable + dt if good else 0.0
        if self.stable >= self.stable_seconds:
            self.history.append({"stage": self.stage, "time": self.elapsed})
            self.stage += 1
            self.stable = 0.0
            self.success = self.task != "reactor_assembly" or self.stage == 8
        return self.result()

    def result(self):
        return {
            "task": self.task,
            "mode": self.mode,
            "stage": self.stage,
            "success": self.success,
            "elapsed_seconds": self.elapsed,
            "failure_reason": self.failure_reason,
            "stage_history": list(self.history),
        }
