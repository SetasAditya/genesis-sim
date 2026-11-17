"""Conservative robot-action trials. A failed grasp is a failed trial, never a teleport."""

import math
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .config import Action, CONTROL_DT
from .tasks import distance


def quat(r):
    x, y, z, w = Rotation.from_matrix(np.array(r, dtype=float)).as_quat()
    return [w, x, y, z]


PINCH_DOWN = quat([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
MAGNET_DOWN = quat([[0, 1, 0], [1, 0, 0], [0, 0, -1]])


class ScriptedTrial:
    def __init__(self, env):
        self.env = env
        self.log = []
        self.last_info = {}
        self.on_step = None

    def step(self, action):
        result = self.env.step(action)
        self.last_info = result[-1]
        if self.env.steps % 30 == 0:
            (self.env.output / "live_state.json").write_text(
                json.dumps(result[0], indent=2)
            )
        if self.on_step:
            self.on_step()
        if self.last_info.get("failure_reason"):
            raise RuntimeError(self.last_info["failure_reason"])
        return result

    def wait(self, seconds=1.0, **kwargs):
        for _ in range(int(seconds / CONTROL_DT)):
            self.step(Action(**kwargs))

    def move(
        self,
        position,
        side="right",
        orientation=PINCH_DOWN,
        seconds=4.0,
        settle_seconds=2.0,
        base_velocity=(0.0, 0.0, 0.0),
    ):
        print("WRIST_MOVE", side, np.asarray(position).round(4).tolist(), flush=True)
        start = self.env.controller.tcp(side)[:3, 3].copy()
        goal = np.array(position)
        initial_rotation = Rotation.from_matrix(self.env.controller.tcp(side)[:3, :3])
        target_rotation = Rotation.from_quat(
            [orientation[1], orientation[2], orientation[3], orientation[0]]
        )
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_quat([initial_rotation.as_quat(), target_rotation.as_quat()]),
        )
        count = max(
            int(seconds / CONTROL_DT), int(np.linalg.norm(goal - start) / 0.002)
        )
        for i in range(count):
            fraction = (i + 1) / count
            p = start + (goal - start) * fraction
            x, y, z, w = slerp([fraction]).as_quat()[0]
            self.step(
                Action(
                    wrist_targets={
                        side: {"position": p.tolist(), "orientation": [w, x, y, z]}
                    },
                    base_velocity=base_velocity,
                )
            )
        for _ in range(int(settle_seconds / CONTROL_DT)):
            self.step(
                Action(
                    wrist_targets={
                        side: {"position": goal.tolist(), "orientation": orientation}
                    },
                    base_velocity=base_velocity,
                )
            )
        actual = self.env.controller.tcp(side)[:3, 3]
        error = distance(actual, goal)
        if settle_seconds and error > 0.006:
            raise RuntimeError(f"wrist_tracking_error_{error:.4f}m")

    def drive_to(self, x, y, seconds=25.0):
        print("BASE_MOVE", x, y, flush=True)
        self.env.controller.wrist_goals.clear()
        for _ in range(int(seconds / CONTROL_DT)):
            p, q = self.env.robot.get_world_pose()
            yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]
            delta = np.array([x, y]) - p[:2]
            angle = math.atan2(math.sin(math.pi / 2 - yaw), math.cos(math.pi / 2 - yaw))
            if np.linalg.norm(delta) < 0.008 and abs(angle) < 0.03:
                self.wait(0.7)
                return
            c, s = math.cos(yaw), math.sin(yaw)
            v = np.clip(
                1.4
                * np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]),
                -0.10,
                0.10,
            )
            self.step(
                Action(
                    base_velocity=(
                        float(v[0]),
                        float(v[1]),
                        float(np.clip(2 * angle, -0.2, 0.2)),
                    )
                )
            )
        raise RuntimeError("base_waypoint_not_reached")

    def prepare_bench(self):
        for side in ("left", "right"):
            t = self.env.controller.tcp(side)
            orientation = quat(t[:3, :3])
            self.move(
                t[:3, 3] + [0, 0, 0.50],
                side,
                orientation,
                seconds=8.0,
                settle_seconds=4.0,
            )
        p, _ = self.env.robot.get_world_pose()
        self.drive_to(float(p[0]), 2.90)

    def grasp_point(self, name):
        p, q = self.env._pose(name)
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        if name == "lid":
            return p + r.apply([0.0, -0.050, 0.031])
        if name.startswith("bolt"):
            return p + r.apply([0.0, 0.0, 0.039])
        return p + r.apply([0.0, 0.0, self.env.sizes[name][2] * 0.65])

    def pick(self, name, side="right"):
        p = self.grasp_point(name)
        before = self.env._pose(name)[0].copy()
        self.wait(2.8, grippers={side: 0.0})
        # Rotate the hand clear of the bench before crossing its front edge.
        t = self.env.controller.tcp(side)
        self.move(t[:3, 3], side, seconds=10.0, settle_seconds=4.0)
        self.move(p + [0, 0, 0.22], side, seconds=6.0, settle_seconds=4.0)
        self.move(p, side, seconds=2.0)
        self.wait(3.0, grippers={side: 1.6})
        self.move(p + [0, 0, 0.10], side, seconds=2.0)
        lifted = self.env._pose(name)[0][2] - before[2]
        self.log.append({"action": "pick", "object": name, "lift_m": float(lifted)})
        if lifted < 0.04:
            raise RuntimeError(f"physical_grasp_failed:{name}")
        return self.env.controller.tcp(side)[:3, 3] - self.env._pose(name)[0]

    def place(self, name, target, offset, side="right"):
        release_gap = (
            0.012 if name.startswith("bolt") else 0.030 if name == "well" else 0.0
        )
        p = np.array(target) + offset + [0, 0, release_gap]
        self.move(p + [0, 0, 0.12], side)
        self.move(p, side, seconds=3.0)
        self.wait(2.8, grippers={side: 0.0})
        self.move(p + [0, 0, 0.12], side, seconds=2.0)
        self.wait(1.0)
        self.log.append(
            {
                "action": "place",
                "object": name,
                "actual": self.env._pose(name)[0].tolist(),
                "target": list(target),
            }
        )

    def transfer(self, name, target, side="right"):
        offset = self.pick(name, side)
        self.place(name, target, offset, side)

    def magnetic_transfer(self, name, target):
        side = "left"
        p = self.env._pose(name)[0] + [
            0,
            0,
            (0.0092 if name == "basket" else self.env.sizes[name][2]) + 0.066,
        ]
        self.move(p + [0, 0, 0.12], side, MAGNET_DOWN)
        self.move(p, side, MAGNET_DOWN, seconds=3.0)
        self.wait(0.5, magnetic=True)
        if name not in self.env.attachments:
            raise RuntimeError(f"magnet_contact_gate_failed:{name}")
        offset = self.env.controller.tcp(side)[:3, 3] - self.env._pose(name)[0]
        self.move(p + [0, 0, 0.12], side, MAGNET_DOWN)
        self.move(np.array(target) + offset + [0, 0, 0.12], side, MAGNET_DOWN)
        self.move(np.array(target) + offset, side, MAGNET_DOWN)
        self.wait(0.5, magnetic=False)
        self.move(np.array(target) + offset + [0, 0, 0.12], side, MAGNET_DOWN)
        self.wait(1.0)

    def run(self):
        try:
            task = self.env.task
            if task != "fridge_retrieval":
                self.prepare_bench()
            if task == "petri":
                self.transfer(
                    "lid",
                    self.env.targets[
                        "lid_park" if self.env.mode == "open" else "lid_seated"
                    ],
                    "left",
                )
            elif task == "tube_insertion":
                self.transfer("tube", self.env.targets["tube_seated"])
            elif task == "reactor_assembly":
                self.magnetic_transfer("basket", self.env.targets["basket"])
                self.wait(0.5, fixture_clamp=True)
                for n in ("well", "insert"):
                    self.transfer(n, self.env.targets[n])
                self.magnetic_transfer("cover", self.env.targets["cover"])
                for i in range(4):
                    self.transfer("bolt" + str(i), self.env.targets["bolt" + str(i)])
            elif task == "fridge_retrieval":
                # Pull the handle along the passive hinge arc; never command door state.
                fp = self.env.fridge_position
                hinge = fp + [-0.3, -0.30, 0.0]
                p = hinge + [0.52, -0.04, 0.65]
                self.wait(2.8, grippers={"right": 0.0})
                self.move(p + [0, -0.10, 0.0])
                self.move(p)
                self.wait(3.0, grippers={"right": 1.6})
                for angle in np.linspace(0, -math.radians(80), 80):
                    c, s = math.cos(angle), math.sin(angle)
                    self.move(
                        hinge + [c * 0.52 + s * 0.04, s * 0.52 - c * 0.04, 0.65],
                        seconds=0.08,
                        settle_seconds=0.0,
                        base_velocity=(-0.025, 0, 0),
                    )
                self.wait(2.8, grippers={"right": 0.0})
                if not self.env.evaluator.door_opened:
                    raise RuntimeError("door_contact_pull_failed")
                tcp = self.env.controller.tcp("right")[:3, 3]
                self.move(tcp + [0.20, -0.10, 0.10])
                self.drive_to(-6.65, 1.20)
                self.drive_to(-6.45, 1.20)
                self.drive_to(-6.45, 1.80)
                offset = self.pick("sample", "left")
                tcp = self.env.controller.tcp("left")[:3, 3]
                self.move([tcp[0], 1.90, tcp[2]], "left")
                self.drive_to(-5.95, 1.80)
                self.drive_to(-5.95, 2.80)
                self.place("sample", self.env.targets["sample_target"], offset, "left")
            elif task == "pipetting":
                self.pipette_cycle()
            self.wait(1.0)
            passed = self.env.evaluator.success and not self.env.debug_used
            return {
                "passed": bool(passed),
                "task": task,
                "seed": self.env.seed,
                "mode": self.env.mode,
                "failure_reason": None if passed else "success_condition_not_met",
                "events": self.log,
                "task_state": self.env.evaluator.result(),
                "debug_used": self.env.debug_used,
            }
        except (ValueError, RuntimeError) as e:
            try:
                tcp = {
                    s: self.env.controller.tcp(s)[:3, 3].tolist()
                    for s in ("left", "right")
                }
            except Exception:
                tcp = None
            return {
                "passed": False,
                "task": self.env.task,
                "seed": self.env.seed,
                "mode": self.env.mode,
                "failure_reason": str(e),
                "events": self.log,
                "task_state": self.env.evaluator.result(),
                "debug_used": self.env.debug_used,
                "ik": self.env.controller.last_ik,
                "tcp": tcp,
            }

    def pipette_cycle(self):
        offset = self.pick("pipette", "right")
        tip, _ = self.env._pose("tip")
        self.move(tip + [0, 0, 0.04] + offset + [0, 0, 0.05])
        self.move(tip + [0, 0, 0.04] + offset, seconds=3.0)
        self.wait(0.5)
        if not self.env.tip_attached:
            raise RuntimeError("tip_engagement_failed")

        # Hold the pipette with one arm; press its spring-return plunger with the other.
        def press(fraction):
            pp, pq = self.env._pose("pipette")
            button = pp + np.array([0, 0, 0.165 - 0.008 * fraction])
            self.move(button, "left", PINCH_DOWN, seconds=2.0)
            for _ in range(120):
                actual = -float(self.env.pipette_art.get_joint_positions()[0]) / 0.008
                if abs(actual - fraction) < 0.005:
                    return
                button[2] -= float(
                    np.clip(0.008 * (fraction - actual) * 0.4, -0.00025, 0.00025)
                )
                self.step(
                    Action(
                        wrist_targets={
                            "left": {
                                "position": button.tolist(),
                                "orientation": PINCH_DOWN,
                            }
                        }
                    )
                )
            raise RuntimeError("plunger_contact_tracking_failed")

        def lower_pressed(goal, fraction):
            start = self.env._pose("pipette")[0]
            for a in np.linspace(0.0, 1.0, 120):
                pp = start + (np.array(goal) - start) * a
                self.step(
                    Action(
                        wrist_targets={
                            "right": {
                                "position": (pp + offset).tolist(),
                                "orientation": PINCH_DOWN,
                            },
                            "left": {
                                "position": (
                                    pp + [0, 0, 0.165 - 0.008 * fraction]
                                ).tolist(),
                                "orientation": PINCH_DOWN,
                            },
                        }
                    )
                )

        source = self.env._pose("source_vessel")[0]
        self.move(source + [0, 0, 0.065] + offset)
        fraction = self.env.volume.requested / self.env.volume.capacity
        press(fraction)
        lower_pressed(source + [0, 0, 0.048], fraction)
        press(fraction)
        pp, _ = self.env._pose("pipette")
        self.move(pp + [0, 0, 0.22], "left", PINCH_DOWN, seconds=2.0)
        self.wait(1.0)
        target = self.env._pose("target_vessel")[0]
        self.move(target + [0, 0, 0.060] + offset)
        press(fraction)
        self.wait(1.0)
        self.move(np.array(self.env.initial["tip_bin"]) + [0, 0, 0.10] + offset)
        press(1.0)
        self.wait(1.0)
