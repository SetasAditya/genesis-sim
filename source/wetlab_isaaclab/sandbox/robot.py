"""Native Galbot joint control and joint-limited Cartesian inverse kinematics."""

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.types import ArticulationAction


def matrix(m):
    return np.array(m, dtype=float).T


def quat_matrix(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


class GalbotController:
    def __init__(self, stage, articulation):
        self.stage = stage
        self.robot = articulation
        self.names = list(articulation.dof_names)
        self.joints = {
            p.GetName(): UsdPhysics.Joint(p)
            for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Galbot"))
            if p.IsA(UsdPhysics.RevoluteJoint)
        }
        self.targets = self.robot.get_joint_positions().copy()
        self.wrist_goals = {}
        self.gripper_goals = {}
        self._base_was_moving = False
        self.base_anchor = None
        self.wheel_rows = []
        self.last_ik = {}
        self.tcp_offsets = {
            side: np.array([0.0, 0.0, 0.0]) for side in ("left", "right")
        }
        # Drive only real actuators; coupled fingers and omni-wheel rollers stay passive.
        kp = []
        kd = []
        for n in self.names:
            passive = n.startswith(("wheel_passive_", "gripper_"))
            kp.append(
                0.0
                if passive or n.startswith("wheel")
                else 200.0
                if "gripper" in n
                else 10000.0
                if "_arm_joint" in n
                else 60000.0
            )
            kd.append(
                0.0
                if passive
                else 10000.0
                if n.startswith("wheel")
                else 20.0
                if "gripper" in n
                else 300.0
                if "_arm_joint" in n
                else 600.0
            )
        articulation.get_articulation_controller().set_gains(
            kps=np.array(kp), kds=np.array(kd)
        )
        self.wrists = {}
        for side in ("left", "right"):
            self.wrists[side] = str(
                self.joints[side + "_arm_joint7"].GetBody1Rel().GetTargets()[0]
            )
        self._chains = {
            side: [self.joints[side + "_arm_joint" + str(i)] for i in range(1, 8)]
            for side in ("left", "right")
        }
        for i in range(1, 5):
            j = self.joints[f"wheel{i}_joint"]
            f = self._frame(j, "0")
            axle = f[:3, :3] @ np.array([1.0, 0.0, 0.0])
            tangent = np.cross(axle, [0.0, 0.0, 1.0])
            tangent /= np.linalg.norm(tangent)
            p = f[:3, 3]
            lever = p[0] * tangent[1] - p[1] * tangent[0]
            wp = stage.GetPrimAtPath(j.GetBody1Rel().GetTargets()[0])
            box = (
                UsdGeom.BBoxCache(0, ["default", "render"])
                .ComputeRelativeBound(
                    wp, stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
                )
                .ComputeAlignedRange()
            )
            radius = max(0.04, float(box.GetSize()[2]) / 2)
            self.wheel_rows.append(np.array([tangent[0], tangent[1], lever]) / radius)
        self._edges = []
        for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Galbot")):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            if b0 and b1:
                self._edges.append(
                    (
                        str(b0[0]),
                        str(b1[0]),
                        p.GetName(),
                        self._frame(j, "0"),
                        np.linalg.inv(self._frame(j, "1")),
                    )
                )
        # Model the TCP at the central pinch region in the real gripper's flange frame.
        self.tcp_matrices = {}
        for side in ("left", "right"):
            wrist = self.stage.GetPrimAtPath(self.wrists[side])
            flange = next(
                p for p in Usd.PrimRange(wrist) if p.GetName() == "gripper_flange"
            )
            f = matrix(UsdGeom.Xformable(flange).ComputeLocalToWorldTransform(0))
            w = matrix(UsdGeom.Xformable(wrist).ComputeLocalToWorldTransform(0))
            tcp = f.copy()
            tcp[:3, 3] = f[:3, :3] @ np.array([0.14, 0.0, 0.0]) + f[:3, 3]
            self.tcp_matrices[side] = np.linalg.inv(w) @ tcp

    def _frame(self, j, side):
        q = j.GetPrim().GetAttribute("physics:localRot" + side).Get()
        p = j.GetPrim().GetAttribute("physics:localPos" + side).Get()
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        t[:3, 3] = p
        return t

    def tcp(self, side):
        return self.link_world(self.wrists[side]) @ self.tcp_matrices[side]

    def magnetic_tcp(self):
        t = self.tcp("left").copy()
        t[:3, 3] += t[:3, 2] * 0.066
        return t

    def check_kinematics(self):
        from isaacsim.core.prims import SingleRigidPrim

        result = {}
        for side, path in self.wrists.items():
            body = SingleRigidPrim(
                path, name="fk_check_" + side, reset_xform_properties=False
            )
            body.initialize()
            p, q = body.get_world_pose()
            computed = self.link_world(path)
            error = float(np.linalg.norm(computed[:3, 3] - p))
            angular = float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        quat_matrix(q).T @ computed[:3, :3]
                    ).as_rotvec()
                )
            )
            result[side] = {
                "position_error_m": error,
                "angular_error_rad": angular,
                "passed": error < 0.002 and angular < 0.02,
            }
        return result

    def link_world(self, path):
        """Use actual PhysX root/joint state; USD transforms can lag headless steps."""
        position, quaternion = self.robot.get_world_pose()
        root = np.eye(4)
        root[:3, :3] = quat_matrix(quaternion)
        root[:3, 3] = position
        values = dict(zip(self.names, self.robot.get_joint_positions()))
        transforms = {"/World/Galbot/OmniChassis/base_link": root}
        pending = list(self._edges)
        while pending:
            progress = False
            for edge in pending[:]:
                b0, b1, n, f0, f1 = edge
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    [0, 0, values.get(n, 0.0)]
                ).as_matrix()
                delta = f0 @ motion @ f1
                if b0 in transforms:
                    transforms[b1] = transforms[b0] @ delta
                elif b1 in transforms:
                    transforms[b0] = transforms[b1] @ np.linalg.inv(delta)
                else:
                    continue
                pending.remove(edge)
                progress = True
                if path in transforms:
                    return transforms[path]
            if not progress:
                break
        if path in transforms:
            return transforms[path]
        raise ValueError("Disconnected robot link " + path)

    def solve(self, side, position, orientation=None):
        chain = self._chains[side]
        root = self.link_world(str(chain[0].GetBody0Rel().GetTargets()[0]))
        frames = [
            (self._frame(j, "0"), np.linalg.inv(self._frame(j, "1"))) for j in chain
        ]
        indices = [self.names.index(side + "_arm_joint" + str(i)) for i in range(1, 8)]
        current_raw = self.robot.get_joint_positions()[indices]
        current = np.arctan2(np.sin(current_raw), np.cos(current_raw))

        def fk(q):
            t = root.copy()
            for angle, (f0, f1) in zip(q, frames):
                r = np.eye(4)
                r[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
                t = t @ f0 @ r @ f1
            return t @ self.tcp_matrices[side]

        def fun(q):
            t = fk(q)
            r = 50 * (t[:3, 3] - np.asarray(position))
            if orientation is not None:
                r = np.r_[
                    r,
                    0.3
                    * Rotation.from_matrix(
                        quat_matrix(orientation).T @ t[:3, :3]
                    ).as_rotvec(),
                ]
            return np.r_[r, 0.002 * (q - current)]

        bounds = (
            np.array(
                [
                    [
                        float(j.GetPrim().GetAttribute("physics:lowerLimit").Get()),
                        float(j.GetPrim().GetAttribute("physics:upperLimit").Get()),
                    ]
                    for j in chain
                ]
            ).T
            * math.pi
            / 180
        )
        result = least_squares(
            fun,
            np.clip(current, bounds[0] + 1e-5, bounds[1] - 1e-5),
            bounds=bounds,
            max_nfev=100,
        )
        error = float(np.linalg.norm(fk(result.x)[:3, 3] - position))
        if error > 0.004:
            other = least_squares(
                fun,
                np.clip(np.zeros(7), bounds[0] + 1e-5, bounds[1] - 1e-5),
                bounds=bounds,
                max_nfev=160,
            )
            if np.linalg.norm(fun(other.x)) < np.linalg.norm(fun(result.x)):
                result = other
            error = float(np.linalg.norm(fk(result.x)[:3, 3] - position))
        angular_error = (
            None
            if orientation is None
            else float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        quat_matrix(orientation).T @ fk(result.x)[:3, :3]
                    ).as_rotvec()
                )
            )
        )
        self.last_ik[side] = {
            "position_error_m": error,
            "angular_error_rad": angular_error,
            "converged": error < 0.004,
        }
        if error > 0.015:
            raise ValueError(f"{side} wrist target unreachable: {error:.4f} m")
        return indices, result.x

    def command(self, action, dt):
        self.wrist_goals.update(action.wrist_targets)
        self.gripper_goals.update(action.grippers)
        for side, target in self.wrist_goals.items():
            if side not in ("left", "right"):
                raise ValueError("Arm must be left or right")
            indices, q = self.solve(side, target["position"], target.get("orientation"))
            # Limit command speed, not tracking error: gravity must not prevent
            # the commanded position from advancing toward its bounded IK goal.
            self.targets[indices] += np.clip(
                q - self.targets[indices], -dt * 0.15, dt * 0.15
            )
        for side, value in self.gripper_goals.items():
            if not 0.0 <= value <= 1.6:
                raise ValueError("Gripper target range is 0..1.6 rad")
            i = self.names.index(side + "_gripper_joint")
            self.targets[i] += float(
                np.clip(value - self.targets[i], -dt * 0.6, dt * 0.6)
            )
        vx, vy, w = action.base_velocity
        if max(abs(vx), abs(vy)) > 0.15 or abs(w) > 0.3:
            raise ValueError("Base velocity exceeds sandbox limit")
        # Derive the four omni-wheel axes from the USD rather than assume a mecanum layout.
        p, q = self.robot.get_world_pose()
        yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]
        moving = any(abs(v) > 1e-8 for v in (vx, vy, w))
        if moving or self.base_anchor is None:
            self.base_anchor = np.array([p[0], p[1], yaw])
        else:
            delta = self.base_anchor[:2] - p[:2]
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy = np.clip(
                3.0
                * np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]),
                -0.10,
                0.10,
            )
            w = float(
                np.clip(
                    3.0
                    * math.atan2(
                        math.sin(self.base_anchor[2] - yaw),
                        math.cos(self.base_anchor[2] - yaw),
                    ),
                    -0.20,
                    0.20,
                )
            )
        wheel = np.array(self.wheel_rows) @ np.array([vx, vy, w])
        self.robot.apply_action(ArticulationAction(joint_positions=self.targets))
        indices = np.array([self.names.index(f"wheel{k + 1}_joint") for k in range(4)])
        self.robot.apply_action(
            ArticulationAction(joint_velocities=wheel, joint_indices=indices)
        )
        self._base_was_moving = moving
