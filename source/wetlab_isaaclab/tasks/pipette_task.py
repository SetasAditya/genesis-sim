"""Physical observations and event-based scoring for the cabinet pipette."""

import math

import torch

from .pipette_events import SustainedPress, nozzle_aligned


class CabinetPipetteTask:
    """Read the scene at control time after stepping it. Reset matching world IDs."""

    def __init__(self, scene, control_dt):
        self.scene = scene
        self.robot = scene["robot"]
        self.pipette = scene["pipette"]
        self.cabinet = scene["cabinet"]
        self.tube = scene["tube"]
        self.contact = scene.sensors.get("press_contact")
        self.palm = next(
            (self.robot.body_names.index(name) for name in ("rl_dg_mount", "base_link")
             if name in self.robot.body_names),
            None,
        )
        self.press = SustainedPress(scene.num_envs, control_dt, device=scene.device)
        self.tolerance = torch.tensor(
            [1e-4 if name in ("root_x_axis_joint", "root_y_axis_joint", "root_z_axis_joint")
             else math.radians(1) for name in self.robot.joint_names], device=scene.device,
        )
        self.initial_cabinet = torch.zeros((scene.num_envs, 7), device=scene.device)
        self.start_body = torch.zeros_like(self.initial_cabinet)
        self.press_offset = torch.zeros((scene.num_envs, 3), device=scene.device)
        self.press_started = torch.zeros(scene.num_envs, dtype=torch.bool, device=scene.device)

    def reset(self, env_ids):
        self.initial_cabinet[env_ids] = self.cabinet.data.root_link_pose_w.torch[env_ids]
        self.start_body[env_ids] = self.pipette.data.root_link_pose_w.torch[env_ids]
        self.press_started[env_ids] = False
        self.press.reset(env_ids)

    def observe(self):
        from isaaclab.utils.math import quat_apply

        def rotate(pose, vector):
            return quat_apply(pose[:, 3:], pose.new_tensor(vector).expand_as(pose[:, :3]))

        robot = self.robot.data
        body = self.pipette.data.root_link_pose_w.torch
        tube = self.tube.data.root_link_pose_w.torch
        travel = self.pipette.data.joint_pos.torch[:, 0]
        axis = rotate(body, (-0.1061736643, -0.0271494649, -0.9939768909))
        if self.contact is None:
            axial = torch.zeros_like(travel)
        else:
            contacts = self.contact.data
            force = (contacts.normal_force_matrix_w.torch + contacts.friction_force_matrix_w.torch)
            axial = (force.sum(dim=(1, 2)) * axis).sum(dim=-1)
        return {
            "robot_joint_pos": robot.joint_pos.torch,
            "robot_joint_vel": robot.joint_vel.torch,
            "palm_pose": None if self.palm is None else robot.body_link_pose_w.torch[:, self.palm],
            "pipette_pose": body,
            "tube_pose": tube,
            "cabinet_pose": self.cabinet.data.root_link_pose_w.torch,
            "tube_opening": tube[:, :3] + rotate(tube, (0, 0, 0.112)),
            "tube_axis": rotate(tube, (0, 0, 1)),
            "nozzle": body[:, :3] + rotate(body, (0.0016495, 0.0000139, -0.14165)),
            "plunger_travel": travel,
            "axial_force": axial,
        }

    def update(self):
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

        obs = self.observe()
        body, cabinet = obs["pipette_pose"], obs["cabinet_pose"]
        axis, opening = obs["tube_axis"], obs["tube_opening"]
        travel, axial = obs["plunger_travel"], obs["axial_force"]
        q = obs["robot_joint_pos"]
        limits = self.robot.data.joint_pos_limits.torch
        valid = ((q >= limits[..., 0] - self.tolerance) &
                 (q <= limits[..., 1] + self.tolerance)).all(dim=-1)
        for key, data in obs.items():
            if data is not None:
                valid &= torch.isfinite(data).reshape(q.shape[0], -1).all(dim=-1)
        valid &= travel >= -1e-4
        fixture_stable = (cabinet[:, :3] - self.initial_cabinet[:, :3]).norm(dim=-1) <= 0.10
        fixture_stable &= (cabinet[:, 3:] * self.initial_cabinet[:, 3:]).sum(dim=-1).abs() >= math.cos(math.pi / 6)

        initial_q = self.start_body[:, 3:]
        local_up = axis.new_tensor((0, 0, 1)).expand_as(axis)
        initial_up = quat_apply(initial_q, local_up)
        cross = torch.cross(initial_up, axis, dim=-1)
        cosine = (initial_up * axis).sum(dim=-1)
        correction = torch.cat((cross, (1 + cosine)[:, None]), dim=-1)
        fallback = torch.cross(initial_up, axis.new_tensor((1, 0, 0)).expand_as(axis), dim=-1)
        correction = torch.where((cosine < -0.999999)[:, None],
                                 torch.cat((fallback, torch.zeros_like(cosine[:, None])), dim=-1), correction)
        goal_q = quat_mul(torch.nn.functional.normalize(correction, dim=-1), initial_q)
        local_tip = axis.new_tensor((0.0016495, 0.0000139, -0.14165)).expand_as(axis)
        goal_pos = opening + 0.025 * axis - quat_apply(goal_q, local_tip)
        goal_reached = (body[:, :3] - goal_pos).norm(dim=-1) < 0.05
        goal_reached &= (body[:, 3:] * goal_q).sum(dim=-1).abs() > math.cos(math.pi / 12)
        aligned = nozzle_aligned(obs["nozzle"], opening, axis)
        upright = quat_apply(body[:, 3:], local_up)[:, 2] > 0.5

        pressing = valid & upright & fixture_stable & (travel >= 0.015) & (axial > 0.1)
        palm = obs["palm_pose"]
        if palm is None:
            pressing &= False
        else:
            local_cap = axis.new_tensor((0.0017477, 0.0003256, 0.1415928)).expand_as(axis)
            cap = body[:, :3] + quat_apply(body[:, 3:], local_cap)
            offset = quat_apply_inverse(palm[:, 3:], palm[:, :3] - cap)
            self.press_offset.copy_(torch.where((pressing & ~self.press_started)[:, None], offset, self.press_offset))
            pressing &= (offset - self.press_offset).norm(dim=-1) < 0.04
        self.press_started.copy_(pressing)
        held = self.press.update(travel, axial, pressing)
        return {"valid": valid, "fixture_stable": fixture_stable, "goal_reached": goal_reached,
                "aligned": aligned, "sustained_press": held,
                "success": valid & fixture_stable & goal_reached & aligned & held}
