"""Vectorized Gymnasium interface for the rack pipette task."""

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from wetlab_isaaclab.robots.dg5f import dg5f_robot_cfg

from .pipette_layout import PipetteLayout, default_robot_base_pose
from .pipette_placement import screen_pipette_layouts
from .pipette_scene import cabinet_pipette_scene_cfg, reset_cabinet_pipette
from .pipette_task import CabinetPipetteTask


class RackPipetteEnv(DirectRLEnv):
    """Command absolute joint positions in native order at 20 Hz.

    State observations are returned as a flat `policy` tensor. Use
    `observation_slices` to read named fields and `joint_names` to build actions.
    Positions are relative to each bench, and quaternions are XYZW.
    """

    def __init__(self, *, assets, robot_kind="mounted", num_envs=1, device="cuda:0",
                 seed=0, layout=None, render_mode=None):
        if robot_kind not in ("mounted", "floating"):
            raise ValueError(f"Unsupported rack robot: {robot_kind}")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        assets = Path(assets)

        def usd(name):
            paths = sorted((assets / name).glob("*/*.usda"))
            if len(paths) != 1:
                raise ValueError(f"Expected one converted {name} USD under {assets}. Run asset preparation first.")
            return paths[0]

        layout = PipetteLayout() if layout is None else layout
        if layout.holder != "rack":
            raise ValueError("The rack benchmark requires holder='rack'")
        base = default_robot_base_pose(robot=robot_kind, holder="rack")
        if layout.randomize:
            layout = screen_pipette_layouts(
                layout, base_position=base[:3], base_rotation=base[3:],
                floating=robot_kind == "floating", seed=seed,
            )
        self.layout = layout
        self._fixed_layouts = None
        self._rng = torch.Generator(device=device).manual_seed(seed)
        robot = dg5f_robot_cfg(
            usd(robot_kind), floating=robot_kind == "floating",
            position=base[:3], orientation=base[3:],
        )
        scene = cabinet_pipette_scene_cfg(
            robot=robot, pipette_usd=usd("pipette"), tube_usd=usd("tube"),
            num_envs=num_envs, env_spacing=3.0, layout=layout,
        )
        # The spaces are resolved from the actual articulation before the base
        # class configures the vectorized Gym spaces.
        cfg = DirectRLEnvCfg(
            seed=seed, episode_length_s=25.0, decimation=5, scene=scene,
            action_space=1, observation_space=1, compute_final_obs=True,
            ui_window_class_type=None,
            sim=SimulationCfg(device=device, dt=0.01, render_interval=5,
                              physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=2.0)),
        )
        super().__init__(cfg, render_mode=render_mode)

    def _configure_gym_env_spaces(self):
        self.robot = self.scene["robot"]
        self.task = CabinetPipetteTask(self.scene, control_dt=self.step_dt)
        self.joint_names = tuple(self.robot.joint_names)
        count = len(self.joint_names)
        fields = {
            "robot_joint_pos": count, "robot_joint_vel": count, "palm_pose": 7,
            "pipette_pose": 7, "rack_pose": 7, "tube_pose": 7, "cabinet_pose": 7,
            "tube_opening": 3, "tube_axis": 3, "nozzle": 3,
            "plunger_travel": 1, "axial_force": 1,
        }
        self.observation_slices = {}
        start = 0
        for name, size in fields.items():
            self.observation_slices[name] = slice(start, start + size)
            start += size
        limits = self.robot.data.joint_pos_limits.torch[0].cpu().numpy()
        self.cfg.action_space = gym.spaces.Box(low=limits[:, 0], high=limits[:, 1])
        self.cfg.observation_space = start
        super()._configure_gym_env_spaces()

    def reset(self, *, seed=None, options=None):
        """Reset all worlds, optionally to one explicit rack layout per world."""
        if seed is not None:
            self._rng.manual_seed(seed)
        self.extras.clear()
        layouts = None if options is None else options.get("layouts")
        self._fixed_layouts = (_FixedRackLayouts(self.layout, layouts, self.num_envs, self.device)
                               if layouts is not None else None)
        return super().reset(seed=seed)

    def reset_indices(self, env_ids):
        """Reset only the selected worlds without advancing physics."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        self._reset_idx(ids)
        return self._get_observations()

    def _reset_idx(self, env_ids):
        poses = self._fixed_layouts.for_indices(env_ids) if self._fixed_layouts is not None else None
        reset_cabinet_pipette(self.scene, env_ids, self.layout, generator=self._rng, poses=poses)
        self.task.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.sim.render_context.reset_scene_state_cadence()

    def _pre_physics_step(self, action):
        self.extras.pop("final_obs", None)
        limits = self.robot.data.joint_pos_limits.torch
        self.actions = action.clamp(limits[..., 0], limits[..., 1])

    def _apply_action(self):
        self.robot.actuators.target_command.set_position_index(value=self.actions)

    def _get_observations(self):
        obs = self.task.observe()
        obs["rack_pose"] = self.scene["rack"].data.root_link_pose_w.torch
        for name in ("palm_pose", "pipette_pose", "rack_pose", "tube_pose", "cabinet_pose",
                     "tube_opening", "nozzle"):
            value = obs[name]
            pos = value[:, :3] - self.scene.env_origins
            obs[name] = torch.cat((pos, value[:, 3:]), dim=1)
        return {"policy": torch.cat([obs[name].reshape(self.num_envs, -1)
                                     for name in self.observation_slices], dim=1)}

    def _get_dones(self):
        events = self.task.update()
        self.extras["events"] = events
        terminated = events["success"] | ~events["valid"]
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        return terminated, truncated

    def _get_rewards(self):
        return self.extras["events"]["success"].float()


class _FixedRackLayouts:
    """Bench-relative layout assignments indexed like the vectorized worlds."""

    def __init__(self, layout, layouts, num_envs, device):
        if len(layouts) != num_envs:
            raise ValueError("Provide exactly one rack layout per world")
        nominal = layout.nominal if hasattr(layout, "nominal") else layout
        rows = []
        for item in layouts:
            if any(len(item[name]) != 7 for name in ("rack", "tube")):
                raise ValueError("Each rack and tube pose needs seven elements")
            # Derive the pipette seat from the supplied rack transform.
            fixed = replace(nominal, rack=tuple(item["rack"]), tube=tuple(item["tube"]),
                            pipette=None, randomize=False)
            poses = fixed.sample(1, device="cpu")
            rows.append({name: value[0] for name, value in poses.items()})
        self.poses = {name: torch.stack([row[name] for row in rows]) for name in rows[0]}
        if not all(torch.isfinite(tensor).all() for tensor in self.poses.values()):
            raise ValueError("Nonfinite rack layout")
        for tensor in self.poses.values():
            if not torch.allclose(tensor[:, 3:].norm(dim=1), torch.ones(num_envs), atol=1e-5):
                raise ValueError("Layout orientations must be unit WXYZ quaternions")
        self.poses = {name: tensor.to(device) for name, tensor in self.poses.items()}

    def for_indices(self, env_ids):
        return {name: tensor[env_ids.long()] for name, tensor in self.poses.items()}
