"""Vectorized physical events for the pipette task."""

import math

import torch


def nozzle_aligned(nozzle, opening, axis):
    """Test a 7 mm bore and a 50 mm open height interval."""
    delta = nozzle - opening
    height = (delta * axis).sum(dim=-1)
    lateral = delta - height[..., None] * axis
    return (lateral.square().sum(dim=-1) < 0.007**2) & (height > 0) & (height < 0.05)


class SustainedPress:
    """Track a continuous one-second press for each environment."""

    def __init__(self, num_envs: int, control_dt: float, *, device):
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        self.required_samples = math.ceil(1.0 / control_dt)
        self.samples = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.samples.zero_()
        else:
            self.samples[env_ids] = 0

    def update(self, travel, axial_force, valid_and_retained, *, size_scale=1.0):
        pressed = (
            valid_and_retained
            & torch.isfinite(travel)
            & torch.isfinite(axial_force)
            & (travel >= 0.015 * size_scale)
            & (axial_force > 0.1)
        )
        self.samples.copy_(
            torch.where(pressed, (self.samples + 1).clamp_max(self.required_samples), 0)
        )
        return self.samples >= self.required_samples
