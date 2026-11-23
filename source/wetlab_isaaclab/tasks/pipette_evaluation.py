"""Score one attempt per supplied rack layout using the environment's events."""

from collections.abc import Callable, Sequence

import torch


EVENTS = ("goal_reached", "aligned", "sustained_press", "success")


def score_rack_layouts(env, layouts: Sequence[dict], *,
                       controller_factory: Callable,
                       seed: int = 0):
    """Return first-attempt outcomes, one per layout.

    `controller_factory()` creates a fresh batched controller, which receives
    `(observation, active_mask)` and returns the position targets.
    """
    if len(layouts) != env.num_envs:
        raise ValueError("Expected one layout per world")

    obs, _ = env.reset(seed=seed, options={"layouts": layouts})
    controller = controller_factory()
    scene = env.unwrapped
    fields = scene.observation_slices
    seated_z = obs["policy"][:, fields["pipette_pose"]][:, 2].clone()
    active = torch.ones_like(seated_z, dtype=torch.bool)
    counts = torch.zeros_like(seated_z, dtype=torch.int64)
    events_seen = {event: torch.zeros_like(active) for event in EVENTS}
    lift = torch.zeros_like(active)
    valid = active.clone()
    fixture_stable = active.clone()
    timeout = torch.zeros_like(active)
    for _ in range(scene.max_episode_length):
        action = controller(obs, active.clone())
        obs, _, terminated, truncated, extras = env.step(action)
        events = extras["events"]
        done = terminated | truncated
        # Isaac Lab same-step autoreset returns new states for finished worlds.
        state = (torch.where(done[:, None], extras["final_obs"]["policy"], obs["policy"])
                 if "final_obs" in extras else obs["policy"])
        height = state[:, fields["pipette_pose"]][:, 2]
        lift |= active & ((height - seated_z) >= 0.10)
        for name in EVENTS:
            events_seen[name] |= active & events[name]
        valid &= ~active | events["valid"]
        fixture_stable &= ~active | events["fixture_stable"]
        timeout |= active & truncated
        counts += active
        active &= ~done
        if not active.any():
            break

    records = zip(
        counts.tolist(), lift.tolist(), valid.tolist(), fixture_stable.tolist(),
        timeout.tolist(), *(events_seen[name].tolist() for name in EVENTS), strict=True,
    )
    fields = ("control_steps", "lift", "valid", "fixture_stable", "timeout", *EVENTS)
    return [dict(layout=layout.get("id", index), **dict(zip(fields, values, strict=True)))
            for index, (layout, values) in enumerate(zip(layouts, records, strict=True))]
