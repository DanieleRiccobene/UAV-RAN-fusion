"""Agent interface + shared helpers (action masks, DRL state vector).

All agents implement the same protocol so the episodic loop is agent-agnostic:

    begin_episode(obs) -> None
    act(obs)           -> dict {UAV_k: "ON"|"OFF"}
    observe(obs, action, reward_info, next_obs, done) -> None
    end_episode()      -> None

`reward_info` is the env's decomposed reward (see rl/uav_env.compute_reward_info).
Single-agent agents read `reward_info["reward_drl"]` (DRL) or `["total"]` (MAB
base); CLARA reads `reward_info["per_uav"]`.
"""

from __future__ import annotations

import itertools


def enumerate_masks(num_uavs, min_active=1):
    """All ON/OFF masks with >= min_active UAVs active (excludes all-off)."""
    masks = []
    for bits in itertools.product([0, 1], repeat=num_uavs):
        if sum(bits) >= max(1, min_active):
            masks.append(tuple(bool(b) for b in bits))
    return masks


def mask_to_action(mask, uav_ids):
    return {uav_ids[i]: ("ON" if mask[i] else "OFF") for i in range(len(uav_ids))}


def action_to_mask(action, uav_ids):
    return tuple(
        (action.get(uav_ids[i], "ON").upper() == "ON") if isinstance(action.get(uav_ids[i]), str)
        else bool(action.get(uav_ids[i], True))
        for i in range(len(uav_ids))
    )


def global_state_vector(obs, env):
    """DRL global feature vector, length 5N + 4 (see docs/STATE_SPACE.md).

    Per UAV: [is_active, capacity_utilization, free_slot_fraction,
              throughput_norm, off_duration_norm].
    Global tail: [episode_progress, outage_fraction, active_fraction, agg_tput_norm].
    """
    K = float(env.max_ues_per_uav)
    per_uav_ref = env.per_ue_throughput_mbps * K
    ue_count = max(1, obs["ue_count"])
    agg_ref = env.per_ue_throughput_mbps * ue_count
    off_norm_den = 50.0  # normalization horizon for off-duration (steps)

    vec = []
    for uav_id in env.uav_ids:
        u = obs["uavs"][uav_id]
        served = u["served"]
        vec.extend([
            1.0 if u["is_active"] else 0.0,
            served / K,
            max(0.0, (K - served) / K),
            u["throughput_mbps"] / per_uav_ref if per_uav_ref else 0.0,
            min(1.0, u["off_duration"] / off_norm_den),
        ])
    vec.extend([
        obs["episode_progress"],
        obs["disconnected_ue_count"] / ue_count,
        obs["active_uav_count"] / float(env.num_uavs),
        obs["aggregate_throughput_mbps"] / agg_ref if agg_ref else 0.0,
    ])
    return vec


def state_dim(num_uavs):
    return 5 * num_uavs + 4
