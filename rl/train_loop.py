"""Episodic training loop + per-episode metrics (see docs/EVALUATION.md)."""

from __future__ import annotations


def run_episode(env, agent, step_sink=None):
    """Run one episode. If `step_sink` is a list, append per-step INSTANTANEOUS
    values (not averaged) to it — used to build per-step CDFs."""
    obs = env.reset()
    agent.begin_episode(obs)
    acc = {
        "return_base": 0.0, "return_cmp": 0.0, "switch_total": 0.0,
        "tput": 0.0, "active": 0.0, "served": 0.0, "outage": 0.0,
        "ue": 0.0, "util": 0.0,
    }
    steps = 0
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward_info, done, info = env.step(action)
        agent.observe(obs, action, reward_info, next_obs, done)
        m = reward_info["metrics"]
        acc["return_base"] += reward_info["total"]
        acc["return_cmp"] += reward_info["reward_cmp"]
        acc["switch_total"] += reward_info["switch_total"]
        acc["tput"] += m["aggregate_throughput_mbps"]
        acc["active"] += m["active_uav_count"]
        acc["served"] += m["served_ue_count"]
        acc["outage"] += m["disconnected_ue_count"]
        acc["ue"] += max(1, m["ue_count"])
        active = max(1, m["active_uav_count"])
        acc["util"] += m["served_ue_count"] / (active * env.max_ues_per_uav)
        if step_sink is not None:
            step_sink.append({
                "step": steps,
                "active_uavs": m["active_uav_count"],
                "throughput_mbps": m["aggregate_throughput_mbps"],
                "rlf": m["disconnected_ue_count"],
                "switch_cost": reward_info["switch_total"],
                "reward": reward_info["total"],
                "reward_cmp": reward_info["reward_cmp"],
            })
        obs = next_obs
        steps += 1
    agent.end_episode()
    steps = max(1, steps)
    return {
        "return_base": acc["return_base"],
        "return_cmp": acc["return_cmp"],
        "switch_cost_total": acc["switch_total"],
        "mean_throughput_mbps": acc["tput"] / steps,
        "mean_active_uavs": acc["active"] / steps,
        "mean_served_ues": acc["served"] / steps,
        "mean_rlf": acc["outage"] / steps,  # radio link failures = mean disconnected UEs/step
        "outage_rate": acc["outage"] / acc["ue"],
        "mean_capacity_utilization": acc["util"] / steps,
    }


def run_training(env, agent, num_episodes, on_episode=None, collect_steps=False):
    """Returns (history, step_rows). `step_rows` is a flat list of per-step dicts
    (each tagged with its episode) pooled across all episodes, empty unless
    `collect_steps`."""
    history = []
    step_rows = []
    for episode in range(1, num_episodes + 1):
        ep_steps = [] if collect_steps else None
        metrics = run_episode(env, agent, step_sink=ep_steps)
        metrics["episode"] = episode
        history.append(metrics)
        if collect_steps:
            for s in ep_steps:
                s["episode"] = episode
                step_rows.append(s)
        if on_episode is not None:
            on_episode(episode, metrics)
    return history, step_rows
