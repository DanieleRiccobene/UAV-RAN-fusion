"""CLARA — game-theoretic contextual MAB (multi-agent), the flagship method.

Vendored from the user's `GTcmab.py` (`MultiAgentGameMAB`) with one change: the
all-off guard forces the **center** UAV on (topology invariant) instead of index 0.
Wrapped in an Agent adapter (see rl/agents/base.py).
"""

from __future__ import annotations

import random

import numpy as np

from rl.agents.base import mask_to_action


class MultiAgentGameMAB:
    def __init__(self, num_agents=7, epsilon_start=1.0, epsilon_end=0.01,
                 decay_steps=1000, center_index=0, seed=None):
        self.num_agents = num_agents
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.center_index = center_index
        self.t = 0
        self.q_tables = [{} for _ in range(num_agents)]
        self.action_counts = [{} for _ in range(num_agents)]
        self.last_context_vectors = [None] * num_agents
        self.last_actions = [0] * num_agents
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def current_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.t / self.decay_steps)

    def get_context_vector(self, agent_idx, current_actions):
        return [current_actions[i] for i in range(self.num_agents) if i != agent_idx]

    def get_q_values(self, agent_idx, context_vector):
        state_key = tuple(context_vector)
        if state_key not in self.q_tables[agent_idx]:
            self.q_tables[agent_idx][state_key] = np.zeros(2)
            self.action_counts[agent_idx][state_key] = np.zeros(2)
        return self.q_tables[agent_idx][state_key]

    def negotiate(self, previous_actions, cost_vector, max_steps=10):
        self.t += 1
        eps = self.current_epsilon()
        if isinstance(previous_actions, np.ndarray):
            current_actions = previous_actions.copy()
        else:
            current_actions = list(previous_actions)

        for _ in range(max_steps):
            changes = 0
            for agent_idx in np.random.permutation(self.num_agents):
                ctx_vector = self.get_context_vector(agent_idx, current_actions)
                if random.random() < eps:
                    selected_action = random.choice([0, 1])
                else:
                    q_values = np.asarray(self.get_q_values(agent_idx, ctx_vector), dtype=np.float64)
                    raw_cost = cost_vector[agent_idx] if agent_idx < len(cost_vector) else 0.0
                    base_cost = float(raw_cost) if np.isfinite(raw_cost) else 0.0
                    action_costs = np.full(2, base_cost, dtype=np.float64)
                    prev_action = int(previous_actions[agent_idx])
                    if prev_action in (0, 1):
                        action_costs[prev_action] = 0.0
                    net_utilities = q_values - action_costs
                    finite_idx = np.where(np.isfinite(net_utilities))[0]
                    if finite_idx.size == 0:
                        selected_action = prev_action if prev_action in (0, 1) else 0
                    else:
                        finite_vals = net_utilities[finite_idx]
                        candidates = finite_idx[np.isclose(finite_vals, np.max(finite_vals))]
                        selected_action = int(np.random.choice(candidates))
                if int(current_actions[agent_idx]) != selected_action:
                    current_actions[agent_idx] = selected_action
                    changes += 1
            if changes == 0:
                break

        if np.sum(current_actions) == 0:
            current_actions[self.center_index] = 1  # force CENTER on (topology invariant)

        for i in range(self.num_agents):
            ctx_vector = self.get_context_vector(i, current_actions)
            self.last_context_vectors[i] = tuple(ctx_vector)
            self.last_actions[i] = int(current_actions[i])
        return current_actions, (changes == 0)

    def update(self, rewards_dict, cell_list):
        for i, cell_id in enumerate(cell_list):
            ctx_key = self.last_context_vectors[i]
            act = self.last_actions[i]
            if ctx_key is None:
                continue
            if ctx_key not in self.action_counts[i]:
                self.action_counts[i][ctx_key] = np.zeros(2)
                self.q_tables[i][ctx_key] = np.zeros(2)
            self.action_counts[i][ctx_key][act] += 1
            alpha = 1.0 / self.action_counts[i][ctx_key][act]
            agent_reward = float(rewards_dict[cell_id][0] + rewards_dict[cell_id][1])
            old_val = self.q_tables[i][ctx_key][act]
            self.q_tables[i][ctx_key][act] = old_val + alpha * (agent_reward - old_val)


class ClaraAgent:
    """Agent adapter around MultiAgentGameMAB."""

    def __init__(self, env, decay_steps=1000, seed=None):
        self.env = env
        self.uav_ids = env.uav_ids
        self.mab = MultiAgentGameMAB(
            num_agents=env.num_uavs,
            decay_steps=decay_steps,
            center_index=env.center_index,
            seed=seed,
        )
        self.previous_actions = [1] * env.num_uavs

    def begin_episode(self, obs):
        self.previous_actions = [1] * self.env.num_uavs

    def act(self, obs):
        cost_vector = self.env.cost_vector()
        actions, _ = self.mab.negotiate(self.previous_actions, cost_vector)
        self.previous_actions = list(int(a) for a in actions)
        mask = tuple(bool(a) for a in self.previous_actions)
        return mask_to_action(mask, self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        rewards_dict = {
            uav_id: [reward_info["per_uav"][uav_id]["global_k"],
                     reward_info["per_uav"][uav_id]["individual_k"]]
            for uav_id in self.uav_ids
        }
        self.mab.update(rewards_dict, self.uav_ids)

    def end_episode(self):
        pass
