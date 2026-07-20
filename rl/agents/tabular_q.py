"""El Amine et al. [20] — multi-sleeping control via tabular Q-learning.

Reference: A. E. Amine et al., "Energy optimization with multi-sleeping control
in 5G heterogeneous networks using reinforcement learning," IEEE TNSM 2022.

Standalone baseline (separate from CLARA/UCB). In our binary ON/OFF UAV env the
multi-sleep levels collapse to two states (active / sleep), so this is a
single-agent, state-aware tabular Q-learning controller over the joint ON/OFF
mask: epsilon-greedy action selection, coarse discretized network state, and
temporal-difference bootstrapping on the DRL reward (which includes switching
cost). Torch-free. Contrast with `ucb` (contextless bandit, no bootstrapping).
See docs/BASELINES_OREO_TABLE1.md.
"""

from __future__ import annotations

import math
import random

from rl.agents.base import enumerate_masks, mask_to_action


class ElAmineQLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.9, epsilon_start=1.0,
                 epsilon_end=0.01, decay_steps=2000, seed=None):
        self.env = env
        self.uav_ids = env.uav_ids
        self.masks = enumerate_masks(env.num_uavs, env.min_active_uavs)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.eps_start = float(epsilon_start)
        self.eps_end = float(epsilon_end)
        self.decay = float(decay_steps)
        self.t = 0
        self.q = {}  # state_key -> list[float] over masks
        self.rng = random.Random(seed)
        self._last_state = None
        self._last_idx = None

    def _state_key(self, obs):
        """Coarse discretized state: per-UAV activation + a UE-load bin."""
        active = tuple(1 if obs["uavs"][u]["is_active"] else 0 for u in self.uav_ids)
        k = max(1, self.env.max_ues_per_uav)
        ue_bin = min(self.env.num_uavs + 1, obs["ue_count"] // k)
        return (active, ue_bin)

    def _q_row(self, state_key):
        row = self.q.get(state_key)
        if row is None:
            row = [0.0] * len(self.masks)
            self.q[state_key] = row
        return row

    def _epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.t / self.decay)

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        state_key = self._state_key(obs)
        row = self._q_row(state_key)
        self.t += 1
        if self.rng.random() < self._epsilon():
            idx = self.rng.randrange(len(self.masks))
        else:
            best = max(row)
            idx = self.rng.choice([i for i, v in enumerate(row) if v == best])
        self._last_state, self._last_idx = state_key, idx
        return mask_to_action(self.masks[idx], self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        r = reward_info["reward_drl"]
        row = self._q_row(self._last_state)
        target = r if done else r + self.gamma * max(self._q_row(self._state_key(next_obs)))
        row[self._last_idx] += self.lr * (target - row[self._last_idx])

    def end_episode(self):
        pass
