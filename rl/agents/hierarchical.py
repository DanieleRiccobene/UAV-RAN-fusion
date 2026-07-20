"""Xu et al. [18] — hierarchical RL for energy-efficient BS activation.

Reference: D. Xu et al., "Dynamic hierarchical reinforcement learning framework
for energy-efficient 5G base stations in urban environments," IEEE TMC 2025.

Standalone baseline — a lightweight, faithful-in-spirit adaptation of the
hierarchical structure (not a re-implementation of the deep GAMA/QMIX variants):

- HIGH level: a tabular Q-policy chooses an *activation budget* k (how many UAVs
  to keep ON) from a coarse network state, learning with TD bootstrapping on the
  DRL reward.
- LOW level: a fixed load-following rule realizes the budget by activating the
  k UAVs with the highest recent served load (center-optional, matching CLARA's
  action set; uses the previous step's observation only — no peeking at the
  current frame).

Torch-free. See docs/BASELINES_OREO_TABLE1.md.
"""

from __future__ import annotations

import math
import random

from rl.agents.base import mask_to_action


class XuHierarchicalAgent:
    def __init__(self, env, lr=0.1, gamma=0.9, epsilon_start=1.0,
                 epsilon_end=0.01, decay_steps=2000, seed=None):
        self.env = env
        self.uav_ids = env.uav_ids
        self.center_index = env.center_index
        self.n = env.num_uavs
        self.min_active = max(1, env.min_active_uavs)
        # High-level actions: activation budget k in [min_active, N].
        self.budgets = list(range(self.min_active, self.n + 1))
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.eps_start = float(epsilon_start)
        self.eps_end = float(epsilon_end)
        self.decay = float(decay_steps)
        self.t = 0
        self.q = {}  # state_key -> list[float] over budgets
        self.rng = random.Random(seed)
        self._last_state = None
        self._last_bidx = None

    def _state_key(self, obs):
        k = max(1, self.env.max_ues_per_uav)
        ue_bin = min(self.n + 1, obs["ue_count"] // k)
        return (ue_bin,)

    def _q_row(self, state_key):
        row = self.q.get(state_key)
        if row is None:
            row = [0.0] * len(self.budgets)
            self.q[state_key] = row
        return row

    def _epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.t / self.decay)

    def _low_level_mask(self, obs, k):
        """The k UAVs with the highest recent served load (center-optional)."""
        order = sorted(
            range(self.n),
            key=lambda i: (-obs["uavs"][self.uav_ids[i]]["served"], i),
        )
        active = [False] * self.n
        for i in order[:k]:
            active[i] = True
        return tuple(active)

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        state_key = self._state_key(obs)
        row = self._q_row(state_key)
        self.t += 1
        if self.rng.random() < self._epsilon():
            bidx = self.rng.randrange(len(self.budgets))
        else:
            best = max(row)
            bidx = self.rng.choice([i for i, v in enumerate(row) if v == best])
        self._last_state, self._last_bidx = state_key, bidx
        mask = self._low_level_mask(obs, self.budgets[bidx])
        return mask_to_action(mask, self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        r = reward_info["reward_drl"]
        row = self._q_row(self._last_state)
        target = r if done else r + self.gamma * max(self._q_row(self._state_key(next_obs)))
        row[self._last_bidx] += self.lr * (target - row[self._last_bidx])

    def end_episode(self):
        pass
