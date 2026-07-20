"""Single-agent MAB baselines: stateless UCB1 over ON/OFF masks, and Random."""

from __future__ import annotations

import math
import random

from rl.agents.base import enumerate_masks, mask_to_action


class UCBStatelessAgent:
    """UCB1 bandit over enumerated valid ON/OFF masks (MAB baseline)."""

    def __init__(self, env, exploration_coef=2.0, seed=None):
        self.env = env
        self.uav_ids = env.uav_ids
        self.masks = enumerate_masks(env.num_uavs, env.min_active_uavs)
        self.counts = [0] * len(self.masks)
        self.totals = [0.0] * len(self.masks)
        self.total_pulls = 0
        self.c = float(exploration_coef)
        self.rng = random.Random(seed)
        self._last_arm = 0

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        untried = [i for i, n in enumerate(self.counts) if n == 0]
        if untried:
            arm = self.rng.choice(untried)
        else:
            log_total = math.log(max(1, self.total_pulls))
            scores = [
                self.totals[i] / self.counts[i] + self.c * math.sqrt(log_total / self.counts[i])
                for i in range(len(self.masks))
            ]
            best = max(scores)
            arm = self.rng.choice([i for i, s in enumerate(scores) if s == best])
        self._last_arm = arm
        return mask_to_action(self.masks[arm], self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        r = reward_info["total"]  # MAB base reward (no switch cost)
        self.counts[self._last_arm] += 1
        self.totals[self._last_arm] += r
        self.total_pulls += 1

    def end_episode(self):
        pass


class RandomAgent:
    def __init__(self, env, seed=None):
        self.env = env
        self.uav_ids = env.uav_ids
        self.masks = enumerate_masks(env.num_uavs, env.min_active_uavs)
        self.rng = random.Random(seed)

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        return mask_to_action(self.rng.choice(self.masks), self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        pass

    def end_episode(self):
        pass
