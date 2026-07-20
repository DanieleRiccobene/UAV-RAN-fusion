"""Min-UAV oracle baseline ([5]) — a standalone, non-learning reference agent.

Offline optimality reference (genie): at each step it peeks at the current
frame's UE positions and picks the *smallest active set that serves everyone*
(zero outage), falling back to max-coverage when the frame is oversubscribed.
It is the lower envelope of the active-RU curves at iso-outage.

This is a self-contained baseline; it does NOT touch or subclass CLARA. It
implements the same Agent protocol (begin_episode/act/observe/end_episode) so
the episodic loop treats it like any other agent.

Design: switching-cost-blind (canonical [5]); it re-optimizes coverage every
frame, so it can churn. The tertiary churn tie-break only trims *free* toggles,
keeping it a true min-UAV oracle. See docs/BASELINE_MIN_UAV_ORACLE.md.
"""

from __future__ import annotations

from rl.agents.base import enumerate_masks, mask_to_action


class MinUavOracleAgent:
    """Per-frame min-active-UAV coverage optimizer (brute force over 2^N masks)."""

    def __init__(self, env, seed=None):
        self.env = env
        self.uav_ids = env.uav_ids
        # Candidate masks: any with >= min_active active (center-optional, exactly
        # CLARA's reachable action set — fair comparison).
        self.masks = enumerate_masks(env.num_uavs, env.min_active_uavs)
        # frame_index -> chosen mask. The oracle is deterministic per frame, so
        # episode 1 fills this and episodes 2..K are cache hits.
        self._cache = {}

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        idx = self.env.current_index
        mask = self._cache.get(idx)
        if mask is None:
            frame = self.env._ue_by_index[idx]
            mask = self._solve(frame)
            self._cache[idx] = mask
        return mask_to_action(mask, self.uav_ids)

    def _solve(self, frame):
        """Lexicographic min: (outage, active_count, -throughput, churn)."""
        current = self.env.active
        best_key = None
        best_mask = None
        n_ue = len(frame)
        for mask in self.masks:
            assignment, counts = self.env._associate(mask, frame)
            outage = n_ue - sum(counts.values())
            tput, _ = self.env._throughput(assignment)
            churn = sum(1 for i in range(self.env.num_uavs) if mask[i] != current[i])
            key = (outage, sum(mask), -tput, churn)
            if best_key is None or key < best_key:
                best_key, best_mask = key, mask
        return best_mask

    def observe(self, obs, action, reward_info, next_obs, done):
        pass

    def end_episode(self):
        pass
