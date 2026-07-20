import json
import math
import random

from rl.fidelity_provider import FidelityLevel


ARM_TO_FIDELITY = {
    0: FidelityLevel.HIGH,
    1: FidelityLevel.MEDIUM,
    2: FidelityLevel.LOW,
}


class FidelityMabController:
    def __init__(self, *, algorithm="epsilon_greedy", epsilon=0.1, seed=7, min_initial_pulls_per_arm=5):
        normalized_algorithm = str(algorithm or "epsilon_greedy").strip().lower()
        if normalized_algorithm not in {"epsilon_greedy", "ucb1"}:
            raise ValueError(
                f"Unsupported MAB algorithm '{algorithm}'. Expected one of: epsilon_greedy, ucb1."
            )
        self.algorithm = normalized_algorithm
        self.epsilon = float(epsilon)
        if self.epsilon < 0.0:
            raise ValueError("MAB epsilon must be non-negative.")
        self.min_initial_pulls_per_arm = max(1, int(min_initial_pulls_per_arm))
        self.rng = random.Random(seed)
        self.arm_counts = {arm: 0 for arm in ARM_TO_FIDELITY}
        self.arm_total_rewards = {arm: 0.0 for arm in ARM_TO_FIDELITY}
        self.total_pulls = 0

    def select_arm(self):
        under_sampled_arms = [
            arm for arm, count in self.arm_counts.items()
            if count < self.min_initial_pulls_per_arm
        ]
        if under_sampled_arms:
            min_count = min(self.arm_counts[arm] for arm in under_sampled_arms)
            candidate_arms = [arm for arm in under_sampled_arms if self.arm_counts[arm] == min_count]
            return self.rng.choice(candidate_arms)
        if self.algorithm == "ucb1":
            return self._select_arm_ucb1()
        return self._select_arm_epsilon_greedy()

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        if arm not in self.arm_counts:
            raise ValueError(f"Unknown MAB arm '{arm}'.")
        self.arm_counts[arm] += 1
        self.arm_total_rewards[arm] += reward
        self.total_pulls += 1

    def arm_to_fidelity(self, arm):
        try:
            return ARM_TO_FIDELITY[int(arm)]
        except KeyError as exc:
            raise ValueError(f"Unknown MAB arm '{arm}'.") from exc

    def mean_reward(self, arm):
        count = self.arm_counts[int(arm)]
        if count <= 0:
            return 0.0
        return self.arm_total_rewards[int(arm)] / count

    def get_statistics(self):
        arms = {}
        for arm in sorted(ARM_TO_FIDELITY):
            arms[str(arm)] = {
                "fidelity": self.arm_to_fidelity(arm).value,
                "count": self.arm_counts[arm],
                "total_reward": self.arm_total_rewards[arm],
                "mean_reward": self.mean_reward(arm),
            }
        return {
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "min_initial_pulls_per_arm": self.min_initial_pulls_per_arm,
            "total_pulls": self.total_pulls,
            "arms": arms,
        }

    def statistics_json(self):
        return json.dumps(self.get_statistics(), sort_keys=True)

    def _select_arm_epsilon_greedy(self):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(sorted(ARM_TO_FIDELITY))
        mean_rewards = {arm: self.mean_reward(arm) for arm in ARM_TO_FIDELITY}
        best_reward = max(mean_rewards.values())
        best_arms = [arm for arm, value in mean_rewards.items() if value == best_reward]
        return self.rng.choice(best_arms)

    def _select_arm_ucb1(self):
        log_total = math.log(max(1, self.total_pulls))
        scored_arms = []
        for arm in sorted(ARM_TO_FIDELITY):
            mean_reward = self.mean_reward(arm)
            bonus = math.sqrt((2.0 * log_total) / self.arm_counts[arm])
            scored_arms.append((mean_reward + bonus, arm))
        best_score = max(score for score, _arm in scored_arms)
        best_arms = [arm for score, arm in scored_arms if score == best_score]
        return self.rng.choice(best_arms)
