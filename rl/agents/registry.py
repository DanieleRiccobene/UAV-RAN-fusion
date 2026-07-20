"""Agent registry: name -> Agent adapter. torch agents are imported lazily."""

from __future__ import annotations

AGENT_NAMES = ("clara", "ucb", "random", "oracle", "dqn", "ppo",
               "el_amine", "xu", "rezaei", "masrur", "wang")


def build_agent(name, env, seed=None):
    name = name.lower()
    if name == "clara":
        from rl.agents.clara import ClaraAgent
        return ClaraAgent(env, seed=seed)
    if name == "ucb":
        from rl.agents.bandits import UCBStatelessAgent
        return UCBStatelessAgent(env, seed=seed)
    if name == "random":
        from rl.agents.bandits import RandomAgent
        return RandomAgent(env, seed=seed)
    if name == "oracle":
        from rl.agents.oracle import MinUavOracleAgent
        return MinUavOracleAgent(env, seed=seed)
    if name == "dqn":
        from rl.agents.dqn import DQNAdapter  # lazy: needs torch
        return DQNAdapter(env, seed=seed)
    if name == "ppo":
        from rl.agents.ppo import PPOAdapter  # lazy: needs torch
        return PPOAdapter(env, seed=seed)
    # --- OREO Table 1 baselines (RU-activation-only; no user association) ---
    if name == "el_amine":  # [20] tabular Q-learning multi-sleeping control
        from rl.agents.tabular_q import ElAmineQLearningAgent
        return ElAmineQLearningAgent(env, seed=seed)
    if name == "xu":  # [18] hierarchical RL (budget + load-following)
        from rl.agents.hierarchical import XuHierarchicalAgent
        return XuHierarchicalAgent(env, seed=seed)
    if name == "rezaei":  # [19] multi-agent DQN cell switch-off
        from rl.agents.marl_dqn import MARLDQNAgent  # lazy: needs torch
        return MARLDQNAgent(env, double=False, seed=seed)
    if name == "masrur":  # [16] MARL Double-DQN sleep mode
        from rl.agents.marl_dqn import MARLDQNAgent  # lazy: needs torch
        return MARLDQNAgent(env, double=True, seed=seed)
    if name == "wang":  # [12] DQN xApp with the paper's training hyperparameters
        from rl.agents.wang_dqn import WangDQNAdapter  # lazy: needs torch
        return WangDQNAdapter(env, seed=seed)
    raise ValueError(f"Unknown agent {name!r}; available: {AGENT_NAMES}")
