"""Wang et al. [12] — DQN energy-saving xApp (radio-card ON/OFF).

Reference: Q. Wang, S. Chetty, A. Al-Tahmeesschi, X. Liang, Y. Chu, and H. Ahmadi,
"Energy saving in 6G O-RAN using DQN-based xApp," IEEE CAMAD 2024.

Standalone baseline. Keeps the SAME state (global 5N+4) and action space
(enumerated ON/OFF masks) as rl/agents/dqn.py, and reuses that module's network
and replay buffer — only the training hyperparameters are replaced with the
paper's reported values:

    optimizer            AdamW
    learning rate        1e-4
    discount gamma       0.99
    mini-batch size      64
    replay memory        300000
    target update        every 50 steps
    epsilon              0.9 -> 0.05 (decayed)

The paper trains 30000 episodes x 100 steps (3M steps); our campaigns are far
shorter (900 steps x ~100 episodes ~ 90k steps), so `epsilon_decay` (the decay
time-constant, not in the excerpt) defaults to 20000 steps to span a campaign.
Adjust if you train much longer/shorter. torch at module load -> imported lazily
by the registry. See docs/BASELINES_OREO_TABLE1.md.
"""

from __future__ import annotations

import torch.optim as optim

from rl.agents.base import state_dim
from rl.agents.dqn import DQNAdapter, DQNAgent, ReplayBuffer


class WangDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_list, epsilon_decay=20000):
        # Base builds QNetwork/target/Adam; we pass Wang's lr/gamma/epsilon range.
        super().__init__(state_dim, action_list, lr=1e-4, gamma=0.99,
                         epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=epsilon_decay)
        # Override the remaining base defaults with the paper's values.
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(capacity=300000)
        self.batch_size = 64
        self.update_target_every = 50


class WangDQNAdapter(DQNAdapter):
    """Same act/observe loop as DQNAdapter, but drives a WangDQNAgent."""

    def __init__(self, env, seed=None):
        super().__init__(env, seed=seed)
        self.agent = WangDQNAgent(state_dim(env.num_uavs), list(range(len(self.masks))))
