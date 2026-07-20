"""Multi-agent DQN baselines: Rezaei [19] and Masrur [16].

References:
- Z. Rezaei & B. S. Ghahfarokhi, "Energy and spectrum efficient cell switch-off
  ... A deep reinforcement learning approach," Computer Networks 2023.
  -> multi-agent DQN for cell switch-off.  (`rezaei`, double=False)
- S. Masrur et al., "Energy-efficient sleep mode optimization in 5G mmWave
  networks via multi-agent deep reinforcement learning," IEEE TGCN 2026.
  -> MARL Double-DQN for sleep mode.       (`masrur`, double=True)

Both are one independent DQN per UAV over the binary {OFF, ON} action, observing
the shared global state (5N+4, see rl/agents/base.global_state_vector) and trained
on the shared team reward (reward_info["reward_drl"]). The minimum-active
constraint (center-optional, same as CLARA / env._enforce_min_active) is applied
after the per-agent decisions.

torch is imported at module load, so the registry imports this lazily (only when
`rezaei` or `masrur` is selected). Requires torch — not smoke-tested on machines
without it, same status as rl/agents/dqn.py and ppo.py. See
docs/BASELINES_OREO_TABLE1.md.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.agents.base import global_state_vector, mask_to_action, state_dim


class _QNet(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),  # Q(OFF), Q(ON)
        )

    def forward(self, x):
        return self.net(x)


class _PerAgentDQN:
    def __init__(self, in_dim, double, lr, gamma, device):
        self.double = double
        self.gamma = gamma
        self.device = device
        self.q = _QNet(in_dim).to(device)
        self.target = _QNet(in_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = deque(maxlen=100000)
        self.batch_size = 256
        self.update_target_every = 200
        self._updates = 0

    def act(self, state, epsilon, rng):
        if rng.random() < epsilon:
            return rng.randint(0, 1)
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q(st).argmax(dim=1).item())

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        q_val = self.q(s).gather(1, a)
        with torch.no_grad():
            if self.double:  # Double DQN target (Masrur [16])
                next_a = self.q(ns).argmax(dim=1, keepdim=True)
                next_q = self.target(ns).gather(1, next_a)
            else:            # vanilla DQN target (Rezaei [19])
                next_q = self.target(ns).max(dim=1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)
        loss = nn.MSELoss()(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._updates += 1
        if self._updates % self.update_target_every == 0:
            self.target.load_state_dict(self.q.state_dict())


class MARLDQNAgent:
    """Independent per-UAV DQN with a shared team reward; center kept ON."""

    def __init__(self, env, double=False, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=1000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.env = env
        self.uav_ids = env.uav_ids
        self.center_index = env.center_index
        self.min_active = max(1, env.min_active_uavs)
        self.n = env.num_uavs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_dim = state_dim(self.n)
        self.agents = [_PerAgentDQN(in_dim, double, lr, gamma, device) for _ in range(self.n)]
        self.eps_start, self.eps_end, self.eps_decay = epsilon_start, epsilon_end, epsilon_decay
        self.steps = 0
        self.rng = random.Random(seed)
        self._last_state = None
        self._last_actions = None

    def _epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps / self.eps_decay)

    def _enforce(self, bits):
        """Center-optional: mirror env._enforce_min_active — only fill up to
        min_active (center first) when below it, matching CLARA's action set."""
        bits = list(bits)
        if sum(bits) >= self.min_active:
            return bits
        order = [self.center_index] + [i for i in range(self.n) if i != self.center_index]
        for i in order:
            if sum(bits) >= self.min_active:
                break
            bits[i] = 1
        return bits

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        state = np.asarray(global_state_vector(obs, self.env), dtype=np.float32)
        eps = self._epsilon()
        self.steps += 1
        bits = [self.agents[i].act(state, eps, self.rng) for i in range(self.n)]
        bits = self._enforce(bits)
        self._last_state, self._last_actions = state, bits
        return mask_to_action(tuple(bool(b) for b in bits), self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        r = reward_info["reward_drl"]  # shared team reward
        next_state = np.asarray(global_state_vector(next_obs, self.env), dtype=np.float32)
        for i in range(self.n):
            self.agents[i].push(self._last_state, self._last_actions[i], r, next_state, done)
            self.agents[i].update()

    def end_episode(self):
        pass
