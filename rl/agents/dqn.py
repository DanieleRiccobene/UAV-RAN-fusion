"""DQN (Double DQN) — vendored from the user's `discrete_dqn.py` + Agent adapter.

torch is imported at module load, so this module is imported lazily (only when the
`dqn` agent is selected) — see rl/agents/registry.py.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_float32_matmul_precision("high")

from rl.agents.base import enumerate_masks, global_state_vector, mask_to_action, state_dim


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_index, reward, next_state, done):
        self.buffer.append((state, action_index, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_list, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.action_list = action_list
        self.n_actions = len(action_list)
        self.q_net = QNetwork(state_dim, self.n_actions)
        self.target_net = QNetwork(state_dim, self.n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 512
        self.gamma = gamma
        self.update_target_every = 100
        self.steps = 0
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = epsilon_start, epsilon_end, epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.steps / self.epsilon_decay)
        self.steps += 1
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(state_tensor).argmax(dim=1).item())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        q_val = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
            q_target_val = self.target_net(ns).gather(1, next_actions)
            target = r + self.gamma * q_target_val * (1 - d)
        loss = nn.MSELoss()(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class DQNAdapter:
    def __init__(self, env, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.env = env
        self.uav_ids = env.uav_ids
        self.masks = enumerate_masks(env.num_uavs, env.min_active_uavs)
        self.agent = DQNAgent(state_dim(env.num_uavs), list(range(len(self.masks))))
        self._last_state = None
        self._last_idx = None

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        state = np.asarray(global_state_vector(obs, self.env), dtype=np.float32)
        idx = self.agent.select_action(state)
        self._last_state, self._last_idx = state, idx
        return mask_to_action(self.masks[idx], self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        next_state = np.asarray(global_state_vector(next_obs, self.env), dtype=np.float32)
        self.agent.replay_buffer.push(
            self._last_state, self._last_idx, reward_info["reward_drl"], next_state, done
        )
        self.agent.update()

    def end_episode(self):
        pass
