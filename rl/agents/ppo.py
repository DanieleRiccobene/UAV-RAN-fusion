"""PPO (PPO-Clip) — vendored from the user's `ppo_discrete.py` + Agent adapter.

PPO updates **once per episode** (per the user's `energy_saving_drl.py` loop).
torch imported at load; module imported lazily (see rl/agents/registry.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

torch.set_float32_matmul_precision("high")

from rl.agents.base import enumerate_masks, global_state_vector, mask_to_action, state_dim


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    logprob: float
    value: float
    next_value: float


class RolloutBuffer:
    def __init__(self, gamma, gae_lambda):
        self.gamma, self.lmbda = gamma, gae_lambda
        self.data: List[Transition] = []

    def add(self, *args, **kwargs):
        self.data.append(Transition(*args, **kwargs))

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def compute(self):
        states = np.array([t.state for t in self.data], dtype=np.float32)
        actions = np.array([t.action for t in self.data], dtype=np.int64)
        rewards = np.array([t.reward for t in self.data], dtype=np.float32)
        dones = np.array([t.done for t in self.data], dtype=np.float32)
        values = np.array([t.value for t in self.data], dtype=np.float32)
        next_values = np.array([t.next_value for t in self.data], dtype=np.float32)
        old_logprobs = np.array([t.logprob for t in self.data], dtype=np.float32)
        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lmbda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return (torch.from_numpy(states), torch.from_numpy(actions),
                torch.from_numpy(old_logprobs), torch.from_numpy(returns),
                torch.from_numpy(advantages))


class PPOAgent:
    def __init__(self, state_dim, n_actions, learning_rate=1e-5, batch_size=64,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=10,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(state_dim, n_actions).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
        self.batch_size, self.gamma, self.lmbda = batch_size, gamma, gae_lambda
        self.clip_range, self.n_epochs = clip_range, n_epochs
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self._last_state = self._last_action = self._last_logprob = self._last_value = None

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select_action(self, state):
        s = self._to_tensor(state).unsqueeze(0)
        action, logp, value = self.net.act(s)
        self._last_state, self._last_action = state.copy(), action
        self._last_logprob, self._last_value = logp, value
        return action

    @torch.no_grad()
    def _estimate_value(self, next_state, done):
        if done:
            return 0.0
        _, v = self.net.forward(self._to_tensor(next_state).unsqueeze(0))
        return float(v.item())

    def store(self, reward, done, next_state):
        next_val = self._estimate_value(next_state, done)
        self.buffer.add(state=self._last_state, action=self._last_action,
                        reward=float(reward), done=bool(done),
                        logprob=float(self._last_logprob), value=float(self._last_value),
                        next_value=float(next_val))

    def update(self):
        if len(self.buffer) == 0:
            return {}
        states, actions, old_logprobs, returns, advantages = self.buffer.compute()
        states, actions = states.to(self.device), actions.to(self.device)
        old_logprobs, returns = old_logprobs.to(self.device), returns.to(self.device)
        advantages = advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = states.size(0)
        idx = np.arange(n)
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = torch.as_tensor(idx[start:start + self.batch_size], dtype=torch.long, device=self.device)
                new_logp, entropy, values = self.net.evaluate(states[mb], actions[mb])
                ratio = (new_logp - old_logprobs[mb]).exp()
                unclipped = ratio * advantages[mb]
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages[mb]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.MSELoss()(values, returns[mb])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()
        self.buffer.clear()
        return {}


class PPOAdapter:
    def __init__(self, env, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.env = env
        self.uav_ids = env.uav_ids
        self.masks = enumerate_masks(env.num_uavs, env.min_active_uavs)
        self.agent = PPOAgent(state_dim(env.num_uavs), len(self.masks))
        self._last_idx = None

    def begin_episode(self, obs):
        pass

    def act(self, obs):
        state = np.asarray(global_state_vector(obs, self.env), dtype=np.float32)
        idx = self.agent.select_action(state)
        self._last_idx = idx
        return mask_to_action(self.masks[idx], self.uav_ids)

    def observe(self, obs, action, reward_info, next_obs, done):
        next_state = np.asarray(global_state_vector(next_obs, self.env), dtype=np.float32)
        self.agent.store(reward_info["reward_drl"], done, next_state)

    def end_episode(self):
        self.agent.update()  # PPO updates once per episode
