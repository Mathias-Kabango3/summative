from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReinforceAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.policy = Policy(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=6e-4)

        self.gamma = 0.99
        self.entropy_coef = 0.01

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []

    def act(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        self.states.append(obs)
        self.actions.append(int(a.item()))
        return int(a.item())

    def act_greedy(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x)
            return int(torch.argmax(logits, dim=1).item())

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def _returns(self) -> np.ndarray:
        out = []
        run = 0.0
        for r in reversed(self.rewards):
            run = r + self.gamma * run
            out.append(run)
        out.reverse()
        arr = np.array(out, dtype=np.float32)
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    def update(self) -> float:
        if not self.states:
            return 0.0

        s_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        ret_t = torch.tensor(self._returns(), dtype=torch.float32, device=self.device)

        logits = self.policy(s_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a_t)
        entropy = dist.entropy().mean()

        loss = -(logp * ret_t).mean() - self.entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)
