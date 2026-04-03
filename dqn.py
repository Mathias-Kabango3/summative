from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
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


class DQNAgent:
    """Shared DQN policy used by all drones each timestep."""

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.act_dim = act_dim

        self.q = QNet(obs_dim, act_dim).to(self.device)
        self.target = QNet(obs_dim, act_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=8e-4)
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.batch_size = 128
        self.replay = deque(maxlen=100000)
        self.update_steps = 0
        self.target_update = 500

    def act(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.act_dim)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def act_greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def remember(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.replay.append((s, a, r, ns, float(done)))

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s_t = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.tensor(np.array(ns), dtype=torch.float32, device=self.device)
        d_t = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.q(s_t).gather(1, a_t)
        with torch.no_grad():
            na = self.q(ns_t).argmax(dim=1, keepdim=True)
            nq = self.target(ns_t).gather(1, na)
            target = r_t + (1.0 - d_t) * self.gamma * nq

        loss = self.loss_fn(q_vals, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state)
        self.target.load_state_dict(self.q.state_dict())
