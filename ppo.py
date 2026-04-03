from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.pi(h), self.v(h)


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.epochs = 4

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.logp: list[float] = []
        self.values: list[float] = []

    def act(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(x)
            dist = Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a)

        self.states.append(obs)
        self.actions.append(int(a.item()))
        self.logp.append(float(lp.item()))
        self.values.append(float(value.item()))
        return int(a.item())

    def act_greedy(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(x)
            return int(torch.argmax(logits, dim=1).item())

    def store_outcome(self, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        self.dones.append(float(done))

    def _gae(self) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [0.0], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        ret = adv + values[:-1]
        return adv, ret

    def update(self) -> float:
        if not self.states:
            return 0.0

        adv, ret = self._gae()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_lp_t = torch.tensor(self.logp, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)

        loss_value = 0.0
        for _ in range(self.epochs):
            logits, values = self.model(s_t)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(a_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            actor = -torch.min(s1, s2).mean()
            critic = (ret_t - values.squeeze(-1)).pow(2).mean()
            loss = actor + self.value_coef * critic - self.entropy_coef * entropy

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            loss_value = float(loss.item())

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logp.clear()
        self.values.clear()

        return loss_value

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
