from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class EnvConfig:
    grid_w: int = 12
    grid_h: int = 12
    max_steps: int = 700
    max_battery: float = 180.0
    recharge_rate: float = 12.0
    obstacle_density: float = 0.10


class DroneDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(42)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

        self.charger_cells = self._build_charger_cells()
        self.obstacles = self._build_obstacles()
        self.free_cells = self._build_free_cells()

        self.drone_pos = (1, 1)
        self.drone_dir = "UP"
        self.battery = self.cfg.max_battery
        self.has_package = False
        self.package_pos: tuple[int, int] | None = None
        self.destination_pos: tuple[int, int] | None = None

        self.step_count = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self.total_recharges = 0

        self._renderer = None

    def _build_charger_cells(self) -> set[tuple[int, int]]:
        return {
            (1, 1),
            (self.cfg.grid_w - 2, 1),
            (1, self.cfg.grid_h - 2),
            (self.cfg.grid_w - 2, self.cfg.grid_h - 2),
        }

    def _build_obstacles(self) -> set[tuple[int, int]]:
        obstacles: set[tuple[int, int]] = set()
        for x in range(self.cfg.grid_w):
            for y in range(self.cfg.grid_h):
                cell = (x, y)
                if cell in self.charger_cells:
                    continue
                if x in (0, self.cfg.grid_w - 1) or y in (0, self.cfg.grid_h - 1):
                    continue
                if x in (self.cfg.grid_w // 2, self.cfg.grid_w // 2 - 1):
                    continue
                if y in (self.cfg.grid_h // 2, self.cfg.grid_h // 2 - 1):
                    continue
                if self.rng.random() < self.cfg.obstacle_density:
                    obstacles.add(cell)
        return obstacles

    def _build_free_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.cfg.grid_w)
            for y in range(self.cfg.grid_h)
            if (x, y) not in self.obstacles
        }

    def _nearest_charger(self, gx: int, gy: int) -> tuple[int, int]:
        return min(self.charger_cells, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))

    def _sample_free_cell(self) -> tuple[int, int]:
        candidates = [c for c in self.free_cells if c not in self.charger_cells]
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _spawn_task(self) -> None:
        self.package_pos = self._sample_free_cell()
        self.destination_pos = self._sample_free_cell()
        tries = 0
        while (
            self.destination_pos == self.package_pos
            or abs(self.destination_pos[0] - self.package_pos[0]) + abs(self.destination_pos[1] - self.package_pos[1]) < 5
        ) and tries < 80:
            self.destination_pos = self._sample_free_cell()
            tries += 1

    def _current_target(self) -> tuple[int, int]:
        if self.has_package and self.destination_pos is not None:
            return self.destination_pos
        if (not self.has_package) and self.package_pos is not None:
            return self.package_pos
        return self.drone_pos

    def _obs(self) -> np.ndarray:
        gx, gy = self.drone_pos
        tx, ty = self._current_target()
        cx, cy = self._nearest_charger(gx, gy)

        near_obstacles = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = gx + dx, gy + dy
            blocked = int((nx, ny) in self.obstacles or not (0 <= nx < self.cfg.grid_w and 0 <= ny < self.cfg.grid_h))
            near_obstacles.append(blocked)

        dist_target = abs(gx - tx) + abs(gy - ty)
        dist_charger = abs(gx - cx) + abs(gy - cy)

        return np.array(
            [
                gx / max(1, self.cfg.grid_w - 1),
                gy / max(1, self.cfg.grid_h - 1),
                tx / max(1, self.cfg.grid_w - 1),
                ty / max(1, self.cfg.grid_h - 1),
                cx / max(1, self.cfg.grid_w - 1),
                cy / max(1, self.cfg.grid_h - 1),
                self.battery / max(1.0, self.cfg.max_battery),
                float(self.has_package),
                dist_target / max(1, self.cfg.grid_w + self.cfg.grid_h),
                dist_charger / max(1, self.cfg.grid_w + self.cfg.grid_h),
                float(self.battery < 0.2 * self.cfg.max_battery),
                float((gx, gy) in self.charger_cells),
                *near_obstacles,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self.total_recharges = 0

        self.drone_pos = (1, 1)
        self.drone_dir = "UP"
        self.battery = self.cfg.max_battery
        self.has_package = False
        self._spawn_task()

        return self._obs(), self._info(0, 0, 0, 0)

    def _info(self, pickups: int, deliveries: int, recharges: int, collisions: int) -> dict[str, Any]:
        return {
            "pickups": pickups,
            "deliveries": deliveries,
            "recharges": recharges,
            "collisions": collisions,
            "package_position": None if self.has_package else self.package_pos,
            "destination_position": self.destination_pos,
            "chargers": sorted(self.charger_cells),
            "obstacles": sorted(self.obstacles),
            "battery": self.battery,
            "has_package": self.has_package,
            "total_pickups": self.total_pickups,
            "total_deliveries": self.total_deliveries,
            "total_collisions": self.total_collisions,
            "total_recharges": self.total_recharges,
        }

    def step(self, action: int):
        self.step_count += 1
        gx, gy = self.drone_pos
        tx, ty = self._current_target()
        old_dist_target = abs(gx - tx) + abs(gy - ty)
        old_charger = self._nearest_charger(gx, gy)
        old_dist_charger = abs(gx - old_charger[0]) + abs(gy - old_charger[1])

        reward = -0.06
        pickups = 0
        deliveries = 0
        recharges = 0
        collisions = 0

        if action == 5:
            if (gx, gy) in self.charger_cells:
                prev = self.battery
                self.battery = min(self.cfg.max_battery, self.battery + self.cfg.recharge_rate)
                gained = self.battery - prev
                recharges = 1 if gained > 0 else 0
                reward += 0.25 + 0.03 * gained
                if self.battery < 0.4 * self.cfg.max_battery:
                    reward += 0.5
                self.total_recharges += recharges
            else:
                self.battery = max(0.0, self.battery - 0.4)
                reward -= 0.6
        else:
            dx, dy = {
                0: (0, -1),
                1: (0, 1),
                2: (-1, 0),
                3: (1, 0),
                4: (0, 0),
            }.get(int(action), (0, 0))
            nx, ny = gx + dx, gy + dy

            if dx == 1:
                self.drone_dir = "RIGHT"
            elif dx == -1:
                self.drone_dir = "LEFT"
            elif dy == 1:
                self.drone_dir = "DOWN"
            elif dy == -1:
                self.drone_dir = "UP"

            invalid = not (0 <= nx < self.cfg.grid_w and 0 <= ny < self.cfg.grid_h)
            blocked = (nx, ny) in self.obstacles
            if invalid or blocked:
                reward -= 3.0
                collisions = 1
                nx, ny = gx, gy

            self.battery = max(0.0, self.battery - (1.2 if action in (0, 1, 2, 3) else 0.4))
            self.drone_pos = (nx, ny)

        gx, gy = self.drone_pos
        ntx, nty = self._current_target()
        new_dist_target = abs(gx - ntx) + abs(gy - nty)
        if new_dist_target < old_dist_target:
            reward += 0.25
        elif new_dist_target > old_dist_target and action != 5:
            reward -= 0.08

        new_charger = self._nearest_charger(gx, gy)
        new_dist_charger = abs(gx - new_charger[0]) + abs(gy - new_charger[1])
        low_battery = self.battery < 0.25 * self.cfg.max_battery
        if low_battery:
            if (gx, gy) in self.charger_cells and action == 5:
                reward += 0.8
            elif new_dist_charger < old_dist_charger:
                reward += 0.2
            else:
                reward -= 0.35

        if (not self.has_package) and self.package_pos is not None and self.drone_pos == self.package_pos:
            self.has_package = True
            pickups = 1
            self.total_pickups += 1
            reward += 7.0

        if self.has_package and self.destination_pos is not None and self.drone_pos == self.destination_pos:
            self.has_package = False
            deliveries = 1
            self.total_deliveries += 1
            reward += 14.0
            self._spawn_task()

        self.total_collisions += collisions

        if self.battery <= 0.0:
            reward -= 8.0

        terminated = self.battery <= 0.0
        truncated = self.step_count >= self.cfg.max_steps

        obs = self._obs()
        info = self._info(pickups, deliveries, recharges, collisions)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self._renderer is None:
            from environment.rendering import GridRenderer

            self._renderer = GridRenderer(self)
        self._renderer.render(self._info(0, 0, 0, 0))
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
