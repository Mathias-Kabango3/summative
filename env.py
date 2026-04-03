from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone import Drone
from utils import ACTIONS, ensure_assets


@dataclass
class EnvConfig:
    grid_w: int = 12
    grid_h: int = 12
    n_drones: int = 1
    max_steps: int = 700
    max_battery: float = 180.0
    recharge_rate: float = 12.0
    obstacle_density: float = 0.10


class DroneDeliveryEnv:
    def __init__(self, config: EnvConfig | None = None, seed: int = 42) -> None:
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.assets = ensure_assets(Path("assets"))

        self.cell_size = 64
        self.width = self.cfg.grid_w * self.cell_size
        self.height = self.cfg.grid_h * self.cell_size

        self.charger_cells = self._build_charger_cells()
        self.obstacles = self._build_obstacles()
        self.roads = self._build_roads()
        # Compatibility fields used by renderer/training flow.
        self.buildings = set(self.obstacles)
        self.trees: set[tuple[int, int]] = set()
        self.building_style: dict[tuple[int, int], str] = {}
        self.blocked_cells = set(self.obstacles)

        self.drones: list[Drone] = []
        self.pickup_cell: tuple[int, int] | None = None
        self.dropoff_cell: tuple[int, int] | None = None
        self.step_count = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self.total_recharges = 0

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
                # Keep center corridors clean for easier navigation.
                if x in (self.cfg.grid_w // 2, self.cfg.grid_w // 2 - 1):
                    continue
                if y in (self.cfg.grid_h // 2, self.cfg.grid_h // 2 - 1):
                    continue
                if self.rng.random() < self.cfg.obstacle_density:
                    obstacles.add(cell)
        return obstacles

    def _build_roads(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.cfg.grid_w)
            for y in range(self.cfg.grid_h)
            if (x, y) not in self.obstacles
        }

    def _nearest_charger(self, gx: int, gy: int) -> tuple[int, int]:
        return min(self.charger_cells, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))

    def _sample_road_cell(self) -> tuple[int, int]:
        candidates = [c for c in self.roads if c not in self.charger_cells]
        idx = int(self.rng.integers(0, len(candidates)))
        return candidates[idx]

    def _spawn_task(self) -> None:
        self.pickup_cell = self._sample_road_cell()
        self.dropoff_cell = self._sample_road_cell()
        tries = 0
        while (
            self.dropoff_cell == self.pickup_cell
            or abs(self.dropoff_cell[0] - self.pickup_cell[0]) + abs(self.dropoff_cell[1] - self.pickup_cell[1]) < 5
        ) and tries < 50:
            self.dropoff_cell = self._sample_road_cell()
            tries += 1

    def _current_target(self, drone: Drone) -> tuple[int, int]:
        if drone.has_package and self.dropoff_cell is not None:
            return self.dropoff_cell
        if not drone.has_package and self.pickup_cell is not None:
            return self.pickup_cell
        return drone.gx, drone.gy

    def _local_obs(self, idx: int) -> np.ndarray:
        d = self.drones[idx]
        tx, ty = self._current_target(d)
        cx, cy = self._nearest_charger(d.gx, d.gy)

        near_obstacles = []
        for a in [0, 1, 2, 3]:
            dx, dy = ACTIONS[a]
            nx, ny = d.gx + dx, d.gy + dy
            blocked = int((nx, ny) in self.blocked_cells or not (0 <= nx < self.cfg.grid_w and 0 <= ny < self.cfg.grid_h))
            near_obstacles.append(blocked)

        dist_target = abs(d.gx - tx) + abs(d.gy - ty)
        dist_charger = abs(d.gx - cx) + abs(d.gy - cy)

        return np.array(
            [
                d.gx / max(1, self.cfg.grid_w - 1),
                d.gy / max(1, self.cfg.grid_h - 1),
                tx / max(1, self.cfg.grid_w - 1),
                ty / max(1, self.cfg.grid_h - 1),
                cx / max(1, self.cfg.grid_w - 1),
                cy / max(1, self.cfg.grid_h - 1),
                d.battery / max(1.0, self.cfg.max_battery),
                float(d.has_package),
                dist_target / max(1, self.cfg.grid_w + self.cfg.grid_h),
                dist_charger / max(1, self.cfg.grid_w + self.cfg.grid_h),
                float(d.battery < 0.2 * self.cfg.max_battery),
                float((d.gx, d.gy) in self.charger_cells),
                *near_obstacles,
            ],
            dtype=np.float32,
        )

    @property
    def obs_dim(self) -> int:
        return 16

    @property
    def n_actions(self) -> int:
        return 6

    def reset(self) -> tuple[list[np.ndarray], dict[str, Any]]:
        self.step_count = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self.total_recharges = 0

        self.drones = [
            Drone(
                drone_id=0,
                gx=1,
                gy=1,
                battery=self.cfg.max_battery,
                has_package=False,
                sprite_path=self.assets["drone"],
                cell_size=self.cell_size,
            )
        ]
        self._spawn_task()

        obs = [self._local_obs(0)]
        return obs, {}

    def step(self, actions: list[int]) -> tuple[list[np.ndarray], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        d = self.drones[0]
        a = int(actions[0]) if actions else 4
        target = self._current_target(d)
        old_dist_target = abs(d.gx - target[0]) + abs(d.gy - target[1])
        nearest_charger = self._nearest_charger(d.gx, d.gy)
        old_dist_charger = abs(d.gx - nearest_charger[0]) + abs(d.gy - nearest_charger[1])

        reward = -0.06
        collisions = 0
        pickups = 0
        deliveries = 0
        recharges = 0

        if a == 5:
            if (d.gx, d.gy) in self.charger_cells:
                prev = d.battery
                d.battery = min(self.cfg.max_battery, d.battery + self.cfg.recharge_rate)
                gained = d.battery - prev
                recharges = 1 if gained > 0 else 0
                reward += 0.25 + 0.03 * gained
                if d.battery < 0.4 * self.cfg.max_battery:
                    reward += 0.5
                self.total_recharges += recharges
            else:
                d.battery = max(0.0, d.battery - 0.4)
                reward -= 0.6
        else:
            dx, dy = ACTIONS.get(a, (0, 0))
            nx, ny = d.gx + dx, d.gy + dy
            direction = d.direction
            if dx == 1:
                direction = "RIGHT"
            elif dx == -1:
                direction = "LEFT"
            elif dy == 1:
                direction = "DOWN"
            elif dy == -1:
                direction = "UP"

            invalid_cell = not (0 <= nx < self.cfg.grid_w and 0 <= ny < self.cfg.grid_h)
            hit_building = (nx, ny) in self.blocked_cells
            if invalid_cell or hit_building:
                reward -= 3.0
                collisions = 1
                nx, ny = d.gx, d.gy
            d.battery = max(0.0, d.battery - (1.2 if a in (0, 1, 2, 3) else 0.4))
            d.set_grid_position(nx, ny, direction)

        new_target = self._current_target(d)
        new_dist_target = abs(d.gx - new_target[0]) + abs(d.gy - new_target[1])
        if new_dist_target < old_dist_target:
            reward += 0.25
        elif new_dist_target > old_dist_target and a != 5:
            reward -= 0.08

        nearest_charger = self._nearest_charger(d.gx, d.gy)
        new_dist_charger = abs(d.gx - nearest_charger[0]) + abs(d.gy - nearest_charger[1])
        low_battery = d.battery < 0.25 * self.cfg.max_battery
        if low_battery:
            if (d.gx, d.gy) in self.charger_cells and a == 5:
                reward += 0.8
            elif new_dist_charger < old_dist_charger:
                reward += 0.2
            else:
                reward -= 0.35

        if not d.has_package and self.pickup_cell is not None and (d.gx, d.gy) == self.pickup_cell:
            d.has_package = True
            pickups = 1
            self.total_pickups += 1
            reward += 7.0

        if d.has_package and self.dropoff_cell is not None and (d.gx, d.gy) == self.dropoff_cell:
            d.has_package = False
            deliveries = 1
            self.total_deliveries += 1
            reward += 14.0
            self._spawn_task()

        self.total_collisions += collisions

        if d.battery <= 0.0:
            reward -= 8.0

        total_reward = float(reward)
        done = self.step_count >= self.cfg.max_steps or d.battery <= 0.0
        truncated = False

        obs = [self._local_obs(0)]
        info = {
            "pickups": pickups,
            "deliveries": deliveries,
            "recharges": recharges,
            "collisions": collisions,
            "active_task": {"pickup": self.pickup_cell, "dropoff": self.dropoff_cell},
            "package_position": None if d.has_package else self.pickup_cell,
            "destination_position": self.dropoff_cell,
            "chargers": sorted(self.charger_cells),
            "obstacles": sorted(self.obstacles),
            "battery": d.battery,
            "has_package": d.has_package,
            "per_drone_rewards": [total_reward],
            "total_pickups": self.total_pickups,
            "total_deliveries": self.total_deliveries,
            "total_collisions": self.total_collisions,
            "total_recharges": self.total_recharges,
        }
        return obs, total_reward, done, truncated, info
