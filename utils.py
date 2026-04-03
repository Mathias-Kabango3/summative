from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
import pygame

ACTIONS = {
    0: (0, -1),  # up
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
    4: (0, 0),   # hover
    5: (0, 0),   # recharge (valid at charger cell)
}

DIR_ANGLE = {
    "UP": 0,
    "RIGHT": 270,
    "DOWN": 180,
    "LEFT": 90,
}


def ensure_assets(asset_root: Path) -> dict[str, Path]:
    asset_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "drone": asset_root / "drone.png",
        "road": asset_root / "road_tile.png",
        "building": asset_root / "building_tile.png",
        "building_alt": asset_root / "building_tile_alt.png",
        "tree": asset_root / "tree_tile.png",
        "ground": asset_root / "ground_tile.png",
        "delivery": asset_root / "delivery_tile.png",
        "package": asset_root / "package_tile.png",
        "destination": asset_root / "destination_tile.png",
    }

    if not pygame.get_init():
        pygame.init()

    if not paths["drone"].exists():
        surf = pygame.Surface((44, 44), pygame.SRCALPHA)
        pygame.draw.circle(surf, (45, 55, 70), (22, 22), 15)
        pygame.draw.circle(surf, (95, 175, 245), (22, 22), 10)
        pygame.draw.circle(surf, (220, 245, 255), (22, 22), 4)
        for x, y in [(7, 7), (37, 7), (7, 37), (37, 37)]:
            pygame.draw.circle(surf, (30, 30, 30), (x, y), 4)
            pygame.draw.line(surf, (40, 40, 40), (22, 22), (x, y), 2)
        pygame.draw.polygon(surf, (255, 200, 70), [(20, 3), (24, 3), (22, 11)])
        pygame.image.save(surf, paths["drone"])

    if not paths["road"].exists():
        road = pygame.Surface((64, 64), pygame.SRCALPHA)
        road.fill((54, 58, 64))
        for y in range(0, 64, 16):
            pygame.draw.line(road, (235, 235, 235), (30, y), (34, min(63, y + 8)), 2)
        pygame.image.save(road, paths["road"])

    if not paths["building"].exists():
        b = pygame.Surface((64, 64), pygame.SRCALPHA)
        b.fill((98, 102, 112))
        pygame.draw.rect(b, (120, 125, 135), (5, 5, 54, 54), border_radius=4)
        for r in range(4):
            for c in range(3):
                pygame.draw.rect(b, (190, 210, 230), (11 + c * 15, 11 + r * 12, 8, 7), border_radius=2)
        pygame.image.save(b, paths["building"])

    if not paths["building_alt"].exists():
        b2 = pygame.Surface((64, 64), pygame.SRCALPHA)
        b2.fill((92, 96, 104))
        pygame.draw.rect(b2, (112, 116, 126), (4, 4, 56, 56), border_radius=5)
        for r in range(3):
            for c in range(4):
                pygame.draw.rect(b2, (170, 195, 220), (9 + c * 12, 10 + r * 15, 7, 10), border_radius=2)
        pygame.draw.rect(b2, (70, 75, 82), (27, 42, 10, 18), border_radius=2)
        pygame.image.save(b2, paths["building_alt"])

    if not paths["ground"].exists():
        g = pygame.Surface((64, 64), pygame.SRCALPHA)
        g.fill((86, 148, 94))
        for _ in range(40):
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            g.set_at((x, y), (92, 160, 100, 255))
        pygame.image.save(g, paths["ground"])

    if not paths["tree"].exists():
        t = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.rect(t, (92, 66, 35), (29, 34, 8, 20), border_radius=2)
        pygame.draw.circle(t, (48, 120, 58), (32, 26), 20)
        pygame.draw.circle(t, (56, 138, 66), (23, 29), 13)
        pygame.draw.circle(t, (56, 138, 66), (40, 30), 12)
        pygame.draw.circle(t, (72, 156, 82), (32, 18), 9)
        pygame.image.save(t, paths["tree"])

    if not paths["delivery"].exists():
        d = pygame.Surface((64, 64), pygame.SRCALPHA)
        d.fill((230, 190, 72))
        pygame.draw.rect(d, (255, 240, 170), (10, 10, 44, 44), border_radius=8)
        pygame.draw.line(d, (145, 105, 40), (10, 32), (54, 32), 3)
        pygame.draw.line(d, (145, 105, 40), (32, 10), (32, 54), 3)
        pygame.image.save(d, paths["delivery"])

    if not paths["package"].exists():
        p = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.rect(p, (210, 156, 88), (12, 14, 40, 36), border_radius=6)
        pygame.draw.rect(p, (240, 188, 120), (12, 14, 40, 10), border_radius=4)
        pygame.draw.line(p, (120, 88, 50), (32, 14), (32, 50), 2)
        pygame.draw.line(p, (120, 88, 50), (12, 32), (52, 32), 2)
        pygame.image.save(p, paths["package"])

    if not paths["destination"].exists():
        m = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.circle(m, (247, 118, 66), (32, 30), 18)
        pygame.draw.circle(m, (255, 214, 200), (32, 30), 8)
        pygame.draw.polygon(m, (247, 118, 66), [(32, 58), (23, 42), (41, 42)])
        pygame.image.save(m, paths["destination"])

    return paths


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    deliveries: int
    collisions: int


class CsvLogger:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with self.file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "reward", "deliveries", "collisions"])
                writer.writeheader()

    def append(self, row: EpisodeLog) -> None:
        with self.file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "reward", "deliveries", "collisions"])
            writer.writerow(
                {
                    "episode": row.episode,
                    "reward": f"{row.reward:.4f}",
                    "deliveries": row.deliveries,
                    "collisions": row.collisions,
                }
            )
