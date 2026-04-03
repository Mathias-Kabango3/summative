from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import pygame

from utils import DIR_ANGLE


@dataclass
class Drone:
    drone_id: int
    gx: int
    gy: int
    battery: float
    has_package: bool
    sprite_path: Path
    cell_size: int
    speed_px: float = 260.0

    _cache: ClassVar[dict[tuple[str, str], pygame.Surface]] = {}

    def __post_init__(self) -> None:
        self.direction = "UP"
        self.render_x = self.gx * self.cell_size
        self.render_y = self.gy * self.cell_size
        self.target_x = self.render_x
        self.target_y = self.render_y

    def _sprite(self) -> pygame.Surface:
        key = (str(self.sprite_path), self.direction)
        if key not in Drone._cache:
            base = pygame.image.load(str(self.sprite_path))
            Drone._cache[key] = pygame.transform.rotate(base, DIR_ANGLE[self.direction])
        return Drone._cache[key]

    def set_grid_position(self, gx: int, gy: int, direction: str) -> None:
        self.gx = gx
        self.gy = gy
        self.direction = direction
        self.target_x = gx * self.cell_size
        self.target_y = gy * self.cell_size

    def update_render(self, dt: float) -> None:
        # Smooth interpolation toward target cell for pleasant motion.
        dx = self.target_x - self.render_x
        dy = self.target_y - self.render_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-3:
            self.render_x = self.target_x
            self.render_y = self.target_y
            return

        step = self.speed_px * dt
        if step >= dist:
            self.render_x = self.target_x
            self.render_y = self.target_y
        else:
            self.render_x += dx / dist * step
            self.render_y += dy / dist * step

    def draw(self, screen: pygame.Surface) -> None:
        sprite = self._sprite()
        # Center sprite in cell.
        x = self.render_x + (self.cell_size - sprite.get_width()) / 2
        y = self.render_y + (self.cell_size - sprite.get_height()) / 2
        screen.blit(sprite, (x, y))
