from __future__ import annotations

from pathlib import Path
import pygame


class GridRenderer:
    def __init__(self, env) -> None:
        pygame.init()
        self.env = env
        self.cell = 64
        self.width = env.cfg.grid_w * self.cell
        self.height = env.cfg.grid_h * self.cell
        self.screen = pygame.display.set_mode((self.width, self.height + 92))
        pygame.display.set_caption("Drone Delivery Grid")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("verdana", 21)
        self.small = pygame.font.SysFont("verdana", 13)

        # Prefer user-provided obstacle sprites. Fall back to geometric obstacle rendering.
        self.building_tiles = self._load_building_tiles()

    def _load_building_tiles(self) -> list[pygame.Surface]:
        tiles: list[pygame.Surface] = []
        asset_dir = Path("assets")
        for name in ("building-1.jpg", "building-2.jpg", "building-3.jpg"):
            path = asset_dir / name
            if not path.exists():
                continue
            img = pygame.image.load(str(path)).convert()
            tiles.append(pygame.transform.smoothscale(img, (self.cell - 8, self.cell - 8)))
        return tiles

    def _draw_battery(self, x: int, y: int) -> None:
        px, py = x * self.cell, y * self.cell
        body = pygame.Rect(px + 19, py + 21, 28, 22)
        tip = pygame.Rect(px + 47, py + 28, 4, 8)
        pygame.draw.rect(self.screen, (28, 92, 46), body, border_radius=3)
        pygame.draw.rect(self.screen, (28, 92, 46), tip, border_radius=2)
        pygame.draw.rect(self.screen, (134, 236, 150), (px + 22, py + 24, 22, 16), border_radius=2)

    def _draw_grid(self) -> None:
        for y in range(self.env.cfg.grid_h):
            for x in range(self.env.cfg.grid_w):
                px, py = x * self.cell, y * self.cell
                shade = 247 if (x + y) % 2 == 0 else 243
                pygame.draw.rect(self.screen, (shade, shade, 252), (px, py, self.cell, self.cell))
                pygame.draw.rect(self.screen, (208, 218, 230), (px, py, self.cell, self.cell), 1)

    def _draw_obstacles(self) -> None:
        for ox, oy in self.env.obstacles:
            px, py = ox * self.cell, oy * self.cell
            if self.building_tiles:
                idx = (ox + oy) % len(self.building_tiles)
                self.screen.blit(self.building_tiles[idx], (px + 4, py + 4))
                pygame.draw.rect(self.screen, (50, 60, 74), (px + 4, py + 4, self.cell - 8, self.cell - 8), 1, border_radius=6)
            else:
                pygame.draw.rect(self.screen, (72, 82, 98), (px + 6, py + 6, self.cell - 12, self.cell - 12), border_radius=8)
                pygame.draw.line(self.screen, (170, 180, 198), (px + 16, py + 16), (px + self.cell - 16, py + self.cell - 16), 3)
                pygame.draw.line(self.screen, (170, 180, 198), (px + self.cell - 16, py + 16), (px + 16, py + self.cell - 16), 3)

    def _draw_chargers(self) -> None:
        for cx, cy in self.env.charger_cells:
            px, py = cx * self.cell, cy * self.cell
            pygame.draw.rect(self.screen, (86, 212, 126), (px + 6, py + 6, self.cell - 12, self.cell - 12), 3, border_radius=8)
            self._draw_battery(cx, cy)
            lbl = self.small.render("CHG", True, (20, 96, 42))
            self.screen.blit(lbl, (px + 20, py + 4))

    def _draw_package_and_destination(self, info: dict) -> None:
        package = info.get("package_position")
        destination = info.get("destination_position")

        if package is not None:
            px, py = package[0] * self.cell, package[1] * self.cell
            pygame.draw.rect(self.screen, (213, 158, 88), (px + 14, py + 16, 36, 32), border_radius=6)
            pygame.draw.rect(self.screen, (240, 188, 120), (px + 14, py + 16, 36, 10), border_radius=4)
            pygame.draw.rect(self.screen, (65, 165, 245), (px + 6, py + 6, self.cell - 12, self.cell - 12), 3, border_radius=6)
            lbl = self.small.render("PKG", True, (33, 111, 174))
            self.screen.blit(lbl, (px + 19, py + 4))

        if destination is not None:
            px, py = destination[0] * self.cell, destination[1] * self.cell
            pygame.draw.circle(self.screen, (244, 118, 68), (px + 32, py + 28), 16)
            pygame.draw.circle(self.screen, (255, 222, 208), (px + 32, py + 28), 7)
            pygame.draw.polygon(self.screen, (244, 118, 68), [(px + 32, py + 53), (px + 24, py + 39), (px + 40, py + 39)])
            pygame.draw.rect(self.screen, (244, 118, 68), (px + 6, py + 6, self.cell - 12, self.cell - 12), 3, border_radius=6)
            lbl = self.small.render("DEST", True, (180, 74, 41))
            self.screen.blit(lbl, (px + 15, py + 4))

    def _draw_drone(self) -> None:
        x, y = self.env.drone_pos
        px, py = x * self.cell, y * self.cell
        pygame.draw.circle(self.screen, (40, 52, 70), (px + 32, py + 32), 14)
        pygame.draw.circle(self.screen, (96, 176, 245), (px + 32, py + 32), 9)
        pygame.draw.circle(self.screen, (225, 245, 255), (px + 32, py + 32), 4)

    def _draw_hud(self, info: dict) -> None:
        y0 = self.height
        pygame.draw.rect(self.screen, (17, 20, 26), (0, y0, self.width, 92))

        line1 = (
            f"Step {self.env.step_count}/{self.env.cfg.max_steps}  "
            f"Pickups {info.get('total_pickups', 0)}  Deliveries {info.get('total_deliveries', 0)}"
        )
        line2 = (
            f"Battery {info.get('battery', 0.0):.1f}/{self.env.cfg.max_battery:.0f}  "
            f"Recharges {info.get('total_recharges', 0)}  Collisions {info.get('total_collisions', 0)}"
        )

        self.screen.blit(self.font.render(line1, True, (240, 242, 246)), (14, y0 + 12))
        self.screen.blit(self.font.render(line2, True, (240, 242, 246)), (14, y0 + 46))

    def render(self, info: dict) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill((236, 242, 248))
        self._draw_grid()
        self._draw_obstacles()
        self._draw_chargers()
        self._draw_package_and_destination(info)
        self._draw_drone()
        self._draw_hud(info)

        pygame.display.flip()
        self.clock.tick(30)
        return True

    def close(self) -> None:
        pygame.quit()
