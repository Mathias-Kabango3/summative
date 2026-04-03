from __future__ import annotations

from pathlib import Path
import pygame


class CityVisualization:
    def __init__(self, env) -> None:
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((env.width, env.height + 90))
        pygame.display.set_caption("Drone Delivery RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("verdana", 22)
        self.small = pygame.font.SysFont("verdana", 17)

        self.package_img = pygame.image.load(str(env.assets["package"]))
        self.destination_img = pygame.image.load(str(env.assets["destination"]))

        self.building_tiles = self._load_building_tiles()

        self.flash_timer = 0.0

    def _load_building_tiles(self) -> list[pygame.Surface]:
        tiles: list[pygame.Surface] = []
        asset_dir = Path("assets")
        cell = self.env.cell_size
        for name in ("building-1.jpg", "building-2.jpg", "building-3.jpg"):
            path = asset_dir / name
            if not path.exists():
                continue
            img = pygame.image.load(str(path)).convert()
            tiles.append(pygame.transform.smoothscale(img, (cell - 8, cell - 8)))
        return tiles

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _draw_battery_symbol(self, x: int, y: int) -> None:
        cell = self.env.cell_size
        px, py = x * cell, y * cell
        body = pygame.Rect(px + 18, py + 20, 28, 22)
        tip = pygame.Rect(px + 46, py + 27, 4, 8)
        pygame.draw.rect(self.screen, (30, 90, 45), body, border_radius=3)
        pygame.draw.rect(self.screen, (30, 90, 45), tip, border_radius=2)
        pygame.draw.rect(self.screen, (120, 230, 140), (px + 21, py + 23, 22, 16), border_radius=2)

    def render(self, reward: float, info: dict) -> None:
        cell = self.env.cell_size
        self.screen.fill((236, 242, 248))

        # Base clean grid.
        for y in range(self.env.cfg.grid_h):
            for x in range(self.env.cfg.grid_w):
                px, py = x * cell, y * cell
                shade = 247 if (x + y) % 2 == 0 else 243
                pygame.draw.rect(self.screen, (shade, shade, 252), (px, py, cell, cell))
                pygame.draw.rect(self.screen, (210, 218, 230), (px, py, cell, cell), 1)

        # Obstacles that the agent must avoid.
        for ox, oy in getattr(self.env, "obstacles", set()):
            px, py = ox * cell, oy * cell
            if self.building_tiles:
                idx = (ox + oy) % len(self.building_tiles)
                self.screen.blit(self.building_tiles[idx], (px + 4, py + 4))
                pygame.draw.rect(self.screen, (50, 60, 74), (px + 4, py + 4, cell - 8, cell - 8), 1, border_radius=6)
            else:
                pygame.draw.rect(self.screen, (72, 82, 98), (px + 6, py + 6, cell - 12, cell - 12), border_radius=8)
                pygame.draw.line(self.screen, (170, 180, 198), (px + 16, py + 16), (px + cell - 16, py + cell - 16), 3)
                pygame.draw.line(self.screen, (170, 180, 198), (px + cell - 16, py + 16), (px + 16, py + cell - 16), 3)

        # Charging stations with battery symbols.
        for cx, cy in getattr(self.env, "charger_cells", set()):
            pygame.draw.rect(
                self.screen,
                (86, 212, 126),
                (cx * cell + 6, cy * cell + 6, cell - 12, cell - 12),
                3,
                border_radius=8,
            )
            self._draw_battery_symbol(cx, cy)

        # Pickup and dropoff markers for the active package task.
        pickup = info.get("package_position")
        dropoff = info.get("destination_position")
        if pickup is not None:
            self.screen.blit(self.package_img, (pickup[0] * cell, pickup[1] * cell))
            pygame.draw.rect(
                self.screen,
                (65, 165, 245),
                (pickup[0] * cell + 6, pickup[1] * cell + 6, cell - 12, cell - 12),
                3,
                border_radius=6,
            )
        if dropoff is not None:
            self.screen.blit(self.destination_img, (dropoff[0] * cell, dropoff[1] * cell))
            pygame.draw.rect(
                self.screen,
                (244, 118, 68),
                (dropoff[0] * cell + 6, dropoff[1] * cell + 6, cell - 12, cell - 12),
                3,
                border_radius=6,
            )

        # Animate drones smoothly.
        dt = 1.0 / 60.0
        for d in self.env.drones:
            d.update_render(dt)
            d.draw(self.screen)

        # Positive reward visual feedback.
        if reward > 0:
            self.flash_timer = 0.25
        if self.flash_timer > 0:
            glow = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
            glow.fill((60, 200, 80, 45))
            self.screen.blit(glow, (0, 0))
            self.flash_timer -= dt

        # HUD panel.
        pygame.draw.rect(self.screen, (16, 18, 24), (0, self.env.height, self.env.width, 90))
        line1 = (
            f"Step: {self.env.step_count}/{self.env.cfg.max_steps}   Reward: {reward:.2f}   "
            f"Pickups: {info.get('total_pickups', 0)}   Deliveries: {info.get('total_deliveries', 0)}"
        )
        line2 = (
            f"Collisions: {info.get('total_collisions', 0)}   Recharges: {info.get('total_recharges', 0)}   "
            f"Carrying package: {'yes' if info.get('has_package', False) else 'no'}"
        )

        txt1 = self.font.render(line1, True, (240, 240, 240))
        txt2 = self.font.render(line2, True, (240, 240, 240))
        self.screen.blit(txt1, (16, self.env.height + 14))
        self.screen.blit(txt2, (16, self.env.height + 46))

        battery = info.get("battery", self.env.drones[0].battery)
        bx = self.env.width - 300
        by = self.env.height + 14
        t = self.small.render(f"Battery: {battery:.1f}/{self.env.cfg.max_battery:.0f}", True, (180, 220, 255))
        self.screen.blit(t, (bx, by))
        legend = self.small.render("Blue box=package  Orange marker=destination  Green battery=charger  Dark X=obstacle", True, (200, 220, 200))
        self.screen.blit(legend, (bx - 200, by + 24))

        pygame.display.flip()
        self.clock.tick(60)
