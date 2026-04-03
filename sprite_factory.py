from __future__ import annotations

from pathlib import Path
import pygame


def _build_sprite(color: tuple[int, int, int]) -> pygame.Surface:
    """Create a simple top-down car sprite with transparency."""
    surf = pygame.Surface((28, 52), pygame.SRCALPHA)

    # Body polygon (kept slightly tapered to look less like a rectangle).
    pygame.draw.polygon(
        surf,
        color,
        [(6, 50), (22, 50), (26, 38), (24, 10), (20, 3), (8, 3), (4, 10), (2, 38)],
    )

    # Windshield and rear window.
    pygame.draw.polygon(surf, (180, 220, 240), [(8, 14), (20, 14), (18, 22), (10, 22)])
    pygame.draw.polygon(surf, (130, 170, 200), [(9, 30), (19, 30), (18, 40), (10, 40)])

    # Wheels.
    pygame.draw.circle(surf, (25, 25, 25), (5, 14), 3)
    pygame.draw.circle(surf, (25, 25, 25), (23, 14), 3)
    pygame.draw.circle(surf, (25, 25, 25), (5, 38), 3)
    pygame.draw.circle(surf, (25, 25, 25), (23, 38), 3)
    return surf


def ensure_car_sprites(asset_dir: Path) -> dict[str, str]:
    """Generate PNG sprites once so cars are loaded from files at runtime."""
    asset_dir.mkdir(parents=True, exist_ok=True)

    sprite_specs = {
        "car_blue.png": (50, 130, 255),
        "car_orange.png": (240, 145, 40),
        "car_green.png": (65, 190, 120),
    }

    if not pygame.get_init():
        pygame.init()

    for name, color in sprite_specs.items():
        path = asset_dir / name
        if not path.exists():
            surf = _build_sprite(color)
            pygame.image.save(surf, path)

    return {k: str(asset_dir / k) for k in sprite_specs}
