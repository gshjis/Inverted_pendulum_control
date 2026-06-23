"""Renderer: класс-обёртка для функций отрисовки."""

from __future__ import annotations

import pygame
from typing import Any

from .draw import draw_cart, draw_pendulums, draw_force_arrow, draw_hud, draw_record_button
from .constants import WIDTH, HEIGHT, TRACK_Y, CART_H, FPS


class Renderer:
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font, controller: Any = None) -> None:
        self.screen = screen
        self.font = font
        self.controller = controller

    def render(self, plant: Any, applied_force: float, recording: bool) -> None:
        # Собрать необходимые данные
        q = plant.q
        dq = plant.dq
        is_single = plant.single_pendulum_mode
        x = q[0]
        th1 = q[1]
        th2 = q[2] if not is_single else 0.0
        dx = dq[0]
        dth1 = dq[1]
        dth2 = dq[2] if not is_single else 0.0

        cart_x_px = int(WIDTH // 2 + x * 200.0)
        cart_y_px = TRACK_Y - CART_H // 2

        # Очистка экрана
        self.screen.fill((10, 10, 10))

        # Рельс
        pygame.draw.line(self.screen, (100, 100, 100), (0, TRACK_Y), (WIDTH, TRACK_Y), 2)

        draw_cart(self.screen, cart_x_px, cart_y_px)
        draw_pendulums(
            self.screen, cart_x_px, cart_y_px,
            th1, th2, is_single,
            l1=plant._l1, l2=plant._l2,
        )
        draw_force_arrow(self.screen, applied_force, cart_x_px, cart_y_px)

        # HUD
        mode = "PID" if self.controller else "РУЧНОЕ"
        gains_str = ""
        if self.controller and hasattr(self.controller, "gains"):
            g = self.controller.gains  # type: ignore[attr-defined]
            gains_str = f"  Kp={g[0]:.1f}  Ki={g[1]:.1f}  Kd={g[2]:.1f}  Kx={g[3]:.1f}"

        lines = [
            f"Сила: {applied_force:+.1f} Н  [{mode}]",
            f"x  = {x:+.3f} м      θ1 = {__import__('numpy').degrees(th1):+7.1f}°",
        ]
        if not is_single:
            lines.append(f"θ2 = {__import__('numpy').degrees(th2):+7.1f}°")
        lines.append(f"ẋ  = {dx:+.3f} м/с   θ̇1 = {__import__('numpy').degrees(dth1):+7.1f}°/с")
        if not is_single:
            lines.append(f"θ̇2 = {__import__('numpy').degrees(dth2):+7.1f}°/с")
        if gains_str:
            lines.append(gains_str)

        draw_hud(self.screen, self.font, lines)
        draw_record_button(self.screen, self.font, recording)

        pygame.display.flip()
        self.clock_tick()

    def clock_tick(self) -> None:
        pygame.time.Clock().tick(FPS)

