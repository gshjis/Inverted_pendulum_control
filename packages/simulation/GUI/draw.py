"""Функции отрисовки для PendulumViewer."""

from __future__ import annotations

import numpy as np
import pygame
from typing import Tuple

from .constants import (
    BLACK,
    WHITE,
    RED,
    GREEN,
    GRAY,
    ORANGE,
    WIDTH,
    HEIGHT,
    SCALE,
    CART_W,
    CART_H,
    WHEEL_R,
    PEND_R,
    TRACK_Y,
    FORCE_SCALE,
    FPS,
)


def draw_cart(screen: pygame.Surface, cart_x_px: int, cart_y_px: int) -> None:
    """Draw cart at pixel coordinates (cart_x_px, cart_y_px).

    cart_y_px should be provided by the caller to keep coordinates consistent
    with other draw_* functions.
    """
    pygame.draw.rect(
        screen, WHITE,
        (cart_x_px - CART_W // 2, cart_y_px - CART_H // 2, CART_W, CART_H),
        2,
    )
    for offset in (-CART_W // 4, CART_W // 4):
        pygame.draw.circle(screen, WHITE, (cart_x_px + offset, cart_y_px + WHEEL_R), WHEEL_R, 2)


def draw_pendulums(
    screen: pygame.Surface,
    cart_x_px: int,
    cart_y_px: int,
    th1: float,
    th2: float,
    is_single: bool,
    l1: float = 1.0,
    l2: float = 1.0,
) -> None:
    """Рисует подвесы относительно пиксельной позиции тележки (cart_x_px, cart_y_px).

    Parameters
    ----------
    l1 : float
        Реальная длина первого звена (м). Умножается на SCALE для перевода в пиксели.
    l2 : float
        Реальная длина второго звена (м). Умножается на SCALE для перевода в пиксели.
    """
    pivot1 = (cart_x_px, cart_y_px - CART_H // 2)
    pend1_x = pivot1[0] + l1 * SCALE * np.sin(th1)
    pend1_y = pivot1[1] + l1 * SCALE * np.cos(th1)
    pygame.draw.line(screen, ORANGE, pivot1, (pend1_x, pend1_y), 4)
    pygame.draw.circle(screen, RED, (int(pend1_x), int(pend1_y)), PEND_R)

    if not is_single:
        pivot2 = (pend1_x, pend1_y)
        pend2_x = pivot2[0] + l2 * SCALE * np.sin(th1 + th2)
        pend2_y = pivot2[1] + l2 * SCALE * np.cos(th1 + th2)
        pygame.draw.line(screen, ORANGE, pivot2, (pend2_x, pend2_y), 4)
        pygame.draw.circle(screen, RED, (int(pend2_x), int(pend2_y)), PEND_R)


def draw_force_arrow(screen: pygame.Surface, applied_force: float, cart_x_px: int, cart_y_px: int) -> None:
    if abs(applied_force) > 0.5:
        arrow_len = float(np.clip(abs(applied_force) * FORCE_SCALE, 10, 150))
        direction = 1 if applied_force > 0 else -1
        start_x = cart_x_px
        end_x = cart_x_px + int(direction * arrow_len)
        color = GREEN if applied_force > 0 else RED
        pygame.draw.line(screen, color, (start_x, cart_y_px), (end_x, cart_y_px), 4)
        tip = 10
        pygame.draw.line(screen, color, (end_x, cart_y_px), (end_x - direction * tip, cart_y_px - tip // 2), 3)
        pygame.draw.line(screen, color, (end_x, cart_y_px), (end_x - direction * tip, cart_y_px + tip // 2), 3)


def draw_hud(screen: pygame.Surface, font: pygame.font.Font, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        surf = font.render(line, True, GREEN)
        screen.blit(surf, (20, 20 + i * 22))


def draw_record_button(screen: pygame.Surface, font: pygame.font.Font, recording: bool) -> None:
    """Record button removed by user request — keep stub for compatibility."""
    return


def draw_controller_button(screen: pygame.Surface, font: pygame.font.Font, enabled: bool) -> None:
    """Draw controller enable/disable button in the top-right near record button.

    Button moved closer to right edge for visibility.
    """
    btn_rect = (WIDTH - 150, 10, 120, 30)
    # higher-contrast border and fill
    border = (200, 200, 200)
    fill_on = (30, 120, 30)
    fill_off = (80, 80, 80)
    pygame.draw.rect(screen, border, btn_rect)
    inner = (btn_rect[0] + 2, btn_rect[1] + 2, btn_rect[2] - 4, btn_rect[3] - 4)
    pygame.draw.rect(screen, fill_on if enabled else fill_off, inner)
    label = "CTRL: ON" if enabled else "CTRL: OFF"
    surf = font.render(label, True, (255, 255, 255) if enabled else (210, 210, 210))
    screen.blit(surf, (btn_rect[0] + 10, btn_rect[1] + 6))


def draw_target_marker(screen: pygame.Surface, cart_x_px: int, cart_y_px: int, color: Tuple[int, int, int], w: int, h: int, value_str: str) -> None:
    """Draw the target marker (a vertical rectangle) at cart_x_px, aligned to cart_y_px.

    value_str is rendered above the marker.
    """
    # draw a round marker (dot) on the rail line at cart_x_px
    radius = max(4, w // 2)
    pygame.draw.circle(screen, color, (cart_x_px, cart_y_px), radius)
    # small label under the dot showing the X coordinate
    font = pygame.font.SysFont("Consolas", 14, bold=False)
    surf = font.render(value_str, True, (200, 200, 200))
    screen.blit(surf, (cart_x_px - surf.get_width() // 2, cart_y_px + radius + 4))
