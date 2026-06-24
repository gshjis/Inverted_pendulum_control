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
    SINE_GRAPH_X,
    SINE_GRAPH_Y,
    SINE_GRAPH_W,
    SINE_GRAPH_H,
    SINE_COLOR,
    SINE_COLOR2,
    SINE_BG,
    SINE_GRID,
    ERR_GRAPH_X,
    ERR_GRAPH_Y,
    ERR_GRAPH_W,
    ERR_GRAPH_H,
    ERR_COLOR,
    ERR_BG,
    ERR_GRID,
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


def draw_sine_graph(
    screen: pygame.Surface,
    font: pygame.font.Font,
    sin_history: list[float],
    sin2_history: list[float] | None,
    is_single: bool,
) -> None:
    """
    Осциллограф: скроллирующийся график sin(θ₁) и (опционально) sin(θ₂).

    Размещается в правом верхнем углу (константы SINE_GRAPH_*).

    Parameters
    ----------
    sin_history : list[float]
        История sin(θ₁) — последние N отсчётов.
    sin2_history : list[float] | None
        История sin(θ₂) или None для однозвенного режима.
    is_single : bool
        Флаг однозвенного режима (не рисуем sin(θ₂)).
    """
    gx, gy = SINE_GRAPH_X, SINE_GRAPH_Y
    gw, gh = SINE_GRAPH_W, SINE_GRAPH_H

    # Фон
    pygame.draw.rect(screen, SINE_BG, (gx, gy, gw, gh))
    pygame.draw.rect(screen, SINE_GRID, (gx, gy, gw, gh), 1)

    # Сетка: нулевая линия и ±0.5
    zero_y = gy + gh // 2
    pygame.draw.line(screen, SINE_GRID, (gx, zero_y), (gx + gw, zero_y), 1)
    for frac in (0.25, 0.75):
        y = gy + int(gh * frac)
        pygame.draw.line(screen, SINE_GRID, (gx, y), (gx + gw, y), 1)

    # Подпись оси Y
    label = font.render("sin(θ)", True, SINE_GRID)
    screen.blit(label, (gx + 4, gy + 2))

    if not sin_history:
        return

    # Масштаб: вся ширина графика = вся история
    n = len(sin_history)
    step_x = gw / max(n - 1, 1)

    def _draw_trace(history: list[float], color: tuple[int, int, int]) -> None:
        points: list[tuple[int, int]] = []
        for i, val in enumerate(history):
            x = gx + int(i * step_x)
            # val в [-1, 1] → y в [gy, gy+gh]
            y = zero_y - int(val * (gh // 2))
            y = int(np.clip(y, gy + 1, gy + gh - 1))
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)

    # Первое звено — cyan
    _draw_trace(sin_history, SINE_COLOR)

    # Второе звено — orange (если двухзвенный режим)
    if not is_single and sin2_history:
        _draw_trace(sin2_history, SINE_COLOR2)


def draw_error_graph(
    screen: pygame.Surface,
    font: pygame.font.Font,
    err_history: list[float],
) -> None:
    """
    График ошибки положения тележки: e = target_x - current_x.

    Размещается под графиком sin(θ) (константы ERR_GRAPH_*).

    Parameters
    ----------
    err_history : list[float]
        История ошибки по X (м) — последние N отсчётов.
    """
    gx, gy = ERR_GRAPH_X, ERR_GRAPH_Y
    gw, gh = ERR_GRAPH_W, ERR_GRAPH_H

    # Фон
    pygame.draw.rect(screen, ERR_BG, (gx, gy, gw, gh))
    pygame.draw.rect(screen, ERR_GRID, (gx, gy, gw, gh), 1)

    # Нулевая линия
    zero_y = gy + gh // 2
    pygame.draw.line(screen, ERR_GRID, (gx, zero_y), (gx + gw, zero_y), 1)
    for frac in (0.25, 0.75):
        y = gy + int(gh * frac)
        pygame.draw.line(screen, ERR_GRID, (gx, y), (gx + gw, y), 1)

    # Подпись
    label = font.render("err X (м)", True, ERR_GRID)
    screen.blit(label, (gx + 4, gy + 2))

    if not err_history:
        return

    # Автомасштаб по Y: ищем макс. абсолютное значение
    max_abs = max(abs(v) for v in err_history) if err_history else 1.0
    if max_abs < 1e-6:
        max_abs = 1.0  # защита от деления на ноль

    n = len(err_history)
    step_x = gw / max(n - 1, 1)

    points: list[tuple[int, int]] = []
    for i, val in enumerate(err_history):
        x = gx + int(i * step_x)
        # val в [-max_abs, max_abs] → y в [gy, gy+gh]
        y = zero_y - int((val / max_abs) * (gh // 2))
        y = int(np.clip(y, gy + 1, gy + gh - 1))
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(screen, ERR_COLOR, False, points, 2)


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
