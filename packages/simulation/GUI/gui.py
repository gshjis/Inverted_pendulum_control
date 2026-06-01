"""
Pygame-визуализация перевёрнутого маятника.

Класс PendulumViewer
    - Принимает ObjectOfControl, SensorBlock, NoiseForce
    - Опционально Controller (если None — ручное управление ← →)
    - Метод use() запускает pygame-окно
"""

from __future__ import annotations

import sys

import numpy as np
import pygame

from packages.simulation.CO import (
    Controller,
    MeasuredState,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
    State,
)

# ═══════════════════════════════════════════════════════════════════════════
# Константы отрисовки
# ═══════════════════════════════════════════════════════════════════════════

BLACK = (10, 10, 10)
WHITE = (220, 220, 220)
RED = (255, 60, 60)
GREEN = (60, 255, 60)
GRAY = (100, 100, 100)
ORANGE = (255, 180, 30)

WIDTH, HEIGHT = 1200, 700
FPS = 60
SCALE = 200.0
CART_W = 60
CART_H = 30
WHEEL_R = 8
PEND_R = 6
TRACK_Y = 550
FORCE_SCALE = 3.0

PHYSICS_DT = 0.0005
SUBTICKS = int(round(1.0 / (FPS * PHYSICS_DT)))  # 33


# ═══════════════════════════════════════════════════════════════════════════
# PendulumViewer
# ═══════════════════════════════════════════════════════════════════════════

class PendulumViewer:
    """
    Pygame-визуализация перевёрнутого маятника на тележке.

    Parameters
    ----------
    plant : ObjectOfControl
        Объект управления (физика маятника).
    sensor : SensorBlock
        Измерительная подсистема (шум + квантование).
    noise : NoiseForce
        Внешнее возмущение.
    controller : Controller | None
        Регулятор. Если ``None`` — ручное управление стрелками.
    target_state : State | None
        Целевое состояние для вычисления ошибки (по умолч. маятник вверх).
    """

    def __init__(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        controller: Controller | None = None,
        target_state: State | None = None,
    ) -> None:
        self._plant = ObjectOfControl(plant_config)
        self._sensor = SensorBlock(sensor_config)
        self._noise = noise
        self._controller = controller
        self._target = target_state or State(x=0.0, theta1=np.pi, theta2=0.0)

        self._l1: float = 1.0  # длина звена (для отрисовки)

        # Pygame
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Consolas", 16, bold=True)

        title = "Перевёрнутый маятник"
        title += " — PID-регулятор" if controller else " — ручное управление"
        title += "  (Пробел сброс, Q / ESC выход)"
        pygame.display.set_caption(title)

    # ── Публичный метод ───────────────────────────────────────────────────

    def use(self) -> None:
        """Запустить главный цикл визуализации (блокирующий)."""
        running = True
        manual_force = 0.0
        force_per_frame = 80.0

        while running:
            # ── События ────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                running = False

            # ── Управление ─────────────────────────────────────────────
            if self._controller is not None:
                # PID-управление
                measured_arr = self._sensor.get_telemetry(
                    self._plant.q, self._plant.dq,
                )
                ms = MeasuredState(
                    x=measured_arr[0],
                    theta1=measured_arr[1],
                    theta2=measured_arr[2],
                    x_dot=measured_arr[3],
                    theta1_dot=measured_arr[4],
                    theta2_dot=measured_arr[5],
                )
                F = self._controller.compute_control(ms, self._target)
                viz_force = F
            else:
                # Ручное управление
                if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                    manual_force = -force_per_frame
                elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
                    manual_force = force_per_frame
                else:
                    manual_force = 0.0
                F = manual_force
                viz_force = manual_force

            # ── Сброс ──────────────────────────────────────────────────
            if keys[pygame.K_SPACE]:
                self._reset()

            # ── Физика ─────────────────────────────────────────────────
            for _ in range(SUBTICKS):
                self._plant.update_physics(F, self._noise, PHYSICS_DT)

            # ── Отрисовка ──────────────────────────────────────────────
            self._draw(viz_force)

        pygame.quit()
        sys.exit(0)

    # ── Сброс ─────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        """Сброс не поддерживается — пересоздайте PendulumViewer."""
        pass

    # ── Отрисовка ─────────────────────────────────────────────────────────

    def _draw(self, applied_force: float) -> None:
        self._screen.fill(BLACK)

        x = self._plant.q[0]
        th1 = self._plant.q[1]
        dx = self._plant.dq[0]
        dth1 = self._plant.dq[1]

        cart_x_px = int(WIDTH // 2 + x * SCALE)
        cart_y_px = TRACK_Y - CART_H // 2

        # Рельс
        pygame.draw.line(
            self._screen, GRAY, (0, TRACK_Y), (WIDTH, TRACK_Y), 2,
        )

        # Тележка
        pygame.draw.rect(
            self._screen, WHITE,
            (cart_x_px - CART_W // 2, cart_y_px - CART_H // 2, CART_W, CART_H),
            2,
        )
        for offset in (-CART_W // 4, CART_W // 4):
            pygame.draw.circle(
                self._screen, WHITE,
                (cart_x_px + offset, TRACK_Y + WHEEL_R), WHEEL_R, 2,
            )

        # Маятник
        pivot = (cart_x_px, cart_y_px - CART_H // 2)
        pend_x = pivot[0] + self._l1 * SCALE * np.sin(th1)
        pend_y = pivot[1] + self._l1 * SCALE * np.cos(th1)

        pygame.draw.line(self._screen, ORANGE, pivot, (pend_x, pend_y), 4)
        pygame.draw.circle(self._screen, RED, (int(pend_x), int(pend_y)), PEND_R)

        # Стрелка силы
        if abs(applied_force) > 0.5:
            arrow_len = float(np.clip(
                abs(applied_force) * FORCE_SCALE, 10, 150,
            ))
            direction = 1 if applied_force > 0 else -1
            start_x = cart_x_px
            end_x = cart_x_px + int(direction * arrow_len)
            color = GREEN if applied_force > 0 else RED
            pygame.draw.line(
                self._screen, color,
                (start_x, cart_y_px), (end_x, cart_y_px), 4,
            )
            tip = 10
            pygame.draw.line(
                self._screen, color,
                (end_x, cart_y_px),
                (end_x - direction * tip, cart_y_px - tip // 2), 3,
            )
            pygame.draw.line(
                self._screen, color,
                (end_x, cart_y_px),
                (end_x - direction * tip, cart_y_px + tip // 2), 3,
            )

        # HUD
        mode = "PID" if self._controller else "РУЧНОЕ"
        gains_str = ""
        if self._controller:
            if hasattr(self._controller, "gains"):
                g = self._controller.gains  # type: ignore[attr-defined]
                gains_str = f"  Kp={g[0]:.1f}  Ki={g[1]:.1f}  Kd={g[2]:.1f}  Kx={g[3]:.1f}"
        lines = [
            f"Сила: {applied_force:+.1f} Н  [{mode}]",
            f"x  = {x:+.3f} м      θ = {np.degrees(th1):+7.1f}°",
            f"ẋ  = {dx:+.3f} м/с   θ̇ = {np.degrees(dth1):+7.1f}°/с",
        ]
        if gains_str:
            lines.append(gains_str)

        for i, line in enumerate(lines):
            surf = self._font.render(line, True, GREEN)
            self._screen.blit(surf, (20, 20 + i * 22))

        hint = self._font.render(
            "Пробел : сброс | Q / ESC : выход",
            True, GRAY,
        )
        self._screen.blit(hint, (20, HEIGHT - 40))

        pygame.display.flip()
        self._clock.tick(FPS)
