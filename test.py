"""
Симуляция перевёрнутого маятника на тележке с ручным управлением.

Использует готовые классы из пакета ``packages.simulation.CO``:
    - ObjectOfControl  — физика (уравнения Лагранжа + RK4)
    - NoiseForce       — датакласс внешнего возмущения

Масштаб времени: 1:1 (1 с симуляции = 1 с реального времени).
Физика интегрируется с микрошагом 0.5 мс, рендер — 60 FPS.

Управление:
    ← →  — приложить силу к тележке (влево / вправо)
    Пробел — сбросить симуляцию (маятник вверх, тележка по центру)
    Q / ESC — выход
"""

from __future__ import annotations

import sys

import numpy as np
import pygame

from packages.simulation.CO import NoiseForce, ObjectOfControl


# ═══════════════════════════════════════════════════════════════════════════
# Визуализация (pygame)
# ═══════════════════════════════════════════════════════════════════════════

# Цвета (R, G, B)
BLACK = (10, 10, 10)
WHITE = (220, 220, 220)
RED = (255, 60, 60)
GREEN = (60, 255, 60)
GRAY = (100, 100, 100)
ORANGE = (255, 180, 30)

# Размеры окна
WIDTH, HEIGHT = 1200, 700
FPS = 60

# Параметры рендера
SCALE = 200.0          # пикселей на метр
CART_W = 60            # ширина тележки (px)
CART_H = 30            # высота тележки (px)
WHEEL_R = 8            # радиус колеса (px)
PEND_R = 6             # радиус грузика маятника (px)
TRACK_Y = 550          # y-координата рельса
FORCE_SCALE = 3.0      # масштаб стрелки силы

# Физика — масштаб 1:1 с реальным временем
# За один кадр (1/FPS ≈ 16.7 мс) делаем столько микрошагов,
# чтобы симуляция не отставала и не убегала от реального времени.
PHYSICS_DT = 0.0005     # микрошаг физики (с) — 0.5 мс
SUBTICKS = int(round(1.0 / (FPS * PHYSICS_DT)))  # 33 шага на кадр

# Геометрия маятника (для отрисовки — берём из конфига ОУ)
L1 = 1.0               # длина первого звена (м)


class PendulumViewer:
    """Отрисовка маятника на тележке средствами pygame."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(
            "Перевёрнутый маятник — ручное управление  "
            "(← → сила, Пробел сброс, Q выход)"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16, bold=True)

    def draw(
        self,
        plant: ObjectOfControl,
        applied_force: float,
    ) -> None:
        """Отрисовать один кадр."""
        self.screen.fill(BLACK)

        x = plant.q[0]
        th1 = plant.q[1]
        dx = plant.dq[0]
        dth1 = plant.dq[1]

        cart_x_px = int(WIDTH // 2 + x * SCALE)
        cart_y_px = TRACK_Y - CART_H // 2

        # ── Рельс ─────────────────────────────────────────────────────
        pygame.draw.line(
            self.screen, GRAY,
            (0, TRACK_Y), (WIDTH, TRACK_Y), 2,
        )

        # ── Тележка ───────────────────────────────────────────────────
        # Корпус
        pygame.draw.rect(
            self.screen, WHITE,
            (cart_x_px - CART_W // 2, cart_y_px - CART_H // 2, CART_W, CART_H),
            2,
        )
        # Колёса
        for offset in (-CART_W // 4, CART_W // 4):
            pygame.draw.circle(
                self.screen, WHITE,
                (cart_x_px + offset, TRACK_Y + WHEEL_R), WHEEL_R, 2,
            )

        # ── Маятник ───────────────────────────────────────────────────
        pivot = (cart_x_px, cart_y_px - CART_H // 2)

        pend_x = pivot[0] + L1 * SCALE * np.sin(th1)
        pend_y = pivot[1] + L1 * SCALE * np.cos(th1)

        pygame.draw.line(self.screen, ORANGE, pivot, (pend_x, pend_y), 4)
        pygame.draw.circle(self.screen, RED, (int(pend_x), int(pend_y)), PEND_R)

        # ── Стрелка силы ──────────────────────────────────────────────
        if abs(applied_force) > 0.5:
            arrow_len = float(np.clip(
                abs(applied_force) * FORCE_SCALE, 10, 150,
            ))
            direction = 1 if applied_force > 0 else -1
            start_x = cart_x_px
            end_x = cart_x_px + int(direction * arrow_len)
            color = GREEN if applied_force > 0 else RED
            pygame.draw.line(
                self.screen, color,
                (start_x, cart_y_px), (end_x, cart_y_px), 4,
            )
            tip = 10
            pygame.draw.line(
                self.screen, color,
                (end_x, cart_y_px),
                (end_x - direction * tip, cart_y_px - tip // 2), 3,
            )
            pygame.draw.line(
                self.screen, color,
                (end_x, cart_y_px),
                (end_x - direction * tip, cart_y_px + tip // 2), 3,
            )

        # ── Информация ────────────────────────────────────────────────
        lines = [
            f"Сила: {applied_force:+.1f} Н",
            f"x  = {x:+.3f} м      θ = {np.degrees(th1):+7.1f}°",
            f"ẋ  = {dx:+.3f} м/с   θ̇ = {np.degrees(dth1):+7.1f}°/с",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, GREEN)
            self.screen.blit(surf, (20, 20 + i * 22))

        # Подсказки
        hints = [
            "← → : сила на тележку",
            "Пробел : сброс",
            "Q / ESC : выход",
        ]
        for i, hint in enumerate(hints):
            surf = self.font.render(hint, True, GRAY)
            self.screen.blit(surf, (20, HEIGHT - 80 + i * 22))

        pygame.display.flip()
        self.clock.tick(FPS)


# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация физики ОУ
# ═══════════════════════════════════════════════════════════════════════════

PLANT_CONFIG = {
    # Массы
    "M": 1.0,      # масса тележки, кг
    "m1": 0.3,     # масса маятника, кг
    "m2": 0.0,     # второе звено отключено
    # Длины
    "l1": L1,      # длина маятника, м
    "l2": 0.0,
    "L1": 0.7,     # расстояние до ЦМ маятника, м
    "L2": 0.0,
    # Инерция
    "J1": 0.02,    # момент инерции маятника, кг·м²
    "J2": 0.0,
    # Демпфирование
    "b_c": 0.05,   # вязкое трение тележки
    "b_1": 0.05,  # вязкое трение в шарнире
    "b_2": 0.0,
    # Режимы
    "single_pendulum_mode": True,
    "backslash_mode": False,
    # Начальное состояние
    "init_q": [0.0, np.pi, 0.0],
    "init_dq": [0.0, 0.0, 0.0],
}

MAX_FORCE = 10.0   # макс. сила мотора, Н
FORCE_PER_FRAME = 30.0  # Н при полном нажатии клавиши


# ═══════════════════════════════════════════════════════════════════════════
# Главный цикл
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    plant = ObjectOfControl(PLANT_CONFIG)
    viewer = PendulumViewer()

    applied_force = 0.0
    F_noise = NoiseForce(value=0.0001)  # без внешнего возмущения
    running = True

    while running:
        # ── Обработка событий ─────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            running = False

        # Ручное управление
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            applied_force = -FORCE_PER_FRAME
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            applied_force = FORCE_PER_FRAME
        else:
            applied_force = 0.0

        # Сброс
        if keys[pygame.K_SPACE]:
            plant = ObjectOfControl(PLANT_CONFIG)
            applied_force = 0.0

        # ── Шаг физики (субдискретизация, масштаб 1:1) ────────────────
        F_clipped = float(np.clip(applied_force, -MAX_FORCE, MAX_FORCE))
        for _ in range(SUBTICKS):
            plant.update_physics(F_clipped, F_noise, PHYSICS_DT)

        # ── Отрисовка ─────────────────────────────────────────────────
        viewer.draw(plant, applied_force)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
