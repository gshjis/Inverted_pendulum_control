"""
Pygame-визуализация перевёрнутого маятника.

Класс PendulumViewer
    - Принимает ObjectOfControl, SensorBlock, NoiseForce
    - Опционально Controller (если None — ручное управление ← →)
    - Метод use() запускает pygame-окно
"""

from __future__ import annotations

import sys
import os
from typing import Any, Callable

import numpy as np
import pygame

from packages.simulation.CO import (
    Controller,
    MeasuredState,
    MotorInertia,
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
SUBTICKS = int(PHYSICS_DT/0.00005)


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
        plant: ObjectOfControl,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: MeasuredState,
        controller: Controller | None = None,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
    ) -> None:
        self._plant = plant
        self._init_q = plant.q.copy()
        self._init_dq = plant.dq.copy()
        self._sensor = SensorBlock(sensor_config)
        self._noise = noise
        self._controller = controller
        self._target = target_state or MeasuredState(x=0.0, theta1=np.pi, theta2=0.0)
        self._terminate_condition = terminate_condition

        self._motor_inertia = MotorInertia(time_constant=0.1) if controller is None else None
        self._terminated = False
        self._elapsed_when_terminated: int | None = None

        # Pygame
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Consolas", 16, bold=True)

        title = "Перевёрнутый маятник"
        title += f" — {controller.name}" if controller else " — ручное управление"
        title += "  (Пробел сброс, Q / ESC выход)"
        pygame.display.set_caption(title)
        # Запись видео — папка по умолчанию: корень проекта (абсолютный путь)
        self._recording: bool = False
        self._record_dir: str | None = os.path.abspath(".")
        self._frame_index: int = 0
        # Флаг — нужно собрать видео после завершения записи
        self._need_compile: bool = False

    # ── Публичный метод ───────────────────────────────────────────────────

    def use(self) -> None:
        """Запустить главный цикл визуализации (блокирующий)."""
        # Перед стартом показать GUI-диалог для подтверждения записи (в окне pygame)
        def _ask_recording() -> bool:
            # Отрисовать простое окно с текстом и ожидать Y/N клавишу или клик по кнопкам
            asking = True
            choice = False
            font = pygame.font.SysFont("Consolas", 20, bold=True)
            # Прямоугольник диалога
            dialog_w, dialog_h = 520, 140
            dialog_rect = pygame.Rect((WIDTH - dialog_w) // 2, (HEIGHT - dialog_h) // 2, dialog_w, dialog_h)
            btn_yes = pygame.Rect(dialog_rect.right - 180, dialog_rect.bottom - 50, 70, 32)
            btn_no = pygame.Rect(dialog_rect.right - 90, dialog_rect.bottom - 50, 70, 32)

            while asking:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_y:
                            return True
                        if event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                            return False
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        if btn_yes.collidepoint(mx, my):
                            return True
                        if btn_no.collidepoint(mx, my):
                            return False

                # Рисуем диалог
                self._screen.fill(BLACK)
                pygame.draw.rect(self._screen, (30, 30, 30), dialog_rect)
                txt = font.render("Record simulation to video?", True, GREEN)
                self._screen.blit(txt, (dialog_rect.x + 20, dialog_rect.y + 20))
                hint = font.render("Press Y / N or click a button", True, GRAY)
                self._screen.blit(hint, (dialog_rect.x + 20, dialog_rect.y + 60))

                pygame.draw.rect(self._screen, (50, 120, 50), btn_yes)
                pygame.draw.rect(self._screen, (120, 50, 50), btn_no)
                yes_s = font.render("Yes", True, WHITE)
                no_s = font.render("No", True, WHITE)
                self._screen.blit(yes_s, (btn_yes.x + 18, btn_yes.y + 6))
                self._screen.blit(no_s, (btn_no.x + 22, btn_no.y + 6))

                pygame.display.flip()
                self._clock.tick(FPS)

            return choice

        if _ask_recording():
            import tempfile

            d = tempfile.mkdtemp(prefix="pendulum_rec_", dir=self._record_dir)
            self._record_dir = d
            self._frame_index = 0
            self._recording = True
            self._need_compile = True
        else:
            self._recording = False
        running = True
        manual_force = 0.0
        force_per_frame = 20.0
        # Засечь момент старта (ms)
        self._start_ticks = pygame.time.get_ticks()

        # Таймеры для управления/физики
        control_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
        control_acc_ms = 0
        # Для сохранения кадров синхронизированно с шагом симуляции
        last_save_acc_ms = 0

        prev_ticks = pygame.time.get_ticks()

        while running:
            # ── События ────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    # Кнопка записи: правая верхняя область
                    btn_rect = (WIDTH - 120, 10, 110, 28)
                    if btn_rect[0] <= mx <= btn_rect[0] + btn_rect[2] and btn_rect[1] <= my <= btn_rect[1] + btn_rect[3]:
                        # переключить запись
                        self._recording = not self._recording
                        if self._recording:
                            import tempfile

                            # Создать директорию для кадров внутри корня проекта по умолчанию
                            # (self._record_dir уже содержит абсолютный путь к корню проекта)
                            d = tempfile.mkdtemp(prefix="pendulum_rec_", dir=self._record_dir)
                            self._record_dir = d
                            self._frame_index = 0
                            self._need_compile = False
                        else:
                            # при остановке записи — сначала сохранить дополнительный финальный кадр,
                            # затем собрать видео из накопленных кадров
                            try:
                                if self._record_dir is not None:
                                    # сохранить финальный кадр, если возможно
                                    fname = f"frame_{self._frame_index:06d}.png"
                                    path = os.path.join(self._record_dir, fname)
                                    try:
                                        pygame.image.save(self._screen, path)
                                        self._frame_index += 1
                                    except Exception:
                                        pass

                                    if self._frame_index > 0:
                                        # пометить запрос на компиляцию — фактическая сборка выполнится после цикла
                                        self._need_compile = True
                            except Exception:
                                # не фатально — пропускаем ошибки сохранения/сборки
                                pass
                            # Не удаляем self._record_dir здесь — сброс произойдёт после основного цикла

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                running = False
            # ── Вычислить прошедшее время с предыдущего кадра
            now = pygame.time.get_ticks()
            dt_ms = now - prev_ticks
            prev_ticks = now
            control_acc_ms += dt_ms

            # По накоплению времени контроля выполнить шаг управления и соответствующее число шагов физики
            F = 0.0
            if not self._terminated:
                if control_acc_ms >= control_interval_ms:
                    # сколько контрольных шагов провести (обычно 1)
                    steps_ctrl = max(1, control_acc_ms // control_interval_ms)
                    control_acc_ms = control_acc_ms % control_interval_ms

                    # Для каждого контрольного шага: получить телеметрию, вычислить управление и интегрировать физику
                    F = 0.0
                    for _ in range(int(steps_ctrl)):
                        if self._controller is not None:
                            ms = self._sensor.get_telemetry(self._plant.q, self._plant.dq)
                            F = self._controller.compute_control(ms, self._target)
                        else:
                            # ручное — читаем текущие клавиши и применяем инерционность
                            if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                                manual_force = -force_per_frame
                            elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
                                manual_force = force_per_frame
                            else:
                                manual_force = 0.0
                            F = self._motor_inertia.update(manual_force, control_interval_ms / 1000.0) if self._motor_inertia else manual_force

                        # выполнить физику с мелким шагом PHYSICS_DT в течение control interval
                        steps_ph = int(round((control_interval_ms / 1000.0) / PHYSICS_DT))
                        for __ in range(max(1, steps_ph)):
                            self._plant.update_physics(F, self._noise, PHYSICS_DT)

                    viz_force = F
                else:
                    # между управляющими тактами — визуализация без обновления управления/физики
                    viz_force = 0.0
            else:
                F = 0.0
                viz_force = 0.0

            # ── Сброс ──────────────────────────────────────────────────
            if keys[pygame.K_SPACE]:
                self._reset()

            # ── Физика ─────────────────────────────────────────────────
            if not self._terminated:
                for _ in range(SUBTICKS):
                    self._plant.update_physics(F, self._noise, PHYSICS_DT)

                # Проверка терминального состояния
                if self._terminate_condition is not None and self._terminate_condition(self._plant):
                    self._terminated = True
                    # Зафиксировать прошедшее время при остановке
                    self._elapsed_when_terminated = pygame.time.get_ticks() - self._start_ticks

        # ── Отрисовка ──────────────────────────────────────────────
            self._draw(viz_force)

            # Сохранение кадра при записи (каждый кадр цикла)
            # Сохраняем кадры синхронно с управляющим тактом (чтобы в видео fps == симуляции)
            if self._recording and self._record_dir is not None:
                # аккумулировать прошедшее время для сохранения
                last_save_acc_ms += dt_ms
                try:
                    sim_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
                except Exception:
                    sim_interval_ms = int(1.0 / FPS * 1000.0)

                # сохранять один кадр на каждый управляющий такт
                if last_save_acc_ms >= sim_interval_ms:
                    last_save_acc_ms = last_save_acc_ms % sim_interval_ms
                    fname = f"frame_{self._frame_index:06d}.png"
                    path = os.path.join(self._record_dir, fname)
                    try:
                        pygame.image.save(self._screen, path)
                        self._frame_index += 1
                    except Exception:
                        # если не удалось сохранить кадр, продолжаем без прерывания
                        pass

            # Если запись была остановлена и помечена для компиляции — выполнить сборку
            if not self._recording and self._need_compile and self._record_dir is not None:
                try:
                    try:
                        sim_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
                    except Exception:
                        sim_interval_ms = int(1.0 / FPS * 1000.0)
                    # Вычислить фактическую частоту кадров: frames / simulated_seconds
                    try:
                        import glob

                        frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
                        n_frames = len(frames)
                        # длительность симуляции в секундах — от старта до текущего момента или до остановки
                        if self._elapsed_when_terminated is not None:
                            sim_seconds = self._elapsed_when_terminated / 1000.0
                        else:
                            sim_seconds = (pygame.time.get_ticks() - self._start_ticks) / 1000.0
                        if sim_seconds > 0 and n_frames > 0:
                            sim_fps = max(1, round(n_frames / sim_seconds))
                        else:
                            sim_fps = max(1, round(1000.0 / sim_interval_ms))
                    except Exception:
                        sim_fps = max(1, round(1000.0 / sim_interval_ms))
                    self._compile_video(self._record_dir, sim_fps)
                finally:
                    # сброс состояния
                    self._need_compile = False
                    self._record_dir = None

        # Перед выходом: если были накоплены кадры, спросить сохранить ли видео
        if self._record_dir is not None:
            # подсчитать кадры
            try:
                import glob

                frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
            except Exception:
                frames = []
            if frames:
                # Показать GUI-диалог для подтверждения сохранения видео
                def _ask_save_video(n_frames: int) -> bool:
                    asking = True
                    font = pygame.font.SysFont("Consolas", 18, bold=True)
                    dialog_w, dialog_h = 560, 140
                    dialog_rect = pygame.Rect((WIDTH - dialog_w) // 2, (HEIGHT - dialog_h) // 2, dialog_w, dialog_h)
                    btn_yes = pygame.Rect(dialog_rect.right - 180, dialog_rect.bottom - 50, 70, 32)
                    btn_no = pygame.Rect(dialog_rect.right - 90, dialog_rect.bottom - 50, 70, 32)

                    while asking:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_y:
                                    return True
                                if event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                                    return False
                            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                                mx, my = event.pos
                                if btn_yes.collidepoint(mx, my):
                                    return True
                                if btn_no.collidepoint(mx, my):
                                    return False

                        # Рисуем диалог
                        self._screen.fill(BLACK)
                        pygame.draw.rect(self._screen, (30, 30, 30), dialog_rect)
                        txt = font.render(f"Save recorded video from {n_frames} frames?", True, GREEN)
                        self._screen.blit(txt, (dialog_rect.x + 20, dialog_rect.y + 20))
                        hint = font.render("Press Y / N or click a button", True, GRAY)
                        self._screen.blit(hint, (dialog_rect.x + 20, dialog_rect.y + 60))

                        pygame.draw.rect(self._screen, (50, 120, 50), btn_yes)
                        pygame.draw.rect(self._screen, (120, 50, 50), btn_no)
                        yes_s = font.render("Yes", True, WHITE)
                        no_s = font.render("No", True, WHITE)
                        self._screen.blit(yes_s, (btn_yes.x + 18, btn_yes.y + 6))
                        self._screen.blit(no_s, (btn_no.x + 22, btn_no.y + 6))

                        pygame.display.flip()
                        self._clock.tick(FPS)

                    return False

                try:
                    if _ask_save_video(len(frames)):
                        # скомпилировать и затем удалить кадры, оставив только ролик
                        try:
                            try:
                                sim_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
                            except Exception:
                                sim_interval_ms = int(1.0 / FPS * 1000.0)
                            try:
                                import glob

                                frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
                                n_frames = len(frames)
                                if self._elapsed_when_terminated is not None:
                                    sim_seconds = self._elapsed_when_terminated / 1000.0
                                else:
                                    sim_seconds = (pygame.time.get_ticks() - self._start_ticks) / 1000.0
                                if sim_seconds > 0 and n_frames > 0:
                                    sim_fps = max(1, round(n_frames / sim_seconds))
                                else:
                                    sim_fps = max(1, round(1000.0 / sim_interval_ms))
                            except Exception:
                                sim_fps = max(1, round(1000.0 / sim_interval_ms))
                            self._compile_video(self._record_dir, sim_fps)
                            # удалить png-файлы
                            for p in frames:
                                try:
                                    os.remove(p)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

        pygame.quit()
        sys.exit(0)

    # ── Сброс ─────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        """Сброс состояния симуляции."""
        self._plant._q = self._init_q.copy()
        self._plant._dq = self._init_dq.copy()
        if self._controller:
            self._controller.reset()
        if self._motor_inertia:
            self._motor_inertia.reset()
        self._terminated = False
        # Сброс времени старта
        self._start_ticks = pygame.time.get_ticks()
        self._elapsed_when_terminated = None

    def _compile_video(self, record_dir: str, fps: int) -> None:
        """Собрать видео из кадров с помощью ffmpeg. Выводит абсолютный путь к результату."""
        try:
            import subprocess
            import glob

            # Отладочный вывод: сколько файлов до сборки
            frames_before = sorted(glob.glob(os.path.join(record_dir, "frame_*.png")))
            print(f"Compiling video from {len(frames_before)} frames in {os.path.abspath(record_dir)}")

            # В ffmpeg предпочтительнее указывать -framerate перед -i
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i",
                os.path.join(record_dir, "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "output.mp4",
            ]
            subprocess.run(cmd, check=True, cwd=record_dir)
            out_mp4 = os.path.join(record_dir, "output.mp4")
            print(f"Video saved: {os.path.abspath(out_mp4)}")
            # Отладочный вывод: перечислить файлы после сборки (кадры должны остаться)
            frames_after = sorted(glob.glob(os.path.join(record_dir, "frame_*.png")))
            print(f"Frames after compile: {len(frames_after)} (sample: {frames_after[:3]})")
        except Exception:
            # Не мешаем работе симуляции при ошибке сборки
            pass

    # ── Отрисовка ─────────────────────────────────────────────────────────

    def _draw(self, applied_force: float) -> None:
        self._screen.fill(BLACK)

        # Получение состояний из объекта управления
        q = self._plant.q
        dq = self._plant.dq
        
        # Определение количества звеньев
        is_single = self._plant.single_pendulum_mode
        
        x = q[0]
        th1 = q[1]
        th2 = q[2] if not is_single else 0.0
        
        dx = dq[0]
        dth1 = dq[1]
        dth2 = dq[2] if not is_single else 0.0

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

        # Маятник (звено 1)
        pivot1 = (cart_x_px, cart_y_px - CART_H // 2)
        pend1_x = pivot1[0] + 1.0 * SCALE * np.sin(th1)
        pend1_y = pivot1[1] + 1.0 * SCALE * np.cos(th1)

        pygame.draw.line(self._screen, ORANGE, pivot1, (pend1_x, pend1_y), 4)
        pygame.draw.circle(self._screen, RED, (int(pend1_x), int(pend1_y)), PEND_R)

        # Маятник (звено 2, если есть)
        if not is_single:
            pivot2 = (pend1_x, pend1_y)
            pend2_x = pivot2[0] + 1.0 * SCALE * np.sin(th1 + th2)
            pend2_y = pivot2[1] + 1.0 * SCALE * np.cos(th1 + th2)
            pygame.draw.line(self._screen, ORANGE, pivot2, (pend2_x, pend2_y), 4)
            pygame.draw.circle(self._screen, RED, (int(pend2_x), int(pend2_y)), PEND_R)

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
            f"x  = {x:+.3f} м      θ1 = {np.degrees(th1):+7.1f}°",
        ]
        if not is_single:
            lines.append(f"θ2 = {np.degrees(th2):+7.1f}°")
        
        lines.append(f"ẋ  = {dx:+.3f} м/с   θ̇1 = {np.degrees(dth1):+7.1f}°/с")
        if not is_single:
            lines.append(f"θ̇2 = {np.degrees(dth2):+7.1f}°/с")

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

        # Вывести прошедшее время с старта
        if hasattr(self, '_start_ticks'):
            if self._terminated and self._elapsed_when_terminated is not None:
                elapsed_ms = self._elapsed_when_terminated
            else:
                elapsed_ms = pygame.time.get_ticks() - self._start_ticks
            elapsed_s = elapsed_ms / 1000.0
            time_s = f"Время: {elapsed_s:.2f} с"
            time_surf = self._font.render(time_s, True, GREEN)
            self._screen.blit(time_surf, (WIDTH - 220, 20))

        if self._terminated:
            term_surf = self._font.render("СИМУЛЯЦИЯ ОСТАНОВЛЕНА (Пробел - рестарт)", True, RED)
            self._screen.blit(term_surf, (WIDTH // 2 - term_surf.get_width() // 2, HEIGHT // 2))

        # Кнопка записи (правый верхний угол)
        rec_text = "REC" if self._recording else "REC"
        rec_color = RED if self._recording else GRAY
        btn_rect = (WIDTH - 120, 10, 110, 28)
        pygame.draw.rect(self._screen, (40, 40, 40), btn_rect)
        rec_surf = self._font.render(f"[{rec_text}] Record", True, rec_color)
        self._screen.blit(rec_surf, (btn_rect[0] + 8, btn_rect[1] + 6))

        pygame.display.flip()
        self._clock.tick(FPS)
