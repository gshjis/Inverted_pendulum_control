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
    MotorInertia,
    NoiseForce,
    ObjectOfControl,
    SensorBlock,
    SensorConfig,
)
from .constants import *
from .dialogs import ask_recording, ask_save_video
from .dialogs import ask_recording, ask_save_video, ask_input_target
from .recorder import compile_video
from .draw import draw_cart, draw_pendulums, draw_force_arrow, draw_hud, draw_record_button, draw_target_marker, draw_controller_button
from .event_controller import EventController
from .physics_runner import PhysicsRunner

# Константы импортируются напрямую из .constants, не следует переприсваивать их


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
        target_state: np.ndarray,
        controller: Controller | None = None,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
    ) -> None:
        self._plant = plant
        self._init_q = plant.q.copy()
        self._init_dq = plant.dq.copy()
        self._sensor = SensorBlock(sensor_config)
        self._noise = noise
        self._controller = controller
        self._target = np.array([0.0, np.pi, 0])
        self._terminate_condition = terminate_condition

        self._motor_inertia = MotorInertia(time_constant=0.1) if controller is None else None
        self._terminated = False
        self._elapsed_when_terminated: int | None = None

        # Pygame
        pygame.init()
        # Открыть окно в оконном режиме (возвращено по запросу)
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Consolas", 16, bold=True)

        title = "Перевёрнутый маятник"
        title += f" — {controller.name}" if controller else " — ручное управление"
        title += "  (Пробел сброс, Q / ESC выход)"
        pygame.display.set_caption(title)
        # Запись видео — папка по умолчанию: корень проекта (абсолютный путь)
        self._recording: bool = False
        self._record_dir: str = os.path.abspath(".")
        self._frame_index: int = 0
        # Флаг — нужно собрать видео после завершения записи
        self._need_compile: bool = False

        # Подсистемы (разделение ответственности)
        self._event_controller = EventController(self._screen, self._clock)
        self._physics_runner = PhysicsRunner(self._plant)
        from .renderer import Renderer
        from .recorder_obj import Recorder

        self._renderer = Renderer(self._screen, self._font, self._controller)
        # Используем единый интерфейс записи через объект Recorder
        self._recorder = Recorder(self._record_dir)
        # Target marker state (only shown in manual mode)
        from .constants import MARKER_COLOR, MARKER_HOVER_COLOR, MARKER_SPEED, MARKER_THROTTLE_MS, MARKER_MIN_X, MARKER_MAX_X, MARKER_W, MARKER_H
        self._marker_color = MARKER_COLOR
        self._marker_hover_color = MARKER_HOVER_COLOR
        self._marker_speed = MARKER_SPEED
        self._marker_throttle_ms = MARKER_THROTTLE_MS
        self._marker_min_x = MARKER_MIN_X
        self._marker_max_x = MARKER_MAX_X
        self._marker_w = MARKER_W
        self._marker_h = MARKER_H
        # Marker logical position (meters) initialized from target_state
        self._marker_x = float(self._target[0])
        self._marker_dragging = False
        self._last_marker_update_ms = 0
        # Controller enabled flag (visible button)
        self._controller_enabled = True if self._controller is not None else False

    # ── Публичный метод ───────────────────────────────────────────────────

    def use(self) -> None:
        """Запустить главный цикл визуализации (блокирующий)."""
        # Перед стартом показать GUI-диалог для подтверждения записи (в окне pygame)
        if ask_recording(self._screen, self._clock):
            try:
                self._recorder.start()
                self._record_dir = self._recorder.record_dir
                self._frame_index = self._recorder.frame_index
                self._recording = self._recorder.recording
                self._need_compile = self._recorder.need_compile
            except Exception:
                self._recording = False
        else:
            self._recording = False
        running = True
        manual_force = 0.0
        # force_per_frame вынесена в constants
        from .constants import FORCE_PER_FRAME
        force_per_frame = FORCE_PER_FRAME
        # Засечь момент старта (ms)
        self._start_ticks = pygame.time.get_ticks()

        # Таймеры для управления/физики
        control_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
        control_acc_ms = 0
        # Для сохранения кадров синхронизированно с шагом симуляции
        last_save_acc_ms = 0

        prev_ticks = pygame.time.get_ticks()

        while running:
            # ── События (сейчас обрабатываются EventController)
            actions = self._event_controller.poll()
            # Обработка стрелок через события KEYDOWN (надёжнее чем get_pressed)
            # raw events доступны в actions['events']
            # Поддержка непрерывного удержания клавиш: используем get_pressed
            if self._controller is not None:
                keys = pygame.key.get_pressed()
                changed = False
                if keys[pygame.K_LEFT]:
                    if self._marker_min_x is None:
                        self._marker_x -= 0.05
                    else:
                        self._marker_x = max(self._marker_min_x, self._marker_x - 0.05)
                    changed = True
                if keys[pygame.K_RIGHT]:
                    if self._marker_max_x is None:
                        self._marker_x += 0.05
                    else:
                        self._marker_x = min(self._marker_max_x, self._marker_x + 0.05)
                    changed = True

                if changed:
                    self._target.x = float(self._marker_x)
                    self._last_marker_update_ms = pygame.time.get_ticks()
            # Сохранить сырые события для обработки маркера и других модалей
            self._last_events = actions.get("events", [])
            if not actions.get("running", True):
                running = False
                if actions.get("quit", False):
                    break

            if actions.get("toggle_record", False):
                # переключить запись
                self._toggle_recording()
            if actions.get("toggle_controller", False):
                # toggle controller via key 'c'
                self._controller_enabled = not getattr(self, '_controller_enabled', True)
                if not self._controller_enabled:
                    self._controller_backup = self._controller
                    if self._controller is not None and hasattr(self._controller, 'reset'):
                        try:
                            self._controller.reset()
                        except Exception:
                            pass
                    self._controller = None
                else:
                    self._controller = getattr(self, '_controller_backup', self._controller)
                    if self._controller is not None and hasattr(self._controller, 'reset'):
                        try:
                            self._controller.reset()
                        except Exception:
                            pass
            # переключить контроллер по клику на кнопке (верхняя панель)
            mx_my = actions.get("mouse_pos")
            if mx_my:
                mx, my = mx_my
                # кнопка теперь рисуется ближе к правому краю (WIDTH - 150)
                ctrl_btn = (WIDTH - 150, 10, 120, 30)
                if ctrl_btn[0] <= mx <= ctrl_btn[0] + ctrl_btn[2] and ctrl_btn[1] <= my <= ctrl_btn[1] + ctrl_btn[3]:
                    # toggle controller enabled flag
                    self._controller_enabled = not getattr(self, '_controller_enabled', True)
                    # debug log for click
                    try:
                        print(f"[GUI] Controller button clicked -> enabled={self._controller_enabled}")
                    except Exception:
                        pass
                    if not self._controller_enabled:
                        # disable controller: back it up, reset internals and set to None
                        self._controller_backup = self._controller
                        if self._controller is not None and hasattr(self._controller, 'reset'):
                            try:
                                self._controller.reset()
                            except Exception:
                                pass
                        self._controller = None
                    else:
                        # restore and reset internals
                        self._controller = getattr(self, '_controller_backup', self._controller)
                        if self._controller is not None and hasattr(self._controller, 'reset'):
                            try:
                                self._controller.reset()
                            except Exception:
                                pass

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
                        self._physics_runner.step(F, self._noise, PHYSICS_DT, steps_ph)

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
                    self._plant.update_physics(F, self._noise)

                # Проверка терминального состояния

                if self._terminate_condition is not None and self._terminate_condition(self._plant):
                    self._terminated = True
                    # Зафиксировать прошедшее время при остановке
                    self._elapsed_when_terminated = pygame.time.get_ticks() - self._start_ticks

            # ── Отрисовка ──────────────────────────────────────────────
            self._draw(viz_force)
            # Обработка ввода мыши для маркера цели
            self._handle_marker_events()

            # Сохранение кадра при записи (каждый кадр цикла)
            # Сохраняем кадры синхронно с управляющим тактом (чтобы в видео fps == симуляции)
            if self._recording:
                # аккумулировать прошедшее время для сохранения
                last_save_acc_ms += dt_ms
                # делегируем сохранение кадра рекордеру
                last_save_acc_ms = self._save_frame_if_recording(last_save_acc_ms)

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

                    try:
                        # используем Recorder как единый интерфейс
                        self._recorder.compile(sim_fps)
                    except Exception:
                        # не фатально — пропускаем ошибки сборки
                        pass
                finally:
                    # сброс состояния
                    self._need_compile = False

        # Перед выходом: если были накоплены кадры, спросить сохранить ли видео
        if self._record_dir is not None:
            # подсчитать кадры
            try:
                import glob

                if self._record_dir is None:
                    frames = []
                else:
                    frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
            except Exception:
                frames = []
            if frames:
                # Показать GUI-диалог для подтверждения сохранения видео
                if ask_save_video(self._screen, self._clock, len(frames)):
                    try:
                        try:
                            sim_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
                        except Exception:
                            sim_interval_ms = int(1.0 / FPS * 1000.0)
                        try:
                            sim_fps = self._compute_sim_fps(sim_interval_ms)
                        except Exception:
                            sim_fps = max(1, round(1000.0 / sim_interval_ms))

                        try:
                            if hasattr(self, '_recorder') and self._recorder is not None:
                                self._recorder.compile(sim_fps)
                            else:
                                compile_video(self._record_dir, sim_fps)
                        except Exception:
                            pass

                        # удалить png-файлы
                        for p in frames:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    except Exception:
                        pass

        pygame.quit()
        sys.exit(0)

    def _handle_marker_events(self) -> None:
        """Обрабатывать события мыши для перетаскивания маркера и обновлять target_state.x.

        Использует события, собранные EventController.poll() и сохранённые в
        self._last_events (чтобы избежать двойного чтения очереди событий).
        """
        events = getattr(self, '_last_events', [])
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # проверим попадание в маркер по пиксельной позиции
                cart_x_px = int(WIDTH // 2 + self._plant.q[0] * SCALE)
                cart_y_px = TRACK_Y - CART_H // 2
                rect = pygame.Rect(cart_x_px - self._marker_w // 2, cart_y_px - self._marker_h, self._marker_w, self._marker_h)
                if rect.collidepoint(mx, my) and self._controller is not None:
                    self._marker_dragging = True
                    self._drag_offset_x = mx - rect.x
                # кнопка ввода координаты при правом клике
                elif rect.collidepoint(mx, my) and event.button == 3 and self._controller is not None:
                    # вызвать диалог ввода
                    val = ask_input_target(self._screen, self._clock, self._marker_x, self._marker_min_x, self._marker_max_x)
                    if val is not None:
                        self._marker_x = val
                        self._target.x = float(val)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if getattr(self, '_marker_dragging', False):
                    self._marker_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if getattr(self, '_marker_dragging', False):
                    mx, my = event.pos
                    # вычислить новую x в метрах, учитывая ограничение скорости
                    new_px = mx - getattr(self, '_drag_offset_x', self._marker_w // 2)
                    new_x = (new_px - WIDTH // 2) / SCALE
                    # ограничение диапазона (если задан)
                    if self._marker_min_x is not None:
                        new_x = max(self._marker_min_x, new_x)
                    if self._marker_max_x is not None:
                        new_x = min(self._marker_max_x, new_x)
                    # ограничение скорости (интерполяция по времени)
                    now_ms = pygame.time.get_ticks()
                    dt = max(1, now_ms - getattr(self, '_last_marker_update_ms', now_ms)) / 1000.0
                    max_dx = self._marker_speed * dt
                    dx = new_x - self._marker_x
                    if abs(dx) > max_dx:
                        dx = max_dx if dx > 0 else -max_dx
                    self._marker_x += dx
                    self._last_marker_update_ms = now_ms
                    # обновить target_state и послать в контроллер с throttle
                    self._throttled_send_target()

    def _throttled_send_target(self) -> None:
        """Отправляет обновление target_state.x в контроллер с учетом throttle времени."""
        now_ms = pygame.time.get_ticks()
        if now_ms - getattr(self, '_last_marker_update_ms', 0) >= self._marker_throttle_ms:
                    try:
                        # Обновляем локальное целевое состояние; контроллер будет читать его при следующем шаге
                        self._target.x = float(self._marker_x)
                    finally:
                        self._last_marker_update_ms = now_ms

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

    # Удалён: старый метод _compile_video — сборка теперь делегируется Recorder.compile

    # Вспомогательные методы для управления записью и кадрированием
    def _toggle_recording(self) -> None:
        import tempfile

        if not self._recording:
            d = tempfile.mkdtemp(prefix="pendulum_rec_", dir=self._record_dir)
            self._record_dir = d
            self._frame_index = 0
            self._recording = True
            self._need_compile = False
            # обновить рекордер
            if hasattr(self, '_recorder') and self._recorder is not None:
                    # Recorder не обязательно имеет set_dir; синхронизируем значения напрямую
                    try:
                        self._recorder.record_dir = self._record_dir
                        self._recorder.frame_index = self._frame_index
                        self._recorder.recording = self._recording
                        self._recorder.need_compile = self._need_compile
                    except Exception:
                        pass
        else:
            # при остановке записи — сохранить финальный кадр и пометить на сборку
            try:
                if self._record_dir is not None:
                    fname = f"frame_{self._frame_index:06d}.png"
                    path = os.path.join(self._record_dir, fname)
                    try:
                        pygame.image.save(self._screen, path)
                        self._frame_index += 1
                    except Exception:
                        pass

                    if self._frame_index > 0:
                        self._need_compile = True
            except Exception:
                pass
            self._recording = False

    def _save_frame_if_recording(self, last_save_acc_ms: int) -> int:
        """Сохраняет кадр если накоплено достаточно времени. Возвращает остаток аккумулятора времени."""
        try:
            sim_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
        except Exception:
            sim_interval_ms = int(1.0 / FPS * 1000.0)

        if last_save_acc_ms >= sim_interval_ms and self._recording and self._record_dir is not None:
            last_save_acc_ms = last_save_acc_ms % sim_interval_ms
            fname = f"frame_{self._frame_index:06d}.png"
            path = os.path.join(self._record_dir, fname)
            try:
                pygame.image.save(self._screen, path)
                self._frame_index += 1
            except Exception:
                pass
        return last_save_acc_ms

    def _compute_sim_fps(self, sim_interval_ms: int) -> int:
        try:
            import glob

            frames = sorted(glob.glob(os.path.join(self._record_dir, "frame_*.png")))
            n_frames = len(frames)
            if self._elapsed_when_terminated is not None:
                sim_seconds = self._elapsed_when_terminated / 1000.0
            else:
                sim_seconds = (pygame.time.get_ticks() - self._start_ticks) / 1000.0
            if sim_seconds > 0 and n_frames > 0:
                return max(1, round(n_frames / sim_seconds))
        except Exception:
            pass
        return max(1, round(1000.0 / sim_interval_ms))

    def _compile_video_dir(self) -> None:
        try:
            try:
                sim_interval_ms = int(self._controller._dt * 1000.0) if self._controller is not None else int(1.0 / FPS * 1000.0)
            except Exception:
                sim_interval_ms = int(1.0 / FPS * 1000.0)
            sim_fps = self._compute_sim_fps(sim_interval_ms)
            # Используем единый интерфейс записи
            if hasattr(self, '_recorder') and self._recorder is not None:
                try:
                    self._recorder.compile(sim_fps)
                except Exception:
                    # fallback to module-level compile
                    try:
                        compile_video(self._record_dir, sim_fps)
                    except Exception:
                        pass
            else:
                try:
                    compile_video(self._record_dir, sim_fps)
                except Exception:
                    pass
        finally:
            self._need_compile = False

    # (старый wrapper удалён — используется _compile_video или Recorder.compile)

    # ── Отрисовка ─────────────────────────────────────────────────────────

    def _draw(self, applied_force: float) -> None:
        # Получение состояний из объекта управления
        q = self._plant.q
        dq = self._plant.dq
        is_single = self._plant.single_pendulum_mode
        x = q[0]
        th1 = q[1]
        th2 = q[2] if not is_single else 0.0
        dx = dq[0]
        dth1 = dq[1]
        dth2 = dq[2] if not is_single else 0.0

        cart_x_px = int(WIDTH // 2 + x * SCALE)
        cart_y_px = TRACK_Y - CART_H // 2

        # Очистка и отрисовка сцены
        self._screen.fill(BLACK)
        pygame.draw.line(self._screen, GRAY, (0, TRACK_Y), (WIDTH, TRACK_Y), 2)
        draw_cart(self._screen, cart_x_px, cart_y_px)
        draw_pendulums(self._screen, cart_x_px, cart_y_px, th1, th2, is_single)
        draw_force_arrow(self._screen, applied_force, cart_x_px, cart_y_px)

        # HUD
        mode = "PID" if self._controller else "РУЧНОЕ"
        gains_str = ""
        if self._controller and hasattr(self._controller, "gains"):
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

        draw_hud(self._screen, self._font, lines)
        # controller enable/disable button
        draw_controller_button(self._screen, self._font, getattr(self, '_controller_enabled', False))
        # Рисуем маркер цели, только если передан контроллер (в этом режиме GUI управляет target_state.x)
        if self._controller is not None:
            # вычислить пиксельную позицию маркера
            marker_px = int(WIDTH // 2 + self._marker_x * SCALE)
            marker_py = cart_y_px
            # цвет зависит от hover — простая проверка позиции курсора
            mx, my = pygame.mouse.get_pos()
            rect = pygame.Rect(marker_px - self._marker_w // 2, marker_py - self._marker_h, self._marker_w, self._marker_h)
            color = self._marker_color
            if rect.collidepoint(mx, my):
                color = self._marker_hover_color
            draw_target_marker(self._screen, marker_px, marker_py, color, self._marker_w, self._marker_h, f"{self._marker_x:.3f}")
            # Отладочный HUD: показать текущее значение маркера и его пикс. позицию
            debug_s = f"MARKER x={self._marker_x:.3f} m  px={marker_px}  (use ←/→ to nudge by 0.05m)"
            dbg_surf = self._font.render(debug_s, True, (180, 180, 180))
            self._screen.blit(dbg_surf, (20, HEIGHT - 70))
            # tiny indicator showing whether arrows were pressed recently
            if getattr(self, '_last_marker_update_ms', 0) + 500 > pygame.time.get_ticks():
                hint = self._font.render("[marker updated]", True, (200, 200, 100))
                self._screen.blit(hint, (20, HEIGHT - 94))

        # coordinates display removed per user request

        # Время и статус: смещаем влево, чтобы освободить место для кнопки
        elapsed_s = self._get_elapsed_time()
        time_surf = self._font.render(f"Время: {elapsed_s:.2f} с", True, GREEN)
        self._screen.blit(time_surf, (WIDTH - 360, 20))

        # Вывод ошибки (target - текущие измерения) рядом с кнопкой контроллера
        # Ошибки отключены по запросу пользователя (убрано отображение err_x/err_th)

        if self._terminated:
            term_surf = self._font.render("СИМУЛЯЦИЯ ОСТАНОВЛЕНА (Пробел - рестарт)", True, RED)
            self._screen.blit(term_surf, (WIDTH // 2 - term_surf.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        self._clock.tick(FPS)

    def _get_elapsed_time(self) -> float:
        """Возвращает прошедшее время симуляции в секундах."""
        if hasattr(self, '_start_ticks'):
            if self._terminated and self._elapsed_when_terminated is not None:
                elapsed_ms = self._elapsed_when_terminated
            else:
                elapsed_ms = pygame.time.get_ticks() - self._start_ticks
            return elapsed_ms / 1000.0
        return 0.0
