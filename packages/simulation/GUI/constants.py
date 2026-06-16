"""Константы и настройки отрисовки для PendulumViewer."""

from __future__ import annotations

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

PHYSICS_DT = 0.005
SUBTICKS = int(PHYSICS_DT / 0.005)

# --- Target marker defaults ---
# Цвет маркера цели (по умолчанию серый) и при hover
MARKER_COLOR = (0, 200, 0)      # green token
MARKER_HOVER_COLOR = (0, 160, 0)
# Скорость перемещения маркера (ед./с, в единицах координат) — по умолчанию 100
MARKER_SPEED = 100.0
# Throttle для отправки обновлений в контроллер (мс)
MARKER_THROTTLE_MS = 100
# Допустимый диапазон X (метры)
# маркер теперь не ограничен по X (может двигаться куда угодно)
MARKER_MIN_X = None
MARKER_MAX_X = None
# Размер маркера в пикселях
MARKER_W = 12
MARKER_H = 28
# Сила при ручном управлении (перенесено из gui.py)
FORCE_PER_FRAME = 20.0
