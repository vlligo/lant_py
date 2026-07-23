"""
Port of AntFieldWidget (the Qt C++ custom-painted canvas) to PySide6.

Rendering strategy mirrors the original:
  - A cached QPixmap buffer is only redrawn when something invalidates it
    (pan/zoom/resize/step), not on every paintEvent.
  - When cells are >= 1px on screen: draw grid + per-cell QPainter fillRect,
    only over the visible chunk range.
  - When zoomed out far enough that cells are sub-pixel: switch to a
    vectorized NumPy "paint by pixel" path (equivalent to the C++
    QtConcurrent per-pixel image fill), which is what makes huge patterns
    (millions of cells) still render at interactive speed.
"""
import math
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (QColor, QCursor, QImage, QPainter, QPen,
                            QPixmap, QPolygonF, QTransform)
from PySide6.QtWidgets import QWidget

from engine.common import CHUNK_AREA, CHUNK_SHIFT, CHUNK_SIZE, chunk_index
from engine.loader import AntEngine


class DisplayStyle(IntEnum):
    JUST_COLORS = 0
    VISITS = 1
    ROTATIONS = 2
    ARCS = 3
    DIAGONALS = 4


class AntFieldWidget(QWidget):
    antMoved = Signal(int, int, int, int)          # x, y, direction, steps
    zoomChanged = Signal(float)
    stepsChanged = Signal(int)
    mouseOverCell = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        self.engine = AntEngine()

        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom_factor = 1.0
        self.cell_size = 6

        self.dragging = False
        self.last_mouse_pos = QPoint()
        self.last_mouse_cell_pos = (0, 0)

        self.needs_redraw = True
        self.buffer_pixmap = QPixmap()

        self.current_style = DisplayStyle.JUST_COLORS
        self._state_color_cache: list[QColor] = []
        self._rgb_color_cache: Optional[np.ndarray] = None  # (N,3) uint8, for the fast path

        self.reset()

        self._mouse_update_timer = QTimer(self)
        self._mouse_update_timer.setInterval(50)
        self._mouse_update_timer.timeout.connect(self._poll_mouse_position)
        self._mouse_update_timer.start()

    # ------------------------------------------------------------------ #
    # Rules / lifecycle (delegates to the engine)
    # ------------------------------------------------------------------ #
    def set_rules(self, rules_str: str):
        self.engine.set_rules(rules_str)
        self._update_state_colors()
        self.reset()

    def get_rules(self) -> str:
        return self.engine.get_rules()

    def reset(self):
        self.engine.reset()
        self.offset_x = self.width() / 2.0
        self.offset_y = self.height() / 2.0
        self.needs_redraw = True
        self.update()
        self.antMoved.emit(0, 0, 0, 0)
        self.stepsChanged.emit(0)

    def next_step(self, steps: int = 1):
        if not self.engine.rules or steps <= 0:
            return
        self.engine.next_step(steps)
        self.needs_redraw = True
        self.center_on_ant()
        self.antMoved.emit(self.engine.ant_x, self.engine.ant_y, self.engine.ant_dir, self.engine.step_count)
        self.stepsChanged.emit(self.engine.step_count)

    def set_display_style(self, style: DisplayStyle):
        if self.current_style != style:
            self.current_style = style
            self.needs_redraw = True
            self.update()

    def set_cell_size(self, size: int):
        self.cell_size = max(1, min(50, int(size)))
        self.needs_redraw = True
        self.update()

    def get_cell_size(self) -> int:
        return self.cell_size

    def get_zoom(self) -> float:
        return self.zoom_factor

    def set_zoom(self, zoom: float):
        center_point = self._screen_to_field(QPoint(self.width() // 2, self.height() // 2))
        old_center_screen = self._field_to_screen(center_point)
        self.zoom_factor = max(0.00001, min(50.0, zoom))
        new_center_screen = self._field_to_screen(center_point)
        self.offset_x += old_center_screen[0] - new_center_screen[0]
        self.offset_y += old_center_screen[1] - new_center_screen[1]
        self.needs_redraw = True
        self.update()
        self.zoomChanged.emit(self.zoom_factor)

    def randomize_area(self, radius: int):
        self.engine.randomize_area(radius)
        self.needs_redraw = True
        self.update()

    @staticmethod
    def estimate_randomize_area_bytes(radius: int) -> int:
        return AntEngine.estimate_randomize_area_bytes(radius)

    # -- statistics passthroughs ----------------------------------------#
    def get_visit_count(self, x, y):
        return self.engine.get_visit_count(x, y)

    def get_most_visited_cell(self):
        return self.engine.get_most_visited_cell()

    def get_statistics_summary(self):
        return self.engine.get_statistics_summary()

    def get_top_visited_cells(self, count):
        return self.engine.get_top_visited_cells(count)

    def is_statistics_enabled(self):
        return self.engine.statistics_enabled

    def set_statistics_enabled(self, enabled: bool):
        self.engine.set_statistics_enabled(enabled)
        self.needs_redraw = True
        self.update()

    def reset_statistics(self):
        self.engine.reset_statistics()
        self.needs_redraw = True
        self.update()

    def save_state(self, filename: str) -> bool:
        return self.engine.save_state(filename)

    def load_state(self, filename: str) -> bool:
        ok = self.engine.load_state(filename)
        if ok:
            self._update_state_colors()
            self.needs_redraw = True
            self.update()
            self.antMoved.emit(self.engine.ant_x, self.engine.ant_y, self.engine.ant_dir, self.engine.step_count)
            self.stepsChanged.emit(self.engine.step_count)
            self.zoomChanged.emit(self.zoom_factor)
        return ok

    # ------------------------------------------------------------------ #
    # View centering / movement
    # ------------------------------------------------------------------ #
    def center_on_ant(self):
        scaled = self.cell_size * self.zoom_factor
        self.offset_x = self.width() / 2.0 - self.engine.ant_x * scaled
        self.offset_y = self.height() / 2.0 - self.engine.ant_y * scaled
        self.needs_redraw = True
        self.update()

    def center_on_point(self, x: int, y: int):
        scaled = self.cell_size * self.zoom_factor
        self.offset_x = self.width() / 2.0 - x * scaled
        self.offset_y = self.height() / 2.0 - y * scaled
        self.needs_redraw = True
        self.update()

    def move_view(self, dx: int, dy: int):
        self.offset_x += dx
        self.offset_y += dy
        self.needs_redraw = True
        self.update()

    # ------------------------------------------------------------------ #
    # Coordinate conversions
    # ------------------------------------------------------------------ #
    def _screen_to_field(self, pos: QPoint) -> Tuple[int, int]:
        scaled = self.cell_size * self.zoom_factor
        if abs(scaled) < 1e-12:
            return 0, 0
        return (math.floor((pos.x() - self.offset_x) / scaled),
                math.floor((pos.y() - self.offset_y) / scaled))

    def _field_to_screen(self, field_pos: Tuple[int, int]) -> Tuple[int, int]:
        scaled = self.cell_size * self.zoom_factor
        return (round(field_pos[0] * scaled + self.offset_x),
                round(field_pos[1] * scaled + self.offset_y))

    # ------------------------------------------------------------------ #
    # Colors
    # ------------------------------------------------------------------ #
    def _update_state_colors(self):
        self._state_color_cache.clear()
        max_states = max(2, len(self.engine.rules))
        rgb = np.zeros((max_states, 3), dtype=np.uint8)
        for state in range(max_states):
            ratio = state / (max_states - 1 if max_states > 1 else 1)
            hue = int(ratio * 360) % 360
            color = QColor.fromHsv(hue, 200, 230)
            self._state_color_cache.append(color)
            rgb[state] = (color.red(), color.green(), color.blue())
        self._rgb_color_cache = rgb

    def _state_to_color(self, state: int) -> QColor:
        if not self._state_color_cache:
            return QColor.fromHsv(0, 200, 230)
        return self._state_color_cache[state % len(self._state_color_cache)]

    def _previous_state(self, state: int) -> int:
        n = len(self.engine.rules)
        return n - 1 if state == 0 else state - 1

    def _next_state(self, state: int) -> int:
        return (state + 1) % len(self.engine.rules)

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event):
        if self.needs_redraw or self.buffer_pixmap.size() != self.size():
            self._redraw_buffer()
            self.needs_redraw = False
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.buffer_pixmap)

    def _redraw_buffer(self):
        self.buffer_pixmap = QPixmap(self.size())
        self.buffer_pixmap.fill(Qt.GlobalColor.white)

        scaled = self.cell_size * self.zoom_factor
        if scaled <= 0:
            return

        start_x = math.floor(-self.offset_x / scaled) - 1
        end_x = math.ceil((self.width() - self.offset_x) / scaled) + 1
        start_y = math.floor(-self.offset_y / scaled) - 1
        end_y = math.ceil((self.height() - self.offset_y) / scaled) + 1

        start_cx, end_cx = chunk_index(start_x), chunk_index(end_x)
        start_cy, end_cy = chunk_index(start_y), chunk_index(end_y)

        use_optimized = scaled < 1.0
        painter = QPainter(self.buffer_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if use_optimized:
            painter.end()
            self._redraw_zoomed_out(start_cx, end_cx, start_cy, end_cy, scaled)
            painter = QPainter(self.buffer_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        else:
            if scaled >= 4:
                painter.setPen(QPen(QColor(220, 220, 220), 0.5))
                for x in range(start_x, end_x + 1):
                    sx = self.offset_x + x * scaled
                    painter.drawLine(QPointF(sx, self.offset_y + start_y * scaled),
                                      QPointF(sx, self.offset_y + end_y * scaled))
                for y in range(start_y, end_y + 1):
                    sy = self.offset_y + y * scaled
                    painter.drawLine(QPointF(self.offset_x + start_x * scaled, sy),
                                      QPointF(self.offset_x + end_x * scaled, sy))

            for cy in range(start_cy, end_cy + 1):
                for cx in range(start_cx, end_cx + 1):
                    chunk = self.engine.chunks.get((cx, cy))
                    if chunk is None:
                        continue
                    states = chunk.states
                    nz = np.nonzero(states)[0]
                    for i in nz:
                        i = int(i)
                        lx, ly = i % CHUNK_SIZE, i // CHUNK_SIZE
                        gx, gy = (cx << CHUNK_SHIFT) + lx, (cy << CHUNK_SHIFT) + ly
                        if start_x <= gx <= end_x and start_y <= gy <= end_y:
                            color = self._state_to_color(int(states[i]))
                            sx = self.offset_x + gx * scaled
                            sy = self.offset_y + gy * scaled
                            painter.fillRect(QRectF(sx, sy, scaled, scaled), color)

            if self.engine.statistics_enabled and scaled >= 8 and self.current_style != DisplayStyle.JUST_COLORS:
                self._draw_statistics_overlay(painter, start_cx, end_cx, start_cy, end_cy, scaled)

        self._draw_ant(painter, scaled)
        painter.end()

    def _draw_statistics_overlay(self, painter, start_cx, end_cx, start_cy, end_cy, scaled):
        painter.setPen(QPen(Qt.GlobalColor.black, 3))
        rules = self.engine.rules

        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                stat_chunk = self.engine.stat_chunks.get((cx, cy))
                if stat_chunk is None:
                    continue
                chunk = self.engine.chunks.get((cx, cy))
                nz = np.nonzero(stat_chunk.visits)[0]

                for i in nz:
                    i = int(i)
                    gx, gy = (cx << CHUNK_SHIFT) + (i % CHUNK_SIZE), (cy << CHUNK_SHIFT) + (i // CHUNK_SIZE)
                    cell_rect = QRectF(self.offset_x + gx * scaled, self.offset_y + gy * scaled, scaled, scaled)

                    if self.current_style == DisplayStyle.VISITS:
                        text = str(int(stat_chunk.visits[i]))
                        self._draw_scaled_text(painter, cell_rect, text)

                    elif self.current_style == DisplayStyle.ROTATIONS:
                        corners = stat_chunk.corners[i]
                        half_w, half_h = cell_rect.width() / 2, cell_rect.height() / 2
                        positions = [
                            (cell_rect.left(), cell_rect.top()),                    # top-left
                            (cell_rect.center().x(), cell_rect.top()),              # top-right
                            (cell_rect.center().x(), cell_rect.center().y()),       # bottom-right
                            (cell_rect.left(), cell_rect.center().y()),             # bottom-left
                        ]
                        for c_idx in range(4):
                            value = int(corners[c_idx])
                            if value == 0:
                                continue
                            px, py = positions[c_idx]
                            corner_rect = QRectF(px, py, half_w, half_h)
                            self._draw_scaled_text(painter, corner_rect, str(value), margin=0.85)

                    elif self.current_style == DisplayStyle.DIAGONALS and chunk is not None:
                        state = int(chunk.states[i])
                        if rules[state] != rules[self._next_state(state)]:
                            parity = (i % CHUNK_SIZE + i // CHUNK_SIZE) % 2 != 0
                            if (rules[self._previous_state(state)] == 'R') ^ parity:
                                painter.drawLine(cell_rect.bottomLeft(), cell_rect.topRight())
                            else:
                                painter.drawLine(cell_rect.bottomRight(), cell_rect.topLeft())

                    elif self.current_style == DisplayStyle.ARCS and chunk is not None:
                        state = int(chunk.states[i])
                        parity = (i % CHUNK_SIZE + i // CHUNK_SIZE) % 2 == 0
                        w, h = cell_rect.width() / 2.0, cell_rect.height() / 2.0
                        if (rules[self._previous_state(state)] == 'L') ^ parity:
                            painter.drawArc(cell_rect.adjusted(w, h, w, h), 180 * 16, -90 * 16)
                            painter.drawArc(cell_rect.adjusted(-w, -h, -w, -h), 0 * 16, -90 * 16)
                        else:
                            painter.drawArc(cell_rect.adjusted(w, -h, w, -h), 270 * 16, -90 * 16)
                            painter.drawArc(cell_rect.adjusted(-w, h, -w, h), 90 * 16, -90 * 16)

    @staticmethod
    def _draw_scaled_text(painter: QPainter, rect: QRectF, text: str, margin: float = 0.8):
        font = painter.font()
        font.setPointSizeF(1)
        painter.setFont(font)
        text_rect = painter.fontMetrics().boundingRect(text)
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            return
        scale = min((rect.width() * margin) / text_rect.width(),
                    (rect.height() * margin) / text_rect.height())
        font.setPointSizeF(max(1.0, scale))
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _redraw_zoomed_out(self, start_cx, end_cx, start_cy, end_cy, scaled):
        """Vectorized NumPy per-pixel fill for far-zoomed-out views (many
        cells per pixel or many pixels per chunk with sparse content)."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        if self._rgb_color_cache is None or len(self._rgb_color_cache) == 0:
            self._update_state_colors()
        color_cache = self._rgb_color_cache
        n_colors = len(color_cache)

        for cx, cy in list(self.engine.chunks.keys()):
            if cx < start_cx or cx > end_cx or cy < start_cy or cy > end_cy:
                continue
            chunk = self.engine.chunks[(cx, cy)]
            states = chunk.states
            nz_idx = np.nonzero(states)[0]
            if nz_idx.size == 0:
                continue

            lx = (nz_idx % CHUNK_SIZE).astype(np.int64)
            ly = (nz_idx // CHUNK_SIZE).astype(np.int64)
            gx = (cx << CHUNK_SHIFT) + lx
            gy = (cy << CHUNK_SHIFT) + ly

            screen_x = np.floor(self.offset_x + gx * scaled).astype(np.int64)
            screen_y = np.floor(self.offset_y + gy * scaled).astype(np.int64)

            valid = (screen_x >= 0) & (screen_x < w) & (screen_y >= 0) & (screen_y < h)
            if not np.any(valid):
                continue

            sx, sy = screen_x[valid], screen_y[valid]
            state_vals = states[nz_idx[valid]].astype(np.int64) % n_colors
            canvas[sy, sx] = color_cache[state_vals]

        image = QImage(canvas.data, w, h, w * 3, QImage.Format.Format_RGB888)
        painter = QPainter(self.buffer_pixmap)
        painter.drawImage(0, 0, image)
        painter.end()

    def _draw_ant(self, painter: QPainter, scaled: float):
        ant_screen_x = self.offset_x + self.engine.ant_x * scaled
        ant_screen_y = self.offset_y + self.engine.ant_y * scaled
        center = QPointF(ant_screen_x + scaled / 2, ant_screen_y + scaled / 2)

        if scaled >= 2:
            painter.setBrush(Qt.GlobalColor.red)
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            r = scaled * 0.4
            triangle = QPolygonF([
                center + QPointF(0, -r),
                center + QPointF(r * 0.7, r * 0.7),
                center + QPointF(-r * 0.7, r * 0.7),
            ])
            transform = QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(self.engine.ant_dir * 90.0)
            transform.translate(-center.x(), -center.y())
            painter.drawPolygon(transform.map(triangle))
        else:
            painter.setBrush(Qt.GlobalColor.red)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, scaled / 2, scaled / 2)

    # ------------------------------------------------------------------ #
    # Mouse / wheel / resize events
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()
            self.dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        self._update_mouse_position(event.pos())
        if self.dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.pos() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.pos()
            self.needs_redraw = True
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        zoom_step = 1.2
        old_zoom = self.zoom_factor
        new_zoom = old_zoom * zoom_step if event.angleDelta().y() > 0 else old_zoom / zoom_step
        if new_zoom < 0.00001 or new_zoom > 50.0:
            return

        mouse_pos = event.position()
        ratio = new_zoom / old_zoom
        self.offset_x = mouse_pos.x() - (mouse_pos.x() - self.offset_x) * ratio
        self.offset_y = mouse_pos.y() - (mouse_pos.y() - self.offset_y) * ratio
        self.zoom_factor = new_zoom

        self.needs_redraw = True
        self.update()
        self.zoomChanged.emit(self.zoom_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.needs_redraw = True

    def enterEvent(self, event):
        super().enterEvent(event)
        pos = event.position().toPoint() if hasattr(event, "position") else self.mapFromGlobal(QCursor.pos())
        self._update_mouse_position(pos)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.last_mouse_cell_pos = (0, 0)
        self.mouseOverCell.emit(0, 0)

    def _poll_mouse_position(self):
        if self.underMouse():
            self._update_mouse_position(self.mapFromGlobal(QCursor.pos()))

    def _update_mouse_position(self, pos: QPoint):
        field_pos = self._screen_to_field(pos)
        if field_pos != self.last_mouse_cell_pos:
            self.last_mouse_cell_pos = field_pos
            self.mouseOverCell.emit(field_pos[0], field_pos[1])
