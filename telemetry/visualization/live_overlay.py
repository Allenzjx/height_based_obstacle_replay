"""Best-effort Isaac viewport debug overlay."""

from __future__ import annotations

import math
from collections import deque
from typing import Any


class LiveTelemetryOverlay:
    def __init__(self, config: Any, *, headless: bool = False):
        self.config = config
        self.headless = bool(headless)
        self.enabled = bool(getattr(config, "live_enabled", False)) and not self.headless
        self.available = False
        self.error = ""
        self._draw = None
        self._last_draw_time_s = -math.inf
        self._trail: deque[list[float]] = deque(maxlen=max(2, int(getattr(config, "marker_limit", 256))))
        if self.enabled:
            self._initialize()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "available": bool(self.available),
            "error": self.error,
        }

    def update(self, row: dict[str, Any], stability: Any | None, contacts: Any | None, equilibrium: dict[str, Any] | None) -> None:
        if not self.enabled or not self.available:
            return
        time_s = _float(row.get("time_s"))
        interval = 1.0 / max(0.1, float(getattr(self.config, "update_hz", 20.0)))
        if time_s - self._last_draw_time_s < interval:
            return
        self._last_draw_time_s = time_s
        try:
            self._clear()
            com = [row.get("com_x_m"), row.get("com_y_m"), row.get("com_z_m")]
            if _finite_vec(com):
                self._trail.append([float(com[0]), float(com[1]), float(com[2])])
                if bool(getattr(self.config, "show_com", True)):
                    self._draw_points([com], [(1.0, 0.15, 0.10, 1.0)], [10.0])
                if bool(getattr(self.config, "show_com_projection", True)):
                    proj = [row.get("com_projection_x_m"), row.get("com_projection_y_m"), row.get("com_projection_z_m")]
                    if _finite_vec(proj):
                        self._draw_lines([com], [proj], [(1.0, 0.2, 0.1, 1.0)], [2.0])
                        self._draw_points([proj], [(1.0, 0.75, 0.15, 1.0)], [7.0])
            if bool(getattr(self.config, "show_com_trail", True)) and len(self._trail) >= 2:
                starts = list(self._trail)[:-1]
                ends = list(self._trail)[1:]
                self._draw_lines(starts, ends, [(0.0, 0.8, 1.0, 1.0)] * len(starts), [2.0] * len(starts))
            if bool(getattr(self.config, "show_support_polygon", True)) and stability is not None:
                vertices = list(getattr(stability, "support_vertices_w", []) or [])
                if len(vertices) >= 2:
                    starts = vertices
                    ends = vertices[1:] + vertices[:1]
                    self._draw_lines(starts, ends, [(0.1, 1.0, 0.35, 1.0)] * len(starts), [3.0] * len(starts))
            if bool(getattr(self.config, "show_equilibrium_region", True)) and equilibrium:
                vertices = list(equilibrium.get("vertices_w", []) or [])
                if len(vertices) >= 2:
                    starts = vertices
                    ends = vertices[1:] + vertices[:1]
                    self._draw_lines(starts, ends, [(0.25, 0.45, 1.0, 1.0)] * len(starts), [2.0] * len(starts))
            if bool(getattr(self.config, "show_contact_points", True)) and contacts is not None:
                points = [row.get("contact_point_w", []) for row in getattr(contacts, "contacts", [])]
                points = [point for point in points if _finite_vec(point)]
                if points:
                    self._draw_points(points, [(1.0, 1.0, 0.05, 1.0)] * len(points), [6.0] * len(points))
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def clear(self) -> None:
        if self._draw is None:
            return
        try:
            self._clear()
        except Exception:
            pass

    def _initialize(self) -> None:
        try:
            from isaacsim.util.debug_drawing import _debug_draw  # type: ignore

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self.available = self._draw is not None
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def _clear(self) -> None:
        if hasattr(self._draw, "clear_lines"):
            self._draw.clear_lines()
        if hasattr(self._draw, "clear_points"):
            self._draw.clear_points()

    def _draw_points(self, points: list[Any], colors: list[Any], sizes: list[float]) -> None:
        if hasattr(self._draw, "draw_points"):
            self._draw.draw_points(points, colors, sizes)

    def _draw_lines(self, starts: list[Any], ends: list[Any], colors: list[Any], widths: list[float]) -> None:
        if hasattr(self._draw, "draw_lines"):
            self._draw.draw_lines(starts, ends, colors, widths)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite_vec(value: Any) -> bool:
    try:
        values = [float(item) for item in value]
    except Exception:
        return False
    return len(values) >= 3 and all(math.isfinite(item) for item in values[:3])
