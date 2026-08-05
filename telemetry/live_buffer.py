"""Bounded live telemetry buffers for non-blocking UI display."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiveTelemetryBuffer:
    """Keep only recent display frames while preserving recent events."""

    max_frames: int = 5
    max_events: int = 1000
    dropped_frames: int = 0
    frames: deque[dict[str, Any]] = field(default_factory=deque)
    events: deque[dict[str, Any]] = field(default_factory=deque)

    def push_frame(self, frame: dict[str, Any]) -> None:
        if not isinstance(frame, dict):
            return
        limit = max(1, int(self.max_frames))
        while len(self.frames) >= limit:
            self.frames.popleft()
            self.dropped_frames += 1
        self.frames.append(dict(frame))

    def push_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        limit = max(1, int(self.max_events))
        while len(self.events) >= limit:
            self.events.popleft()
        self.events.append(dict(event))

    def latest_frame(self) -> dict[str, Any]:
        return dict(self.frames[-1]) if self.frames else {}

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        return [dict(event) for event in list(self.events)[-limit:]]

    def status(self) -> dict[str, Any]:
        return {
            "queued_live_frames": len(self.frames),
            "queued_live_events": len(self.events),
            "dropped_live_frames": int(self.dropped_frames),
            "max_live_frames": max(1, int(self.max_frames)),
        }
