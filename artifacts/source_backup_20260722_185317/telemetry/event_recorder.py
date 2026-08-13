"""JSONL event recorder with stateful de-duplication."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EventRecorder:
    output_path: Path
    dedupe_interval_s: float = 0.25
    events: list[dict[str, Any]] = field(default_factory=list)
    _active_states: set[tuple[str, str]] = field(default_factory=set)
    _last_event_time: dict[tuple[str, str], float] = field(default_factory=dict)

    def record(
        self,
        simulation_time_s: float,
        event_type: str,
        *,
        severity: str = "info",
        body: str = "",
        joint: str = "",
        obstacle: str = "",
        value: Any = None,
        threshold: Any = None,
        message: str = "",
        step_sequence: str = "",
        phase: str = "",
        key: str = "",
        extra: dict[str, Any] | None = None,
    ) -> bool:
        dedupe_key = (str(event_type), str(key or body or joint or obstacle or "global"))
        last = self._last_event_time.get(dedupe_key)
        if last is not None and float(simulation_time_s) - float(last) < float(self.dedupe_interval_s):
            return False
        event = {
            "simulation_time_s": float(simulation_time_s),
            "event_type": str(event_type),
            "severity": str(severity),
            "body": str(body or ""),
            "joint": str(joint or ""),
            "obstacle": str(obstacle or ""),
            "value": value,
            "threshold": threshold,
            "message": str(message or ""),
            "associated_step_sequence": str(step_sequence or ""),
            "associated_phase": str(phase or ""),
            "wall_time": time.time(),
        }
        if extra:
            event.update(dict(extra))
        self.events.append(event)
        self._last_event_time[dedupe_key] = float(simulation_time_s)
        return True

    def record_state(
        self,
        simulation_time_s: float,
        event_type: str,
        active: bool,
        *,
        key: str,
        enter_severity: str = "warning",
        exit_severity: str = "info",
        enter_message: str = "",
        exit_message: str = "",
        **kwargs: Any,
    ) -> None:
        state_key = (str(event_type), str(key))
        if active and state_key not in self._active_states:
            self._active_states.add(state_key)
            self.record(
                simulation_time_s,
                event_type,
                severity=enter_severity,
                key=key,
                message=enter_message,
                **kwargs,
            )
        elif not active and state_key in self._active_states:
            self._active_states.remove(state_key)
            self.record(
                simulation_time_s,
                event_type + "_cleared",
                severity=exit_severity,
                key=key,
                message=exit_message,
                **kwargs,
            )

    def flush(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for event in self.events:
                stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
