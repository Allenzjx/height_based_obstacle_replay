"""Replay context and tracking helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayContext:
    active: bool = False
    sequence_name: str = ""
    started_sim_time_s: float = 0.0
    final_time_s: float = 0.0
    event_count: int = 0
    event_index: int = 0
    step_index: int | None = None
    phase: str = "idle"
    last_command: str = ""
    interruptions: int = 0
    retries: int = 0
    success: bool | None = None
    failure_reason: str = ""

    def snapshot(self, sim_time_s: float) -> dict[str, Any]:
        elapsed = max(0.0, float(sim_time_s) - float(self.started_sim_time_s)) if self.active else 0.0
        progress = elapsed / self.final_time_s if self.final_time_s > 0.0 else 0.0
        return {
            "selected_step_sequence": self.sequence_name,
            "step_sequence_phase": self.phase,
            "step_index": self.step_index,
            "step_progress": max(0.0, min(1.0, progress)),
            "replay_state": "active" if self.active else "idle",
            "replay_event_index": int(self.event_index),
            "replay_event_count": int(self.event_count),
            "last_replay_command": self.last_command,
            "sequence_interruption": int(self.interruptions),
            "sequence_retry": int(self.retries),
            "sequence_success": self.success,
            "failure_reason": self.failure_reason,
        }


def command_targets_from_adapter(adapter: Any, joint_names: list[str]) -> tuple[list[float], list[float], str]:
    pos = [math.nan] * len(joint_names)
    vel = [math.nan] * len(joint_names)
    source = "adapter.command_state"
    try:
        target_state = adapter.get_target_joint_state()
        servos = target_state.get("servos", {}) if isinstance(target_state, dict) else {}
        for index, name in enumerate(joint_names):
            if name in servos:
                value = servos[name].get("target_actual_rad")
                if value is not None:
                    pos[index] = float(value)
    except Exception:
        pass
    try:
        wheel_targets = adapter._wheel_velocity_target_by_name()
        for index, name in enumerate(joint_names):
            if name in wheel_targets:
                vel[index] = float(wheel_targets[name])
    except Exception:
        pass
    return pos, vel, source
