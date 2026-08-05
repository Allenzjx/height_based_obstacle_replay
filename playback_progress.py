"""Worker-derived playback progress shared by scheduler, manager and GUI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class PlaybackState(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    RESTORING = "RESTORING"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class PlaybackProgress:
    playback_state: str = PlaybackState.IDLE.value
    status_text: str = "Idle"
    current_step_index: int = 0
    total_steps: int = 0
    current_step_id: str = ""
    current_command_index_in_step: int = 0
    commands_in_current_step: int = 0
    global_command_index: int = 0
    total_commands: int = 0
    elapsed_time: float = 0.0
    estimated_remaining_time: float = 0.0
    playback_profile: str = "raw"
    last_error: str = ""
    command_phase: str = "idle"
    selected_playback: bool = False
    scheduler_lateness_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PlaybackProgress":
        data = dict(payload or {})
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in valid})
