"""Single-owner operation state for recording, playback, and scene changes."""

from __future__ import annotations

from enum import Enum
from threading import RLock


class OperationState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PLAYBACK = "PLAYBACK"
    SCENE_UPDATE = "SCENE_UPDATE"
    RESPAWNING = "RESPAWNING"


_BUSY_REASONS = {
    OperationState.RECORDING: "Recording is active.",
    OperationState.PLAYBACK: "Playback is already active.",
    OperationState.SCENE_UPDATE: "Scene update is in progress.",
    OperationState.RESPAWNING: "Robot respawn is in progress.",
}


class OperationCoordinator:
    """Own the one mutually-exclusive operation state used by every task."""

    def __init__(self) -> None:
        self._state = OperationState.IDLE
        self._detail = ""
        self._lock = RLock()

    @property
    def state(self) -> OperationState:
        with self._lock:
            return self._state

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    @property
    def idle(self) -> bool:
        return self.state is OperationState.IDLE

    @property
    def reason(self) -> str:
        with self._lock:
            return self._detail or _BUSY_REASONS.get(self._state, "")

    def begin(self, state: OperationState, *, detail: str = "") -> bool:
        state = OperationState(state)
        if state is OperationState.IDLE:
            raise ValueError("Use finish() to return to IDLE.")
        with self._lock:
            if self._state is not OperationState.IDLE:
                return False
            self._state = state
            self._detail = str(detail or "")
            return True

    def transition(self, expected: OperationState, target: OperationState, *, detail: str = "") -> bool:
        expected = OperationState(expected)
        target = OperationState(target)
        with self._lock:
            if self._state is not expected:
                return False
            self._state = target
            self._detail = str(detail or "")
            return True

    def enter_playback(self, *, detail: str = "") -> bool:
        with self._lock:
            if self._state is OperationState.IDLE:
                self._state = OperationState.PLAYBACK
                self._detail = str(detail or "")
                return True
            if self._state is OperationState.RESPAWNING:
                self._state = OperationState.PLAYBACK
                self._detail = str(detail or "")
                return True
            return self._state is OperationState.PLAYBACK

    def finish(self, expected: OperationState | None = None) -> bool:
        with self._lock:
            if expected is not None and self._state is not OperationState(expected):
                return False
            self._state = OperationState.IDLE
            self._detail = ""
            return True

