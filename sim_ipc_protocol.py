"""Newline-delimited JSON protocol for the Isaac worker subprocess."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


MESSAGE_TYPES = {
    "hello",
    "status",
    "command",
    "set_height",
    "set_height_respawn",
    "recalibrate_ground_reference",
    "set_speed_scale",
    "apply_motion_batch",
    "respawn",
    "stop_wheels",
    "start_playback_plan",
    "pause_playback",
    "resume_playback",
    "stop_playback",
    "restore_sim_state",
    "request_state",
    "shutdown",
    "error",
    "log",
    "operation_ack",
    "stop_ack",
    "save_result",
}


def make_message(message_type: str, **payload: Any) -> dict[str, Any]:
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"Unsupported IPC message type: {message_type}")
    message = {"type": message_type}
    message.update(payload)
    return message


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")


def decode_line(line: bytes | str) -> dict[str, Any] | None:
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
        text = text.strip()
        if not text:
            return None
        message = json.loads(text)
        if not isinstance(message, dict):
            return make_message("error", error="IPC JSON message is not an object", raw=text)
        message_type = str(message.get("type", ""))
        if message_type not in MESSAGE_TYPES:
            return make_message("error", error=f"Unsupported IPC message type: {message_type}", raw=text)
        return message
    except Exception as exc:
        return make_message("error", error=f"Invalid IPC JSON: {exc}", raw=_safe_text(line))


@dataclass
class JsonLineBuffer:
    """Incrementally decodes newline-delimited JSON bytes."""

    text: str = ""

    def feed(self, chunk: bytes | str) -> list[dict[str, Any]]:
        self.text += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
        messages: list[dict[str, Any]] = []
        while "\n" in self.text:
            line, self.text = self.text.split("\n", 1)
            message = decode_line(line)
            if message is not None:
                messages.append(message)
        return messages


def _safe_text(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
