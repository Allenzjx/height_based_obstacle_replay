"""Small sim transport facade used in place of serial transport."""

from __future__ import annotations

import time
import uuid
from typing import Any

from command_model import CommandMessage
from playback import PlaybackPlan, playback_plan_to_payload
from sim_robot_adapter import NullSimRobotAdapter


class SimTransport:
    def __init__(self, adapter: Any | None = None):
        self.adapter = adapter or NullSimRobotAdapter()
        self.process_client: Any | None = None
        self.ready = False
        self.no_sim = isinstance(self.adapter, NullSimRobotAdapter)
        self.last_command = ""
        self.last_error = ""
        self.last_worker_status: dict[str, Any] = {}
        self.wheel_generation = 0

    def attach(self, adapter: Any, *, ready: bool) -> None:
        self.adapter = adapter
        self.process_client = None
        self.ready = bool(ready)
        self.no_sim = isinstance(adapter, NullSimRobotAdapter)
        self.last_error = ""

    def attach_process_client(self, process_client: Any) -> None:
        self.process_client = process_client
        self.ready = False
        self.no_sim = False
        self.last_error = ""

    def send(self, command: str, *, source: str = "ui", message: CommandMessage | None = None) -> None:
        self.last_command = str(command)
        outgoing = message or CommandMessage(text=str(command), source=source)
        if _is_wheel_command(str(command)) and outgoing.wheel_generation is None:
            outgoing.wheel_generation = self.wheel_generation
        outgoing.command_id = outgoing.command_id or uuid.uuid4().hex
        outgoing.requested_wall_time = outgoing.requested_wall_time or time.time()
        payload = _telemetry_payload_from_message(message)
        payload.update(_wheel_payload_from_message(outgoing))
        try:
            if self.process_client is not None:
                self.process_client.send_command(str(command), source=source, **payload)
            else:
                self.adapter.handle_command(CommandMessage(text=str(command), source=source, **payload))
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def stop_wheels(self, *, reason: str = "safety_stop") -> dict[str, Any]:
        self.wheel_generation += 1
        requested = time.time()
        command_id = uuid.uuid4().hex
        if self.process_client is not None:
            result = self.process_client.stop_wheels(
                generation=self.wheel_generation,
                command_id=command_id,
                requested_wall_time=requested,
            )
        else:
            self.adapter.stop_wheels(generation=self.wheel_generation, command_id=command_id, requested_wall_time=requested)
            result = {"wheel_generation": self.wheel_generation, "command_id": command_id}
        result["reason"] = str(reason)
        return result

    def start_playback_plan(
        self,
        plan: PlaybackPlan,
        *,
        start_delay_sim_s: float = 0.0,
        plan_id: str = "",
    ) -> None:
        payload = playback_plan_to_payload(plan)
        if self.process_client is not None:
            self.process_client.start_playback_plan(
                payload,
                start_delay_sim_s=start_delay_sim_s,
                plan_id=plan_id,
            )
        elif hasattr(self.adapter, "play_plan_blocking"):
            self.adapter.play_plan_blocking(plan)
        else:
            raise RuntimeError("Playback requires an Isaac worker or adapter playback support.")

    def pause_playback(self) -> None:
        if self.process_client is not None:
            self.process_client.pause_playback()

    def resume_playback(self) -> None:
        if self.process_client is not None:
            self.process_client.resume_playback()

    def stop_playback(self, *, reason: str = "stopped", stop_wheels: bool = True) -> None:
        if self.process_client is not None:
            self.process_client.stop_playback(reason=reason, stop_wheels=stop_wheels)
        else:
            if stop_wheels:
                self.adapter.stop_wheels()

    def respawn(self) -> None:
        if self.process_client is not None:
            self.process_client.respawn()
        else:
            self.adapter.respawn_robot()

    def set_height_mm(self, height_mm: int, **payload: Any) -> None:
        if self.process_client is not None:
            self.process_client.set_height_mm(height_mm, **payload)
        else:
            raise RuntimeError("Height changes require an attached simulation worker.")

    def set_height_respawn(self, height_mm: int, **payload: Any) -> None:
        if self.process_client is not None:
            self.process_client.set_height_respawn(height_mm, **payload)
        else:
            raise RuntimeError("Height+respawn requires an attached simulation worker.")

    def recalibrate(self, **payload: Any) -> None:
        if self.process_client is not None:
            self.process_client.recalibrate(**payload)
        else:
            raise RuntimeError("Recalibration requires an attached simulation worker.")

    def set_speed_scale(self, speed_percent: float, **payload: Any) -> None:
        if self.process_client is not None:
            self.process_client.set_speed_scale(speed_percent, **payload)
        else:
            self.adapter.set_speed_percent(speed_percent)

    def apply_motion_batch(self, payload: dict[str, Any]) -> str:
        batch = dict(payload or {})
        batch_id = str(batch.get("batch_id", "") or uuid.uuid4().hex)
        batch["batch_id"] = batch_id
        if self.process_client is not None:
            return self.process_client.apply_motion_batch(batch)
        self.adapter.apply_motion_batch(batch)
        return batch_id

    def request_state(self, *, detailed: bool = False) -> None:
        if self.process_client is not None:
            self.process_client.request_state(detailed=detailed)

    def capture_sim_state(self) -> dict[str, Any]:
        status_state = self.last_worker_status.get("sim_state")
        if isinstance(status_state, dict):
            return dict(status_state)
        if hasattr(self.adapter, "capture_sim_state"):
            return self.adapter.capture_sim_state()
        return {"command_state": self.capture_command_state()}

    def restore_sim_state(self, sim_state: dict[str, Any]) -> None:
        if self.process_client is not None:
            self.process_client.restore_sim_state(sim_state)
        elif hasattr(self.adapter, "restore_sim_state"):
            self.adapter.restore_sim_state(sim_state)

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        status_state = self.last_worker_status.get("command_state")
        if isinstance(status_state, dict):
            return status_state
        return self.adapter.capture_command_state()

    def update_worker_status(self, status: dict[str, Any]) -> None:
        self.last_worker_status = dict(status)
        self.ready = bool(status.get("runtime_ready", status.get("ready", False)))
        self.last_error = str(status.get("error", "") or "")

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "no_sim": self.no_sim,
            "adapter": type(self.adapter).__name__,
            "process_worker": self.process_client is not None,
            "worker_status": self.last_worker_status,
            "last_command": self.last_command,
            "last_error": self.last_error,
        }


def _telemetry_payload_from_message(message: CommandMessage | None) -> dict[str, Any]:
    if message is None:
        return {}
    return {
        "playback_label": str(getattr(message, "playback_label", "") or ""),
        "playback_event_index": getattr(message, "playback_event_index", None),
        "playback_event_count": int(getattr(message, "playback_event_count", 0) or 0),
        "playback_final_time_s": float(getattr(message, "playback_final_time_s", 0.0) or 0.0),
        "source_step": getattr(message, "source_step", None),
    }


def _wheel_payload_from_message(message: CommandMessage) -> dict[str, Any]:
    return {
        "wheel_generation": message.wheel_generation,
        "command_id": message.command_id,
        "high_priority": bool(message.high_priority),
        "requested_wall_time": float(message.requested_wall_time or 0.0),
        "enqueued_wall_time": float(message.enqueued_wall_time or 0.0),
    }


def _is_wheel_command(command: str) -> bool:
    for part in str(command).split(";"):
        tokens = part.strip().lower().split()
        if tokens and tokens[0] in {"w", "a", "s", "d", "x", "stop", "wheel", "wheels", "speed"}:
            return True
    return False
