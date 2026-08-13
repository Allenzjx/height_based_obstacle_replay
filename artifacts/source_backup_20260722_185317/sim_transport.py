"""Small sim transport facade used in place of serial transport."""

from __future__ import annotations

from typing import Any

from command_model import CommandMessage
from sim_robot_adapter import NullSimRobotAdapter


class SimTransport:
    def __init__(self, adapter: Any | None = None):
        self.adapter = adapter or NullSimRobotAdapter()
        self.worker: Any | None = None
        self.process_client: Any | None = None
        self.ready = False
        self.no_sim = isinstance(self.adapter, NullSimRobotAdapter)
        self.last_command = ""
        self.last_error = ""
        self.last_worker_status: dict[str, Any] = {}

    def attach(self, adapter: Any, *, ready: bool) -> None:
        self.adapter = adapter
        self.worker = None
        self.process_client = None
        self.ready = bool(ready)
        self.no_sim = isinstance(adapter, NullSimRobotAdapter)
        self.last_error = ""

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker
        self.process_client = None
        self.ready = False
        self.no_sim = False
        self.last_error = ""

    def attach_process_client(self, process_client: Any) -> None:
        self.process_client = process_client
        self.worker = None
        self.ready = False
        self.no_sim = False
        self.last_error = ""

    def send(self, command: str, *, source: str = "ui", message: CommandMessage | None = None) -> None:
        self.last_command = str(command)
        payload = _telemetry_payload_from_message(message)
        try:
            if self.process_client is not None:
                self.process_client.send_command(str(command), source=source, **payload)
            elif self.worker is not None:
                self.worker.send_command(str(command), source=source, **payload)
            else:
                self.adapter.handle_command(CommandMessage(text=str(command), source=source, **payload))
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def stop_wheels(self) -> None:
        if self.process_client is not None:
            self.process_client.stop_wheels()
        elif self.worker is not None:
            self.worker.send_command("wheel stop", source="transport")
        else:
            self.adapter.stop_wheels()

    def respawn(self) -> None:
        if self.process_client is not None:
            self.process_client.respawn()
        elif self.worker is not None and hasattr(self.worker, "respawn"):
            self.worker.respawn()
        else:
            self.adapter.respawn_robot()

    def request_state(self) -> None:
        if self.process_client is not None:
            self.process_client.request_state()
        elif self.worker is not None and hasattr(self.worker, "request_state"):
            self.worker.request_state()

    def vision_control(self, action: str, **payload: Any) -> None:
        if self.process_client is not None and hasattr(self.process_client, "vision_control"):
            self.process_client.vision_control(action, **payload)
        elif self.worker is not None and hasattr(self.worker, "vision_control"):
            self.worker.vision_control(action, **payload)

    def set_vision_enabled(self, enabled: bool) -> None:
        self.vision_control("enable" if enabled else "disable")

    def request_vision_detection_once(self) -> None:
        self.vision_control("detect_once")

    def reset_vision_filter(self) -> None:
        self.vision_control("reset_filter")

    def save_vision_debug_frame(self) -> None:
        self.vision_control("save_debug_frame")

    def validate_camera(self) -> None:
        self.vision_control("validate_camera")

    def validate_current_height(self, expected_height_cm: int) -> None:
        self.vision_control("validate_current_height", expected_height_cm=int(expected_height_cm))

    def save_rgbd_diagnostic(self, expected_height_cm: int | None = None) -> None:
        payload: dict[str, Any] = {}
        if expected_height_cm is not None:
            payload["expected_height_cm"] = int(expected_height_cm)
        self.vision_control("save_rgbd_diagnostic", **payload)

    def clear_validation_result(self) -> None:
        self.vision_control("clear_validation_result")

    def validate_mount_after_next_respawn(self) -> None:
        self.vision_control("validate_mount_after_next_respawn")

    def show_camera_view(self, **payload: Any) -> None:
        self.vision_control("open_camera_viewport", **payload)

    def open_camera_viewport(self, **payload: Any) -> None:
        self.vision_control("open_camera_viewport", **payload)

    def return_main_view_to_perspective(self, **payload: Any) -> None:
        self.vision_control("return_main_view_to_perspective", **payload)

    def close_camera_viewport(self, **payload: Any) -> None:
        self.vision_control("close_camera_viewport", **payload)

    def restore_camera_view(self, **payload: Any) -> None:
        self.vision_control("restore_camera_view", **payload)

    def validate_robot_ground_contact(self) -> None:
        self.vision_control("validate_robot_ground_contact")

    def calibrate_ground_reference(self) -> None:
        self.vision_control("calibrate_ground_reference")

    def respawn_and_validate_ground(self) -> None:
        self.vision_control("respawn_validate_ground")

    def set_vision_source_mode(self, source_mode: str) -> None:
        self.vision_control("set_source_mode", source_mode=str(source_mode))

    def validate_camera_geometry(self) -> None:
        self.vision_control("validate_camera_geometry")

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
        elif self.worker is not None and hasattr(self.worker, "restore_sim_state"):
            self.worker.restore_sim_state(sim_state)
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
            "worker": self.worker is not None,
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
