from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sequence_model import empty_command_state, make_event, make_step


def make_args(store_root: Path, *, no_sim: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=no_sim,
        store_root=store_root,
        height_cm=10,
        sim_launch_mode="disabled" if no_sim else "subprocess",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        global_motion_speed_scale=1.0,
        wheel_speed_scale=1.0,
        servo_command_scale=1.0,
        playback_speed_scale=1.0,
        preserve_wheel_distance=True,
        apply_speed_scale_to_manual=True,
        apply_speed_scale_to_playback=True,
        ui_refresh_ms=100,
        sim_status_refresh_ms=250,
        full_refresh_ms=1000,
        no_continuous_sim_step=False,
        record_event_min_interval_ms=50.0,
        record_event_max_hz=20.0,
        record_coalesce_slider_events=True,
        record_max_events_per_step=2000,
        max_text_widget_chars=2000,
        disable_auto_sim_state_json=False,
        sim_state_json_on_demand=True,
        playback_pre_step_settle_s=0.0,
        respawn_play_settle_s=0.0,
        restore_step_start_state_before_selected_playback=True,
        restore_full_sim_pose_if_available=True,
        fallback_to_command_state_before=True,
        vision_auto_replay=True,
        vision_confidence_threshold=0.75,
        vision_stable_frames=5,
        vision_window_size=7,
        vision_height_tolerance_cm=2.0,
        vision_auto_replay_cooldown_s=0.0,
        vision_respawn_before_replay=True,
        onboard_camera=True,
        headless=False,
    )


def motion_step(index: int = 1, *, height_cm: int = 5, command: str = "servo front_left_hip 10") -> dict[str, Any]:
    before = empty_command_state()
    event = make_event(0.0, command, kind="test")
    return make_step(
        index=index,
        step_type="recorded",
        duration=0.2,
        events=[event],
        command_state_before=before,
        command_state_after=before,
        name=f"step_{height_cm}_{index}",
        note=f"height={height_cm}cm",
        extra={"height_cm": height_cm},
    )


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def stop_wheels(self) -> None:
        self.calls.append(("stop_wheels", {}))

    def respawn(self) -> None:
        self.calls.append(("respawn", {}))

    def request_state(self) -> None:
        self.calls.append(("request_state", {}))

    def reset_vision_filter(self) -> None:
        self.calls.append(("vision_control", {"action": "reset_filter"}))

    def request_vision_detection_once(self) -> None:
        self.calls.append(("vision_control", {"action": "detect_once"}))

    def set_vision_enabled(self, enabled: bool) -> None:
        self.calls.append(("vision_control", {"action": "enable" if enabled else "disable"}))

    def save_vision_debug_frame(self) -> None:
        self.calls.append(("vision_control", {"action": "save_debug_frame"}))

    def clear_validation_result(self) -> None:
        self.calls.append(("vision_control", {"action": "clear_validation_result"}))

    def validate_current_height(self, expected_height_cm: int) -> None:
        self.calls.append(("vision_control", {"action": "validate_current_height", "expected_height_cm": int(expected_height_cm)}))

    def validate_camera(self) -> None:
        self.calls.append(("vision_control", {"action": "validate_camera"}))

    def show_camera_view(self, **payload: Any) -> None:
        self.calls.append(("vision_control", {"action": "show_camera_view", **payload}))

    def open_camera_viewport(self, **payload: Any) -> None:
        self.calls.append(("vision_control", {"action": "open_camera_viewport", **payload}))

    def return_main_view_to_perspective(self, **payload: Any) -> None:
        self.calls.append(("vision_control", {"action": "return_main_view_to_perspective", **payload}))

    def close_camera_viewport(self, **payload: Any) -> None:
        self.calls.append(("vision_control", {"action": "close_camera_viewport", **payload}))

    def restore_camera_view(self, **payload: Any) -> None:
        self.calls.append(("vision_control", {"action": "restore_camera_view", **payload}))

    def set_vision_source_mode(self, source_mode: str) -> None:
        self.calls.append(("vision_control", {"action": "set_source_mode", "source_mode": str(source_mode)}))

    def validate_camera_geometry(self) -> None:
        self.calls.append(("vision_control", {"action": "validate_camera_geometry"}))

    def validate_robot_ground_contact(self) -> None:
        self.calls.append(("vision_control", {"action": "validate_robot_ground_contact"}))

    def calibrate_ground_reference(self) -> None:
        self.calls.append(("vision_control", {"action": "calibrate_ground_reference"}))

    def respawn_and_validate_ground(self) -> None:
        self.calls.append(("vision_control", {"action": "respawn_validate_ground"}))

    def capture_command_state(self) -> dict[str, Any]:
        return empty_command_state()
