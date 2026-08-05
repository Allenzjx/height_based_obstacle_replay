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
        self.wheel_generation = 0

    def stop_wheels(self, *, reason: str = "test_stop") -> dict[str, Any]:
        self.wheel_generation += 1
        result = {
            "command_id": f"test-stop-{self.wheel_generation}",
            "wheel_generation": self.wheel_generation,
            "reason": reason,
            "high_priority": True,
        }
        self.calls.append(("stop_wheels", dict(result)))
        return result

    def respawn(self) -> None:
        self.calls.append(("respawn", {}))

    def request_state(self) -> None:
        self.calls.append(("request_state", {}))

    def capture_command_state(self) -> dict[str, Any]:
        return empty_command_state()

    def status(self) -> dict[str, Any]:
        return {"fake": True, "calls": len(self.calls)}
