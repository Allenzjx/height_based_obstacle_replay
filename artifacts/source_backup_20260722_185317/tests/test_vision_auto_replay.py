from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sequence_model import empty_command_state, make_event, make_step  # noqa: E402
from sim_ui_controller import HeightReplayController, MODE_E_STOP, MODE_RECORDING_STEP  # noqa: E402


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=10,
        sim_launch_mode="disabled",
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
    )


def motion_step(index: int = 1, *, height_cm: int = 20) -> dict[str, Any]:
    before = empty_command_state()
    event = make_event(0.0, "servo front_left_hip 10", kind="test")
    return make_step(
        index=index,
        step_type="recorded",
        duration=0.2,
        events=[event],
        command_state_before=before,
        command_state_after=before,
        name=f"vision_step_{height_cm}_{index}",
        note=f"height={height_cm}cm",
        extra={"height_cm": height_cm},
    )


class VisionAutoReplayTest(unittest.TestCase):
    def test_blocks_unsafe_states_and_missing_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._ready_controller(Path(tmp), saved_height=20)
            self._set_detection(controller, height=20, revision=1)

            controller.mode = MODE_RECORDING_STEP
            self.assertFalse(controller.can_auto_replay_detected_height(20, 1)[0])
            controller.mode = "TEST"

            controller.pending_step = motion_step(height_cm=10)
            self.assertFalse(controller.can_auto_replay_detected_height(20, 1)[0])
            controller.pending_step = None

            controller.pending_replacement = motion_step(height_cm=10)
            self.assertFalse(controller.can_auto_replay_detected_height(20, 1)[0])
            controller.pending_replacement = None

            controller.manager.dirty = True
            self.assertTrue(controller.can_auto_replay_detected_height(20, 1)[0])
            self.assertTrue(controller.vision_steps_ready)
            self.assertEqual(controller.manager.count, 0)
            controller.manager.dirty = False

            controller.servo_wheel_staging_active = True
            controller.servo_wheel_staged_dirty = True
            self.assertFalse(controller.can_auto_replay_detected_height(20, 1)[0])
            controller.servo_wheel_staged_dirty = False

            controller.mode = MODE_E_STOP
            self.assertFalse(controller.can_auto_replay_detected_height(20, 1)[0])
            controller.mode = "TEST"

            controller.playback.active = True
            self.assertFalse(controller.can_auto_replay_detected_height(20, 1)[0])
            controller.playback.active = False

            self.assertFalse(controller.can_auto_replay_detected_height(25, 2)[0])

    def test_replay_detected_height_uses_vision_steps_without_regenerating_obstacle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._ready_controller(Path(tmp), saved_height=20)
            self._set_detection(controller, height=20, revision=7)
            calls: list[str] = []

            def forbidden_generate() -> None:
                calls.append("generate")
                raise AssertionError("generate_or_update_height_obstacle must not be called")

            controller.generate_or_update_height_obstacle = forbidden_generate  # type: ignore[method-assign]

            ok = controller.replay_detected_height(20, 7)
            self.assertTrue(ok)
            self.assertEqual(calls, [])
            self.assertEqual(controller.current_height_cm, 10)
            self.assertEqual(controller.manager.count, 0)
            self.assertEqual(len(controller.vision_steps), 1)
            self.assertEqual(controller.vision_steps_height_cm, 20)
            self.assertEqual(controller.vision_last_consumed_detection_revision, 7)
            self.assertFalse(controller.vision_auto_replay_armed)
            self.assertTrue(controller.playback.active)
            self.assertFalse(controller.can_auto_replay_detected_height(20, 7)[0])

    def test_update_consumes_new_detection_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._ready_controller(Path(tmp), saved_height=20)
            self._set_detection(controller, height=20, revision=3)
            controller.update()
            self.assertEqual(controller.vision_last_consumed_detection_revision, 3)
            started_at = controller.vision_last_replay_started_at
            controller.playback.stop(silent=True)
            controller.vision_auto_replay_armed = True
            controller.update()
            self.assertEqual(controller.vision_last_consumed_detection_revision, 3)
            self.assertEqual(controller.vision_last_replay_started_at, started_at)

    def _ready_controller(self, root: Path, *, saved_height: int) -> HeightReplayController:
        controller = HeightReplayController(make_args(root))
        controller.no_sim = False
        controller.sim_ready = True
        controller.vision_auto_replay_enabled = True
        controller.vision_auto_replay_armed = True
        controller.store.save_steps(saved_height, [motion_step(height_cm=saved_height)])
        return controller

    def _set_detection(self, controller: HeightReplayController, *, height: int, revision: int) -> None:
        controller.latest_sim_status["vision"] = {
            "enabled": True,
            "camera_ready": True,
            "stable": True,
            "detection_revision": revision,
            "detected_height_cm": height,
            "raw_height_cm": float(height),
            "confidence": 0.92,
            "stable_count": 5,
            "stable_required": 5,
            "failure_reason": "",
        }


if __name__ == "__main__":
    unittest.main()
