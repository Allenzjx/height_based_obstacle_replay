from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_onboard_camera  # noqa: E402
from obstacle_height_vision import HeightDetection  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class Camera:
    def __init__(self):
        self.data = SimpleNamespace(
            output={
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "distance_to_image_plane": np.linspace(0.4, 1.2, 64, dtype=float).reshape(8, 8),
            },
            intrinsic_matrices=np.array([[120.0, 0.0, 4.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]], dtype=float),
            pos_w=np.array([0.0, 0.0, 0.2], dtype=float),
            quat_w_ros=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

    def update(self, _dt: float) -> None:
        return None


Camera.__module__ = "isaaclab.sensors.camera.camera"


class FakeTransport:
    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def validate_current_height(self, expected_height_cm: int) -> None:
        self.calls.append(("validate_current_height", expected_height_cm))

    def validate_camera(self) -> None:
        self.calls.append(("validate_camera", None))

    def save_rgbd_diagnostic(self, expected_height_cm: int | None = None) -> None:
        self.calls.append(("save_rgbd_diagnostic", expected_height_cm))

    def clear_validation_result(self) -> None:
        self.calls.append(("clear_validation_result", None))

    def set_vision_source_mode(self, source_mode: str) -> None:
        self.calls.append(("set_vision_source_mode", source_mode))


class CameraGroundTruthIsolationTest(unittest.TestCase):
    def test_expected_height_never_enters_estimate_height_from_depth(self) -> None:
        calls: list[dict[str, Any]] = []
        original = sim_onboard_camera.estimate_height_from_depth

        def spy(*args: Any, **kwargs: Any) -> HeightDetection:
            calls.append({"args": args, "kwargs": kwargs})
            return HeightDetection(
                valid=True,
                raw_height_cm=10.0,
                detected_height_cm=10,
                confidence=0.9,
                point_count=200,
                top_plane_mad_m=0.001,
                quantization_error_cm=0.0,
                reason="ok",
                timestamp=1.0,
            )

        sim_onboard_camera.estimate_height_from_depth = spy
        try:
            processor = sim_onboard_camera.OnboardCameraProcessor(self._scene())
            processor.request_height_validation(5)
            processor.update(dt=0.1, sim_time=0.1, wall_time=1.0)
        finally:
            sim_onboard_camera.estimate_height_from_depth = original
        self.assertEqual(len(calls), 1)
        self.assertNotIn("expected_height_cm", calls[0]["kwargs"])
        self.assertNotIn("height_cm", calls[0]["kwargs"])
        self.assertEqual(processor.last_detection.detected_height_cm, 10)

    def test_validate_current_height_does_not_mutate_scene_or_start_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(self._args(Path(tmp)))
            transport = FakeTransport()
            controller.transport = transport  # type: ignore[assignment]
            forbidden: list[str] = []

            def mark(name: str):
                def _inner(*_args: Any, **_kwargs: Any) -> None:
                    forbidden.append(name)
                    raise AssertionError(f"{name} must not be called")

                return _inner

            controller.generate_or_update_height_obstacle = mark("generate_or_update_height_obstacle")  # type: ignore[method-assign]
            controller.start_playback = mark("start_playback")  # type: ignore[method-assign]
            controller.replay_detected_height = mark("replay_detected_height")  # type: ignore[method-assign]
            controller.start_vision_task("generated_test_obstacle")
            controller.vision_generated_height_cm = 10
            controller.vision_detection_baseline_revision = 1
            controller.vision_frame_baseline_revision = 1
            controller.vision_scene_obstacle_revision = 1
            controller.latest_sim_status = {
                "scene_height_cm": 10,
                "obstacle_revision": 1,
                "vision": {
                    "camera_ready": True,
                    "stable": True,
                    "detected_height_cm": 10,
                    "raw_height_cm": 10.0,
                    "confidence": 0.98,
                    "detection_revision": 2,
                    "frame_revision": 2,
                },
            }
            ok = controller.validate_current_generated_height()
            self.assertTrue(ok)
            self.assertIn(("validate_current_height", 10), transport.calls)
            self.assertEqual(forbidden, [])

    def test_validation_is_blocked_without_vision_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(self._args(Path(tmp)))
            transport = FakeTransport()
            controller.transport = transport  # type: ignore[assignment]
            controller.vision_auto_replay_armed = True
            self.assertFalse(controller.validate_current_generated_height())
            self.assertEqual(transport.calls, [])

    def _scene(self) -> SimpleNamespace:
        config = SimpleNamespace(
            onboard_camera_enabled=True,
            camera_update_period_s=0.1,
            obstacle_x=1.55,
            camera_near_clip_m=0.05,
            camera_far_clip_m=6.0,
        )
        return SimpleNamespace(
            config=config,
            camera=Camera(),
            camera_error="",
            camera_parent_prim="/World/WLRRobot/base_link",
            camera_prim_path="/World/WLRRobot/base_link/onboard_rgbd_camera",
            stage=None,
        )

    def _args(self, store_root: Path) -> argparse.Namespace:
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
            vision_auto_replay=False,
            vision_confidence_threshold=0.75,
            vision_stable_frames=5,
            vision_window_size=7,
            vision_height_tolerance_cm=2.0,
            vision_auto_replay_cooldown_s=0.0,
            vision_respawn_before_replay=True,
            onboard_camera=True,
        )


if __name__ == "__main__":
    unittest.main()
