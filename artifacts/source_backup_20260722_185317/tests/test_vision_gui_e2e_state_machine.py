from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from vision_gui_e2e import VisionGuiE2ERunner  # noqa: E402


class FakeRoot:
    def __init__(self) -> None:
        self.callbacks: list[object] = []
        self.destroyed = False

    def after(self, _delay_ms: int, callback: object) -> None:
        self.callbacks.append(callback)

    def destroy(self) -> None:
        self.destroyed = True

    def pump(self, runner: VisionGuiE2ERunner, limit: int = 200) -> None:
        count = 0
        while self.callbacks and not runner._stopping and count < limit:
            callback = self.callbacks.pop(0)
            callback()
            count += 1


class FakePlayback:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self, *_args: object) -> None:
        self.stop_calls += 1


class FakeController:
    def __init__(self) -> None:
        self.runtime_ready = True
        self.vision_respawn_before_replay = True
        self.playback = FakePlayback()
        self.calls: list[str] = []
        self.shutdown_called = False

    def start_vision_task(self, source_mode: str | None = None) -> bool:
        self.calls.append(f"start_vision_task:{source_mode}")
        return True

    def generate_vision_test_obstacle(self, height_cm: int) -> bool:
        self.calls.append(f"generate:{height_cm}")
        return True

    def set_vision_enabled(self, enabled: bool) -> None:
        self.calls.append(f"set_vision_enabled:{enabled}")

    def validate_current_generated_height(self) -> bool:
        self.calls.append("validate_height")
        return True

    def validate_robot_ground_contact(self) -> bool:
        self.calls.append("validate_ground")
        return True

    def calibrate_ground_reference(self) -> bool:
        self.calls.append("calibrate_ground")
        return True

    def playback_readiness(self, *, respawn_first: bool = False) -> tuple[bool, str]:
        self.calls.append(f"playback_readiness:{respawn_first}")
        return False, "motion is not ready"

    def replay_validated_vision_steps(self, height_cm: int, detection_revision: int, *, require_auto_replay: bool = False) -> bool:
        self.calls.append(f"replay:{height_cm}:{detection_revision}:{require_auto_replay}")
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def snapshot(self) -> dict[str, object]:
        return {
            "vision_scene_ready": True,
            "vision_steps_ready": True,
            "vision_steps_count": 35,
            "playback_block_reason": "motion is not ready",
            "vision": {
                "scene_ready": True,
                "stable": True,
                "detected_height_cm": 5,
                "confidence": 0.98,
                "detection_revision": 7,
                "height_validation": {"checked": True, "passed": True},
                "camera_view": {"pending": False},
            },
            "robot_ground": {
                "checked": True,
                "classification": "OK",
                "physical_ground_safe": True,
                "reasons": [],
            },
        }


class FakeUi:
    def __init__(self) -> None:
        self.root = FakeRoot()
        self.closed = False

    def _window_close(self) -> None:
        self.closed = True
        self.root.destroy()


class VisionGuiE2EStateMachineTest(unittest.TestCase):
    def test_runner_uses_controller_methods_and_reports_playback_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ui = FakeUi()
            controller = FakeController()
            args = SimpleNamespace(
                e2e_height_cm=5,
                e2e_timeout_s=30.0,
                e2e_output=str(Path(tmpdir) / "e2e.json"),
                e2e_open_camera_viewport=False,
                e2e_test_camera_fallback=False,
                e2e_playback_probe_s=0.0,
                e2e_keep_open_on_failure=False,
                e2e_save_screenshots=False,
                vision_confidence_threshold=0.75,
            )
            runner = VisionGuiE2ERunner(ui, controller, args)

            runner.start()
            ui.root.pump(runner)

            self.assertIn("start_vision_task:generated", controller.calls)
            self.assertIn("generate:5", controller.calls)
            self.assertIn("set_vision_enabled:True", controller.calls)
            self.assertIn("validate_height", controller.calls)
            self.assertIn("validate_ground", controller.calls)
            self.assertIn("calibrate_ground", controller.calls)
            self.assertIn("playback_readiness:True", controller.calls)
            self.assertNotIn("replay:5:7:False", controller.calls)
            self.assertTrue(ui.closed)
            self.assertEqual(runner.exit_code, 1)
            self.assertTrue(Path(args.e2e_output).exists())
            text = Path(args.e2e_output).read_text(encoding="utf-8")
            self.assertIn("PLAYBACK_PROBE", text)
            self.assertIn("motion is not ready", text)


if __name__ == "__main__":
    unittest.main()
