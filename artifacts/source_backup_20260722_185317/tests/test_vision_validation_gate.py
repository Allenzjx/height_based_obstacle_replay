from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for path in (MODULE_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from controller_test_utils import FakeTransport, make_args, motion_step  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class VisionValidationGateTest(unittest.TestCase):
    def test_steps_load_only_after_matching_validation_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = HeightReplayController(make_args(root, no_sim=True))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.store.save_steps(5, [motion_step(height_cm=5)])
            controller.store.save_steps(10, [motion_step(height_cm=10)])

            self.assertTrue(controller.start_vision_task("generated_test_obstacle"))
            self.assertTrue(controller.generate_vision_test_obstacle(5))
            controller.latest_sim_status = self._status(height=5, revision=1)
            controller.update()
            self.assertFalse(controller.vision_steps_ready)
            self.assertEqual(controller.get_visible_steps(), [])

            controller.validate_isaac_camera()
            controller.request_vision_detection_once()
            controller.update()
            self.assertFalse(controller.vision_steps_ready)

            controller.latest_sim_status = self._status(
                height=10,
                revision=2,
                validation={"checked": True, "passed": False, "expected_height_cm": 5, "detected_height_cm": 10, "detection_revision": 2},
            )
            controller.update()
            self.assertFalse(controller.vision_steps_ready)
            self.assertEqual(controller.get_visible_steps(), [])

            controller.latest_sim_status = self._status(
                height=5,
                revision=0,
                validation={"checked": True, "passed": True, "expected_height_cm": 5, "detected_height_cm": 5, "detection_revision": 0},
            )
            controller.update()
            self.assertFalse(controller.vision_steps_ready)

            controller.latest_sim_status = self._status(
                height=5,
                revision=3,
                validation={"checked": True, "passed": True, "expected_height_cm": 5, "detected_height_cm": 5, "detection_revision": 3},
            )
            controller.update()

            self.assertTrue(controller.vision_steps_ready)
            self.assertEqual(controller.vision_steps_height_cm, 5)
            self.assertIn("height_05cm", str(controller.vision_steps_path))
            self.assertNotIn("height_10cm", str(controller.vision_steps_path))
            self.assertEqual(len(controller.get_visible_steps()), 1)
            self.assertFalse(controller.playback.active)

    def test_validation_pass_with_armed_auto_replay_starts_vision_playback_only_after_steps_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = HeightReplayController(make_args(root, no_sim=False))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.sim_ready = True
            controller.vision_respawn_before_replay = False
            controller.store.save_steps(5, [motion_step(height_cm=5)])
            controller.current_height_cm = 10

            self.assertTrue(controller.start_vision_task("generated_test_obstacle"))
            controller.vision_generated_height_cm = 5
            controller.vision_detection_baseline_revision = 1
            controller.vision_frame_baseline_revision = 1
            controller.vision_scene_obstacle_revision = 1
            controller.vision_auto_replay_enabled = True
            controller.vision_auto_replay_armed = True
            controller.latest_sim_status = self._status(
                height=5,
                revision=2,
                validation={"checked": True, "passed": True, "expected_height_cm": 5, "detected_height_cm": 5, "detection_revision": 2},
            )

            controller.update()

            self.assertTrue(controller.vision_steps_ready)
            self.assertTrue(controller.playback.active)
            self.assertFalse(controller.vision_auto_replay_armed)
            self.assertEqual(controller.current_height_cm, 10)
            self.assertEqual(controller.manager.count, 0)

    def _status(self, *, height: int, revision: int, validation: dict | None = None) -> dict:
        vision = {
            "enabled": True,
            "camera_ready": True,
            "stable": True,
            "detected_height_cm": height,
            "raw_height_cm": 5.143 if height == 5 else float(height),
            "confidence": 0.9835,
            "stable_count": 7,
            "stable_required": 5,
            "detection_revision": revision,
            "frame_revision": revision,
        }
        if validation is not None:
            vision["height_validation"] = validation
        return {"scene_height_cm": 5, "obstacle_revision": 1, "last_set_height_request_id": "", "vision": vision}


if __name__ == "__main__":
    unittest.main()
