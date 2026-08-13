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

from controller_test_utils import FakeTransport, make_args  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class ValidationReadinessReasonTest(unittest.TestCase):
    def test_specific_reasons_before_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.transport = FakeTransport()  # type: ignore[assignment]

            self.assertEqual(controller.can_validate_current_generated_height()[1], "Start Vision Task first.")
            controller.start_vision_task("generated")
            self.assertEqual(controller.can_validate_current_generated_height()[1], "Generate a Vision test obstacle first.")

            controller.vision_generated_height_cm = 5
            controller.latest_sim_status = {"vision": {"camera_ready": False}}
            self.assertEqual(controller.can_validate_current_generated_height()[1], "Camera health not ready.")

            controller.latest_sim_status = {
                "scene_height_cm": 5,
                "obstacle_revision": 1,
                "vision": {
                    "camera_ready": True,
                    "stable": True,
                    "detected_height_cm": 5,
                    "confidence": 0.95,
                    "detection_revision": 1,
                    "frame_revision": 0,
                },
            }
            controller.vision_detection_baseline_revision = 0
            controller.vision_frame_baseline_revision = 0
            controller.vision_scene_obstacle_revision = 1
            self.assertEqual(controller.can_validate_current_generated_height()[1], "Waiting for a new Camera frame.")

    def test_ready_after_scene_new_frame_new_detection_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.start_vision_task("generated")
            controller.vision_generated_height_cm = 5
            controller.vision_detection_baseline_revision = 1
            controller.vision_frame_baseline_revision = 1
            controller.vision_scene_obstacle_revision = 2
            controller.latest_sim_status = {
                "scene_height_cm": 5,
                "obstacle_revision": 2,
                "last_set_height_request_id": "",
                "vision": {
                    "camera_ready": True,
                    "stable": True,
                    "detected_height_cm": 5,
                    "confidence": 0.95,
                    "detection_revision": 2,
                    "frame_revision": 2,
                },
            }

            self.assertEqual(controller.can_validate_current_generated_height(), (True, ""))


if __name__ == "__main__":
    unittest.main()
