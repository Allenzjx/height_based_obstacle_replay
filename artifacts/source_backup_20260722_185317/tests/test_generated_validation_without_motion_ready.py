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
from robot_ground_diagnostics import COLLIDER_RESOLUTION_FAILED  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402
from sim_worker_runtime import enrich_runtime_readiness  # noqa: E402


class GeneratedValidationWithoutMotionReadyTest(unittest.TestCase):
    def test_validation_and_step_load_do_not_require_motion_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = HeightReplayController(make_args(root, no_sim=False))
            transport = FakeTransport()
            controller.transport = transport  # type: ignore[assignment]
            controller.store.save_steps(5, [motion_step(height_cm=5)])
            controller.sim_ready = True
            controller.latest_sim_status = enrich_runtime_readiness(
                {
                    "ready": True,
                    "runtime_ready": True,
                    "phase": "running",
                    "robot_ground": {"checked": True, "classification": COLLIDER_RESOLUTION_FAILED},
                    "grounded_reference_valid": False,
                    "scene_height_cm": 5,
                    "obstacle_revision": 2,
                    "last_set_height_request_id": "",
                    "vision": {
                        "camera_ready": True,
                        "stable": True,
                        "detected_height_cm": 5,
                        "confidence": 0.96,
                        "detection_revision": 3,
                        "frame_revision": 3,
                        "height_validation": {
                            "checked": True,
                            "passed": True,
                            "expected_height_cm": 5,
                            "detected_height_cm": 5,
                            "detection_revision": 3,
                        },
                    },
                }
            )

            self.assertTrue(controller.start_vision_task("generated"))
            controller.vision_generated_height_cm = 5
            controller.vision_detection_baseline_revision = 1
            controller.vision_frame_baseline_revision = 1
            controller.vision_scene_obstacle_revision = 2
            self.assertTrue(controller.can_validate_current_generated_height()[0])
            controller.update()

            self.assertTrue(controller.vision_steps_ready)
            self.assertFalse(controller.can_playback()[0])


if __name__ == "__main__":
    unittest.main()
