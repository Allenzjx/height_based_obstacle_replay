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
from robot_ground_diagnostics import COLLIDER_RESOLUTION_FAILED  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402
from sim_worker_runtime import enrich_runtime_readiness  # noqa: E402


class RuntimeReadyMotionReadyTest(unittest.TestCase):
    def test_ground_unverified_does_not_clear_runtime_ready(self) -> None:
        status = enrich_runtime_readiness(
            {
                "ready": True,
                "phase": "running",
                "robot_ground": {"checked": True, "classification": COLLIDER_RESOLUTION_FAILED},
                "grounded_reference_valid": False,
                "vision": {"camera_ready": True, "camera_prim_path": "/World/Camera"},
            }
        )

        self.assertTrue(status["ready"])
        self.assertTrue(status["runtime_ready"])
        self.assertEqual(status["ground_state"], "UNVERIFIED")
        self.assertFalse(status["motion_ready"])
        self.assertIn("ground", status["motion_block_reason"])

    def test_controller_blocks_playback_but_allows_camera_and_validation_when_ground_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.sim_ready = True
            controller.latest_sim_status = enrich_runtime_readiness(
                {
                    "ready": True,
                    "runtime_ready": True,
                    "phase": "running",
                    "effective_headless": False,
                    "robot_ground": {"checked": True, "classification": COLLIDER_RESOLUTION_FAILED},
                    "grounded_reference_valid": False,
                    "vision": {
                        "camera_ready": True,
                        "camera_prim_path": "/World/WLRRobot/base_link/onboard_rgbd_camera",
                        "stable": True,
                        "detected_height_cm": 5,
                        "confidence": 0.95,
                        "detection_revision": 2,
                        "frame_revision": 2,
                    },
                    "scene_height_cm": 5,
                    "obstacle_revision": 1,
                }
            )
            controller.start_vision_task("generated")
            controller.vision_generated_height_cm = 5
            controller.vision_detection_baseline_revision = 1
            controller.vision_frame_baseline_revision = 1
            controller.vision_scene_obstacle_revision = 1

            self.assertFalse(controller.can_playback()[0])
            self.assertTrue(controller.camera_view_readiness()[0])
            self.assertTrue(controller.can_validate_current_generated_height()[0])


if __name__ == "__main__":
    unittest.main()
