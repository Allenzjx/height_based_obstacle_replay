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


class CameraViewportWithoutMotionReadyTest(unittest.TestCase):
    def test_open_camera_viewport_sends_action_when_motion_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            transport = FakeTransport()
            controller.transport = transport  # type: ignore[assignment]
            controller.sim_ready = True
            controller.latest_sim_status = enrich_runtime_readiness(
                {
                    "ready": True,
                    "runtime_ready": True,
                    "effective_headless": False,
                    "robot_ground": {"checked": True, "classification": COLLIDER_RESOLUTION_FAILED},
                    "grounded_reference_valid": False,
                    "vision": {
                        "camera_ready": True,
                        "camera_prim_path": "/World/WLRRobot/base_link/onboard_rgbd_camera",
                    },
                }
            )

            self.assertTrue(controller.open_onboard_camera_viewport())

            actions = [call for call in transport.calls if call[0] == "vision_control"]
            self.assertEqual(actions[-1][1]["action"], "open_camera_viewport")
            self.assertFalse(any(call[0] == "respawn" for call in transport.calls))


if __name__ == "__main__":
    unittest.main()
