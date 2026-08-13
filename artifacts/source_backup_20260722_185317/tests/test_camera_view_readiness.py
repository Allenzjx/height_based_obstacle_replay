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


class CameraViewReadinessTest(unittest.TestCase):
    def test_ground_unverified_does_not_block_camera_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.transport = FakeTransport()  # type: ignore[assignment]
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
            controller.sim_ready = True

            ok, reason = controller.camera_view_readiness()

            self.assertTrue(ok, reason)

    def test_nonzero_wheel_command_blocks_camera_view_idle_guard(self) -> None:
        class MovingTransport(FakeTransport):
            def capture_command_state(self) -> dict:
                state = super().capture_command_state()
                state["wheels"]["front_left_ankle"] = 0.1
                return state

        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.transport = MovingTransport()  # type: ignore[assignment]
            controller.latest_sim_status = {
                "ready": True,
                "runtime_ready": True,
                "effective_headless": False,
                "vision": {
                    "camera_ready": True,
                    "camera_prim_path": "/World/WLRRobot/base_link/onboard_rgbd_camera",
                },
            }
            controller.sim_ready = True

            ok, reason = controller.camera_view_readiness()

            self.assertFalse(ok)
            self.assertIn("wheel", reason)


if __name__ == "__main__":
    unittest.main()
