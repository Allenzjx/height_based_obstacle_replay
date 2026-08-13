from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker_runtime  # noqa: E402
from robot_ground_diagnostics import COLLIDER_RESOLUTION_FAILED, default_robot_ground_diagnostics  # noqa: E402


class AdapterWithoutMotionReady:
    grounded_reference_valid = False
    robot_ground_diagnostics = {
        **default_robot_ground_diagnostics("resolver failed"),
        "checked": True,
        "classification": COLLIDER_RESOLUTION_FAILED,
    }


class VisionProcessor:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_filter(self) -> None:
        self.reset_count += 1


class VisionObstacleUpdateRespawnSplitTest(unittest.TestCase):
    def test_if_motion_ready_keeps_scene_ready_when_respawn_is_skipped(self) -> None:
        calls: list[float] = []
        original_update = sim_worker_runtime.update_obstacle_height
        sim_worker_runtime.update_obstacle_height = lambda _scene, height_m: calls.append(float(height_m))  # type: ignore[assignment]
        try:
            vision = VisionProcessor()
            result = sim_worker_runtime.handle_set_height(
                adapter=AdapterWithoutMotionReady(),
                scene_handle=SimpleNamespace(),
                vision_processor=vision,
                height_cm=5,
                source="vision_task",
                request_id="req",
                obstacle_revision=9,
                respawn_policy="if_motion_ready",
            )
        finally:
            sim_worker_runtime.update_obstacle_height = original_update  # type: ignore[assignment]

        self.assertTrue(result["ok"])
        self.assertTrue(result["obstacle_updated"])
        self.assertTrue(result["scene_ready"])
        self.assertEqual(result["scene_height_cm"], 5)
        self.assertTrue(result["respawn_requested"])
        self.assertFalse(result["respawned"])
        self.assertFalse(result["motion_ready"])
        self.assertIn("motion_ready=false", result["respawn_warning"])
        self.assertEqual(vision.reset_count, 1)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
