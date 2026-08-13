from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for path in (MODULE_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from controller_test_utils import FakeTransport, make_args, motion_step  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class FakeSimClient:
    def __init__(self) -> None:
        self.set_height_calls: list[tuple[int, dict[str, Any]]] = []

    def set_height(self, height_cm: int, **payload: Any) -> None:
        self.set_height_calls.append((int(height_cm), dict(payload)))


class VisionObstacleGenerationTest(unittest.TestCase):
    def test_generated_obstacle_uses_existing_set_height_payload_without_height_task_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.sim_ready = True
            client = FakeSimClient()
            controller.sim_client = client  # type: ignore[assignment]
            controller.current_height_cm = 10
            controller.manager.adopt_steps([motion_step(height_cm=10)], dirty=False)
            controller.latest_sim_status = {
                "obstacle_revision": 4,
                "vision": {"detection_revision": 9, "frame_revision": 11},
            }

            self.assertTrue(controller.start_vision_task("generated_test_obstacle"))
            self.assertTrue(controller.generate_vision_test_obstacle(5))

            self.assertEqual(controller.current_height_cm, 10)
            self.assertEqual(controller.manager.count, 1)
            self.assertFalse(controller.manager.dirty)
            self.assertEqual(len(client.set_height_calls), 1)
            height, payload = client.set_height_calls[0]
            self.assertEqual(height, 5)
            self.assertEqual(payload["source"], "vision_task")
            self.assertEqual(payload["generation_revision"], controller.vision_generation_revision)
            self.assertTrue(str(payload["request_id"]).startswith("vision-"))
            self.assertEqual(controller.vision_detection_baseline_revision, 9)
            self.assertEqual(controller.vision_frame_baseline_revision, 11)
            self.assertEqual(controller.vision_scene_obstacle_revision, 5)
            self.assertFalse(controller.vision_steps_ready)

    def test_invalid_or_external_generation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            self.assertTrue(controller.start_vision_task("external"))
            self.assertFalse(controller.generate_vision_test_obstacle(5))
            controller.finish_vision_task()
            self.assertTrue(controller.start_vision_task("generated"))
            self.assertFalse(controller.generate_vision_test_obstacle(7))


if __name__ == "__main__":
    unittest.main()
