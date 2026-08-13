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


class VisionTaskIsolationTest(unittest.TestCase):
    def test_height_task_dirty_or_active_blocks_vision_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            controller.manager.adopt_steps([motion_step(height_cm=10)], dirty=True)
            ok, reason = controller.can_start_vision_task()
            self.assertFalse(ok)
            self.assertIn("unsaved", reason.lower())

            controller.manager.dirty = False
            controller.height_task_active = True
            ok, reason = controller.can_start_vision_task()
            self.assertFalse(ok)
            self.assertIn("height task", reason.lower())

    def test_vision_task_blocks_height_task_until_finished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            self.assertTrue(controller.start_vision_task("generated_test_obstacle"))
            with self.assertRaises(RuntimeError):
                controller.start_height_task("recording", [5])
            controller.finish_vision_task()
            controller.start_height_task("recording", [5])
            self.assertTrue(controller.height_task_active)

    def test_vision_generation_does_not_modify_height_manager_or_current_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.current_height_cm = 10
            original = motion_step(height_cm=10)
            controller.manager.adopt_steps([original], dirty=False)

            self.assertTrue(controller.start_vision_task("generated_test_obstacle"))
            self.assertTrue(controller.generate_vision_test_obstacle(5))

            self.assertEqual(controller.current_height_cm, 10)
            self.assertEqual(controller.manager.count, 1)
            self.assertFalse(controller.manager.dirty)
            self.assertEqual(controller.manager.steps[0]["name"], original["name"])
            self.assertEqual(controller.get_visible_steps(), [])
            self.assertEqual(controller.get_visible_steps_source(), "Vision - waiting for validation")

            controller.finish_vision_task()
            self.assertEqual(len(controller.get_visible_steps()), 1)
            self.assertEqual(controller.get_visible_steps_source(), "Height - 10cm")


if __name__ == "__main__":
    unittest.main()
