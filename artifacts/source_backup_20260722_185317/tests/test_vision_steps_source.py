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

from controller_test_utils import make_args, motion_step  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class VisionStepsSourceTest(unittest.TestCase):
    def test_visible_steps_switch_between_height_and_vision_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            controller.manager.adopt_steps([motion_step(height_cm=10)], dirty=True)

            self.assertEqual(controller.active_steps_view(), "height")
            self.assertEqual(len(controller.get_visible_steps()), 1)
            self.assertFalse(controller.visible_steps_are_read_only())

            controller.manager.dirty = False
            self.assertTrue(controller.start_vision_task("generated_test_obstacle"))
            self.assertEqual(controller.active_steps_view(), "vision")
            self.assertEqual(controller.get_visible_steps(), [])
            self.assertEqual(controller.get_visible_steps_source(), "Vision - waiting for validation")
            self.assertTrue(controller.visible_steps_are_read_only())
            self.assertFalse(controller.can_save()[0])
            self.assertFalse(controller.can_combine()[0])
            self.assertFalse(controller.can_prepare_replacement()[0])

            controller.store.save_steps(5, [motion_step(height_cm=5)])
            count = controller.load_validated_vision_steps(5, detection_revision=4)
            self.assertEqual(count, 1)
            self.assertEqual(len(controller.get_visible_steps()), 1)
            self.assertEqual(controller.get_visible_steps_source(), "Vision - detected 5cm")
            self.assertEqual(controller.manager.count, 1)
            self.assertFalse(controller.manager.dirty)

            controller.finish_vision_task()
            self.assertEqual(controller.active_steps_view(), "height")
            self.assertEqual(len(controller.get_visible_steps()), 1)
            self.assertEqual(controller.get_visible_steps_source(), "Height - 10cm")

    def test_missing_vision_steps_file_keeps_ready_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            controller.start_vision_task("generated")
            count = controller.load_validated_vision_steps(25, detection_revision=2)
            self.assertEqual(count, 0)
            self.assertFalse(controller.vision_steps_ready)
            self.assertIn("No saved steps found for 25cm", controller.vision_steps_error)


if __name__ == "__main__":
    unittest.main()
