from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from vision_task_state import (  # noqa: E402
    PHASE_INACTIVE,
    PHASE_READY,
    VISION_SOURCE_EXTERNAL,
    VISION_SOURCE_GENERATED,
    VisionTaskState,
    inactive_state,
    normalize_source_mode,
    ready_state,
)


class VisionTaskStateTest(unittest.TestCase):
    def test_ready_and_inactive_states_are_pure_data(self) -> None:
        inactive = inactive_state()
        ready = ready_state(VISION_SOURCE_EXTERNAL)

        self.assertFalse(inactive.active)
        self.assertEqual(inactive.phase, PHASE_INACTIVE)
        self.assertTrue(ready.active)
        self.assertEqual(ready.phase, PHASE_READY)
        self.assertEqual(ready.source_mode, VISION_SOURCE_EXTERNAL)
        self.assertIsInstance(VisionTaskState().to_dict(), dict)

    def test_source_mode_normalization(self) -> None:
        self.assertEqual(normalize_source_mode("External / Unknown Obstacle"), VISION_SOURCE_EXTERNAL)
        self.assertEqual(normalize_source_mode("external_unknown"), VISION_SOURCE_EXTERNAL)
        self.assertEqual(normalize_source_mode("Generated Test Obstacle"), VISION_SOURCE_GENERATED)


if __name__ == "__main__":
    unittest.main()
