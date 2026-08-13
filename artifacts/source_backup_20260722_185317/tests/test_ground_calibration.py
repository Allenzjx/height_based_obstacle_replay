from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_worker_runtime import handle_vision_control  # noqa: E402


class CalibrationAdapter:
    def __init__(self) -> None:
        self.calibrated = 0
        self.respawned = 0

    def calibrate_grounded_reference(self) -> dict:
        self.calibrated += 1
        return {"grounded_reference_valid": True, "grounded_reference_diagnostics": {"classification": "OK"}}

    def respawn_robot(self, *, settle: bool = True) -> dict:
        self.respawned += 1
        return {"ok": True, "respawned": True}


class GroundCalibrationTest(unittest.TestCase):
    def test_calibration_action_does_not_respawn(self) -> None:
        adapter = CalibrationAdapter()

        result = handle_vision_control(
            args=SimpleNamespace(headless=False),
            adapter=adapter,
            scene_handle=SimpleNamespace(),
            vision_processor=None,
            action="calibrate_ground_reference",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(adapter.calibrated, 1)
        self.assertEqual(adapter.respawned, 0)


if __name__ == "__main__":
    unittest.main()
