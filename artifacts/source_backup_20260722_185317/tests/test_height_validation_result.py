from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from camera_validation import compare_detection_to_expected  # noqa: E402


def valid_vision_status(**overrides):
    status = {
        "camera_ready": True,
        "frame_revision": 3442,
        "detection_revision": 2,
        "raw_height_cm": 5.143,
        "detected_height_cm": 5,
        "candidate_height_cm": 5,
        "confidence": 0.9804,
        "stable": True,
        "stable_count": 7,
        "stable_required": 5,
        "valid_point_count": 18731,
        "top_plane_mad_m": 6.72e-05,
        "quantization_error_cm": 0.143,
        "camera_health": {
            "checked": True,
            "ok": True,
            "frames_advanced_during_check": 3,
            "stale_frame": False,
        },
    }
    status.update(overrides)
    return status


class HeightValidationResultTest(unittest.TestCase):
    def test_current_5cm_case_passes(self) -> None:
        result = compare_detection_to_expected(valid_vision_status(), 5)
        self.assertTrue(result.passed, result)
        self.assertAlmostEqual(result.absolute_error_cm or 0.0, 0.143, places=3)
        self.assertAlmostEqual(result.absolute_error_mm or 0.0, 1.43, places=2)
        self.assertAlmostEqual(result.relative_error_percent or 0.0, 2.86, places=2)

    def test_raw_height_outside_tolerance_and_no_detected_fails(self) -> None:
        result = compare_detection_to_expected(
            valid_vision_status(raw_height_cm=7.1, detected_height_cm=None, candidate_height_cm=None, stable=False),
            5,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("raw height error" in reason for reason in result.reasons))
        self.assertTrue(any("stable detected height missing" in reason for reason in result.reasons))

    def test_detected_bucket_mismatch_fails(self) -> None:
        result = compare_detection_to_expected(valid_vision_status(detected_height_cm=10), 5)
        self.assertFalse(result.passed)
        self.assertTrue(any("detected height 10cm" in reason for reason in result.reasons))

    def test_low_confidence_fails(self) -> None:
        result = compare_detection_to_expected(valid_vision_status(confidence=0.4), 5)
        self.assertFalse(result.passed)
        self.assertTrue(any("confidence" in reason for reason in result.reasons))

    def test_unstable_fails(self) -> None:
        result = compare_detection_to_expected(valid_vision_status(stable=False, stable_count=3), 5)
        self.assertFalse(result.passed)
        self.assertTrue(any("not stable" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
