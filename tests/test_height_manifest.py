from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_manifest import (  # noqa: E402
    HEIGHT_ERROR_MESSAGE,
    SUPPORTED_HEIGHTS_CM,
    HeightManifest,
    HeightValidationError,
    height_folder_name,
    normalize_height_cm,
    obstacle_height_m,
)


class HeightManifestTest(unittest.TestCase):
    def test_height_validation(self) -> None:
        self.assertEqual(normalize_height_cm("10"), 10)
        self.assertEqual(obstacle_height_m(10), 0.10)
        self.assertEqual(height_folder_name(5), "height_05cm")
        with self.assertRaisesRegex(HeightValidationError, HEIGHT_ERROR_MESSAGE):
            normalize_height_cm(12)
        with self.assertRaisesRegex(HeightValidationError, HEIGHT_ERROR_MESSAGE):
            normalize_height_cm(45)

    def test_manifest_ensure_creates_supported_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = HeightManifest(Path(tmp) / "manifest.json")
            data = manifest.ensure()
            self.assertEqual(data["supported_heights_cm"], list(SUPPORTED_HEIGHTS_CM))
            for height_cm in SUPPORTED_HEIGHTS_CM:
                self.assertTrue((Path(tmp) / height_folder_name(height_cm)).is_dir())
                entry = data["heights"][str(height_cm)]
                self.assertFalse(entry["recorded"])
                self.assertEqual(entry["step_count"], 0)

    def test_update_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = HeightManifest(Path(tmp) / "manifest.json")
            manifest.ensure()
            data = manifest.update_height(10, step_count=3, saved_at="2026-01-01T00:00:00-05:00")
            entry = data["heights"]["10"]
            self.assertTrue(entry["recorded"])
            self.assertEqual(entry["step_count"], 3)
            self.assertTrue(entry["steps_path"].endswith(r"height_10cm\accepted_steps.jsonl") or entry["steps_path"].endswith("height_10cm/accepted_steps.jsonl"))


if __name__ == "__main__":
    unittest.main()

