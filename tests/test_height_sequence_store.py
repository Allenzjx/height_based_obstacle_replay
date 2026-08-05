from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_sequence_store import HeightSequenceStore  # noqa: E402
from sequence_model import empty_command_state, make_event, make_step  # noqa: E402


def sample_step() -> dict:
    before = empty_command_state()
    event = make_event(0.0, "servo front_left_hip 10", command_state_before=before)
    after = empty_command_state()
    after["servos"]["front_left_hip"] = 10.0
    return make_step(
        index=1,
        step_type="recorded",
        duration=0.25,
        events=[event],
        command_state_before=before,
        command_state_after=after,
        note="test",
    )


class HeightSequenceStoreTest(unittest.TestCase):
    def test_refresh_manifest_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HeightSequenceStore(Path(tmpdir))
            manifest_path = Path(tmpdir) / "manifest.json"
            before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            snapshot = store.refresh_manifest()

            after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(snapshot["heights"]["5"]["step_count"], 0)

    def test_save_load_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightSequenceStore(Path(tmp))
            path = store.save_steps(10, [sample_step()])
            self.assertTrue(path.exists())
            loaded = store.load_steps(10)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["command_state_after"]["servos"]["front_left_hip"], 10.0)
            entry = store.manifest.entry(10)
            self.assertTrue(entry["recorded"])
            self.assertEqual(entry["step_count"], 1)

    def test_new_empty_sequence_does_not_overwrite_until_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightSequenceStore(Path(tmp))
            path = store.new_empty_sequence(5, write_file=False)
            self.assertFalse(path.exists())
            path = store.new_empty_sequence(5, write_file=True)
            self.assertTrue(path.exists())
            self.assertEqual(store.load_steps(5), [])
            self.assertFalse(store.manifest.entry(5)["recorded"])

    def test_invalid_height_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightSequenceStore(Path(tmp))
            with self.assertRaises(ValueError):
                store.save_steps(12, [])


if __name__ == "__main__":
    unittest.main()
