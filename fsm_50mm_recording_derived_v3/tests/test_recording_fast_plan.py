from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from command_model import WHEEL_JOINT_NAMES

from fsm_50mm_recording_derived_v3.recording_fast_plan import write_fast_plan


class RecordingFastPlanExportTests(unittest.TestCase):
    @staticmethod
    def _steps() -> list[dict]:
        return [
            {
                "index": 1,
                "name": "deterministic-export-step",
                "duration": 0.5,
                "events": [
                    {
                        "time": 0.0,
                        "command": "servo front_left_hip 15",
                        "kind": "command",
                    },
                    {
                        "time": 0.2,
                        "command": "wheel all 1",
                        "kind": "command",
                    },
                ],
                "command_state_before": {
                    "servos": {},
                    "wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES},
                },
            }
        ]

    def test_repeated_writes_omit_build_clock_but_preserve_live_plan_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch(
                "playback.time.monotonic",
                side_effect=(100.0, 101.0, 200.0, 201.0),
            ):
                first = write_fast_plan(
                    output_dir=root / "first",
                    source_version="v003_test",
                    steps=self._steps(),
                    max_wheel_speed=3.0,
                )
                second = write_fast_plan(
                    output_dir=root / "second",
                    source_version="v003_test",
                    steps=self._steps(),
                    max_wheel_speed=3.0,
                )

            first_bytes = first["json_path"].read_bytes()
            second_bytes = second["json_path"].read_bytes()
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                hashlib.sha256(first_bytes).hexdigest(),
                hashlib.sha256(second_bytes).hexdigest(),
            )

            first_payload = json.loads(first_bytes)
            second_payload = json.loads(second_bytes)
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(first["plan"].plan_sha256, second["plan"].plan_sha256)
            self.assertEqual(first["rows"], second["rows"])
            self.assertEqual(first_payload["segments"], first["rows"])

            self.assertEqual(100.0, first["plan"].timing["plan_build_start"])
            self.assertEqual(101.0, first["plan"].timing["plan_build_end"])
            self.assertEqual(200.0, second["plan"].timing["plan_build_start"])
            self.assertEqual(201.0, second["plan"].timing["plan_build_end"])
            self.assertNotIn("plan_build_start", first_payload["timing"])
            self.assertNotIn("plan_build_end", first_payload["timing"])
            self.assertEqual(
                first_payload["timing"],
                {
                    key: value
                    for key, value in first["plan"].timing.items()
                    if key not in {"plan_build_start", "plan_build_end"}
                },
            )
            self.assertEqual(
                second_payload["timing"],
                {
                    key: value
                    for key, value in second["plan"].timing.items()
                    if key not in {"plan_build_start", "plan_build_end"}
                },
            )


if __name__ == "__main__":
    unittest.main()
