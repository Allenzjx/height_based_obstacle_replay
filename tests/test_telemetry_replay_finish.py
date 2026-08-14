from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from telemetry.collector import TelemetryCollector
from telemetry.config import RuntimeTelemetryConfig


class TelemetryReplayFinishTest(unittest.TestCase):
    def test_finish_replay_checkpoints_then_episode_exports_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RuntimeTelemetryConfig()
            cfg.telemetry.output_root = tmpdir
            cfg.telemetry.save_csv = False
            cfg.telemetry.save_npz = False
            cfg.telemetry.save_contacts = False
            cfg.telemetry.save_events = True
            collector = TelemetryCollector(cfg)
            run_dir = collector.start_episode(obstacle_height_cm=5, sequence_label="unit") or Path(tmpdir)

            collector.start_replay(label="unit replay", event_count=2, final_time_s=1.0, started_sim_time_s=0.0)
            collector.record_event(
                0.5,
                "unavailable_measurement",
                value=float("nan"),
                threshold=float("inf"),
                extra={"reason": "sensor unavailable"},
            )
            collector.finish_replay(success=True, sim_time_s=1.0)
            self.assertFalse((run_dir / "events.jsonl").exists())
            commits = list(
                (run_dir / ".telemetry_journal").glob("checkpoint-*/commit.json")
            )
            self.assertEqual(len(commits), 1)

            collector.finish_episode(success=True, reason="complete")

            events = [
                json.loads(
                    line,
                    parse_constant=lambda value: self.fail(
                        f"non-finite JSON constant {value}"
                    ),
                )
                for line in (run_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            finish_events = [row for row in events if row.get("event_type") == "replay_finished"]
            self.assertEqual(len(finish_events), 1)
            self.assertEqual(finish_events[0].get("message"), "Replay complete")
            unavailable = [
                row
                for row in events
                if row.get("event_type") == "unavailable_measurement"
            ]
            self.assertEqual(len(unavailable), 1)
            self.assertIsNone(unavailable[0]["value"])
            self.assertIsNone(unavailable[0]["threshold"])
            self.assertEqual(unavailable[0]["reason"], "sensor unavailable")


if __name__ == "__main__":
    unittest.main()
