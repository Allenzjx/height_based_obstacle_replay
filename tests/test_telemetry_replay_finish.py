from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from telemetry.collector import TelemetryCollector
from telemetry.config import RuntimeTelemetryConfig


class TelemetryReplayFinishTest(unittest.TestCase):
    def test_finish_replay_flushes_replay_finished_event(self) -> None:
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
            collector.finish_replay(success=True, sim_time_s=1.0)

            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            finish_events = [row for row in events if row.get("event_type") == "replay_finished"]
            self.assertEqual(len(finish_events), 1)
            self.assertEqual(finish_events[0].get("message"), "Replay complete")


if __name__ == "__main__":
    unittest.main()
