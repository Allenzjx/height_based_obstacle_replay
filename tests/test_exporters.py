from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from telemetry.config import RuntimeTelemetryConfig  # noqa: E402
from telemetry.exporters import (  # noqa: E402
    create_run_dir,
    summarize_run,
    write_csv,
    write_json,
    write_jsonl,
    write_metadata,
    write_npz,
)
from telemetry.visualization.report_generator import generate_report  # noqa: E402


class ExportersTest(unittest.TestCase):
    def test_json_writers_replace_nonfinite_values_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "valid": False,
                "reason": "measurement unavailable",
                "missing": float("nan"),
                "nested": [
                    np.float64(float("inf")),
                    np.asarray([1.0, float("-inf")]),
                ],
                "path": root / "evidence.json",
            }
            write_json(root / "evidence.json", payload)
            write_jsonl(root / "evidence.jsonl", [payload])

            json_text = (root / "evidence.json").read_text(encoding="utf-8")
            jsonl_text = (root / "evidence.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("NaN", json_text + jsonl_text)
            self.assertNotIn("Infinity", json_text + jsonl_text)
            decoded = json.loads(json_text, parse_constant=self._reject_constant)
            decoded_row = json.loads(
                jsonl_text.strip(), parse_constant=self._reject_constant
            )
            for value in (decoded, decoded_row):
                self.assertIsNone(value["missing"])
                self.assertIsNone(value["nested"][0])
                self.assertIsNone(value["nested"][1][1])
                self.assertFalse(value["valid"])
                self.assertEqual(value["reason"], "measurement unavailable")

    def test_json_writers_preserve_existing_file_when_encoding_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, writer, payload in (
                (
                    "evidence.json",
                    write_json,
                    {1: "numeric", "1": "text"},
                ),
                (
                    "evidence.jsonl",
                    write_jsonl,
                    [{"ok": True}, {1: "numeric", "1": "text"}],
                ),
            ):
                path = root / name
                original = '{"sentinel":true}\n'
                path.write_text(original, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "keys collide"):
                    writer(path, payload)
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                self.assertEqual(list(root.glob(f".{name}.*.tmp")), [])

    def test_csv_nonfinite_representation_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "values.csv"
            write_csv(path, [{"value": float("nan")}])
            value = path.read_text(encoding="utf-8").splitlines()[1]
            self.assertTrue(math.isnan(float(value)))

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise AssertionError(f"non-finite JSON constant {value}")

    def test_export_files_and_summary_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, height_cm=5, sequence_label="unit test")
            cfg = RuntimeTelemetryConfig()
            cfg.visualization.live_enabled = False
            write_metadata(run_dir, {"run_label": "unit test"}, cfg)
            rows = [
                {"time_s": 0.0, "static_stability_margin_m": 0.1, "dynamic_stability_margin_m": 0.08, "equilibrium_stability_margin_m": 0.07, "sample_overhead_ms": 0.2},
                {"time_s": 0.1, "static_stability_margin_m": -0.01, "dynamic_stability_margin_m": -0.02, "equilibrium_stability_margin_m": 0.01, "sample_overhead_ms": 0.3},
            ]
            joint_rows = [{"time_s": 0.0, "joint_name": "joint_a", "torque_utilization": 0.4}]
            contact_rows = [{"time_s": 0.0, "body_name": "wheel", "normal_force_n": 5.0}]
            events = [{"simulation_time_s": 0.1, "severity": "warning", "event_type": "stability_margin_low"}]
            write_csv(run_dir / "telemetry_samples.csv", rows)
            write_csv(run_dir / "joint_timeseries.csv", joint_rows)
            write_csv(run_dir / "contacts.csv", contact_rows)
            write_jsonl(run_dir / "events.jsonl", events)
            write_npz(run_dir / "telemetry_timeseries.npz", rows)
            summary = summarize_run(rows, [], joint_rows, contact_rows, events, started_wall=time.time(), finished_wall=time.time())
            self.assertEqual(summary["sample_count"], 2)
            self.assertAlmostEqual(summary["min_static_margin_m"], -0.01)
            (run_dir / "stability_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            report = generate_report(run_dir)
            self.assertTrue(Path(report["dashboard"]).exists())
            self.assertTrue((run_dir / "telemetry_timeseries.npz").exists())
            self.assertTrue((run_dir / "data_quality.json").exists())
            self.assertTrue((run_dir / "static_baseline.json").exists())
            self.assertTrue((run_dir / "dynamic_consistency.json").exists())
            html = Path(report["dashboard"]).read_text(encoding="utf-8")
            self.assertIn("Data Quality Verdict", html)
            self.assertIn("overall_verdict", (run_dir / "data_quality.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
