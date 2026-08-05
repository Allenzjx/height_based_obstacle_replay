from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from telemetry.physical_success import analyze_physical_success  # noqa: E402


class PhysicalSuccessAnalysisTest(unittest.TestCase):
    def test_passes_when_replay_crosses_obstacle_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_run(Path(tmpdir), final_x=0.70, max_x=0.72, z_peak=0.14, finished=True, last_index=11, duration=12.0)

            result = analyze_physical_success(run_dir, expected_height_cm=5, expected_event_count=12, expected_final_time_s=11.5)

            self.assertTrue(result["ok"], result["failure_reasons"])
            self.assertEqual(result["verdict"], "PASS")
            self.assertTrue((run_dir / "physical_success.json").exists())

    def test_fails_when_replay_does_not_clear_obstacle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_run(Path(tmpdir), final_x=0.10, max_x=0.12, z_peak=0.14, finished=True, last_index=11, duration=12.0)

            result = analyze_physical_success(run_dir, expected_height_cm=5, expected_event_count=12, expected_final_time_s=11.5)

            self.assertFalse(result["ok"])
            self.assertTrue(any("front_of_robot_reached_obstacle" in reason for reason in result["failure_reasons"]))

    def test_contact_unavailable_does_not_override_pose_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_run(
                Path(tmpdir),
                final_x=0.70,
                max_x=0.72,
                z_peak=0.14,
                finished=True,
                last_index=11,
                duration=12.0,
                active_contacts=0,
                data_quality_warnings=["contact normal forces unavailable"],
            )

            result = analyze_physical_success(run_dir, expected_height_cm=5, expected_event_count=12, expected_final_time_s=11.5)

            self.assertTrue(result["ok"], result["failure_reasons"])
            self.assertTrue(result["data_quality"]["contact_data_unavailable"])

    def test_valid_contact_data_without_support_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_run(
                Path(tmpdir),
                final_x=0.70,
                max_x=0.72,
                z_peak=0.14,
                finished=True,
                last_index=11,
                duration=12.0,
                active_contacts=0,
            )

            result = analyze_physical_success(run_dir, expected_height_cm=5, expected_event_count=12, expected_final_time_s=11.5)

            self.assertFalse(result["ok"])
            self.assertTrue(any("contact_support_observed_or_unavailable" in reason for reason in result["failure_reasons"]))

    def test_finish_event_duration_covers_plan_when_last_sample_precedes_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_run(
                Path(tmpdir),
                final_x=0.70,
                max_x=0.72,
                z_peak=0.14,
                finished=True,
                last_index=11,
                duration=12.0,
                sample_duration=10.9,
            )

            result = analyze_physical_success(run_dir, expected_height_cm=5, expected_event_count=12, expected_final_time_s=11.5)

            self.assertTrue(result["ok"], result["failure_reasons"])
            duration_check = next(row for row in result["checks"] if row["name"] == "sim_duration_covers_plan")
            self.assertEqual(duration_check["actual"]["events"], 12.0)

    def _make_run(
        self,
        root: Path,
        *,
        final_x: float,
        max_x: float,
        z_peak: float,
        finished: bool,
        last_index: int,
        duration: float,
        sample_duration: float | None = None,
        active_contacts: int = 4,
        data_quality_warnings: list[str] | None = None,
    ) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "obstacle_height_cm": 5,
                    "obstacle_x_m": 1.55,
                    "obstacle_length_m": 1.65,
                    "robot_length_m": 0.55,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "stability_summary.json").write_text(json.dumps({"sample_count": 40}), encoding="utf-8")
        (run_dir / "data_quality.json").write_text(json.dumps({"overall_verdict": "PASS", "warnings": data_quality_warnings or []}), encoding="utf-8")
        rows = []
        sample_span = duration if sample_duration is None else float(sample_duration)
        for index in range(40):
            frac = index / 39.0
            base_x = min(max_x, final_x * frac)
            if index == 39:
                base_x = final_x
            rows.append(
                {
                    "time_s": sample_span * frac,
                    "base_x_m": base_x,
                    "base_z_m": 0.10 + max(0.0, z_peak - 0.10) * (1.0 - abs(0.5 - frac) * 2.0),
                    "base_roll_rad": 0.05,
                    "base_pitch_rad": 0.04,
                    "static_stability_margin_m": 0.04,
                    "active_contact_count": active_contacts,
                    "replay_state": "idle" if index == 39 else "active",
                    "replay_event_index": last_index,
                    "sequence_success": "True" if finished and index == 39 else "",
                }
            )
        with (run_dir / "telemetry_samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        events = [{"simulation_time_s": 0.0, "event_type": "replay_started", "severity": "info", "message": "start"}]
        events.extend(
            {
                "simulation_time_s": float(index),
                "event_type": "replay_command",
                "severity": "info",
                "message": f"cmd {index}",
                "playback_event_index": index,
            }
            for index in range(last_index + 1)
        )
        if finished:
            events.append({"simulation_time_s": duration, "event_type": "replay_finished", "severity": "info", "message": "Replay complete"})
        (run_dir / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")
        (run_dir / "contacts.csv").write_text("time_s,normal_force_n\n0,1\n", encoding="utf-8")
        (run_dir / "joint_timeseries.csv").write_text("time_s,torque_utilization\n0,0.1\n", encoding="utf-8")
        return run_dir


if __name__ == "__main__":
    unittest.main()
