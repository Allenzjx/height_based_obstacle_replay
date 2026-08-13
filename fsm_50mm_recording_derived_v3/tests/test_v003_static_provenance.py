from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from command_model import WHEEL_JOINT_NAMES

from fsm_50mm_recording_derived_v3 import recording_fast_plan
from fsm_50mm_recording_derived_v3 import v003_static_provenance as provenance


class V003StaticProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = provenance.build_v003_static_provenance()
        cls.payload = cls.result.payload

    def test_explicit_v003_source_hash_counts_and_schema(self) -> None:
        source = self.payload["source_audit"]
        counts = source["counts"]
        self.assertEqual(provenance.V003_VERSION_ID, source["version_id"])
        self.assertEqual(
            "explicit_v003_only_no_active_pointer_fallback",
            source["selection_policy"],
        )
        self.assertTrue(source["accepted_steps_sha256_matches_metadata"])
        self.assertTrue(source["robot_asset_sha256_matches_metadata"])
        self.assertEqual(
            "06e13153b7ba75a4283e117d875f1da4895748835a9032c6faadef2bda25b394",
            source["accepted_steps_sha256"],
        )
        self.assertEqual(24, counts["step_count"])
        self.assertEqual(136, counts["raw_source_event_count"])
        self.assertEqual(136, counts["metadata_command_count"])
        self.assertEqual(202, counts["decoded_source_command_count"])
        self.assertEqual(138, counts["decoded_source_servo_count"])
        self.assertEqual(64, counts["decoded_source_wheel_count"])
        self.assertEqual([], source["schema_audit"]["missing_required_step_fields"])
        self.assertEqual([], source["schema_audit"]["missing_required_event_fields"])
        self.assertEqual([], source["schema_audit"]["nonfinite_numeric_paths"])
        self.assertEqual(
            {"FULL_VALID": 16, "PLACEHOLDER_NO_SIM": 32},
            source["schema_audit"]["snapshot_classification_counts"],
        )

    def test_runtime_plan_calls_production_adapter(self) -> None:
        with mock.patch.object(
            provenance.recording_fast_plan,
            "fast_plan_rows",
            wraps=recording_fast_plan.fast_plan_rows,
        ) as production_adapter:
            rebuilt = provenance.build_v003_static_provenance()
        production_adapter.assert_called_once()
        kwargs = production_adapter.call_args.kwargs
        self.assertEqual(provenance.V003_VERSION_ID, kwargs["source_version"])
        self.assertEqual(24, len(kwargs["steps"]))
        self.assertEqual(
            "fsm_50mm_recording_derived_v3.recording_fast_plan.fast_plan_rows",
            rebuilt.payload["production_compiler"]["runtime_entrypoint"],
        )
        self.assertEqual(
            "playback.plan_from_steps",
            rebuilt.payload["production_compiler"]["planner_entrypoint"],
        )
        self.assertFalse(
            rebuilt.payload["production_compiler"]["approximate_compiler_used"]
        )
        self.assertFalse(
            rebuilt.payload["production_compiler"]["endpoint_fallback_used"]
        )

    def test_atomic_batches_remain_single_concurrent_segments(self) -> None:
        counts = self.payload["source_audit"]["counts"]
        self.assertEqual(6, counts["atomic_batch_count"])
        self.assertEqual(72, counts["atomic_expanded_command_count"])
        self.assertEqual(160, counts["production_event_count"])
        self.assertEqual(112, counts["production_segment_count"])
        self.assertEqual(42, counts["production_semantic_noop_count"])
        self.assertEqual(0, counts["source_mapping_error_count"])
        for batch in self.payload["atomic_batches"]:
            with self.subTest(batch=batch["batch_id"]):
                self.assertEqual(12, batch["source_command_counts"]["total"])
                self.assertEqual(8, batch["source_command_counts"]["servo"])
                self.assertEqual(4, batch["source_command_counts"]["wheel"])
                self.assertEqual(
                    9, batch["production_retained_command_counts"]["total"]
                )
                self.assertEqual(3, len(batch["production_elided_semantic_noops"]))
                self.assertTrue(batch["single_segment_preserved"])
                self.assertTrue(batch["servo_wheel_concurrent"])
        atomic_plan_rows = [
            row for row in self.payload["segments"] if row["source_atomic_event"]
        ]
        self.assertEqual(6, len(atomic_plan_rows))
        self.assertTrue(all(row["concurrent_servo_wheel"] for row in atomic_plan_rows))

    def test_30_vs_150_is_full_production_trajectory_comparison(self) -> None:
        timing = self.payload["timing_provenance"]
        recording_timing = timing["recording_timing"]
        speeds = timing["speed_sources"]
        comparison = timing["full_trajectory_comparison"]
        runtime = timing["production_runtime_150"]
        thirty = timing["production_counterfactual_30"]
        self.assertEqual(24, recording_timing["step_duration_field_count"])
        self.assertEqual(
            24, recording_timing["step_recording_timing_actual_duration_count"]
        )
        self.assertEqual(
            24, recording_timing["step_motion_semantics_actual_duration_count"]
        )
        self.assertEqual(
            0.0, recording_timing["maximum_step_vs_recording_timing_duration_delta_s"]
        )
        self.assertEqual(
            0.0,
            recording_timing[
                "maximum_step_vs_motion_semantics_duration_delta_s"
            ],
        )
        self.assertEqual({}, speeds["top_level_metadata_explicit_servo_speed_fields"])
        self.assertEqual(
            ["fixed-linear-command-space-30deg-s-v1"],
            speeds["recording_payload_bookkeeping_profile_ids"],
        )
        self.assertEqual(30.0, speeds["recording_bookkeeping_reference_velocity_deg_s"])
        self.assertEqual([150.0], speeds["source_event_canonical_servo_velocity_deg_s"])
        self.assertEqual(150.0, speeds["runtime_motion_reference_servo_velocity_deg_s"])
        self.assertEqual(150.0, speeds["production_plan_servo_velocity_deg_s"])
        self.assertAlmostEqual(78.52266666679965, runtime["final_time_s"], places=12)
        self.assertAlmostEqual(93.48333333345098, thirty["final_time_s"], places=12)
        self.assertAlmostEqual(
            14.960666666651335,
            comparison["final_time_delta_30_minus_150_s"],
            places=12,
        )
        self.assertTrue(comparison["same_command_and_target_path"])
        self.assertEqual(112, comparison["timing_changed_segment_count"])
        self.assertFalse(comparison["endpoint_only_comparison_used"])
        self.assertTrue(
            all(
                "counterfactual_30_start_s" in row
                and "counterfactual_30_end_s" in row
                and "delta_30_minus_150_duration_s" in row
                for row in self.payload["segments"]
            )
        )

    def test_production_wheel_integral_matches_applied_source(self) -> None:
        wheel = self.payload["wheel_integral_rad"]
        for name in WHEEL_JOINT_NAMES:
            with self.subTest(wheel=name):
                self.assertAlmostEqual(
                    wheel["source_applied_derived"][name],
                    wheel["production_runtime_150"][name],
                    places=12,
                )
                self.assertAlmostEqual(
                    0.0,
                    wheel["production_minus_source_applied"][name],
                    places=12,
                )
        self.assertAlmostEqual(
            7.836170889951477e-06,
            wheel["source_requested_canonical"]["front_right_ankle"]
            - wheel["source_applied_derived"]["front_right_ankle"],
            places=12,
        )
        for step in self.payload["steps"]:
            with self.subTest(step=step["source_step_index"]):
                for delta in step["production_minus_source_applied_rad"].values():
                    self.assertAlmostEqual(0.0, delta, places=12)

    def test_generation_writes_requested_reports_without_touching_source(self) -> None:
        steps_path = provenance.DEFAULT_V003_DIRECTORY / "accepted_steps.jsonl"
        metadata_path = provenance.DEFAULT_V003_DIRECTORY / "metadata.json"
        before = (
            provenance.sha256_file(steps_path),
            provenance.sha256_file(metadata_path),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = provenance.write_v003_static_provenance(
                report_root=Path(directory)
            )
            expected_names = {
                "V003_FAST_REPLAY_PLAN.json",
                "V003_FAST_REPLAY_PLAN.csv",
                "V003_SOURCE_AUDIT.md",
                "V003_TIMING_PROVENANCE.md",
            }
            self.assertEqual(
                expected_names,
                {Path(path).name for path in output["paths"].values()},
            )
            self.assertTrue(output["source_unchanged"])
            payload = json.loads(
                (Path(directory) / "V003_FAST_REPLAY_PLAN.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(provenance.STATIC_STATUS, payload["status"])
            self.assertEqual(provenance.PHYSICAL_STATUS, payload["physical_replay_status"])
            self.assertEqual(
                provenance.DISPATCH_STATUS, payload["live_dispatch_trace_status"]
            )
            self.assertFalse(payload["physical_pass_claimed"])
            with (Path(directory) / "V003_FAST_REPLAY_PLAN.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(112, len(rows))
            source_md = (Path(directory) / "V003_SOURCE_AUDIT.md").read_text(
                encoding="utf-8"
            )
            timing_md = (
                Path(directory) / "V003_TIMING_PROVENANCE.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Physical Fast Replay: `NOT_RUN`", source_md)
            self.assertIn("Live dispatch trace: `NOT_COLLECTED`", timing_md)
        after = (
            provenance.sha256_file(steps_path),
            provenance.sha256_file(metadata_path),
        )
        self.assertEqual(before, after)

    def test_non_v003_directory_is_refused_instead_of_silent_fallback(self) -> None:
        v012 = (
            provenance.DEFAULT_V003_DIRECTORY.parent
            / "v012_20260807_043048_973374_manual"
        )
        with self.assertRaisesRegex(ValueError, "explicit v003 source required"):
            provenance.build_v003_static_provenance(v012)


if __name__ == "__main__":
    unittest.main()
