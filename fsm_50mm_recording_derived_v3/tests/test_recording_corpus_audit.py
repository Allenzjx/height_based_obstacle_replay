from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES

from fsm_50mm_recording_derived_v3.recording_corpus_audit import (
    PRIMARY_PRIORITY,
    REFERENCE_PRIORITY,
    RecordingCorpusAudit,
    discover_versions,
)


SERVO_NAMES = tuple(SERVO_JOINT_NAMES)
WHEEL_NAMES = tuple(WHEEL_JOINT_NAMES)


def _targets(*, hip: float = 0.0, wheel: float = 0.0) -> dict[str, dict[str, float]]:
    servos = {name: 0.0 for name in SERVO_NAMES}
    wheels = {name: 0.0 for name in WHEEL_NAMES}
    servos["front_left_hip"] = hip
    wheels["front_left_ankle"] = wheel
    return {"servos": servos, "wheels": wheels}


def _snapshot(targets: dict[str, dict[str, float]], sim_time: float) -> dict[str, object]:
    names = list(SERVO_NAMES + WHEEL_NAMES)
    return {
        "capture_source": "SimRobotAdapter",
        "sim_time": sim_time,
        "root_pose": [[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]],
        "root_velocity": [[0.0] * 6],
        "joint_names": names,
        "joint_pos": [[0.0] * len(names)],
        "joint_vel": [[0.0] * len(names)],
        "command_state": copy.deepcopy(targets),
        "actual_joint_state": {
            "servos": {
                name: {"deg": float(targets["servos"][name])}
                for name in SERVO_NAMES
            },
            "wheels": {
                name: {"rad_s": float(targets["wheels"][name])}
                for name in WHEEL_NAMES
            },
        },
    }


def _atomic_event(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, object]:
    batch_id = "batch-001"
    return {
        "time": 0.0,
        "actual_recording_time_s": 0.0,
        "command": "servo_wheel launch",
        "kind": "servo_wheel_launch",
        "expanded_commands": [
            "servo front_left_hip 1.0",
            "wheel fl 0.5",
        ],
        "command_state_before": copy.deepcopy(before),
        "command_state_after": copy.deepcopy(after),
        "command_start_sim_time": 10.0,
        "recording_timing_source": "simulation_time",
        "actuator_command_semantics": "fixed-100-percent-direct-v1",
        "applied_sim_step": 100,
        "applied_sim_time": 10.0,
        "batch_id": batch_id,
        "canonical_servo_target_deg": {"front_left_hip": 1.0},
        "canonical_servo_velocity_deg_s": 150.0,
        "canonical_wheel_velocity_rad_s": copy.deepcopy(after["wheels"]),
        "staged_state_before": copy.deepcopy(before),
        "staged_state_after": copy.deepcopy(after),
        "wheel_active_duration_s": None,
        "batch_ack": {
            "type": "operation_ack",
            "source": "servo_wheel_launch",
            "operation": "apply_motion_batch",
            "batch_id": batch_id,
            "applied_sim_step": 100,
            "first_physics_step": 101,
            "applied_sim_time": 10.0,
            "batch_applied_sim_time": 10.0,
            "servo_motion_start_sim_time": 10.0 + 1.0 / 120.0,
            "wheel_motion_start_sim_time": 10.0 + 1.0 / 120.0,
            "motion_start_skew_s": 0.0,
            "physics_dt_s": 1.0 / 120.0,
            "servo_applied": True,
            "wheel_applied": True,
            "error": "",
            "servo_targets_applied": copy.deepcopy(after["servos"]),
            "wheel_targets_applied": copy.deepcopy(after["wheels"]),
            "canonical_servo_targets_deg": copy.deepcopy(after["servos"]),
            "canonical_wheel_velocity_rad_s": copy.deepcopy(after["wheels"]),
        },
    }


def _stop_event(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    *,
    time_s: float,
    kind: str = "recording_stop_boundary",
) -> dict[str, object]:
    return {
        "time": time_s,
        "actual_recording_time_s": time_s,
        "command": "wheel stop",
        "kind": kind,
        "command_state_before": copy.deepcopy(before),
        "command_state_after": copy.deepcopy(after),
        "command_start_sim_time": 10.0 + time_s,
        "recording_timing_source": "simulation_time",
        "actuator_command_semantics": "fixed-100-percent-direct-v1",
        "canonical_servo_target_deg": {},
        "canonical_servo_velocity_deg_s": 150.0,
        "canonical_wheel_velocity_rad_s": copy.deepcopy(after["wheels"]),
        "wheel_active_duration_s": None,
        "final_stop_command": "wheel stop" if kind == "recording_stop_boundary" else None,
    }


def _make_step(*, index: int = 1, pattern_findings: bool = False) -> dict[str, object]:
    initial = _targets()
    active = _targets(hip=1.0, wheel=0.5)
    stopped = _targets(hip=1.0, wheel=0.0)
    events: list[dict[str, object]] = [_atomic_event(initial, active)]
    active_duration = 2.0
    if pattern_findings:
        # Ordered same-time waypoint on the same servo: explicit overlap, not
        # an atomic batch.  It is a semantic no-op and preserves final state.
        events.append(
            {
                "time": 0.0,
                "actual_recording_time_s": 0.0,
                "command": "servo front_left_hip 1.0",
                "kind": "ui",
                "command_state_before": copy.deepcopy(active),
                "command_state_after": copy.deepcopy(active),
                "command_start_sim_time": 10.0,
                "recording_timing_source": "simulation_time",
                "actuator_command_semantics": "fixed-100-percent-direct-v1",
                "canonical_servo_target_deg": {"front_left_hip": 1.0},
                "canonical_servo_velocity_deg_s": 150.0,
                "canonical_wheel_velocity_rad_s": copy.deepcopy(active["wheels"]),
                "wheel_active_duration_s": None,
            }
        )
        events.append(_stop_event(active, stopped, time_s=1.0, kind="ui"))
        events.append(
            {
                "time": 2.0,
                "actual_recording_time_s": 2.0,
                "command": "wait 0",
                "kind": "ui",
                "command_state_before": copy.deepcopy(stopped),
                "command_state_after": copy.deepcopy(stopped),
                "command_start_sim_time": 12.0,
                "recording_timing_source": "simulation_time",
                "actuator_command_semantics": "fixed-100-percent-direct-v1",
                "canonical_servo_target_deg": {},
                "canonical_servo_velocity_deg_s": 150.0,
                "canonical_wheel_velocity_rad_s": copy.deepcopy(stopped["wheels"]),
                "wheel_active_duration_s": None,
            }
        )
        events.append(_stop_event(stopped, stopped, time_s=2.0))
        active_duration = 1.0
    else:
        events.append(_stop_event(active, stopped, time_s=2.0))
    theta = {name: 0.0 for name in WHEEL_NAMES}
    theta["front_left_ankle"] = 0.5 * active_duration
    return {
        "index": index,
        "name": f"step_{index:03d}",
        "type": "recorded",
        "duration": 2.0,
        "height_m": 0.05,
        "height_mm": 50,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "note": "",
        "record_coalesce": {},
        "recording_baseline": {},
        "pose_checkpoint_capture": {"pose_restore_eligible": True},
        "command_state_before": copy.deepcopy(initial),
        "command_state_after": copy.deepcopy(stopped),
        "events": events,
        "recording_timing": {
            "actual_duration_s": 2.0,
            "actuator_command_semantics": "fixed-100-percent-direct-v1",
            "source": "simulation_time",
            "wheel_active_duration_s": active_duration,
        },
        "motion_semantics": {
            "actual_recording_duration_s": 2.0,
            "actuator_command_semantics": "fixed-100-percent-direct-v1",
            "canonical_wheel_angular_displacement_rad": copy.deepcopy(theta),
            "derived_wheel_angular_displacement_rad": copy.deepcopy(theta),
            "reference_duration_s": 2.0,
            "servo_records": [],
            "servo_start_deg": copy.deepcopy(initial["servos"]),
            "servo_target_deg": copy.deepcopy(stopped["servos"]),
            "wheel_active_duration_clock": "actual_recording_time",
            "wheel_angular_displacement_rad": copy.deepcopy(theta),
            "wheel_displacement_source": "articulation_joint_state",
            "wheel_records": [],
        },
        "sim_state_before": _snapshot(initial, 20.0 + index * 3.0),
        "sim_state_after": _snapshot(stopped, 22.0 + index * 3.0),
        "wheel_stop_status": {"state": "zero_target_applied"},
    }


def _write_version(
    recording_root: Path,
    version_id: str,
    steps: list[dict[str, object]],
) -> Path:
    directory = recording_root / "versions" / version_id
    directory.mkdir(parents=True, exist_ok=True)
    accepted = directory / "accepted_steps.jsonl"
    accepted.write_text(
        "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps),
        encoding="utf-8",
    )
    digest = hashlib.sha256(accepted.read_bytes()).hexdigest()
    metadata = {
        "accepted_steps_sha256": digest,
        "actuator_baseline_id": "baseline",
        "actuator_command_semantics": "fixed-100-percent-direct-v1",
        "command_count": sum(len(step["events"]) for step in steps),
        "created_at": "2026-01-01T00:00:00Z",
        "environment_baseline_id": "environment",
        "height_mm": 50,
        "motion_profile_id": "motion",
        "motion_profile_mode": "fixed_100_percent",
        "note": "",
        "obstacle_front_face_x_m": 0.5,
        "obstacle_length_m": 2.0,
        "obstacle_width_m": 2.0,
        "parent_version_id": "",
        "robot_asset_path": "robot.usd",
        "robot_asset_sha256": "0" * 64,
        "robot_respawn_pose": None,
        "schema_version": "height-steps-versions.v2",
        "step_count": len(steps),
        "updated_at": "2026-01-01T00:00:00Z",
        "version_id": version_id,
        "version_name": "manual",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


class RecordingCorpusAuditTests(unittest.TestCase):
    def test_dynamic_membership_and_v003_priority_ignore_active_v012(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            _write_version(root, "v003_first_complete", [_make_step()])
            _write_version(root, "v012_active_short", [_make_step()])
            _write_version(root, "v777_new_directory", [_make_step()])
            (root / "active_version.json").write_text(
                json.dumps({"version_id": "v012_active_short"}), encoding="utf-8"
            )

            discovered = discover_versions(root)
            self.assertEqual(3, len(discovered))
            report_root = Path(temporary) / "reports"
            result = RecordingCorpusAudit(root, report_root).run()
            payload = result["payload"]
            by_id = {row["version_id"]: row for row in payload["versions"]}
            self.assertEqual(PRIMARY_PRIORITY, by_id["v003_first_complete"]["priority"])
            self.assertEqual(REFERENCE_PRIORITY, by_id["v012_active_short"]["priority"])
            self.assertTrue(by_id["v012_active_short"]["active_pointer"])
            self.assertFalse(payload["selection_policy"]["active_pointer_is_selection"])
            self.assertIsNone(payload["selection_policy"]["automatic_selected_version_id"])
            self.assertEqual("PASS", payload["aggregate"]["corpus_static_status"])

            for path in result["paths"].values():
                self.assertTrue(path.is_file())
            with result["paths"]["csv"].open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(3, len(rows))
            self.assertIn("priority", rows[0])

    def test_atomic_counts_and_wheel_integral_have_source_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            _write_version(root, "v003_first_complete", [_make_step()])
            version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                "versions"
            ][0]
            self.assertEqual(2, version["source_event_count"])
            self.assertEqual(3, version["decoded_command_count"])
            self.assertEqual(1, version["servo_command_count"])
            self.assertEqual(2, version["wheel_command_count"])
            self.assertEqual(1, version["atomic_event_count"])
            self.assertEqual(1, version["atomic_valid_count"])
            self.assertEqual(1, version["source_wheel_segment_count"])
            self.assertAlmostEqual(
                1.0,
                version["wheel_integral_theta_rad"]["front_left_ankle"],
                places=12,
            )
            segment = version["source_wheel_segments"][0]
            self.assertEqual(0, segment["source_event_start_index"])
            self.assertEqual(1, segment["source_event_end_index"])
            self.assertAlmostEqual(0.5, segment["speed_rad_s"]["front_left_ankle"])
            self.assertAlmostEqual(2.0, segment["duration_s"])
            self.assertAlmostEqual(1.0, segment["theta_rad"]["front_left_ankle"])
            self.assertEqual(0, version["source_event_index_error_count"])

    def test_atomic_canonical_targets_must_match_applied_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            step = _make_step()
            step["events"][0]["batch_ack"]["canonical_servo_targets_deg"][
                "front_left_hip"
            ] = 999.0
            _write_version(root, "v003_first_complete", [step])
            version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                "versions"
            ][0]
            self.assertEqual(1, version["atomic_event_count"])
            self.assertEqual(0, version["atomic_valid_count"])
            self.assertIn("ATOMIC_EVIDENCE_INVALID", version["error_codes"])

    def test_atomic_event_ack_and_next_tick_time_chain_must_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            step = _make_step()
            atomic = step["events"][0]
            atomic["applied_sim_step"] = 999
            atomic["applied_sim_time"] = 77.0
            atomic["batch_ack"]["servo_motion_start_sim_time"] = 20.0
            atomic["batch_ack"]["wheel_motion_start_sim_time"] = 20.0
            _write_version(root, "v003_first_complete", [step])
            version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                "versions"
            ][0]
            self.assertEqual(0, version["atomic_valid_count"])
            errors = version["atomic_events"][0]["errors"]
            self.assertTrue(any("applied_sim_step" in error for error in errors))
            self.assertTrue(any("applied_sim_time" in error for error in errors))
            self.assertTrue(any("physics_dt_s" in error for error in errors))

    def test_duplicate_empty_wait_overlap_and_mid_stop_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            first = _make_step(pattern_findings=True)
            second = copy.deepcopy(first)
            second["index"] = 2
            second["name"] = "step_002"
            second["sim_state_before"] = _snapshot(_targets(), 30.0)
            second["sim_state_after"] = _snapshot(_targets(hip=1.0), 32.0)
            _write_version(root, "v003_first_complete", [first, second])
            version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                "versions"
            ][0]
            self.assertEqual(1, version["duplicate_step_count"])
            self.assertGreaterEqual(version["empty_wait_count"], 2)
            self.assertGreaterEqual(version["same_timestamp_overlap_count"], 2)
            self.assertEqual(2, version["mid_step_wheel_stop_count"])
            self.assertAlmostEqual(
                1.0,
                version["wheel_integral_theta_rad"]["front_left_ankle"],
                places=12,
            )

    def test_hash_snapshot_nonfinite_and_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            _write_version(root, "v003_first_complete", [_make_step()])
            damaged = _write_version(root, "v005_damaged", [_make_step()])
            accepted = damaged / "accepted_steps.jsonl"
            step = json.loads(accepted.read_text(encoding="utf-8").strip())
            step.pop("recording_timing")
            step["duration"] = float("nan")
            step["sim_state_before"] = {
                "capture_source": "NullSimRobotAdapter",
                "command_state": _targets(),
            }
            # Leave metadata SHA unchanged to exercise independent hash failure.
            accepted.write_text(json.dumps(step) + "\n", encoding="utf-8")
            metadata_path = damaged / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["height_mm"] = float("nan")
            metadata["step_count"] = float("inf")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            result = RecordingCorpusAudit(root, Path(temporary) / "reports").run()
            payload = result["payload"]
            damaged_row = next(
                row for row in payload["versions"] if row["version_id"] == "v005_damaged"
            )
            self.assertEqual("FAIL", damaged_row["static_integrity_status"])
            self.assertFalse(damaged_row["sha256_matches"])
            self.assertGreater(damaged_row["missing_required_field_count"], 0)
            self.assertGreater(damaged_row["nonfinite_numeric_count"], 0)
            self.assertGreater(damaged_row["schema_drift_count"], 0)
            self.assertGreater(damaged_row["snapshot_incomplete_count"], 0)
            self.assertIn("ACCEPTED_STEPS_HASH_MISMATCH", damaged_row["error_codes"])
            self.assertIn("NONFINITE_NUMERIC_VALUE", damaged_row["error_codes"])
            self.assertIn("SCHEMA_DRIFT", damaged_row["error_codes"])
            json.loads(
                result["paths"]["json"].read_text(encoding="utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )

    def test_bad_event_shape_retains_raw_index_and_empty_marker_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            step = _make_step()
            step["events"].insert(0, 7)
            _write_version(root, "v003_first_complete", [step])
            version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                "versions"
            ][0]
            self.assertEqual([0, 1, 2], [row["source_event_index"] for row in version["source_events"]])
            self.assertFalse(version["source_events"][0]["source_event_valid"])
            self.assertEqual(2, version["source_event_index_valid_count"])
            self.assertGreater(version["source_event_index_error_count"], 0)

            marker = _make_step()
            marker["events"] = [
                {
                    "time": 2.0,
                    "actual_recording_time_s": 2.0,
                    "command": "record_stop",
                    "kind": "recording_stop_boundary",
                    "command_state_before": _targets(),
                    "command_state_after": _targets(),
                    "command_start_sim_time": 12.0,
                    "recording_timing_source": "simulation_time",
                    "actuator_command_semantics": "fixed-100-percent-direct-v1",
                }
            ]
            _write_version(root, "v005_marker_only", [marker])
            payload = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()
            marker_row = next(row for row in payload["versions"] if row["version_id"] == "v005_marker_only")
            self.assertEqual(1, marker_row["empty_step_count"])

    def test_final_stop_followed_by_same_tick_marker_is_not_mid_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            step = _make_step()
            marker = copy.deepcopy(step["events"][-1])
            marker["command"] = "record_stop"
            marker["kind"] = "recording_stop_boundary"
            step["events"].append(marker)
            _write_version(root, "v003_first_complete", [step])
            version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                "versions"
            ][0]
            self.assertEqual(0, version["mid_step_wheel_stop_count"])

    def test_wrong_events_type_required_null_and_bad_manifest_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            directory = _write_version(root, "v003_first_complete", [_make_step()])
            accepted = directory / "accepted_steps.jsonl"
            step = json.loads(accepted.read_text(encoding="utf-8"))
            step["events"] = 1
            accepted.write_text(json.dumps(step) + "\n", encoding="utf-8")
            metadata_path = directory / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["actuator_baseline_id"] = None
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            manifest_path = root.parent / "manifest.json"
            manifest_path.write_text(json.dumps({"heights": []}), encoding="utf-8")

            result = RecordingCorpusAudit(root, Path(temporary) / "reports").run()
            version = result["payload"]["versions"][0]
            self.assertEqual("FAIL", version["static_integrity_status"])
            self.assertIn("STEP_EVENTS_WRONG_TYPE", version["error_codes"])
            self.assertIn("METADATA_REQUIRED_FIELD_MISSING", version["error_codes"])
            self.assertTrue(result["payload"]["manifest_inventory"]["issues"])

    def test_reader_failures_become_issues_instead_of_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "height_050mm"
            directory = _write_version(root, "v003_first_complete", [_make_step()])
            (directory / "metadata.json").write_text(
                '{"huge":' + "9" * 5000 + "}", encoding="utf-8"
            )
            result = RecordingCorpusAudit(root, Path(temporary) / "reports").run()
            self.assertEqual("FAIL", result["payload"]["versions"][0]["static_integrity_status"])
            self.assertIn("JSON_OBJECT_INVALID", result["payload"]["versions"][0]["error_codes"])

            (directory / "accepted_steps.jsonl").write_bytes(b"\xff\xfe\x00")
            result = RecordingCorpusAudit(root, Path(temporary) / "reports").run()
            self.assertEqual("FAIL", result["payload"]["versions"][0]["static_integrity_status"])
            self.assertIn("JSONL_READ_FAILED", result["payload"]["versions"][0]["error_codes"])

    def test_unknown_or_malformed_command_is_fail_closed_and_not_actionable(self) -> None:
        for command in ("frobnicate", "servo bogus", "wheel bogus", "wait nope"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "height_050mm"
                step = _make_step()
                event = step["events"][0]
                event.pop("expanded_commands", None)
                event["command"] = command
                event["kind"] = "ui"
                event.pop("batch_ack", None)
                event.pop("batch_id", None)
                event["command_state_after"] = copy.deepcopy(event["command_state_before"])
                step["events"] = [event]
                step["command_state_after"] = copy.deepcopy(step["command_state_before"])
                _write_version(root, "v003_first_complete", [step])
                version = RecordingCorpusAudit(root, Path(temporary) / "reports").audit()[
                    "versions"
                ][0]
                self.assertEqual("FAIL", version["static_integrity_status"])
                self.assertIn("COMMAND_INVALID", version["error_codes"])
                self.assertEqual(1, version["empty_step_count"])
                self.assertEqual(1, version["other_command_count"])


if __name__ == "__main__":
    unittest.main()
