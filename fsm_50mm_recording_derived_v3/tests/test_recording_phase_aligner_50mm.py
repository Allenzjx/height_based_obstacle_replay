from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fsm_50mm_recording_derived_v3.recording_phase_aligner_50mm import (
    GateAIncompleteError,
    LEG_ORDER,
    LoadedRun,
    SUCCESS_RESULTS,
    _crossing_index,
    _final_support_departure,
    _lift_clearance_index,
    _stable_top_index,
    _telemetry_groups,
    analyze_gate_b,
    assign_strategy_profiles,
    emit_reports,
)


def _telemetry_row(
    index: int,
    *,
    segment: int,
    fr_class: str = "GROUND",
    fr_face: float = -0.1,
    fr_top: float = -0.05,
) -> dict:
    classes = {leg: "GROUND" for leg in LEG_ORDER}
    classes["FR"] = fr_class
    face = {leg: -0.2 for leg in LEG_ORDER}
    face["FR"] = fr_face
    top = {leg: -0.05 for leg in LEG_ORDER}
    top["FR"] = fr_top
    return {
        "sim_time_s": 1.0 + index * 0.1,
        "sim_step": index,
        "segment_cursor": segment,
        "event_cursor": segment,
        "source_step": 1,
        "source_version": "v003_test",
        "scheduler_state": "PLAYING",
        "robot_state_finite": True,
        "base_roll_rad": 0.0,
        "base_pitch_rad": 0.0,
        "base_position_m": {"x": index * 0.01, "y": 0.0, "z": 0.1},
        "wheel_contact_classes": classes,
        "wheel_front_face_clearance_m": face,
        "wheel_top_clearance_m": top,
        "wheel_center_w_m": {leg: [0.0, 0.0, 0.05] for leg in LEG_ORDER},
    }


def _loaded_run(version: str, *, structure_variant: bool = False) -> LoadedRun:
    plan_rows = []
    for index in range(4):
        servo = {}
        if index == 1:
            servo = {
                ("front_left_hip" if structure_variant else "front_right_hip"): 10.0
            }
        wheel = {f"{prefix}_ankle": 0.0 for prefix in (
            "front_left", "front_right", "rear_left", "rear_right"
        )}
        if index == 2:
            wheel["front_right_ankle"] = 0.3
        plan_rows.append(
            {
                "decoded_segment_index": index,
                "source_step_index": index + 1,
                "servo_target_deg": servo,
                "wheel_target_rad_s": wheel,
                "concurrent": bool(servo and index == 2),
            }
        )
    task_inputs = {
        "physical_evidence": {
            "traversal": {
                "legs": {
                    leg: {"front_face_crossing_s": float(i + 1)}
                    for i, leg in enumerate(LEG_ORDER)
                }
            }
        }
    }
    return LoadedRun(
        version=version,
        recording_dir=Path(version),
        run_dir=Path("run") / version,
        result={"assessment": {}},
        task_result="REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE",
        plan_rows=tuple(plan_rows),
        telemetry=tuple(_telemetry_row(i, segment=min(i, 3)) for i in range(5)),
        task_inputs=task_inputs,
        accepted_steps_sha256="a" * 64,
        plan_sha256="b" * 64,
        task_inputs_sha256="c" * 64,
        telemetry_sha256="d" * 64,
        video_sha256="e" * 64,
    )


class GeometryLandmarkTests(unittest.TestCase):
    def test_uses_final_support_departure_not_first_air(self) -> None:
        rows = [
            _telemetry_row(0, segment=0, fr_class="GROUND"),
            _telemetry_row(1, segment=0, fr_class="AIR"),
            _telemetry_row(2, segment=1, fr_class="GROUND"),
            _telemetry_row(3, segment=1, fr_class="FRONT_FACE", fr_top=-0.02),
            _telemetry_row(4, segment=2, fr_class="AIR", fr_top=0.01),
            _telemetry_row(5, segment=2, fr_class="AIR", fr_face=0.01, fr_top=0.02),
            _telemetry_row(6, segment=3, fr_class="TOP", fr_face=0.02, fr_top=0.0),
        ]
        crossing = _crossing_index(rows, "FR", 0.5)
        self.assertEqual(5, crossing)
        self.assertEqual(
            3,
            _final_support_departure(
                rows, "FR", lower_bound=0, crossing_index=crossing
            ),
        )
        self.assertEqual(
            4,
            _lift_clearance_index(
                rows,
                "FR",
                departure_index=3,
                crossing_index=crossing,
            ),
        )
        self.assertEqual(6, _stable_top_index(rows, "FR", crossing))

    def test_wheel_groups_follow_production_segment_targets(self) -> None:
        run = _loaded_run("v003_test")
        groups = _telemetry_groups(
            run,
            lambda segment: any(
                abs(float(value)) > 0.0
                for value in segment["wheel_target_rad_s"].values()
            ),
            lower=0,
            upper=4,
        )
        self.assertEqual([(2, 2)], groups)


class StrategyClusterTests(unittest.TestCase):
    def _row(self, version: str, phase: str, joint: str) -> dict:
        return {
            "version": version,
            "phase": phase,
            "structure": {
                "servo_joint_sequence": [joint],
                "wheel_assist_joint_sequence": [],
                "servo_wheel_concurrent": False,
                "phase_status": "GEOMETRY_EVENT_ANCHORED",
            },
        }

    def test_v003_is_primary_and_unlike_structures_are_not_merged(self) -> None:
        primary = _loaded_run("v003_primary")
        same = _loaded_run("v008_same")
        alternate = _loaded_run("v009_alternate", structure_variant=True)
        rows = []
        for run, joint in (
            (primary, "front_right_hip"),
            (same, "front_right_hip"),
            (alternate, "front_left_hip"),
        ):
            rows.extend(
                [
                    self._row(run.version, "FR_FACE_CROSS", joint),
                    self._row(run.version, "FINAL_POSTURE_RECOVERY", joint),
                ]
            )
        roles, clusters, recovery = assign_strategy_profiles(
            [primary, same, alternate], rows
        )
        self.assertEqual("PRIMARY_PROFILE", roles[primary.version])
        self.assertEqual("PRIMARY_PROFILE", roles[same.version])
        self.assertTrue(roles[alternate.version].startswith("ALTERNATE_PROFILE_"))
        self.assertEqual(2, len(clusters))
        self.assertTrue(recovery[primary.version].startswith("RECOVERY_PROFILE_"))


class ReportEmissionTests(unittest.TestCase):
    def test_incomplete_gate_a_analysis_creates_no_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_root = root / "recordings"
            version = recording_root / "versions" / "v003_missing_run"
            version.mkdir(parents=True)
            (version / "accepted_steps.jsonl").write_text("{}\n", encoding="utf-8")
            (version / "metadata.json").write_text("{}\n", encoding="utf-8")
            report_root = root / "reports"
            with self.assertRaises(GateAIncompleteError):
                analyze_gate_b(
                    recording_root=recording_root,
                    run_root=root / "runs",
                )
            self.assertFalse(report_root.exists())

    def test_bundle_contains_requested_files_and_explicit_exclusion(self) -> None:
        success = _loaded_run("v003_primary")
        failure = LoadedRun(
            **{
                **success.__dict__,
                "version": "v005_failed",
                "task_result": "REPLAY_TASK_FAIL",
                "result": {
                    "assessment": {"first_actual_failure_phase": "FL_LIFT"}
                },
            }
        )
        row = {
            "version": success.version,
            "task_result": success.task_result,
            "strategy_profile": "PRIMARY_PROFILE",
            "phase": "FR_FACE_CROSS",
            "phase_status": "GEOMETRY_EVENT_ANCHORED",
            "evidence_basis": "WHEEL_CLEARANCE_AND_FRONT_FACE_GEOMETRY",
            "functional_windows_may_overlap": True,
            "run_dir": str(success.run_dir),
            "accepted_steps_sha256": success.accepted_steps_sha256,
            "plan_sha256": success.plan_sha256,
            "task_inputs_sha256": success.task_inputs_sha256,
            "telemetry_sha256": success.telemetry_sha256,
            "video_sha256": success.video_sha256,
            "replay_start_s": 0.0,
            "replay_end_s": 1.0,
            "duration_s": 1.0,
            "source_step_range": "1:2",
            "fast_segment_range": "0:1",
            "active_leg": "FR",
            "candidate_support_legs": ["FL", "RR", "RL"],
            "geometry_support_fraction": {"FL": 1.0},
            "support_evidence_status": "GEOMETRY_ONLY_NO_CONTACT_LOAD",
            "servo_commands": [],
            "servo_target_range_deg": {},
            "wheel_commands": [],
            "wheel_target_range_rad_s": {},
            "servo_wheel_concurrent": False,
            "concurrent_segment_count": 0,
            "candidate_com_target_direction": "PHASE_LOCAL",
            "observed_base_delta_m": {"x": 0.1, "y": 0.0, "z": 0.0},
            "observed_base_direction": "+X",
            "com_evidence_status": "BASE_TRANSLATION_PROXY_ONLY_NO_COM_TELEMETRY",
            "entry_event": "lift",
            "completion_event": "cross",
            "active_wheel_clearance_range_m": {},
            "peak_abs_roll_rad": 0.1,
            "peak_abs_pitch_rad": 0.1,
            "suitable_for_first_fsm": "YES_EVENT_GUARD_CANDIDATE",
            "notes": "",
            "structure": {
                "servo_joint_sequence": [],
                "wheel_assist_joint_sequence": [],
                "servo_wheel_concurrent": False,
                "phase_status": "GEOMETRY_EVENT_ANCHORED",
            },
        }
        # Fill the remaining canonical phases as explicitly unresolved so the
        # Markdown generator exercises the honest no-invention path.
        rows = []
        for phase in (
            "INITIAL_APPROACH",
            "PRE_FR_COM_SHIFT",
            "FR_UNLOAD_AND_LIFT",
            "FR_FACE_CROSS",
            "FR_TOP_PLACE",
            "FL_UNLOAD_AND_LIFT",
            "FL_FACE_CROSS",
            "FL_TOP_PLACE",
            "FRONT_PAIR_ADVANCE",
            "PRE_RR_COM_SHIFT",
            "RR_UNLOAD_AND_LIFT",
            "RR_FACE_CROSS",
            "RR_TOP_PLACE",
            "PRE_RL_SUPPORT_SETUP",
            "PRE_RL_COM_SHIFT",
            "RL_UNLOAD_AND_LIFT",
            "RL_FACE_CROSS",
            "RL_TOP_PLACE",
            "FINAL_ADVANCE",
            "FINAL_POSTURE_RECOVERY",
        ):
            item = dict(row)
            item["phase"] = phase
            rows.append(item)
        analysis = {
            "successful_runs": [success],
            "excluded_runs": [failure],
            "rows": rows,
            "clusters": [
                {
                    "role": "PRIMARY_PROFILE",
                    "fingerprint_sha256": "f" * 64,
                    "versions": [success.version],
                    "structure": {
                        "traversal_order": list(LEG_ORDER),
                        "phases": [
                            {"phase": "FR_FACE_CROSS", **row["structure"]}
                        ],
                    },
                }
            ],
            "version_roles": {success.version: "PRIMARY_PROFILE"},
            "recovery_roles": {success.version: "RECOVERY_PROFILE_1"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = emit_reports(analysis, Path(temporary))
            self.assertEqual(
                {
                    "50MM_COMMON_PHASE_ALIGNMENT.csv",
                    "50MM_COMMON_PHASES.md",
                    "50MM_VERSION_STRATEGY_CLUSTERS.md",
                },
                set(paths),
            )
            clusters = paths["50MM_VERSION_STRATEGY_CLUSTERS.md"].read_text()
            self.assertIn("v005_failed", clusters)
            self.assertIn("never averaged", clusters)

    def test_success_result_names_are_exact(self) -> None:
        self.assertEqual(
            {
                "REPLAY_TASK_SUCCESS",
                "REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE",
            },
            set(SUCCESS_RESULTS),
        )
        self.assertTrue(issubclass(GateAIncompleteError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
