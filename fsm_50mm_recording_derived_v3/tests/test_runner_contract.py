from __future__ import annotations

import hashlib
import json
import inspect
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from fsm_50mm_recording_derived_v3 import run_fsm50 as run_fsm50_module
from fsm_50mm_recording_derived_v3.recording_audit import RecordingAudit, VersionFiles
from fsm_50mm_recording_derived_v3.run_fsm50 import (
    ChildSupervisorHandshake,
    ReplaySingletonLock,
    _atomic_write_json,
    _batch_command_exit_code,
    _compare_source_freezes,
    _close_simulation_with_explicit_policy,
    _deserialize_replay_args,
    _existing_simulator_processes,
    _fail_closed_recording_audits,
    _find_reliable_completed_replays,
    _first_failure,
    _generate_fsm50_visualization,
    _result_payload,
    _monitor_supervised_child,
    _new_directory,
    _preclose_evidence_manifest,
    _process_returncode_after_close,
    _replace_with_windows_retry,
    _record_shutdown_outcome,
    _runtime_environment_equivalence,
    _run_recording_replays_locked,
    _serialize_replay_args,
    _select_versions,
    _snapshot_preclose_files,
    _validate_preclose_closure,
    _write_checksums,
    build_parser,
)
from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES


class RunnerContractTests(unittest.TestCase):
    def test_atomic_replace_retries_transient_windows_access_denied(self) -> None:
        attempts = []

        def replace(_source, _destination):
            attempts.append(1)
            if len(attempts) < 3:
                error = PermissionError("sharing violation")
                error.winerror = 5
                raise error

        with (
            mock.patch.object(run_fsm50_module.os, "name", "nt"),
            mock.patch.object(run_fsm50_module.os, "replace", replace),
            mock.patch.object(run_fsm50_module.time, "sleep"),
            mock.patch.object(
                run_fsm50_module.time,
                "monotonic",
                side_effect=[0.0, 0.1, 0.2],
            ),
        ):
            _replace_with_windows_retry(Path("source"), Path("destination"))

        self.assertEqual(len(attempts), 3)

    def test_atomic_replace_does_not_retry_nonsharing_error(self) -> None:
        with (
            mock.patch.object(run_fsm50_module.os, "name", "nt"),
            mock.patch.object(
                run_fsm50_module.os,
                "replace",
                side_effect=FileNotFoundError("missing source"),
            ) as replace,
            mock.patch.object(run_fsm50_module.time, "sleep") as sleep,
        ):
            with self.assertRaises(FileNotFoundError):
                _replace_with_windows_retry(Path("source"), Path("destination"))

        replace.assert_called_once()
        sleep.assert_not_called()

    def test_fast_child_process_return_is_separate_from_command_result(self) -> None:
        self.assertEqual(
            _process_returncode_after_close(
                command_exit_code=1,
                shutdown_mode="fast",
                close_error="",
                supervised=True,
            ),
            0,
        )
        self.assertEqual(
            _process_returncode_after_close(
                command_exit_code=1,
                shutdown_mode="fast",
                close_error="",
                supervised=False,
            ),
            1,
        )
        self.assertEqual(
            _process_returncode_after_close(
                command_exit_code=0,
                shutdown_mode="fast",
                close_error="close failed",
                supervised=True,
            ),
            1,
        )

    def test_grounding_preclose_is_self_contained_without_global_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "batch_request.json",
                "batch_results.json",
                "batch_finalization.json",
                "batch_results.preclose.json",
                "batch_finalization.preclose.json",
                "source_integrity.json",
                "checksums.sha256",
                "checksums.preclose.sha256",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(
                    run_fsm50_module,
                    "VERSION_MATRIX_PATH",
                    root / "missing_version_matrix.csv",
                ),
                mock.patch.object(
                    run_fsm50_module,
                    "DIAGONAL_TIMELINE_PATH",
                    root / "missing_diagonal_timeline.csv",
                ),
            ):
                evidence = _preclose_evidence_manifest(
                    root,
                    results=[],
                    batch_source_comparison={"equal": True},
                    batch_finalization={"strict_success": False},
                    include_global_analysis_reports=False,
                )

            self.assertEqual(evidence["physics_result_count"], 0)
            self.assertNotIn(
                str((root / "missing_version_matrix.csv").resolve()),
                evidence["evidence_files"],
            )

    def test_batch_checksum_covers_nested_manifests_without_preclose_self_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            (child / "checksums.sha256").write_text(
                "nested\n", encoding="utf-8"
            )
            for _source, snapshot_name in (
                ("batch_results.json", "batch_results.preclose.json"),
                (
                    "batch_finalization.json",
                    "batch_finalization.preclose.json",
                ),
                ("checksums.sha256", "checksums.preclose.sha256"),
            ):
                (root / snapshot_name).write_text("snapshot\n", encoding="utf-8")

            _write_checksums(root, exclude_preclose_snapshots=True)
            rows = (root / "checksums.sha256").read_text(encoding="utf-8")

            self.assertIn("child/checksums.sha256", rows)
            self.assertNotIn("batch_results.preclose.json", rows)
            self.assertNotIn("batch_finalization.preclose.json", rows)
            self.assertNotIn("checksums.preclose.sha256", rows)

    def test_fast_process_success_is_separate_from_command_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _atomic_write_json(
                root / "batch_finalization.json",
                {"phase": "SHUTDOWN_COMPLETE", "strict_success": False},
            )
            self.assertEqual(
                _batch_command_exit_code(root, fallback=0),
                1,
            )
            _atomic_write_json(
                root / "batch_finalization.json",
                {"phase": "SHUTDOWN_COMPLETE", "strict_success": True},
            )
            self.assertEqual(
                _batch_command_exit_code(root, fallback=1),
                0,
            )

    def test_short_version_selector_is_unambiguous(self) -> None:
        versions = RecordingAudit().enumerate_versions()
        selected = _select_versions(versions, ["v010", "v012"])
        self.assertEqual(["v010", "v012"], [item.version_id.split("_", 1)[0] for item in selected])

    def test_all_selector_keeps_every_physical_directory(self) -> None:
        versions = RecordingAudit().enumerate_versions()
        self.assertEqual(versions, _select_versions(versions, ["all"]))

    def test_scheduler_and_physical_fields_are_independent_cli_contract(self) -> None:
        args = build_parser().parse_args(["replay-recordings", "--versions", "v012", "--headless"])
        self.assertEqual("replay-recordings", args.command)
        self.assertEqual(["v012"], args.versions)
        self.assertTrue(args.headless)
        self.assertFalse(args.resume)

    def test_grounding_only_cli_is_one_fresh_trial_with_fast_exit_default(self) -> None:
        args = build_parser().parse_args(
            ["grounding-only", "--trial-id", "3"]
        )

        self.assertEqual(args.command, "grounding-only")
        self.assertEqual(args.trial_id, 3)
        self.assertEqual(args.shutdown_mode, "fast")
        self.assertFalse(args.headless)
        self.assertFalse(args.no_video)

    def test_replay_resume_flag_and_version_directories_are_unique(self) -> None:
        args = build_parser().parse_args(
            ["replay-recordings", "--versions", "v012", "--resume"]
        )
        self.assertTrue(args.resume)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _new_directory(root, "clean_fast_replay")
            second = _new_directory(root, "clean_fast_replay")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    @staticmethod
    def _write_resume_fixture(
        output_root: Path,
        item: VersionFiles,
        accepted_steps_sha256: str,
        *,
        marker: str = ".finalized",
        classification: str = "PHYSICAL_FAILURE",
        complete_shutdown: bool = True,
    ) -> tuple[Path, Path]:
        artifact_root = (
            output_root
            / "batch"
            / item.version_id
            / "unique_clean_fast_replay"
        )
        run_dir = artifact_root / "telemetry_run"
        run_dir.mkdir(parents=True)
        video_path = (run_dir / "fsm50_viewport.mp4").resolve()
        video_path.write_bytes(
            b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
        )
        video_sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest()
        viewport_manifest_path = (run_dir / "viewport_video_manifest.json").resolve()
        viewport_manifest = {
            "schema_version": "fsm50.recording_viewport_video.v1",
            "contact_mode": "instrumented",
            "capture_requested": True,
            "diagnostic_only": False,
            "valid": True,
            "artifact_valid": True,
            "actual_viewport_video": True,
            "not_camera_video": False,
            "source": "actual_active_isaac_gui_viewport_render_product",
            "render_product_path": "/Render/Product/ActiveViewport",
            "frame_count": 2,
            "video_path": str(video_path),
            "video_sha256": video_sha256,
            "error": "",
        }
        _atomic_write_json(viewport_manifest_path, viewport_manifest)
        viewport_manifest_sha256 = hashlib.sha256(
            viewport_manifest_path.read_bytes()
        ).hexdigest()
        video = {
            **viewport_manifest,
            "manifest_path": str(viewport_manifest_path),
            "manifest_sha256": viewport_manifest_sha256,
        }
        result = {
            "schema_version": "fsm50.recording_replay_result.v1",
            "created_utc": "2026-08-08T00:00:00+00:00",
            "source_version": item.version_id,
            "accepted_steps_sha256": accepted_steps_sha256,
            "classification": classification,
            "strict_full_success": False,
            "artifact_valid": True,
            "artifact_root": str(artifact_root.resolve()),
            "run_dir": str(run_dir.resolve()),
            "contact_mode": "instrumented",
            "actual_viewport_video": True,
            "video_path": str(video_path),
            "video_sha256": video_sha256,
            "video": video,
            "source_integrity": {"ok": True},
            "visualization": {"ok": True},
            "lifecycle": {
                "finalized": True,
                "failed": False,
                "strict_success": False,
            },
        }
        _atomic_write_json(run_dir / "result.json", result)
        _atomic_write_json(
            run_dir / "failure_diagnostics.json",
            {
                "classification": classification,
                "artifact_valid": True,
            },
        )
        _atomic_write_json(
            run_dir / "runtime_environment.json",
            {
                "contact_mode": "instrumented",
                "actual_viewport_video": True,
                "video_path": str(video_path),
                "video_sha256": video_sha256,
                "video": video,
            },
        )
        _atomic_write_json(
            run_dir / "visual_recording_manifest.json",
            {
                "contact_mode": "instrumented",
                "actual_viewport_video": True,
                "not_camera_video": False,
                "video_path": str(video_path),
                "video_sha256": video_sha256,
                "video": video,
                "artifact_valid": True,
            },
        )
        _atomic_write_json(
            run_dir / "physical_evidence.json",
            {"strict_success": False, "classification": classification},
        )
        (run_dir / "fsm50_telemetry.csv").write_text(
            "time_s,classification\n0.0,GROUND\n", encoding="utf-8"
        )
        (run_dir / "fsm50_telemetry.jsonl").write_text(
            '{"time_s":0.0,"classification":"GROUND"}\n', encoding="utf-8"
        )
        (run_dir / "state_timeline.csv").write_text(
            "start_time_s,end_time_s\n0.0,0.0\n", encoding="utf-8"
        )
        _atomic_write_json(
            artifact_root / "artifact_pointer.json",
            {"run_dir": str(run_dir.resolve())},
        )
        _write_checksums(run_dir)
        (artifact_root / marker).write_text(marker.removeprefix(".") + "\n", encoding="utf-8")

        # Mirror the producer's durable ordering: write the successful
        # preclose documents and checksum, snapshot them immutably, create the
        # batch marker/handshake evidence, then let the parent record that
        # SimulationApp.close() returned normally and refresh live checksums.
        batch_root = artifact_root.parent.parent.resolve()
        batch_source_integrity = {"equal": True, "failures": []}
        _atomic_write_json(
            batch_root / "batch_request.json",
            {
                "schema_version": "fsm50.recording_replay_batch_request.v1",
                "created_utc": "2026-08-08T00:00:00+00:00",
                "versions": [item.version_id],
            },
        )
        _atomic_write_json(
            batch_root / "source_integrity.json", batch_source_integrity
        )
        _atomic_write_json(batch_root / "batch_results.json", [result])
        preclose_finalization = {
            "artifact_root": str(batch_root),
            "finalized": True,
            "failed": False,
            "strict_success": False,
            "batch_error": "",
            "close_error": "PENDING_SIMULATION_CLOSE",
            "phase": "PRECLOSE_FINALIZED",
            "source_integrity": batch_source_integrity,
            "finalization_errors": [],
        }
        _atomic_write_json(
            batch_root / "batch_finalization.json", preclose_finalization
        )
        _write_checksums(batch_root)
        snapshot_errors = _snapshot_preclose_files(batch_root)
        if snapshot_errors:
            raise AssertionError(f"resume fixture preclose snapshot failed: {snapshot_errors}")
        preclose_evidence = _preclose_evidence_manifest(
            batch_root,
            results=[result],
            batch_source_comparison=batch_source_integrity,
            batch_finalization=preclose_finalization,
        )
        (batch_root / ".finalized").write_text("finalized\n", encoding="utf-8")
        _atomic_write_json(
            batch_root / "preclose_complete.json",
            {
                "schema_version": "fsm50.preclose_complete.v1",
                "created_utc": "2026-08-08T00:00:01+00:00",
                "token": "resume-fixture",
                "parent_pid": 100,
                "child_pid": 101,
                "batch_root": str(batch_root),
                "evidence": preclose_evidence,
            },
        )
        if complete_shutdown:
            _record_shutdown_outcome(
                batch_root,
                {
                    "schema_version": "fsm50.shutdown_outcome.v1",
                    "created_utc": "2026-08-08T00:00:02+00:00",
                    "status": "NORMAL_EXIT",
                    "parent_pid": 100,
                    "child_pid": 101,
                    "child_returncode": 0,
                    "preclose_observed": True,
                    "handshake_state": "CLOSE_RETURNED",
                },
            )
        return artifact_root, run_dir

    def test_resume_accepts_source_matched_finalized_physical_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "runs"
            recording = Path(directory) / "recording"
            recording.mkdir()
            steps = recording / "accepted_steps.jsonl"
            metadata = recording / "metadata.json"
            steps.write_text("{}\n", encoding="utf-8")
            metadata.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(steps.read_bytes()).hexdigest()
            item = VersionFiles("v_test", recording, steps, metadata)
            _artifact_root, run_dir = self._write_resume_fixture(
                output_root,
                item,
                digest,
            )
            completed = _find_reliable_completed_replays(
                output_root,
                [item],
                {item.version_id: digest},
            )
            self.assertEqual([item.version_id], list(completed))
            self.assertEqual(
                (run_dir / "result.json").resolve(),
                Path(completed[item.version_id]["result_path"]),
            )

    def test_resume_rejects_stale_partial_and_incomplete_evidence(self) -> None:
        cases = ("partial", "missing_diagnostics", "wrong_source_hash")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                output_root = Path(directory) / "runs"
                recording = Path(directory) / "recording"
                recording.mkdir()
                steps = recording / "accepted_steps.jsonl"
                metadata = recording / "metadata.json"
                steps.write_text("{}\n", encoding="utf-8")
                metadata.write_text("{}\n", encoding="utf-8")
                digest = hashlib.sha256(steps.read_bytes()).hexdigest()
                item = VersionFiles("v_test", recording, steps, metadata)
                marker = ".partial" if case == "partial" else ".finalized"
                _artifact_root, run_dir = self._write_resume_fixture(
                    output_root,
                    item,
                    digest,
                    marker=marker,
                )
                if case == "missing_diagnostics":
                    (run_dir / "failure_diagnostics.json").unlink()
                expected = "0" * 64 if case == "wrong_source_hash" else digest
                self.assertEqual(
                    {},
                    _find_reliable_completed_replays(
                        output_root,
                        [item],
                        {item.version_id: expected},
                    ),
                )

    def test_resume_rejects_incomplete_or_tampered_batch_shutdown_closure(self) -> None:
        cases = ("missing_shutdown", "shutdown_timeout", "tampered_preclose")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                output_root = Path(directory) / "runs"
                recording = Path(directory) / "recording"
                recording.mkdir()
                steps = recording / "accepted_steps.jsonl"
                metadata = recording / "metadata.json"
                steps.write_text("{}\n", encoding="utf-8")
                metadata.write_text("{}\n", encoding="utf-8")
                digest = hashlib.sha256(steps.read_bytes()).hexdigest()
                item = VersionFiles("v_test", recording, steps, metadata)
                artifact_root, _run_dir = self._write_resume_fixture(
                    output_root,
                    item,
                    digest,
                    complete_shutdown=case != "missing_shutdown",
                )
                batch_root = artifact_root.parent.parent
                if case == "shutdown_timeout":
                    _record_shutdown_outcome(
                        batch_root,
                        {
                            "schema_version": "fsm50.shutdown_outcome.v1",
                            "created_utc": "2026-08-08T00:00:03+00:00",
                            "status": "SIMULATION_CLOSE_TIMEOUT",
                            "parent_pid": 100,
                            "child_pid": 101,
                            "child_returncode": None,
                            "preclose_observed": True,
                            "handshake_state": "PRECLOSE_COMPLETE",
                        },
                    )
                elif case == "tampered_preclose":
                    with (batch_root / "batch_finalization.preclose.json").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(" \n")

                self.assertEqual(
                    {},
                    _find_reliable_completed_replays(
                        output_root,
                        [item],
                        {item.version_id: digest},
                    ),
                )

    def test_first_failure_never_uses_final_top_as_lift_proof(self) -> None:
        evidence = {
            "traversal": {
                "legs": {
                    "FR": {
                        "unload_start_s": None,
                        "airborne_start_s": None,
                        "front_face_crossing_s": 1.0,
                        "top_contact_s": 1.1,
                        "top_load_confirm_s": 1.3,
                        "illegal_reasons": ["GROUND/FRONT_FACE to TOP transition without AIR"],
                    }
                }
            },
            "final_all_top": True,
            "final_all_loaded": True,
        }
        self.assertEqual("FR_UNLOAD_NOT_OBSERVED", _first_failure(evidence))

    def test_source_freeze_comparison_reports_exact_drift(self) -> None:
        before = {
            "created_utc": "before",
            "files": {
                "a.py": {"sha256": "a", "size_bytes": 1},
                "gone.py": {"sha256": "g", "size_bytes": 2},
            },
        }
        after = {
            "created_utc": "after",
            "files": {
                "a.py": {"sha256": "changed", "size_bytes": 1},
                "new.py": {"sha256": "n", "size_bytes": 3},
            },
        }
        result = _compare_source_freezes(before, after)
        self.assertFalse(result["equal"])
        self.assertEqual(["a.py"], result["changed"])
        self.assertEqual(["gone.py"], result["missing"])
        self.assertEqual(["new.py"], result["added"])

    def test_process_preflight_identifies_kit_and_sim_worker_only(self) -> None:
        rows = [
            {"pid": os.getpid(), "name": "python.exe", "command_line": "isaac-sim"},
            {"pid": 10101, "name": "kit.exe", "command_line": "kit.exe"},
            {
                "pid": 10102,
                "name": "python.exe",
                "command_line": "python -m sim_worker_runtime --port 1 ",
            },
            {"pid": 10103, "name": "python.exe", "command_line": "python tests.py"},
        ]
        self.assertEqual(
            [10101, 10102],
            [row["pid"] for row in _existing_simulator_processes(rows)],
        )

    def test_singleton_lock_is_atomic_and_owner_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runner.lock"
            first = ReplaySingletonLock(path)
            second = ReplaySingletonLock(path)
            first.acquire()
            with self.assertRaises(RuntimeError):
                second.acquire()
            first.release()
            second.acquire()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(os.getpid(), payload["pid"])
            second.release()
            self.assertFalse(path.exists())

    def test_singleton_lock_reclaims_a_dead_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runner.lock"
            path.write_text(
                json.dumps(
                    {
                        "pid": 2147483647,
                        "token": "dead-owner",
                        "created_utc": "past",
                    }
                ),
                encoding="utf-8",
            )
            lock = ReplaySingletonLock(path)
            lock.acquire()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(os.getpid(), payload["pid"])
            self.assertNotEqual("dead-owner", payload["token"])
            lock.release()

    def test_recording_audit_fails_closed_on_hash_mismatch(self) -> None:
        item = SimpleNamespace(version_id="v_bad", steps_path=Path("missing.jsonl"))

        class BadAudit:
            @staticmethod
            def audit_version(_item):
                return {
                    "version": "v_bad",
                    "valid": False,
                    "metadata_sha256_matches": False,
                    "missing_required_files": [],
                    "actual_step_count": 1,
                    "accepted_steps_sha256": "deadbeef",
                }

        with self.assertRaisesRegex(RuntimeError, "preflight rejected"):
            _fail_closed_recording_audits(BadAudit(), [item])

    def test_final_joint_error_excludes_wheel_joints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = root / "accepted_steps.jsonl"
            metadata = root / "metadata.json"
            steps.write_text("{}\n", encoding="utf-8")
            metadata.write_text("{}\n", encoding="utf-8")
            item = VersionFiles("v_test", root, steps, metadata)
            service = SimpleNamespace(
                stop_reason="complete",
                status_dict=lambda **_kwargs: {},
            )
            physical = {
                "physical_success": False,
                "traversal": {"legs": {}},
            }
            collector = SimpleNamespace(
                last_row={"time_s": 1.0},
                fsm50_rows=[
                    {
                        "base_roll_rad": 0.0,
                        "base_pitch_rad": 0.0,
                        "measured_joint_position_rad": {
                            "front_left_hip": 0.1,
                            "front_left_ankle": 100.0,
                        },
                        "actual_joint_target_rad": {
                            "front_left_hip": 0.0,
                            "front_left_ankle": 0.0,
                        },
                    }
                ],
                physical_evidence=lambda: physical,
            )
            plan = SimpleNamespace(
                profile="motion_only",
                plan_sha256="plan",
                events=[],
                segments=[],
                final_time_s=1.0,
            )
            result = _result_payload(
                item=item,
                service=service,
                collector=collector,
                respawn={},
                plan=plan,
                run_dir=root,
                timed_out=False,
            )
            self.assertAlmostEqual(0.1, result["final_joint_target_error_rad"])

    def test_dedicated_visualization_consumes_fsm_rows_timeline_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visualization = _generate_fsm50_visualization(
                root,
                fsm50_rows=[
                    {
                        "time_s": 0.0,
                        "base_roll_rad": 0.01,
                        "base_pitch_rad": -0.02,
                        "wheel_support_polygon_margin_m": 0.03,
                        "two_leg_corridor_distance_m": 0.01,
                        "source_step": 1,
                        "wheel_contact_force_up_n": {
                            "FL": 1.0,
                            "FR": 2.0,
                            "RL": 3.0,
                            "RR": 4.0,
                        },
                    }
                ],
                state_timeline_rows=[
                    {
                        "start_time_s": 0.0,
                        "end_time_s": 0.1,
                        "source_fast_segment": 0,
                        "source_step": 1,
                        "support_legs": ["FL", "RR"],
                        "primary_diagonal": "FL_RR",
                        "wheel_contact_classes": {"FL": "GROUND"},
                    }
                ],
                strict_result={
                    "classification": "PHYSICAL_FAILURE",
                    "strict_full_success": False,
                },
            )
            self.assertTrue(visualization["ok"])
            self.assertTrue(Path(visualization["png"]).is_file())
            html_text = Path(visualization["html"]).read_text(encoding="utf-8")
            self.assertIn("Strict success", html_text)
            self.assertIn("FL_RR", html_text)

    def test_runtime_environment_equivalence_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            robot = Path(directory) / "robot.usd"
            robot.write_bytes(b"robot")
            digest = hashlib.sha256(b"robot").hexdigest()
            lock = {
                "selected_environment": {
                    "robot_usd_sha256": digest,
                    "physics_dt_s": 1.0 / 120.0,
                    "render_interval_physics_steps": 8,
                    "servo_stiffness": 600.0,
                    "servo_damping": 60.0,
                    "wheel_damping": 20.0,
                    "wheel_maximum_velocity_rad_s": 2.0,
                    "wheel_reference_velocity_rad_s": 0.5,
                    "servo_reference_velocity_deg_s": 150.0,
                    "obstacle_height_m": 0.05,
                    "obstacle_front_face_x_m": 0.5,
                    "obstacle_length_m": 2.0,
                    "obstacle_width_m": 2.0,
                    "obstacle_bottom_z_m": 0.0,
                    "robot_initial_root_position_m": [0.0, 0.0, 0.1],
                    "robot_initial_root_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "source_sha256": {str(robot): digest},
            }
            config = SimpleNamespace(
                render_interval=8,
                servo_stiffness=600.0,
                servo_damping=60.0,
                wheel_damping=20.0,
                max_wheel_speed=2.0,
            )
            live_baseline = {"robot_root_pose": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]}
            live_obstacle = {
                "height_m": 0.05,
                "front_face_x_m": 0.5,
                "length_m": 2.0,
                "width_m": 2.0,
                "bottom_z_m": 0.0,
                "prim_valid": True,
                "visual_valid": True,
                "collision_valid": True,
            }
            motion = SimpleNamespace(
                wheel_reference_velocity_rad_s=0.5,
                servo_reference_velocity_deg_s=150.0,
            )
            result = _runtime_environment_equivalence(
                lock=lock,
                scene_config=config,
                live_baseline=live_baseline,
                live_obstacle=live_obstacle,
                motion=motion,
                robot_usd=robot,
                physics_dt_s=1.0 / 120.0,
            )
            self.assertTrue(result["ok"])
            live_obstacle["width_m"] = 1.5
            self.assertFalse(
                _runtime_environment_equivalence(
                    lock=lock,
                    scene_config=config,
                    live_baseline=live_baseline,
                    live_obstacle=live_obstacle,
                    motion=motion,
                    robot_usd=robot,
                    physics_dt_s=1.0 / 120.0,
                )["ok"]
            )

    def test_formal_replay_grounding_never_writes_historical_locked_pose(self) -> None:
        source = inspect.getsource(_run_recording_replays_locked)
        self.assertFalse(
            hasattr(run_fsm50_module, "_seed_adapter_from_locked_ground_pose")
        )
        self.assertIn("formal_worker_unseeded_ground_reference", source)
        self.assertLess(
            source.index("adapter = SimRobotAdapter("),
            source.index("ground = initialize_adapter_ground_reference(adapter)"),
        )

    def test_replay_args_round_trip_for_supervised_child(self) -> None:
        original = build_parser().parse_args(
            [
                "replay-recordings",
                "--versions",
                "v012",
                "--recording-root",
                "C:/recordings",
                "--output-root",
                "C:/runs",
                "--headless",
                "--resume",
            ]
        )
        restored = _deserialize_replay_args(_serialize_replay_args(original))
        self.assertEqual(original.command, restored.command)
        self.assertEqual(original.versions, restored.versions)
        self.assertEqual(original.recording_root, restored.recording_root)
        self.assertEqual(original.output_root, restored.output_root)
        self.assertTrue(restored.headless)
        self.assertTrue(restored.resume)

    @staticmethod
    def _write_supervisor_files(
        root: Path,
        *,
        token: str,
        parent_pid: int,
        child_pid: int,
        preclose: bool,
        state: str = "PRECLOSE_COMPLETE",
        intended_returncode: int = 0,
    ) -> tuple[Path, Path]:
        batch_root = root / "batch"
        batch_root.mkdir()
        source_integrity = {"equal": True, "failures": []}
        finalization = {
            "artifact_root": str(batch_root.resolve()),
            "finalized": True,
            "failed": False,
            "strict_success": True,
            "batch_error": "",
            "close_error": "",
            "phase": "PRECLOSE_FINALIZED",
            "source_integrity": source_integrity,
            "finalization_errors": [],
        }
        _atomic_write_json(batch_root / "batch_request.json", {"command": "test"})
        _atomic_write_json(batch_root / "batch_results.json", [])
        _atomic_write_json(batch_root / "batch_finalization.json", finalization)
        _atomic_write_json(batch_root / "source_integrity.json", source_integrity)
        _write_checksums(batch_root)
        errors = _snapshot_preclose_files(batch_root)
        if errors:
            raise AssertionError(errors)
        evidence = _preclose_evidence_manifest(
            batch_root,
            results=[],
            batch_source_comparison=source_integrity,
            batch_finalization=finalization,
        )
        handshake = root / "handshake.json"
        extras = {"intended_returncode": intended_returncode}
        if state in {"FAST_EXIT_REQUESTED", "FAST_CLOSE_RETURNED"}:
            extras.update(
                shutdown_mode="fast",
                runtime_version="5.1.0.0",
                close_kwargs={
                    "wait_for_replicator": False,
                    "skip_cleanup": True,
                },
            )
        elif state == "GRACEFUL_CLOSE_RETURNED":
            extras.update(
                shutdown_mode="graceful",
                close_kwargs={
                    "wait_for_replicator": False,
                    "skip_cleanup": False,
                },
            )
        elif state == "CLOSE_ERROR":
            extras.update(shutdown_mode="fast", close_error="unit-test")
        _atomic_write_json(
            handshake,
            {
                "schema_version": "fsm50.supervisor_handshake.v1",
                "token": token,
                "parent_pid": parent_pid,
                "child_pid": child_pid,
                "batch_root": str(batch_root.resolve()),
                "state": state,
                "sequence": 2,
                **extras,
            },
        )
        if preclose:
            _atomic_write_json(
                batch_root / "preclose_complete.json",
                {
                    "schema_version": "fsm50.preclose_complete.v1",
                    "token": token,
                    "parent_pid": parent_pid,
                    "child_pid": child_pid,
                    "batch_root": str(batch_root.resolve()),
                    "evidence": evidence,
                },
            )
        return handshake, batch_root

    def test_supervisor_normal_exit_after_durable_preclose(self) -> None:
        class Child:
            pid = 43001

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handshake, batch_root = self._write_supervisor_files(
                root,
                token="token",
                parent_pid=123,
                child_pid=Child.pid,
                preclose=True,
                state="FAST_EXIT_REQUESTED",
            )
            outcome, observed = _monitor_supervised_child(
                Child(),
                handshake_path=handshake,
                token="token",
                parent_pid=123,
                output_root=root,
                clock=lambda: 1.0,
                sleep=lambda _seconds: None,
            )
            self.assertEqual("FAST_EXIT_VERIFIED", outcome["status"])
            self.assertEqual(batch_root.resolve(), observed)
            self.assertTrue(outcome["preclose_observed"])
            self.assertTrue(outcome["preclose_verification"]["ok"])
            self.assertEqual(outcome["runtime_version"], "5.1.0.0")
            self.assertTrue(outcome["process_returned_normally"])

    def test_graceful_exit_has_a_distinct_verified_state(self) -> None:
        class Child:
            pid = 43011

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handshake, _batch_root = self._write_supervisor_files(
                root,
                token="token",
                parent_pid=123,
                child_pid=Child.pid,
                preclose=True,
                state="GRACEFUL_CLOSE_RETURNED",
            )
            outcome, _observed = _monitor_supervised_child(
                Child(),
                handshake_path=handshake,
                token="token",
                parent_pid=123,
                output_root=root,
                clock=lambda: 1.0,
                sleep=lambda _seconds: None,
            )

            self.assertEqual(outcome["status"], "GRACEFUL_EXIT")
            self.assertEqual(outcome["handshake_state"], "GRACEFUL_CLOSE_RETURNED")

    def test_fast_exit_rejects_tampered_preclose_checksum(self) -> None:
        class Child:
            pid = 43012

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handshake, batch_root = self._write_supervisor_files(
                root,
                token="token",
                parent_pid=123,
                child_pid=Child.pid,
                preclose=True,
                state="FAST_EXIT_REQUESTED",
            )
            with (batch_root / "batch_results.preclose.json").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(" \n")

            outcome, _observed = _monitor_supervised_child(
                Child(),
                handshake_path=handshake,
                token="token",
                parent_pid=123,
                output_root=root,
                clock=lambda: 1.0,
                sleep=lambda _seconds: None,
            )

            self.assertEqual(outcome["status"], "PRECLOSE_CLOSURE_INVALID")
            self.assertFalse(outcome["preclose_verification"]["ok"])

    def test_close_policy_uses_exact_fast_kwargs_without_graceful_alias(self) -> None:
        calls = []
        states = []

        class App:
            def close(self, **kwargs):
                calls.append(kwargs)

        class Supervisor:
            def mark_fast_exit_requested(self, **kwargs):
                states.append(("FAST_EXIT_REQUESTED", kwargs))

            def mark_fast_close_returned(self, **kwargs):
                states.append(("FAST_CLOSE_RETURNED", kwargs))

        mode = _close_simulation_with_explicit_policy(
            simulation_app=App(),
            scene_handle=None,
            args=SimpleNamespace(shutdown_mode="fast"),
            supervisor=Supervisor(),
            intended_returncode=0,
        )

        self.assertEqual(mode, "fast")
        self.assertEqual(
            calls,
            [{"wait_for_replicator": False, "skip_cleanup": True}],
        )
        self.assertEqual(
            [state for state, _payload in states],
            ["FAST_EXIT_REQUESTED", "FAST_CLOSE_RETURNED"],
        )
        self.assertNotIn("CLOSE_RETURNED", [state for state, _payload in states])

    def test_fast_close_policy_rejects_non_isaac_5_1_runtime(self) -> None:
        with mock.patch.object(
            run_fsm50_module,
            "_runtime_versions",
            return_value={"packages": {"isaacsim": "4.5.0"}},
        ):
            with self.assertRaisesRegex(RuntimeError, "restricted to.*Isaac 5.1"):
                _close_simulation_with_explicit_policy(
                    simulation_app=SimpleNamespace(close=lambda **_kwargs: None),
                    scene_handle=None,
                    args=SimpleNamespace(shutdown_mode="fast"),
                    supervisor=None,
                    intended_returncode=0,
                )

    def test_supervisor_post_preclose_exit_requires_successful_close_return(self) -> None:
        cases = (
            ("PRECLOSE_COMPLETE", 0, "CHILD_EXIT_BEFORE_CLOSE_RETURNED"),
            ("FAST_EXIT_REQUESTED", 7, "FAST_EXIT_FAILED"),
            ("CLOSE_ERROR", 0, "SIMULATION_CLOSE_ERROR"),
        )
        for index, (state, returncode, expected_status) in enumerate(cases):
            with self.subTest(state=state, returncode=returncode):
                child_pid = 43100 + index

                class Child:
                    pid = child_pid

                    @staticmethod
                    def poll():
                        return returncode

                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    handshake, batch_root = self._write_supervisor_files(
                        root,
                        token="token",
                        parent_pid=123,
                        child_pid=Child.pid,
                        preclose=True,
                        state=state,
                    )
                    outcome, observed = _monitor_supervised_child(
                        Child(),
                        handshake_path=handshake,
                        token="token",
                        parent_pid=123,
                        output_root=root,
                        clock=lambda: 1.0,
                        sleep=lambda _seconds: None,
                    )
                    self.assertEqual(expected_status, outcome["status"])
                    self.assertNotIn(
                        outcome["status"],
                        {"GRACEFUL_EXIT", "FAST_EXIT_VERIFIED"},
                    )
                    self.assertEqual(returncode, outcome["child_returncode"])
                    self.assertEqual(state, outcome["handshake_state"])
                    self.assertEqual(batch_root.resolve(), observed)
                    if state == "CLOSE_ERROR":
                        self.assertEqual(outcome["close_error"], "unit-test")
                        self.assertTrue(outcome["preclose_verification"]["ok"])

    def test_supervisor_timeout_starts_only_after_preclose_and_owns_exact_pid(self) -> None:
        class Child:
            pid = 43002
            returncode = None

            def poll(self):
                return self.returncode

        class Clock:
            value = -61.0

            def __call__(self):
                self.value += 61.0
                return self.value

        terminated: list[int] = []

        def terminate(child):
            terminated.append(child.pid)
            child.returncode = -9
            return {"pid": child.pid, "method": "unit-test"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handshake, batch_root = self._write_supervisor_files(
                root,
                token="token",
                parent_pid=123,
                child_pid=Child.pid,
                preclose=True,
            )
            child = Child()
            outcome, observed = _monitor_supervised_child(
                child,
                handshake_path=handshake,
                token="token",
                parent_pid=123,
                output_root=root,
                close_grace_s=60.0,
                poll_interval_s=0.0,
                clock=Clock(),
                sleep=lambda _seconds: None,
                terminate_tree=terminate,
            )
            self.assertEqual("SIMULATION_CLOSE_TIMEOUT", outcome["status"])
            self.assertEqual([Child.pid], terminated)
            self.assertEqual(batch_root.resolve(), observed)

    def test_no_global_timeout_before_preclose_marker(self) -> None:
        class Child:
            pid = 43003
            polls = iter([None, None, 7])

            @classmethod
            def poll(cls):
                return next(cls.polls)

        class Clock:
            value = -1000.0

            def __call__(self):
                self.value += 1000.0
                return self.value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handshake, _batch_root = self._write_supervisor_files(
                root,
                token="token",
                parent_pid=123,
                child_pid=Child.pid,
                preclose=False,
                state="BATCH_ALLOCATED",
            )
            outcome, _observed = _monitor_supervised_child(
                Child(),
                handshake_path=handshake,
                token="token",
                parent_pid=123,
                output_root=root,
                close_grace_s=0.001,
                poll_interval_s=0.0,
                clock=Clock(),
                sleep=lambda _seconds: None,
                terminate_tree=lambda _child: self.fail("must not terminate before preclose"),
            )
            self.assertEqual("CHILD_EXIT_BEFORE_PRECLOSE", outcome["status"])

    def test_close_timeout_marks_lifecycle_failed_without_reclassifying_physics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / "batch"
            run_dir = batch_root / "version" / "run"
            artifact_root = batch_root / "version" / "artifact"
            run_dir.mkdir(parents=True)
            artifact_root.mkdir(parents=True)
            (artifact_root / ".finalized").write_text("finalized\n", encoding="utf-8")
            result = {
                "source_version": "v012",
                "classification": "FULL_SUCCESS",
                "physical_success": True,
                "strict_full_success": True,
                "artifact_valid": True,
                "run_dir": str(run_dir),
                "artifact_root": str(artifact_root),
                "lifecycle": {
                    "finalized": True,
                    "failed": False,
                    "strict_success": True,
                },
            }
            _atomic_write_json(run_dir / "result.json", result)
            _atomic_write_json(batch_root / "batch_results.json", [result])
            _atomic_write_json(
                batch_root / "batch_finalization.json",
                {"finalized": True, "failed": False, "strict_success": True},
            )
            (batch_root / "checksums.preclose.sha256").write_text(
                "immutable-preclose-checksum\n", encoding="utf-8"
            )
            _atomic_write_json(
                batch_root / "source_integrity.json", {"equal": True}
            )
            _record_shutdown_outcome(
                batch_root,
                {
                    "status": "SIMULATION_CLOSE_TIMEOUT",
                    "child_pid": 43004,
                },
            )
            updated = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual("FULL_SUCCESS", updated["classification"])
            self.assertTrue(updated["physical_success"])
            self.assertTrue(updated["strict_full_success"])
            self.assertTrue(updated["lifecycle"]["failed"])
            self.assertEqual(
                "SIMULATION_CLOSE_TIMEOUT",
                updated["lifecycle"]["failure_reason"],
            )
            self.assertTrue((artifact_root / ".failed").is_file())
            self.assertTrue((batch_root / ".failed").is_file())
            self.assertEqual(
                "immutable-preclose-checksum\n",
                (batch_root / "checksums.preclose.sha256").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(
                {"equal": True},
                json.loads(
                    (batch_root / "source_integrity.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )

    def test_batch_marker_is_created_only_after_immutable_preclose_snapshot(self) -> None:
        source = inspect.getsource(_run_recording_replays_locked)
        snapshot_index = source.index(
            "immutable_preclose_errors = _snapshot_preclose_files(batch_root)"
        )
        failure_index = source.index(
            "batch_valid = False",
            snapshot_index,
        )
        marker_index = source.index(
            "_mark_artifact_root(batch_root, valid=batch_valid)",
            snapshot_index,
        )
        self.assertLess(snapshot_index, failure_index)
        self.assertLess(failure_index, marker_index)

    def test_preclose_snapshot_reports_each_copy_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            (batch_root / "batch_results.json").write_text("[]\n", encoding="utf-8")
            (batch_root / "batch_finalization.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (batch_root / "checksums.sha256").write_text("\n", encoding="utf-8")
            # A directory at the destination makes os.replace fail without
            # relying on permissions or platform-specific file locking.
            (batch_root / "batch_finalization.preclose.json").mkdir()
            errors = _snapshot_preclose_files(batch_root)
            self.assertEqual(1, len(errors))
            self.assertIn("batch_finalization.json", errors[0])
            self.assertTrue((batch_root / "batch_results.preclose.json").is_file())
            self.assertTrue((batch_root / "checksums.preclose.sha256").is_file())

    def test_normal_shutdown_outcome_does_not_rewrite_run_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / "batch"
            run_dir = batch_root / "run"
            run_dir.mkdir(parents=True)
            result = {
                "classification": "PHYSICAL_FAILURE",
                "physical_success": False,
                "strict_full_success": False,
                "run_dir": str(run_dir),
                "lifecycle": {"finalized": True, "failed": False},
            }
            _atomic_write_json(run_dir / "result.json", result)
            _atomic_write_json(batch_root / "batch_results.json", [result])
            _atomic_write_json(
                batch_root / "batch_finalization.json",
                {"finalized": True, "failed": False},
            )
            _record_shutdown_outcome(
                batch_root,
                {"status": "NORMAL_EXIT", "child_returncode": 0},
            )
            unchanged = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result, unchanged)


if __name__ == "__main__":
    unittest.main()
