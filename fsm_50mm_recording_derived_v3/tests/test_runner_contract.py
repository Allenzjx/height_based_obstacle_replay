from __future__ import annotations

import ast
import hashlib
import json
import inspect
import os
import tempfile
import textwrap
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
    _apply_batch_source_drift,
    _apply_recording_artifact_policy,
    _batch_command_exit_code,
    _batch_owned_command_context,
    _batch_owned_segment_cursor,
    _build_motion_start_readiness_evidence,
    _compare_source_freezes,
    _close_formal_recording_worker,
    _close_simulation_with_explicit_policy,
    _deserialize_replay_args,
    _evaluate_motion_start_window,
    _existing_simulator_processes,
    _fail_closed_recording_audits,
    _find_reliable_completed_replays,
    _finalize_worker_recording_batch_preclose,
    _first_failure,
    _generate_fsm50_visualization,
    _result_payload,
    _monitor_supervised_child,
    _new_directory,
    _normalize_recording_replay_role,
    _preclose_evidence_manifest,
    _process_returncode_after_close,
    _replace_with_windows_retry,
    _record_shutdown_outcome,
    _recording_artifact_request,
    _recording_worker_args,
    _runtime_environment_equivalence,
    _run_recording_replays_locked,
    _serialize_replay_args,
    _select_versions,
    _snapshot_preclose_files,
    _strict_json_equal,
    _validate_preclose_closure,
    _validate_recording_worker_ready_status,
    _validate_worker_artifact_complete_ack,
    _wait_for_recording_worker_ready,
    _wait_for_worker_operation_ack,
    _wait_for_worker_stop_ack,
    _viewport_preclose_evidence_paths,
    _write_checksums,
    build_parser,
)
from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES


class RunnerContractTests(unittest.TestCase):
    def test_strict_json_identity_does_not_coerce_bool_int_or_float(self) -> None:
        self.assertTrue(_strict_json_equal({"value": True}, {"value": True}))
        self.assertFalse(_strict_json_equal({"value": True}, {"value": 1}))
        self.assertFalse(_strict_json_equal({"value": 1}, {"value": 1.0}))

    def test_formal_worker_ready_status_binds_request_session_and_adapter(self) -> None:
        request = {
            "request_id": "request-1",
            "source_version": "v003",
            "trial_id": 1,
            "contact_mode": "instrumented",
            "environment_equivalence_role": "",
            "diagnostic_role": "",
            "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
            "gate1_eligible": True,
            "gate1_physical_qualification_eligible": True,
            "environment_equivalence_eligible": False,
            "artifact_root": str(Path("C:/batch/v003/artifact").resolve()),
            "plan_sha256": "a" * 64,
            "expected_root_state_write_count": 0,
        }
        session = {
            "request_id": "request-1",
            "worker_session_id": "session-1",
            "source_version": "v003",
            "trial_id": 1,
            "contact_mode": "instrumented",
            "environment_equivalence_role": "",
            "diagnostic_role": "",
            "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
            "gate1_eligible": True,
            "gate1_physical_qualification_eligible": True,
            "environment_equivalence_eligible": False,
            "artifact_root": request["artifact_root"],
            "adapter_runtime_instance_id": "adapter-1",
            "root_state_write_count": 0,
            "motion_start_ready": True,
            "readiness_frame_count": 10,
            "readiness_frame_count_required": 10,
            "readiness_sample_stride_physics_ticks": 8,
            "environment_equivalence": {"ok": True, "failed_checks": []},
            "contact_sensor_type": "WheelAndNonWheelContactSensorBank",
            "contact_sensor_error": "",
        }
        status = {
            "ready": True,
            "artifact_preflight_ready": True,
            "worker_pid": 1234,
            "worker_session_id": "session-1",
            "worker_artifact_session": session,
            "worker_artifact_preflight": {
                "artifact_request_sha256": "b" * 64,
                "expected_plan_sha256": "a" * 64,
                "environment_equivalence_role": "",
                "diagnostic_role": "",
                "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
                "gate1_eligible": True,
                "gate1_physical_qualification_eligible": True,
                "environment_equivalence_eligible": False,
            },
        }

        binding = _validate_recording_worker_ready_status(
            status,
            request=request,
            request_sha256="b" * 64,
            worker_pid=1234,
        )

        self.assertEqual("session-1", binding["worker_session_id"])
        self.assertEqual("adapter-1", binding["adapter_runtime_instance_id"])
        bad = json.loads(json.dumps(status))
        bad["worker_artifact_session"]["root_state_write_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "root-state write"):
            _validate_recording_worker_ready_status(
                bad,
                request=request,
                request_sha256="b" * 64,
                worker_pid=1234,
            )

    def test_worker_ready_waiter_raises_immediately_for_current_artifact_failure(
        self,
    ) -> None:
        failure_ack = {
            "type": "operation_ack",
            "operation": "recording_artifact",
            "phase": "ARTIFACT_FAILED",
            "request_id": "request-current",
            "accepted": False,
            "artifact_complete": False,
            "error": "artifact preflight exploded",
        }

        class Process:
            @staticmethod
            def poll():
                return None

        class Client:
            pid = 1234
            process = Process()

            def __init__(self) -> None:
                self.poll_count = 0

            def poll(self):
                self.poll_count += 1
                return []

            @staticmethod
            def status():
                return {
                    "ready": False,
                    "artifact_preflight_ready": False,
                    "starting": True,
                    "last_artifact_ack": failure_ack,
                    "artifact_ack_history": [failure_ack],
                }

        client = Client()
        with mock.patch.object(
            run_fsm50_module.time,
            "sleep",
            side_effect=AssertionError("current artifact failure must not sleep"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "ARTIFACT_FAILED|artifact preflight exploded",
            ):
                _wait_for_recording_worker_ready(
                    client,
                    request={"request_id": "request-current"},
                    request_sha256="b" * 64,
                    timeout_s=30.0,
                )
        self.assertEqual(1, client.poll_count)

    def test_worker_ready_waiter_ignores_other_request_artifact_failure(
        self,
    ) -> None:
        other_failure_ack = {
            "type": "operation_ack",
            "operation": "recording_artifact",
            "phase": "ARTIFACT_FAILED",
            "request_id": "request-other",
            "accepted": False,
            "artifact_complete": False,
            "error": "other request failed",
        }

        class Process:
            @staticmethod
            def poll():
                return None

        class Client:
            pid = 1234
            process = Process()

            def __init__(self) -> None:
                self.poll_count = 0

            def poll(self):
                self.poll_count += 1
                return []

            @staticmethod
            def status():
                return {
                    "ready": False,
                    "artifact_preflight_ready": False,
                    "starting": True,
                    "last_artifact_ack": other_failure_ack,
                    "artifact_ack_history": [other_failure_ack],
                }

        client = Client()
        clock_values = [0.0, 0.0]

        def clock() -> float:
            return clock_values.pop(0) if clock_values else 2.0

        with mock.patch.object(
            run_fsm50_module.time,
            "monotonic",
            side_effect=clock,
        ), mock.patch.object(run_fsm50_module.time, "sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                _wait_for_recording_worker_ready(
                    client,
                    request={"request_id": "request-current"},
                    request_sha256="b" * 64,
                    timeout_s=0.01,
                    poll_interval_s=0.001,
                )
        self.assertEqual(1, client.poll_count)

    def test_operation_ack_waiter_selects_exact_request_from_history(self) -> None:
        class Process:
            @staticmethod
            def poll():
                return None

        class Client:
            process = Process()

            @staticmethod
            def poll():
                return []

            @staticmethod
            def status():
                return {
                    "operation_ack_history": [
                        {
                            "operation": "start_playback_plan",
                            "request_id": "old",
                            "accepted": True,
                        },
                        {
                            "operation": "start_playback_plan",
                            "request_id": "request-1",
                            "accepted": True,
                        },
                    ]
                }

        ack = _wait_for_worker_operation_ack(
            Client(),
            operation="start_playback_plan",
            request_id="request-1",
            timeout_s=0.1,
            poll_interval_s=0.001,
        )
        self.assertEqual("request-1", ack["request_id"])

    def test_formal_worker_fast_close_uses_preclose_acks_and_process_return(
        self,
    ) -> None:
        binding = {
            "worker_pid": 1234,
            "worker_session_id": "worker-session",
            "adapter_runtime_instance_id": "adapter-instance",
            "artifact_request_id": "artifact-request",
        }
        fast_kwargs = {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        }

        class Process:
            returncode = None

            def poll(self):
                return self.returncode

        class Client:
            def __init__(self, *, include_close_requested: bool) -> None:
                self.process = Process()
                self.include_close_requested = include_close_requested
                self.shutdown_request_id = ""
                self._status: dict = {}
                self.closed = False

            def request_shutdown(self, *, mode: str, request_id: str) -> None:
                self.shutdown_request_id = request_id
                self.mode = mode

            def poll(self):
                if not self._status:
                    common = {
                        "request_id": self.shutdown_request_id,
                        **binding,
                        "root_state_write_count": 0,
                        "mode": "fast",
                        "accepted": True,
                        "error": "",
                        "close_kwargs": fast_kwargs,
                        "runtime_version": "5.1.0-test",
                    }
                    shutdown_ack = {
                        **common,
                        "type": "operation_ack",
                        "operation": "shutdown",
                    }
                    self._status = {
                        "operation_ack_history": [shutdown_ack],
                        "close_requested_ack": (
                            {**common, "type": "close_requested"}
                            if self.include_close_requested
                            else {}
                        ),
                        "close_returned_ack": {},
                    }
                    self.process.returncode = 0
                return []

            def status(self):
                return self._status

            def close(self) -> None:
                self.closed = True

        class Supervisor:
            def __init__(self) -> None:
                self.evidence = None

            def mark_fast_worker_process_returned(self, **kwargs) -> None:
                self.evidence = kwargs

        client = Client(include_close_requested=True)
        supervisor = Supervisor()
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            mode = _close_formal_recording_worker(
                client,
                supervisor=supervisor,
                worker_binding=binding,
                artifact_complete=True,
            )

        self.assertEqual("fast", mode)
        self.assertTrue(client.closed)
        self.assertIsNotNone(supervisor.evidence)
        self.assertTrue(supervisor.evidence["worker_close_requested"])
        self.assertFalse(supervisor.evidence["worker_close_returned"])
        self.assertEqual({}, supervisor.evidence["worker_close_returned_ack"])

        missing = Client(include_close_requested=False)
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "close_requested ACK is missing"):
                _close_formal_recording_worker(
                    missing,
                    supervisor=Supervisor(),
                    worker_binding=binding,
                    artifact_complete=True,
                )
        self.assertFalse(missing.closed)

    def test_failed_worker_without_startup_binding_closes_from_terminal_ack(
        self,
    ) -> None:
        request = {
            "request_id": "artifact-request",
            "contact_mode": "disabled",
            "environment_equivalence_role": "",
            "diagnostic_role": "U",
            "qualification_scope": "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC",
            "gate1_eligible": False,
            "gate1_physical_qualification_eligible": False,
            "environment_equivalence_eligible": False,
        }
        identity = {
            "worker_pid": 4321,
            "worker_session_id": "failed-session",
            "adapter_runtime_instance_id": "",
            "artifact_request_id": "artifact-request",
            "root_state_write_count": 0,
            "contact_mode": "disabled",
            "environment_equivalence_role": "",
            "diagnostic_role": "U",
            "qualification_scope": "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC",
            "gate1_eligible": False,
            "gate1_physical_qualification_eligible": False,
            "environment_equivalence_eligible": False,
        }
        terminal_ack = {
            "type": "operation_ack",
            "operation": "recording_artifact",
            "phase": "ARTIFACT_FAILED",
            "accepted": False,
            "artifact_complete": False,
            "request_id": "artifact-request",
            "error": "preflight failed",
            **identity,
        }

        class Process:
            pid = 4321
            returncode = None

            def poll(self):
                return self.returncode

        class Client:
            def __init__(self, *, failed_ack: dict) -> None:
                self.process = Process()
                self.pid = 4321
                self.failed_ack = dict(failed_ack)
                self.shutdown_request_id = ""
                self._status = {
                    "artifact_ack_history": [self.failed_ack],
                    "last_artifact_ack": self.failed_ack,
                }
                self.closed = False

            def request_shutdown(self, *, mode: str, request_id: str) -> None:
                self.shutdown_request_id = request_id
                self.mode = mode

            def poll(self):
                if self.process.returncode is None and self.shutdown_request_id:
                    shutdown_ack = {
                        "type": "operation_ack",
                        "operation": "shutdown",
                        "request_id": self.shutdown_request_id,
                        "mode": "normal",
                        "accepted": True,
                        "error": "",
                        "close_kwargs": {},
                        "runtime_version": "5.1.0-test",
                        **identity,
                    }
                    self._status = {
                        "operation_ack_history": [shutdown_ack],
                        "last_operation_ack": shutdown_ack,
                    }
                    self.process.returncode = 0
                return []

            def status(self):
                return self._status

            def close(self) -> None:
                self.closed = True

        class Supervisor:
            def __init__(self) -> None:
                self.evidence = None

            def mark_graceful_close_returned(self, **kwargs) -> None:
                self.evidence = kwargs

        client = Client(failed_ack=terminal_ack)
        supervisor = Supervisor()
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            mode = _close_formal_recording_worker(
                client,
                supervisor=supervisor,
                worker_binding={},
                artifact_complete=False,
                artifact_request=request,
            )
        self.assertEqual("normal", mode)
        self.assertTrue(client.closed)
        self.assertEqual("normal", client.mode)
        self.assertIsNone(supervisor.evidence)

        self_exit = Client(failed_ack=terminal_ack)
        self_exit.process.returncode = 1
        self_exit_supervisor = Supervisor()
        mode = _close_formal_recording_worker(
            self_exit,
            supervisor=self_exit_supervisor,
            worker_binding={},
            artifact_complete=False,
            artifact_request=request,
        )
        self.assertEqual("normal", mode)
        self.assertTrue(self_exit.closed)
        self.assertEqual("", self_exit.shutdown_request_id)
        self.assertIsNone(self_exit_supervisor.evidence)

        self_exit_zero = Client(failed_ack=terminal_ack)
        self_exit_zero.process.returncode = 0
        with self.assertRaisesRegex(
            RuntimeError, "exited zero without accepting normal shutdown"
        ):
            _close_formal_recording_worker(
                self_exit_zero,
                supervisor=Supervisor(),
                worker_binding={},
                artifact_complete=False,
                artifact_request=request,
            )
        self.assertTrue(self_exit_zero.closed)
        self.assertEqual("", self_exit_zero.shutdown_request_id)

        class SelfExitAfterRequestClient(Client):
            def poll(self):
                if self.shutdown_request_id:
                    self.process.returncode = 1
                return []

        exits_after_request = SelfExitAfterRequestClient(failed_ack=terminal_ack)
        exits_after_supervisor = Supervisor()
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            mode = _close_formal_recording_worker(
                exits_after_request,
                supervisor=exits_after_supervisor,
                worker_binding={},
                artifact_complete=False,
                artifact_request=request,
            )
        self.assertEqual("normal", mode)
        self.assertTrue(exits_after_request.closed)
        self.assertEqual("normal", exits_after_request.mode)
        self.assertIsNone(exits_after_supervisor.evidence)

        class ZeroExitWithoutAckClient(SelfExitAfterRequestClient):
            def poll(self):
                if self.shutdown_request_id:
                    self.process.returncode = 0
                return []

        zero_without_ack = ZeroExitWithoutAckClient(failed_ack=terminal_ack)
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            with self.assertRaisesRegex(
                RuntimeError, "exited zero without accepting normal shutdown"
            ):
                _close_formal_recording_worker(
                    zero_without_ack,
                    supervisor=Supervisor(),
                    worker_binding={},
                    artifact_complete=False,
                    artifact_request=request,
                )
        self.assertTrue(zero_without_ack.closed)

        class DelayedTerminalClient(Client):
            def __init__(self) -> None:
                super().__init__(failed_ack=terminal_ack)
                self._status = {}
                self.terminal_delivered = False

            def poll(self):
                if not self.terminal_delivered:
                    self._status = {
                        "artifact_ack_history": [self.failed_ack],
                        "last_artifact_ack": self.failed_ack,
                    }
                    self.terminal_delivered = True
                    return []
                return super().poll()

        delayed = DelayedTerminalClient()
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            mode = _close_formal_recording_worker(
                delayed,
                supervisor=Supervisor(),
                worker_binding={},
                artifact_complete=False,
                artifact_request=request,
            )
        self.assertEqual("normal", mode)
        self.assertTrue(delayed.terminal_delivered)
        self.assertTrue(delayed.closed)

        for label, mutation in (
            ("no exact terminal", {}),
            ("wrong worker PID", {**terminal_ack, "worker_pid": 4322}),
        ):
            with self.subTest(label=label):
                rejected = Client(failed_ack=mutation)
                with mock.patch.object(
                    run_fsm50_module.time, "sleep", return_value=None
                ):
                    with self.assertRaises(RuntimeError):
                        _close_formal_recording_worker(
                            rejected,
                            supervisor=Supervisor(),
                            worker_binding={},
                            artifact_complete=False,
                            artifact_request=request,
                        )
                self.assertTrue(rejected.shutdown_request_id)
                self.assertEqual("normal", rejected.mode)
                self.assertEqual(0, rejected.process.returncode)
                self.assertTrue(rejected.closed)

        pid_mismatch = Client(failed_ack=terminal_ack)
        pid_mismatch.process.pid = 4322
        with self.assertRaisesRegex(RuntimeError, "client/process PID mismatch"):
            _close_formal_recording_worker(
                pid_mismatch,
                supervisor=Supervisor(),
                worker_binding={},
                artifact_complete=False,
                artifact_request=request,
            )

        class TamperedShutdownClient(Client):
            def poll(self):
                if self.process.returncode is None and self.shutdown_request_id:
                    shutdown_ack = {
                        "type": "operation_ack",
                        "operation": "shutdown",
                        "request_id": self.shutdown_request_id,
                        "mode": "normal",
                        "accepted": True,
                        "error": "",
                        "close_kwargs": {},
                        "runtime_version": "5.1.0-test",
                        **identity,
                        "root_state_write_count": 1,
                    }
                    self._status = {
                        "operation_ack_history": [shutdown_ack],
                        "last_operation_ack": shutdown_ack,
                    }
                    self.process.returncode = 0
                return []

        tampered_shutdown = TamperedShutdownClient(failed_ack=terminal_ack)
        with mock.patch.object(
            run_fsm50_module.time, "sleep", return_value=None
        ):
            with self.assertRaisesRegex(
                RuntimeError, "shutdown ACK root_state_write_count mismatch"
            ):
                _close_formal_recording_worker(
                    tampered_shutdown,
                    supervisor=Supervisor(),
                    worker_binding={},
                    artifact_complete=False,
                    artifact_request=request,
                )
        self.assertFalse(tampered_shutdown.closed)

    def test_preplay_stop_waiter_binds_command_worker_and_zero_target(self) -> None:
        class Process:
            @staticmethod
            def poll():
                return None

        class Client:
            process = Process()

            @staticmethod
            def poll():
                return []

            @staticmethod
            def status():
                return {
                    "operation_ack_history": [
                        {
                            "type": "stop_ack",
                            "command_id": "stop-1",
                            "worker_pid": 1234,
                            "worker_session_id": "session-1",
                            "root_state_write_count": 0,
                            "zero_target_applied": True,
                            "error": "",
                        }
                    ]
                }

        ack = _wait_for_worker_stop_ack(
            Client(),
            command_id="stop-1",
            worker_pid=1234,
            worker_session_id="session-1",
            timeout_s=0.1,
            poll_interval_s=0.001,
        )
        self.assertTrue(ack["zero_target_applied"])

    def test_recording_worker_args_freeze_production_scene_and_gate_request(self) -> None:
        args = SimpleNamespace(
            device="cuda:0",
            headless=False,
            livestream=0,
            experience="",
        )
        motion = SimpleNamespace(
            wheel_velocity_limit_rad_s=2.0943951023931953,
            wheel_reference_velocity_rad_s=0.5235987755982988,
        )
        worker = _recording_worker_args(
            args,
            robot_usd=Path("C:/robot.usd"),
            motion=motion,
            artifact_request_path=Path("C:/batch/request.json"),
        )
        self.assertEqual(50, worker.height_mm)
        self.assertEqual(1.0 / 120.0, worker.physics_dt)
        self.assertEqual(8, worker.render_interval)
        self.assertFalse(worker.save_scene)
        self.assertFalse(worker.no_continuous_sim_step)
        self.assertFalse(worker.robot_auto_ground_correction)
        self.assertEqual(0.003, worker.robot_ground_penetration_tolerance_m)
        self.assertEqual(
            Path("C:/batch/request.json").resolve(),
            Path(worker.fsm50_gate_request_path),
        )

    def test_worker_artifact_complete_ack_revalidates_sealed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory).resolve()
            artifact_root = batch_root / "v003" / "artifact"
            run_dir = artifact_root / "run"
            run_dir.mkdir(parents=True)
            request = {
                "request_id": "request-1",
                "plan_id": "plan-1",
                "plan_sha256": "a" * 64,
                "artifact_root": str(artifact_root),
                "source_version": "v003",
                "trial_id": 1,
                "accepted_steps_sha256": "b" * 64,
                "contact_mode": "instrumented",
                "environment_equivalence_role": "",
                "diagnostic_role": "",
                "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
                "gate1_eligible": True,
                "gate1_physical_qualification_eligible": True,
                "environment_equivalence_eligible": False,
                "expected_accepted_steps_sha256": "b" * 64,
            }
            request_sha = "c" * 64
            telemetry = run_dir / "telemetry_finalization.json"
            telemetry.write_text('{"canonical_complete":true}\n', encoding="utf-8")
            visual = run_dir / "visual_recording_manifest.json"
            visual.write_text('{"artifact_valid":true}\n', encoding="utf-8")
            result_path = run_dir / "result.json"
            result = {
                "artifact_owner": "sim_worker_process",
                "execution_path": "sim_worker_process_ipc",
                "artifact_request_sha256": request_sha,
                "request_id": "request-1",
                "plan_id": "plan-1",
                "artifact_root": str(artifact_root),
                "worker_pid": 1234,
                "worker_session_id": "worker-session",
                "adapter_runtime_instance_id": "adapter-1",
                "root_state_write_count": 0,
                "run_dir": str(run_dir),
                "source_version": "v003",
                "trial_id": 1,
                "plan_sha256": "a" * 64,
                "accepted_steps_sha256": "b" * 64,
                "contact_mode": "instrumented",
                "environment_equivalence_role": "",
                "environment_equivalence_diagnostic": False,
                "environment_equivalence_diagnostic_complete": False,
                "diagnostic_role": "",
                "ordinary_ui_diagnostic": False,
                "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
                "gate1_eligible": True,
                "gate1_physical_qualification_eligible": True,
                "environment_equivalence_eligible": False,
                "respawn": {"root_state_write_count": 0},
                "artifact_valid": False,
                "classification": "ARTIFACT_INVALID",
                "scheduler_complete": True,
                "physical_success": True,
                "strict_full_success": False,
                "lifecycle": {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                },
                "required_evidence_paths": [
                    str(result_path),
                    str(telemetry),
                    str(visual),
                    str(run_dir / "checksums.sha256"),
                ],
            }
            _atomic_write_json(result_path, result)
            pointer_path = artifact_root / "artifact_pointer.json"
            _atomic_write_json(pointer_path, {"run_dir": str(run_dir)})
            _write_checksums(run_dir)
            checksums = run_dir / "checksums.sha256"
            (artifact_root / ".failed").write_text(
                "failed\n", encoding="utf-8"
            )
            ack = {
                "type": "operation_ack",
                "operation": "recording_artifact",
                "phase": "ARTIFACT_COMPLETE",
                "accepted": True,
                "artifact_complete": True,
                "error": "",
                "artifact_owner": "sim_worker_process",
                "request_id": "request-1",
                "artifact_request_id": "request-1",
                "plan_id": "plan-1",
                "plan_sha256": "a" * 64,
                "artifact_request_sha256": request_sha,
                "worker_pid": 1234,
                "worker_session_id": "worker-session",
                "adapter_runtime_instance_id": "adapter-1",
                "source_version": "v003",
                "trial_id": 1,
                "contact_mode": "instrumented",
                "environment_equivalence_role": "",
                "environment_equivalence_diagnostic": False,
                "diagnostic_role": "",
                "ordinary_ui_diagnostic": False,
                "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
                "gate1_eligible": True,
                "gate1_physical_qualification_eligible": True,
                "environment_equivalence_eligible": False,
                "accepted_steps_sha256": "b" * 64,
                "root_state_write_count": 0,
                "artifact_root": str(artifact_root),
                "run_dir": str(run_dir),
                "result_path": str(result_path),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "artifact_pointer_path": str(pointer_path),
                "artifact_pointer_sha256": hashlib.sha256(pointer_path.read_bytes()).hexdigest(),
                "checksums_path": str(checksums),
                "checksums_sha256": hashlib.sha256(checksums.read_bytes()).hexdigest(),
                "telemetry_finalization_path": str(telemetry),
                "telemetry_finalization_sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
                "visual_manifest_path": str(visual),
                "visual_manifest_sha256": hashlib.sha256(visual.read_bytes()).hexdigest(),
                "artifact_valid": False,
                "classification": "ARTIFACT_INVALID",
                "scheduler_complete": True,
                "physical_success": True,
                "strict_full_success": False,
                "environment_equivalence_diagnostic_complete": False,
            }
            loaded = _validate_worker_artifact_complete_ack(
                ack,
                request=request,
                request_sha256=request_sha,
                batch_root=batch_root,
                worker_pid=1234,
                worker_session_id="worker-session",
                adapter_runtime_instance_id="adapter-1",
            )
            self.assertEqual(result, loaded)
            bad = dict(ack)
            bad["worker_pid"] = 9999
            with self.assertRaisesRegex(RuntimeError, "worker_pid mismatch"):
                _validate_worker_artifact_complete_ack(
                    bad,
                    request=request,
                    request_sha256=request_sha,
                    batch_root=batch_root,
                    worker_pid=1234,
                    worker_session_id="worker-session",
                    adapter_runtime_instance_id="adapter-1",
                )

    def test_failed_viewport_capture_preserves_diagnostics_without_missing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            wrapper = run_dir / "viewport_video_manifest.json"
            low_level = run_dir / "viewport_buffer_video_manifest.json"
            ledger = run_dir / "viewport_frame_ledger.jsonl"
            wrapper.write_text('{"valid":false}\n', encoding="utf-8")
            low_level.write_text('{"valid":false}\n', encoding="utf-8")
            ledger.write_text("", encoding="utf-8")
            video = {
                "actual_viewport_video": False,
                "manifest_path": str(wrapper),
                "video_path": str(run_dir / "actual_viewport_video.mp4"),
            }

            paths = _viewport_preclose_evidence_paths(run_dir, video)

            self.assertEqual({wrapper, low_level, ledger}, set(paths))
            self.assertNotIn(run_dir / "viewport_first_frame.png", paths)
            self.assertNotIn(run_dir / "viewport_last_frame.png", paths)
            self.assertNotIn(run_dir / "actual_viewport_video.mp4", paths)

    def test_valid_viewport_claim_requires_complete_direct_capture_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            wrapper = run_dir / "viewport_video_manifest.json"
            video_path = run_dir / "actual_viewport_video.mp4"
            video = {
                "actual_viewport_video": True,
                "manifest_path": str(wrapper),
                "video_path": str(video_path),
            }

            paths = _viewport_preclose_evidence_paths(run_dir, video)

            self.assertEqual(
                {
                    wrapper,
                    run_dir / "viewport_buffer_video_manifest.json",
                    run_dir / "viewport_frame_ledger.jsonl",
                    run_dir / "viewport_first_frame.png",
                    run_dir / "viewport_last_frame.png",
                    video_path,
                },
                set(paths),
            )

    def test_atomic_json_writer_replaces_nonfinite_values_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            _atomic_write_json(
                path,
                {
                    "value": float("nan"),
                    "nested": [float("inf"), float("-inf")],
                    "valid": False,
                    "reason": "unavailable",
                },
            )

            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            decoded = json.loads(
                text,
                parse_constant=lambda value: self.fail(
                    f"non-finite JSON constant {value}"
                ),
            )
            self.assertIsNone(decoded["value"])
            self.assertEqual(decoded["nested"], [None, None])
            self.assertFalse(decoded["valid"])
            self.assertEqual(decoded["reason"], "unavailable")

    def test_artifact_policy_requires_hashed_canonical_telemetry_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            marker_path = run_dir / "telemetry_finalization.json"
            _atomic_write_json(
                marker_path,
                {
                    "schema_version": "telemetry.canonical_finalization.v1",
                    "canonical_export_attempted": True,
                    "canonical_complete": True,
                    "stream_counts": {},
                    "canonical_files": {},
                    "journal": {
                        "removed_after_success": True,
                        "errors": [],
                    },
                    "errors": [],
                },
            )
            finalization = {
                "schema_version": "telemetry.canonical_finalization.v1",
                "canonical_export_attempted": True,
                "canonical_complete": True,
                "stream_counts": {},
                "canonical_files": {},
                "journal": {
                    "removed_after_success": True,
                    "errors": [],
                },
                "errors": [],
                "marker_path": str(marker_path),
                "marker_sha256": hashlib.sha256(
                    marker_path.read_bytes()
                ).hexdigest(),
            }
            result = {
                "run_dir": str(run_dir),
                "source_integrity": {"ok": True},
                "motion_start_ready": True,
                "dispatch_complete": True,
                "wheel_target_integral_verdict": "PASS",
                "strict_full_success": False,
                "telemetry_finalization": finalization,
            }
            self.assertTrue(
                _apply_recording_artifact_policy(
                    result,
                    video={"actual_viewport_video": True},
                    visualization={"ok": True},
                )
            )

            journal_dir = run_dir / ".telemetry_journal"
            journal_dir.mkdir()
            result["classification"] = "PHYSICAL_FAILURE"
            self.assertFalse(
                _apply_recording_artifact_policy(
                    result,
                    video={"actual_viewport_video": True},
                    visualization={"ok": True},
                )
            )
            journal_dir.rmdir()

            marker_path.unlink()
            result["classification"] = "PHYSICAL_FAILURE"
            self.assertFalse(
                _apply_recording_artifact_policy(
                    result,
                    video={"actual_viewport_video": True},
                    visualization={"ok": True},
                )
            )
            self.assertEqual(
                result["first_failure_phase"],
                "TELEMETRY_FINALIZATION_INCOMPLETE",
            )

    def test_motion_start_window_uses_formal_render_loop_cadence(self) -> None:
        adapter = SimpleNamespace(
            runtime_instance_id="adapter-1",
            root_state_write_count=0,
            config=SimpleNamespace(
                ground_vertical_speed_threshold_m_s=0.01,
                ground_wheel_speed_threshold_rad_s=0.20,
                ground_penetration_tolerance_m=0.003,
            ),
            _render_step_timing=lambda: (8.0 / 120.0, 8),
        )
        frames = [{"sim_step": 180 + 8 * index} for index in range(10)]
        with mock.patch.object(
            run_fsm50_module,
            "evaluate_motion_start_ready",
            return_value={"ready": True, "failed_checks": [], "status": "PASS"},
        ):
            result = _evaluate_motion_start_window(
                adapter=adapter,
                startup_ground={},
                frames=frames,
                plan_identity={"plan_sha256": "a" * 64},
                required_frames=10,
            )

        self.assertTrue(result["ready"])
        self.assertFalse(result["contiguous_sim_steps"])
        self.assertTrue(result["production_sampling_cadence_valid"])
        self.assertEqual(result["production_step_stride_physics_ticks"], 8)
        self.assertEqual(result["observed_sim_step_deltas"], [8] * 9)
        self.assertEqual(result["window_failed_checks"], [])

    def test_motion_start_window_uses_observed_dispatch_idle_evidence(self) -> None:
        adapter = SimpleNamespace(
            runtime_instance_id="adapter-1",
            root_state_write_count=0,
            config=SimpleNamespace(
                ground_vertical_speed_threshold_m_s=0.01,
                ground_wheel_speed_threshold_rad_s=0.20,
                ground_penetration_tolerance_m=0.003,
            ),
            _render_step_timing=lambda: (8.0 / 120.0, 8),
        )
        frames = [{"sim_step": 180 + 8 * index} for index in range(10)]
        evidence = {
            "direct_dispatch_attempt_count": 1,
            "source_segment_start_count": 0,
        }

        def evaluate(**kwargs):
            idle = kwargs["command_dispatch_idle"]
            return {
                "ready": bool(idle),
                "failed_checks": [] if idle else [
                    "plan_identity_bound_and_no_prior_dispatch"
                ],
                "status": "PASS" if idle else "FAIL",
            }

        with mock.patch.object(
            run_fsm50_module,
            "evaluate_motion_start_ready",
            side_effect=evaluate,
        ) as evaluator:
            result = _evaluate_motion_start_window(
                adapter=adapter,
                startup_ground={},
                frames=frames,
                plan_identity={"plan_sha256": "a" * 64},
                required_frames=10,
                command_dispatch_idle=False,
                command_dispatch_evidence=evidence,
            )

        self.assertFalse(result["ready"])
        self.assertFalse(result["command_dispatch_idle"])
        self.assertEqual(evidence, result["command_dispatch_evidence"])
        self.assertEqual(10, evaluator.call_count)
        self.assertTrue(
            all(
                call.kwargs["command_dispatch_idle"] is False
                for call in evaluator.call_args_list
            )
        )
    def test_motion_start_window_rejects_a_missing_production_sample(self) -> None:
        adapter = SimpleNamespace(
            runtime_instance_id="adapter-1",
            root_state_write_count=0,
            config=SimpleNamespace(
                ground_vertical_speed_threshold_m_s=0.01,
                ground_wheel_speed_threshold_rad_s=0.20,
                ground_penetration_tolerance_m=0.003,
            ),
            _render_step_timing=lambda: (8.0 / 120.0, 8),
        )
        steps = [180, 188, 196, 204, 212, 220, 228, 236, 252, 260]
        frames = [{"sim_step": step} for step in steps]
        with mock.patch.object(
            run_fsm50_module,
            "evaluate_motion_start_ready",
            return_value={"ready": True, "failed_checks": [], "status": "PASS"},
        ):
            result = _evaluate_motion_start_window(
                adapter=adapter,
                startup_ground={},
                frames=frames,
                plan_identity={"plan_sha256": "a" * 64},
                required_frames=10,
            )

        self.assertFalse(result["ready"])
        self.assertFalse(result["production_sampling_cadence_valid"])
        self.assertIn(16, result["observed_sim_step_deltas"])
        self.assertTrue(result["window_failed_checks"])

    def test_motion_start_window_fails_closed_without_valid_cadence(self) -> None:
        base_adapter = {
            "runtime_instance_id": "adapter-1",
            "root_state_write_count": 0,
            "config": SimpleNamespace(
                ground_vertical_speed_threshold_m_s=0.01,
                ground_wheel_speed_threshold_rad_s=0.20,
                ground_penetration_tolerance_m=0.003,
            ),
        }
        adapters = {
            "missing": SimpleNamespace(**base_adapter),
            "raises": SimpleNamespace(
                **base_adapter,
                _render_step_timing=lambda: (_ for _ in ()).throw(
                    ValueError("unavailable")
                ),
            ),
            "invalid": SimpleNamespace(
                **base_adapter,
                _render_step_timing=lambda: (0.0, 0),
            ),
        }
        frames = [{"sim_step": 180 + 8 * index} for index in range(10)]
        with mock.patch.object(
            run_fsm50_module,
            "evaluate_motion_start_ready",
            return_value={"ready": True, "failed_checks": [], "status": "PASS"},
        ):
            for label, adapter in adapters.items():
                with self.subTest(label=label):
                    result = _evaluate_motion_start_window(
                        adapter=adapter,
                        startup_ground={},
                        frames=frames,
                        plan_identity={"plan_sha256": "a" * 64},
                        required_frames=10,
                    )
                    self.assertFalse(result["ready"])
                    self.assertFalse(
                        result["production_sampling_cadence_valid"]
                    )
                    self.assertIsNone(
                        result["production_step_stride_physics_ticks"]
                    )
                    self.assertTrue(
                        result["production_sampling_cadence_error"]
                    )
                    self.assertTrue(result["window_failed_checks"])

    def test_motion_start_token_evidence_is_json_safe_and_not_self_referential(self) -> None:
        readiness = {
            "ready": True,
            "frames": [{"sim_step": 42, "joint": {"q": [0.0]}}],
        }
        evidence, token = _build_motion_start_readiness_evidence(
            readiness=readiness,
            source_version="v003_20260805_224517_157723_manual",
            trial_id=1,
            plan_identity={"plan_sha256": "a" * 64},
            adapter_runtime_instance_id="adapter-1",
            root_state_write_count=0,
            pre_first_dispatch_sim_step=42,
        )

        encoded = json.dumps(evidence, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(len(token), 64)
        self.assertEqual(decoded["readiness_token_sha256"], token)
        self.assertEqual(
            decoded["token_payload"]["pre_first_dispatch_readiness"],
            readiness,
        )
        self.assertNotIn(
            "token_payload",
            decoded["token_payload"]["pre_first_dispatch_readiness"],
        )

    def test_applied_batch_owns_zero_duration_segment_telemetry_cursor(self) -> None:
        plan = SimpleNamespace(
            segments=[
                SimpleNamespace(source_step=3),
                SimpleNamespace(source_step=4),
            ]
        )
        self.assertEqual(
            _batch_owned_segment_cursor(
                scheduler_segment_index=1,
                current_motion_batch={
                    "segment_index": 0,
                    "source_step_index": 3,
                    "dispatch_kind": "source_segment_start",
                },
                plan=plan,
            ),
            (0, 3),
        )
        self.assertEqual(
            _batch_owned_segment_cursor(
                scheduler_segment_index=1,
                current_motion_batch={},
                plan=plan,
            ),
            (1, 4),
        )
        self.assertEqual(
            _batch_owned_segment_cursor(
                scheduler_segment_index=2,
                current_motion_batch={},
                plan=plan,
            ),
            (1, 4),
        )

    def test_applied_batch_owns_command_provenance_and_runtime_stop_clears_it(self) -> None:
        timing_commands = [
            {
                "global_command_index": 7,
                "command": "wheel front_left_ankle stop",
                "planned_start_sim_time": 1.0,
                "actual_start_sim_time": 1.008333333,
                "atomic_batch_id": "source-7",
            },
            {
                "global_command_index": 8,
                "command": "servo front_left_hip 12",
                "planned_start_sim_time": 1.1,
                "actual_start_sim_time": 1.108333333,
                "atomic_batch_id": "source-8",
            },
        ]
        context = _batch_owned_command_context(
            current_motion_batch={
                "batch_id": "source-7",
                "dispatch_kind": "source_segment_start",
                "global_command_indices": [7],
                "plan_event_indices": [6],
            },
            timing_commands=timing_commands,
            source_event_by_plan_index={6: 23, 7: 24},
        )
        self.assertEqual(context["command_cursor"], 7)
        self.assertEqual(context["source_event_index"], 23)
        self.assertEqual(
            context["source_command"], "wheel front_left_ankle stop"
        )
        self.assertEqual(context["atomic_batch_id"], "source-7")

        final_stop = _batch_owned_command_context(
            current_motion_batch={
                "batch_id": "final-stop",
                "dispatch_kind": "final_safety_stop",
                "global_command_indices": [],
                "plan_event_indices": [],
            },
            timing_commands=timing_commands,
            source_event_by_plan_index={6: 23, 7: 24},
        )
        self.assertIsNone(final_stop["command_cursor"])
        self.assertEqual(final_stop["source_command"], "")
        self.assertIsNone(final_stop["source_event_index"])
        self.assertEqual(final_stop["atomic_batch_id"], "final-stop")
        self.assertEqual(final_stop["dispatch_kind"], "final_safety_stop")

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
            journal = child / ".telemetry_journal" / "checkpoint-000001-test"
            journal.mkdir(parents=True)
            (journal / "commit.json").write_text("{}\n", encoding="utf-8")
            (child / ".telemetry.json.1.tmp").write_text(
                "temporary\n", encoding="utf-8"
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
            self.assertNotIn(".telemetry_journal", rows)
            self.assertNotIn(".telemetry.json.1.tmp", rows)
            self.assertNotIn("batch_results.preclose.json", rows)
            self.assertNotIn("batch_finalization.preclose.json", rows)
            self.assertNotIn("checksums.preclose.sha256", rows)

    def test_fast_process_success_is_separate_from_command_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _atomic_write_json(
                root / "batch_finalization.json",
                {
                    "phase": "SHUTDOWN_COMPLETE",
                    "finalized": True,
                    "failed": False,
                    "strict_success": False,
                    "batch_error": "",
                    "close_error": "",
                    "finalization_errors": [],
                },
            )
            (root / ".finalized").write_text("finalized\n", encoding="utf-8")
            self.assertEqual(
                _batch_command_exit_code(root, fallback=0),
                1,
            )
            _atomic_write_json(
                root / "batch_finalization.json",
                {
                    "phase": "SHUTDOWN_COMPLETE",
                    "finalized": True,
                    "failed": False,
                    "strict_success": True,
                    "batch_error": "",
                    "close_error": "",
                    "finalization_errors": [],
                },
            )
            self.assertEqual(
                _batch_command_exit_code(root, fallback=1),
                0,
            )

            for field, value in (
                ("failed", True),
                ("batch_error", "late artifact write failed"),
                ("finalization_errors", ["checksum failed"]),
            ):
                payload = {
                    "phase": "SHUTDOWN_COMPLETE",
                    "finalized": True,
                    "failed": False,
                    "strict_success": True,
                    "batch_error": "",
                    "close_error": "",
                    "finalization_errors": [],
                }
                payload[field] = value
                _atomic_write_json(root / "batch_finalization.json", payload)
                self.assertEqual(_batch_command_exit_code(root, fallback=0), 1)

    def test_environment_diagnostic_exit_depends_on_sealed_capture_not_physics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".finalized").write_text("finalized\n", encoding="utf-8")
            payload = {
                "phase": "SHUTDOWN_COMPLETE",
                "finalized": True,
                "failed": False,
                "strict_success": False,
                "environment_equivalence_role": "A1",
                "environment_equivalence_diagnostic_complete": True,
                "command_success": True,
                "batch_error": "",
                "close_error": "",
                "finalization_errors": [],
            }
            _atomic_write_json(root / "batch_finalization.json", payload)

            self.assertEqual(_batch_command_exit_code(root, fallback=1), 0)

            payload["environment_equivalence_diagnostic_complete"] = False
            payload["command_success"] = False
            _atomic_write_json(root / "batch_finalization.json", payload)
            self.assertEqual(_batch_command_exit_code(root, fallback=0), 1)

    def test_short_version_selector_is_unambiguous(self) -> None:
        versions = RecordingAudit().enumerate_versions()
        selected = _select_versions(versions, ["v010", "v012"])
        self.assertEqual(["v010", "v012"], [item.version_id.split("_", 1)[0] for item in selected])

    def test_all_selector_keeps_every_physical_directory(self) -> None:
        versions = RecordingAudit().enumerate_versions()
        self.assertEqual(versions, _select_versions(versions, ["all"]))

    def test_scheduler_and_physical_fields_are_independent_cli_contract(self) -> None:
        args = build_parser().parse_args(
            [
                "replay-recordings",
                "--versions",
                "v012",
                "--trial-id",
                "1",
                "--headless",
            ]
        )
        self.assertEqual("replay-recordings", args.command)
        self.assertEqual(["v012"], args.versions)
        self.assertTrue(args.headless)
        self.assertFalse(args.resume)

    def test_environment_equivalence_role_selects_exact_contact_mode(self) -> None:
        for role, expected_mode in (
            ("A1", "formal"),
            ("A2", "formal"),
            ("B", "instrumented"),
        ):
            args = build_parser().parse_args(
                [
                    "replay-recordings",
                    "--versions",
                    "v003",
                    "--trial-id",
                    "1",
                    "--environment-equivalence-role",
                    role,
                ]
            )
            _normalize_recording_replay_role(args)
            self.assertEqual(args.environment_equivalence_role, role)
            self.assertEqual(args.contact_mode, expected_mode)

        gate1 = build_parser().parse_args(
            ["replay-recordings", "--versions", "v003", "--trial-id", "1"]
        )
        _normalize_recording_replay_role(gate1)
        self.assertEqual(gate1.environment_equivalence_role, "")
        self.assertEqual(gate1.contact_mode, "instrumented")

    def test_environment_equivalence_role_rejects_conflicting_contact_mode(self) -> None:
        for role, contact_mode in (
            ("A1", "instrumented"),
            ("A2", "instrumented"),
            ("B", "formal"),
        ):
            args = build_parser().parse_args(
                [
                    "replay-recordings",
                    "--versions",
                    "v003",
                    "--trial-id",
                    "1",
                    "--environment-equivalence-role",
                    role,
                    "--contact-mode",
                    contact_mode,
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "requires --contact-mode"):
                _normalize_recording_replay_role(args)

        plain_formal = build_parser().parse_args(
            [
                "replay-recordings",
                "--versions",
                "v003",
                "--trial-id",
                "1",
                "--contact-mode",
                "formal",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "explicit.*role"):
            _normalize_recording_replay_role(plain_formal)

    def test_ordinary_ui_u_role_derives_disabled_and_is_mutually_exclusive(self) -> None:
        args = build_parser().parse_args(
            [
                "replay-recordings",
                "--versions",
                "v003",
                "--trial-id",
                "1",
                "--diagnostic-role",
                "U",
            ]
        )
        _normalize_recording_replay_role(args)
        self.assertEqual(args.diagnostic_role, "U")
        self.assertEqual(args.environment_equivalence_role, "")
        self.assertEqual(args.contact_mode, "disabled")

        conflicts = (
            ["--environment-equivalence-role", "A1"],
            ["--contact-mode", "formal"],
            ["--resume"],
        )
        for suffix in conflicts:
            conflict = build_parser().parse_args(
                [
                    "replay-recordings",
                    "--versions",
                    "v003",
                    "--trial-id",
                    "1",
                    "--diagnostic-role",
                    "U",
                    *suffix,
                ]
            )
            with self.assertRaises(RuntimeError, msg=str(suffix)):
                _normalize_recording_replay_role(conflict)

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
            [
                "replay-recordings",
                "--versions",
                "v012",
                "--trial-id",
                "1",
                "--resume",
            ]
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
        # A valid direct-buffer claim carries the complete viewport closure;
        # resume admission itself remains intentionally lightweight here.
        _atomic_write_json(
            run_dir / "viewport_buffer_video_manifest.json",
            {"schema_version": "fsm50.active_viewport_buffer_video.v1", "valid": True},
        )
        (run_dir / "viewport_frame_ledger.jsonl").write_text(
            '{"render_sequence":0}\n{"render_sequence":1}\n',
            encoding="utf-8",
        )
        (run_dir / "viewport_first_frame.png").write_bytes(b"synthetic-first-png")
        (run_dir / "viewport_last_frame.png").write_bytes(b"synthetic-last-png")
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
        (run_dir / "telemetry_samples.csv").write_text(
            "time_s\n0.0\n", encoding="utf-8"
        )
        (run_dir / "body_com_timeseries.csv").write_text(
            "time_s\n", encoding="utf-8"
        )
        (run_dir / "joint_timeseries.csv").write_text(
            "time_s\n", encoding="utf-8"
        )
        (run_dir / "contacts.csv").write_text("time_s\n", encoding="utf-8")
        (run_dir / "events.jsonl").write_text(
            '{"event_type":"finished"}\n', encoding="utf-8"
        )
        _atomic_write_json(
            run_dir / "stability_summary.json", {"sample_count": 1}
        )
        (run_dir / "wheel_filtered_contacts.jsonl").write_text(
            "", encoding="utf-8"
        )
        (run_dir / "nonwheel_obstacle_contacts.csv").write_text(
            "time_s\n", encoding="utf-8"
        )
        (run_dir / "nonwheel_obstacle_contacts.jsonl").write_text(
            "", encoding="utf-8"
        )
        canonical_counts = {
            "telemetry_samples.csv": 1,
            "body_com_timeseries.csv": 0,
            "joint_timeseries.csv": 0,
            "contacts.csv": 0,
            "events.jsonl": 1,
            "stability_summary.json": 1,
            "fsm50_telemetry.csv": 1,
            "fsm50_telemetry.jsonl": 1,
            "wheel_filtered_contacts.jsonl": 0,
            "nonwheel_obstacle_contacts.csv": 0,
            "nonwheel_obstacle_contacts.jsonl": 0,
            "state_timeline.csv": 1,
            "physical_evidence.json": 1,
        }
        canonical_files = {
            name: {
                "record_count": count,
                "size_bytes": (run_dir / name).stat().st_size,
                "sha256": hashlib.sha256(
                    (run_dir / name).read_bytes()
                ).hexdigest(),
            }
            for name, count in canonical_counts.items()
        }
        telemetry_marker = {
            "schema_version": "telemetry.canonical_finalization.v1",
            "created_wall_time_s": 1.0,
            "canonical_export_attempted": True,
            "canonical_complete": True,
            "stream_counts": {"fsm50_telemetry": 1},
            "canonical_files": canonical_files,
            "journal": {
                "committed_checkpoint_count": 1,
                "final_cursors": {"fsm50_telemetry": 1},
                "removed_after_success": True,
                "errors": [],
            },
            "errors": [],
        }
        telemetry_marker_path = run_dir / "telemetry_finalization.json"
        _atomic_write_json(telemetry_marker_path, telemetry_marker)
        result["telemetry_finalization"] = {
            **telemetry_marker,
            "marker_path": str(telemetry_marker_path.resolve()),
            "marker_sha256": hashlib.sha256(
                telemetry_marker_path.read_bytes()
            ).hexdigest(),
        }
        _atomic_write_json(run_dir / "result.json", result)
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

    def test_shutdown_failure_never_rewrites_worker_owned_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory).resolve()
            run_dir = batch_root / "v003" / "artifact" / "run"
            artifact_root = run_dir.parent
            run_dir.mkdir(parents=True)
            result = {
                "artifact_owner": "sim_worker_process",
                "artifact_root": str(artifact_root),
                "run_dir": str(run_dir),
                "classification": "PHYSICAL_FAILURE",
                "artifact_valid": True,
                "lifecycle": {"finalized": True, "failed": False},
            }
            _atomic_write_json(run_dir / "result.json", result)
            _atomic_write_json(run_dir / "failure_diagnostics.json", result)
            _atomic_write_json(batch_root / "batch_results.json", [result])
            _atomic_write_json(
                batch_root / "batch_finalization.json",
                {"finalized": True, "failed": False, "strict_success": False},
            )
            (artifact_root / ".finalized").write_text(
                "finalized\n", encoding="utf-8"
            )
            before_result = (run_dir / "result.json").read_bytes()
            before_batch_results = (batch_root / "batch_results.json").read_bytes()

            _record_shutdown_outcome(
                batch_root,
                {
                    "status": "SIMULATION_CLOSE_TIMEOUT",
                    "close_error": "worker close timed out",
                },
            )

            self.assertEqual(before_result, (run_dir / "result.json").read_bytes())
            self.assertEqual(
                before_batch_results,
                (batch_root / "batch_results.json").read_bytes(),
            )
            self.assertTrue((artifact_root / ".finalized").is_file())
            finalization = json.loads(
                (batch_root / "batch_finalization.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(finalization["failed"])
            self.assertFalse(finalization["strict_success"])

    def test_batch_source_drift_never_rewrites_worker_owned_result(self) -> None:
        result = {
            "artifact_owner": "sim_worker_process",
            "classification": "FULL_SUCCESS",
            "strict_full_success": True,
            "artifact_valid": True,
        }
        before = json.loads(json.dumps(result))

        _apply_batch_source_drift(
            [result],
            {"equal": False, "failures": ["source changed"]},
        )

        self.assertEqual(before, result)

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
                motion_start_readiness={
                    "ready": True,
                    "shared_production_worker_gate": {
                        "motion_start_ready": True
                    },
                },
                pre_first_dispatch_readiness={"ready": True},
                dispatch_ledger={"complete": True},
                wheel_integral_evidence={
                    "target_integral_verdict": "PASS",
                    "measured_tracking_verdict": "NOT_EVALUABLE",
                },
                trial_id=1,
            )
            self.assertAlmostEqual(0.1, result["final_joint_target_error_rad"])

    def test_wheel_target_integral_is_a_strict_replay_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = root / "accepted_steps.jsonl"
            metadata = root / "metadata.json"
            steps.write_text("{}\n", encoding="utf-8")
            metadata.write_text("{}\n", encoding="utf-8")
            item = VersionFiles("v003", root, steps, metadata)
            service = SimpleNamespace(
                stop_reason="complete",
                status_dict=lambda **_kwargs: {},
            )
            physical = {
                "physical_success": True,
                "traversal": {"legs": {}},
            }
            collector = SimpleNamespace(
                last_row={"time_s": 1.0},
                fsm50_rows=[],
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
                motion_start_readiness={
                    "ready": True,
                    "shared_production_worker_gate": {
                        "motion_start_ready": True
                    },
                },
                pre_first_dispatch_readiness={"ready": True},
                dispatch_ledger={"complete": True},
                wheel_integral_evidence={
                    "target_integral_verdict": "NOT_EVALUABLE",
                    "measured_tracking_verdict": "NOT_EVALUABLE",
                },
                trial_id=1,
            )

            self.assertFalse(result["strict_full_success"])
            self.assertFalse(result["wheel_target_integral_complete"])
            self.assertEqual("WHEEL_INTEGRAL_FAILURE", result["classification"])
            self.assertEqual("WHEEL_TARGET_INTEGRAL", result["first_failure_phase"])

    def test_u_result_readiness_does_not_require_physical_sensor_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = root / "accepted_steps.jsonl"
            metadata = root / "metadata.json"
            steps.write_text("{}\n", encoding="utf-8")
            metadata.write_text("{}\n", encoding="utf-8")
            item = VersionFiles("v003", root, steps, metadata)
            service = SimpleNamespace(
                stop_reason="complete",
                status_dict=lambda **_kwargs: {},
            )
            collector = SimpleNamespace(
                last_row={"time_s": 1.0},
                fsm50_rows=[],
                physical_evidence=lambda: {
                    "physical_success": False,
                    "full_physical_verdict": "NOT_EVALUABLE",
                    "traversal": {"legs": {}},
                },
            )
            plan = SimpleNamespace(
                profile="motion_only",
                plan_sha256="plan",
                events=[],
                segments=[],
                final_time_s=1.0,
            )
            common = dict(
                item=item,
                service=service,
                collector=collector,
                respawn={},
                plan=plan,
                run_dir=root,
                timed_out=False,
                motion_start_readiness={
                    "ready": True,
                    "shared_production_worker_gate": {
                        "motion_start_ready": False
                    },
                },
                pre_first_dispatch_readiness={"ready": True},
                dispatch_ledger={"complete": True},
                wheel_integral_evidence={
                    "target_integral_verdict": "PASS",
                    "measured_tracking_verdict": "NOT_EVALUABLE",
                },
                trial_id=1,
            )

            diagnostic = _result_payload(**common, diagnostic_role="U")
            physical = _result_payload(**common, diagnostic_role="")

            self.assertTrue(diagnostic["motion_start_ready"])
            self.assertEqual(
                diagnostic["motion_start_readiness_scope"],
                "SENSOR_INDEPENDENT_TRAJECTORY_ADMISSION",
            )
            self.assertEqual(
                diagnostic["physical_motion_start_verdict"],
                "NOT_EVALUABLE",
            )
            self.assertFalse(physical["motion_start_ready"])
            self.assertEqual(
                physical["motion_start_readiness_scope"],
                "PHYSICAL_MOTION_ADMISSION",
            )
            self.assertEqual(
                physical["physical_motion_start_verdict"],
                "FAIL",
            )

    def test_artifact_policy_rejects_unevaluable_wheel_target_integral(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            marker = run_dir / "telemetry_finalization.json"
            marker.write_text("{}\n", encoding="utf-8")
            result = {
                "run_dir": str(run_dir),
                "source_integrity": {"ok": True},
                "motion_start_ready": True,
                "dispatch_complete": True,
                "wheel_target_integral_verdict": "NOT_EVALUABLE",
                "strict_full_success": False,
                "telemetry_finalization": {},
            }

            valid = _apply_recording_artifact_policy(
                result,
                video={"actual_viewport_video": True},
                visualization={"ok": True},
            )

            self.assertFalse(valid)
            self.assertEqual("ARTIFACT_INVALID", result["classification"])
            self.assertEqual(
                "WHEEL_TARGET_INTEGRAL_NOT_EVALUABLE",
                result["first_failure_phase"],
            )

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
            lock["source_sha256"] = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in run_fsm50_module._source_files(robot)
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
        self.assertIn("client = SimProcessClient(worker_args)", source)
        self.assertIn("transport.start_playback_plan(", source)
        self.assertNotIn("ensure_simulation_app", source)
        self.assertNotIn("SimRobotAdapter(", source)
        self.assertNotIn("initialize_adapter_ground_reference", source)

    def test_replay_args_round_trip_for_supervised_child(self) -> None:
        original = build_parser().parse_args(
            [
                "replay-recordings",
                "--versions",
                "v012",
                "--trial-id",
                "1",
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
        worker_binding: dict | None = None,
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
        (batch_root / ".finalized").write_text(
            "finalized\n", encoding="utf-8"
        )
        handshake = root / "handshake.json"
        extras = {"intended_returncode": intended_returncode}
        if state in {
            "FAST_EXIT_REQUESTED",
            "FAST_CLOSE_RETURNED",
            "FAST_WORKER_PROCESS_RETURNED",
        }:
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
                **dict(worker_binding or {}),
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

    def test_worker_bound_fast_exit_requires_close_request_and_worker_rc_zero(self) -> None:
        class Child:
            pid = 43021

            @staticmethod
            def poll():
                return 0

        binding = {
            "formal_worker_pid": 53021,
            "formal_worker_session_id": "worker-session-21",
            "adapter_runtime_instance_id": "adapter-21",
            "artifact_request_id": "artifact-request-21",
            "artifact_request_sha256": "a" * 64,
        }
        shutdown_request_id = "shutdown-21"
        fast_kwargs = {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        }
        shutdown_ack = {
            "type": "operation_ack",
            "operation": "shutdown",
            "accepted": True,
            "error": "",
            "mode": "fast",
            "request_id": shutdown_request_id,
            "worker_pid": binding["formal_worker_pid"],
            "worker_session_id": binding["formal_worker_session_id"],
            "adapter_runtime_instance_id": binding[
                "adapter_runtime_instance_id"
            ],
            "artifact_request_id": binding["artifact_request_id"],
            "root_state_write_count": 0,
            "close_kwargs": fast_kwargs,
            "runtime_version": "5.1.0.0",
        }
        close_requested_ack = {
            "type": "close_requested",
            "mode": "fast",
            "accepted": True,
            "error": "",
            **{key: shutdown_ack[key] for key in (
                "request_id",
                "worker_pid",
                "worker_session_id",
                "adapter_runtime_instance_id",
                "artifact_request_id",
                "root_state_write_count",
                "close_kwargs",
                "runtime_version",
            )},
        }
        cases = (
            (
                "FAST_EXIT_REQUESTED",
                {},
                "FAST_EXIT_FAILED",
            ),
            (
                "FAST_CLOSE_RETURNED",
                {
                    "worker_returncode": 0,
                    "worker_process_returned_normally": True,
                    "worker_shutdown_accepted": True,
                    "worker_close_requested": True,
                    "worker_close_returned": False,
                    "worker_forced_termination": False,
                    "worker_shutdown_request_id": shutdown_request_id,
                    "worker_shutdown_ack": shutdown_ack,
                    "worker_close_requested_ack": close_requested_ack,
                    "worker_close_returned_ack": {},
                },
                "FAST_EXIT_FAILED",
            ),
            (
                "FAST_WORKER_PROCESS_RETURNED",
                {
                    "worker_returncode": 0,
                    "worker_process_returned_normally": True,
                    "worker_shutdown_accepted": True,
                    "worker_close_requested": True,
                    "worker_close_returned": False,
                    "worker_forced_termination": False,
                    "worker_shutdown_request_id": shutdown_request_id,
                    "worker_shutdown_ack": shutdown_ack,
                    "worker_close_requested_ack": close_requested_ack,
                    "worker_close_returned_ack": {},
                },
                "FAST_EXIT_VERIFIED",
            ),
            (
                "FAST_WORKER_PROCESS_RETURNED",
                {
                    "worker_returncode": 7,
                    "worker_process_returned_normally": True,
                    "worker_shutdown_accepted": True,
                    "worker_close_requested": True,
                    "worker_close_returned": False,
                    "worker_forced_termination": False,
                    "worker_shutdown_request_id": shutdown_request_id,
                    "worker_shutdown_ack": shutdown_ack,
                    "worker_close_requested_ack": close_requested_ack,
                    "worker_close_returned_ack": {},
                },
                "FAST_EXIT_FAILED",
            ),
        )
        for index, (state, close_evidence, expected) in enumerate(cases):
            with self.subTest(state=state, expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    handshake, _batch_root = self._write_supervisor_files(
                        root,
                        token=f"token-{index}",
                        parent_pid=123,
                        child_pid=Child.pid,
                        preclose=True,
                        state=state,
                        worker_binding={**binding, **close_evidence},
                    )
                    with mock.patch.object(
                        run_fsm50_module,
                        "_validate_preclose_closure",
                        return_value={
                            "ok": True,
                            "formal_worker_identity": {
                                "worker_pid": binding["formal_worker_pid"],
                                "worker_session_id": binding[
                                    "formal_worker_session_id"
                                ],
                                "adapter_runtime_instance_id": binding[
                                    "adapter_runtime_instance_id"
                                ],
                                "artifact_request_id": binding[
                                    "artifact_request_id"
                                ],
                                "artifact_request_sha256": binding[
                                    "artifact_request_sha256"
                                ],
                            },
                        },
                    ):
                        outcome, _observed = _monitor_supervised_child(
                            Child(),
                            handshake_path=handshake,
                            token=f"token-{index}",
                            parent_pid=123,
                            output_root=root,
                            clock=lambda: 1.0,
                            sleep=lambda _seconds: None,
                        )
                    self.assertEqual(expected, outcome["status"])
                    self.assertEqual(
                        binding["formal_worker_pid"],
                        outcome["formal_worker_pid"],
                    )
                    self.assertEqual(
                        binding["formal_worker_session_id"],
                        outcome["formal_worker_session_id"],
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_close = dict(cases[1][1])
            handshake, _batch_root = self._write_supervisor_files(
                root,
                token="missing-worker-pid",
                parent_pid=123,
                child_pid=Child.pid,
                preclose=True,
                state="FAST_WORKER_PROCESS_RETURNED",
                worker_binding={**binding, **valid_close},
            )
            payload = json.loads(handshake.read_text(encoding="utf-8"))
            payload.pop("formal_worker_pid")
            _atomic_write_json(handshake, payload)
            with mock.patch.object(
                run_fsm50_module,
                "_validate_preclose_closure",
                return_value={
                    "ok": True,
                    "formal_worker_identity": {
                        "worker_pid": binding["formal_worker_pid"],
                        "worker_session_id": binding[
                            "formal_worker_session_id"
                        ],
                        "adapter_runtime_instance_id": binding[
                            "adapter_runtime_instance_id"
                        ],
                        "artifact_request_id": binding[
                            "artifact_request_id"
                        ],
                        "artifact_request_sha256": binding[
                            "artifact_request_sha256"
                        ],
                    },
                },
            ):
                outcome, _observed = _monitor_supervised_child(
                    Child(),
                    handshake_path=handshake,
                    token="missing-worker-pid",
                    parent_pid=123,
                    output_root=root,
                    clock=lambda: 1.0,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual("FAST_EXIT_FAILED", outcome["status"])

    def test_preclose_closure_rejects_missing_or_conflicting_lifecycle_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _handshake, batch_root = self._write_supervisor_files(
                root,
                token="token",
                parent_pid=123,
                child_pid=43020,
                preclose=True,
                state="FAST_EXIT_REQUESTED",
            )
            (batch_root / ".finalized").unlink()
            with self.assertRaisesRegex(RuntimeError, "lifecycle marker"):
                _validate_preclose_closure(batch_root)

            (batch_root / ".finalized").write_text(
                "finalized\n", encoding="utf-8"
            )
            (batch_root / ".failed").write_text("failed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lifecycle marker"):
                _validate_preclose_closure(batch_root)

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

        with mock.patch.object(
            run_fsm50_module,
            "_runtime_versions",
            return_value={"packages": {"isaacsim": "5.1.0.0"}},
        ):
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
        source = inspect.getsource(_finalize_worker_recording_batch_preclose)
        snapshot_index = source.index(
            "immutable_errors = _snapshot_preclose_files(root)"
        )
        failure_index = source.index(
            "batch_valid = False",
            snapshot_index,
        )
        marker_index = source.index(
            "_mark_artifact_root(root, valid=batch_valid)",
            snapshot_index,
        )
        self.assertLess(snapshot_index, failure_index)
        self.assertLess(failure_index, marker_index)

    def test_formal_worker_postprocessing_always_closes_owned_worker(self) -> None:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(_run_recording_replays_locked))
        )

        def calls(statements, function_name: str) -> bool:
            return any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == function_name
                for statement in statements
                for node in ast.walk(statement)
            )

        postprocessing = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and calls(node.body, "_source_freeze")
            and calls(node.body, "_finalize_worker_recording_batch_preclose")
        ]
        self.assertEqual(1, len(postprocessing))
        self.assertTrue(
            calls(
                postprocessing[0].finalbody,
                "_close_formal_recording_worker",
            )
        )

    def test_empty_failed_batch_keeps_immutable_diagnostic_role_identity(
        self,
    ) -> None:
        cases = (
            (
                "U",
                "",
                "U",
                "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC",
                False,
                False,
            ),
            ("A1", "A1", "", "TRAJECTORY_COMPARISON", False, True),
            ("A2", "A2", "", "TRAJECTORY_COMPARISON", False, True),
            ("B", "B", "", "TRAJECTORY_COMPARISON", False, True),
        )

        class Supervisor:
            evidence: dict = {}

            def mark_preclose(self, evidence: dict) -> None:
                self.evidence = evidence

        for (
            label,
            equivalence_role,
            diagnostic_role,
            scope,
            gate1_eligible,
            equivalence_eligible,
        ) in cases:
            with self.subTest(role=label), tempfile.TemporaryDirectory() as directory:
                batch_root = Path(directory)
                request = {
                    "environment_equivalence_role": equivalence_role,
                    "diagnostic_role": diagnostic_role,
                    "qualification_scope": scope,
                    "gate1_physical_qualification_eligible": gate1_eligible,
                    "gate1_eligible": gate1_eligible,
                    "environment_equivalence_eligible": equivalence_eligible,
                }
                _atomic_write_json(batch_root / "batch_request.json", request)
                _atomic_write_json(
                    batch_root / "worker_artifact_request.json", request
                )
                _atomic_write_json(
                    batch_root / "source_integrity.json", {"equal": True}
                )
                (batch_root / ".partial").write_text(
                    "running\n", encoding="utf-8"
                )
                supervisor = Supervisor()

                finalization, valid = _finalize_worker_recording_batch_preclose(
                    batch_root,
                    artifact_request=request,
                    results=[],
                    batch_error="RuntimeError: worker pre-start failure",
                    source_integrity={"equal": True},
                    supervisor=supervisor,
                )

                self.assertFalse(valid)
                self.assertTrue(finalization["failed"])
                self.assertEqual(
                    "RuntimeError: worker pre-start failure",
                    finalization["batch_error"],
                )
                self.assertEqual(
                    equivalence_role,
                    finalization["environment_equivalence_role"],
                )
                self.assertEqual(
                    diagnostic_role, finalization["diagnostic_role"]
                )
                self.assertEqual(scope, finalization["qualification_scope"])
                self.assertIs(
                    finalization["gate1_eligible"], gate1_eligible
                )
                self.assertIs(
                    finalization["gate1_physical_qualification_eligible"],
                    gate1_eligible,
                )
                self.assertIs(
                    finalization["environment_equivalence_eligible"],
                    equivalence_eligible,
                )
                self.assertEqual(0, finalization["worker_artifact_result_count"])
                self.assertEqual(
                    [],
                    json.loads(
                        (batch_root / "batch_results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )
                self.assertEqual(
                    0, supervisor.evidence["physics_result_count"]
                )

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
