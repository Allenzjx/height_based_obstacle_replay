from __future__ import annotations

import ast
import hashlib
import json
import math
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from fsm_50mm_recording_derived_v3.worker_recording_session import (  # noqa: E402
    REQUEST_SCHEMA,
    REQUEST_REQUIRED_KEYS,
    WorkerRecordingSession,
    _write_json,
    apply_ordinary_ui_diagnostic_classification,
    configure_scene_for_worker_recording,
    environment_equivalence_diagnostic_status,
    load_worker_recording_gate_request,
    ordinary_ui_diagnostic_status,
    validate_worker_plan_binding,
)
from sim_ipc_protocol import make_message  # noqa: E402
from sim_process_client import SimProcessClient, build_worker_config  # noqa: E402
from sim_transport import SimTransport  # noqa: E402
from sim_worker_process import (  # noqa: E402
    WorkerIpc,
    _wait_for_close_receipt,
    _worker_ack_identity,
    main as worker_main,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorkerRecordingIpcContractTest(unittest.TestCase):
    def _request(self, root: Path) -> tuple[Path, Path]:
        steps = root / "accepted_steps.jsonl"
        metadata = root / "metadata.json"
        robot = root / "robot.usd"
        lock = root / "environment_lock.json"
        steps.write_text('{"events": []}\n', encoding="utf-8")
        metadata.write_text("{}\n", encoding="utf-8")
        robot.write_text("#usda 1.0\n", encoding="utf-8")
        lock.write_text("{}\n", encoding="utf-8")
        artifact = root / "worker_owned_artifact"
        payload = {
            "schema_version": REQUEST_SCHEMA,
            "enabled": True,
            "artifact_owner": "sim_worker_process",
            "request_id": "request-1",
            "plan_id": "plan-1",
            "plan_sha256": "a" * 64,
            "expected_plan_sha256": "a" * 64,
            "plan_event_count": 1,
            "plan_segment_count": 1,
            "source_version": "v003",
            "trial_id": 1,
            "contact_mode": "instrumented",
            "environment_equivalence_role": "",
            "diagnostic_role": "",
            "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
            "gate1_eligible": True,
            "gate1_physical_qualification_eligible": True,
            "environment_equivalence_eligible": False,
            "height_mm": 50,
            "artifact_root": str(artifact),
            "accepted_steps_path": str(steps),
            "metadata_path": str(metadata),
            "accepted_steps_sha256": _sha(steps),
            "expected_accepted_steps_sha256": _sha(steps),
            "robot_usd_path": str(robot),
            "expected_robot_usd_sha256": _sha(robot),
            "environment_lock_path": str(lock),
            "environment_lock_sha256": _sha(lock),
            "post_run_settle_s": 0.5,
            "timeout_s": 600.0,
            "timeout_scale": 1.0,
            "telemetry_rate_hz": 120.0,
            "video_fps": 15.0,
            "capture_video": True,
            "headless": False,
            "expected_root_state_write_count": 0,
        }
        request = root / "request.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        return request, artifact

    def test_request_is_opt_in_and_binds_nonexistent_worker_owned_root(self) -> None:
        self.assertIsNone(load_worker_recording_gate_request(""))
        with tempfile.TemporaryDirectory() as tmp:
            request_path, artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            self.assertIsNotNone(request)
            assert request is not None
            self.assertEqual(request.artifact_root, artifact.resolve())
            self.assertFalse(artifact.exists())
            self.assertEqual(request.artifact_request_sha256, _sha(request_path))

    def test_request_v1_requires_every_exact_field_and_rejects_extras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            base = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(set(base), set(REQUEST_REQUIRED_KEYS))

            for key in sorted(REQUEST_REQUIRED_KEYS):
                payload = dict(base)
                del payload[key]
                request_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "missing required keys"):
                    load_worker_recording_gate_request(request_path)

            request_path.write_text(
                json.dumps({**base, "unexpected_contract_field": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected v1 keys"):
                load_worker_recording_gate_request(request_path)

    def test_request_v1_uses_strict_json_types_and_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            base = json.loads(request_path.read_text(encoding="utf-8"))
            for key, invalid in (
                ("enabled", 1),
                ("height_mm", 50.0),
                ("trial_id", True),
                ("plan_event_count", 1.0),
                ("plan_segment_count", False),
                ("expected_root_state_write_count", 0.0),
                ("capture_video", 1),
                ("headless", 0),
                ("telemetry_rate_hz", 120),
                ("video_fps", 15),
                ("post_run_settle_s", 0),
                ("timeout_s", 600),
                ("timeout_scale", 1),
                ("contact_mode", True),
                ("environment_equivalence_role", None),
                ("diagnostic_role", 0),
            ):
                request_path.write_text(
                    json.dumps({**base, key: invalid}), encoding="utf-8"
                )
                with self.assertRaises(ValueError, msg=key):
                    load_worker_recording_gate_request(request_path)

            request_path.write_text(
                json.dumps(base)[:-1] + ',"request_id":"duplicate"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_worker_recording_gate_request(request_path)

            request_path.write_text(
                json.dumps(base).replace('"timeout_s": 600.0', '"timeout_s": NaN'),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                load_worker_recording_gate_request(request_path)

    def test_request_rejects_precreated_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, artifact = self._request(Path(tmp))
            artifact.mkdir()
            with self.assertRaises(FileExistsError):
                load_worker_recording_gate_request(request_path)

    def test_request_role_contact_matrix_is_explicit_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request_path, _artifact = self._request(root)
            base = json.loads(request_path.read_text(encoding="utf-8"))

            for role, contact_mode in (
                ("A1", "formal"),
                ("A2", "formal"),
                ("B", "instrumented"),
            ):
                payload = {
                    **base,
                    "environment_equivalence_role": role,
                    "contact_mode": contact_mode,
                    "qualification_scope": "TRAJECTORY_COMPARISON",
                    "gate1_eligible": False,
                    "gate1_physical_qualification_eligible": False,
                    "environment_equivalence_eligible": True,
                }
                request_path.write_text(json.dumps(payload), encoding="utf-8")
                request = load_worker_recording_gate_request(request_path)
                assert request is not None
                self.assertEqual(request.environment_equivalence_role, role)
                self.assertEqual(request.contact_mode, contact_mode)
                self.assertEqual(
                    request.preflight_payload()["environment_equivalence_role"],
                    role,
                )

            for role, contact_mode in (
                ("", "formal"),
                ("A1", "instrumented"),
                ("A2", "instrumented"),
                ("B", "formal"),
                ("C", "instrumented"),
            ):
                payload = {
                    **base,
                    "environment_equivalence_role": role,
                    "contact_mode": contact_mode,
                }
                request_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "environment-equivalence|contact_mode"):
                    load_worker_recording_gate_request(request_path)

    def test_scene_contact_bank_is_selected_by_bound_evidence_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request_path, _artifact = self._request(root)
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            payload.update(
                environment_equivalence_role="A1",
                contact_mode="formal",
                qualification_scope="TRAJECTORY_COMPARISON",
                gate1_eligible=False,
                gate1_physical_qualification_eligible=False,
                environment_equivalence_eligible=True,
            )
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            formal_request = load_worker_recording_gate_request(request_path)
            assert formal_request is not None
            aggregate_factory_sentinel = object()
            formal_config = SimpleNamespace(
                telemetry_contact_sensors_enabled=False,
                contact_sensor_factory=aggregate_factory_sentinel,
            )

            configured = configure_scene_for_worker_recording(
                formal_config, formal_request
            )

            self.assertIs(configured, formal_config)
            self.assertTrue(formal_config.telemetry_contact_sensors_enabled)
            self.assertIsNone(formal_config.contact_sensor_factory)

            payload.update(
                environment_equivalence_role="B",
                contact_mode="instrumented",
            )
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            instrumented_request = load_worker_recording_gate_request(request_path)
            assert instrumented_request is not None
            instrumented_config = SimpleNamespace(
                telemetry_contact_sensors_enabled=False,
                contact_sensor_factory=None,
            )
            with mock.patch(
                "fsm_50mm_recording_derived_v3.nonwheel_obstacle_contact."
                "configure_scene_for_wheel_and_nonwheel_contacts"
            ) as configure_combined:
                configure_scene_for_worker_recording(
                    instrumented_config, instrumented_request
                )
            configure_combined.assert_called_once()

    def test_u_request_scene_and_sensor_independent_role_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            payload.update(
                diagnostic_role="U",
                environment_equivalence_role="",
                contact_mode="disabled",
                qualification_scope="PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC",
                gate1_eligible=False,
                gate1_physical_qualification_eligible=False,
                environment_equivalence_eligible=False,
            )
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            self.assertEqual(request.diagnostic_role, "U")
            self.assertEqual(request.contact_mode, "disabled")
            self.assertEqual(request.environment_equivalence_role, "")

            config = SimpleNamespace(
                telemetry_contact_sensors_enabled=False,
                contact_sensor_factory=None,
            )
            self.assertIs(
                configure_scene_for_worker_recording(config, request),
                config,
            )
            self.assertIs(config.telemetry_contact_sensors_enabled, False)
            self.assertIsNone(config.contact_sensor_factory)

            bad_enabled = SimpleNamespace(
                telemetry_contact_sensors_enabled=True,
                contact_sensor_factory=None,
            )
            with self.assertRaisesRegex(ValueError, "incoming production default"):
                configure_scene_for_worker_recording(bad_enabled, request)
            bad_factory = SimpleNamespace(
                telemetry_contact_sensors_enabled=False,
                contact_sensor_factory=object(),
            )
            with self.assertRaisesRegex(ValueError, "incoming production default"):
                configure_scene_for_worker_recording(bad_factory, request)

            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-u",
            )
            self.assertTrue(
                session.motion_start_ready_for_role(
                    {"ready": True},
                    {"motion_start_ready": False},
                )
            )
            self.assertFalse(
                session.motion_start_ready_for_role(
                    {"ready": False},
                    {"motion_start_ready": True},
                )
            )

            for field, value in (
                ("environment_equivalence_role", "A1"),
                ("contact_mode", "instrumented"),
                ("gate1_eligible", True),
            ):
                invalid = {**payload, field: value}
                request_path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError, msg=field):
                    load_worker_recording_gate_request(request_path)

    def test_u_completion_and_classification_are_nonphysical(self) -> None:
        base = {
            "role": "U",
            "contact_mode": "disabled",
            "artifact_valid": True,
            "scheduler_complete": True,
            "dispatch_complete": True,
            "wheel_target_integral_verdict": "PASS",
            "trajectory_valid": True,
            "contact_sensor_disabled": True,
            "source_integrity_ok": True,
            "root_state_write_count": 0,
        }
        complete = ordinary_ui_diagnostic_status(**base)
        self.assertTrue(complete["ordinary_ui_diagnostic_complete"])
        self.assertFalse(complete["gate1_eligible"])
        self.assertFalse(complete["environment_equivalence_eligible"])
        for key, failing in (
            ("artifact_valid", False),
            ("scheduler_complete", False),
            ("dispatch_complete", False),
            ("wheel_target_integral_verdict", "FAIL"),
            ("trajectory_valid", False),
            ("contact_sensor_disabled", False),
            ("source_integrity_ok", False),
            ("root_state_write_count", 1),
        ):
            self.assertFalse(
                ordinary_ui_diagnostic_status(**{**base, key: failing})[
                    "ordinary_ui_diagnostic_complete"
                ],
                key,
            )

        result = {
            "classification": "PHYSICAL_FAILURE",
            "first_failure_phase": "physical_evidence",
            "physical_success": False,
            "full_physical_verdict": "NOT_EVALUABLE",
        }
        apply_ordinary_ui_diagnostic_classification(result, role="U")
        self.assertEqual(
            result["classification_before_diagnostic_scope"],
            "PHYSICAL_FAILURE",
        )
        self.assertEqual(result["classification"], "TRAJECTORY_DIAGNOSTIC_ONLY")
        self.assertEqual(result["first_failure_phase"], "")

    def test_diagnostic_completion_is_capture_closure_not_physical_qualification(self) -> None:
        status = environment_equivalence_diagnostic_status(
            role="A1",
            contact_mode="formal",
            artifact_valid=True,
            scheduler_complete=True,
            dispatch_complete=True,
            source_integrity_ok=True,
        )
        self.assertEqual(status["qualification_scope"], "TRAJECTORY_COMPARISON")
        self.assertTrue(status["environment_equivalence_diagnostic_complete"])

        for failed_field in (
            "artifact_valid",
            "scheduler_complete",
            "dispatch_complete",
            "source_integrity_ok",
        ):
            values = {
                "artifact_valid": True,
                "scheduler_complete": True,
                "dispatch_complete": True,
                "source_integrity_ok": True,
            }
            values[failed_field] = False
            failed = environment_equivalence_diagnostic_status(
                role="B",
                contact_mode="instrumented",
                **values,
            )
            self.assertFalse(
                failed["environment_equivalence_diagnostic_complete"],
                failed_field,
            )

        gate1 = environment_equivalence_diagnostic_status(
            role="",
            contact_mode="instrumented",
            artifact_valid=True,
            scheduler_complete=True,
            dispatch_complete=True,
            source_integrity_ok=True,
        )
        self.assertFalse(gate1["environment_equivalence_diagnostic"])
        self.assertFalse(gate1["environment_equivalence_diagnostic_complete"])
        self.assertEqual(gate1["qualification_scope"], "GATE1_PHYSICAL_QUALIFICATION")

    def test_session_build_plan_identity_binds_verified_request_source_and_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            initial_state = {
                "servos": {"front_left_hip": 0.0},
                "wheels": {"front_left_ankle": 0.0},
            }
            initial_state_sha = hashlib.sha256(
                json.dumps(
                    initial_state,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            plan = SimpleNamespace(
                plan_sha256=request.expected_plan_sha256,
                events=[SimpleNamespace()],
                segments=[SimpleNamespace()],
                timing={
                    "source_initial_command_state": initial_state,
                    "source_initial_command_state_sha256": initial_state_sha,
                },
            )
            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-1",
            )
            session.item = SimpleNamespace(
                version_id=request.source_version,
                steps_path=request.accepted_steps_path,
            )

            identity = session._build_plan_identity(plan)

            self.assertEqual(identity["source_version"], request.source_version)
            self.assertEqual(identity["source_sha256"], request.accepted_steps_sha256)
            self.assertEqual(identity["source_sha256"], _sha(request.accepted_steps_path))
            self.assertEqual(identity["plan_sha256"], request.expected_plan_sha256)
            self.assertEqual(
                identity["validated_plan_sha256"], request.expected_plan_sha256
            )
            self.assertEqual(identity["plan_id"], request.plan_id)
            self.assertEqual(identity["request_id"], request.request_id)
            self.assertEqual(identity["worker_session_id"], "worker-session-1")
            self.assertEqual(
                identity["requested_worker_session_id"], "worker-session-1"
            )
            self.assertEqual(identity["event_count"], 1)
            self.assertEqual(identity["validated_event_count"], 1)
            self.assertEqual(identity["segment_count"], 1)
            self.assertEqual(identity["validated_segment_count"], 1)
            self.assertIs(identity["integrity_ok"], True)
            self.assertEqual(identity["source_initial_command_state"], initial_state)
            self.assertEqual(
                identity["source_initial_command_state_sha256"], initial_state_sha
            )
            request.accepted_steps_path.write_text(
                '{"events": [{"tampered": true}]}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "source hash differs from the immutable request"
            ):
                session._build_plan_identity(plan)

    def test_gate_rejects_direct_dispatch_and_environment_mutation_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-1",
            )
            blocked = [
                {"type": "command", "command": "servo front_left_hip 1"},
                {"type": "apply_motion_batch", "batch_id": "direct-batch-1"},
                {"type": "set_height", "height_mm": 50},
                {"type": "set_height_respawn", "height_mm": 50},
                {"type": "respawn", "request_id": "respawn-1"},
                {
                    "type": "restore_sim_state",
                    "request_id": "restore-1",
                    "sim_state": {},
                },
                {"type": "recalibrate_ground_reference"},
            ]

            rejected = session.reject_direct_dispatch_messages(blocked)

            self.assertEqual(rejected, blocked)
            self.assertEqual(session.state, "preflight")
            self.assertFalse(session.terminal)
            status = session.status_dict()
            attempts = list(status.get("rejected_direct_dispatch_attempts", []) or [])
            self.assertEqual(
                [str(row.get("type", "")) for row in attempts],
                [str(row["type"]) for row in blocked],
            )
            self.assertTrue(
                all(row.get("rejected") is True for row in attempts),
                attempts,
            )

    def test_gate_dispatch_guard_allows_only_formal_control_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-1",
            )
            allowed = [
                {"type": "stop_wheels", "command_id": "stop-1"},
                {
                    "type": "start_playback_plan",
                    "request_id": request.request_id,
                    "plan_id": request.plan_id,
                    "worker_session_id": "worker-session-1",
                },
                {"type": "request_state", "request_id": "state-1"},
                {"type": "shutdown", "request_id": "shutdown-1"},
            ]

            rejected = session.reject_direct_dispatch_messages(allowed)

            self.assertEqual(rejected, [])
            self.assertEqual(session.state, "preflight")
            self.assertFalse(session.terminal)
            self.assertEqual(
                list(
                    session.status_dict().get(
                        "rejected_direct_dispatch_attempts", []
                    )
                    or []
                ),
                [],
            )

    def test_artifact_pre_first_token_binds_complete_rich_evidence_once(self) -> None:
        """The artifact token must cover the rich gate before source dispatch."""

        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            order: list[str] = []
            bound_tokens: list[tuple[str, int]] = []

            class FakeService:
                timing_trace = {
                    "motion_batches": [
                        {
                            "dispatch_kind": "playback_start_boundary",
                            "batch_id": "boundary-1",
                        }
                    ]
                }
                last_motion_batch_ack = {
                    "batch_id": "boundary-1",
                    "dispatch_kind": "playback_start_boundary",
                    "applied_sim_step": 87,
                    "first_physics_step": 88,
                }
                last_error = ""

                @staticmethod
                def bind_motion_start_readiness(
                    token: str, *, current_sim_step: int
                ) -> bool:
                    order.append("bind")
                    bound_tokens.append((token, current_sim_step))
                    return True

            adapter = SimpleNamespace(
                runtime_instance_id="adapter-runtime-1234",
                root_state_write_count=0,
                sim_steps=88,
            )
            identity = {
                "source_version": request.source_version,
                "source_sha256": request.accepted_steps_sha256,
                "plan_sha256": request.expected_plan_sha256,
                "validated_plan_sha256": request.expected_plan_sha256,
                "plan_id": request.plan_id,
                "request_id": request.request_id,
                "worker_session_id": "worker-session-1",
                "requested_worker_session_id": "worker-session-1",
                "event_count": 1,
                "validated_event_count": 1,
                "segment_count": 1,
                "validated_segment_count": 1,
                "integrity_ok": True,
            }
            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-1",
            )
            session.state = "attached"
            session.run_dir = Path(tmp) / "run"
            session.run_dir.mkdir()
            session.adapter = adapter
            session.scene_handle = SimpleNamespace()
            session.live_obstacle = {}
            session.service = FakeService()
            session.admitted_plan_identity = identity
            session.readiness_frames = [
                {"frame": index} for index in range(session.required_readiness_frames)
            ]
            session.boundary_frame = {"frame": "boundary"}
            shared = {
                "motion_start_ready": True,
                "adapter_runtime_instance_id": adapter.runtime_instance_id,
                "root_state_write_count": 0,
                "current_sim_step": adapter.sim_steps,
                "identity": identity,
            }
            rich = {
                "schema_version": "fsm50.motion_start_readiness_window.v1",
                "gate": "MOTION_START_READY_PRE_FIRST_DISPATCH",
                "ready": True,
                "status": "PASS",
                "writes_robot_state": False,
                "state_writes_performed": False,
                "root_state_write_count": 0,
                "root_state_write_events": [],
                "adapter_runtime_instance_id": adapter.runtime_instance_id,
                "plan_identity": identity,
            }

            from fsm_50mm_recording_derived_v3 import run_fsm50

            real_builder = run_fsm50._build_motion_start_readiness_evidence

            def evaluate(*_args, **_kwargs):
                order.append("rich")
                return dict(rich)

            def build_token(**kwargs):
                order.append("token")
                return real_builder(**kwargs)

            with (
                mock.patch(
                    "fsm_50mm_recording_derived_v3.motion_start_readiness."
                    "capture_live_motion_start_snapshot",
                    return_value={"frame": "post-boundary"},
                ),
                mock.patch.object(
                    run_fsm50,
                    "_evaluate_motion_start_window",
                    side_effect=evaluate,
                ),
                mock.patch.object(
                    run_fsm50,
                    "_build_motion_start_readiness_evidence",
                    side_effect=build_token,
                ),
            ):
                ready = session.record_pre_first_dispatch(shared)

            self.assertIs(ready, True)
            self.assertEqual(order, ["rich", "token", "bind"])
            self.assertEqual(len(bound_tokens), 1)
            evidence = session.pre_first_dispatch_readiness
            payload = evidence["token_payload"]
            self.assertEqual(
                payload["schema_version"],
                "fsm50.motion_start_readiness_token.v1",
            )
            self.assertEqual(payload["source_version"], request.source_version)
            self.assertEqual(payload["trial_id"], request.trial_id)
            self.assertEqual(payload["plan_identity"], identity)
            self.assertEqual(
                payload["adapter_runtime_instance_id"],
                adapter.runtime_instance_id,
            )
            self.assertEqual(payload["root_state_write_count"], 0)
            self.assertEqual(
                payload["pre_first_dispatch_sim_step"], adapter.sim_steps
            )
            self.assertTrue(
                payload["pre_first_dispatch_readiness"]["ready"]
            )
            token = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(evidence["readiness_token_sha256"], token)
            self.assertEqual(bound_tokens, [(token, adapter.sim_steps)])
            self.assertEqual(session.readiness_token, token)
            self.assertIs(evidence["readiness_token_bound"], True)

    def test_closed_physical_or_artifact_fail_is_still_artifact_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-1",
                finalize_callback=lambda _session: {
                    "finalization_complete": True,
                    "artifact_valid": False,
                    "classification": "PHYSICAL_FAILURE",
                },
            )
            terminal = session.finalize()
            self.assertEqual(terminal["type"], "artifact_complete")
            self.assertEqual(terminal["state"], "complete")
            self.assertFalse(terminal["artifact_valid"])

    def test_artifact_preflight_ready_is_sticky_through_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            artifact.mkdir()
            session = WorkerRecordingSession(
                request,
                worker_session_id="worker-session-1",
                finalize_callback=lambda _session: {
                    "finalization_complete": True,
                    "artifact_valid": False,
                },
            )
            session.run_dir = artifact
            session.state = "readiness_window"
            session.readiness_frames = [
                {"frame": index} for index in range(session.required_readiness_frames)
            ]
            session.expected_plan_identity = {"plan_id": request.plan_id}
            session.adapter = SimpleNamespace(
                sim_steps=10,
                root_state_write_count=0,
                runtime_instance_id="adapter-1",
            )
            session.scene_handle = SimpleNamespace(
                config=SimpleNamespace(render_interval=8)
            )
            with (
                mock.patch(
                    "fsm_50mm_recording_derived_v3.run_fsm50."
                    "_evaluate_motion_start_window",
                    return_value={"ready": True},
                ),
                mock.patch(
                    "sim_worker_runtime.capture_worker_motion_start_readiness",
                    return_value={"motion_start_ready": True},
                ),
            ):
                self.assertIsNone(session.after_adapter_step())

            self.assertEqual(session.state, "ready_for_plan")
            self.assertIs(session.status_dict()["artifact_preflight_ready"], True)
            for state in ("attached", "running", "postsettle"):
                session.state = state
                self.assertIs(
                    session.status_dict()["artifact_preflight_ready"], True
                )
            terminal = session.finalize()
            self.assertEqual(terminal["state"], "complete")
            self.assertIs(terminal["artifact_preflight_ready"], True)

    def test_worker_result_declares_app_running_at_preclose(self) -> None:
        source = (
            MODULE_ROOT
            / "fsm_50mm_recording_derived_v3"
            / "worker_recording_session.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "update":
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "result":
                continue
            values.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "simulation_app_stopped"
            )
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], ast.Constant)
        self.assertIs(values[0].value, False)

    def test_worker_runtime_environment_binds_viewport_manifest_at_top_level(self) -> None:
        source = (
            MODULE_ROOT
            / "fsm_50mm_recording_derived_v3"
            / "worker_recording_session.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        payloads = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "write_json":
                continue
            path = node.args[0]
            if not (
                isinstance(path, ast.BinOp)
                and isinstance(path.op, ast.Div)
                and isinstance(path.right, ast.Constant)
                and path.right.value == "runtime_environment.json"
                and isinstance(node.args[1], ast.Dict)
            ):
                continue
            payloads.append(
                {
                    key.value: value
                    for key, value in zip(node.args[1].keys, node.args[1].values)
                    if isinstance(key, ast.Constant)
                }
            )

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(
            ast.unparse(payload["viewport_video_manifest_path"]),
            "str(self.video.get('manifest_path', '') or '')",
        )
        self.assertEqual(
            ast.unparse(payload["viewport_video_manifest_sha256"]),
            "str(self.video.get('manifest_sha256', '') or '')",
        )

    def test_worker_result_uses_recording_version_source_integrity_scope(self) -> None:
        source = (
            MODULE_ROOT
            / "fsm_50mm_recording_derived_v3"
            / "worker_recording_session.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        scopes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "result"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "source_integrity"
                and isinstance(node.value, ast.Dict)
            ):
                continue
            payload = {
                key.value: value
                for key, value in zip(node.value.keys, node.value.values)
                if isinstance(key, ast.Constant)
            }
            scopes.append(payload.get("scope"))
        self.assertEqual(len(scopes), 1)
        self.assertIsInstance(scopes[0], ast.Constant)
        self.assertEqual(scopes[0].value, "recording_version")

    def test_session_atomic_json_is_strict_and_normalizes_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strict.json"
            _write_json(path, {"nan": math.nan, "inf": math.inf})
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            self.assertEqual(json.loads(text), {"inf": None, "nan": None})

    def test_default_worker_config_does_not_enable_gate(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        config = build_worker_config(args, host="127.0.0.1", port=1234)
        self.assertNotIn("fsm50_gate_request_path", config)

    def test_explicit_worker_config_preserves_gate_request_path(self) -> None:
        args = SimpleNamespace(
            robot_usd="robot.usd",
            save_usd="scene.usd",
            fsm50_gate_request_path="C:/batch/worker_artifact_request.json",
        )
        config = build_worker_config(args, host="127.0.0.1", port=1234)
        self.assertEqual(
            config["fsm50_gate_request_path"],
            "C:/batch/worker_artifact_request.json",
        )

    def test_worker_main_preserves_default_sensor_state_before_role_hook(self) -> None:
        with mock.patch(
            "sim_worker_process.run_worker", return_value=0
        ) as run_worker:
            self.assertEqual(
                worker_main(
                    [
                        "--fsm50-gate-request-path",
                        "C:/batch/worker_artifact_request.json",
                    ]
                ),
                0,
            )

        launched_args = run_worker.call_args.args[0]
        self.assertEqual(
            launched_args.fsm50_gate_request_path,
            "C:/batch/worker_artifact_request.json",
        )
        self.assertIs(
            launched_args.telemetry_contact_sensors_enabled,
            False,
        )

    def test_client_and_transport_preserve_worker_session_binding(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        client = SimProcessClient(args)
        client.start_playback_plan(
            {"events": [], "segments": [], "plan_sha256": "a" * 64},
            plan_id="plan-1",
            request_id="request-1",
            worker_session_id="worker-session-1",
        )
        self.assertEqual(
            client.pending_messages[-1]["worker_session_id"],
            "worker-session-1",
        )

        class Plan:
            plan_sha256 = "a" * 64

        class BoundClient:
            def __init__(self) -> None:
                self.payload = {}

            def start_playback_plan(self, _plan_payload, **payload) -> None:
                self.payload = payload

        # Verify the public transport signature carries the exact binding.  A
        # real PlaybackPlan serialization is covered by playback unit tests.
        self.assertIn("worker_session_id", SimTransport.start_playback_plan.__annotations__)

    def test_preplay_stop_reason_is_preserved_in_ipc(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        client = SimProcessClient(args)
        request = client.stop_wheels(reason="playback_start_boundary")
        self.assertEqual(request["reason"], "playback_start_boundary")
        self.assertEqual(
            client.pending_messages[-1]["command_id"], request["command_id"]
        )

    def test_gate_ack_identity_is_exact_and_used_by_all_three_terminal_stages(self) -> None:
        adapter = SimpleNamespace(
            runtime_instance_id="adapter-1",
            root_state_write_count=0,
        )
        with mock.patch("sim_worker_process.os.getpid", return_value=4321):
            self.assertEqual(
                _worker_ack_identity(
                    adapter,
                    worker_session_id="worker-session-1",
                    artifact_request_id="request-1",
                ),
                {
                    "worker_pid": 4321,
                    "worker_session_id": "worker-session-1",
                    "adapter_runtime_instance_id": "adapter-1",
                    "artifact_request_id": "request-1",
                    "root_state_write_count": 0,
                },
            )

        tree = ast.parse((MODULE_ROOT / "sim_worker_process.py").read_text(encoding="utf-8"))
        protected_ack_kinds: set[str] = set()
        protected_ack_fields: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "make_message":
                continue
            uses_identity = any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Call)
                and isinstance(keyword.value.func, ast.Name)
                and keyword.value.func.id == "_worker_ack_identity"
                for keyword in node.keywords
            )
            if not uses_identity:
                continue
            message_type = (
                node.args[0].value
                if node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                else ""
            )
            operation = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "operation"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                "",
            )
            ack_kind = operation or message_type
            protected_ack_kinds.add(ack_kind)
            protected_ack_fields[ack_kind] = {
                str(keyword.arg)
                for keyword in node.keywords
                if keyword.arg is not None
            }
        self.assertTrue(
            {
                "stop_ack",
                "start_playback_plan",
                "recording_artifact",
                "shutdown",
                "close_requested",
                "close_returned",
            }
            <= protected_ack_kinds
        )
        required_close_fields = {
            "accepted",
            "error",
            "request_id",
            "mode",
            "close_kwargs",
            "runtime_version",
        }
        for ack_kind in ("shutdown", "close_requested", "close_returned"):
            self.assertTrue(
                required_close_fields <= protected_ack_fields[ack_kind],
                f"{ack_kind} is missing closure contract fields",
            )

    def test_playback_start_ack_explicitly_carries_execution_semantics(self) -> None:
        tree = ast.parse(
            (MODULE_ROOT / "sim_worker_process.py").read_text(encoding="utf-8")
        )
        start_ack_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "make_message":
                continue
            operation = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "operation"
                    and isinstance(keyword.value, ast.Constant)
                ),
                "",
            )
            if operation == "start_playback_plan":
                start_ack_calls.append(node)
        self.assertEqual(len(start_ack_calls), 1)
        self.assertIn(
            "execution_semantics",
            {keyword.arg for keyword in start_ack_calls[0].keywords},
        )

    def test_wrong_or_missing_worker_session_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_path, _artifact = self._request(Path(tmp))
            request = load_worker_recording_gate_request(request_path)
            assert request is not None
            self.assertEqual(
                validate_worker_plan_binding(
                    request,
                    request_id="request-1",
                    plan_id="plan-1",
                    worker_session_id="session-1",
                    expected_worker_session_id="session-1",
                ),
                [],
            )
            errors = validate_worker_plan_binding(
                request,
                request_id="request-1",
                plan_id="plan-1",
                worker_session_id="",
                expected_worker_session_id="session-1",
            )
            self.assertTrue(any("worker_session_id mismatch" in row for row in errors))

    def test_wait_returns_only_terminal_recording_operation_ack(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        client = SimProcessClient(args)
        client._handle_message(
            make_message(
                "artifact_complete",
                request_id="request-1",
                artifact_valid=False,
            )
        )
        self.assertEqual(client.latest_artifact_ack, {})
        terminal = make_message(
            "operation_ack",
            operation="recording_artifact",
            phase="ARTIFACT_COMPLETE",
            request_id="request-1",
            accepted=True,
            artifact_complete=True,
            artifact_valid=False,
        )
        client._handle_message(terminal)
        self.assertEqual(
            client.wait_for_artifact(timeout_s=None, request_id="request-1"),
            terminal,
        )

    def test_shutdown_no_force_returns_live_pid_without_tree_kill(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        client = SimProcessClient(args)

        class Process:
            pid = 4321

            @staticmethod
            def poll():
                return None

        client.process = Process()  # type: ignore[assignment]
        client.stop_wheels = mock.Mock()  # type: ignore[method-assign]
        client.request_shutdown = mock.Mock(return_value={})  # type: ignore[method-assign]
        client.poll = mock.Mock(return_value=[])  # type: ignore[method-assign]
        client._terminate_process_tree = mock.Mock()  # type: ignore[method-assign]
        outcome = client.shutdown(
            timeout_s=0.01,
            mode="fast",
            force_on_timeout=False,
            request_id="close-1",
        )
        self.assertTrue(outcome["timed_out"])
        self.assertFalse(outcome["forced_termination"])
        self.assertEqual(outcome["pid"], 4321)
        client._terminate_process_tree.assert_not_called()

    def test_close_events_remain_raw_and_request_bound(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        client = SimProcessClient(args)
        requested = make_message(
            "close_requested",
            request_id="close-1",
            close_kwargs={"wait_for_replicator": False, "skip_cleanup": True},
        )
        returned = make_message(
            "close_returned",
            request_id="close-1",
            close_kwargs={"wait_for_replicator": False, "skip_cleanup": True},
        )
        client._handle_message(requested)
        client._handle_message(returned)
        self.assertEqual(client.latest_status["close_requested_ack"], requested)
        self.assertEqual(client.latest_status["close_returned_ack"], returned)

    def test_fast_close_receipt_barrier_retains_ordered_shutdown_acks(self) -> None:
        args = SimpleNamespace(robot_usd="robot.usd", save_usd="scene.usd")
        client = SimProcessClient(args)
        worker_socket, client_socket = socket.socketpair()
        worker_socket.setblocking(False)
        client_socket.setblocking(False)
        ipc = WorkerIpc("", 0)
        ipc.sock = worker_socket
        client.conn = client_socket
        identity = {
            "worker_pid": 4321,
            "worker_session_id": "session-1",
            "adapter_runtime_instance_id": "adapter-1",
            "artifact_request_id": "artifact-1",
            "root_state_write_count": 0,
        }
        close_kwargs = {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        }
        shutdown = make_message(
            "operation_ack",
            operation="shutdown",
            request_id="close-1",
            mode="fast",
            accepted=True,
            error="",
            close_kwargs=close_kwargs,
            runtime_version="5.1.0",
            **identity,
        )
        requested = make_message(
            "close_requested",
            request_id="close-1",
            mode="fast",
            accepted=True,
            error="",
            close_kwargs=close_kwargs,
            runtime_version="5.1.0",
            **identity,
        )
        received: dict[str, dict] = {}

        try:
            ipc.send(shutdown)
            ipc.send(requested)

            def wait_for_receipt() -> None:
                received["receipt"] = _wait_for_close_receipt(
                    ipc,
                    requested,
                    timeout_s=1.0,
                )

            waiter = threading.Thread(target=wait_for_receipt)
            waiter.start()
            deadline = time.monotonic() + 1.0
            while waiter.is_alive() and time.monotonic() < deadline:
                client.poll()
                time.sleep(0.002)
            waiter.join(timeout=1.0)

            self.assertFalse(waiter.is_alive())
            self.assertEqual(
                received["receipt"]["close_event_type"], "close_requested"
            )
            self.assertEqual(received["receipt"]["request_id"], "close-1")
            self.assertEqual(
                client.latest_status["close_requested_ack"], requested
            )
            shutdown_rows = [
                row
                for row in client.latest_status["operation_ack_history"]
                if row.get("operation") == "shutdown"
            ]
            self.assertEqual(shutdown_rows, [shutdown])
            self.assertEqual(
                client.latest_status["close_requested_receipt"]["worker_session_id"],
                identity["worker_session_id"],
            )
        finally:
            client.close()
            ipc.close()

    def test_close_receipt_binding_rejects_json_equal_wrong_types(self) -> None:
        close_event = make_message(
            "close_requested",
            request_id="close-1",
            mode="fast",
            accepted=True,
            error="",
            worker_pid=4321,
            worker_session_id="session-1",
            adapter_runtime_instance_id="adapter-1",
            artifact_request_id="artifact-1",
            root_state_write_count=0,
            close_kwargs={
                "wait_for_replicator": False,
                "skip_cleanup": True,
            },
            runtime_version="5.1.0",
        )
        exact = {
            **close_event,
            "type": "close_receipt",
            "close_event_type": "close_requested",
            "received": True,
        }
        wrong_values = (
            ("accepted", 1),
            ("worker_pid", 4321.0),
            (
                "close_kwargs",
                {"wait_for_replicator": 0, "skip_cleanup": 1},
            ),
        )
        for key, wrong in wrong_values:
            with self.subTest(key=key), mock.patch.object(
                time, "sleep", return_value=None
            ):
                ipc = mock.Mock()
                malformed = dict(exact)
                malformed[key] = wrong
                ipc.poll.return_value = [malformed]
                receipt = _wait_for_close_receipt(
                    ipc,
                    close_event,
                    timeout_s=0.001,
                )
                self.assertEqual(receipt, {})
                self.assertGreater(ipc.poll.call_count, 0)

    def test_formal_fast_close_python_exception_is_reraised(self) -> None:
        source = (MODULE_ROOT / "sim_worker_process.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        run_worker = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_worker"
        )
        close_try = next(
            node
            for node in ast.walk(run_worker)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "close"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "simulation_app"
                for statement in node.body
                for child in ast.walk(statement)
            )
            and node.handlers
        )
        handler = close_try.handlers[0]
        fast_guard = next(
            node for node in handler.body if isinstance(node, ast.If)
        )
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(fast_guard)),
            "formal-fast close exceptions must override run_worker return 0",
        )


if __name__ == "__main__":
    unittest.main()
