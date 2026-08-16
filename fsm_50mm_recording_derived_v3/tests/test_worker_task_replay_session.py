from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsm_50mm_recording_derived_v3.worker_task_replay_session import (  # noqa: E402
    REQUEST_SCHEMA,
    WorkerTaskReplaySession,
    load_worker_task_replay_request,
    validate_worker_task_plan_binding,
)
from completion_aware_segment import RECORDED_TIMELINE_OPEN_LOOP_V1  # noqa: E402
from fsm_50mm_recording_derived_v3.v003_static_provenance import (  # noqa: E402
    DEFAULT_V003_DIRECTORY,
    V003_VERSION_ID,
)
from motion_speed import load_motion_reference  # noqa: E402
from playback import (  # noqa: E402
    PlaybackEvent,
    PlaybackPlan,
    PlaybackSegment,
    SimTimePlaybackService,
    plan_fingerprint,
    plan_from_steps,
)
from sequence_model import load_steps_jsonl  # noqa: E402
from fsm_50mm_recording_derived_v3.fsm50_task_success import (  # noqa: E402
    EVALUATED,
    NOT_EVALUATED,
    REPLAY_TASK_FAIL,
    REPLAY_TASK_NOT_EVALUATED,
    REPLAY_TASK_SUCCESS,
    classify_replay_task,
)


class _FakeRecorder:
    def __init__(self, root, *, enabled, fps):
        self.root = Path(root)
        self.video_path = self.root / "actual_viewport_video.mp4"
        self.enabled = enabled
        self.fps = fps
        self.error = ""
        self.finalized = False

    def start(self):
        return True

    def before_render(self, *, sim_step, sim_time_s):
        return None

    def after_render(self):
        return None

    def finalize(self):
        self.finalized = True
        self.root.mkdir(parents=True, exist_ok=True)
        self.video_path.write_bytes(b"synthetic-mp4")
        return {
            "schema_version": "fsm50.active_viewport_buffer_video.v1",
            "valid": True,
            "artifact_valid": True,
            "actual_viewport_video": True,
            "video_path": str(self.video_path),
            "frame_count": 8,
            "full_decode": {"valid": True, "decoded_frame_count": 8},
            "full_decode_all_frames": True,
            "error": "",
        }


class _FakeInvalidRecorder(_FakeRecorder):
    def finalize(self):
        self.finalized = True
        return {
            "schema_version": "fsm50.active_viewport_buffer_video.v1",
            "valid": False,
            "artifact_valid": False,
            "actual_viewport_video": False,
            "video_path": str(self.video_path),
            "full_decode": {"valid": False},
            "full_decode_all_frames": False,
            "error": "synthetic video failure",
        }


class _FakeAdapter:
    def __init__(self):
        joint_names = [
            "front_left_hip",
            "front_left_knee",
            "front_left_ankle",
            "front_right_hip",
            "front_right_knee",
            "front_right_ankle",
            "rear_left_hip",
            "rear_left_knee",
            "rear_left_ankle",
            "rear_right_hip",
            "rear_right_knee",
            "rear_right_ankle",
        ]
        body_names = [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ]
        self.robot = SimpleNamespace(
            joint_names=joint_names,
            body_names=body_names,
            data=SimpleNamespace(
                root_pose_w=np.asarray([[0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]]),
                root_vel_w=np.zeros((1, 6)),
                joint_pos=np.zeros((1, len(joint_names))),
                joint_vel=np.zeros((1, len(joint_names))),
                body_link_state_w=np.asarray(
                    [
                        [
                            [0.8, -0.2, 0.05] + [0.0] * 10,
                            [0.8, 0.2, 0.05] + [0.0] * 10,
                            [0.4, -0.2, 0.05] + [0.0] * 10,
                            [0.4, 0.2, 0.05] + [0.0] * 10,
                        ]
                    ]
                ),
            ),
        )
        self.sim_time = 0.0
        self.sim_steps = 0
        self.max_wheel_speed = 2.1
        self.telemetry_collector = None
        self.artifact_render_observer = None
        self.servo_applied_command_deg = {
            name: 0.0 for name in joint_names if not name.endswith("ankle")
        }
        self.wheel_speeds = {
            name: 0.0 for name in joint_names if name.endswith("ankle")
        }
        self.safe_joint_limit_records = {
            name: {"min_rad": -2.0, "max_rad": 2.0}
            for name in joint_names
            if not name.endswith("ankle")
        }

    def attach_artifact_render_observer(self, observer):
        assert self.artifact_render_observer is None
        self.artifact_render_observer = observer

    def detach_artifact_render_observer(self, observer):
        assert self.artifact_render_observer is observer
        self.artifact_render_observer = None

    def attach_telemetry(self, collector):
        self.telemetry_collector = collector

    def move_all_wheels(self, *, x: float, z: float):
        for body in self.robot.data.body_link_state_w[0]:
            body[0] = x
            body[2] = z

    def apply_motion_batch(self, payload):
        servo = dict(payload.get("servo_targets_deg", {}) or {})
        wheels = dict(payload.get("wheel_targets_rad_s", {}) or {})
        self.servo_applied_command_deg.update(servo)
        self.wheel_speeds.update(wheels)
        return {
            "batch_id": str(payload.get("batch_id", "") or ""),
            "error": "",
            "applied_sim_step": int(self.sim_steps),
            "first_physics_step": int(self.sim_steps) + 1,
            "motion_start_skew_s": 0.0,
            "servo_applied": bool(servo),
            "wheel_applied": bool(wheels),
            "servo_targets_applied": servo,
            "wheel_targets_applied": wheels,
            "recording_metadata": dict(
                payload.get("recording_metadata", {}) or {}
            ),
        }

    def stop_wheels(self):
        for name in self.wheel_speeds:
            self.wheel_speeds[name] = 0.0

    def apply_commands_to_robot(self):
        return None

    @staticmethod
    def command_to_actual_target_deg(_joint_name: str, command_deg: float) -> float:
        return float(command_deg)

    @staticmethod
    def get_final_target_limits_deg(_joint_name: str) -> tuple[float, float]:
        return (-120.0, 120.0)


class _FakeService:
    def __init__(self, request_id: str, plan_id: str, sha: str):
        self.request_id = request_id
        self.plan_id = plan_id
        self.sha = sha
        self.events_sent = 0
        self.segment_index = 0
        self.stop_reason = ""
        self.first_command_applied = False

    def status_dict(self, **_kwargs):
        return {
            "active": not bool(self.stop_reason),
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "plan_sha256": self.sha,
            "events_sent": self.events_sent,
            "count": 2,
            "event_count": 2,
            "segment_index": self.segment_index,
            "segment_count": 2,
            "index": self.events_sent,
            "stop_reason": self.stop_reason,
            "first_command_applied": self.first_command_applied,
            "progress_detail": {
                "playback_state": "completed" if self.stop_reason else "playing",
                "current_step_index": 1,
                "global_command_index": self.events_sent,
            },
        }


def _request_payload(run_dir: Path, steps_path: Path) -> dict:
    return {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": "request-v003",
        "plan_id": "plan-v003",
        "plan_sha256": "a" * 64,
        "execution_semantics": RECORDED_TIMELINE_OPEN_LOOP_V1,
        "plan_event_count": 2,
        "plan_segment_count": 2,
        "source_version": "v003_test",
        "height_mm": 50,
        "step_count": 1,
        "run_dir": str(run_dir),
        "accepted_steps_path": str(steps_path),
        "accepted_steps_sha256": hashlib.sha256(steps_path.read_bytes()).hexdigest(),
        "telemetry_hz": 15.0,
        "video_fps": 15.0,
        "capture_video": True,
        "post_run_settle_s": 0.1,
        "timeout_s": 10.0,
        "filtered_contact_bank_enabled": False,
    }


class WorkerTaskReplaySessionTests(unittest.TestCase):
    def _loaded_request(self, root: Path):
        steps = root / "accepted_steps.jsonl"
        steps.write_text('{"index":1,"events":[]}\n', encoding="utf-8")
        payload = _request_payload(root / "run", steps)
        request_path = root / "request.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")
        request = load_worker_task_replay_request(request_path)
        self.assertIsNotNone(request)
        return request

    @staticmethod
    def _runtime_fixture(request):
        adapter = _FakeAdapter()
        scene = SimpleNamespace(
            config=SimpleNamespace(
                obstacle_front_x=1.0,
                obstacle_height_m=0.05,
                obstacle_length=1.0,
                obstacle_width=2.0,
                ground_z_m=0.0,
                telemetry_contact_sensors_enabled=False,
            )
        )
        service = _FakeService(request.request_id, request.plan_id, request.plan_sha256)
        wheel_targets = {
            name: 0.0
            for name in (
                "front_left_ankle",
                "front_right_ankle",
                "rear_left_ankle",
                "rear_right_ankle",
            )
        }
        plan = SimpleNamespace(
            profile="motion_only",
            execution_semantics=request.execution_semantics,
            plan_sha256=request.plan_sha256,
            events=[1, 2],
            segments=[
                SimpleNamespace(
                    segment_index=index,
                    servo_targets={},
                    wheel_requested_velocity_rad_s=dict(wheel_targets),
                    wheel_applied_target_rad_s=dict(wheel_targets),
                )
                for index in range(2)
            ],
        )
        return adapter, scene, service, plan

    @staticmethod
    def _record_replay_callbacks(
        session,
        adapter,
        indices,
        *,
        callback_event_count=2,
    ):
        if callback_event_count is not None:
            session.start_replay(
                label="callback admission fixture",
                event_count=callback_event_count,
                final_time_s=0.2,
                started_sim_time_s=0.0,
            )
        for index in indices:
            session.record_replay_event(
                adapter,
                SimpleNamespace(
                    global_command_index=index + 1,
                    source_step=1,
                    segment_index=min(max(index, 0), 1),
                    command=f"event {index}",
                ),
                index,
            )

    def _complete_with_replay_callbacks(
        self,
        root,
        indices,
        *,
        callback_event_count=2,
    ):
        request = self._loaded_request(root)
        adapter, scene, service, plan = self._runtime_fixture(request)
        session = WorkerTaskReplaySession(
            request,
            worker_session_id="worker-session",
            recorder_factory=_FakeRecorder,
        )
        session.attach_verified_plan(
            plan=plan,
            service=service,
            adapter=adapter,
            scene_handle=scene,
        )
        self._record_replay_callbacks(
            session,
            adapter,
            indices,
            callback_event_count=callback_event_count,
        )
        service.events_sent = 2
        service.segment_index = 2
        service.first_command_applied = True
        service.stop_reason = "complete"
        service.completed_at_sim_s = 0.0
        session.finish_replay(success=True, reason="", sim_time_s=0.0)
        adapter.sim_time = 0.11
        terminal = session.after_adapter_step()
        self.assertIsNotNone(terminal)
        return terminal

    def _assert_callback_infrastructure_failure(self, terminal, token):
        self.assertEqual(terminal["type"], "task_replay_failed")
        self.assertIn(token, terminal["error"])
        completed = terminal["task_inputs"]["completed_result"]
        callback_status = completed["telemetry_callback_status"]
        self.assertFalse(callback_status["valid"])
        self.assertFalse(callback_status["event_count_complete"])
        self.assertTrue(completed["scheduler_complete"])
        self.assertTrue(completed["dispatch_complete"])
        self.assertEqual(
            completed["lifecycle"]["failure_kind"], "INFRASTRUCTURE"
        )
        assessment = classify_replay_task(
            completed_result=completed,
            physical_evidence=terminal["task_inputs"]["physical_evidence"],
            final_telemetry_row=terminal["task_inputs"][
                "final_telemetry_row"
            ],
        )
        self.assertEqual(assessment.evaluation_status, NOT_EVALUATED)
        self.assertEqual(assessment.task_result, REPLAY_TASK_NOT_EVALUATED)

    def test_empty_request_keeps_hook_disabled_and_request_forbids_filtered_bank(self):
        self.assertIsNone(load_worker_task_replay_request(""))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            steps = root / "accepted_steps.jsonl"
            steps.write_text("{}\n", encoding="utf-8")
            payload = _request_payload(root / "run", steps)
            payload["filtered_contact_bank_enabled"] = True
            path = root / "request.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbids the filtered"):
                load_worker_task_replay_request(path)

            payload = _request_payload(root / "run", steps)
            payload["execution_semantics"] = "measured_endpoint_v1"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "execution_semantics"):
                load_worker_task_replay_request(path)

    def test_plan_binding_requires_exact_production_fast_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            plan = SimpleNamespace(
                profile="motion_only",
                execution_semantics=request.execution_semantics,
                plan_sha256="a" * 64,
                events=[1, 2],
                segments=[1, 2],
            )
            self.assertEqual(
                validate_worker_task_plan_binding(
                    request,
                    plan=plan,
                    request_id="request-v003",
                    plan_id="plan-v003",
                ),
                [],
            )
            plan.profile = "raw"
            self.assertIn(
                "production Fast profile motion_only",
                "; ".join(
                    validate_worker_task_plan_binding(
                        request,
                        plan=plan,
                        request_id="request-v003",
                        plan_id="plan-v003",
                    )
                ),
            )
            plan.profile = "motion_only"
            plan.execution_semantics = "measured_endpoint_v1"
            self.assertIn(
                "execution_semantics",
                "; ".join(
                    validate_worker_task_plan_binding(
                        request,
                        plan=plan,
                        request_id="request-v003",
                        plan_id="plan-v003",
                    )
                ),
            )

    def test_production_session_writes_finite_classifier_inputs_and_quiesces_video(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            # The same production service/adapter objects are observed; the
            # session owns no compiler or scheduler.
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            self._record_replay_callbacks(session, adapter, [0, 1])
            self.assertIs(adapter.telemetry_collector, session)
            self.assertFalse(scene.config.telemetry_contact_sensors_enabled)

            adapter.sim_time = 0.10
            adapter.sim_steps = 12
            adapter.move_all_wheels(x=0.95, z=0.14)
            session.on_step(adapter, 1.0 / 120.0)

            adapter.sim_time = 0.20
            adapter.sim_steps = 24
            adapter.robot.data.root_pose_w[0, 0] = 1.25
            adapter.move_all_wheels(x=1.20, z=0.10)
            service.events_sent = 2
            service.segment_index = 2
            service.first_command_applied = True
            service.stop_reason = "complete"
            service.completed_at_sim_s = 0.20
            session.on_step(adapter, 1.0 / 120.0)

            adapter.sim_time = 0.31
            adapter.sim_steps = 37
            session.on_step(adapter, 1.0 / 120.0)
            terminal = session.after_adapter_step()
            self.assertIsNotNone(terminal)
            self.assertEqual(terminal["type"], "task_replay_complete")
            self.assertTrue(session.fast_close_ready)
            self.assertTrue(session.video_writer_quiesced)
            self.assertIsNone(adapter.telemetry_collector)
            self.assertIsNone(adapter.artifact_render_observer)

            task_inputs = terminal["task_inputs"]
            # This is the exact boundary consumed by fsm50_task_success.
            self.assertEqual(
                set(task_inputs),
                {
                    "schema_version",
                    "completed_result",
                    "physical_evidence",
                    "final_telemetry_row",
                },
            )
            completed = task_inputs["completed_result"]
            physical = task_inputs["physical_evidence"]
            final = task_inputs["final_telemetry_row"]
            self.assertTrue(completed["dispatch_complete"])
            self.assertTrue(completed["scheduler_complete"])
            self.assertEqual(completed["scheduler_stop_reason"], "complete")
            self.assertEqual(completed["plan_event_count"], 2)
            self.assertEqual(completed["sent_event_count"], 2)
            self.assertEqual(completed["completed_segment_count"], 2)
            self.assertTrue(completed["video_valid"])
            self.assertTrue(completed["video_full_decode_valid"])
            self.assertTrue(physical["required_leg_lift_completed"])
            self.assertFalse(physical["traversal"]["any_illegal_drive_up"])
            self.assertEqual(
                set(physical["final_wheel_contact_classes"]), set(("FL", "FR", "RL", "RR"))
            )
            self.assertEqual(
                final["wheel_contact_classes"],
                final["final_wheel_contact_classes"],
            )
            self.assertNotIn("active_contact_count", final)
            self.assertFalse(final["active_contact_count_available"])
            self.assertEqual(final["wheel_contact_load_n"], {leg: None for leg in ("FL", "FR", "RL", "RR")})
            self.assertIsNone(physical["dangerous_collision"])
            self.assertIsNone(physical["severe_penetration"])
            self.assertFalse(physical["unsafe_joint_target"])
            self.assertTrue(
                physical["unsafe_joint_target_evidence_available"]
            )
            for key in (
                "root_pose_w",
                "root_linear_velocity_w",
                "root_angular_velocity_w",
                "measured_joint_position_rad",
                "measured_joint_velocity_rad_s",
                "command_space_servo_target_deg",
                "wheel_command_velocity_rad_s",
            ):
                self.assertIn(key, final)
            # Classifier inputs can never carry NaN/Inf, including unavailable loads.
            json.dumps(task_inputs, allow_nan=False)
            self.assertTrue((request.run_dir / "minimal_telemetry.jsonl").is_file())
            self.assertTrue((request.run_dir / "minimal_telemetry.csv").is_file())
            self.assertTrue((request.run_dir / "task_inputs.json").is_file())
            self.assertTrue((request.run_dir / "worker_task_replay_result.json").is_file())

            assessment = classify_replay_task(
                completed_result=completed,
                physical_evidence=physical,
                final_telemetry_row=final,
                video_verdict={
                    "task_completed": True,
                    "body_crossed_front_face": True,
                    "required_leg_lift_completed": True,
                    "final_recoverable": True,
                    "posture_incomplete": False,
                    "robot_fell": False,
                    "body_stuck": False,
                    "wheel_drive_up_without_required_lift": False,
                    "dangerous_body_collision": False,
                    "joint_limit_violation": False,
                    "severe_penetration": False,
                    "irrecoverable": False,
                },
            )
            self.assertEqual(assessment.evaluation_status, EVALUATED)
            self.assertEqual(assessment.task_result, REPLAY_TASK_SUCCESS)

    def test_production_scheduler_telemetry_callbacks_do_not_set_last_error(self):
        with tempfile.TemporaryDirectory() as temp:
            base_request = self._loaded_request(Path(temp))
            adapter = _FakeAdapter()
            wheel_targets = {
                name: 0.5 for name in adapter.wheel_speeds
            }
            event = PlaybackEvent(
                time_s=0.0,
                command="wheel all 0.5",
                source_step=1,
                source_step_id="step-1",
                global_command_index=1,
                segment_index=0,
                channel="wheel",
                dispatch_command=True,
                wheel_requested_velocity_rad_s=tuple(wheel_targets.values()),
                wheel_applied_target_rad_s=tuple(wheel_targets.values()),
                wheel_active_duration_s=0.1,
            )
            segment = PlaybackSegment(
                segment_index=0,
                source_step=1,
                source_step_id="step-1",
                event_start_index=0,
                event_count=1,
                planned_start_s=0.0,
                planned_end_s=0.1,
                base_duration_s=0.1,
                servo_base_duration_s=0.0,
                servo_duration_s=0.0,
                wheel_active_duration_s=0.1,
                wheel_base_velocity=dict(wheel_targets),
                wheel_requested_velocity_rad_s=dict(wheel_targets),
                wheel_applied_target_rad_s=dict(wheel_targets),
            )
            plan = PlaybackPlan(
                path=None,
                events=[event],
                segments=[segment],
                final_time_s=0.1,
                label="normal task callback regression",
                profile="motion_only",
                execution_semantics=RECORDED_TIMELINE_OPEN_LOOP_V1,
                total_steps=1,
            )
            plan.plan_sha256 = plan_fingerprint(plan)
            request = replace(
                base_request,
                plan_sha256=plan.plan_sha256,
                plan_event_count=1,
                plan_segment_count=1,
            )
            scene = SimpleNamespace(
                config=SimpleNamespace(
                    obstacle_front_x=1.0,
                    obstacle_height_m=0.05,
                    obstacle_length=1.0,
                    obstacle_width=2.0,
                    ground_z_m=0.0,
                    telemetry_contact_sensors_enabled=False,
                )
            )
            service = SimTimePlaybackService()
            self.assertTrue(
                service.start_plan(
                    plan,
                    current_sim_time_s=0.0,
                    current_wall_time_s=0.0,
                    plan_id=request.plan_id,
                    request_id=request.request_id,
                    worker_session_id="worker-session",
                )
            )
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )

            service.update(
                adapter,
                current_sim_time_s=0.0,
                current_sim_step=adapter.sim_steps,
                current_wall_time_s=0.0,
            )
            session.record_command(
                adapter,
                SimpleNamespace(
                    source="playback",
                    playback_event_index=0,
                    source_step=1,
                ),
                event.command,
            )

            self.assertEqual(service.last_error, "")
            self.assertTrue(session.replay_callback_started)
            self.assertEqual(session.replay_callback_expected_event_count, 1)
            self.assertEqual(session.replay_event_indices_seen, {0})
            self.assertEqual(session.last_replay_event["command"], event.command)
            self.assertEqual(session.command_callback_count, 1)
            adapter.sim_time = 0.11
            adapter.sim_steps = 1
            service.update(
                adapter,
                current_sim_time_s=adapter.sim_time,
                current_sim_step=adapter.sim_steps,
                current_wall_time_s=0.11,
            )
            self.assertEqual(service.stop_reason, "complete")
            adapter.sim_time = 0.22
            adapter.sim_steps = 2
            terminal = session.after_adapter_step()
            self.assertEqual(terminal["type"], "task_replay_complete")
            callback_status = terminal["task_inputs"]["completed_result"][
                "telemetry_callback_status"
            ]
            self.assertTrue(callback_status["event_count_complete"])
            self.assertEqual(callback_status["recorded_event_count"], 1)

    def test_callback_admission_requires_start_and_matching_plan_count(self):
        cases = (
            ("missing-start", None, [0, 1], "start_replay callback"),
            ("wrong-count", 3, [0, 1], "does not match the admitted plan"),
        )
        for name, callback_count, indices, token in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                terminal = self._complete_with_replay_callbacks(
                    Path(temp),
                    indices,
                    callback_event_count=callback_count,
                )
                self._assert_callback_infrastructure_failure(terminal, token)

    def test_callback_admission_rejects_missing_duplicate_and_out_of_range(self):
        cases = (
            ("missing", [0], "missing replay event indices"),
            ("duplicate", [0, 1, 1], "duplicate replay event indices"),
            ("out-of-range", [0, 2], "out-of-range replay event indices"),
        )
        for name, indices, token in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                terminal = self._complete_with_replay_callbacks(
                    Path(temp), indices
                )
                self._assert_callback_infrastructure_failure(terminal, token)

    def test_unsafe_servo_target_is_rejected_before_recorder_or_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            plan.segments[0].servo_targets = {"front_left_hip": 999.0}
            recorder_calls = []

            def recorder_factory(*args, **kwargs):
                recorder_calls.append((args, kwargs))
                return _FakeRecorder(*args, **kwargs)

            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=recorder_factory,
            )
            with self.assertRaisesRegex(RuntimeError, "unsafe applied target"):
                session.attach_verified_plan(
                    plan=plan,
                    service=service,
                    adapter=adapter,
                    scene_handle=scene,
                )
            self.assertTrue(session.unsafe_joint_target_detected)
            self.assertEqual(recorder_calls, [])
            self.assertIsNone(adapter.telemetry_collector)
            self.assertIsNone(adapter.artifact_render_observer)
            self.assertEqual(session.state, "ready_for_plan")

            terminal = session.fail("unsafe plan rejected before boundary")
            completed = terminal["task_inputs"]["completed_result"]
            physical = terminal["task_inputs"]["physical_evidence"]
            self.assertFalse(completed["scheduler_complete"])
            self.assertIsNone(completed["motion_start_ready"])
            self.assertIsNone(completed["actuator_targets_applied"])
            self.assertNotIn("artifact_valid", completed)
            self.assertTrue(physical["unsafe_joint_target"])
            self.assertTrue(
                physical["unsafe_joint_target_evidence_available"]
            )
            self.assertEqual(
                physical["plan_target_audit"]["violations"][0]["kind"],
                "servo_target_outside_safe_envelope",
            )
            assessment = classify_replay_task(
                completed_result=completed,
                physical_evidence=physical,
                final_telemetry_row=terminal["task_inputs"]["final_telemetry_row"],
            )
            self.assertEqual(assessment.evaluation_status, EVALUATED)
            self.assertEqual(assessment.task_result, REPLAY_TASK_FAIL)
            self.assertIn("UNSAFE_JOINT_TARGET", assessment.hard_failure_reasons)

    def test_extreme_requested_wheel_target_is_safe_when_production_clamp_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            segment = plan.segments[0]
            joint = "front_right_ankle"
            segment.wheel_requested_velocity_rad_s[joint] = 1.0e6
            segment.wheel_applied_target_rad_s[joint] = adapter.max_wheel_speed
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )

            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            audit = session.plan_target_audit

            self.assertTrue(audit["available"])
            self.assertFalse(audit["unsafe"])
            self.assertEqual(audit["violations"], [])
            self.assertEqual(audit["clamped_wheel_target_count"], 1)
            detail = audit["clamped_wheel_targets"][0]
            self.assertEqual(detail["joint"], joint)
            self.assertEqual(
                detail["expected_clamped_target_rad_s"], adapter.max_wheel_speed
            )
            self.assertEqual(
                detail["applied_target_rad_s"], adapter.max_wheel_speed
            )
            self.assertEqual(session.state, "recording")
            session.fail("test cleanup")

    def test_malicious_applied_wheel_target_mismatch_remains_unsafe(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            segment = plan.segments[0]
            joint = "front_right_ankle"
            segment.wheel_requested_velocity_rad_s[joint] = 1.0e6
            segment.wheel_applied_target_rad_s[joint] = (
                adapter.max_wheel_speed - 0.01
            )
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )

            with self.assertRaisesRegex(RuntimeError, "unsafe applied target"):
                session.attach_verified_plan(
                    plan=plan,
                    service=service,
                    adapter=adapter,
                    scene_handle=scene,
                )
            audit = session.plan_target_audit

            self.assertTrue(audit["available"])
            self.assertTrue(audit["unsafe"])
            self.assertEqual(audit["clamped_wheel_target_count"], 1)
            self.assertEqual(len(audit["violations"]), 1)
            self.assertEqual(
                audit["violations"][0]["kind"],
                "wheel_applied_target_mismatch",
            )
            self.assertIsNone(adapter.telemetry_collector)
            self.assertIsNone(adapter.artifact_render_observer)
            session.fail("test cleanup")

    def test_out_of_bound_and_nonfinite_applied_wheel_targets_remain_unsafe(self):
        cases = (
            (2.2, "wheel_applied_target_outside_safe_envelope"),
            (float("nan"), "nonfinite_wheel_target"),
        )
        for applied, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind), tempfile.TemporaryDirectory() as temp:
                request = self._loaded_request(Path(temp))
                adapter, _scene, _service, plan = self._runtime_fixture(request)
                joint = "front_right_ankle"
                plan.segments[0].wheel_requested_velocity_rad_s[joint] = 0.0
                plan.segments[0].wheel_applied_target_rad_s[joint] = applied
                session = WorkerTaskReplaySession(
                    request,
                    worker_session_id="worker-session",
                    recorder_factory=_FakeRecorder,
                )

                audit = session._audit_plan_targets(plan, adapter)

                self.assertTrue(audit["unsafe"])
                self.assertEqual(audit["violations"][0]["kind"], expected_kind)
                json.dumps(audit, allow_nan=False)

    def test_real_v003_production_plan_clamp_is_diagnostic_not_unsafe(self):
        with tempfile.TemporaryDirectory() as temp:
            base_request = self._loaded_request(Path(temp))
            steps_path = DEFAULT_V003_DIRECTORY / "accepted_steps.jsonl"
            steps = load_steps_jsonl(steps_path)
            max_wheel_speed = float(
                load_motion_reference().wheel_velocity_limit_rad_s
            )
            plan = plan_from_steps(
                steps,
                profile="fast",
                max_wheel_speed=max_wheel_speed,
                label="v003 target-audit regression",
                sequence_total_steps=len(steps),
            )
            request = replace(
                base_request,
                source_version=V003_VERSION_ID,
                step_count=len(steps),
                accepted_steps_path=steps_path.resolve(),
                accepted_steps_sha256=hashlib.sha256(
                    steps_path.read_bytes()
                ).hexdigest(),
                plan_sha256=plan.plan_sha256,
                plan_event_count=len(plan.events),
                plan_segment_count=len(plan.segments),
            )
            adapter, scene, service, _synthetic_plan = self._runtime_fixture(request)
            adapter.max_wheel_speed = max_wheel_speed
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )

            self.assertEqual(
                validate_worker_task_plan_binding(
                    request,
                    plan=plan,
                    request_id=request.request_id,
                    plan_id=request.plan_id,
                ),
                [],
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            audit = session.plan_target_audit

            self.assertTrue(audit["available"])
            self.assertFalse(audit["unsafe"])
            self.assertEqual(audit["violations"], [])
            self.assertEqual(audit["clamped_wheel_target_count"], 1)
            detail = audit["clamped_wheel_targets"][0]
            self.assertEqual(detail["segment_index"], 56)
            self.assertEqual(detail["joint"], "front_right_ankle")
            self.assertEqual(detail["requested_target_rad_s"], 2.0944)
            self.assertEqual(
                detail["expected_clamped_target_rad_s"], max_wheel_speed
            )
            self.assertEqual(detail["applied_target_rad_s"], max_wheel_speed)
            self.assertEqual(session.state, "recording")
            self.assertIs(adapter.telemetry_collector, session)
            self.assertIsNotNone(adapter.artifact_render_observer)
            session.fail("test cleanup")

    def test_video_infrastructure_failure_preserves_actual_scheduler_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeInvalidRecorder,
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            self._record_replay_callbacks(session, adapter, [0, 1])
            service.events_sent = 2
            service.segment_index = 2
            service.first_command_applied = True
            service.stop_reason = "complete"
            service.completed_at_sim_s = 0.0
            session.finish_replay(success=True, reason="", sim_time_s=0.0)
            adapter.sim_time = 0.11
            terminal = session.after_adapter_step()

            self.assertEqual(terminal["type"], "task_replay_failed")
            completed = terminal["task_inputs"]["completed_result"]
            self.assertTrue(completed["dispatch_complete"])
            self.assertTrue(completed["scheduler_complete"])
            self.assertTrue(completed["actuator_targets_applied"])
            self.assertFalse(completed["artifact_valid"])
            self.assertTrue(completed["lifecycle"]["failed"])
            self.assertEqual(
                completed["lifecycle"]["failure_kind"], "INFRASTRUCTURE"
            )
            assessment = classify_replay_task(
                completed_result=completed,
                physical_evidence=terminal["task_inputs"]["physical_evidence"],
                final_telemetry_row=terminal["task_inputs"]["final_telemetry_row"],
            )
            self.assertEqual(assessment.evaluation_status, NOT_EVALUATED)
            self.assertEqual(assessment.task_result, REPLAY_TASK_NOT_EVALUATED)

    def test_short_root_state_is_a_nonfinite_core_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            adapter.robot.data.root_pose_w = np.zeros((1, 6))
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            self.assertTrue(session.nonfinite_state_detected)
            self.assertFalse(session.rows[-1]["robot_state_finite"])
            service.events_sent = 2
            service.segment_index = 2
            service.first_command_applied = True
            service.stop_reason = "complete"
            terminal = session.fail("invalid state fixture")
            self.assertEqual(terminal["type"], "task_replay_failed")
            json.dumps(terminal["task_inputs"], allow_nan=False)
            assessment = classify_replay_task(
                completed_result=terminal["task_inputs"]["completed_result"],
                physical_evidence=terminal["task_inputs"]["physical_evidence"],
                final_telemetry_row=terminal["task_inputs"]["final_telemetry_row"],
                video_verdict={
                    "task_completed": False,
                    "body_crossed_front_face": False,
                    "required_leg_lift_completed": False,
                    "final_recoverable": False,
                },
            )
            self.assertEqual(assessment.evaluation_status, EVALUATED)
            self.assertEqual(assessment.task_result, REPLAY_TASK_FAIL)
            self.assertTrue(
                any(
                    reason.startswith("NONFINITE_CORE_STATE")
                    for reason in assessment.hard_failure_reasons
                )
            )

    def test_missing_one_of_twelve_joint_states_is_a_core_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            adapter.robot.joint_names = adapter.robot.joint_names[:-1]
            adapter.robot.data.joint_pos = adapter.robot.data.joint_pos[:, :-1]
            adapter.robot.data.joint_vel = adapter.robot.data.joint_vel[:, :-1]
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            self.assertTrue(session.nonfinite_state_detected)
            self.assertFalse(session.rows[-1]["robot_state_finite"])
            session.fail("missing joint state fixture")

    def test_multi_environment_tensor_shape_cannot_flatten_into_valid_state(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            adapter.robot.data.root_pose_w = np.repeat(
                adapter.robot.data.root_pose_w, 2, axis=0
            )
            adapter.robot.data.root_vel_w = np.repeat(
                adapter.robot.data.root_vel_w, 2, axis=0
            )
            adapter.robot.data.joint_pos = np.repeat(
                adapter.robot.data.joint_pos, 2, axis=0
            )
            adapter.robot.data.joint_vel = np.repeat(
                adapter.robot.data.joint_vel, 2, axis=0
            )
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            self.assertTrue(session.nonfinite_state_detected)
            self.assertFalse(session.rows[-1]["robot_state_finite"])
            terminal = session.fail("multi-environment state fixture")
            self.assertTrue(
                terminal["task_inputs"]["completed_result"][
                    "nonfinite_core_state_detected"
                ]
            )

    def test_unavailable_plan_target_audit_rejects_admission_as_not_evaluated(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            del adapter.max_wheel_speed
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            with self.assertRaisesRegex(RuntimeError, "target-limit audit"):
                session.attach_verified_plan(
                    plan=plan,
                    service=service,
                    adapter=adapter,
                    scene_handle=scene,
                )
            self.assertIsNone(session.unsafe_joint_target_detected)
            terminal = session.fail("plan target audit unavailable")
            completed = terminal["task_inputs"]["completed_result"]
            self.assertTrue(completed["lifecycle"]["failed"])
            self.assertEqual(
                completed["lifecycle"]["failure_kind"],
                "PLAN_TARGET_EVIDENCE_UNAVAILABLE",
            )

    def test_missing_servo_safe_limit_record_is_unavailable_not_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            adapter, scene, service, plan = self._runtime_fixture(request)
            del adapter.safe_joint_limit_records["front_left_hip"]
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            session.attach_verified_plan(
                plan=plan,
                service=service,
                adapter=adapter,
                scene_handle=scene,
            )
            row = session.rows[-1]
            self.assertFalse(row["joint_limit_evidence_available"])
            self.assertIsNone(row["joint_limit_violation"])
            terminal = session.fail("fixture complete")
            physical = terminal["task_inputs"]["physical_evidence"]
            self.assertFalse(physical["joint_limit_evidence_available"])
            self.assertIsNone(physical["joint_limit_violation"])
            self.assertIsNone(physical["joint_limit_violation_count"])
            self.assertNotIn("joint_limits_safe", physical["criteria"])

    def test_failed_terminal_is_also_safe_for_receipted_fast_close(self):
        with tempfile.TemporaryDirectory() as temp:
            request = self._loaded_request(Path(temp))
            session = WorkerTaskReplaySession(
                request,
                worker_session_id="worker-session",
                recorder_factory=_FakeRecorder,
            )
            terminal = session.fail("scheduler admission failed")
            self.assertEqual(terminal["type"], "task_replay_failed")
            self.assertFalse(terminal["accepted"])
            self.assertTrue(session.video_writer_quiesced)
            self.assertTrue(session.fast_close_ready)
            self.assertTrue(Path(terminal["worker_result_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
