from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
try:
    import pytest
except ImportError:  # pragma: no cover - lets the stdlib smoke harness run too
    class _Raises:
        def __init__(self, error, match=""):
            self.error = error
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, kind, value, _tb):
            if kind is None or not issubclass(kind, self.error):
                return False
            if self.match and self.match not in str(value):
                raise AssertionError(f"{self.match!r} not in {value!r}")
            return True

    class _PytestFallback:
        raises = _Raises

        class mark:
            @staticmethod
            def parametrize(*_args, **_kwargs):
                return lambda function: function

    pytest = _PytestFallback()

from command_model import (
    JOINT_COMMAND_SIGN,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    command_limits_for_servo,
)
from fsm_50mm_recording_derived_v3.fsm50_direct_command_residual import (
    RESIDUAL_ACTION_NAMES,
    ZERO_RESIDUAL_ACTION,
    ResidualPhaseContract,
    ZeroResidualPolicy,
    canonical_mapping_sha256,
)
from fsm_50mm_recording_derived_v3.fsm50_macro_controller import (
    FeedbackRecoveryObservation,
    MacroObservation,
)
from fsm_50mm_recording_derived_v3.filtered_wheel_contact import (
    FILTERED_SURFACES,
    FilteredWheelContactSensorBank,
    wheel_contact_sensor_specs,
)
from fsm_50mm_recording_derived_v3.nonwheel_obstacle_contact import (
    NonWheelObstacleContactSensorBank,
    NonWheelRigidBodySpec,
    WheelAndNonWheelContactSensorBank,
)
from fsm_50mm_recording_derived_v3.worker_macro_fsm_session import (
    AUTHORIZED_GATE_D_SOURCE_VERSIONS,
    CANONICAL_GATE_C_SOURCE_VERSION,
    GATE_D_TRIAL_KIND,
    REQUEST_SCHEMA,
    WorkerMacroFSMSession,
    configure_scene_for_macro_fsm,
    load_worker_macro_fsm_request,
    validate_worker_macro_start_binding,
)


class _Recorder:
    def __init__(self, root, *, enabled, fps):
        self.root = Path(root)
        self.enabled = enabled
        self.fps = fps
        self.error = ""
        self.video_path = self.root / "actual_viewport_video.mp4"

    def start(self):
        return True

    def finalize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.video_path.write_bytes(b"macro-video")
        return {
            "valid": True,
            "video_path": str(self.video_path),
            "full_decode": {"valid": True},
            "full_decode_all_frames": True,
            "error": "",
        }


class _Adapter:
    def __init__(self):
        names = list(SERVO_JOINT_NAMES) + list(WHEEL_JOINT_NAMES)
        bodies = [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ]
        self.write_calls = 0
        body_points = np.asarray(
            [
                [0.80, -0.20, 0.05],
                [0.80, 0.20, 0.05],
                [0.40, -0.20, 0.05],
                [0.40, 0.20, 0.05],
            ],
            dtype=float,
        )
        body_pose = np.concatenate(
            (body_points, np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]]), (4, 1))),
            axis=1,
        )
        data = SimpleNamespace(
            root_pose_w=np.asarray([[0.5, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0]]),
            root_vel_w=np.zeros((1, 6)),
            joint_pos=np.zeros((1, len(names))),
            joint_vel=np.zeros((1, len(names))),
            body_link_state_w=np.asarray(
                [
                    [
                        [0.80, -0.20, 0.05] + [0.0] * 10,
                        [0.80, 0.20, 0.05] + [0.0] * 10,
                        [0.40, -0.20, 0.05] + [0.0] * 10,
                        [0.40, 0.20, 0.05] + [0.0] * 10,
                    ]
                ]
            ),
            body_names=bodies,
            _sim_timestamp=0.0,
            _body_com_pose_w=SimpleNamespace(timestamp=0.0),
            _body_com_vel_w=SimpleNamespace(timestamp=0.0),
            _body_com_acc_w=SimpleNamespace(timestamp=0.0),
            body_com_pos_w=body_points.reshape(1, 4, 3),
            body_com_pose_w=body_pose.reshape(1, 4, 7),
            body_com_lin_vel_w=np.zeros((1, 4, 3)),
            body_com_vel_w=np.zeros((1, 4, 6)),
            body_com_lin_acc_w=np.zeros((1, 4, 3)),
            body_com_acc_w=np.zeros((1, 4, 6)),
        )
        root_view = SimpleNamespace(
            shared_metatype=SimpleNamespace(link_names=bodies),
            get_masses=lambda: np.ones((1, 4), dtype=float),
            get_inertias=lambda: np.tile(
                np.asarray([[0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]]),
                (1, 4, 1),
            ),
        )
        self.robot = SimpleNamespace(
            joint_names=names,
            body_names=bodies,
            num_bodies=4,
            num_instances=1,
            root_physx_view=root_view,
            write_data_to_sim=self._write,
            data=data,
        )
        self._sim_time = 0.0
        self.sim_steps = 0
        self.physics_dt_s = 1.0 / 120.0
        self.sim = SimpleNamespace(get_physics_dt=lambda: self.physics_dt_s)
        self.config = SimpleNamespace(ground_penetration_tolerance_m=0.003)
        self.motion_reference = SimpleNamespace(
            servo_reference_velocity_deg_s=150.0,
            servo_velocity_limit_deg_s=None,
        )
        self.max_wheel_speed = 2.1
        self.joint_command_deg = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self.servo_applied_command_deg = {
            name: 0.0 for name in SERVO_JOINT_NAMES
        }
        self.wheel_speeds = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self.wheel_generation = 0
        self.root_state_write_count = 0
        self.root_state_write_events = []
        self.runtime_instance_id = "adapter-id"
        self.grounded_reference_valid = True
        self.telemetry_collector = None
        self.artifact_render_observer = None
        self.batch_calls: list[dict] = []
        self.batch_call_sim_steps: list[int] = []
        self.motion_batch_status = {}
        self.readback_unavailable = False
        self.readback_target_mismatch = False
        self.safe_stop_failure = ""
        self.begin_tracking_calls: list[dict[str, float]] = []
        self.begin_tracking_exception: Exception | None = None
        self.end_tracking_calls: list[dict[str, float]] = []
        self.end_tracking_result: object = {"ended": True}
        self.end_tracking_exception: Exception | None = None
        self.safe_joint_limit_records = {
            name: {"min_rad": -2.0, "max_rad": 2.0}
            for name in SERVO_JOINT_NAMES
        }
        self.event_log = None

    def _write(self):
        self.write_calls += 1

    @property
    def sim_time(self):
        return self._sim_time

    @sim_time.setter
    def sim_time(self, value):
        self._sim_time = float(value)
        if hasattr(self, "robot"):
            self._sync_physics_buffers()

    def _sync_physics_buffers(self):
        self.robot.data._sim_timestamp = self.sim_time
        for name in ("_body_com_pose_w", "_body_com_vel_w", "_body_com_acc_w"):
            getattr(self.robot.data, name).timestamp = self.sim_time

    def _render_step_timing(self):
        return (8.0 / 120.0, 8)

    def attach_telemetry(self, value):
        self.telemetry_collector = value

    def attach_artifact_render_observer(self, value):
        assert self.artifact_render_observer is None
        self.artifact_render_observer = value

    def detach_artifact_render_observer(self, value):
        assert self.artifact_render_observer is value
        self.artifact_render_observer = None

    def apply_motion_batch(self, payload):
        row = dict(payload)
        self.batch_calls.append(row)
        self.batch_call_sim_steps.append(self.sim_steps)
        if row.get("source") == "fsm50_macro_safe_stop" and self.safe_stop_failure == "apply":
            raise RuntimeError("injected safe-stop apply failure")
        servos = dict(row["servo_targets_deg"])
        wheels = dict(row["wheel_targets_rad_s"])
        self.joint_command_deg.update(servos)
        self.servo_applied_command_deg.update(servos)
        self.wheel_speeds.update(wheels)
        ack = {
            "batch_id": row["batch_id"],
            "source": row.get("source", ""),
            "error": "",
            "applied_sim_step": self.sim_steps,
            "first_physics_step": self.sim_steps + 1,
            "motion_start_skew_s": 0.0,
            "physics_dt_s": self.physics_dt_s,
            "servo_applied": True,
            "wheel_applied": True,
            "servo_targets_applied": servos,
            "wheel_targets_applied": wheels,
            "recording_metadata": dict(row.get("recording_metadata", {}) or {}),
        }
        if row.get("source") == "fsm50_macro_safe_stop" and self.safe_stop_failure == "ack":
            ack["wheel_applied"] = False
        self.motion_batch_status = dict(ack)
        return ack

    def capture_motion_start_base_evidence(self):
        if self.readback_unavailable:
            raise RuntimeError("injected target readback unavailable")
        canonical_servos = dict(self.joint_command_deg)
        if self.readback_target_mismatch:
            canonical_servos[SERVO_JOINT_NAMES[0]] += 0.5
        servo_rad = {
            name: math.radians(
                float(JOINT_COMMAND_SIGN[name]) * float(canonical_servos[name])
            )
            for name in SERVO_JOINT_NAMES
        }
        position_targets = {
            **servo_rad,
            **{name: 0.0 for name in WHEEL_JOINT_NAMES},
        }
        return {
            "adapter_runtime_instance_id": self.runtime_instance_id,
            "root_state_write_count": self.root_state_write_count,
            "root_state_write_events": list(self.root_state_write_events),
            "sim_step": self.sim_steps,
            "sim_time_s": self.sim_time,
            "physics_dt_s": self.physics_dt_s,
            "command_state": {
                "servos": canonical_servos,
                "wheels": dict(self.wheel_speeds),
            },
            "joint_state_evidence_valid": True,
            "joint_state_evidence_error": "",
            "joint_position_target_by_name": position_targets,
            "joint_position_target_buffer_by_name": position_targets,
            "servo_command_target_by_name": servo_rad,
            "wheel_target_evidence_valid": True,
            "wheel_target_evidence_error": "",
            "wheel_target_velocity_by_name": dict(self.wheel_speeds),
            "wheel_target_readback_velocity_by_name": dict(self.wheel_speeds),
        }

    def capture_macro_safety_evidence(self, *, scene_handle):
        if self.event_log is not None:
            self.event_log.append(("capture_macro_safety_evidence", self.sim_steps))
        return {
            "available": True,
            "dangerous_body_collision": False,
            "severe_penetration": False,
            "source": "TEST_LIVE_INITIAL_GEOMETRY",
            "sample_sim_step": self.sim_steps,
            "geometry": {
                "obstacle_bounds_min_m": [1.0, -1.0, 0.0],
                "obstacle_bounds_max_m": [2.0, 1.0, 0.05],
                "robot_collision_bounds_min_m": [0.2, -0.3, 0.0],
                "robot_collision_bounds_max_m": [0.85, 0.3, 0.3],
                "wheel_collision_centers": [
                    {"leg": leg, "center_m": [0.4, 0.0, 0.05]}
                    for leg in ("FL", "FR", "RL", "RR")
                ],
            },
            "ground_diagnostics": {"maximum_collision_penetration_m": 0.0},
            "robot_root_pose": [0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "obstacle_bounds_min_m": [1.0, -1.0, 0.0],
            "obstacle_bounds_max_m": [2.0, 1.0, 0.05],
            "robot_collision_bounds_min_m": [0.2, -0.3, 0.0],
            "robot_collision_bounds_max_m": [0.85, 0.3, 0.3],
            "initial_robot_to_obstacle_clearance_m": 0.15,
            "initial_maximum_ground_penetration_m": 0.0,
            "ground_penetration_tolerance_m": 0.003,
            "error": "",
        }

    def stop_wheels(self):
        self.wheel_generation += 1
        self.wheel_speeds = {name: 0.0 for name in WHEEL_JOINT_NAMES}

    def apply_commands_to_robot(self):
        return None

    @staticmethod
    def command_to_actual_target_deg(name, value):
        return float(JOINT_COMMAND_SIGN[name]) * float(value)

    @staticmethod
    def get_final_target_limits_deg(_name):
        return (-135.0, 135.0)

    def begin_servo_tracking(self, targets):
        self.begin_tracking_calls.append(dict(targets))
        if self.begin_tracking_exception is not None:
            raise self.begin_tracking_exception

    def servo_tracking_completion_evidence(self, targets):
        return {
            "supported": True,
            "converged": True,
            "joints": {name: {"converged": True} for name in targets},
        }

    def end_servo_tracking(self, targets):
        self.end_tracking_calls.append(dict(targets))
        if self.end_tracking_exception is not None:
            raise self.end_tracking_exception
        if isinstance(self.end_tracking_result, dict):
            return dict(self.end_tracking_result)
        return self.end_tracking_result


class _FakeRawContactView:
    def __init__(self, data):
        self._data = data
        self.sensor_count = int(np.asarray(data.force_matrix_w).shape[0])
        self.filter_count = int(np.asarray(data.force_matrix_w).shape[2])
        self.max_contact_data_count = 8

    @staticmethod
    def _layout(vector_rows, point_rows):
        capacity = 8
        values = np.full((capacity, 3), np.nan, dtype=float)
        points = np.full((capacity, 3), np.nan, dtype=float)
        counts = np.zeros((1, 2), dtype=np.int64)
        starts = np.zeros((1, 2), dtype=np.int64)
        cursor = 0
        for filter_index, (vector, point) in enumerate(zip(vector_rows, point_rows)):
            starts[0, filter_index] = cursor
            if vector is None:
                continue
            values[cursor] = np.asarray(vector, dtype=float)
            points[cursor] = np.asarray(point, dtype=float)
            counts[0, filter_index] = 1
            cursor += 1
        return values, points, counts, starts

    def check(self):
        return True

    def get_contact_data(self, *, dt):
        assert float(dt) > 0.0
        aggregate_forces = np.asarray(self._data.force_matrix_w, dtype=float)[0, 0]
        aggregate_points = np.asarray(self._data.contact_pos_w, dtype=float)[0, 0]
        raw_forces = []
        raw_points = []
        for force, point in zip(aggregate_forces, aggregate_points):
            magnitude = float(np.linalg.norm(force))
            if magnitude > 0.0:
                raw_forces.append(np.asarray(force, dtype=float))
                raw_points.append(np.asarray(point, dtype=float))
            else:
                raw_forces.append(None)
                raw_points.append(None)
        vectors, points, counts, starts = self._layout(raw_forces, raw_points)
        magnitudes = np.full((self.max_contact_data_count, 1), np.nan, dtype=float)
        normals = np.full((self.max_contact_data_count, 3), np.nan, dtype=float)
        separations = np.full((self.max_contact_data_count, 1), np.nan, dtype=float)
        for index in range(int(np.sum(counts))):
            magnitude = float(np.linalg.norm(vectors[index]))
            magnitudes[index, 0] = magnitude
            normals[index] = vectors[index] / magnitude
            separations[index, 0] = 0.0
        return magnitudes, points, normals, separations, counts, starts

    def get_friction_data(self, *, dt):
        assert float(dt) > 0.0
        aggregate_forces = np.asarray(self._data.friction_forces_w, dtype=float)[0, 0]
        aggregate_points = np.asarray(self._data.contact_pos_w, dtype=float)[0, 0]
        raw_forces = []
        raw_points = []
        for force, point in zip(aggregate_forces, aggregate_points):
            if float(np.linalg.norm(force)) > 0.0:
                raw_forces.append(np.asarray(force, dtype=float))
                raw_points.append(np.asarray(point, dtype=float))
            else:
                raw_forces.append(None)
                raw_points.append(None)
        return self._layout(raw_forces, raw_points)


class _FakeContactSensor:
    def __init__(self, data, *, prim_path, filter_paths, event_log=None):
        self.data = data
        self._timestamp = np.asarray([0.0], dtype=float)
        self._timestamp_last_update = np.asarray([0.0], dtype=float)
        self.cfg = SimpleNamespace(
            prim_path=str(prim_path),
            filter_prim_paths_expr=list(filter_paths),
        )
        self.event_log = event_log
        self.contact_physx_view = _FakeRawContactView(data)

    def update(self, dt, force_recompute=False):
        if self.event_log is not None:
            self.event_log.append(
                ("child_contact_update", float(dt), bool(force_recompute))
            )
        self._timestamp += float(dt)
        self._timestamp_last_update[:] = self._timestamp

    def reset(self, _env_ids=None):
        self._timestamp[:] = 0.0
        self._timestamp_last_update[:] = 0.0


class _FakeCombinedContactBank(WheelAndNonWheelContactSensorBank):
    def __init__(self, *, event_log=None):
        points = {
            "FL": (0.80, -0.20, 0.0),
            "FR": (0.80, 0.20, 0.0),
            "RL": (0.40, -0.20, 0.0),
            "RR": (0.40, 0.20, 0.0),
        }
        wheel_sensors = {}
        for spec in wheel_contact_sensor_specs():
            normal = np.zeros((1, 1, 2, 3), dtype=float)
            normal[0, 0, 0, :] = (0.0, 0.0, 9.81)
            contact = np.full((1, 1, 2, 3), np.nan, dtype=float)
            contact[0, 0, 0, :] = points[spec.leg]
            friction = np.zeros((1, 1, 2, 3), dtype=float)
            wheel_sensors[spec.leg] = _FakeContactSensor(
                SimpleNamespace(
                    net_forces_w=np.asarray([[[0.0, 0.0, 9.81]]]),
                    force_matrix_w=normal,
                    contact_pos_w=contact,
                    friction_forces_w=friction,
                ),
                prim_path=spec.prim_path,
                filter_paths=[path for _name, path in FILTERED_SURFACES],
                event_log=event_log,
            )
        wheel_bank = FilteredWheelContactSensorBank(
            wheel_sensors,
            wheel_contact_sensor_specs(),
            force_threshold_n=1.0,
        )
        nonwheel_spec = NonWheelRigidBodySpec(
            body_name="base_link",
            prim_path="/World/WLRRobot/base_link",
        )
        nonwheel_sensor = _FakeContactSensor(
            SimpleNamespace(
                net_forces_w=np.zeros((1, 1, 3), dtype=float),
                force_matrix_w=np.zeros((1, 1, 1, 3), dtype=float),
                contact_pos_w=np.full((1, 1, 1, 3), np.nan, dtype=float),
                friction_forces_w=np.zeros((1, 1, 1, 3), dtype=float),
            ),
            prim_path=nonwheel_spec.prim_path,
            filter_paths=["/World/Obstacle"],
            event_log=event_log,
        )
        nonwheel_bank = NonWheelObstacleContactSensorBank(
            {nonwheel_spec.prim_path: nonwheel_sensor},
            (nonwheel_spec,),
            obstacle_prim_path="/World/Obstacle",
            force_threshold_n=1.0,
        )
        super().__init__(wheel_bank, nonwheel_bank)
        self.event_log = event_log

    def update(self, dt, force_recompute=False):
        if self.event_log is not None:
            self.event_log.append(
                ("combined_contact_update", float(dt), bool(force_recompute))
            )
        super().update(dt, force_recompute=force_recompute)


def _source_provenance(
    *,
    segment: int,
    step: int,
    source_time_s: float,
    events: tuple[int, ...],
    commands: tuple[str, ...],
    sequence: int,
    source_version: str = CANONICAL_GATE_C_SOURCE_VERSION,
    dispatch_kind: str = "segment_start",
):
    identity_payload = {
        "schema_version": "fsm50.source_action_identity.v1",
        "source_version": source_version,
        "source_segment_index": segment,
        "source_step_index": step,
        "source_time_s": source_time_s,
        "source_event_indices": list(events),
        "commands": list(commands),
        "dispatch_kind": dispatch_kind,
        "sequence_index": sequence,
    }
    identity = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "SOURCE_ACTION",
        "source_action_identity": identity,
        **{
            key: value
            for key, value in identity_payload.items()
            if key != "schema_version"
        },
        "recovery_stage": "",
        "recovery_action": "",
        "recovery_evidence_sha256": "",
        "recovery_centroidal_evidence_sha256": "",
        "recovery_feedback_observation_sha256": "",
        "recovery_target_map_sha256": "",
        "recovery_direction_sign": None,
        "recovery_attempt": None,
        "recovery_leg": "",
        "recovery_joint": "",
        "recovery_configuration_sha256": "",
    }


_SOURCE0_PROVENANCE = _source_provenance(
    segment=0,
    step=1,
    source_time_s=0.0,
    events=(0,),
    commands=("atomic source action",),
    sequence=0,
)


def _completion_spec(
    *,
    segment: int,
    step: int,
    servo_targets: dict[str, float] | None = None,
    servo_duration_s: float = 0.0,
    wheel_duration_s: float = 0.0,
    hold_s: float = 0.0,
):
    targets = dict(servo_targets or {})
    return {
        "segment_index": segment,
        "source_step": step,
        "source_step_id": f"test-step-{step}",
        "servo_targets_deg": targets,
        "servo_duration_s": servo_duration_s,
        "servo_tolerance_deg": 1.0,
        "recorded_servo_residual_deg": {
            name: 0.0 for name in targets
        },
        "legacy_missing_endpoint": False,
        "wheel_active_duration_s": wheel_duration_s,
        "explicit_hold_s": hold_s,
    }


def _segment_binding(
    *,
    segment: int,
    step: int,
    plan_sha: str,
    servo_targets: dict[str, float] | None = None,
    servo_duration_s: float = 0.0,
    wheel_duration_s: float = 0.0,
    hold_s: float = 0.0,
):
    spec = _completion_spec(
        segment=segment,
        step=step,
        servo_targets=servo_targets,
        servo_duration_s=servo_duration_s,
        wheel_duration_s=wheel_duration_s,
        hold_s=hold_s,
    )
    return {
        "schema_version": "fsm50.playback_segment_binding.v1",
        "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
        "source_plan_sha256": plan_sha,
        "source_plan_payload_sha256": "1" * 64,
        "accepted_steps_sha256": "2" * 64,
        "segment": {
            "segment_index": segment,
            "source_step": step,
            "source_step_id": spec["source_step_id"],
        },
        "events": [],
        "completion_spec": spec,
    }


class _Profiles:
    library_id = "profiles-v1"
    sha256 = "b" * 64
    plan_sha256 = "d" * 64

    @staticmethod
    def to_mapping():
        return {
            "segment_ownership": [
                {
                    "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
                    "state_id": "S1_APPROACH_AND_PRE_FR_SHIFT",
                    "phase_source_state_id": "S1_APPROACH_AND_PRE_FR_SHIFT",
                    "first_segment": 0,
                    "last_segment": 0,
                    "source_plan_sha256": _Profiles.plan_sha256,
                    "evidence_basis": "test canonical owner",
                }
            ],
            "profiles": [
                {
                    "profile_id": "profile-state",
                    "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
                    "state_id": "S1_APPROACH_AND_PRE_FR_SHIFT",
                    "strategy": "PRIMARY_PROFILE",
                    "source_plan_sha256": _Profiles.plan_sha256,
                    "source_segment_range": [0, 0],
                    "segment_bindings": [
                        _segment_binding(
                            segment=0,
                            step=1,
                            plan_sha=_Profiles.plan_sha256,
                            servo_targets={
                                SERVO_JOINT_NAMES[0]: 1.0
                            },
                        )
                    ],
                    "keyframes": [
                        {
                            "source_segment_index": 0,
                            "source_step_index": 1,
                            "source_time_s": 0.0,
                            "source_event_indices": [0],
                            "commands": ["atomic source action"],
                            "dispatch_kind": "segment_start",
                            "sequence_index": 0,
                            "servo_targets_deg": {
                                name: 1.0 for name in SERVO_JOINT_NAMES
                            },
                            "wheel_targets_rad_s": {
                                name: 0.25 for name in WHEEL_JOINT_NAMES
                            },
                        }
                    ]
                }
            ]
        }


class _Graph:
    graph_id = "graph-v1"
    sha256 = "a" * 64

    _NEXT_STATE = {
        "S0_INITIALIZE": "S1_APPROACH_AND_PRE_FR_SHIFT",
        "S1_APPROACH_AND_PRE_FR_SHIFT": "S2_FR_TRAVERSE",
        "S2_FR_TRAVERSE": "S3_FL_TRAVERSE",
        "S3_FL_TRAVERSE": "S4_FRONT_PAIR_ADVANCE",
        "S4_FRONT_PAIR_ADVANCE": "S5_PRE_RR_COM_SHIFT",
        "S5_PRE_RR_COM_SHIFT": "S6_RR_TRAVERSE",
        "S6_RR_TRAVERSE": "S7_PRE_RL_SUPPORT_SETUP",
        "S7_PRE_RL_SUPPORT_SETUP": "S8_RL_COM_SHIFT_AND_TRAVERSE",
        "S8_RL_COM_SHIFT_AND_TRAVERSE": "S9_FINAL_ADVANCE",
        "S9_FINAL_ADVANCE": "S10_POSTURE_RECOVERY",
        "S10_POSTURE_RECOVERY": "SUCCESS",
    }

    @staticmethod
    def get(state):
        text = str(state)
        active = (
            "FR"
            if text.startswith(("S1_", "S2_"))
            else "RL"
            if text.startswith(("S7_", "S8_"))
            else ""
        )
        return SimpleNamespace(
            active_leg=active,
            next_state=_Graph._NEXT_STATE.get(text, "SAFE_STOP"),
        )


class _Bundle:
    graph = _Graph()
    profiles = _Profiles()
    graph_sha256 = _Graph.sha256
    profile_library_sha256 = _Profiles.sha256
    bundle_sha256 = "c" * 64

    @staticmethod
    def to_mapping():
        return {"bundle_sha256": _Bundle.bundle_sha256}


def _decision(
    *,
    state="S1_APPROACH_AND_PRE_FR_SHIFT",
    epoch=0,
    changed=False,
    servos=None,
    wheels=None,
    events=(),
    terminal=False,
    outcome="RUNNING",
    provenance=None,
    source_action_consumed=None,
    target_changed=None,
    profile_id="profile-state",
    profile_source_version=CANONICAL_GATE_C_SOURCE_VERSION,
    profile_strategy="PRIMARY_PROFILE",
    completion_control=None,
    subphase="PRELOAD",
):
    command_provenance = dict(
        provenance
        if provenance is not None
        else {
            "kind": "NONE",
            "source_action_identity": "",
            "source_version": "",
            "source_segment_index": None,
            "source_step_index": None,
            "source_time_s": None,
            "source_event_indices": [],
            "commands": [],
            "dispatch_kind": "",
            "sequence_index": None,
            "recovery_stage": "",
            "recovery_action": "",
            "recovery_evidence_sha256": "",
            "recovery_centroidal_evidence_sha256": "",
            "recovery_feedback_observation_sha256": "",
            "recovery_target_map_sha256": "",
            "recovery_direction_sign": None,
            "recovery_attempt": None,
            "recovery_leg": "",
            "recovery_joint": "",
            "recovery_configuration_sha256": "",
        }
    )
    consumed = (
        command_provenance["kind"] == "SOURCE_ACTION"
        if source_action_consumed is None
        else source_action_consumed
    )
    return {
        "macro_state": state,
        "subphase": subphase,
        "profile_id": profile_id,
        "profile_source_version": profile_source_version,
        "profile_strategy": profile_strategy,
        "phase_elapsed_s": 0.0,
        "profile_fraction": 0.0,
        "servo_targets_deg": dict(
            servos if servos is not None else {name: 0.0 for name in SERVO_JOINT_NAMES}
        ),
        "wheel_targets_rad_s": dict(
            wheels if wheels is not None else {name: 0.0 for name in WHEEL_JOINT_NAMES}
        ),
        "command_epoch": epoch,
        "command_changed": changed,
        "source_action_consumed": consumed,
        "target_changed": changed if target_changed is None else target_changed,
        "command_provenance": command_provenance,
        "segment_completion_control": dict(
            completion_control
            if completion_control is not None
            else _empty_completion_control()
        ),
        "transition_events": list(events),
        "reason": outcome,
        "retry_count": 0,
        "terminal": terminal,
        "terminal_outcome": outcome,
    }


def _empty_completion_control():
    return {
        "schema_version": "fsm50.macro_segment_completion_control.v1",
        "kind": "NONE",
        "profile_id": "",
        "profile_source_version": "",
        "owner_state": "",
        "source_plan_sha256": "",
        "source_plan_payload_sha256": "",
        "accepted_steps_sha256": "",
        "source_segment_index": None,
        "source_step_index": None,
        "source_step_id": "",
        "start_command_epoch": None,
        "completion_spec": {},
        "source_action_identity": "",
        "source_action": False,
        "completion_token_sha256": "",
    }


def _start_completion_control(
    *,
    provenance,
    spec,
    epoch,
    profile_id="profile-state",
    plan_sha=_Profiles.plan_sha256,
    owner_state="S1_APPROACH_AND_PRE_FR_SHIFT",
):
    return {
        "schema_version": "fsm50.macro_segment_completion_control.v1",
        "kind": "START",
        "profile_id": profile_id,
        "profile_source_version": CANONICAL_GATE_C_SOURCE_VERSION,
        "owner_state": owner_state,
        "source_plan_sha256": plan_sha,
        "source_plan_payload_sha256": "1" * 64,
        "accepted_steps_sha256": "2" * 64,
        "source_segment_index": provenance["source_segment_index"],
        "source_step_index": provenance["source_step_index"],
        "source_step_id": spec["source_step_id"],
        "start_command_epoch": epoch,
        "completion_spec": dict(spec),
        "source_action_identity": provenance["source_action_identity"],
        "source_action": True,
        "completion_token_sha256": "",
    }


class _Controller:
    def __init__(self, *, terminal_tick=119):
        self.reset_calls = []
        self.tick_calls = []
        self.terminal_tick = terminal_tick
        self.timeline = []
        self.servos = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self.wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self.entered_profile_state = False
        self.segment_started = False

    def reset(self, observation, *, sim_time_s, profile_id, source_version):
        self.reset_calls.append((observation, sim_time_s, profile_id, source_version))
        return _decision(
            state="S0_INITIALIZE",
            events=("RESET:S0_INITIALIZE",),
            servos=self.servos,
            wheels=self.wheels,
        )

    def tick(
        self,
        observation,
        *,
        sim_time_s,
        segment_completion_token=None,
        source_cursor_permit=False,
    ):
        self.tick_calls.append(
            (
                observation,
                sim_time_s,
                segment_completion_token,
                source_cursor_permit,
            )
        )
        if not self.entered_profile_state:
            self.entered_profile_state = True
            return _decision(
                events=("EXIT:S0_INITIALIZE", "ENTER:S1_APPROACH_AND_PRE_FR_SHIFT"),
                servos=self.servos,
                wheels=self.wheels,
            )
        if not self.segment_started and source_cursor_permit:
            self.segment_started = True
            self.servos = {name: 1.0 for name in SERVO_JOINT_NAMES}
            self.wheels = {name: 0.25 for name in WHEEL_JOINT_NAMES}
            spec = _completion_spec(
                segment=0,
                step=1,
                servo_targets={SERVO_JOINT_NAMES[0]: 1.0},
            )
            return _decision(
                epoch=1,
                changed=True,
                servos=self.servos,
                wheels=self.wheels,
                provenance=_SOURCE0_PROVENANCE,
                completion_control=_start_completion_control(
                    provenance=_SOURCE0_PROVENANCE,
                    spec=spec,
                    epoch=1,
                ),
            )
        terminal = bool(
            segment_completion_token is not None
            and str(getattr(segment_completion_token, "kind", "")) == "COMPLETE"
        )
        return _decision(
            epoch=1 if self.segment_started else 0,
            changed=False,
            servos=self.servos,
            wheels=self.wheels,
            terminal=terminal,
            outcome=(
                "TASK_SUCCESS_POSTURE_INCOMPLETE"
                if terminal
                else "RUNNING"
            ),
        )


class _ControllerException(_Controller):
    def reset(self, observation, *, sim_time_s, profile_id, source_version):
        raise RuntimeError("injected controller exception")


def _request_payload(root: Path) -> dict:
    alignment = root / "alignment.csv"
    table = root / "success.csv"
    alignment.write_text("a,b\n1,2\n", encoding="utf-8")
    table.write_text(
        "version,evaluation_status,task_result,notes\n"
        f'{CANONICAL_GATE_C_SOURCE_VERSION},EVALUATED,'
        'REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE,'
        '"No fall, dangerous body collision, or severe penetration was visible."\n',
        encoding="utf-8",
    )
    return {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": "request-v003",
        "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
        "profile_id": _Profiles.library_id,
        "graph_id": _Graph.graph_id,
        "graph_sha256": _Graph.sha256,
        "profile_library_sha256": _Profiles.sha256,
        "bundle_sha256": _Bundle.bundle_sha256,
        "height_mm": 50,
        "run_dir": str((root / "run").resolve()),
        "alignment_path": str(alignment.resolve()),
        "alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
        "task_success_table_path": str(table.resolve()),
        "task_success_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "trial_kind": "baseline",
        "trial_index": 0,
        "telemetry_hz": 15.0,
        "video_fps": 15.0,
        "capture_video": True,
        "post_run_settle_s": 1.0 / 120.0,
        "timeout_s": 10.0,
        "filtered_contact_bank_enabled": True,
    }


def _load_request(root: Path):
    payload = _request_payload(root)
    path = root / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    request = load_worker_macro_fsm_request(path)
    assert request is not None
    return request, payload


def _bind_fake_contact_scene(session, scene, request, *, event_log=None):
    scene.config.telemetry_contact_sensors_enabled = False
    scene.config.contact_sensor_factory = None
    configure_scene_for_macro_fsm(scene.config, request)
    scene.contact_sensor = _FakeCombinedContactBank(event_log=event_log)
    scene.contact_sensor_error = ""
    session.bind_filtered_contact_bank_scene(scene)


def _runtime(
    request,
    controller=None,
    bundle=None,
    *,
    residual_policy=None,
    residual_contract_provider=None,
    contact_event_log=None,
):
    adapter = _Adapter()
    adapter.event_log = contact_event_log
    scene = SimpleNamespace(
        config=SimpleNamespace(
            obstacle_front_x=1.0,
            obstacle_height_m=0.05,
            obstacle_length=1.0,
            obstacle_width=2.0,
            ground_z_m=0.0,
            ground_static_friction=1.25,
            ground_dynamic_friction=1.05,
            obstacle_static_friction=1.20,
            obstacle_dynamic_friction=1.00,
        )
    )
    chosen = controller or _Controller()
    chosen_bundle = bundle or _Bundle()
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session",
        bundle_builder=lambda _root, _request: chosen_bundle,
        controller_factory=lambda _bundle: chosen,
        observation_factory=lambda payload: dict(payload),
        recorder_factory=_Recorder,
        residual_policy=residual_policy,
        residual_contract_provider=residual_contract_provider,
    )
    _bind_fake_contact_scene(
        session,
        scene,
        request,
        event_log=contact_event_log,
    )
    session.prepare_after_adapter(
        adapter=adapter,
        scene_handle=scene,
        project_root=Path.cwd(),
    )
    return session, adapter, chosen


def _on_step(session, adapter, dt=1.0 / 120.0):
    if session.outer_render_substeps_remaining == 0:
        completed_step = adapter.sim_steps
        adapter.sim_steps = completed_step - 1
        try:
            session.before_adapter_step()
        finally:
            adapter.sim_steps = completed_step
    session.on_step(adapter, dt)


def _prime_current_target_readback(session, adapter):
    servos = dict(adapter.joint_command_deg)
    wheels = dict(adapter.wheel_speeds)
    session.last_target_readback = session._capture_and_validate_target_readback(
        adapter,
        servo_targets=servos,
        wheel_targets=wheels,
        expected_sim_step=adapter.sim_steps,
    )
    session.last_verified_servo_targets = servos
    session.last_verified_wheel_targets = wheels
    observed_epoch = int(session.last_epoch if session.last_epoch is not None else 0)
    session.last_epoch = observed_epoch
    session.last_verified_command_epoch = observed_epoch


def _advance_until_terminal(session, adapter, *, first_step=1, last_step=128):
    for step in range(first_step, last_step + 1):
        adapter.sim_steps = step
        adapter.sim_time = step / 120.0
        _on_step(session, adapter, 1.0 / 120.0)
        terminal = session.after_adapter_step()
        if terminal is not None:
            return terminal, step
    raise AssertionError("Macro session did not reach a terminal result")


def _observe_completion_boundary(session, adapter, *, step):
    adapter.sim_steps = step
    adapter.sim_time = step / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        token = session._observe_active_segment_completion(
            adapter=adapter,
            payload=session._observation_payload(adapter),
        )
    finally:
        session.outer_render_boundary_permit = False
    session.last_segment_completion_token = token
    return token


def _non_source_provenance(kind="NONE"):
    return {
        "kind": kind,
        "source_action_identity": "",
        "source_version": "",
        "source_segment_index": None,
        "source_step_index": None,
        "source_time_s": None,
        "source_event_indices": [],
        "commands": [],
        "dispatch_kind": "",
        "sequence_index": None,
        "recovery_stage": "",
        "recovery_action": "",
        "recovery_evidence_sha256": "",
        "recovery_centroidal_evidence_sha256": "",
        "recovery_feedback_observation_sha256": "",
        "recovery_target_map_sha256": "",
        "recovery_direction_sign": None,
        "recovery_attempt": None,
        "recovery_leg": "",
        "recovery_joint": "",
        "recovery_configuration_sha256": "",
    }


def _residual_action(**values):
    unknown = set(values) - set(RESIDUAL_ACTION_NAMES)
    assert not unknown
    return tuple(float(values.get(name, 0.0)) for name in RESIDUAL_ACTION_NAMES)


class _ResidualPolicy:
    policy_id = "test.worker_direct_command_residual.v1"

    def __init__(self, action=ZERO_RESIDUAL_ACTION, *, actions_by_state=None):
        self.action = tuple(action)
        self.actions_by_state = dict(actions_by_state or {})
        self.observations = []

    @property
    def policy_sha256(self):
        return canonical_mapping_sha256(
            {
                "schema_version": "test.worker_direct_command_residual_policy.v1",
                "policy_id": self.policy_id,
            }
        )

    def act(self, observation):
        self.observations.append(dict(observation))
        return tuple(
            self.actions_by_state.get(
                observation["macro_state"], self.action
            )
        )


class _ResidualContractProvider:
    def __init__(
        self,
        *,
        empty_profile_strategy=None,
        forced_profile_strategy=None,
    ):
        assert empty_profile_strategy is None or forced_profile_strategy is None
        self.contexts = []
        self.empty_profile_strategy = empty_profile_strategy
        self.forced_profile_strategy = forced_profile_strategy

    def __call__(self, context):
        self.contexts.append(dict(context))
        servo_count = len(SERVO_JOINT_NAMES)
        profile_strategy = context["profile_strategy"]
        if self.forced_profile_strategy is not None:
            profile_strategy = self.forced_profile_strategy
        elif not profile_strategy and self.empty_profile_strategy is not None:
            profile_strategy = self.empty_profile_strategy
        return ResidualPhaseContract(
            source_version=context["source_version"],
            profile_strategy=profile_strategy,
            macro_state=context["macro_state"],
            subphase=context["subphase"],
            enabled_mask=(True,) * len(RESIDUAL_ACTION_NAMES),
            residual_min_command_units=tuple(
                -2.0 if index < servo_count else -0.2
                for index in range(len(RESIDUAL_ACTION_NAMES))
            ),
            residual_max_command_units=tuple(
                2.0 if index < servo_count else 0.2
                for index in range(len(RESIDUAL_ACTION_NAMES))
            ),
            maximum_rate_command_units_per_s=(100.0,)
            * len(RESIDUAL_ACTION_NAMES),
        )


def _bundle_for_actions(actions):
    plan_sha = "e" * 64
    keyframes = []
    bindings = []
    bound_segments = set()
    for action in actions:
        provenance = dict(action["provenance"])
        keyframes.append(
            {
                "source_segment_index": provenance["source_segment_index"],
                "source_step_index": provenance["source_step_index"],
                "source_time_s": provenance["source_time_s"],
                "source_event_indices": list(provenance["source_event_indices"]),
                "commands": list(provenance["commands"]),
                "dispatch_kind": provenance["dispatch_kind"],
                "sequence_index": provenance["sequence_index"],
                "servo_targets_deg": dict(action["servos"]),
                "wheel_targets_rad_s": dict(action["wheels"]),
            }
        )
        segment_index = provenance["source_segment_index"]
        if segment_index not in bound_segments:
            bound_segments.add(segment_index)
            sparse_targets = dict(
                action.get("completion_servo_targets", action["servos"])
            )
            bindings.append(
                _segment_binding(
                    segment=provenance["source_segment_index"],
                    step=provenance["source_step_index"],
                    plan_sha=plan_sha,
                    servo_targets=sparse_targets,
                    servo_duration_s=float(action.get("servo_duration_s", 0.0)),
                    wheel_duration_s=float(action.get("wheel_duration_s", 0.0)),
                    hold_s=float(action.get("hold_s", 0.0)),
                )
            )
    last = max(
        int(action["provenance"]["source_segment_index"])
        for action in actions
    )
    mapping = {
        "segment_ownership": [
            {
                "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
                "state_id": "S1_APPROACH_AND_PRE_FR_SHIFT",
                "phase_source_state_id": "S1_APPROACH_AND_PRE_FR_SHIFT",
                "first_segment": 0,
                "last_segment": last,
                "source_plan_sha256": plan_sha,
                "evidence_basis": "test ordered source actions",
            }
        ],
        "profiles": [
            {
                "profile_id": "profile-state",
                "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
                "state_id": "S1_APPROACH_AND_PRE_FR_SHIFT",
                "strategy": "PRIMARY_PROFILE",
                "source_plan_sha256": plan_sha,
                "source_segment_range": [0, last],
                "segment_bindings": bindings,
                "keyframes": keyframes,
            }
        ],
    }
    profiles = SimpleNamespace(
        library_id=_Profiles.library_id,
        sha256=_Profiles.sha256,
        to_mapping=lambda: mapping,
    )
    return SimpleNamespace(
        graph=_Graph(),
        profiles=profiles,
        graph_sha256=_Graph.sha256,
        profile_library_sha256=_Profiles.sha256,
        bundle_sha256=_Bundle.bundle_sha256,
        to_mapping=lambda: {"bundle_sha256": _Bundle.bundle_sha256},
    )


def _prime_manual_session(
    request,
    *,
    bundle,
    residual_policy=None,
    residual_contract_provider=None,
):
    session, adapter, _controller = _runtime(
        request,
        bundle=bundle,
        residual_policy=residual_policy,
        residual_contract_provider=residual_contract_provider,
    )
    session.start()
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.last_epoch == 0
    assert session.last_verified_command_epoch == 0
    first = session.expected_source_actions[0]
    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session._process_decision(
        adapter,
        session._observation_payload(adapter),
        _decision(
            state=first["owner_state"],
            events=("EXIT:S0_INITIALIZE", f"ENTER:{first['owner_state']}"),
            profile_id=first["profile_id"],
            profile_source_version=first["profile_source_version"],
            profile_strategy=first["profile_strategy"],
        ),
    )
    return session, adapter


def _process_expected_source_action(session, adapter, expected, *, step, epoch):
    adapter.sim_steps = step
    adapter.sim_time = step / 120.0
    _prime_current_target_readback(session, adapter)
    provenance = expected["command_provenance"]
    if provenance["dispatch_kind"] == "segment_start":
        binding = expected["segment_completion_binding"]
        control = _start_completion_control(
            provenance=provenance,
            spec=binding["completion_spec"],
            epoch=epoch,
            profile_id=expected["profile_id"],
            plan_sha=expected["source_plan_sha256"],
            owner_state=expected["owner_state"],
        )
        control["source_plan_payload_sha256"] = binding[
            "source_plan_payload_sha256"
        ]
        control["accepted_steps_sha256"] = binding["accepted_steps_sha256"]
    else:
        row = session._active_completion_row()
        token = session.last_segment_completion_token
        assert token is not None and token.kind == "WHEEL_STOP_DUE"
        control = {
            "schema_version": "fsm50.macro_segment_completion_control.v1",
            "kind": "WHEEL_STOP",
            "profile_id": row["profile_id"],
            "profile_source_version": row["profile_source_version"],
            "owner_state": row["owner_state"],
            "source_plan_sha256": row["source_plan_sha256"],
            "source_plan_payload_sha256": row[
                "source_plan_payload_sha256"
            ],
            "accepted_steps_sha256": row["accepted_steps_sha256"],
            "source_segment_index": row["source_segment_index"],
            "source_step_index": row["source_step_index"],
            "source_step_id": row["source_step_id"],
            "start_command_epoch": row["start_command_epoch"],
            "completion_spec": row["completion_spec"],
            "source_action_identity": provenance["source_action_identity"],
            "source_action": True,
            "completion_token_sha256": token.sha256,
        }
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state=expected["owner_state"],
                epoch=epoch,
                changed=(
                    expected["servo_targets_deg"] != session.last_servo_targets
                    or expected["wheel_targets_rad_s"] != session.last_wheel_targets
                ),
                servos=expected["servo_targets_deg"],
                wheels=expected["wheel_targets_rad_s"],
                provenance=provenance,
                profile_id=expected["profile_id"],
                profile_source_version=expected["profile_source_version"],
                profile_strategy=expected["profile_strategy"],
                completion_control=control,
            ),
        )
    finally:
        session.outer_render_boundary_permit = False


def _coalesced_transition_case(
    tmp_path: Path,
    *,
    same_target: bool,
    residual_policy=None,
    residual_contract_provider=None,
):
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    next_s = {
        name: (0.0 if same_target else 1.0)
        for name in SERVO_JOINT_NAMES
    }
    actions = [
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.0,
                events=(0,),
                commands=("old final zero target",),
                sequence=0,
            ),
            "servos": zeros_s,
            "wheels": zeros_w,
        },
        {
            "provenance": _source_provenance(
                segment=1,
                step=2,
                source_time_s=1.0,
                events=(1,),
                commands=("next state first target",),
                sequence=1,
            ),
            "servos": next_s,
            "wheels": zeros_w,
        },
    ]
    request, _payload = _load_request(tmp_path)
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions(actions),
        residual_policy=residual_policy,
        residual_contract_provider=residual_contract_provider,
    )
    old_action = session.expected_source_actions[0]
    _process_expected_source_action(
        session, adapter, old_action, step=3, epoch=0
    )
    if session.pending_readback is not None:
        expected_readback_step = int(
            session.pending_readback["expected_sim_step"]
        )
        adapter.sim_steps = expected_readback_step
        adapter.sim_time = expected_readback_step / 120.0
        session._verify_pending_readback(
            adapter, sim_step=expected_readback_step
        )
    assert session.pending_readback is None
    assert session.active_segment_completion_row_index == 0

    step = 8
    adapter.sim_steps = step
    adapter.sim_time = step / 120.0
    session.last_target_readback = session._capture_and_validate_target_readback(
        adapter,
        servo_targets=session.last_applied_servo_targets,
        wheel_targets=session.last_applied_wheel_targets,
        expected_sim_step=step,
    )
    if session.residual_enabled:
        for index, name in enumerate(SERVO_JOINT_NAMES):
            adapter.robot.data.joint_pos[0, index] = math.radians(
                float(session.last_applied_servo_targets[name])
            )
    token = _observe_completion_boundary(session, adapter, step=step)
    assert token is not None and token.kind == "COMPLETE"
    assert token.owner_state == "S1_APPROACH_AND_PRE_FR_SHIFT"
    assert session.active_segment_completion_row_index is None
    assert session.segment_completion_rows[-1]["tracking_lifecycle_closed"] is True

    next_action = session.expected_source_actions[1]
    next_action["owner_state"] = "S2_FR_TRAVERSE"
    provenance = next_action["command_provenance"]
    binding = next_action["segment_completion_binding"]
    epoch = 0 if same_target else 1
    control = _start_completion_control(
        provenance=provenance,
        spec=binding["completion_spec"],
        epoch=epoch,
        profile_id=next_action["profile_id"],
        plan_sha=next_action["source_plan_sha256"],
        owner_state=next_action["owner_state"],
    )
    control["source_plan_payload_sha256"] = binding[
        "source_plan_payload_sha256"
    ]
    control["accepted_steps_sha256"] = binding["accepted_steps_sha256"]
    decision = _decision(
        state="S2_FR_TRAVERSE",
        epoch=epoch,
        changed=not same_target,
        servos=next_action["servo_targets_deg"],
        wheels=next_action["wheel_targets_rad_s"],
        events=(
            "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
            "ENTER:S2_FR_TRAVERSE",
        ),
        provenance=provenance,
        profile_id=next_action["profile_id"],
        profile_source_version=next_action["profile_source_version"],
        profile_strategy=next_action["profile_strategy"],
        completion_control=control,
    )
    return session, adapter, decision, step


def test_combined_contact_refresh_is_atomic_once_per_tick_and_tracks_dwell(
    tmp_path: Path,
):
    legs = ("FL", "FR", "RL", "RR")
    request, _payload = _load_request(tmp_path)
    events: list[tuple] = []
    session, adapter, _controller = _runtime(
        request,
        contact_event_log=events,
    )

    def capture_runtime_safety(*, scene_handle):
        assert scene_handle is session.scene_handle
        assert session.filtered_contact_last_sim_step == adapter.sim_steps
        events.append(("capture_macro_runtime_safety_evidence", adapter.sim_steps))
        return {
            "available": False,
            "dangerous_body_collision": None,
            "severe_penetration": None,
            "source": "TEST_UNAVAILABLE_RUNTIME_SAFETY",
            "sample_sim_step": adapter.sim_steps,
            "error": "test runtime safety is unavailable",
        }

    adapter.capture_macro_runtime_safety_evidence = capture_runtime_safety

    events.clear()
    session.start()
    assert [row for row in events if row[0] == "combined_contact_update"] == [
        ("combined_contact_update", 0.0, True)
    ]
    assert events.index(("combined_contact_update", 0.0, True)) < events.index(
        ("capture_macro_runtime_safety_evidence", 0)
    )
    first = dict(session.filtered_contact_sample)
    assert first["sample_epoch"] == 1
    assert first["sample_sim_step"] == 0
    assert first["surface_kind_by_leg"] == {leg: "GROUND" for leg in legs}
    assert first["surface_dwell_lower_bound_s_by_leg"] == {
        leg: 0.0 for leg in legs
    }
    assert session.filtered_contact_frame.available is False

    # Multiple payload consumers at one completed physics tick reuse the
    # immutable published sample and never update either child bank again.
    session._observation_payload(adapter)
    session._observation_payload(adapter)
    assert len([row for row in events if row[0] == "combined_contact_update"]) == 1

    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session._refresh_filtered_contact_evidence(adapter)
    session._observation_payload(adapter)
    session._observation_payload(adapter)
    updates = [row for row in events if row[0] == "combined_contact_update"]
    assert updates == [
        ("combined_contact_update", 0.0, True),
        ("combined_contact_update", 1.0 / 120.0, True),
    ]
    assert events.index(
        ("combined_contact_update", 1.0 / 120.0, True)
    ) < events.index(("capture_macro_runtime_safety_evidence", 1))
    second = session.filtered_contact_sample
    assert second["sample_epoch"] == 2
    assert second["sample_sim_step"] == 1
    assert second["surface_dwell_lower_bound_s_by_leg"] == {
        leg: 1.0 / 120.0 for leg in legs
    }
    assert session.filtered_contact_frame.available is True
    assert session.filtered_contact_com.available is True
    assert session.last_centroidal_support_evidence.sim_step == 1


def test_combined_contact_refresh_rejects_rebound_bank_before_update(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    events: list[tuple] = []
    session, _adapter, _controller = _runtime(
        request,
        contact_event_log=events,
    )
    events.clear()
    replacement_events: list[tuple] = []
    session.scene_handle.contact_sensor = _FakeCombinedContactBank(
        event_log=replacement_events
    )
    with pytest.raises(RuntimeError, match="cannot be updated"):
        session._refresh_filtered_contact_evidence(_adapter)
    assert replacement_events == []
    assert events == []
    assert session.filtered_contact_sample_epoch == 0
    assert session.filtered_contact_sample == {}


@pytest.mark.parametrize(
    "tamper",
    ("extra_wheel", "aliased_child", "nonwheel_prim", "nonwheel_filter"),
)
def test_combined_contact_bind_requires_exact_unique_child_mapping(
    tmp_path: Path,
    tamper: str,
):
    request, _payload = _load_request(tmp_path)
    bound, _adapter, _controller = _runtime(request)
    scene = bound.scene_handle
    bank = _FakeCombinedContactBank()
    if tamper == "extra_wheel":
        bank.wheel_bank.sensors["EXTRA"] = next(
            iter(bank.wheel_bank.sensors.values())
        )
    else:
        nonwheel_key = next(iter(bank.nonwheel_bank.sensors))
        if tamper == "aliased_child":
            bank.nonwheel_bank.sensors[nonwheel_key] = bank.wheel_bank.sensors["FL"]
        elif tamper == "nonwheel_prim":
            bank.nonwheel_bank.sensors[nonwheel_key].cfg.prim_path = (
                "/World/WLRRobot/wrong_body"
            )
        else:
            bank.nonwheel_bank.sensors[
                nonwheel_key
            ].cfg.filter_prim_paths_expr = ["/World/defaultGroundPlane"]
    scene.contact_sensor = bank
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session-exact-children",
        bundle_builder=lambda _root, _request: _Bundle(),
        controller_factory=lambda _bundle: _Controller(),
        observation_factory=lambda payload: dict(payload),
        recorder_factory=_Recorder,
    )
    with pytest.raises(RuntimeError, match="child"):
        session.bind_filtered_contact_bank_scene(scene)


@pytest.mark.parametrize("tamper", ("spec_object", "child_prim", "child_filter"))
def test_combined_contact_refresh_rejects_nonwheel_contract_drift_before_update(
    tmp_path: Path,
    tamper: str,
):
    request, _payload = _load_request(tmp_path)
    events: list[tuple] = []
    session, adapter, _controller = _runtime(
        request,
        contact_event_log=events,
    )
    bank = session.filtered_contact_nonwheel_bank
    if tamper == "spec_object":
        original = bank.specs[0]
        bank.specs = (
            NonWheelRigidBodySpec(
                body_name=original.body_name,
                prim_path=original.prim_path,
            ),
        )
    else:
        child = next(iter(bank.sensors.values()))
        if tamper == "child_prim":
            child.cfg.prim_path = "/World/WLRRobot/wrong_body"
        else:
            child.cfg.filter_prim_paths_expr = ["/World/defaultGroundPlane"]
    events.clear()
    with pytest.raises(RuntimeError, match="cannot be updated"):
        session._refresh_filtered_contact_evidence(adapter)
    assert events == []
    assert session.filtered_contact_sample_epoch == 0


def test_combined_contact_clock_reset_cannot_inherit_pair_dwell(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session._refresh_filtered_contact_evidence(adapter)
    assert session.filtered_contact_dwell_s_by_leg["FL"] == 1.0 / 120.0
    for child_bank in (
        session.filtered_contact_wheel_bank,
        session.filtered_contact_nonwheel_bank,
    ):
        for sensor in child_bank.sensors.values():
            sensor.reset()
    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="clock regressed"):
        session._refresh_filtered_contact_evidence(adapter)
    assert session.filtered_contact_sample_epoch == 2
    assert session.filtered_contact_last_sim_step == 1
    assert session.filtered_contact_dwell_s_by_leg["FL"] == 1.0 / 120.0


def test_surface_pair_changes_and_tick_gaps_reset_dwell(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    sensor = session.filtered_contact_wheel_bank.sensors["FL"]

    def set_filters(ground_force, obstacle_force, *, obstacle_point):
        sensor.data.force_matrix_w[0, 0, 0, :] = ground_force
        sensor.data.force_matrix_w[0, 0, 1, :] = obstacle_force
        sensor.data.net_forces_w[0, 0, :] = np.asarray(
            ground_force, dtype=float
        ) + np.asarray(obstacle_force, dtype=float)
        sensor.data.contact_pos_w[0, 0, 0, :] = (
            (0.80, -0.20, 0.0)
            if np.linalg.norm(ground_force) > 0.0
            else (np.nan, np.nan, np.nan)
        )
        sensor.data.contact_pos_w[0, 0, 1, :] = (
            obstacle_point
            if np.linalg.norm(obstacle_force) > 0.0
            else (np.nan, np.nan, np.nan)
        )

    def refresh(step):
        adapter.sim_steps = step
        adapter.sim_time = step / 120.0
        session._refresh_filtered_contact_evidence(adapter)
        return (
            session.filtered_contact_surface_kind_by_leg["FL"],
            session.filtered_contact_dwell_s_by_leg["FL"],
        )

    set_filters((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), obstacle_point=(1.0, -0.2, 0.025))
    assert refresh(1) == ("FRONT_FACE", 0.0)
    set_filters((0.0, 0.0, 0.0), (0.0, 0.0, 9.81), obstacle_point=(1.1, -0.2, 0.05))
    assert refresh(2) == ("OBSTACLE_TOP", 0.0)
    assert refresh(3) == ("OBSTACLE_TOP", 1.0 / 120.0)
    set_filters((0.0, 0.0, 9.81), (0.0, 0.0, 0.0), obstacle_point=(np.nan, np.nan, np.nan))
    assert refresh(4) == ("GROUND", 0.0)
    set_filters((0.0, 0.0, 9.81), (0.0, 0.0, 9.81), obstacle_point=(1.1, -0.2, 0.05))
    assert refresh(5) == ("UNKNOWN", 0.0)
    set_filters((0.0, 0.0, 9.81), (0.0, 0.0, 0.0), obstacle_point=(np.nan, np.nan, np.nan))
    assert refresh(7) == ("GROUND", 0.0)


def test_worker_uses_mass_weighted_whole_body_com_without_root_proxy(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    masses = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    positions = np.asarray(
        [[[0.0, 0.0, 0.1], [1.0, 0.0, 0.2], [2.0, 0.0, 0.3], [4.0, 0.0, 0.4]]],
        dtype=float,
    )
    adapter.robot.root_physx_view.get_masses = lambda: masses
    adapter.robot.data.body_com_pos_w = positions
    adapter.robot.data.body_com_pose_w[0, :, :3] = positions[0]
    adapter.robot.data.root_pose_w[0, :3] = (99.0, 98.0, 97.0)
    session._refresh_filtered_contact_evidence(adapter)
    expected = tuple(
        float(value)
        for value in np.sum(positions[0] * masses[0, :, None], axis=0)
        / np.sum(masses)
    )
    assert session.filtered_contact_com.available is True
    assert session.filtered_contact_com.position_w_m == expected
    assert session.filtered_contact_com.position_w_m != (99.0, 98.0, 97.0)
    evidence = session._build_centroidal_support_evidence(adapter, active_leg="")
    assert evidence.whole_body_com.position_w_m == expected
    assert evidence.sim_step == adapter.sim_steps
    assert evidence.wheel_contacts.physics_tick == adapter.sim_steps


def test_feedback_joint_state_uses_canonical_rear_command_sign_and_zero_offset():
    adapter = _Adapter()
    offsets = {
        name: float(index + 1) for index, name in enumerate(SERVO_JOINT_NAMES)
    }
    adapter.command_to_actual_target_deg = (
        lambda name, value: offsets[name]
        + float(JOINT_COMMAND_SIGN[name]) * float(value)
    )
    joint_q = {
        name: math.radians(offsets[name] + 10.0)
        for name in SERVO_JOINT_NAMES
    }
    joint_qd = {
        name: math.radians(2.0) for name in SERVO_JOINT_NAMES
    }
    positions, velocities, margins = WorkerMacroFSMSession._command_space_servo_state(
        adapter,
        joint_q,
        joint_qd,
    )
    for name in SERVO_JOINT_NAMES:
        sign = float(JOINT_COMMAND_SIGN[name])
        assert positions[name] == pytest.approx(sign * 10.0)
        assert velocities[name] == pytest.approx(sign * 2.0)
        assert margins[name] > 0.0


def test_feedback_probe_matrix_is_complete_and_chooses_only_measured_best_response():
    pairs = (
        ("front_right_hip", 1),
        ("front_right_hip", -1),
        ("front_right_knee", 1),
        ("front_right_knee", -1),
    )

    def result(joint, sign, *, dz, sign_valid=True, baseline=True):
        return {
            "joint": joint,
            "direction_sign": sign,
            "dq_deg": 0.25 * sign if sign_valid else 0.0,
            "dx_m": 0.0,
            "dz_m": dz,
            "sign_response_valid": sign_valid,
            "baseline_preserved": baseline,
            "unsafe_reasons": [],
            "n_plus_one_response_verified": True,
        }

    complete = {
        "completed_probe_pairs": list(pairs),
        "probe_results": [
            result(*pairs[0], dz=-0.0005),
            result(*pairs[1], dz=-0.0010),
            result(*pairs[2], dz=-0.0007),
            result(*pairs[3], dz=-0.0004),
        ],
    }
    assert WorkerMacroFSMSession._feedback_probe_matrix_choice(
        sequence=complete,
        safe_pairs=pairs,
    ) == pairs[1]

    missing = {
        "completed_probe_pairs": list(pairs[:-1]),
        "probe_results": complete["probe_results"][:-1],
    }
    with pytest.raises(RuntimeError, match="complete independently verified"):
        WorkerMacroFSMSession._feedback_probe_matrix_choice(
            sequence=missing,
            safe_pairs=pairs,
        )

    no_motion_best = {
        **complete,
        "probe_results": [
            result(*pairs[0], dz=-0.0020, sign_valid=False),
            *complete["probe_results"][1:],
        ],
    }
    assert WorkerMacroFSMSession._feedback_probe_matrix_choice(
        sequence=no_motion_best,
        safe_pairs=pairs,
    ) == pairs[1]

    no_descent = {
        **complete,
        "probe_results": [
            result(joint, sign, dz=0.0)
            for joint, sign in pairs
        ],
    }
    with pytest.raises(RuntimeError, match="no independently measured safe descent"):
        WorkerMacroFSMSession._feedback_probe_matrix_choice(
            sequence=no_descent,
            safe_pairs=pairs,
        )


def test_feedback_command_map_comparison_is_exact_not_tolerance_based():
    expected = {"joint": 1.0}
    assert WorkerMacroFSMSession._feedback_target_maps_equal(expected, expected)
    assert not WorkerMacroFSMSession._feedback_target_maps_equal(
        {"joint": 1.0 + 5.0e-10},
        expected,
    )
    assert not WorkerMacroFSMSession._feedback_target_maps_equal(
        {"joint": 1.0, "extra": 0.0},
        expected,
    )


def test_feedback_recovery_dispatch_is_config_bound_atomic_and_exact_n_plus_one(
    tmp_path: Path,
):
    reference = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zero_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    actions = [
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.0,
                events=(0,),
                commands=("initial source reference",),
                sequence=0,
            ),
            "servos": reference,
            "wheels": zero_wheels,
        },
        {
            "provenance": _source_provenance(
                segment=1,
                step=2,
                source_time_s=1.0,
                events=(1,),
                commands=("S10 immutable recovery reference",),
                sequence=1,
            ),
            "servos": reference,
            "wheels": zero_wheels,
        },
    ]
    base_bundle = _bundle_for_actions(actions)
    profile_mapping = base_bundle.profiles.to_mapping()
    s1_profile = profile_mapping["profiles"][0]
    s10_keyframe = dict(s1_profile["keyframes"][1])
    s10_keyframe["sequence_index"] = 0
    s10_profile = {
        **s1_profile,
        "profile_id": "profile-s10-reference",
        "state_id": "S10_POSTURE_RECOVERY",
        "source_segment_range": [1, 1],
        "segment_bindings": [s1_profile["segment_bindings"][1]],
        "keyframes": [s10_keyframe],
    }
    s1_profile["source_segment_range"] = [0, 0]
    s1_profile["segment_bindings"] = [s1_profile["segment_bindings"][0]]
    s1_profile["keyframes"] = [s1_profile["keyframes"][0]]
    profile_mapping["segment_ownership"][0]["last_segment"] = 0
    profile_mapping["segment_ownership"].append(
        {
            **profile_mapping["segment_ownership"][0],
            "state_id": "S10_POSTURE_RECOVERY",
            "phase_source_state_id": "S10_POSTURE_RECOVERY",
            "first_segment": 1,
            "last_segment": 1,
        }
    )
    profile_mapping["profiles"].append(s10_profile)
    profiles = SimpleNamespace(
        library_id=_Profiles.library_id,
        sha256=_Profiles.sha256,
        to_mapping=lambda: profile_mapping,
    )
    bundle = SimpleNamespace(
        graph=_Graph(),
        profiles=profiles,
        graph_sha256=_Graph.sha256,
        profile_library_sha256=_Profiles.sha256,
        bundle_sha256=_Bundle.bundle_sha256,
        to_mapping=lambda: {"bundle_sha256": _Bundle.bundle_sha256},
    )
    request, _payload = _load_request(tmp_path)
    session, adapter = _prime_manual_session(request, bundle=bundle)
    epoch = len(session.expected_source_actions)
    session.next_source_action_index = len(session.expected_source_actions)
    session.last_macro_state = "S10_POSTURE_RECOVERY"
    session.last_epoch = epoch
    session.last_servo_targets = dict(reference)
    session.last_wheel_targets = dict(zero_wheels)
    session.last_applied_servo_targets = dict(reference)
    session.last_applied_wheel_targets = dict(zero_wheels)
    session.last_applied_residual = ZERO_RESIDUAL_ACTION
    session.last_verified_servo_targets = dict(reference)
    session.last_verified_wheel_targets = dict(zero_wheels)
    session.last_verified_command_epoch = epoch
    adapter.joint_command_deg.update(reference)
    adapter.servo_applied_command_deg.update(reference)
    adapter.wheel_speeds.update(zero_wheels)
    fr_sensor = session.filtered_contact_wheel_sensors["FR"]
    fr_sensor.data.net_forces_w[...] = 0.0
    fr_sensor.data.force_matrix_w[...] = 0.0
    fr_sensor.data.contact_pos_w[...] = np.nan
    fr_sensor.data.friction_forces_w[...] = 0.0
    # Preserve the exact four-body gravity wrench on the remaining three
    # support contacts while FR is the deliberate AIR recovery candidate.
    for support_leg in ("FL", "RL", "RR"):
        support_sensor = session.filtered_contact_wheel_sensors[support_leg]
        support_sensor.data.net_forces_w[0, 0, :] = (0.0, 0.0, 13.08)
        support_sensor.data.force_matrix_w[0, 0, 0, :] = (0.0, 0.0, 13.08)
    support_centroid = (0.5333333333333333, -0.06666666666666667, 0.05)
    adapter.robot.data.body_com_pos_w[0, :, :] = support_centroid
    adapter.robot.data.body_com_pose_w[0, :, :3] = support_centroid
    front_right_body_index = adapter.robot.body_names.index("front_right_wheel")
    adapter.robot.data.body_link_state_w[
        0, front_right_body_index, :3
    ] = (1.05, 0.20, 0.055)
    payload = None
    first_bootstrap_step = int(session.filtered_contact_last_sim_step or 0) + 1
    for bootstrap_step in range(
        first_bootstrap_step,
        first_bootstrap_step + 13,
    ):
        adapter.sim_steps = bootstrap_step
        adapter.sim_time = bootstrap_step / 120.0
        _prime_current_target_readback(session, adapter)
        payload = session._observation_payload(adapter)
    assert payload is not None
    probe_dispatch_step = adapter.sim_steps
    feedback = FeedbackRecoveryObservation.from_mapping(
        payload["feedback_recovery_observation"]
    )
    centroidal_sha = payload["centroidal_support_evidence"]["payload_sha256"]
    feedback_sha = feedback.payload_sha256
    joint = "front_right_hip"
    targets = dict(reference)
    targets[joint] = 0.25
    target_sha = canonical_mapping_sha256(
        {
            "schema_version": "fsm50.feedback_recovery_target_map.v1",
            "servo_targets_deg": targets,
            "wheel_targets_rad_s": zero_wheels,
        }
    )
    evidence_sha = canonical_mapping_sha256(
        {
            "schema_version": "fsm50.feedback_recovery_evidence_binding.v1",
            "centroidal_support_evidence_sha256": centroidal_sha,
            "feedback_recovery_observation_sha256": feedback_sha,
        }
    )
    configuration_payload = {
        "schema_version": "fsm50.feedback_recovery_configuration.v1",
        "leg": "FR",
        "macro_state": "S10_POSTURE_RECOVERY",
        "selected_source_version": request.source_version,
        "reference_profile_id": s10_profile["profile_id"],
        "reference_profile_source_version": s10_profile["source_version"],
        "reference_profile_source_plan_sha256": s10_profile["source_plan_sha256"],
        "centroidal_evidence_sha256": centroidal_sha,
        "feedback_observation_sha256": feedback_sha,
        "servo_reference_targets_deg": reference,
        "measured_servo_positions_deg": dict(
            feedback.measured_servo_positions_deg
        ),
        "wheel_center_w_m": {
            leg: list(feedback.wheel_center_w_m[leg])
            for leg in ("FR", "FL", "RR", "RL")
        },
        "body_crossed_front_face": feedback.body_crossed_front_face,
        "final_recoverable": feedback.final_recoverable,
        "posture_complete": feedback.posture_complete,
    }
    configuration_sha = canonical_mapping_sha256(configuration_payload)
    provenance = _non_source_provenance()
    provenance.update(
        kind="FEEDBACK_RECOVERY",
        recovery_stage="SAFE_PROBE",
        recovery_action="CONSERVATIVE_DIAGNOSTIC_PROBE",
        recovery_evidence_sha256=evidence_sha,
        recovery_centroidal_evidence_sha256=centroidal_sha,
        recovery_feedback_observation_sha256=feedback_sha,
        recovery_target_map_sha256=target_sha,
        recovery_direction_sign=1,
        recovery_attempt=1,
        recovery_leg="FR",
        recovery_joint=joint,
        recovery_configuration_sha256=configuration_sha,
    )
    wrong_leg_targets = dict(reference)
    wrong_leg_targets["front_left_hip"] = 0.25
    wrong_leg_target_sha = canonical_mapping_sha256(
        {
            "schema_version": "fsm50.feedback_recovery_target_map.v1",
            "servo_targets_deg": wrong_leg_targets,
            "wheel_targets_rad_s": zero_wheels,
        }
    )
    wrong_leg_configuration = dict(configuration_payload)
    wrong_leg_configuration["leg"] = "FL"
    wrong_leg_provenance = dict(provenance)
    wrong_leg_provenance.update(
        recovery_target_map_sha256=wrong_leg_target_sha,
        recovery_leg="FL",
        recovery_joint="front_left_hip",
        recovery_configuration_sha256=canonical_mapping_sha256(
            wrong_leg_configuration
        ),
    )
    wrong_leg_batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="deterministic current AIR"):
            session._process_decision(
                adapter,
                payload,
                _decision(
                    state="S10_POSTURE_RECOVERY",
                    epoch=epoch + 1,
                    changed=True,
                    servos=wrong_leg_targets,
                    wheels=zero_wheels,
                    provenance=wrong_leg_provenance,
                    profile_id=s10_profile["profile_id"],
                    profile_source_version=s10_profile["source_version"],
                    profile_strategy=s10_profile["strategy"],
                    subphase="SAFE_PROBE",
                ),
            )
    finally:
        session.outer_render_boundary_permit = False
    assert session.pending_readback is None
    assert len(adapter.batch_calls) == wrong_leg_batches_before
    class _AckDiagnosticBomb:
        @property
        def value(self):
            raise RuntimeError("ACK_NORMALIZATION_BOMB_AFTER_APPLY")

    original_apply_motion_batch = adapter.apply_motion_batch

    def apply_with_irrelevant_diagnostic(payload):
        ack = original_apply_motion_batch(payload)
        ack["adapter_specific_diagnostic"] = _AckDiagnosticBomb()
        return ack

    adapter.apply_motion_batch = apply_with_irrelevant_diagnostic
    batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            payload,
            _decision(
                state="S10_POSTURE_RECOVERY",
                epoch=epoch + 1,
                changed=True,
                servos=targets,
                wheels=zero_wheels,
                provenance=provenance,
                profile_id=s10_profile["profile_id"],
                profile_source_version=s10_profile["source_version"],
                profile_strategy=s10_profile["strategy"],
                subphase="SAFE_PROBE",
            ),
        )
    finally:
        session.outer_render_boundary_permit = False
        adapter.apply_motion_batch = original_apply_motion_batch
    assert len(adapter.batch_calls) == batches_before + 1
    assert session.next_source_action_index == len(session.expected_source_actions)
    assert len(session.feedback_recovery_action_rows) == 1
    assert session.feedback_recovery_sequence_by_configuration
    assert session.dispatch_rows
    assert session.pending_readback is not None
    action_row = session.feedback_recovery_action_rows[0]
    assert action_row["attempt"] == 1
    assert action_row["n_plus_one_verified"] is False
    assert session.pending_readback["feedback_recovery_action_index"] == 0
    adapter.sim_steps = probe_dispatch_step + 1
    adapter.sim_time = adapter.sim_steps / 120.0
    joint_index = adapter.robot.joint_names.index(joint)
    adapter.robot.data.joint_pos[0, joint_index] = math.radians(0.25)
    adapter.robot.data.body_link_state_w[
        0, front_right_body_index, 2
    ] = 0.0545
    session._verify_pending_readback(adapter, sim_step=adapter.sim_steps)
    assert action_row["n_plus_one_verified"] is True
    assert action_row["n_plus_one_verified_sim_step"] == adapter.sim_steps
    assert len(action_row["n_plus_one_readback_sha256"]) == 64
    assert session.feedback_recovery_verified_action_count == 1

    payload = session._observation_payload(adapter)
    session._capture_feedback_recovery_n_plus_one_response(
        payload=payload,
        sim_step=adapter.sim_steps,
    )
    assert action_row["physical_response_verified"] is True
    assert action_row["physical_response"]["sign_response_valid"] is True
    assert action_row["physical_response"]["dz_m"] <= -0.00025
    assert action_row["n_plus_one_readback"]
    assert action_row["dispatch_centroidal_support_evidence"]
    assert action_row["dispatch_feedback_recovery_observation"]
    assert action_row["physical_response_centroidal_support_evidence"]
    assert action_row["physical_response_feedback_recovery_observation"]
    feedback = FeedbackRecoveryObservation.from_mapping(
        payload["feedback_recovery_observation"]
    )
    centroidal_sha = payload["centroidal_support_evidence"]["payload_sha256"]
    feedback_sha = feedback.payload_sha256
    return_target_sha = canonical_mapping_sha256(
        {
            "schema_version": "fsm50.feedback_recovery_target_map.v1",
            "servo_targets_deg": reference,
            "wheel_targets_rad_s": zero_wheels,
        }
    )
    return_evidence_sha = canonical_mapping_sha256(
        {
            "schema_version": "fsm50.feedback_recovery_evidence_binding.v1",
            "centroidal_support_evidence_sha256": centroidal_sha,
            "feedback_recovery_observation_sha256": feedback_sha,
        }
    )
    return_provenance = _non_source_provenance()
    return_provenance.update(
        kind="FEEDBACK_RECOVERY",
        recovery_stage="RETURN_TO_REFERENCE",
        recovery_action="RETURN_TO_IMMUTABLE_REFERENCE",
        recovery_evidence_sha256=return_evidence_sha,
        recovery_centroidal_evidence_sha256=centroidal_sha,
        recovery_feedback_observation_sha256=feedback_sha,
        recovery_target_map_sha256=return_target_sha,
        recovery_direction_sign=1,
        recovery_attempt=2,
        recovery_leg="FR",
        recovery_joint=joint,
        recovery_configuration_sha256=configuration_sha,
    )

    def return_decision(provenance, servos):
        return _decision(
            state="S10_POSTURE_RECOVERY",
            epoch=session.last_epoch + 1,
            changed=True,
            servos=servos,
            wheels=zero_wheels,
            provenance=provenance,
            profile_id=s10_profile["profile_id"],
            profile_source_version=s10_profile["source_version"],
            profile_strategy=s10_profile["strategy"],
            subphase="RETURN_TO_REFERENCE",
        )

    gap_provenance = dict(return_provenance)
    gap_provenance["recovery_attempt"] = 3
    batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="attempts must start at one"):
            session._process_decision(
                adapter,
                payload,
                return_decision(gap_provenance, reference),
            )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batches_before
    assert session.pending_readback is None

    wrong_return = dict(reference)
    wrong_return[joint] = 0.1
    wrong_target_provenance = dict(return_provenance)
    wrong_target_provenance["recovery_target_map_sha256"] = canonical_mapping_sha256(
        {
            "schema_version": "fsm50.feedback_recovery_target_map.v1",
            "servo_targets_deg": wrong_return,
            "wheel_targets_rad_s": zero_wheels,
        }
    )
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="target map differs"):
            session._process_decision(
                adapter,
                payload,
                return_decision(wrong_target_provenance, wrong_return),
            )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batches_before
    assert session.pending_readback is None

    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            payload,
            return_decision(return_provenance, reference),
        )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batches_before + 1
    assert session.feedback_recovery_action_rows[-1]["stage"] == "RETURN_TO_REFERENCE"
    adapter.sim_steps += 1
    adapter.sim_time = adapter.sim_steps / 120.0
    adapter.robot.data.joint_pos[0, joint_index] = 0.0
    adapter.robot.data.body_link_state_w[
        0, front_right_body_index, 2
    ] = 0.055
    session._verify_pending_readback(adapter, sim_step=adapter.sim_steps)
    assert session.feedback_recovery_action_rows[-1]["n_plus_one_verified"] is True
    assert session.feedback_recovery_verified_action_count == 2
    payload = session._observation_payload(adapter)
    session._capture_feedback_recovery_n_plus_one_response(
        payload=payload,
        sim_step=adapter.sim_steps,
    )
    assert session.feedback_recovery_action_rows[-1][
        "physical_response_verified"
    ] is True

    # This hermetic fixture jumps directly to the post-profile S10 state. Add
    # the durable preceding physical reference dispatch that production gets
    # while consuming the selected source profile, then shift the two
    # feedback-dispatch identities forward by one.
    for dispatch_row in session.dispatch_rows:
        dispatch_row["dispatch_index"] += 1
    for feedback_row in session.feedback_recovery_action_rows:
        feedback_row["dispatch_index"] += 1
    reference_dispatch = json.loads(json.dumps(session.dispatch_rows[0]))
    canonical_reference_action = max(
        (
            action
            for action in session.expected_source_actions
            if action["owner_state"] == "S10_POSTURE_RECOVERY"
        ),
        key=lambda action: action["source_action_index"],
    )
    reference_batch_id = f"{request.request_id}:macro:{epoch:06d}"
    reference_ack = json.loads(json.dumps(reference_dispatch["ack"]))
    reference_ack.update(
        batch_id=reference_batch_id,
        applied_sim_step=probe_dispatch_step - 1,
        first_physics_step=probe_dispatch_step,
        motion_start_skew_s=0.0,
        physics_dt_s=1.0 / 120.0,
        servo_targets_applied=dict(reference),
        wheel_targets_applied=dict(zero_wheels),
    )
    reference_ack["recording_metadata"].update(
        source_version=request.source_version,
        command_epoch=epoch,
        bundle_sha256=request.bundle_sha256,
        command_provenance=dict(
            canonical_reference_action["command_provenance"]
        ),
        source_plan_sha256=canonical_reference_action[
            "source_plan_sha256"
        ],
        source_action_consumption_index=canonical_reference_action[
            "source_action_index"
        ],
    )
    reference_dispatch.update(
        dispatch_index=0,
        batch_id=reference_batch_id,
        command_epoch=epoch,
        sim_step=probe_dispatch_step - 1,
        sim_time_s=(probe_dispatch_step - 1) / 120.0,
        servo_targets_deg=dict(reference),
        wheel_targets_rad_s=dict(zero_wheels),
        n_plus_one_verified=True,
        n_plus_one_verified_sim_step=probe_dispatch_step,
        n_plus_one_readback_sha256="f" * 64,
        command_provenance=dict(
            canonical_reference_action["command_provenance"]
        ),
        source_action_consumption_index=canonical_reference_action[
            "source_action_index"
        ],
        ack=reference_ack,
    )
    session.dispatch_rows.insert(0, reference_dispatch)

    assert session._feedback_recovery_action_closure_errors(
        index=0,
        row=session.feedback_recovery_action_rows[0],
    ) == []
    assert any(
        "ended before the complete safe probe/return matrix" in error
        for error in session._feedback_recovery_durable_sequence_errors()
    )
    saved_feedback_rows = session.feedback_recovery_action_rows
    synthetic_matrix = []
    for matrix_joint, matrix_sign in (
        ("front_right_hip", 1),
        ("front_right_hip", -1),
        ("front_right_knee", 1),
        ("front_right_knee", -1),
    ):
        probe_copy = json.loads(json.dumps(saved_feedback_rows[0]))
        probe_copy.update(
            joint=matrix_joint,
            direction_sign=matrix_sign,
            action="CONSERVATIVE_DIAGNOSTIC_PROBE",
        )
        probe_copy["physical_response"].update(
            joint=matrix_joint,
            direction_sign=matrix_sign,
            dq_deg=0.0,
            dx_m=0.0,
            dz_m=0.0,
            sign_response_valid=False,
            baseline_preserved=True,
            unsafe_reasons=[],
        )
        return_copy = json.loads(json.dumps(saved_feedback_rows[1]))
        return_copy.update(
            joint=matrix_joint,
            direction_sign=matrix_sign,
            action="RETURN_TO_IMMUTABLE_REFERENCE",
        )
        return_copy["physical_response"].update(
            joint=matrix_joint,
            direction_sign=matrix_sign,
            return_error_deg=0.0,
        )
        synthetic_matrix.extend((probe_copy, return_copy))
    session.feedback_recovery_action_rows = synthetic_matrix
    assert session._feedback_recovery_durable_sequence_errors() == []
    session.feedback_recovery_action_rows = saved_feedback_rows
    probe_row = session.feedback_recovery_action_rows[0]
    probe_dispatch = session.dispatch_rows[probe_row["dispatch_index"]]

    reference_dispatch = session.dispatch_rows[probe_row["dispatch_index"] - 1]
    saved_reference_wheels = dict(reference_dispatch["wheel_targets_rad_s"])
    reference_dispatch["wheel_targets_rad_s"][WHEEL_JOINT_NAMES[0]] = 0.25
    assert any(
        "first-action physical reference binding" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0,
            row=probe_row,
        )
    )
    reference_dispatch["wheel_targets_rad_s"] = saved_reference_wheels

    saved_attempt = probe_row["attempt"]
    probe_row["attempt"] = 64
    assert any(
        "core action identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["attempt"] = saved_attempt

    saved_batch_id = probe_row["batch_id"]
    probe_row["batch_id"] = "coherently-tampered-batch"
    assert any(
        "dispatch" in error or "ACK" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["batch_id"] = saved_batch_id

    saved_provenance = dict(probe_row["command_provenance"])
    probe_row["command_provenance"] = {}
    assert any(
        "provenance" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["command_provenance"] = saved_provenance

    saved_readback = json.loads(json.dumps(probe_row["n_plus_one_readback"]))
    saved_readback_sha = probe_row["n_plus_one_readback_sha256"]
    saved_dispatch_readback_sha = probe_dispatch["n_plus_one_readback_sha256"]
    probe_row["n_plus_one_readback"] = {"arbitrary": True}
    probe_row["n_plus_one_readback_sha256"] = canonical_mapping_sha256(
        probe_row["n_plus_one_readback"]
    )
    probe_dispatch["n_plus_one_readback_sha256"] = probe_row[
        "n_plus_one_readback_sha256"
    ]
    assert any(
        "readback preimage/full identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["n_plus_one_readback"] = saved_readback
    probe_row["n_plus_one_readback_sha256"] = saved_readback_sha
    probe_dispatch["n_plus_one_readback_sha256"] = saved_dispatch_readback_sha

    saved_actual_servos = dict(
        probe_row["n_plus_one_readback"]["actual_servo_drive_targets_rad"]
    )
    first_servo = SERVO_JOINT_NAMES[0]
    probe_row["n_plus_one_readback"]["actual_servo_drive_targets_rad"][
        first_servo
    ] += 1.0
    probe_row["n_plus_one_readback_sha256"] = canonical_mapping_sha256(
        probe_row["n_plus_one_readback"]
    )
    probe_dispatch["n_plus_one_readback_sha256"] = probe_row[
        "n_plus_one_readback_sha256"
    ]
    assert any(
        "readback preimage/full identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["n_plus_one_readback"]["actual_servo_drive_targets_rad"] = (
        saved_actual_servos
    )
    probe_row["n_plus_one_readback_sha256"] = saved_readback_sha
    probe_dispatch["n_plus_one_readback_sha256"] = saved_dispatch_readback_sha

    saved_expected_servos = dict(
        probe_row["n_plus_one_readback"][
            "expected_servo_drive_targets_rad"
        ]
    )
    probe_row["n_plus_one_readback"]["actual_servo_drive_targets_rad"][
        first_servo
    ] += 1.0
    probe_row["n_plus_one_readback"]["expected_servo_drive_targets_rad"][
        first_servo
    ] += 1.0
    probe_row["n_plus_one_readback_sha256"] = canonical_mapping_sha256(
        probe_row["n_plus_one_readback"]
    )
    probe_dispatch["n_plus_one_readback_sha256"] = probe_row[
        "n_plus_one_readback_sha256"
    ]
    assert any(
        "readback preimage/full identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["n_plus_one_readback"]["actual_servo_drive_targets_rad"] = (
        saved_actual_servos
    )
    probe_row["n_plus_one_readback"]["expected_servo_drive_targets_rad"] = (
        saved_expected_servos
    )
    probe_row["n_plus_one_readback_sha256"] = saved_readback_sha
    probe_dispatch["n_plus_one_readback_sha256"] = saved_dispatch_readback_sha

    saved_runtime_id = probe_row["n_plus_one_readback"][
        "adapter_runtime_instance_id"
    ]
    probe_row["n_plus_one_readback"][
        "adapter_runtime_instance_id"
    ] = "forged-runtime"
    probe_row["n_plus_one_readback_sha256"] = canonical_mapping_sha256(
        probe_row["n_plus_one_readback"]
    )
    probe_dispatch["n_plus_one_readback_sha256"] = probe_row[
        "n_plus_one_readback_sha256"
    ]
    assert any(
        "readback preimage/full identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["n_plus_one_readback"][
        "adapter_runtime_instance_id"
    ] = saved_runtime_id
    probe_row["n_plus_one_readback_sha256"] = saved_readback_sha
    probe_dispatch["n_plus_one_readback_sha256"] = saved_dispatch_readback_sha

    saved_ack = json.loads(json.dumps(probe_row["dispatch_ack"]))
    saved_ack_sha = probe_row["ack_sha256"]
    saved_dispatch_ack = json.loads(json.dumps(probe_dispatch["ack"]))
    tampered_ack = json.loads(json.dumps(saved_ack))
    tampered_ack["motion_start_skew_s"] = 0.5
    tampered_ack["physics_dt_s"] = 0.5
    probe_row["dispatch_ack"] = tampered_ack
    probe_row["ack_sha256"] = canonical_mapping_sha256(tampered_ack)
    probe_dispatch["ack"] = json.loads(json.dumps(tampered_ack))
    assert any(
        "durable atomic ACK" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["dispatch_ack"] = saved_ack
    probe_row["ack_sha256"] = saved_ack_sha
    probe_dispatch["ack"] = saved_dispatch_ack

    tampered_ack = json.loads(json.dumps(saved_ack))
    tampered_ack["recording_metadata"].update(
        {
            "source_version": "forged-source",
            "macro_state": "S3_REAR_LEFT_LIFT",
            "subphase": "INCREMENT",
            "bundle_sha256": "0" * 64,
        }
    )
    probe_row["dispatch_ack"] = tampered_ack
    probe_row["ack_sha256"] = canonical_mapping_sha256(tampered_ack)
    probe_dispatch["ack"] = json.loads(json.dumps(tampered_ack))
    assert any(
        "ACK preimage/identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["dispatch_ack"] = saved_ack
    probe_row["ack_sha256"] = saved_ack_sha
    probe_dispatch["ack"] = saved_dispatch_ack

    saved_actual_wheels = dict(
        probe_row["n_plus_one_readback"][
            "actual_wheel_drive_targets_rad_s"
        ]
    )
    first_wheel = WHEEL_JOINT_NAMES[0]
    probe_row["n_plus_one_readback"][
        "actual_wheel_drive_targets_rad_s"
    ][first_wheel] = 1.0
    probe_row["n_plus_one_readback_sha256"] = canonical_mapping_sha256(
        probe_row["n_plus_one_readback"]
    )
    probe_dispatch["n_plus_one_readback_sha256"] = probe_row[
        "n_plus_one_readback_sha256"
    ]
    assert any(
        "readback preimage/full identity" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["n_plus_one_readback"][
        "actual_wheel_drive_targets_rad_s"
    ] = saved_actual_wheels
    probe_row["n_plus_one_readback_sha256"] = saved_readback_sha
    probe_dispatch["n_plus_one_readback_sha256"] = saved_dispatch_readback_sha

    saved_probe_response = dict(probe_row["physical_response"])
    sequence_for_tamper = dict(
        session.feedback_recovery_sequence_by_configuration[configuration_sha]
    )
    saved_sequence_for_tamper = json.loads(json.dumps(sequence_for_tamper))
    tampered_baseline = dict(sequence_for_tamper["physical_baseline"])
    tampered_positions = dict(
        tampered_baseline["measured_servo_positions_deg"]
    )
    tampered_positions[joint] += 0.1
    tampered_baseline["measured_servo_positions_deg"] = tampered_positions
    sequence_for_tamper["physical_baseline"] = tampered_baseline
    session.feedback_recovery_sequence_by_configuration[
        configuration_sha
    ] = sequence_for_tamper
    probe_row["physical_response"]["dq_deg"] -= 0.1
    assert any(
        "physical response differs" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0, row=probe_row
        )
    )
    probe_row["physical_response"] = saved_probe_response
    session.feedback_recovery_sequence_by_configuration[
        configuration_sha
    ] = saved_sequence_for_tamper
    saved_response_evidence = json.loads(
        json.dumps(
            session.feedback_recovery_action_rows[0][
                "physical_response_feedback_recovery_observation"
            ]
        )
    )
    session.feedback_recovery_action_rows[0][
        "physical_response_feedback_recovery_observation"
    ]["payload"]["sim_step"] += 1
    assert any(
        "strict evidence envelope is invalid" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=0,
            row=session.feedback_recovery_action_rows[0],
        )
    )
    session.feedback_recovery_action_rows[0][
        "physical_response_feedback_recovery_observation"
    ] = saved_response_evidence
    return_row = session.feedback_recovery_action_rows[1]
    saved_return_evidence = json.loads(
        json.dumps(
            return_row["physical_response_feedback_recovery_observation"]
        )
    )
    saved_return_sha = return_row[
        "physical_response_feedback_observation_sha256"
    ]
    saved_return_response = dict(return_row["physical_response"])
    return_payload = return_row[
        "physical_response_feedback_recovery_observation"
    ]["payload"]
    return_payload["measured_servo_positions_deg"][joint] = 0.3
    return_row["physical_response_feedback_recovery_observation"][
        "payload_sha256"
    ] = canonical_mapping_sha256(return_payload)
    return_row["physical_response_feedback_observation_sha256"] = (
        return_row["physical_response_feedback_recovery_observation"][
            "payload_sha256"
        ]
    )
    return_row["physical_response"]["return_error_deg"] = 0.3
    assert any(
        "return did not close" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=1,
            row=return_row,
        )
    )
    return_row["physical_response_feedback_recovery_observation"] = (
        saved_return_evidence
    )
    return_row["physical_response_feedback_observation_sha256"] = (
        saved_return_sha
    )
    return_row["physical_response"] = saved_return_response

    # A later durable row may not point backward (or sideways) in the
    # physical-dispatch ledger even when all of its other row-local hashes are
    # self-consistent.  Cross-action chronology is an independent invariant.
    saved_return_dispatch_index = return_row["dispatch_index"]
    return_row["dispatch_index"] = probe_row["dispatch_index"]
    assert any(
        "physical dispatch chronology" in error
        for error in session._feedback_recovery_action_closure_errors(
            index=1,
            row=return_row,
        )
    )
    return_row["dispatch_index"] = saved_return_dispatch_index

    sequence = dict(
        session.feedback_recovery_sequence_by_configuration[configuration_sha]
    )
    sequence["probe_index"] = 4
    session.feedback_recovery_sequence_by_configuration[configuration_sha] = sequence
    payload = session._observation_payload(adapter)
    feedback = FeedbackRecoveryObservation.from_mapping(
        payload["feedback_recovery_observation"]
    )
    centroidal_sha = payload["centroidal_support_evidence"]["payload_sha256"]
    feedback_sha = feedback.payload_sha256
    unproven_increment = dict(reference)
    unproven_increment[joint] = -0.25
    unproven_increment_provenance = _non_source_provenance()
    unproven_increment_provenance.update(
        kind="FEEDBACK_RECOVERY",
        recovery_stage="INCREMENT",
        recovery_action="BOUNDED_DESCENT_INCREMENT",
        recovery_evidence_sha256=canonical_mapping_sha256(
            {
                "schema_version": "fsm50.feedback_recovery_evidence_binding.v1",
                "centroidal_support_evidence_sha256": centroidal_sha,
                "feedback_recovery_observation_sha256": feedback_sha,
            }
        ),
        recovery_centroidal_evidence_sha256=centroidal_sha,
        recovery_feedback_observation_sha256=feedback_sha,
        recovery_target_map_sha256=canonical_mapping_sha256(
            {
                "schema_version": "fsm50.feedback_recovery_target_map.v1",
                "servo_targets_deg": unproven_increment,
                "wheel_targets_rad_s": zero_wheels,
            }
        ),
        recovery_direction_sign=-1,
        recovery_attempt=3,
        recovery_leg="FR",
        recovery_joint=joint,
        recovery_configuration_sha256=configuration_sha,
    )
    batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="complete independently verified"):
            session._process_decision(
                adapter,
                payload,
                _decision(
                    state="S10_POSTURE_RECOVERY",
                    epoch=session.last_epoch + 1,
                    changed=True,
                    servos=unproven_increment,
                    wheels=zero_wheels,
                    provenance=unproven_increment_provenance,
                    profile_id=s10_profile["profile_id"],
                    profile_source_version=s10_profile["source_version"],
                    profile_strategy=s10_profile["strategy"],
                    subphase="INCREMENT",
                ),
            )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batches_before
    assert session.pending_readback is None

    _assert_feedback_increment_margin_is_fail_closed(
        session=session,
        adapter=adapter,
        configuration_sha=configuration_sha,
        reference=reference,
        zero_wheels=zero_wheels,
        joint=joint,
        s10_profile=s10_profile,
    )


def _assert_feedback_increment_margin_is_fail_closed(
    *,
    session,
    adapter,
    configuration_sha,
    reference,
    zero_wheels,
    joint,
    s10_profile,
):
    _lower, upper = command_limits_for_servo(joint)
    near_limit_reference = dict(reference)
    near_limit_reference[joint] = upper - 1.3
    session.last_servo_targets = dict(near_limit_reference)
    session.last_applied_servo_targets = dict(near_limit_reference)
    session.last_verified_servo_targets = dict(near_limit_reference)
    adapter.joint_command_deg.update(near_limit_reference)
    adapter.servo_applied_command_deg.update(near_limit_reference)
    _prime_current_target_readback(session, adapter)
    sequence = dict(
        session.feedback_recovery_sequence_by_configuration[configuration_sha]
    )
    probe_pairs = [
        ("front_right_hip", 1),
        ("front_right_hip", -1),
        ("front_right_knee", 1),
        ("front_right_knee", -1),
    ]
    sequence.update(
        reference_targets_deg=dict(near_limit_reference),
        probe_index=4,
        awaiting_return=False,
        completed_probe_pairs=list(probe_pairs),
        probe_results=[
            {
                "joint": probe_joint,
                "direction_sign": probe_sign,
                "dq_deg": 0.25,
                "dx_m": 0.0,
                "dz_m": (
                    -0.001
                    if (probe_joint, probe_sign) == (joint, 1)
                    else -0.0005
                ),
                "sign_response_valid": True,
                "baseline_preserved": True,
                "unsafe_reasons": [],
                "n_plus_one_response_verified": True,
            }
            for probe_joint, probe_sign in probe_pairs
        ],
        increment_joint=joint,
        increment_sign=1,
        increment_count=1,
    )
    session.feedback_recovery_sequence_by_configuration[configuration_sha] = sequence
    payload = session._observation_payload(adapter)
    feedback = FeedbackRecoveryObservation.from_mapping(
        payload["feedback_recovery_observation"]
    )
    centroidal_sha = payload["centroidal_support_evidence"]["payload_sha256"]
    feedback_sha = feedback.payload_sha256
    target = dict(near_limit_reference)
    target[joint] += 0.50
    provenance = _non_source_provenance()
    provenance.update(
        kind="FEEDBACK_RECOVERY",
        recovery_stage="INCREMENT",
        recovery_action="BOUNDED_DESCENT_INCREMENT",
        recovery_evidence_sha256=canonical_mapping_sha256(
            {
                "schema_version": "fsm50.feedback_recovery_evidence_binding.v1",
                "centroidal_support_evidence_sha256": centroidal_sha,
                "feedback_recovery_observation_sha256": feedback_sha,
            }
        ),
        recovery_centroidal_evidence_sha256=centroidal_sha,
        recovery_feedback_observation_sha256=feedback_sha,
        recovery_target_map_sha256=canonical_mapping_sha256(
            {
                "schema_version": "fsm50.feedback_recovery_target_map.v1",
                "servo_targets_deg": target,
                "wheel_targets_rad_s": zero_wheels,
            }
        ),
        recovery_direction_sign=1,
        recovery_attempt=len(session.feedback_recovery_action_rows) + 1,
        recovery_leg="FR",
        recovery_joint=joint,
        recovery_configuration_sha256=configuration_sha,
    )
    batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="command-limit margin"):
            session._process_decision(
                adapter,
                payload,
                _decision(
                    state="S10_POSTURE_RECOVERY",
                    epoch=session.last_epoch + 1,
                    changed=True,
                    servos=target,
                    wheels=zero_wheels,
                    provenance=provenance,
                    profile_id=s10_profile["profile_id"],
                    profile_source_version=s10_profile["source_version"],
                    profile_strategy=s10_profile["strategy"],
                    subphase="INCREMENT",
                ),
            )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batches_before
    assert session.pending_readback is None


def test_runtime_safety_rehashes_contact_sample_and_rejects_stale_provider(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    original = json.loads(json.dumps(session.filtered_contact_sample))
    session.filtered_contact_sample["nonwheel_rows"][0]["active"] = True
    with pytest.raises(RuntimeError, match="current exact non-wheel contact sample"):
        session._capture_runtime_safety_evidence(adapter)
    session.filtered_contact_sample = original
    session.filtered_contact_nonwheel_rows = json.loads(
        json.dumps(original["nonwheel_rows"])
    )
    session._capture_runtime_safety_evidence(adapter)

    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": False,
        "source": "TEST_STALE_RUNTIME_PROVIDER",
        "sample_sim_step": 0,
        "error": "",
    }
    with pytest.raises(RuntimeError, match="provider sample is stale"):
        session._capture_runtime_safety_evidence(adapter)


def test_unverified_provider_true_collision_claim_stops_without_false_attribution(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "dangerous_body_collision": True,
    }
    with pytest.raises(RuntimeError, match="true collision claim is not a valid"):
        session._capture_runtime_safety_evidence(adapter)
    physical = session._task_inputs(success=False, error="unverified provider claim")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False
    assert physical["dangerous_collision_validation_source"] == (
        "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    )
    assert physical["runtime_collision_penetration_classification"] == (
        "UNVERIFIED_PROVIDER_TRUE_CLAIM_HARD_STOP"
    )
    claim = physical["unverified_provider_collision_claim"]
    assert claim["classification"] == "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    assert len(claim["provider_payload_sha256"]) == 64
    assert claim["provider_payload_sha256_error"] == ""
    assert "unavailable" in claim["provider_contract_error"]
    assert claim["combined_contact_sample_sha256"] == (
        session.filtered_contact_sample["sample_sha256"]
    )


@pytest.mark.parametrize(
    "malformed",
    (
        {"severe_penetration": None, "error": ""},
        {"severe_penetration": False, "error": "provider reported corruption"},
        {
            "available": "malformed",
            "severe_penetration": False,
            "error": "",
        },
    ),
)
def test_provider_true_malformed_contract_cannot_be_finalized_as_collision_clear(
    tmp_path: Path,
    malformed: dict[str, object],
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": True,
        "source": "PROVIDER_TRUE_BUT_MALFORMED",
        "sample_sim_step": adapter.sim_steps,
        **malformed,
    }
    with pytest.raises(RuntimeError, match="true collision claim is not a valid"):
        session._capture_runtime_safety_evidence(adapter)
    physical = session._task_inputs(success=False, error="malformed provider claim")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False
    assert physical["runtime_collision_penetration_classification"] == (
        "UNVERIFIED_PROVIDER_TRUE_CLAIM_HARD_STOP"
    )
    claim = physical["unverified_provider_collision_claim"]
    assert claim["classification"] == "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    assert claim["reported_source"] == "PROVIDER_TRUE_BUT_MALFORMED"
    assert claim["provider_contract_error"]


def test_valid_provider_true_is_sticky_before_later_observation_failure(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": True,
        "severe_penetration": False,
        "source": "VERIFIED_CURRENT_PROVIDER",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
    }
    evidence = session._capture_runtime_safety_evidence(adapter)
    assert evidence["dangerous_body_collision"] is True
    assert session.dangerous_collision_detected is True
    # Model an exception in any later observation field before hard safety.
    physical = session._task_inputs(success=False, error="later observation failure")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is True
    assert physical["dangerous_collision_available"] is True
    detection = physical["dangerous_collision_detection_evidence"]
    assert detection["sample_sim_step"] == adapter.sim_steps
    assert "VERIFIED_CURRENT_PROVIDER" in detection["source"]
    assert len(detection["runtime_safety_evidence_sha256"]) == 64


def test_provider_true_nonserializable_payload_records_claim_before_hash_failure(
    tmp_path: Path,
):
    class _NonSerializable:
        def __str__(self) -> str:
            raise ValueError("intentional provider serialization failure")

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": True,
        "severe_penetration": False,
        "source": "PROVIDER_WITH_UNSERIALIZABLE_DIAGNOSTIC",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
        "diagnostic": _NonSerializable(),
    }
    with pytest.raises(ValueError, match="serialization failure"):
        session._capture_runtime_safety_evidence(adapter)
    claim = session.unverified_provider_collision_claim
    assert claim["classification"] == "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    assert claim["provider_payload_sha256"] == ""
    assert "serialization failure" in claim["provider_payload_sha256_error"]
    physical = session._task_inputs(success=False, error="provider hash failure")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False


def test_provider_true_penetration_is_sticky_before_normalization_failure(
    tmp_path: Path,
):
    class _ExplodingValue:
        @property
        def value(self):
            raise RuntimeError("intentional penetration normalization failure")

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": True,
        "source": "PROVIDER_PENETRATION_TRUE_WITH_BAD_DIAGNOSTIC",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
        "diagnostic": _ExplodingValue(),
    }
    with pytest.raises(RuntimeError, match="penetration normalization failure"):
        session._capture_runtime_safety_evidence(adapter)
    physical = session._task_inputs(
        success=False, error="penetration normalization failure"
    )["physical_evidence"]
    assert physical["severe_penetration"] is True
    assert physical["penetration_evidence_available"] is True
    claim = physical["unverified_provider_penetration_claim"]
    assert claim["classification"] == (
        "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM"
    )
    assert claim["reported_sample_sim_step"] == adapter.sim_steps
    assert claim["reported_source"] == (
        "PROVIDER_PENETRATION_TRUE_WITH_BAD_DIAGNOSTIC"
    )
    assert physical["penetration_validation_source"] == (
        "PROVIDER_PENETRATION_TRUE_WITH_BAD_DIAGNOSTIC"
    )
    assert "normalization failure" in claim["provider_payload_sha256_error"]


def test_verified_provider_penetration_preserves_first_hit_provenance(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": True,
        "source": "VALID_PENETRATION_SENSOR",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
    }
    evidence = session._capture_runtime_safety_evidence(adapter)
    assert evidence["severe_penetration"] is True
    physical = session._task_inputs(success=False, error="penetration detected")[
        "physical_evidence"
    ]
    assert physical["severe_penetration"] is True
    assert physical["penetration_evidence_available"] is True
    assert physical["penetration_validation_source"].startswith(
        "VALID_PENETRATION_SENSOR;"
        "COMBINED_FILTERED_NONWHEEL_OBSTACLE_CONTACT_CURRENT_TICK:"
    )
    detection = physical["severe_penetration_detection_evidence"]
    assert detection["sample_sim_step"] == adapter.sim_steps
    assert "VALID_PENETRATION_SENSOR" in detection["source"]
    assert len(detection["runtime_safety_evidence_sha256"]) == 64


def test_later_verified_penetration_does_not_replace_first_unverified_hit(
    tmp_path: Path,
):
    class _ExplodingValue:
        @property
        def value(self):
            raise RuntimeError("first penetration sample normalization failed")

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": True,
        "source": "FIRST_UNVERIFIED_PENETRATION_HIT",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
        "diagnostic": _ExplodingValue(),
    }
    with pytest.raises(RuntimeError, match="first penetration sample"):
        session._capture_runtime_safety_evidence(adapter)
    first_claim = dict(session.unverified_provider_penetration_claim)

    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": True,
        "source": "LATER_VERIFIED_PENETRATION_HIT",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
    }
    evidence = session._capture_runtime_safety_evidence(adapter)
    assert evidence["severe_penetration"] is True
    assert session.unverified_provider_penetration_claim == first_claim
    assert session.severe_penetration_detection_evidence["sample_sim_step"] == 1
    assert "LATER_VERIFIED_PENETRATION_HIT" in (
        session.severe_penetration_detection_evidence["source"]
    )
    physical = session._task_inputs(success=False, error="penetration detected")[
        "physical_evidence"
    ]
    assert physical["penetration_validation_source"] == (
        "FIRST_UNVERIFIED_PENETRATION_HIT"
    )
    assert physical["unverified_provider_penetration_claim"] == first_claim


def test_provider_true_hash_bomb_records_claim_before_canonical_hash_failure(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": True,
        "severe_penetration": False,
        "source": "PROVIDER_WITH_HASH_BOMB_DIAGNOSTIC",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
        "diagnostic_integer": 10**5000,
    }
    with pytest.raises(ValueError):
        session._capture_runtime_safety_evidence(adapter)
    claim = session.unverified_provider_collision_claim
    assert claim["classification"] == "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    assert claim["reported_source"] == "PROVIDER_WITH_HASH_BOMB_DIAGNOSTIC"
    assert claim["provider_payload_sha256"] == ""
    physical = session._task_inputs(success=False, error="provider hash bomb")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False
    assert physical["runtime_collision_penetration_classification"] == (
        "UNVERIFIED_PROVIDER_TRUE_CLAIM_HARD_STOP"
    )


def test_provider_true_cannot_flip_false_during_mapping_snapshot(
    tmp_path: Path,
):
    class _FlippingCollisionMapping(dict):
        def __init__(self, payload: dict[str, object]):
            super().__init__(payload)
            self._first_collision_read = True

        def get(self, key, default=None):
            if key == "dangerous_body_collision" and self._first_collision_read:
                self._first_collision_read = False
                return True
            return super().get(key, default)

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    provider_mapping = _FlippingCollisionMapping(
        {
            "available": True,
            "dangerous_body_collision": False,
            "severe_penetration": False,
            "source": "FLIPPING_PROVIDER_MAPPING",
            "sample_sim_step": adapter.sim_steps,
            "error": "",
        }
    )
    adapter.capture_macro_runtime_safety_evidence = (
        lambda **_kwargs: provider_mapping
    )
    with pytest.raises(RuntimeError, match="claim drifted during snapshot"):
        session._capture_runtime_safety_evidence(adapter)
    claim = session.unverified_provider_collision_claim
    assert claim["classification"] == "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    assert "changed while" in claim["provider_contract_error"]
    physical = session._task_inputs(success=False, error="provider snapshot drift")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False
    assert physical["runtime_collision_penetration_classification"] == (
        "UNVERIFIED_PROVIDER_TRUE_CLAIM_HARD_STOP"
    )


def test_provider_snapshot_true_is_recorded_before_unrelated_normalization_failure(
    tmp_path: Path,
):
    class _ExplodingValue:
        @property
        def value(self):
            raise RuntimeError("intentional post-snapshot normalization failure")

    class _SnapshotTrueMapping(dict):
        def get(self, key, default=None):
            if key == "dangerous_body_collision":
                return False
            return super().get(key, default)

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    provider_mapping = _SnapshotTrueMapping(
        {
            "available": True,
            "dangerous_body_collision": True,
            "severe_penetration": False,
            "source": "SNAPSHOT_TRUE_PROVIDER_MAPPING",
            "sample_sim_step": adapter.sim_steps,
            "error": "",
            "diagnostic": _ExplodingValue(),
        }
    )
    adapter.capture_macro_runtime_safety_evidence = (
        lambda **_kwargs: provider_mapping
    )
    with pytest.raises(RuntimeError, match="normalization failure"):
        session._capture_runtime_safety_evidence(adapter)
    claim = session.unverified_provider_collision_claim
    assert claim["classification"] == "UNVERIFIED_PROVIDER_TRUE_CLAIM"
    assert "normalization failure" in claim["provider_payload_sha256_error"]
    physical = session._task_inputs(success=False, error="provider normalization failure")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False
    assert physical["runtime_collision_penetration_classification"] == (
        "UNVERIFIED_PROVIDER_TRUE_CLAIM_HARD_STOP"
    )


def test_final_readback_failure_cannot_reuse_prior_tick_as_collision_clear(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    assert session.runtime_contact_collision_evidence["last_sample_sim_step"] == 0
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0

    def fail_readback():
        raise RuntimeError("injected final readback failure")

    adapter.capture_motion_start_base_evidence = fail_readback
    terminal = session._finalize(success=False, error="injected failure")
    physical = terminal["task_inputs"]["physical_evidence"]
    collision = physical["runtime_nonwheel_collision_evidence"]
    assert collision["terminal_adapter_sim_step"] == 1
    assert collision["last_sample_sim_step"] == 0
    assert collision["coverage_complete"] is False
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False
    for field in ("body_stuck", "active_leg_trapped"):
        assert physical[field] is None
        assert physical[f"{field}_available"] is False
        assert physical[f"{field}_last_sample_sim_step"] == 0
        assert physical[f"{field}_terminal_adapter_sim_step"] == 1
        assert physical[f"{field}_complete_live_coverage"] is False


def test_terminal_time_drift_cannot_reuse_same_step_as_collision_clear(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    assert session.runtime_contact_collision_evidence["last_sample_sim_step"] == 0
    adapter.sim_time = 1.0 / 120.0

    def fail_readback():
        raise RuntimeError("injected final readback failure with time drift")

    adapter.capture_motion_start_base_evidence = fail_readback
    terminal = session._finalize(success=False, error="injected failure")
    physical = terminal["task_inputs"]["physical_evidence"]
    collision = physical["runtime_nonwheel_collision_evidence"]
    assert collision["terminal_adapter_sim_step"] == 0
    assert collision["terminal_adapter_sim_time_s"] == 1.0 / 120.0
    assert collision["last_published_contact_sim_time_s"] == 0.0
    assert collision["terminal_time_identity_valid"] is False
    assert collision["coverage_complete"] is False
    assert physical["dangerous_collision"] is None
    assert physical["dangerous_collision_available"] is False


@pytest.mark.parametrize("tamper", ("nan_friction", "nan_active_point"))
def test_combined_contact_refresh_fails_closed_before_partial_publication(
    tmp_path: Path,
    tamper: str,
):
    request, _payload = _load_request(tmp_path)
    session, _adapter, _controller = _runtime(request)
    sensor = session.scene_handle.contact_sensor.wheel_bank.sensors["FL"]
    if tamper == "nan_friction":
        sensor.data.friction_forces_w[0, 0, 0, 0] = np.nan
    else:
        sensor.data.contact_pos_w[0, 0, 0, :] = np.nan
    with pytest.raises(RuntimeError):
        session._refresh_filtered_contact_evidence(_adapter)
    assert session.filtered_contact_sample_epoch == 0
    assert session.filtered_contact_sample == {}
    assert session.filtered_contact_frame is None


def test_exact_one_newton_filtered_row_is_not_active_support(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, _adapter, _controller = _runtime(request)
    sensor = session.scene_handle.contact_sensor.wheel_bank.sensors["FL"]
    sensor.data.force_matrix_w[0, 0, 0, :] = (0.0, 0.0, 1.0)
    sensor.data.net_forces_w[0, 0, :] = (0.0, 0.0, 1.0)
    session._refresh_filtered_contact_evidence(_adapter)
    rows = [
        row
        for row in session.filtered_contact_wheel_rows
        if row["leg"] == "FL" and row["surface"] == "ground"
    ]
    assert len(rows) == 1
    assert rows[0]["bank_active"] is True
    assert rows[0]["active"] is False
    assert session.filtered_contact_surface_kind_by_leg["FL"] == "AIR"
    assert session.filtered_contact_frame.by_leg()["FL"].measurement.active is False


def test_exact_one_newton_nonwheel_row_is_not_a_collision(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    sensor = next(iter(session.filtered_contact_nonwheel_bank.sensors.values()))
    sensor.data.force_matrix_w[0, 0, 0, :] = (0.0, 0.0, 1.0)
    sensor.data.net_forces_w[0, 0, :] = (0.0, 0.0, 1.0)
    sensor.data.contact_pos_w[0, 0, 0, :] = (0.55, 0.0, 0.05)
    session._refresh_filtered_contact_evidence(adapter)
    assert session.filtered_contact_nonwheel_rows[0]["active"] is False
    evidence = session._capture_runtime_safety_evidence(adapter)
    assert evidence["filtered_nonwheel_collision"] is False
    assert evidence["dangerous_body_collision"] is False


def test_current_nonwheel_obstacle_contact_is_a_same_tick_hard_stop(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, _adapter, _controller = _runtime(request)
    sensor = next(
        iter(session.scene_handle.contact_sensor.nonwheel_bank.sensors.values())
    )
    sensor.data.force_matrix_w[0, 0, 0, :] = (0.0, 0.0, 5.0)
    sensor.data.net_forces_w[0, 0, :] = (0.0, 0.0, 5.0)
    sensor.data.contact_pos_w[0, 0, 0, :] = (0.55, 0.0, 0.05)
    with pytest.raises(RuntimeError, match="dangerous body collision"):
        session.start()
    assert session.filtered_contact_nonwheel_rows[0]["active"] is True
    assert session.filtered_contact_sample["sample_sim_step"] == 0


def test_contact_collision_captured_before_later_observation_failure_cannot_false_clear(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    sensor = next(iter(session.filtered_contact_nonwheel_bank.sensors.values()))
    sensor.data.force_matrix_w[0, 0, 0, :] = (0.0, 0.0, 5.0)
    sensor.data.net_forces_w[0, 0, :] = (0.0, 0.0, 5.0)
    sensor.data.contact_pos_w[0, 0, 0, :] = (0.55, 0.0, 0.05)
    session._refresh_filtered_contact_evidence(adapter)
    evidence = session._capture_runtime_safety_evidence(adapter)
    assert evidence["dangerous_body_collision"] is True
    # Model any later payload-construction exception by finalizing evidence
    # before _validate_hard_safety has had a chance to set its sticky flag.
    assert session.dangerous_collision_detected is False
    physical = session._task_inputs(success=False, error="injected observation failure")[
        "physical_evidence"
    ]
    assert physical["dangerous_collision"] is True
    assert physical["dangerous_collision_available"] is True
    assert physical["runtime_nonwheel_collision_evidence"][
        "detected_sample_count"
    ] == 1
    assert physical["runtime_nonwheel_collision_evidence"][
        "coverage_complete"
    ] is False
    assert physical["dangerous_collision_detection_evidence"][
        "combined_contact_sample_sha256"
    ] == session.filtered_contact_sample["sample_sha256"]


def test_current_complete_coalesces_adjacent_changed_start_with_one_n_plus_one_batch(
    tmp_path: Path,
):
    session, adapter, decision, step = _coalesced_transition_case(
        tmp_path, same_target=False
    )
    physical_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            decision,
        )
    finally:
        session.outer_render_boundary_permit = False

    assert len(adapter.batch_calls) == physical_before + 1
    assert len(adapter.batch_call_sim_steps) == len(set(adapter.batch_call_sim_steps))
    assert adapter.batch_call_sim_steps[-1] == step
    assert session.transition_rows[-1]["from_state"] == (
        "S1_APPROACH_AND_PRE_FR_SHIFT"
    )
    assert session.transition_rows[-1]["to_state"] == "S2_FR_TRAVERSE"
    assert session.transition_rows[-1]["sim_step"] == step
    assert session.transition_rows[-1]["events"] == [
        "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
        "ENTER:S2_FR_TRAVERSE",
    ]
    source_row = session.source_action_consumption_rows[-1]
    assert source_row["source_action_index"] == 1
    assert source_row["macro_state"] == "S2_FR_TRAVERSE"
    assert source_row["sim_step"] == step
    assert source_row["physical_dispatch_applied"] is True
    assert source_row["physical_dispatch_index"] == len(session.dispatch_rows) - 1
    assert session.dispatch_rows[-1]["sim_step"] == step
    assert session.dispatch_rows[-1]["macro_state"] == "S2_FR_TRAVERSE"
    assert session.dispatch_rows[-1]["command_provenance"]["kind"] == (
        "SOURCE_ACTION"
    )
    assert session.pending_readback is not None
    assert session.segment_completion_rows[-1]["start_sim_step"] == step
    assert session.segment_completion_rows[-1]["start_physical_dispatch"] is True

    adapter.sim_steps = step + 1
    adapter.sim_time = (step + 1) / 120.0
    session._verify_pending_readback(adapter, sim_step=step + 1)
    assert session.dispatch_rows[-1]["n_plus_one_verified"] is True
    assert session.dispatch_rows[-1]["n_plus_one_verified_sim_step"] == step + 1
    assert source_row["n_plus_one_verified"] is True
    assert source_row["n_plus_one_verified_sim_step"] == step + 1
    assert session.segment_completion_rows[-1]["start_readback_verified"] is True
    assert (
        session.segment_completion_rows[-1]["start_readback_verified_sim_step"]
        == step + 1
    )


def test_current_complete_coalesces_adjacent_same_target_start_without_batch(
    tmp_path: Path,
):
    session, adapter, decision, step = _coalesced_transition_case(
        tmp_path, same_target=True
    )
    physical_before = len(adapter.batch_calls)
    dispatch_before = len(session.dispatch_rows)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            decision,
        )
    finally:
        session.outer_render_boundary_permit = False

    assert len(adapter.batch_calls) == physical_before
    assert len(session.dispatch_rows) == dispatch_before
    assert session.pending_readback is None
    assert session.transition_rows[-1]["sim_step"] == step
    source_row = session.source_action_consumption_rows[-1]
    assert source_row["sim_step"] == step
    assert source_row["target_changed"] is False
    assert source_row["dispatch_epoch"] == 0
    assert source_row["physical_dispatch_required"] is False
    assert source_row["physical_dispatch_applied"] is False
    assert source_row["physical_dispatch_index"] is None
    completion = session.segment_completion_rows[-1]
    assert completion["owner_state"] == "S2_FR_TRAVERSE"
    assert completion["start_sim_step"] == step
    assert completion["start_physical_dispatch"] is False
    assert completion["start_readback_verified"] is True
    assert completion["start_readback_verified_sim_step"] == step
    assert completion["retained_epoch_same_target"] is True


def test_transition_source_coalescing_rejects_every_missing_contract_condition(
    tmp_path: Path,
):
    cases = (
        ("off-boundary", "current outer boundary"),
        ("missing-token", "current COMPLETE token"),
        ("stale-token", "old final COMPLETE"),
        ("open-completion", "old segment to be closed"),
        ("unclosed-row", "completion row"),
        ("wrong-edge", "adjacent graph edge"),
        ("wrong-events", "adjacent graph edge"),
        ("moving-boundary", "verified zero-wheel boundary"),
        ("pending-readback", "pending readback"),
        ("occupied-batch", "occupied batch slot"),
        ("wrong-start-owner", "next-state START control"),
        ("not-first-next-action", "first action"),
    )
    for case, message in cases:
        case_root = tmp_path / case
        case_root.mkdir()
        session, adapter, raw_decision, step = _coalesced_transition_case(
            case_root, same_target=False
        )
        decision = json.loads(json.dumps(raw_decision))
        session.outer_render_boundary_permit = True
        if case == "off-boundary":
            session.outer_render_boundary_permit = False
        elif case == "missing-token":
            session.last_segment_completion_token = None
        elif case == "stale-token":
            token = session.last_segment_completion_token
            session.last_segment_completion_token = SimpleNamespace(
                kind="COMPLETE",
                sim_step=step - 1,
                to_mapping=lambda token=token: token.to_mapping(),
            )
        elif case == "open-completion":
            session.active_segment_completion_row_index = 0
        elif case == "unclosed-row":
            session.segment_completion_rows[-1]["tracking_lifecycle_closed"] = False
        elif case == "wrong-edge":
            session.bundle.graph = SimpleNamespace(
                get=lambda _state: SimpleNamespace(
                    active_leg="", next_state="S3_FL_TRAVERSE"
                )
            )
        elif case == "wrong-events":
            decision["transition_events"] = [
                "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                "ENTER:S3_FL_TRAVERSE",
            ]
        elif case == "moving-boundary":
            moving = dict(session.last_wheel_targets)
            moving[WHEEL_JOINT_NAMES[0]] = 0.1
            session.last_wheel_targets = moving
            session.last_verified_wheel_targets = dict(moving)
            session.last_target_readback[
                "canonical_wheel_targets_rad_s"
            ] = dict(moving)
        elif case == "pending-readback":
            session.pending_readback = {"kind": "controller"}
        elif case == "occupied-batch":
            session.last_batch_attempt_sim_step = step
        elif case == "wrong-start-owner":
            decision["segment_completion_control"]["owner_state"] = (
                "S1_APPROACH_AND_PRE_FR_SHIFT"
            )
        elif case == "not-first-next-action":
            session.expected_source_actions[0]["owner_state"] = "S2_FR_TRAVERSE"
        with pytest.raises(RuntimeError, match=message):
            session._process_decision(
                adapter,
                session._observation_payload(adapter),
                decision,
            )
        assert len(session.source_action_consumption_rows) == 1
        assert len(session.dispatch_rows) == 0


def test_request_is_strict_sha_bound_and_start_binding_is_exact(tmp_path: Path):
    request, payload = _load_request(tmp_path)
    message = {
        key: value
        for key, value in {
            **payload,
            "worker_session_id": "worker-session",
        }.items()
        if key
        in {
            "request_id",
            "worker_session_id",
            "source_version",
            "profile_id",
            "graph_id",
            "graph_sha256",
            "profile_library_sha256",
            "bundle_sha256",
        }
    }
    assert validate_worker_macro_start_binding(
        request, message, expected_worker_session_id="worker-session"
    ) == []
    message["bundle_sha256"] = "d" * 64
    assert "bundle_sha256" in "; ".join(
        validate_worker_macro_start_binding(
            request, message, expected_worker_session_id="worker-session"
        )
    )
    payload["unexpected"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_worker_macro_fsm_request(bad)
    payload.pop("unexpected")
    payload["height_mm"] = 49
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="height_mm must equal 50"):
        load_worker_macro_fsm_request(bad)
    payload["height_mm"] = 50
    payload["source_version"] = "v009_unauthorized"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not authorized"):
        load_worker_macro_fsm_request(bad)


def test_request_source_trial_matrix_is_closed_for_gate_c_and_gate_d(tmp_path: Path):
    payload = _request_payload(tmp_path)
    request_path = tmp_path / "source-trial-request.json"
    for source_version in AUTHORIZED_GATE_D_SOURCE_VERSIONS:
        payload["source_version"] = source_version
        payload["trial_kind"] = GATE_D_TRIAL_KIND
        request_path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_worker_macro_fsm_request(request_path)
        assert loaded is not None
        assert loaded.source_version == source_version
        assert loaded.trial_kind == GATE_D_TRIAL_KIND

        payload["trial_kind"] = "baseline"
        request_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="require trial_kind cross_version"):
            load_worker_macro_fsm_request(request_path)

    payload["source_version"] = CANONICAL_GATE_C_SOURCE_VERSION
    payload["trial_kind"] = GATE_D_TRIAL_KIND
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline or repeat"):
        load_worker_macro_fsm_request(request_path)


def test_bundle_identity_mismatch_fails_before_controller_or_video(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    adapter = _Adapter()
    scene = SimpleNamespace(config=SimpleNamespace())
    wrong = SimpleNamespace(
        graph=_Graph(),
        profiles=_Profiles(),
        graph_sha256=_Graph.sha256,
        profile_library_sha256=_Profiles.sha256,
        bundle_sha256="d" * 64,
    )
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session",
        bundle_builder=lambda _root, _request: wrong,
        recorder_factory=_Recorder,
    )
    with pytest.raises(RuntimeError, match="bundle identity mismatch"):
        session.prepare_after_adapter(
            adapter=adapter, scene_handle=scene, project_root=tmp_path
        )
    assert adapter.telemetry_collector is None
    assert adapter.artifact_render_observer is None


def test_startup_without_exact_clean_geometry_evidence_is_no_go(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    for raw in (
        {
            "available": False,
            "dangerous_body_collision": None,
            "severe_penetration": None,
            "source": "UNAVAILABLE",
            "sample_sim_step": 0,
            "error": "geometry unavailable",
        },
        {
            "available": True,
            "dangerous_body_collision": True,
            "severe_penetration": False,
            "source": "TEST_LIVE_INITIAL_GEOMETRY",
            "sample_sim_step": 0,
            "error": "",
        },
    ):
        adapter = _Adapter()
        adapter.capture_macro_safety_evidence = lambda **_kwargs: dict(raw)
        session = WorkerMacroFSMSession(
            request,
            worker_session_id="worker-session",
            bundle_builder=lambda _root, _request: _Bundle(),
            controller_factory=lambda _bundle: _Controller(),
            recorder_factory=_Recorder,
        )
        with pytest.raises(RuntimeError, match="evidence"):
            session.prepare_after_adapter(
                adapter=adapter,
                scene_handle=SimpleNamespace(config=SimpleNamespace()),
                project_root=tmp_path,
            )
        assert adapter.telemetry_collector is None


def test_physics_tick_control_atomic_epochs_boundary_and_15hz_evidence(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, controller = _runtime(request)
    hard_safety_steps = []
    validate_hard_safety = session._validate_hard_safety

    def record_hard_safety(payload, *, startup=False):
        hard_safety_steps.append(int(payload["sim_step"]))
        return validate_hard_safety(payload, startup=startup)

    session._validate_hard_safety = record_hard_safety
    start = session.start()
    assert start["first_controller_tick_physics_step"] == 1
    assert start["earliest_profile_dispatch_physics_step"] == 8
    assert start["earliest_profile_actuation_physics_step"] == 9
    assert len(adapter.batch_calls) == 1
    boundary = adapter.batch_calls[0]
    assert boundary["source"] == "fsm50_macro_start_boundary"
    assert set(boundary["servo_targets_deg"]) == set(SERVO_JOINT_NAMES)
    assert boundary["wheel_targets_rad_s"] == {
        name: 0.0 for name in WHEEL_JOINT_NAMES
    }

    terminal = None
    for step in range(1, 124):
        adapter.sim_steps = step
        adapter.sim_time = step / 120.0
        _on_step(session, adapter, 1.0 / 120.0)
        candidate = session.after_adapter_step()
        if candidate is not None:
            terminal = candidate
            break

    assert terminal is not None
    assert terminal["type"] == "macro_fsm_complete"
    assert terminal["controller_terminal_outcome"] == "TASK_SUCCESS_POSTURE_INCOMPLETE"
    assert session.controller_tick_count == 16
    assert len(controller.reset_calls) == 1
    assert len(controller.tick_calls) == 15
    assert [call[3] for call in controller.tick_calls] == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert [
        None if call[2] is None else call[2].kind
        for call in controller.tick_calls
    ] == [None] * 14 + ["COMPLETE"]
    # One zero boundary, exactly one changed controller epoch, and one full
    # atomic terminal safe stop.  Unchanged controller decisions did not write.
    assert len(adapter.batch_calls) == 3
    dispatched = adapter.batch_calls[1]
    assert dispatched["batch_id"].endswith(":macro:000001")
    assert set(dispatched["servo_targets_deg"]) == set(SERVO_JOINT_NAMES)
    assert set(dispatched["wheel_targets_rad_s"]) == set(WHEEL_JOINT_NAMES)
    assert all(value == 1.0 for value in dispatched["servo_targets_deg"].values())
    assert all(value == 0.25 for value in dispatched["wheel_targets_rad_s"].values())
    assert session.dispatch_rows[0]["concurrent"] is True
    assert session.dispatch_rows[0]["sim_step"] == 8
    assert session.dispatch_rows[0]["n_plus_one_verified"] is True
    assert session.dispatch_rows[0]["n_plus_one_verified_sim_step"] == 9
    assert len(session.dispatch_rows[0]["n_plus_one_readback_sha256"]) == 64
    assert session.source_action_consumption_rows[0]["target_changed"] is True
    assert (
        session.source_action_consumption_rows[0]["physical_dispatch_applied"]
        is True
    )
    assert session.source_action_consumption_rows[0]["physical_dispatch_index"] == 0
    assert session.source_action_consumption_rows[0]["n_plus_one_verified"] is True
    assert (
        session.source_action_consumption_rows[0][
            "n_plus_one_readback_sha256"
        ]
        == session.dispatch_rows[0]["n_plus_one_readback_sha256"]
    )
    safe_stop = adapter.batch_calls[2]
    assert safe_stop["source"] == "fsm50_macro_safe_stop"
    assert set(safe_stop["servo_targets_deg"]) == set(SERVO_JOINT_NAMES)
    assert safe_stop["wheel_targets_rad_s"] == {
        name: 0.0 for name in WHEEL_JOINT_NAMES
    }
    assert terminal["safe_stop_status"] == "VERIFIED"
    assert terminal["safe_stop_verified"] is True
    safe_ack = terminal["safe_stop_ack"]
    safe_readback = terminal["safe_stop_readback"]
    safe_readback_sha256 = terminal["safe_stop_readback_sha256"]
    assert set(safe_readback) == {
        "sim_step",
        "command_epoch",
        "batch_id",
        "canonical_servo_targets_deg",
        "canonical_wheel_targets_rad_s",
        "servo_command_transform",
        "servo_command_transform_sha256",
        "expected_servo_drive_targets_rad",
        "actual_servo_drive_targets_rad",
        "actual_wheel_drive_targets_rad_s",
        "adapter_runtime_instance_id",
        "root_state_write_count",
        "physics_dt_s",
    }
    assert safe_readback["batch_id"] == safe_ack["batch_id"]
    assert safe_readback["sim_step"] == safe_ack["first_physics_step"]
    assert safe_readback["sim_step"] == safe_ack["applied_sim_step"] + 1
    assert safe_readback["command_epoch"] == safe_ack["recording_metadata"][
        "command_epoch"
    ]
    assert set(safe_readback["canonical_servo_targets_deg"]) == set(
        SERVO_JOINT_NAMES
    )
    assert safe_readback["canonical_wheel_targets_rad_s"] == {
        name: 0.0 for name in WHEEL_JOINT_NAMES
    }
    assert set(safe_readback["actual_servo_drive_targets_rad"]) == set(
        SERVO_JOINT_NAMES
    )
    assert set(safe_readback["expected_servo_drive_targets_rad"]) == set(
        SERVO_JOINT_NAMES
    )
    assert set(safe_readback["actual_wheel_drive_targets_rad_s"]) == set(
        WHEEL_JOINT_NAMES
    )
    assert safe_readback_sha256 == canonical_mapping_sha256(safe_readback)
    assert terminal["last_target_readback"]["sim_step"] > safe_readback["sim_step"]
    assert terminal["last_target_readback"]["batch_id"] == safe_ack["batch_id"]
    durable_result = json.loads(Path(terminal["worker_result_path"]).read_text())
    assert durable_result["safe_stop_readback"] == safe_readback
    assert durable_result["safe_stop_readback_sha256"] == safe_readback_sha256
    assert len(session.rows) == 4
    assert set(range(0, 19)).issubset(hard_safety_steps)
    completion = session.segment_completion_rows[0]
    assert completion["dynamic_servo_duration_s"] == 1.0 / 150.0
    assert completion["effective_servo_reference_velocity_deg_s"] == 150.0
    assert completion["start_sim_step"] == 8
    assert completion["start_readback_verified_sim_step"] == 9
    assert completion["terminal_sim_step"] == 16
    assert completion["tracking_begin_count"] == 1
    assert completion["tracking_end_count"] == 1
    assert len(adapter.begin_tracking_calls) == 1
    assert len(adapter.end_tracking_calls) == 1
    sparse_names = {SERVO_JOINT_NAMES[0]}
    assert set(adapter.begin_tracking_calls[0]) == sparse_names
    assert set(completion["last_decision"]["servo_errors_deg"]) == sparse_names
    active_telemetry = next(row for row in session.rows if row["sim_step"] == 16)
    assert set(active_telemetry["canonical_servo_endpoint_targets_deg"]) == set(
        SERVO_JOINT_NAMES
    )
    assert set(active_telemetry["applied_servo_drive_command_deg"]) == set(
        SERVO_JOINT_NAMES
    )
    assert set(active_telemetry["measured_servo_actual_deg"]) == set(
        SERVO_JOINT_NAMES
    )
    assert set(active_telemetry["canonical_servo_actual_error_deg"]) == set(
        SERVO_JOINT_NAMES
    )
    assert set(active_telemetry["active_completion_sparse_servo_targets_deg"]) == (
        sparse_names
    )
    assert set(
        active_telemetry["active_completion_sparse_servo_actual_error_deg"]
    ) == sparse_names
    assert session.fast_close_ready is True
    assert terminal["task_inputs"]["completed_result"]["dispatch_complete"] is True
    completed = terminal["task_inputs"]["completed_result"]
    assert completed["expected_segment_count"] == 1
    assert completed["completed_segment_count"] == 1
    assert completed["expected_event_count"] == 1
    assert completed["sent_event_count"] == 1
    assert completed["expected_step_count"] == 1
    consumption_path = Path(terminal["worker_result_path"]).with_name(
        "macro_source_action_consumption.jsonl"
    )
    assert consumption_path.is_file()
    completion_path = Path(terminal["segment_completion_path"])
    assert completion_path.is_file()
    assert terminal["segment_completion_sha256"] == hashlib.sha256(
        completion_path.read_bytes()
    ).hexdigest()
    completion_rows = [
        json.loads(line)
        for line in completion_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(completion_rows) == 1
    assert completion_rows[0]["terminal_kind"] == "COMPLETE"
    result = json.loads(Path(terminal["worker_result_path"]).read_text())
    assert result["segment_completion_count"] == 1
    assert result["segment_completion_coverage_complete"] is True
    assert result["segment_completion_path"] == str(completion_path)
    assert result["segment_completion_sha256"] == terminal[
        "segment_completion_sha256"
    ]
    ledger_bytes = completion_path.read_bytes()
    completion_path.write_bytes(ledger_bytes + b"{}\n")
    assert hashlib.sha256(completion_path.read_bytes()).hexdigest() != result[
        "segment_completion_sha256"
    ]
    # The wheels never reached TOP in this fixture: posture-incomplete success
    # must still settle and finalize as success.
    assert terminal["task_inputs"]["physical_evidence"]["final_all_top"] is False
    physical = terminal["task_inputs"]["physical_evidence"]
    for field in ("body_stuck", "active_leg_trapped"):
        assert physical[field] is None
        assert physical[f"{field}_available"] is False
        assert physical[f"{field}_source"] == (
            "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
        )
        assert physical[f"{field}_validation_source"] == (
            "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
        )
    assert physical["dangerous_collision"] is False
    assert physical["severe_penetration"] is None
    assert physical["dangerous_collision_available"] is True
    collision = physical["runtime_nonwheel_collision_evidence"]
    assert collision["coverage_complete"] is True
    assert collision["sample_count"] > 0
    assert collision["sample_count"] == collision["clear_sample_count"]
    assert collision["detected_sample_count"] == 0
    assert collision["sample_count"] == collision["published_contact_sample_epoch"]
    assert len(collision["first_sample_sha256"]) == 64
    assert len(collision["last_sample_sha256"]) == 64
    assert physical["penetration_evidence_available"] is False
    assert physical["runtime_collision_penetration_classification"] == (
        "DANGEROUS_COLLISION_CLEAR_BY_FULL_CURRENT_TICK_FILTERED_"
        "CONTACT_COVERAGE;PENETRATION_NOT_EVALUATED_REQUIRES_"
        "SHA_BOUND_FULL_VIDEO_REVIEW"
    )
    assert physical["initial_deployment_collision_penetration_clear"] is True
    assert len(physical["initial_deployment_evidence_sha256"]) == 64
    session.runtime_contact_collision_evidence["last_sample_sim_step"] += 1
    gap_physical = session._task_inputs(success=True, error="")["physical_evidence"]
    assert gap_physical["dangerous_collision"] is None
    assert gap_physical["dangerous_collision_available"] is False
    assert gap_physical["runtime_nonwheel_collision_evidence"][
        "coverage_complete"
    ] is False
    assert gap_physical["runtime_collision_penetration_classification"] == (
        "NOT_EVALUATED_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
    )


def test_changed_epoch_ack_mismatch_fails_closed(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    original = adapter.apply_motion_batch

    def wrong_ack(payload):
        result = original(payload)
        if payload.get("source") == "fsm50_macro_controller":
            result["motion_start_skew_s"] = 0.01
        return result

    adapter.apply_motion_batch = wrong_ack
    terminal = None
    for step in range(1, 11):
        adapter.sim_steps = step
        adapter.sim_time = step / 120.0
        _on_step(session, adapter, 1.0 / 120.0)
        terminal = session.after_adapter_step()
        if terminal:
            break
    assert terminal is not None
    assert terminal["type"] == "macro_fsm_failed"
    assert "skew" in terminal["error"]
    assert all(value == 0.0 for value in adapter.wheel_speeds.values())


def test_same_target_source_action_is_consumed_without_physical_dispatch(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("wheel stop",),
            sequence=0,
        ),
        "servos": zeros_s,
        "wheels": zeros_w,
    }
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    physical_before = len(adapter.batch_calls)
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=0
    )
    assert len(adapter.batch_calls) == physical_before
    assert session.command_dispatch_count == 0
    assert session.next_source_action_index == 1
    row = session.source_action_consumption_rows[0]
    assert row["target_changed"] is False
    assert row["dispatch_epoch"] == 0
    assert row["physical_dispatch_required"] is False
    assert row["physical_dispatch_applied"] is False
    assert row["physical_dispatch_index"] is None
    assert row["n_plus_one_verified"] is False
    assert row["n_plus_one_verified_sim_step"] is None
    assert row["n_plus_one_readback_sha256"] == ""
    assert row["pre_action_verified_command_epoch"] == 0
    assert len(row["pre_action_verified_readback_sha256"]) == 64
    completion = session.segment_completion_rows[0]
    assert completion["start_physical_dispatch"] is False
    assert completion["retained_epoch_same_target"] is True
    assert completion["dynamic_servo_duration_s"] == 0.0
    assert completion["start_readback_verified"] is True
    assert len(adapter.begin_tracking_calls) == 1
    complete_token = _observe_completion_boundary(
        session, adapter, step=8
    )
    assert complete_token is not None and complete_token.kind == "COMPLETE"
    assert completion["tracking_begin_count"] == 1
    assert completion["tracking_end_count"] == 1
    assert len(adapter.end_tracking_calls) == 1

    saved_servo_targets = dict(row["servo_targets_deg"])
    row["servo_targets_deg"][SERVO_JOINT_NAMES[0]] = 7.0
    row["target_changed"] = True
    row["physical_dispatch_required"] = True
    assert any(
        "canonical source action" in error
        or "physical dispatch completion flag" in error
        for error in session._source_action_coverage_errors()
    )
    row["servo_targets_deg"] = saved_servo_targets
    row["target_changed"] = False
    row["physical_dispatch_required"] = False
    assert session._source_action_coverage_errors() == []

    physical_action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("physical source action",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": zeros_w,
    }
    other, other_adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([physical_action])
    )
    _process_expected_source_action(
        other, other_adapter, other.expected_source_actions[0], step=3, epoch=1
    )
    other_adapter.sim_steps = 4
    other_adapter.sim_time = 4.0 / 120.0
    other._verify_pending_readback(other_adapter, sim_step=4)
    assert other._source_action_coverage_errors() == []
    physical_row = other.source_action_consumption_rows[0]
    saved_physical_identity = {
        key: physical_row[key]
        for key in (
            "target_changed",
            "physical_dispatch_required",
            "physical_dispatch_applied",
            "physical_dispatch_index",
            "batch_id",
            "n_plus_one_verified",
            "n_plus_one_verified_sim_step",
            "n_plus_one_readback_sha256",
        )
    }
    physical_row.update(
        target_changed=False,
        physical_dispatch_required=False,
        physical_dispatch_applied=False,
        physical_dispatch_index=None,
        batch_id="",
        n_plus_one_verified=False,
        n_plus_one_verified_sim_step=None,
        n_plus_one_readback_sha256="",
    )
    assert any(
        "target-change flag differs" in error
        for error in other._source_action_coverage_errors()
    )
    physical_row.update(saved_physical_identity)
    assert other._source_action_coverage_errors() == []

    saved_pre_epoch = physical_row["pre_action_verified_command_epoch"]
    physical_row["pre_action_verified_command_epoch"] = 999
    assert any(
        "scalar/readback identity" in error
        for error in other._source_action_coverage_errors()
    )
    physical_row["pre_action_verified_command_epoch"] = saved_pre_epoch

    saved_step = physical_row["sim_step"]
    saved_time = physical_row["sim_time_s"]
    physical_row["sim_step"] = 999
    physical_row["sim_time_s"] = 999.0 / 120.0
    physical_row["n_plus_one_verified_sim_step"] = 1000
    assert any(
        "exact source-action consumption" in error
        for error in other._dispatch_ownership_errors()
    )
    physical_row["sim_step"] = saved_step
    physical_row["sim_time_s"] = saved_time
    physical_row["n_plus_one_verified_sim_step"] = saved_physical_identity[
        "n_plus_one_verified_sim_step"
    ]

    saved_pre_action = json.loads(
        json.dumps(physical_row["pre_action_verified_readback"])
    )
    forged_pre_action = json.loads(json.dumps(saved_pre_action))
    forged_pre_action["canonical_servo_targets_deg"] = dict(
        physical_action["servos"]
    )
    transform = forged_pre_action["servo_command_transform"]
    for name in SERVO_JOINT_NAMES:
        expected_rad = math.radians(
            float(transform["standing_pose_deg_by_servo"][name])
            + float(transform["command_sign_by_servo"][name])
            * float(physical_action["servos"][name])
        )
        forged_pre_action["expected_servo_drive_targets_rad"][name] = expected_rad
        forged_pre_action["actual_servo_drive_targets_rad"][name] = expected_rad
    physical_row["pre_action_verified_readback"] = forged_pre_action
    physical_row["pre_action_verified_readback_sha256"] = canonical_mapping_sha256(
        forged_pre_action
    )
    assert any(
        "readback preimage/full identity" in error
        for error in other._source_action_coverage_errors()
    )
    physical_row["pre_action_verified_readback"] = saved_pre_action
    physical_row["pre_action_verified_readback_sha256"] = canonical_mapping_sha256(
        saved_pre_action
    )
    assert other._source_action_coverage_errors() == []
    assert other._dispatch_ownership_errors() == []
    phantom = json.loads(json.dumps(other.dispatch_rows[0]))
    phantom["dispatch_index"] = 0
    session.dispatch_rows.append(phantom)
    session.command_dispatch_count = 1
    assert any(
        "exact source-action consumption" in error
        for error in session._dispatch_ownership_errors()
    )


def test_wheel_completion_action_is_ordered_consumed_and_not_counted_as_segment(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    ones_s = {name: 1.0 for name in SERVO_JOINT_NAMES}
    moving_w = {name: 0.25 for name in WHEEL_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    actions = [
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.0,
                events=(0,),
                commands=("start wheel channel",),
                sequence=0,
            ),
            "servos": ones_s,
            "wheels": moving_w,
            "wheel_duration_s": 0.5,
            "hold_s": 0.51,
        },
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.5,
                events=(),
                commands=(),
                sequence=1,
                dispatch_kind="wheel_channel_completion_stop",
            ),
            "servos": ones_s,
            "wheels": zeros_w,
        },
    ]
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions(actions)
    )
    assert len(session.expected_source_actions) == 2

    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    adapter.sim_steps = 64
    adapter.sim_time = 64.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        due_token = session._observe_active_segment_completion(
            adapter=adapter,
            payload=session._observation_payload(adapter),
        )
    finally:
        session.outer_render_boundary_permit = False
    assert due_token is not None and due_token.kind == "WHEEL_STOP_DUE"
    session.last_segment_completion_token = due_token
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[1], step=64, epoch=2
    )
    adapter.sim_steps = 65
    adapter.sim_time = 65.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=65)
    adapter.sim_steps = 72
    adapter.sim_time = 72.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        complete_token = session._observe_active_segment_completion(
            adapter=adapter,
            payload=session._observation_payload(adapter),
        )
    finally:
        session.outer_render_boundary_permit = False
    assert complete_token is not None and complete_token.kind == "COMPLETE"

    assert session.command_dispatch_count == 2
    assert all(row["n_plus_one_verified"] for row in session.dispatch_rows)
    assert all(
        row["n_plus_one_verified"]
        for row in session.source_action_consumption_rows
    )
    assert [
        row["command_provenance"]["dispatch_kind"]
        for row in session.source_action_consumption_rows
    ] == ["segment_start", "wheel_channel_completion_stop"]
    counts = session._task_inputs(success=False, error="")["completed_result"]
    assert counts["expected_segment_count"] == 1
    assert counts["completed_segment_count"] == 1
    assert counts["expected_event_count"] == 1
    assert counts["sent_event_count"] == 1
    assert counts["expected_step_count"] == 1
    assert counts["step_count"] == 1
    assert counts["source_action_coverage_complete"] is True
    assert counts["macro_controller"]["expected_source_action_count"] == 2
    assert counts["macro_controller"]["source_action_consumption_count"] == 2


def test_generated_completion_wheel_stop_has_independent_epoch_and_n_plus_one(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    ones_s = {name: 1.0 for name in SERVO_JOINT_NAMES}
    moving_w = {name: 0.25 for name in WHEEL_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("start wheel channel",),
            sequence=0,
        ),
        "servos": ones_s,
        "wheels": moving_w,
        "completion_servo_targets": {SERVO_JOINT_NAMES[0]: 1.0},
        "wheel_duration_s": 0.5,
        "hold_s": 0.51,
    }
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    due = _observe_completion_boundary(session, adapter, step=64)
    assert due is not None and due.kind == "WHEEL_STOP_DUE"
    row = session._active_completion_row()
    control = {
        "schema_version": "fsm50.macro_segment_completion_control.v1",
        "kind": "WHEEL_STOP",
        "profile_id": row["profile_id"],
        "profile_source_version": row["profile_source_version"],
        "owner_state": row["owner_state"],
        "source_plan_sha256": row["source_plan_sha256"],
        "source_plan_payload_sha256": row["source_plan_payload_sha256"],
        "accepted_steps_sha256": row["accepted_steps_sha256"],
        "source_segment_index": row["source_segment_index"],
        "source_step_index": row["source_step_index"],
        "source_step_id": row["source_step_id"],
        "start_command_epoch": row["start_command_epoch"],
        "completion_spec": row["completion_spec"],
        "source_action_identity": "",
        "source_action": False,
        "completion_token_sha256": due.sha256,
    }
    provenance = _non_source_provenance("COMPLETION_WHEEL_STOP")

    def decision(stop_control):
        return _decision(
            state=row["owner_state"],
            epoch=2,
            changed=True,
            servos=ones_s,
            wheels=zeros_w,
            provenance=provenance,
            profile_id=row["profile_id"],
            profile_source_version=row["profile_source_version"],
            completion_control=stop_control,
        )

    tampered = dict(control)
    tampered["completion_token_sha256"] = "0" * 64
    batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="completion_token_sha256 identity"):
            session._process_decision(
                adapter,
                session._observation_payload(adapter),
                decision(tampered),
            )
        assert len(adapter.batch_calls) == batches_before
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            decision(control),
        )
    finally:
        session.outer_render_boundary_permit = False

    assert len(adapter.batch_calls) == batches_before + 1
    generated = adapter.batch_calls[-1]
    assert generated["source"] == "fsm50_macro_controller"
    assert set(generated["servo_targets_deg"]) == set(SERVO_JOINT_NAMES)
    assert generated["servo_targets_deg"] == ones_s
    assert generated["wheel_targets_rad_s"] == zeros_w
    assert generated["recording_metadata"]["command_provenance"] == provenance
    assert session.command_dispatch_count == 2
    assert len(session.source_action_consumption_rows) == 1
    assert session.next_source_action_index == 1
    assert session.dispatch_rows[-1]["command_epoch"] == 2
    assert session.dispatch_rows[-1]["sim_step"] == 64
    assert session.dispatch_rows[-1]["command_provenance"]["kind"] == (
        "COMPLETION_WHEEL_STOP"
    )
    assert row["wheel_stop"]["generated"] is True
    assert row["wheel_stop"]["source_action"] is False
    assert row["wheel_stop"]["applied_sim_step"] == 64
    assert row["wheel_stop"]["first_physics_step"] == 65
    assert row["wheel_stop"]["n_plus_one_verified"] is False

    adapter.sim_steps = 65
    adapter.sim_time = 65.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=65)
    assert row["wheel_stop"]["n_plus_one_verified"] is True
    assert row["wheel_stop"]["n_plus_one_verified_sim_step"] == 65
    assert len(row["wheel_stop"]["n_plus_one_readback_sha256"]) == 64
    complete = _observe_completion_boundary(session, adapter, step=72)
    assert complete is not None and complete.kind == "COMPLETE"
    assert row["terminal_kind"] == "COMPLETE"
    assert row["tracking_begin_count"] == 1
    assert row["tracking_end_count"] == 1
    counts = session._task_inputs(success=False, error="")["completed_result"]
    assert counts["source_action_coverage_complete"] is True
    assert counts["expected_segment_completion_count"] == 1
    assert counts["segment_completion_count"] == 1
    assert counts["physical_command_dispatch_count"] == 2


def test_shared_helper_wait_and_fail_tokens_close_tracking_exactly_once(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    servos = {name: 0.0 for name in SERVO_JOINT_NAMES}
    target_name = SERVO_JOINT_NAMES[0]
    servos[target_name] = 10.0
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("unreachable sparse endpoint",),
            sequence=0,
        ),
        "servos": servos,
        "wheels": zeros_w,
        "completion_servo_targets": {target_name: 10.0},
    }
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)

    wait = _observe_completion_boundary(session, adapter, step=8)
    assert wait is not None and wait.kind == "WAIT"
    assert wait.decision["servo_errors_deg"] == {}
    assert session.active_segment_completion_row_index == 0
    terminal_token = wait
    for step in range(16, 113, 8):
        terminal_token = _observe_completion_boundary(
            session, adapter, step=step
        )
        assert set(terminal_token.decision["servo_errors_deg"]) == {
            target_name
        }
        if terminal_token.kind == "FAIL":
            break
    assert terminal_token.kind == "FAIL"
    assert terminal_token.decision["failure_reason"] == "actuator_limit"
    assert terminal_token.decision["failure_code"] == "stalled_near_zero"
    row = session.segment_completion_rows[0]
    assert row["terminal_kind"] == "FAIL"
    assert row["tracking_begin_count"] == 1
    assert row["tracking_end_count"] == 1
    assert row["tracking_lifecycle_closed"] is True
    assert session.active_segment_completion_row_index is None
    assert len(adapter.begin_tracking_calls) == 1
    assert len(adapter.end_tracking_calls) == 1


def test_end_tracking_false_is_diagnostic_and_helper_complete_is_retained(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    adapter.end_tracking_result = {
        "ended": False,
        "reason": "production tracker keeps its aggregate window",
    }
    session.start()
    terminal, _step = _advance_until_terminal(session, adapter)
    assert terminal["type"] == "macro_fsm_complete"
    assert len(adapter.end_tracking_calls) == 1
    row = session.segment_completion_rows[0]
    assert row["terminal_kind"] == "COMPLETE"
    assert row["tracking_end_count"] == 1
    assert row["tracking_lifecycle_closed"] is True
    assert row["tracking_end_evidence"]["ended"] is False
    assert row["tracking_end_evidence"]["tracking_completion_deferred"] is True
    ledger = Path(terminal["segment_completion_ledger_path"])
    persisted = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert persisted["terminal_kind"] == "COMPLETE"
    assert persisted["tracking_end_evidence"]["ended"] is False


def test_end_tracking_malformed_or_exception_fails_once_and_persists_ledger(
    tmp_path: Path,
):
    cases = (
        ("missing", {"other": True}, None, "explicit boolean ended"),
        ("nonbool", {"ended": 1}, None, "explicit boolean ended"),
        ("exception", {"ended": True}, RuntimeError("injected end failure"), "injected end failure"),
    )
    for label, result, exception, expected_error in cases:
        root = tmp_path / label
        root.mkdir()
        request, _payload = _load_request(root)
        session, adapter, _controller = _runtime(request)
        adapter.end_tracking_result = result
        adapter.end_tracking_exception = exception
        session.start()
        terminal, _step = _advance_until_terminal(session, adapter)
        assert terminal["type"] == "macro_fsm_failed"
        assert expected_error in terminal["error"]
        assert len(adapter.end_tracking_calls) == 1
        assert len(session.segment_completion_rows) == 1
        row = session.segment_completion_rows[0]
        assert row["tracking_end_attempt_count"] == 1
        assert row["tracking_end_count"] == 0
        assert row["tracking_lifecycle_closed"] is False
        assert row["terminal_kind"] == "ABORTED"
        ledger = Path(terminal["segment_completion_ledger_path"])
        assert ledger.is_file()
        assert terminal["segment_completion_ledger_sha256"] == hashlib.sha256(
            ledger.read_bytes()
        ).hexdigest()
        persisted = json.loads(
            ledger.read_text(encoding="utf-8").splitlines()[0]
        )
        assert persisted["terminal_kind"] == "ABORTED"
        assert terminal["segment_completion_coverage_complete"] is False


def test_post_ack_completion_start_failure_defers_safe_stop_to_next_tick(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)

    def fail_executor_start(*_args, **_kwargs):
        raise RuntimeError("injected completion executor start failure")

    session.segment_completion_executor.start = fail_executor_start
    session.start()
    terminal, terminal_step = _advance_until_terminal(session, adapter)
    assert terminal["type"] == "macro_fsm_failed"
    assert "injected completion executor start failure" in terminal["error"]
    assert terminal_step == 10
    assert [call["source"] for call in adapter.batch_calls] == [
        "fsm50_macro_start_boundary",
        "fsm50_macro_controller",
        "fsm50_macro_safe_stop",
    ]
    assert adapter.batch_call_sim_steps == [0, 8, 9]
    assert len(adapter.batch_call_sim_steps) == len(
        set(adapter.batch_call_sim_steps)
    )
    assert len(adapter.begin_tracking_calls) == 1
    assert len(adapter.end_tracking_calls) == 1
    assert terminal["safe_stop_verified"] is True


def test_source_dispatch_kind_enum_and_conditional_shape_fail_closed(
    tmp_path: Path,
):
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}

    unknown_root = tmp_path / "unknown-kind"
    unknown_root.mkdir()
    request, _payload = _load_request(unknown_root)
    unknown = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("unknown action",),
            sequence=0,
            dispatch_kind="invented_dispatch",
        ),
        "servos": zeros_s,
        "wheels": zeros_w,
    }
    with pytest.raises(RuntimeError, match="dispatch_kind is invalid"):
        _runtime(request, bundle=_bundle_for_actions([unknown]))

    completion_root = tmp_path / "malformed-completion"
    completion_root.mkdir()
    request, _payload = _load_request(completion_root)
    malformed_completion = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("must be empty",),
            sequence=0,
            dispatch_kind="wheel_channel_completion_stop",
        ),
        "servos": zeros_s,
        "wheels": zeros_w,
    }
    with pytest.raises(RuntimeError, match="requires empty commands/events"):
        _runtime(request, bundle=_bundle_for_actions([malformed_completion]))

    start_root = tmp_path / "malformed-start"
    start_root.mkdir()
    request, _payload = _load_request(start_root)
    malformed_start = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(),
            commands=(),
            sequence=0,
        ),
        "servos": zeros_s,
        "wheels": zeros_w,
    }
    with pytest.raises(RuntimeError, match="requires commands and event indices"):
        _runtime(request, bundle=_bundle_for_actions([malformed_start]))


def test_source_action_duplicate_missing_out_of_order_and_malformed_fail_closed(
    tmp_path: Path,
):
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    actions = [
        {
            "provenance": _source_provenance(
                segment=index,
                step=index + 1,
                source_time_s=float(index),
                events=(0,),
                commands=(f"wheel stop {index}",),
                sequence=index,
            ),
            "servos": zeros_s,
            "wheels": zeros_w,
        }
        for index in range(2)
    ]

    out_of_order_root = tmp_path / "out-of-order"
    out_of_order_root.mkdir()
    request, _payload = _load_request(out_of_order_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions(actions)
    )
    with pytest.raises(RuntimeError, match="out of order"):
        _process_expected_source_action(
            session, adapter, session.expected_source_actions[1], step=3, epoch=0
        )
    assert session.source_action_consumption_rows == []

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    request, _payload = _load_request(duplicate_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions(actions)
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=0
    )
    with pytest.raises(RuntimeError, match="expected canonical segment 1"):
        _process_expected_source_action(
            session, adapter, session.expected_source_actions[0], step=4, epoch=0
        )
    assert len(session.source_action_consumption_rows) == 1

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    request, _payload = _load_request(malformed_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions(actions)
    )
    malformed = dict(session.expected_source_actions[0]["command_provenance"])
    malformed["source_action_identity"] = "0" * 64
    adapter.sim_steps = 3
    adapter.sim_time = 3.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="identity hash mismatch"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                epoch=0,
                servos=zeros_s,
                wheels=zeros_w,
                provenance=malformed,
            ),
        )
    assert session.source_action_consumption_rows == []


def test_source_target_change_without_provenance_and_incomplete_success_fail(
    tmp_path: Path,
):
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    ones_s = {name: 1.0 for name in SERVO_JOINT_NAMES}
    moving_w = {name: 0.25 for name in WHEEL_JOINT_NAMES}
    actions = [
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.0,
                events=(0,),
                commands=("atomic source action",),
                sequence=0,
            ),
            "servos": ones_s,
            "wheels": moving_w,
        },
        {
            "provenance": _source_provenance(
                segment=1,
                step=2,
                source_time_s=1.0,
                events=(0,),
                commands=("wheel stop",),
                sequence=1,
            ),
            "servos": ones_s,
            "wheels": zeros_w,
        },
    ]
    no_provenance_root = tmp_path / "no-provenance"
    no_provenance_root.mkdir()
    request, _payload = _load_request(no_provenance_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions(actions)
    )
    adapter.sim_steps = 3
    adapter.sim_time = 3.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="require exact command provenance"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                epoch=1,
                changed=True,
                servos=ones_s,
                wheels=moving_w,
                provenance=_non_source_provenance(),
            ),
        )

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    request, _payload = _load_request(missing_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions(actions)
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    with pytest.raises(RuntimeError, match="coverage"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                epoch=1,
                servos=ones_s,
                wheels=moving_w,
                terminal=True,
                outcome="TASK_SUCCESS_POSTURE_INCOMPLETE",
            ),
        )


def test_non_source_zero_wheel_provenance_cannot_preapply_source_motion(
    tmp_path: Path,
):
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    ones_s = {name: 1.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    moving_w = {name: 0.25 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("source motion",),
            sequence=0,
        ),
        "servos": ones_s,
        "wheels": moving_w,
    }

    retained_root = tmp_path / "forged-retained-servos"
    retained_root.mkdir()
    request, _payload = _load_request(retained_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    adapter.sim_steps = 3
    adapter.sim_time = 3.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="retain all servo targets"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state="S2_FORGED_BOUNDARY",
                epoch=1,
                changed=True,
                servos=ones_s,
                wheels=zeros_w,
                events=(
                    "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                    "ENTER:S2_FORGED_BOUNDARY",
                ),
                provenance=_non_source_provenance("BOUNDARY_ZERO_WHEELS"),
            ),
        )
    assert session.source_action_consumption_rows == []
    assert session.command_dispatch_count == 0

    shared_root = tmp_path / "source-shared-boundary-slot"
    shared_root.mkdir()
    request, _payload = _load_request(shared_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    session.last_macro_state = "S0_INITIALIZE"
    adapter.sim_steps = 3
    adapter.sim_time = 3.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="current outer boundary"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state="S1_APPROACH_AND_PRE_FR_SHIFT",
                epoch=1,
                changed=True,
                servos=ones_s,
                wheels=moving_w,
                events=(
                    "EXIT:S0_INITIALIZE",
                    "ENTER:S1_APPROACH_AND_PRE_FR_SHIFT",
                ),
                provenance=session.expected_source_actions[0][
                    "command_provenance"
                ],
            ),
        )
    assert session.source_action_consumption_rows == []
    assert session.command_dispatch_count == 0

    wheel_root = tmp_path / "forged-nonzero-wheels"
    wheel_root.mkdir()
    request, _payload = _load_request(wheel_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    adapter.sim_steps = 3
    adapter.sim_time = 3.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="zero all wheel targets"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state="S2_FORGED_BOUNDARY",
                epoch=1,
                changed=True,
                servos=zeros_s,
                wheels=moving_w,
                events=(
                    "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                    "ENTER:S2_FORGED_BOUNDARY",
                ),
                provenance=_non_source_provenance("BOUNDARY_ZERO_WHEELS"),
            ),
        )
    assert session.source_action_consumption_rows == []
    assert session.command_dispatch_count == 0

    context_root = tmp_path / "forged-context"
    context_root.mkdir()
    request, _payload = _load_request(context_root)
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    adapter.sim_steps = 5
    adapter.sim_time = 5.0 / 120.0
    _prime_current_target_readback(session, adapter)
    with pytest.raises(RuntimeError, match="exact state/event context"):
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state="S2_FORGED_BOUNDARY",
                epoch=2,
                changed=True,
                servos=ones_s,
                wheels=zeros_w,
                events=(
                    "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                    "ENTER:S2_FORGED_BOUNDARY",
                ),
                provenance=_non_source_provenance("HOLD_ZERO_WHEELS"),
            ),
        )
    assert session.command_dispatch_count == 1


def test_same_target_source_action_requires_verified_retained_epoch(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("wheel stop",),
            sequence=0,
        ),
        "servos": zeros_s,
        "wheels": zeros_w,
    }
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    expected = session.expected_source_actions[0]
    adapter.sim_steps = 3
    adapter.sim_time = 3.0 / 120.0
    _prime_current_target_readback(session, adapter)
    payload = session._observation_payload(adapter)
    session.last_verified_command_epoch = -1
    with pytest.raises(RuntimeError, match="verified retained command epoch"):
        session._process_decision(
            adapter,
            payload,
            _decision(
                state=expected["owner_state"],
                epoch=0,
                servos=zeros_s,
                wheels=zeros_w,
                provenance=expected["command_provenance"],
                completion_control=_start_completion_control(
                    provenance=expected["command_provenance"],
                    spec=expected["segment_completion_binding"]["completion_spec"],
                    epoch=0,
                    profile_id=expected["profile_id"],
                    plan_sha=expected["source_plan_sha256"],
                    owner_state=expected["owner_state"],
                ),
                profile_id=expected["profile_id"],
                profile_source_version=expected["profile_source_version"],
                profile_strategy=expected["profile_strategy"],
            ),
        )


def test_real_v003_s1_to_s2_segment7_noop_and_canonical_counts(tmp_path: Path):
    from fsm_50mm_recording_derived_v3.fsm50_macro_controller import (
        MacroFSMController,
    )
    from fsm_50mm_recording_derived_v3.fsm50_motion_profiles import (
        build_profile_library,
    )

    project_root = Path(__file__).resolve().parents[2]
    real_profiles = build_profile_library(project_root)
    bundle = SimpleNamespace(
        graph=_Graph(),
        profiles=real_profiles,
        graph_sha256=_Graph.sha256,
        profile_library_sha256=_Profiles.sha256,
        bundle_sha256=_Bundle.bundle_sha256,
        to_mapping=lambda: {"bundle_sha256": _Bundle.bundle_sha256},
    )
    request, _payload = _load_request(tmp_path)
    session, adapter = _prime_manual_session(request, bundle=bundle)
    assert len(session.expected_source_actions) == 112

    s2_profile = next(
        profile
        for profile in real_profiles.profiles
        if profile.source_version == CANONICAL_GATE_C_SOURCE_VERSION
        and profile.state_id.value == "S2_FR_TRAVERSE"
    )
    controller_provenance = MacroFSMController._source_action_provenance(
        s2_profile, s2_profile.keyframes[0]
    )
    controller_json = json.loads(json.dumps(controller_provenance))
    assert controller_json == session.expected_source_actions[7][
        "command_provenance"
    ]

    # This test isolates the real S1->S2 retained-target boundary.  The worker
    # cursor and verified adapter readback are seeded to the canonical segment
    # 6 endpoint so completion handling for segments 0..6 is not bypassed by
    # pretending that source consumption itself meant completion.
    previous = session.expected_source_actions[6]
    session.next_source_action_index = 7
    session.last_epoch = 7
    session.last_servo_targets = dict(previous["servo_targets_deg"])
    session.last_wheel_targets = dict(previous["wheel_targets_rad_s"])
    session.last_verified_servo_targets = dict(previous["servo_targets_deg"])
    session.last_verified_wheel_targets = dict(previous["wheel_targets_rad_s"])
    session.last_verified_command_epoch = 7
    adapter.joint_command_deg.update(previous["servo_targets_deg"])
    adapter.servo_applied_command_deg.update(previous["servo_targets_deg"])
    adapter.wheel_speeds.update(previous["wheel_targets_rad_s"])
    session.last_target_readback = session._capture_and_validate_target_readback(
        adapter,
        servo_targets=previous["servo_targets_deg"],
        wheel_targets=previous["wheel_targets_rad_s"],
        expected_sim_step=adapter.sim_steps,
    )
    epoch = 8
    step = 3

    segment7 = session.expected_source_actions[7]
    boundary_servos = dict(session.last_servo_targets)
    boundary_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    adapter.sim_steps = step
    adapter.sim_time = step / 120.0
    _prime_current_target_readback(session, adapter)
    session._process_decision(
        adapter,
        session._observation_payload(adapter),
        _decision(
            state="S2_FR_TRAVERSE",
            epoch=epoch,
            changed=True,
            servos=boundary_servos,
            wheels=boundary_wheels,
            events=(
                "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                "ENTER:S2_FR_TRAVERSE",
            ),
            provenance=_non_source_provenance("BOUNDARY_ZERO_WHEELS"),
            profile_id=segment7["profile_id"],
            profile_source_version=segment7["profile_source_version"],
            profile_strategy=segment7["profile_strategy"],
        ),
    )
    step += 1
    adapter.sim_steps = step
    adapter.sim_time = step / 120.0
    session._verify_pending_readback(adapter, sim_step=step)
    boundary_dispatch = session.dispatch_rows[-1]
    assert boundary_dispatch["command_provenance"]["kind"] == (
        "BOUNDARY_ZERO_WHEELS"
    )
    assert boundary_dispatch["n_plus_one_verified"] is True
    assert boundary_dispatch["n_plus_one_verified_sim_step"] == step
    assert len(boundary_dispatch["n_plus_one_readback_sha256"]) == 64
    physical_before_segment7 = len(adapter.batch_calls)

    step += 1
    _process_expected_source_action(
        session, adapter, segment7, step=step, epoch=epoch
    )
    assert len(adapter.batch_calls) == physical_before_segment7
    assert session.source_action_consumption_rows[-1]["target_changed"] is False
    assert (
        session.source_action_consumption_rows[-1]["physical_dispatch_applied"]
        is False
    )
    assert session.source_action_consumption_rows[-1]["dispatch_epoch"] == 8
    assert (
        session.source_action_consumption_rows[-1][
            "pre_action_verified_readback_sha256"
        ]
        == canonical_mapping_sha256(session.last_target_readback)
    )
    assert session.last_target_readback["sim_step"] == step
    assert [
        row["command_provenance"]["source_segment_index"]
        for row in session.source_action_consumption_rows
    ] == [7]

    counts = session._task_inputs(success=False, error="")["completed_result"]
    assert counts["expected_step_count"] == 24
    assert counts["expected_event_count"] == 160
    assert counts["expected_segment_count"] == 112
    assert counts["consumed_segment_start_count"] == 1
    assert counts["completed_segment_count"] == 0


def test_initial_rl_air_cannot_authorize_later_s8_crossing_without_lift(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.started_sim_time_s = 0.0

    # v003-like startup transient: RL is AIR before any RL active-leg phase.
    session.last_macro_state = "S0_INITIALIZE"
    adapter.robot.data.body_link_state_w[0, 2, 2] = 0.18
    adapter.sim_steps = 120
    adapter.sim_time = 1.0
    _prime_current_target_readback(session, adapter)
    startup = session._observation_payload(adapter)
    assert startup["wheel_contact_classes"]["RL"] == "AIR"
    assert session.traversal["RL"]["airborne_seen_before_crossing"] is True
    assert session.phase_traversal["RL"]["airborne_seen_before_crossing"] is False

    # It returns to support.  Entering S7 begins a new RL phase-local episode.
    adapter.robot.data.body_link_state_w[0, 2, 2] = 0.05
    adapter.sim_steps = 240
    adapter.sim_time = 2.0
    _prime_current_target_readback(session, adapter)
    session._observation_payload(adapter)
    session.last_macro_state = "S7_PRE_RL_SUPPORT_SETUP"
    adapter.sim_steps = 8400
    adapter.sim_time = 70.0
    _prime_current_target_readback(session, adapter)
    entry = session._observation_payload(adapter)
    assert entry["active_traversal_leg"] == "RL"
    assert session.phase_traversal["RL"]["airborne_seen_before_crossing"] is False

    # No S7/S8 lift occurs; RL crosses the face at ground height.  The stale
    # startup AIR must not suppress the live hard failure.
    session.last_macro_state = "S8_RL_COM_SHIFT_AND_TRAVERSE"
    adapter.robot.data.body_link_state_w[0, 2, 0] = 1.20
    adapter.sim_steps = 8520
    adapter.sim_time = 71.0
    _prime_current_target_readback(session, adapter)
    crossing = session._observation_payload(adapter)
    crossing["actuator_targets_applied"] = True
    assert crossing["wheel_drive_up_without_required_lift"] is True
    assert session.phase_traversal["RL"]["illegal_drive_up"] is True
    with pytest.raises(RuntimeError, match="phase-local lift"):
        session._validate_hard_safety(crossing)


def test_physics_dt_is_bound_at_prepare_ack_and_every_callback(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    adapter = _Adapter()
    adapter.physics_dt_s = 1.0 / 60.0
    scene = SimpleNamespace(config=SimpleNamespace())
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session",
        bundle_builder=lambda _root, _request: _Bundle(),
        controller_factory=lambda _bundle: _Controller(),
        recorder_factory=_Recorder,
    )
    with pytest.raises(RuntimeError, match="1/120"):
        session.prepare_after_adapter(
            adapter=adapter, scene_handle=scene, project_root=tmp_path
        )

    session, adapter, _controller = _runtime(request)
    session.start()
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 60.0)
    assert session.after_adapter_step() is None
    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    terminal = session.after_adapter_step()
    assert terminal["type"] == "macro_fsm_failed"
    assert "physics dt drift" in terminal["error"]
    assert terminal["task_inputs"]["completed_result"]["lifecycle"][
        "failure_kind"
    ] == "INFRASTRUCTURE"


def test_render_interval_must_be_exact_production_eight_substeps(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    adapter = _Adapter()
    adapter._render_step_timing = lambda: (7.0 / 120.0, 7)
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session",
        bundle_builder=lambda _root, _request: _Bundle(),
        controller_factory=lambda _bundle: _Controller(),
        recorder_factory=_Recorder,
    )
    with pytest.raises(RuntimeError, match="render_interval=8"):
        session.prepare_after_adapter(
            adapter=adapter,
            scene_handle=SimpleNamespace(config=SimpleNamespace()),
            project_root=tmp_path,
        )


def test_prestart_outer_callbacks_do_not_arm_but_active_cycles_stay_exact(
    tmp_path: Path,
):
    exact_root = tmp_path / "exact"
    exact_root.mkdir()
    request, _payload = _load_request(exact_root)
    session, adapter, controller = _runtime(request)
    assert session.state == "ready_for_start"

    # Reproduce the production worker ordering: before + eight callbacks +
    # after can occur repeatedly while the session is only prepared.  Those
    # ignored callbacks carry no active cadence debt into start().
    for _outer in range(2):
        session.before_adapter_step()
        assert session.outer_render_substeps_remaining == 0
        assert session.outer_render_cycle_index == -1
        for _substep in range(8):
            adapter.sim_steps += 1
            adapter.sim_time = adapter.sim_steps / 120.0
            session.on_step(adapter, 1.0 / 120.0)
        assert session.after_adapter_step() is None
    session.before_adapter_step()
    assert session.outer_render_substeps_remaining == 0

    start = session.start()
    assert start["first_controller_tick_physics_step"] == 17
    assert start["earliest_profile_dispatch_physics_step"] == 24
    session.before_adapter_step()
    for step in range(17, 25):
        adapter.sim_steps = step
        adapter.sim_time = step / 120.0
        session.on_step(adapter, 1.0 / 120.0)
    assert session.outer_render_substeps_remaining == 0
    assert session.dispatch_rows[0]["sim_step"] == 24
    assert [call[3] for call in controller.tick_calls] == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]

    # Exactly eight active callbacks permit the next outer cycle; seven do
    # not.  The pending segment-start target is still verified at exact N+1.
    session.before_adapter_step()
    for step in range(25, 32):
        adapter.sim_steps = step
        adapter.sim_time = step / 120.0
        session.on_step(adapter, 1.0 / 120.0)
    assert session.dispatch_rows[0]["n_plus_one_verified_sim_step"] == 25
    assert session.outer_render_substeps_remaining == 1
    with pytest.raises(RuntimeError, match="did not deliver exactly eight"):
        session.before_adapter_step()

    # A ninth callback without a new before hook also fails closed.  It may
    # schedule one safe-stop batch on that ninth callback, never a source
    # cursor advance or a second same-step batch.
    ninth_root = tmp_path / "ninth"
    ninth_root.mkdir()
    request, _payload = _load_request(ninth_root)
    ninth, ninth_adapter, _controller = _runtime(request)
    ninth.start()
    ninth.before_adapter_step()
    for step in range(1, 9):
        ninth_adapter.sim_steps = step
        ninth_adapter.sim_time = step / 120.0
        ninth.on_step(ninth_adapter, 1.0 / 120.0)
    assert ninth.outer_render_substeps_remaining == 0
    ninth_adapter.sim_steps = 9
    ninth_adapter.sim_time = 9.0 / 120.0
    ninth.on_step(ninth_adapter, 1.0 / 120.0)
    assert "outside before_adapter_step() cadence" in ninth.error
    assert ninth.state == "safe_stop_pending_readback"
    assert ninth_adapter.batch_call_sim_steps == [0, 8, 9]
    assert len(ninth_adapter.batch_call_sim_steps) == len(
        set(ninth_adapter.batch_call_sim_steps)
    )


def test_stopped_simulation_terminalizes_queued_same_step_stop_unverified(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    assert adapter.batch_call_sim_steps == [0]

    # A failure in the same callback as the boundary batch queues its stop to
    # preserve one-batch-per-tick.  If the app then stops, no N+1 or later
    # callback exists; fail() must durably finalize without inventing a stop.
    session._request_failure(
        "injected post-boundary failure",
        infrastructure_failure=True,
    )
    assert session.terminal_stop_request is not None
    assert session.state == "boundary_pending"
    assert adapter.batch_call_sim_steps == [0]
    terminal = session.fail(
        "simulation app stopped",
        infrastructure_failure=True,
        simulation_app_stopped=True,
    )
    assert terminal is not None
    assert terminal["type"] == "macro_fsm_failed"
    assert terminal["safe_stop_status"] == (
        "SAFE_STOP_NOT_APPLIED_SIMULATION_STOPPED"
    )
    assert terminal["safe_stop_verified"] is False
    assert "applied or verified" in terminal["safe_stop_error"]
    assert "SAFE_STOP_UNVERIFIED" in terminal["error"]
    assert adapter.batch_call_sim_steps == [0]
    assert session.pending_readback is None
    assert session.terminal_stop_request is None
    assert session.fast_close_ready is True
    assert Path(terminal["worker_result_path"]).is_file()
    ledger = Path(terminal["segment_completion_ledger_path"])
    assert ledger.is_file()
    assert terminal["segment_completion_ledger_sha256"] == hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()
    assert terminal["segment_completion_coverage_complete"] is False
    assert terminal["segment_completion_coverage_errors"]
    assert terminal["task_inputs"]["completed_result"][
        "simulation_app_stopped"
    ] is True
    assert session.fail("duplicate close", simulation_app_stopped=True) == terminal


def test_effective_servo_rate_is_strict_finite_positive_and_gate_locked(
    tmp_path: Path,
):
    del tmp_path
    resolve = WorkerMacroFSMSession._effective_servo_reference_velocity_deg_s
    for limit in (None, 150.0, 200.0):
        adapter = SimpleNamespace(
            motion_reference=SimpleNamespace(
                servo_reference_velocity_deg_s=150.0,
                servo_velocity_limit_deg_s=limit,
            )
        )
        assert resolve(adapter) == 150.0

    invalid_references = (True, "150", float("nan"), float("inf"), 0.0, -1.0)
    for reference in invalid_references:
        adapter = SimpleNamespace(
            motion_reference=SimpleNamespace(
                servo_reference_velocity_deg_s=reference,
                servo_velocity_limit_deg_s=None,
            )
        )
        with pytest.raises(RuntimeError, match="finite and positive"):
            resolve(adapter)

    for limit in (True, "150", float("nan"), float("inf"), 0.0, -1.0):
        adapter = SimpleNamespace(
            motion_reference=SimpleNamespace(
                servo_reference_velocity_deg_s=150.0,
                servo_velocity_limit_deg_s=limit,
            )
        )
        with pytest.raises(RuntimeError, match="finite and positive"):
            resolve(adapter)

    missing_reference = SimpleNamespace(motion_reference=SimpleNamespace())
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve(missing_reference)
    with pytest.raises(RuntimeError, match="unchanged effective 150"):
        resolve(
            SimpleNamespace(
                motion_reference=SimpleNamespace(
                    servo_reference_velocity_deg_s=150.0,
                    servo_velocity_limit_deg_s=75.0,
                )
            )
        )


def test_shift_air_that_returns_ground_cannot_authorize_traverse_crossing(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.started_sim_time_s = 0.0

    session.last_macro_state = "S1_APPROACH_AND_PRE_FR_SHIFT"
    adapter.robot.data.body_link_state_w[0, 1, 2] = 0.18
    adapter.sim_steps = 1200
    adapter.sim_time = 10.0
    _prime_current_target_readback(session, adapter)
    session._observation_payload(adapter)
    assert session.phase_traversal["FR"]["airborne_seen_before_crossing"] is True

    # FR is back on support before the S2 entry boundary.  State-local reset
    # must drop the earlier transient rather than sharing it across S1/S2.
    adapter.robot.data.body_link_state_w[0, 1, 2] = 0.05
    session.last_macro_state = "S2_FR_TRAVERSE"
    adapter.sim_steps = 1320
    adapter.sim_time = 11.0
    _prime_current_target_readback(session, adapter)
    session._observation_payload(adapter)
    assert session.phase_traversal["FR"]["airborne_seen_before_crossing"] is False
    assert session.phase_traversal["FR"]["phase_entry_state"] == "S2_FR_TRAVERSE"

    adapter.robot.data.body_link_state_w[0, 1, 0] = 1.20
    adapter.sim_steps = 1440
    adapter.sim_time = 12.0
    _prime_current_target_readback(session, adapter)
    crossing = session._observation_payload(adapter)
    crossing["actuator_targets_applied"] = True
    assert crossing["wheel_drive_up_without_required_lift"] is True
    with pytest.raises(RuntimeError, match="phase-local lift"):
        session._validate_hard_safety(crossing)


def test_transition_tick_live_air_seeds_next_state_without_one_tick_gap(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.started_sim_time_s = 0.0
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    session.last_macro_state = "S1_APPROACH_AND_PRE_FR_SHIFT"
    session.active_traversal_state = session.last_macro_state
    session.active_traversal_leg = "FR"
    session.last_epoch = 0
    session.last_servo_targets = zeros_s
    session.last_wheel_targets = zeros_w

    # AIR exists on the exact S1->S2 transition observation.
    adapter.robot.data.body_link_state_w[0, 1, 2] = 0.18
    adapter.sim_steps = 20
    adapter.sim_time = 20.0 / 120.0
    _prime_current_target_readback(session, adapter)
    boundary_payload = session._observation_payload(adapter)
    session._process_decision(
        adapter,
        boundary_payload,
        _decision(
            state="S2_FR_TRAVERSE",
            servos=zeros_s,
            wheels=zeros_w,
            events=(
                "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                "ENTER:S2_FR_TRAVERSE",
            ),
        ),
    )
    assert session.active_traversal_state == "S2_FR_TRAVERSE"
    assert session.phase_traversal["FR"]["airborne_seen_before_crossing"] is True

    # It is grounded on the next tick; the transition-tick seed must survive
    # because S2 was already initialized from the same payload as controller.
    adapter.robot.data.body_link_state_w[0, 1, 2] = 0.05
    adapter.sim_steps = 21
    adapter.sim_time = 21.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session._observation_payload(adapter)
    assert session.phase_traversal["FR"]["airborne_seen_before_crossing"] is True

    adapter.robot.data.body_link_state_w[0, 1, 0] = 1.20
    adapter.sim_steps = 22
    adapter.sim_time = 22.0 / 120.0
    _prime_current_target_readback(session, adapter)
    crossing = session._observation_payload(adapter)
    assert crossing["wheel_drive_up_without_required_lift"] is False
    assert session.phase_traversal["FR"]["front_face_crossing_s"] is not None


def test_real_observation_contract_accepts_only_complete_adapter_snapshot(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    adapter = _Adapter()
    scene = SimpleNamespace(
        config=SimpleNamespace(
            obstacle_front_x=1.0,
            obstacle_height_m=0.05,
            obstacle_length=1.0,
                obstacle_width=2.0,
                ground_z_m=0.0,
                ground_static_friction=1.25,
                ground_dynamic_friction=1.05,
                obstacle_static_friction=1.20,
                obstacle_dynamic_friction=1.00,
            )
    )
    controller = _Controller()
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session",
        bundle_builder=lambda _root, _request: _Bundle(),
        controller_factory=lambda _bundle: controller,
        recorder_factory=_Recorder,
    )
    _bind_fake_contact_scene(session, scene, request)
    session.prepare_after_adapter(
        adapter=adapter, scene_handle=scene, project_root=tmp_path
    )
    session.start()
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.state == "running"
    assert isinstance(controller.reset_calls[0][0], MacroObservation)


def test_controller_exception_uses_full_atomic_stop_and_n_plus_1_verification(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(
        request, controller=_ControllerException()
    )
    session.start()

    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.after_adapter_step() is None
    assert session.state == "safe_stop_pending_readback"
    assert [row["source"] for row in adapter.batch_calls] == [
        "fsm50_macro_start_boundary",
        "fsm50_macro_safe_stop",
    ]
    stop = adapter.batch_calls[-1]
    assert set(stop["servo_targets_deg"]) == set(SERVO_JOINT_NAMES)
    assert stop["wheel_targets_rad_s"] == {
        name: 0.0 for name in WHEEL_JOINT_NAMES
    }

    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    terminal = session.after_adapter_step()
    assert terminal["type"] == "macro_fsm_failed"
    assert "injected controller exception" in terminal["error"]
    assert terminal["safe_stop_status"] == "VERIFIED"
    assert terminal["safe_stop_verified"] is True
    assert terminal["safe_stop_readback"]["batch_id"] == terminal[
        "safe_stop_ack"
    ]["batch_id"]
    assert terminal["safe_stop_readback"]["sim_step"] == terminal[
        "safe_stop_ack"
    ]["applied_sim_step"] + 1
    assert terminal["safe_stop_readback_sha256"] == canonical_mapping_sha256(
        terminal["safe_stop_readback"]
    )


def test_pending_n_plus_1_target_mismatch_fails_closed_then_verifies_stop(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    adapter.readback_target_mismatch = True
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.state == "safe_stop_pending_readback"
    assert "canonical servo target drift" in session.error

    adapter.readback_target_mismatch = False
    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    terminal = session.after_adapter_step()
    assert terminal["type"] == "macro_fsm_failed"
    assert terminal["safe_stop_verified"] is True


def test_missing_n_plus_1_readback_fails_closed_then_verifies_stop(tmp_path: Path):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    adapter.readback_unavailable = True
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.state == "safe_stop_pending_readback"
    assert "readback unavailable" in session.error

    adapter.readback_unavailable = False
    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    terminal = session.after_adapter_step()
    assert terminal["safe_stop_verified"] is True
    assert terminal["type"] == "macro_fsm_failed"


def test_safe_stop_apply_or_ack_failure_never_claims_verified(tmp_path: Path):
    for mode in ("apply", "ack"):
        case = tmp_path / mode
        case.mkdir()
        request, _payload = _load_request(case)
        session, adapter, _controller = _runtime(
            request, controller=_ControllerException()
        )
        session.start()
        adapter.safe_stop_failure = mode
        adapter.sim_steps = 1
        adapter.sim_time = 1.0 / 120.0
        _on_step(session, adapter, 1.0 / 120.0)
        terminal = session.after_adapter_step()
        assert terminal["type"] == "macro_fsm_failed"
        assert terminal["safe_stop_status"] == "SAFE_STOP_APPLICATION_FAILED"
        assert terminal["safe_stop_verified"] is False
        assert terminal["safe_stop_readback"] == {}
        assert terminal["safe_stop_readback_sha256"] == ""
        assert "SAFE_STOP_APPLICATION_FAILED" in terminal["error"]


def test_root_write_drift_enters_safe_stop_but_cannot_be_falsely_verified(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    adapter.root_state_write_count = 1
    adapter.root_state_write_events = [{"operation": "injected", "sim_step": 1}]
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.state == "safe_stop_pending_readback"
    assert "root_state_write_count" in session.error

    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    terminal = session.after_adapter_step()
    assert terminal["type"] == "macro_fsm_failed"
    assert terminal["safe_stop_status"] == "SAFE_STOP_READBACK_FAILED"
    assert terminal["safe_stop_verified"] is False
    assert terminal["safe_stop_readback"] == {}
    assert terminal["safe_stop_readback_sha256"] == ""


def test_runtime_collision_unknown_stays_none_and_true_hard_stops_same_tick(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session.start()
    unknown = session._observation_payload(adapter)
    assert unknown["dangerous_body_collision"] is False
    assert unknown["severe_penetration"] is None
    assert unknown["runtime_safety_evidence"]["available"] is False
    assert len(
        unknown["runtime_safety_evidence"]["combined_contact_sample_sha256"]
    ) == 64
    for field in ("body_stuck", "active_leg_trapped"):
        assert unknown[field] is None
        assert unknown[f"{field}_available"] is False
        assert unknown[f"{field}_source"] == (
            "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
        )
        assert unknown[f"{field}_sample_sim_step"] == adapter.sim_steps

    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: {
        "available": True,
        "dangerous_body_collision": True,
        "severe_penetration": False,
        "source": "TEST_RUNTIME_SENSOR",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
    }
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    assert session.state == "safe_stop_pending_readback"
    assert "dangerous body collision" in session.error
    assert session.dangerous_collision_detected is True
    adapter.sim_steps = 2
    adapter.sim_time = 2.0 / 120.0
    _on_step(session, adapter, 1.0 / 120.0)
    terminal = session.after_adapter_step()
    assert terminal["type"] == "macro_fsm_failed"
    physical = terminal["task_inputs"]["physical_evidence"]
    assert physical["dangerous_collision"] is True
    assert physical["dangerous_collision_available"] is True
    assert "TEST_RUNTIME_SENSOR" in physical[
        "dangerous_collision_validation_source"
    ]
    detection = physical["dangerous_collision_detection_evidence"]
    assert detection["sample_sim_step"] == 1
    assert "TEST_RUNTIME_SENSOR" in detection["source"]
    assert len(detection["runtime_safety_evidence_sha256"]) == 64


def test_optional_body_and_trapped_live_producer_true_hard_stops_same_tick(
    tmp_path: Path,
):
    for field in ("body_stuck", "active_leg_trapped"):
        case = tmp_path / field
        case.mkdir()
        request, _payload = _load_request(case)
        session, adapter, _controller = _runtime(request)
        session.start()

        def capture(**_kwargs):
            row = {}
            for name in ("body_stuck", "active_leg_trapped"):
                row[name] = name == field
                row[f"{name}_available"] = True
                row[f"{name}_source"] = "TEST_LIVE_RUNTIME_PRODUCER"
                row[f"{name}_sample_sim_step"] = adapter.sim_steps
            return row

        adapter.capture_macro_runtime_safety_evidence = capture
        adapter.sim_steps = 1
        adapter.sim_time = 1.0 / 120.0
        _on_step(session, adapter, 1.0 / 120.0)
        assert session.state == "safe_stop_pending_readback"
        assert field.replace("_", " ") in session.error
        assert adapter.batch_calls[-1]["source"] == "fsm50_macro_safe_stop"
        assert (
            session.body_stuck_detected
            if field == "body_stuck"
            else session.active_leg_trapped_detected
        ) is True


@pytest.mark.parametrize("field", ("body_stuck", "active_leg_trapped"))
def test_optional_true_is_sticky_before_peer_or_later_observation_failure(
    tmp_path: Path,
    field: str,
):
    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)

    def complete_row(*, true_field: str | None = None):
        row = {}
        for name in ("body_stuck", "active_leg_trapped"):
            row[name] = name == true_field
            row[f"{name}_available"] = True
            row[f"{name}_source"] = "TEST_OPTIONAL_STICKY_PROVIDER"
            row[f"{name}_sample_sim_step"] = adapter.sim_steps
        return row

    adapter.capture_macro_runtime_safety_evidence = (
        lambda **_kwargs: complete_row()
    )
    session.start()
    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session._refresh_filtered_contact_evidence(adapter)
    if field == "body_stuck":
        malformed = complete_row(true_field=field)
        malformed.pop("active_leg_trapped_source")
        adapter.capture_macro_runtime_safety_evidence = (
            lambda **_kwargs: malformed
        )
        with pytest.raises(RuntimeError, match="producer contract is incomplete"):
            session._observation_payload(adapter)
    else:
        adapter.capture_macro_runtime_safety_evidence = (
            lambda **_kwargs: complete_row(true_field=field)
        )

        def fail_root(_adapter):
            raise RuntimeError("injected later root-state failure")

        session._root_and_joint_state = fail_root
        with pytest.raises(RuntimeError, match="later root-state failure"):
            session._observation_payload(adapter)
    physical = session._task_inputs(success=False, error="later observation failure")[
        "physical_evidence"
    ]
    assert physical[field] is True
    assert physical[f"{field}_available"] is True
    assert physical[f"{field}_detected_sample_count"] == 1


@pytest.mark.parametrize("field", ("body_stuck", "active_leg_trapped"))
def test_optional_true_is_latched_before_provider_normalization_failure(
    tmp_path: Path,
    field: str,
):
    class _ExplodingValue:
        @property
        def value(self):
            raise RuntimeError("intentional optional-provider normalization failure")

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)

    def complete_row(*, true_field: str | None = None):
        row = {}
        for name in ("body_stuck", "active_leg_trapped"):
            row[name] = name == true_field
            row[f"{name}_available"] = True
            row[f"{name}_source"] = "TEST_OPTIONAL_PRENORMALIZATION_PROVIDER"
            row[f"{name}_sample_sim_step"] = adapter.sim_steps
        return row

    adapter.capture_macro_runtime_safety_evidence = (
        lambda **_kwargs: complete_row()
    )
    session.start()
    session._refresh_filtered_contact_evidence(adapter)
    row = complete_row(true_field=field)
    row["diagnostic"] = _ExplodingValue()
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: row
    with pytest.raises(RuntimeError, match="normalization failure"):
        session._capture_runtime_safety_evidence(adapter)

    physical = session._task_inputs(success=False, error="normalization failure")[
        "physical_evidence"
    ]
    assert physical[field] is True
    assert physical[f"{field}_available"] is True
    assert physical[f"{field}_complete_live_coverage"] is False
    claim = physical[f"{field}_unverified_true_claim"]
    assert claim["field"] == field
    assert claim["classification"] == (
        "UNVERIFIED_PROVIDER_" + field.upper() + "_TRUE_CLAIM"
    )
    assert claim["reported_sample_sim_step"] == adapter.sim_steps
    assert claim["reported_source"] == (
        "TEST_OPTIONAL_PRENORMALIZATION_PROVIDER"
    )
    assert "normalization failure" in claim["provider_payload_sha256_error"]


@pytest.mark.parametrize("field", ("body_stuck", "active_leg_trapped"))
def test_later_verified_optional_true_does_not_replace_first_unverified_hit(
    tmp_path: Path,
    field: str,
):
    class _ExplodingValue:
        @property
        def value(self):
            raise RuntimeError("first optional sample normalization failed")

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)

    def optional_row(*, source: str, exploding: bool):
        row = {
            "available": True,
            "dangerous_body_collision": False,
            "severe_penetration": False,
            "source": source,
            "sample_sim_step": adapter.sim_steps,
            "error": "",
        }
        for name in ("body_stuck", "active_leg_trapped"):
            row[name] = name == field
            row[f"{name}_available"] = True
            row[f"{name}_source"] = source
            row[f"{name}_sample_sim_step"] = adapter.sim_steps
        if exploding:
            row["diagnostic"] = _ExplodingValue()
        return row

    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: optional_row(
        source="FIRST_UNVERIFIED_OPTIONAL_HIT", exploding=True
    )
    with pytest.raises(RuntimeError, match="first optional sample"):
        session._capture_runtime_safety_evidence(adapter)
    first_claim = dict(session.unverified_optional_runtime_true_claims[field])

    adapter.sim_steps = 1
    adapter.sim_time = 1.0 / 120.0
    session._refresh_filtered_contact_evidence(adapter)
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: optional_row(
        source="LATER_VERIFIED_OPTIONAL_HIT", exploding=False
    )
    evidence = session._capture_runtime_safety_evidence(adapter)
    normalized = session._normalize_optional_runtime_bool(
        evidence,
        field=field,
        expected_sim_step=adapter.sim_steps,
    )
    assert normalized[field] is True
    assert session.unverified_optional_runtime_true_claims[field] == first_claim
    detection = session.optional_runtime_evidence[field]["first_detection_evidence"]
    assert detection["sample_sim_step"] == 1
    assert detection["source"] == "LATER_VERIFIED_OPTIONAL_HIT"
    physical = session._task_inputs(success=False, error="optional safety hit")[
        "physical_evidence"
    ]
    assert physical[f"{field}_validation_source"] == (
        "FIRST_UNVERIFIED_OPTIONAL_HIT"
    )
    assert physical[f"{field}_unverified_true_claim"] == first_claim


@pytest.mark.parametrize(
    "field",
    (
        "dangerous_body_collision",
        "severe_penetration",
        "body_stuck",
        "active_leg_trapped",
    ),
)
def test_scalar_wrapper_true_is_latched_before_sibling_normalization_failure(
    tmp_path: Path,
    field: str,
):
    class _ExplodingValue:
        @property
        def value(self):
            raise RuntimeError("intentional wrapped-true normalization failure")

    request, _payload = _load_request(tmp_path)
    session, adapter, _controller = _runtime(request)
    session._refresh_filtered_contact_evidence(adapter)
    row = {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": False,
        "source": "WRAPPED_TRUE_PROVIDER",
        "sample_sim_step": adapter.sim_steps,
        "error": "",
        "body_stuck": False,
        "body_stuck_available": True,
        "body_stuck_source": "WRAPPED_TRUE_PROVIDER",
        "body_stuck_sample_sim_step": adapter.sim_steps,
        "active_leg_trapped": False,
        "active_leg_trapped_available": True,
        "active_leg_trapped_source": "WRAPPED_TRUE_PROVIDER",
        "active_leg_trapped_sample_sim_step": adapter.sim_steps,
    }
    row[field] = np.bool_(True)
    row["diagnostic"] = _ExplodingValue()
    adapter.capture_macro_runtime_safety_evidence = lambda **_kwargs: row
    with pytest.raises(RuntimeError, match="wrapped-true normalization failure"):
        session._capture_runtime_safety_evidence(adapter)
    physical = session._task_inputs(success=False, error="normalization failure")[
        "physical_evidence"
    ]
    if field == "dangerous_body_collision":
        assert physical["dangerous_collision"] is None
        claim = physical["unverified_provider_collision_claim"]
    elif field == "severe_penetration":
        assert physical["severe_penetration"] is True
        claim = physical["unverified_provider_penetration_claim"]
    else:
        assert physical[field] is True
        claim = physical[f"{field}_unverified_true_claim"]
    assert claim["reported_sample_sim_step"] == adapter.sim_steps
    assert claim["reported_source"] == "WRAPPED_TRUE_PROVIDER"


def test_optional_runtime_producer_contract_is_strict_and_fail_closed(
    tmp_path: Path,
):
    malformed_rows = (
        {"body_stuck": False},
        {
            "body_stuck": False,
            "body_stuck_available": 1,
            "body_stuck_source": "TEST",
            "body_stuck_sample_sim_step": 1,
        },
        {
            "active_leg_trapped": False,
            "active_leg_trapped_available": True,
            "active_leg_trapped_source": "TEST",
            "active_leg_trapped_sample_sim_step": 999,
        },
        {
            "active_leg_trapped": False,
            "active_leg_trapped_available": False,
            "active_leg_trapped_source": (
                "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
            ),
            "active_leg_trapped_sample_sim_step": 1,
        },
    )
    for index, malformed in enumerate(malformed_rows):
        case = tmp_path / f"malformed-{index}"
        case.mkdir()
        request, _payload = _load_request(case)
        session, adapter, _controller = _runtime(request)
        session.start()
        adapter.capture_macro_runtime_safety_evidence = (
            lambda **_kwargs: dict(malformed)
        )
        adapter.sim_steps = 1
        adapter.sim_time = 1.0 / 120.0
        _on_step(session, adapter, 1.0 / 120.0)
        assert session.state == "safe_stop_pending_readback"
        assert "runtime" in session.error
        assert adapter.batch_calls[-1]["source"] == "fsm50_macro_safe_stop"


@pytest.mark.parametrize(
    "penetration_m",
    (
        0.00105,
        0.0010487644,  # Real v003 pre-action ground-evidence value.
    ),
)
def test_grounded_rest_failure_and_high_qd_do_not_block_safe_geometry_admission(
    tmp_path: Path,
    penetration_m: float,
):
    request, _payload = _load_request(tmp_path)
    adapter = _Adapter()
    adapter.capture_macro_safety_evidence = None
    adapter.grounded_reference_valid = False
    adapter.robot.data.joint_vel[:] = 0.5
    adapter.validate_robot_ground_contact = lambda **_kwargs: {
        "maximum_collision_penetration_m": penetration_m,
        "strict_rest_valid": False,
        "strict_rest_failure": "servo_qd_above_0.02",
    }
    scene = SimpleNamespace(
        config=SimpleNamespace(
            # Deliberately conflicting and non-authoritative.  Macro
            # admission must bind the adapter's deployed tolerance instead.
            robot_ground_penetration_tolerance_m=0.0001,
            obstacle_front_x=1.0,
            obstacle_height_m=0.05,
            obstacle_length=1.0,
            obstacle_width=2.0,
            ground_z_m=0.0,
        )
    )
    geometry = {
        "available": True,
        "robot_root_pose": [0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
        "obstacle_bounds_min_m": [1.0, -1.0, 0.0],
        "obstacle_bounds_max_m": [2.0, 1.0, 0.05],
        "robot_collision_bounds_min_m": [0.2, -0.3, 0.0],
        "robot_collision_bounds_max_m": [0.85, 0.3, 0.3],
        "robot_collision_front_to_obstacle_front_m": 0.15,
        "wheel_collision_centers": [
            {
                "prim_path": f"/Robot/{leg}/wheel_collision",
                "center_m": [0.4, 0.0, 0.05],
                "bounds_min_m": [0.35, -0.05, 0.0],
                "bounds_max_m": [0.45, 0.05, 0.1],
                "radius_m": 0.05,
            }
            for leg in ("FL", "FR", "RL", "RR")
        ],
    }
    session = WorkerMacroFSMSession(
        request,
        worker_session_id="worker-session",
        bundle_builder=lambda _root, _request: _Bundle(),
        controller_factory=lambda _bundle: _Controller(),
        recorder_factory=_Recorder,
    )
    with mock.patch(
        "sim_obstacle_scene.measure_scene_baseline", return_value=geometry
    ):
        session.prepare_after_adapter(
            adapter=adapter, scene_handle=scene, project_root=tmp_path
        )
    assert session.state == "ready_for_start"
    evidence = session.deployment_safety_evidence
    assert evidence["available"] is True
    assert evidence["dangerous_body_collision"] is False
    assert evidence["severe_penetration"] is False
    assert evidence["ground_diagnostics"]["strict_rest_valid"] is False
    assert evidence["initial_maximum_ground_penetration_m"] == penetration_m
    assert evidence["ground_penetration_tolerance_m"] == 0.003
    assert evidence["ground_penetration_tolerance_source"] == (
        "adapter.config.ground_penetration_tolerance_m"
    )
    assert max(abs(value) for value in evidence["initial_joint_velocity_rad_s"].values()) == 0.5


def test_initial_geometry_or_penetration_absence_still_fails_admission(
    tmp_path: Path,
):
    for index, (geometry_available, penetration, tolerance) in enumerate(
        (
            (False, 0.0, 0.003),
            (True, 0.00301, 0.003),
            (True, 0.00105, None),
            (True, 0.00105, float("nan")),
            (True, 0.00105, float("inf")),
            (True, 0.00105, 0.0),
            (True, 0.00105, -0.003),
        )
    ):
        case = tmp_path / f"admission-{index}"
        case.mkdir()
        request, _payload = _load_request(case)
        adapter = _Adapter()
        adapter.capture_macro_safety_evidence = None
        adapter.config = (
            SimpleNamespace()
            if tolerance is None
            else SimpleNamespace(ground_penetration_tolerance_m=tolerance)
        )
        adapter.validate_robot_ground_contact = lambda **_kwargs: {
            "maximum_collision_penetration_m": penetration
        }
        scene = SimpleNamespace(
            # This field must never rescue a missing/invalid adapter binding.
            config=SimpleNamespace(robot_ground_penetration_tolerance_m=0.003)
        )
        geometry = {
            "available": geometry_available,
            "robot_root_pose": [0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "obstacle_bounds_min_m": [1.0, -1.0, 0.0],
            "obstacle_bounds_max_m": [2.0, 1.0, 0.05],
            "robot_collision_bounds_min_m": [0.2, -0.3, 0.0],
            "robot_collision_bounds_max_m": [0.85, 0.3, 0.3],
            "robot_collision_front_to_obstacle_front_m": 0.15,
            "wheel_collision_centers": [
                {
                    "center_m": [0.4, 0.0, 0.05],
                    "bounds_min_m": [0.35, -0.05, 0.0],
                    "bounds_max_m": [0.45, 0.05, 0.1],
                    "radius_m": 0.05,
                }
                for _leg in ("FL", "FR", "RL", "RR")
            ],
        }
        session = WorkerMacroFSMSession(
            request,
            worker_session_id="worker-session",
            bundle_builder=lambda _root, _request: _Bundle(),
            controller_factory=lambda _bundle: _Controller(),
            recorder_factory=_Recorder,
        )
        with mock.patch(
            "sim_obstacle_scene.measure_scene_baseline", return_value=geometry
        ):
            with pytest.raises(RuntimeError, match="evidence"):
                session.prepare_after_adapter(
                    adapter=adapter,
                    scene_handle=scene,
                    project_root=case,
                )


def test_residual_hook_default_is_disabled_and_preserves_legacy_ledger_shape(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("nominal command",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.25 for name in WHEEL_JOINT_NAMES},
    }
    session, adapter = _prime_manual_session(
        request, bundle=_bundle_for_actions([action])
    )
    assert session.residual_enabled is False
    assert "direct_command_residual" not in session.status_dict()
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    assert len(adapter.batch_calls) == 2
    assert adapter.batch_calls[-1]["servo_targets_deg"] == action["servos"]
    assert adapter.batch_calls[-1]["wheel_targets_rad_s"] == action["wheels"]
    assert not any(
        key.startswith(("residual_", "physical_command_", "nominal_target_"))
        for key in adapter.batch_calls[-1]["recording_metadata"]
    )
    assert "physical_command_epoch" not in session.dispatch_rows[-1]
    assert "effective_completion_servo_targets_deg" not in (
        session.segment_completion_rows[-1]
    )


def test_zero_residual_is_exact_nominal_identity_without_an_extra_batch(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("identity command",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.25 for name in WHEEL_JOINT_NAMES},
        "completion_servo_targets": {SERVO_JOINT_NAMES[0]: 1.0},
    }
    provider = _ResidualContractProvider()
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=ZeroResidualPolicy(),
        residual_contract_provider=provider,
    )
    assert provider.contexts == []
    batches_before = len(adapter.batch_calls)
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    assert len(adapter.batch_calls) == batches_before + 1
    assert adapter.batch_calls[-1]["servo_targets_deg"] == action["servos"]
    assert adapter.batch_calls[-1]["wheel_targets_rad_s"] == action["wheels"]
    assert session.last_applied_servo_targets == action["servos"]
    assert session.last_applied_wheel_targets == action["wheels"]
    assert session.last_residual_transform["zero_identity"] is True
    assert session.last_residual_transform["applied_residual"] == list(
        ZERO_RESIDUAL_ACTION
    )
    assert session.physical_command_epoch == 1
    assert session.last_epoch == 1

    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    batch_count = len(adapter.batch_calls)
    adapter.sim_steps = 8
    adapter.sim_time = 8.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                epoch=1,
                servos=action["servos"],
                wheels=action["wheels"],
            ),
        )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batch_count
    assert session.command_dispatch_count == 1
    assert session.residual_transform_count == 2
    assert all(context["outer_render_boundary_permit"] for context in provider.contexts)


def test_zero_residual_empty_profile_non_source_boundary_uses_contract_identity_without_batch(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("identity before profile-free boundary",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.25 for name in WHEEL_JOINT_NAMES},
    }
    policy = _ResidualPolicy()
    provider = _ResidualContractProvider(
        empty_profile_strategy="NO_ACTIVE_PROFILE"
    )
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=policy,
        residual_contract_provider=provider,
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)

    batch_count = len(adapter.batch_calls)
    physical_epoch = session.physical_command_epoch
    transform_count = session.residual_transform_count
    adapter.sim_steps = 8
    adapter.sim_time = 8.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                epoch=1,
                servos=action["servos"],
                wheels=action["wheels"],
                profile_id="",
                profile_source_version="",
                profile_strategy="",
            ),
        )
    finally:
        session.outer_render_boundary_permit = False

    assert provider.contexts[-1]["command_provenance"]["kind"] == "NONE"
    assert provider.contexts[-1]["profile_strategy"] == ""
    assert policy.observations[-1]["contract"]["profile_strategy"] == (
        "NO_ACTIVE_PROFILE"
    )
    assert len(adapter.batch_calls) == batch_count
    assert session.physical_command_epoch == physical_epoch
    assert session.residual_transform_count == transform_count + 1
    assert session.last_residual_transform["profile_strategy"] == (
        "NO_ACTIVE_PROFILE"
    )
    assert session.last_residual_transform["zero_identity"] is True
    assert session.last_applied_servo_targets == action["servos"]
    assert session.last_applied_wheel_targets == action["wheels"]


def test_source_action_rejects_provider_profile_strategy_rebinding(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("source profile binding must remain exact",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.25 for name in WHEEL_JOINT_NAMES},
    }
    policy = _ResidualPolicy()
    provider = _ResidualContractProvider(
        forced_profile_strategy="NO_ACTIVE_PROFILE"
    )
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=policy,
        residual_contract_provider=provider,
    )
    batch_count = len(adapter.batch_calls)

    with pytest.raises(RuntimeError, match="profile strategy differs"):
        _process_expected_source_action(
            session,
            adapter,
            session.expected_source_actions[0],
            step=3,
            epoch=1,
        )

    assert provider.contexts[-1]["command_provenance"]["kind"] == (
        "SOURCE_ACTION"
    )
    assert provider.contexts[-1]["profile_strategy"] == "PRIMARY_PROFILE"
    assert policy.observations == []
    assert len(adapter.batch_calls) == batch_count
    assert session.next_source_action_index == 0


def test_residual_injection_is_paired_and_policy_observation_cannot_mutate_nominal(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    with pytest.raises(ValueError, match="supplied together"):
        WorkerMacroFSMSession(
            request,
            worker_session_id="worker-session",
            residual_policy=ZeroResidualPolicy(),
        )

    first_servo = SERVO_JOINT_NAMES[0]

    class MutatingObservationPolicy(_ResidualPolicy):
        def act(self, observation):
            observation["nominal_servo_targets_deg"][first_servo] = 99.0
            observation["command_provenance"]["kind"] = "NONE"
            return ZERO_RESIDUAL_ACTION

    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("immutable nominal",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.25 for name in WHEEL_JOINT_NAMES},
    }
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=MutatingObservationPolicy(),
        residual_contract_provider=_ResidualContractProvider(),
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    assert session.last_servo_targets[first_servo] == 1.0
    assert session.last_applied_servo_targets[first_servo] == 1.0
    assert adapter.batch_calls[-1]["servo_targets_deg"][first_servo] == 1.0
    assert session.source_action_consumption_rows[-1][
        "command_provenance"
    ]["kind"] == "SOURCE_ACTION"


def test_nonzero_residual_uses_one_atomic_batch_and_latches_completion_target(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    first_servo = SERVO_JOINT_NAMES[0]
    first_wheel = WHEEL_JOINT_NAMES[0]
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("residual segment",),
            sequence=0,
        ),
        "servos": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.25 for name in WHEEL_JOINT_NAMES},
        "completion_servo_targets": {first_servo: 1.0},
    }
    policy = _ResidualPolicy(
        _residual_action(**{first_servo: 0.5, first_wheel: 0.5})
    )
    provider = _ResidualContractProvider()
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=policy,
        residual_contract_provider=provider,
    )
    batches_before = len(adapter.batch_calls)
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    assert len(adapter.batch_calls) == batches_before + 1
    applied = adapter.batch_calls[-1]
    assert applied["servo_targets_deg"][first_servo] == 2.0
    assert applied["wheel_targets_rad_s"][first_wheel] == 0.35
    assert set(applied["servo_targets_deg"]) == set(SERVO_JOINT_NAMES)
    assert set(applied["wheel_targets_rad_s"]) == set(WHEEL_JOINT_NAMES)
    completion = session.segment_completion_rows[-1]
    assert completion["completion_spec"]["servo_targets_deg"] == {
        first_servo: 1.0
    }
    assert completion["effective_completion_servo_targets_deg"] == {
        first_servo: 2.0
    }
    assert completion["latched_servo_residual_deg"] == {first_servo: 1.0}
    assert adapter.begin_tracking_calls[-1] == {first_servo: 2.0}

    batch_count = len(adapter.batch_calls)
    source_count = len(session.source_action_consumption_rows)
    session.outer_render_boundary_permit = True
    try:
        with pytest.raises(RuntimeError, match="pending"):
            session._process_decision(
                adapter,
                session._observation_payload(adapter),
                _decision(
                    epoch=1,
                    servos=action["servos"],
                    wheels=action["wheels"],
                ),
            )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batch_count
    assert len(session.source_action_consumption_rows) == source_count

    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    policy.action = ZERO_RESIDUAL_ACTION
    adapter.sim_steps = 8
    adapter.sim_time = 8.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                epoch=1,
                servos=action["servos"],
                wheels=action["wheels"],
            ),
        )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batch_count + 1
    residual_only = session.dispatch_rows[-1]
    assert residual_only["dispatch_cause"] == "RESIDUAL_ONLY"
    assert residual_only["command_epoch"] == 1
    assert residual_only["physical_command_epoch"] == 2
    assert residual_only["servo_targets_deg"][first_servo] == 2.0
    assert residual_only["wheel_targets_rad_s"][first_wheel] == 0.25
    assert residual_only["residual_transform"]["zero_identity"] is False
    assert "ACTIVE_COMPLETION_LATCH" in residual_only[
        "residual_transform"
    ]["clip_reasons_by_action"][first_servo]

    with pytest.raises(RuntimeError, match="exact N"):
        session._verify_pending_readback(adapter, sim_step=8)
    adapter.sim_steps = 9
    adapter.sim_time = 9.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=9)
    adapter.robot.data.joint_pos[0, 0] = math.radians(2.0)
    token = _observe_completion_boundary(session, adapter, step=16)
    assert token is not None and token.kind == "COMPLETE"
    assert adapter.end_tracking_calls[-1] == {first_servo: 2.0}
    assert session.active_completion_latched_servo_residual_deg == {}


def test_same_nominal_target_can_require_a_residual_only_physical_epoch(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    first_servo = SERVO_JOINT_NAMES[0]
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("same nominal target",),
            sequence=0,
        ),
        "servos": zeros_s,
        "wheels": zeros_w,
        "completion_servo_targets": {first_servo: 0.0},
    }
    policy = _ResidualPolicy(_residual_action(**{first_servo: 0.5}))
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=policy,
        residual_contract_provider=_ResidualContractProvider(),
    )
    batches_before = len(adapter.batch_calls)
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=0
    )
    assert len(adapter.batch_calls) == batches_before + 1
    row = session.source_action_consumption_rows[-1]
    assert row["target_changed"] is False
    assert row["nominal_target_changed"] is False
    assert row["applied_target_changed"] is True
    assert row["physical_dispatch_required"] is True
    assert row["dispatch_epoch"] == 0
    assert row["physical_command_epoch"] == 1
    assert row["batch_id"].endswith(":residual:000001")
    assert session.last_epoch == 0
    assert session.physical_command_epoch == 1


def test_boundary_and_terminal_paths_force_residual_wheels_to_zero(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    first_servo = SERVO_JOINT_NAMES[0]
    first_wheel = WHEEL_JOINT_NAMES[0]
    zeros_s = {name: 0.0 for name in SERVO_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    moving_w = {name: 0.25 for name in WHEEL_JOINT_NAMES}
    action = {
        "provenance": _source_provenance(
            segment=0,
            step=1,
            source_time_s=0.0,
            events=(0,),
            commands=("moving segment",),
            sequence=0,
        ),
        "servos": zeros_s,
        "wheels": moving_w,
        "completion_servo_targets": {},
    }
    policy = _ResidualPolicy(
        _residual_action(**{first_servo: 0.5, first_wheel: 0.5})
    )
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions([action]),
        residual_policy=policy,
        residual_contract_provider=_ResidualContractProvider(),
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    token = _observe_completion_boundary(session, adapter, step=8)
    assert token is not None and token.kind == "COMPLETE"

    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state="S2_FR_TRAVERSE",
                epoch=2,
                changed=True,
                servos=zeros_s,
                wheels=zeros_w,
                events=(
                    "EXIT:S1_APPROACH_AND_PRE_FR_SHIFT",
                    "ENTER:S2_FR_TRAVERSE",
                ),
                provenance=_non_source_provenance("BOUNDARY_ZERO_WHEELS"),
            ),
        )
    finally:
        session.outer_render_boundary_permit = False
    boundary = session.dispatch_rows[-1]
    assert boundary["residual_transform"]["force_zero_wheels"] is True
    assert all(value == 0.0 for value in boundary["wheel_targets_rad_s"].values())
    assert boundary["servo_targets_deg"][first_servo] == 1.0

    adapter.sim_steps = 9
    adapter.sim_time = 9.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=9)
    adapter.sim_steps = 16
    adapter.sim_time = 16.0 / 120.0
    _prime_current_target_readback(session, adapter)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            _decision(
                state="SAFE_STOP",
                epoch=2,
                changed=False,
                servos=zeros_s,
                wheels=zeros_w,
                events=("SAFE_STOP:S2_FR_TRAVERSE",),
                terminal=True,
                outcome="SAFE_STOP",
            ),
        )
    finally:
        session.outer_render_boundary_permit = False
    terminal_dispatch = session.dispatch_rows[-1]
    assert terminal_dispatch["residual_transform"]["force_zero_residual"] is True
    assert all(value == 0.0 for value in terminal_dispatch["servo_targets_deg"].values())
    assert all(value == 0.0 for value in terminal_dispatch["wheel_targets_rad_s"].values())
    assert session.state == "terminal_command_pending_readback"


def test_completion_wheel_stop_forces_residual_wheel_channels_to_zero(
    tmp_path: Path,
):
    request, _payload = _load_request(tmp_path)
    first_wheel = WHEEL_JOINT_NAMES[0]
    ones_s = {name: 1.0 for name in SERVO_JOINT_NAMES}
    moving_w = {name: 0.25 for name in WHEEL_JOINT_NAMES}
    zeros_w = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    actions = [
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.0,
                events=(0,),
                commands=("start wheel channel",),
                sequence=0,
            ),
            "servos": ones_s,
            "wheels": moving_w,
            "wheel_duration_s": 0.5,
            "hold_s": 0.51,
        },
        {
            "provenance": _source_provenance(
                segment=0,
                step=1,
                source_time_s=0.5,
                events=(),
                commands=(),
                sequence=1,
                dispatch_kind="wheel_channel_completion_stop",
            ),
            "servos": ones_s,
            "wheels": zeros_w,
        },
    ]
    policy = _ResidualPolicy(_residual_action(**{first_wheel: 0.5}))
    session, adapter = _prime_manual_session(
        request,
        bundle=_bundle_for_actions(actions),
        residual_policy=policy,
        residual_contract_provider=_ResidualContractProvider(),
    )
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[0], step=3, epoch=1
    )
    adapter.sim_steps = 4
    adapter.sim_time = 4.0 / 120.0
    session._verify_pending_readback(adapter, sim_step=4)
    due = _observe_completion_boundary(session, adapter, step=64)
    assert due is not None and due.kind == "WHEEL_STOP_DUE"
    _process_expected_source_action(
        session, adapter, session.expected_source_actions[1], step=64, epoch=2
    )
    row = session.dispatch_rows[-1]
    assert row["residual_transform"]["force_zero_wheels"] is True
    assert all(value == 0.0 for value in row["wheel_targets_rad_s"].values())
    assert all(
        value == 0.0
        for name, value in zip(
            RESIDUAL_ACTION_NAMES,
            row["residual_transform"]["applied_residual"],
        )
        if name in WHEEL_JOINT_NAMES
    )


def test_coalesced_start_composes_only_in_the_next_state_context(tmp_path: Path):
    first_servo, second_servo = SERVO_JOINT_NAMES[:2]
    policy = _ResidualPolicy(
        actions_by_state={
            "S1_APPROACH_AND_PRE_FR_SHIFT": _residual_action(
                **{first_servo: 0.5}
            ),
            "S2_FR_TRAVERSE": _residual_action(**{second_servo: 0.5}),
        }
    )
    provider = _ResidualContractProvider()
    session, adapter, decision, step = _coalesced_transition_case(
        tmp_path,
        same_target=False,
        residual_policy=policy,
        residual_contract_provider=provider,
    )
    assert session.segment_completion_rows[-1][
        "latched_servo_residual_deg"
    ][first_servo] == 1.0
    assert session.active_completion_latched_servo_residual_deg == {}
    batches_before = len(adapter.batch_calls)
    session.outer_render_boundary_permit = True
    try:
        session._process_decision(
            adapter,
            session._observation_payload(adapter),
            decision,
        )
    finally:
        session.outer_render_boundary_permit = False
    assert len(adapter.batch_calls) == batches_before + 1
    assert provider.contexts[-1]["macro_state"] == "S2_FR_TRAVERSE"
    assert policy.observations[-1]["macro_state"] == "S2_FR_TRAVERSE"
    completion = session.segment_completion_rows[-1]
    assert completion["owner_state"] == "S2_FR_TRAVERSE"
    assert completion["effective_completion_servo_targets_deg"][first_servo] == 1.0
    assert completion["effective_completion_servo_targets_deg"][second_servo] == 2.0
    assert completion["latched_servo_residual_deg"][first_servo] == 0.0
    assert completion["latched_servo_residual_deg"][second_servo] == 1.0
    assert session.dispatch_rows[-1]["sim_step"] == step
