from __future__ import annotations

from fractions import Fraction
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from command_model import (
    SERVO_JOINT_NAMES,
    WHEEL_FORWARD_SIGN,
    WHEEL_JOINT_NAMES,
)
from playback import PlaybackPlan, PlaybackSegment
from telemetry.collector import TelemetryCollector

from fsm_50mm_recording_derived_v3.filtered_wheel_contact import FILTERED_SURFACES
from fsm_50mm_recording_derived_v3.fsm50_telemetry import (
    FILTERED_FORCE_SOURCE,
    FSM50TelemetryCollector,
    LEG_TO_WHEEL_BODY,
    LEG_TO_WHEEL_JOINT,
)
from fsm_50mm_recording_derived_v3.physx_contact_separation import (
    PHYSX_SEPARATION_SOURCE,
)
from fsm_50mm_recording_derived_v3.support_classifier import (
    ContactPersistenceTracker,
    ObstacleGeometry,
    TraversalEvidenceTracker,
)
from fsm_50mm_recording_derived_v3.wheel_integral_evidence import (
    PASS,
    evaluate_wheel_integral_evidence,
)


LEGS = ("FL", "FR", "RL", "RR")


class _TraversalResult:
    def __init__(self, *, all_valid: bool = True, illegal: bool = False) -> None:
        self.payload = {
            "legs": {leg: {} for leg in LEGS},
            "all_legs_valid": bool(all_valid),
            "any_illegal_drive_up": bool(illegal),
        }

    def result(self):
        return dict(self.payload)


def _drift_sample(
    leg: str,
    *,
    drift: float = 0.0,
    mode: str = "ZERO_TARGET_LOADED_CONTACT_EPOCH",
):
    return {
        "leg": leg,
        "mode": mode,
        "contact_class": "TOP",
        "evidence_valid": True,
        "zero_target_contact_point_displacement_m": (
            drift if mode == "ZERO_TARGET_LOADED_CONTACT_EPOCH" else None
        ),
        "physical_anchoring_proven": False,
        "material_point_identity_available": False,
        "contact_point_displacement_semantics": (
            "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT"
        ),
    }


def _qualification_row(time_s: float) -> dict[str, object]:
    return {
        "time_s": float(time_s),
        "base_roll_rad": 0.0,
        "base_pitch_rad": 0.0,
        "base_vx_m_s": 0.0,
        "base_vy_m_s": 0.0,
        "base_vz_m_s": 0.0,
        "base_wx_rad_s": 0.0,
        "base_wy_rad_s": 0.0,
        "base_wz_rad_s": 0.0,
        "wheel_net_force_valid": True,
        "wheel_contact_force_up_n": {leg: 10.0 for leg in LEGS},
        "wheel_canonical_forward_angle_rad": {leg: 0.0 for leg in LEGS},
        "wheel_centers_w": {leg: [0.0, 0.0, 0.1] for leg in LEGS},
        "wheel_contact_classes": {leg: "TOP" for leg in LEGS},
        "filtered_contact_layout_valid": True,
        "filtered_contact_force_valid": True,
        "filtered_contact_geometry_valid": True,
        "filtered_contact_consistency_valid": True,
        "joint_limit_evidence_valid": True,
        "joint_limit_violation": False,
        "collision_evidence_valid": True,
        "dangerous_collision": False,
        "physx_drive_target_evidence_valid": True,
        "penetration_evidence_valid": True,
        "penetration_evidence_source": PHYSX_SEPARATION_SOURCE,
        "maximum_collision_penetration_m": 0.0,
        "force_impulse_evidence_valid": True,
        "wheel_contact_persistence_diagnostic": {
            leg: _drift_sample(leg) for leg in LEGS
        },
    }


def _evidence_collector(*, contact_mode: str) -> FSM50TelemetryCollector:
    collector = object.__new__(FSM50TelemetryCollector)
    collector.contact_mode = contact_mode
    collector.environment_equivalence_role = ""
    collector.source_version = "v003-test"
    collector.fsm50_rows = [
        _qualification_row(0.0),
        _qualification_row(0.125),
        _qualification_row(0.25),
    ]
    collector.traversal = _TraversalResult()
    collector.force_threshold_n = 2.0
    collector.maximum_roll_rad = math.radians(45.0)
    collector.maximum_pitch_rad = math.radians(45.0)
    collector.maximum_angular_velocity_rad_s = 3.0
    collector.maximum_allowed_contact_drift_m = 0.03
    collector.final_linear_velocity_m_s = 0.03
    collector.final_angular_velocity_rad_s = 0.15
    collector.final_stable_dwell_s = 0.25
    collector.maximum_penetration_m = 0.003
    collector.maximum_contact_drift_m = 0.0
    collector.minimum_corridor_margin_m = float("inf")
    collector.filtered_contact_errors = []
    collector.dangerous_collision_rows = []
    collector.collision_evidence_errors = []
    collector.joint_limit_violation_rows = []
    collector.integrated_wheel_force_impulse_w = {
        leg: [0.0, 0.0, 0.0] for leg in LEGS
    }
    collector.integrated_wheel_upward_impulse_n_s = {leg: 0.0 for leg in LEGS}
    return collector


def test_formal_role_pass_full_not_evaluable_with_all_common_evidence() -> None:
    evidence = _evidence_collector(contact_mode="formal").physical_evidence()

    assert evidence["schema_version"] == "fsm50.physical_evidence.v2"
    assert len(evidence["criterion_records"]) == 11
    assert evidence["role_capture_verdict"] == "PASS"
    assert evidence["full_physical_verdict"] == "NOT_EVALUABLE"
    assert evidence["physical_success"] is False
    unavailable = {
        name
        for name, record in evidence["criteria"].items()
        if record["availability"] == "UNAVAILABLE_BY_ROLE"
    }
    assert unavailable == {
        "contact_evidence_valid",
        "collision_safe",
        "contact_drift_safe",
    }
    assert evidence["criteria"]["penetration_safe"]["scope"] == "COMMON"


def test_contact_mode_is_an_explicit_required_constructor_argument() -> None:
    parameter = inspect.signature(FSM50TelemetryCollector).parameters["contact_mode"]
    assert parameter.default is inspect.Parameter.empty


def test_physical_evidence_carries_explicit_environment_capture_role() -> None:
    collector = _evidence_collector(contact_mode="formal")
    collector.environment_equivalence_role = "A1"

    evidence = collector.physical_evidence()

    assert evidence["contact_mode"] == "formal"
    assert evidence["environment_equivalence_role"] == "A1"
    assert evidence["physical_success"] is False


def test_u_role_marks_every_physical_criterion_unavailable() -> None:
    collector = _evidence_collector(contact_mode="disabled")
    collector.diagnostic_role = "U"

    evidence = collector.physical_evidence()

    assert evidence["diagnostic_role"] == "U"
    assert evidence["environment_equivalence_role"] == ""
    assert evidence["role_capture_verdict"] == "NOT_EVALUABLE"
    assert evidence["full_physical_verdict"] == "NOT_EVALUABLE"
    assert evidence["physical_success"] is False
    assert evidence["physical_qualification_eligible"] is False
    assert evidence["environment_equivalence_eligible"] is False
    assert evidence["physical_anchoring_proven"] is False
    assert evidence["material_point_identity_available"] is False
    assert evidence["zero_target_loaded_contact_displacement_sample_count"] == 0
    assert len(evidence["criteria"]) == 11
    assert all(
        row["availability"] == "UNAVAILABLE_BY_ROLE"
        and row["passed"] is None
        for row in evidence["criteria"].values()
    )


def test_known_common_failure_precedes_missing_common_evidence() -> None:
    collector = _evidence_collector(contact_mode="formal")
    collector.fsm50_rows[0]["base_roll_rad"] = math.radians(50.0)
    collector.fsm50_rows[1]["penetration_evidence_valid"] = False
    collector.fsm50_rows[1]["penetration_evidence_source"] = ""
    collector.fsm50_rows[1]["maximum_collision_penetration_m"] = None

    evidence = collector.physical_evidence()

    assert evidence["criteria"]["attitude_safe"]["passed"] is False
    assert evidence["criteria"]["penetration_safe"]["passed"] is None
    assert evidence["role_capture_verdict"] == "FAIL"
    assert evidence["full_physical_verdict"] == "FAIL"


def test_instrumented_full_pass_and_missing_b_evidence_is_not_evaluable() -> None:
    passed = _evidence_collector(contact_mode="instrumented").physical_evidence()
    assert passed["role_capture_verdict"] == "PASS"
    assert passed["full_physical_verdict"] == "PASS"
    assert passed["physical_success"] is True
    assert all(value is True for value in passed["strict_criteria"].values())
    assert passed["physical_anchoring_proven"] is False
    assert passed["material_point_identity_available"] is False
    assert passed["contact_point_displacement_semantics"] == (
        "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT"
    )
    assert passed["zero_target_loaded_contact_displacement_sample_count"] == 12

    missing = _evidence_collector(contact_mode="instrumented")
    missing.fsm50_rows[1]["collision_evidence_valid"] = False
    missing.fsm50_rows[1]["dangerous_collision"] = None
    result = missing.physical_evidence()
    assert result["criteria"]["collision_safe"]["availability"] == "MISSING"
    assert result["criteria"]["collision_safe"]["passed"] is None
    assert result["role_capture_verdict"] == "NOT_EVALUABLE"
    assert result["full_physical_verdict"] == "NOT_EVALUABLE"
    assert result["physical_success"] is False


def test_zero_target_loaded_contact_displacement_over_limit_remains_fail() -> None:
    collector = _evidence_collector(contact_mode="instrumented")
    collector.fsm50_rows[1]["wheel_contact_persistence_diagnostic"]["RL"] = (
        _drift_sample("RL", drift=0.03445764763205546)
    )

    evidence = collector.physical_evidence()

    criterion = evidence["criteria"]["contact_drift_safe"]
    assert criterion["availability"] == "AVAILABLE"
    assert criterion["passed"] is False
    assert "zero-target loaded measured contact-point displacement" in criterion["reason"]
    assert evidence["maximum_contact_drift_m"] == pytest.approx(
        0.03445764763205546
    )
    assert evidence["maximum_zero_target_contact_point_displacement_m"] == pytest.approx(
        0.03445764763205546
    )
    assert evidence["physical_anchoring_proven"] is False
    assert evidence["material_point_identity_available"] is False
    assert evidence["full_physical_verdict"] == "FAIL"
    assert evidence["physical_success"] is False


def test_legacy_anchored_mode_is_not_accepted_as_new_producer_evidence() -> None:
    collector = _evidence_collector(contact_mode="instrumented")
    legacy = _drift_sample("FL")
    legacy.update({"mode": "ANCHORED", "anchored_drift_m": 0.0})
    collector.fsm50_rows[0]["wheel_contact_persistence_diagnostic"]["FL"] = legacy

    evidence = collector.physical_evidence()

    criterion = evidence["criteria"]["contact_drift_safe"]
    assert criterion["availability"] == "MISSING"
    assert criterion["passed"] is None
    assert evidence["contact_drift_evidence_valid"] is False
    assert evidence["physical_success"] is False


def test_final_velocity_dwell_accepts_only_120hz_timestamp_roundoff_boundary() -> None:
    start_time_s = 5.450000000000061
    final_time_s = 5.700000000000073
    physics_dt_s = float(Fraction(1, 120))
    rows = []
    for offset, sim_step in enumerate(range(654, 685)):
        time_s = start_time_s + offset * physics_dt_s
        if sim_step == 684:
            time_s = final_time_s
        row = _qualification_row(time_s)
        row.update(
            {
                "sim_step": sim_step,
                "physics_dt_s": physics_dt_s,
                "base_vx_m_s": 0.012804773567587436,
                "base_wz_rad_s": 0.041666324024381418,
            }
        )
        rows.append(row)

    # This is the observed tick-654 boundary: it is 12.4 fs below the cutoff
    # solely because sim time was accumulated in binary floating point.
    assert rows[0]["time_s"] < final_time_s - 0.25
    assert rows[-1]["time_s"] - rows[0]["time_s"] >= 0.25

    collector = _evidence_collector(contact_mode="instrumented")
    collector.fsm50_rows = rows
    evidence = collector.physical_evidence()
    assert collector.final_stable_dwell_s == 0.25
    assert evidence["criteria"]["final_velocity_stable"]["passed"] is True

    # The guard must not turn a genuinely short, 30-sample interval into a
    # complete dwell; the measured first-to-last span remains authoritative.
    short = _evidence_collector(contact_mode="instrumented")
    short.fsm50_rows = rows[1:]
    short_evidence = short.physical_evidence()
    assert rows[-1]["time_s"] - rows[1]["time_s"] < 0.25
    assert short_evidence["criteria"]["final_velocity_stable"]["passed"] is False


def test_physx_penetration_failure_unknown_and_aabb_never_upgrades() -> None:
    penetrated = _evidence_collector(contact_mode="formal")
    penetrated.fsm50_rows[0]["maximum_collision_penetration_m"] = 0.004
    result = penetrated.physical_evidence()
    assert result["criteria"]["penetration_safe"]["passed"] is False
    assert result["role_capture_verdict"] == "FAIL"

    unknown = _evidence_collector(contact_mode="formal")
    for row in unknown.fsm50_rows:
        row["penetration_evidence_valid"] = False
        row["penetration_evidence_source"] = ""
        row["maximum_collision_penetration_m"] = None
        row["wheel_ground_aabb_proxy"] = {
            "valid": True,
            "maximum_collision_penetration_m": 0.0,
            "physical_qualification_eligible": False,
        }
    unavailable = unknown.physical_evidence()
    criterion = unavailable["criteria"]["penetration_safe"]
    assert criterion["scope"] == "COMMON"
    assert criterion["availability"] == "MISSING"
    assert criterion["passed"] is None
    assert unavailable["role_capture_verdict"] == "NOT_EVALUABLE"
    assert unavailable["physical_success"] is False


def test_combined_bank_signed_separation_is_preserved_and_unknown_fails_closed() -> None:
    valid_sensor = SimpleNamespace(
        is_filtered_wheel_contact_bank=True,
        is_nonwheel_obstacle_contact_bank=True,
        separation_observations=lambda **_kwargs: [
            {
                "pair_id": "wheel-ground",
                "valid": True,
                "signed_separations_m": [-0.001],
                "maximum_penetration_m": 0.001,
            }
        ],
        separation_evidence=lambda **_kwargs: {
            "schema_version": "fsm50.physx_contact_separation.v1",
            "valid": True,
            "status": "AVAILABLE",
            "pair_count": 1,
            "pair_ids": ["wheel-ground"],
            "unknown_pair_ids": [],
            "maximum_physx_penetration_m": 0.001,
            "source": PHYSX_SEPARATION_SOURCE,
            "errors": [],
        },
    )
    collector = object.__new__(FSM50TelemetryCollector)
    collector.scene_handle = SimpleNamespace(contact_sensor=valid_sensor)
    rows, evidence = collector._physx_separation_sample()
    assert rows[0]["signed_separations_m"] == [-0.001]
    assert evidence["valid"] is True
    assert evidence["maximum_physx_penetration_m"] == 0.001

    unknown_sensor = SimpleNamespace(
        is_filtered_wheel_contact_bank=True,
        is_nonwheel_obstacle_contact_bank=True,
        separation_observations=lambda **_kwargs: [
            {"pair_id": "wheel-ground", "valid": False}
        ],
        separation_evidence=lambda **_kwargs: {
            "schema_version": "fsm50.physx_contact_separation.v1",
            "valid": False,
            "status": "UNKNOWN",
            "pair_count": 1,
            "pair_ids": ["wheel-ground"],
            "unknown_pair_ids": ["wheel-ground"],
            "maximum_physx_penetration_m": None,
            "source": PHYSX_SEPARATION_SOURCE,
            "errors": ["capacity exhaustion"],
        },
    )
    collector.scene_handle = SimpleNamespace(contact_sensor=unknown_sensor)
    _rows, unknown_evidence = collector._physx_separation_sample()
    assert unknown_evidence["valid"] is False
    assert unknown_evidence["maximum_physx_penetration_m"] is None


def _fast_plan() -> PlaybackPlan:
    duration = float(Fraction(2, 120))
    segment = PlaybackSegment(
        segment_index=0,
        source_step=0,
        source_step_id="step-0",
        event_start_index=0,
        event_count=1,
        planned_start_s=0.0,
        planned_end_s=duration,
        base_duration_s=duration,
        servo_base_duration_s=0.0,
        servo_duration_s=0.0,
        servo_targets={},
        wheel_active_duration_s=duration,
        wheel_base_velocity={name: 1.0 for name in WHEEL_JOINT_NAMES},
        wheel_requested_velocity_rad_s={name: 1.0 for name in WHEEL_JOINT_NAMES},
        wheel_applied_target_rad_s={name: 1.0 for name in WHEEL_JOINT_NAMES},
    )
    return PlaybackPlan(
        path=Path("v003.jsonl"),
        events=[],
        final_time_s=duration,
        label="collector-wheel-integral",
        plan_sha256="a" * 64,
        segments=[segment],
    )


def _collector_for_live_rows(monkeypatch: pytest.MonkeyPatch):
    collector = object.__new__(FSM50TelemetryCollector)
    collector.enabled = True
    collector.rows = []
    collector.contact_rows = []
    collector.fsm50_rows = []
    collector.state_timeline_rows = []
    collector.filtered_surface_rows = []
    collector.nonwheel_obstacle_rows = []
    collector.dangerous_collision_rows = []
    collector.collision_evidence_errors = []
    collector.joint_limit_violation_rows = []
    collector.filtered_contact_errors = []
    collector.last_timeline_key = None
    collector.runtime_context = {}
    collector.source_version = "v003-test"
    collector.contact_mode = "instrumented"
    collector.obstacle = ObstacleGeometry(
        front_face_x_m=0.5,
        top_z_m=0.05,
        bottom_z_m=0.0,
        rear_face_x_m=2.5,
        width_m=2.0,
    )
    collector.wheel_radius_m = 0.05
    collector.force_threshold_n = 2.0
    collector.maximum_contact_drift_m = 0.0
    collector.maximum_allowed_contact_drift_m = 0.03
    collector.minimum_corridor_margin_m = float("inf")
    collector.previous_wheel_angle = {}
    collector.integrated_wheel_rotation = {leg: 0.0 for leg in LEGS}
    collector.integrated_wheel_travel = {leg: 0.0 for leg in LEGS}
    collector.integrated_wheel_force_impulse_w = {
        leg: [0.0, 0.0, 0.0] for leg in LEGS
    }
    collector.integrated_wheel_upward_impulse_n_s = {leg: 0.0 for leg in LEGS}
    collector.initial_com_position_w = None
    collector.previous_sample_time_s = None
    collector.dangerous_nonwheel_contact_force_n = 5.0
    collector.wheel_forward_sign = {
        leg: float(WHEEL_FORWARD_SIGN[LEG_TO_WHEEL_JOINT[leg]]) for leg in LEGS
    }
    collector.persistence = ContactPersistenceTracker(load_confirm_force_n=2.0)
    collector.traversal = TraversalEvidenceTracker(
        unload_force_n=1.0,
        load_confirm_force_n=2.0,
        top_load_dwell_s=0.10,
        loaded_front_face_rotation_limit_rad=0.15,
    )
    collector.plan = None
    collector.plan_rows_by_segment = {}
    collector.scene_handle = SimpleNamespace(contact_sensor=object())
    collector.defer_periodic_checkpoint = True
    collector._maybe_checkpoint = lambda _time: None

    adapter = SimpleNamespace(
        sim_steps=0,
        sim_time=0.0,
        wheel_direction=1.0,
        wheel_speeds={name: 0.0 for name in WHEEL_JOINT_NAMES},
        joint_command_deg={name: 0.0 for name in SERVO_JOINT_NAMES},
    )
    physical_position = {name: 0.0 for name in WHEEL_JOINT_NAMES}

    def drive_evidence():
        physx_target = {
            name: adapter.wheel_speeds[name] * float(WHEEL_FORWARD_SIGN[name])
            for name in WHEEL_JOINT_NAMES
        }
        positions = {
            **{name: 0.0 for name in SERVO_JOINT_NAMES},
            **physical_position,
        }
        velocities = {name: 0.0 for name in positions}
        position_targets = {name: 0.0 for name in positions}
        velocity_targets = {**{name: 0.0 for name in SERVO_JOINT_NAMES}, **physx_target}
        return {
            "valid": True,
            "error": "",
            "joint_position_by_name": positions,
            "joint_velocity_by_name": velocities,
            "joint_position_target_by_name": position_targets,
            "joint_position_target_buffer_by_name": position_targets,
            "joint_velocity_target_by_name": velocity_targets,
            "joint_velocity_target_buffer_by_name": velocity_targets,
            "servo_command_target_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
            "servo_command_to_readback_error_by_name": {
                name: 0.0 for name in SERVO_JOINT_NAMES
            },
            "joint_target_minus_position_by_name": {
                name: 0.0 for name in positions
            },
        }

    adapter.capture_joint_drive_evidence = drive_evidence
    body_positions = {
        LEG_TO_WHEEL_BODY[leg]: (0.0, (-0.2, 0.2, -0.2, 0.2)[index], 0.05)
        for index, leg in enumerate(LEGS)
    }
    collector._body_positions = lambda _adapter: dict(body_positions)
    collector._joint_vectors = lambda _adapter: (
        dict(drive_evidence()["joint_position_by_name"]),
        dict(drive_evidence()["joint_velocity_by_name"]),
    )
    collector._joint_safety_evidence = lambda *_args: {
        "joint_limit_evidence_valid": True,
        "joint_limit_evidence_error": "",
        "joint_limit_margin_rad": {name: 1.0 for name in SERVO_JOINT_NAMES},
        "minimum_joint_limit_margin_rad": 1.0,
        "joint_limit_violation": False,
        "actuator_target_error_rad": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "maximum_actuator_target_error_rad": 0.0,
    }
    filtered = []
    for leg in LEGS:
        for filter_index, (surface, other_prim_path) in enumerate(FILTERED_SURFACES):
            active = surface == "ground"
            filtered.append(
                {
                    "leg": leg,
                    "surface": surface,
                    "filter_index": filter_index,
                    "active": active,
                    "force_valid": True,
                    "contact_point_valid": active,
                    "contact_point_w": [0.0, 0.0, 0.0] if active else None,
                    "normal_force_w": [0.0, 0.0, 10.0 if active else 0.0],
                    "normal_force_n": 10.0 if active else 0.0,
                    "upward_force_n": 10.0 if active else 0.0,
                    "total_force_n": 10.0 if active else 0.0,
                    "wheel_body_name": LEG_TO_WHEEL_BODY[leg],
                    "wheel_prim_path": f"/World/WLRRobot/{LEG_TO_WHEEL_BODY[leg]}",
                    "other_prim_path": other_prim_path,
                    "source": FILTERED_FORCE_SOURCE,
                }
            )
    collector._filtered_contacts = lambda: (list(filtered), "")
    collector._nonwheel_collision_sample = lambda _time: ([], True, False, "")
    collector._common_wheel_net_force_by_leg = lambda _sensor: (
        {
            leg: {
                "upward_force_n": 10.0,
                "total_force_n": 10.0,
                "vector_w": [0.0, 0.0, 10.0],
                "source": "isaaclab.ContactSensor.net_forces_w",
            }
            for leg in LEGS
        },
        {
            "wheel_net_force_layout_valid": True,
            "wheel_net_force_valid": True,
            "wheel_net_force_error": "",
            "wheel_contact_force_common_source": "isaaclab.ContactSensor.net_forces_w",
        },
    )
    collector._physx_separation_sample = lambda: (
        [
            {
                "pair_id": "all",
                "valid": True,
                "signed_separations_m": [],
                "maximum_penetration_m": 0.0,
            }
        ],
        {
            "schema_version": "fsm50.physx_contact_separation.v1",
            "valid": True,
            "status": "AVAILABLE",
            "pair_count": 1,
            "pair_ids": ["all"],
            "unknown_pair_ids": [],
            "source": PHYSX_SEPARATION_SOURCE,
            "maximum_physx_penetration_m": 0.0,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "fsm_50mm_recording_derived_v3.fsm50_telemetry.wheel_ground_aabb_proxy_snapshot",
        lambda _adapter: {
            "valid": True,
            "maximum_collision_penetration_m": 0.0,
            "wheel_penetration_m": {name: 0.0 for name in WHEEL_JOINT_NAMES},
        },
    )

    def base_on_step(self, current_adapter, _dt):
        base = {
            "time_s": current_adapter.sim_time,
            "sim_step": current_adapter.sim_steps,
            "base_x_m": 0.0,
            "base_y_m": 0.0,
            "base_z_m": 0.2,
            "base_qw": 1.0,
            "base_qx": 0.0,
            "base_qy": 0.0,
            "base_qz": 0.0,
            "base_roll_rad": 0.0,
            "base_pitch_rad": 0.0,
            "base_vx_m_s": 0.0,
            "base_vy_m_s": 0.0,
            "base_vz_m_s": 0.0,
            "base_wx_rad_s": 0.0,
            "base_wy_rad_s": 0.0,
            "base_wz_rad_s": 0.0,
            "com_x_m": 0.0,
            "com_y_m": 0.0,
            "com_z_m": 0.2,
            "com_vx_m_s": 0.0,
            "com_vy_m_s": 0.0,
            "com_vz_m_s": 0.0,
        }
        self.rows.append(base)
        self.last_row = base

    monkeypatch.setattr(TelemetryCollector, "on_step", base_on_step)
    return collector, adapter, physical_position


def test_direct_collector_rows_are_accepted_by_wheel_integral_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector, adapter, positions = _collector_for_live_rows(monkeypatch)
    dt = float(Fraction(1, 120))
    batches = [
        {
            "batch_id": "segment-0",
            "dispatch_kind": "source_segment_start",
            "segment_index": 0,
            "wheel_targets_rad_s": {name: 1.0 for name in WHEEL_JOINT_NAMES},
            "ack_present": True,
            "ack_valid": True,
            "ack_error": "",
            "adapter_called": True,
            "applied_sim_step": 9,
            "first_physics_step": 10,
            "motion_start_skew_s": 0.0,
        },
        {
            "batch_id": "final-stop",
            "dispatch_kind": "final_safety_stop",
            "segment_index": 0,
            "wheel_targets_rad_s": {name: 0.0 for name in WHEEL_JOINT_NAMES},
            "ack_present": True,
            "ack_valid": True,
            "ack_error": "",
            "adapter_called": True,
            "applied_sim_step": 11,
            "first_physics_step": 12,
            "motion_start_skew_s": 0.0,
        },
    ]
    for step in (9, 10, 11, 12):
        adapter.sim_steps = step
        adapter.sim_time = step * dt
        target = 1.0 if step in (10, 11) else 0.0
        adapter.wheel_speeds = {name: target for name in WHEEL_JOINT_NAMES}
        for name in WHEEL_JOINT_NAMES:
            positions[name] = (
                max(0, min(step - 9, 2)) * dt * float(WHEEL_FORWARD_SIGN[name])
            )
        collector.on_step(adapter, dt)

    assert all(row["physics_dt_s"] == float(Fraction(1, 120)) for row in collector.fsm50_rows)
    assert all(
        set(row["wheel_forward_sign"]) == set(WHEEL_JOINT_NAMES)
        for row in collector.fsm50_rows
    )
    assert collector.fsm50_rows[1]["observed_sample_dt_s"] == pytest.approx(dt)
    assert all(
        row["wheel_legacy_simple_integration_authoritative"] is False
        for row in collector.fsm50_rows
    )
    assert collector.fsm50_rows[0]["wheel_contact_persistence_mode"] == {
        leg: "ZERO_TARGET_LOADED_CONTACT_EPOCH" for leg in LEGS
    }
    assert collector.fsm50_rows[1]["wheel_contact_persistence_mode"] == {
        leg: "ACTIVE_ROLLING" for leg in LEGS
    }
    assert collector.fsm50_rows[2]["wheel_contact_persistence_mode"] == {
        leg: "ACTIVE_ROLLING" for leg in LEGS
    }
    assert collector.fsm50_rows[3]["wheel_contact_persistence_mode"] == {
        leg: "ZERO_TARGET_LOADED_CONTACT_EPOCH" for leg in LEGS
    }
    assert collector.maximum_contact_drift_m == 0.0

    evidence = evaluate_wheel_integral_evidence(
        plan=_fast_plan(),
        timing_trace={"motion_batches": batches},
        telemetry_rows=collector.fsm50_rows,
    )
    assert evidence["target_integral_verdict"] == PASS
    assert not evidence["target_not_evaluable_reasons"]
