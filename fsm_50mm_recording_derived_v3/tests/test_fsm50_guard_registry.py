from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3.com_transfer_primitives import Leg
from fsm_50mm_recording_derived_v3.fsm50_guard_registry import (
    FSM50GuardRegistry,
    GuardEvaluationContext,
    LIFECYCLE_GUARDS,
    STATE_GUARDS,
)
from fsm_50mm_recording_derived_v3.fsm50_observation import (
    COMTargetDirection,
    FSM50Observation,
    ObservationIssue,
)
from fsm_50mm_recording_derived_v3.fsm50_state_model import FSM50StateTable


MODULE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = MODULE_ROOT / "fsm50_config.yaml"
LEGS = ("FL", "FR", "RL", "RR")


@pytest.fixture(scope="module")
def table() -> FSM50StateTable:
    return FSM50StateTable.load(CONFIG_PATH)


@pytest.fixture
def registry() -> FSM50GuardRegistry:
    return FSM50GuardRegistry()


def _thresholds(table: FSM50StateTable) -> dict[str, object]:
    return dict(table.metadata["thresholds"])


def _context(
    table: FSM50StateTable,
    state_id: str,
    observation: FSM50Observation,
    *,
    inherited: GuardEvaluationContext | None = None,
) -> GuardEvaluationContext:
    return GuardEvaluationContext.enter(
        table.get(state_id), observation, _thresholds(table), inherited=inherited
    )


def _active_observation(
    time_s: float,
    *,
    leg: str = "FR",
    contact_class: str = "GROUND",
    load_n: float = 5.0,
    rotation_rad: float = 0.0,
    top_clearance_m: float = -0.05,
    face_clearance_m: float = -0.20,
) -> FSM50Observation:
    classes = {name: "GROUND" for name in LEGS}
    classes[leg] = contact_class
    loads = {name: 5.0 for name in LEGS}
    loads[leg] = load_n
    rotations = {name: 0.0 for name in LEGS}
    rotations[leg] = rotation_rad
    top_clearance = {name: -0.05 for name in LEGS}
    top_clearance[leg] = top_clearance_m
    face_clearance = {name: -0.20 for name in LEGS}
    face_clearance[leg] = face_clearance_m
    support_legs = [
        name
        for name in LEGS
        if loads[name] >= 2.0 and classes[name] not in {"AIR", "UNKNOWN"}
    ]
    return FSM50Observation.fake(
        time_s=time_s,
        wheel_contact_class=classes,
        filtered_ground_force_n=loads,
        filtered_obstacle_force_n={name: 0.0 for name in LEGS},
        integrated_wheel_rotation_rad=rotations,
        wheel_clearance_over_top_m=top_clearance,
        wheel_front_face_clearance_m=face_clearance,
        support_legs=support_legs,
    )


def _diagonal_observation(time_s: float, *, applicable: bool, valid: bool) -> FSM50Observation:
    return FSM50Observation.fake(
        time_s=time_s,
        wheel_contact_class={
            "FL": "GROUND",
            "FR": "AIR",
            "RL": "AIR",
            "RR": "GROUND",
        },
        filtered_ground_force_n={"FL": 5.0, "FR": 0.0, "RL": 0.0, "RR": 5.0},
        filtered_obstacle_force_n={name: 0.0 for name in LEGS},
        support_legs=["FL", "RR"],
        primary_diagonal="FL_RR",
        two_leg_corridor_applicable=applicable,
        two_leg_corridor_valid=valid,
        two_leg_corridor_distance_m=0.01,
        two_leg_corridor_fraction=0.5,
        two_leg_corridor_within_longitudinal_bounds=True,
        two_leg_corridor_within_width=True,
    )


def test_observation_contract_fields_are_all_accounted_for() -> None:
    assert {item.name for item in fields(FSM50Observation)} == {
        "time_s",
        "root_position_w",
        "root_orientation_wxyz",
        "root_linear_velocity_w",
        "root_angular_velocity_w",
        "com_position_w",
        "com_velocity_w",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "servo_target_rad",
        "wheel_target_rad_s",
        "measured_wheel_velocity_rad_s",
        "integrated_wheel_rotation_rad",
        "integrated_wheel_travel_m",
        "wheel_center_w",
        "filtered_ground_force_n",
        "filtered_obstacle_force_n",
        "wheel_contact_point_w",
        "wheel_contact_class",
        "support_legs",
        "light_support_legs",
        "primary_diagonal",
        "diagonal_load_share",
        "support_polygon_margin_m",
        "support_polygon_valid",
        "two_leg_corridor",
        "contact_drift_m",
        "wheel_clearance_over_top_m",
        "wheel_front_face_clearance_m",
        "nonwheel_obstacle_contact",
        "nonwheel_contact_evidence_valid",
        "joint_limit_margin_rad",
        "actuator_target_error_rad",
        "com_target",
        "issues",
    }
    assert FSM50Observation.fake().control_ready


def test_every_configured_guard_resolves_to_a_real_predicate(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    registry.validate_states(table.states)
    configured_state_guards = {state.guard for state in table.states}
    configured_lifecycle_guards = {
        guard.kind
        for state in table.states
        for guard in (
            state.entry_guard,
            state.progress_guard,
            state.exit_guard,
            state.abort_guard,
        )
    }
    assert configured_state_guards == STATE_GUARDS
    assert configured_lifecycle_guards == LIFECYCLE_GUARDS
    assert configured_state_guards | configured_lifecycle_guards == registry.names


@pytest.mark.parametrize(
    "name", ["UNKNOWN_GUARD", "ALWAYS_TRUE", "PLACEHOLDER", "PENDING_REPLAY"]
)
def test_unknown_and_placeholder_guards_are_rejected(
    name: str, table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    observation = FSM50Observation.fake()
    context = _context(table, "A0_RESET_AND_SETTLE", observation)
    with pytest.raises(KeyError, match="unknown or unimplemented guard"):
        registry.evaluate(name, observation, context)
    with pytest.raises(ValueError, match="unknown or unimplemented guards"):
        registry.validate_states((replace(context.state, guard=name),))


def test_lifecycle_placeholder_cannot_replace_a_concrete_state_predicate(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    state = replace(table.get("A0_RESET_AND_SETTLE"), guard="STATE_PHYSICAL_EVENT")
    with pytest.raises(ValueError, match="unknown or unimplemented guards"):
        registry.validate_states((state,))
    observation = FSM50Observation.fake()
    context = GuardEvaluationContext.enter(state, observation, _thresholds(table))
    with pytest.raises(KeyError, match="requires a concrete state predicate"):
        registry.evaluate("STATE_PHYSICAL_EVENT", observation, context)


def test_missing_critical_telemetry_fails_closed_before_predicate_access(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    valid = FSM50Observation.fake()
    missing = replace(
        valid,
        com_velocity_w=None,
        issues=(
            ObservationIssue(
                "com_velocity_w", "MISSING", "measured COM velocity unavailable"
            ),
        ),
    )
    context = _context(table, "B3_FR_PUSH_TOWARD_OBSTACLE", valid)

    decision = registry.evaluate("COM_DIRECTION", missing, context)
    lifecycle = registry.evaluate("LIVE_EVIDENCE_COMPLETE", missing, context)
    safety = registry.evaluate("FAIL_CLOSED_SAFETY_STOP", missing, context)

    assert not missing.control_ready
    assert decision.abort and not decision.satisfied
    assert lifecycle.abort and not lifecycle.satisfied
    assert safety.abort and not safety.satisfied
    assert decision.unmet == ("com_velocity_w",)


def test_com_direction_uses_live_world_geometry_not_yaml_hint(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    contacts = dict(FSM50Observation.fake().wheel_contact_point_w)
    contacts["FR"] = (0.0, 1.0, 0.0)
    yaw_90_wxyz = (
        math.cos(math.pi / 4.0),
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
    )
    baseline = FSM50Observation.fake(
        root_orientation_wxyz=yaw_90_wxyz,
        com_position_w=(0.0, 0.0, 0.08),
        wheel_contact_point_w=contacts,
    )
    moved = FSM50Observation.fake(
        time_s=0.10,
        root_orientation_wxyz=yaw_90_wxyz,
        com_position_w=(0.0, 0.01, 0.08),
        com_velocity_w=(0.0, 0.01, 0.0),
        wheel_contact_point_w=contacts,
    )
    state = replace(
        table.get("B3_FR_PUSH_TOWARD_OBSTACLE"),
        target_com_direction=(1.0, 0.0),
    )
    context = GuardEvaluationContext.enter(state, baseline, _thresholds(table))

    target = moved.target_direction_for(Leg.FR)
    decision = registry.evaluate("COM_DIRECTION", moved, context)

    assert isinstance(target, COMTargetDirection)
    assert target.target_leg == "FR"
    assert target.direction_w[:2] == pytest.approx((0.0, 1.0), abs=1.0e-12)
    assert decision.satisfied and not decision.abort
    assert decision.metrics["target_direction_world_xy"] == pytest.approx(
        (0.0, 1.0), abs=1.0e-12
    )
    assert decision.metrics["projected_com_displacement_m"] == pytest.approx(0.01)


def test_loaded_front_face_forward_rotation_aborts_fail_closed(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    baseline = _active_observation(
        0.0, contact_class="FRONT_FACE", load_n=3.0, rotation_rad=0.0
    )
    context = _context(table, "A5_VERIFY_FR_UNLOAD_READY", baseline)
    initial = registry.evaluate("FAIL_CLOSED_SAFETY_STOP", baseline, context)
    driven = _active_observation(
        0.1, contact_class="FRONT_FACE", load_n=3.0, rotation_rad=0.16
    )
    decision = registry.evaluate("FAIL_CLOSED_SAFETY_STOP", driven, context)

    assert not initial.abort
    assert decision.abort and not decision.satisfied
    assert "ILLEGAL_DRIVE_UP" in decision.unmet
    assert context.illegal_drive_up[Leg.FR]
    assert context.loaded_front_face_rotation_rad[Leg.FR] == pytest.approx(0.16)


def test_two_leg_diagonal_guard_requires_applicable_valid_corridor(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    first = _diagonal_observation(0.0, applicable=True, valid=True)
    context = _context(table, "E2_PREP_FL_RR_DIAGONAL", first)
    waiting = registry.evaluate("DIAGONAL_SUPPORT", first, context)
    stable = registry.evaluate(
        "DIAGONAL_SUPPORT",
        _diagonal_observation(0.09, applicable=True, valid=True),
        context,
    )
    invalid = registry.evaluate(
        "DIAGONAL_SUPPORT",
        _diagonal_observation(0.10, applicable=True, valid=False),
        context,
    )

    assert not waiting.satisfied
    assert stable.satisfied and not stable.abort
    assert stable.metrics["corridor_applicable"] is True
    assert stable.metrics["corridor_valid"] is True
    assert not invalid.satisfied

    not_applicable = _diagonal_observation(0.0, applicable=False, valid=True)
    other_context = _context(table, "E2_PREP_FL_RR_DIAGONAL", not_applicable)
    assert not registry.evaluate(
        "DIAGONAL_SUPPORT", not_applicable, other_context
    ).satisfied


def test_traversal_requires_ordered_unload_air_clear_place_and_top_load(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    loaded = _active_observation(0.0)
    unload_context = _context(table, "A6_UNLOAD_FR", loaded)
    unloaded = _active_observation(0.10, load_n=0.2)
    assert not registry.evaluate("UNLOADED", unloaded, unload_context).satisfied
    unload_confirm = registry.evaluate(
        "UNLOADED", _active_observation(0.13, load_n=0.2), unload_context
    )
    assert unload_confirm.satisfied

    air = _active_observation(
        0.20, contact_class="AIR", load_n=0.0, top_clearance_m=0.01
    )
    air_context = _context(table, "A7_LIFT_FR", air, inherited=unload_context)
    assert not registry.evaluate("AIRBORNE", air, air_context).satisfied
    air_confirm = registry.evaluate(
        "AIRBORNE",
        _active_observation(
            0.26, contact_class="AIR", load_n=0.0, top_clearance_m=0.01
        ),
        air_context,
    )
    assert air_confirm.satisfied

    clear = _active_observation(
        0.27,
        contact_class="AIR",
        load_n=0.0,
        top_clearance_m=0.01,
        face_clearance_m=0.01,
    )
    clear_context = _context(
        table, "A8_ADVANCE_FR_CONCURRENT", clear, inherited=air_context
    )
    assert registry.evaluate("FACE_CLEARED", clear, clear_context).satisfied

    placed = _active_observation(
        0.30,
        contact_class="TOP",
        load_n=0.2,
        top_clearance_m=0.0,
        face_clearance_m=0.01,
    )
    place_context = _context(table, "A10_PLACE_FR", placed, inherited=clear_context)
    assert registry.evaluate("TOP_CONTACT", placed, place_context).satisfied

    top_loaded = _active_observation(
        0.31,
        contact_class="TOP",
        load_n=3.0,
        top_clearance_m=0.0,
        face_clearance_m=0.01,
    )
    load_context = _context(
        table, "A11_CONFIRM_FR_TOP_SUPPORT", top_loaded, inherited=place_context
    )
    assert not registry.evaluate("TOP_LOAD", top_loaded, load_context).satisfied
    load_confirm = registry.evaluate(
        "TOP_LOAD",
        _active_observation(
            0.40,
            contact_class="TOP",
            load_n=3.0,
            top_clearance_m=0.0,
            face_clearance_m=0.01,
        ),
        load_context,
    )

    assert load_confirm.satisfied and not load_confirm.abort
    assert load_context.traversal_valid[Leg.FR]
    assert load_context.traversal_phase[Leg.FR] == "TOP_LOAD_CONFIRMED"
    assert (
        load_context.unload_start_s[Leg.FR]
        <= load_context.airborne_start_s[Leg.FR]
        <= load_context.front_face_crossing_s[Leg.FR]
        <= load_context.top_contact_s[Leg.FR]
        <= load_context.top_load_confirm_s[Leg.FR]
    )
    assert load_context.unload_end_s[Leg.FR] == load_context.airborne_start_s[Leg.FR]
    assert load_context.airborne_end_s[Leg.FR] == load_context.top_contact_s[Leg.FR]


def test_out_of_order_air_and_direct_top_never_create_traversal_evidence(
    table: FSM50StateTable, registry: FSM50GuardRegistry
) -> None:
    loaded = _active_observation(0.0)
    air_context = _context(table, "A7_LIFT_FR", loaded)
    premature_air = _active_observation(
        0.10, contact_class="AIR", load_n=0.0, top_clearance_m=0.01
    )
    registry.evaluate("AIRBORNE", premature_air, air_context)
    air_decision = registry.evaluate(
        "AIRBORNE",
        _active_observation(
            0.20, contact_class="AIR", load_n=0.0, top_clearance_m=0.01
        ),
        air_context,
    )
    assert not air_decision.satisfied
    assert not air_context.airborne_seen[Leg.FR]
    assert air_context.traversal_phase[Leg.FR] == "LOADED_SUPPORT"

    place_context = _context(table, "A10_PLACE_FR", loaded)
    direct_top = _active_observation(
        0.10,
        contact_class="TOP",
        load_n=3.0,
        top_clearance_m=0.0,
        face_clearance_m=0.01,
    )
    top_decision = registry.evaluate("TOP_CONTACT", direct_top, place_context)
    assert top_decision.abort and not top_decision.satisfied
    assert place_context.illegal_drive_up[Leg.FR]
    assert not place_context.traversal_valid[Leg.FR]
    assert Leg.FR not in place_context.top_contact_s
