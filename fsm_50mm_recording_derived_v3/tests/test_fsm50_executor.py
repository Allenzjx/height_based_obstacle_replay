from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fsm_50mm_recording_derived_v3.com_transfer_primitives import Leg, LegIKCandidate
from fsm_50mm_recording_derived_v3.fsm50_executor import FSM50CommandExecutor


SERVO_NAMES = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_NAMES = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)


@dataclass
class FakeState:
    state_id: str
    state_name: str = "TEST"
    active_leg: Leg | None = None
    swing_leg: Leg | None = None
    support_legs: tuple[Leg, ...] = ()
    servo_start_target: dict[str, float] = field(default_factory=dict)
    servo_end_target: dict[str, float] = field(default_factory=dict)
    servo_trajectory_type: str = "LINEAR"
    wheel_start_target: dict[str, float] = field(default_factory=dict)
    wheel_end_target: dict[str, float] = field(default_factory=dict)
    wheel_trajectory_type: str = "LINEAR_RAMP"
    atomic_concurrent: bool = False
    explicit_hold_s: float = 0.0
    servo_duration_s: float | None = None
    wheel_duration_s: float | None = None
    servo_recording_profile: object | None = None
    wheel_recording_profile: object | None = None


class FakeAdapter:
    def __init__(self) -> None:
        self.servos = {name: 0.0 for name in SERVO_NAMES}
        self.wheels = {name: 0.0 for name in WHEEL_NAMES}
        self.apply_calls: list[dict[str, object]] = []
        self.sim_step = 10
        self.bad_atomic_ack = False

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return {"servos": dict(self.servos), "wheels": dict(self.wheels)}

    def apply_motion_batch(self, payload: dict[str, object]) -> dict[str, object]:
        self.apply_calls.append(payload)
        servo = dict(payload["servo_targets_deg"])
        wheel = dict(payload["wheel_targets_rad_s"])
        self.servos.update(servo)
        self.wheels.update(wheel)
        wheel_step = self.sim_step + 1 if self.bad_atomic_ack else self.sim_step
        ack = {
            "error": "",
            "servo_applied": True,
            "wheel_applied": True,
            "applied_sim_step": self.sim_step,
            "first_physics_step": self.sim_step + 1,
            "servo_applied_sim_step": self.sim_step,
            "wheel_applied_sim_step": wheel_step,
            "servo_first_physics_step": self.sim_step + 1,
            "wheel_first_physics_step": self.sim_step + 1,
            "motion_start_skew_s": 0.0,
            "servo_targets_applied": servo,
            "wheel_targets_applied": wheel,
        }
        self.sim_step += 1
        return ack


def test_atomic_state_uses_exactly_one_full_servo_wheel_batch() -> None:
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(
        adapter, servo_rate_deg_s=100.0, wheel_acceleration_rad_s2=20.0
    )
    state = FakeState(
        "A8_ADVANCE_FR_CONCURRENT",
        servo_end_target={"front_right_hip": 5.0},
        wheel_end_target={"front_left_ankle": 1.0},
        atomic_concurrent=True,
    )
    executor.enter_state(state, time_s=0.0)
    result = executor.update(time_s=0.1)

    assert result.ok
    assert result.atomic_evidence["verified"]
    assert len(adapter.apply_calls) == 1
    payload = adapter.apply_calls[0]
    assert set(payload["servo_targets_deg"]) == set(SERVO_NAMES)
    assert set(payload["wheel_targets_rad_s"]) == set(WHEEL_NAMES)
    assert result.applied["payload"]["servo_targets_deg"] == result.applied["servo_targets_deg"]
    assert result.applied["payload"]["wheel_targets_rad_s"] == result.applied["wheel_targets_rad_s"]
    with pytest.raises(TypeError):
        result.applied["servo_targets_deg"] = {}  # type: ignore[index]


def test_servo_rate_wheel_acceleration_entry_capture_and_cross_state_continuity() -> None:
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(
        adapter, servo_rate_deg_s=10.0, wheel_acceleration_rad_s2=1.0
    )
    first = FakeState(
        "MOVE_OUT",
        servo_end_target={"front_left_hip": 10.0},
        wheel_end_target={"front_left_ankle": 2.0},
        explicit_hold_s=0.25,
    )
    executor.enter_state(first, time_s=0.0)
    halfway = executor.update(time_s=0.5)
    assert halfway.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(5.0)
    assert halfway.applied["wheel_targets_rad_s"]["front_left_ankle"] == pytest.approx(0.5)
    endpoint = executor.update(time_s=2.0)
    assert endpoint.endpoint_reached
    assert not endpoint.hold_complete
    held = executor.update(time_s=2.3)
    assert held.hold_complete
    assert held.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(10.0)
    assert held.applied["wheel_targets_rad_s"]["front_left_ankle"] == pytest.approx(2.0)

    second = FakeState(
        "RAMP_BACK",
        servo_end_target={"front_left_hip": 0.0},
        wheel_end_target={"front_left_ankle": 0.0},
    )
    entry = executor.enter_state(second, time_s=2.3)
    assert entry["captured_command_state"]["servos"]["front_left_hip"] == pytest.approx(10.0)
    same_tick = executor.update(time_s=2.3)
    assert same_tick.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(10.0)
    assert same_tick.applied["wheel_targets_rad_s"]["front_left_ankle"] == pytest.approx(2.0)
    later = executor.update(time_s=2.8)
    assert later.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(5.0)
    assert later.applied["wheel_targets_rad_s"]["front_left_ankle"] == pytest.approx(1.5)


def test_all_declared_trajectory_families_have_bounded_deterministic_semantics() -> None:
    def one_tick(state: FakeState, *, time_s: float = 0.5) -> tuple[float, float]:
        adapter = FakeAdapter()
        executor = FSM50CommandExecutor(
            adapter, servo_rate_deg_s=100.0, wheel_acceleration_rad_s2=100.0
        )
        executor.enter_state(state, time_s=0.0)
        result = executor.update(time_s=time_s)
        return (
            result.applied["servo_targets_deg"]["front_left_hip"],
            result.applied["wheel_targets_rad_s"]["front_left_ankle"],
        )

    cubic_servo, linear_wheel = one_tick(
        FakeState(
            "CURVES",
            servo_end_target={"front_left_hip": 8.0},
            servo_trajectory_type="CUBIC",
            servo_duration_s=1.0,
            wheel_end_target={"front_left_ankle": 2.0},
            wheel_trajectory_type="LINEAR_RAMP",
            wheel_duration_s=1.0,
        )
    )
    assert cubic_servo == pytest.approx(4.0)
    assert linear_wheel == pytest.approx(1.0)

    held_servo, held_wheel = one_tick(
        FakeState(
            "HOLDS",
            servo_end_target={"front_left_hip": 8.0},
            servo_trajectory_type="HOLD",
            wheel_end_target={"front_left_ankle": 2.0},
            wheel_trajectory_type="HOLD",
        )
    )
    assert held_servo == pytest.approx(0.0)
    assert held_wheel == pytest.approx(0.0)

    recorded_servo, recorded_wheel = one_tick(
        FakeState(
            "RECORDED",
            servo_end_target={"front_left_hip": 8.0},
            servo_trajectory_type="RECORDING_RAMP",
            servo_duration_s=1.0,
            servo_recording_profile=((0.0, 0.0), (0.5, 0.25), (1.0, 1.0)),
            wheel_end_target={"front_left_ankle": 2.0},
            wheel_trajectory_type="RECORDING_PROFILE",
            wheel_duration_s=1.0,
            wheel_recording_profile=((0.0, 0.0), (0.5, 0.25), (1.0, 1.0)),
        )
    )
    assert recorded_servo == pytest.approx(2.0)
    assert recorded_wheel == pytest.approx(0.5)

    # STEP selects the endpoint immediately, while the mandatory acceleration
    # limiter still prevents a discontinuous wheel command.
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(adapter, wheel_acceleration_rad_s2=1.0)
    executor.enter_state(
        FakeState(
            "STEP",
            servo_trajectory_type="HOLD",
            wheel_end_target={"front_left_ankle": 2.0},
            wheel_trajectory_type="STEP",
        ),
        time_s=0.0,
    )
    stepped = executor.update(time_s=0.5)
    assert stepped.requested["wheel_profile_target_rad_s"]["front_left_ankle"] == 2.0
    assert stepped.applied["wheel_targets_rad_s"]["front_left_ankle"] == pytest.approx(0.5)


def test_active_leg_isolation_protects_swing_and_only_support_legs_receive_correction() -> None:
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(
        adapter,
        servo_rate_deg_s=100.0,
        wheel_acceleration_rad_s2=10.0,
        correction_rate_deg_s=100.0,
    )
    state = FakeState(
        "E13_SWING_RL_CLEAR_FRONT_FACE",
        state_name="SWING_RL_CLEAR_FRONT_FACE",
        active_leg=Leg.RL,
        swing_leg=Leg.RL,
        support_legs=(Leg.FL, Leg.RR),
        servo_end_target={name: 0.0 for name in SERVO_NAMES},
        servo_trajectory_type="HOLD",
        wheel_trajectory_type="HOLD",
    )
    corrections = {
        leg: {
            next(name for name in SERVO_NAMES if name.startswith(
                {Leg.FL: "front_left", Leg.FR: "front_right", Leg.RL: "rear_left", Leg.RR: "rear_right"}[leg]
            ) and name.endswith("_hip")): 5.0
        }
        for leg in Leg
    }
    executor.enter_state(state, time_s=0.0, whole_body_corrections_deg=corrections)
    result = executor.update(time_s=0.1)

    assert result.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(5.0)
    assert result.applied["servo_targets_deg"]["rear_right_hip"] == pytest.approx(5.0)
    assert result.applied["servo_targets_deg"]["rear_left_hip"] == pytest.approx(0.0)
    assert result.applied["servo_targets_deg"]["front_right_hip"] == pytest.approx(0.0)
    assert result.isolation_evidence["scales"]["RL"] == 0.0
    assert result.isolation_evidence["scales"]["FL"] == 1.0


def test_place_blends_active_leg_correction() -> None:
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(
        adapter, servo_rate_deg_s=100.0, correction_rate_deg_s=100.0
    )
    state = FakeState(
        "E14_PLACE_RL",
        state_name="PLACE_RL",
        active_leg=Leg.RL,
        support_legs=(Leg.FL, Leg.RR),
        servo_end_target={name: 0.0 for name in SERVO_NAMES},
        servo_trajectory_type="HOLD",
        wheel_trajectory_type="HOLD",
    )
    corrections = {Leg.RL: {"rear_left_hip": 8.0}}
    executor.enter_state(
        state,
        time_s=0.0,
        whole_body_corrections_deg=corrections,
        place_confirm_blend=0.25,
    )
    result = executor.update(time_s=0.1)
    assert result.applied["servo_targets_deg"]["rear_left_hip"] == pytest.approx(2.0)
    assert result.isolation_evidence["scales"]["RL"] == pytest.approx(0.25)


def test_correction_offset_is_carried_across_state_without_double_apply_or_zero_jump() -> None:
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(
        adapter, servo_rate_deg_s=100.0, correction_rate_deg_s=100.0
    )
    correction = {Leg.FL: {"front_left_hip": 5.0}}
    first = FakeState(
        "FIRST_SUPPORT",
        support_legs=(Leg.FL,),
        servo_trajectory_type="HOLD",
        wheel_trajectory_type="HOLD",
    )
    executor.enter_state(
        first, time_s=0.0, whole_body_corrections_deg=correction
    )
    applied = executor.update(time_s=0.1)
    assert applied.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(5.0)

    second = FakeState(
        "SECOND_SUPPORT",
        support_legs=(Leg.FL,),
        servo_trajectory_type="HOLD",
        wheel_trajectory_type="HOLD",
    )
    executor.enter_state(second, time_s=0.1)
    continuous = executor.update(time_s=0.1)
    assert continuous.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(5.0)
    assert continuous.continuity_evidence["entry_correction_offsets_deg"]["front_left_hip"] == pytest.approx(5.0)


def test_per_leg_ik_accepts_valid_leg_and_holds_only_rejected_leg() -> None:
    adapter = FakeAdapter()
    executor = FSM50CommandExecutor(
        adapter, servo_rate_deg_s=100.0, wheel_acceleration_rad_s2=10.0
    )
    state = FakeState(
        "IK_STATE",
        servo_end_target={"front_left_hip": 2.0, "rear_right_knee": 2.0},
        wheel_trajectory_type="HOLD",
    )
    executor.enter_state(state, time_s=0.0)
    candidates = {
        Leg.FL: LegIKCandidate(
            Leg.FL,
            requested_targets_deg={"front_left_hip": 3.0},
            reference_targets_deg={"front_left_hip": 0.0},
            joint_limits_deg={"front_left_hip": (-10.0, 10.0)},
        ),
        Leg.RR: LegIKCandidate(
            Leg.RR,
            requested_targets_deg={"rear_right_knee": 50.0},
            reference_targets_deg={"rear_right_knee": 0.0},
            joint_limits_deg={"rear_right_knee": (-10.0, 10.0)},
        ),
    }
    result = executor.update(time_s=0.1, ik_candidates=candidates)

    assert result.ok
    assert result.applied["servo_targets_deg"]["front_left_hip"] == pytest.approx(3.0)
    assert result.applied["servo_targets_deg"]["rear_right_knee"] == pytest.approx(0.0)
    assert result.ik_evidence["decisions"]["FL"]["accepted"]
    assert not result.ik_evidence["decisions"]["RR"]["accepted"]
    assert result.ik_evidence["rejected_legs"] == ("RR",)


def test_atomic_ack_mismatch_fails_closed_without_second_batch() -> None:
    adapter = FakeAdapter()
    adapter.bad_atomic_ack = True
    executor = FSM50CommandExecutor(adapter)
    state = FakeState(
        "ATOMIC_BAD_ACK",
        servo_end_target={"front_left_hip": 1.0},
        wheel_end_target={"front_left_ankle": 1.0},
        atomic_concurrent=True,
    )
    executor.enter_state(state, time_s=0.0)
    result = executor.update(time_s=0.1)

    assert not result.ok
    assert result.fail_closed
    assert not result.atomic_evidence["verified"]
    assert "applied_sim_step differ" in result.error
    assert len(adapter.apply_calls) == 1
    latched = executor.update(time_s=0.2)
    assert latched.fail_closed
    assert len(adapter.apply_calls) == 1
