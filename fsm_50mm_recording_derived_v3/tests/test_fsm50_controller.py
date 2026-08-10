from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3.com_transfer_primitives import Leg
from fsm_50mm_recording_derived_v3.fsm50_controller import (
    ControllerStatus,
    FSM50Controller,
    validate_happy_path_reachability,
)
from fsm_50mm_recording_derived_v3.fsm50_observation import FSM50Observation
from fsm_50mm_recording_derived_v3.fsm50_state_model import (
    FSM50StateTable,
    RetryPolicy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "fsm50_config.yaml"
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


class FakeAdapter:
    def __init__(self) -> None:
        self.servos = {name: 0.0 for name in SERVO_NAMES}
        self.wheels = {name: 0.0 for name in WHEEL_NAMES}
        self.calls: list[dict[str, object]] = []
        self.sim_step = 0

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return {"servos": dict(self.servos), "wheels": dict(self.wheels)}

    def apply_motion_batch(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        servo = {str(k): float(v) for k, v in dict(payload["servo_targets_deg"]).items()}
        wheel = {str(k): float(v) for k, v in dict(payload["wheel_targets_rad_s"]).items()}
        self.servos.update(servo)
        self.wheels.update(wheel)
        ack = {
            "error": "",
            "servo_applied": bool(servo),
            "wheel_applied": bool(wheel),
            "applied_sim_step": self.sim_step,
            "first_physics_step": self.sim_step + 1,
            "motion_start_skew_s": 0.0,
            "servo_targets_applied": servo,
            "wheel_targets_applied": wheel,
        }
        self.sim_step += 1
        return ack


def _small_table(
    *,
    first_guard: str = "INITIAL_STABLE",
    first_max_s: float = 1.0,
    retry_policy: RetryPolicy | None = None,
    active_leg: Leg | None = None,
    target_com_leg: Leg | None = None,
) -> FSM50StateTable:
    source = FSM50StateTable.load(CONFIG)
    first = replace(
        source.get("A0_RESET_AND_SETTLE"),
        guard=first_guard,
        active_leg=active_leg,
        swing_leg=active_leg,
        target_com_leg=target_com_leg,
        support_legs=tuple(leg for leg in Leg if leg != active_leg),
        min_duration=0.0,
        settle_duration=0.0,
        max_duration=first_max_s,
        required_consecutive_samples=1,
        retry_policy=retry_policy or RetryPolicy(),
    )
    terminal = replace(
        source.get("F5_SUCCESS"),
        min_duration=0.0,
        settle_duration=0.0,
        max_duration=1.0,
        required_consecutive_samples=1,
    )
    safe = replace(
        source.get("SAFE_STOP"),
        min_duration=0.0,
        settle_duration=0.0,
        required_consecutive_samples=1,
    )
    return FSM50StateTable(
        states=(first, terminal, safe),
        schema_version=source.schema_version,
        metadata=source.metadata,
    )


def _top_observation(time_s: float) -> FSM50Observation:
    return FSM50Observation.fake(
        time_s=time_s,
        wheel_contact_class={leg.value: "TOP" for leg in Leg},
        filtered_ground_force_n={leg.value: 0.0 for leg in Leg},
        filtered_obstacle_force_n={leg.value: 5.0 for leg in Leg},
    )


def test_actual_config_happy_path_is_reachable_and_guard_complete() -> None:
    identifiers = validate_happy_path_reachability(CONFIG)
    assert len(identifiers) == 56
    assert identifiers[0] == "A0_RESET_AND_SETTLE"
    assert identifiers[-1] == "F5_SUCCESS"
    assert "SAFE_STOP" not in identifiers
    assert len(set(identifiers)) == len(identifiers)


def test_controller_runs_on_enter_update_on_exit_and_never_uses_endpoint_as_exit() -> None:
    adapter = FakeAdapter()
    controller = FSM50Controller(adapter, _small_table())
    started = controller.start(FSM50Observation.fake(time_s=0.0))
    assert started.state_id == "A0_RESET_AND_SETTLE"
    assert not adapter.calls  # on_enter latches state but does not write early.

    controller.context.airborne_seen = {leg: True for leg in Leg}
    controller.context.traversal_valid = {leg: True for leg in Leg}
    controller.context.concurrent_home_recovery_verified = True
    transitioned = controller.update(FSM50Observation.fake(time_s=0.01))
    assert transitioned.transitioned
    assert transitioned.state_id == "F5_SUCCESS"
    assert transitioned.status == ControllerStatus.RUNNING
    assert len(adapter.calls) == 1

    finished = controller.update(_top_observation(0.02))
    assert finished.succeeded
    events = [row["event"] for row in controller.timeline]
    assert events.count("on_enter") == 2
    assert "update" in events
    assert events.count("on_exit") == 2


def test_reaching_f5_label_is_not_success_without_all_physical_history() -> None:
    controller = FSM50Controller(FakeAdapter(), _small_table())
    controller.start(FSM50Observation.fake(time_s=0.0))
    at_f5 = controller.update(FSM50Observation.fake(time_s=0.01))
    assert at_f5.state_id == "F5_SUCCESS"
    not_success = controller.update(_top_observation(0.02))
    assert not_success.status == ControllerStatus.RUNNING
    assert not not_success.succeeded
    assert "traversal not verified" in not_success.exit_guard["reason"]


def test_timeout_enters_safe_stop_and_holds_servos_with_zero_wheels() -> None:
    table = _small_table(
        first_guard="COM_DIRECTION", first_max_s=0.01, target_com_leg=Leg.RL
    )
    adapter = FakeAdapter()
    controller = FSM50Controller(adapter, table)
    controller.start(FSM50Observation.fake(time_s=0.0))
    result = controller.update(FSM50Observation.fake(time_s=0.02))
    assert result.fail_closed
    assert result.state_id == "SAFE_STOP"
    assert "timeout" in result.transition_reason
    safe_payload = adapter.calls[-1]
    assert safe_payload["source"] == "fsm50_controller_safe_stop"
    assert set(safe_payload["servo_targets_deg"]) == set(SERVO_NAMES)
    assert all(value == 0.0 for value in safe_payload["wheel_targets_rad_s"].values())


def test_timeout_retries_exactly_policy_count_then_safe_stops() -> None:
    table = _small_table(
        first_guard="COM_DIRECTION",
        first_max_s=0.01,
        target_com_leg=Leg.RL,
        retry_policy=RetryPolicy(
            maximum_retries=1,
            compensation_scale=0.1,
            retry_state_id="A0_RESET_AND_SETTLE",
        ),
    )
    controller = FSM50Controller(FakeAdapter(), table)
    controller.start(FSM50Observation.fake(time_s=0.0))
    retried = controller.update(FSM50Observation.fake(time_s=0.02))
    assert retried.status == ControllerStatus.RUNNING
    assert retried.retry_index == 1
    assert retried.transitioned
    assert controller.retry_counts["A0_RESET_AND_SETTLE"] == 1

    exhausted = controller.update(FSM50Observation.fake(time_s=0.04))
    assert exhausted.fail_closed
    assert controller.retry_counts["A0_RESET_AND_SETTLE"] == 1


def test_missing_critical_telemetry_fails_closed_before_motion() -> None:
    adapter = FakeAdapter()
    controller = FSM50Controller(adapter, _small_table())
    incomplete = FSM50Observation.from_mapping({}, strict=False)
    result = controller.start(incomplete)
    assert result.fail_closed
    assert result.state_id == "SAFE_STOP"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["source"] == "fsm50_controller_safe_stop"


def test_loaded_front_face_rotation_is_illegal_drive_up_abort() -> None:
    table = _small_table(
        first_guard="FACE_CLEARED", active_leg=Leg.FR, first_max_s=1.0
    )
    adapter = FakeAdapter()
    controller = FSM50Controller(adapter, table)
    controller.start(FSM50Observation.fake(time_s=0.0))
    face = {
        leg.value: ("FRONT_FACE" if leg == Leg.FR else "GROUND") for leg in Leg
    }
    controller.update(
        FSM50Observation.fake(
            time_s=0.01,
            wheel_contact_class=face,
            integrated_wheel_rotation_rad={leg.value: 0.0 for leg in Leg},
        )
    )
    result = controller.update(
        FSM50Observation.fake(
            time_s=0.02,
            wheel_contact_class=face,
            integrated_wheel_rotation_rad={
                leg.value: (0.2 if leg == Leg.FR else 0.0) for leg in Leg
            },
        )
    )
    assert result.fail_closed
    assert "ILLEGAL_DRIVE_UP" in result.transition_reason


def test_non_a0_start_requires_trusted_restore_or_verified_prefix() -> None:
    with pytest.raises(ValueError, match="trusted restore/prefix provenance"):
        FSM50Controller(
            FakeAdapter(),
            FSM50StateTable.load(CONFIG),
            start_state_id="A7_LIFT_FR",
        )

    provenance = {
        "method": "TRUSTED_SIM_STATE_RESTORE",
        "validated": True,
        "state_id": "A7_LIFT_FR",
        "source_run_directory": "runs/state_restore/example",
        "source_sha256": "a" * 64,
    }
    controller = FSM50Controller(
        FakeAdapter(),
        FSM50StateTable.load(CONFIG),
        start_state_id="A7_LIFT_FR",
        restore_provenance=provenance,
    )
    assert controller.start_state_id == "A7_LIFT_FR"


def test_controller_result_is_deeply_immutable() -> None:
    controller = FSM50Controller(FakeAdapter(), _small_table())
    result = controller.start(FSM50Observation.fake(time_s=0.0))
    with pytest.raises(TypeError):
        result.evidence["support_legs"] = []  # type: ignore[index]
