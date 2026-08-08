from __future__ import annotations

import csv

import pytest

from fsm_50mm_recording_derived_v3.com_transfer_primitives import (
    COMTransferMethod,
    Leg,
)
from fsm_50mm_recording_derived_v3.fsm50_state_model import (
    FSM50State,
    FSM50StateTable,
    ProvenanceStatus,
    REQUIRED_STATE_FIELDS,
    TransitionDwellTracker,
    make_pending_candidate_state,
)


def _state_row() -> dict[str, object]:
    return {
        "state_id": "E5",
        "state_name": "RL_DOWNWARD_REACTION_PULSE",
        "description": "Recording-derived reaction candidate; replay validation pending.",
        "source_recording_version": "v010_20260806_220745_363972_manual",
        "source_step_indices": [21, 22],
        "source_event_indices": [130, 131],
        "source_telemetry_time_range": [10.0, 12.0],
        "active_leg": "RL",
        "swing_leg": "RL",
        "impulse_leg": "RL",
        "support_legs": ["FL", "RR"],
        "primary_diagonal": "FL_RR",
        "secondary_support": ["FR"],
        "target_com_direction": [1.0, -1.0],
        "com_transfer_method": "HYBRID",
        "servo_start_target": {"rear_left_knee": 22.6},
        "servo_end_target": {"rear_left_knee": 35.3},
        "servo_trajectory_type": "RECORDING_RAMP",
        "wheel_start_target": {"front_left_ankle": 0.0},
        "wheel_end_target": {"front_left_ankle": -0.3},
        "wheel_trajectory_type": "LINEAR_RAMP",
        "atomic_concurrent": True,
        "min_duration": 0.2,
        "settle_duration": 0.1,
        "max_duration": 2.0,
        "entry_guard": {"kind": "IMPULSE_PRELOAD", "parameters": {"load_n": 5.0}},
        "progress_guard": {"kind": "COM_DIRECTION", "parameters": {"dx_m": 0.01}},
        "exit_guard": {"kind": "COM_SETTLED", "parameters": {"speed_m_s": 0.01}},
        "abort_guard": {"kind": "SAFE_STOP", "parameters": {"roll_deg": 15.0}},
        "hysteresis": {"load_n": 0.5},
        "required_consecutive_samples": 2,
        "retry_policy": {
            "maximum_retries": 1,
            "compensation_scale": 0.25,
            "abort_on_exhaustion": True,
        },
        "expected_com_displacement": [0.01, -0.01],
        "expected_com_velocity_direction": [1.0, -1.0],
        "expected_load_transfer": {"FR": 4.0, "RL": -4.0},
        "expected_contact_change": {"RL": "LOADED_TO_UNLOADED"},
        "expected_clearance": {"RL": 0.012},
        "expected_body_response": {"roll_direction": "toward_FR", "settle_required": True},
        "allowed_contact_drift": 0.01,
        "allowed_roll_pitch": [15.0, 15.0],
        "allowed_angular_velocity": 20.0,
        "joint_limit_margin": 2.0,
        "notes": "Thresholds are test values, not recording-derived validation.",
        "provenance_status": "CANDIDATE",
        "provenance_note": "candidate endpoint evidence; clean replay pending",
    }


def test_state_schema_contains_every_user_required_field() -> None:
    row = _state_row()
    state = FSM50State.from_mapping(row)
    exported = state.to_mapping()
    assert set(REQUIRED_STATE_FIELDS).issubset(exported)
    assert state.active_leg == Leg.RL
    assert state.support_legs == (Leg.FL, Leg.RR)
    assert state.com_transfer_method == COMTransferMethod.HYBRID
    assert state.provenance_status == ProvenanceStatus.CANDIDATE
    assert state.pending_physical_replay


def test_strict_state_loading_rejects_an_omitted_required_column() -> None:
    row = _state_row()
    del row["expected_clearance"]
    with pytest.raises(ValueError, match="expected_clearance"):
        FSM50State.from_mapping(row)


def test_atomic_concurrent_requires_both_servo_and_wheel_motion() -> None:
    row = _state_row()
    row["wheel_end_target"] = {"front_left_ankle": 0.0}
    with pytest.raises(ValueError, match="atomic_concurrent"):
        FSM50State.from_mapping(row)


def test_physically_verified_provenance_requires_event_and_telemetry_evidence() -> None:
    row = _state_row()
    row["provenance_status"] = "PHYSICALLY_VERIFIED"
    row["source_event_indices"] = []
    row["source_telemetry_time_range"] = None
    with pytest.raises(ValueError, match="PHYSICALLY_VERIFIED"):
        FSM50State.from_mapping(row)


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".csv"])
def test_state_table_round_trips_json_yaml_and_csv(tmp_path, suffix: str) -> None:
    original = FSM50StateTable(
        states=(FSM50State.from_mapping(_state_row()),),
        metadata={"physical_validation": "pending"},
    )
    path = original.export(tmp_path / f"state_table{suffix}")
    loaded = FSM50StateTable.load(path)
    assert loaded.states[0].to_mapping() == original.states[0].to_mapping()
    assert loaded.states[0].provenance_status == ProvenanceStatus.CANDIDATE
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream))
        assert set(REQUIRED_STATE_FIELDS).issubset(header)


def test_state_table_rejects_duplicate_identifiers() -> None:
    state = FSM50State.from_mapping(_state_row())
    with pytest.raises(ValueError, match="unique"):
        FSM50StateTable((state, state))


def test_transition_tracker_enforces_minimum_stable_dwell_samples_and_timeout() -> None:
    state = FSM50State.from_mapping(_state_row())
    tracker = TransitionDwellTracker(state, entered_at_s=0.0)
    assert not tracker.update(time_s=0.1, exit_condition=True).satisfied
    assert not tracker.update(time_s=0.5, exit_condition=True).satisfied
    passed = tracker.update(time_s=0.61, exit_condition=True)
    assert passed.satisfied
    assert passed.metrics["consecutive_samples"] == 2

    tracker.reset(0.0)
    timed_out = tracker.update(time_s=2.01, exit_condition=False)
    assert timed_out.abort
    assert "timeout" in timed_out.reason


def test_pending_candidate_helper_never_claims_physical_validation() -> None:
    state = make_pending_candidate_state(
        state_id="E10",
        state_name="UNLOAD_RL",
        source_recording_version="v010_20260806_220745_363972_manual",
        source_step_indices=(21, 24),
        active_leg=Leg.RL,
        support_legs=(Leg.FL, Leg.RR),
        com_transfer_method=COMTransferMethod.IMPULSE_REACTION_TRANSFER,
        target_com_direction=(1.0, -1.0),
    )
    assert state.provenance_status == ProvenanceStatus.PENDING_REPLAY
    assert state.source_event_indices == ()
    assert state.source_telemetry_time_range is None
    assert "physical" in state.provenance_note
    assert state.pending_physical_replay


def test_nonzero_transfer_requires_a_nonzero_com_direction() -> None:
    row = _state_row()
    row["target_com_direction"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="target_com_direction"):
        FSM50State.from_mapping(row)
