from __future__ import annotations

from pathlib import Path

import yaml

from fsm_50mm_recording_derived_v3.fsm50_state_model import FSM50StateTable
from motion_speed import load_motion_reference


MODULE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = MODULE_ROOT / "fsm50_config.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_actual_controller_yaml_compiles_profiles_into_commands() -> None:
    table = FSM50StateTable.load(CONFIG_PATH)
    assert len(table.states) == 57
    assert table.states[0].state_id == "A0_RESET_AND_SETTLE"
    assert table.states[-2].state_id == "F5_SUCCESS"
    assert table.states[-1].state_id == "SAFE_STOP"
    atomic = table.get("A8_ADVANCE_FR_CONCURRENT")
    assert atomic.command_profile == "s04_fr_cross_atomic"
    assert len(atomic.servo_end_target) == 8
    assert atomic.wheel_end_target["front_left_ankle"] == -0.79
    assert atomic.atomic_concurrent


def test_every_profile_is_used_or_explicitly_deprecated() -> None:
    raw = _raw()
    used = {row["command_profile"] for row in raw["states"]}
    profiles = raw["command_profiles"]
    unexplained = {
        name
        for name, profile in profiles.items()
        if name not in used
        and not (
            profile.get("deprecated") is True
            and str(profile.get("deprecation_reason", "")).strip()
        )
    }
    assert not unexplained
    assert {name for name in profiles if name not in used} == {
        "s02_approach",
        "s17_all_top_advance",
    }


def test_com_transfer_states_name_a_world_target_leg() -> None:
    table = FSM50StateTable.load(CONFIG_PATH)
    missing = [
        state.state_id
        for state in table.states
        if state.target_com_direction is not None and state.target_com_leg is None
    ]
    assert not missing


def test_happy_path_list_is_reachable_and_safe_stop_is_an_abort_target() -> None:
    table = FSM50StateTable.load(CONFIG_PATH)
    happy = [state.state_id for state in table.states if state.state_id != "SAFE_STOP"]
    assert happy[0].startswith("A0_")
    assert happy[-1] == "F5_SUCCESS"
    assert len(happy) == len(set(happy))
    assert table.get("SAFE_STOP").guard == "TERMINAL_SAFE_STOP"


def test_runtime_servo_rate_is_authoritative_motion_reference() -> None:
    raw = _raw()
    configured = float(raw["actuators"]["servo_max_rate_deg_s"])
    assert configured == load_motion_reference().servo_reference_velocity_deg_s
    assert configured == 150.0


def test_no_state_claims_unbacked_physical_provenance() -> None:
    table = FSM50StateTable.load(CONFIG_PATH)
    assert all(state.provenance_status.value == "PENDING_REPLAY" for state in table.states)
    assert all(not state.source_event_indices for state in table.states)
    assert all(state.source_telemetry_time_range is None for state in table.states)
