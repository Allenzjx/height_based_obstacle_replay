"""Pure Playback button availability rules shared by controller and UI."""

from __future__ import annotations

from dataclasses import dataclass

from operation_coordinator import OperationState


@dataclass(frozen=True)
class PlaybackAvailability:
    can_start: bool
    can_respawn_start: bool
    can_play_selected: bool
    can_pause: bool
    can_resume: bool
    can_stop: bool
    can_analyze: bool
    can_export: bool
    reason: str


def evaluate_playback_availability(
    *,
    sim_connected: bool,
    sim_ready: bool,
    sequence_valid: bool,
    sequence_count: int,
    selected_step_valid: bool,
    operation_state: OperationState,
    playback_active: bool,
    playback_paused: bool,
    playback_scheduled: bool,
) -> PlaybackAvailability:
    """Return every Playback control state from one immutable snapshot."""

    has_sequence = bool(sequence_valid and int(sequence_count) > 0)
    reason = ""
    if playback_active or playback_scheduled or operation_state is OperationState.PLAYBACK:
        reason = "Playback is already active."
    elif operation_state is OperationState.RECORDING:
        reason = "Recording is active."
    elif operation_state is OperationState.SCENE_UPDATE:
        reason = "Scene update is in progress."
    elif operation_state is OperationState.RESPAWNING:
        reason = "Robot respawn is in progress."
    elif operation_state is not OperationState.IDLE:
        reason = f"Operation {operation_state.value} is active."
    elif not sim_connected:
        reason = "Simulation is not connected."
    elif not sim_ready:
        reason = "Simulation worker is not ready."
    elif not has_sequence:
        reason = "No valid sequence is loaded."

    can_start = not reason
    busy = operation_state in {
        OperationState.RECORDING,
        OperationState.SCENE_UPDATE,
        OperationState.RESPAWNING,
    }
    can_respawn_start = bool(
        sim_connected
        and has_sequence
        and not busy
        and not playback_active
        and not playback_scheduled
        and operation_state is OperationState.IDLE
    )
    return PlaybackAvailability(
        can_start=can_start,
        can_respawn_start=can_respawn_start,
        can_play_selected=bool(can_start and selected_step_valid),
        can_pause=bool(playback_active and not playback_paused),
        can_resume=bool(playback_active and playback_paused),
        can_stop=bool(playback_active or playback_scheduled),
        can_analyze=has_sequence,
        can_export=has_sequence,
        reason=reason,
    )
