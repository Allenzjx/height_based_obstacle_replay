"""Timed playback for height-indexed accepted steps."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from command_model import (
    CommandMessage,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    clamp_servo_command,
    resolve_servo_targets_for_command,
    resolve_servo_targets_for_group_part,
    validate_motion_command,
)
from sequence_model import clone_command_state, event_playback_commands, load_steps_jsonl, normalize_step, semantic_motion_groups
from playback_progress import PlaybackProgress, PlaybackState
from motion_speed import load_motion_reference


MOTION_REFERENCE = load_motion_reference()
SERVO_REFERENCE_VELOCITY_DEG_S = MOTION_REFERENCE.servo_reference_velocity_deg_s


@dataclass
class PlaybackEvent:
    time_s: float
    command: str
    source_step: int | None = None
    source_step_id: str = ""
    command_index_in_step: int = 0
    commands_in_step: int = 0
    global_command_index: int = 0
    base_command: str = ""
    base_time_s: float = 0.0
    base_duration_s: float = 0.0
    planned_duration_s: float = 0.0
    base_servo_target: float | None = None
    base_wheel_velocity: tuple[float, ...] = ()
    base_wheel_distance: tuple[float, ...] = ()
    segment_index: int = 0
    channel: str = "other"
    dispatch_command: bool = True
    planned_end_s: float = 0.0
    gap_reason: str = "none"
    servo_targets: tuple[tuple[str, float], ...] = ()
    servo_base_velocity_deg_s: float = SERVO_REFERENCE_VELOCITY_DEG_S
    servo_duration_s: float = 0.0
    wheel_requested_velocity_rad_s: tuple[float, ...] = ()
    wheel_applied_target_rad_s: tuple[float, ...] = ()
    wheel_active_duration_s: float = 0.0
    wheel_displacement: tuple[float, ...] = ()


@dataclass
class PlaybackSegment:
    segment_index: int
    source_step: int
    source_step_id: str
    event_start_index: int
    event_count: int
    planned_start_s: float
    planned_end_s: float
    base_duration_s: float
    servo_base_duration_s: float
    servo_duration_s: float
    servo_targets: dict[str, float] = field(default_factory=dict)
    wheel_active_duration_s: float = 0.0
    wheel_base_velocity: dict[str, float] = field(default_factory=dict)
    wheel_requested_velocity_rad_s: dict[str, float] = field(default_factory=dict)
    wheel_applied_target_rad_s: dict[str, float] = field(default_factory=dict)
    explicit_hold_s: float = 0.0
    implicit_idle_before_s: float = 0.0
    gap_reason: str = "none"
    servo_tolerance_deg: float = 1.0
    recorded_servo_residual_deg: dict[str, float] = field(default_factory=dict)
    legacy_missing_endpoint: bool = False


@dataclass
class PlaybackPlan:
    path: Path | None
    events: list[PlaybackEvent]
    final_time_s: float
    label: str = ""
    plan_sha256: str = ""
    source_sha256: str = ""
    profile: str = "raw"
    total_steps: int = 0
    selected_playback: bool = False
    final_pad_s: float = 0.0
    timing: dict[str, Any] = field(default_factory=dict)
    segments: list[PlaybackSegment] = field(default_factory=list)


def plan_from_steps(
    steps: list[dict[str, Any]],
    *,
    profile: str = "fast",
    max_wheel_speed: float | None = None,
    servo_reference_velocity_deg_s: float = SERVO_REFERENCE_VELOCITY_DEG_S,
    label: str = "accepted steps",
    sequence_total_steps: int | None = None,
) -> PlaybackPlan:
    build_started = time.monotonic()
    normalized_profile = "raw" if str(profile).lower() == "raw" else "motion_only"
    servo_reference_velocity_deg_s = max(1.0e-9, float(servo_reference_velocity_deg_s))
    normalized_steps = [normalize_step(step) for step in steps]
    normalized_by_index = {
        int(step.get("index", index) or index): step
        for index, step in enumerate(normalized_steps, start=1)
    }
    groups, step_timing = semantic_motion_groups(normalized_steps)
    raw_sequence_duration = sum(float(step.get("duration", 0.0)) for step in normalized_steps)
    fast_end = max([float(row.get("fast_sequence_end_s", 0.0)) for row in step_timing] + [0.0])
    command_counts: dict[int, int] = {}
    for group in groups:
        step_index = int(group["source_step"])
        command_counts[step_index] = command_counts.get(step_index, 0) + len(group["commands"])

    events: list[PlaybackEvent] = []
    segments: list[PlaybackSegment] = []
    per_step_seen: dict[int, int] = {}
    global_index = 0
    cursor = 0.0
    state = clone_command_state(normalized_steps[0].get("command_state_before") if normalized_steps else None)
    previous_raw_time = 0.0
    previous_recorded_motion_duration = 0.0

    for group_index, group in enumerate(groups):
        base_time = float(group.get("base_time", 0.0))
        raw_time = float(group.get("raw_time", base_time))
        if group_index == 0:
            implicit_idle_before = max(0.0, raw_time)
        else:
            raw_gap = max(0.0, raw_time - previous_raw_time)
            implicit_idle_before = max(0.0, raw_gap - previous_recorded_motion_duration)
        if normalized_profile != "raw":
            implicit_idle_before = 0.0
        cursor += implicit_idle_before

        step_index = int(group["source_step"])
        step_id = str(group["source_step_id"])
        commands = [str(command).strip() for command in group["commands"] if str(command).strip()]
        next_base = float(groups[group_index + 1]["base_time"]) if group_index + 1 < len(groups) else fast_end
        base_interval = max(0.0, next_base - base_time)
        explicit_hold = max([_explicit_wait_duration(command) for command in commands] + [0.0])

        servo_targets: dict[str, float] = {}
        servo_delta = 0.0
        wheel_continuations: dict[str, bool] = {}
        state_before_group = clone_command_state(state)
        for base_command in commands:
            targets = _servo_targets_from_command(base_command)
            for name, target in targets.items():
                servo_targets[name] = target
                servo_delta = max(servo_delta, abs(float(target) - float(state["servos"].get(name, 0.0))))
            before_wheels = dict(state["wheels"])
            from sequence_model import apply_command_to_state

            apply_command_to_state(state, base_command)
            if _wheel_values_from_command(base_command):
                wheel_continuations[base_command] = before_wheels == state["wheels"] and any(
                    abs(float(value)) > 1.0e-9 for value in state["wheels"].values()
                )

        servo_base_velocity = servo_reference_velocity_deg_s
        servo_base_duration = servo_delta / servo_base_velocity if servo_delta > 0.0 else 0.0
        servo_duration = servo_base_duration
        wheel_active = any(abs(float(value)) > 1.0e-9 for value in state["wheels"].values())
        wheel_duration = base_interval if wheel_active else 0.0
        wheel_base = {name: float(state["wheels"].get(name, 0.0)) for name in WHEEL_JOINT_NAMES}
        wheel_requested = dict(wheel_base)
        wheel_applied = {
            name: _clamp_wheel_value(value, max_wheel_speed)
            for name, value in wheel_requested.items()
        }

        segment_duration = max(servo_duration, wheel_duration, explicit_hold)
        planned_start = cursor
        planned_end = planned_start + segment_duration
        gap_reason = "implicit_timestamp_gap" if implicit_idle_before > 0.0 else ("explicit_hold" if explicit_hold > 0.0 else "none")
        event_start = len(events)
        for base_command in commands:
            global_index += 1
            per_step_seen[step_index] = per_step_seen.get(step_index, 0) + 1
            command_servo_targets = _servo_targets_from_command(base_command)
            wheel_values = _wheel_values_from_command(base_command)
            channel = "servo" if command_servo_targets else "wheel" if wheel_values else "timing" if _is_timing_only_command(base_command) else "other"
            validated = validate_motion_command(
                base_command,
                default_wheel_speed_rad_s=0.0,
                max_wheel_speed_rad_s=max_wheel_speed if max_wheel_speed is not None else float("inf"),
            )
            planned_command = validated.command
            requested_values = tuple(wheel_values)
            effective_values = tuple(_clamp_wheel_value(value, max_wheel_speed) for value in requested_values)
            channel_duration = servo_duration if channel == "servo" else wheel_duration if channel == "wheel" else explicit_hold if channel == "timing" else segment_duration
            channel_base_duration = servo_base_duration if channel == "servo" else wheel_duration if channel == "wheel" else explicit_hold if channel == "timing" else base_interval
            target = _servo_target_from_command(base_command)
            events.append(
                PlaybackEvent(
                    time_s=planned_start,
                    command=planned_command,
                    source_step=step_index,
                    source_step_id=step_id,
                    command_index_in_step=per_step_seen[step_index],
                    commands_in_step=command_counts.get(step_index, 0),
                    global_command_index=global_index,
                    base_command=base_command,
                    base_time_s=base_time,
                    base_duration_s=channel_base_duration,
                    planned_duration_s=channel_duration,
                    base_servo_target=target,
                    base_wheel_velocity=tuple(wheel_values),
                    base_wheel_distance=tuple(value * wheel_duration for value in wheel_values),
                    segment_index=group_index,
                    channel=channel,
                    dispatch_command=not bool(wheel_continuations.get(base_command, False)),
                    planned_end_s=planned_start + channel_duration,
                    gap_reason=gap_reason,
                    servo_targets=tuple(command_servo_targets.items()),
                    servo_base_velocity_deg_s=servo_base_velocity,
                    servo_duration_s=servo_duration if command_servo_targets else 0.0,
                    wheel_requested_velocity_rad_s=requested_values,
                    wheel_applied_target_rad_s=effective_values,
                    wheel_active_duration_s=wheel_duration if channel == "wheel" else 0.0,
                    wheel_displacement=tuple(value * wheel_duration for value in effective_values),
                )
            )
        segments.append(
            PlaybackSegment(
                segment_index=group_index,
                source_step=step_index,
                source_step_id=step_id,
                event_start_index=event_start,
                event_count=len(events) - event_start,
                planned_start_s=planned_start,
                planned_end_s=planned_end,
                base_duration_s=base_interval,
                servo_base_duration_s=servo_base_duration,
                servo_duration_s=servo_duration,
                servo_targets=servo_targets,
                wheel_active_duration_s=wheel_duration,
                wheel_base_velocity=wheel_base,
                wheel_requested_velocity_rad_s=wheel_requested,
                wheel_applied_target_rad_s=wheel_applied,
                explicit_hold_s=explicit_hold,
                implicit_idle_before_s=implicit_idle_before,
                gap_reason=gap_reason,
                **_recorded_servo_completion_policy(
                    normalized_by_index.get(step_index, {}),
                    servo_targets,
                ),
            )
        )
        cursor = planned_end
        previous_raw_time = raw_time
        previous_recorded_motion_duration = max(servo_base_duration, wheel_duration, explicit_hold)

    final_idle_s = 0.0
    if groups and normalized_profile == "raw":
        final_idle_s = max(0.0, raw_sequence_duration - float(groups[-1].get("raw_time", 0.0)) - previous_recorded_motion_duration)
        cursor += final_idle_s
    total_steps = int(sequence_total_steps) if sequence_total_steps is not None else max([int(step.get("index", 0)) for step in normalized_steps] + [len(normalized_steps)])
    input_step_indices = [int(step.get("index", index) or index) for index, step in enumerate(normalized_steps, start=1)]
    required_step_indices = [
        int(step.get("index", index) or index)
        for index, step in enumerate(normalized_steps, start=1)
        if event_playback_commands(step) or float(step.get("duration", 0.0) or 0.0) > 0.0
    ]
    represented_step_indices = sorted({int(event.source_step) for event in events if event.source_step is not None})
    plan = PlaybackPlan(
        path=None,
        events=events,
        segments=segments,
        final_time_s=cursor,
        label=label,
        profile=normalized_profile,
        total_steps=total_steps,
        selected_playback=bool(len(normalized_steps) == 1 and total_steps > 1),
        final_pad_s=0.0,
        timing={
            "plan_build_start": build_started,
            "plan_build_end": time.monotonic(),
            "step_diagnostics": copy.deepcopy(step_timing),
            "raw_sequence_duration_s": raw_sequence_duration,
            "compacted_sequence_duration_s": fast_end,
            "final_implicit_idle_s": final_idle_s,
            "fixed_step_gap_s": 0.0,
            "scheduler_model": "continuous_worker_completion_aware_v2",
            "actuator_command_semantics": "direct_recorded_values_v1",
            "wheel_duration_source": "recorded_simulation_time",
            "servo_reference_velocity_deg_s": servo_reference_velocity_deg_s,
            "plan_integrity": {
                "input_step_count": len(normalized_steps),
                "input_step_indices": input_step_indices,
                "required_step_indices": required_step_indices,
                "represented_step_indices": represented_step_indices,
                "missing_required_step_indices": sorted(set(required_step_indices) - set(represented_step_indices)),
                "event_count": len(events),
                "segment_count": len(segments),
            },
        },
    )
    plan.plan_sha256 = plan_fingerprint(plan)
    return plan


def _recorded_servo_completion_policy(step: dict[str, Any], targets: dict[str, float]) -> dict[str, Any]:
    """Use the recorded loaded endpoint as contact-load evidence, never as a target rewrite."""

    sim_state_after = dict(step.get("sim_state_after", {}) or {})
    actual = dict(sim_state_after.get("actual_joint_state", {}) or {})
    actual_servos = dict(actual.get("servos", {}) or {})
    target_state = dict(dict(step.get("sim_state_after", {}) or {}).get("target_joint_state", {}) or {})
    recorded_targets = dict(target_state.get("servos", {}) or {})
    residuals: dict[str, float] = {}
    for name in targets:
        actual_row = actual_servos.get(name, {})
        target_row = recorded_targets.get(name, {})
        if not isinstance(actual_row, dict) or not isinstance(target_row, dict):
            continue
        measured = actual_row.get("deg")
        expected = target_row.get("target_actual_deg", target_row.get("actual_target_deg"))
        try:
            if measured is not None and expected is not None:
                residuals[name] = float(measured) - float(expected)
        except (TypeError, ValueError):
            continue
    recorded_max = max([abs(value) for value in residuals.values()] or [0.0])
    if targets and set(residuals) != set(targets):
        return {
            "servo_tolerance_deg": 1.0,
            "recorded_servo_residual_deg": {},
            "legacy_missing_endpoint": True,
        }
    # 1 degree remains the normal threshold.  Real formal-replay measurements
    # showed that a 0.5 degree margin sat on the edge of normal contact jitter,
    # so loaded endpoints use a measured 0.75 degree margin.  The independent
    # hard 3 degree safety ceiling is unchanged.
    tolerance = min(3.0, max(1.0, recorded_max + 0.75))
    return {
        "servo_tolerance_deg": tolerance,
        "recorded_servo_residual_deg": residuals,
        "legacy_missing_endpoint": False,
    }


def validate_plan_integrity(
    plan: PlaybackPlan,
    *,
    expected_plan_sha256: str = "",
    expected_event_count: int | None = None,
    expected_segment_count: int | None = None,
) -> dict[str, Any]:
    actual_sha = plan_fingerprint(plan)
    integrity = dict(plan.timing.get("plan_integrity", {}) or {})
    missing = [int(value) for value in list(integrity.get("missing_required_step_indices", []) or [])]
    errors: list[str] = []
    if not plan.events:
        errors.append("plan is empty")
    if not plan.segments:
        errors.append("plan has no segments")
    if expected_event_count is not None and len(plan.events) != int(expected_event_count):
        errors.append(f"event count mismatch expected={int(expected_event_count)} decoded={len(plan.events)}")
    if expected_segment_count is not None and len(plan.segments) != int(expected_segment_count):
        errors.append(f"segment count mismatch expected={int(expected_segment_count)} decoded={len(plan.segments)}")
    if expected_plan_sha256 and actual_sha != str(expected_plan_sha256):
        errors.append(f"plan sha mismatch expected={expected_plan_sha256} decoded={actual_sha}")
    if missing:
        errors.append(f"missing required step boundaries: {missing}")
    for index, segment in enumerate(plan.segments):
        start = int(segment.event_start_index)
        stop = start + int(segment.event_count)
        if start < 0 or stop > len(plan.events) or stop < start:
            errors.append(f"segment {index} event range is invalid: {start}:{stop}/{len(plan.events)}")
            break
    return {
        "ok": not errors,
        "errors": errors,
        "plan_sha256": actual_sha,
        "event_count": len(plan.events),
        "segment_count": len(plan.segments),
        "input_step_count": int(integrity.get("input_step_count", plan.total_steps) or 0),
        "represented_step_indices": list(integrity.get("represented_step_indices", []) or []),
        "missing_required_step_indices": missing,
    }


def _explicit_wait_duration(command: str) -> float:
    tokens = command_tokens(command)
    if len(tokens) >= 2 and tokens[0].lower() in {"wait", "hold", "sleep"}:
        try:
            return max(0.0, float(tokens[1]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _is_timing_only_command(command: str) -> bool:
    tokens = command_tokens(command)
    return bool(tokens and tokens[0].lower() in {"wait", "hold", "sleep"})


def _servo_target_from_command(command: str) -> float | None:
    tokens = command_tokens(command)
    if not tokens or tokens[0].lower() not in {"servo", "angle"}:
        return None
    try:
        return float(tokens[-1])
    except (TypeError, ValueError):
        return None


def _servo_targets_from_command(command: str) -> dict[str, float]:
    tokens = command_tokens(command)
    if not tokens or tokens[0].lower() not in {"servo", "angle"}:
        return {}
    try:
        if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
            names = resolve_servo_targets_for_group_part(tokens[1], tokens[2])
            target = float(tokens[3])
        elif len(tokens) == 3:
            names = resolve_servo_targets_for_command(tokens[1])
            target = float(tokens[2])
        else:
            return {}
    except (TypeError, ValueError):
        return {}
    return {name: clamp_servo_command(name, target) for name in names}


def _clamp_wheel_value(value: float, max_wheel_speed: float | None) -> float:
    if max_wheel_speed is None or float(max_wheel_speed) <= 0.0:
        return float(value)
    limit = abs(float(max_wheel_speed))
    return max(-limit, min(limit, float(value)))


def _is_float(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _wheel_values_from_command(command: str) -> list[float]:
    tokens = command_tokens(command)
    if not tokens or tokens[0].lower() not in {"wheel", "wheels", "speed"}:
        return []
    try:
        if len(tokens) >= 2 and tokens[1].lower() == "stop":
            return [0.0]
        if tokens[0].lower() in {"wheels", "speed"} and len(tokens) == 3:
            return [float(tokens[1]), float(tokens[2])]
        if tokens[0].lower() == "wheel" and len(tokens) == 3:
            if _is_float(tokens[1]):
                return [float(tokens[1]), float(tokens[2])]
            return [float(tokens[2])]
    except (TypeError, ValueError):
        return []
    return []


def _apply_wheel_command_to_state(state: dict[str, float], command: str) -> None:
    tokens = command_tokens(command)
    if not tokens:
        return
    verb = tokens[0].lower()
    try:
        if (verb == "wheel" and len(tokens) >= 2 and tokens[1].lower() == "stop") or (verb in {"wheels", "speed"} and len(tokens) >= 2 and tokens[1].lower() == "stop"):
            state.clear()
        elif verb in {"wheels", "speed"} and len(tokens) == 3:
            state.update(left=float(tokens[1]), right=float(tokens[2]))
        elif verb == "wheel" and len(tokens) == 3 and _is_float(tokens[1]):
            state.update(left=float(tokens[1]), right=float(tokens[2]))
        elif verb == "wheel" and len(tokens) == 3:
            key = str(tokens[1]).lower()
            value = float(tokens[2])
            if key == "all":
                state.clear()
                state["all"] = value
            else:
                state[key] = value
    except (TypeError, ValueError):
        return


def plan_from_jsonl(path: str | Path, **kwargs: Any) -> PlaybackPlan:
    source = Path(path)
    plan = plan_from_steps(load_steps_jsonl(source), label=str(source), **kwargs)
    plan.path = source
    plan.source_sha256 = file_sha256(source)
    plan.plan_sha256 = plan_fingerprint(plan)
    return plan


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_fingerprint(plan: PlaybackPlan) -> str:
    payload = {
        "final_time_s": round(float(plan.final_time_s), 9),
        "profile": plan.profile,
        "total_steps": int(plan.total_steps),
        "events": [
            {
                "time_s": round(float(event.time_s), 9),
                "command": str(event.command),
                "source_step": event.source_step,
                "source_step_id": event.source_step_id,
                "command_index_in_step": event.command_index_in_step,
                "commands_in_step": event.commands_in_step,
                "global_command_index": event.global_command_index,
                "base_command": event.base_command,
                "base_duration_s": round(float(event.base_duration_s), 9),
                "planned_duration_s": round(float(event.planned_duration_s), 9),
                "segment_index": int(event.segment_index),
                "channel": event.channel,
                "dispatch_command": bool(event.dispatch_command),
            }
            for event in plan.events
        ],
        "segments": [
            {
                "segment_index": int(segment.segment_index),
                "source_step": int(segment.source_step),
                "event_start_index": int(segment.event_start_index),
                "event_count": int(segment.event_count),
                "servo_tolerance_deg": round(float(segment.servo_tolerance_deg), 9),
                "recorded_servo_residual_deg": {
                    name: round(float(value), 9)
                    for name, value in sorted(segment.recorded_servo_residual_deg.items())
                },
                "legacy_missing_endpoint": bool(segment.legacy_missing_endpoint),
            }
            for segment in plan.segments
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def playback_plan_to_payload(plan: PlaybackPlan) -> dict[str, Any]:
    return {
        "path": str(plan.path) if plan.path is not None else "",
        "label": str(plan.label or ""),
        "final_time_s": float(plan.final_time_s),
        "plan_sha256": plan.plan_sha256 or plan_fingerprint(plan),
        "declared_event_count": len(plan.events),
        "declared_segment_count": len(plan.segments),
        "source_sha256": str(plan.source_sha256 or ""),
        "profile": str(plan.profile or "raw"),
        "total_steps": int(plan.total_steps),
        "selected_playback": bool(plan.selected_playback),
        "final_pad_s": float(plan.final_pad_s),
        "timing": copy.deepcopy(plan.timing),
        "segments": [
            {
                "segment_index": int(segment.segment_index),
                "source_step": int(segment.source_step),
                "source_step_id": str(segment.source_step_id),
                "event_start_index": int(segment.event_start_index),
                "event_count": int(segment.event_count),
                "planned_start_s": float(segment.planned_start_s),
                "planned_end_s": float(segment.planned_end_s),
                "base_duration_s": float(segment.base_duration_s),
                "servo_base_duration_s": float(segment.servo_base_duration_s),
                "servo_duration_s": float(segment.servo_duration_s),
                "servo_targets": dict(segment.servo_targets),
                "wheel_active_duration_s": float(segment.wheel_active_duration_s),
                "wheel_base_velocity": dict(segment.wheel_base_velocity),
                "wheel_requested_velocity_rad_s": dict(segment.wheel_requested_velocity_rad_s),
                "wheel_applied_target_rad_s": dict(segment.wheel_applied_target_rad_s),
                "explicit_hold_s": float(segment.explicit_hold_s),
                "implicit_idle_before_s": float(segment.implicit_idle_before_s),
                "gap_reason": str(segment.gap_reason),
                "servo_tolerance_deg": float(segment.servo_tolerance_deg),
                "recorded_servo_residual_deg": dict(segment.recorded_servo_residual_deg),
                "legacy_missing_endpoint": bool(segment.legacy_missing_endpoint),
            }
            for segment in plan.segments
        ],
        "events": [
            {
                "time_s": float(event.time_s),
                "command": str(event.command),
                "source_step": event.source_step,
                "source_step_id": event.source_step_id,
                "command_index_in_step": int(event.command_index_in_step),
                "commands_in_step": int(event.commands_in_step),
                "global_command_index": int(event.global_command_index),
                "base_command": str(event.base_command or event.command),
                "base_time_s": float(event.base_time_s),
                "base_duration_s": float(event.base_duration_s),
                "planned_duration_s": float(event.planned_duration_s),
                "base_servo_target": event.base_servo_target,
                "base_wheel_velocity": list(event.base_wheel_velocity),
                "base_wheel_distance": list(event.base_wheel_distance),
                "segment_index": int(event.segment_index),
                "channel": str(event.channel),
                "dispatch_command": bool(event.dispatch_command),
                "planned_end_s": float(event.planned_end_s),
                "gap_reason": str(event.gap_reason),
                "servo_targets": [[name, value] for name, value in event.servo_targets],
                "servo_base_velocity_deg_s": float(event.servo_base_velocity_deg_s),
                "servo_duration_s": float(event.servo_duration_s),
                "wheel_requested_velocity_rad_s": list(event.wheel_requested_velocity_rad_s),
                "wheel_applied_target_rad_s": list(event.wheel_applied_target_rad_s),
                "wheel_active_duration_s": float(event.wheel_active_duration_s),
                "wheel_displacement": list(event.wheel_displacement),
            }
            for event in plan.events
        ],
    }


def playback_plan_from_payload(payload: dict[str, Any]) -> PlaybackPlan:
    data = dict(payload or {})
    events = [
        PlaybackEvent(
            time_s=float(row.get("time_s", row.get("time", 0.0)) or 0.0),
            command=str(row.get("command", "") or ""),
            source_step=_optional_int(row.get("source_step")),
            source_step_id=str(row.get("source_step_id", "") or ""),
            command_index_in_step=int(row.get("command_index_in_step", 0) or 0),
            commands_in_step=int(row.get("commands_in_step", 0) or 0),
            global_command_index=int(row.get("global_command_index", 0) or 0),
            base_command=str(row.get("base_command", row.get("command", "")) or ""),
            base_time_s=float(row.get("base_time_s", 0.0) or 0.0),
            base_duration_s=float(row.get("base_duration_s", 0.0) or 0.0),
            planned_duration_s=float(row.get("planned_duration_s", 0.0) or 0.0),
            base_servo_target=float(row["base_servo_target"]) if row.get("base_servo_target") is not None else None,
            base_wheel_velocity=tuple(float(value) for value in list(row.get("base_wheel_velocity", []) or [])),
            base_wheel_distance=tuple(float(value) for value in list(row.get("base_wheel_distance", []) or [])),
            segment_index=int(row.get("segment_index", 0) or 0),
            channel=str(row.get("channel", "other") or "other"),
            dispatch_command=bool(row.get("dispatch_command", True)),
            planned_end_s=float(row.get("planned_end_s", 0.0) or 0.0),
            gap_reason=str(row.get("gap_reason", "none") or "none"),
            servo_targets=tuple((str(item[0]), float(item[1])) for item in list(row.get("servo_targets", []) or []) if isinstance(item, (list, tuple)) and len(item) == 2),
            servo_base_velocity_deg_s=float(row.get("servo_base_velocity_deg_s", SERVO_REFERENCE_VELOCITY_DEG_S) or SERVO_REFERENCE_VELOCITY_DEG_S),
            servo_duration_s=float(row.get("servo_duration_s", 0.0) or 0.0),
            wheel_requested_velocity_rad_s=tuple(float(value) for value in list(row.get("wheel_requested_velocity_rad_s", []) or [])),
            wheel_applied_target_rad_s=tuple(float(value) for value in list(row.get("wheel_applied_target_rad_s", []) or [])),
            wheel_active_duration_s=float(row.get("wheel_active_duration_s", 0.0) or 0.0),
            wheel_displacement=tuple(float(value) for value in list(row.get("wheel_displacement", []) or [])),
        )
        for row in list(data.get("events", []) or [])
        if str(row.get("command", "") if isinstance(row, dict) else "").strip()
    ]
    events.sort(key=lambda event: (event.segment_index, event.global_command_index, event.time_s))
    segments = [
        PlaybackSegment(
            segment_index=int(row.get("segment_index", index) or 0),
            source_step=int(row.get("source_step", 0) or 0),
            source_step_id=str(row.get("source_step_id", "") or ""),
            event_start_index=int(row.get("event_start_index", 0) or 0),
            event_count=int(row.get("event_count", 0) or 0),
            planned_start_s=float(row.get("planned_start_s", 0.0) or 0.0),
            planned_end_s=float(row.get("planned_end_s", 0.0) or 0.0),
            base_duration_s=float(row.get("base_duration_s", 0.0) or 0.0),
            servo_base_duration_s=float(row.get("servo_base_duration_s", 0.0) or 0.0),
            servo_duration_s=float(row.get("servo_duration_s", 0.0) or 0.0),
            servo_targets={str(name): float(value) for name, value in dict(row.get("servo_targets", {}) or {}).items()},
            wheel_active_duration_s=float(row.get("wheel_active_duration_s", 0.0) or 0.0),
            wheel_base_velocity={str(name): float(value) for name, value in dict(row.get("wheel_base_velocity", {}) or {}).items()},
            wheel_requested_velocity_rad_s={str(name): float(value) for name, value in dict(row.get("wheel_requested_velocity_rad_s", {}) or {}).items()},
            wheel_applied_target_rad_s={str(name): float(value) for name, value in dict(row.get("wheel_applied_target_rad_s", {}) or {}).items()},
            explicit_hold_s=float(row.get("explicit_hold_s", 0.0) or 0.0),
            implicit_idle_before_s=float(row.get("implicit_idle_before_s", 0.0) or 0.0),
            gap_reason=str(row.get("gap_reason", "none") or "none"),
            servo_tolerance_deg=float(row.get("servo_tolerance_deg", 1.0) or 1.0),
            recorded_servo_residual_deg={
                str(name): float(value)
                for name, value in dict(row.get("recorded_servo_residual_deg", {}) or {}).items()
            },
            legacy_missing_endpoint=bool(row.get("legacy_missing_endpoint", False)),
        )
        for index, row in enumerate(list(data.get("segments", []) or []))
        if isinstance(row, dict)
    ]
    path_text = str(data.get("path", "") or "")
    plan = PlaybackPlan(
        path=Path(path_text) if path_text else None,
        events=events,
        final_time_s=float(data.get("final_time_s", 0.0) or 0.0),
        label=str(data.get("label", "") or ""),
        source_sha256=str(data.get("source_sha256", "") or ""),
        profile=str(data.get("profile", "raw") or "raw"),
        total_steps=int(data.get("total_steps", 0) or 0),
        selected_playback=bool(data.get("selected_playback", False)),
        final_pad_s=float(data.get("final_pad_s", 0.0) or 0.0),
        timing=copy.deepcopy(dict(data.get("timing", {}) or {})),
        segments=segments,
    )
    plan.plan_sha256 = str(data.get("plan_sha256", "") or "") or plan_fingerprint(plan)
    return plan


class SimTimePlaybackService:
    """Non-blocking playback scheduler keyed to adapter simulation time."""

    def __init__(self, *, max_events_per_step: int = 1000):
        self.max_events_per_step = max(1, int(max_events_per_step))
        self.plan: PlaybackPlan | None = None
        self.plan_id = ""
        self.request_id = ""
        self.worker_session_id = ""
        self.active = False
        self.paused = False
        self.started = False
        self.index = 0
        self.events_sent = 0
        self.start_sim_time_s = 0.0
        self.start_wall_time_s = 0.0
        self.pause_started_sim_time_s = 0.0
        self.paused_sim_accum_s = 0.0
        self.completed_at_wall_s = 0.0
        self.completed_at_sim_s = 0.0
        self.stop_reason = ""
        self.last_info = ""
        self.last_error = ""
        self.last_event_command = ""
        self.last_event_time_s = 0.0
        self.last_event_sim_time_s = 0.0
        self.max_dispatch_jitter_s = 0.0
        self.dispatch_jitter_samples: list[float] = []
        self.progress = PlaybackProgress()
        self.timing_trace: dict[str, Any] = {}
        self.active_wheel_command = ""
        self.resume_wheel_command_pending = False
        self.wheel_was_stopped_for_pause = False
        self.paused_wheel_state: dict[str, float] = {}
        self.segment_index = 0
        self.segment_active = False
        self.segment_start_elapsed_s = 0.0
        self.next_segment_start_elapsed_s = 0.0
        self.segment_wheel_stopped = False
        self.wheel_stopped_for_channel_completion = False
        self.servo_position_tolerance_deg = 1.0
        self.current_servo_errors: dict[str, float] = {}
        self.previous_step_end_sim_time_s: float | None = None
        self.segment_start_joint_positions: dict[str, float] = {}
        self.resume_servo_motion_pending = False
        self.segment_best_servo_error_deg = float("inf")
        self.segment_last_servo_improvement_elapsed_s = 0.0
        self.first_command_applied = False
        self.first_command_applied_sim_time_s = 0.0
        self.first_command_applied_sim_step = 0
        self.first_motion_planned_s = 0.0
        self.servo_residual_warnings: list[dict[str, Any]] = []
        self.contact_residual_grace_s = 0.25
        self.contact_residual_stable_s = 0.10
        self.actuator_divergence_window_s = 1.50
        self.contact_residual_hard_cap_deg = 3.0
        self.segment_last_servo_error_deg: float | None = None
        self.segment_last_servo_change_elapsed_s = 0.0
        self.segment_last_servo_worsening_elapsed_s = 0.0
        self.segment_contact_within_tolerance_since_s: float | None = None
        self.segment_contact_error_history: list[tuple[float, float]] = []
        # A genuinely stalled actuator now fails quickly instead of making a
        # short command look like a multi-second UI/playback hang.
        self.servo_stall_window_s = 0.75
        self.servo_improvement_epsilon_deg = 0.05

    def start_plan(
        self,
        plan: PlaybackPlan,
        *,
        current_sim_time_s: float,
        current_wall_time_s: float,
        start_delay_sim_s: float = 0.0,
        plan_id: str = "",
        request_id: str = "",
        worker_session_id: str = "",
    ) -> bool:
        self.plan = copy.deepcopy(plan)
        self.plan.plan_sha256 = self.plan.plan_sha256 or plan_fingerprint(self.plan)
        self.plan_id = str(plan_id or self.plan.plan_sha256[:16])
        self.request_id = str(request_id or "")
        self.worker_session_id = str(worker_session_id or "")
        self.active = bool(self.plan.events)
        self.paused = False
        self.started = False
        self.index = 0
        self.events_sent = 0
        self.start_sim_time_s = float(current_sim_time_s) + max(0.0, float(start_delay_sim_s))
        self.start_wall_time_s = float(current_wall_time_s)
        self.pause_started_sim_time_s = 0.0
        self.paused_sim_accum_s = 0.0
        self.completed_at_wall_s = 0.0
        self.completed_at_sim_s = 0.0
        self.stop_reason = ""
        self.last_error = "" if self.active else "empty playback plan"
        self.last_info = (
            f"worker playback scheduled plan={self.plan_id} events={len(self.plan.events)}"
            if self.active
            else self.last_error
        )
        self.last_event_command = ""
        self.last_event_time_s = 0.0
        self.last_event_sim_time_s = 0.0
        self.max_dispatch_jitter_s = 0.0
        self.dispatch_jitter_samples = []
        self.timing_trace = copy.deepcopy(self.plan.timing)
        self.timing_trace["worker_ready_time"] = float(current_wall_time_s)
        self.timing_trace.setdefault("commands", [])
        self.timing_trace.setdefault("steps", [])
        self.active_wheel_command = ""
        self.resume_wheel_command_pending = False
        self.wheel_was_stopped_for_pause = False
        self.paused_wheel_state = {}
        self.segment_index = 0
        self.segment_active = False
        self.segment_start_elapsed_s = 0.0
        self.next_segment_start_elapsed_s = float(self.plan.segments[0].planned_start_s) if self.plan.segments else 0.0
        self.segment_wheel_stopped = False
        self.wheel_stopped_for_channel_completion = False
        self.current_servo_errors = {}
        self.previous_step_end_sim_time_s = None
        self.segment_start_joint_positions = {}
        self.resume_servo_motion_pending = False
        self.segment_best_servo_error_deg = float("inf")
        self.segment_last_servo_improvement_elapsed_s = 0.0
        self.first_command_applied = False
        self.first_command_applied_sim_time_s = 0.0
        self.first_command_applied_sim_step = 0
        motion_events = [event for event in self.plan.events if not _is_timing_only_command(event.command)]
        self.first_motion_planned_s = float(motion_events[0].time_s) if motion_events else 0.0
        self.servo_residual_warnings = []
        self.segment_last_servo_error_deg = None
        self.segment_last_servo_change_elapsed_s = 0.0
        self.segment_last_servo_worsening_elapsed_s = 0.0
        self.segment_contact_within_tolerance_since_s = None
        self.segment_contact_error_history = []
        first = self.plan.events[0] if self.plan.events else None
        selected_step_index = int(first.source_step or 0) if first is not None else 0
        total_steps = int(self.plan.total_steps or len({event.source_step for event in self.plan.events if event.source_step is not None}))
        preparing_text = (
            f"Restore verified. Waiting to play Selected Step {selected_step_index} / {total_steps}..."
            if self.plan.selected_playback and self.active
            else "Waiting for simulation ready..." if self.active else "Error"
        )
        self.progress = PlaybackProgress(
            playback_state=PlaybackState.PREPARING.value if self.active else PlaybackState.ERROR.value,
            status_text=preparing_text,
            current_step_index=selected_step_index,
            total_steps=total_steps,
            current_step_id=str(first.source_step_id if first is not None else ""),
            current_command_index_in_step=0,
            commands_in_current_step=int(first.commands_in_step if first is not None else 0),
            global_command_index=0,
            total_commands=len(self.plan.events),
            playback_profile=str(self.plan.profile or "raw"),
            last_error=self.last_error,
            command_phase="waiting",
            selected_playback=bool(self.plan.selected_playback),
        )
        return self.active

    def update(self, adapter: Any, *, current_sim_time_s: float, current_sim_step: int, current_wall_time_s: float) -> None:
        if not self.active or self.paused or self.plan is None:
            return
        sim_time = float(current_sim_time_s)
        if sim_time + 1.0e-9 < self.start_sim_time_s:
            starts_in = max(0.0, self.start_sim_time_s - sim_time)
            self.last_info = f"worker playback scheduled; starts in {starts_in:.3f}s sim"
            return
        if not self.started:
            self.started = True
            collector = getattr(adapter, "telemetry_collector", None)
            if collector is not None:
                try:
                    collector.start_replay(
                        label=self.plan.label or "worker playback",
                        event_count=len(self.plan.events),
                        final_time_s=float(self.plan.final_time_s),
                        started_sim_time_s=float(self.start_sim_time_s),
                    )
                except Exception as exc:
                    self.last_error = f"telemetry replay start failed: {exc}"
            self.last_info = f"worker playback running plan={self.plan_id}"
            self.progress.playback_state = PlaybackState.PLAYING.value
            self.progress.status_text = (
                f"Restore verified. Playing Selected Step {self.progress.current_step_index} / {self.progress.total_steps}"
                if self.plan.selected_playback
                else "Playing..."
            )
        if self.resume_wheel_command_pending and self.paused_wheel_state:
            if hasattr(adapter, "apply_motion_batch"):
                adapter.apply_motion_batch(
                    {
                        "batch_id": f"playback-resume-{self.plan_id}-{uuid.uuid4().hex[:8]}",
                        "source": "playback",
                        "servo_targets_deg": {},
                        "wheel_targets_rad_s": dict(self.paused_wheel_state),
                    }
                )
            else:
                for name, value in self.paused_wheel_state.items():
                    adapter.handle_command(CommandMessage(text=f"wheel {name} {value:.9g}", source="playback", log_history=False, quiet=True))
            self.resume_wheel_command_pending = False
            self.paused_wheel_state = {}
        if self.resume_servo_motion_pending:
            if hasattr(adapter, "servo_motion_enabled"):
                adapter.servo_motion_enabled = True
            self.resume_servo_motion_pending = False
        elapsed = max(0.0, sim_time - self.start_sim_time_s - self.paused_sim_accum_s)
        if self.plan.segments:
            self._update_segments(
                adapter,
                elapsed=elapsed,
                sim_time=sim_time,
                current_sim_step=int(current_sim_step),
                wall_time=float(current_wall_time_s),
            )
            return
        sent_this_step = 0
        while self.index < len(self.plan.events):
            if sent_this_step >= self.max_events_per_step:
                self.last_info = f"worker playback batch limit at {self.index}/{len(self.plan.events)}"
                break
            event = self.plan.events[self.index]
            if float(event.time_s) > elapsed + 1.0e-6:
                break
            collector = getattr(adapter, "telemetry_collector", None)
            if collector is not None:
                try:
                    collector.record_replay_event(adapter, event, self.index)
                except Exception as exc:
                    self.last_error = f"telemetry replay event failed: {exc}"
            wheel_event = bool(_wheel_values_from_command(event.command))
            pre_dispatch_joint_pos: Any = None
            if wheel_event and hasattr(adapter, "capture_sim_state"):
                try:
                    pre_dispatch_joint_pos = copy.deepcopy(adapter.capture_sim_state().get("joint_pos"))
                except Exception as exc:
                    self.last_error = f"wheel dispatch state capture failed: {exc}"
            if not _is_timing_only_command(event.command):
                adapter.handle_command(
                    CommandMessage(
                        text=event.command,
                        source="playback",
                        log_history=False,
                        quiet=True,
                        playback_label=self.plan.label,
                        playback_event_index=self.index,
                        playback_event_count=len(self.plan.events),
                        playback_final_time_s=float(self.plan.final_time_s),
                        source_step=event.source_step,
                    )
                )
            if wheel_event:
                self.active_wheel_command = "" if "stop" in command_tokens(event.command)[1:] else event.command
            jitter = max(0.0, elapsed - float(event.time_s))
            self.dispatch_jitter_samples.append(float(jitter))
            self.max_dispatch_jitter_s = max(self.max_dispatch_jitter_s, float(jitter))
            self.last_event_command = event.command
            self.last_event_time_s = float(current_wall_time_s)
            self.last_event_sim_time_s = sim_time
            self._record_event_progress(
                event,
                elapsed=elapsed,
                sim_time=sim_time,
                wall_time=float(current_wall_time_s),
                jitter=jitter,
                pre_dispatch_joint_pos=pre_dispatch_joint_pos,
            )
            self.index += 1
            self.events_sent += 1
            sent_this_step += 1
        if self.index >= len(self.plan.events) and elapsed + 1.0e-6 >= float(self.plan.final_time_s):
            self._finish(adapter, success=True, reason="complete", current_sim_time_s=sim_time, current_wall_time_s=current_wall_time_s)

    def _update_segments(
        self,
        adapter: Any,
        *,
        elapsed: float,
        sim_time: float,
        current_sim_step: int,
        wall_time: float,
    ) -> None:
        """Advance a preloaded, completion-aware plan entirely in the worker."""

        if self.plan is None:
            return
        if self.segment_index >= len(self.plan.segments):
            if elapsed + 1.0e-9 >= self.next_segment_start_elapsed_s:
                self._finish(adapter, success=True, reason="complete", current_sim_time_s=sim_time, current_wall_time_s=wall_time)
            return
        cycles = 0
        while self.segment_index < len(self.plan.segments) and cycles < self.max_events_per_step:
            cycles += 1
            segment = self.plan.segments[self.segment_index]
            if not self.segment_active:
                if elapsed + 1.0e-9 < self.next_segment_start_elapsed_s:
                    return
                self._start_segment(
                    adapter,
                    segment,
                    elapsed=elapsed,
                    sim_time=sim_time,
                    current_sim_step=current_sim_step,
                    wall_time=wall_time,
                )

            segment_elapsed = max(0.0, elapsed - self.segment_start_elapsed_s)
            servo_planned_done = segment_elapsed + 1.0e-9 >= float(segment.servo_duration_s)
            servo_done, errors = (
                self._servo_targets_complete(
                    adapter,
                    segment.servo_targets,
                    tolerance_deg=float(segment.servo_tolerance_deg),
                )
                if servo_planned_done
                else (False, {})
            )
            if not segment.servo_targets:
                servo_done = True
            self.current_servo_errors = errors
            max_error_now = max([abs(float(value)) for value in errors.values()] or [0.0])
            if any(not math.isfinite(float(value)) for value in errors.values()):
                self.last_error = (
                    f"invalid_joint_state: non-finite servo error step={segment.source_step} "
                    f"segment={segment.segment_index} errors={errors}"
                )
                self._finish(
                    adapter,
                    success=False,
                    reason="invalid_joint_state",
                    current_sim_time_s=sim_time,
                    current_wall_time_s=wall_time,
                )
                return
            previous_error = self.segment_last_servo_error_deg
            if previous_error is None or abs(max_error_now - previous_error) > self.servo_improvement_epsilon_deg:
                self.segment_last_servo_change_elapsed_s = float(elapsed)
            if previous_error is not None and max_error_now > previous_error + self.servo_improvement_epsilon_deg:
                self.segment_last_servo_worsening_elapsed_s = float(elapsed)
            self.segment_last_servo_error_deg = max_error_now
            contact_candidate = (
                servo_done
                and max_error_now > self.servo_position_tolerance_deg
                and max_error_now <= float(segment.servo_tolerance_deg)
                and bool(segment.recorded_servo_residual_deg)
            )
            if contact_candidate:
                if self.segment_contact_within_tolerance_since_s is None:
                    self.segment_contact_within_tolerance_since_s = float(elapsed)
            else:
                self.segment_contact_within_tolerance_since_s = None
            if servo_planned_done and segment.recorded_servo_residual_deg:
                self.segment_contact_error_history.append((float(elapsed), max_error_now))
                window_start = float(elapsed) - max(
                    self.contact_residual_stable_s,
                    self.actuator_divergence_window_s,
                )
                while len(self.segment_contact_error_history) > 1 and self.segment_contact_error_history[1][0] <= window_start:
                    self.segment_contact_error_history.pop(0)
            contact_extension = max(0.0, segment_elapsed - float(segment.servo_duration_s))
            contact_window_errors = [value for _at, value in self.segment_contact_error_history]
            contact_window_duration_s = (
                self.segment_contact_error_history[-1][0] - self.segment_contact_error_history[0][0]
                if len(self.segment_contact_error_history) >= 2
                else 0.0
            )
            contact_window_ready = bool(
                len(self.segment_contact_error_history) >= 2
                and contact_window_duration_s >= self.contact_residual_stable_s * 0.90
            )
            # Contact-loaded articulated joints can alternate by a fraction of
            # a degree at 120 Hz even though their position residual is bounded.
            # Treat the position as stable only when a complete recent window
            # stays under the independent 3 degree cap and has no material net
            # worsening.  The current sample must still be inside the narrower
            # recording-derived effective tolerance.
            contact_window_cap = min(
                float(self.contact_residual_hard_cap_deg),
                float(segment.servo_tolerance_deg) + 0.5,
            )
            contact_window_min = min(contact_window_errors or [max_error_now])
            contact_window_max = max(contact_window_errors or [max_error_now])
            contact_window_slope = (
                (contact_window_errors[-1] - contact_window_errors[0]) / contact_window_duration_s
                if contact_window_duration_s > 1.0e-9
                else 0.0
            )
            contact_stable = (
                contact_candidate
                and contact_window_ready
                and max(contact_window_errors or [float("inf")]) <= contact_window_cap
                and contact_window_errors[-1] <= contact_window_errors[0] + 0.25
            )
            if contact_candidate:
                # A recorded, contact-loaded residual that is finite and
                # currently inside its bounded effective tolerance completes
                # after one finite grace interval.  Recent jitter is retained
                # as diagnostic evidence; it is not a reason to abort while
                # the current residual remains within the <=3 degree cap.
                servo_done = bool(
                    contact_extension >= self.contact_residual_grace_s
                    and max_error_now <= min(
                        float(segment.servo_tolerance_deg),
                        float(self.contact_residual_hard_cap_deg),
                    )
                )
            wheel_done = segment_elapsed + 1.0e-9 >= float(segment.wheel_active_duration_s)
            hold_done = segment_elapsed + 1.0e-9 >= float(segment.explicit_hold_s)
            segment_done = servo_done and wheel_done and hold_done

            if wheel_done and segment.wheel_active_duration_s > 0.0 and not segment_done and not self.segment_wheel_stopped:
                adapter.stop_wheels()
                if hasattr(adapter, "apply_commands_to_robot"):
                    adapter.apply_commands_to_robot()
                self.segment_wheel_stopped = True
                self.wheel_stopped_for_channel_completion = True
                self.active_wheel_command = ""

            if not segment_done:
                # A mixed servo+wheel segment remains active until both channels
                # finish.  Never run servo-stall logic after the servo channel is
                # already inside tolerance merely because the wheel timer is live.
                if servo_planned_done and segment.servo_targets and not servo_done:
                    extension = max(0.0, segment_elapsed - float(segment.servo_duration_s))
                    max_error = max([abs(float(value)) for value in errors.values()] or [0.0])
                    within_recorded_contact_tolerance = (
                        max_error > self.servo_position_tolerance_deg
                        and max_error <= float(segment.servo_tolerance_deg)
                        and bool(segment.recorded_servo_residual_deg)
                    )
                    if within_recorded_contact_tolerance:
                        self.last_info = (
                            f"contact_residual_grace step={segment.source_step} segment={segment.segment_index} "
                            f"extension={extension:.3f}s bounded={True} stable={contact_stable} "
                            f"error={max_error:.3f}deg recent_min={contact_window_min:.3f} "
                            f"recent_max={contact_window_max:.3f} slope={contact_window_slope:.3f}deg/s"
                        )
                        self.progress.command_phase = "contact_residual_grace"
                        return
                    if max_error + self.servo_improvement_epsilon_deg < self.segment_best_servo_error_deg:
                        self.segment_best_servo_error_deg = max_error
                        self.segment_last_servo_improvement_elapsed_s = float(elapsed)
                    stalled_for = max(0.0, float(elapsed) - self.segment_last_servo_improvement_elapsed_s)
                    velocities = self._servo_target_velocities_deg_s(adapter, segment.servo_targets)
                    velocities_near_zero = bool(velocities) and all(
                        abs(float(value)) <= 0.5 for value in velocities.values()
                    )
                    worst_joint = max(errors, key=lambda name: abs(float(errors[name]))) if errors else "unknown"
                    worst_joint_velocity = velocities.get(worst_joint)
                    worst_joint_error = errors.get(worst_joint)
                    divergence_low_response = bool(
                        worst_joint_velocity is None or abs(float(worst_joint_velocity)) <= 5.0
                    )
                    divergence_not_recovering = bool(
                        worst_joint_velocity is None
                        or abs(float(worst_joint_velocity)) <= 0.5
                        or (
                            worst_joint_error is not None
                            and float(worst_joint_error) * float(worst_joint_velocity) > 0.0
                        )
                    )
                    worsening_outside_hard_tolerance = bool(
                        max_error > float(self.contact_residual_hard_cap_deg)
                        and contact_window_duration_s >= self.actuator_divergence_window_s * 0.90
                        and contact_window_errors[-1] > contact_window_errors[0] + 0.25
                        and contact_window_errors[-1] >= contact_window_max - 0.25
                        and contact_window_slope > 1.0
                        and divergence_low_response
                        and divergence_not_recovering
                    )
                    if worsening_outside_hard_tolerance:
                        self.last_error = (
                            f"actuator_unstable: error is worsening outside the hard safety tolerance; "
                            f"step={segment.source_step} segment={segment.segment_index} joint={worst_joint} "
                            f"error_deg={errors.get(worst_joint)} max_error_deg={max_error:.6f} "
                            f"tolerance_deg={segment.servo_tolerance_deg:.6f} "
                            f"recent_min_deg={contact_window_min:.6f} recent_max_deg={contact_window_max:.6f} "
                            f"recent_slope_deg_s={contact_window_slope:.6f} "
                            f"joint_velocity_deg_s={worst_joint_velocity}"
                        )
                        self._finish(
                            adapter,
                            success=False,
                            reason="actuator_unstable",
                            current_sim_time_s=sim_time,
                            current_wall_time_s=wall_time,
                        )
                        return
                    if extension > 0.0 and stalled_for >= self.servo_stall_window_s and velocities_near_zero:
                        worst_joint = max(errors, key=lambda name: abs(float(errors[name]))) if errors else "unknown"
                        command_target = segment.servo_targets.get(worst_joint)
                        actual_target = (
                            float(adapter.command_to_actual_target_deg(worst_joint, command_target))
                            if command_target is not None and hasattr(adapter, "command_to_actual_target_deg")
                            else command_target
                        )
                        actual_deg = None if actual_target is None else float(actual_target) + float(errors.get(worst_joint, 0.0))
                        self.last_error = (
                            f"actuator_limit: servo target did not improve for {stalled_for:.3f}s sim; "
                            f"step={segment.source_step} segment={segment.segment_index} joint={worst_joint} "
                            f"requested_command_deg={command_target} expected_actual_deg={actual_target} "
                            f"measured_actual_deg={actual_deg} error_deg={errors.get(worst_joint)} "
                            f"max_error_deg={max_error:.6f} tolerance_deg={segment.servo_tolerance_deg:.6f} "
                            f"recent_min_deg={contact_window_min:.6f} recent_max_deg={contact_window_max:.6f} "
                            f"recent_slope_deg_s={contact_window_slope:.6f} "
                            f"target_joint_velocity_deg_s={velocities.get(worst_joint)} "
                            f"legacy_missing_endpoint={segment.legacy_missing_endpoint}"
                        )
                        self._finish(
                            adapter,
                            success=False,
                            reason="actuator_limit",
                            current_sim_time_s=sim_time,
                            current_wall_time_s=wall_time,
                        )
                        return
                    self.last_info = f"servo_completion_extension={extension:.4f}s segment={segment.segment_index}"
                    self.progress.command_phase = "servo_completion_extension"
                return

            max_completed_error = max([abs(float(value)) for value in errors.values()] or [0.0])
            if max_completed_error > self.servo_position_tolerance_deg:
                completed_velocities = self._servo_target_velocities_deg_s(adapter, segment.servo_targets)
                warning = {
                    "warning": "contact_residual_accepted",
                    "step_index": int(segment.source_step),
                    "segment_index": int(segment.segment_index),
                    "max_error_deg": max_completed_error,
                    "effective_tolerance_deg": float(segment.servo_tolerance_deg),
                    "recorded_residual_deg": dict(segment.recorded_servo_residual_deg),
                    "measured_errors_deg": dict(errors),
                    "stability_basis": "bounded_recorded_contact_tolerance",
                    "stability_window_s": float(self.contact_residual_stable_s),
                    "stability_window_cap_deg": float(contact_window_cap),
                    "recent_min_deg": float(contact_window_min),
                    "recent_max_deg": float(contact_window_max),
                    "recent_slope_deg_s": float(contact_window_slope),
                    "joint_velocity_deg_s": dict(completed_velocities),
                }
                self.servo_residual_warnings.append(warning)
                self.last_info = (
                    f"contact_residual_accepted step={segment.source_step} segment={segment.segment_index} "
                    f"error={max_completed_error:.3f}deg tolerance={segment.servo_tolerance_deg:.3f}deg"
                )
            self._finish_segment(adapter, segment, elapsed=elapsed, sim_time=sim_time, wall_time=wall_time, servo_errors=errors)
            self.segment_index += 1
            self.segment_active = False
            self.segment_wheel_stopped = False
            if self.segment_index >= len(self.plan.segments):
                final_idle = float(self.plan.timing.get("final_implicit_idle_s", 0.0) or 0.0)
                if final_idle > 0.0:
                    self.next_segment_start_elapsed_s = elapsed + final_idle
                    self.plan.timing["final_implicit_idle_s"] = 0.0
                    return
                self._finish(adapter, success=True, reason="complete", current_sim_time_s=sim_time, current_wall_time_s=wall_time)
                return

            next_segment = self.plan.segments[self.segment_index]
            self.next_segment_start_elapsed_s = elapsed + float(next_segment.implicit_idle_before_s)
            if next_segment.implicit_idle_before_s > 0.0:
                if any(abs(value) > 1.0e-9 for value in segment.wheel_applied_target_rad_s.values()):
                    adapter.stop_wheels()
                    if hasattr(adapter, "apply_commands_to_robot"):
                        adapter.apply_commands_to_robot()
                    self.active_wheel_command = ""
                return
            # No UI/IPC/progress barrier: loop and start the next segment in
            # this same scheduler cycle (zero simulation ticks of fixed pad).

    def _start_segment(
        self,
        adapter: Any,
        segment: PlaybackSegment,
        *,
        elapsed: float,
        sim_time: float,
        current_sim_step: int,
        wall_time: float,
    ) -> None:
        if self.plan is None:
            return
        if segment.servo_targets:
            current_commands = dict(getattr(adapter, "joint_command_deg", {}) or {})
            servo_delta = max(
                [abs(float(target) - float(current_commands.get(name, 0.0))) for name, target in segment.servo_targets.items()]
                or [0.0]
            )
            reference = float(getattr(getattr(adapter, "motion_reference", MOTION_REFERENCE), "servo_reference_velocity_deg_s", SERVO_REFERENCE_VELOCITY_DEG_S))
            velocity_limit = getattr(getattr(adapter, "motion_reference", MOTION_REFERENCE), "servo_velocity_limit_deg_s", None)
            effective = reference if velocity_limit is None else min(reference, float(velocity_limit))
            segment.servo_base_duration_s = servo_delta / max(reference, 1.0e-9)
            segment.servo_duration_s = servo_delta / max(effective, 1.0e-9)
        self.segment_active = True
        self.segment_start_elapsed_s = float(elapsed)
        self.segment_wheel_stopped = False
        self.segment_best_servo_error_deg = float("inf")
        self.segment_last_servo_improvement_elapsed_s = float(elapsed)
        self.segment_last_servo_error_deg = None
        self.segment_last_servo_change_elapsed_s = float(elapsed)
        self.segment_last_servo_worsening_elapsed_s = float(elapsed)
        self.segment_contact_within_tolerance_since_s = None
        self.segment_contact_error_history = []
        self.segment_start_joint_positions = self._joint_positions_by_name(adapter)
        start = int(segment.event_start_index)
        stop = start + int(segment.event_count)
        segment_events = self.plan.events[start:stop]
        dispatched_wheel = False
        atomic_motion_applied = False
        if hasattr(adapter, "apply_motion_batch") and (segment.servo_targets or segment.wheel_base_velocity):
            adapter.apply_motion_batch(
                {
                    "batch_id": f"playback-{self.plan_id}-{segment.segment_index}-{uuid.uuid4().hex[:8]}",
                    "source": "playback",
                    "servo_targets_deg": dict(segment.servo_targets),
                    "wheel_targets_rad_s": dict(segment.wheel_base_velocity),
                    "recording_metadata": {
                        "plan_id": self.plan_id,
                        "source_step": int(segment.source_step),
                        "source_step_id": segment.source_step_id,
                    },
                }
            )
            atomic_motion_applied = True
            dispatched_wheel = any(event.channel == "wheel" and event.dispatch_command for event in segment_events)
        for event in segment_events:
            collector = getattr(adapter, "telemetry_collector", None)
            if collector is not None:
                try:
                    collector.record_replay_event(adapter, event, self.index)
                except Exception as exc:
                    self.last_error = f"telemetry replay event failed: {exc}"
            if event.dispatch_command and not atomic_motion_applied and not _is_timing_only_command(event.command):
                adapter.handle_command(
                    CommandMessage(
                        text=event.command,
                        source="playback",
                        log_history=False,
                        quiet=True,
                        playback_label=self.plan.label,
                        playback_event_index=self.index,
                        playback_event_count=len(self.plan.events),
                        playback_final_time_s=float(self.plan.final_time_s),
                        source_step=event.source_step,
                    )
                )
                dispatched_wheel = dispatched_wheel or event.channel == "wheel"
            if event.channel == "wheel" and event.dispatch_command:
                tokens = command_tokens(event.command)
                self.active_wheel_command = "" if "stop" in tokens[1:] else event.command
            jitter = max(0.0, elapsed - float(event.time_s))
            self.dispatch_jitter_samples.append(float(jitter))
            self.max_dispatch_jitter_s = max(self.max_dispatch_jitter_s, float(jitter))
            self.last_event_command = event.command
            self.last_event_time_s = float(wall_time)
            self.last_event_sim_time_s = float(sim_time)
            self._record_event_progress(event, elapsed=elapsed, sim_time=sim_time, wall_time=wall_time, jitter=jitter)
            self.index += 1
            self.events_sent += 1

        if not self.first_command_applied and any(not _is_timing_only_command(event.command) for event in segment_events):
            self.first_command_applied = True
            self.first_command_applied_sim_time_s = float(sim_time)
            self.first_command_applied_sim_step = int(current_sim_step)

        if segment.servo_targets and hasattr(adapter, "begin_servo_tracking"):
            adapter.begin_servo_tracking(segment.servo_targets)

        if self.wheel_stopped_for_channel_completion and not dispatched_wheel:
            active = {name: value for name, value in segment.wheel_applied_target_rad_s.items() if abs(value) > 1.0e-9}
            if active:
                resume = "; ".join(f"wheel {name} {value:.9g}" for name, value in active.items())
                if hasattr(adapter, "apply_motion_batch"):
                    adapter.apply_motion_batch(
                        {
                            "batch_id": f"playback-resume-{self.plan_id}-{segment.segment_index}",
                            "source": "playback",
                            "servo_targets_deg": {},
                            "wheel_targets_rad_s": dict(segment.wheel_base_velocity),
                        }
                    )
                else:
                    adapter.handle_command(CommandMessage(text=resume, source="playback", log_history=False, quiet=True))
                self.active_wheel_command = resume
                self.wheel_stopped_for_channel_completion = False
        self.last_info = f"worker segment {segment.segment_index} started"

    def _servo_targets_complete(
        self,
        adapter: Any,
        targets: dict[str, float],
        *,
        tolerance_deg: float | None = None,
    ) -> tuple[bool, dict[str, float]]:
        if not targets:
            return True, {}
        try:
            actual_state = adapter.get_actual_joint_state()
            actual_servos = dict(actual_state.get("servos", {}) or {})
        except Exception:
            return False, {"__joint_state_unreadable__": float("nan")}
        errors: dict[str, float] = {}
        any_measured = False
        for name, command_target in targets.items():
            row = actual_servos.get(name, {})
            actual = row.get("deg") if isinstance(row, dict) else None
            if actual is None:
                continue
            any_measured = True
            try:
                expected = (
                    float(adapter.command_to_actual_target_deg(name, command_target))
                    if hasattr(adapter, "command_to_actual_target_deg")
                    else float(command_target)
                )
                errors[name] = float(actual) - expected
            except (TypeError, ValueError):
                continue
        if not any_measured:
            return False, {"__joint_state_unreadable__": float("nan")}
        tolerance = self.servo_position_tolerance_deg if tolerance_deg is None else min(3.0, max(1.0, float(tolerance_deg)))
        return bool(errors) and max(abs(value) for value in errors.values()) <= tolerance, errors

    @staticmethod
    def _servo_target_velocities_deg_s(adapter: Any, targets: dict[str, float]) -> dict[str, float]:
        try:
            actual_servos = dict(adapter.get_actual_joint_state().get("servos", {}) or {})
        except Exception:
            return {}
        velocities: dict[str, float] = {}
        for name in targets:
            row = actual_servos.get(name, {})
            if not isinstance(row, dict):
                return {}
            value = row.get("velocity_deg_s")
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return {}
            if not math.isfinite(parsed):
                return {}
            velocities[name] = parsed
        return velocities

    def _finish_segment(
        self,
        adapter: Any,
        segment: PlaybackSegment,
        *,
        elapsed: float,
        sim_time: float,
        wall_time: float,
        servo_errors: dict[str, float],
    ) -> None:
        if segment.servo_targets and hasattr(adapter, "end_servo_tracking"):
            adapter.end_servo_tracking(segment.servo_targets)
        actual_duration = max(0.0, elapsed - self.segment_start_elapsed_s)
        extension = max(0.0, actual_duration - max(segment.servo_duration_s, segment.wheel_active_duration_s, segment.explicit_hold_s))
        servo_reference_velocity = SERVO_REFERENCE_VELOCITY_DEG_S if segment.servo_targets else 0.0
        servo_motion_deg = float(segment.servo_base_duration_s) * SERVO_REFERENCE_VELOCITY_DEG_S
        servo_measured_average_velocity = (
            servo_motion_deg / actual_duration if segment.servo_targets and actual_duration > 0.0 else servo_reference_velocity
        )
        end_joint_positions = self._joint_positions_by_name(adapter)
        measured_wheel_displacement = {
            name: float(end_joint_positions[name]) - float(self.segment_start_joint_positions[name])
            for name in WHEEL_JOINT_NAMES
            if name in end_joint_positions and name in self.segment_start_joint_positions
        }
        wheel_displacement = measured_wheel_displacement or {
            name: float(value) * float(segment.wheel_active_duration_s)
            for name, value in segment.wheel_applied_target_rad_s.items()
        }
        completion_warning = next(
            (
                dict(row)
                for row in reversed(self.servo_residual_warnings)
                if int(row.get("step_index", -1)) == int(segment.source_step)
                and int(row.get("segment_index", -1)) == int(segment.segment_index)
            ),
            {},
        )
        trace = self.timing_trace.setdefault("segments", [])
        trace.append(
            {
                "segment_index": int(segment.segment_index),
                "step_index": int(segment.source_step),
                "step_id": segment.source_step_id,
                "planned_start_sim_time": float(self.start_sim_time_s + segment.planned_start_s),
                "actual_start_sim_time": float(self.start_sim_time_s + self.segment_start_elapsed_s),
                "planned_end_sim_time": float(self.start_sim_time_s + segment.planned_end_s),
                "actual_end_sim_time": float(sim_time),
                "previous_step_end_time": self.previous_step_end_sim_time_s,
                "inter_step_gap_ms": 0.0,
                "gap_reason": "servo_completion_extension" if extension > 1.0e-9 else segment.gap_reason,
                "servo_target": dict(segment.servo_targets),
                "servo_actual_at_transition": {
                    name: float(target) + float(servo_errors.get(name, 0.0)) for name, target in segment.servo_targets.items()
                },
                "servo_target_error": dict(servo_errors),
                "completion_decision": (
                    "contact_residual_accepted" if completion_warning else "exact_completion"
                ),
                "completion_warning": completion_warning,
                "servo_completion_extension": extension,
                "servo_reference_velocity_deg_s": servo_reference_velocity,
                "servo_measured_average_velocity_deg_s": servo_measured_average_velocity,
                "wheel_requested_velocity_rad_s": dict(segment.wheel_requested_velocity_rad_s),
                "wheel_applied_target_rad_s": dict(segment.wheel_applied_target_rad_s),
                "wheel_active_duration": float(segment.wheel_active_duration_s),
                "wheel_displacement": wheel_displacement,
                "wheel_displacement_source": "articulation_joint_state" if measured_wheel_displacement else "applied_target_x_active_duration",
                "end_wall_time": float(wall_time),
            }
        )
        for command_row in self.timing_trace.get("commands", []):
            if int(command_row.get("segment_index", -1)) == int(segment.segment_index):
                command_row["actual_end_sim_time"] = float(sim_time)
                command_row["end_sim_time_s"] = float(sim_time)
                command_row["end_wall_time"] = float(wall_time)
                command_row["servo_target_error"] = dict(servo_errors)
                command_row["servo_actual_at_transition"] = {
                    name: float(target) + float(servo_errors.get(name, 0.0))
                    for name, target in segment.servo_targets.items()
                }
        for step_row in reversed(self.timing_trace.get("steps", [])):
            if int(step_row.get("source_step_index", -1)) == int(segment.source_step):
                step_row["planned_end_sim_time"] = float(self.start_sim_time_s + segment.planned_end_s)
                step_row["actual_end_sim_time"] = float(sim_time)
                step_row["end_sim_time_s"] = float(sim_time)
                step_row["end_wall_time"] = float(wall_time)
                break
        next_segment = self.plan.segments[self.segment_index + 1] if self.plan and self.segment_index + 1 < len(self.plan.segments) else None
        if next_segment is None or next_segment.source_step != segment.source_step:
            self.previous_step_end_sim_time_s = float(sim_time)

    @staticmethod
    def _joint_positions_by_name(adapter: Any) -> dict[str, float]:
        if not hasattr(adapter, "capture_sim_state"):
            return {}
        try:
            state = dict(adapter.capture_sim_state() or {})
            names = [str(name) for name in list(state.get("joint_names", []) or [])]
            values = state.get("joint_pos")
            while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            if not names or not isinstance(values, list):
                return {}
            return {name: float(values[index]) for index, name in enumerate(names) if index < len(values)}
        except Exception:
            return {}

    def pause(self, *, current_sim_time_s: float, adapter: Any | None = None) -> None:
        if not self.active or self.paused:
            return
        self.paused = True
        self.pause_started_sim_time_s = float(current_sim_time_s)
        self.last_info = "worker playback paused"
        self.progress.playback_state = PlaybackState.PAUSED.value
        self.progress.status_text = "Paused"
        self.progress.command_phase = "paused_at_boundary"
        if adapter is not None and self.active_wheel_command:
            try:
                self.paused_wheel_state = dict(adapter.capture_command_state().get("wheels", {}) or {})
            except Exception:
                self.paused_wheel_state = {}
            adapter.stop_wheels()
            if hasattr(adapter, "apply_commands_to_robot"):
                adapter.apply_commands_to_robot()
            self.wheel_was_stopped_for_pause = True
        if adapter is not None and hasattr(adapter, "servo_motion_enabled"):
            self.resume_servo_motion_pending = True
            adapter.servo_motion_enabled = False

    def resume(self, *, current_sim_time_s: float) -> None:
        if not self.active or not self.paused:
            return
        self.paused_sim_accum_s += max(0.0, float(current_sim_time_s) - self.pause_started_sim_time_s)
        self.pause_started_sim_time_s = 0.0
        self.paused = False
        self.last_info = "worker playback resumed"
        self.resume_wheel_command_pending = bool(self.paused_wheel_state and self.wheel_was_stopped_for_pause)
        self.wheel_was_stopped_for_pause = False
        self.progress.playback_state = PlaybackState.PLAYING.value
        self.progress.status_text = (
            f"Restore verified. Playing Selected Step {int(self.progress.current_step_index or 0)} / {self.progress.total_steps}"
            if self.plan and self.plan.selected_playback
            else "Playing..."
        )
        self.progress.command_phase = "resuming"

    def stop(
        self,
        adapter: Any,
        *,
        current_sim_time_s: float,
        current_wall_time_s: float,
        reason: str = "stopped",
        stop_wheels: bool = True,
    ) -> None:
        if stop_wheels and adapter is not None:
            adapter.stop_wheels()
            if hasattr(adapter, "apply_commands_to_robot"):
                adapter.apply_commands_to_robot()
        if self.active:
            self._finish(
                adapter,
                success=False,
                reason=str(reason or "stopped"),
                current_sim_time_s=float(current_sim_time_s),
                current_wall_time_s=float(current_wall_time_s),
            )
        else:
            self.stop_reason = str(reason or "stopped")
            self.last_info = "worker playback stopped"
            self.progress.playback_state = PlaybackState.IDLE.value
            self.progress.status_text = "Stopped"
            self.progress.command_phase = "stopped"

    def status_dict(
        self,
        *,
        current_sim_time_s: float | None = None,
        current_wall_time_s: float | None = None,
        compact: bool = False,
    ) -> dict[str, Any]:
        plan = self.plan
        sim_time = float(current_sim_time_s if current_sim_time_s is not None else self.completed_at_sim_s)
        elapsed = 0.0
        if self.active and plan is not None:
            elapsed = max(0.0, sim_time - self.start_sim_time_s - self.paused_sim_accum_s)
        elif self.completed_at_sim_s > 0.0:
            elapsed = float(plan.final_time_s if plan is not None else 0.0)
        count = len(plan.events) if plan is not None else 0
        progress = (elapsed / float(plan.final_time_s)) if plan is not None and float(plan.final_time_s) > 0.0 else (1.0 if count and not self.active and self.stop_reason == "complete" else 0.0)
        self.progress.elapsed_time = float(elapsed)
        self.progress.estimated_remaining_time = max(0.0, float(plan.final_time_s if plan is not None else 0.0) - float(elapsed))
        self.progress.scheduler_lateness_s = float(self.max_dispatch_jitter_s)
        self.progress.last_error = self.last_error
        status = {
            "active": bool(self.active),
            "paused": bool(self.paused),
            "scheduled": bool(self.active and not self.started),
            "started": bool(self.started),
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "worker_session_id": self.worker_session_id,
            "plan_sha256": str(plan.plan_sha256 if plan is not None else ""),
            "source_sha256": str(plan.source_sha256 if plan is not None else ""),
            "label": str(plan.label if plan is not None else ""),
            "profile": str(plan.profile if plan is not None else "raw"),
            "path": str(plan.path) if plan is not None and plan.path is not None else "",
            "index": int(self.index),
            "count": int(count),
            "event_count": int(count),
            "segment_count": len(plan.segments) if plan is not None else 0,
            "segment_index": int(self.segment_index),
            "events_sent": int(self.events_sent),
            "final_time_s": float(plan.final_time_s if plan is not None else 0.0),
            "sim_elapsed_s": float(elapsed),
            "wall_elapsed_s": max(0.0, float(current_wall_time_s if current_wall_time_s is not None else time.time()) - self.start_wall_time_s) if self.start_wall_time_s else 0.0,
            "progress": max(0.0, min(1.0, float(progress))),
            "last_event_command": self.last_event_command,
            "last_event_time": self.last_event_time_s,
            "last_event_sim_time_s": self.last_event_sim_time_s,
            "max_dispatch_jitter_s": float(self.max_dispatch_jitter_s),
            "dispatch_clock": "simulation_time",
            "completed_at": self.completed_at_wall_s,
            "completed_at_sim_time_s": self.completed_at_sim_s,
            "stop_reason": self.stop_reason,
            "last_error": self.last_error,
            "last_info": self.last_info,
            "first_motion_planned_s": float(self.first_motion_planned_s),
            "first_command_applied": bool(self.first_command_applied),
            "first_command_applied_sim_time_s": float(self.first_command_applied_sim_time_s),
            "first_command_applied_sim_step": int(self.first_command_applied_sim_step),
            "current_servo_errors": dict(self.current_servo_errors),
            "servo_residual_warning_count": len(self.servo_residual_warnings),
            "last_servo_residual_warning": dict(self.servo_residual_warnings[-1]) if self.servo_residual_warnings else {},
            "progress_detail": self.progress.to_dict(),
            "timing": {} if compact else copy.deepcopy(self.timing_trace),
        }
        if compact:
            status["progress_detail"] = {
                "playback_state": self.progress.playback_state,
                "status_text": self.progress.status_text,
                "current_step_index": self.progress.current_step_index,
                "total_steps": self.progress.total_steps,
                "global_command_index": self.progress.global_command_index,
                "total_commands": self.progress.total_commands,
                "playback_profile": self.progress.playback_profile,
                "selected_playback": self.progress.selected_playback,
                "command_phase": self.progress.command_phase,
                "last_error": self.progress.last_error,
            }
        return status

    def _finish(self, adapter: Any, *, success: bool, reason: str, current_sim_time_s: float, current_wall_time_s: float) -> None:
        if adapter is not None:
            adapter.stop_wheels()
            if hasattr(adapter, "apply_commands_to_robot"):
                adapter.apply_commands_to_robot()
        collector = getattr(adapter, "telemetry_collector", None) if adapter is not None else None
        if collector is not None:
            try:
                collector.finish_replay(success=bool(success), reason="" if success else str(reason), sim_time_s=float(current_sim_time_s))
            except Exception as exc:
                self.last_error = f"telemetry replay finish failed: {exc}"
        self.active = False
        self.paused = False
        self.completed_at_wall_s = float(current_wall_time_s)
        self.completed_at_sim_s = float(current_sim_time_s)
        self.stop_reason = "complete" if success else str(reason or "stopped")
        self.last_info = "worker playback complete" if success else f"worker playback stopped: {self.stop_reason}"
        self.timing_trace["final_completion_time"] = float(current_wall_time_s)
        if self.timing_trace.get("commands"):
            self.timing_trace["commands"][-1]["end_sim_time_s"] = float(current_sim_time_s)
            self.timing_trace["commands"][-1]["end_wall_time"] = float(current_wall_time_s)
        if self.timing_trace.get("steps"):
            self.timing_trace["steps"][-1]["end_sim_time_s"] = float(current_sim_time_s)
            self.timing_trace["steps"][-1]["end_wall_time"] = float(current_wall_time_s)
        self.progress.playback_state = PlaybackState.COMPLETED.value if success else PlaybackState.IDLE.value
        self.progress.status_text = "Completed" if success else "Stopped"
        self.progress.command_phase = "completed" if success else "stopped"
        self.progress.elapsed_time = float(self.plan.final_time_s if success and self.plan is not None else max(0.0, current_sim_time_s - self.start_sim_time_s - self.paused_sim_accum_s))
        self.progress.estimated_remaining_time = 0.0 if success else max(0.0, float(self.plan.final_time_s if self.plan else 0.0) - self.progress.elapsed_time)

    def _record_event_progress(
        self,
        event: PlaybackEvent,
        *,
        elapsed: float,
        sim_time: float,
        wall_time: float,
        jitter: float,
        pre_dispatch_joint_pos: Any = None,
    ) -> None:
        commands = self.timing_trace.setdefault("commands", [])
        if commands:
            commands[-1]["end_sim_time_s"] = float(sim_time)
            commands[-1]["end_wall_time"] = float(wall_time)
        commands.append(
            {
                "segment_index": int(event.segment_index),
                "global_command_index": int(event.global_command_index or self.index + 1),
                "step_index": int(event.source_step or 0),
                "step_id": event.source_step_id,
                "command_index": int(event.command_index_in_step or 0),
                "source_step_index": int(event.source_step or 0),
                "command_index_in_step": int(event.command_index_in_step or 0),
                "command": event.command,
                "scheduled_time_s": float(event.time_s),
                "planned_start_sim_time": float(self.start_sim_time_s + event.time_s),
                "actual_start_sim_time": float(sim_time),
                "planned_end_sim_time": float(self.start_sim_time_s + event.planned_end_s),
                "actual_end_sim_time": None,
                "previous_step_end_time": self.previous_step_end_sim_time_s,
                "inter_step_gap_ms": (
                    max(0.0, float(sim_time) - float(self.previous_step_end_sim_time_s)) * 1000.0
                    if self.previous_step_end_sim_time_s is not None
                    else 0.0
                ),
                "gap_reason": event.gap_reason,
                "start_sim_time_s": float(sim_time),
                "start_wall_time": float(wall_time),
                "intentional_motion_duration": float(event.planned_duration_s),
                "implicit_idle_gap": 0.0 if self.plan and self.plan.profile == "motion_only" else float(event.planned_duration_s),
                "scheduler_lateness": float(jitter),
                "pre_dispatch_joint_pos": copy.deepcopy(pre_dispatch_joint_pos),
                "servo_target": dict(event.servo_targets),
                "servo_actual_at_transition": None,
                "servo_target_error": None,
                "wheel_requested_velocity_rad_s": list(event.wheel_requested_velocity_rad_s),
                "wheel_applied_target_rad_s": list(event.wheel_applied_target_rad_s),
                "wheel_active_duration": float(event.wheel_active_duration_s),
                "wheel_displacement": list(event.wheel_displacement),
                "dispatch_command": bool(event.dispatch_command),
            }
        )
        if "first_command_start" not in self.timing_trace:
            self.timing_trace["first_command_start"] = float(wall_time)
        steps = self.timing_trace.setdefault("steps", [])
        if not steps or int(steps[-1].get("source_step_index", -1)) != int(event.source_step or 0):
            if steps:
                steps[-1]["end_sim_time_s"] = float(sim_time)
                steps[-1]["end_wall_time"] = float(wall_time)
            steps.append(
                {
                    "step_index": int(event.source_step or 0),
                    "step_id": event.source_step_id,
                    "planned_start_sim_time": float(self.start_sim_time_s + event.time_s),
                    "actual_start_sim_time": float(sim_time),
                    "planned_end_sim_time": float(self.start_sim_time_s + event.planned_end_s),
                    "actual_end_sim_time": None,
                    "previous_step_end_time": self.previous_step_end_sim_time_s,
                    "inter_step_gap_ms": (
                        max(0.0, float(sim_time) - float(self.previous_step_end_sim_time_s)) * 1000.0
                        if self.previous_step_end_sim_time_s is not None
                        else 0.0
                    ),
                    "gap_reason": event.gap_reason,
                    "source_step_index": int(event.source_step or 0),
                    "source_step_id": event.source_step_id,
                    "start_sim_time_s": float(sim_time),
                    "start_wall_time": float(wall_time),
                }
            )
        self.progress.playback_state = PlaybackState.PLAYING.value
        self.progress.status_text = "Playing..."
        self.progress.current_step_index = int(event.source_step or 0)
        self.progress.current_step_id = str(event.source_step_id or "")
        self.progress.current_command_index_in_step = int(event.command_index_in_step or 0)
        self.progress.commands_in_current_step = int(event.commands_in_step or 0)
        self.progress.global_command_index = int(event.global_command_index or self.index + 1)
        self.progress.total_commands = len(self.plan.events) if self.plan else 0
        self.progress.elapsed_time = float(elapsed)
        self.progress.estimated_remaining_time = max(0.0, float(self.plan.final_time_s if self.plan else 0.0) - float(elapsed))
        self.progress.command_phase = "started"
        self.progress.scheduler_lateness_s = float(jitter)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


class PlaybackManager:
    def __init__(self, controller: Any):
        self.controller = controller
        self.active = False
        self.paused = False
        self.plan: PlaybackPlan | None = None
        self.index = 0
        self.start_time = 0.0
        self.scheduled_start_at = 0.0
        self.pause_started = 0.0
        self.started_at = 0.0
        self.events_sent = 0
        self.last_event_command = ""
        self.last_event_time = 0.0
        self.completed_at = 0.0
        self.stop_reason = ""
        self.last_error = ""
        self.last_info = ""
        self.profile = "raw"
        self.max_events_per_update = 50
        self.worker_managed = False
        self.worker_plan_id = ""
        self.worker_request_id = ""
        self.worker_session_id = ""
        self.worker_acknowledged = False
        self.operation_owner_id = ""
        self.start_requested = False
        self.worker_requested_at = 0.0
        self.worker_ack_timeout_s = 10.0
        self.first_command_watchdog_s = 2.0
        self.dispatch_clock = "wall_clock"
        self.max_dispatch_jitter_s = 0.0
        self.progress = PlaybackProgress()
        self.timing_trace: dict[str, Any] = {}

    def _enter_operation(self, label: str) -> bool:
        coordinator = getattr(self.controller, "operation", None)
        if coordinator is None:
            return True
        if coordinator.enter_playback(detail=f"Playback is active: {label}"):
            return True
        self.last_error = coordinator.reason or "Another operation is active."
        self.last_info = self.last_error
        return False

    def _finish_operation(self) -> None:
        coordinator = getattr(self.controller, "operation", None)
        if coordinator is not None:
            try:
                from operation_coordinator import OperationState

                if not coordinator.finish(OperationState.PLAYBACK):
                    coordinator.finish(OperationState.RESPAWNING)
            except Exception:
                coordinator.finish()

    def start_steps(self, steps: list[dict[str, Any]], *, label: str = "accepted steps", start_delay_s: float = 0.0) -> bool:
        try:
            plan = plan_from_steps(
                steps,
                profile=self.profile,
                max_wheel_speed=getattr(self.controller, "max_wheel_speed", None),
                label=label,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return False
        return self.start_plan(plan, start_delay_s=start_delay_s)

    def start_worker_plan(
        self,
        plan: PlaybackPlan,
        *,
        start_delay_s: float = 0.0,
        operation_already_owned: bool = False,
        operation_owner_id: str = "",
    ) -> bool:
        label = plan.label or "plan"
        if not plan.events:
            return self.start_plan(plan, start_delay_s=start_delay_s)
        plan = copy.deepcopy(plan)
        plan.plan_sha256 = plan.plan_sha256 or plan_fingerprint(plan)
        integrity = validate_plan_integrity(
            plan,
            expected_plan_sha256=plan.plan_sha256,
            expected_event_count=len(plan.events),
            expected_segment_count=len(plan.segments),
        )
        if not integrity["ok"]:
            self.last_error = "Playback plan integrity failed: " + "; ".join(integrity["errors"])
            self.last_info = self.last_error
            return False
        playback_request_id = uuid.uuid4().hex
        plan_id = f"{plan.plan_sha256[:12]}-{uuid.uuid4().hex[:12]}"
        if operation_already_owned:
            coordinator = getattr(self.controller, "operation", None)
            expected_owner = str(plan.timing.get("selected_restore_request_id", "") or "")
            if (
                coordinator is None
                or str(getattr(coordinator.state, "value", coordinator.state)) != "PLAYBACK"
                or not operation_owner_id
                or str(operation_owner_id) != expected_owner
            ):
                self.last_error = "Selected playback does not own the active PLAYBACK operation."
                self.last_info = self.last_error
                return False
        elif not self._enter_operation(label):
            return False
        try:
            self.controller.transport.start_playback_plan(
                plan,
                start_delay_sim_s=max(0.0, float(start_delay_s)),
                plan_id=plan_id,
                request_id=playback_request_id,
                plan_sha256=plan.plan_sha256,
            )
        except Exception as exc:
            self.last_error = str(exc)
            self.last_info = self.last_error
            self._finish_operation()
            return False
        self.plan = plan
        self.worker_managed = True
        self.worker_plan_id = plan_id
        self.worker_request_id = playback_request_id
        self.worker_session_id = ""
        self.worker_acknowledged = False
        self.operation_owner_id = str(operation_owner_id or playback_request_id)
        self.start_requested = True
        self.worker_requested_at = time.monotonic()
        self.dispatch_clock = "simulation_time"
        self.active = False
        self.paused = False
        self.index = 0
        self.events_sent = 0
        self.scheduled_start_at = 0.0
        self.start_time = 0.0
        self.started_at = 0.0
        self.pause_started = 0.0
        self.completed_at = 0.0
        self.stop_reason = ""
        self.last_error = ""
        self.last_info = f"worker playback start requested: {label} ({len(plan.events)} events)"
        self.last_event_command = ""
        self.last_event_time = 0.0
        self.max_dispatch_jitter_s = 0.0
        first = plan.events[0]
        self.progress = PlaybackProgress(
            playback_state=PlaybackState.START_REQUESTED.value,
            status_text="Start requested; waiting for worker acceptance...",
            current_step_index=int(first.source_step or 0),
            total_steps=int(plan.total_steps),
            current_step_id=first.source_step_id,
            commands_in_current_step=int(first.commands_in_step),
            total_commands=len(plan.events),
            playback_profile=plan.profile,
            selected_playback=plan.selected_playback,
            command_phase="start_handshake",
        )
        self.timing_trace = copy.deepcopy(plan.timing)
        return True

    def start_plan(self, plan: PlaybackPlan, *, start_delay_s: float = 0.0) -> bool:
        label = plan.label or "plan"
        if not plan.events:
            self.plan = copy.deepcopy(plan)
            self.index = 0
            self.active = False
            self.paused = False
            self.scheduled_start_at = 0.0
            self.start_time = 0.0
            self.started_at = 0.0
            self.events_sent = 0
            self.last_event_command = ""
            self.last_event_time = 0.0
            self.completed_at = 0.0
            self.stop_reason = "empty_plan"
            self.last_error = f"Selected step/plan {label} has no motion events to play."
            self.last_info = self.last_error
            self.worker_managed = False
            self.worker_plan_id = ""
            self.worker_request_id = ""
            self.worker_session_id = ""
            self.worker_acknowledged = False
            self.start_requested = False
            self.worker_requested_at = 0.0
            self.dispatch_clock = "wall_clock"
            self.progress = PlaybackProgress(
                playback_state=PlaybackState.ERROR.value,
                status_text="Error",
                last_error=self.last_error,
            )
            self._finish_operation()
            return False
        if not self._enter_operation(label):
            return False
        self.plan = copy.deepcopy(plan)
        self.plan.plan_sha256 = self.plan.plan_sha256 or plan_fingerprint(self.plan)
        self.index = 0
        self.active = True
        self.paused = False
        now = time.monotonic()
        delay_s = max(0.0, float(start_delay_s))
        self.scheduled_start_at = now + delay_s if delay_s > 0.0 else 0.0
        self.start_time = self.scheduled_start_at or now
        self.started_at = 0.0 if self.scheduled_start_at else time.time()
        self.pause_started = 0.0
        self.events_sent = 0
        self.last_event_command = ""
        self.last_event_time = 0.0
        self.completed_at = 0.0
        self.stop_reason = ""
        self.last_error = ""
        self.worker_managed = False
        self.worker_plan_id = ""
        self.worker_request_id = ""
        self.worker_session_id = ""
        self.worker_acknowledged = False
        self.start_requested = False
        self.worker_requested_at = 0.0
        self.dispatch_clock = "wall_clock"
        self.max_dispatch_jitter_s = 0.0
        first = self.plan.events[0]
        self.progress = PlaybackProgress(
            playback_state=PlaybackState.PREPARING.value if self.scheduled_start_at else PlaybackState.PLAYING.value,
            status_text="Waiting for simulation ready..." if self.scheduled_start_at else "Playing...",
            current_step_index=int(first.source_step or 0),
            total_steps=int(self.plan.total_steps),
            current_step_id=first.source_step_id,
            commands_in_current_step=int(first.commands_in_step),
            total_commands=len(self.plan.events),
            playback_profile=self.plan.profile,
            selected_playback=self.plan.selected_playback,
        )
        self.timing_trace = copy.deepcopy(self.plan.timing)
        if self.scheduled_start_at:
            self.last_info = f"playback scheduled; starts in {delay_s:.2f}s: {label} ({len(plan.events)} events)"
        else:
            self.last_info = f"playing {label} ({len(plan.events)} events)"
        return True

    def update(self) -> None:
        if self.worker_managed:
            self.sync_worker_status()
            return
        if not self.active or self.paused or self.plan is None:
            return
        now = time.monotonic()
        if self.scheduled_start_at > 0.0:
            remaining = self.scheduled_start_at - now
            if remaining > 0.0:
                self.last_info = f"playback scheduled; starts in {remaining:.2f}s"
                return
            self.started_at = time.time()
            self.scheduled_start_at = 0.0
            self.last_info = f"playing {self.plan.label or 'plan'} ({len(self.plan.events)} events)"
        elapsed = now - self.start_time
        sent_this_update = 0
        while self.index < len(self.plan.events):
            if sent_this_update >= int(self.max_events_per_update):
                self.last_info = f"playback batching at event {self.index}/{len(self.plan.events)}"
                break
            event = self.plan.events[self.index]
            if event.time_s > elapsed + 1.0e-4:
                break
            self.controller.handle_command(
                CommandMessage(
                    text=event.command,
                    source="playback",
                    log_history=False,
                    quiet=True,
                    playback_label=self.plan.label,
                    playback_event_index=self.index,
                    playback_event_count=len(self.plan.events),
                    playback_final_time_s=float(self.plan.final_time_s),
                    source_step=event.source_step,
                )
            )
            self.index += 1
            sent_this_update += 1
            self.events_sent += 1
            self.last_event_command = event.command
            self.last_event_time = time.time()
            self.progress.playback_state = PlaybackState.PLAYING.value
            self.progress.status_text = "Playing..."
            self.progress.current_step_index = int(event.source_step or 0)
            self.progress.current_step_id = event.source_step_id
            self.progress.current_command_index_in_step = int(event.command_index_in_step or 0)
            self.progress.commands_in_current_step = int(event.commands_in_step or 0)
            self.progress.global_command_index = int(event.global_command_index or self.index)
            self.progress.elapsed_time = float(elapsed)
            self.progress.estimated_remaining_time = max(0.0, float(self.plan.final_time_s) - float(elapsed))
            self.progress.command_phase = "started"
        if self.index >= len(self.plan.events) and elapsed >= self.plan.final_time_s:
            self.completed_at = time.time()
            self.stop(silent=True, reason="complete")
            self.last_info = "playback complete"

    def sync_worker_status(
        self,
        status: dict[str, Any] | None = None,
        *,
        operation_ack: dict[str, Any] | None = None,
        worker_status_age_s: float = 0.0,
    ) -> None:
        if not self.worker_managed:
            return
        if status is None:
            latest = getattr(self.controller, "latest_sim_status", {}) or {}
            status = latest.get("worker_playback", latest.get("playback_service", {}))
            operation_ack = dict(latest.get("last_operation_ack", {}) or {})
        status = dict(status or {})
        ack = dict(operation_ack or {})
        now = time.monotonic()

        if self.start_requested:
            ack_matches = (
                str(ack.get("operation", "") or "") == "start_playback_plan"
                and str(ack.get("request_id", "") or "") == self.worker_request_id
            )
            if ack_matches:
                accepted = bool(ack.get("accepted", False))
                identity_ok = (
                    str(ack.get("plan_id", "") or "") == self.worker_plan_id
                    and str(ack.get("plan_sha256", "") or "") == str(self.plan.plan_sha256 if self.plan else "")
                    and int(ack.get("event_count", -1) or -1) == len(self.plan.events if self.plan else [])
                    and int(ack.get("segment_count", -1) or -1) == len(self.plan.segments if self.plan else [])
                    and bool(str(ack.get("worker_session_id", "") or ""))
                )
                if not accepted or not identity_ok:
                    rejection = str(ack.get("rejection_reason", ack.get("error", "")) or "worker rejected playback")
                    if accepted and not identity_ok:
                        rejection = "worker acceptance identity/count mismatch"
                    self.active = False
                    self.paused = False
                    self.start_requested = False
                    self.worker_managed = False
                    self.stop_reason = "start_rejected"
                    self.last_error = rejection
                    self.last_info = f"Playback start failed: {rejection}"
                    self.progress.playback_state = PlaybackState.ERROR.value
                    self.progress.status_text = "Error"
                    self.progress.last_error = rejection
                    self._finish_operation()
                    return
                self.worker_acknowledged = True
                self.start_requested = False
                self.worker_session_id = str(ack.get("worker_session_id", "") or "")
                self.progress.playback_state = PlaybackState.PREPARING.value
                self.progress.status_text = "Worker accepted; preparing..."
                self.progress.command_phase = "accepted_waiting_for_start"
                self.last_info = f"worker accepted playback plan={self.worker_plan_id}"
            elif now - float(self.worker_requested_at or now) >= float(self.worker_ack_timeout_s):
                self.active = False
                self.paused = False
                self.start_requested = False
                self.worker_managed = False
                self.stop_reason = "worker_ack_timeout"
                self.last_error = (
                    f"Worker did not explicitly accept request {self.worker_request_id} "
                    f"within {self.worker_ack_timeout_s:.1f}s."
                )
                self.last_info = self.last_error
                self.progress.playback_state = PlaybackState.ERROR.value
                self.progress.status_text = "Error"
                self.progress.last_error = self.last_error
                self._finish_operation()
                return
            else:
                self.active = False
                self.paused = False
                return

        if not self.worker_acknowledged or not status:
            return
        if worker_status_age_s > 3.0:
            self.last_info = f"waiting for fresh worker playback status ({worker_status_age_s:.2f}s old)"
            return
        if (
            str(status.get("plan_id", "") or "") != self.worker_plan_id
            or str(status.get("request_id", "") or "") != self.worker_request_id
            or str(status.get("worker_session_id", "") or "") != self.worker_session_id
        ):
            self.last_info = "ignoring stale playback status from another request/session"
            return

        self.active = bool(status.get("active", False))
        self.paused = bool(status.get("paused", False))
        self.index = int(status.get("index", self.index) or 0)
        self.events_sent = int(status.get("events_sent", self.events_sent) or 0)
        self.last_event_command = str(status.get("last_event_command", self.last_event_command) or "")
        self.last_event_time = float(status.get("last_event_time", self.last_event_time) or 0.0)
        self.completed_at = float(status.get("completed_at", self.completed_at) or 0.0)
        self.stop_reason = str(status.get("stop_reason", self.stop_reason) or "")
        self.last_error = str(status.get("last_error", self.last_error) or "")
        self.last_info = str(status.get("last_info", self.last_info) or "")
        self.dispatch_clock = str(status.get("dispatch_clock", self.dispatch_clock) or self.dispatch_clock)
        self.max_dispatch_jitter_s = float(status.get("max_dispatch_jitter_s", self.max_dispatch_jitter_s) or 0.0)
        if bool(status.get("started", False)) and not self.started_at:
            self.started_at = time.time()
        if isinstance(status.get("progress_detail"), dict):
            self.progress = PlaybackProgress.from_dict(status.get("progress_detail"))
        if isinstance(status.get("timing"), dict):
            self.timing_trace = copy.deepcopy(status.get("timing"))

        first_deadline = float(status.get("first_motion_planned_s", 0.0) or 0.0) + float(self.first_command_watchdog_s)
        if (
            bool(status.get("started", False))
            and self.active
            and not bool(status.get("first_command_applied", False))
            and float(status.get("sim_elapsed_s", 0.0) or 0.0) > first_deadline
        ):
            self.last_error = (
                f"First motion command was not applied within {self.first_command_watchdog_s:.1f}s "
                f"of its planned simulation time."
            )
            self.last_info = self.last_error
            self.progress.playback_state = PlaybackState.ERROR.value
            self.progress.status_text = "Error"
            self.progress.last_error = self.last_error
            try:
                self.controller.transport.stop_playback(reason="first_command_watchdog", stop_wheels=True)
            except Exception:
                pass
            self.active = False
            self.worker_managed = False
            self.stop_reason = "first_command_watchdog"
            self._finish_operation()
            return

        if not self.active and self.stop_reason:
            if self.stop_reason != "complete" and self.last_error:
                stopped_step = int(self.progress.current_step_index or status.get("current_step", 0) or 0)
                stopped_segment = int(status.get("segment_index", 0) or 0)
                failure_fields = {}
                for label, key in (
                    ("Joint", "joint"),
                    ("Requested", "requested_command_deg"),
                    ("Actual", "measured_actual_deg"),
                    ("Error", "error_deg"),
                    ("Tolerance", "tolerance_deg"),
                    ("Recent Min", "recent_min_deg"),
                    ("Recent Max", "recent_max_deg"),
                    ("Recent Slope", "recent_slope_deg_s"),
                    ("Joint Velocity", "target_joint_velocity_deg_s"),
                ):
                    match = re.search(rf"\b{key}=([^\s]+)", self.last_error)
                    if match:
                        failure_fields[label] = match.group(1)
                failure_lines = [
                    f"Playback stopped at Step {stopped_step} / Segment {stopped_segment}",
                    f"Reason: {self.stop_reason}",
                ]
                failure_lines.extend(f"{label}: {value}" for label, value in failure_fields.items())
                failure_lines.append(self.last_error)
                self.progress.playback_state = PlaybackState.ERROR.value
                self.progress.status_text = "\n".join(failure_lines)
                self.progress.last_error = self.last_error
                self.last_info = self.progress.status_text
            self.worker_managed = False
            self.worker_acknowledged = False
            self.start_requested = False
            self.scheduled_start_at = 0.0
            self._finish_operation()

    def stop(self, *, silent: bool = False, stop_wheels: bool = True, reason: str = "stopped") -> None:
        if self.worker_managed:
            self.progress.playback_state = PlaybackState.STOPPING.value
            self.progress.status_text = "Stopping..."
            try:
                self.controller.transport.stop_playback(reason=reason, stop_wheels=stop_wheels)
            except Exception as exc:
                self.last_error = str(exc)
            was_active = self.active
            self.active = False
            self.paused = False
            self.worker_managed = False
            self.worker_acknowledged = False
            self.start_requested = False
            self.worker_requested_at = 0.0
            self.index = 0
            self.scheduled_start_at = 0.0
            self.plan = None
            self._finish_operation()
            if was_active or not silent:
                self.stop_reason = reason
                self.last_info = "worker playback stop requested"
            self.progress.playback_state = PlaybackState.IDLE.value
            self.progress.status_text = "Stopped"
            self.progress.command_phase = "stopped"
            return
        was_active = self.active
        was_scheduled = self.scheduled_start_at > 0.0
        self.active = False
        self.paused = False
        self.index = 0
        self.scheduled_start_at = 0.0
        self.plan = None
        self._finish_operation()
        if stop_wheels and hasattr(self.controller, "stop_wheels"):
            self.controller.stop_wheels()
        if was_active or was_scheduled or not silent:
            self.stop_reason = reason
            self.last_info = "playback stopped"
        self.progress.playback_state = PlaybackState.COMPLETED.value if reason == "complete" else PlaybackState.IDLE.value
        self.progress.status_text = "Completed" if reason == "complete" else "Stopped"
        self.progress.command_phase = "completed" if reason == "complete" else "stopped"

    def pause(self) -> None:
        if not self.active or self.paused:
            return
        if self.worker_managed:
            try:
                self.controller.transport.pause_playback()
            except Exception as exc:
                self.last_error = str(exc)
            self.paused = True
            self.last_info = "worker playback pause requested"
            self.progress.playback_state = PlaybackState.PAUSED.value
            self.progress.status_text = "Paused"
            return
        self.paused = True
        self.pause_started = time.monotonic()
        self.last_info = "playback paused"
        self.progress.playback_state = PlaybackState.PAUSED.value
        self.progress.status_text = "Paused"

    def resume(self) -> None:
        latest = dict(getattr(self.controller, "latest_sim_status", {}) or {})
        worker = dict(latest.get("worker_playback", latest.get("playback_service", {})) or {})
        matching_worker_pause = bool(
            self.worker_managed
            and worker.get("active", False)
            and worker.get("paused", False)
            and str(worker.get("plan_id", "") or "") == str(self.worker_plan_id or "")
            and str(worker.get("request_id", "") or "") == str(self.worker_request_id or "")
        )
        if not self.active or (not self.paused and not matching_worker_pause):
            return
        if self.worker_managed:
            try:
                self.controller.transport.resume_playback()
            except Exception as exc:
                self.last_error = str(exc)
            self.paused = False
            self.last_info = "worker playback resume requested"
            self.progress.playback_state = PlaybackState.PLAYING.value
            self.progress.status_text = "Playing..."
            return
        paused_for = time.monotonic() - self.pause_started
        self.start_time += paused_for
        if self.scheduled_start_at:
            self.scheduled_start_at += paused_for
        self.pause_started = 0.0
        self.paused = False
        self.last_info = "playback resumed"
        self.progress.playback_state = PlaybackState.PLAYING.value
        self.progress.status_text = "Playing..."

    def set_profile(self, profile: str) -> None:
        normalized = profile.lower()
        if normalized in {"motion-only", "motion_only", "fast"}:
            self.profile = "fast"
        elif normalized == "raw":
            self.profile = "raw"
        else:
            raise ValueError("Playback profile must be fast or raw.")

    def analyze_steps(self, steps: list[dict[str, Any]]) -> str:
        plan = plan_from_steps(
            steps,
            profile=self.profile,
            max_wheel_speed=getattr(self.controller, "max_wheel_speed", None),
        )
        return f"Playback timing: profile={self.profile} events={len(plan.events)} duration={plan.final_time_s:.3f}s"

    def status_dict(self) -> dict[str, Any]:
        now = time.monotonic()
        starts_in_s = max(0.0, self.scheduled_start_at - now) if self.scheduled_start_at else 0.0
        return {
            "active": self.active,
            "start_requested": bool(self.start_requested),
            "paused": self.paused,
            "scheduled": bool(self.active and self.scheduled_start_at > 0.0),
            "scheduled_start_at": self.scheduled_start_at,
            "starts_in_s": starts_in_s,
            "index": self.index,
            "count": len(self.plan.events) if self.plan else 0,
            "label": self.plan.label if self.plan else "",
            "profile": self.profile,
            "started_at": self.started_at,
            "events_sent": self.events_sent,
            "last_event_command": self.last_event_command,
            "last_event_time": self.last_event_time,
            "completed_at": self.completed_at,
            "stop_reason": self.stop_reason,
            "last_error": self.last_error,
            "last_info": self.last_info,
            "worker_managed": bool(self.worker_managed),
            "worker_plan_id": self.worker_plan_id,
            "worker_request_id": self.worker_request_id,
            "worker_session_id": self.worker_session_id,
            "worker_acknowledged": bool(self.worker_acknowledged),
            "operation_owner_id": self.operation_owner_id,
            "dispatch_clock": self.dispatch_clock,
            "plan_sha256": self.plan.plan_sha256 if self.plan else "",
            "source_sha256": self.plan.source_sha256 if self.plan else "",
            "max_dispatch_jitter_s": float(self.max_dispatch_jitter_s),
            "progress_detail": self.progress.to_dict(),
            "timing": copy.deepcopy(self.timing_trace),
        }


def command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return str(command).split()
