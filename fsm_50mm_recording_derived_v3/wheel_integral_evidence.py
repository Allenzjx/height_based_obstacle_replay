"""Pure-Python wheel target-integral and measured-rotation evidence.

The evaluator deliberately derives target epochs from independently ACKed
``timing_trace.motion_batches``.  A batch's ``first_physics_step`` is inclusive
and the next batch's first step is exclusive.  Runtime cursor fields are not
used as timing authority.

Measured wheel positions are phase-unwrapped with
``atan2(sin(delta), cos(delta))`` only when contiguous 120 Hz samples and the
available velocity bound make the principal delta unique.  No measured
tracking tolerance is defined here; even complete measured metrics therefore
remain ``NOT_EVALUABLE`` as a tracking verdict until a registered threshold is
supplied elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

from command_model import (
    DEFAULT_MAX_WHEEL_SPEED_RAD_S,
    WHEEL_FORWARD_SIGN,
    WHEEL_JOINT_NAMES,
)


PHYSICS_DT = Fraction(1, 120)
PHYSICS_DT_S = float(PHYSICS_DT)

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"
AVAILABLE = "AVAILABLE"


@dataclass(frozen=True)
class _PlanSegment:
    index: int
    duration: Fraction
    targets: Mapping[str, Fraction]


@dataclass(frozen=True)
class _Batch:
    index: int
    batch_id: str
    dispatch_kind: str
    segment_index: int | None
    applied_step: int
    first_step: int
    targets: Mapping[str, Fraction]


@dataclass(frozen=True)
class _Tick:
    step: int
    target: Mapping[str, Fraction] | None
    position: Mapping[str, float] | None
    velocity: Mapping[str, float] | None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fraction(value: Any) -> Fraction | None:
    parsed = _finite_float(value)
    if parsed is None:
        return None
    return Fraction(str(parsed))


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _sign(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or parsed == 0.0:
        return None
    return -1 if parsed < 0.0 else 1


def _ratio(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _float_map(values: Mapping[str, Fraction]) -> dict[str, float]:
    return {name: float(values[name]) for name in WHEEL_JOINT_NAMES}


def _parse_full_wheel_map(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> dict[str, Fraction] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} is not a mapping")
        return None
    keys = {str(key) for key in value}
    expected = set(WHEEL_JOINT_NAMES)
    if keys != expected:
        errors.append(
            f"{label} wheel identity mismatch: expected={sorted(expected)} got={sorted(keys)}"
        )
        return None
    result: dict[str, Fraction] = {}
    for name in WHEEL_JOINT_NAMES:
        parsed = _fraction(value.get(name))
        if parsed is None:
            errors.append(f"{label}.{name} is missing or non-finite")
            return None
        result[name] = parsed
    return result


def _parse_plan(plan: Any) -> tuple[list[_PlanSegment], list[str]]:
    errors: list[str] = []
    raw_segments = getattr(plan, "segments", None)
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        return [], ["PlaybackPlan.segments is unavailable"]
    if not raw_segments:
        return [], ["PlaybackPlan.segments is empty"]
    segments: list[_PlanSegment] = []
    for position, segment in enumerate(raw_segments):
        index = _strict_int(getattr(segment, "segment_index", None))
        if index is None or index != position:
            errors.append(
                f"plan segment identity mismatch at position={position}: index={index}"
            )
            continue
        duration = _fraction(getattr(segment, "wheel_active_duration_s", None))
        if duration is None or duration < 0:
            errors.append(f"plan segment={index} has invalid wheel_active_duration_s")
            continue
        targets = _parse_full_wheel_map(
            getattr(segment, "wheel_applied_target_rad_s", None),
            label=f"plan.segment[{index}].wheel_applied_target_rad_s",
            errors=errors,
        )
        if targets is None:
            continue
        segments.append(_PlanSegment(index=index, duration=duration, targets=targets))
    if len(segments) != len(raw_segments):
        errors.append("one or more authoritative plan segments could not be parsed")
    return segments, errors


def _parse_batches(
    timing_trace: Mapping[str, Any],
    segments: Sequence[_PlanSegment],
) -> tuple[list[_Batch], dict[int, _Batch], list[str]]:
    errors: list[str] = []
    raw_batches = timing_trace.get("motion_batches")
    if not isinstance(raw_batches, Sequence) or isinstance(raw_batches, (str, bytes)):
        return [], {}, ["timing_trace.motion_batches is unavailable"]
    if not raw_batches:
        return [], {}, ["timing_trace.motion_batches is empty"]

    batches: list[_Batch] = []
    batch_ids: set[str] = set()
    previous_first: int | None = None
    for index, raw in enumerate(raw_batches):
        if not isinstance(raw, Mapping):
            errors.append(f"motion_batch[{index}] is not a mapping")
            continue
        batch_id = str(raw.get("batch_id", "") or "")
        if not batch_id:
            errors.append(f"motion_batch[{index}] has no batch_id")
        elif batch_id in batch_ids:
            errors.append(f"motion_batch[{index}] duplicates batch_id={batch_id!r}")
        batch_ids.add(batch_id)
        if raw.get("ack_valid") is not True:
            errors.append(f"motion_batch[{index}] ACK is not valid")
        if str(raw.get("ack_error", "") or ""):
            errors.append(
                f"motion_batch[{index}] ACK error: {raw.get('ack_error')}"
            )
        if raw.get("ack_present") is not True:
            errors.append(f"motion_batch[{index}] ACK is absent")
        if raw.get("adapter_called") is not True:
            errors.append(f"motion_batch[{index}] was not applied by the adapter")
        applied = _strict_int(raw.get("applied_sim_step"))
        first = _strict_int(raw.get("first_physics_step"))
        if applied is None or first is None or first != applied + 1:
            errors.append(
                f"motion_batch[{index}] first_physics_step is not applied_sim_step + 1"
            )
            continue
        if previous_first is not None and first <= previous_first:
            errors.append(
                f"motion_batch[{index}] first_physics_step={first} is not strictly after {previous_first}"
            )
        previous_first = first
        skew = _finite_float(raw.get("motion_start_skew_s"))
        if skew is None or skew != 0.0:
            errors.append(f"motion_batch[{index}] motion_start_skew_s is not exactly zero")
        targets = _parse_full_wheel_map(
            raw.get("wheel_targets_rad_s"),
            label=f"motion_batch[{index}].wheel_targets_rad_s",
            errors=errors,
        )
        if targets is None:
            continue
        segment_index_raw = raw.get("segment_index")
        segment_index = (
            _strict_int(segment_index_raw) if segment_index_raw is not None else None
        )
        if segment_index_raw is not None and segment_index is None:
            errors.append(f"motion_batch[{index}] segment_index is invalid")
        batches.append(
            _Batch(
                index=index,
                batch_id=batch_id,
                dispatch_kind=str(raw.get("dispatch_kind", "") or ""),
                segment_index=segment_index,
                applied_step=applied,
                first_step=first,
                targets=targets,
            )
        )

    source_by_segment: dict[int, list[_Batch]] = {}
    for batch in batches:
        if batch.dispatch_kind == "source_segment_start" and batch.segment_index is not None:
            source_by_segment.setdefault(batch.segment_index, []).append(batch)
    source_batches: dict[int, _Batch] = {}
    for segment in segments:
        matches = source_by_segment.get(segment.index, [])
        if len(matches) != 1:
            errors.append(
                f"plan segment={segment.index} expected one source_segment_start ACK, got {len(matches)}"
            )
            continue
        batch = matches[0]
        source_batches[segment.index] = batch
        if dict(batch.targets) != dict(segment.targets):
            errors.append(
                f"plan segment={segment.index} ACK wheel target identity mismatch"
            )
    extra_source_indices = set(source_by_segment) - {segment.index for segment in segments}
    if extra_source_indices:
        errors.append(
            f"source_segment_start ACKs reference unknown segments={sorted(extra_source_indices)}"
        )
    ordered_starts = [
        source_batches[index].first_step
        for index in sorted(source_batches)
        if index in source_batches
    ]
    if any(right <= left for left, right in zip(ordered_starts, ordered_starts[1:])):
        errors.append("source segment ACK first_physics_step order is invalid")

    final_batches = [
        batch for batch in batches if batch.dispatch_kind == "final_safety_stop"
    ]
    if len(final_batches) != 1:
        errors.append(
            f"expected exactly one final_safety_stop ACK, got {len(final_batches)}"
        )
    else:
        final_batch = final_batches[0]
        if not batches or final_batch != batches[-1]:
            errors.append("final_safety_stop is not the final motion batch")
        if any(value != 0 for value in final_batch.targets.values()):
            errors.append("final_safety_stop wheel targets are not exactly zero")
        if ordered_starts and final_batch.first_step <= ordered_starts[-1]:
            errors.append("final_safety_stop does not close the last plan segment")
    if len(batches) != len(raw_batches):
        errors.append("one or more motion batches could not be parsed")
    return batches, source_batches, errors


def _parse_rows(
    telemetry_rows: Sequence[Mapping[str, Any]],
    *,
    direction: int,
    required_target_start: int,
    required_target_end: int,
    required_measured_start: int,
    required_measured_end: int,
) -> tuple[dict[int, _Tick], list[str], list[str], list[str]]:
    common_errors: list[str] = []
    target_errors: list[str] = []
    measured_errors: list[str] = []
    rows_by_step: dict[int, Mapping[str, Any]] = {}
    previous_step: int | None = None
    for row_index, row in enumerate(telemetry_rows):
        if not isinstance(row, Mapping):
            common_errors.append(f"telemetry row[{row_index}] is not a mapping")
            continue
        step = _strict_int(row.get("sim_step"))
        if step is None:
            common_errors.append(f"telemetry row[{row_index}] sim_step is not an integer")
            continue
        if step in rows_by_step:
            common_errors.append(f"telemetry sim_step={step} is duplicated")
        if previous_step is not None and step != previous_step + 1:
            common_errors.append(
                f"telemetry ticks are not continuous: previous={previous_step} current={step}"
            )
        previous_step = step
        rows_by_step[step] = row

    for step in range(required_target_start, required_target_end + 1):
        if step not in rows_by_step:
            target_errors.append(f"telemetry target tick sim_step={step} is missing")
    for step in range(required_measured_start, required_measured_end + 1):
        if step not in rows_by_step:
            measured_errors.append(f"telemetry measured tick sim_step={step} is missing")

    ticks: dict[int, _Tick] = {}
    relevant_start = min(required_target_start, required_measured_start)
    relevant_end = max(required_target_end, required_measured_end)
    expected_signs = {
        name: float(WHEEL_FORWARD_SIGN[name]) for name in WHEEL_JOINT_NAMES
    }
    for step in range(relevant_start, relevant_end + 1):
        row = rows_by_step.get(step)
        if row is None:
            continue
        dt_value = _finite_float(row.get("physics_dt_s"))
        if dt_value is None or dt_value != PHYSICS_DT_S:
            common_errors.append(
                f"telemetry sim_step={step} physics_dt_s is not exactly 1/120"
            )
        row_direction = _sign(row.get("wheel_direction"))
        if row_direction != direction:
            common_errors.append(
                f"telemetry sim_step={step} wheel_direction identity mismatch"
            )
        row_signs = row.get("wheel_forward_sign")
        if not isinstance(row_signs, Mapping) or {
            str(key) for key in row_signs
        } != set(WHEEL_JOINT_NAMES):
            common_errors.append(
                f"telemetry sim_step={step} wheel_forward_sign identity is unavailable"
            )
        else:
            for name in WHEEL_JOINT_NAMES:
                value = _finite_float(row_signs.get(name))
                if value != expected_signs[name]:
                    common_errors.append(
                        f"telemetry sim_step={step} wheel_forward_sign[{name}] mismatch"
                    )

        canonical_target: dict[str, Fraction] | None = None
        if required_target_start <= step <= required_target_end:
            if row.get("physx_drive_target_evidence_valid") is not True:
                target_errors.append(
                    f"telemetry sim_step={step} independent PhysX target evidence is invalid"
                )
            raw_target_errors: list[str] = []
            raw_targets = _parse_full_wheel_map(
                {
                    name: dict(row.get("physx_joint_velocity_target_rad_s", {}) or {}).get(name)
                    for name in WHEEL_JOINT_NAMES
                }
                if isinstance(row.get("physx_joint_velocity_target_rad_s"), Mapping)
                else row.get("physx_joint_velocity_target_rad_s"),
                label=f"telemetry[{step}].physx_joint_velocity_target_rad_s",
                errors=raw_target_errors,
            )
            if raw_target_errors:
                target_errors.extend(raw_target_errors)
            if raw_targets is not None:
                canonical_target = {
                    name: raw_targets[name]
                    * direction
                    * int(WHEEL_FORWARD_SIGN[name])
                    for name in WHEEL_JOINT_NAMES
                }

        position: dict[str, float] | None = None
        velocity: dict[str, float] | None = None
        if required_measured_start <= step <= required_measured_end:
            raw_positions = row.get("measured_joint_position_rad")
            raw_velocities = row.get("measured_joint_velocity_rad_s")
            if not isinstance(raw_positions, Mapping):
                measured_errors.append(
                    f"telemetry sim_step={step} measured positions are unavailable"
                )
            if not isinstance(raw_velocities, Mapping):
                measured_errors.append(
                    f"telemetry sim_step={step} measured velocities are unavailable"
                )
            if isinstance(raw_positions, Mapping) and isinstance(raw_velocities, Mapping):
                parsed_position: dict[str, float] = {}
                parsed_velocity: dict[str, float] = {}
                for name in WHEEL_JOINT_NAMES:
                    q = _finite_float(raw_positions.get(name))
                    qd = _finite_float(raw_velocities.get(name))
                    if q is None:
                        measured_errors.append(
                            f"telemetry sim_step={step} measured q[{name}] is missing or non-finite"
                        )
                    if qd is None:
                        measured_errors.append(
                            f"telemetry sim_step={step} measured qd[{name}] is missing or non-finite"
                        )
                    if q is not None and qd is not None:
                        sign = direction * int(WHEEL_FORWARD_SIGN[name])
                        parsed_position[name] = sign * q
                        parsed_velocity[name] = sign * qd
                if len(parsed_position) == len(WHEEL_JOINT_NAMES):
                    position = parsed_position
                    velocity = parsed_velocity
        ticks[step] = _Tick(
            step=step,
            target=canonical_target,
            position=position,
            velocity=velocity,
        )
    return ticks, common_errors, target_errors, measured_errors


def _comparison(
    *,
    planned: Fraction,
    actual: Fraction,
    allowed: Fraction,
) -> dict[str, Any]:
    error = actual - planned
    absolute_error = abs(error)
    if allowed == 0:
        passed = absolute_error == 0
        comparison_limit = Fraction(0)
    else:
        # One nextafter is the entire representational allowance.  It is not a
        # physical or empirical tolerance.
        comparison_limit = Fraction.from_float(
            math.nextafter(float(allowed), math.inf)
        )
        passed = absolute_error <= comparison_limit
    return {
        "planned_integral_rad": float(planned),
        "planned_integral_fraction": _ratio(planned),
        "physx_target_integral_rad": float(actual),
        "physx_target_integral_fraction": _ratio(actual),
        "error_rad": float(error),
        "error_fraction": _ratio(error),
        "allowed_one_tick_error_rad": float(allowed),
        "allowed_one_tick_error_fraction": _ratio(allowed),
        "comparison_limit_rad": float(comparison_limit),
        "zero_target_requires_exact_zero": allowed == 0,
        "verdict": PASS if passed else FAIL,
    }


def _unavailable_comparison(
    *, planned: Fraction, allowed: Fraction, reasons: Sequence[str]
) -> dict[str, Any]:
    return {
        "planned_integral_rad": float(planned),
        "planned_integral_fraction": _ratio(planned),
        "physx_target_integral_rad": None,
        "physx_target_integral_fraction": None,
        "error_rad": None,
        "error_fraction": None,
        "allowed_one_tick_error_rad": float(allowed),
        "allowed_one_tick_error_fraction": _ratio(allowed),
        "comparison_limit_rad": None,
        "zero_target_requires_exact_zero": allowed == 0,
        "verdict": NOT_EVALUABLE,
        "not_evaluable_reasons": list(reasons),
    }


def _empty_measured_metrics() -> dict[str, dict[str, float]]:
    return {
        name: {
            "net_rotation_rad": 0.0,
            "total_variation_rad": 0.0,
            "forward_rotation_rad": 0.0,
            "reverse_rotation_rad": 0.0,
            "zero_target_total_variation_rad": 0.0,
            "qd_right_riemann_integral_rad": 0.0,
        }
        for name in WHEEL_JOINT_NAMES
    }


def evaluate_wheel_integral_evidence(
    *,
    plan: Any,
    timing_trace: Mapping[str, Any],
    telemetry_rows: Sequence[Mapping[str, Any]],
    wheel_direction: float = 1.0,
) -> dict[str, Any]:
    """Evaluate exact plan/ACK/PhysX integrals and measured phase metrics."""

    direction = _sign(wheel_direction)
    plan_segments, plan_errors = _parse_plan(plan)
    batches: list[_Batch] = []
    source_batches: dict[int, _Batch] = {}
    batch_errors: list[str] = []
    if direction is None:
        plan_errors.append("wheel_direction must be finite and non-zero")
    if plan_segments:
        batches, source_batches, batch_errors = _parse_batches(
            timing_trace, plan_segments
        )

    base = {
        "schema_version": 1,
        "physics_dt_fraction": "1/120",
        "physics_dt_s": PHYSICS_DT_S,
        "wheel_direction": direction,
        "wheel_forward_sign": {
            name: float(WHEEL_FORWARD_SIGN[name]) for name in WHEEL_JOINT_NAMES
        },
        "wheel_joint_names": list(WHEEL_JOINT_NAMES),
        "plan_sha256": str(getattr(plan, "plan_sha256", "") or ""),
    }
    structural_errors = [*plan_errors, *batch_errors]
    if structural_errors or not plan_segments or not batches or direction is None:
        return {
            **base,
            "structural_errors": structural_errors,
            "target_integral_verdict": NOT_EVALUABLE,
            "target_not_evaluable_reasons": structural_errors,
            "measured_unwrap_status": NOT_EVALUABLE,
            "measured_not_evaluable_reasons": structural_errors,
            "measured_tracking_verdict": NOT_EVALUABLE,
            "measured_tracking_reason": "no registered measured tracking/repeatability threshold",
            "overall_verdict": NOT_EVALUABLE,
            "physical_success": False,
            "segments": [],
            "ticks": [],
            "measured_delta_ticks": [],
        }

    first_batch_step = batches[0].first_step
    last_batch_step = batches[-1].first_step
    first_source_step = source_batches[0].first_step
    final_stop_step = batches[-1].first_step
    segment_windows: dict[int, tuple[int, int]] = {}
    for position, segment in enumerate(plan_segments):
        start = source_batches[segment.index].first_step
        end = (
            source_batches[plan_segments[position + 1].index].first_step
            if position + 1 < len(plan_segments)
            else final_stop_step
        )
        if end <= start:
            structural_errors.append(
                f"segment={segment.index} ACK interval is empty or reversed"
            )
        segment_windows[segment.index] = (start, end)
    if structural_errors:
        return {
            **base,
            "structural_errors": structural_errors,
            "target_integral_verdict": NOT_EVALUABLE,
            "target_not_evaluable_reasons": structural_errors,
            "measured_unwrap_status": NOT_EVALUABLE,
            "measured_not_evaluable_reasons": structural_errors,
            "measured_tracking_verdict": NOT_EVALUABLE,
            "measured_tracking_reason": "no registered measured tracking/repeatability threshold",
            "overall_verdict": NOT_EVALUABLE,
            "physical_success": False,
            "segments": [],
            "ticks": [],
            "measured_delta_ticks": [],
        }

    ticks, common_errors, target_errors, measured_errors = _parse_rows(
        telemetry_rows,
        direction=direction,
        required_target_start=first_batch_step,
        required_target_end=last_batch_step,
        required_measured_start=first_source_step - 1,
        required_measured_end=final_stop_step - 1,
    )
    target_reasons = [*common_errors, *target_errors]
    measured_reasons = [*common_errors, *measured_errors]

    active_batch_by_step: dict[int, _Batch] = {}
    batch_cursor = 0
    for step in range(first_batch_step, last_batch_step + 1):
        while (
            batch_cursor + 1 < len(batches)
            and batches[batch_cursor + 1].first_step <= step
        ):
            batch_cursor += 1
        active_batch_by_step[step] = batches[batch_cursor]

    segment_by_step: dict[int, int] = {}
    for segment_index, (start, end) in segment_windows.items():
        for step in range(start, end):
            segment_by_step[step] = segment_index

    tick_rows: list[dict[str, Any]] = []
    actual_by_segment: dict[int, dict[str, Fraction]] = {
        segment.index: {name: Fraction(0) for name in WHEEL_JOINT_NAMES}
        for segment in plan_segments
    }
    if not target_reasons:
        for step in range(first_batch_step, last_batch_step + 1):
            tick = ticks.get(step)
            if tick is None or tick.target is None:
                target_reasons.append(
                    f"telemetry sim_step={step} has no parsed PhysX target"
                )
                continue
            batch = active_batch_by_step[step]
            segment_index = segment_by_step.get(step)
            if segment_index is not None:
                for name in WHEEL_JOINT_NAMES:
                    actual_by_segment[segment_index][name] += (
                        tick.target[name] * PHYSICS_DT
                    )
            tick_rows.append(
                {
                    "sim_step": step,
                    "batch_index": batch.index,
                    "batch_id": batch.batch_id,
                    "dispatch_kind": batch.dispatch_kind,
                    "segment_index": segment_index,
                    "ack_target_rad_s": _float_map(batch.targets),
                    "physx_target_rad_s": _float_map(tick.target),
                    "target_error_rad_s": {
                        name: float(tick.target[name] - batch.targets[name])
                        for name in WHEEL_JOINT_NAMES
                    },
                }
            )

    segment_rows: list[dict[str, Any]] = []
    full_planned = {
        name: sum(
            (segment.targets[name] * segment.duration for segment in plan_segments),
            start=Fraction(0),
        )
        for name in WHEEL_JOINT_NAMES
    }
    full_actual = {name: Fraction(0) for name in WHEEL_JOINT_NAMES}
    target_failed = False
    for segment in plan_segments:
        wheel_rows: dict[str, Any] = {}
        for name in WHEEL_JOINT_NAMES:
            planned = segment.targets[name] * segment.duration
            allowed = abs(segment.targets[name]) * PHYSICS_DT
            if target_reasons:
                comparison = _unavailable_comparison(
                    planned=planned, allowed=allowed, reasons=target_reasons
                )
            else:
                actual = actual_by_segment[segment.index][name]
                comparison = _comparison(
                    planned=planned, actual=actual, allowed=allowed
                )
                target_failed = target_failed or comparison["verdict"] == FAIL
                full_actual[name] += actual
            wheel_rows[name] = comparison
        start, end = segment_windows[segment.index]
        segment_rows.append(
            {
                "segment_index": segment.index,
                "first_physics_step_inclusive": start,
                "next_segment_or_final_stop_step_exclusive": end,
                "physics_tick_count": end - start,
                "plan_wheel_active_duration_s": float(segment.duration),
                "plan_wheel_active_duration_fraction": _ratio(segment.duration),
                "plan_target_rad_s": _float_map(segment.targets),
                "target_integrals": wheel_rows,
                "target_integral_verdict": (
                    NOT_EVALUABLE
                    if target_reasons
                    else FAIL
                    if any(row["verdict"] == FAIL for row in wheel_rows.values())
                    else PASS
                ),
            }
        )

    full_target_rows: dict[str, Any] = {}
    for name in WHEEL_JOINT_NAMES:
        max_plan_speed = max(
            (abs(segment.targets[name]) for segment in plan_segments),
            default=Fraction(0),
        )
        if target_reasons:
            full_target_rows[name] = _unavailable_comparison(
                planned=full_planned[name],
                allowed=max_plan_speed * PHYSICS_DT,
                reasons=target_reasons,
            )
        else:
            comparison = _comparison(
                planned=full_planned[name],
                actual=full_actual[name],
                allowed=max_plan_speed * PHYSICS_DT,
            )
            full_target_rows[name] = comparison
            target_failed = target_failed or comparison["verdict"] == FAIL
    target_verdict = (
        NOT_EVALUABLE
        if target_reasons
        else FAIL
        if target_failed
        else PASS
    )

    measured_by_segment: dict[int, dict[str, dict[str, float]]] = {
        segment.index: _empty_measured_metrics() for segment in plan_segments
    }
    measured_total = _empty_measured_metrics()
    measured_delta_rows: list[dict[str, Any]] = []
    if not measured_reasons:
        for segment in plan_segments:
            start, end = segment_windows[segment.index]
            for step in range(start, end):
                previous = ticks.get(step - 1)
                current = ticks.get(step)
                if (
                    previous is None
                    or current is None
                    or previous.position is None
                    or previous.velocity is None
                    or current.position is None
                    or current.velocity is None
                    or current.target is None
                ):
                    measured_reasons.append(
                        f"segment={segment.index} measured interval ending sim_step={step} is incomplete"
                    )
                    continue
                delta_map: dict[str, float] = {}
                bound_map: dict[str, float] = {}
                for name in WHEEL_JOINT_NAMES:
                    raw_delta = current.position[name] - previous.position[name]
                    delta = math.atan2(math.sin(raw_delta), math.cos(raw_delta))
                    speed_bound = max(
                        abs(previous.velocity[name]),
                        abs(current.velocity[name]),
                        abs(float(current.target[name])),
                        float(DEFAULT_MAX_WHEEL_SPEED_RAD_S),
                    )
                    bound_map[name] = speed_bound
                    maximum_unique_delta = speed_bound * PHYSICS_DT_S
                    if maximum_unique_delta >= math.pi:
                        measured_reasons.append(
                            f"segment={segment.index} sim_step={step} {name} speed bound cannot uniquely unwrap phase"
                        )
                        continue
                    if abs(delta) > maximum_unique_delta:
                        measured_reasons.append(
                            f"segment={segment.index} sim_step={step} {name} phase delta exceeds sampled speed bound"
                        )
                        continue
                    delta_map[name] = delta
                    for destination in (
                        measured_by_segment[segment.index][name],
                        measured_total[name],
                    ):
                        destination["net_rotation_rad"] += delta
                        destination["total_variation_rad"] += abs(delta)
                        destination["forward_rotation_rad"] += max(0.0, delta)
                        destination["reverse_rotation_rad"] += max(0.0, -delta)
                        if active_batch_by_step[step].targets[name] == 0:
                            destination["zero_target_total_variation_rad"] += abs(delta)
                        destination["qd_right_riemann_integral_rad"] += (
                            current.velocity[name] * PHYSICS_DT_S
                        )
                measured_delta_rows.append(
                    {
                        "segment_index": segment.index,
                        "sim_step": step,
                        "previous_sim_step": step - 1,
                        "canonical_unwrapped_delta_rad": delta_map,
                        "sampled_speed_bound_rad_s": bound_map,
                        "ack_target_rad_s": _float_map(
                            active_batch_by_step[step].targets
                        ),
                    }
                )

    measured_status = AVAILABLE if not measured_reasons else NOT_EVALUABLE
    for segment_row in segment_rows:
        index = int(segment_row["segment_index"])
        segment_row["measured_rotation"] = measured_by_segment[index]
        segment_row["measured_unwrap_status"] = measured_status
        segment_row["measured_tracking_verdict"] = NOT_EVALUABLE
        segment_row["measured_tracking_reason"] = (
            "no registered measured tracking/repeatability threshold"
        )

    overall = FAIL if target_verdict == FAIL else NOT_EVALUABLE
    return {
        **base,
        "structural_errors": [],
        "target_integral_verdict": target_verdict,
        "target_not_evaluable_reasons": target_reasons,
        "measured_unwrap_status": measured_status,
        "measured_not_evaluable_reasons": measured_reasons,
        "measured_tracking_verdict": NOT_EVALUABLE,
        "measured_tracking_reason": "no registered measured tracking/repeatability threshold",
        "full_target_integrals": full_target_rows,
        "measured_rotation": measured_total,
        "segments": segment_rows,
        "ticks": tick_rows,
        "measured_delta_ticks": measured_delta_rows,
        "overall_verdict": overall,
        "physical_success": False,
    }


build_wheel_integral_evidence = evaluate_wheel_integral_evidence


__all__ = [
    "AVAILABLE",
    "FAIL",
    "NOT_EVALUABLE",
    "PASS",
    "PHYSICS_DT",
    "PHYSICS_DT_S",
    "build_wheel_integral_evidence",
    "evaluate_wheel_integral_evidence",
]
