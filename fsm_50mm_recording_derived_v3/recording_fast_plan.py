"""Lossless exports of the project's authoritative Fast Replay planner.

This module deliberately calls :func:`playback.plan_from_steps`.  It does not
reimplement or approximate the timing semantics used by the simulator.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from command_model import WHEEL_JOINT_NAMES
from playback import PlaybackPlan, plan_from_steps
from sequence_model import event_playback_commands, is_record_marker, normalize_events


FAST_PLAN_COLUMNS = (
    "source_version",
    "source_step_index",
    "source_event_indices",
    "decoded_segment_index",
    "command_start_s",
    "command_end_s",
    "commands",
    "servo_target_deg",
    "wheel_target_rad_s",
    "concurrent",
    "servo_duration_s",
    "wheel_duration_s",
    "explicit_hold_s",
    "final_segment_duration_s",
    "expected_wheel_displacement_rad",
    "command_provenance",
)

DISPATCH_TRACE_COLUMNS = (
    "source_version",
    "source_command_cursor",
    "source_step_index",
    "source_event_index",
    "source_expanded_index",
    "source_command",
    "source_event_time_s",
    "compiler_outcome",
    "plan_event_index",
    "compiled_segment_index",
    "plan_global_command_index",
    "planned_dispatch_time_s",
    "actual_dispatch_time_s",
    "dispatch_lateness_s",
    "servo_target_deg",
    "wheel_target_rad_s",
    "atomic_batch_id",
    "atomic_batch_applied_sim_step",
    "atomic_batch_first_physics_step",
    "atomic_batch_motion_start_skew_s",
    "atomic_batch_ack_valid",
    "atomic_batch_ack_error",
    "command_cursor",
    "segment_cursor",
    "macro_state_cursor",
)


def _source_command_rows(steps: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Flatten source events without losing their original event index."""

    result: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        step_index = int(step.get("index", 0) or 0)
        flattened: list[dict[str, Any]] = []
        for event_index, event in enumerate(normalize_events(step.get("events", []))):
            if is_record_marker(str(event.get("command", ""))):
                continue
            for expanded_index, command in enumerate(event_playback_commands(event)):
                command = str(command).strip()
                if command:
                    flattened.append(
                        {
                            "event_index": event_index,
                            "expanded_index": expanded_index,
                            "command": command,
                            "event_time_s": float(event.get("time", 0.0) or 0.0),
                            "kind": str(event.get("kind", "command") or "command"),
                        }
                    )
        result[step_index] = flattened
    return result


def _map_plan_events_to_source(
    steps: list[dict[str, Any]], plan: PlaybackPlan
) -> dict[int, dict[str, Any]]:
    """Map planner event indices back to source event/expanded-command indices.

    The production planner intentionally drops semantic no-ops.  Matching is
    therefore monotonic within each source step and may skip source commands,
    but it may never reorder them.
    """

    source = _source_command_rows(steps)
    cursor = {step_index: 0 for step_index in source}
    mapped: dict[int, dict[str, Any]] = {}
    for plan_index, event in enumerate(plan.events):
        step_index = int(event.source_step or 0)
        candidates = source.get(step_index, [])
        start = cursor.get(step_index, 0)
        match_index: int | None = None
        for index in range(start, len(candidates)):
            if candidates[index]["command"] == event.base_command:
                match_index = index
                break
        if match_index is None:
            # Keep the loss explicit.  A report may not silently invent a
            # source event number when normalization changes a command.
            mapped[plan_index] = {
                "event_index": None,
                "expanded_index": None,
                "command": event.base_command,
                "mapping_error": "source command not found monotonically",
            }
            continue
        mapped[plan_index] = dict(candidates[match_index])
        cursor[step_index] = match_index + 1
    return mapped


def fast_plan_rows(
    *, source_version: str, steps: list[dict[str, Any]], max_wheel_speed: float
) -> tuple[PlaybackPlan, list[dict[str, Any]]]:
    """Return the authoritative Fast plan and a stable, audit-friendly table."""

    plan = plan_from_steps(
        steps,
        profile="fast",
        max_wheel_speed=max_wheel_speed,
        label=f"50 mm {source_version} fast audit",
        sequence_total_steps=len(steps),
    )
    source_map = _map_plan_events_to_source(steps, plan)
    rows: list[dict[str, Any]] = []
    for segment in plan.segments:
        plan_event_indices = range(
            segment.event_start_index,
            segment.event_start_index + segment.event_count,
        )
        plan_events = [plan.events[index] for index in plan_event_indices]
        source_rows = [source_map[index] for index in plan_event_indices]
        event_indices = sorted(
            {
                int(row["event_index"])
                for row in source_rows
                if row.get("event_index") is not None
            }
        )
        wheel_targets = {
            name: float(segment.wheel_applied_target_rad_s.get(name, 0.0))
            for name in WHEEL_JOINT_NAMES
        }
        wheel_displacement = {
            name: target * float(segment.wheel_active_duration_s)
            for name, target in wheel_targets.items()
        }
        final_duration = float(segment.planned_end_s - segment.planned_start_s)
        rows.append(
            {
                "source_version": source_version,
                "source_step_index": int(segment.source_step),
                "source_event_indices": event_indices,
                "decoded_segment_index": int(segment.segment_index),
                "command_start_s": float(segment.planned_start_s),
                "command_end_s": float(segment.planned_end_s),
                "commands": [event.base_command for event in plan_events],
                "servo_target_deg": {
                    name: float(value) for name, value in segment.servo_targets.items()
                },
                "wheel_target_rad_s": wheel_targets,
                "concurrent": bool(
                    segment.servo_targets
                    and any(abs(value) > 1.0e-12 for value in wheel_targets.values())
                ),
                "servo_duration_s": float(segment.servo_duration_s),
                "wheel_duration_s": float(segment.wheel_active_duration_s),
                "explicit_hold_s": float(segment.explicit_hold_s),
                "final_segment_duration_s": final_duration,
                "expected_wheel_displacement_rad": wheel_displacement,
                "command_provenance": [
                    {
                        "plan_event_index": index,
                        "source_event_index": source_map[index].get("event_index"),
                        "source_expanded_index": source_map[index].get("expanded_index"),
                        "source_step_index": int(segment.source_step),
                        "base_command": plan.events[index].base_command,
                        "dispatch_command": bool(plan.events[index].dispatch_command),
                        "mapping_error": source_map[index].get("mapping_error", ""),
                    }
                    for index in plan_event_indices
                ],
            }
        )
    return plan, rows


def build_source_dispatch_ledger(
    *,
    source_version: str,
    steps: list[dict[str, Any]],
    plan: PlaybackPlan,
    timing_trace: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Account for every expanded source command against live production ACKs.

    Commands intentionally removed by the production compiler are retained as
    explicit ``SEMANTIC_NOOP`` rows.  Every retained command must resolve to
    exactly one plan event, one timing row, and a valid source-segment motion
    batch ACK; otherwise the returned summary fails closed.
    """

    source_by_step = _source_command_rows(steps)
    source_rows = [
        {"source_step_index": step_index, **row}
        for step_index, rows in source_by_step.items()
        for row in rows
    ]
    source_rows.sort(
        key=lambda row: (
            int(row["source_step_index"]),
            int(row["event_index"]),
            int(row["expanded_index"]),
        )
    )
    source_map = _map_plan_events_to_source(steps, plan)
    plan_by_source: dict[tuple[int, int, int], list[int]] = {}
    mapping_errors: list[str] = []
    for plan_index, event in enumerate(plan.events):
        mapped = dict(source_map.get(plan_index, {}) or {})
        if mapped.get("mapping_error"):
            mapping_errors.append(
                f"plan_event={plan_index}: {mapped.get('mapping_error')}"
            )
            continue
        key = (
            int(event.source_step or 0),
            int(mapped.get("event_index")),
            int(mapped.get("expanded_index")),
        )
        plan_by_source.setdefault(key, []).append(plan_index)

    timing_commands = list(timing_trace.get("commands", []) or [])
    timing_by_global: dict[int, list[dict[str, Any]]] = {}
    for raw in timing_commands:
        row = dict(raw or {})
        try:
            cursor = int(row.get("global_command_index"))
        except (TypeError, ValueError):
            continue
        timing_by_global.setdefault(cursor, []).append(row)
    motion_batches = [
        dict(row or {}) for row in list(timing_trace.get("motion_batches", []) or [])
    ]
    allowed_runtime_dispatch_kinds = {
        "playback_start_boundary",
        "wheel_channel_completion_stop",
        "wheel_channel_resume",
        "pause_stop",
        "pause_resume",
        "implicit_idle_wheel_stop",
        "final_safety_stop",
    }
    dispatch_kind_counts: dict[str, int] = {}
    runtime_batch_errors: list[str] = []
    applied_steps: dict[int, list[int]] = {}
    for batch_index, row in enumerate(motion_batches):
        kind = str(row.get("dispatch_kind", "") or "")
        dispatch_kind_counts[kind] = dispatch_kind_counts.get(kind, 0) + 1
        if kind != "source_segment_start" and kind not in allowed_runtime_dispatch_kinds:
            runtime_batch_errors.append(
                f"motion_batch={batch_index} has unsupported dispatch_kind={kind!r}"
            )
        if row.get("ack_valid") is not True or str(row.get("ack_error", "") or ""):
            runtime_batch_errors.append(
                f"motion_batch={batch_index} kind={kind} has invalid ACK: "
                f"{row.get('ack_error', '')}"
            )
        try:
            applied_step = int(row.get("applied_sim_step"))
            first_step = int(row.get("first_physics_step"))
            if first_step != applied_step + 1:
                runtime_batch_errors.append(
                    f"motion_batch={batch_index} first physics step is not applied+1"
                )
            applied_steps.setdefault(applied_step, []).append(batch_index)
        except (TypeError, ValueError):
            runtime_batch_errors.append(
                f"motion_batch={batch_index} applied/first physics step is invalid"
            )
        try:
            skew = float(row.get("motion_start_skew_s"))
            if not math.isfinite(skew) or abs(skew) > 1.0e-12:
                runtime_batch_errors.append(
                    f"motion_batch={batch_index} motion_start_skew_s is not zero"
                )
        except (TypeError, ValueError):
            runtime_batch_errors.append(
                f"motion_batch={batch_index} motion_start_skew_s is invalid"
            )
        try:
            wheel_targets = {
                str(name): float(value)
                for name, value in dict(
                    row.get("wheel_targets_rad_s", {}) or {}
                ).items()
            }
        except (TypeError, ValueError):
            wheel_targets = {}
        if set(wheel_targets) != set(WHEEL_JOINT_NAMES) or not all(
            math.isfinite(value) for value in wheel_targets.values()
        ):
            runtime_batch_errors.append(
                f"motion_batch={batch_index} does not contain one finite full wheel map"
            )
    for applied_step, indices in sorted(applied_steps.items()):
        if len(indices) != 1:
            runtime_batch_errors.append(
                f"physics step {applied_step} has {len(indices)} motion batches: {indices}"
            )
    final_stops = [
        row
        for row in motion_batches
        if str(row.get("dispatch_kind", "") or "") == "final_safety_stop"
    ]
    if len(final_stops) != 1:
        runtime_batch_errors.append(
            f"expected exactly one final_safety_stop, got {len(final_stops)}"
        )
    else:
        final_stop = final_stops[0]
        try:
            final_wheels = {
                str(name): float(value)
                for name, value in dict(
                    final_stop.get("wheel_targets_rad_s", {}) or {}
                ).items()
            }
        except (TypeError, ValueError):
            final_wheels = {}
        if (
            set(final_wheels) != set(WHEEL_JOINT_NAMES)
            or any(abs(value) > 1.0e-12 for value in final_wheels.values())
            or dict(final_stop.get("servo_targets_deg", {}) or {})
        ):
            runtime_batch_errors.append(
                "final_safety_stop is not an all-zero wheel-only batch"
            )
        if not motion_batches or final_stop is not motion_batches[-1]:
            runtime_batch_errors.append("final_safety_stop is not the final motion batch")
    start_boundaries = [
        row
        for row in motion_batches
        if str(row.get("dispatch_kind", "") or "")
        == "playback_start_boundary"
    ]
    if len(start_boundaries) != 1:
        runtime_batch_errors.append(
            f"expected exactly one playback_start_boundary, got {len(start_boundaries)}"
        )
    elif not motion_batches or start_boundaries[0] is not motion_batches[0]:
        runtime_batch_errors.append(
            "playback_start_boundary is not the first motion batch"
        )
    primary_batch_by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in motion_batches:
        if str(row.get("dispatch_kind", "")) != "source_segment_start":
            continue
        try:
            segment_index = int(row.get("segment_index"))
        except (TypeError, ValueError):
            continue
        primary_batch_by_segment.setdefault(segment_index, []).append(row)
    readiness_tokens: set[str] = set()
    if plan.timing.get("requires_motion_start_readiness_token") is True:
        for segment_index, batches in sorted(primary_batch_by_segment.items()):
            for batch in batches:
                metadata = dict(batch.get("recording_metadata", {}) or {})
                token = str(
                    metadata.get("motion_start_readiness_token", "") or ""
                ).lower()
                try:
                    bound_step = int(
                        metadata.get("motion_start_readiness_bound_sim_step")
                    )
                    applied_step = int(batch.get("applied_sim_step"))
                except (TypeError, ValueError):
                    bound_step = applied_step = -1
                if len(token) != 64 or any(
                    character not in "0123456789abcdef" for character in token
                ):
                    runtime_batch_errors.append(
                        f"segment={segment_index} has no valid MOTION_START_READY token"
                    )
                else:
                    readiness_tokens.add(token)
                if bound_step < 0 or applied_step < bound_step:
                    runtime_batch_errors.append(
                        f"segment={segment_index} readiness binding is after dispatch"
                    )
        if len(readiness_tokens) != 1:
            runtime_batch_errors.append(
                "source motion batches do not share exactly one readiness token"
            )

    ledger: list[dict[str, Any]] = []
    errors: list[str] = list(mapping_errors) + runtime_batch_errors
    retained_count = 0
    semantic_noop_count = 0
    for source_cursor, source in enumerate(source_rows, start=1):
        key = (
            int(source["source_step_index"]),
            int(source["event_index"]),
            int(source["expanded_index"]),
        )
        matches = list(plan_by_source.get(key, []))
        common = {
            "source_version": str(source_version),
            "source_command_cursor": int(source_cursor),
            "source_step_index": key[0],
            "source_event_index": key[1],
            "source_expanded_index": key[2],
            "source_command": str(source["command"]),
            "source_event_time_s": float(source["event_time_s"]),
            "macro_state_cursor": "RECORDING_FAST_REPLAY",
        }
        if not matches:
            semantic_noop_count += 1
            ledger.append(
                {
                    **common,
                    "compiler_outcome": "SEMANTIC_NOOP_ELIDED_BY_PRODUCTION_COMPILER",
                    "plan_event_index": None,
                    "compiled_segment_index": None,
                    "plan_global_command_index": None,
                    "planned_dispatch_time_s": None,
                    "actual_dispatch_time_s": None,
                    "dispatch_lateness_s": None,
                    "servo_target_deg": {},
                    "wheel_target_rad_s": {},
                    "atomic_batch_id": "",
                    "atomic_batch_applied_sim_step": None,
                    "atomic_batch_first_physics_step": None,
                    "atomic_batch_motion_start_skew_s": None,
                    "atomic_batch_ack_valid": True,
                    "atomic_batch_ack_error": "",
                    "command_cursor": None,
                    "segment_cursor": None,
                }
            )
            continue
        if len(matches) != 1:
            errors.append(f"source={key} resolved {len(matches)} plan events")
        plan_index = int(matches[0])
        event = plan.events[plan_index]
        global_cursor = int(event.global_command_index)
        timing_matches = timing_by_global.get(global_cursor, [])
        if len(timing_matches) != 1:
            errors.append(
                f"source={key} global_command={global_cursor} resolved "
                f"{len(timing_matches)} timing rows"
            )
        timing = dict(timing_matches[0] if timing_matches else {})
        segment_index = int(event.segment_index)
        batch_matches = primary_batch_by_segment.get(segment_index, [])
        if len(batch_matches) != 1:
            errors.append(
                f"source={key} segment={segment_index} resolved "
                f"{len(batch_matches)} primary batch ACK rows"
            )
        batch = dict(batch_matches[0] if batch_matches else {})
        batch_valid = batch.get("ack_valid") is True
        if not batch_valid:
            errors.append(
                f"source={key} segment={segment_index} has invalid batch ACK: "
                f"{batch.get('ack_error', '')}"
            )
        retained_count += 1
        ledger.append(
            {
                **common,
                "compiler_outcome": "APPLIED_VIA_PRODUCTION_SEGMENT_BATCH",
                "plan_event_index": plan_index,
                "compiled_segment_index": segment_index,
                "plan_global_command_index": global_cursor,
                "planned_dispatch_time_s": timing.get("planned_start_sim_time"),
                "actual_dispatch_time_s": timing.get("actual_start_sim_time"),
                "dispatch_lateness_s": timing.get("scheduler_lateness"),
                "servo_target_deg": dict(event.servo_targets),
                "wheel_target_rad_s": {
                    name: float(event.wheel_applied_target_rad_s[index])
                    for index, name in enumerate(WHEEL_JOINT_NAMES)
                    if index < len(event.wheel_applied_target_rad_s)
                },
                "atomic_batch_id": str(batch.get("batch_id", "") or ""),
                "atomic_batch_applied_sim_step": batch.get("applied_sim_step"),
                "atomic_batch_first_physics_step": batch.get("first_physics_step"),
                "atomic_batch_motion_start_skew_s": batch.get(
                    "motion_start_skew_s"
                ),
                "atomic_batch_ack_valid": batch_valid,
                "atomic_batch_ack_error": str(batch.get("ack_error", "") or ""),
                "command_cursor": global_cursor,
                "segment_cursor": segment_index,
            }
        )

    source_cursors = [int(row["source_command_cursor"]) for row in ledger]
    retained_plan_indices = sorted(
        int(row["plan_event_index"])
        for row in ledger
        if row.get("plan_event_index") is not None
    )
    if source_cursors != list(range(1, len(ledger) + 1)):
        errors.append("source command cursor is not contiguous")
    if retained_plan_indices != list(range(len(plan.events))):
        errors.append("retained plan-event coverage is incomplete or duplicated")
    if len(timing_commands) != len(plan.events):
        errors.append(
            f"live timing row count {len(timing_commands)} != plan events {len(plan.events)}"
        )
    summary = {
        "schema_version": "fsm50.source_dispatch_ledger.v1",
        "source_version": str(source_version),
        "source_command_count": len(source_rows),
        "retained_plan_event_count": retained_count,
        "semantic_noop_count": semantic_noop_count,
        "plan_event_count": len(plan.events),
        "plan_segment_count": len(plan.segments),
        "live_timing_command_count": len(timing_commands),
        "primary_motion_batch_count": sum(
            len(rows) for rows in primary_batch_by_segment.values()
        ),
        "runtime_generated_motion_batch_count": len(motion_batches)
        - sum(len(rows) for rows in primary_batch_by_segment.values()),
        "dispatch_kind_counts": dispatch_kind_counts,
        "one_motion_batch_per_physics_tick": not any(
            len(indices) != 1 for indices in applied_steps.values()
        ),
        "final_safety_stop_count": len(final_stops),
        "playback_start_boundary_count": len(start_boundaries),
        "motion_start_readiness_token_count": len(readiness_tokens),
        "motion_start_readiness_token": (
            next(iter(readiness_tokens)) if len(readiness_tokens) == 1 else ""
        ),
        "complete": not errors,
        "errors": errors,
    }
    return ledger, summary


def write_source_dispatch_ledger(
    *,
    csv_path: Path,
    json_path: Path,
    source_version: str,
    steps: list[dict[str, Any]],
    plan: PlaybackPlan,
    timing_trace: dict[str, Any],
) -> dict[str, Any]:
    rows, summary = build_source_dispatch_ledger(
        source_version=source_version,
        steps=steps,
        plan=plan,
        timing_trace=timing_trace,
    )
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(DISPATCH_TRACE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json_cell(row.get(key))
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in DISPATCH_TRACE_COLUMNS
                }
            )
    Path(json_path).write_text(
        json.dumps(
            {**summary, "rows": rows, "motion_batches": timing_trace.get("motion_batches", [])},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**summary, "csv_path": str(Path(csv_path)), "json_path": str(Path(json_path))}


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_fast_plan(
    *, output_dir: Path, source_version: str, steps: list[dict[str, Any]], max_wheel_speed: float
) -> dict[str, Any]:
    """Write JSON and CSV exports without touching the source recording."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plan, rows = fast_plan_rows(
        source_version=source_version,
        steps=steps,
        max_wheel_speed=max_wheel_speed,
    )
    json_path = output_dir / f"{source_version}_fast_plan.json"
    csv_path = output_dir / f"{source_version}_fast_plan.csv"
    payload = {
        "schema_version": "fsm50.recording_fast_plan.v1",
        "source_version": source_version,
        "profile_requested": "fast",
        "profile_normalized": plan.profile,
        "plan_sha256": plan.plan_sha256,
        "event_count": len(plan.events),
        "segment_count": len(plan.segments),
        "final_time_s": float(plan.final_time_s),
        "timing": plan.timing,
        "segments": rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FAST_PLAN_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json_cell(row[key])
                    if isinstance(row[key], (dict, list, tuple))
                    else row[key]
                    for key in FAST_PLAN_COLUMNS
                }
            )
    return {
        "plan": plan,
        "rows": rows,
        "json_path": json_path,
        "csv_path": csv_path,
    }
