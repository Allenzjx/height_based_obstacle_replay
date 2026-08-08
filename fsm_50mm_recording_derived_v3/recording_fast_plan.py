"""Lossless exports of the project's authoritative Fast Replay planner.

This module deliberately calls :func:`playback.plan_from_steps`.  It does not
reimplement or approximate the timing semantics used by the simulator.
"""

from __future__ import annotations

import csv
import dataclasses
import json
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

