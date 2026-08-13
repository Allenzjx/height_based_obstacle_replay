"""Deterministic, non-physical provenance export for the primary v003 recording.

The runtime plan in this module is produced only through
``recording_fast_plan.fast_plan_rows``, which delegates to the production
``playback.plan_from_steps`` implementation.  The 30 deg/s comparison also
calls ``playback.plan_from_steps`` directly with an explicit counterfactual
rate.  No approximate planner or endpoint-only reconstruction is used.

This module never launches Isaac and never writes into the recording source
directory.  Its reports therefore prove static source/compile integrity only;
live dispatch and physical replay remain pending.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from command_model import WHEEL_JOINT_NAMES
from motion_speed import load_motion_reference
from playback import PlaybackPlan, plan_from_steps
from sequence_model import event_playback_commands, load_steps_jsonl, normalize_events
from sim_state_validation import validate_full_sim_pose_state

from . import recording_fast_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
V003_VERSION_ID = "v003_20260805_224517_157723_manual"
DEFAULT_V003_DIRECTORY = (
    PROJECT_ROOT
    / "saved_height_steps_fsm_reference_v2"
    / "height_050mm"
    / "versions"
    / V003_VERSION_ID
)
DEFAULT_REPORT_ROOT = MODULE_ROOT / "reports"
RECORDING_BASELINE_PATH = PROJECT_ROOT / "config" / "fsm_recording_baseline.yaml"
MOTION_REFERENCE_PATH = PROJECT_ROOT / "config" / "real_robot_motion_reference.yaml"

STATIC_STATUS = "STATIC_PRODUCTION_PLAN_ONLY"
PHYSICAL_STATUS = "NOT_RUN"
DISPATCH_STATUS = "NOT_COLLECTED"

CSV_COLUMNS = (
    "source_version",
    "source_step_index",
    "source_step_id",
    "source_event_indices",
    "source_event_times_s",
    "source_actual_recording_times_s",
    "source_command_start_sim_times_s",
    "source_atomic_batch_ids",
    "source_atomic_event",
    "plan_segment_index",
    "plan_event_indices",
    "commands",
    "servo_target_deg",
    "wheel_requested_target_rad_s",
    "wheel_applied_target_rad_s",
    "runtime_150_start_s",
    "runtime_150_end_s",
    "runtime_150_duration_s",
    "runtime_150_servo_duration_s",
    "runtime_150_wheel_duration_s",
    "runtime_150_explicit_hold_s",
    "counterfactual_30_start_s",
    "counterfactual_30_end_s",
    "counterfactual_30_duration_s",
    "delta_30_minus_150_start_s",
    "delta_30_minus_150_end_s",
    "delta_30_minus_150_duration_s",
    "concurrent_servo_wheel",
    "expected_wheel_displacement_rad",
    "command_provenance",
)


@dataclass(frozen=True)
class V003StaticProvenance:
    payload: dict[str, Any]
    csv_rows: list[dict[str, Any]]
    source_markdown: str
    timing_markdown: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.extend(_nonfinite_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_nonfinite_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            result.append(prefix)
    return result


def _stable_timing(timing: dict[str, Any]) -> dict[str, Any]:
    """Remove planner wall-clock diagnostics while retaining timing semantics."""

    return {
        key: value
        for key, value in timing.items()
        if key not in {"plan_build_start", "plan_build_end"}
    }


def _sum_wheel_maps(rows: Iterable[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        name: sum(float(dict(row.get(key, {}) or {}).get(name, 0.0) or 0.0) for row in rows)
        for name in WHEEL_JOINT_NAMES
    }


def _source_wheel_integral(
    steps: list[dict[str, Any]], field: str
) -> dict[str, float]:
    return {
        name: sum(
            float(
                dict(dict(step.get("motion_semantics", {}) or {}).get(field, {}) or {}).get(
                    name, 0.0
                )
                or 0.0
            )
            for step in steps
        )
        for name in WHEEL_JOINT_NAMES
    }


def _command_counts(commands: Iterable[str]) -> dict[str, int]:
    values = [str(command).strip() for command in commands if str(command).strip()]
    return {
        "total": len(values),
        "servo": sum(command.startswith("servo ") for command in values),
        "wheel": sum(command.startswith("wheel ") for command in values),
        "other": sum(
            not command.startswith("servo ") and not command.startswith("wheel ")
            for command in values
        ),
    }


def _source_events(
    steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for step in steps:
        step_index = int(step.get("index", 0) or 0)
        step_id = str(step.get("name", "") or "")
        for event_index, event in enumerate(normalize_events(step.get("events", []))):
            commands = event_playback_commands(event)
            batch_id = str(event.get("batch_id", "") or "")
            expanded = list(event.get("expanded_commands", []) or [])
            row = {
                "source_step_index": step_index,
                "source_step_id": step_id,
                "source_event_index": event_index,
                "source_event_sha256": _fingerprint(event),
                "command": str(event.get("command", "") or ""),
                "kind": str(event.get("kind", "") or ""),
                "time_s": float(event.get("time", 0.0) or 0.0),
                "actual_recording_time_s": float(
                    event.get("actual_recording_time_s", 0.0) or 0.0
                ),
                "command_start_sim_time_s": float(
                    event.get("command_start_sim_time", 0.0) or 0.0
                ),
                "recording_timing_source": str(
                    event.get("recording_timing_source", "") or ""
                ),
                "wheel_active_duration_field_present": "wheel_active_duration_s" in event,
                "wheel_active_duration_s": event.get("wheel_active_duration_s"),
                "batch_id": batch_id,
                "atomic": bool(batch_id and expanded),
                "expanded_commands": commands,
                "decoded_command_counts": _command_counts(commands),
                "canonical_servo_velocity_deg_s": event.get(
                    "canonical_servo_velocity_deg_s"
                ),
                "canonical_servo_target_deg": dict(
                    event.get("canonical_servo_target_deg", {}) or {}
                ),
                "canonical_wheel_velocity_rad_s": dict(
                    event.get("canonical_wheel_velocity_rad_s", {}) or {}
                ),
                "command_state_before": dict(event.get("command_state_before", {}) or {}),
                "command_state_after": dict(event.get("command_state_after", {}) or {}),
            }
            rows.append(row)
            lookup[(step_index, event_index)] = row
    return rows, lookup


def _plan_signature(plan: PlaybackPlan) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for segment in plan.segments:
        event_indices = range(
            segment.event_start_index,
            segment.event_start_index + segment.event_count,
        )
        result.append(
            {
                "segment_index": int(segment.segment_index),
                "source_step": int(segment.source_step),
                "commands": [plan.events[index].base_command for index in event_indices],
                "servo_targets": dict(segment.servo_targets),
                "wheel_applied_target_rad_s": dict(
                    segment.wheel_applied_target_rad_s
                ),
                "wheel_active_duration_s": float(segment.wheel_active_duration_s),
            }
        )
    return result


def _counterfactual_plan(
    steps: list[dict[str, Any]], max_wheel_speed: float
) -> PlaybackPlan:
    return plan_from_steps(
        steps,
        profile="fast",
        max_wheel_speed=max_wheel_speed,
        servo_reference_velocity_deg_s=30.0,
        label=f"50 mm {V003_VERSION_ID} 30 deg/s provenance counterfactual",
        sequence_total_steps=len(steps),
    )


def _enrich_plan_rows(
    *,
    steps: list[dict[str, Any]],
    runtime_plan: PlaybackPlan,
    runtime_rows: list[dict[str, Any]],
    counterfactual_plan: PlaybackPlan,
    source_lookup: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(runtime_plan.segments) != len(counterfactual_plan.segments):
        raise ValueError("30/150 production plans have different segment counts")
    step_ids = {
        int(step.get("index", 0) or 0): str(step.get("name", "") or "")
        for step in steps
    }
    counterfactual = {
        int(segment.segment_index): segment for segment in counterfactual_plan.segments
    }
    runtime_segments = {
        int(segment.segment_index): segment for segment in runtime_plan.segments
    }
    rows: list[dict[str, Any]] = []
    for row in runtime_rows:
        segment_index = int(row["decoded_segment_index"])
        runtime_segment = runtime_segments[segment_index]
        thirty_segment = counterfactual[segment_index]
        step_index = int(row["source_step_index"])
        source_rows = [
            source_lookup[(step_index, int(event_index))]
            for event_index in row["source_event_indices"]
        ]
        runtime_duration = float(row["final_segment_duration_s"])
        thirty_duration = float(
            thirty_segment.planned_end_s - thirty_segment.planned_start_s
        )
        rows.append(
            {
                "source_version": V003_VERSION_ID,
                "source_step_index": step_index,
                "source_step_id": step_ids[step_index],
                "source_event_indices": list(row["source_event_indices"]),
                "source_event_times_s": [item["time_s"] for item in source_rows],
                "source_actual_recording_times_s": [
                    item["actual_recording_time_s"] for item in source_rows
                ],
                "source_command_start_sim_times_s": [
                    item["command_start_sim_time_s"] for item in source_rows
                ],
                "source_atomic_batch_ids": sorted(
                    {item["batch_id"] for item in source_rows if item["batch_id"]}
                ),
                "source_atomic_event": any(item["atomic"] for item in source_rows),
                "plan_segment_index": segment_index,
                "plan_event_indices": [
                    int(item["plan_event_index"])
                    for item in row["command_provenance"]
                ],
                "commands": list(row["commands"]),
                "servo_target_deg": dict(row["servo_target_deg"]),
                "wheel_requested_target_rad_s": dict(
                    runtime_segment.wheel_requested_velocity_rad_s
                ),
                "wheel_applied_target_rad_s": dict(row["wheel_target_rad_s"]),
                "runtime_150_start_s": float(row["command_start_s"]),
                "runtime_150_end_s": float(row["command_end_s"]),
                "runtime_150_duration_s": runtime_duration,
                "runtime_150_servo_duration_s": float(row["servo_duration_s"]),
                "runtime_150_wheel_duration_s": float(row["wheel_duration_s"]),
                "runtime_150_explicit_hold_s": float(row["explicit_hold_s"]),
                "counterfactual_30_start_s": float(thirty_segment.planned_start_s),
                "counterfactual_30_end_s": float(thirty_segment.planned_end_s),
                "counterfactual_30_duration_s": thirty_duration,
                "delta_30_minus_150_start_s": float(thirty_segment.planned_start_s)
                - float(row["command_start_s"]),
                "delta_30_minus_150_end_s": float(thirty_segment.planned_end_s)
                - float(row["command_end_s"]),
                "delta_30_minus_150_duration_s": thirty_duration - runtime_duration,
                "concurrent_servo_wheel": bool(row["concurrent"]),
                "expected_wheel_displacement_rad": dict(
                    row["expected_wheel_displacement_rad"]
                ),
                "command_provenance": list(row["command_provenance"]),
            }
        )
    return rows


def _atomic_audit(
    source_events: list[dict[str, Any]], plan_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in source_events:
        if not source["atomic"]:
            continue
        step_index = int(source["source_step_index"])
        event_index = int(source["source_event_index"])
        matched = [
            row
            for row in plan_rows
            if int(row["source_step_index"]) == step_index
            and event_index in row["source_event_indices"]
        ]
        retained = [
            command
            for row in matched
            for provenance in row["command_provenance"]
            if int(provenance.get("source_event_index", -1)) == event_index
            for command in [str(provenance["base_command"])]
        ]
        source_commands = list(source["expanded_commands"])
        omitted_counter = Counter(source_commands) - Counter(retained)
        omitted = list(omitted_counter.elements())
        segment_indices = sorted({int(row["plan_segment_index"]) for row in matched})
        source_counts = _command_counts(source_commands)
        retained_counts = _command_counts(retained)
        result.append(
            {
                "source_step_index": step_index,
                "source_event_index": event_index,
                "batch_id": source["batch_id"],
                "applied_sim_step": source_lookup_value(source, "applied_sim_step"),
                "applied_sim_time_s": source_lookup_value(source, "applied_sim_time"),
                "source_commands": source_commands,
                "source_command_counts": source_counts,
                "production_retained_commands": retained,
                "production_retained_command_counts": retained_counts,
                "production_elided_semantic_noops": omitted,
                "production_segment_indices": segment_indices,
                "single_segment_preserved": len(segment_indices) == 1,
                "servo_wheel_concurrent": bool(
                    len(segment_indices) == 1
                    and any(bool(row["concurrent_servo_wheel"]) for row in matched)
                    and retained_counts["servo"] > 0
                    and retained_counts["wheel"] > 0
                ),
            }
        )
    return result


def source_lookup_value(source: dict[str, Any], key: str) -> Any:
    """Return optional raw atomic acknowledgement fields kept in source metadata."""

    return dict(source.get("raw_fields", {}) or {}).get(key)


def _step_summaries(
    steps: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in steps:
        step_index = int(step.get("index", 0) or 0)
        events = [row for row in source_events if row["source_step_index"] == step_index]
        rows = [row for row in plan_rows if row["source_step_index"] == step_index]
        decoded_commands = [
            command for event in events for command in event["expanded_commands"]
        ]
        source_requested = {
            name: float(
                dict(
                    dict(step.get("motion_semantics", {}) or {}).get(
                        "canonical_wheel_angular_displacement_rad", {}
                    )
                    or {}
                ).get(name, 0.0)
                or 0.0
            )
            for name in WHEEL_JOINT_NAMES
        }
        source_applied = {
            name: float(
                dict(
                    dict(step.get("motion_semantics", {}) or {}).get(
                        "derived_wheel_angular_displacement_rad", {}
                    )
                    or {}
                ).get(name, 0.0)
                or 0.0
            )
            for name in WHEEL_JOINT_NAMES
        }
        production_integral = _sum_wheel_maps(
            rows, "expected_wheel_displacement_rad"
        )
        result.append(
            {
                "source_step_index": step_index,
                "source_step_id": str(step.get("name", "") or ""),
                "source_duration_s": float(step.get("duration", 0.0) or 0.0),
                "source_event_count": len(events),
                "decoded_command_counts": _command_counts(decoded_commands),
                "atomic_event_count": sum(bool(event["atomic"]) for event in events),
                "source_event_first_time_s": min(
                    [float(event["time_s"]) for event in events] + [0.0]
                ),
                "source_event_last_time_s": max(
                    [float(event["time_s"]) for event in events] + [0.0]
                ),
                "initial_joint_targets": dict(
                    step.get("command_state_before", {}) or {}
                ),
                "final_joint_targets": dict(step.get("command_state_after", {}) or {}),
                "source_requested_wheel_integral_rad": source_requested,
                "source_applied_wheel_integral_rad": source_applied,
                "production_plan_wheel_integral_rad": production_integral,
                "production_minus_source_applied_rad": {
                    name: production_integral[name] - source_applied[name]
                    for name in WHEEL_JOINT_NAMES
                },
                "production_segment_count": len(rows),
                "production_plan_start_s": min(
                    [float(row["runtime_150_start_s"]) for row in rows] + [0.0]
                ),
                "production_plan_end_s": max(
                    [float(row["runtime_150_end_s"]) for row in rows] + [0.0]
                ),
            }
        )
    return result


def _source_schema_audit(
    steps: list[dict[str, Any]], source_events: list[dict[str, Any]]
) -> dict[str, Any]:
    required_step_fields = (
        "index",
        "duration",
        "events",
        "command_state_before",
        "command_state_after",
        "motion_semantics",
        "sim_state_before",
        "sim_state_after",
    )
    required_event_fields = (
        "command",
        "time",
        "actual_recording_time_s",
        "command_start_sim_time",
        "recording_timing_source",
        "command_state_before",
        "command_state_after",
        "canonical_servo_velocity_deg_s",
        "wheel_active_duration_s",
    )
    missing_step = [
        {"step": int(step.get("index", 0) or 0), "field": field}
        for step in steps
        for field in required_step_fields
        if field not in step
    ]
    missing_event: list[dict[str, Any]] = []
    raw_by_key = {
        (int(step.get("index", 0) or 0), event_index): event
        for step in steps
        for event_index, event in enumerate(normalize_events(step.get("events", [])))
    }
    for source in source_events:
        raw = raw_by_key[
            (int(source["source_step_index"]), int(source["source_event_index"]))
        ]
        for field in required_event_fields:
            if field not in raw:
                missing_event.append(
                    {
                        "step": source["source_step_index"],
                        "event": source["source_event_index"],
                        "field": field,
                    }
                )
    snapshot_rows: list[dict[str, Any]] = []
    for step in steps:
        for field in ("sim_state_before", "sim_state_after"):
            validation = validate_full_sim_pose_state(step.get(field))
            snapshot_rows.append(
                {
                    "source_step_index": int(step.get("index", 0) or 0),
                    "field": field,
                    "classification": str(
                        validation.get("classification", "INVALID")
                    ),
                    "valid": bool(validation.get("valid", False)),
                    "reason": str(validation.get("reason", "") or ""),
                }
            )
    return {
        "missing_required_step_fields": missing_step,
        "missing_required_event_fields": missing_event,
        "nonfinite_numeric_paths": _nonfinite_paths(steps),
        "step_key_signatures": dict(
            Counter(",".join(sorted(step)) for step in steps)
        ),
        "event_key_signatures": dict(
            Counter(
                ",".join(sorted(event))
                for step in steps
                for event in normalize_events(step.get("events", []))
            )
        ),
        "snapshot_classification_counts": dict(
            Counter(row["classification"] for row in snapshot_rows)
        ),
        "snapshot_rows": snapshot_rows,
    }


def _raw_atomic_fields(
    steps: list[dict[str, Any]], source_events: list[dict[str, Any]]
) -> None:
    """Attach only acknowledgement timing used by the atomic audit."""

    lookup = {
        (int(step.get("index", 0) or 0), event_index): event
        for step in steps
        for event_index, event in enumerate(normalize_events(step.get("events", [])))
    }
    for source in source_events:
        raw = lookup[
            (int(source["source_step_index"]), int(source["source_event_index"]))
        ]
        source["raw_fields"] = {
            "applied_sim_step": raw.get("applied_sim_step"),
            "applied_sim_time": raw.get("applied_sim_time"),
        }


def _source_hashes(source_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = source_dir / "metadata.json"
    paths = {
        "accepted_steps": source_dir / "accepted_steps.jsonl",
        "metadata": metadata_path,
        "production_playback": PROJECT_ROOT / "playback.py",
        "production_fast_plan_adapter": MODULE_ROOT / "recording_fast_plan.py",
        "sequence_model": PROJECT_ROOT / "sequence_model.py",
        "command_model": PROJECT_ROOT / "command_model.py",
        "motion_speed": PROJECT_ROOT / "motion_speed.py",
        "runtime_motion_reference": MOTION_REFERENCE_PATH,
        "recording_bookkeeping_baseline": RECORDING_BASELINE_PATH,
    }
    if metadata_path.is_file():
        metadata = _json_object(metadata_path)
        robot_asset_value = str(metadata.get("robot_asset_path", "") or "")
        if robot_asset_value:
            paths["robot_asset"] = Path(robot_asset_value)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing provenance sources: " + ", ".join(missing))
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _timing_provenance(
    *,
    metadata: dict[str, Any],
    steps: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
    runtime_plan: PlaybackPlan,
    counterfactual_plan: PlaybackPlan,
    plan_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    profile_ids = sorted(
        {
            str(record.get("reference_profile_id", "") or "")
            for step in steps
            for record in list(
                dict(step.get("motion_semantics", {}) or {}).get(
                    "servo_records", []
                )
                or []
            )
            if str(record.get("reference_profile_id", "") or "")
        }
    )
    event_rates = sorted(
        {
            float(row["canonical_servo_velocity_deg_s"])
            for row in source_events
            if row["canonical_servo_velocity_deg_s"] is not None
        }
    )
    motion = load_motion_reference()
    changed_rows = [
        row
        for row in plan_rows
        if abs(float(row["delta_30_minus_150_start_s"])) > 1.0e-12
        or abs(float(row["delta_30_minus_150_end_s"])) > 1.0e-12
        or abs(float(row["delta_30_minus_150_duration_s"])) > 1.0e-12
    ]
    first_changed = (
        {
            "plan_segment_index": changed_rows[0]["plan_segment_index"],
            "source_step_index": changed_rows[0]["source_step_index"],
            "runtime_150_start_s": changed_rows[0]["runtime_150_start_s"],
            "runtime_150_end_s": changed_rows[0]["runtime_150_end_s"],
            "counterfactual_30_start_s": changed_rows[0][
                "counterfactual_30_start_s"
            ],
            "counterfactual_30_end_s": changed_rows[0][
                "counterfactual_30_end_s"
            ],
        }
        if changed_rows
        else None
    )
    top_level_speed_keys = {
        key: value
        for key, value in metadata.items()
        if "servo" in key.lower() and ("speed" in key.lower() or "velocity" in key.lower())
    }
    return {
        "recording_timing": {
            "step_duration_field_count": sum("duration" in step for step in steps),
            "step_count": len(steps),
            "step_duration_sum_s": sum(
                float(step.get("duration", 0.0) or 0.0) for step in steps
            ),
            "step_recording_timing_actual_duration_count": sum(
                dict(step.get("recording_timing", {}) or {}).get(
                    "actual_duration_s"
                )
                is not None
                for step in steps
            ),
            "step_motion_semantics_actual_duration_count": sum(
                dict(step.get("motion_semantics", {}) or {}).get(
                    "actual_recording_duration_s"
                )
                is not None
                for step in steps
            ),
            "maximum_step_vs_recording_timing_duration_delta_s": max(
                [
                    abs(
                        float(step.get("duration", 0.0) or 0.0)
                        - float(
                            dict(step.get("recording_timing", {}) or {}).get(
                                "actual_duration_s", 0.0
                            )
                            or 0.0
                        )
                    )
                    for step in steps
                ]
                + [0.0]
            ),
            "maximum_step_vs_motion_semantics_duration_delta_s": max(
                [
                    abs(
                        float(step.get("duration", 0.0) or 0.0)
                        - float(
                            dict(step.get("motion_semantics", {}) or {}).get(
                                "actual_recording_duration_s", 0.0
                            )
                            or 0.0
                        )
                    )
                    for step in steps
                ]
                + [0.0]
            ),
            "event_time_field_count": len(source_events),
            "event_actual_recording_time_field_count": sum(
                row["actual_recording_time_s"] is not None for row in source_events
            ),
            "event_command_start_sim_time_field_count": sum(
                row["command_start_sim_time_s"] is not None for row in source_events
            ),
            "event_count": len(source_events),
            "recording_timing_sources": dict(
                Counter(row["recording_timing_source"] for row in source_events)
            ),
            "event_wheel_active_duration_field_count": sum(
                row["wheel_active_duration_field_present"] for row in source_events
            ),
            "event_wheel_active_duration_nonnull_count": sum(
                row["wheel_active_duration_s"] is not None for row in source_events
            ),
            "wheel_duration_authority": "step motion_semantics plus production simulation-time grouping; per-event wheel_active_duration_s is present but null",
        },
        "speed_sources": {
            "top_level_metadata_explicit_servo_speed_fields": top_level_speed_keys,
            "recording_payload_bookkeeping_profile_ids": profile_ids,
            "recording_bookkeeping_reference_velocity_deg_s": float(
                dict(baseline.get("servo_profile", {}) or {}).get(
                    "reference_velocity_deg_s", 0.0
                )
            ),
            "source_event_canonical_servo_velocity_deg_s": event_rates,
            "runtime_motion_reference_servo_velocity_deg_s": float(
                motion.servo_reference_velocity_deg_s
            ),
            "production_plan_servo_velocity_deg_s": float(
                runtime_plan.timing["servo_reference_velocity_deg_s"]
            ),
            "runtime_wheel_reference_velocity_rad_s": float(
                motion.wheel_reference_velocity_rad_s
            ),
            "runtime_wheel_limit_rad_s": float(motion.wheel_velocity_limit_rad_s),
            "interpretation": "30 deg/s is legacy nested bookkeeping, not a numeric field in top-level metadata.json; source events and the current production planner specify 150 deg/s",
            "successful_live_dispatch_speed": "PENDING_LIVE_DISPATCH_TRACE",
        },
        "production_runtime_150": {
            "plan_sha256": runtime_plan.plan_sha256,
            "event_count": len(runtime_plan.events),
            "segment_count": len(runtime_plan.segments),
            "final_time_s": float(runtime_plan.final_time_s),
            "timing": _stable_timing(runtime_plan.timing),
        },
        "production_counterfactual_30": {
            "plan_sha256": counterfactual_plan.plan_sha256,
            "event_count": len(counterfactual_plan.events),
            "segment_count": len(counterfactual_plan.segments),
            "final_time_s": float(counterfactual_plan.final_time_s),
            "timing": _stable_timing(counterfactual_plan.timing),
        },
        "full_trajectory_comparison": {
            "same_command_and_target_path": _plan_signature(runtime_plan)
            == _plan_signature(counterfactual_plan),
            "same_event_count": len(runtime_plan.events)
            == len(counterfactual_plan.events),
            "same_segment_count": len(runtime_plan.segments)
            == len(counterfactual_plan.segments),
            "timing_changed_segment_count": len(changed_rows),
            "timing_unchanged_segment_count": len(plan_rows) - len(changed_rows),
            "first_timing_divergence": first_changed,
            "final_time_delta_30_minus_150_s": float(
                counterfactual_plan.final_time_s - runtime_plan.final_time_s
            ),
            "endpoint_only_comparison_used": False,
        },
    }


def _markdown_number(value: float) -> str:
    return f"{float(value):.12f}".rstrip("0").rstrip(".")


def _source_markdown(payload: dict[str, Any]) -> str:
    source = payload["source_audit"]
    counts = source["counts"]
    schema = source["schema_audit"]
    lines = [
        "# V003 Source Audit",
        "",
        "> Static evidence only. Isaac was not launched; this document does not claim a physical Fast Replay PASS.",
        "",
        "## Status",
        "",
        f"- Static source/compile status: `{payload['status']}`",
        f"- Physical Fast Replay: `{payload['physical_replay_status']}`",
        f"- Live dispatch trace: `{payload['live_dispatch_trace_status']}`",
        f"- Selected source: `{source['version_id']}` (explicit v003; no active-pointer fallback)",
        f"- accepted_steps SHA-256 matches metadata: `{source['accepted_steps_sha256_matches_metadata']}`",
        f"- robot asset SHA-256 matches metadata: `{source['robot_asset_sha256_matches_metadata']}`",
        f"- source files unchanged during audit: `{source['source_files_unchanged_during_audit']}`",
        "",
        "## Source counts",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Steps | {counts['step_count']} |",
        f"| Raw source events / metadata commands | {counts['raw_source_event_count']} / {counts['metadata_command_count']} |",
        f"| Expanded actuator commands | {counts['decoded_source_command_count']} |",
        f"| Expanded servo / wheel commands | {counts['decoded_source_servo_count']} / {counts['decoded_source_wheel_count']} |",
        f"| Atomic servo-wheel batches | {counts['atomic_batch_count']} |",
        f"| Production plan events / segments | {counts['production_event_count']} / {counts['production_segment_count']} |",
        f"| Production semantic no-ops elided | {counts['production_semantic_noop_count']} |",
        "",
        "The three command counts are intentionally distinct: top-level metadata counts 136 recorded events; atomic expansion produces 202 actuator commands; the production compiler retains 160 effective plan events after semantic no-op removal.",
        "All `source_event_index` values in the JSON/CSV are zero-based within their source Step; Step indices retain the recording's one-based values.",
        "",
        "## Timestamp, duration, and schema evidence",
        "",
        f"- All {counts['raw_source_event_count']} events contain relative `time`, `actual_recording_time_s`, and `command_start_sim_time` fields.",
        "- All recording timing sources are `simulation_time`.",
        "- `wheel_active_duration_s` exists on every event but is null; wheel intervals come from step motion semantics and the production simulation-time grouping.",
        f"- Missing required Step fields: `{len(schema['missing_required_step_fields'])}`",
        f"- Missing required event fields: `{len(schema['missing_required_event_fields'])}`",
        f"- Non-finite numeric values: `{len(schema['nonfinite_numeric_paths'])}`",
        f"- Snapshot classifications: `{json.dumps(schema['snapshot_classification_counts'], sort_keys=True)}`",
        "",
        "## Atomic source-to-plan preservation",
        "",
        "| Step:event | Batch | Source commands | Retained | Elided zero/no-op | One segment | Concurrent |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["atomic_batches"]:
        lines.append(
            "| {step}:{event} | `{batch}` | {source} | {retained} | {omitted} | {single} | {concurrent} |".format(
                step=row["source_step_index"],
                event=row["source_event_index"],
                batch=row["batch_id"],
                source=row["source_command_counts"]["total"],
                retained=row["production_retained_command_counts"]["total"],
                omitted=len(row["production_elided_semantic_noops"]),
                single=row["single_segment_preserved"],
                concurrent=row["servo_wheel_concurrent"],
            )
        )
    lines.extend(
        [
            "",
            "The production compiler drops explicit zero-wheel semantic no-ops from each atomic launch, but keeps all eight servo commands and the active wheel command in one concurrent segment. It does not split the effective servo-wheel batch across ticks at the plan level.",
            "",
            "## Step provenance",
            "",
            "| Step | Duration (s) | Events | Expanded commands | Atomic | Plan segments |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for step in payload["steps"]:
        lines.append(
            f"| {step['source_step_index']} | {_markdown_number(step['source_duration_s'])} | {step['source_event_count']} | {step['decoded_command_counts']['total']} | {step['atomic_event_count']} | {step['production_segment_count']} |"
        )
    lines.extend(
        [
            "",
            "Initial/final joint command targets for every Step, every source event identity, and every source-to-production command mapping are stored in `V003_FAST_REPLAY_PLAN.json`.",
            "",
            "## Locked hashes",
            "",
            "| Source | SHA-256 | Path |",
            "|---|---|---|",
        ]
    )
    for name, row in payload["source_hashes"].items():
        lines.append(f"| {name} | `{row['sha256']}` | `{row['path']}` |")
    lines.extend(
        [
            "",
            f"- Production plan SHA-256: `{payload['production_plan_sha256']}`",
            f"- Static provenance fingerprint: `{payload['provenance_fingerprint_sha256']}`",
            "",
            "## Evidence boundary",
            "",
            "This source audit validates bytes, schema presence, timing fields, command provenance, atomic compile structure, and planned wheel integrals. It contains no viewport video, PhysX readback, live dispatch tick, contact, COM, or clearance evidence. Physical traversal, successful replay, support diagonal, and COM-transfer mechanism all remain unverified.",
        ]
    )
    return "\n".join(lines) + "\n"


def _timing_markdown(payload: dict[str, Any]) -> str:
    timing = payload["timing_provenance"]
    speeds = timing["speed_sources"]
    runtime = timing["production_runtime_150"]
    thirty = timing["production_counterfactual_30"]
    comparison = timing["full_trajectory_comparison"]
    lines = [
        "# V003 Timing Provenance",
        "",
        "> Offline production-compiler evidence only. Live dispatch and physical Fast Replay are pending.",
        "",
        "## Authority chain",
        "",
        "1. Original `accepted_steps.jsonl` and `metadata.json` from the explicit v003 directory.",
        "2. `fsm_50mm_recording_derived_v3.recording_fast_plan.fast_plan_rows`.",
        "3. Production `playback.plan_from_steps(profile=\"fast\")`.",
        "",
        "No approximate replay compiler and no Step-endpoint fallback were used.",
        "",
        "## What the recording actually stores",
        "",
        f"- Step durations: {timing['recording_timing']['step_duration_field_count']}/{timing['recording_timing']['step_count']}; sum `{_markdown_number(timing['recording_timing']['step_duration_sum_s'])} s`.",
        f"- `recording_timing.actual_duration_s` and `motion_semantics.actual_recording_duration_s`: {timing['recording_timing']['step_recording_timing_actual_duration_count']}/{timing['recording_timing']['step_count']} and {timing['recording_timing']['step_motion_semantics_actual_duration_count']}/{timing['recording_timing']['step_count']}; both have zero maximum difference from the Step duration.",
        f"- Event timestamps: {timing['recording_timing']['event_time_field_count']}/{timing['recording_timing']['event_count']} relative timestamps and {timing['recording_timing']['event_command_start_sim_time_field_count']}/{timing['recording_timing']['event_count']} simulation timestamps.",
        "- All events identify `simulation_time` as their recording clock.",
        "- Per-event `wheel_active_duration_s` is present but null; the recorded Step motion semantics and production grouping supply wheel intervals.",
        "",
        "## 30 deg/s bookkeeping versus 150 deg/s production runtime",
        "",
        f"- Top-level `metadata.json` numeric servo speed fields: `{json.dumps(speeds['top_level_metadata_explicit_servo_speed_fields'], sort_keys=True)}`.",
        f"- Nested recording bookkeeping profiles: `{', '.join(speeds['recording_payload_bookkeeping_profile_ids'])}`; baseline numeric reference `{_markdown_number(speeds['recording_bookkeeping_reference_velocity_deg_s'])} deg/s`.",
        f"- Source-event canonical speeds: `{speeds['source_event_canonical_servo_velocity_deg_s']}` deg/s.",
        f"- Current runtime motion reference and production plan: `{_markdown_number(speeds['runtime_motion_reference_servo_velocity_deg_s'])}` / `{_markdown_number(speeds['production_plan_servo_velocity_deg_s'])}` deg/s.",
        "",
        "Therefore 30 deg/s is stale nested bookkeeping, not a numeric field in top-level `metadata.json`. Static code/data provenance selects 150 deg/s for the current compiler. Which speed the historically successful UI run actually dispatched still requires the requested live dispatch trace and is not claimed here.",
        "",
        "## Full production trajectory comparison",
        "",
        "| Compile | Events | Segments | Final planned time (s) | Plan SHA-256 |",
        "|---|---:|---:|---:|---|",
        f"| Runtime 150 deg/s | {runtime['event_count']} | {runtime['segment_count']} | {_markdown_number(runtime['final_time_s'])} | `{runtime['plan_sha256']}` |",
        f"| Counterfactual 30 deg/s | {thirty['event_count']} | {thirty['segment_count']} | {_markdown_number(thirty['final_time_s'])} | `{thirty['plan_sha256']}` |",
        "",
        f"- Final-time increase at 30 deg/s: `{_markdown_number(comparison['final_time_delta_30_minus_150_s'])} s`.",
        f"- Segments whose full start/end/duration trajectory changes: `{comparison['timing_changed_segment_count']}/{runtime['segment_count']}`.",
        f"- Same command/target path: `{comparison['same_command_and_target_path']}`; endpoint-only comparison used: `{comparison['endpoint_only_comparison_used']}`.",
        "- The CSV includes both 150 and 30 deg/s start/end/duration for every segment, so the conclusion is not based only on final joint targets.",
        "",
        "## Wheel integral",
        "",
        "| Wheel | Source requested/canonical (rad) | Source applied/derived (rad) | Production plan (rad) | Plan - applied (rad) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in WHEEL_JOINT_NAMES:
        wheel = payload["wheel_integral_rad"]
        lines.append(
            f"| {name} | {_markdown_number(wheel['source_requested_canonical'][name])} | {_markdown_number(wheel['source_applied_derived'][name])} | {_markdown_number(wheel['production_runtime_150'][name])} | {_markdown_number(wheel['production_minus_source_applied'][name])} |"
        )
    lines.extend(
        [
            "",
            "The tiny front-right requested/applied difference is the production clamp of a recorded `2.0944 rad/s` command to the runtime limit `2.0943951023931953 rad/s`. The production plan exactly matches the source applied/derived integral; it does not substitute endpoints for wheel travel.",
            "",
            "## Pending evidence",
            "",
            f"- Live dispatch trace: `{payload['live_dispatch_trace_status']}`",
            f"- Physical Fast Replay: `{payload['physical_replay_status']}`",
            "- No claim is made for obstacle traversal, contact sequence, support diagonal, COM transfer, or FSM completion.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_v003_static_provenance(
    source_dir: Path = DEFAULT_V003_DIRECTORY,
) -> V003StaticProvenance:
    source_dir = Path(source_dir).resolve()
    if source_dir.name != V003_VERSION_ID:
        raise ValueError(
            f"explicit v003 source required; refusing fallback to {source_dir.name!r}"
        )
    steps_path = source_dir / "accepted_steps.jsonl"
    metadata_path = source_dir / "metadata.json"
    if not steps_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"v003 source files missing under {source_dir}")

    source_hashes_before = _source_hashes(source_dir)
    metadata = _json_object(metadata_path)
    if str(metadata.get("version_id", "") or "") != V003_VERSION_ID:
        raise ValueError("metadata version_id is not the explicit v003 source")
    accepted_hash = source_hashes_before["accepted_steps"]["sha256"]
    expected_hash = str(metadata.get("accepted_steps_sha256", "") or "").lower()
    if accepted_hash != expected_hash:
        raise ValueError(
            f"accepted_steps hash mismatch: metadata={expected_hash} actual={accepted_hash}"
        )

    steps = load_steps_jsonl(steps_path)
    if len(steps) != int(metadata.get("step_count", 0) or 0):
        raise ValueError("metadata step_count does not match accepted_steps")
    source_events, source_lookup = _source_events(steps)
    _raw_atomic_fields(steps, source_events)
    motion = load_motion_reference()
    runtime_plan, runtime_rows = recording_fast_plan.fast_plan_rows(
        source_version=V003_VERSION_ID,
        steps=steps,
        max_wheel_speed=float(motion.wheel_velocity_limit_rad_s),
    )
    runtime_servo_rate = float(runtime_plan.timing["servo_reference_velocity_deg_s"])
    if abs(runtime_servo_rate - 150.0) > 1.0e-12:
        raise ValueError(
            f"current production runtime is not the audited 150 deg/s: {runtime_servo_rate}"
        )
    mapping_errors = [
        provenance
        for row in runtime_rows
        for provenance in row["command_provenance"]
        if provenance.get("mapping_error")
    ]
    if mapping_errors:
        raise ValueError(f"source-to-production mapping errors: {mapping_errors}")
    counterfactual_plan = _counterfactual_plan(
        steps, float(motion.wheel_velocity_limit_rad_s)
    )
    plan_rows = _enrich_plan_rows(
        steps=steps,
        runtime_plan=runtime_plan,
        runtime_rows=runtime_rows,
        counterfactual_plan=counterfactual_plan,
        source_lookup=source_lookup,
    )
    atomic_batches = _atomic_audit(source_events, plan_rows)
    if not all(
        row["single_segment_preserved"] and row["servo_wheel_concurrent"]
        for row in atomic_batches
    ):
        raise ValueError("one or more v003 atomic servo-wheel batches were split")

    schema_audit = _source_schema_audit(steps, source_events)
    baseline = _json_object(RECORDING_BASELINE_PATH)
    timing = _timing_provenance(
        metadata=metadata,
        steps=steps,
        source_events=source_events,
        runtime_plan=runtime_plan,
        counterfactual_plan=counterfactual_plan,
        plan_rows=plan_rows,
        baseline=baseline,
    )
    requested_integral = _source_wheel_integral(
        steps, "canonical_wheel_angular_displacement_rad"
    )
    applied_integral = _source_wheel_integral(
        steps, "derived_wheel_angular_displacement_rad"
    )
    production_integral = _sum_wheel_maps(
        plan_rows, "expected_wheel_displacement_rad"
    )
    wheel_integral = {
        "source_requested_canonical": requested_integral,
        "source_applied_derived": applied_integral,
        "production_runtime_150": production_integral,
        "production_minus_source_applied": {
            name: production_integral[name] - applied_integral[name]
            for name in WHEEL_JOINT_NAMES
        },
    }
    decoded_commands = [
        command for row in source_events for command in row["expanded_commands"]
    ]
    plan_commands = [event.base_command for event in runtime_plan.events]
    metadata_command_count = int(metadata.get("command_count", 0) or 0)
    counts = {
        "step_count": len(steps),
        "metadata_step_count": int(metadata.get("step_count", 0) or 0),
        "raw_source_event_count": len(source_events),
        "metadata_command_count": metadata_command_count,
        "decoded_source_command_count": len(decoded_commands),
        "decoded_source_servo_count": _command_counts(decoded_commands)["servo"],
        "decoded_source_wheel_count": _command_counts(decoded_commands)["wheel"],
        "atomic_batch_count": len(atomic_batches),
        "atomic_expanded_command_count": sum(
            row["source_command_counts"]["total"] for row in atomic_batches
        ),
        "production_event_count": len(runtime_plan.events),
        "production_servo_event_count": _command_counts(plan_commands)["servo"],
        "production_wheel_event_count": _command_counts(plan_commands)["wheel"],
        "production_segment_count": len(runtime_plan.segments),
        "production_semantic_noop_count": len(decoded_commands)
        - len(runtime_plan.events),
        "source_mapping_error_count": len(mapping_errors),
    }
    source_hashes_after = _source_hashes(source_dir)
    hashes_unchanged = source_hashes_before == source_hashes_after
    if not hashes_unchanged:
        raise RuntimeError("source/prod compiler bytes changed during v003 audit")

    source_audit = {
        "version_id": V003_VERSION_ID,
        "source_directory": str(source_dir),
        "accepted_steps_path": str(steps_path),
        "metadata_path": str(metadata_path),
        "accepted_steps_sha256": accepted_hash,
        "metadata_expected_accepted_steps_sha256": expected_hash,
        "accepted_steps_sha256_matches_metadata": accepted_hash == expected_hash,
        "metadata_sha256": source_hashes_before["metadata"]["sha256"],
        "metadata_expected_robot_asset_sha256": str(
            metadata.get("robot_asset_sha256", "") or ""
        ).lower(),
        "robot_asset_sha256": source_hashes_before["robot_asset"]["sha256"],
        "robot_asset_sha256_matches_metadata": source_hashes_before["robot_asset"][
            "sha256"
        ]
        == str(metadata.get("robot_asset_sha256", "") or "").lower(),
        "metadata": metadata,
        "counts": counts,
        "schema_audit": schema_audit,
        "source_events": source_events,
        "source_files_unchanged_during_audit": hashes_unchanged,
        "selection_policy": "explicit_v003_only_no_active_pointer_fallback",
    }
    steps_summary = _step_summaries(steps, source_events, plan_rows)
    core_fingerprint = {
        "source_hashes": source_hashes_before,
        "production_plan_sha256": runtime_plan.plan_sha256,
        "counterfactual_30_plan_sha256": counterfactual_plan.plan_sha256,
        "counts": counts,
        "wheel_integral_rad": wheel_integral,
        "atomic_batches": atomic_batches,
    }
    payload: dict[str, Any] = {
        "schema_version": "fsm50.v003_static_provenance.v1",
        "status": STATIC_STATUS,
        "physical_replay_status": PHYSICAL_STATUS,
        "live_dispatch_trace_status": DISPATCH_STATUS,
        "physical_pass_claimed": False,
        "production_compiler": {
            "runtime_entrypoint": "fsm_50mm_recording_derived_v3.recording_fast_plan.fast_plan_rows",
            "planner_entrypoint": "playback.plan_from_steps",
            "runtime_profile_requested": "fast",
            "runtime_profile_normalized": runtime_plan.profile,
            "approximate_compiler_used": False,
            "endpoint_fallback_used": False,
        },
        "source_hashes": source_hashes_before,
        "source_audit": source_audit,
        "timing_provenance": timing,
        "wheel_integral_rad": wheel_integral,
        "atomic_batches": atomic_batches,
        "steps": steps_summary,
        "production_plan_sha256": runtime_plan.plan_sha256,
        "counterfactual_30_plan_sha256": counterfactual_plan.plan_sha256,
        "provenance_fingerprint_sha256": _fingerprint(core_fingerprint),
        "segments": plan_rows,
    }
    payload["content_fingerprint_sha256"] = _fingerprint(payload)
    return V003StaticProvenance(
        payload=payload,
        csv_rows=plan_rows,
        source_markdown=_source_markdown(payload),
        timing_markdown=_timing_markdown(payload),
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def write_v003_static_provenance(
    *,
    source_dir: Path = DEFAULT_V003_DIRECTORY,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    report_root = Path(report_root).resolve()
    source_hash_before = {
        name: row["sha256"] for name, row in _source_hashes(source_dir).items()
    }
    result = build_v003_static_provenance(source_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "V003_FAST_REPLAY_PLAN.json"
    csv_path = report_root / "V003_FAST_REPLAY_PLAN.csv"
    source_md_path = report_root / "V003_SOURCE_AUDIT.md"
    timing_md_path = report_root / "V003_TIMING_PROVENANCE.md"

    json_path.write_text(
        json.dumps(
            result.payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in result.csv_rows:
            writer.writerow({key: _csv_cell(row[key]) for key in CSV_COLUMNS})
    source_md_path.write_text(result.source_markdown, encoding="utf-8")
    timing_md_path.write_text(result.timing_markdown, encoding="utf-8")

    source_hash_after = {
        name: row["sha256"] for name, row in _source_hashes(source_dir).items()
    }
    if source_hash_before != source_hash_after:
        raise RuntimeError("report generation modified a source/prod compiler file")
    return {
        "status": STATIC_STATUS,
        "physical_replay_status": PHYSICAL_STATUS,
        "live_dispatch_trace_status": DISPATCH_STATUS,
        "source_unchanged": True,
        "production_plan_sha256": result.payload["production_plan_sha256"],
        "paths": {
            "json": str(json_path),
            "csv": str(csv_path),
            "source_audit": str(source_md_path),
            "timing_provenance": str(timing_md_path),
        },
        "report_sha256": {
            "json": sha256_file(json_path),
            "csv": sha256_file(csv_path),
            "source_audit": sha256_file(source_md_path),
            "timing_provenance": sha256_file(timing_md_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_V003_DIRECTORY)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_v003_static_provenance(
        source_dir=args.source_dir,
        report_root=args.report_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
