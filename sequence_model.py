"""Accepted-step JSONL helpers compatible with real_robot_ui_controller."""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from command_model import (
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_SHORT_NAMES,
    clamp_servo_command,
    is_float_token,
    resolve_servo_targets_for_command,
    resolve_servo_targets_for_group_part,
    resolve_wheel_name,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def empty_command_state() -> dict[str, dict[str, float]]:
    return {
        "servos": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES},
    }


def clone_command_state(state: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    cloned = empty_command_state()
    if not isinstance(state, dict):
        return cloned
    servos = state.get("servos", {})
    wheels = state.get("wheels", {})
    if isinstance(servos, dict):
        for name in SERVO_JOINT_NAMES:
            if name in servos:
                cloned["servos"][name] = float(servos[name])
    if isinstance(wheels, dict):
        for name in WHEEL_JOINT_NAMES:
            if name in wheels:
                cloned["wheels"][name] = float(wheels[name])
        for short_name, full_name in WHEEL_SHORT_NAMES.items():
            if short_name in wheels:
                cloned["wheels"][full_name] = float(wheels[short_name])
    return cloned


def make_event(
    time_s: float,
    command: str,
    *,
    kind: str = "command",
    command_state_before: dict[str, Any] | None = None,
    command_state_after: dict[str, Any] | None = None,
    expanded_commands: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "time": max(0.0, float(time_s)),
        "command": str(command).strip(),
        "kind": kind,
        "expanded_commands": list(expanded_commands or []),
        "command_state_before": clone_command_state(command_state_before),
        "command_state_after": clone_command_state(command_state_after),
    }


def normalize_event(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return make_event(0.0, raw)
    if not isinstance(raw, dict):
        return make_event(0.0, str(raw))
    expanded = raw.get("expanded_commands", raw.get("playback_commands", []))
    if isinstance(expanded, str):
        expanded = [expanded]
    if not isinstance(expanded, list):
        expanded = []
    event = make_event(
        float(raw.get("time", raw.get("time_s", raw.get("relative_time", raw.get("t", 0.0))))),
        str(raw.get("command", raw.get("text", ""))).strip(),
        kind=str(raw.get("kind", raw.get("source", "command"))),
        command_state_before=raw.get("command_state_before") or raw.get("before_state"),
        command_state_after=raw.get("command_state_after") or raw.get("after_state") or raw.get("state"),
        expanded_commands=[str(item).strip() for item in expanded if str(item).strip()],
    )
    # Unknown legacy metadata remains opaque during normalization.  Playback
    # derives actuator commands only from command/event values and timestamps.
    for key, value in raw.items():
        if key not in event:
            event[key] = copy.deepcopy(value)
    return event


def normalize_events(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    return sorted((normalize_event(event) for event in events), key=lambda event: float(event.get("time", 0.0)))


def make_step(
    *,
    index: int,
    step_type: str,
    duration: float,
    events: list[dict[str, Any]],
    command_state_before: dict[str, Any] | None,
    command_state_after: dict[str, Any] | None,
    name: str | None = None,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created = now_iso()
    duration_s = max(0.0, float(duration))
    step_name = name or f"step_{int(index):03d}_{step_type}_dur_{duration_s:.2f}s"
    step: dict[str, Any] = {
        "index": int(index),
        "name": step_name,
        "type": str(step_type),
        "duration": duration_s,
        "note": str(note),
        "events": normalize_events(events),
        "command_state_before": clone_command_state(command_state_before),
        "command_state_after": clone_command_state(command_state_after),
        "created_at": created,
        "updated_at": created,
    }
    if extra:
        step.update(copy.deepcopy(extra))
    return step


def normalize_step(raw: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    source = dict(raw or {})
    events = normalize_events(source.get("events", source.get("commands", [])))
    derived_duration = max([float(event.get("time", 0.0)) for event in events] + [0.0])
    duration = float(source.get("duration", source.get("duration_s", derived_duration)))
    step_index = int(source.get("index", index if index is not None else 0))
    before = (
        source.get("command_state_before")
        or source.get("state_before")
        or source.get("robot_state_before")
        or empty_command_state()
    )
    after = (
        source.get("command_state_after")
        or source.get("state_after")
        or source.get("robot_state_after")
        or apply_events_to_state(before, events)
    )
    step = {
        "index": step_index,
        "name": str(source.get("name") or f"step_{step_index:03d}"),
        "type": str(source.get("type") or source.get("kind") or "recorded"),
        "duration": max(duration, derived_duration),
        "note": str(source.get("note", "")),
        "events": events,
        "command_state_before": clone_command_state(before),
        "command_state_after": clone_command_state(after),
        "created_at": str(source.get("created_at") or now_iso()),
        "updated_at": str(source.get("updated_at") or source.get("created_at") or now_iso()),
    }
    for key, value in source.items():
        if key not in step:
            step[key] = copy.deepcopy(value)
    return step


def reindex_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reindexed: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        normalized = normalize_step(step, index=index)
        normalized["index"] = index
        reindexed.append(normalized)
    return reindexed


def rebuild_sequence_continuity(
    steps: list[dict[str, Any]],
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = reindex_steps(copy.deepcopy(steps))
    if not normalized:
        return [], {"ok": True, "start_index": 1, "steps_rebuilt": 0, "errors": []}
    start = max(1, min(int(start_index), len(normalized)))
    if start > 1:
        state = clone_command_state(normalized[start - 2].get("command_state_after"))
    else:
        state = clone_command_state(normalized[0].get("command_state_before"))

    rebuilt: list[dict[str, Any]] = []
    for index, step in enumerate(normalized, start=1):
        if index < start:
            rebuilt.append(step)
            continue
        current = copy.deepcopy(step)
        current["index"] = index
        current["command_state_before"] = clone_command_state(state)
        rebuilt_events: list[dict[str, Any]] = []
        for event in normalize_events(current.get("events", [])):
            event_before = clone_command_state(state)
            for command in event_playback_commands(event):
                apply_command_to_state(state, command)
            event_after = clone_command_state(state)
            rebuilt_event = normalize_event(event)
            rebuilt_event["command_state_before"] = event_before
            rebuilt_event["command_state_after"] = event_after
            rebuilt_events.append(rebuilt_event)
        current["events"] = rebuilt_events
        current["command_state_after"] = clone_command_state(state)
        current["duration"] = max(float(current.get("duration", 0.0)), _events_duration(rebuilt_events))
        current["updated_at"] = now_iso()
        rebuilt.append(normalize_step(current, index=index))

    errors: list[str] = []
    for left, right in zip(rebuilt, rebuilt[1:]):
        if clone_command_state(left.get("command_state_after")) != clone_command_state(right.get("command_state_before")):
            errors.append(f"continuity mismatch between step {left.get('index')} and {right.get('index')}")
    return rebuilt, {
        "ok": not errors,
        "start_index": start,
        "steps_rebuilt": len(rebuilt) - start + 1,
        "errors": errors,
    }


def event_playback_commands(event: dict[str, Any]) -> list[str]:
    expanded = event.get("expanded_commands") or event.get("playback_commands") or []
    if isinstance(expanded, str):
        expanded = [expanded]
    if isinstance(expanded, list) and expanded:
        return [str(command).strip() for command in expanded if str(command).strip()]
    command = str(event.get("command", "")).strip()
    return [command] if command else []


def is_record_marker(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return bool(tokens and tokens[0].lower() in {"record_start", "record_stop"})


def motion_only_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [normalize_event(event) for event in events if not is_record_marker(str(event.get("command", "")))]
    if not usable:
        return []
    first_time = min(float(event.get("time", 0.0)) for event in usable)
    shifted: list[dict[str, Any]] = []
    for event in usable:
        moved = copy.deepcopy(event)
        moved["time"] = max(0.0, float(moved.get("time", 0.0)) - first_time)
        shifted.append(moved)
    return sorted(shifted, key=lambda event: float(event.get("time", 0.0)))


def semantic_motion_groups(steps: list[dict[str, Any]], *, epsilon: float = 1.0e-6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compact a sequence by actuator semantics, not recorded UI timestamps.

    Non-zero wheel state preserves elapsed time because it produces angular
    displacement.  Consecutive targets for the same servo inside one recorded
    step preserve their trajectory interval.  Other implicit gaps are removed;
    explicit wait/hold/sleep commands are retained.  Step boundaries remain
    metadata and never introduce a scheduling delay.
    """

    normalized_steps = [normalize_step(step) for step in steps]
    if not normalized_steps:
        return [], []
    state = clone_command_state(normalized_steps[0].get("command_state_before"))
    raw_cursor = 0.0
    raw_groups: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for normalized in normalized_steps:
        step_index = int(normalized.get("index", 0))
        step_id = str(normalized.get("name", "") or f"step_{step_index:03d}")
        events = normalize_events(normalized.get("events", []))
        first_raw = min([float(event.get("time", 0.0)) for event in events] + [0.0])
        last_raw = max([float(event.get("time", 0.0)) for event in events] + [0.0])
        diagnostics.append(
            {
                "step_index": step_index,
                "step_id": step_id,
                "raw_first_event_s": raw_cursor + first_raw,
                "raw_last_event_s": raw_cursor + last_raw,
                "raw_duration_s": max(float(normalized.get("duration", 0.0)), last_raw),
                "first_motion_time_s": None,
                "last_motion_time_s": None,
                "implicit_ui_gap_removed_s": 0.0,
                "explicit_motion_duration_s": 0.0,
                "inter_step_scheduler_gap_s": 0.0,
                "kept_command_count": 0,
                "removed_noop_count": 0,
            }
        )
        for event in events:
            if is_record_marker(str(event.get("command", ""))):
                continue
            commands = event_playback_commands(event) or [str(event.get("command", ""))]
            usable = [str(command).strip() for command in commands if str(command).strip()]
            if usable:
                raw_groups.append(
                    {
                        "raw_time": raw_cursor + max(0.0, float(event.get("time", 0.0))),
                        "source_step": step_index,
                        "source_step_id": step_id,
                        "commands": usable,
                        "event_kind": str(event.get("kind", "command") or "command").lower(),
                    }
                )
        raw_cursor += max(float(normalized.get("duration", 0.0)), last_raw)

    diag_by_step = {int(row["step_index"]): row for row in diagnostics}
    compacted: list[dict[str, Any]] = []
    fast_cursor = 0.0
    previous_kept_raw_time: float | None = None
    previous_servo_keys: set[str] = set()
    previous_source_step: int | None = None
    explicit_wait_due = 0.0
    for raw_group in sorted(raw_groups, key=lambda row: float(row["raw_time"])):
        meaningful: list[str] = []
        changed_servo_keys: set[str] = set()
        wait_duration = 0.0
        timing_boundary = False
        row_diag = diag_by_step[int(raw_group["source_step"])]
        wheels_active_before = any(abs(float(value)) > epsilon for value in state["wheels"].values())
        for command in raw_group["commands"]:
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = []
            verb = tokens[0].lower() if tokens else ""
            if verb in {"wait", "hold", "sleep"}:
                meaningful.append(command)
                timing_boundary = True
                try:
                    wait_duration = max(wait_duration, max(0.0, float(tokens[1])))
                except (IndexError, TypeError, ValueError):
                    pass
                continue
            before = clone_command_state(state)
            apply_command_to_state(state, command)
            servo_changes = {
                name
                for name in SERVO_JOINT_NAMES
                if abs(float(state["servos"][name]) - float(before["servos"][name])) > epsilon
            }
            wheel_changes = {
                name
                for name in WHEEL_JOINT_NAMES
                if abs(float(state["wheels"][name]) - float(before["wheels"][name])) > epsilon
            }
            known_actuator_command = verb in {"servo", "angle", "wheel", "wheels", "speed", "w", "s", "a", "d", "x", "stop", "home"}
            # A repeated non-zero wheel command is a semantic step/progress
            # boundary inside one continuous run.  Keep it in the plan so the
            # worker can advance step metadata, but the planner marks it as a
            # no-dispatch continuation (no stop/restart).
            repeated_active_wheel_boundary = bool(
                verb in {"wheel", "wheels", "speed", "w", "s", "a", "d"}
                and wheels_active_before
                and not wheel_changes
            )
            recorded_servo_waypoint = verb in {"servo", "angle"}
            if servo_changes or recorded_servo_waypoint or wheel_changes or repeated_active_wheel_boundary or not known_actuator_command:
                meaningful.append(command)
                changed_servo_keys.update(servo_changes)
            else:
                row_diag["removed_noop_count"] += 1
        if not meaningful:
            continue

        raw_time = float(raw_group["raw_time"])
        raw_gap = 0.0 if previous_kept_raw_time is None else max(0.0, raw_time - previous_kept_raw_time)
        same_step_servo_trajectory = bool(
            (changed_servo_keys.intersection(previous_servo_keys) or (timing_boundary and previous_servo_keys))
            and previous_source_step == int(raw_group["source_step"])
        )
        if previous_kept_raw_time is not None:
            if wheels_active_before or same_step_servo_trajectory:
                kept_gap = raw_gap
            else:
                kept_gap = explicit_wait_due
            fast_cursor += kept_gap
            row_diag["implicit_ui_gap_removed_s"] += max(0.0, raw_gap - kept_gap)
            row_diag["explicit_motion_duration_s"] += kept_gap
        compacted.append(
            {
                "base_time": fast_cursor,
                "raw_time": raw_time,
                "source_step": int(raw_group["source_step"]),
                "source_step_id": str(raw_group["source_step_id"]),
                "commands": meaningful,
            }
        )
        row_diag["kept_command_count"] += len(meaningful)
        if row_diag["first_motion_time_s"] is None:
            row_diag["first_motion_time_s"] = fast_cursor
        row_diag["last_motion_time_s"] = fast_cursor
        previous_kept_raw_time = raw_time
        previous_servo_keys = changed_servo_keys
        previous_source_step = int(raw_group["source_step"])
        explicit_wait_due = wait_duration

    if explicit_wait_due > 0.0:
        fast_cursor += explicit_wait_due
    for left, right in zip(compacted, compacted[1:]):
        if int(left["source_step"]) != int(right["source_step"]):
            diag_by_step[int(right["source_step"])]["inter_step_scheduler_gap_s"] = max(
                0.0, float(right["base_time"]) - float(left["base_time"])
            )
    for row in diagnostics:
        if row["first_motion_time_s"] is None:
            row["first_motion_time_s"] = 0.0
            row["last_motion_time_s"] = 0.0
        row["fast_sequence_end_s"] = fast_cursor
    return compacted, diagnostics


def accepted_rows(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in reindex_steps(steps):
        rows.append(
            {
                "index": int(step["index"]),
                "name": str(step.get("name", "")),
                "type": str(step.get("type", "")),
                "duration": float(step.get("duration", 0.0)),
                "events_count": len(step.get("events", [])),
                "note": str(step.get("note", "")),
            }
        )
    return rows


def step_summary(step: dict[str, Any]) -> str:
    normalized = normalize_step(step)
    return (
        f"{int(normalized['index']):03d} {normalized['name']} "
        f"type={normalized['type']} duration={float(normalized['duration']):.3f}s "
        f"events={len(normalized['events'])} note={normalized['note']}"
    )


def load_steps_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    steps: list[dict[str, Any]] = []
    if not source.exists():
        return steps
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{source}:{line_number}: each JSONL line must be an object")
        steps.append(normalize_step(loaded, index=len(steps) + 1))
    return reindex_steps(steps)


def save_steps_jsonl(steps: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for step in reindex_steps(steps):
            stream.write(json.dumps(step, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_save_steps_jsonl(steps: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    """Atomically save and validate steps, preserving a timestamped backup."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = reindex_steps(steps)
    token = uuid.uuid4().hex
    tmp_path = target.with_name(f".{target.name}.{token}.tmp")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = target.with_name(f"{target.name}.backup_{timestamp}") if target.exists() else None
    original_existed = target.exists()
    replaced = False
    try:
        save_steps_jsonl(normalized, tmp_path)
        loaded_tmp = load_steps_jsonl(tmp_path) if tmp_path.stat().st_size > 0 else []
        _validate_saved_steps(normalized, loaded_tmp, tmp_path)
        if backup_path is not None:
            shutil.copy2(target, backup_path)
            with backup_path.open("r+b") as backup_stream:
                os.fsync(backup_stream.fileno())
        os.replace(tmp_path, target)
        replaced = True
        loaded_final = load_steps_jsonl(target) if target.stat().st_size > 0 else []
        _validate_saved_steps(normalized, loaded_final, target)
        return {
            "path": str(target),
            "backup_path": str(backup_path) if backup_path is not None else "",
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step_count": len(normalized),
            "command_count": sum(len(step.get("events", []) or []) for step in normalized),
            "atomic_replace": True,
            "fsync": True,
            "validated": True,
        }
    except Exception:
        if replaced:
            if backup_path is not None and backup_path.exists():
                shutil.copy2(backup_path, target)
                with target.open("r+b") as restored_stream:
                    os.fsync(restored_stream.fileno())
            elif not original_existed and target.exists():
                target.unlink()
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _validate_saved_steps(expected: list[dict[str, Any]], loaded: list[dict[str, Any]], source: Path) -> None:
    if len(loaded) != len(expected):
        raise IOError(f"Saved step count mismatch for {source}: wrote {len(expected)}, reloaded {len(loaded)}")
    required = {"index", "name", "type", "duration", "events", "command_state_before", "command_state_after"}
    for index, step in enumerate(loaded, start=1):
        missing = sorted(required.difference(step))
        if missing:
            raise IOError(f"Saved step {index} in {source} is missing required fields: {missing}")
        if int(step.get("index", 0)) != index or not isinstance(step.get("events"), list):
            raise IOError(f"Saved step {index} in {source} failed structural validation")


def apply_events_to_state(
    command_state_before: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    state = clone_command_state(command_state_before)
    for event in normalize_events(events):
        commands = event_playback_commands(event)
        if not commands:
            commands = [str(event.get("command", ""))]
        for command in commands:
            apply_command_to_state(state, command)
    return state


def coalesce_record_events(
    events: list[dict[str, Any]],
    *,
    min_interval_s: float = 0.05,
    max_events: int = 2000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = sorted(normalize_events(events), key=lambda event: float(event.get("time", 0.0)))
    if not normalized:
        return [], {"original_count": 0, "coalesced_count": 0, "coalesced": False, "dropped_count": 0}

    output: list[dict[str, Any]] = []
    first_index_for_key: dict[str, int] = {}
    last_index_for_key: dict[str, int] = {}
    last_time_for_key: dict[str, float] = {}
    min_interval = max(0.0, float(min_interval_s))

    for event in normalized:
        keys = sorted(_event_control_keys(event)) or [f"event:{len(output)}"]
        key = "|".join(keys)
        event_time = float(event.get("time", 0.0))
        if key not in last_index_for_key:
            first_index_for_key[key] = len(output)
            last_index_for_key[key] = len(output)
            last_time_for_key[key] = event_time
            output.append(event)
            continue
        last_index = last_index_for_key[key]
        should_append = event_time - last_time_for_key[key] >= min_interval
        if should_append or last_index == first_index_for_key[key]:
            last_index_for_key[key] = len(output)
            last_time_for_key[key] = event_time
            output.append(event)
        else:
            output[last_index] = event

    capped = output
    dropped = 0
    max_count = max(1, int(max_events))
    if len(capped) > max_count:
        required_indices: set[int] = {0, len(output) - 1}
        first_by_key: dict[str, int] = {}
        last_by_key: dict[str, int] = {}
        for index, event in enumerate(output):
            keys = sorted(_event_control_keys(event)) or [f"event:{index}"]
            key = "|".join(keys)
            first_by_key.setdefault(key, index)
            last_by_key[key] = index
        required_indices.update(first_by_key.values())
        required_indices.update(last_by_key.values())
        if len(required_indices) > max_count:
            keep = sorted(required_indices)[: max_count - 1]
            keep.append(len(output) - 1)
            required_indices = set(keep)
        for index in range(len(output)):
            if len(required_indices) >= max_count:
                break
            required_indices.add(index)
        capped = [output[index] for index in sorted(required_indices)]
        dropped = len(output) - len(capped)

    stats = {
        "original_count": len(normalized),
        "coalesced_count": len(capped),
        "coalesced": len(capped) != len(normalized),
        "dropped_count": dropped,
        "min_interval_s": min_interval,
        "max_events": max_count,
    }
    return capped, stats


def apply_command_to_state(state: dict[str, dict[str, float]], command: str) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    if not tokens:
        return
    verb = tokens[0].lower()
    try:
        if verb in {"servo", "angle"}:
            _apply_servo_tokens(state, tokens)
        elif verb in {"wheel", "wheels", "speed"}:
            _apply_wheel_tokens(state, tokens)
        elif verb == "w":
            _set_all_wheels(state, 1.0)
        elif verb == "s":
            _set_all_wheels(state, -1.0)
        elif verb == "a":
            _set_left_right_wheels(state, -1.0, 1.0)
        elif verb == "d":
            _set_left_right_wheels(state, 1.0, -1.0)
        elif verb in {"x", "stop"}:
            _set_all_wheels(state, 0.0)
        elif verb == "home":
            for name in SERVO_JOINT_NAMES:
                state["servos"][name] = 0.0
            for name in WHEEL_JOINT_NAMES:
                state["wheels"][name] = 0.0
    except Exception:
        return


def build_combined_step(
    steps: list[dict[str, Any]],
    *,
    allow_conflicts: bool = False,
    name: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    if len(steps) < 2:
        raise ValueError("Combine requires at least two steps.")
    normalized_steps = [normalize_step(step) for step in steps]
    first_state = normalized_steps[0].get("command_state_before")
    merged: list[tuple[float, int, dict[str, Any], set[str]]] = []
    conflicts: list[str] = []
    for order, step in enumerate(normalized_steps):
        for event in motion_only_events(step.get("events", [])):
            time_key = round(float(event.get("time", 0.0)), 3)
            keys = _event_control_keys(event)
            overlapping = [
                existing
                for existing in merged
                if round(existing[0], 3) == time_key and existing[3].intersection(keys)
            ]
            if overlapping and not allow_conflicts:
                control_text = ", ".join(sorted(set().union(*(item[3].intersection(keys) for item in overlapping))))
                conflicts.append(f"t={time_key:.3f}: {control_text}")
                continue
            if overlapping and allow_conflicts:
                merged = [
                    item
                    for item in merged
                    if not (round(item[0], 3) == time_key and item[3].intersection(keys))
                ]
            merged.append((float(event.get("time", 0.0)), order, copy.deepcopy(event), keys))
    if conflicts:
        raise ValueError("Combine conflict detected: " + "; ".join(conflicts))
    merged.sort(key=lambda item: (item[0], item[1]))
    events = [item[2] for item in merged]
    duration = max([float(event.get("time", 0.0)) for event in events] + [0.0])
    after = apply_events_to_state(first_state, events)
    source_indices = [int(step.get("index", 0)) for step in normalized_steps]
    return make_step(
        index=min(source_indices) if source_indices else 1,
        step_type="combined",
        duration=duration,
        events=events,
        command_state_before=first_state,
        command_state_after=after,
        name=name or f"step_{min(source_indices):03d}_combined_{len(normalized_steps)}_steps_dur_{duration:.2f}s",
        note=note,
        extra={
            "combined_from": source_indices,
            "combine_source_count": len(normalized_steps),
            "allow_conflicts": bool(allow_conflicts),
        },
    )


def _apply_servo_tokens(state: dict[str, dict[str, float]], tokens: list[str]) -> None:
    if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
        targets = resolve_servo_targets_for_group_part(tokens[1], tokens[2])
        value = float(tokens[3])
    elif len(tokens) == 3:
        targets = resolve_servo_targets_for_command(tokens[1])
        value = float(tokens[2])
    else:
        return
    for target in targets:
        state["servos"][target] = clamp_servo_command(target, value)


def _apply_wheel_tokens(state: dict[str, dict[str, float]], tokens: list[str]) -> None:
    verb = tokens[0].lower()
    args = tokens[1:]
    if verb in {"wheels", "speed"} or (verb == "wheel" and len(args) == 2 and is_float_token(args[0])):
        if len(args) != 2:
            return
        _set_left_right_wheels(state, float(args[0]), float(args[1]))
        return
    if verb != "wheel" or not args:
        return
    sub = args[0].lower()
    if sub == "stop":
        _set_all_wheels(state, 0.0)
    elif sub == "all" and len(args) == 2:
        _set_all_wheels(state, float(args[1]))
    elif len(args) == 2:
        state["wheels"][resolve_wheel_name(sub)] = float(args[1])


def _set_all_wheels(state: dict[str, dict[str, float]], value: float) -> None:
    for name in WHEEL_JOINT_NAMES:
        state["wheels"][name] = float(value)


def _set_left_right_wheels(state: dict[str, dict[str, float]], left: float, right: float) -> None:
    state["wheels"]["front_left_ankle"] = float(left)
    state["wheels"]["rear_left_ankle"] = float(left)
    state["wheels"]["front_right_ankle"] = float(right)
    state["wheels"]["rear_right_ankle"] = float(right)


def _event_control_keys(event: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for command in event_playback_commands(event):
        keys.update(_command_control_keys(command))
    return keys


def _command_control_keys(command: str) -> set[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {f"command:{command}"}
    if not tokens:
        return set()
    verb = tokens[0].lower()
    try:
        if verb in {"servo", "angle"}:
            return {f"servo:{name}" for name in _servo_targets_from_tokens(tokens)}
        if verb in {"wheel", "wheels", "speed", "w", "s", "a", "d", "x", "stop"}:
            return {f"wheel:{name}" for name in WHEEL_JOINT_NAMES}
        if verb == "home":
            return {f"servo:{name}" for name in SERVO_JOINT_NAMES} | {f"wheel:{name}" for name in WHEEL_JOINT_NAMES}
    except Exception:
        pass
    return {f"command:{' '.join(tokens)}"}


def _servo_targets_from_tokens(tokens: list[str]) -> list[str]:
    if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
        return resolve_servo_targets_for_group_part(tokens[1], tokens[2])
    if len(tokens) == 3:
        return resolve_servo_targets_for_command(tokens[1])
    return []


def _events_duration(events: list[dict[str, Any]]) -> float:
    return max([float(event.get("time", 0.0)) for event in events] + [0.0])


def format_step_json(step: dict[str, Any]) -> str:
    return json.dumps(normalize_step(step), indent=2, ensure_ascii=False, sort_keys=True)


def parse_step_header(line: str) -> dict[str, Any]:
    match = re.match(r"#\s*step\s+(\d+)\s*(.*)$", line)
    if not match:
        return {}
    result: dict[str, Any] = {"index": int(match.group(1))}
    rest = match.group(2)
    for key in ("name", "type", "duration", "note"):
        key_match = re.search(rf"\b{key}=([^=]*?)(?=\s+\w+=|$)", rest)
        if key_match:
            value = key_match.group(1).strip()
            result[key] = float(value) if key == "duration" else ("" if value == "-" else value)
    return result
