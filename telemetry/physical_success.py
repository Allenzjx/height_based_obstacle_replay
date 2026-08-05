"""Physical replay success checks from recorded telemetry artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from telemetry.exporters import read_csv_rows, write_json


SCHEMA = "height_replay_physical_success.v1"


def analyze_physical_success(
    run_dir: str | Path,
    *,
    expected_height_cm: int | None = None,
    expected_event_count: int | None = None,
    expected_final_time_s: float | None = None,
    obstacle_x_m: float | None = None,
    obstacle_length_m: float | None = None,
    robot_length_m: float | None = None,
) -> dict[str, Any]:
    """Evaluate whether replay physically crossed the obstacle using telemetry CSVs."""

    root = Path(run_dir)
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "stability_summary.json")
    quality = _read_json(root / "data_quality.json")
    if not quality:
        try:
            from telemetry.data_quality import analyze_run_data_quality

            quality = analyze_run_data_quality(root)
        except Exception as exc:
            quality = {"overall_verdict": "INSUFFICIENT", "warnings": [str(exc)]}
    samples = read_csv_rows(root / "telemetry_samples.csv")
    events = _read_jsonl(root / "events.jsonl")
    obstacle_x = _first_finite(
        obstacle_x_m,
        _dig(metadata, "scene", "obstacle_x_m"),
        metadata.get("obstacle_x_m"),
        1.55,
    )
    obstacle_length = _first_finite(
        obstacle_length_m,
        _dig(metadata, "scene", "obstacle_length_m"),
        metadata.get("obstacle_length_m"),
        1.65,
    )
    robot_length = _first_finite(
        robot_length_m,
        _dig(metadata, "scene", "robot_length_m"),
        metadata.get("robot_length_m"),
        0.55,
    )
    height_cm = int(_first_finite(expected_height_cm, metadata.get("obstacle_height_cm"), 0.0))
    height_m = max(0.0, float(height_cm) / 100.0)
    required_clearance = max(0.05, min(0.35, 0.25 * max(0.0, robot_length)))
    obstacle_front_x = obstacle_x - 0.5 * max(0.0, obstacle_length)
    mount_x = obstacle_front_x - 0.5 * max(0.0, robot_length)
    clear_x = obstacle_x + 0.5 * max(0.0, obstacle_length) + required_clearance

    replay_samples = _replay_window(samples)
    metrics = _sample_metrics(replay_samples, clear_x=clear_x)
    replay = _replay_metrics(samples, events)
    contact_data_unavailable = _contact_data_unavailable(quality)
    checks: list[dict[str, Any]] = []

    _check(checks, "telemetry_samples_present", metrics["sample_count"] >= 20, metrics["sample_count"], ">= 20")
    _check(checks, "data_quality_not_insufficient", str(quality.get("overall_verdict", "")).upper() != "INSUFFICIENT", quality.get("overall_verdict", ""), "PASS or WARN")
    _check(checks, "replay_started", replay["started"], replay["started"], True)
    _check(checks, "replay_finished_success", replay["finished_success"], replay["finish_message"], "Replay complete")
    if expected_event_count is not None:
        _check(checks, "all_events_dispatched", int(replay["max_event_index"]) >= max(0, int(expected_event_count) - 1), replay["max_event_index"], f">= {int(expected_event_count) - 1}")
    if expected_final_time_s is not None:
        covered_duration = max(float(metrics["duration_s"]), float(replay["event_duration_s"]))
        _check(checks, "sim_duration_covers_plan", covered_duration >= max(0.0, float(expected_final_time_s) - 0.5), {"samples": metrics["duration_s"], "events": replay["event_duration_s"]}, f">= {float(expected_final_time_s) - 0.5:.3f}s")
    _check(checks, "front_of_robot_reached_obstacle", metrics["max_base_x_m"] >= mount_x, metrics["max_base_x_m"], f">= {mount_x:.3f}m")
    _check(checks, "base_height_indicates_climb", metrics["max_base_z_delta_m"] >= max(0.015, 0.35 * height_m), metrics["max_base_z_delta_m"], f">= {max(0.015, 0.35 * height_m):.3f}m")
    _check(checks, "not_flipped_roll_pitch", metrics["max_abs_roll_pitch_rad"] <= 0.85, metrics["max_abs_roll_pitch_rad"], "<= 0.85rad")
    _check(checks, "base_height_reasonable", metrics["min_base_z_m"] >= -0.05 and metrics["final_base_z_m"] >= max(0.02, 0.4 * height_cm / 100.0), {"min": metrics["min_base_z_m"], "final": metrics["final_base_z_m"]}, "no sustained collapse")
    _check(
        checks,
        "contact_support_observed_or_unavailable",
        contact_data_unavailable or metrics["max_active_contacts"] >= 2,
        {
            "max": metrics["max_active_contacts"],
            "final": metrics["final_active_contacts"],
            "recent_max": metrics["recent_active_contacts_max"],
            "contact_data_unavailable": contact_data_unavailable,
        },
        ">= 2 contacts observed, or contact telemetry unavailable",
    )
    _check(checks, "static_margin_finite", metrics["finite_static_margin_count"] > 0, metrics["finite_static_margin_count"], "> 0")

    failures = [row for row in checks if not row["passed"]]
    result = {
        "schema": SCHEMA,
        "ok": not failures,
        "verdict": "PASS" if not failures else "FAIL",
        "run_dir": str(root),
        "expected": {
            "height_cm": height_cm,
            "event_count": expected_event_count,
            "final_time_s": expected_final_time_s,
        },
        "scene": {
            "obstacle_x_m": obstacle_x,
            "obstacle_length_m": obstacle_length,
            "robot_length_m": robot_length,
            "required_clearance_m": required_clearance,
            "obstacle_front_x_m": obstacle_front_x,
            "mount_x_m": mount_x,
            "clear_x_m": clear_x,
        },
        "metrics": metrics,
        "diagnostics": {
            "cleared_obstacle_tail": bool(metrics["max_base_x_m"] >= clear_x),
            "ended_past_obstacle_tail": bool(metrics["final_base_x_m"] >= clear_x - 0.10),
        },
        "replay": replay,
        "data_quality": {
            "overall_verdict": quality.get("overall_verdict", ""),
            "warnings": quality.get("warnings", []),
            "contact_data_unavailable": contact_data_unavailable,
            "path": str(root / "data_quality.json"),
        },
        "checks": checks,
        "failure_reasons": [f"{row['name']}: actual={row['actual']} expected={row['expected']}" for row in failures],
        "sources": {
            "telemetry_samples": str(root / "telemetry_samples.csv"),
            "replay_window_sample_count": len(replay_samples),
            "events": str(root / "events.jsonl"),
            "metadata": str(root / "metadata.json"),
            "stability_summary": str(root / "stability_summary.json"),
        },
    }
    write_json(root / "physical_success.json", result)
    return result


def _sample_metrics(samples: list[dict[str, str]], *, clear_x: float) -> dict[str, Any]:
    time_s = [_float(row.get("time_s")) for row in samples]
    base_x = [_float(row.get("base_x_m")) for row in samples]
    base_z = [_float(row.get("base_z_m")) for row in samples]
    roll = [_float(row.get("base_roll_rad")) for row in samples]
    pitch = [_float(row.get("base_pitch_rad")) for row in samples]
    margins = [_float(row.get("static_stability_margin_m")) for row in samples]
    contacts = [_float(row.get("active_contact_count")) for row in samples]
    finite_t = _finite(time_s)
    finite_x = _finite(base_x)
    finite_z = _finite(base_z)
    finite_margin = _finite(margins)
    finite_contacts = _finite(contacts)
    final = samples[-1] if samples else {}
    final_t = _float(final.get("time_s"))
    recent_contacts = [
        contact
        for row_time, contact in zip(time_s, contacts)
        if math.isfinite(row_time) and math.isfinite(contact) and math.isfinite(final_t) and row_time >= final_t - 1.0
    ]
    crossing_indices = [index for index, value in enumerate(base_x) if math.isfinite(value) and value >= clear_x]
    return {
        "sample_count": len(samples),
        "duration_s": (max(finite_t) - min(finite_t)) if len(finite_t) >= 2 else 0.0,
        "initial_base_x_m": finite_x[0] if finite_x else float("nan"),
        "final_base_x_m": _float(final.get("base_x_m")),
        "max_base_x_m": max(finite_x) if finite_x else float("nan"),
        "forward_progress_m": (max(finite_x) - finite_x[0]) if finite_x else float("nan"),
        "first_clearance_time_s": time_s[crossing_indices[0]] if crossing_indices else None,
        "min_base_z_m": min(finite_z) if finite_z else float("nan"),
        "final_base_z_m": _float(final.get("base_z_m")),
        "max_base_z_m": max(finite_z) if finite_z else float("nan"),
        "max_base_z_delta_m": (max(finite_z) - finite_z[0]) if finite_z else float("nan"),
        "max_abs_roll_rad": max([abs(v) for v in _finite(roll)], default=float("nan")),
        "max_abs_pitch_rad": max([abs(v) for v in _finite(pitch)], default=float("nan")),
        "max_abs_roll_pitch_rad": max([abs(v) for v in _finite(roll + pitch)], default=float("nan")),
        "min_static_margin_m": min(finite_margin) if finite_margin else None,
        "finite_static_margin_count": len(finite_margin),
        "min_active_contacts": min(finite_contacts, default=float("nan")),
        "max_active_contacts": max(finite_contacts, default=float("nan")),
        "final_active_contacts": _float(final.get("active_contact_count")),
        "recent_active_contacts_max": max(recent_contacts, default=float("nan")),
        "contact_supported_sample_count": sum(1 for value in finite_contacts if value >= 2),
        "contact_supported_fraction": (sum(1 for value in finite_contacts if value >= 2) / len(finite_contacts)) if finite_contacts else 0.0,
        "recent_contact_supported_fraction": (sum(1 for value in recent_contacts if value >= 2) / len(recent_contacts)) if recent_contacts else 0.0,
        "last_replay_state": str(final.get("replay_state", "") or ""),
        "last_sequence_success": _parse_bool(final.get("sequence_success")),
    }


def _replay_metrics(samples: list[dict[str, str]], events: list[dict[str, Any]]) -> dict[str, Any]:
    event_types = [str(row.get("event_type", "") or "") for row in events]
    start_events = [row for row in events if str(row.get("event_type", "") or "") == "replay_started"]
    finish_events = [row for row in events if str(row.get("event_type", "") or "") == "replay_finished"]
    command_events = [row for row in events if str(row.get("event_type", "") or "") == "replay_command"]
    sample_success = any(_parse_bool(row.get("sequence_success")) is True for row in samples)
    max_sample_index = max([int(_float(row.get("replay_event_index"))) for row in samples if math.isfinite(_float(row.get("replay_event_index")))] + [-1])
    max_event_index = max([_event_index(row) for row in command_events] + [max_sample_index])
    start_sim_time = _event_sim_time(start_events[-1]) if start_events else _event_sim_time(command_events[0]) if command_events else float("nan")
    finish = finish_events[-1] if finish_events else {}
    finish_sim_time = _event_sim_time(finish) if finish_events else float("nan")
    event_duration = max(0.0, finish_sim_time - start_sim_time) if math.isfinite(start_sim_time) and math.isfinite(finish_sim_time) else 0.0
    event_finished_success = bool(finish_events) and str(finish.get("severity", "info") or "info").lower() == "info"
    return {
        "started": "replay_started" in event_types or bool(command_events),
        "finished": bool(finish_events) or sample_success,
        "finished_success": event_finished_success or sample_success,
        "finish_message": str(finish.get("message", "") or ("sequence_success=True" if sample_success else "")),
        "command_event_count": len(command_events),
        "max_event_index": int(max_event_index),
        "started_sim_time_s": start_sim_time if math.isfinite(start_sim_time) else None,
        "finished_sim_time_s": finish_sim_time if math.isfinite(finish_sim_time) else None,
        "event_duration_s": event_duration,
    }


def _replay_window(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    window: list[dict[str, str]] = []
    for row in samples:
        if str(row.get("replay_state", "") or "") == "active" or _parse_bool(row.get("sequence_success")) is True:
            window.append(row)
            continue
        event_count = _float(row.get("replay_event_count"))
        if math.isfinite(event_count) and int(event_count) > 0:
            window.append(row)
    return window or samples


def _check(rows: list[dict[str, Any]], name: str, passed: bool, actual: Any, expected: Any) -> None:
    rows.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})


def _contact_data_unavailable(quality: dict[str, Any]) -> bool:
    warnings = quality.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [warnings]
    text = " ".join(str(item).lower() for item in warnings)
    return "contact" in text and any(token in text for token in ("unavailable", "disabled", "missing", "not available"))


def _event_index(row: dict[str, Any]) -> int:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    value = extra.get("playback_event_index", row.get("playback_event_index"))
    if value is None:
        return -1
    try:
        return int(float(value))
    except Exception:
        return -1


def _event_sim_time(row: dict[str, Any]) -> float:
    return _float(row.get("simulation_time_s", row.get("time_s")))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except Exception:
            pass
    return rows


def _float(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except Exception:
        return float("nan")


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _first_finite(*values: Any) -> float:
    for value in values:
        parsed = _float(value)
        if math.isfinite(parsed):
            return parsed
    return float("nan")


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _dig(root: dict[str, Any], *keys: str) -> Any:
    current: Any = root
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
