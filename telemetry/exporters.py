"""File exporters for replay telemetry runs."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import RuntimeTelemetryConfig, config_snapshot_dict


def create_run_dir(
    output_root: str | Path,
    *,
    height_cm: int | None = None,
    sequence_label: str = "",
    run_name: str = "",
) -> Path:
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if run_name:
        run_dir = root / _safe_slug(run_name)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        height = "unknown" if height_cm is None else f"{int(height_cm):02d}cm"
        label = _safe_slug(sequence_label or "session")[:56]
        run_dir = root / f"{stamp}_obstacle-{height}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def write_metadata(run_dir: Path, metadata: dict[str, Any], config: RuntimeTelemetryConfig) -> None:
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config_snapshot.json", config_snapshot_dict(config))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = _field_order(rows)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def write_npz(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key in (
        "time_s",
        "base_x_m",
        "base_y_m",
        "base_z_m",
        "base_roll_rad",
        "base_pitch_rad",
        "base_yaw_rad",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "com_vx_m_s",
        "com_vy_m_s",
        "com_vz_m_s",
        "static_stability_margin_m",
        "dynamic_stability_margin_m",
        "equilibrium_stability_margin_m",
        "support_area_m2",
        "active_contact_count",
        "max_contact_force_n",
        "tracking_rmse_rad",
        "max_torque_utilization",
        "sample_overhead_ms",
    ):
        arrays[key] = np.asarray([_float(row.get(key)) for row in rows], dtype=float)
    np.savez_compressed(destination, **arrays)


def summarize_run(
    rows: list[dict[str, Any]],
    body_rows: list[dict[str, Any]] | None,
    joint_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    started_wall: float,
    finished_wall: float,
    dropped_samples: int = 0,
) -> dict[str, Any]:
    margins = [_float(row.get("static_stability_margin_m")) for row in rows]
    dyn_margins = [_float(row.get("dynamic_stability_margin_m")) for row in rows]
    eq_margins = [_float(row.get("equilibrium_stability_margin_m")) for row in rows]
    overhead = [_float(row.get("sample_overhead_ms")) for row in rows]
    states: dict[str, int] = {}
    for row in rows:
        state = str(row.get("stability_state", "") or "")
        if state:
            states[state] = states.get(state, 0) + 1
    severities: dict[str, int] = {}
    for event in events:
        severity = str(event.get("severity", "info") or "info")
        severities[severity] = severities.get(severity, 0) + 1
    return {
        "sample_count": len(rows),
        "body_com_sample_count": len(body_rows or []),
        "joint_sample_count": len(joint_rows),
        "contact_sample_count": len(contact_rows),
        "event_count": len(events),
        "dropped_samples": int(dropped_samples),
        "duration_sim_s": _duration(rows),
        "duration_wall_s": max(0.0, float(finished_wall) - float(started_wall)),
        "min_static_margin_m": _finite_min(margins),
        "min_dynamic_margin_m": _finite_min(dyn_margins),
        "min_equilibrium_margin_m": _finite_min(eq_margins),
        "max_sample_overhead_ms": _finite_max(overhead),
        "mean_sample_overhead_ms": _finite_mean(overhead),
        "stability_state_counts": states,
        "event_severity_counts": severities,
        "max_contact_force_n": _finite_max([_float(row.get("max_contact_force_n")) for row in rows]),
        "max_torque_utilization": _finite_max([_float(row.get("torque_utilization")) for row in joint_rows]),
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _field_order(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _finite_min(values: list[float]) -> float:
    finite = _finite(values)
    return min(finite) if finite else float("nan")


def _finite_max(values: list[float]) -> float:
    finite = _finite(values)
    return max(finite) if finite else float("nan")


def _finite_mean(values: list[float]) -> float:
    finite = _finite(values)
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _duration(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    first = _float(rows[0].get("time_s"))
    last = _float(rows[-1].get("time_s"))
    return max(0.0, last - first) if math.isfinite(first) and math.isfinite(last) else 0.0


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "").strip()).strip("-")
    return slug or "run"


def python_runtime_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "executable": sys.executable,
    }
