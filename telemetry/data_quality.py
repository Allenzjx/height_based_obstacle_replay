"""Lightweight telemetry data-quality checks for stability reports."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from telemetry.exporters import read_csv_rows, write_json


def analyze_run_data_quality(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    samples = read_csv_rows(root / "telemetry_samples.csv")
    contacts = read_csv_rows(root / "contacts.csv")
    joints = read_csv_rows(root / "joint_timeseries.csv")
    time_s = _series(samples, "time_s")
    com_x = _series(samples, "com_x_m")
    com_y = _series(samples, "com_y_m")
    com_z = _series(samples, "com_z_m")
    static_margin = _series(samples, "static_stability_margin_m")
    dynamic_margin = _series(samples, "dynamic_stability_margin_m")
    equilibrium_margin = _series(samples, "equilibrium_stability_margin_m")
    support_area = _series(samples, "support_area_m2")
    torque_util = _series(joints, "torque_utilization")
    normal_forces = _series(contacts, "normal_force_n")
    warnings: list[str] = []
    if len(samples) < 2:
        warnings.append("not enough telemetry samples")
    if _finite_count(static_margin) == 0:
        warnings.append("static stability margin unavailable")
    if _finite_count(dynamic_margin) == 0:
        warnings.append("dynamic stability margin unavailable")
    if _finite_count(equilibrium_margin) == 0:
        warnings.append("equilibrium stability margin unavailable")
    if _constant_like(static_margin):
        warnings.append("static margin appears constant or placeholder-like")
    if _constant_like(equilibrium_margin):
        warnings.append("equilibrium margin appears constant or placeholder-like")
    if _finite_count(normal_forces) == 0:
        warnings.append("contact normal forces unavailable")
    if _finite_count(torque_util) == 0:
        warnings.append("joint torque utilization unavailable")
    static = {
        "sample_count": len(samples),
        "duration_s": (max(time_s) - min(time_s)) if _finite_count(time_s) >= 2 else 0.0,
        "com_jitter_xy_m": _range(com_x) + _range(com_y),
        "com_z_range_m": _range(com_z),
        "static_margin_mean_m": _mean(static_margin),
        "static_margin_std_m": _std(static_margin),
        "support_area_mean_m2": _mean(support_area),
        "support_area_std_m2": _std(support_area),
    }
    dynamic = {
        "dynamic_margin_mean_m": _mean(dynamic_margin),
        "dynamic_margin_std_m": _std(dynamic_margin),
        "equilibrium_margin_mean_m": _mean(equilibrium_margin),
        "equilibrium_margin_std_m": _std(equilibrium_margin),
        "max_torque_utilization": _max_abs(torque_util),
        "max_contact_normal_force_n": _max_abs(normal_forces),
        "finite_torque_rows": _finite_count(torque_util),
        "finite_contact_rows": _finite_count(normal_forces),
    }
    verdict = "PASS"
    if warnings:
        verdict = "WARN"
    if len(samples) < 2 or _finite_count(static_margin) == 0:
        verdict = "INSUFFICIENT"
    result = {
        "schema": "telemetry.data_quality.v1",
        "overall_verdict": verdict,
        "stability_conclusion": "Insufficient valid data for stability conclusion"
        if verdict == "INSUFFICIENT"
        else "Stability metrics are usable with warnings" if verdict == "WARN" else "Stability metrics are usable",
        "warnings": warnings,
        "static_baseline": static,
        "dynamic_consistency": dynamic,
        "sources": {
            "telemetry_samples": str(root / "telemetry_samples.csv"),
            "contacts": str(root / "contacts.csv"),
            "joints": str(root / "joint_timeseries.csv"),
        },
    }
    write_json(root / "data_quality.json", result)
    write_json(root / "static_baseline.json", static)
    write_json(root / "dynamic_consistency.json", dynamic)
    return result


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
            values.append(value if math.isfinite(value) else float("nan"))
        except Exception:
            values.append(float("nan"))
    return values


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]


def _finite_count(values: list[float]) -> int:
    return len(_finite(values))


def _mean(values: list[float]) -> float | None:
    finite = _finite(values)
    return statistics.fmean(finite) if finite else None


def _std(values: list[float]) -> float | None:
    finite = _finite(values)
    return statistics.pstdev(finite) if len(finite) >= 2 else 0.0 if finite else None


def _range(values: list[float]) -> float:
    finite = _finite(values)
    return max(finite) - min(finite) if finite else float("nan")


def _max_abs(values: list[float]) -> float | None:
    finite = _finite(values)
    return max(abs(value) for value in finite) if finite else None


def _constant_like(values: list[float]) -> bool:
    finite = _finite(values)
    if len(finite) < 10:
        return False
    return (max(finite) - min(finite)) <= 1.0e-9


def write_quality_report_copy(run_dir: str | Path, target: str | Path) -> None:
    result = analyze_run_data_quality(run_dir)
    Path(target).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
