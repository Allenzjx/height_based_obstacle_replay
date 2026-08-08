"""Offline alignment of all physical 50 mm recording directories.

The accepted-step files contain commands and before/after articulation
snapshots.  They do not contain continuous COM, contact-force, or support
telemetry.  This module therefore exports endpoint candidates only and never
promotes a row to a mechanically verified phase.

The module is Isaac-free.  It calls the project's existing pure-Python Fast
Replay planner so segment ranges and timing have the same semantics as a later
replay, but it does not start or import Isaac.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from sim_state_validation import (
    ROOT_ORIENTATION_TOLERANCE_DEG,
    ROOT_POSITION_TOLERANCE_M,
    SERVO_JOINT_POSITION_TOLERANCE_DEG,
    WHEEL_JOINT_POSITION_TOLERANCE_RAD,
)

from .recording_fast_plan import fast_plan_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDING_ROOT = (
    PROJECT_ROOT
    / "saved_height_steps_fsm_reference_v2"
    / "height_050mm"
)
DEFAULT_REPORT_ROOT = PACKAGE_ROOT / "reports"
BASELINE_PATH = PROJECT_ROOT / "config" / "fsm_recording_baseline.yaml"

ENDPOINT_CANDIDATE = "ENDPOINT_CANDIDATE"
PENDING_REPLAY = "PENDING_REPLAY"
NO_CONTINUOUS_TELEMETRY = "NO_CONTINUOUS_COM_CONTACT_TELEMETRY"

ALIGNMENT_COLUMNS = (
    "source_version",
    "source_directory",
    "fast_plan_sha256",
    "step_index",
    "step_name",
    "source_duration_s",
    "source_event_count",
    "fast_segment_range",
    "fast_segment_first_index",
    "fast_segment_last_index",
    "fast_segment_count",
    "fast_start_s",
    "fast_end_s",
    "fast_duration_s",
    "fast_gap_from_previous_step_s",
    "recorded_snapshot_gap_from_previous_step_s",
    "commands",
    "command_shape",
    "concurrent_segment_count",
    "expected_wheel_displacement_rad",
    "candidate_phase_evidence",
    "candidate_phase_status",
    "continuous_telemetry_status",
    "start_boundary_status",
    "end_boundary_status",
    "root_position_start_m",
    "root_position_end_m",
    "root_orientation_start_wxyz",
    "root_orientation_end_wxyz",
    "attitude_rpy_start_rad",
    "attitude_rpy_end_rad",
    "root_linear_velocity_start_m_s",
    "root_linear_velocity_end_m_s",
    "root_angular_velocity_start_rad_s",
    "root_angular_velocity_end_rad_s",
    "joint_position_start_rad",
    "joint_position_end_rad",
    "joint_velocity_start_rad_s",
    "joint_velocity_end_rad_s",
    "adjacent_joint_schema_compatible",
    "adjacent_command_state_compatible",
    "adjacent_root_position_gap_m",
    "adjacent_attitude_gap_rad",
    "adjacent_servo_position_gap_deg",
    "adjacent_wheel_position_gap_rad",
    "adjacent_joint_velocity_gap_rad_s",
    "adjacent_root_linear_velocity_gap_m_s",
    "adjacent_root_angular_velocity_gap_rad_s",
    "adjacent_endpoint_compatibility",
)


def _single_row(value: Any) -> list[Any] | None:
    row = value
    while (
        isinstance(row, (list, tuple))
        and len(row) == 1
        and isinstance(row[0], (list, tuple))
    ):
        row = row[0]
    return list(row) if isinstance(row, (list, tuple)) else None


def finite_vector(value: Any, *, width: int | None = None) -> list[float] | None:
    """Return a finite flat vector, accepting the recording's one-row shape."""

    row = _single_row(value)
    if row is None or (width is not None and len(row) != int(width)):
        return None
    try:
        result = [float(item) for item in row]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def quaternion_to_rpy_wxyz(
    quaternion_wxyz: Sequence[float],
) -> tuple[float, float, float]:
    """Convert a finite wxyz quaternion to roll, pitch, yaw in radians."""

    if len(quaternion_wxyz) != 4:
        raise ValueError("expected a wxyz quaternion")
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must have finite nonzero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def quaternion_distance_rad(
    left_wxyz: Sequence[float], right_wxyz: Sequence[float]
) -> float:
    """Shortest orientation distance; q and -q are treated as identical."""

    if len(left_wxyz) != 4 or len(right_wxyz) != 4:
        raise ValueError("expected two wxyz quaternions")
    left = [float(value) for value in left_wxyz]
    right = [float(value) for value in right_wxyz]
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if (
        not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
        or left_norm <= 1.0e-12
        or right_norm <= 1.0e-12
    ):
        raise ValueError("quaternions must have finite nonzero norms")
    dot = abs(
        sum(a * b for a, b in zip(left, right, strict=True))
        / (left_norm * right_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _vector_norm_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector widths differ")
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))
    )


def _joint_map(
    state: dict[str, Any], field: str
) -> tuple[list[str], dict[str, float]] | None:
    names = [str(name) for name in list(state.get("joint_names", []) or [])]
    values = finite_vector(state.get(field))
    if (
        not names
        or values is None
        or len(names) != len(values)
        or len(set(names)) != len(names)
    ):
        return None
    return names, dict(zip(names, values, strict=True))


def boundary_snapshot(state: Any) -> dict[str, Any]:
    """Extract endpoint fields without inventing values for legacy snapshots."""

    source = dict(state or {}) if isinstance(state, dict) else {}
    root_pose = finite_vector(source.get("root_pose"), width=7)
    root_velocity = finite_vector(source.get("root_velocity"), width=6)
    joint_position = _joint_map(source, "joint_pos")
    joint_velocity = _joint_map(source, "joint_vel")

    missing: list[str] = []
    if root_pose is None:
        missing.append("root_pose")
    if root_velocity is None:
        missing.append("root_velocity")
    if joint_position is None:
        missing.append("joint_pos")
    if joint_velocity is None:
        missing.append("joint_vel")

    orientation = root_pose[3:7] if root_pose is not None else None
    try:
        attitude = (
            list(quaternion_to_rpy_wxyz(orientation))
            if orientation is not None
            else None
        )
    except ValueError:
        attitude = None
        if "root_pose" not in missing:
            missing.append("root_orientation")

    return {
        "status": "FULL_BOUNDARY"
        if not missing
        else "MISSING:" + "|".join(sorted(missing)),
        "root_position_m": root_pose[:3] if root_pose is not None else None,
        "root_orientation_wxyz": orientation,
        "attitude_rpy_rad": attitude,
        "root_linear_velocity_m_s": (
            root_velocity[:3] if root_velocity is not None else None
        ),
        "root_angular_velocity_rad_s": (
            root_velocity[3:6] if root_velocity is not None else None
        ),
        "joint_names": joint_position[0] if joint_position is not None else [],
        "joint_position_rad": (
            joint_position[1] if joint_position is not None else None
        ),
        "joint_velocity_rad_s": (
            joint_velocity[1] if joint_velocity is not None else None
        ),
        "command_state": source.get("command_state"),
        "sim_time_s": _finite_scalar(source.get("sim_time")),
    }


def _finite_scalar(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _max_named_delta(
    names: Iterable[str],
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    selected = [name for name in names if name in left and name in right]
    return max((abs(right[name] - left[name]) for name in selected), default=0.0)


def compare_endpoint_boundaries(
    previous_after: Any, current_before: Any
) -> dict[str, Any]:
    """Compare adjacent recorded endpoints using restorable-pose tolerances.

    Velocity gaps are reported but intentionally excluded from pose
    compatibility.  A later replay must decide whether a velocity reset or
    retained momentum is mechanically correct.
    """

    previous = boundary_snapshot(previous_after)
    current = boundary_snapshot(current_before)
    empty = {
        "joint_schema_compatible": None,
        "command_state_compatible": None,
        "root_position_gap_m": None,
        "attitude_gap_rad": None,
        "servo_position_gap_deg": None,
        "wheel_position_gap_rad": None,
        "joint_velocity_gap_rad_s": None,
        "root_linear_velocity_gap_m_s": None,
        "root_angular_velocity_gap_rad_s": None,
        "endpoint_compatibility": "MISSING_ENDPOINT",
    }
    required = (
        previous["root_position_m"],
        previous["root_orientation_wxyz"],
        previous["joint_position_rad"],
        current["root_position_m"],
        current["root_orientation_wxyz"],
        current["joint_position_rad"],
    )
    if any(value is None for value in required):
        return empty

    previous_names = set(previous["joint_names"])
    current_names = set(current["joint_names"])
    schema_compatible = bool(previous_names and previous_names == current_names)
    if not schema_compatible:
        return {
            **empty,
            "joint_schema_compatible": False,
            "command_state_compatible": (
                previous["command_state"] == current["command_state"]
            ),
            "endpoint_compatibility": "JOINT_SCHEMA_MISMATCH",
        }

    previous_position = dict(previous["joint_position_rad"])
    current_position = dict(current["joint_position_rad"])
    root_position_gap = _vector_norm_delta(
        previous["root_position_m"], current["root_position_m"]
    )
    attitude_gap = quaternion_distance_rad(
        previous["root_orientation_wxyz"],
        current["root_orientation_wxyz"],
    )
    servo_position_gap_deg = math.degrees(
        _max_named_delta(
            SERVO_JOINT_NAMES, previous_position, current_position
        )
    )
    wheel_position_gap_rad = _max_named_delta(
        WHEEL_JOINT_NAMES, previous_position, current_position
    )

    velocity_gap: float | None = None
    previous_velocity = previous["joint_velocity_rad_s"]
    current_velocity = current["joint_velocity_rad_s"]
    if previous_velocity is not None and current_velocity is not None:
        velocity_gap = _max_named_delta(
            previous_names, dict(previous_velocity), dict(current_velocity)
        )

    linear_velocity_gap = None
    if (
        previous["root_linear_velocity_m_s"] is not None
        and current["root_linear_velocity_m_s"] is not None
    ):
        linear_velocity_gap = _vector_norm_delta(
            previous["root_linear_velocity_m_s"],
            current["root_linear_velocity_m_s"],
        )
    angular_velocity_gap = None
    if (
        previous["root_angular_velocity_rad_s"] is not None
        and current["root_angular_velocity_rad_s"] is not None
    ):
        angular_velocity_gap = _vector_norm_delta(
            previous["root_angular_velocity_rad_s"],
            current["root_angular_velocity_rad_s"],
        )

    pose_compatible = bool(
        root_position_gap <= ROOT_POSITION_TOLERANCE_M
        and math.degrees(attitude_gap) <= ROOT_ORIENTATION_TOLERANCE_DEG
        and servo_position_gap_deg <= SERVO_JOINT_POSITION_TOLERANCE_DEG
        and wheel_position_gap_rad <= WHEEL_JOINT_POSITION_TOLERANCE_RAD
    )
    return {
        "joint_schema_compatible": True,
        "command_state_compatible": (
            previous["command_state"] == current["command_state"]
        ),
        "root_position_gap_m": root_position_gap,
        "attitude_gap_rad": attitude_gap,
        "servo_position_gap_deg": servo_position_gap_deg,
        "wheel_position_gap_rad": wheel_position_gap_rad,
        "joint_velocity_gap_rad_s": velocity_gap,
        "root_linear_velocity_gap_m_s": linear_velocity_gap,
        "root_angular_velocity_gap_rad_s": angular_velocity_gap,
        "endpoint_compatibility": (
            "POSE_COMPATIBLE_ENDPOINTS"
            if pose_compatible
            else "POSE_GAP_EXCEEDS_RESTORE_TOLERANCE"
        ),
    }


def _first_step_adjacency() -> dict[str, Any]:
    return {
        "joint_schema_compatible": None,
        "command_state_compatible": None,
        "root_position_gap_m": None,
        "attitude_gap_rad": None,
        "servo_position_gap_deg": None,
        "wheel_position_gap_rad": None,
        "joint_velocity_gap_rad_s": None,
        "root_linear_velocity_gap_m_s": None,
        "root_angular_velocity_gap_rad_s": None,
        "endpoint_compatibility": "FIRST_STEP",
    }


def summarize_fast_segments(
    segments: Sequence[dict[str, Any]],
    *,
    previous_fast_end_s: float | None,
) -> dict[str, Any]:
    """Aggregate authoritative Fast segments that belong to one source step."""

    if not segments:
        return {
            "segment_range": "",
            "segment_first_index": None,
            "segment_last_index": None,
            "segment_count": 0,
            "start_s": None,
            "end_s": None,
            "duration_s": None,
            "gap_from_previous_s": None,
            "commands": [],
            "command_shape": "NO_FAST_SEGMENT",
            "concurrent_segment_count": 0,
            "expected_wheel_displacement_rad": {
                name: 0.0 for name in WHEEL_JOINT_NAMES
            },
        }

    ordered = sorted(
        segments, key=lambda row: int(row["decoded_segment_index"])
    )
    first_index = int(ordered[0]["decoded_segment_index"])
    last_index = int(ordered[-1]["decoded_segment_index"])
    start_s = min(float(row["command_start_s"]) for row in ordered)
    end_s = max(float(row["command_end_s"]) for row in ordered)
    commands = [
        str(command)
        for row in ordered
        for command in list(row.get("commands", []) or [])
    ]
    displacement = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    has_servo = False
    has_nonzero_wheel = False
    concurrent_count = 0
    for row in ordered:
        has_servo = has_servo or bool(row.get("servo_target_deg"))
        concurrent_count += int(bool(row.get("concurrent", False)))
        for name, value in dict(
            row.get("expected_wheel_displacement_rad", {}) or {}
        ).items():
            if name in displacement:
                displacement[name] += float(value)
        wheel_targets = dict(row.get("wheel_target_rad_s", {}) or {})
        if any(abs(float(value)) > 1.0e-12 for value in wheel_targets.values()):
            has_nonzero_wheel = True

    if has_servo and has_nonzero_wheel:
        shape = "SERVO_WHEEL_COMMANDS"
    elif has_servo:
        shape = "SERVO_COMMANDS"
    elif has_nonzero_wheel:
        shape = "WHEEL_COMMANDS"
    else:
        shape = "ZERO_OR_HOLD_COMMANDS"

    return {
        "segment_range": f"{first_index}:{last_index}",
        "segment_first_index": first_index,
        "segment_last_index": last_index,
        "segment_count": len(ordered),
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "gap_from_previous_s": (
            None
            if previous_fast_end_s is None
            else start_s - float(previous_fast_end_s)
        ),
        "commands": commands,
        "command_shape": shape,
        "concurrent_segment_count": concurrent_count,
        "expected_wheel_displacement_rad": displacement,
    }


def enumerate_recording_directories(
    recording_root: Path = DEFAULT_RECORDING_ROOT,
) -> list[Path]:
    """Enumerate physical directories, not stale manifest-only entries."""

    versions_root = Path(recording_root) / "versions"
    if not versions_root.is_dir():
        raise FileNotFoundError(versions_root)
    return sorted(
        directory
        for directory in versions_root.iterdir()
        if directory.is_dir()
        and (directory / "accepted_steps.jsonl").is_file()
        and (directory / "metadata.json").is_file()
    )


def load_steps_jsonl(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        steps.append(value)
    return steps


def load_max_wheel_speed(path: Path = BASELINE_PATH) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = float(payload["wheel_actuator"]["velocity_limit_rad_s"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("wheel velocity limit must be finite and positive")
    return value


def align_version(
    directory: Path,
    *,
    max_wheel_speed: float,
) -> list[dict[str, Any]]:
    steps = load_steps_jsonl(directory / "accepted_steps.jsonl")
    plan, fast_rows = fast_plan_rows(
        source_version=directory.name,
        steps=steps,
        max_wheel_speed=max_wheel_speed,
    )
    fast_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in fast_rows:
        fast_by_step[int(row["source_step_index"])].append(row)

    result: list[dict[str, Any]] = []
    previous_after: dict[str, Any] | None = None
    previous_fast_end_s: float | None = None
    for step in steps:
        step_index = int(step.get("index", 0) or 0)
        fast = summarize_fast_segments(
            fast_by_step.get(step_index, []),
            previous_fast_end_s=previous_fast_end_s,
        )
        if fast["end_s"] is not None:
            previous_fast_end_s = float(fast["end_s"])

        before_state = dict(step.get("sim_state_before", {}) or {})
        after_state = dict(step.get("sim_state_after", {}) or {})
        before = boundary_snapshot(before_state)
        after = boundary_snapshot(after_state)
        adjacency = (
            _first_step_adjacency()
            if previous_after is None
            else compare_endpoint_boundaries(previous_after, before_state)
        )
        recorded_gap = None
        if previous_after is not None:
            previous_time = boundary_snapshot(previous_after)["sim_time_s"]
            current_time = before["sim_time_s"]
            if previous_time is not None and current_time is not None:
                recorded_gap = current_time - previous_time

        result.append(
            {
                "source_version": directory.name,
                "source_directory": str(directory.resolve()),
                "fast_plan_sha256": str(plan.plan_sha256),
                "step_index": step_index,
                "step_name": str(step.get("name", "") or ""),
                "source_duration_s": float(step.get("duration", 0.0) or 0.0),
                "source_event_count": len(list(step.get("events", []) or [])),
                "fast_segment_range": fast["segment_range"],
                "fast_segment_first_index": fast["segment_first_index"],
                "fast_segment_last_index": fast["segment_last_index"],
                "fast_segment_count": fast["segment_count"],
                "fast_start_s": fast["start_s"],
                "fast_end_s": fast["end_s"],
                "fast_duration_s": fast["duration_s"],
                "fast_gap_from_previous_step_s": fast[
                    "gap_from_previous_s"
                ],
                "recorded_snapshot_gap_from_previous_step_s": recorded_gap,
                "commands": fast["commands"],
                "command_shape": fast["command_shape"],
                "concurrent_segment_count": fast[
                    "concurrent_segment_count"
                ],
                "expected_wheel_displacement_rad": fast[
                    "expected_wheel_displacement_rad"
                ],
                "candidate_phase_evidence": ENDPOINT_CANDIDATE,
                "candidate_phase_status": PENDING_REPLAY,
                "continuous_telemetry_status": NO_CONTINUOUS_TELEMETRY,
                "start_boundary_status": before["status"],
                "end_boundary_status": after["status"],
                "root_position_start_m": before["root_position_m"],
                "root_position_end_m": after["root_position_m"],
                "root_orientation_start_wxyz": before[
                    "root_orientation_wxyz"
                ],
                "root_orientation_end_wxyz": after[
                    "root_orientation_wxyz"
                ],
                "attitude_rpy_start_rad": before["attitude_rpy_rad"],
                "attitude_rpy_end_rad": after["attitude_rpy_rad"],
                "root_linear_velocity_start_m_s": before[
                    "root_linear_velocity_m_s"
                ],
                "root_linear_velocity_end_m_s": after[
                    "root_linear_velocity_m_s"
                ],
                "root_angular_velocity_start_rad_s": before[
                    "root_angular_velocity_rad_s"
                ],
                "root_angular_velocity_end_rad_s": after[
                    "root_angular_velocity_rad_s"
                ],
                "joint_position_start_rad": before["joint_position_rad"],
                "joint_position_end_rad": after["joint_position_rad"],
                "joint_velocity_start_rad_s": before[
                    "joint_velocity_rad_s"
                ],
                "joint_velocity_end_rad_s": after[
                    "joint_velocity_rad_s"
                ],
                "adjacent_joint_schema_compatible": adjacency[
                    "joint_schema_compatible"
                ],
                "adjacent_command_state_compatible": adjacency[
                    "command_state_compatible"
                ],
                "adjacent_root_position_gap_m": adjacency[
                    "root_position_gap_m"
                ],
                "adjacent_attitude_gap_rad": adjacency[
                    "attitude_gap_rad"
                ],
                "adjacent_servo_position_gap_deg": adjacency[
                    "servo_position_gap_deg"
                ],
                "adjacent_wheel_position_gap_rad": adjacency[
                    "wheel_position_gap_rad"
                ],
                "adjacent_joint_velocity_gap_rad_s": adjacency[
                    "joint_velocity_gap_rad_s"
                ],
                "adjacent_root_linear_velocity_gap_m_s": adjacency[
                    "root_linear_velocity_gap_m_s"
                ],
                "adjacent_root_angular_velocity_gap_rad_s": adjacency[
                    "root_angular_velocity_gap_rad_s"
                ],
                "adjacent_endpoint_compatibility": adjacency[
                    "endpoint_compatibility"
                ],
            }
        )
        previous_after = after_state
    return result


def build_alignment(
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    *,
    expected_version_count: int | None = 9,
) -> tuple[list[Path], list[dict[str, Any]]]:
    directories = enumerate_recording_directories(recording_root)
    if (
        expected_version_count is not None
        and len(directories) != int(expected_version_count)
    ):
        raise ValueError(
            f"expected {expected_version_count} physical recording directories, "
            f"found {len(directories)}"
        )
    max_wheel_speed = load_max_wheel_speed()
    rows = [
        row
        for directory in directories
        for row in align_version(
            directory, max_wheel_speed=max_wheel_speed
        )
    ]
    return directories, rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def write_alignment_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ALIGNMENT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _csv_value(row.get(key)) for key in ALIGNMENT_COLUMNS}
            )


def _version_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shapes = Counter(str(row["command_shape"]) for row in rows)
    adjacency = Counter(
        str(row["adjacent_endpoint_compatibility"]) for row in rows
    )
    full_boundaries = sum(
        row["start_boundary_status"] == "FULL_BOUNDARY"
        and row["end_boundary_status"] == "FULL_BOUNDARY"
        for row in rows
    )
    wheel_steps = sum(
        any(
            abs(float(value)) > 1.0e-12
            for value in dict(
                row["expected_wheel_displacement_rad"]
            ).values()
        )
        for row in rows
    )
    concurrent_steps = sum(
        int(row["concurrent_segment_count"]) > 0 for row in rows
    )
    return {
        "steps": len(rows),
        "full_boundaries": full_boundaries,
        "wheel_steps": wheel_steps,
        "concurrent_steps": concurrent_steps,
        "shapes": shapes,
        "adjacency": adjacency,
    }


def write_method_comparison(
    path: Path,
    *,
    directories: Sequence[Path],
    rows: Sequence[dict[str, Any]],
    alignment_csv: Path,
) -> None:
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_version[str(row["source_version"])].append(row)

    total_adjacencies = sum(
        row["adjacent_endpoint_compatibility"] != "FIRST_STEP"
        for row in rows
    )
    adjacency_counts = Counter(
        str(row["adjacent_endpoint_compatibility"])
        for row in rows
        if row["adjacent_endpoint_compatibility"] != "FIRST_STEP"
    )
    full_endpoint_rows = sum(
        row["start_boundary_status"] == "FULL_BOUNDARY"
        and row["end_boundary_status"] == "FULL_BOUNDARY"
        for row in rows
    )
    wheel_rows = sum(
        any(
            abs(float(value)) > 1.0e-12
            for value in dict(
                row["expected_wheel_displacement_rad"]
            ).values()
        )
        for row in rows
    )
    concurrent_rows = sum(
        int(row["concurrent_segment_count"]) > 0 for row in rows
    )

    lines = [
        "# 50 mm COM Transfer Method Comparison",
        "",
        "## Evidence boundary",
        "",
        "This is an offline endpoint comparison. No Isaac process was started. "
        "The nine recording directories contain commands and before/after "
        "articulation snapshots, but no standalone continuous COM, support "
        "force/class, contact-point drift, or video telemetry.",
        "",
        f"- Physical version directories enumerated: {len(directories)}",
        f"- Source steps aligned: {len(rows)}",
        f"- Steps with complete start/end root, attitude, joint, and velocity boundaries: {full_endpoint_rows}",
        f"- Adjacent step boundaries compared: {total_adjacencies}",
        f"- Endpoint compatibility counts: {json.dumps(dict(sorted(adjacency_counts.items())), sort_keys=True)}",
        f"- Steps with nonzero expected wheel displacement in the authoritative Fast plan: {wheel_rows}",
        f"- Steps containing at least one concurrent servo/wheel Fast segment: {concurrent_rows}",
        f"- Detailed alignment CSV: {alignment_csv.resolve()}",
        "",
        "Every row is marked ENDPOINT_CANDIDATE and PENDING_REPLAY. These labels "
        "must not be promoted using endpoint data alone.",
        "",
        "## Physical recording directories",
        "",
    ]
    lines.extend(f"- {directory.resolve()}" for directory in directories)
    lines.extend(
        [
            "",
            "## Per-version endpoint inventory",
            "",
            "| Version | Steps | Full boundaries | Nonzero wheel-displacement steps | Concurrent steps | Endpoint adjacency results |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for directory in directories:
        summary = _version_summary(by_version[directory.name])
        adjacency_text = ", ".join(
            f"{name}={count}"
            for name, count in sorted(summary["adjacency"].items())
        )
        lines.append(
            "| {version} | {steps} | {full} | {wheel} | {concurrent} | {adjacency} |".format(
                version=directory.name,
                steps=summary["steps"],
                full=summary["full_boundaries"],
                wheel=summary["wheel_steps"],
                concurrent=summary["concurrent_steps"],
                adjacency=adjacency_text,
            )
        )

    lines.extend(
        [
            "",
            "## Method comparison",
            "",
            "| Method | What endpoints can show | What endpoints cannot establish | Relevant audited failure constraints | Required clean-replay evidence | Current status |",
            "|---|---|---|---|---|---|",
            "| Impulse | Short successive Fast segments, command deltas, endpoint velocity before/after, and source timing can identify an impulse-shaped command candidate. | Endpoint velocity does not prove a contact-force impulse, momentum transfer, preload, coast, or causal COM shift during the step. | Old-failure audit findings 2, 5, 7, 14, and 16: static compression loses dynamics; entry steps and transition discontinuities matter; time alone is not a physical guard. | High-rate COM position/velocity, per-wheel force and contact class, contact-point drift, joint velocity, command dispatch time, and pre/post impulse dwell. | ENDPOINT_CANDIDATE / PENDING_REPLAY |",
            "| Anchored support-angle | Servo-only posture changes with zero commanded wheel speed identify support-angle endpoint candidates and expose joint/root boundary changes. | Zero wheel command does not prove an anchored contact, and an endpoint root shift does not prove the support points stayed fixed or loaded. | Findings 3, 9, 10, 11, 12, 13, 15, and 18: isolate the active leg; verify contact anchoring/load; use per-leg IK/signs; relatch COM targets; use a diagonal corridor rather than a fabricated polygon. | Relative COM to measured support contacts, support-point drift, retained upward load with dwell, per-leg applied IK, and diagonal segment coordinates s and d_perp. | ENDPOINT_CANDIDATE / PENDING_REPLAY |",
            "| Wheel assist | Nonzero wheel targets, Fast active duration, expected angular displacement, and servo/wheel concurrency are preserved and reported per step. | Wheel angular displacement does not prove body-relative COM improvement; the support boundary and body may translate together, and slip is unknown. | Findings 5, 6, 7, 8, 9, and 19: no phase-entry speed step, rear-local ramping only, preserve sustained travel, and never treat wheel travel as relative COM transfer. | Wheel/world displacement, body displacement, slip, contact-point drift, support-boundary motion, active-wheel zero-command compliance, and COM relative to the moving support set. | ENDPOINT_CANDIDATE / PENDING_REPLAY |",
            "| Hybrid | A step with concurrent or sequential servo and wheel commands can be identified without discarding command order, Fast duration, or expected wheel displacement. | Endpoints cannot attribute the observed root/joint change to the angle action, wheel action, their order, an impulse, or uncontrolled contact dynamics. | All constraints above, especially findings 2-8 and 14: preserve timing/concurrency, isolate the active leg, scope ramps locally, and maintain boundary continuity. | Primitive-isolation A/B replays first, then a preregistered hybrid replay with the same full telemetry, unchanged safety gates, and component-wise ablation. | ENDPOINT_CANDIDATE / PENDING_REPLAY |",
            "",
            "## Interpretation",
            "",
            "The endpoint inventory is sufficient to define replay candidates and "
            "to reject incompatible step joins. It is not sufficient to select "
            "a winning COM-transfer mechanism.",
            "",
            "A conservative experimental order is:",
            "",
            "1. verify an anchored support-angle primitive while measuring actual contact anchoring and load;",
            "2. verify wheel assist separately using relative COM and support-boundary motion, not wheel travel;",
            "3. replay impulse-shaped candidates with high-rate force and velocity telemetry;",
            "4. evaluate a hybrid only after each component has an isolated, repeatable mechanical effect.",
            "",
            "This order is a safety and identifiability recommendation, not a "
            "claim that any recording has already demonstrated one of these "
            "mechanisms.",
            "",
            "## Compatibility semantics",
            "",
            "POSE_COMPATIBLE_ENDPOINTS uses the repository's existing restore "
            f"tolerances: root position <= {ROOT_POSITION_TOLERANCE_M} m, "
            f"root orientation <= {ROOT_ORIENTATION_TOLERANCE_DEG} deg, "
            f"servo position <= {SERVO_JOINT_POSITION_TOLERANCE_DEG} deg, and "
            f"wheel position <= {WHEEL_JOINT_POSITION_TOLERANCE_RAD} rad.",
            "",
            "Velocity gaps are reported but do not decide endpoint compatibility. "
            "Whether momentum should be retained or reset is precisely one of "
            "the mechanisms that requires replay. MISSING_ENDPOINT is retained "
            "for legacy v003 snapshots and is never silently filled.",
            "",
            "## Evidence sources",
            "",
            f"- Endpoint alignment: {alignment_csv.resolve()}",
            f"- Old-failure evidence audit: {(DEFAULT_REPORT_ROOT / 'OLD_FSM_FAILURES_TO_AVOID.md').resolve()}",
            f"- Recording audit: {(DEFAULT_REPORT_ROOT / 'RECORDING_AUDIT_50MM.md').resolve()}",
            f"- Environment lock: {(DEFAULT_REPORT_ROOT / 'environment_lock_50mm.json').resolve()}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    expected_version_count: int | None = 9,
) -> dict[str, Any]:
    directories, rows = build_alignment(
        recording_root,
        expected_version_count=expected_version_count,
    )
    csv_path = Path(report_root) / "RECORDING_PHASE_ALIGNMENT_50MM.csv"
    markdown_path = (
        Path(report_root) / "COM_TRANSFER_METHOD_COMPARISON.md"
    )
    write_alignment_csv(csv_path, rows)
    write_method_comparison(
        markdown_path,
        directories=directories,
        rows=rows,
        alignment_csv=csv_path,
    )
    return {
        "directories": directories,
        "rows": rows,
        "csv_path": csv_path,
        "markdown_path": markdown_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align all physical 50 mm recording endpoints offline."
    )
    parser.add_argument(
        "--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT
    )
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_REPORT_ROOT
    )
    parser.add_argument("--expected-version-count", type=int, default=9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        recording_root=args.recording_root,
        report_root=args.report_root,
        expected_version_count=args.expected_version_count,
    )
    print(
        json.dumps(
            {
                "version_count": len(result["directories"]),
                "step_count": len(result["rows"]),
                "csv": str(result["csv_path"].resolve()),
                "comparison": str(result["markdown_path"].resolve()),
                "candidate_phase_evidence": ENDPOINT_CANDIDATE,
                "candidate_phase_status": PENDING_REPLAY,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
