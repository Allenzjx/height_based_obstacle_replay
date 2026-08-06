"""Strict, Isaac-free validation and comparison for restorable simulation poses."""

from __future__ import annotations

import math
from typing import Any

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES


FULL_VALID = "FULL_VALID"
COMMAND_ONLY = "COMMAND_ONLY"
PLACEHOLDER_NO_SIM = "PLACEHOLDER_NO_SIM"
INVALID = "INVALID"

ROOT_POSITION_TOLERANCE_M = 0.005
ROOT_ORIENTATION_TOLERANCE_DEG = 0.5
SERVO_JOINT_POSITION_TOLERANCE_DEG = 1.0
WHEEL_JOINT_POSITION_TOLERANCE_RAD = 0.05


def _single_row(value: Any, *, width: int | None = None) -> list[Any] | None:
    row = value
    while isinstance(row, (list, tuple)) and len(row) == 1 and isinstance(row[0], (list, tuple)):
        row = row[0]
    if not isinstance(row, (list, tuple)):
        return None
    result = list(row)
    if width is not None and len(result) != int(width):
        return None
    return result


def _finite_row(value: Any, *, width: int | None = None) -> list[float] | None:
    row = _single_row(value, width=width)
    if row is None:
        return None
    try:
        result = [float(item) for item in row]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _complete_command_state(state: Any) -> tuple[bool, list[str]]:
    if not isinstance(state, dict):
        return False, ["command_state"]
    missing: list[str] = []
    servos = dict(state.get("servos", {}) or {})
    wheels = dict(state.get("wheels", {}) or {})
    for name in SERVO_JOINT_NAMES:
        try:
            if name not in servos or not math.isfinite(float(servos[name])):
                missing.append(f"command_state.servos.{name}")
        except (TypeError, ValueError):
            missing.append(f"command_state.servos.{name}")
    for name in WHEEL_JOINT_NAMES:
        try:
            if name not in wheels or not math.isfinite(float(wheels[name])):
                missing.append(f"command_state.wheels.{name}")
        except (TypeError, ValueError):
            missing.append(f"command_state.wheels.{name}")
    return not missing, missing


def _complete_actual_state(state: Any) -> tuple[bool, list[str]]:
    if not isinstance(state, dict):
        return False, ["actual_joint_state"]
    missing: list[str] = []
    servos = dict(state.get("servos", {}) or {})
    wheels = dict(state.get("wheels", {}) or {})
    for name in SERVO_JOINT_NAMES:
        row = dict(servos.get(name, {}) or {})
        try:
            if row.get("deg") is None or not math.isfinite(float(row["deg"])):
                missing.append(f"actual_joint_state.servos.{name}.deg")
        except (TypeError, ValueError):
            missing.append(f"actual_joint_state.servos.{name}.deg")
    for name in WHEEL_JOINT_NAMES:
        row = dict(wheels.get(name, {}) or {})
        try:
            if row.get("rad_s") is None or not math.isfinite(float(row["rad_s"])):
                missing.append(f"actual_joint_state.wheels.{name}.rad_s")
        except (TypeError, ValueError):
            missing.append(f"actual_joint_state.wheels.{name}.rad_s")
    return not missing, missing


def _no_sim_marker(state: dict[str, Any]) -> bool:
    if str(state.get("capture_source", "") or "").lower() in {
        "nullsimrobotadapter",
        "no_sim",
        "no-sim",
    }:
        return True
    text = " ".join(
        str(state.get(key, "") or "")
        for key in ("grounded_reference_diagnostics", "robot_ground_diagnostics", "diagnostics")
    ).lower()
    return "no-sim adapter" in text or "nullsim" in text


def validate_full_sim_pose_state(
    state: Any,
    expected_joint_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Classify a state without mutating it or accepting command-only placeholders."""

    if not isinstance(state, dict):
        return {
            "valid": False,
            "classification": INVALID,
            "missing_fields": ["state"],
            "invalid_fields": [],
            "source": "",
            "joint_count": 0,
            "reason": "state is not a dictionary",
            "joint_reorder_indices": [],
        }
    source = str(state.get("capture_source", state.get("adapter_type", "")) or "")
    command_valid, command_missing = _complete_command_state(state.get("command_state"))
    actual_valid, actual_missing = _complete_actual_state(state.get("actual_joint_state"))
    pose_empty = all(state.get(key) in (None, [], {}) for key in ("root_pose", "root_velocity", "joint_pos", "joint_vel", "joint_names"))
    if _no_sim_marker(state):
        missing = [
            key
            for key in ("root_pose", "root_velocity", "joint_pos", "joint_vel", "joint_names")
            if state.get(key) in (None, [], {})
        ]
        return {
            "valid": False,
            "classification": PLACEHOLDER_NO_SIM,
            "missing_fields": missing + actual_missing,
            "invalid_fields": [],
            "source": source or "NullSimRobotAdapter",
            "joint_count": 0,
            "reason": "state is a NullSim/no-sim placeholder and is not pose-restore eligible",
            "joint_reorder_indices": [],
        }
    if pose_empty and command_valid:
        return {
            "valid": False,
            "classification": COMMAND_ONLY,
            "missing_fields": ["root_pose", "root_velocity", "joint_pos", "joint_vel", "joint_names"],
            "invalid_fields": [],
            "source": source,
            "joint_count": 0,
            "reason": "command_state is present but a full Isaac pose is absent",
            "joint_reorder_indices": [],
        }

    missing: list[str] = []
    invalid: list[str] = []
    root_pose = _finite_row(state.get("root_pose"), width=7)
    root_velocity = _finite_row(state.get("root_velocity"), width=6)
    names_value = state.get("joint_names")
    names = [str(name) for name in list(names_value or [])] if isinstance(names_value, (list, tuple)) else []
    joint_pos = _finite_row(state.get("joint_pos"))
    joint_vel = _finite_row(state.get("joint_vel"))
    if state.get("root_pose") is None:
        missing.append("root_pose")
    elif root_pose is None:
        invalid.append("root_pose")
    if state.get("root_velocity") is None:
        missing.append("root_velocity")
    elif root_velocity is None:
        invalid.append("root_velocity")
    if not names:
        missing.append("joint_names")
    elif len(names) != len(set(names)):
        invalid.append("joint_names.duplicate")
    if state.get("joint_pos") is None:
        missing.append("joint_pos")
    elif joint_pos is None:
        invalid.append("joint_pos")
    if state.get("joint_vel") is None:
        missing.append("joint_vel")
    elif joint_vel is None:
        invalid.append("joint_vel")
    if names and joint_pos is not None and len(joint_pos) != len(names):
        invalid.append("joint_pos.count")
    if names and joint_vel is not None and len(joint_vel) != len(names):
        invalid.append("joint_vel.count")
    reorder: list[int] = list(range(len(names)))
    expected = [str(name) for name in list(expected_joint_names or [])]
    if expected:
        if set(names) != set(expected) or len(names) != len(expected):
            invalid.append("joint_names.articulation_mismatch")
        else:
            reorder = [names.index(name) for name in expected]
    if not command_valid:
        missing.extend(command_missing)
    if not actual_valid:
        missing.extend(actual_missing)
    valid = not missing and not invalid
    reason = "full Isaac pose checkpoint is valid" if valid else "; ".join(missing + invalid)
    return {
        "valid": valid,
        "classification": FULL_VALID if valid else INVALID,
        "missing_fields": sorted(set(missing)),
        "invalid_fields": sorted(set(invalid)),
        "source": source,
        "joint_count": len(names),
        "reason": reason,
        "joint_reorder_indices": reorder if valid else [],
    }


def _quaternion_orientation_error_deg(left: list[float], right: list[float]) -> float:
    q1 = left[3:7]
    q2 = right[3:7]
    n1 = math.sqrt(sum(value * value for value in q1))
    n2 = math.sqrt(sum(value * value for value in q2))
    if n1 <= 1.0e-12 or n2 <= 1.0e-12:
        return float("inf")
    dot = abs(sum(a * b for a, b in zip(q1, q2)) / (n1 * n2))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def verify_restored_full_sim_pose(
    expected_state: Any,
    measured_state: Any,
    expected_joint_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    expected_validation = validate_full_sim_pose_state(expected_state, expected_joint_names)
    names = list(expected_joint_names or (expected_state.get("joint_names", []) if isinstance(expected_state, dict) else []))
    measured_validation = validate_full_sim_pose_state(measured_state, names)
    result = {
        "verified": False,
        "expected_validation": expected_validation,
        "measured_validation": measured_validation,
        "root_position_error_m": None,
        "root_orientation_error_deg": None,
        "servo_joint_position_max_error_deg": None,
        "wheel_joint_position_max_error_rad": None,
        "root_position_tolerance_m": ROOT_POSITION_TOLERANCE_M,
        "root_orientation_tolerance_deg": ROOT_ORIENTATION_TOLERANCE_DEG,
        "servo_joint_position_tolerance_deg": SERVO_JOINT_POSITION_TOLERANCE_DEG,
        "wheel_joint_position_tolerance_rad": WHEEL_JOINT_POSITION_TOLERANCE_RAD,
        "reason": "",
    }
    if not expected_validation["valid"] or not measured_validation["valid"]:
        result["reason"] = "expected or measured state is not FULL_VALID"
        return result
    expected_root = _finite_row(expected_state.get("root_pose"), width=7) or []
    measured_root = _finite_row(measured_state.get("root_pose"), width=7) or []
    root_position_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(expected_root[:3], measured_root[:3])))
    orientation_error = _quaternion_orientation_error_deg(expected_root, measured_root)
    expected_names = [str(name) for name in list(expected_state.get("joint_names", []) or [])]
    measured_names = [str(name) for name in list(measured_state.get("joint_names", []) or [])]
    expected_pos = _finite_row(expected_state.get("joint_pos")) or []
    measured_pos = _finite_row(measured_state.get("joint_pos")) or []
    expected_by_name = {name: expected_pos[index] for index, name in enumerate(expected_names)}
    measured_by_name = {name: measured_pos[index] for index, name in enumerate(measured_names)}
    servo_errors = [
        abs(measured_by_name[name] - expected_by_name[name])
        for name in SERVO_JOINT_NAMES
        if name in expected_by_name and name in measured_by_name
    ]
    wheel_errors = [
        abs(measured_by_name[name] - expected_by_name[name])
        for name in WHEEL_JOINT_NAMES
        if name in expected_by_name and name in measured_by_name
    ]
    servo_max_deg = math.degrees(max(servo_errors or [0.0]))
    wheel_max_rad = max(wheel_errors or [0.0])
    result.update(
        root_position_error_m=root_position_error,
        root_orientation_error_deg=orientation_error,
        servo_joint_position_max_error_deg=servo_max_deg,
        wheel_joint_position_max_error_rad=wheel_max_rad,
    )
    result["verified"] = bool(
        root_position_error <= ROOT_POSITION_TOLERANCE_M
        and orientation_error <= ROOT_ORIENTATION_TOLERANCE_DEG
        and servo_max_deg <= SERVO_JOINT_POSITION_TOLERANCE_DEG
        and wheel_max_rad <= WHEEL_JOINT_POSITION_TOLERANCE_RAD
    )
    result["reason"] = "pose verified" if result["verified"] else "pose error exceeds configured tolerance"
    return result
