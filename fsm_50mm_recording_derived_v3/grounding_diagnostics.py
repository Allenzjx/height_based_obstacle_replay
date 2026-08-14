"""Streaming, fail-closed evidence for formal clean-reset grounding probes.

The module is deliberately independent of Isaac imports.  It reads the live
objects passed by the shared adapter callback, writes one durable JSONL row per
physics tick, and never mutates robot state or command targets.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from telemetry.exporters import strict_json_dumps


LEG_TO_WHEEL_BODY = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}


def _first_row(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value[0].detach().cpu().tolist())
    except Exception:
        try:
            return list(value[0])
        except Exception:
            return []


def _finite_vector(value: Any, expected: int) -> tuple[list[float], str]:
    try:
        values = [
            float(item.item() if hasattr(item, "item") else item)
            for item in list(value)
        ]
    except Exception as exc:
        return [], f"vector conversion failed: {type(exc).__name__}: {exc}"
    if len(values) != expected:
        return values, f"expected {expected} values, got {len(values)}"
    if not all(math.isfinite(item) for item in values):
        return values, "vector contains non-finite values"
    return values, ""


def _root_pose(adapter: Any) -> tuple[list[float], str]:
    row = _first_row(
        getattr(getattr(getattr(adapter, "robot", None), "data", None), "root_pose_w", None)
    )
    return _finite_vector(row, 7)


def _roll_pitch_from_wxyz(quaternion: Sequence[float]) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    return float(roll), float(pitch)


def _contact_snapshot(
    scene_handle: Any,
    *,
    dt: float,
    active_force_threshold_n: float = 1.0e-6,
) -> dict[str, Any]:
    sensor = getattr(scene_handle, "contact_sensor", None)
    if sensor is None:
        return {
            "valid": False,
            "error": "formal contact sensor is unavailable",
            "wheel_net_forces_w": {},
            "nonwheel_contacts": [],
            "nonwheel_net_forces_w": {},
        }
    try:
        sensor.update(float(dt), force_recompute=True)
    except Exception as exc:
        return {
            "valid": False,
            "error": f"contact sensor update failed: {type(exc).__name__}: {exc}",
            "wheel_net_forces_w": {},
            "nonwheel_contacts": [],
            "nonwheel_net_forces_w": {},
        }
    body_names = [str(name) for name in (getattr(sensor, "body_names", []) or [])]
    raw_forces = getattr(getattr(sensor, "data", None), "net_forces_w", None)
    raw_shape = getattr(raw_forces, "shape", None)
    if raw_shape is not None:
        try:
            shape = tuple(int(value) for value in raw_shape)
        except Exception:
            shape = ()
        if shape != (1, len(body_names), 3):
            return {
                "valid": False,
                "error": (
                    "net_forces_w shape must be exactly "
                    f"(1, {len(body_names)}, 3), got {shape}"
                ),
                "wheel_net_forces_w": {},
                "nonwheel_contacts": [],
                "nonwheel_net_forces_w": {},
            }
    else:
        try:
            environment_rows = list(raw_forces)
        except Exception:
            environment_rows = []
        if len(environment_rows) != 1:
            return {
                "valid": False,
                "error": (
                    "net_forces_w must contain exactly one environment row; "
                    f"got {len(environment_rows)}"
                ),
                "wheel_net_forces_w": {},
                "nonwheel_contacts": [],
                "nonwheel_net_forces_w": {},
            }
    rows = _first_row(raw_forces)
    if not body_names or len(rows) != len(body_names):
        return {
            "valid": False,
            "error": (
                "net_forces_w/body_names layout mismatch: "
                f"rows={len(rows)} names={len(body_names)}"
            ),
            "wheel_net_forces_w": {},
            "nonwheel_contacts": [],
            "nonwheel_net_forces_w": {},
        }
    vectors: dict[str, list[float]] = {}
    errors: list[str] = []
    for index, name in enumerate(body_names):
        vector, error = _finite_vector(rows[index], 3)
        if error:
            errors.append(f"{name}: {error}")
        else:
            vectors[name] = vector
    normalized = {
        name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower(): name
        for name in body_names
    }
    if len(normalized) != len(body_names):
        errors.append("contact sensor body names are not unique by leaf name")
    wheel_forces: dict[str, list[float]] = {}
    wheel_body_names: set[str] = set()
    for leg, expected in LEG_TO_WHEEL_BODY.items():
        matches = [
            name
            for name in body_names
            if name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
            == expected
        ]
        if len(matches) != 1 or matches[0] not in vectors:
            errors.append(
                f"{leg}/{expected} resolved {len(matches)} finite body rows; expected one"
            )
            continue
        wheel_body_names.add(matches[0])
        wheel_forces[leg] = list(vectors[matches[0]])
    nonwheel_vectors = {
        name: vector
        for name, vector in vectors.items()
        if name not in wheel_body_names
    }
    nonwheel_contacts = []
    for name, vector in nonwheel_vectors.items():
        total = math.sqrt(sum(component * component for component in vector))
        if total > float(active_force_threshold_n):
            nonwheel_contacts.append(
                {
                    "body_name": name,
                    "net_force_w": list(vector),
                    "total_force_n": float(total),
                    "active": True,
                    "source": "isaaclab.ContactSensor.net_forces_w",
                }
            )
    filtered_nonwheel = getattr(sensor, "nonwheel_obstacle_observations", None)
    if callable(filtered_nonwheel):
        try:
            observation_rows = list(
                filtered_nonwheel(
                    env_id=0,
                    force_threshold_n=float(active_force_threshold_n),
                )
                or []
            )
        except Exception as exc:
            errors.append(
                "non-wheel obstacle contact read failed: "
                f"{type(exc).__name__}: {exc}"
            )
            observation_rows = []
        if not observation_rows:
            errors.append("non-wheel obstacle contact layout is empty")
        for raw in observation_rows:
            if isinstance(raw, Mapping):
                row = dict(raw)
            elif callable(getattr(raw, "as_dict", None)):
                row = dict(raw.as_dict())
            else:
                errors.append("non-wheel obstacle contact row is not serializable")
                continue
            name = str(row.get("body_name", "") or "")
            if not name or row.get("force_valid") is not True:
                errors.append(f"{name or '<unnamed>'}: non-wheel force is invalid")
                continue
            vector, vector_error = _finite_vector(
                list(row.get("total_force_w", []) or []), 3
            )
            if vector_error:
                errors.append(f"{name}: {vector_error}")
                continue
            nonwheel_vectors[name] = vector
            if row.get("active") is True:
                nonwheel_contacts.append(
                    {
                        **row,
                        "body_name": name,
                        "net_force_w": vector,
                        "total_force_n": float(
                            math.sqrt(sum(component * component for component in vector))
                        ),
                        "active": True,
                        "source": "filtered non-wheel obstacle ContactSensor",
                    }
                )
    return {
        "valid": not errors and len(wheel_forces) == 4,
        "error": "; ".join(errors),
        "source": "isaaclab.ContactSensor.net_forces_w",
        "body_names": body_names,
        "wheel_net_forces_w": wheel_forces,
        "nonwheel_contacts": nonwheel_contacts,
        "nonwheel_net_forces_w": nonwheel_vectors,
    }


def penetration_snapshot(adapter: Any) -> dict[str, Any]:
    """Read fail-closed live wheel/ground penetration without mutating state."""

    try:
        diagnostics = dict(adapter.validate_robot_ground_contact(apply_correction=False) or {})
    except Exception as exc:
        return {
            "valid": False,
            "error": f"ground penetration inspection failed: {type(exc).__name__}: {exc}",
            "maximum_collision_penetration_m": float("nan"),
            "wheel_penetration_m": {},
        }
    wheel_rows = list(diagnostics.get("wheels", []) or [])
    wheel_penetration: dict[str, float] = {}
    wheel_clearance: dict[str, float] = {}
    errors: list[str] = []
    expected_wheels = set(WHEEL_JOINT_NAMES)
    if len(wheel_rows) != len(expected_wheels):
        errors.append(
            f"expected {len(expected_wheels)} wheel diagnostic rows, got {len(wheel_rows)}"
        )
    for row in wheel_rows:
        if not isinstance(row, Mapping):
            errors.append("ground diagnostic wheel row is not a mapping")
            continue
        name = str(row.get("wheel_name", "") or "")
        joint_name = str(row.get("joint_name", "") or "")
        if not name or name not in expected_wheels:
            errors.append(f"unexpected/empty wheel_name: {name!r}")
            continue
        if name in wheel_penetration:
            errors.append(f"duplicate wheel_name: {name}")
            continue
        if joint_name and joint_name != name:
            errors.append(
                f"{name}: joint_name mismatch {joint_name!r}"
            )
            continue
        if row.get("bounds_valid") is not True:
            errors.append(f"{name}: bounds_valid is not true")
        if row.get("bounds_finite") is not True:
            errors.append(f"{name}: bounds_finite is not true")
        if not str(row.get("bounds_source", "") or ""):
            errors.append(f"{name}: bounds_source is unavailable")
        if str(row.get("collision_resolution_state", "") or "") != "OK":
            errors.append(f"{name}: collision_resolution_state is not OK")
        try:
            value = float(row.get("collision_penetration_m"))
        except (TypeError, ValueError):
            errors.append(f"{name or '<unnamed>'}: penetration is unavailable")
            continue
        if not math.isfinite(value):
            errors.append(f"{name or '<unnamed>'}: penetration is non-finite")
            continue
        try:
            clearance = float(
                row.get(
                    "collision_ground_clearance_m",
                    row.get("clearance_m"),
                )
            )
        except (TypeError, ValueError):
            errors.append(f"{name}: collision clearance is unavailable")
            continue
        if not math.isfinite(clearance):
            errors.append(f"{name}: collision clearance is non-finite")
            continue
        wheel_penetration[name] = value
        wheel_clearance[name] = clearance
    try:
        maximum = float(diagnostics.get("maximum_collision_penetration_m"))
    except (TypeError, ValueError):
        maximum = float("nan")
    if not math.isfinite(maximum):
        errors.append("maximum collision penetration is unavailable/non-finite")
    elif set(wheel_penetration) == expected_wheels:
        computed_maximum = max(wheel_penetration.values(), default=0.0)
        if not math.isclose(
            maximum,
            computed_maximum,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            errors.append(
                "maximum collision penetration does not match wheel rows: "
                f"{maximum:.9g} != {computed_maximum:.9g}"
            )
    valid = bool(
        diagnostics.get("checked", False)
        and set(wheel_penetration) == expected_wheels
        and not diagnostics.get("missing_collision_wheels")
        and not diagnostics.get("unresolved_collision_wheels")
        and not errors
    )
    return {
        "valid": valid,
        "error": "; ".join(errors),
        "classification": str(diagnostics.get("classification", "") or ""),
        "physical_ground_safe": diagnostics.get("physical_ground_safe"),
        "maximum_collision_penetration_m": maximum,
        "wheel_penetration_m": wheel_penetration,
        "wheel_clearance_m": wheel_clearance,
        "missing_collision_wheels": list(
            diagnostics.get("missing_collision_wheels", []) or []
        ),
        "unresolved_collision_wheels": list(
            diagnostics.get("unresolved_collision_wheels", []) or []
        ),
    }


def enrich_grounding_tick(
    adapter: Any,
    scene_handle: Any,
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach pose, contact, and penetration evidence to one settle tick."""

    enriched = dict(frame)
    root_pose, root_error = _root_pose(adapter)
    roll = float("nan")
    pitch = float("nan")
    if not root_error:
        roll, pitch = _roll_pitch_from_wxyz(root_pose[3:7])
    root_velocity, velocity_error = _finite_vector(
        list(enriched.get("root_velocity", []) or []), 6
    )
    dt = float(enriched.get("physics_dt_s", 0.0) or 0.0)
    contact = _contact_snapshot(scene_handle, dt=dt)
    penetration = penetration_snapshot(adapter)
    errors = [
        error
        for error in (
            root_error,
            velocity_error,
            str(enriched.get("root_velocity_evidence_error", "") or ""),
            str(enriched.get("joint_state_evidence_error", "") or ""),
            str(enriched.get("wheel_target_evidence_error", "") or ""),
            str(contact.get("error", "") or ""),
            str(penetration.get("error", "") or ""),
        )
        if error
    ]
    enriched.update(
        {
            "root_pose_w": root_pose,
            "root_position_w": root_pose[:3] if len(root_pose) == 7 else [],
            "root_orientation_wxyz": root_pose[3:7] if len(root_pose) == 7 else [],
            "root_linear_velocity_w": root_velocity[:3] if len(root_velocity) == 6 else [],
            "root_angular_velocity_w": root_velocity[3:6] if len(root_velocity) == 6 else [],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "wheel_net_forces_w": dict(contact.get("wheel_net_forces_w", {}) or {}),
            "wheel_force_evidence_valid": bool(contact.get("valid", False)),
            "wheel_force_evidence_error": str(contact.get("error", "") or ""),
            "nonwheel_contacts": list(contact.get("nonwheel_contacts", []) or []),
            "nonwheel_net_forces_w": dict(
                contact.get("nonwheel_net_forces_w", {}) or {}
            ),
            "penetration": penetration,
            "diagnostic_evidence_valid": bool(
                not errors
                and enriched.get("root_velocity_evidence_valid") is True
                and enriched.get("joint_state_evidence_valid") is True
                and enriched.get("wheel_target_evidence_valid") is True
                and contact.get("valid") is True
                and penetration.get("valid") is True
            ),
            "diagnostic_evidence_error": "; ".join(dict.fromkeys(errors)),
        }
    )
    return enriched


class GroundingTraceWriter:
    """Durably append every observed tick while retaining rows for analysis."""

    def __init__(self, path: Path, *, adapter: Any, scene_handle: Any) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.scene_handle = scene_handle
        self.rows: list[dict[str, Any]] = []
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")

    def __call__(self, frame: dict[str, Any]) -> dict[str, Any]:
        enriched = enrich_grounding_tick(self.adapter, self.scene_handle, frame)
        self.rows.append(enriched)
        self._stream.write(
            strict_json_dumps(enriched, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return enriched

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


def write_grounding_trace_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "local_tick",
        "sim_step",
        "sim_time_s",
        "physics_dt_s",
        "root_x_m",
        "root_y_m",
        "root_z_m",
        "root_linear_velocity_w_json",
        "root_angular_velocity_w_json",
        "roll_rad",
        "pitch_rad",
        "max_servo_speed_rad_s",
        "max_wheel_speed_rad_s",
        "stable_count",
        "strict_tick_stable",
        "rolling_window_metrics_json",
        "joint_state_json",
        "wheel_net_forces_w_json",
        "nonwheel_contacts_json",
        "penetration_json",
        "diagnostic_evidence_valid",
        "diagnostic_evidence_error",
    )
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            position = list(row.get("root_position_w", []) or [])
            joint_names = sorted(
                set(dict(row.get("joint_position_by_name", {}) or {}))
                | set(dict(row.get("joint_velocity_by_name", {}) or {}))
                | set(dict(row.get("joint_position_target_by_name", {}) or {}))
            )
            joints = {
                name: {
                    "q_rad": dict(row.get("joint_position_by_name", {}) or {}).get(name),
                    "qd_rad_s": dict(row.get("joint_velocity_by_name", {}) or {}).get(name),
                    "q_target_rad": dict(
                        row.get("joint_position_target_by_name", {}) or {}
                    ).get(name),
                    "q_target_minus_q_rad": dict(
                        row.get("joint_target_minus_position_by_name", {}) or {}
                    ).get(name),
                    "wheel_velocity_target_rad_s": dict(
                        row.get("wheel_target_velocity_by_name", {}) or {}
                    ).get(name),
                    "wheel_velocity_target_readback_rad_s": dict(
                        row.get("wheel_target_readback_velocity_by_name", {})
                        or {}
                    ).get(name),
                    "wheel_target_command_to_readback_error_rad_s": dict(
                        row.get(
                            "wheel_target_command_to_readback_error_by_name",
                            {},
                        )
                        or {}
                    ).get(name),
                }
                for name in joint_names
            }
            writer.writerow(
                {
                    "local_tick": row.get("local_tick"),
                    "sim_step": row.get("sim_step"),
                    "sim_time_s": row.get("sim_time_s"),
                    "physics_dt_s": row.get("physics_dt_s"),
                    "root_x_m": position[0] if len(position) == 3 else None,
                    "root_y_m": position[1] if len(position) == 3 else None,
                    "root_z_m": position[2] if len(position) == 3 else None,
                    "root_linear_velocity_w_json": json.dumps(
                        row.get("root_linear_velocity_w", []), sort_keys=True
                    ),
                    "root_angular_velocity_w_json": json.dumps(
                        row.get("root_angular_velocity_w", []), sort_keys=True
                    ),
                    "roll_rad": row.get("roll_rad"),
                    "pitch_rad": row.get("pitch_rad"),
                    "max_servo_speed_rad_s": row.get("servo_joint_speed"),
                    "max_wheel_speed_rad_s": row.get("wheel_joint_speed"),
                    "stable_count": row.get("stable_count"),
                    "strict_tick_stable": row.get("strict_tick_stable"),
                    "rolling_window_metrics_json": json.dumps(
                        row.get("rolling_window_metrics", {}), sort_keys=True
                    ),
                    "joint_state_json": json.dumps(joints, sort_keys=True),
                    "wheel_net_forces_w_json": json.dumps(
                        row.get("wheel_net_forces_w", {}), sort_keys=True
                    ),
                    "nonwheel_contacts_json": json.dumps(
                        row.get("nonwheel_contacts", []), sort_keys=True
                    ),
                    "penetration_json": json.dumps(
                        row.get("penetration", {}), sort_keys=True
                    ),
                    "diagnostic_evidence_valid": row.get(
                        "diagnostic_evidence_valid"
                    ),
                    "diagnostic_evidence_error": row.get(
                        "diagnostic_evidence_error", ""
                    ),
                }
            )


def analyze_joint_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    servo_threshold_rad_s: float,
    wheel_threshold_rad_s: float,
) -> dict[str, Any]:
    """Return per-joint velocity and reset target-readback mismatch evidence."""

    joint_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    analysis: dict[str, Any] = {}
    for name in joint_names:
        velocities: list[float] = []
        target_errors: list[float] = []
        command_readback_errors: list[float] = []
        for row in rows:
            try:
                velocity = float(
                    dict(row.get("joint_velocity_by_name", {}) or {})[name]
                )
                target_error = float(
                    dict(
                        row.get("joint_target_minus_position_by_name", {}) or {}
                    )[name]
                )
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(velocity) and math.isfinite(target_error):
                velocities.append(velocity)
                target_errors.append(target_error)
            if name in SERVO_JOINT_NAMES:
                try:
                    command_error = float(
                        dict(
                            row.get(
                                "servo_command_to_readback_error_by_name", {}
                            )
                            or {}
                        )[name]
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(command_error):
                    command_readback_errors.append(command_error)
        threshold = (
            float(servo_threshold_rad_s)
            if name in SERVO_JOINT_NAMES
            else float(wheel_threshold_rad_s)
        )
        analysis[name] = {
            "group": "servo" if name in SERVO_JOINT_NAMES else "wheel",
            "sample_count": len(velocities),
            "threshold_rad_s": threshold,
            "peak_abs_velocity_rad_s": max(
                (abs(value) for value in velocities), default=float("nan")
            ),
            "final_velocity_rad_s": (
                velocities[-1] if velocities else float("nan")
            ),
            "ticks_over_velocity_threshold": sum(
                abs(value) > threshold for value in velocities
            ),
            "initial_target_minus_q_rad": (
                target_errors[0] if target_errors else float("nan")
            ),
            "peak_abs_target_minus_q_rad": max(
                (abs(value) for value in target_errors), default=float("nan")
            ),
            "final_target_minus_q_rad": (
                target_errors[-1] if target_errors else float("nan")
            ),
            "initial_command_to_readback_error_rad": (
                command_readback_errors[0]
                if command_readback_errors
                else None
            ),
            "peak_abs_command_to_readback_error_rad": (
                max(abs(value) for value in command_readback_errors)
                if command_readback_errors
                else None
            ),
            "final_command_to_readback_error_rad": (
                command_readback_errors[-1]
                if command_readback_errors
                else None
            ),
        }
    final_offending_servos = [
        name
        for name in SERVO_JOINT_NAMES
        if analysis[name]["sample_count"]
        and abs(float(analysis[name]["final_velocity_rad_s"]))
        > float(servo_threshold_rad_s)
    ]
    return {
        "schema_version": "fsm50.grounding_joint_velocity_analysis.v1",
        "tick_count": len(rows),
        "servo_threshold_rad_s": float(servo_threshold_rad_s),
        "wheel_threshold_rad_s": float(wheel_threshold_rad_s),
        "all_ticks_diagnostic_evidence_valid": bool(
            rows
            and all(row.get("diagnostic_evidence_valid") is True for row in rows)
        ),
        "final_offending_servos": final_offending_servos,
        "joints": analysis,
    }
