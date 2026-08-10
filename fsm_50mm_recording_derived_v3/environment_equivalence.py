"""Isaac-free environment fingerprinting and A/A--A/B equivalence checks.

The static fingerprint records what the formal source tree requests.  It does
not claim that PhysX created the same runtime objects: version, USD/prim, and
physics readbacks remain explicitly pending until a simulator worker supplies
them.  The comparison helpers are intentionally standard-library only so the
equivalence gate can be tested without importing or starting Isaac Sim.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from command_model import (
    HIP_LIMIT_DEG,
    JOINT_COMMAND_SIGN,
    KNEE_LIMIT_DEG,
    SERVO_JOINT_NAMES,
    WHEEL_FORWARD_SIGN,
    WHEEL_JOINT_NAMES,
)
from motion_speed import load_motion_reference
from sim_obstacle_scene import (
    DEFAULT_ROBOT_USD_PATH,
    GROUND_PRIM_PATH,
    OBSTACLE_PRIM_PATH,
    ROBOT_PRIM_PATH,
    SERVO_TORQUE_LIMIT_NM,
    SimSceneConfig,
)


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
ENVIRONMENT_REFERENCE_PATH = PROJECT_ROOT / "config" / "environment_reference.yaml"
MOTION_REFERENCE_PATH = PROJECT_ROOT / "config" / "real_robot_motion_reference.yaml"
LEGACY_BASELINE_PATH = PROJECT_ROOT / "config" / "fsm_recording_baseline.yaml"
FSM50_CONFIG_PATH = MODULE_ROOT / "fsm50_config.yaml"

UNKNOWN_RUNTIME_VERSION = "unknown_pending_runtime_readback"

# These are the only scene-configuration fields an instrumented A/B run may
# change.  Sensor observations are passed separately to the normalizer and may
# differ; no physical setting with a sensor-like name is silently discarded.
ALLOWED_INSTRUMENTATION_CONFIG_FIELDS = frozenset(
    {"telemetry_contact_sensors_enabled", "contact_sensor_factory"}
)

TRAJECTORY_METRICS = (
    "root_trajectory",
    "joint_trajectory",
    "wheel_rotation",
    "wheel_travel",
    "final_pose",
    "obstacle_geometry",
    "contact_class",
    "contact_force",
)

# Floors cover serialization/readback quantization only.  A/A self-error is
# multiplied below and is normally the governing tolerance.
DEFAULT_ABSOLUTE_FLOORS: dict[str, float] = {
    "root_trajectory": 1.0e-6,
    "joint_trajectory": 1.0e-6,
    "wheel_rotation": 1.0e-6,
    "wheel_travel": 1.0e-7,
    "final_pose": 1.0e-6,
    "obstacle_geometry": 1.0e-7,
    "contact_class": 0.0,
    "contact_force": 1.0e-6,
}

FORMAL_SOURCE_RELATIVE_PATHS = (
    "sim_obstacle_scene.py",
    "sim_robot_adapter.py",
    "command_model.py",
    "motion_speed.py",
    "playback.py",
    "sequence_model.py",
    "config/environment_reference.yaml",
    "config/real_robot_motion_reference.yaml",
    "config/fsm_recording_baseline.yaml",
    "fsm_50mm_recording_derived_v3/recording_fast_plan.py",
    "fsm_50mm_recording_derived_v3/fsm50_config.yaml",
    "fsm_50mm_recording_derived_v3/run_fsm50.py",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of *path*, raising if the formal input is absent."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _source_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = completed.stdout.strip()
        return value if len(value) == 40 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _callable_name(value: Any) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
    return f"{module}.{qualname}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if callable(value):
        return {"callable": _callable_name(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(vars(value).items())
            if not str(key).startswith("_")
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _object_mapping(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: getattr(config, field.name) for field in fields(config)}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {key: value for key, value in vars(config).items() if not str(key).startswith("_")}
    raise TypeError("configuration must be a dataclass, mapping, or object with attributes")


def _recursive_differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Return exact differing JSON paths; physical config is not fuzzy."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_recursive_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}.length" if path else "length"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child = f"{path}[{index}]" if path else f"[{index}]"
            differences.extend(_recursive_differences(left_item, right_item, child))
        return differences
    return [] if left == right else [path or "$" ]


def normalize_physical_scene_config(config: Any) -> dict[str, Any]:
    """Remove only the two explicitly authorized instrumentation switches."""

    values = _object_mapping(config)
    return _json_safe(
        {
            key: value
            for key, value in values.items()
            if key not in ALLOWED_INSTRUMENTATION_CONFIG_FIELDS
        }
    )


def normalize_instrumentation_config(
    config: Any,
    *,
    sensor_readback: Any | None = None,
) -> dict[str, Any]:
    """Separate physical config, allowed instrumentation, and sensor output."""

    values = _object_mapping(config)
    return {
        "physical_config": normalize_physical_scene_config(config),
        "instrumentation": {
            "telemetry_contact_sensors_enabled": bool(
                values.get("telemetry_contact_sensors_enabled", False)
            ),
            "contact_sensor_factory": _callable_name(values.get("contact_sensor_factory")),
        },
        "sensor_readback": _json_safe(sensor_readback),
    }


def compare_instrumentation_configs(
    baseline_config: Any,
    instrumented_config: Any,
    *,
    baseline_sensor_readback: Any | None = None,
    instrumented_sensor_readback: Any | None = None,
) -> dict[str, Any]:
    """Verify that instrumentation changed no physical scene configuration."""

    baseline = normalize_instrumentation_config(
        baseline_config, sensor_readback=baseline_sensor_readback
    )
    instrumented = normalize_instrumentation_config(
        instrumented_config, sensor_readback=instrumented_sensor_readback
    )
    physical_differences = _recursive_differences(
        baseline["physical_config"], instrumented["physical_config"]
    )
    instrumentation_differences = _recursive_differences(
        baseline["instrumentation"], instrumented["instrumentation"]
    )
    sensor_differences = _recursive_differences(
        baseline["sensor_readback"], instrumented["sensor_readback"]
    )
    return {
        "schema_version": "fsm50.instrumentation_equivalence.v1",
        "ok": not physical_differences,
        "allowed_config_fields": sorted(ALLOWED_INSTRUMENTATION_CONFIG_FIELDS),
        "physical_config": baseline["physical_config"],
        "physical_differences": physical_differences,
        "allowed_instrumentation_differences": instrumentation_differences,
        "allowed_sensor_readback_differences": sensor_differences,
        "baseline": baseline,
        "instrumented": instrumented,
    }


def build_static_environment_fingerprint(
    *,
    project_root: str | Path = PROJECT_ROOT,
    robot_usd_path: str | Path | None = None,
    obstacle_height_m: float = 0.050,
    source_commit: str | None = None,
    runtime_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a hash-backed fingerprint of the formal 50 mm environment.

    Static values are taken from the same importable constants/config files as
    production.  Live USD geometry, grounded pose, and engine versions remain
    pending readback and therefore cannot by themselves establish equivalence.
    """

    root = Path(project_root).resolve()
    environment_reference = _read_json(root / "config" / "environment_reference.yaml")
    legacy_baseline = _read_json(root / "config" / "fsm_recording_baseline.yaml")
    motion_reference = load_motion_reference(root / "config" / "real_robot_motion_reference.yaml")
    scene = SimSceneConfig(obstacle_height_m=float(obstacle_height_m))

    usd_path = Path(robot_usd_path) if robot_usd_path is not None else Path(DEFAULT_ROBOT_USD_PATH)
    usd_path = usd_path.resolve()
    source_files: dict[str, dict[str, Any]] = {}
    for relative in FORMAL_SOURCE_RELATIVE_PATHS:
        source = root / relative
        source_files[relative] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }

    front_x = float(environment_reference["obstacle_front_face_x_m"])
    length = float(environment_reference["obstacle_length_m"])
    width = float(environment_reference["obstacle_width_m"])
    ground_z = float(scene.ground_z_m)
    height = float(obstacle_height_m)
    center_x = front_x + 0.5 * length
    center_y = float(environment_reference.get("obstacle_center_y_m", 0.0))
    center_z = ground_z + 0.5 * height

    legacy_servo = float(legacy_baseline["servo_profile"]["reference_velocity_deg_s"])
    legacy_render = int(legacy_baseline["physics"]["render_interval_physics_steps"])
    selected_servo = float(motion_reference.servo_reference_velocity_deg_s)
    selected_render = int(scene.render_interval)
    legacy_ground_pose = legacy_baseline.get("respawn", {}).get(
        "grounded_settled_reference_root_pose_wxyz"
    )
    wheel_radius = legacy_baseline.get("wheel_actuator", {}).get("wheel_radius_m")

    versions = {
        "python": platform.python_version(),
        "isaac_sim": UNKNOWN_RUNTIME_VERSION,
        "isaac_lab": UNKNOWN_RUNTIME_VERSION,
        "physx": UNKNOWN_RUNTIME_VERSION,
        "torch": UNKNOWN_RUNTIME_VERSION,
    }
    if runtime_versions is not None:
        for key, value in runtime_versions.items():
            versions[str(key)] = _json_safe(value)

    servo_limits = {
        name: list(KNEE_LIMIT_DEG if name.endswith("_knee") else HIP_LIMIT_DEG)
        for name in SERVO_JOINT_NAMES
    }
    initial_servos = {name: 0.0 for name in SERVO_JOINT_NAMES}
    initial_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}

    return {
        "schema_version": "fsm50.environment_static_fingerprint.v1",
        "status": "STATIC_LOCK_RUNTIME_READBACK_PENDING",
        "environment_equivalent": False,
        "source_commit": source_commit or _source_commit(root),
        "source_commit_scope": "git HEAD; per-file hashes bind dirty working-tree content",
        "source_files": source_files,
        "environment_reference": {
            "path": str((root / "config" / "environment_reference.yaml").resolve()),
            "profile_id": environment_reference.get("profile_id"),
        },
        "motion_reference": motion_reference.to_dict(),
        "robot_usd": {
            "path": str(usd_path),
            "sha256": sha256_file(usd_path),
            "size_bytes": usd_path.stat().st_size,
        },
        "prims": {
            "robot": ROBOT_PRIM_PATH,
            "ground": GROUND_PRIM_PATH,
            "obstacle": OBSTACLE_PRIM_PATH,
        },
        "initial_state": {
            "raw_spawn_root_pose_wxyz": [
                0.0,
                0.0,
                float(scene.spawn_z),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "servo_command_deg": initial_servos,
            "wheel_target_rad_s": initial_wheels,
            "formal_policy": "clean reset, zero velocities, settle, then lock live grounded readback",
            "legacy_grounded_pose_metadata_wxyz": legacy_ground_pose,
            "grounded_root_pose_runtime_readback": None,
        },
        "obstacle": {
            "height_m": height,
            "front_face_x_m": front_x,
            "rear_face_x_m": front_x + length,
            "length_m": length,
            "width_m": width,
            "bottom_z_m": ground_z,
            "top_z_m": ground_z + height,
            "center_xyz_m": [center_x, center_y, center_z],
            "bounds_min_xyz_m": [front_x, center_y - 0.5 * width, ground_z],
            "bounds_max_xyz_m": [front_x + length, center_y + 0.5 * width, ground_z + height],
            "kinematic": True,
            "disable_gravity": True,
            "contact_offset_m": 0.005,
            "rest_offset_m": 0.0,
            "material": {
                "static_friction": 1.20,
                "dynamic_friction": 1.00,
                "restitution": 0.0,
                "friction_combine_mode": "max",
                "restitution_combine_mode": "min",
            },
        },
        "ground": {
            "z_m": ground_z,
            "size_xy_m": [6.0, 6.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "material": {
                "static_friction": 1.25,
                "dynamic_friction": 1.05,
                "restitution": 0.0,
                "friction_combine_mode": "max",
                "restitution_combine_mode": "min",
            },
        },
        "physics": {
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "physics_dt_s": float(scene.physics_dt),
            "render_interval_physics_steps": selected_render,
            "render_dt_s": float(scene.physics_dt) * selected_render,
            "substeps": 1,
            "solver_type": "PhysX-default",
            "solver_position_iterations": 8,
            "solver_velocity_iterations": 2,
            "robot_max_depenetration_velocity_m_s": 1.0,
            "obstacle_contact_offset_m": 0.005,
            "obstacle_rest_offset_m": 0.0,
        },
        "actuators": {
            "servo": {
                "mode": "implicit_position",
                "effort_limit_nm": float(SERVO_TORQUE_LIMIT_NM),
                "velocity_limit_deg_s": motion_reference.servo_velocity_limit_deg_s,
                "reference_velocity_deg_s": selected_servo,
                "stiffness": float(scene.servo_stiffness),
                "damping": float(scene.servo_damping),
                "armature": 0.005,
                "command_sign": dict(JOINT_COMMAND_SIGN),
                "command_limits_deg": servo_limits,
            },
            "wheel": {
                "mode": "implicit_velocity",
                "effort_limit_nm": None,
                "velocity_limit_rad_s": float(motion_reference.wheel_velocity_limit_rad_s),
                "reference_velocity_rad_s": float(motion_reference.wheel_reference_velocity_rad_s),
                "stiffness": 0.0,
                "damping": float(scene.wheel_damping),
                "armature": 0.002,
                "radius_m": wheel_radius,
                "radius_source": "legacy live_collision_local_mesh metadata; revalidate by runtime readback",
                "forward_direction": dict(WHEEL_FORWARD_SIGN),
            },
        },
        "fast_replay": {
            "profile_requested": "fast",
            "profile_normalized": "motion_only",
            "servo_reference_velocity_deg_s": selected_servo,
            "servo_effective_velocity_deg_s": selected_servo
            if motion_reference.servo_velocity_limit_deg_s is None
            else min(selected_servo, float(motion_reference.servo_velocity_limit_deg_s)),
            "servo_delta_rule": "maximum absolute command-space joint delta in the semantic motion group",
            "servo_duration_rule": "servo_delta_deg / effective_servo_velocity_deg_s",
            "wheel_duration_rule": "recorded semantic base interval only while any wheel target is active",
            "segment_duration_rule": "max(servo_duration_s, wheel_duration_s, explicit_hold_s)",
            "runtime_recompute": "servo duration is recomputed at segment start from the adapter's current logical command",
            "implicit_timestamp_gaps": "removed in Fast/motion_only profile",
        },
        "runtime_selection": {
            "servo_reference_velocity_deg_s": selected_servo,
            "render_interval_physics_steps": selected_render,
            "expected_current_values_match": {
                "servo_150_deg_s": math.isclose(selected_servo, 150.0, rel_tol=0.0, abs_tol=1.0e-12),
                "render_interval_8": selected_render == 8,
            },
        },
        "legacy_metadata_differences": {
            "servo_reference_velocity_deg_s": {
                "legacy_baseline_metadata": legacy_servo,
                "selected_runtime": selected_servo,
                "classification": "metadata_only_not_runtime_selection",
            },
            "render_interval_physics_steps": {
                "legacy_baseline_metadata": legacy_render,
                "selected_runtime": selected_render,
                "classification": "metadata_only_not_runtime_selection",
            },
        },
        "instrumentation_policy": {
            "allowed_config_differences": sorted(ALLOWED_INSTRUMENTATION_CONFIG_FIELDS),
            "sensor_readback_differences_allowed": True,
            "all_physical_config_differences": "forbidden_fail_closed",
        },
        "runtime_versions": versions,
        "runtime_readback_required": [
            "runtime_versions",
            "loaded_stage_and_usd_identifiers",
            "articulation_root_and_joint_order",
            "live_grounded_root_pose",
            "live_obstacle_and_ground_bounds",
            "live_material_and_solver_properties",
            "live_actuator_properties",
            "live_wheel_collision_radius_and_axis",
        ],
    }


def _flatten_metric(
    value: Any,
    *,
    path: str = "$",
) -> tuple[dict[str, float], dict[str, Any], list[str]]:
    numeric: dict[str, float] = {}
    categorical: dict[str, Any] = {}
    errors: list[str] = []

    def visit(item: Any, item_path: str) -> None:
        if isinstance(item, bool) or item is None or isinstance(item, str):
            categorical[item_path] = item
            return
        if isinstance(item, (int, float)):
            number = float(item)
            if not math.isfinite(number):
                errors.append(f"{item_path}: non-finite numeric value")
            else:
                numeric[item_path] = number
            return
        if isinstance(item, Mapping):
            if not item:
                categorical[f"{item_path}.__empty_mapping__"] = True
            for key in sorted(item, key=str):
                visit(item[key], f"{item_path}.{key}")
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if not item:
                categorical[f"{item_path}.__empty_sequence__"] = True
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")
            return
        errors.append(f"{item_path}: unsupported value type {type(item).__name__}")

    visit(value, path)
    return numeric, categorical, errors


def _numeric_metric_error(left: Any, right: Any) -> dict[str, Any]:
    left_numeric, left_categories, left_errors = _flatten_metric(left)
    right_numeric, right_categories, right_errors = _flatten_metric(right)
    errors = left_errors + right_errors
    if set(left_numeric) != set(right_numeric):
        errors.append("numeric shape/key mismatch")
    if left_categories != right_categories:
        errors.append("categorical structure/metadata mismatch")
    if not left_numeric:
        errors.append("metric has no numeric samples")
    if errors:
        return {"valid": False, "error": None, "max_abs_error": None, "rms_error": None, "sample_count": 0, "reasons": errors}
    deltas = [abs(left_numeric[key] - right_numeric[key]) for key in sorted(left_numeric)]
    rms = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
    return {
        "valid": True,
        "error": max(deltas),
        "max_abs_error": max(deltas),
        "rms_error": rms,
        "sample_count": len(deltas),
        "reasons": [],
    }


def _categorical_metric_error(left: Any, right: Any) -> dict[str, Any]:
    left_numeric, left_categories, left_errors = _flatten_metric(left)
    right_numeric, right_categories, right_errors = _flatten_metric(right)
    errors = left_errors + right_errors
    if left_numeric or right_numeric:
        errors.append("contact_class must contain categorical values only")
    if set(left_categories) != set(right_categories):
        errors.append("categorical shape/key mismatch")
    if not left_categories:
        errors.append("metric has no categorical samples")
    if errors:
        return {"valid": False, "error": None, "mismatch_rate": None, "mismatch_count": None, "sample_count": 0, "reasons": errors}
    keys = sorted(left_categories)
    mismatch_count = sum(left_categories[key] != right_categories[key] for key in keys)
    mismatch_rate = mismatch_count / len(keys)
    return {
        "valid": True,
        "error": mismatch_rate,
        "mismatch_rate": mismatch_rate,
        "mismatch_count": mismatch_count,
        "sample_count": len(keys),
        "reasons": [],
    }


def _pair_metric_error(metric: str, left: Any, right: Any) -> dict[str, Any]:
    return (
        _categorical_metric_error(left, right)
        if metric == "contact_class"
        else _numeric_metric_error(left, right)
    )


def compare_trajectory_equivalence(
    baseline_a1: Mapping[str, Any],
    baseline_a2: Mapping[str, Any],
    instrumented_b: Mapping[str, Any],
    *,
    self_error_multiplier: float = 3.0,
    absolute_floors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare duplicate A runs with B and derive every tolerance from A/A.

    B is compared to both A runs and the worse error is used.  Missing metrics,
    shape drift, non-finite samples, and invalid tolerances fail closed.
    """

    multiplier = float(self_error_multiplier)
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("self_error_multiplier must be finite and >= 1.0")
    floors = dict(DEFAULT_ABSOLUTE_FLOORS)
    if absolute_floors is not None:
        unknown = sorted(set(absolute_floors) - set(TRAJECTORY_METRICS))
        if unknown:
            raise ValueError(f"Unknown trajectory tolerance metric(s): {unknown}")
        floors.update({key: float(value) for key, value in absolute_floors.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in floors.values()):
        raise ValueError("absolute tolerance floors must be finite and non-negative")

    metrics: dict[str, Any] = {}
    failures: list[str] = []
    for metric in TRAJECTORY_METRICS:
        missing = [
            label
            for label, run in (("A1", baseline_a1), ("A2", baseline_a2), ("B", instrumented_b))
            if metric not in run
        ]
        if missing:
            metrics[metric] = {
                "ok": False,
                "reason": f"missing metric in {', '.join(missing)}",
                "baseline_self_error": None,
                "instrumented_error": None,
                "tolerance": None,
            }
            failures.append(metric)
            continue

        aa = _pair_metric_error(metric, baseline_a1[metric], baseline_a2[metric])
        ab1 = _pair_metric_error(metric, baseline_a1[metric], instrumented_b[metric])
        ab2 = _pair_metric_error(metric, baseline_a2[metric], instrumented_b[metric])
        valid = bool(aa["valid"] and ab1["valid"] and ab2["valid"])
        if valid:
            aa_error = float(aa["error"])
            ab_error = max(float(ab1["error"]), float(ab2["error"]))
            tolerance = max(float(floors[metric]), multiplier * aa_error)
            ok = ab_error <= tolerance
            ratio = 0.0 if ab_error == 0.0 else (math.inf if tolerance == 0.0 else ab_error / tolerance)
            reason = "within A/A-derived tolerance" if ok else "B exceeds A/A-derived tolerance"
        else:
            aa_error = None
            ab_error = None
            tolerance = None
            ratio = None
            ok = False
            reason = "invalid/misaligned metric data"
        metrics[metric] = {
            "ok": ok,
            "reason": reason,
            "baseline_self_error": aa_error,
            "instrumented_error": ab_error,
            "tolerance": tolerance,
            "absolute_floor": float(floors[metric]),
            "self_error_multiplier": multiplier,
            "overrun_ratio": ratio,
            "A1_vs_A2": aa,
            "A1_vs_B": ab1,
            "A2_vs_B": ab2,
        }
        if not ok:
            failures.append(metric)

    return {
        "schema_version": "fsm50.trajectory_equivalence.v1",
        "comparison_design": "A/A baseline self-error sets tolerance; B is checked against both A runs",
        "ok": not failures,
        "fail_closed": True,
        "self_error_multiplier": multiplier,
        "metrics": metrics,
        "failed_metrics": failures,
    }


def write_environment_equivalence_report(
    path: str | Path,
    *,
    fingerprint: Mapping[str, Any],
    instrumentation_comparison: Mapping[str, Any] | None = None,
    trajectory_comparison: Mapping[str, Any] | None = None,
    runtime_readback: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write an atomic JSON report without upgrading a pending result to PASS."""

    checks = [
        check
        for check in (instrumentation_comparison, trajectory_comparison)
        if check is not None
    ]
    runtime_failed = bool(
        runtime_readback is not None
        and (
            runtime_readback.get("ok") is False
            or str(runtime_readback.get("status", "")).upper() == "FAIL"
        )
    )
    runtime_complete = bool(
        runtime_readback is not None
        and runtime_readback.get("readback_complete") is True
    )
    runtime_ok = bool(
        runtime_readback is not None and runtime_readback.get("ok") is True
    )
    runtime_complete_but_not_ok = bool(runtime_complete and not runtime_ok)
    conversion_present = bool(
        isinstance(extra, Mapping) and "artifact_conversion" in extra
    )
    conversion = extra.get("artifact_conversion") if conversion_present else None
    conversion_is_mapping = isinstance(conversion, Mapping)
    conversion_ok = bool(
        conversion_is_mapping and conversion.get("ok") is True
    )
    conversion_failed = bool(
        conversion_present
        and (
            not conversion_is_mapping
            or conversion.get("ok") is False
        )
    )
    if (
        any(check.get("ok") is not True for check in checks)
        or runtime_failed
        or runtime_complete_but_not_ok
        or conversion_failed
    ):
        status = "FAIL"
    elif (
        instrumentation_comparison is None
        or trajectory_comparison is None
        or not runtime_complete
        or not runtime_ok
        or not conversion_ok
    ):
        status = "PENDING_RUNTIME_A_B"
    else:
        status = "PASS"
    payload = {
        "schema_version": "fsm50.environment_equivalence_report.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "environment_equivalent": status == "PASS",
        "static_fingerprint": _json_safe(fingerprint),
        "instrumentation_comparison": _json_safe(instrumentation_comparison),
        "trajectory_comparison": _json_safe(trajectory_comparison),
        "runtime_readback": _json_safe(runtime_readback),
        "extra": _json_safe(extra),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


# Clear aliases for callers that use the shorter nouns.
build_static_fingerprint = build_static_environment_fingerprint
build_environment_fingerprint = build_static_environment_fingerprint
compare_instrumentation_equivalence = compare_instrumentation_configs
compare_trajectories = compare_trajectory_equivalence
write_json_report = write_environment_equivalence_report


__all__ = [
    "ALLOWED_INSTRUMENTATION_CONFIG_FIELDS",
    "DEFAULT_ABSOLUTE_FLOORS",
    "FORMAL_SOURCE_RELATIVE_PATHS",
    "TRAJECTORY_METRICS",
    "build_environment_fingerprint",
    "build_static_environment_fingerprint",
    "build_static_fingerprint",
    "compare_instrumentation_configs",
    "compare_instrumentation_equivalence",
    "compare_trajectories",
    "compare_trajectory_equivalence",
    "normalize_instrumentation_config",
    "normalize_physical_scene_config",
    "sha256_file",
    "write_environment_equivalence_report",
    "write_json_report",
]
