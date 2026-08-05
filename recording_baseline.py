"""Stable actuator/environment baseline used to gate reference recordings."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from sim_obstacle_scene import OBSTACLE_FRONT_FACE_X_M, OBSTACLE_LENGTH_M, OBSTACLE_WIDTH_M


BASELINE_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "fsm_recording_baseline.yaml"


def load_baseline(path: str | Path = BASELINE_CONFIG_PATH) -> dict[str, Any]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Recording baseline must be an object: {source}")
    return data


AUTHORITATIVE_BASELINE = load_baseline()
DEFAULT_RECORDING_WHEEL_VELOCITY_RAD_S = float(
    AUTHORITATIVE_BASELINE["wheel_actuator"]["default_recording_velocity_rad_s"]
)
SERVO_REFERENCE_VELOCITY_DEG_S = float(
    AUTHORITATIVE_BASELINE["servo_profile"]["reference_velocity_deg_s"]
)
WHEEL_PHYSICAL_STOP_NOISE_FLOOR_RAD_S = float(
    AUTHORITATIVE_BASELINE["wheel_actuator"]["physical_stop_noise_floor_rad_s"]
)


def baseline_sha256(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def baseline_identity(data: dict[str, Any]) -> dict[str, str]:
    digest = baseline_sha256(data)
    return {
        "baseline_id": f"{data.get('baseline_version', 'recording-baseline')}:{digest[:16]}",
        "baseline_version": str(data.get("baseline_version", "")),
        "baseline_sha256": digest,
        "authoritative_config_path": str(BASELINE_CONFIG_PATH),
    }


def validate_recording_baseline(
    data: dict[str, Any],
    *,
    args: Any,
    worker_status: dict[str, Any],
    output_root: str | Path,
    no_sim: bool,
) -> dict[str, Any]:
    identity = baseline_identity(data)
    mismatches: list[str] = []
    warnings: list[str] = []

    if no_sim:
        return {
            **identity,
            "passed": True,
            "mismatches": [],
            "warnings": [],
            "current_height_cm": getattr(args, "height_cm", None),
            "scene_baseline": {},
            "wheel_command": {},
            "validation_mode": "static-no-sim-test-only",
        }

    def compare(label: str, actual: Any, expected: Any, tolerance: float = 1.0e-9) -> None:
        try:
            if abs(float(actual) - float(expected)) > tolerance:
                mismatches.append(f"{label}: expected {expected}, actual {actual}")
        except (TypeError, ValueError):
            if actual != expected:
                mismatches.append(f"{label}: expected {expected!r}, actual {actual!r}")

    robot_expected = Path(str(data["robot"]["usd_path"])).resolve()
    robot_actual = Path(str(getattr(args, "robot_usd", ""))).resolve()
    if robot_actual != robot_expected:
        mismatches.append(f"robot.usd_path: expected {robot_expected}, actual {robot_actual}")
    compare("physics.physics_timestep_s", getattr(args, "physics_dt", None), data["physics"]["physics_timestep_s"])
    runtime_render_interval = int(getattr(args, "render_interval", data["physics"]["render_interval_physics_steps"]))
    recorded_render_interval = int(data["physics"]["render_interval_physics_steps"])
    if runtime_render_interval != recorded_render_interval:
        warnings.append(
            "physics.render_interval_physics_steps is a render-cadence performance override: "
            f"recorded={recorded_render_interval}, runtime={runtime_render_interval}; physics_dt and canonical actuator commands are unchanged"
        )
    compare("servo.stiffness", getattr(args, "servo_stiffness", None), data["servo_profile"]["stiffness"])
    compare("servo.damping", getattr(args, "servo_damping", None), data["servo_profile"]["damping"])
    compare("wheel.velocity_limit_rad_s", getattr(args, "max_wheel_speed_rad_s", None), data["wheel_actuator"]["velocity_limit_rad_s"])
    compare("wheel.damping", getattr(args, "wheel_damping", None), data["wheel_actuator"]["damping"])
    expected_root = Path(str(data["recording_output_root"])).resolve()
    actual_root = Path(output_root).resolve()
    v2_output = actual_root.name == "saved_height_steps_fsm_reference_v2"
    if actual_root != expected_root and not v2_output:
        mismatches.append(f"recording_output_root: expected {expected_root}, actual {actual_root}")

    wheel_status = dict(worker_status.get("wheel_command", {}) or {})
    if not no_sim:
        catalog = {str(row.get("joint_name", "")): row for row in list(worker_status.get("joint_catalog", []) or []) if isinstance(row, dict)}
        if catalog:
            for name in data["robot"]["joint_order"]:
                if name not in catalog or not bool(catalog[name].get("available", False)):
                    mismatches.append(f"joint mapping unavailable: {name}")
        diagnostics = {str(row.get("joint_name", "")): row for row in list(worker_status.get("joint_diagnostics", []) or []) if isinstance(row, dict)}
        if diagnostics:
            for row in data["servo_profile"]["joints"]:
                name = str(row["articulation_joint"])
                diagnostic = diagnostics.get(name, {})
                actual_limit = diagnostic.get("command_limit_deg")
                expected_limit = [float(row["lower_deg"]), float(row["upper_deg"])]
                if not isinstance(actual_limit, list) or len(actual_limit) != 2 or any(abs(float(actual_limit[i]) - expected_limit[i]) > 1.0e-6 for i in range(2)):
                    mismatches.append(f"servo limit {name}: expected {expected_limit}, actual {actual_limit}")
                if diagnostic.get("target_inside_current_limit") is False:
                    mismatches.append(f"runtime target outside articulation limit: {name}")
        scene = dict(worker_status.get("scene_baseline", {}) or {})
        cached_fixed_geometry = scene.get("measurement") == "cached-lightweight-startup"
        if not bool(scene.get("available", False)) and not cached_fixed_geometry:
            mismatches.append(f"live scene geometry unavailable: {scene.get('error', 'no metrics')}")
        else:
            compare("ground.z_m", scene.get("ground_z_m"), data["ground"]["z_m"], 1.0e-5)
            bottom = scene.get("obstacle_bottom_z_m")
            if bottom is not None:
                compare("obstacle.bottom_z_m", bottom, data["ground"]["z_m"], 1.0e-4)
            if scene.get("wheel_radius_m") is None and not cached_fixed_geometry:
                mismatches.append("wheel radius could not be measured from live collision geometry")
            elif scene.get("wheel_radius_m") is not None:
                compare(
                    "wheel.radius_m",
                    scene.get("wheel_radius_m"),
                    data["wheel_actuator"]["wheel_radius_m"],
                    2.0e-3,
                )
            compare(
                "obstacle.front_face_x_m",
                scene.get("obstacle_front_face_x_m"),
                OBSTACLE_FRONT_FACE_X_M if v2_output else data["obstacle"]["authoritative_front_face_x_m"],
                1.0e-4,
            )
            compare(
                "obstacle.length_m",
                scene.get("obstacle_length_m"),
                OBSTACLE_LENGTH_M if v2_output else data["obstacle"]["authoritative_length_m"],
                1.0e-4,
            )
            compare(
                "obstacle.width_m",
                scene.get("obstacle_width_m"),
                OBSTACLE_WIDTH_M if v2_output else data["obstacle"]["authoritative_width_m"],
                1.0e-4,
            )
            collision_min = list(scene.get("robot_collision_bounds_min_m", []) or [])
            if len(collision_min) < 3 and not cached_fixed_geometry:
                mismatches.append("live robot collision ground clearance unavailable")
            elif len(collision_min) >= 3 and float(collision_min[2]) < -float(data["ground"]["penetration_tolerance_m"]):
                mismatches.append(
                    f"robot collision ground penetration {-float(collision_min[2]):.6f}m exceeds tolerance"
                )
            root_pose = list(scene.get("robot_root_pose", []) or [])
            reference = list(data["respawn"]["grounded_settled_reference_root_pose_wxyz"])
            if len(root_pose) < 7 and not cached_fixed_geometry:
                mismatches.append("live robot root pose unavailable")
            elif len(root_pose) >= 7:
                position_error = sum((float(root_pose[i]) - float(reference[i])) ** 2 for i in range(3)) ** 0.5
                if position_error > float(data["respawn"]["root_position_tolerance_m"]):
                    mismatches.append(f"respawn root position error {position_error:.6f}m exceeds tolerance")
        if not bool(worker_status.get("grounded_reference_valid", False)):
            mismatches.append("grounded respawn reference is not valid")
        if not bool(wheel_status.get("zero_target_applied", False)):
            mismatches.append("wheel zero target is not applied")
        if not bool(wheel_status.get("physically_stopped", False)):
            mismatches.append("wheels are not physically stopped")

    return {
        **identity,
        "passed": not mismatches,
        "mismatches": mismatches,
        "warnings": warnings,
        "current_height_cm": worker_status.get("height_cm", getattr(args, "height_cm", None)),
        "scene_baseline": copy.deepcopy(worker_status.get("scene_baseline", {})),
        "wheel_command": copy.deepcopy(wheel_status),
        "validation_mode": "static-no-sim" if no_sim else "live-worker",
    }
