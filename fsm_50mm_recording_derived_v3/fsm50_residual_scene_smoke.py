"""Executable, SHA-bound real-Isaac smoke for the formal residual scene.

Importing this module is Isaac-free.  The CLI imports and owns AppLauncher only
inside :func:`main`; callers must execute it with the configured ``env_isaaclab``
Python.  Tests exercise the request/result contracts without launching Kit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from motion_speed import load_motion_reference
from fsm_50mm_recording_derived_v3.fsm50_residual_scene import (
    DEFAULT_ENV_SPACING_M,
    DIRECT_RL_DECIMATION,
    PHYSICS_DT_S,
    RENDER_INTERVAL_PHYSICS_STEPS,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    build_isaaclab_scene_bundle,
    load_formal_scene_spec,
    spawn_global_scene_assets,
)


REQUEST_SCHEMA_VERSION = "fsm50-residual-scene-smoke-request-v1"
RESULT_SCHEMA_VERSION = "fsm50-residual-scene-smoke-result-v1"

SMOKE_SOURCE_PATH = Path(__file__).resolve()
REPLAY_ROOT = SMOKE_SOURCE_PATH.parents[1]
MOTION_SPEED_SOURCE_PATH = REPLAY_ROOT / "motion_speed.py"
MOTION_REFERENCE_PATH = REPLAY_ROOT / "config" / "real_robot_motion_reference.yaml"

FROZEN_SCENE_SOURCE_SHA256 = "76d1117f2ad780226e4b24421d79a76af5efcb0178c3800a78318ebf594390c0"
FROZEN_SCENE_MANIFEST_SHA256 = "af3e05b4294eebdfa14e1688187da89c82826941565f815f6c478a46715ee891"
FROZEN_MOTION_SPEED_SOURCE_SHA256 = "5e41f5501b841ff7764ee6c2a9975afb256d6c121d22519ffc38842737b49421"
FROZEN_MOTION_REFERENCE_SHA256 = "143bbb71fe50fcde22360901552f36b397b9d102bde21172213bf3a316333747"

SMOKE_NUM_ENVS = 2
SMOKE_ENV_SPACING_M = DEFAULT_ENV_SPACING_M
SMOKE_PHYSICS_STEPS = 24
SERVO_REFERENCE_VELOCITY_DEG_S = 150.0
SERVO_MAX_DELTA_DEG_PER_STEP = SERVO_REFERENCE_VELOCITY_DEG_S * PHYSICS_DT_S
RATE_LIMIT_PROBE_REQUEST_DEG = 10.0

WHEEL_BODY_NAMES = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
JOINT_COMMAND_SIGN = (
    1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
)
WHEEL_FORWARD_SIGN = (-1.0, 1.0, -1.0, 1.0)

_REQUEST_ENVELOPE_KEYS = frozenset({"schema_version", "payload", "payload_sha256"})
_REQUEST_PAYLOAD_KEYS = frozenset(
    {
        "request_id",
        "result_path",
        "device",
        "headless_required",
        "num_envs",
        "env_spacing_m",
        "physics_steps",
        "servo_reference_velocity_deg_s",
        "expected_smoke_source_sha256",
        "expected_scene_source_sha256",
        "expected_scene_manifest_sha256",
        "expected_robot_usd_sha256",
        "expected_motion_speed_source_sha256",
        "expected_motion_reference_sha256",
    }
)
_RESULT_ENVELOPE_KEYS = frozenset({"schema_version", "payload", "payload_sha256"})
_RESULT_PAYLOAD_KEYS = frozenset(
    {
        "request_id",
        "request_path",
        "request_file_sha256",
        "request_payload_sha256",
        "status",
        "completed_utc",
        "smoke_source_sha256",
        "scene_source_sha256",
        "scene_manifest_sha256",
        "robot_usd_sha256",
        "motion_speed_source_sha256",
        "motion_reference_sha256",
        "runtime",
        "closure",
        "error",
    }
)
_SUCCESS_RUNTIME_KEYS = frozenset(
    {
        "device",
        "num_envs",
        "env_spacing_m",
        "physics_dt_s",
        "direct_rl_decimation",
        "render_interval_physics_steps",
        "physics_steps_requested",
        "physics_steps_completed",
        "scene_env_prim_paths",
        "scene_prim_validation",
        "collision_filter",
        "servo_joint_ids",
        "servo_joint_names",
        "wheel_joint_ids",
        "wheel_joint_names",
        "wheel_body_ids",
        "wheel_body_names",
        "actuation",
        "finite_state",
        "package_versions",
        "process_id",
    }
)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEVICE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")


class SceneSmokeContractError(ValueError):
    """A request, result, or runtime identity violated the smoke contract."""


@dataclass(frozen=True)
class ValidatedSceneSmokeRequest:
    request_id: str
    result_path: Path
    device: str
    headless_required: bool
    num_envs: int
    env_spacing_m: float
    physics_steps: int
    servo_reference_velocity_deg_s: float
    expected_smoke_source_sha256: str
    expected_scene_source_sha256: str
    expected_scene_manifest_sha256: str
    expected_robot_usd_sha256: str
    expected_motion_speed_source_sha256: str
    expected_motion_reference_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class LoadedSceneSmokeRequest:
    request: ValidatedSceneSmokeRequest
    request_path: Path
    request_file_sha256: str


def build_smoke_request(
    *,
    request_id: str,
    result_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build the one allowed formal smoke request and bind all source identities."""

    spec = load_formal_scene_spec()
    payload = {
        "request_id": request_id,
        "result_path": str(Path(result_path)),
        "device": device,
        "headless_required": True,
        "num_envs": SMOKE_NUM_ENVS,
        "env_spacing_m": SMOKE_ENV_SPACING_M,
        "physics_steps": SMOKE_PHYSICS_STEPS,
        "servo_reference_velocity_deg_s": SERVO_REFERENCE_VELOCITY_DEG_S,
        "expected_smoke_source_sha256": _sha256_file(SMOKE_SOURCE_PATH),
        "expected_scene_source_sha256": FROZEN_SCENE_SOURCE_SHA256,
        "expected_scene_manifest_sha256": FROZEN_SCENE_MANIFEST_SHA256,
        "expected_robot_usd_sha256": spec.robot_usd_sha256,
        "expected_motion_speed_source_sha256": FROZEN_MOTION_SPEED_SOURCE_SHA256,
        "expected_motion_reference_sha256": FROZEN_MOTION_REFERENCE_SHA256,
    }
    envelope = _make_envelope(REQUEST_SCHEMA_VERSION, payload)
    validate_smoke_request(envelope)
    return envelope


def validate_smoke_request(
    envelope: Mapping[str, Any],
    *,
    request_path: str | Path | None = None,
    verify_files: bool = True,
) -> ValidatedSceneSmokeRequest:
    """Validate exact request keys, canonical payload SHA, paths, and frozen inputs."""

    row = _require_mapping("request envelope", envelope)
    _require_exact_keys("request envelope", row, _REQUEST_ENVELOPE_KEYS)
    if row["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise SceneSmokeContractError("request schema_version is invalid")
    payload = _require_mapping("request payload", row["payload"])
    _require_exact_keys("request payload", payload, _REQUEST_PAYLOAD_KEYS)
    payload_sha = _require_sha("request payload_sha256", row["payload_sha256"])
    if payload_sha != _sha256_payload(payload):
        raise SceneSmokeContractError("request payload SHA256 mismatch")

    request_id = _require_request_id(payload["request_id"])
    result_path = _require_absolute_json_path("result_path", payload["result_path"])
    if request_path is not None:
        source = Path(request_path).resolve()
        if not source.is_file():
            raise SceneSmokeContractError(f"request file is missing: {source}")
        if result_path == source:
            raise SceneSmokeContractError("result_path must differ from request_path")
    device = _require_device(payload["device"])
    if payload["headless_required"] is not True:
        raise SceneSmokeContractError("headless_required must be exactly true")
    if _require_exact_int("num_envs", payload["num_envs"]) != SMOKE_NUM_ENVS:
        raise SceneSmokeContractError(f"num_envs must be exactly {SMOKE_NUM_ENVS}")
    if _require_finite("env_spacing_m", payload["env_spacing_m"]) != SMOKE_ENV_SPACING_M:
        raise SceneSmokeContractError(f"env_spacing_m must be exactly {SMOKE_ENV_SPACING_M}")
    if _require_exact_int("physics_steps", payload["physics_steps"]) != SMOKE_PHYSICS_STEPS:
        raise SceneSmokeContractError(f"physics_steps must be exactly {SMOKE_PHYSICS_STEPS}")
    if (
        _require_finite("servo_reference_velocity_deg_s", payload["servo_reference_velocity_deg_s"])
        != SERVO_REFERENCE_VELOCITY_DEG_S
    ):
        raise SceneSmokeContractError(
            f"servo_reference_velocity_deg_s must be exactly {SERVO_REFERENCE_VELOCITY_DEG_S}"
        )

    smoke_sha = _require_sha("expected_smoke_source_sha256", payload["expected_smoke_source_sha256"])
    scene_sha = _require_sha("expected_scene_source_sha256", payload["expected_scene_source_sha256"])
    scene_manifest_sha = _require_sha(
        "expected_scene_manifest_sha256", payload["expected_scene_manifest_sha256"]
    )
    robot_sha = _require_sha("expected_robot_usd_sha256", payload["expected_robot_usd_sha256"])
    motion_source_sha = _require_sha(
        "expected_motion_speed_source_sha256", payload["expected_motion_speed_source_sha256"]
    )
    motion_reference_sha = _require_sha(
        "expected_motion_reference_sha256", payload["expected_motion_reference_sha256"]
    )
    if scene_sha != FROZEN_SCENE_SOURCE_SHA256:
        raise SceneSmokeContractError("request is not bound to the frozen scene source")
    if scene_manifest_sha != FROZEN_SCENE_MANIFEST_SHA256:
        raise SceneSmokeContractError("request is not bound to the frozen scene manifest")
    if motion_source_sha != FROZEN_MOTION_SPEED_SOURCE_SHA256:
        raise SceneSmokeContractError("request is not bound to the frozen motion-speed source")
    if motion_reference_sha != FROZEN_MOTION_REFERENCE_SHA256:
        raise SceneSmokeContractError("request is not bound to the frozen motion reference")

    spec = load_formal_scene_spec(verify_files=verify_files)
    if spec.manifest_sha256 != scene_manifest_sha:
        raise SceneSmokeContractError("live scene manifest does not match request")
    if spec.robot_usd_sha256 != robot_sha:
        raise SceneSmokeContractError("live robot USD identity does not match request")
    if verify_files:
        _require_file_sha(SMOKE_SOURCE_PATH, smoke_sha, "smoke source")
        _require_file_sha(MOTION_SPEED_SOURCE_PATH, motion_source_sha, "motion-speed source")
        _require_file_sha(MOTION_REFERENCE_PATH, motion_reference_sha, "motion reference")

    return ValidatedSceneSmokeRequest(
        request_id=request_id,
        result_path=result_path,
        device=device,
        headless_required=True,
        num_envs=SMOKE_NUM_ENVS,
        env_spacing_m=SMOKE_ENV_SPACING_M,
        physics_steps=SMOKE_PHYSICS_STEPS,
        servo_reference_velocity_deg_s=SERVO_REFERENCE_VELOCITY_DEG_S,
        expected_smoke_source_sha256=smoke_sha,
        expected_scene_source_sha256=scene_sha,
        expected_scene_manifest_sha256=scene_manifest_sha,
        expected_robot_usd_sha256=robot_sha,
        expected_motion_speed_source_sha256=motion_source_sha,
        expected_motion_reference_sha256=motion_reference_sha,
        payload_sha256=payload_sha,
    )


def load_smoke_request(path: str | Path) -> LoadedSceneSmokeRequest:
    source = Path(path).resolve()
    if not source.is_file():
        raise SceneSmokeContractError(f"request file is missing: {source}")
    try:
        envelope = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SceneSmokeContractError(f"request JSON is unreadable: {exc}") from exc
    request = validate_smoke_request(envelope, request_path=source)
    return LoadedSceneSmokeRequest(request, source, _sha256_file(source))


def build_smoke_result(
    loaded: LoadedSceneSmokeRequest,
    *,
    status: str,
    runtime: Mapping[str, Any],
    closure: Mapping[str, Any],
    error: Mapping[str, Any] | None = None,
    completed_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate a request-bound PASS or FAIL result envelope."""

    if status not in {"PASS", "FAIL"}:
        raise SceneSmokeContractError("result status must be PASS or FAIL")
    payload = {
        "request_id": loaded.request.request_id,
        "request_path": str(loaded.request_path),
        "request_file_sha256": loaded.request_file_sha256,
        "request_payload_sha256": loaded.request.payload_sha256,
        "status": status,
        "completed_utc": completed_utc or datetime.now(timezone.utc).isoformat(),
        "smoke_source_sha256": loaded.request.expected_smoke_source_sha256,
        "scene_source_sha256": loaded.request.expected_scene_source_sha256,
        "scene_manifest_sha256": loaded.request.expected_scene_manifest_sha256,
        "robot_usd_sha256": loaded.request.expected_robot_usd_sha256,
        "motion_speed_source_sha256": loaded.request.expected_motion_speed_source_sha256,
        "motion_reference_sha256": loaded.request.expected_motion_reference_sha256,
        "runtime": dict(runtime),
        "closure": dict(closure),
        "error": None if error is None else dict(error),
    }
    envelope = _make_envelope(RESULT_SCHEMA_VERSION, payload)
    validate_smoke_result(envelope, loaded=loaded)
    return envelope


def validate_smoke_result(
    envelope: Mapping[str, Any],
    *,
    loaded: LoadedSceneSmokeRequest | None = None,
) -> dict[str, Any]:
    row = _require_mapping("result envelope", envelope)
    _require_exact_keys("result envelope", row, _RESULT_ENVELOPE_KEYS)
    if row["schema_version"] != RESULT_SCHEMA_VERSION:
        raise SceneSmokeContractError("result schema_version is invalid")
    payload = _require_mapping("result payload", row["payload"])
    _require_exact_keys("result payload", payload, _RESULT_PAYLOAD_KEYS)
    payload_sha = _require_sha("result payload_sha256", row["payload_sha256"])
    if payload_sha != _sha256_payload(payload):
        raise SceneSmokeContractError("result payload SHA256 mismatch")
    _require_request_id(payload["request_id"])
    _require_absolute_json_path("request_path", payload["request_path"])
    for name in (
        "request_file_sha256",
        "request_payload_sha256",
        "smoke_source_sha256",
        "scene_source_sha256",
        "scene_manifest_sha256",
        "robot_usd_sha256",
        "motion_speed_source_sha256",
        "motion_reference_sha256",
    ):
        _require_sha(name, payload[name])
    _require_aware_iso8601("completed_utc", payload["completed_utc"])
    status = payload["status"]
    if status not in {"PASS", "FAIL"}:
        raise SceneSmokeContractError("result status must be PASS or FAIL")
    runtime = _require_mapping("runtime", payload["runtime"])
    closure = _require_mapping("closure", payload["closure"])
    _require_exact_keys(
        "closure",
        closure,
        frozenset({"simulation_context_cleared", "application_close_requested"}),
    )
    if not all(isinstance(closure[name], bool) for name in closure):
        raise SceneSmokeContractError("closure values must be booleans")
    if status == "PASS":
        if payload["error"] is not None:
            raise SceneSmokeContractError("PASS result must have error=null")
        if closure != {
            "simulation_context_cleared": True,
            "application_close_requested": True,
        }:
            raise SceneSmokeContractError(
                "PASS result requires cleared SimulationContext and an application close request"
            )
        _validate_success_runtime(runtime, loaded.request if loaded is not None else None)
    else:
        error = _require_mapping("FAIL error", payload["error"])
        _require_exact_keys("FAIL error", error, frozenset({"type", "message", "traceback_sha256"}))
        if not isinstance(error["type"], str) or not error["type"]:
            raise SceneSmokeContractError("FAIL error type is invalid")
        if not isinstance(error["message"], str) or not error["message"]:
            raise SceneSmokeContractError("FAIL error message is invalid")
        _require_sha("FAIL traceback_sha256", error["traceback_sha256"])
    if loaded is not None:
        request = loaded.request
        exact = {
            "request_id": request.request_id,
            "request_path": str(loaded.request_path),
            "request_file_sha256": loaded.request_file_sha256,
            "request_payload_sha256": request.payload_sha256,
            "smoke_source_sha256": request.expected_smoke_source_sha256,
            "scene_source_sha256": request.expected_scene_source_sha256,
            "scene_manifest_sha256": request.expected_scene_manifest_sha256,
            "robot_usd_sha256": request.expected_robot_usd_sha256,
            "motion_speed_source_sha256": request.expected_motion_speed_source_sha256,
            "motion_reference_sha256": request.expected_motion_reference_sha256,
        }
        for name, expected in exact.items():
            if payload[name] != expected:
                raise SceneSmokeContractError(f"result {name} does not match request")
    return dict(payload)


def write_json_atomic(path: str | Path, envelope: Mapping[str, Any], *, refuse_existing: bool = True) -> None:
    """Write one canonical JSON artifact through a same-directory atomic replace."""

    target = Path(path).resolve()
    if target.suffix.lower() != ".json":
        raise SceneSmokeContractError("atomic JSON target must end in .json")
    if not target.parent.is_dir():
        raise SceneSmokeContractError(f"atomic JSON parent directory is missing: {target.parent}")
    if refuse_existing and target.exists():
        raise SceneSmokeContractError(f"atomic JSON target already exists: {target}")
    encoded = (json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse_existing and target.exists():
            raise SceneSmokeContractError(f"atomic JSON target appeared during write: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def rate_limit_probe(
    *,
    current_deg: float = 0.0,
    requested_deg: float = RATE_LIMIT_PROBE_REQUEST_DEG,
    dt_s: float = PHYSICS_DT_S,
    velocity_deg_s: float = SERVO_REFERENCE_VELOCITY_DEG_S,
) -> float:
    """Pure scalar form of the exact per-physics-step endpoint slew used below."""

    current = _require_finite("current_deg", current_deg)
    requested = _require_finite("requested_deg", requested_deg)
    dt = _require_finite("dt_s", dt_s)
    velocity = _require_finite("velocity_deg_s", velocity_deg_s)
    if dt <= 0.0 or velocity <= 0.0:
        raise SceneSmokeContractError("rate-limit dt and velocity must be positive")
    maximum = velocity * dt
    delta = min(max(requested - current, -maximum), maximum)
    return current + delta


def _validate_success_runtime(
    runtime: Mapping[str, Any], request: ValidatedSceneSmokeRequest | None
) -> None:
    _require_exact_keys("PASS runtime", runtime, _SUCCESS_RUNTIME_KEYS)
    expected_device = request.device if request is not None else runtime["device"]
    if runtime["device"] != expected_device:
        raise SceneSmokeContractError("PASS runtime device mismatches request")
    expected_num_envs = request.num_envs if request is not None else SMOKE_NUM_ENVS
    expected_steps = request.physics_steps if request is not None else SMOKE_PHYSICS_STEPS
    exact = {
        "num_envs": expected_num_envs,
        "env_spacing_m": SMOKE_ENV_SPACING_M,
        "physics_dt_s": PHYSICS_DT_S,
        "direct_rl_decimation": DIRECT_RL_DECIMATION,
        "render_interval_physics_steps": RENDER_INTERVAL_PHYSICS_STEPS,
        "physics_steps_requested": expected_steps,
        "physics_steps_completed": expected_steps,
        "servo_joint_names": list(SERVO_JOINT_NAMES),
        "wheel_joint_names": list(WHEEL_JOINT_NAMES),
        "wheel_body_names": list(WHEEL_BODY_NAMES),
    }
    for name, expected in exact.items():
        if runtime[name] != expected:
            raise SceneSmokeContractError(f"PASS runtime {name} is invalid")
    for name, expected_count in (
        ("servo_joint_ids", 8),
        ("wheel_joint_ids", 4),
        ("wheel_body_ids", 4),
    ):
        values = runtime[name]
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            or len(set(values)) != expected_count
        ):
            raise SceneSmokeContractError(f"PASS runtime {name} is invalid")
    if set(runtime["servo_joint_ids"]) & set(runtime["wheel_joint_ids"]):
        raise SceneSmokeContractError("servo and wheel joint IDs overlap")
    expected_env_paths = [f"/World/envs/env_{index}" for index in range(expected_num_envs)]
    if runtime["scene_env_prim_paths"] != expected_env_paths:
        raise SceneSmokeContractError("PASS runtime environment prim paths are invalid")
    prims = _require_mapping("scene_prim_validation", runtime["scene_prim_validation"])
    if prims.get("all_expected_prims_valid") is not True:
        raise SceneSmokeContractError("PASS runtime did not validate all scene prims")
    collision = _require_mapping("collision_filter", runtime["collision_filter"])
    if collision.get("clone_called") is not True or collision.get("filter_called") is not True:
        raise SceneSmokeContractError("PASS runtime clone/filter evidence is incomplete")
    if collision.get("global_collision_prim_paths") != ["/World/defaultGroundPlane"]:
        raise SceneSmokeContractError("PASS runtime collision-filter globals are invalid")
    actuation = _require_mapping("actuation", runtime["actuation"])
    required_actuation = {
        "zero_servo_command_verified",
        "zero_wheel_command_verified",
        "standing_target_verified_all_steps",
        "target_echo_verified_all_steps",
        "servo_target_write_count",
        "wheel_target_write_count",
        "servo_reference_velocity_deg_s",
        "max_delta_deg_per_physics_step",
        "rate_limit_probe_requested_deg",
        "rate_limit_probe_applied_deg",
        "rate_limit_probe_applied_to_robot",
        "max_abs_servo_target_echo_error_rad",
        "max_abs_wheel_target_echo_error_rad_s",
        "final_max_abs_standing_position_error_rad",
    }
    _require_exact_keys("actuation", actuation, frozenset(required_actuation))
    for name in (
        "zero_servo_command_verified",
        "zero_wheel_command_verified",
        "standing_target_verified_all_steps",
        "target_echo_verified_all_steps",
    ):
        if actuation[name] is not True:
            raise SceneSmokeContractError(f"PASS actuation {name} must be true")
    if actuation["rate_limit_probe_applied_to_robot"] is not False:
        raise SceneSmokeContractError("rate-limit probe must never be applied to the robot")
    if actuation["servo_target_write_count"] != expected_steps:
        raise SceneSmokeContractError("servo target write count is invalid")
    if actuation["wheel_target_write_count"] != expected_steps:
        raise SceneSmokeContractError("wheel target write count is invalid")
    numeric_exact = {
        "servo_reference_velocity_deg_s": SERVO_REFERENCE_VELOCITY_DEG_S,
        "max_delta_deg_per_physics_step": SERVO_MAX_DELTA_DEG_PER_STEP,
        "rate_limit_probe_requested_deg": RATE_LIMIT_PROBE_REQUEST_DEG,
        "rate_limit_probe_applied_deg": SERVO_MAX_DELTA_DEG_PER_STEP,
        "max_abs_servo_target_echo_error_rad": 0.0,
        "max_abs_wheel_target_echo_error_rad_s": 0.0,
    }
    for name, expected in numeric_exact.items():
        if _require_finite(name, actuation[name]) != expected:
            raise SceneSmokeContractError(f"PASS actuation {name} is invalid")
    _require_finite(
        "final_max_abs_standing_position_error_rad",
        actuation["final_max_abs_standing_position_error_rad"],
    )
    finite = _require_mapping("finite_state", runtime["finite_state"])
    if finite.get("all_finite") is not True or finite.get("physics_frames_checked") != expected_steps:
        raise SceneSmokeContractError("PASS finite-state evidence is invalid")
    versions = _require_mapping("package_versions", runtime["package_versions"])
    for name in ("isaaclab", "isaacsim", "torch"):
        if not isinstance(versions.get(name), str) or not versions[name]:
            raise SceneSmokeContractError(f"PASS package version {name} is missing")
    _require_exact_int("process_id", runtime["process_id"])


def _execute_real_isaac_smoke(
    request: ValidatedSceneSmokeRequest,
    *,
    simulation_app: Any,
) -> tuple[dict[str, Any], bool]:
    """Run the active-stage smoke.  AppLauncher must already own ``simulation_app``."""

    import torch  # type: ignore
    from isaaclab.assets import Articulation, RigidObject  # type: ignore
    from isaaclab.scene import InteractiveScene  # type: ignore
    from isaaclab.sim import SimulationContext  # type: ignore

    motion = load_motion_reference()
    if float(motion.servo_reference_velocity_deg_s) != request.servo_reference_velocity_deg_s:
        raise SceneSmokeContractError("live motion reference servo velocity is not exactly 150 deg/s")
    bundle = build_isaaclab_scene_bundle(
        num_envs=request.num_envs,
        env_spacing_m=request.env_spacing_m,
        device=request.device,
    )
    if bundle.spec.manifest_sha256 != request.expected_scene_manifest_sha256:
        raise SceneSmokeContractError("constructed bundle manifest mismatches request")
    if bundle.direct_rl_decimation != DIRECT_RL_DECIMATION:
        raise SceneSmokeContractError("constructed bundle decimation is not one")

    sim: Any | None = None
    interactive_scene: Any | None = None
    robot: Any | None = None
    obstacle: Any | None = None
    context_cleared = False
    runtime: dict[str, Any] = {}
    try:
        sim = SimulationContext(bundle.simulation_cfg)
        if float(sim.get_physics_dt()) != PHYSICS_DT_S:
            raise SceneSmokeContractError("live SimulationContext physics dt drifted")
        interactive_scene = InteractiveScene(bundle.interactive_scene_cfg)
        # InteractiveScene expands ``{ENV_REGEX_NS}`` only for asset configs
        # declared on its configclass.  This smoke registers the frozen bundle
        # manually, so it must perform the same exact expansion before spawning.
        robot_cfg = _expand_namespaced_asset_cfg(
            bundle.robot_cfg, interactive_scene.env_regex_ns, "robot"
        )
        obstacle_cfg = _expand_namespaced_asset_cfg(
            bundle.obstacle_cfg, interactive_scene.env_regex_ns, "obstacle"
        )
        robot = Articulation(robot_cfg)
        obstacle = RigidObject(obstacle_cfg)
        interactive_scene.articulations["robot"] = robot
        interactive_scene.rigid_objects["obstacle"] = obstacle
        spawn_global_scene_assets(bundle)
        interactive_scene.clone_environments(copy_from_source=False)
        interactive_scene.filter_collisions(
            global_prim_paths=list(bundle.global_collision_prim_paths)
        )
        sim.reset()
        interactive_scene.reset()
        interactive_scene.update(PHYSICS_DT_S)

        servo_ids, servo_names = _resolve_exact(
            robot.find_joints,
            SERVO_JOINT_NAMES,
            "servo joints",
        )
        wheel_ids, wheel_names = _resolve_exact(
            robot.find_joints,
            WHEEL_JOINT_NAMES,
            "wheel joints",
        )
        wheel_body_ids, wheel_body_names = _resolve_exact(
            robot.find_bodies,
            WHEEL_BODY_NAMES,
            "wheel bodies",
        )
        if set(servo_ids) & set(wheel_ids):
            raise SceneSmokeContractError("resolved servo and wheel joint IDs overlap")

        env_count = int(robot.data.default_joint_pos.shape[0])
        if env_count != request.num_envs:
            raise SceneSmokeContractError(
                f"robot instance count mismatch: expected {request.num_envs}, got {env_count}"
            )
        standing_raw_rad = robot.data.default_joint_pos[:, servo_ids].clone()
        command_sign = torch.tensor(JOINT_COMMAND_SIGN, dtype=standing_raw_rad.dtype, device=standing_raw_rad.device)
        requested_command_deg = torch.zeros_like(standing_raw_rad)
        applied_command_deg = torch.zeros_like(standing_raw_rad)
        wheel_command_physical = torch.zeros(
            (env_count, len(wheel_ids)), dtype=standing_raw_rad.dtype, device=standing_raw_rad.device
        )
        wheel_forward_sign = torch.tensor(
            WHEEL_FORWARD_SIGN,
            dtype=wheel_command_physical.dtype,
            device=wheel_command_physical.device,
        )
        raw_wheel_target = wheel_command_physical * wheel_forward_sign
        max_delta_deg = request.servo_reference_velocity_deg_s * PHYSICS_DT_S
        target_echo_ok = True
        standing_target_ok = True
        maximum_servo_echo_error = 0.0
        maximum_wheel_echo_error = 0.0
        finite_frames = 0
        state_maxima = {
            "joint_position_rad": 0.0,
            "joint_velocity_rad_s": 0.0,
            "root_position_m": 0.0,
            "root_velocity": 0.0,
            "wheel_body_position_m": 0.0,
            "obstacle_position_m": 0.0,
            "environment_origin_m": 0.0,
        }

        for _step in range(request.physics_steps):
            if hasattr(simulation_app, "is_running") and not simulation_app.is_running():
                raise SceneSmokeContractError("SimulationApp stopped before smoke completed")
            delta_deg = torch.clamp(
                requested_command_deg - applied_command_deg,
                min=-max_delta_deg,
                max=max_delta_deg,
            )
            applied_command_deg = applied_command_deg + delta_deg
            raw_servo_target = standing_raw_rad + command_sign * torch.deg2rad(applied_command_deg)
            robot.set_joint_position_target(raw_servo_target, joint_ids=servo_ids)
            robot.set_joint_velocity_target(raw_wheel_target, joint_ids=wheel_ids)
            servo_echo = robot.data.joint_pos_target[:, servo_ids]
            wheel_echo = robot.data.joint_vel_target[:, wheel_ids]
            servo_echo_error = float(torch.max(torch.abs(servo_echo - raw_servo_target)).item())
            wheel_echo_error = float(torch.max(torch.abs(wheel_echo - raw_wheel_target)).item())
            maximum_servo_echo_error = max(maximum_servo_echo_error, servo_echo_error)
            maximum_wheel_echo_error = max(maximum_wheel_echo_error, wheel_echo_error)
            target_echo_ok = target_echo_ok and bool(
                torch.equal(servo_echo, raw_servo_target) and torch.equal(wheel_echo, raw_wheel_target)
            )
            standing_target_ok = standing_target_ok and bool(torch.equal(raw_servo_target, standing_raw_rad))
            interactive_scene.write_data_to_sim()
            sim.step(render=False)
            interactive_scene.update(PHYSICS_DT_S)

            tensors = {
                "joint_position_rad": robot.data.joint_pos,
                "joint_velocity_rad_s": robot.data.joint_vel,
                "root_position_m": robot.data.root_pos_w,
                "root_velocity": torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1),
                "wheel_body_position_m": robot.data.body_pos_w[:, wheel_body_ids, :],
                "obstacle_position_m": obstacle.data.root_pos_w,
                "environment_origin_m": interactive_scene.env_origins,
            }
            for name, tensor in tensors.items():
                if tensor.numel() < 1 or not bool(torch.isfinite(tensor).all().item()):
                    raise SceneSmokeContractError(f"non-finite or empty live tensor: {name}")
                state_maxima[name] = max(
                    state_maxima[name], float(torch.max(torch.abs(tensor)).item())
                )
            finite_frames += 1

        stage = interactive_scene.stage
        env_prim_paths = [str(path) for path in interactive_scene.env_prim_paths]
        expected_env_paths = [f"/World/envs/env_{index}" for index in range(request.num_envs)]
        if env_prim_paths != expected_env_paths:
            raise SceneSmokeContractError(
                f"environment prim paths mismatch: expected {expected_env_paths}, got {env_prim_paths}"
            )
        per_env_prims = {
            env_path: {
                "robot": bool(stage.GetPrimAtPath(f"{env_path}/Robot").IsValid()),
                "obstacle": bool(stage.GetPrimAtPath(f"{env_path}/Obstacle").IsValid()),
            }
            for env_path in env_prim_paths
        }
        ground_valid = bool(stage.GetPrimAtPath(bundle.spec.ground_prim_path).IsValid())
        collision_filter_prim_valid = bool(stage.GetPrimAtPath("/World/collisions").IsValid())
        all_prims_valid = bool(
            ground_valid
            and collision_filter_prim_valid
            and all(all(values.values()) for values in per_env_prims.values())
        )
        final_standing_error = float(
            torch.max(torch.abs(robot.data.joint_pos[:, servo_ids] - standing_raw_rad)).item()
        )
        runtime = {
            "device": request.device,
            "num_envs": request.num_envs,
            "env_spacing_m": request.env_spacing_m,
            "physics_dt_s": PHYSICS_DT_S,
            "direct_rl_decimation": bundle.direct_rl_decimation,
            "render_interval_physics_steps": RENDER_INTERVAL_PHYSICS_STEPS,
            "physics_steps_requested": request.physics_steps,
            "physics_steps_completed": finite_frames,
            "scene_env_prim_paths": env_prim_paths,
            "scene_prim_validation": {
                "per_environment": per_env_prims,
                "ground": ground_valid,
                "collision_filter_prim": collision_filter_prim_valid,
                "all_expected_prims_valid": all_prims_valid,
            },
            "collision_filter": {
                "clone_called": True,
                "copy_from_source": False,
                "filter_called": True,
                "global_collision_prim_paths": list(bundle.global_collision_prim_paths),
            },
            "servo_joint_ids": servo_ids,
            "servo_joint_names": servo_names,
            "wheel_joint_ids": wheel_ids,
            "wheel_joint_names": wheel_names,
            "wheel_body_ids": wheel_body_ids,
            "wheel_body_names": wheel_body_names,
            "actuation": {
                "zero_servo_command_verified": bool(torch.count_nonzero(requested_command_deg).item() == 0),
                "zero_wheel_command_verified": bool(torch.count_nonzero(wheel_command_physical).item() == 0),
                "standing_target_verified_all_steps": standing_target_ok,
                "target_echo_verified_all_steps": target_echo_ok,
                "servo_target_write_count": request.physics_steps,
                "wheel_target_write_count": request.physics_steps,
                "servo_reference_velocity_deg_s": request.servo_reference_velocity_deg_s,
                "max_delta_deg_per_physics_step": max_delta_deg,
                "rate_limit_probe_requested_deg": RATE_LIMIT_PROBE_REQUEST_DEG,
                "rate_limit_probe_applied_deg": rate_limit_probe(),
                "rate_limit_probe_applied_to_robot": False,
                "max_abs_servo_target_echo_error_rad": maximum_servo_echo_error,
                "max_abs_wheel_target_echo_error_rad_s": maximum_wheel_echo_error,
                "final_max_abs_standing_position_error_rad": final_standing_error,
            },
            "finite_state": {
                "all_finite": finite_frames == request.physics_steps,
                "physics_frames_checked": finite_frames,
                "max_abs_by_tensor": state_maxima,
            },
            "package_versions": {
                "isaaclab": _package_version("isaaclab"),
                "isaacsim": _package_version("isaacsim"),
                "torch": _package_version("torch"),
            },
            "process_id": os.getpid(),
        }
        _validate_success_runtime(runtime, request)
    finally:
        # Mirror DirectRLEnv's order-sensitive shutdown: drop scene/assets,
        # remove callbacks, then clear the SimulationContext singleton.
        obstacle = None
        robot = None
        interactive_scene = None
        if sim is not None:
            clear_callbacks = getattr(sim, "clear_all_callbacks", None)
            if callable(clear_callbacks):
                clear_callbacks()
            sim.clear_instance()
            context_cleared = True
    return runtime, context_cleared


def _run_with_app_launcher(args: argparse.Namespace, loaded: LoadedSceneSmokeRequest) -> int:
    from isaaclab.app import AppLauncher  # type: ignore

    request = loaded.request
    if str(args.device) != request.device:
        raise SceneSmokeContractError(
            f"CLI --device {args.device!r} does not match request device {request.device!r}"
        )
    if request.headless_required and args.headless is not True:
        raise SceneSmokeContractError("the sealed smoke request requires explicit --headless")
    if request.result_path.exists():
        raise SceneSmokeContractError(f"fresh result_path already exists: {request.result_path}")
    if not request.result_path.parent.is_dir():
        raise SceneSmokeContractError(
            f"result parent directory does not exist: {request.result_path.parent}"
        )

    app: Any | None = None
    runtime: dict[str, Any] = {}
    status = "FAIL"
    error: dict[str, Any] | None = None
    context_cleared = False
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        runtime, context_cleared = _execute_real_isaac_smoke(request, simulation_app=app)
        status = "PASS"
    except Exception as exc:
        trace = traceback.format_exc()
        error = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000] or type(exc).__name__,
            "traceback_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
        }
    application_close_requested = app is not None
    if status == "PASS" and not context_cleared:
        status = "FAIL"
        error = {
            "type": "SceneSmokeClosureError",
            "message": "successful runtime did not clear its SimulationContext",
            "traceback_sha256": hashlib.sha256(b"SceneSmokeClosureError").hexdigest(),
        }
    result = build_smoke_result(
        loaded,
        status=status,
        runtime=runtime,
        closure={
            "simulation_context_cleared": context_cleared,
            "application_close_requested": application_close_requested,
        },
        error=error,
    )
    exit_code = 0 if status == "PASS" else 1
    try:
        # SimulationApp.close() may terminate the process and never return.
        # Persist and fsync the complete child evidence before requesting it;
        # process exit/no-survivor are deliberately parent-owned evidence.
        write_json_atomic(request.result_path, result)
        print(
            json.dumps(
                {
                    "status": status,
                    "result_path": str(request.result_path),
                    "result_file_sha256": _sha256_file(request.result_path),
                    "result_payload_sha256": result["payload_sha256"],
                    "application_close_requested": application_close_requested,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if app is not None:
            app.close()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry.  This is the only function that imports AppLauncher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Absolute sealed smoke request JSON path")
    from isaaclab.app import AppLauncher  # type: ignore

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    loaded = load_smoke_request(args.request)
    return _run_with_app_launcher(args, loaded)


def _resolve_exact(
    resolver: Any,
    expected_names: Sequence[str],
    label: str,
) -> tuple[list[int], list[str]]:
    ids, names = resolver(list(expected_names), preserve_order=True)
    normalized_ids = [int(value) for value in ids]
    normalized_names = [str(value) for value in names]
    if normalized_names != list(expected_names):
        raise SceneSmokeContractError(
            f"{label} names mismatch: expected {list(expected_names)}, got {normalized_names}"
        )
    if len(normalized_ids) != len(expected_names) or len(set(normalized_ids)) != len(normalized_ids):
        raise SceneSmokeContractError(f"{label} IDs are incomplete or duplicated: {normalized_ids}")
    if any(value < 0 for value in normalized_ids):
        raise SceneSmokeContractError(f"{label} IDs must be nonnegative: {normalized_ids}")
    return normalized_ids, normalized_names


def _expand_namespaced_asset_cfg(cfg: Any, env_regex_ns: str, label: str) -> Any:
    prim_path = getattr(cfg, "prim_path", None)
    if not isinstance(prim_path, str) or prim_path.count("{ENV_REGEX_NS}") != 1:
        raise SceneSmokeContractError(
            f"{label} cfg must contain exactly one {{ENV_REGEX_NS}} placeholder"
        )
    if not isinstance(env_regex_ns, str) or not env_regex_ns.startswith("/World/envs/env_"):
        raise SceneSmokeContractError(f"{label} env_regex_ns is invalid: {env_regex_ns!r}")
    expanded = prim_path.format(ENV_REGEX_NS=env_regex_ns)
    if "{" in expanded or "}" in expanded or not expanded.startswith(f"{env_regex_ns}/"):
        raise SceneSmokeContractError(f"{label} expanded prim path is invalid: {expanded!r}")
    replace_method = getattr(cfg, "replace", None)
    if not callable(replace_method):
        raise SceneSmokeContractError(f"{label} cfg has no configclass replace() method")
    result = replace_method(prim_path=expanded)
    if getattr(result, "prim_path", None) != expanded:
        raise SceneSmokeContractError(f"{label} cfg replacement did not retain expanded prim path")
    return result


def _make_envelope(schema_version: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    return {
        "schema_version": schema_version,
        "payload": row,
        "payload_sha256": _sha256_payload(row),
    }


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_file_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SceneSmokeContractError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise SceneSmokeContractError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual} ({path})"
        )


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneSmokeContractError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise SceneSmokeContractError(f"{name} keys must be strings")
    return value


def _require_exact_keys(name: str, value: Mapping[str, Any], expected: frozenset[str]) -> None:
    actual = set(value)
    if actual != set(expected):
        raise SceneSmokeContractError(
            f"{name} keys mismatch: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )


def _require_sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SceneSmokeContractError(f"{name} must be a lowercase SHA256")
    return value


def _require_request_id(value: Any) -> str:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise SceneSmokeContractError("request_id is invalid")
    return value


def _require_device(value: Any) -> str:
    if not isinstance(value, str) or _DEVICE.fullmatch(value) is None:
        raise SceneSmokeContractError("device must be cpu, cuda, or cuda:<nonnegative index>")
    return value


def _require_absolute_json_path(name: str, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise SceneSmokeContractError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() != ".json":
        raise SceneSmokeContractError(f"{name} must be an absolute .json path")
    return path.resolve()


def _require_exact_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SceneSmokeContractError(f"{name} must be a nonnegative exact integer")
    return value


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneSmokeContractError(f"{name} must be an exact finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SceneSmokeContractError(f"{name} must be finite")
    return result


def _require_aware_iso8601(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SceneSmokeContractError(f"{name} must be an ISO8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SceneSmokeContractError(f"{name} is not valid ISO8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SceneSmokeContractError(f"{name} must include a timezone")
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "UNKNOWN"


__all__ = [
    "FROZEN_SCENE_MANIFEST_SHA256",
    "FROZEN_SCENE_SOURCE_SHA256",
    "LoadedSceneSmokeRequest",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SMOKE_NUM_ENVS",
    "SMOKE_PHYSICS_STEPS",
    "SceneSmokeContractError",
    "ValidatedSceneSmokeRequest",
    "build_smoke_request",
    "build_smoke_result",
    "load_smoke_request",
    "main",
    "rate_limit_probe",
    "validate_smoke_request",
    "validate_smoke_result",
    "write_json_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
