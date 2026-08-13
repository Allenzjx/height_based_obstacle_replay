"""Robot ground-contact and viewport physics guard diagnostics.

This module is intentionally importable without Isaac Sim.  Isaac/USD access is
resolved lazily inside helper functions, while the dataclasses and comparison
logic remain plain Python so tests can exercise the safety rules in --no-sim.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from command_model import WHEEL_JOINT_NAMES


GROUND_OK = "OK"
VISUAL_ONLY_INTERSECTION = "VISUAL_ONLY_INTERSECTION"
COLLISION_PENETRATION = "COLLISION_PENETRATION"
COLLIDER_CONFIRMED_MISSING = "COLLIDER_CONFIRMED_MISSING"
COLLIDER_RESOLUTION_FAILED = "COLLIDER_RESOLUTION_FAILED"
MISSING_WHEEL_COLLISION = COLLIDER_CONFIRMED_MISSING
ROBOT_STATE_CHANGED_BY_VIEWPORT_ACTION = "ROBOT_STATE_CHANGED_BY_VIEWPORT_ACTION"
TIMELINE_CHANGED_BY_VIEWPORT_ACTION = "TIMELINE_CHANGED_BY_VIEWPORT_ACTION"
RENDER_OR_FABRIC_DESYNC_SUSPECTED = "RENDER_OR_FABRIC_DESYNC_SUSPECTED"
UNKNOWN = "UNKNOWN"
GROUND_STATE_PASS = "PASS"
GROUND_STATE_PASS_WITH_VISUAL_WARNING = "PASS_WITH_VISUAL_WARNING"
GROUND_STATE_FAIL = "FAIL"
GROUND_STATE_UNVERIFIED = "UNVERIFIED"

GROUND_PRIM_PATH = "/World/defaultGroundPlane"
BOUND_SENTINEL_ABS = 3.0e38
MAX_REASONABLE_SCENE_COORD_M = 1000.0
MAX_REASONABLE_BOUND_EXTENT_M = 1000.0

VIEWPORT_ACTIONS = {
    "open_camera_viewport",
    "show_camera_view",
    "return_main_view_to_perspective",
    "close_camera_viewport",
    "restore_camera_view",
}


@dataclass
class WheelGroundDiagnostics:
    wheel_name: str
    joint_name: str = ""
    joint_prim_path: str = ""
    joint_id: int | None = None
    body0_path: str = ""
    body1_path: str = ""
    resolved_body_path: str = ""
    body_name: str = ""
    candidate_body_paths: list[str] = field(default_factory=list)
    candidate_body_scores: list[dict[str, Any]] = field(default_factory=list)
    candidate_collision_paths: list[str] = field(default_factory=list)
    collision_paths: list[str] = field(default_factory=list)
    collision_resolution_state: str = ""
    applied_schemas: list[str] = field(default_factory=list)
    body_prim_path: str = ""
    data_source: str = ""
    body_position_w: list[float] = field(default_factory=list)
    visual_prim_paths: list[str] = field(default_factory=list)
    collision_prim_paths: list[str] = field(default_factory=list)
    collision_local_bounds: list[float] = field(default_factory=list)
    collision_api_present: bool = False
    collision_enabled: bool = False
    physics_collision_enabled: bool | None = None
    collision_approximation: str = ""
    visual_aabb_min_z: float | None = None
    collision_aabb_min_z: float | None = None
    visual_world_min_z: float | None = None
    collision_world_min_z: float | None = None
    minimum_collision_z: float | None = None
    clearance_m: float | None = None
    visual_ground_clearance_m: float | None = None
    collision_ground_clearance_m: float | None = None
    contact_with_ground: bool | None = None
    contact_state: str = ""
    bounds_valid: bool = False
    bounds_empty: bool = False
    bounds_min: list[float] = field(default_factory=list)
    bounds_max: list[float] = field(default_factory=list)
    bounds_extent: list[float] = field(default_factory=list)
    bounds_rejection_reason: str = ""
    bounds_source: str = ""
    bounds_finite: bool = False
    usd_bound_age_or_warning: str = ""
    collision_penetration_m: float = 0.0
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.joint_name:
            self.joint_name = self.wheel_name
        if not self.wheel_name:
            self.wheel_name = self.joint_name
        if self.collision_paths and not self.collision_prim_paths:
            self.collision_prim_paths = list(self.collision_paths)
        if self.collision_prim_paths and not self.collision_paths:
            self.collision_paths = list(self.collision_prim_paths)
        if self.candidate_collision_paths and not self.collision_prim_paths:
            self.collision_prim_paths = list(self.candidate_collision_paths)
            self.collision_paths = list(self.candidate_collision_paths)
        if self.resolved_body_path and not self.body_prim_path:
            self.body_prim_path = self.resolved_body_path
        if self.body_prim_path and not self.resolved_body_path:
            self.resolved_body_path = self.body_prim_path
        if self.body_prim_path and not self.body_name:
            self.body_name = self.body_prim_path.rstrip("/").split("/")[-1]
        if self.visual_world_min_z is None:
            self.visual_world_min_z = self.visual_aabb_min_z
        if self.collision_world_min_z is None:
            self.collision_world_min_z = self.collision_aabb_min_z
        if self.minimum_collision_z is None:
            self.minimum_collision_z = self.collision_world_min_z
        if self.clearance_m is None:
            self.clearance_m = self.collision_ground_clearance_m
        if self.collision_world_min_z is not None and not self.bounds_min:
            self.bounds_min = [0.0, 0.0, float(self.collision_world_min_z)]
        if self.collision_world_min_z is not None and self.bounds_valid is False and not self.bounds_rejection_reason:
            self.bounds_valid = True
            self.bounds_finite = True
        if not self.data_source:
            self.data_source = "usd_bbox"
        if not self.collision_resolution_state:
            if self.collision_api_present and self.collision_prim_paths:
                self.collision_resolution_state = GROUND_OK
            elif (
                self.body_prim_path
                or self.candidate_body_paths
                or self.visual_prim_paths
                or self.visual_aabb_min_z is not None
                or self.visual_ground_clearance_m is not None
            ):
                self.collision_resolution_state = COLLIDER_CONFIRMED_MISSING
            else:
                self.collision_resolution_state = COLLIDER_RESOLUTION_FAILED
        if not self.contact_state:
            if self.collision_ground_clearance_m is None:
                self.contact_state = "unavailable"
            elif self.collision_ground_clearance_m < 0.0:
                self.contact_state = "penetrating"
            else:
                self.contact_state = "clear"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundSurfaceInfo:
    configured_ground_z_m: float = 0.0
    actual_ground_z_m: float | None = None
    ground_z_source: str = ""
    ground_prim_path: str = GROUND_PRIM_PATH
    ground_collision_prim_path: str = ""
    ground_world_transform: list[list[float]] = field(default_factory=list)
    ground_local_translation: list[float] = field(default_factory=list)
    ground_world_translation: list[float] = field(default_factory=list)
    ground_normal_w: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    ground_resolution_ok: bool = False
    ground_z_delta_m: float | None = None
    tolerance_m: float = 1.0e-4
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicsStateSignature:
    checked: bool = False
    root_position_w: list[float] = field(default_factory=list)
    root_quaternion_w: list[float] = field(default_factory=list)
    root_linear_velocity: list[float] = field(default_factory=list)
    root_angular_velocity: list[float] = field(default_factory=list)
    root_z_m: float | None = None
    joint_pos: list[float] = field(default_factory=list)
    joint_vel: list[float] = field(default_factory=list)
    wheel_command_values: dict[str, float] = field(default_factory=dict)
    sim_time: float = 0.0
    sim_steps: int = 0
    timeline_playing: bool | None = None
    timestamp: float = field(default_factory=time.time)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotGroundDiagnostics:
    checked: bool = False
    classification: str = UNKNOWN
    ground_state: str = GROUND_STATE_UNVERIFIED
    ground_z_m: float = 0.0
    ground_surface: dict[str, Any] = field(default_factory=dict)
    ground_resolution_state: str = ""
    physical_ground_safe: bool = False
    visual_ground_safe: bool = False
    root_position_w: list[float] = field(default_factory=list)
    root_quaternion_w: list[float] = field(default_factory=list)
    root_linear_velocity: list[float] = field(default_factory=list)
    root_angular_velocity: list[float] = field(default_factory=list)
    root_z_m: float | None = None
    sim_time: float = 0.0
    sim_steps: int = 0
    timeline_playing: bool | None = None
    wheel_command_values: dict[str, float] = field(default_factory=dict)
    wheels: list[WheelGroundDiagnostics] = field(default_factory=list)
    minimum_visual_clearance_m: float | None = None
    minimum_collision_clearance_m: float | None = None
    maximum_collision_penetration_m: float = 0.0
    missing_collision_wheels: list[str] = field(default_factory=list)
    unresolved_collision_wheels: list[str] = field(default_factory=list)
    root_pose_delta: dict[str, float] = field(default_factory=dict)
    maximum_joint_position_delta: float = 0.0
    maximum_joint_velocity_delta: float = 0.0
    sim_time_delta: float = 0.0
    sim_steps_delta: int = 0
    correction_applied: bool = False
    correction_z_m: float = 0.0
    grounded_respawn_reference_valid: bool = False
    respawn_ready: bool = False
    ground_reference_block_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["wheels"] = [wheel.to_dict() for wheel in self.wheels]
        return data


@dataclass
class ViewportPhysicsGuardResult:
    checked: bool = False
    passed: bool = False
    classification: str = UNKNOWN
    action: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    ground_before: dict[str, Any] = field(default_factory=dict)
    ground_after: dict[str, Any] = field(default_factory=dict)
    root_pose_delta: dict[str, float] = field(default_factory=dict)
    root_pose_delta_m: float = 0.0
    root_rotation_delta_deg: float = 0.0
    maximum_joint_position_delta: float = 0.0
    maximum_joint_velocity_delta: float = 0.0
    sim_time_delta: float = 0.0
    sim_steps_delta: int = 0
    timeline_changed: bool = False
    correction_applied: bool = False
    correction_z_m: float = 0.0
    rtf_before: float | None = None
    rtf_after: float | None = None
    fabric_warning_detected: bool = False
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraViewChangeGuard:
    checked: bool = True
    allowed: bool = False
    reasons: list[str] = field(default_factory=list)
    wheel_command_values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_robot_ground_diagnostics(reason: str = "not checked") -> dict[str, Any]:
    reasons = [str(reason)] if reason else []
    return RobotGroundDiagnostics(checked=False, classification=UNKNOWN, reasons=reasons).to_dict()


def default_viewport_physics_guard(reason: str = "not checked") -> dict[str, Any]:
    reasons = [str(reason)] if reason else []
    return ViewportPhysicsGuardResult(checked=False, passed=False, classification=UNKNOWN, reasons=reasons).to_dict()


def capture_physics_state_signature(
    adapter: Any,
    *,
    sim_time: float | None = None,
    sim_steps: int | None = None,
    timeline_playing: bool | None = None,
) -> PhysicsStateSignature:
    warnings: list[str] = []
    root_pose = _first_row(_read_nested(getattr(getattr(adapter, "robot", None), "data", None), "root_pose_w"))
    root_vel = _first_row(_read_nested(getattr(getattr(adapter, "robot", None), "data", None), "root_vel_w"))
    joint_pos = _first_row(_read_nested(getattr(getattr(adapter, "robot", None), "data", None), "joint_pos"))
    joint_vel = _first_row(_read_nested(getattr(getattr(adapter, "robot", None), "data", None), "joint_vel"))
    if not root_pose or len(root_pose) < 7:
        state = _safe_capture_sim_state(adapter)
        root_pose = _first_row(state.get("root_pose")) or root_pose
        root_vel = _first_row(state.get("root_velocity")) or root_vel
        joint_pos = _first_row(state.get("joint_pos")) or joint_pos
        joint_vel = _first_row(state.get("joint_vel")) or joint_vel
    if not root_pose or len(root_pose) < 7:
        warnings.append("root_pose_w unavailable")
    if not root_vel or len(root_vel) < 6:
        root_vel = [0.0] * 6
        warnings.append("root_vel_w unavailable")
    command_state = _safe_capture_command_state(adapter)
    if sim_time is None:
        sim_time = _safe_float(getattr(getattr(adapter, "sim", None), "current_time", 0.0), 0.0)
    if sim_steps is None:
        sim_steps = int(_safe_float(getattr(adapter, "sim_steps", 0), 0.0))
    if timeline_playing is None:
        timeline_playing = _timeline_is_playing()
    root_position = [_safe_float(value, 0.0) for value in (root_pose[:3] if len(root_pose) >= 3 else [])]
    root_quat = [_safe_float(value, 0.0) for value in (root_pose[3:7] if len(root_pose) >= 7 else [])]
    return PhysicsStateSignature(
        checked=bool(root_position or joint_pos or joint_vel),
        root_position_w=root_position,
        root_quaternion_w=root_quat,
        root_linear_velocity=[_safe_float(value, 0.0) for value in root_vel[:3]],
        root_angular_velocity=[_safe_float(value, 0.0) for value in root_vel[3:6]],
        root_z_m=root_position[2] if len(root_position) >= 3 else None,
        joint_pos=[_safe_float(value, 0.0) for value in joint_pos],
        joint_vel=[_safe_float(value, 0.0) for value in joint_vel],
        wheel_command_values={
            name: _safe_float(command_state.get("wheels", {}).get(name, 0.0), 0.0)
            for name in WHEEL_JOINT_NAMES
        },
        sim_time=float(sim_time or 0.0),
        sim_steps=int(sim_steps or 0),
        timeline_playing=timeline_playing,
        warnings=warnings,
    )


def compare_physics_signatures(
    before: PhysicsStateSignature,
    after: PhysicsStateSignature,
    *,
    root_translation_tolerance_m: float = 1.0e-6,
    root_rotation_tolerance_deg: float = 1.0e-5,
    joint_position_tolerance_rad: float = 1.0e-6,
    joint_velocity_tolerance_rad_s: float = 1.0e-6,
) -> ViewportPhysicsGuardResult:
    translation_delta = _vector_max_abs_delta(before.root_position_w, after.root_position_w)
    rotation_delta = _quat_angle_delta_deg(before.root_quaternion_w, after.root_quaternion_w)
    joint_pos_delta = _vector_max_abs_delta(before.joint_pos, after.joint_pos)
    joint_vel_delta = _vector_max_abs_delta(before.joint_vel, after.joint_vel)
    sim_time_delta = float(after.sim_time) - float(before.sim_time)
    sim_steps_delta = int(after.sim_steps) - int(before.sim_steps)
    timeline_changed = (
        before.timeline_playing is not None
        and after.timeline_playing is not None
        and bool(before.timeline_playing) != bool(after.timeline_playing)
    )
    reasons: list[str] = []
    if translation_delta > root_translation_tolerance_m:
        reasons.append(f"root translation delta {translation_delta:.9g}m exceeds {root_translation_tolerance_m:.9g}m")
    if rotation_delta > root_rotation_tolerance_deg:
        reasons.append(f"root rotation delta {rotation_delta:.9g}deg exceeds {root_rotation_tolerance_deg:.9g}deg")
    if joint_pos_delta > joint_position_tolerance_rad:
        reasons.append(f"joint position delta {joint_pos_delta:.9g}rad exceeds {joint_position_tolerance_rad:.9g}rad")
    if joint_vel_delta > joint_velocity_tolerance_rad_s:
        reasons.append(f"joint velocity delta {joint_vel_delta:.9g}rad/s exceeds {joint_velocity_tolerance_rad_s:.9g}rad/s")
    if sim_time_delta != 0.0:
        reasons.append(f"sim_time changed by {sim_time_delta:.9g}s")
    if sim_steps_delta != 0:
        reasons.append(f"sim_steps changed by {sim_steps_delta}")
    if timeline_changed:
        reasons.append("timeline playing state changed")
    if sim_time_delta != 0.0 or sim_steps_delta != 0 or timeline_changed:
        classification = TIMELINE_CHANGED_BY_VIEWPORT_ACTION
    elif reasons:
        classification = ROBOT_STATE_CHANGED_BY_VIEWPORT_ACTION
    else:
        classification = GROUND_OK
    return ViewportPhysicsGuardResult(
        checked=True,
        passed=not reasons,
        classification=classification,
        before=before.to_dict(),
        after=after.to_dict(),
        root_pose_delta={
            "translation_m": float(translation_delta),
            "rotation_deg": float(rotation_delta),
        },
        root_pose_delta_m=float(translation_delta),
        root_rotation_delta_deg=float(rotation_delta),
        maximum_joint_position_delta=float(joint_pos_delta),
        maximum_joint_velocity_delta=float(joint_vel_delta),
        sim_time_delta=float(sim_time_delta),
        sim_steps_delta=int(sim_steps_delta),
        timeline_changed=bool(timeline_changed),
        reasons=reasons,
    )


def build_robot_ground_diagnostics(
    wheels: list[WheelGroundDiagnostics],
    *,
    ground_z_m: float = 0.0,
    ground_surface: GroundSurfaceInfo | dict[str, Any] | None = None,
    signature: PhysicsStateSignature | None = None,
    penetration_tolerance_m: float = 0.003,
    fabric_warning_detected: bool = False,
    grounded_respawn_reference_valid: bool = False,
    reasons: list[str] | None = None,
    warnings: list[str] | None = None,
) -> RobotGroundDiagnostics:
    rows = list(wheels)
    ground_z = float(ground_z_m)
    visual_clearances = [float(w.visual_ground_clearance_m) for w in rows if w.visual_ground_clearance_m is not None]
    collision_clearances = [float(w.collision_ground_clearance_m) for w in rows if w.collision_ground_clearance_m is not None]
    unresolved = [
        w.wheel_name
        for w in rows
        if str(w.collision_resolution_state or "") == COLLIDER_RESOLUTION_FAILED
    ]
    missing = [
        w.wheel_name
        for w in rows
        if str(w.collision_resolution_state or "") == COLLIDER_CONFIRMED_MISSING
        or (
            str(w.collision_resolution_state or "") not in {GROUND_OK, COLLIDER_RESOLUTION_FAILED}
            and (not w.collision_api_present or not w.collision_prim_paths)
        )
    ]
    max_penetration = max([float(w.collision_penetration_m) for w in rows] + [0.0])
    min_visual = min(visual_clearances) if visual_clearances else None
    min_collision = min(collision_clearances) if collision_clearances else None
    diag_reasons = list(reasons or [])
    diag_warnings = list(warnings or [])
    if unresolved:
        classification = COLLIDER_RESOLUTION_FAILED
        ground_state = GROUND_STATE_UNVERIFIED
        diag_reasons.append("wheel collision resolver failed: " + ", ".join(unresolved))
    elif missing:
        classification = COLLIDER_CONFIRMED_MISSING
        ground_state = GROUND_STATE_FAIL
        diag_reasons.append("wheel collision missing: " + ", ".join(missing))
    elif max_penetration > float(penetration_tolerance_m):
        classification = COLLISION_PENETRATION
        ground_state = GROUND_STATE_FAIL
        diag_reasons.append(
            f"maximum collision penetration {max_penetration:.6f}m exceeds tolerance {float(penetration_tolerance_m):.6f}m"
        )
    elif min_visual is not None and min_visual < 0.0 and (min_collision is None or min_collision >= -float(penetration_tolerance_m)):
        classification = RENDER_OR_FABRIC_DESYNC_SUSPECTED if fabric_warning_detected else VISUAL_ONLY_INTERSECTION
        ground_state = GROUND_STATE_UNVERIFIED if fabric_warning_detected else GROUND_STATE_PASS_WITH_VISUAL_WARNING
        diag_reasons.append("visual AABB intersects ground while collision AABB is within tolerance")
    elif rows:
        classification = GROUND_OK
        ground_state = GROUND_STATE_PASS
    else:
        classification = UNKNOWN
        ground_state = GROUND_STATE_UNVERIFIED
        diag_reasons.append("no wheel ground diagnostics available")
    collision_bounds_reliable = bool(rows) and not unresolved and not missing and all(
        str(w.collision_resolution_state or "") == GROUND_OK
        and w.collision_ground_clearance_m is not None
        and bool(getattr(w, "bounds_valid", False) or w.collision_world_min_z is not None)
        for w in rows
    )
    physical_ground_safe = bool(
        collision_bounds_reliable
        and max_penetration <= float(penetration_tolerance_m)
        and classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}
    )
    visual_ground_safe = bool(physical_ground_safe and classification == GROUND_OK)
    if not physical_ground_safe and collision_bounds_reliable is False and rows and not unresolved and not missing:
        diag_reasons.append("collision bounds are unavailable or invalid")
    sig = signature or PhysicsStateSignature()
    if ground_surface is None:
        surface_dict: dict[str, Any] = {}
        ground_resolution_state = "not_checked"
    elif isinstance(ground_surface, GroundSurfaceInfo):
        surface_dict = ground_surface.to_dict()
        ground_resolution_state = "ok" if ground_surface.ground_resolution_ok else "mismatch"
    else:
        surface_dict = dict(ground_surface)
        ground_resolution_state = "ok" if bool(surface_dict.get("ground_resolution_ok", False)) else "mismatch"
    return RobotGroundDiagnostics(
        checked=True,
        classification=classification,
        ground_state=ground_state,
        ground_z_m=ground_z,
        ground_surface=surface_dict,
        ground_resolution_state=ground_resolution_state,
        physical_ground_safe=physical_ground_safe,
        visual_ground_safe=visual_ground_safe,
        root_position_w=list(sig.root_position_w),
        root_quaternion_w=list(sig.root_quaternion_w),
        root_linear_velocity=list(sig.root_linear_velocity),
        root_angular_velocity=list(sig.root_angular_velocity),
        root_z_m=sig.root_z_m,
        sim_time=float(sig.sim_time),
        sim_steps=int(sig.sim_steps),
        timeline_playing=sig.timeline_playing,
        wheel_command_values=dict(sig.wheel_command_values),
        wheels=rows,
        minimum_visual_clearance_m=min_visual,
        minimum_collision_clearance_m=min_collision,
        maximum_collision_penetration_m=float(max_penetration),
        missing_collision_wheels=missing,
        unresolved_collision_wheels=unresolved,
        grounded_respawn_reference_valid=bool(grounded_respawn_reference_valid),
        respawn_ready=bool(grounded_respawn_reference_valid and physical_ground_safe),
        ground_reference_block_reason="" if bool(grounded_respawn_reference_valid and physical_ground_safe) else "; ".join(diag_reasons),
        warnings=diag_warnings,
        reasons=diag_reasons,
    )


def inspect_robot_ground_contact(
    scene_handle: Any,
    adapter: Any,
    *,
    sim_time: float | None = None,
    sim_steps: int | None = None,
    penetration_tolerance_m: float = 0.003,
    fabric_warning_detected: bool = False,
) -> RobotGroundDiagnostics:
    signature = capture_physics_state_signature(adapter, sim_time=sim_time, sim_steps=sim_steps)
    ground_surface = resolve_ground_surface(scene_handle)
    actual_ground_z = ground_surface.actual_ground_z_m
    ground_z = _safe_float(actual_ground_z, _safe_float(getattr(getattr(scene_handle, "config", None), "ground_z_m", 0.0), 0.0))
    fake_rows = _fake_wheel_rows(scene_handle, ground_z)
    if fake_rows is not None:
        return build_robot_ground_diagnostics(
            fake_rows,
            ground_z_m=ground_z,
            ground_surface=ground_surface,
            signature=signature,
            penetration_tolerance_m=penetration_tolerance_m,
            fabric_warning_detected=fabric_warning_detected,
            grounded_respawn_reference_valid=bool(getattr(adapter, "grounded_reference_valid", False)),
        )
    try:
        rows = _inspect_usd_wheel_rows(scene_handle, adapter, ground_z)
        return build_robot_ground_diagnostics(
            rows,
            ground_z_m=ground_z,
            ground_surface=ground_surface,
            signature=signature,
            penetration_tolerance_m=penetration_tolerance_m,
            fabric_warning_detected=fabric_warning_detected,
            grounded_respawn_reference_valid=bool(getattr(adapter, "grounded_reference_valid", False)),
        )
    except Exception as exc:
        return RobotGroundDiagnostics(
            checked=False,
            classification=UNKNOWN,
            ground_state=GROUND_STATE_UNVERIFIED,
            ground_z_m=ground_z,
            ground_surface=ground_surface.to_dict(),
            ground_resolution_state="ok" if ground_surface.ground_resolution_ok else "mismatch",
            root_position_w=list(signature.root_position_w),
            root_quaternion_w=list(signature.root_quaternion_w),
            root_linear_velocity=list(signature.root_linear_velocity),
            root_angular_velocity=list(signature.root_angular_velocity),
            root_z_m=signature.root_z_m,
            sim_time=float(signature.sim_time),
            sim_steps=int(signature.sim_steps),
            timeline_playing=signature.timeline_playing,
            wheel_command_values=dict(signature.wheel_command_values),
            grounded_respawn_reference_valid=bool(getattr(adapter, "grounded_reference_valid", False)),
            warnings=list(signature.warnings),
            reasons=[f"ground diagnostics unavailable: {exc}"],
        )


def compute_bounded_ground_correction(
    diagnostics: RobotGroundDiagnostics | dict[str, Any],
    *,
    target_clearance_m: float = 0.002,
    max_correction_m: float = 0.10,
) -> tuple[bool, float, str]:
    diag = diagnostics if isinstance(diagnostics, dict) else diagnostics.to_dict()
    if str(diag.get("classification", "")) != COLLISION_PENETRATION:
        return False, 0.0, "correction allowed only for COLLISION_PENETRATION"
    minimum = diag.get("minimum_collision_clearance_m")
    if minimum is None:
        return False, 0.0, "minimum collision clearance unavailable"
    required = float(target_clearance_m) - float(minimum)
    if required <= 0.0:
        return False, 0.0, "collision clearance already meets target"
    clamped = max(0.0, min(float(required), float(max_correction_m)))
    if clamped <= 0.0:
        return False, 0.0, "computed correction is zero"
    if required > float(max_correction_m):
        return True, clamped, f"required correction {required:.6f}m clamped to {clamped:.6f}m"
    return True, clamped, ""


def can_change_camera_view(
    *,
    flags: dict[str, Any] | None = None,
    wheel_command_values: dict[str, float] | None = None,
    wheel_tolerance: float = 1.0e-6,
) -> CameraViewChangeGuard:
    data = dict(flags or {})
    wheels = {str(name): _safe_float(value, 0.0) for name, value in dict(wheel_command_values or {}).items()}
    reasons: list[str] = []
    if bool(data.get("recording", data.get("recording_active", False))):
        reasons.append("recording is active")
    if bool(data.get("playback", data.get("playback_active", False))):
        reasons.append("playback is active")
    if bool(data.get("scheduled_playback", data.get("playback_scheduled", False))):
        reasons.append("playback is scheduled")
    if bool(data.get("operation_busy", False)):
        reasons.append("operation is busy")
    if bool(data.get("pending_step", False)):
        reasons.append("recorded step is pending")
    if bool(data.get("pending_replacement", False)):
        reasons.append("replacement step is pending")
    if bool(data.get("e_stop_exception", data.get("estop_exception", False))):
        reasons.append("E-stop exception handling is active")
    moving = [name for name, value in wheels.items() if abs(float(value)) > float(wheel_tolerance)]
    if moving:
        reasons.append("wheel commands are non-zero: " + ", ".join(moving))
    return CameraViewChangeGuard(allowed=not reasons, reasons=reasons, wheel_command_values=wheels)


def merge_guard_into_camera_view_status(status: dict[str, Any], guard: ViewportPhysicsGuardResult | dict[str, Any]) -> dict[str, Any]:
    merged = dict(status or {})
    guard_dict = guard if isinstance(guard, dict) else guard.to_dict()
    merged["physics_guard"] = dict(guard_dict)
    merged["physics_guard_checked"] = bool(guard_dict.get("checked", False))
    merged["physics_guard_passed"] = bool(guard_dict.get("passed", False))
    merged["root_pose_delta_m"] = float(guard_dict.get("root_pose_delta_m", 0.0) or 0.0)
    merged["root_rotation_delta_deg"] = float(guard_dict.get("root_rotation_delta_deg", 0.0) or 0.0)
    merged["max_joint_delta_rad"] = float(guard_dict.get("maximum_joint_position_delta", 0.0) or 0.0)
    merged["sim_time_delta"] = float(guard_dict.get("sim_time_delta", 0.0) or 0.0)
    merged["sim_steps_delta"] = int(guard_dict.get("sim_steps_delta", 0) or 0)
    before_ground = guard_dict.get("ground_before", {}) if isinstance(guard_dict.get("ground_before", {}), dict) else {}
    after_ground = guard_dict.get("ground_after", {}) if isinstance(guard_dict.get("ground_after", {}), dict) else {}
    merged["ground_classification_before"] = str(before_ground.get("classification", "") or "")
    merged["ground_classification_after"] = str(after_ground.get("classification", "") or "")
    merged["rtf_before"] = guard_dict.get("rtf_before")
    merged["rtf_after"] = guard_dict.get("rtf_after")
    merged["fabric_warning_detected"] = bool(guard_dict.get("fabric_warning_detected", False))
    return merged


def detect_fabric_or_viewport_warning(lines: list[str] | tuple[str, ...]) -> bool:
    needles = ("omni.fabric", "viewport", "render product", "getnamesandtypes", "bucketid")
    for line in lines:
        lowered = str(line).lower()
        if any(needle in lowered for needle in needles) and ("warn" in lowered or "error" in lowered or "not found" in lowered):
            return True
    return False


def run_viewport_action_with_physics_guard(
    *,
    adapter: Any,
    scene_handle: Any,
    action: str,
    callback: Any,
    enabled: bool = True,
    allow_restore_on_failure: bool = False,
    sim_time: float | None = None,
    sim_steps: int | None = None,
    rtf_before: float | None = None,
    rtf_after: float | None = None,
    penetration_tolerance_m: float = 0.003,
) -> tuple[Any, ViewportPhysicsGuardResult]:
    if not bool(enabled):
        result = callback()
        return result, ViewportPhysicsGuardResult(checked=False, passed=True, classification=UNKNOWN, action=str(action))
    before_state = _safe_capture_sim_state(adapter) if bool(allow_restore_on_failure) else {}
    before = capture_physics_state_signature(adapter, sim_time=sim_time, sim_steps=sim_steps)
    before_ground = inspect_robot_ground_contact(
        scene_handle,
        adapter,
        sim_time=sim_time,
        sim_steps=sim_steps,
        penetration_tolerance_m=penetration_tolerance_m,
    ).to_dict()
    result = callback()
    after = capture_physics_state_signature(adapter, sim_time=sim_time, sim_steps=sim_steps)
    after_ground = inspect_robot_ground_contact(
        scene_handle,
        adapter,
        sim_time=sim_time,
        sim_steps=sim_steps,
        penetration_tolerance_m=penetration_tolerance_m,
    ).to_dict()
    guard = compare_physics_signatures(before, after)
    guard.action = str(action)
    guard.ground_before = before_ground
    guard.ground_after = after_ground
    guard.rtf_before = rtf_before
    guard.rtf_after = rtf_after
    if allow_restore_on_failure and not guard.passed and before_state:
        try:
            adapter.restore_sim_state(before_state)
            adapter.stop_wheels()
            guard.warnings.append("explicit test path restored exact pre-viewport sim state after guard failure")
        except Exception as exc:
            guard.warnings.append(f"failed to restore pre-viewport state after guard failure: {exc}")
    return result, guard


def resolve_ground_surface(
    scene_handle: Any,
    *,
    configured_ground_z_m: float | None = None,
    ground_prim_path: str = GROUND_PRIM_PATH,
    tolerance_m: float = 1.0e-4,
) -> GroundSurfaceInfo:
    configured = _safe_float(
        configured_ground_z_m,
        _safe_float(getattr(getattr(scene_handle, "config", None), "ground_z_m", 0.0), 0.0),
    )
    existing = getattr(scene_handle, "ground_surface_info", None)
    if isinstance(existing, GroundSurfaceInfo):
        return existing
    if isinstance(existing, dict) and existing:
        try:
            return GroundSurfaceInfo(**existing)
        except Exception:
            pass
    info = GroundSurfaceInfo(
        configured_ground_z_m=float(configured),
        ground_prim_path=str(getattr(scene_handle, "ground_prim_path", "") or ground_prim_path),
        tolerance_m=float(tolerance_m),
    )
    stage = _resolve_stage(scene_handle)
    if stage is None:
        info.actual_ground_z_m = float(configured)
        info.ground_z_source = "configured_ground_z_stage_unavailable"
        info.ground_z_delta_m = 0.0
        info.ground_resolution_ok = True
        info.warnings.append("USD stage unavailable; using configured ground_z_m")
        return info
    try:
        prim = stage.GetPrimAtPath(info.ground_prim_path)
    except Exception as exc:
        prim = None
        info.warnings.append(f"could not access ground prim {info.ground_prim_path}: {exc}")
    if prim is None or not _prim_valid(prim):
        info.actual_ground_z_m = float(configured)
        info.ground_z_source = "configured_ground_z_prim_unavailable"
        info.ground_z_delta_m = 0.0
        info.ground_resolution_ok = True
        info.reasons.append(f"ground prim unavailable: {info.ground_prim_path}")
        return info

    collision = _find_ground_collision_prim(prim)
    if collision is not None and _prim_valid(collision):
        info.ground_collision_prim_path = str(collision.GetPath())
    else:
        info.ground_collision_prim_path = str(prim.GetPath())
        collision = prim

    world_translation = _prim_world_translation(collision)
    local_translation = _prim_local_translation(collision)
    world_transform = _prim_world_transform_matrix(collision)
    if world_translation is None and collision is not prim:
        world_translation = _prim_world_translation(prim)
        local_translation = _prim_local_translation(prim)
        world_transform = _prim_world_transform_matrix(prim)
    if world_translation is None:
        info.actual_ground_z_m = float(configured)
        info.ground_z_source = "configured_ground_z_world_transform_unavailable"
        info.ground_z_delta_m = 0.0
        info.ground_resolution_ok = True
        info.warnings.append("ground world transform unavailable; using configured ground_z_m")
        return info

    info.ground_world_translation = [float(value) for value in world_translation]
    info.ground_local_translation = [float(value) for value in (local_translation or [])]
    info.ground_world_transform = world_transform or []
    info.actual_ground_z_m = float(world_translation[2])
    info.ground_z_source = "usd_world_transform"
    info.ground_z_delta_m = float(info.actual_ground_z_m) - float(info.configured_ground_z_m)
    info.ground_resolution_ok = abs(float(info.ground_z_delta_m)) <= float(tolerance_m)
    if not info.ground_resolution_ok:
        info.warnings.append(
            "configured ground_z_m differs from actual ground surface: "
            f"configured={info.configured_ground_z_m:.6f} actual={info.actual_ground_z_m:.6f} "
            f"delta={info.ground_z_delta_m:.6f}"
        )
    return info


def ground_state_from_diagnostics(diagnostics: dict[str, Any] | RobotGroundDiagnostics | None) -> str:
    diag = diagnostics.to_dict() if isinstance(diagnostics, RobotGroundDiagnostics) else dict(diagnostics or {})
    explicit = str(diag.get("ground_state", "") or "")
    if explicit in {
        GROUND_STATE_PASS,
        GROUND_STATE_PASS_WITH_VISUAL_WARNING,
        GROUND_STATE_FAIL,
        GROUND_STATE_UNVERIFIED,
    }:
        return explicit
    classification = str(diag.get("classification", "") or UNKNOWN)
    if classification == GROUND_OK:
        return GROUND_STATE_PASS
    if classification == VISUAL_ONLY_INTERSECTION:
        return GROUND_STATE_PASS_WITH_VISUAL_WARNING
    if classification in {COLLISION_PENETRATION, COLLIDER_CONFIRMED_MISSING, MISSING_WHEEL_COLLISION}:
        return GROUND_STATE_FAIL
    return GROUND_STATE_UNVERIFIED


def motion_status_from_worker_status(status: dict[str, Any]) -> tuple[bool, str, str]:
    runtime_ready = bool(status.get("runtime_ready", status.get("ready", False)))
    ground = status.get("robot_ground", {}) if isinstance(status.get("robot_ground"), dict) else {}
    ground_state = ground_state_from_diagnostics(ground)
    physical_ground_safe = bool(ground.get("physical_ground_safe", ground_state in {GROUND_STATE_PASS, GROUND_STATE_PASS_WITH_VISUAL_WARNING}))
    reasons: list[str] = []
    if not runtime_ready:
        reasons.append("runtime is not ready")
    if ground_state == GROUND_STATE_FAIL:
        reasons.append("ground diagnostics failed")
    elif ground_state == GROUND_STATE_UNVERIFIED:
        reasons.append("ground diagnostics are unverified")
    if not physical_ground_safe:
        reasons.append("physical ground contact is not safe")
    motion_ready = runtime_ready and physical_ground_safe and ground_state in {GROUND_STATE_PASS, GROUND_STATE_PASS_WITH_VISUAL_WARNING}
    return bool(motion_ready), "; ".join(reasons), ground_state


def respawn_status_from_worker_status(status: dict[str, Any]) -> tuple[bool, str]:
    runtime_ready = bool(status.get("runtime_ready", status.get("ready", False)))
    ground = status.get("robot_ground", {}) if isinstance(status.get("robot_ground"), dict) else {}
    motion_ready, motion_reason, _ground_state = motion_status_from_worker_status(status)
    grounded_reference_valid = bool(
        status.get(
            "grounded_reference_valid",
            ground.get("grounded_respawn_reference_valid", ground.get("grounded_reference_valid", False)),
        )
    )
    reference_stable = bool(status.get("grounded_reference_stable", ground.get("grounded_reference_stable", grounded_reference_valid)))
    reasons: list[str] = []
    if not runtime_ready:
        reasons.append("runtime is not ready")
    if not motion_ready:
        reasons.append(motion_reason or "motion is not ready")
    if not grounded_reference_valid:
        reasons.append("grounded respawn reference is invalid")
    if not reference_stable:
        reasons.append("grounded respawn reference is not stable")
    block = str(ground.get("ground_reference_block_reason", "") or "")
    if block and not grounded_reference_valid:
        reasons.append(block)
    return bool(runtime_ready and motion_ready and grounded_reference_valid and reference_stable), "; ".join(dict.fromkeys(r for r in reasons if r))


def _inspect_usd_wheel_rows(scene_handle: Any, adapter: Any, ground_z: float) -> list[WheelGroundDiagnostics]:
    from pxr import Usd, UsdGeom  # type: ignore

    stage = _resolve_stage(scene_handle)
    if stage is None:
        raise RuntimeError("USD stage unavailable")
    root_path = str(getattr(scene_handle, "robot_prim_path", "") or "/World/WLRRobot")
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"robot root prim unavailable: {root_path}")
    imageable_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), imageable_purposes, useExtentsHint=True)
    rows: list[WheelGroundDiagnostics] = []
    for wheel_name in WHEEL_JOINT_NAMES:
        rows.append(resolve_wheel_collision_geometry(scene_handle, adapter, wheel_name, ground_z_m=ground_z, bbox_cache=bbox_cache))
    return rows


def resolve_wheel_collision_geometry(
    scene_handle: Any,
    adapter: Any,
    wheel_joint_name: str,
    *,
    ground_z_m: float | None = None,
    bbox_cache: Any | None = None,
) -> WheelGroundDiagnostics:
    stage = _resolve_stage(scene_handle)
    joint_name = str(wheel_joint_name)
    joint_id = _adapter_joint_id(adapter, joint_name)
    ground_z = _safe_float(
        ground_z_m,
        _safe_float(getattr(getattr(scene_handle, "config", None), "ground_z_m", 0.0), 0.0),
    )
    if stage is None:
        return WheelGroundDiagnostics(
            wheel_name=joint_name,
            joint_name=joint_name,
            joint_id=joint_id,
            data_source="stage_unavailable",
            collision_resolution_state=COLLIDER_RESOLUTION_FAILED,
            reasons=["USD stage unavailable"],
        )
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore
        try:
            from pxr import PhysxSchema  # type: ignore
        except Exception:
            PhysxSchema = None  # type: ignore
    except Exception as exc:
        return WheelGroundDiagnostics(
            wheel_name=joint_name,
            joint_name=joint_name,
            joint_id=joint_id,
            data_source="pxr_unavailable",
            collision_resolution_state=COLLIDER_RESOLUTION_FAILED,
            reasons=[f"pxr unavailable: {exc}"],
        )

    if bbox_cache is None:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
    root_path = str(getattr(scene_handle, "robot_prim_path", "") or "/World/WLRRobot")
    root = stage.GetPrimAtPath(root_path)
    if not _prim_valid(root):
        return WheelGroundDiagnostics(
            wheel_name=joint_name,
            joint_name=joint_name,
            joint_id=joint_id,
            data_source="robot_root_unavailable",
            collision_resolution_state=COLLIDER_RESOLUTION_FAILED,
            reasons=[f"robot root prim unavailable: {root_path}"],
        )

    joint_prim, body0_path, body1_path, joint_warnings = _find_joint_relationships(root, joint_name)
    body_names = list(getattr(getattr(adapter, "robot", None), "body_names", []) or [])
    candidate_body_paths = _candidate_body_paths_from_metadata(
        root,
        joint_name,
        body_names=body_names,
        body0_path=body0_path,
        body1_path=body1_path,
    )
    fallback_bodies = _fallback_find_wheel_bodies(root, joint_name)
    for fallback in fallback_bodies:
        if _prim_valid(fallback):
            path = str(fallback.GetPath())
            if path not in candidate_body_paths:
                candidate_body_paths.append(path)
    candidate_scores: list[dict[str, Any]] = []
    body = None
    best_score: float | None = None
    for path in candidate_body_paths:
        candidate = stage.GetPrimAtPath(path)
        if not _prim_valid(candidate):
            candidate_scores.append({"path": path, "valid": False, "score": -math.inf, "reasons": ["prim is not valid"]})
            continue
        score_row = _score_wheel_body_candidate(
            candidate,
            joint_name=joint_name,
            root_path=root_path,
            body0_path=body0_path,
            body1_path=body1_path,
            UsdPhysics=UsdPhysics,
            PhysxSchema=PhysxSchema,
        )
        candidate_scores.append(score_row)
        score = float(score_row.get("score", 0.0) or 0.0)
        if body is None or best_score is None or score > best_score:
            body = candidate
            best_score = score
    if body is not None and str(body.GetPath()) not in {body1_path, body0_path}:
        joint_warnings.append("wheel body resolved by scored subtree/body-name fallback after metadata and joint relationship lookup")
    if not _prim_valid(body):
        return WheelGroundDiagnostics(
            wheel_name=joint_name,
            joint_name=joint_name,
            joint_id=joint_id,
            joint_prim_path=str(joint_prim.GetPath()) if _prim_valid(joint_prim) else "",
            body0_path=body0_path,
            body1_path=body1_path,
            candidate_body_paths=candidate_body_paths,
            candidate_body_scores=candidate_scores,
            data_source="joint_metadata",
            collision_resolution_state=COLLIDER_RESOLUTION_FAILED,
            warnings=joint_warnings,
            reasons=["wheel body prim could not be resolved"],
        )

    body_path = str(body.GetPath())
    visual_paths: list[str] = []
    collision_paths: list[str] = []
    candidate_collision_paths: list[str] = []
    applied_schemas: list[str] = []
    collision_api_present = False
    collision_enabled = False
    visual_min_z: float | None = None
    collision_min_z: float | None = None
    collision_bound_info: dict[str, Any] | None = None
    collision_bound_invalid_reasons: list[str] = []
    visual_bound_invalid_reasons: list[str] = []
    approximation = ""
    for prim in _traverse_prim_subtree(body):
        if not _prim_valid(prim):
            continue
        schemas = _applied_schemas(prim)
        applied_schemas.extend(f"{prim.GetPath()}:{schema}" for schema in schemas)
        has_collision = _prim_has_collision_api(prim, UsdPhysics, PhysxSchema)
        if has_collision:
            path = str(prim.GetPath())
            candidate_collision_paths.append(path)
            enabled = _collision_enabled(prim, UsdPhysics)
            collision_api_present = True
            collision_enabled = collision_enabled or enabled
            approx = _collision_approximation(prim)
            if approx and not approximation:
                approximation = approx
            if enabled:
                collision_paths.append(path)
                bound = safe_prim_world_aabb(prim, bbox_cache, source=path)
                if bool(bound.get("valid", False)):
                    min_z = float((bound.get("min") or [0.0, 0.0, 0.0])[2])
                    collision_min_z = _min_optional(collision_min_z, min_z)
                    if collision_bound_info is None or min_z <= float((collision_bound_info.get("min") or [0.0, 0.0, math.inf])[2]):
                        collision_bound_info = bound
                else:
                    reason = str(bound.get("rejection_reason", "") or "invalid collision world bound")
                    live_bound = live_collider_world_aabb_from_body_pose(adapter, body_path, prim, bbox_cache)
                    if bool(live_bound.get("valid", False)):
                        min_z = float((live_bound.get("min") or [0.0, 0.0, 0.0])[2])
                        collision_min_z = _min_optional(collision_min_z, min_z)
                        if collision_bound_info is None or min_z <= float((collision_bound_info.get("min") or [0.0, 0.0, math.inf])[2]):
                            collision_bound_info = live_bound
                        joint_warnings.append(f"{path}: USD world bound invalid ({reason}); used live body pose/local bound fallback")
                    else:
                        collision_bound_invalid_reasons.append(f"{path}: {reason}; live fallback: {live_bound.get('rejection_reason', 'unavailable')}")
        elif _prim_is_visual(prim):
            path = str(prim.GetPath())
            visual_paths.append(path)
            bound = safe_prim_world_aabb(prim, bbox_cache, source=path)
            if bool(bound.get("valid", False)):
                visual_min_z = _min_optional(visual_min_z, float((bound.get("min") or [0.0, 0.0, 0.0])[2]))
            else:
                reason = str(bound.get("rejection_reason", "") or "invalid visual world bound")
                visual_bound_invalid_reasons.append(f"{path}: {reason}")
    visual_clearance = None if visual_min_z is None else float(visual_min_z) - ground_z
    collision_clearance = None if collision_min_z is None else float(collision_min_z) - ground_z
    if collision_api_present and collision_paths and collision_min_z is not None:
        resolution = GROUND_OK
        reasons: list[str] = []
    elif collision_api_present and collision_paths:
        resolution = COLLIDER_RESOLUTION_FAILED
        reasons = ["enabled collision geometry was found, but world bounds were invalid or unavailable"]
        reasons.extend(collision_bound_invalid_reasons[:4])
    elif candidate_collision_paths:
        resolution = COLLIDER_RESOLUTION_FAILED
        reasons = ["collision API was found, but no enabled collision geometry could be bounded"]
    else:
        resolution = COLLIDER_CONFIRMED_MISSING
        reasons = ["no collision API found on resolved wheel body or descendants"]
    return WheelGroundDiagnostics(
        wheel_name=joint_name,
        joint_name=joint_name,
        joint_prim_path=str(joint_prim.GetPath()) if _prim_valid(joint_prim) else "",
        joint_id=joint_id,
        body0_path=body0_path,
        body1_path=body1_path,
        resolved_body_path=body_path,
        body_prim_path=body_path,
        body_name=body_path.rstrip("/").split("/")[-1],
        data_source=str(collision_bound_info.get("source", "usd_bbox")) if collision_bound_info is not None else "usd_bbox",
        body_position_w=_body_position_w(adapter, body_path),
        candidate_body_paths=candidate_body_paths,
        candidate_body_scores=candidate_scores,
        candidate_collision_paths=candidate_collision_paths,
        collision_paths=collision_paths,
        visual_prim_paths=visual_paths,
        collision_prim_paths=collision_paths,
        collision_api_present=bool(collision_api_present),
        collision_enabled=bool(collision_enabled),
        physics_collision_enabled=bool(collision_enabled) if collision_api_present else None,
        collision_approximation=approximation,
        collision_resolution_state=resolution,
        applied_schemas=sorted(set(applied_schemas)),
        visual_aabb_min_z=visual_min_z,
        collision_aabb_min_z=collision_min_z,
        visual_ground_clearance_m=visual_clearance,
        collision_ground_clearance_m=collision_clearance,
        minimum_collision_z=collision_min_z,
        clearance_m=collision_clearance,
        bounds_valid=bool(collision_bound_info is not None),
        bounds_empty=bool(collision_bound_info.get("empty", False)) if collision_bound_info is not None else False,
        bounds_min=list(collision_bound_info.get("min", [])) if collision_bound_info is not None else [],
        bounds_max=list(collision_bound_info.get("max", [])) if collision_bound_info is not None else [],
        bounds_extent=list(collision_bound_info.get("extent", [])) if collision_bound_info is not None else [],
        bounds_rejection_reason="; ".join(collision_bound_invalid_reasons) if collision_bound_info is None else "",
        bounds_source=str(collision_bound_info.get("source", "")) if collision_bound_info is not None else "",
        bounds_finite=bool(collision_bound_info.get("finite", False)) if collision_bound_info is not None else False,
        collision_penetration_m=max(0.0, ground_z - float(collision_min_z)) if collision_min_z is not None else 0.0,
        warnings=joint_warnings + visual_bound_invalid_reasons[:4],
        reasons=reasons,
    )


def _adapter_joint_id(adapter: Any, joint_name: str) -> int | None:
    for mapping_name in ("wheel_name_to_id", "joint_name_to_id", "servo_name_to_id"):
        mapping = getattr(adapter, mapping_name, None)
        if isinstance(mapping, dict) and joint_name in mapping:
            try:
                return int(mapping[joint_name])
            except Exception:
                return None
    robot = getattr(adapter, "robot", None)
    names = list(getattr(robot, "joint_names", []) or [])
    if joint_name in names:
        return names.index(joint_name)
    return None


def _find_joint_relationships(root: Any, joint_name: str) -> tuple[Any | None, str, str, list[str]]:
    warnings: list[str] = []
    name_tokens = _wheel_name_tokens(joint_name)
    fallback_match = None
    for prim in _traverse_prim_subtree(root):
        if not _prim_valid(prim):
            continue
        name = str(prim.GetName())
        body0 = _relationship_target(prim, ("physics:body0", "body0"))
        body1 = _relationship_target(prim, ("physics:body1", "body1"))
        if name == joint_name:
            return prim, body0, body1, warnings
        joined = " ".join([name, body0, body1]).lower()
        if any(token in joined for token in name_tokens) and (body0 or body1):
            fallback_match = (prim, body0, body1)
    if fallback_match is not None:
        warnings.append("joint prim name did not match joint name; resolved by relationship target tokens")
        return fallback_match[0], fallback_match[1], fallback_match[2], warnings
    return None, "", "", warnings


def _candidate_body_paths_from_metadata(
    root: Any,
    joint_name: str,
    *,
    body_names: list[str],
    body0_path: str,
    body1_path: str,
) -> list[str]:
    paths: list[str] = []
    for path in (body1_path, body0_path):
        if path and path not in paths:
            paths.append(path)
    root_path = str(root.GetPath()) if _prim_valid(root) else ""
    tokens = _wheel_name_tokens(joint_name)
    for body_name in body_names:
        body_text = str(body_name)
        if any(token in body_text.lower() for token in tokens):
            for candidate in (f"{root_path}/{body_text}", body_text if body_text.startswith("/") else ""):
                if candidate and candidate not in paths:
                    paths.append(candidate)
    return paths


def _wheel_name_tokens(joint_name: str) -> list[str]:
    lowered = str(joint_name).lower()
    tokens = [lowered]
    for suffix in ("_ankle", "_wheel", "_joint"):
        if lowered.endswith(suffix):
            tokens.append(lowered[: -len(suffix)])
    tokens.append(lowered.replace("_ankle", "_wheel"))
    tokens.append(lowered.replace("_wheel", "_ankle"))
    return sorted({token for token in tokens if token}, key=len, reverse=True)


def _relationship_target(prim: Any, names: tuple[str, ...]) -> str:
    for rel_name in names:
        try:
            rel = prim.GetRelationship(rel_name)
            if rel:
                targets = list(rel.GetTargets())
                if targets:
                    return str(targets[0])
        except Exception:
            continue
    return ""


def _fallback_find_wheel_body(root: Any, wheel_name: str) -> Any | None:
    matches = _fallback_find_wheel_bodies(root, wheel_name)
    return matches[0] if matches else None


def _fallback_find_wheel_bodies(root: Any, wheel_name: str) -> list[Any]:
    exact = []
    tokens = _wheel_name_tokens(wheel_name)
    for prim in _traverse_prim_subtree(root):
        name = str(prim.GetName())
        if name == wheel_name or any(token in name.lower() for token in tokens):
            exact.append(prim)
    return exact


def _score_wheel_body_candidate(
    prim: Any,
    *,
    joint_name: str,
    root_path: str,
    body0_path: str,
    body1_path: str,
    UsdPhysics: Any,
    PhysxSchema: Any | None,
) -> dict[str, Any]:
    path = str(prim.GetPath())
    leaf = path.rstrip("/").split("/")[-1].lower()
    tokens = _wheel_name_tokens(joint_name)
    score = 0.0
    reasons: list[str] = []
    if path == body1_path:
        score += 50.0
        reasons.append("matches joint body1 child target")
    if path == body0_path:
        score += 10.0
        reasons.append("matches joint body0 parent target")
    if path == root_path:
        score -= 100.0
        reasons.append("candidate is articulation root")
    if any(token in leaf for token in tokens):
        score += 15.0
        reasons.append("name matches wheel/link token")
    if any(token in leaf for token in ("base", "chassis", "trunk")):
        score -= 40.0
        reasons.append("name looks like parent/base body")
    if _prim_has_rigid_body_api(prim, UsdPhysics):
        score += 30.0
        reasons.append("has RigidBodyAPI")
    else:
        reasons.append("RigidBodyAPI not found on candidate")
    collider_paths = [
        str(child.GetPath())
        for child in _traverse_prim_subtree(prim)
        if _prim_valid(child)
        and _prim_has_collision_api(child, UsdPhysics, PhysxSchema)
        and _collision_enabled(child, UsdPhysics)
    ]
    if collider_paths:
        score += 30.0
        reasons.append(f"enabled collider(s) in subtree: {len(collider_paths)}")
    else:
        reasons.append("no enabled collider found in subtree")
    try:
        if bool(prim.IsInstanceProxy()):
            score += 3.0
            reasons.append("candidate is instance proxy")
    except Exception:
        pass
    return {
        "path": path,
        "valid": True,
        "score": score,
        "reasons": reasons,
        "is_body0": bool(path == body0_path),
        "is_body1": bool(path == body1_path),
        "has_rigid_body_api": _prim_has_rigid_body_api(prim, UsdPhysics),
        "enabled_collider_paths": collider_paths[:20],
    }


def _traverse_prim_subtree(root: Any) -> list[Any]:
    if root is None:
        return []
    try:
        from pxr import Usd  # type: ignore

        children = list(root.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        return [root] + [child for child in children for child in _traverse_prim_subtree(child)]
    except Exception:
        pass
    try:
        children = list(root.GetChildren())
        return [root] + children + [child for prim in children for child in _traverse_prim_subtree(prim)]
    except Exception:
        return [root] if root is not None else []


def _prim_valid(prim: Any) -> bool:
    if prim is None:
        return False
    try:
        return bool(prim.IsValid())
    except Exception:
        return bool(prim)


def _prim_has_collision_api(prim: Any, UsdPhysics: Any, PhysxSchema: Any | None = None) -> bool:
    try:
        return bool(prim.HasAPI(UsdPhysics.CollisionAPI))
    except Exception:
        try:
            api = UsdPhysics.CollisionAPI(prim)
            if bool(api):
                return True
        except Exception:
            pass
    if PhysxSchema is not None:
        try:
            if bool(prim.HasAPI(PhysxSchema.PhysxCollisionAPI)):
                return True
        except Exception:
            try:
                if bool(PhysxSchema.PhysxCollisionAPI(prim)):
                    return True
            except Exception:
                pass
    try:
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr:
            return True
    except Exception:
        pass
    return False


def _prim_has_rigid_body_api(prim: Any, UsdPhysics: Any) -> bool:
    try:
        return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
    except Exception:
        try:
            api = UsdPhysics.RigidBodyAPI(prim)
            return bool(api)
        except Exception:
            return False


def _collision_enabled(prim: Any, UsdPhysics: Any) -> bool:
    try:
        api = UsdPhysics.CollisionAPI(prim)
        attr = api.GetCollisionEnabledAttr()
        if attr:
            value = attr.Get()
            return True if value is None else bool(value)
    except Exception:
        pass
    try:
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr:
            value = attr.Get()
            return True if value is None else bool(value)
    except Exception:
        pass
    return True


def _collision_approximation(prim: Any) -> str:
    for name in ("physics:approximation", "physxCollision:contactOffset", "physxCollision:restOffset"):
        try:
            attr = prim.GetAttribute(name)
            if attr:
                value = attr.Get()
                if value is not None:
                    return f"{name}={value}"
        except Exception:
            pass
    return ""


def _applied_schemas(prim: Any) -> list[str]:
    try:
        return [str(item) for item in prim.GetAppliedSchemas()]
    except Exception:
        return []


def _prim_is_visual(prim: Any) -> bool:
    try:
        type_name = str(prim.GetTypeName()).lower()
    except Exception:
        type_name = ""
    return any(token in type_name for token in ("mesh", "cube", "sphere", "cylinder", "capsule"))


def is_valid_world_bound(bound: Any, *, source: str = "") -> bool:
    return bool(sanitize_world_bound(bound, source=source).get("valid", False))


def sanitize_world_bound(
    bound: Any,
    *,
    source: str = "",
    max_abs_coord_m: float = MAX_REASONABLE_SCENE_COORD_M,
    max_extent_m: float = MAX_REASONABLE_BOUND_EXTENT_M,
) -> dict[str, Any]:
    """Return a plain-Python AABB description, rejecting USD empty/sentinel bounds."""

    reasons: list[str] = []
    empty = False
    try:
        empty = bool(bound.IsEmpty())
    except Exception:
        empty = False
    try:
        minimum = [float(v) for v in bound.GetMin()]
        maximum = [float(v) for v in bound.GetMax()]
    except Exception as exc:
        return {
            "valid": False,
            "empty": bool(empty),
            "min": [],
            "max": [],
            "extent": [],
            "finite": False,
            "source": str(source or ""),
            "rejection_reason": f"could not read bound min/max: {exc}",
        }
    if len(minimum) < 3 or len(maximum) < 3:
        reasons.append("bound min/max does not contain 3 coordinates")
    minimum = minimum[:3]
    maximum = maximum[:3]
    finite = all(math.isfinite(v) for v in [*minimum, *maximum])
    if empty:
        reasons.append("bound is empty")
    if not finite:
        reasons.append("bound contains non-finite coordinate")
    extent = [maximum[i] - minimum[i] for i in range(min(3, len(minimum), len(maximum)))]
    if len(extent) == 3:
        for axis, value in zip(("x", "y", "z"), extent):
            if math.isfinite(value) and value < 0.0:
                reasons.append(f"bound min is greater than max on {axis}")
        if any(math.isfinite(v) and v > float(max_extent_m) for v in extent):
            reasons.append(f"bound extent exceeds {float(max_extent_m):.3g}m")
    if any(math.isfinite(v) and abs(v) >= BOUND_SENTINEL_ABS for v in [*minimum, *maximum]):
        reasons.append("bound contains FLT_MAX-style sentinel coordinate")
    if any(math.isfinite(v) and abs(v) > float(max_abs_coord_m) for v in [*minimum, *maximum]):
        reasons.append(f"bound coordinate exceeds reasonable scene range {float(max_abs_coord_m):.3g}m")
    return {
        "valid": not reasons,
        "empty": bool(empty),
        "min": minimum,
        "max": maximum,
        "extent": extent,
        "finite": bool(finite),
        "source": str(source or ""),
        "rejection_reason": "; ".join(reasons),
    }


def safe_prim_world_aabb(prim: Any, bbox_cache: Any, *, source: str = "") -> dict[str, Any]:
    try:
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    except Exception as exc:
        return {
            "valid": False,
            "empty": False,
            "min": [],
            "max": [],
            "extent": [],
            "finite": False,
            "source": str(source or _prim_path(prim)),
            "rejection_reason": f"could not compute world bound: {exc}",
        }
    return sanitize_world_bound(aligned, source=str(source or _prim_path(prim)))


def live_collider_world_aabb_from_body_pose(adapter: Any, body_path: str, collider_prim: Any, bbox_cache: Any) -> dict[str, Any]:
    body_pos = _body_position_w(adapter, body_path)
    body_quat = _body_quaternion_w(adapter, body_path)
    if len(body_pos) < 3:
        return {
            "valid": False,
            "empty": False,
            "min": [],
            "max": [],
            "extent": [],
            "finite": False,
            "source": "live_body_pose_unavailable",
            "rejection_reason": "robot.data.body_pos_w unavailable for resolved body",
        }
    mesh_bound = live_collider_mesh_world_aabb_from_body_pose(adapter, body_path, collider_prim)
    if bool(mesh_bound.get("valid", False)):
        return mesh_bound
    try:
        local = bbox_cache.ComputeLocalBound(collider_prim).ComputeAlignedBox()
    except Exception as exc:
        return {
            "valid": False,
            "empty": False,
            "min": [],
            "max": [],
            "extent": [],
            "finite": False,
            "source": "live_body_pose_local_bound",
            "rejection_reason": f"could not compute local collider bound: {exc}",
        }
    local_info = sanitize_world_bound(local, source="local_collider_bound")
    if not bool(local_info.get("valid", False)):
        return {
            **local_info,
            "valid": False,
            "source": "live_body_pose_local_bound",
            "rejection_reason": str(local_info.get("rejection_reason", "invalid local bound")),
        }
    mins = list(local_info.get("min", []) or [])
    maxs = list(local_info.get("max", []) or [])
    if len(mins) < 3 or len(maxs) < 3:
        return {
            "valid": False,
            "empty": False,
            "min": [],
            "max": [],
            "extent": [],
            "finite": False,
            "source": "live_body_pose_local_bound",
            "rejection_reason": "local bound does not contain 3D min/max",
        }
    corners = [
        [x, y, z]
        for x in (float(mins[0]), float(maxs[0]))
        for y in (float(mins[1]), float(maxs[1]))
        for z in (float(mins[2]), float(maxs[2]))
    ]
    world: list[list[float]] = []
    for corner in corners:
        rotated = _quat_rotate_vec(body_quat, corner) if len(body_quat) >= 4 else list(corner)
        world.append([float(body_pos[i]) + float(rotated[i]) for i in range(3)])
    minimum = [min(point[i] for point in world) for i in range(3)]
    maximum = [max(point[i] for point in world) for i in range(3)]
    return sanitize_world_bound(
        _PlainBound(minimum, maximum),
        source=f"live_body_pose_local_bound:{body_path}:{_prim_path(collider_prim)}",
    )


def live_collider_mesh_world_aabb_from_body_pose(adapter: Any, body_path: str, collider_prim: Any) -> dict[str, Any]:
    """Build a world AABB from collider mesh points and live articulation body pose.

    USD/Fabric world bounds can be stale or sentinel-valued for dynamic articulations.
    The mesh vertices and local transforms are static, so we convert mesh points into
    body-local coordinates once per diagnostic call and then apply robot.data body pose.
    """

    body_pos = _body_position_w(adapter, body_path)
    body_quat = _body_quaternion_w(adapter, body_path)
    if len(body_pos) < 3:
        return _invalid_bound("live_body_mesh_points", "robot.data.body_pos_w unavailable for resolved body")
    stage = None
    body_prim = None
    try:
        stage = collider_prim.GetStage()
        body_prim = stage.GetPrimAtPath(body_path) if stage is not None else None
    except Exception:
        body_prim = None
    if not _prim_valid(body_prim):
        return _invalid_bound("live_body_mesh_points", f"body prim unavailable: {body_path}")
    body_local_bound = collider_mesh_body_local_aabb(body_prim, collider_prim)
    if not bool(body_local_bound.get("valid", False)):
        return body_local_bound
    mins = list(body_local_bound.get("min", []) or [])
    maxs = list(body_local_bound.get("max", []) or [])
    if len(mins) < 3 or len(maxs) < 3:
        return _invalid_bound("live_body_mesh_points", "body-local mesh bound missing min/max")
    corners = [
        [x, y, z]
        for x in (float(mins[0]), float(maxs[0]))
        for y in (float(mins[1]), float(maxs[1]))
        for z in (float(mins[2]), float(maxs[2]))
    ]
    world: list[list[float]] = []
    for corner in corners:
        rotated = _quat_rotate_vec(body_quat, corner) if len(body_quat) >= 4 else list(corner)
        world.append([float(body_pos[i]) + float(rotated[i]) for i in range(3)])
    minimum = [min(point[i] for point in world) for i in range(3)]
    maximum = [max(point[i] for point in world) for i in range(3)]
    info = sanitize_world_bound(
        _PlainBound(minimum, maximum),
        source=f"{body_local_bound.get('source', 'live_body_mesh_points')}:{body_path}:{_prim_path(collider_prim)}",
    )
    if bool(info.get("valid", False)):
        info["mesh_point_count"] = int(body_local_bound.get("mesh_point_count", 0) or 0)
        info["mesh_prim_paths"] = list(body_local_bound.get("mesh_prim_paths", []) or [])
    return info


def collider_mesh_body_local_aabb(body_prim: Any, collider_prim: Any) -> dict[str, Any]:
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore
    except Exception as exc:
        return _invalid_bound("live_body_mesh_points", f"pxr unavailable for mesh points: {exc}")
    meshes = _mesh_prims_under(collider_prim, UsdGeom)
    if not meshes:
        return _invalid_bound("live_body_mesh_points", f"no UsdGeom.Mesh under collider {_prim_path(collider_prim)}")
    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        body_world = cache.GetLocalToWorldTransform(body_prim)
        body_world_inv = body_world.GetInverse()
    except Exception as exc:
        return _invalid_bound("live_body_mesh_points", f"could not compute body transform: {exc}")
    body_points: list[list[float]] = []
    mesh_paths: list[str] = []
    source = "live_body_mesh_points"
    for mesh_prim in meshes:
        points, point_source = _mesh_points_or_extent(mesh_prim, UsdGeom)
        if not points:
            continue
        mesh_paths.append(_prim_path(mesh_prim))
        if point_source == "prototype_mesh_points":
            source = "prototype_mesh_points"
        try:
            mesh_world = cache.GetLocalToWorldTransform(mesh_prim)
        except Exception:
            continue
        for point in points:
            try:
                world_point = mesh_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                body_point = body_world_inv.Transform(world_point)
                body_points.append([float(body_point[0]), float(body_point[1]), float(body_point[2])])
            except Exception:
                continue
    if not body_points:
        return _invalid_bound(source, f"mesh points unavailable for collider {_prim_path(collider_prim)}")
    minimum = [min(point[i] for point in body_points) for i in range(3)]
    maximum = [max(point[i] for point in body_points) for i in range(3)]
    info = sanitize_world_bound(_PlainBound(minimum, maximum), source=source)
    info["mesh_point_count"] = int(len(body_points))
    info["mesh_prim_paths"] = mesh_paths
    return info


def _mesh_prims_under(prim: Any, UsdGeom: Any) -> list[Any]:
    meshes: list[Any] = []
    for candidate in _traverse_prim_subtree(prim):
        if not _prim_valid(candidate):
            continue
        try:
            if candidate.IsA(UsdGeom.Mesh):
                meshes.append(candidate)
                continue
        except Exception:
            pass
        try:
            if str(candidate.GetTypeName()).lower() == "mesh":
                meshes.append(candidate)
        except Exception:
            pass
    return meshes


def _mesh_points_or_extent(mesh_prim: Any, UsdGeom: Any) -> tuple[list[list[float]], str]:
    points = _mesh_points(mesh_prim, UsdGeom)
    if points:
        return points, "live_body_mesh_points"
    proto = None
    try:
        proto = mesh_prim.GetPrimInPrototype()
    except Exception:
        proto = None
    if _prim_valid(proto):
        points = _mesh_points(proto, UsdGeom)
        if points:
            return points, "prototype_mesh_points"
    return [], "live_body_mesh_points"


def _mesh_points(mesh_prim: Any, UsdGeom: Any) -> list[list[float]]:
    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        raw_points = mesh.GetPointsAttr().Get()
        points = _points_to_float_lists(raw_points)
        if points:
            return points
        extent = mesh.GetExtentAttr().Get()
        extent_points = _points_to_float_lists(extent)
        if len(extent_points) >= 2:
            mins = extent_points[0]
            maxs = extent_points[1]
            return [
                [x, y, z]
                for x in (mins[0], maxs[0])
                for y in (mins[1], maxs[1])
                for z in (mins[2], maxs[2])
            ]
    except Exception:
        return []
    return []


def _points_to_float_lists(points: Any) -> list[list[float]]:
    rows: list[list[float]] = []
    if points is None:
        return rows
    try:
        iterator = list(points)
    except Exception:
        return rows
    for point in iterator:
        try:
            row = [float(point[0]), float(point[1]), float(point[2])]
        except Exception:
            continue
        if all(math.isfinite(value) for value in row):
            rows.append(row)
    return rows


def _invalid_bound(source: str, reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "empty": False,
        "min": [],
        "max": [],
        "extent": [],
        "finite": False,
        "source": str(source),
        "rejection_reason": str(reason),
    }


def _prim_aabb_min_z(prim: Any, bbox_cache: Any) -> float | None:
    info = safe_prim_world_aabb(prim, bbox_cache)
    if not bool(info.get("valid", False)):
        return None
    try:
        return float((info.get("min") or [])[2])
    except Exception:
        return None


def _prim_path(prim: Any) -> str:
    try:
        return str(prim.GetPath())
    except Exception:
        return ""


class _PlainBound:
    def __init__(self, minimum: list[float], maximum: list[float]):
        self._minimum = list(minimum)
        self._maximum = list(maximum)

    def IsEmpty(self) -> bool:
        return False

    def GetMin(self) -> list[float]:
        return list(self._minimum)

    def GetMax(self) -> list[float]:
        return list(self._maximum)


def _body_position_w(adapter: Any, body_path: str) -> list[float]:
    return _body_pose_component(adapter, body_path, attr_name="body_pos_w", fallback_size=3)


def _body_quaternion_w(adapter: Any, body_path: str) -> list[float]:
    return _body_pose_component(adapter, body_path, attr_name="body_quat_w", fallback_size=4)


def _body_pose_component(adapter: Any, body_path: str, *, attr_name: str, fallback_size: int) -> list[float]:
    robot = getattr(adapter, "robot", None)
    data = getattr(robot, "data", None)
    index = _body_index_for_path(robot, body_path)
    value = getattr(data, attr_name, None)
    if value is None or index is None:
        return []
    try:
        return [float(item) for item in value[0, index].detach().cpu().reshape(-1).tolist()[:fallback_size]]
    except Exception:
        try:
            return [float(item) for item in value[0][index]][:fallback_size]
        except Exception:
            return []


def _body_index_for_path(robot: Any, body_path: str) -> int | None:
    body_names = list(getattr(robot, "body_names", []) or [])
    if not body_names:
        return None
    path = str(body_path or "")
    leaf = path.rstrip("/").split("/")[-1]
    for index, name in enumerate(body_names):
        text = str(name)
        if text == path or text == leaf or path.endswith("/" + text):
            return int(index)
    return None


def _quat_rotate_vec(quat_wxyz: list[float], vec: list[float]) -> list[float]:
    if len(quat_wxyz) < 4 or len(vec) < 3:
        return list(vec[:3])
    w, x, y, z = [float(value) for value in quat_wxyz[:4]]
    vx, vy, vz = [float(value) for value in vec[:3]]
    # q * v * q^-1, expanded to avoid importing numpy in no-sim tests.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _find_ground_collision_prim(ground_prim: Any) -> Any | None:
    try:
        from pxr import UsdPhysics  # type: ignore
        try:
            from pxr import PhysxSchema  # type: ignore
        except Exception:
            PhysxSchema = None  # type: ignore
    except Exception:
        UsdPhysics = None  # type: ignore
        PhysxSchema = None  # type: ignore
    first_valid = None
    for prim in _traverse_prim_subtree(ground_prim):
        if not _prim_valid(prim):
            continue
        if first_valid is None:
            first_valid = prim
        if UsdPhysics is not None and _prim_has_collision_api(prim, UsdPhysics, PhysxSchema):
            return prim
    return first_valid


def _prim_world_translation(prim: Any) -> list[float] | None:
    for attr_name in ("world_translation", "translation"):
        value = getattr(prim, attr_name, None)
        parsed = _vector3(value)
        if parsed is not None:
            return parsed
    try:
        from pxr import Usd, UsdGeom  # type: ignore

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        transform = cache.GetLocalToWorldTransform(prim)
        translation = transform.ExtractTranslation()
        return [float(translation[0]), float(translation[1]), float(translation[2])]
    except Exception:
        return None


def _prim_local_translation(prim: Any) -> list[float] | None:
    for attr_name in ("local_translation", "translation"):
        value = getattr(prim, attr_name, None)
        parsed = _vector3(value)
        if parsed is not None:
            return parsed
    try:
        from pxr import UsdGeom  # type: ignore

        xformable = UsdGeom.Xformable(prim)
        _reset_stack, ops = xformable.GetOrderedXformOps()
        for op in ops:
            if "translate" in str(op.GetOpName()).lower():
                value = op.Get()
                parsed = _vector3(value)
                if parsed is not None:
                    return parsed
    except Exception:
        pass
    return None


def _prim_world_transform_matrix(prim: Any) -> list[list[float]]:
    value = getattr(prim, "world_transform", None)
    if isinstance(value, list):
        return value
    try:
        from pxr import Usd, UsdGeom  # type: ignore

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        transform = cache.GetLocalToWorldTransform(prim)
        return [[float(transform[row][col]) for col in range(4)] for row in range(4)]
    except Exception:
        return []


def _vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = list(value)
        if len(values) >= 3:
            return [float(values[0]), float(values[1]), float(values[2])]
    except Exception:
        return None
    return None


def _resolve_stage(scene_handle: Any) -> Any | None:
    for owner in (scene_handle, getattr(scene_handle, "sim", None)):
        stage = getattr(owner, "stage", None)
        if stage is not None:
            return stage
    try:
        from isaaclab.sim import utils as sim_utils  # type: ignore

        return sim_utils.get_current_stage()
    except Exception:
        pass
    try:
        import omni.usd  # type: ignore

        return omni.usd.get_context().get_stage()
    except Exception:
        return None


def _fake_wheel_rows(scene_handle: Any, ground_z: float) -> list[WheelGroundDiagnostics] | None:
    rows = getattr(scene_handle, "fake_wheel_ground_rows", None)
    if rows is None:
        return None
    result: list[WheelGroundDiagnostics] = []
    for row in rows:
        if isinstance(row, WheelGroundDiagnostics):
            result.append(row)
        elif isinstance(row, dict):
            data = dict(row)
            if data.get("visual_ground_clearance_m") is None and data.get("visual_aabb_min_z") is not None:
                data["visual_ground_clearance_m"] = float(data["visual_aabb_min_z"]) - ground_z
            if data.get("collision_ground_clearance_m") is None and data.get("collision_aabb_min_z") is not None:
                data["collision_ground_clearance_m"] = float(data["collision_aabb_min_z"]) - ground_z
            if data.get("collision_penetration_m") is None and data.get("collision_aabb_min_z") is not None:
                data["collision_penetration_m"] = max(0.0, ground_z - float(data["collision_aabb_min_z"]))
            result.append(WheelGroundDiagnostics(**data))
    return result


def _safe_capture_sim_state(adapter: Any) -> dict[str, Any]:
    try:
        if hasattr(adapter, "capture_sim_state"):
            value = adapter.capture_sim_state()
            return dict(value or {})
    except Exception:
        pass
    return {}


def _safe_capture_command_state(adapter: Any) -> dict[str, Any]:
    try:
        value = adapter.capture_command_state()
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {"wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES}, "servos": {}}


def _read_nested(owner: Any, attr: str) -> Any:
    if owner is None:
        return None
    return _to_nested_list(getattr(owner, attr, None))


def _to_nested_list(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        pass
    try:
        return list(value)
    except Exception:
        return None


def _first_row(value: Any) -> list[float]:
    value = _to_nested_list(value)
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], (list, tuple)):
        return [_safe_float(item, 0.0) for item in value[0]]
    return [_safe_float(item, 0.0) for item in value]


def _vector_max_abs_delta(a: list[float], b: list[float]) -> float:
    count = min(len(a), len(b))
    if count <= 0:
        return 0.0
    return max(abs(float(a[index]) - float(b[index])) for index in range(count))


def _quat_angle_delta_deg(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    qa = _normalize_quat(a[:4])
    qb = _normalize_quat(b[:4])
    dot = abs(sum(qa[index] * qb[index] for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _normalize_quat(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(value) / norm for value in values]


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except Exception:
        pass
    return float(default)


def _min_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return float(candidate) if current is None else min(float(current), float(candidate))


def _timeline_is_playing() -> bool | None:
    try:
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
        return bool(timeline.is_playing())
    except Exception:
        return None
