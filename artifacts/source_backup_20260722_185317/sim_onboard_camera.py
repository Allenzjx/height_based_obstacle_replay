"""Isaac-worker onboard RGB-D camera helpers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from camera_validation import (
    CameraHealthStatus,
    compare_detection_to_expected,
    default_camera_health_status,
    default_height_validation_result,
    format_height_validation_summary,
)
from sim_camera_viewport import (
    close_onboard_camera_viewport,
    default_camera_viewport_status,
    open_onboard_camera_viewport,
    return_main_view_to_perspective,
    service_pending_camera_viewport,
    restore_previous_isaac_viewport,
)
from obstacle_height_vision import (
    HeightDetection,
    TemporalHeightFilter,
    VisionHeightConfig,
    build_obstacle_mask,
    estimate_height_from_depth,
    _transform_points,
)


ONBOARD_CAMERA_NAME = "onboard_rgbd_camera"


FRAMING_GOOD = "GOOD_OBLIQUE_VIEW"
FRAMING_TOO_TOP_DOWN = "TOO_TOP_DOWN"
FRAMING_TOO_HORIZONTAL = "TOO_HORIZONTAL"
FRAMING_OBSTACLE_NOT_VISIBLE = "OBSTACLE_NOT_VISIBLE"
FRAMING_TOP_PLANE_NOT_VISIBLE = "TOP_PLANE_NOT_VISIBLE"
FRAMING_GROUND_NOT_VISIBLE = "GROUND_NOT_VISIBLE"
FRAMING_POINTS_BACKWARD = "CAMERA_POINTS_BACKWARD"


@dataclass
class CameraGeometryDiagnostics:
    checked: bool = False
    camera_position_w: tuple[float, ...] = ()
    camera_quaternion_world: tuple[float, ...] = ()
    camera_quaternion_ros: tuple[float, ...] = ()
    optical_forward_w: tuple[float, ...] = ()
    optical_up_w: tuple[float, ...] = ()
    center_ray_direction_w: tuple[float, ...] = ()
    target_point_w: tuple[float, ...] = ()
    target_direction_w: tuple[float, ...] = ()
    angular_error_deg: float | None = None
    ground_intersection_w: tuple[float, ...] | None = None
    ground_intersection_distance_m: float | None = None
    obstacle_search_x_m: float | None = None
    top_point_fraction: float = 0.0
    obstacle_point_fraction: float = 0.0
    ground_point_fraction: float = 0.0
    valid_depth_fraction: float = 0.0
    framing_state: str = "NOT_CHECKED"
    reasons: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons or [])
        return data


def default_camera_geometry_diagnostics(reason: str = "not checked") -> dict[str, Any]:
    return CameraGeometryDiagnostics(reasons=[str(reason)]).to_dict()


def default_height_provenance(*, roi_source: str = "generated_scene_x_prior", obstacle_x_m: float | None = None, frame_revision: int = 0) -> dict[str, Any]:
    return {
        "height_source": "isaac_rgbd_depth_geometry",
        "camera_backend": "",
        "depth_data_type": "distance_to_image_plane",
        "depth_frame_revision": int(frame_revision),
        "intrinsics_used": False,
        "camera_world_pose_used": False,
        "ground_reference_source": "configured_world_ground_z",
        "ground_z_m": 0.0,
        "roi_source": str(roi_source),
        "obstacle_x_prior_used": obstacle_x_m is not None,
        "obstacle_x_prior_m": obstacle_x_m,
        "expected_height_used_by_detector": False,
        "generated_height_used_by_detector": False,
        "scene_obstacle_height_used_by_detector": False,
        "usd_geometry_height_used_by_detector": False,
        "detector_input_audit_passed": False,
        "forbidden_input_reason": "",
    }


def default_height_measurement_evidence(
    *,
    roi_source: str = "generated_scene_x_prior",
    obstacle_x_m: float | None = None,
    frame_revision: int = 0,
) -> dict[str, Any]:
    return {
        "height_source": "isaac_rgbd_depth_geometry",
        "depth_frame_revision": int(frame_revision),
        "depth_timestamp": 0.0,
        "depth_fingerprint": "",
        "depth_shape": [],
        "depth_finite_ratio": 0.0,
        "intrinsics_fingerprint": "",
        "intrinsics_used": False,
        "camera_position_w": [],
        "camera_quaternion_w": [],
        "camera_world_pose_used": False,
        "roi_source": str(roi_source),
        "obstacle_x_prior_used": obstacle_x_m is not None,
        "obstacle_x_prior_m": obstacle_x_m,
        "ground_reference_source": "configured_world_ground_z",
        "ground_z_m": 0.0,
        "top_z_m": None,
        "raw_height_cm": None,
        "candidate_height_cm": None,
        "detected_height_cm": None,
        "top_point_count": 0,
        "obstacle_point_count": 0,
        "ground_point_count": 0,
        "expected_height_used_by_detector": False,
        "generated_height_used_by_detector": False,
        "scene_obstacle_height_used_by_detector": False,
        "usd_geometry_height_used_by_detector": False,
        "detector_input_audit_passed": False,
        "forbidden_input_reason": "",
    }


def camera_pitch_quat_wxyz(pitch_deg: float) -> tuple[float, float, float, float]:
    """World-convention local pitch: positive tilts +X forward down toward -Z."""

    half = math.radians(float(pitch_deg)) * 0.5
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def camera_look_at_quat_wxyz(
    *,
    camera_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    roll_deg: float = 0.0,
    convention: str = "world",
) -> tuple[float, float, float, float]:
    """Return an offset quaternion that aims the convention-specific optical axis at a target."""

    import numpy as np  # type: ignore

    pos = np.asarray(camera_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    forward = target - pos
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        return camera_pitch_quat_wxyz(14.0)
    forward = forward / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(world_up, forward))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    side = np.cross(world_up, forward)
    side = side / max(1.0e-9, float(np.linalg.norm(side)))
    up = np.cross(forward, side)
    up = up / max(1.0e-9, float(np.linalg.norm(up)))
    convention_key = str(convention or "world").lower()
    if convention_key == "world":
        # Isaac Lab world convention: local +X is optical forward and local +Z is up.
        local_x = forward
        local_z = up
        local_y = np.cross(local_z, local_x)
    elif convention_key == "ros":
        # ROS optical convention: local +Z is optical forward and local -Y is up.
        local_z = forward
        local_y = -up
        local_x = np.cross(local_y, local_z)
    elif convention_key == "opengl":
        # OpenGL convention: local -Z is optical forward and local +Y is up.
        local_z = -forward
        local_y = up
        local_x = np.cross(local_y, local_z)
    else:
        raise ValueError(f"unsupported camera convention: {convention}")
    local_x = local_x / max(1.0e-9, float(np.linalg.norm(local_x)))
    local_y = local_y / max(1.0e-9, float(np.linalg.norm(local_y)))
    local_z = local_z / max(1.0e-9, float(np.linalg.norm(local_z)))
    rotation = np.column_stack((local_x, local_y, local_z))
    if abs(float(roll_deg)) > 1.0e-9:
        roll = math.radians(float(roll_deg))
        c = math.cos(roll)
        s = math.sin(roll)
        if convention_key == "world":
            roll_m = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
        elif convention_key == "ros":
            roll_m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        else:
            roll_m = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        rotation = rotation @ roll_m
    return _quat_from_matrix(rotation)


def _quat_from_matrix(matrix: Any) -> tuple[float, float, float, float]:
    import numpy as np  # type: ignore

    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (float(w / norm), float(x / norm), float(y / norm), float(z / norm))


def resolve_camera_parent_prim(stage: Any, robot_root: str, explicit_path: str | None = None) -> tuple[str | None, str]:
    """Resolve a robot body prim for mounting the onboard camera."""

    robot_root = str(robot_root or "").rstrip("/")
    explicit = str(explicit_path or "").strip()
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception as exc:
        return None, f"UsdPhysics unavailable while resolving camera parent: {exc}"

    if explicit:
        prim = stage.GetPrimAtPath(explicit)
        if not prim or not prim.IsValid():
            return None, f"--camera-parent-prim does not exist: {explicit}"
        return explicit, ""

    candidates: list[tuple[int, int, str]] = []
    preferred = ("base", "body", "chassis", "trunk")
    try:
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not path.startswith(robot_root + "/") and path != robot_root:
                continue
            try:
                is_body = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            except Exception:
                is_body = False
            if not is_body:
                continue
            name = prim.GetName().lower()
            priority = 0 if any(token in name for token in preferred) else 1
            depth = path.count("/")
            candidates.append((priority, depth, path))
    except Exception as exc:
        return None, f"Could not scan robot rigid bodies under {robot_root}: {exc}"

    if not candidates:
        return None, f"No RigidBodyAPI prim found under {robot_root}; onboard vision disabled."
    candidates.sort()
    return candidates[0][2], ""


def create_onboard_camera(scene_handle: Any, *, robot_root: str) -> tuple[Any | None, str, str, str]:
    """Create a Camera sensor before ``sim.reset`` if enabled in the scene config."""

    config = scene_handle.config
    if not bool(getattr(config, "onboard_camera_enabled", True)):
        return None, "", "", "onboard camera disabled by config"
    try:
        from isaaclab.sensors.camera import Camera, CameraCfg  # type: ignore
        import isaaclab.sim as sim_utils  # type: ignore

        stage = sim_utils.get_current_stage()
        parent, error = resolve_camera_parent_prim(
            stage,
            robot_root,
            str(getattr(config, "camera_parent_prim", "") or ""),
        )
        if not parent:
            return None, "", "", error
        prim_path = f"{parent.rstrip('/')}/{ONBOARD_CAMERA_NAME}"
        offset_pos = tuple(float(v) for v in getattr(config, "camera_offset_pos", (0.35, 0.0, 0.18)))
        convention = str(getattr(config, "camera_offset_convention", "world") or "world")
        if str(getattr(config, "camera_aim_mode", "pitch") or "pitch").lower() == "look-at":
            target_position = (
                float(getattr(config, "camera_target_x", getattr(config, "obstacle_x", 1.55))),
                float(getattr(config, "camera_target_y", 0.0)),
                float(getattr(config, "camera_target_z", 0.02)),
            )
            target_frame = str(getattr(config, "camera_target_frame", "world") or "world").lower()
            if target_frame == "world":
                target_position = _world_point_to_parent_local(stage, parent, target_position)
            elif target_frame != "parent":
                return None, "", "", f"unsupported camera target frame: {target_frame}"
            offset_rot = camera_look_at_quat_wxyz(
                camera_position=offset_pos,
                target_position=target_position,
                roll_deg=float(getattr(config, "camera_look_at_roll_deg", 0.0)),
                convention=convention,
            )
        else:
            offset_rot = tuple(float(v) for v in getattr(config, "camera_offset_rot", camera_pitch_quat_wxyz(14.0)))
        camera_cfg = CameraCfg(
            prim_path=prim_path,
            update_period=float(getattr(config, "camera_update_period_s", 0.1)),
            height=int(getattr(config, "camera_height", 240)),
            width=int(getattr(config, "camera_width", 424)),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(getattr(config, "camera_focal_length", 24.0)),
                focus_distance=400.0,
                horizontal_aperture=float(getattr(config, "camera_horizontal_aperture", 20.955)),
                clipping_range=(
                    float(getattr(config, "camera_near_clip_m", 0.05)),
                    float(getattr(config, "camera_far_clip_m", 6.0)),
                ),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=offset_pos,
                rot=offset_rot,
                convention=convention,
            ),
            update_latest_camera_pose=True,
        )
        camera = Camera(cfg=camera_cfg)
        return camera, prim_path, parent, ""
    except Exception as exc:
        return None, "", "", f"onboard camera creation failed: {exc}"


def _world_point_to_parent_local(stage: Any, parent_prim_path: str, point_w: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        from pxr import Gf, UsdGeom  # type: ignore

        parent_prim = stage.GetPrimAtPath(str(parent_prim_path))
        cache = UsdGeom.XformCache()
        world = cache.GetLocalToWorldTransform(parent_prim)
        inverse = world.GetInverse()
        local = inverse.Transform(Gf.Vec3d(float(point_w[0]), float(point_w[1]), float(point_w[2])))
        return (float(local[0]), float(local[1]), float(local[2]))
    except Exception:
        return (float(point_w[0]), float(point_w[1]), float(point_w[2]))


def reset_camera(scene_handle: Any) -> None:
    camera = getattr(scene_handle, "camera", None)
    if camera is None:
        return
    try:
        camera.reset()
        camera.update(0.0, force_recompute=True)
        _ = camera.data
    except Exception as exc:
        scene_handle.camera_error = f"camera reset failed: {exc}"


class OnboardCameraProcessor:
    """Small worker-side state machine for camera height detection."""

    def __init__(self, scene_handle: Any, vision_config: VisionHeightConfig | None = None):
        self.scene_handle = scene_handle
        self.config = vision_config or VisionHeightConfig()
        self.filter = TemporalHeightFilter(self.config)
        self.enabled = bool(getattr(scene_handle.config, "onboard_camera_enabled", True))
        self.frame_revision = 0
        self.last_frame_at = 0.0
        self.last_update_sim_time = -1.0e9
        self.last_detection = HeightDetection.invalid("no frame processed")
        self.last_filtered_detection = HeightDetection.invalid("no stable detection")
        self.detect_once_requested = False
        self.save_debug_requested = False
        self.debug_expected_height_cm: int | None = None
        self.debug_image_path = ""
        self.debug_sidecar_path = ""
        self.camera_health_status = default_camera_health_status(reason="not checked")
        self.height_validation_result = default_height_validation_result(reason="not checked")
        self.camera_mount_validation: dict[str, Any] = {"checked": False, "passed": False, "pending": False, "reasons": ["not checked"]}
        self.camera_view_status = default_camera_viewport_status("not requested")
        self.source_mode = "generated"
        self.roi_source = "generated_scene_x_prior"
        self.height_provenance = default_height_provenance()
        self.height_measurement_evidence = default_height_measurement_evidence()
        self.camera_geometry_diagnostics = default_camera_geometry_diagnostics()
        self._camera_health_check: dict[str, Any] | None = None
        self._height_validation_expected_cm: int | None = None
        self._pending_mount_check: dict[str, Any] | None = None
        self._last_frame_sample: dict[str, Any] = {}
        self.consecutive_errors = 0
        self.last_error_log_at = 0.0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.filter.reset()

    def reset_filter(self) -> None:
        self.filter.reset()
        self.last_filtered_detection = HeightDetection.invalid("filter reset")

    def detect_once(self) -> None:
        self.detect_once_requested = True

    def save_debug_frame(self, *, expected_height_cm: int | None = None) -> None:
        self.save_debug_requested = True
        self.detect_once_requested = True
        expected = _optional_int(expected_height_cm)
        if expected is None and bool(self.height_validation_result.get("checked", False)):
            expected = _optional_int(self.height_validation_result.get("expected_height_cm"))
        self.debug_expected_height_cm = expected

    def request_camera_validation(self, *, duration_s: float = 1.0, min_frames: int = 2) -> None:
        self._camera_health_check = {
            "start_revision": int(self.frame_revision),
            "start_time": time.time(),
            "deadline": time.time() + max(0.05, float(duration_s)),
            "min_frames": max(1, int(min_frames)),
            "fingerprints": [],
        }
        self.camera_health_status = default_camera_health_status(reason="camera health check running")

    def request_height_validation(self, expected_height_cm: int) -> None:
        self._height_validation_expected_cm = int(expected_height_cm)
        self.height_validation_result = default_height_validation_result(checked=False, reason="height validation pending")
        self.request_camera_validation(duration_s=1.0, min_frames=2)
        self.detect_once_requested = True

    def clear_validation_result(self) -> None:
        self._height_validation_expected_cm = None
        self._camera_health_check = None
        self.camera_health_status = default_camera_health_status(reason="cleared")
        self.height_validation_result = default_height_validation_result(reason="cleared")
        self.camera_mount_validation = {"checked": False, "passed": False, "pending": False, "reasons": ["cleared"]}
        self._pending_mount_check = None

    def request_mount_validation_after_next_respawn(self) -> tuple[bool, str]:
        relative = _camera_relative_matrix(self.scene_handle)
        if relative is None:
            self.camera_mount_validation = {
                "checked": True,
                "passed": False,
                "pending": False,
                "reasons": ["could not read camera relative transform"],
            }
            return False, "could not read camera relative transform"
        self._pending_mount_check = {"initial_relative": relative}
        self.camera_mount_validation = {
            "checked": False,
            "passed": False,
            "pending": True,
            "position_error_m": None,
            "orientation_error_deg": None,
            "position_tolerance_m": 0.002,
            "orientation_tolerance_deg": 0.5,
            "reasons": ["waiting for next explicit respawn"],
        }
        return True, "camera mount validation armed for next respawn"

    def set_source_mode(self, source_mode: str) -> None:
        text = str(source_mode or "").strip().lower()
        if text in {"external", "unknown", "external_unknown", "external_unknown_obstacle"}:
            self.source_mode = "external"
            self.roi_source = "camera_forward_auto"
        else:
            self.source_mode = "generated"
            self.roi_source = "generated_scene_x_prior"

    def _viewport_payload(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_revision = int(payload.get("action_revision", payload.get("request_revision", 0)) or 0)
        request_id = str(payload.get("request_id", "") or "")
        status = default_camera_viewport_status("")
        status.update(
            {
                "requested_action": action,
                "request_id": request_id,
                "request_revision": request_revision,
                "completed_revision": 0,
                "completed": False,
                "action_revision": request_revision,
            }
        )
        return status

    def notify_respawn(self) -> None:
        pending = self._pending_mount_check
        if pending is None:
            return
        initial = pending.get("initial_relative")
        current = _camera_relative_matrix(self.scene_handle)
        if initial is None or current is None:
            self.camera_mount_validation = {
                "checked": True,
                "passed": False,
                "pending": False,
                "reasons": ["could not read camera relative transform after respawn"],
            }
            self._pending_mount_check = None
            return
        position_error_m, orientation_error_deg = _relative_transform_error(initial, current)
        passed = position_error_m <= 0.002 and orientation_error_deg <= 0.5
        reasons = []
        if not passed:
            if position_error_m > 0.002:
                reasons.append(f"relative position error {position_error_m:.6f}m exceeds 0.002m")
            if orientation_error_deg > 0.5:
                reasons.append(f"relative orientation error {orientation_error_deg:.3f}deg exceeds 0.5deg")
        self.camera_mount_validation = {
            "checked": True,
            "passed": bool(passed),
            "pending": False,
            "position_error_m": float(position_error_m),
            "orientation_error_deg": float(orientation_error_deg),
            "position_tolerance_m": 0.002,
            "orientation_tolerance_deg": 0.5,
            "reasons": reasons,
        }
        self._pending_mount_check = None

    def handle_control(self, action: str, **payload: Any) -> tuple[bool, str]:
        action = str(action or "").strip().lower()
        if action == "enable":
            self.set_enabled(True)
            return True, "vision enabled"
        if action == "disable":
            self.set_enabled(False)
            return True, "vision disabled"
        if action == "detect_once":
            self.detect_once()
            return True, "vision detect_once requested"
        if action == "reset_filter":
            self.reset_filter()
            return True, "vision filter reset"
        if action == "save_debug_frame":
            self.save_debug_frame(expected_height_cm=payload.get("expected_height_cm"))
            return True, "vision debug frame requested"
        if action == "validate_camera":
            self.request_camera_validation()
            return True, "camera validation requested"
        if action == "validate_current_height":
            expected = _optional_int(payload.get("expected_height_cm"))
            if expected is None:
                return False, "validate_current_height requires expected_height_cm"
            self.request_height_validation(expected)
            return True, f"height validation requested for {expected}cm"
        if action == "save_rgbd_diagnostic":
            self.save_debug_frame(expected_height_cm=payload.get("expected_height_cm"))
            return True, "RGB-D diagnostic save requested"
        if action == "clear_validation_result":
            self.clear_validation_result()
            return True, "vision validation result cleared"
        if action == "validate_mount_after_next_respawn":
            return self.request_mount_validation_after_next_respawn()
        if action == "set_source_mode":
            self.set_source_mode(str(payload.get("source_mode", "") or "generated"))
            return True, f"vision source mode set to {self.source_mode}"
        if action == "validate_camera_geometry":
            try:
                frame = _read_camera_frame(getattr(self.scene_handle, "camera", None))
                self.camera_geometry_diagnostics = inspect_camera_geometry(self.scene_handle, frame, self.config, roi_source=self.roi_source).to_dict()
                return bool(self.camera_geometry_diagnostics.get("checked", False)), self.camera_geometry_diagnostics.get("framing_state", "camera geometry checked")
            except Exception as exc:
                self.camera_geometry_diagnostics = default_camera_geometry_diagnostics(str(exc))
                return False, str(exc)
        if action in {"show_camera_view", "open_camera_viewport"}:
            self.camera_view_status = self._viewport_payload("open_camera_viewport", payload)
            if bool(payload.get("headless", False)):
                self.camera_view_status = default_camera_viewport_status("headless mode does not support camera viewport")
                self.camera_view_status["requested"] = True
                self.camera_view_status["requested_action"] = "open_camera_viewport"
                self.camera_view_status["request_id"] = str(payload.get("request_id", "") or "")
                self.camera_view_status["request_revision"] = int(payload.get("action_revision", payload.get("request_revision", 0)) or 0)
                self.camera_view_status["completed_revision"] = self.camera_view_status["request_revision"]
                self.camera_view_status["completed"] = True
                return False, "camera viewport unsupported in headless mode"
            camera_path = str(payload.get("camera_prim_path", "") or getattr(self.scene_handle, "camera_prim_path", "") or "")
            viewport_payload = dict(payload)
            viewport_payload.pop("camera_prim_path", None)
            result = open_onboard_camera_viewport(camera_path, **viewport_payload)
            self.camera_view_status = result.to_dict()
            if bool(result.pending):
                return True, result.error or "camera viewport pending"
            return bool(result.supported and result.active), result.error or "camera viewport requested"
        if action == "return_main_view_to_perspective":
            self.camera_view_status = self._viewport_payload("return_main_view_to_perspective", payload)
            result = return_main_view_to_perspective(self.scene_handle, **payload)
            self.camera_view_status = result.to_dict()
            return bool(result.supported and result.completed), result.error or "main viewport returned to Perspective"
        if action == "close_camera_viewport":
            self.camera_view_status = self._viewport_payload("close_camera_viewport", payload)
            result = close_onboard_camera_viewport(**payload)
            self.camera_view_status = result.to_dict()
            return bool(result.completed), result.error or "camera viewport closed"
        if action == "restore_camera_view":
            self.camera_view_status = self._viewport_payload("restore_camera_view", payload)
            result = restore_previous_isaac_viewport(self.scene_handle, **payload)
            self.camera_view_status = result.to_dict()
            return bool(result.supported and result.active), result.error or "camera viewport restored"
        return False, f"unknown vision_control action: {action}"

    def update(self, *, dt: float, sim_time: float, wall_time: float | None = None) -> dict[str, Any]:
        camera = getattr(self.scene_handle, "camera", None)
        if camera is None:
            return self.status(failure_reason=getattr(self.scene_handle, "camera_error", "") or "camera unavailable")
        now = time.time() if wall_time is None else float(wall_time)
        try:
            camera.update(float(dt))
        except Exception as exc:
            self._record_frame_error(f"camera update failed: {exc}")
            return self.status(failure_reason=self.last_detection.reason)
        due = (float(sim_time) - self.last_update_sim_time) >= max(0.0, float(getattr(self.scene_handle.config, "camera_update_period_s", 0.1)))
        explicit_detect = bool(self.detect_once_requested or self.save_debug_requested or self._height_validation_expected_cm is not None)
        health_active = self._camera_health_check is not None
        should_capture = (due and (self.enabled or health_active)) or explicit_detect
        if not should_capture:
            return self.status()

        should_detect = bool((self.enabled and due) or explicit_detect)
        self.detect_once_requested = False
        self.last_update_sim_time = float(sim_time)
        self.frame_revision += 1
        self.last_frame_at = now
        try:
            frame = _read_camera_frame(camera)
            self._last_frame_sample = frame
            self._collect_camera_health_frame(frame)
            if should_detect:
                depth = _require_array(frame, "depth", "camera depth output missing")
                intrinsics = _require_array(frame, "intrinsics", "camera intrinsics missing")
                pos_w = _require_array(frame, "pos_w", "camera world position missing")
                quat_ros = _require_array(frame, "quat_w", "camera world orientation missing")
                obstacle_x_m = self._current_roi_obstacle_x()
                self.height_provenance = build_height_provenance(
                    scene_handle=self.scene_handle,
                    frame=frame,
                    frame_revision=int(self.frame_revision),
                    roi_source=self.roi_source,
                    obstacle_x_m=obstacle_x_m,
                    detector_input_keys={
                        "depth_image",
                        "intrinsic_matrix",
                        "camera_position_w",
                        "camera_quat_wxyz",
                        "config",
                        "obstacle_x_m",
                        "timestamp",
                        "near_clip_m",
                        "far_clip_m",
                    },
                )
                self.camera_geometry_diagnostics = inspect_camera_geometry(
                    self.scene_handle,
                    frame,
                    self.config,
                    roi_source=self.roi_source,
                    obstacle_x_m=obstacle_x_m,
                ).to_dict()
                framing_state = str(self.camera_geometry_diagnostics.get("framing_state", ""))
                if bool(getattr(self.scene_handle.config, "camera_coverage_strict", False)) and framing_state != FRAMING_GOOD:
                    detection = HeightDetection.invalid(f"camera coverage strict: {framing_state}", timestamp=now)
                else:
                    detection = estimate_height_from_depth(
                        depth,
                        intrinsics,
                        pos_w,
                        quat_ros,
                        config=self.config,
                        obstacle_x_m=obstacle_x_m,
                        timestamp=now,
                        near_clip_m=float(getattr(self.scene_handle.config, "camera_near_clip_m", 0.05)),
                        far_clip_m=float(getattr(self.scene_handle.config, "camera_far_clip_m", 6.0)),
                    )
                filtered = self.filter.update(detection)
                self.last_detection = detection
                self.last_filtered_detection = filtered
                self.height_measurement_evidence = build_height_measurement_evidence(
                    scene_handle=self.scene_handle,
                    frame=frame,
                    frame_revision=int(self.frame_revision),
                    detection=detection,
                    filtered_detection=filtered,
                    roi_source=self.roi_source,
                    obstacle_x_m=obstacle_x_m,
                    height_provenance=self.height_provenance,
                )
            self.consecutive_errors = 0
            self._maybe_finish_camera_health_check(now)
            self._maybe_finish_height_validation()
            if self.save_debug_requested:
                detection_for_debug = self.last_filtered_detection if self.last_filtered_detection.stable else self.last_detection
                saved = save_rgbd_diagnostic(
                    frame=frame,
                    detection=detection_for_debug,
                    config=self.config,
                    root=Path(__file__).resolve().parent / "saved_height_steps" / "vision_debug",
                    camera_health=self.camera_health_status,
                    height_validation=self.height_validation_result,
                    expected_height_cm=self.debug_expected_height_cm,
                    camera_prim_path=str(getattr(self.scene_handle, "camera_prim_path", "") or ""),
                    parent_prim_path=str(getattr(self.scene_handle, "camera_parent_prim", "") or ""),
                    frame_revision=int(self.frame_revision),
                    obstacle_x_m=self._current_roi_obstacle_x(),
                    camera_geometry=self.camera_geometry_diagnostics,
                    height_provenance=self.height_provenance,
                    source_mode=self.source_mode,
                    roi_source=self.roi_source,
                    detection_revision=int(self.filter.detection_revision),
                )
                self.debug_image_path = str(saved.get("diagnostic_image_path", ""))
                self.debug_sidecar_path = str(saved.get("json_path", ""))
                self.save_debug_requested = False
                self.debug_expected_height_cm = None
            return self.status()
        except Exception as exc:
            self.save_debug_requested = False
            self.debug_expected_height_cm = None
            self._record_frame_error(f"camera detection failed: {exc}")
            self._maybe_finish_camera_health_check(now, force=True)
            self._maybe_finish_height_validation()
            return self.status(failure_reason=self.last_detection.reason)

    def service_pending_camera_viewport(self) -> None:
        try:
            if not bool(self.camera_view_status.get("pending", False)):
                return
            status = service_pending_camera_viewport().to_dict()
            self.camera_view_status = status
        except Exception as exc:
            current = dict(self.camera_view_status)
            current["pending"] = False
            current["completed"] = True
            current["supported"] = False
            current["active"] = False
            current["error"] = str(exc)
            self.camera_view_status = current

    def _collect_camera_health_frame(self, frame: dict[str, Any]) -> None:
        if self._camera_health_check is None:
            return
        fingerprints = self._camera_health_check.setdefault("fingerprints", [])
        fingerprint = _depth_fingerprint(frame.get("depth"))
        if fingerprint:
            fingerprints.append(fingerprint)
            del fingerprints[:-8]

    def _maybe_finish_camera_health_check(self, now: float, *, force: bool = False) -> None:
        check = self._camera_health_check
        if check is None:
            return
        frames_advanced = max(0, int(self.frame_revision) - int(check.get("start_revision", 0)))
        deadline = float(check.get("deadline", now))
        min_frames = max(1, int(check.get("min_frames", 1)))
        if not force and frames_advanced < min_frames and now < deadline:
            return
        self.camera_health_status = inspect_camera_health(
            self.scene_handle,
            self,
            latest_frame=self._last_frame_sample,
            now=now,
            frames_advanced_during_check=frames_advanced,
            depth_fingerprints=tuple(check.get("fingerprints", ())),
        ).to_dict()
        self._camera_health_check = None

    def _maybe_finish_height_validation(self) -> None:
        expected = self._height_validation_expected_cm
        if expected is None:
            return
        health = self.camera_health_status if isinstance(self.camera_health_status, dict) else {}
        if not bool(health.get("checked", False)):
            return
        result = compare_detection_to_expected(
            self.status(),
            expected,
            raw_height_tolerance_cm=1.0,
            minimum_confidence=float(self.config.minimum_confidence),
            minimum_obstacle_points=int(self.config.minimum_obstacle_points),
            quantization_tolerance_cm=float(self.config.quantization_tolerance_cm),
        )
        self.height_validation_result = result.to_dict()
        self._height_validation_expected_cm = None

    def status(self, *, failure_reason: str | None = None) -> dict[str, Any]:
        camera_ready = getattr(self.scene_handle, "camera", None) is not None and not getattr(self.scene_handle, "camera_error", "")
        stable = self.last_filtered_detection if self.last_filtered_detection.stable else None
        latest = self.last_detection
        reason = str(failure_reason if failure_reason is not None else ("" if camera_ready else getattr(self.scene_handle, "camera_error", "")))
        if not self.enabled and not reason:
            reason = "disabled"
        ground_surface = dict(getattr(self.scene_handle, "ground_surface_info", {}) or {})
        return {
            "enabled": bool(self.enabled),
            "camera_ready": bool(camera_ready),
            "camera_parent_prim": str(getattr(self.scene_handle, "camera_parent_prim", "") or ""),
            "camera_prim_path": str(getattr(self.scene_handle, "camera_prim_path", "") or ""),
            "frame_revision": int(self.frame_revision),
            "detection_revision": int(self.filter.detection_revision),
            "last_frame_at": float(self.last_frame_at),
            "raw_height_cm": latest.raw_height_cm,
            "detected_height_cm": stable.detected_height_cm if stable is not None else None,
            "candidate_height_cm": latest.detected_height_cm,
            "confidence": float(stable.confidence if stable is not None else latest.confidence),
            "stable": bool(stable is not None and stable.valid),
            "stable_count": int(self.filter.stable_count),
            "stable_required": int(self.config.stable_frames_required),
            "valid_point_count": int(latest.point_count),
            "top_plane_mad_m": latest.top_plane_mad_m,
            "quantization_error_cm": latest.quantization_error_cm,
            "method": "rgbd_geometry",
            "failure_reason": reason or latest.reason,
            "debug_image_path": str(self.debug_image_path or ""),
            "debug_sidecar_path": str(self.debug_sidecar_path or ""),
            "debug_folder": str(Path(__file__).resolve().parent / "saved_height_steps" / "vision_debug"),
            "camera_health": dict(self.camera_health_status),
            "height_validation": dict(self.height_validation_result),
            "height_validation_summary": format_height_validation_summary(self.height_validation_result),
            "camera_mount_validation": dict(self.camera_mount_validation),
            "camera_view": dict(self.camera_view_status),
            "camera_geometry": dict(self.camera_geometry_diagnostics),
            "camera_coverage": dict(self.camera_geometry_diagnostics),
            "ground_surface": ground_surface,
            "height_provenance": dict(self.height_provenance),
            "height_measurement_evidence": dict(self.height_measurement_evidence),
            "source_mode": str(self.source_mode),
            "roi_source": str(self.roi_source),
            "consecutive_errors": int(self.consecutive_errors),
        }

    def _record_frame_error(self, reason: str) -> None:
        self.consecutive_errors += 1
        self.last_detection = HeightDetection.invalid(reason)

    def _current_roi_obstacle_x(self) -> float | None:
        if self.source_mode == "external":
            self.roi_source = "camera_forward_auto"
            return None
        self.roi_source = "generated_scene_x_prior"
        return float(getattr(self.scene_handle.config, "obstacle_x", 1.55))


def inspect_camera_health(
    scene_handle: Any,
    processor: OnboardCameraProcessor | None = None,
    *,
    latest_frame: dict[str, Any] | None = None,
    now: float | None = None,
    frames_advanced_during_check: int = 0,
    depth_fingerprints: tuple[str, ...] = (),
) -> CameraHealthStatus:
    stamp = time.time() if now is None else float(now)
    reasons: list[str] = []
    camera = getattr(scene_handle, "camera", None)
    frame = dict(latest_frame or {})
    if not frame and camera is not None:
        frame = _read_camera_frame(camera)

    backend_module = type(camera).__module__ if camera is not None else ""
    backend_class = type(camera).__name__ if camera is not None else ""
    is_isaaclab_camera = backend_class == "Camera" and backend_module.startswith("isaaclab.sensors.camera")
    if camera is None:
        reasons.append("camera object missing")
    elif not is_isaaclab_camera:
        reasons.append(f"camera backend is {backend_module}.{backend_class}, not Isaac Lab Camera")

    camera_path = str(getattr(scene_handle, "camera_prim_path", "") or "")
    parent_path = str(getattr(scene_handle, "camera_parent_prim", "") or "")
    stage = _resolve_stage(scene_handle)
    camera_prim = _get_prim(stage, camera_path)
    parent_prim = _get_prim(stage, parent_path)
    camera_prim_valid = _prim_is_valid(camera_prim)
    parent_prim_valid = _prim_is_valid(parent_prim)
    camera_prim_type = _prim_type_name(camera_prim)
    parent_has_rigid_body_api = _prim_has_rigid_body_api(parent_prim)
    if not camera_path:
        reasons.append("camera prim path missing")
    elif not camera_prim_valid:
        reasons.append(f"camera prim invalid: {camera_path}")
    elif camera_prim_type and "camera" not in camera_prim_type.lower():
        reasons.append(f"camera prim type is {camera_prim_type}, not Camera")
    if not parent_path:
        reasons.append("parent prim path missing")
    elif not parent_prim_valid:
        reasons.append(f"parent prim invalid: {parent_path}")
    elif not parent_has_rigid_body_api:
        reasons.append("parent prim does not have RigidBodyAPI")

    rgb = frame.get("rgb")
    depth = frame.get("depth")
    intrinsics = frame.get("intrinsics")
    pos_w = frame.get("pos_w")
    quat_w = frame.get("quat_w")
    rgb_shape, rgb_dtype = _shape_dtype(rgb)
    depth_shape, depth_dtype = _shape_dtype(depth)
    rgb_available = bool(rgb_shape)
    depth_available = bool(depth_shape)
    if not rgb_available:
        reasons.append("rgb output missing")
    if not depth_available:
        reasons.append("distance_to_image_plane output missing")
    depth_stats = _depth_stats(depth)
    if depth_available and depth_stats["depth_finite_ratio"] <= 0.0:
        reasons.append("depth has no finite pixels")
    if depth_available and depth_stats["depth_positive_ratio"] <= 0.0:
        reasons.append("depth has no positive pixels")
    if depth_available and (depth_stats["depth_std_m"] is None or float(depth_stats["depth_std_m"] or 0.0) <= 1.0e-9):
        reasons.append("depth is blank or constant")

    intrinsics_available = bool(_shape_dtype(intrinsics)[0])
    intrinsics_valid = _intrinsics_valid(intrinsics)
    if not intrinsics_available:
        reasons.append("camera intrinsics missing")
    elif not intrinsics_valid:
        reasons.append("camera intrinsics invalid")
    camera_pose_available = _pose_valid(pos_w, quat_w)
    if not camera_pose_available:
        reasons.append("camera world pose missing or invalid")

    last_frame_at = float(getattr(processor, "last_frame_at", 0.0) if processor is not None else 0.0)
    frame_revision = int(getattr(processor, "frame_revision", 0) if processor is not None else 0)
    frame_age = max(0.0, stamp - last_frame_at) if last_frame_at > 0.0 else 1.0e9
    frames_advanced = int(frames_advanced_during_check)
    stale_frame = frames_advanced <= 0 or frame_age > 2.0
    if frames_advanced <= 0:
        reasons.append("frame revision did not advance during check")
    if frame_age > 2.0:
        reasons.append(f"latest frame is too old: {frame_age:.3f}s")
    if len(depth_fingerprints) >= 2 and len(set(depth_fingerprints)) == 1:
        reasons.append("depth fingerprint unchanged during check; static scene may be okay, but verify if robot or scene moved")

    ok = (
        is_isaaclab_camera
        and camera_prim_valid
        and (not camera_prim_type or "camera" in camera_prim_type.lower())
        and parent_prim_valid
        and parent_has_rigid_body_api
        and rgb_available
        and depth_available
        and float(depth_stats["depth_finite_ratio"]) > 0.0
        and float(depth_stats["depth_positive_ratio"]) > 0.0
        and depth_stats["depth_std_m"] is not None
        and float(depth_stats["depth_std_m"] or 0.0) > 1.0e-9
        and intrinsics_available
        and intrinsics_valid
        and camera_pose_available
        and frames_advanced > 0
        and frame_age <= 2.0
    )
    return CameraHealthStatus(
        checked=True,
        ok=bool(ok),
        backend_module=backend_module,
        backend_class=backend_class,
        is_isaaclab_camera=bool(is_isaaclab_camera),
        camera_prim_path=camera_path,
        camera_prim_valid=bool(camera_prim_valid),
        camera_prim_type=camera_prim_type,
        parent_prim_path=parent_path,
        parent_prim_valid=bool(parent_prim_valid),
        parent_has_rigid_body_api=bool(parent_has_rigid_body_api),
        rgb_available=bool(rgb_available),
        depth_available=bool(depth_available),
        rgb_shape=tuple(int(v) for v in rgb_shape),
        depth_shape=tuple(int(v) for v in depth_shape),
        rgb_dtype=rgb_dtype,
        depth_dtype=depth_dtype,
        depth_finite_ratio=float(depth_stats["depth_finite_ratio"]),
        depth_positive_ratio=float(depth_stats["depth_positive_ratio"]),
        depth_min_m=depth_stats["depth_min_m"],
        depth_median_m=depth_stats["depth_median_m"],
        depth_max_m=depth_stats["depth_max_m"],
        depth_std_m=depth_stats["depth_std_m"],
        depth_fingerprint=str(depth_fingerprints[-1] if depth_fingerprints else depth_stats["depth_fingerprint"]),
        intrinsics_available=bool(intrinsics_available),
        intrinsics_valid=bool(intrinsics_valid),
        camera_pose_available=bool(camera_pose_available),
        frame_revision=frame_revision,
        frame_age_s=float(frame_age),
        frames_advanced_during_check=frames_advanced,
        stale_frame=bool(stale_frame),
        reasons=reasons,
    )


def build_height_provenance(
    *,
    scene_handle: Any,
    frame: dict[str, Any],
    frame_revision: int,
    roi_source: str,
    obstacle_x_m: float | None,
    detector_input_keys: set[str],
) -> dict[str, Any]:
    forbidden = {
        "expected_height_cm",
        "vision_generated_height_cm",
        "generated_height_cm",
        "scene_obstacle_height_m",
        "obstacle_height_m",
        "height_cm",
        "ui_height",
        "worker_height_cm",
        "set_height_cm",
        "obstacle_scale",
        "obstacle_size",
        "obstacle_bounding_box",
        "obstacle_center_z",
    }
    used_forbidden = sorted(str(key) for key in detector_input_keys if str(key) in forbidden)
    backend = ""
    camera = getattr(scene_handle, "camera", None)
    if camera is not None:
        backend = f"{type(camera).__module__}.{type(camera).__name__}"
    intrinsics_used = frame.get("intrinsics") is not None
    pose_used = frame.get("pos_w") is not None and frame.get("quat_w") is not None
    data = default_height_provenance(roi_source=roi_source, obstacle_x_m=obstacle_x_m, frame_revision=frame_revision)
    data.update(
        {
            "camera_backend": backend,
            "intrinsics_used": bool(intrinsics_used),
            "camera_world_pose_used": bool(pose_used),
            "ground_z_m": float(getattr(getattr(scene_handle, "config", None), "ground_z_m", 0.0) or 0.0),
            "obstacle_x_prior_used": obstacle_x_m is not None,
            "obstacle_x_prior_m": obstacle_x_m,
            "detector_input_audit_passed": not used_forbidden and bool(frame.get("depth") is not None and intrinsics_used and pose_used),
            "forbidden_input_reason": "; ".join(used_forbidden),
        }
    )
    return data


def build_height_measurement_evidence(
    *,
    scene_handle: Any,
    frame: dict[str, Any],
    frame_revision: int,
    detection: HeightDetection,
    filtered_detection: HeightDetection | None,
    roi_source: str,
    obstacle_x_m: float | None,
    height_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = dict(height_provenance or {})
    depth = frame.get("depth")
    stats = _depth_stats(depth)
    intrinsics = frame.get("intrinsics")
    chosen = filtered_detection if filtered_detection is not None and bool(filtered_detection.stable) else detection
    stable_detected = (
        filtered_detection.detected_height_cm
        if filtered_detection is not None and bool(filtered_detection.stable)
        else None
    )
    ground_z = chosen.ground_z_m
    if ground_z is None:
        ground_z = provenance.get("ground_z_m", getattr(getattr(scene_handle, "config", None), "ground_z_m", 0.0))
    evidence = default_height_measurement_evidence(
        roi_source=roi_source,
        obstacle_x_m=obstacle_x_m,
        frame_revision=frame_revision,
    )
    evidence.update(
        {
            "depth_timestamp": float(time.time()),
            "depth_fingerprint": str(stats.get("depth_fingerprint", "")),
            "depth_shape": list(_shape_dtype(depth)[0]),
            "depth_finite_ratio": float(stats.get("depth_finite_ratio", 0.0) or 0.0),
            "intrinsics_fingerprint": _intrinsics_fingerprint(intrinsics),
            "intrinsics_used": bool(provenance.get("intrinsics_used", intrinsics is not None)),
            "camera_position_w": _small_array(frame.get("pos_w")) or [],
            "camera_quaternion_w": _small_array(frame.get("quat_w")) or [],
            "camera_world_pose_used": bool(provenance.get("camera_world_pose_used", frame.get("pos_w") is not None and frame.get("quat_w") is not None)),
            "ground_reference_source": str(provenance.get("ground_reference_source", "configured_world_ground_z") or "configured_world_ground_z"),
            "ground_z_m": float(ground_z or 0.0),
            "top_z_m": chosen.top_z_m,
            "raw_height_cm": chosen.raw_height_cm,
            "candidate_height_cm": detection.detected_height_cm,
            "detected_height_cm": stable_detected,
            "top_point_count": int(chosen.top_point_count),
            "obstacle_point_count": int(chosen.obstacle_point_count or chosen.point_count),
            "ground_point_count": int(chosen.ground_point_count),
            "expected_height_used_by_detector": bool(provenance.get("expected_height_used_by_detector", False)),
            "generated_height_used_by_detector": bool(provenance.get("generated_height_used_by_detector", False)),
            "scene_obstacle_height_used_by_detector": bool(provenance.get("scene_obstacle_height_used_by_detector", False)),
            "usd_geometry_height_used_by_detector": bool(provenance.get("usd_geometry_height_used_by_detector", False)),
            "detector_input_audit_passed": bool(provenance.get("detector_input_audit_passed", False)),
            "forbidden_input_reason": str(provenance.get("forbidden_input_reason", "") or ""),
        }
    )
    return evidence


def inspect_camera_geometry(
    scene_handle: Any,
    frame: dict[str, Any],
    config: VisionHeightConfig,
    *,
    roi_source: str = "generated_scene_x_prior",
    obstacle_x_m: float | None = None,
) -> CameraGeometryDiagnostics:
    try:
        import numpy as np  # type: ignore

        depth = _depth_2d(frame.get("depth"))
        intr = _to_numpy(frame.get("intrinsics"))
        pos = _to_numpy(frame.get("pos_w"))
        quat = _to_numpy(frame.get("quat_w"))
        if depth is None:
            return CameraGeometryDiagnostics(checked=True, framing_state=FRAMING_OBSTACLE_NOT_VISIBLE, reasons=["depth unavailable"])
        if intr is None or pos is None or quat is None:
            return CameraGeometryDiagnostics(checked=True, framing_state=FRAMING_OBSTACLE_NOT_VISIBLE, reasons=["camera intrinsics or pose missing"])
        intr = np.asarray(intr, dtype=float)
        if intr.shape == (1, 3, 3):
            intr = intr[0]
        pos = np.asarray(pos, dtype=float).reshape(-1)[:3]
        quat = np.asarray(quat, dtype=float).reshape(-1)[:4]
        height, width = depth.shape
        fx = float(intr[0, 0])
        fy = float(intr[1, 1])
        cx = float(intr[0, 2])
        cy = float(intr[1, 2])
        center_ray_cam = np.array([(width * 0.5 - cx) / max(abs(fx), 1.0e-9), (height * 0.5 - cy) / max(abs(fy), 1.0e-9), 1.0], dtype=float)
        center_ray_cam /= max(1.0e-9, float(np.linalg.norm(center_ray_cam)))
        optical_forward = _rotate_direction((0.0, 0.0, 1.0), quat)
        optical_up = _rotate_direction((0.0, -1.0, 0.0), quat)
        center_ray = _rotate_direction(center_ray_cam, quat)
        target_x = float(obstacle_x_m) if obstacle_x_m is not None else float(pos[0] + getattr(config, "minimum_forward_distance_m", 0.35) + 0.75)
        target = np.array([target_x, 0.0, float(getattr(config, "ground_z_m", 0.0))], dtype=float)
        target_dir = target - pos
        target_dir /= max(1.0e-9, float(np.linalg.norm(target_dir)))
        dot = float(np.clip(np.dot(center_ray, target_dir), -1.0, 1.0))
        angular_error = math.degrees(math.acos(dot))
        ground_intersection = None
        ground_distance = None
        ground_z = float(getattr(config, "ground_z_m", 0.0))
        if abs(float(center_ray[2])) > 1.0e-9:
            t = (ground_z - float(pos[2])) / float(center_ray[2])
            if t > 0:
                hit = pos + center_ray * t
                ground_intersection = tuple(float(v) for v in hit)
                ground_distance = float(t)

        valid = np.isfinite(depth)
        if valid.any():
            rows, cols = np.indices(depth.shape, dtype=float)
            z = depth
            x = (cols - cx) * z / fx
            y = (rows - cy) * z / fy
            points_cam = np.stack((x, y, z), axis=-1)
            points_w = _transform_points(points_cam, pos, quat)
            roi = build_obstacle_mask(
                points_w,
                valid_mask=valid,
                config=config,
                obstacle_x_m=obstacle_x_m,
                camera_position_w=pos,
            )
            ground = roi & (np.abs(points_w[..., 2] - ground_z) <= max(0.01, float(config.no_obstacle_height_threshold_cm) / 100.0))
            obstacle = roi & (points_w[..., 2] > ground_z + max(0.0, float(config.no_obstacle_height_threshold_cm) / 100.0))
            if int(obstacle.sum()) > 0:
                z_values = points_w[..., 2][obstacle]
                cutoff = float(np.percentile(z_values, float(config.top_percentile)))
                top = obstacle & (points_w[..., 2] >= cutoff)
            else:
                top = np.zeros_like(roi, dtype=bool)
            roi_count = max(1, int(roi.sum()))
            obstacle_count = int(obstacle.sum())
            top_fraction = float(top.sum()) / max(1, obstacle_count)
            obstacle_fraction = float(obstacle_count) / roi_count
            ground_fraction = float(ground.sum()) / roi_count
            valid_fraction = float(valid.sum()) / float(max(1, valid.size))
        else:
            top_fraction = obstacle_fraction = ground_fraction = valid_fraction = 0.0

        reasons: list[str] = []
        framing = FRAMING_GOOD
        if float(center_ray[0]) <= 0.0:
            framing = FRAMING_POINTS_BACKWARD
            reasons.append("camera center ray points backward or sideways in world +X")
        elif obstacle_fraction <= 0.0:
            framing = FRAMING_OBSTACLE_NOT_VISIBLE
            reasons.append("no obstacle points in camera ROI")
        elif top_fraction > 0.85:
            framing = FRAMING_TOO_TOP_DOWN
            reasons.append("top surface covers more than 85% of obstacle ROI")
        elif ground_fraction < 0.05:
            framing = FRAMING_GROUND_NOT_VISIBLE
            reasons.append("ground point fraction below 5%")
        elif top_fraction <= 0.0 or obstacle_fraction * valid.size < int(config.minimum_obstacle_points):
            framing = FRAMING_TOP_PLANE_NOT_VISIBLE
            reasons.append("top-plane points below detector minimum")
        elif ground_intersection is None:
            framing = FRAMING_TOO_HORIZONTAL
            reasons.append("center ray does not intersect ground in front of camera")
        if not reasons:
            reasons.append("camera framing contains ground, obstacle, and top-plane points")

        return CameraGeometryDiagnostics(
            checked=True,
            camera_position_w=tuple(float(v) for v in pos),
            camera_quaternion_world=tuple(float(v) for v in quat),
            camera_quaternion_ros=tuple(float(v) for v in quat),
            optical_forward_w=tuple(float(v) for v in optical_forward),
            optical_up_w=tuple(float(v) for v in optical_up),
            center_ray_direction_w=tuple(float(v) for v in center_ray),
            target_point_w=tuple(float(v) for v in target),
            target_direction_w=tuple(float(v) for v in target_dir),
            angular_error_deg=float(angular_error),
            ground_intersection_w=ground_intersection,
            ground_intersection_distance_m=ground_distance,
            obstacle_search_x_m=obstacle_x_m,
            top_point_fraction=float(top_fraction),
            obstacle_point_fraction=float(obstacle_fraction),
            ground_point_fraction=float(ground_fraction),
            valid_depth_fraction=float(valid_fraction),
            framing_state=framing,
            reasons=reasons,
        )
    except Exception as exc:
        return CameraGeometryDiagnostics(checked=True, framing_state="ERROR", reasons=[str(exc)])


def save_rgbd_diagnostic(
    *,
    frame: dict[str, Any],
    detection: HeightDetection,
    config: VisionHeightConfig,
    root: Path,
    camera_health: dict[str, Any],
    height_validation: dict[str, Any],
    expected_height_cm: int | None,
    camera_prim_path: str,
    parent_prim_path: str,
    frame_revision: int,
    obstacle_x_m: float | None = None,
    camera_geometry: dict[str, Any] | None = None,
    height_provenance: dict[str, Any] | None = None,
    source_mode: str = "generated",
    roi_source: str = "generated_scene_x_prior",
    detection_revision: int = 0,
) -> dict[str, str]:
    """Save RGB-D debug artifacts without making image libraries mandatory."""

    import numpy as np  # type: ignore

    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    height_text = "none" if detection.detected_height_cm is None else f"{int(detection.detected_height_cm):02d}cm"
    prefix = root / f"{stamp}_frame{int(frame_revision):05d}_{height_text}_conf{float(detection.confidence):.2f}"
    rgb_u8 = _rgb_u8(frame.get("rgb"))
    depth = _depth_2d(frame.get("depth"))
    if depth is None:
        depth = np.zeros(rgb_u8.shape[:2], dtype=float)
    depth_u8 = _depth_u8(depth)
    depth_rgb = _depth_colormap(depth_u8)
    masks = _diagnostic_masks(frame, config, obstacle_x_m=obstacle_x_m)
    roi = masks.get("roi")
    obstacle = masks.get("obstacle")
    top = masks.get("top")
    roi_u8 = _mask_u8(roi, depth.shape)
    obstacle_u8 = _mask_u8(obstacle, depth.shape)
    top_u8 = _mask_u8(top, depth.shape)

    if rgb_u8.shape[:2] != depth_rgb.shape[:2]:
        rgb_u8 = np.zeros_like(depth_rgb)
    overlay = rgb_u8.copy()
    overlay = _blend_mask(overlay, roi_u8 > 0, (40, 180, 255), alpha=0.25)
    overlay = _blend_mask(overlay, obstacle_u8 > 0, (255, 190, 35), alpha=0.45)
    overlay = _blend_mask(overlay, top_u8 > 0, (255, 40, 90), alpha=0.75)

    mask_panel = np.stack((top_u8, obstacle_u8, roi_u8), axis=-1)
    composite = np.concatenate((overlay, depth_rgb, mask_panel), axis=1)
    validation_checked = bool(height_validation.get("checked", False))
    validation_passed = bool(height_validation.get("passed", False))
    state = "PASS" if validation_checked and validation_passed else ("FAIL" if validation_checked else "UNCHECKED")
    lines = [
        f"RAW {_fmt_optional(detection.raw_height_cm, 3)}CM DET {detection.detected_height_cm if detection.detected_height_cm is not None else '-'}CM",
        f"CONF {float(detection.confidence):.3f} STABLE {int(detection.stable_count)}/{int(getattr(config, 'stable_frames_required', 0))}",
    ]
    if expected_height_cm is not None:
        lines.append(f"EXPECTED {int(expected_height_cm)}CM {state}")
    else:
        lines.append(f"VALIDATION {state}")
    _draw_label_block(composite, lines)

    rgb_path = prefix.with_name(prefix.name + "_rgb.ppm")
    depth_path = prefix.with_name(prefix.name + "_depth.pgm")
    roi_path = prefix.with_name(prefix.name + "_roi_mask.pgm")
    obstacle_path = prefix.with_name(prefix.name + "_obstacle_mask.pgm")
    top_path = prefix.with_name(prefix.name + "_top_plane_mask.pgm")
    composite_path = prefix.with_name(prefix.name + "_diagnostic.ppm")
    json_path = prefix.with_suffix(".json")
    _write_ppm(rgb_path, rgb_u8)
    _write_pgm(depth_path, depth_u8)
    _write_pgm(roi_path, roi_u8)
    _write_pgm(obstacle_path, obstacle_u8)
    _write_pgm(top_path, top_u8)
    _write_ppm(composite_path, composite)
    sidecar = {
        "camera_backend_module": str(camera_health.get("backend_module", "")),
        "camera_backend_class": str(camera_health.get("backend_class", "")),
        "is_isaaclab_camera": bool(camera_health.get("is_isaaclab_camera", False)),
        "camera_prim_path": str(camera_prim_path),
        "parent_prim_path": str(parent_prim_path),
        "rgb_shape": list(_shape_dtype(frame.get("rgb"))[0]),
        "depth_shape": list(_shape_dtype(frame.get("depth"))[0]),
        "depth_finite_ratio": float(camera_health.get("depth_finite_ratio", 0.0) or 0.0),
        "intrinsics": _small_array(frame.get("intrinsics")),
        "camera_world_pose": {
            "pos_w": _small_array(frame.get("pos_w")),
            "quat_w": _small_array(frame.get("quat_w")),
        },
        "frame_revision": int(frame_revision),
        "detection_revision": int(detection_revision),
        "source_mode": str(source_mode),
        "roi_source": str(roi_source),
        "raw_height_cm": detection.raw_height_cm,
        "candidate_height_cm": detection.detected_height_cm,
        "stable_detected_height_cm": height_validation.get("detected_height_cm") if validation_checked else (detection.detected_height_cm if detection.stable else None),
        "confidence": float(detection.confidence),
        "stable_count": int(detection.stable_count),
        "valid_point_count": int(detection.point_count),
        "top_plane_mad_m": detection.top_plane_mad_m,
        "quantization_error_cm": detection.quantization_error_cm,
        "expected_height_cm": expected_height_cm,
        "validation_pass": bool(height_validation.get("passed", False)) if validation_checked else None,
        "failure_reasons": list(height_validation.get("reasons", []) if validation_checked else camera_health.get("reasons", [])),
        "camera_health": dict(camera_health),
        "height_validation": dict(height_validation),
        "camera_geometry": dict(camera_geometry or {}),
        "camera_coverage": dict(camera_geometry or {}),
        "height_provenance": dict(height_provenance or {}),
        "artifacts": {
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "roi_mask": str(roi_path),
            "obstacle_mask": str(obstacle_path),
            "top_plane_mask": str(top_path),
            "diagnostic": str(composite_path),
        },
    }
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "roi_mask_path": str(roi_path),
        "obstacle_mask_path": str(obstacle_path),
        "top_plane_mask_path": str(top_path),
        "diagnostic_image_path": str(composite_path),
        "json_path": str(json_path),
    }


def save_debug_frame_image(*, rgb: Any | None, depth: Any, detection: HeightDetection, root: Path) -> str:
    """Save a small PPM debug mosaic without adding image dependencies."""

    import numpy as np  # type: ignore

    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    height_text = "none" if detection.detected_height_cm is None else f"{int(detection.detected_height_cm):02d}cm"
    filename = f"{stamp}_rev{int(detection.detection_revision):04d}_{height_text}_conf{float(detection.confidence):.2f}.ppm"
    path = root / filename

    depth_arr = np.asarray(depth, dtype=float)
    if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[..., 0]
    finite = np.isfinite(depth_arr)
    if finite.any():
        low = float(np.nanpercentile(depth_arr[finite], 5.0))
        high = float(np.nanpercentile(depth_arr[finite], 95.0))
        scale = max(1.0e-9, high - low)
        depth_u8 = np.clip((depth_arr - low) / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        depth_u8 = np.zeros(depth_arr.shape, dtype=np.uint8)
    depth_rgb = np.stack((depth_u8, depth_u8, depth_u8), axis=-1)

    if rgb is None:
        rgb_u8 = np.zeros_like(depth_rgb)
    else:
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim == 2:
            rgb_arr = np.stack((rgb_arr, rgb_arr, rgb_arr), axis=-1)
        if rgb_arr.shape[-1] > 3:
            rgb_arr = rgb_arr[..., :3]
        if rgb_arr.dtype != np.uint8:
            max_value = 1.0 if float(np.nanmax(rgb_arr)) <= 1.0 else 255.0
            rgb_arr = np.clip(rgb_arr.astype(float) / max_value * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb_u8 = rgb_arr
        if rgb_u8.shape[:2] != depth_rgb.shape[:2]:
            rgb_u8 = np.zeros_like(depth_rgb)

    mosaic = np.concatenate((rgb_u8, depth_rgb), axis=1)
    with path.open("wb") as handle:
        handle.write(f"P6\n{mosaic.shape[1]} {mosaic.shape[0]}\n255\n".encode("ascii"))
        handle.write(mosaic.tobytes())
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(asdict(detection), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _to_numpy(value: Any) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        import numpy as np  # type: ignore

        return np.asarray(value)
    except Exception:
        return None


def _read_camera_frame(camera: Any) -> dict[str, Any]:
    data = camera.data
    output = data.output or {}
    rgb = _to_numpy(output.get("rgb"))
    if rgb is not None and getattr(rgb, "ndim", 0) >= 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    depth = _to_numpy(output.get("distance_to_image_plane"))
    if depth is not None:
        if getattr(depth, "ndim", 0) >= 3 and depth.shape[0] == 1:
            depth = depth[0]
        if getattr(depth, "ndim", 0) == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
    intrinsics = _to_numpy(getattr(data, "intrinsic_matrices", None))
    if intrinsics is not None and getattr(intrinsics, "ndim", 0) == 3:
        intrinsics = intrinsics[0]
    pos_w = _to_numpy(getattr(data, "pos_w", None))
    if pos_w is not None and getattr(pos_w, "ndim", 0) == 2:
        pos_w = pos_w[0]
    quat_w = _to_numpy(getattr(data, "quat_w_ros", None))
    if quat_w is None:
        quat_w = _to_numpy(getattr(data, "quat_w_world", None))
    if quat_w is None:
        quat_w = _to_numpy(getattr(data, "quat_w", None))
    if quat_w is not None and getattr(quat_w, "ndim", 0) == 2:
        quat_w = quat_w[0]
    return {
        "data": data,
        "output": output,
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "pos_w": pos_w,
        "quat_w": quat_w,
    }


def _require_array(frame: dict[str, Any], key: str, message: str) -> Any:
    value = frame.get(key)
    if value is None:
        raise RuntimeError(message)
    return value


def _resolve_stage(scene_handle: Any) -> Any | None:
    stage = getattr(scene_handle, "stage", None)
    if stage is not None:
        return stage
    sim = getattr(scene_handle, "sim", None)
    stage = getattr(sim, "stage", None)
    if stage is not None:
        return stage
    try:
        import isaaclab.sim as sim_utils  # type: ignore

        return sim_utils.get_current_stage()
    except Exception:
        return None


def _get_prim(stage: Any | None, path: str) -> Any | None:
    if stage is None or not path:
        return None
    try:
        return stage.GetPrimAtPath(path)
    except Exception:
        return None


def _prim_is_valid(prim: Any | None) -> bool:
    if prim is None:
        return False
    try:
        return bool(prim.IsValid())
    except Exception:
        return bool(getattr(prim, "valid", False))


def _prim_type_name(prim: Any | None) -> str:
    if prim is None:
        return ""
    try:
        return str(prim.GetTypeName())
    except Exception:
        return str(getattr(prim, "type_name", "") or "")


def _prim_has_rigid_body_api(prim: Any | None) -> bool:
    if not _prim_is_valid(prim):
        return False
    try:
        from pxr import UsdPhysics  # type: ignore

        return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
    except Exception:
        pass
    try:
        return bool(prim.HasAPI("RigidBodyAPI"))
    except Exception:
        return bool(getattr(prim, "has_rigid_body_api", False))


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], str]:
    arr = _to_numpy(value)
    if arr is None:
        return (), ""
    try:
        shape = tuple(int(v) for v in arr.shape)
        dtype = str(arr.dtype)
        return shape, dtype
    except Exception:
        return (), ""


def _rotate_direction(vector: Any, quat_wxyz: Any) -> Any:
    import numpy as np  # type: ignore

    q = np.asarray(quat_wxyz, dtype=float).reshape(-1)[:4]
    v = np.asarray(vector, dtype=float).reshape(-1)[:3]
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-9:
        return v
    w, x, y, z = q / norm
    q_vec = np.array([x, y, z], dtype=float)
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (w * uv + uuv)


def _depth_stats(depth: Any) -> dict[str, Any]:
    import numpy as np  # type: ignore

    arr = _depth_2d(depth)
    if arr is None or arr.size <= 0:
        return {
            "depth_finite_ratio": 0.0,
            "depth_positive_ratio": 0.0,
            "depth_min_m": None,
            "depth_median_m": None,
            "depth_max_m": None,
            "depth_std_m": None,
            "depth_fingerprint": "",
        }
    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    finite_values = arr[finite]
    if finite_values.size <= 0:
        return {
            "depth_finite_ratio": 0.0,
            "depth_positive_ratio": 0.0,
            "depth_min_m": None,
            "depth_median_m": None,
            "depth_max_m": None,
            "depth_std_m": None,
            "depth_fingerprint": _depth_fingerprint(arr),
        }
    return {
        "depth_finite_ratio": float(finite.sum()) / float(arr.size),
        "depth_positive_ratio": float(positive.sum()) / float(arr.size),
        "depth_min_m": float(np.nanmin(finite_values)),
        "depth_median_m": float(np.nanmedian(finite_values)),
        "depth_max_m": float(np.nanmax(finite_values)),
        "depth_std_m": float(np.nanstd(finite_values)),
        "depth_fingerprint": _depth_fingerprint(arr),
    }


def _depth_fingerprint(depth: Any) -> str:
    import hashlib
    import numpy as np  # type: ignore

    arr = _depth_2d(depth)
    if arr is None or arr.size <= 0:
        return ""
    sample = arr[:: max(1, arr.shape[0] // 24), :: max(1, arr.shape[1] // 32)]
    sample = np.nan_to_num(sample.astype(np.float32), nan=-1.0, posinf=-2.0, neginf=-3.0)
    return hashlib.sha1(sample.tobytes()).hexdigest()[:16]


def _intrinsics_fingerprint(intrinsics: Any) -> str:
    import hashlib
    import numpy as np  # type: ignore

    arr = _to_numpy(intrinsics)
    if arr is None:
        return ""
    try:
        matrix = np.asarray(arr, dtype=np.float64)
        if matrix.shape == (1, 3, 3):
            matrix = matrix[0]
        if matrix.shape != (3, 3):
            return ""
        matrix = np.nan_to_num(matrix, nan=-1.0, posinf=-2.0, neginf=-3.0)
        return hashlib.sha1(matrix.tobytes()).hexdigest()[:16]
    except Exception:
        return ""


def _intrinsics_valid(value: Any) -> bool:
    import numpy as np  # type: ignore

    arr = _to_numpy(value)
    if arr is None:
        return False
    try:
        intr = np.asarray(arr, dtype=float)
        if intr.shape == (1, 3, 3):
            intr = intr[0]
        return intr.shape == (3, 3) and np.isfinite(intr).all() and abs(float(intr[0, 0])) > 1.0e-9 and abs(float(intr[1, 1])) > 1.0e-9
    except Exception:
        return False


def _pose_valid(pos_w: Any, quat_w: Any) -> bool:
    import numpy as np  # type: ignore

    try:
        pos = np.asarray(pos_w, dtype=float).reshape(-1)
        quat = np.asarray(quat_w, dtype=float).reshape(-1)
        return pos.size >= 3 and quat.size >= 4 and np.isfinite(pos[:3]).all() and np.isfinite(quat[:4]).all() and float(np.linalg.norm(quat[:4])) > 1.0e-9
    except Exception:
        return False


def _camera_relative_matrix(scene_handle: Any) -> Any | None:
    import numpy as np  # type: ignore

    stage = _resolve_stage(scene_handle)
    camera_path = str(getattr(scene_handle, "camera_prim_path", "") or "")
    parent_path = str(getattr(scene_handle, "camera_parent_prim", "") or "")
    if stage is None or not camera_path or not parent_path:
        return None
    camera_world = _prim_world_matrix(stage, camera_path)
    parent_world = _prim_world_matrix(stage, parent_path)
    if camera_world is None or parent_world is None:
        return None
    try:
        return np.linalg.inv(parent_world) @ camera_world
    except Exception:
        return None


def _prim_world_matrix(stage: Any, path: str) -> Any | None:
    import numpy as np  # type: ignore

    prim = _get_prim(stage, path)
    if not _prim_is_valid(prim):
        return None
    try:
        from pxr import Usd, UsdGeom  # type: ignore

        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows = [[float(matrix[i][j]) for j in range(4)] for i in range(4)]
        return np.asarray(rows, dtype=float)
    except Exception:
        matrix = getattr(prim, "world_matrix", None)
        if matrix is None:
            return None
        arr = np.asarray(matrix, dtype=float)
        return arr.reshape(4, 4) if arr.size == 16 else None


def _relative_transform_error(initial: Any, current: Any) -> tuple[float, float]:
    import numpy as np  # type: ignore

    a = np.asarray(initial, dtype=float).reshape(4, 4)
    b = np.asarray(current, dtype=float).reshape(4, 4)
    pos_error = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation_delta = a[:3, :3].T @ b[:3, :3]
    trace = float(np.trace(rotation_delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    angle_deg = math.degrees(math.acos(cos_angle))
    return pos_error, float(angle_deg)


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(round(number))
    except (TypeError, ValueError):
        return None


def _depth_2d(depth: Any) -> Any | None:
    import numpy as np  # type: ignore

    arr = _to_numpy(depth)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr


def _rgb_u8(rgb: Any) -> Any:
    import numpy as np  # type: ignore

    arr = _to_numpy(rgb)
    if arr is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr), axis=-1)
    if arr.ndim != 3:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        max_value = 1.0
        try:
            finite = arr[np.isfinite(arr)]
            if finite.size and float(np.nanmax(finite)) > 1.0:
                max_value = 255.0
        except Exception:
            max_value = 255.0
        arr = np.clip(arr.astype(float) / max_value * 255.0, 0.0, 255.0).astype(np.uint8)
    return arr


def _depth_u8(depth: Any) -> Any:
    import numpy as np  # type: ignore

    arr = _depth_2d(depth)
    if arr is None:
        return np.zeros((1, 1), dtype=np.uint8)
    finite = np.isfinite(arr)
    if finite.any():
        low = float(np.nanpercentile(arr[finite], 5.0))
        high = float(np.nanpercentile(arr[finite], 95.0))
        scale = max(1.0e-9, high - low)
        return np.clip((arr - low) / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.zeros(arr.shape, dtype=np.uint8)


def _depth_colormap(depth_u8: Any) -> Any:
    import numpy as np  # type: ignore

    d = np.asarray(depth_u8, dtype=np.uint8)
    return np.stack((d, np.clip(255 - d, 0, 255).astype(np.uint8), np.full_like(d, 120)), axis=-1)


def _diagnostic_masks(frame: dict[str, Any], config: VisionHeightConfig, *, obstacle_x_m: float | None = None) -> dict[str, Any]:
    import numpy as np  # type: ignore

    depth = _depth_2d(frame.get("depth"))
    intrinsics = _to_numpy(frame.get("intrinsics"))
    pos_w = _to_numpy(frame.get("pos_w"))
    quat_w = _to_numpy(frame.get("quat_w"))
    if depth is None:
        return {}
    height, width = depth.shape
    fraction = max(0.05, min(1.0, float(config.roi_horizontal_fraction)))
    roi_width = max(1, int(round(width * fraction)))
    start = max(0, (width - roi_width) // 2)
    end = min(width, start + roi_width)
    roi = np.zeros((height, width), dtype=bool)
    roi[:, start:end] = True
    try:
        if intrinsics is None or pos_w is None or quat_w is None:
            return {"roi": roi}
        intr = np.asarray(intrinsics, dtype=float)
        if intr.shape == (1, 3, 3):
            intr = intr[0]
        if intr.shape != (3, 3):
            return {"roi": roi}
        fx = float(intr[0, 0])
        fy = float(intr[1, 1])
        cx = float(intr[0, 2])
        cy = float(intr[1, 2])
        rows, cols = np.indices((height, width), dtype=float)
        z = depth
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        points_cam = np.stack((x, y, z), axis=-1)
        points_w = _transform_points(points_cam, pos_w, quat_w)
        finite = np.isfinite(depth) & (depth > 0.0)
        candidate = build_obstacle_mask(
            points_w,
            valid_mask=finite,
            config=config,
            obstacle_x_m=obstacle_x_m,
            camera_position_w=pos_w,
        )
        obstacle = candidate & (points_w[..., 2] > float(config.ground_z_m) + max(0.0, float(config.no_obstacle_height_threshold_cm) / 100.0))
        top = np.zeros_like(obstacle)
        if obstacle.any():
            z_values = points_w[..., 2][obstacle]
            cutoff = float(np.percentile(z_values, float(config.top_percentile)))
            top = obstacle & (points_w[..., 2] >= cutoff)
        return {"roi": roi, "obstacle": obstacle, "top": top}
    except Exception:
        return {"roi": roi}


def _mask_u8(mask: Any, shape: tuple[int, int]) -> Any:
    import numpy as np  # type: ignore

    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return (arr.astype(np.uint8) * 255).astype(np.uint8)


def _blend_mask(image: Any, mask: Any, color: tuple[int, int, int], *, alpha: float) -> Any:
    import numpy as np  # type: ignore

    out = np.asarray(image, dtype=np.uint8).copy()
    m = np.asarray(mask, dtype=bool)
    if m.shape != out.shape[:2] or not m.any():
        return out
    color_arr = np.asarray(color, dtype=float).reshape(1, 3)
    out[m] = np.clip(out[m].astype(float) * (1.0 - float(alpha)) + color_arr * float(alpha), 0.0, 255.0).astype(np.uint8)
    return out


def _write_ppm(path: Path, image: Any) -> None:
    import numpy as np  # type: ignore

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr), axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    with path.open("wb") as handle:
        handle.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        handle.write(arr.tobytes())


def _write_pgm(path: Path, image: Any) -> None:
    import numpy as np  # type: ignore

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    with path.open("wb") as handle:
        handle.write(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        handle.write(arr.tobytes())


def _small_array(value: Any) -> Any:
    arr = _to_numpy(value)
    if arr is None:
        return None
    try:
        return arr.astype(float).tolist()
    except Exception:
        return None


def _fmt_optional(value: Any, digits: int) -> str:
    try:
        number = float(value)
        if math.isfinite(number):
            return f"{number:.{int(digits)}f}"
    except (TypeError, ValueError):
        pass
    return "-"


def _draw_label_block(image: Any, lines: list[str]) -> None:
    import numpy as np  # type: ignore

    arr = np.asarray(image, dtype=np.uint8)
    y = 4
    for line in lines:
        _draw_text(arr, 4, y, str(line).upper(), color=(255, 255, 255), background=(0, 0, 0))
        y += 9


_FONT_3X5: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "010", "000"),
    "/": ("001", "001", "010", "100", "100"),
    "%": ("101", "001", "010", "100", "101"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "111", "100", "111"),
    "F": ("111", "100", "111", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "001"),
    "R": ("111", "101", "111", "110", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


def _draw_text(image: Any, x: int, y: int, text: str, *, color: tuple[int, int, int], background: tuple[int, int, int]) -> None:
    import numpy as np  # type: ignore

    arr = np.asarray(image, dtype=np.uint8)
    scale = 1
    cursor = int(x)
    y = int(y)
    max_width = arr.shape[1]
    max_height = arr.shape[0]
    box_width = min(max_width - cursor, max(0, len(text) * 4 + 4))
    if box_width > 0 and y + 7 < max_height:
        arr[y : min(max_height, y + 7), cursor : cursor + box_width] = np.asarray(background, dtype=np.uint8)
    for char in text:
        glyph = _FONT_3X5.get(char, _FONT_3X5[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit != "1":
                    continue
                yy = y + 1 + row * scale
                xx = cursor + 1 + col * scale
                if 0 <= yy < max_height and 0 <= xx < max_width:
                    arr[yy, xx] = color
        cursor += 4
        if cursor >= max_width - 4:
            break
