"""Pure camera health and height validation helpers.

This module intentionally avoids Isaac imports at import time.  The worker
passes small status dictionaries in; validation compares those results after
normal RGB-D detection has already produced them.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CameraHealthStatus:
    checked: bool = False
    ok: bool = False
    backend_module: str = ""
    backend_class: str = ""
    is_isaaclab_camera: bool = False
    camera_prim_path: str = ""
    camera_prim_valid: bool = False
    camera_prim_type: str = ""
    parent_prim_path: str = ""
    parent_prim_valid: bool = False
    parent_has_rigid_body_api: bool = False
    rgb_available: bool = False
    depth_available: bool = False
    rgb_shape: tuple[int, ...] = ()
    depth_shape: tuple[int, ...] = ()
    rgb_dtype: str = ""
    depth_dtype: str = ""
    depth_finite_ratio: float = 0.0
    depth_positive_ratio: float = 0.0
    depth_min_m: float | None = None
    depth_median_m: float | None = None
    depth_max_m: float | None = None
    depth_std_m: float | None = None
    depth_fingerprint: str = ""
    intrinsics_available: bool = False
    intrinsics_valid: bool = False
    camera_pose_available: bool = False
    frame_revision: int = 0
    frame_age_s: float = 0.0
    frames_advanced_during_check: int = 0
    stale_frame: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeightValidationResult:
    checked: bool = False
    passed: bool = False
    expected_height_cm: int | None = None
    raw_height_cm: float | None = None
    detected_height_cm: int | None = None
    candidate_height_cm: int | None = None
    absolute_error_cm: float | None = None
    absolute_error_mm: float | None = None
    relative_error_percent: float | None = None
    confidence: float = 0.0
    stable: bool = False
    stable_count: int = 0
    stable_required: int = 0
    valid_point_count: int = 0
    top_plane_mad_m: float | None = None
    quantization_error_cm: float | None = None
    frame_revision: int = 0
    detection_revision: int = 0
    reasons: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_camera_health_status(*, checked: bool = False, reason: str = "") -> dict[str, Any]:
    status = CameraHealthStatus(checked=bool(checked))
    if reason:
        status.reasons.append(str(reason))
    return status.to_dict()


def default_height_validation_result(*, checked: bool = False, reason: str = "") -> dict[str, Any]:
    result = HeightValidationResult(checked=bool(checked), timestamp=time.time())
    if reason:
        result.reasons.append(str(reason))
    return result.to_dict()


def compare_detection_to_expected(
    vision_status: dict[str, Any],
    expected_height_cm: int | None,
    *,
    raw_height_tolerance_cm: float = 1.0,
    minimum_confidence: float = 0.75,
    minimum_obstacle_points: int = 80,
    quantization_tolerance_cm: float = 2.0,
) -> HeightValidationResult:
    """Compare completed RGB-D detection status to an explicit expected height.

    The expected height is used only here, after detector output already exists.
    """

    reasons: list[str] = []
    expected = _optional_int(expected_height_cm)
    raw = _optional_float(vision_status.get("raw_height_cm"))
    detected = _optional_int(vision_status.get("detected_height_cm"))
    candidate = _optional_int(vision_status.get("candidate_height_cm"))
    confidence = _finite_float(vision_status.get("confidence"), default=0.0)
    stable = bool(vision_status.get("stable", False))
    stable_count = _finite_int(vision_status.get("stable_count"), default=0)
    stable_required = _finite_int(vision_status.get("stable_required"), default=0)
    valid_points = _finite_int(vision_status.get("valid_point_count"), default=0)
    quant_error = _optional_float(vision_status.get("quantization_error_cm"))
    frame_revision = _finite_int(vision_status.get("frame_revision"), default=0)
    detection_revision = _finite_int(vision_status.get("detection_revision"), default=0)
    camera_health = vision_status.get("camera_health")

    if expected is None:
        reasons.append("expected height missing")
    if not bool(vision_status.get("camera_ready", False)):
        reasons.append("camera not ready")
    if isinstance(camera_health, dict):
        if not bool(camera_health.get("ok", False)):
            reasons.append("camera health failed")
        if _finite_int(camera_health.get("frames_advanced_during_check"), default=0) <= 0:
            reasons.append("camera frame did not advance during health check")
        if bool(camera_health.get("stale_frame", False)):
            reasons.append("camera frame is stale")
    else:
        reasons.append("camera health not checked")

    if candidate is None:
        reasons.append("candidate height missing")
    elif expected is not None and candidate != expected:
        reasons.append(f"candidate height {candidate}cm != expected {expected}cm")
    if detected is None:
        reasons.append("stable detected height missing")
    elif expected is not None and detected != expected:
        reasons.append(f"detected height {detected}cm != expected {expected}cm")
    if not stable:
        reasons.append("detection is not stable")
    if stable_count < max(1, stable_required):
        reasons.append(f"stable frames {stable_count}/{max(1, stable_required)} below required")
    if confidence < float(minimum_confidence):
        reasons.append(f"confidence {confidence:.3f} below {float(minimum_confidence):.3f}")
    if valid_points < int(minimum_obstacle_points):
        reasons.append(f"valid point count {valid_points} below {int(minimum_obstacle_points)}")
    if quant_error is None:
        reasons.append("quantization error missing")
    elif quant_error > float(quantization_tolerance_cm) + 1.0e-9:
        reasons.append(f"quantization error {quant_error:.3f}cm exceeds {float(quantization_tolerance_cm):.3f}cm")

    absolute_error_cm: float | None = None
    absolute_error_mm: float | None = None
    relative_error_percent: float | None = None
    if raw is None:
        reasons.append("raw height missing")
    elif expected is not None:
        absolute_error_cm = abs(raw - float(expected))
        absolute_error_mm = absolute_error_cm * 10.0
        if expected == 0:
            relative_error_percent = 0.0 if absolute_error_cm <= 1.0e-9 else None
        else:
            relative_error_percent = absolute_error_cm / abs(float(expected)) * 100.0
        if absolute_error_cm > float(raw_height_tolerance_cm) + 1.0e-9:
            reasons.append(f"raw height error {absolute_error_cm:.3f}cm exceeds {float(raw_height_tolerance_cm):.3f}cm")

    return HeightValidationResult(
        checked=True,
        passed=not reasons,
        expected_height_cm=expected,
        raw_height_cm=raw,
        detected_height_cm=detected,
        candidate_height_cm=candidate,
        absolute_error_cm=absolute_error_cm,
        absolute_error_mm=absolute_error_mm,
        relative_error_percent=relative_error_percent,
        confidence=confidence,
        stable=stable,
        stable_count=stable_count,
        stable_required=stable_required,
        valid_point_count=valid_points,
        top_plane_mad_m=_optional_float(vision_status.get("top_plane_mad_m")),
        quantization_error_cm=quant_error,
        frame_revision=frame_revision,
        detection_revision=detection_revision,
        reasons=reasons,
        timestamp=time.time(),
    )


def format_height_validation_summary(result: dict[str, Any] | HeightValidationResult) -> str:
    values = result.to_dict() if isinstance(result, HeightValidationResult) else dict(result or {})
    if not bool(values.get("checked", False)):
        return "Height Validation: NOT CHECKED"
    state = "PASS" if bool(values.get("passed", False)) else "FAIL"
    expected = _optional_float(values.get("expected_height_cm"))
    raw = _optional_float(values.get("raw_height_cm"))
    detected = _optional_int(values.get("detected_height_cm"))
    error_cm = _optional_float(values.get("absolute_error_cm"))
    error_mm = _optional_float(values.get("absolute_error_mm"))
    relative = _optional_float(values.get("relative_error_percent"))
    lines = [state]
    lines.append(f"Expected: {_fmt_cm(expected)}")
    lines.append(f"Measured: {_fmt_cm(raw)}")
    lines.append(f"Absolute error: {_fmt_number(error_cm, 3)}cm / {_fmt_number(error_mm, 2)}mm")
    lines.append(f"Relative error: {_fmt_number(relative, 2)}%")
    lines.append(f"Detected bucket: {'-' if detected is None else str(detected) + 'cm'}")
    lines.append(f"Confidence: {_finite_float(values.get('confidence'), default=0.0):.3f}")
    lines.append(f"Stable frames: {_finite_int(values.get('stable_count'), default=0)}/{_finite_int(values.get('stable_required'), default=0)}")
    lines.append(f"Valid points: {_finite_int(values.get('valid_point_count'), default=0)}")
    reasons = values.get("reasons", [])
    if reasons:
        lines.append("Reasons: " + "; ".join(str(reason) for reason in reasons))
    return "\n".join(lines)


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_float(value: Any, *, default: float) -> float:
    number = _optional_float(value)
    return float(default) if number is None else number


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(number))


def _finite_int(value: Any, *, default: int) -> int:
    number = _optional_int(value)
    return int(default) if number is None else int(number)


def _fmt_cm(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.3f}cm"


def _fmt_number(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{float(value):.{int(digits)}f}"
