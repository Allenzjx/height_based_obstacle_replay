"""Pure Python RGB-D obstacle height estimation.

This module intentionally has no Isaac imports.  The worker passes camera
depth, intrinsics, and camera world pose in; the functions below only perform
geometry, quantization, and temporal filtering.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Any, Iterable


DEFAULT_SUPPORTED_HEIGHTS_CM = tuple(range(0, 41, 5))


@dataclass
class VisionHeightConfig:
    supported_heights_cm: tuple[int, ...] = DEFAULT_SUPPORTED_HEIGHTS_CM
    quantization_tolerance_cm: float = 2.0
    minimum_confidence: float = 0.75
    temporal_window_size: int = 7
    stable_frames_required: int = 5
    minimum_obstacle_points: int = 80
    maximum_top_plane_mad_m: float = 0.012
    minimum_forward_distance_m: float = 0.35
    maximum_forward_distance_m: float = 2.60
    roi_horizontal_fraction: float = 0.72
    no_obstacle_height_threshold_cm: float = 2.0
    obstacle_x_window_m: float = 0.75
    lateral_window_m: float = 1.20
    ground_z_m: float = 0.0
    top_percentile: float = 75.0


@dataclass
class HeightDetection:
    valid: bool
    raw_height_cm: float | None
    detected_height_cm: int | None
    confidence: float
    point_count: int
    top_plane_mad_m: float | None
    quantization_error_cm: float | None
    reason: str
    timestamp: float
    stable_count: int = 0
    stable: bool = False
    detection_revision: int = 0
    ground_z_m: float | None = None
    top_z_m: float | None = None
    top_point_count: int = 0
    obstacle_point_count: int = 0
    ground_point_count: int = 0
    obstacle_area_fraction: float = 0.0

    @classmethod
    def invalid(cls, reason: str, *, timestamp: float | None = None) -> "HeightDetection":
        return cls(
            valid=False,
            raw_height_cm=None,
            detected_height_cm=None,
            confidence=0.0,
            point_count=0,
            top_plane_mad_m=None,
            quantization_error_cm=None,
            reason=str(reason),
            timestamp=time.time() if timestamp is None else float(timestamp),
        )


def quantize_supported_height(
    raw_height_cm: float | int | None,
    supported_heights: Iterable[int] = DEFAULT_SUPPORTED_HEIGHTS_CM,
    tolerance_cm: float = 2.0,
) -> tuple[int | None, float | None]:
    """Return the supported bucket and absolute error, or ``(None, error)``."""

    if raw_height_cm is None:
        return None, None
    try:
        raw = float(raw_height_cm)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(raw):
        return None, None
    supported = [int(value) for value in supported_heights]
    if not supported:
        return None, None
    best = min(supported, key=lambda value: abs(float(value) - raw))
    error = abs(float(best) - raw)
    if error <= float(tolerance_cm) + 1.0e-9:
        return best, error
    return None, error


def estimate_height_from_depth(
    depth_image: Any,
    intrinsic_matrix: Any,
    camera_position_w: Iterable[float],
    camera_quat_wxyz: Iterable[float],
    *,
    config: VisionHeightConfig | None = None,
    obstacle_x_m: float | None = None,
    semantic_mask: Any | None = None,
    timestamp: float | None = None,
    near_clip_m: float = 0.01,
    far_clip_m: float = 10.0,
) -> HeightDetection:
    """Estimate obstacle height from a depth image and camera calibration."""

    cfg = config or VisionHeightConfig()
    stamp = time.time() if timestamp is None else float(timestamp)
    try:
        np = _np()
        depth = np.asarray(depth_image, dtype=float)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            return HeightDetection.invalid("depth image must be HxW", timestamp=stamp)
        height, width = depth.shape
        intr = np.asarray(intrinsic_matrix, dtype=float)
        if intr.shape == (1, 3, 3):
            intr = intr[0]
        if intr.shape != (3, 3):
            return HeightDetection.invalid("intrinsic matrix must be 3x3", timestamp=stamp)
        fx = float(intr[0, 0])
        fy = float(intr[1, 1])
        cx = float(intr[0, 2])
        cy = float(intr[1, 2])
        if min(abs(fx), abs(fy)) <= 1.0e-9:
            return HeightDetection.invalid("invalid camera focal length", timestamp=stamp)

        valid_depth = (
            np.isfinite(depth)
            & (depth >= float(near_clip_m))
            & (depth <= float(far_clip_m))
        )
        if int(valid_depth.sum()) == 0:
            return HeightDetection.invalid("no valid depth pixels", timestamp=stamp)

        rows, cols = np.indices((height, width), dtype=float)
        z = depth
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        points_cam = np.stack((x, y, z), axis=-1)
        points_w = _transform_points(points_cam, camera_position_w, camera_quat_wxyz)
        return estimate_height_from_pointcloud(
            points_w,
            valid_mask=valid_depth,
            config=cfg,
            obstacle_x_m=obstacle_x_m,
            camera_position_w=camera_position_w,
            semantic_mask=semantic_mask,
            timestamp=stamp,
        )
    except Exception as exc:
        return HeightDetection.invalid(f"depth estimation failed: {exc}", timestamp=stamp)


def estimate_height_from_pointcloud(
    points_w: Any,
    *,
    valid_mask: Any | None = None,
    config: VisionHeightConfig | None = None,
    obstacle_x_m: float | None = None,
    camera_position_w: Iterable[float] | None = None,
    semantic_mask: Any | None = None,
    timestamp: float | None = None,
) -> HeightDetection:
    """Estimate obstacle height from world-frame points."""

    cfg = config or VisionHeightConfig()
    stamp = time.time() if timestamp is None else float(timestamp)
    try:
        np = _np()
        points = np.asarray(points_w, dtype=float)
        if points.ndim != 3 or points.shape[-1] != 3:
            return HeightDetection.invalid("pointcloud must be HxWx3", timestamp=stamp)
        finite = np.isfinite(points).all(axis=-1)
        if valid_mask is not None:
            finite &= np.asarray(valid_mask, dtype=bool)
        candidate = build_obstacle_mask(
            points,
            valid_mask=finite,
            config=cfg,
            obstacle_x_m=obstacle_x_m,
            camera_position_w=camera_position_w,
            semantic_mask=semantic_mask,
        )
        if int(candidate.sum()) == 0:
            return _zero_height_detection(cfg, "no points in obstacle search region", stamp)

        ground_z = float(cfg.ground_z_m)
        height_threshold_m = max(0.0, float(cfg.no_obstacle_height_threshold_cm) / 100.0)
        ground_band = candidate & (np.abs(points[..., 2] - ground_z) <= max(0.01, height_threshold_m))
        ground_point_count = int(ground_band.sum())
        non_ground = candidate & (points[..., 2] > ground_z + height_threshold_m)
        obstacle_points = points[non_ground]
        point_count = int(obstacle_points.shape[0])
        if point_count == 0:
            return _zero_height_detection(cfg, "no obstacle points above ground threshold", stamp)
        if point_count < int(cfg.minimum_obstacle_points):
            return HeightDetection(
                valid=False,
                raw_height_cm=None,
                detected_height_cm=None,
                confidence=0.0,
                point_count=point_count,
                top_plane_mad_m=None,
                quantization_error_cm=None,
                reason=f"not enough obstacle points: {point_count}",
                timestamp=stamp,
                ground_z_m=ground_z,
                obstacle_point_count=point_count,
                ground_point_count=ground_point_count,
            )

        z_values = obstacle_points[:, 2]
        cutoff = float(np.percentile(z_values, float(cfg.top_percentile)))
        top_band = z_values[z_values >= cutoff]
        if top_band.size == 0:
            top_band = z_values
        top_z = float(np.median(top_band))
        top_point_count = int(top_band.size)
        top_mad = float(np.median(np.abs(top_band - top_z)))
        raw_height_cm = (top_z - ground_z) * 100.0
        detected, quant_error = quantize_supported_height(
            raw_height_cm,
            cfg.supported_heights_cm,
            cfg.quantization_tolerance_cm,
        )
        if detected is None:
            return HeightDetection(
                valid=False,
                raw_height_cm=raw_height_cm,
                detected_height_cm=None,
                confidence=0.0,
                point_count=point_count,
                top_plane_mad_m=top_mad,
                quantization_error_cm=quant_error,
                reason="height outside quantization tolerance",
                timestamp=stamp,
                ground_z_m=ground_z,
                top_z_m=top_z,
                top_point_count=top_point_count,
                obstacle_point_count=point_count,
                ground_point_count=ground_point_count,
                obstacle_area_fraction=float(non_ground.sum()) / max(1.0, float(candidate.sum())),
            )
        area_fraction = float(non_ground.sum()) / max(1.0, float(candidate.sum()))
        confidence = calculate_detection_confidence(
            point_count=point_count,
            top_plane_mad_m=top_mad,
            quantization_error_cm=float(quant_error or 0.0),
            obstacle_area_fraction=area_fraction,
            config=cfg,
            detected_height_cm=detected,
        )
        valid = confidence >= max(0.0, float(cfg.minimum_confidence) * 0.50)
        return HeightDetection(
            valid=valid,
            raw_height_cm=raw_height_cm,
            detected_height_cm=detected,
            confidence=confidence,
            point_count=point_count,
            top_plane_mad_m=top_mad,
            quantization_error_cm=quant_error,
            reason="ok" if valid else "low single-frame confidence",
            timestamp=stamp,
            ground_z_m=ground_z,
            top_z_m=top_z,
            top_point_count=top_point_count,
            obstacle_point_count=point_count,
            ground_point_count=ground_point_count,
            obstacle_area_fraction=area_fraction,
        )
    except Exception as exc:
        return HeightDetection.invalid(f"pointcloud estimation failed: {exc}", timestamp=stamp)


def build_obstacle_mask(
    points_w: Any,
    *,
    valid_mask: Any,
    config: VisionHeightConfig | None = None,
    obstacle_x_m: float | None = None,
    camera_position_w: Iterable[float] | None = None,
    semantic_mask: Any | None = None,
) -> Any:
    """Build a non-height-coded obstacle search mask."""

    cfg = config or VisionHeightConfig()
    np = _np()
    points = np.asarray(points_w, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool).copy()
    height, width = mask.shape
    fraction = max(0.05, min(1.0, float(cfg.roi_horizontal_fraction)))
    roi_width = max(1, int(round(width * fraction)))
    start = max(0, (width - roi_width) // 2)
    end = min(width, start + roi_width)
    roi = np.zeros_like(mask, dtype=bool)
    roi[:, start:end] = True
    mask &= roi

    if semantic_mask is not None:
        sem = np.asarray(semantic_mask)
        if sem.shape[:2] == mask.shape:
            if sem.ndim == 3:
                sem = sem.any(axis=-1)
            mask &= sem.astype(bool)

    x = points[..., 0]
    y = points[..., 1]
    if obstacle_x_m is not None:
        half = max(0.05, float(cfg.obstacle_x_window_m) * 0.5)
        mask &= (x >= float(obstacle_x_m) - half) & (x <= float(obstacle_x_m) + half)
    elif camera_position_w is not None:
        camera_x = float(list(camera_position_w)[0])
        mask &= (x >= camera_x + float(cfg.minimum_forward_distance_m)) & (
            x <= camera_x + float(cfg.maximum_forward_distance_m)
        )
    else:
        mask &= (x >= float(cfg.minimum_forward_distance_m)) & (x <= float(cfg.maximum_forward_distance_m))
    mask &= np.abs(y) <= max(0.05, float(cfg.lateral_window_m))
    return mask


def calculate_detection_confidence(
    *,
    point_count: int,
    top_plane_mad_m: float | None,
    quantization_error_cm: float | None,
    obstacle_area_fraction: float,
    config: VisionHeightConfig | None = None,
    detected_height_cm: int | None = None,
) -> float:
    cfg = config or VisionHeightConfig()
    if detected_height_cm == 0 and point_count <= 0:
        return 0.85
    min_points = max(1, int(cfg.minimum_obstacle_points))
    points_score = min(1.0, float(point_count) / float(min_points * 3))
    mad = 0.0 if top_plane_mad_m is None else max(0.0, float(top_plane_mad_m))
    flat_score = max(0.0, 1.0 - mad / max(1.0e-9, float(cfg.maximum_top_plane_mad_m)))
    q_error = 0.0 if quantization_error_cm is None else max(0.0, float(quantization_error_cm))
    quant_score = max(0.0, 1.0 - q_error / max(1.0e-9, float(cfg.quantization_tolerance_cm)))
    area_score = min(1.0, max(0.0, float(obstacle_area_fraction)) / 0.02)
    score = 0.30 * points_score + 0.30 * flat_score + 0.25 * quant_score + 0.15 * area_score
    return max(0.0, min(1.0, score))


class TemporalHeightFilter:
    """Sliding-window stability filter for quantized height detections."""

    def __init__(self, config: VisionHeightConfig | None = None):
        self.config = config or VisionHeightConfig()
        self.frames: deque[HeightDetection] = deque(maxlen=max(1, int(self.config.temporal_window_size)))
        self.detection_revision = 0
        self.last_emitted_height_cm: int | None = None
        self.stable_count = 0
        self.last_stable_detection: HeightDetection | None = None

    def reset(self, *, clear_revision: bool = False) -> None:
        self.frames.clear()
        self.stable_count = 0
        self.last_emitted_height_cm = None
        self.last_stable_detection = None
        if clear_revision:
            self.detection_revision = 0

    def update(self, detection: HeightDetection) -> HeightDetection:
        self.frames.append(detection)
        valid_frames = [
            frame
            for frame in self.frames
            if frame.valid
            and frame.detected_height_cm is not None
            and frame.confidence >= float(self.config.minimum_confidence)
        ]
        if not valid_frames:
            self.stable_count = 0
            return replace(
                detection,
                valid=False,
                stable=False,
                stable_count=0,
                detection_revision=self.detection_revision,
                reason=f"unstable: {detection.reason}",
            )
        counts = Counter(int(frame.detected_height_cm) for frame in valid_frames)
        height_cm, count = counts.most_common(1)[0]
        consecutive_count = 0
        for frame in reversed(self.frames):
            if (
                frame.valid
                and frame.detected_height_cm is not None
                and int(frame.detected_height_cm) == int(height_cm)
                and frame.confidence >= float(self.config.minimum_confidence)
            ):
                consecutive_count += 1
                continue
            break
        self.stable_count = int(consecutive_count)
        same_height_frames = [
            frame
            for frame in valid_frames
            if frame.detected_height_cm is not None and int(frame.detected_height_cm) == int(height_cm)
        ]
        confidences = sorted(float(frame.confidence) for frame in same_height_frames)
        median_confidence = confidences[len(confidences) // 2]
        required = max(1, int(self.config.stable_frames_required))
        if consecutive_count < required or median_confidence < float(self.config.minimum_confidence):
            return replace(
                detection,
                valid=False,
                detected_height_cm=None,
                stable=False,
                stable_count=consecutive_count,
                detection_revision=self.detection_revision,
                reason=f"unstable: {consecutive_count}/{required} consecutive frames",
            )

        raw_values = sorted(
            float(frame.raw_height_cm)
            for frame in same_height_frames
            if frame.raw_height_cm is not None and math.isfinite(float(frame.raw_height_cm))
        )
        raw_height = raw_values[len(raw_values) // 2] if raw_values else float(height_cm)
        point_counts = sorted(int(frame.point_count) for frame in same_height_frames)
        point_count = point_counts[len(point_counts) // 2] if point_counts else 0
        mad_values = sorted(
            float(frame.top_plane_mad_m)
            for frame in same_height_frames
            if frame.top_plane_mad_m is not None and math.isfinite(float(frame.top_plane_mad_m))
        )
        mad = mad_values[len(mad_values) // 2] if mad_values else None
        quant_error = abs(float(height_cm) - float(raw_height))
        top_z_values = sorted(
            float(frame.top_z_m)
            for frame in same_height_frames
            if frame.top_z_m is not None and math.isfinite(float(frame.top_z_m))
        )
        top_z = top_z_values[len(top_z_values) // 2] if top_z_values else None
        ground_z_values = sorted(
            float(frame.ground_z_m)
            for frame in same_height_frames
            if frame.ground_z_m is not None and math.isfinite(float(frame.ground_z_m))
        )
        ground_z = ground_z_values[len(ground_z_values) // 2] if ground_z_values else None
        top_counts = sorted(int(frame.top_point_count) for frame in same_height_frames)
        obstacle_counts = sorted(int(frame.obstacle_point_count or frame.point_count) for frame in same_height_frames)
        ground_counts = sorted(int(frame.ground_point_count) for frame in same_height_frames)
        area_values = sorted(float(frame.obstacle_area_fraction) for frame in same_height_frames)
        if self.last_emitted_height_cm != height_cm:
            self.detection_revision += 1
            self.last_emitted_height_cm = height_cm
        stable = HeightDetection(
            valid=True,
            raw_height_cm=raw_height,
            detected_height_cm=height_cm,
            confidence=median_confidence,
            point_count=point_count,
            top_plane_mad_m=mad,
            quantization_error_cm=quant_error,
            reason="stable",
            timestamp=float(detection.timestamp),
            stable_count=consecutive_count,
            stable=True,
            detection_revision=self.detection_revision,
            ground_z_m=ground_z,
            top_z_m=top_z,
            top_point_count=top_counts[len(top_counts) // 2] if top_counts else 0,
            obstacle_point_count=obstacle_counts[len(obstacle_counts) // 2] if obstacle_counts else point_count,
            ground_point_count=ground_counts[len(ground_counts) // 2] if ground_counts else 0,
            obstacle_area_fraction=area_values[len(area_values) // 2] if area_values else 0.0,
        )
        self.last_stable_detection = stable
        return stable


def _zero_height_detection(cfg: VisionHeightConfig, reason: str, timestamp: float) -> HeightDetection:
    detected, quant_error = quantize_supported_height(0.0, cfg.supported_heights_cm, cfg.quantization_tolerance_cm)
    return HeightDetection(
        valid=detected == 0,
        raw_height_cm=0.0,
        detected_height_cm=0 if detected == 0 else None,
        confidence=calculate_detection_confidence(
            point_count=0,
            top_plane_mad_m=0.0,
            quantization_error_cm=0.0,
            obstacle_area_fraction=0.0,
            config=cfg,
            detected_height_cm=0,
        ),
        point_count=0,
        top_plane_mad_m=0.0,
        quantization_error_cm=quant_error,
        reason=reason,
        timestamp=timestamp,
        ground_z_m=float(cfg.ground_z_m),
        top_z_m=float(cfg.ground_z_m),
        top_point_count=0,
        obstacle_point_count=0,
        ground_point_count=0,
        obstacle_area_fraction=0.0,
    )


def _np() -> Any:
    import numpy as np  # type: ignore

    return np


def _transform_points(points_cam: Any, position_w: Iterable[float], quat_wxyz: Iterable[float]) -> Any:
    np = _np()
    points = np.asarray(points_cam, dtype=float)
    position = np.asarray(list(position_w), dtype=float).reshape(1, 1, 3)
    quat = np.asarray(list(quat_wxyz), dtype=float)
    rotated = _quat_rotate(points, quat)
    return rotated + position


def _quat_rotate(points: Any, quat_wxyz: Any) -> Any:
    np = _np()
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = q / norm
    w, x, y, z = q
    q_vec = np.asarray([x, y, z], dtype=float)
    pts = np.asarray(points, dtype=float)
    uv = np.cross(q_vec, pts)
    uuv = np.cross(q_vec, uv)
    return pts + 2.0 * (w * uv + uuv)
