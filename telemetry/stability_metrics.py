"""Geometric and friction-aware support stability metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SupportPlane:
    origin_w: np.ndarray
    normal_w: np.ndarray
    e1_w: np.ndarray
    e2_w: np.ndarray
    source: str = "gravity_aligned"
    fitting_residual_m: float | None = None


@dataclass
class StabilityResult:
    state: str
    margin_m: float
    support_vertices_2d: list[list[float]] = field(default_factory=list)
    support_vertices_w: list[list[float]] = field(default_factory=list)
    com_projection_2d: list[float] = field(default_factory=list)
    com_projection_w: list[float] = field(default_factory=list)
    area_m2: float = 0.0
    perimeter_m: float = 0.0
    active_contact_count: int = 0
    degenerate_support: bool = False
    warnings: list[str] = field(default_factory=list)


def gravity_aligned_support_plane(
    contact_points_w: Any,
    normal_forces_n: Any | None = None,
    *,
    gravity_w: Any = (0.0, 0.0, -9.81),
    fallback_origin_w: Any = (0.0, 0.0, 0.0),
) -> SupportPlane:
    gravity = np.asarray(gravity_w, dtype=float)
    norm_g = float(np.linalg.norm(gravity))
    normal = np.asarray([0.0, 0.0, 1.0], dtype=float) if norm_g <= 1.0e-12 else -gravity / norm_g
    points = _as_points3(contact_points_w)
    if len(points):
        weights = np.asarray(normal_forces_n, dtype=float).reshape(-1) if normal_forces_n is not None else np.ones(len(points))
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        if weights.size != len(points) or float(np.sum(weights)) <= 1.0e-12:
            origin = np.nanmean(points, axis=0)
        else:
            origin = np.sum(points * weights[: len(points), None], axis=0) / float(np.sum(weights[: len(points)]))
    else:
        origin = np.asarray(fallback_origin_w, dtype=float)
    if not np.isfinite(origin).all():
        origin = np.asarray(fallback_origin_w, dtype=float)
    e1 = np.cross(np.asarray([0.0, 1.0, 0.0], dtype=float), normal)
    if float(np.linalg.norm(e1)) <= 1.0e-9:
        e1 = np.cross(np.asarray([1.0, 0.0, 0.0], dtype=float), normal)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-12)
    e2 = np.cross(normal, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-12)
    return SupportPlane(origin_w=origin, normal_w=normal, e1_w=e1, e2_w=e2)


def project_points_to_plane(points_w: Any, plane: SupportPlane) -> np.ndarray:
    points = _as_points3(points_w)
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    rel = points - plane.origin_w[None, :]
    return np.column_stack((rel @ plane.e1_w, rel @ plane.e2_w))


def point_from_plane_coords(point_2d: Any, plane: SupportPlane) -> np.ndarray:
    xy = np.asarray(point_2d, dtype=float).reshape(-1)
    if xy.size < 2:
        return np.full(3, np.nan)
    return plane.origin_w + xy[0] * plane.e1_w + xy[1] * plane.e2_w


def compute_geometric_stability(
    contact_points_w: Any,
    com_w: Any,
    *,
    normal_forces_n: Any | None = None,
    contact_force_threshold_n: float = 2.0,
    safe_margin_m: float = 0.05,
    warning_margin_m: float = 0.02,
    gravity_w: Any = (0.0, 0.0, -9.81),
    support_plane: SupportPlane | None = None,
) -> StabilityResult:
    points_w = _as_points3(contact_points_w)
    if normal_forces_n is not None:
        raw_forces = np.asarray(normal_forces_n, dtype=float).reshape(-1)
        forces = np.full(len(points_w), np.nan, dtype=float)
        count = min(len(points_w), raw_forces.size)
        if count:
            forces[:count] = raw_forces[:count]
        keep = np.isfinite(forces) & (forces >= float(contact_force_threshold_n))
        points_w = points_w[keep]
        forces = forces[keep]
    else:
        forces = None
    warnings: list[str] = []
    if len(points_w) == 0:
        return StabilityResult(state="airborne", margin_m=float("nan"), active_contact_count=0, warnings=["no active contacts"])
    finite = np.isfinite(points_w).all(axis=1)
    if np.any(~finite):
        warnings.append("ignored non-finite contact points")
        points_w = points_w[finite]
        if forces is not None:
            forces = forces[finite]
    if len(points_w) == 0:
        return StabilityResult(state="airborne", margin_m=float("nan"), active_contact_count=0, warnings=warnings + ["all contact points invalid"])
    plane = support_plane or gravity_aligned_support_plane(points_w, forces, gravity_w=gravity_w)
    points_2d = merge_duplicate_points(project_points_to_plane(points_w, plane))
    hull = convex_hull_2d(points_2d)
    com_arr = np.asarray(com_w, dtype=float).reshape(-1)[:3]
    com_2d = project_points_to_plane(com_arr.reshape(1, 3), plane)[0] if com_arr.size == 3 else np.full(2, np.nan)
    margin, degenerate = signed_distance_to_support_region(com_2d, hull)
    state = classify_margin(margin, safe_margin_m=safe_margin_m, warning_margin_m=warning_margin_m)
    vertices_w = [point_from_plane_coords(vertex, plane).tolist() for vertex in hull]
    return StabilityResult(
        state=state,
        margin_m=float(margin),
        support_vertices_2d=hull.tolist(),
        support_vertices_w=vertices_w,
        com_projection_2d=com_2d.tolist(),
        com_projection_w=point_from_plane_coords(com_2d, plane).tolist(),
        area_m2=polygon_area(hull),
        perimeter_m=polygon_perimeter(hull),
        active_contact_count=int(len(points_w)),
        degenerate_support=bool(degenerate),
        warnings=warnings,
    )


def merge_duplicate_points(points: Any, *, tolerance: float = 1.0e-6) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = arr.reshape(-1, 2)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return np.empty((0, 2), dtype=float)
    buckets: dict[tuple[int, int], np.ndarray] = {}
    scale = 1.0 / max(float(tolerance), 1.0e-12)
    for point in arr:
        key = (int(round(point[0] * scale)), int(round(point[1] * scale)))
        buckets.setdefault(key, point)
    return np.asarray(list(buckets.values()), dtype=float)


def convex_hull_2d(points: Any) -> np.ndarray:
    pts = merge_duplicate_points(points)
    if len(pts) <= 1:
        return pts
    pts = np.asarray(sorted((float(x), float(y)) for x, y in pts), dtype=float)

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1.0e-12:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1.0e-12:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) == 0:
        return pts[:1]
    return hull


def signed_distance_to_support_region(point: Any, vertices: Any) -> tuple[float, bool]:
    p = np.asarray(point, dtype=float).reshape(-1)[:2]
    verts = np.asarray(vertices, dtype=float)
    if p.size != 2 or not np.isfinite(p).all():
        return float("nan"), True
    if verts.size == 0:
        return float("nan"), True
    verts = merge_duplicate_points(verts)
    if len(verts) == 1:
        dist = float(np.linalg.norm(p - verts[0]))
        return (0.0 if dist <= 1.0e-9 else -dist), True
    if len(verts) == 2 or abs(polygon_area(verts)) <= 1.0e-12:
        dist = min_distance_to_segments(p, _line_segments_for_degenerate(verts))
        return (0.0 if dist <= 1.0e-9 else -dist), True
    dist = min_distance_to_segments(p, [(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))])
    if point_on_polygon_boundary(p, verts):
        return 0.0, False
    return (dist if point_in_convex_polygon(p, verts) else -dist), False


def classify_margin(margin_m: float, *, safe_margin_m: float, warning_margin_m: float) -> str:
    if not math.isfinite(float(margin_m)):
        return "airborne"
    if margin_m < 0.0:
        return "outside"
    if margin_m < float(warning_margin_m):
        return "critical"
    if margin_m < float(safe_margin_m):
        return "warning"
    return "safe"


def point_in_convex_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    sign = 0
    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        edge = b - a
        rel = point - a
        cross = float(edge[0] * rel[1] - edge[1] * rel[0])
        if abs(cross) <= 1.0e-10:
            continue
        current = 1 if cross > 0.0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return True


def point_on_polygon_boundary(point: np.ndarray, vertices: np.ndarray) -> bool:
    for i in range(len(vertices)):
        if distance_point_to_segment(point, vertices[i], vertices[(i + 1) % len(vertices)]) <= 1.0e-9:
            return True
    return False


def min_distance_to_segments(point: np.ndarray, segments: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not segments:
        return float("nan")
    return float(min(distance_point_to_segment(point, a, b) for a, b in segments))


def distance_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-24:
        return float(np.linalg.norm(point - a))
    t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def polygon_area(vertices: Any) -> float:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or len(verts) < 3:
        return 0.0
    x = verts[:, 0]
    y = verts[:, 1]
    return abs(float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def polygon_perimeter(vertices: Any) -> float:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or len(verts) == 0:
        return 0.0
    if len(verts) == 1:
        return 0.0
    if len(verts) == 2:
        return float(np.linalg.norm(verts[1] - verts[0]))
    return float(sum(np.linalg.norm(verts[(i + 1) % len(verts)] - verts[i]) for i in range(len(verts))))


def compute_equilibrium_region(
    contact_points_w: Any,
    contact_normals_w: Any,
    friction_coefficients: Any,
    *,
    mass_kg: float,
    com_w: Any,
    support_plane: SupportPlane | None = None,
    gravity_w: Any = (0.0, 0.0, -9.81),
    friction_pyramid_sides: int = 8,
    direction_samples: int = 48,
    solver_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    started = time.perf_counter()
    warnings: list[str] = []
    points = _as_points3(contact_points_w)
    if len(points) < 1:
        return _equilibrium_failure("no_contacts", started, warnings)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 1:
        return _equilibrium_failure("invalid_contacts", started, warnings + ["contact points are not finite"])
    gravity = np.asarray(gravity_w, dtype=float)
    if not np.isfinite(gravity).all() or float(np.linalg.norm(gravity)) <= 1.0e-12:
        return _equilibrium_failure("invalid_gravity", started, warnings)
    if not math.isfinite(float(mass_kg)) or float(mass_kg) <= 0.0:
        return _equilibrium_failure("invalid_mass", started, warnings)
    plane = support_plane or gravity_aligned_support_plane(points, gravity_w=gravity)
    normals = _as_points3(contact_normals_w)
    if len(normals) != len(points):
        normals = np.repeat(plane.normal_w.reshape(1, 3), len(points), axis=0)
        warnings.append("contact normals unavailable; using support plane normal")
    normals = np.asarray([_normalize_or_default(n, plane.normal_w) for n in normals], dtype=float)
    for index, normal in enumerate(normals):
        if float(np.dot(normal, plane.normal_w)) < 0.0:
            normals[index] = -normal
            warnings.append("flipped contact normal toward support-plane normal")
    mu = np.asarray(friction_coefficients, dtype=float).reshape(-1)
    if mu.size != len(points):
        mu = np.full(len(points), 0.0 if mu.size == 0 else float(mu[0]), dtype=float)
        warnings.append("friction coefficient count did not match contacts")
    mu = np.where(np.isfinite(mu) & (mu >= 0.0), mu, 0.0)
    try:
        from scipy.optimize import linprog  # type: ignore
    except Exception as exc:
        return _equilibrium_failure("scipy_unavailable", started, warnings + [str(exc)])

    n_contacts = len(points)
    n_vars = n_contacts * 3 + 2
    mg = float(mass_kg) * gravity
    com = np.asarray(com_w, dtype=float).reshape(-1)[:3]
    if com.size != 3 or not np.isfinite(com).all():
        return _equilibrium_failure("invalid_com", started, warnings)
    height = float(np.dot(com - plane.origin_w, plane.normal_w))
    c0 = plane.origin_w + height * plane.normal_w
    a_eq = np.zeros((6, n_vars), dtype=float)
    b_eq = np.zeros(6, dtype=float)
    for i in range(n_contacts):
        a_eq[0:3, 3 * i : 3 * i + 3] = np.eye(3)
        a_eq[3:6, 3 * i : 3 * i + 3] = _skew(points[i])
    a_eq[3:6, -2] = np.cross(plane.e1_w, mg)
    a_eq[3:6, -1] = np.cross(plane.e2_w, mg)
    b_eq[0:3] = -mg
    b_eq[3:6] = -np.cross(c0, mg)
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for i in range(n_contacts):
        normal = normals[i]
        row = np.zeros(n_vars, dtype=float)
        row[3 * i : 3 * i + 3] = -normal
        a_ub.append(row)
        b_ub.append(0.0)
        t1, t2 = _contact_tangent_basis(normal)
        for side in range(max(4, int(friction_pyramid_sides))):
            angle = 2.0 * math.pi * float(side) / float(max(4, int(friction_pyramid_sides)))
            tangent = math.cos(angle) * t1 + math.sin(angle) * t2
            row = np.zeros(n_vars, dtype=float)
            row[3 * i : 3 * i + 3] = tangent - float(mu[i]) * normal
            a_ub.append(row)
            b_ub.append(0.0)
    vertices_2d: list[list[float]] = []
    statuses: list[str] = []
    for sample in range(max(8, int(direction_samples))):
        angle = 2.0 * math.pi * float(sample) / float(max(8, int(direction_samples)))
        direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        objective = np.zeros(n_vars, dtype=float)
        objective[-2:] = -direction
        result = linprog(
            objective,
            A_ub=np.asarray(a_ub, dtype=float),
            b_ub=np.asarray(b_ub, dtype=float),
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(None, None)] * n_vars,
            method="highs",
            options={"primal_feasibility_tolerance": float(solver_tolerance), "dual_feasibility_tolerance": float(solver_tolerance)},
        )
        statuses.append(str(result.status))
        if result.success and result.x is not None:
            vertices_2d.append([float(result.x[-2]), float(result.x[-1])])
        elif result.status == 3:
            return _equilibrium_failure("solver_unbounded", started, warnings + ["LP became unbounded"])
    if len(vertices_2d) < 3:
        return _equilibrium_failure("solver_infeasible", started, warnings + [f"solver_statuses={statuses}"])
    hull = convex_hull_2d(np.asarray(vertices_2d, dtype=float))
    current_2d = project_points_to_plane(com.reshape(1, 3), plane)[0]
    margin, degenerate = signed_distance_to_support_region(current_2d, hull)
    feasible = bool(math.isfinite(float(margin)) and margin >= -1.0e-6)
    return {
        "status": "success",
        "solver_status": "success",
        "solver_statuses": statuses,
        "solver_time_ms": (time.perf_counter() - started) * 1000.0,
        "vertices_2d": hull.tolist(),
        "vertices_w": [point_from_plane_coords(vertex, plane).tolist() for vertex in hull],
        "area_m2": polygon_area(hull),
        "equilibrium_stability_margin_m": float(margin),
        "current_state_feasible": feasible,
        "degenerate_support": bool(degenerate),
        "friction_pyramid_sides": int(friction_pyramid_sides),
        "direction_samples": int(direction_samples),
        "warnings": warnings,
    }


def _equilibrium_failure(status: str, started: float, warnings: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "solver_status": status,
        "solver_time_ms": (time.perf_counter() - started) * 1000.0,
        "vertices_2d": [],
        "vertices_w": [],
        "area_m2": float("nan"),
        "equilibrium_stability_margin_m": float("nan"),
        "current_state_feasible": False,
        "degenerate_support": True,
        "warnings": list(warnings),
    }


def _line_segments_for_degenerate(vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(vertices) < 2:
        return []
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    axis = np.argmax(np.ptp(vertices, axis=0))
    ordered = vertices[np.argsort(vertices[:, axis])]
    return [(ordered[0], ordered[-1])]


def _as_points3(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 3:
        out = np.full((arr.shape[0], 3), np.nan, dtype=float)
        out[:, : arr.shape[-1]] = arr
        return out
    return arr[:, :3]


def _normalize_or_default(vec: Any, default: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)[:3]
    norm = float(np.linalg.norm(arr)) if arr.size == 3 and np.isfinite(arr).all() else 0.0
    return arr / norm if norm > 1.0e-12 else np.asarray(default, dtype=float)


def _contact_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=float)
    t1 = np.cross(normal, reference)
    t1 = t1 / max(float(np.linalg.norm(t1)), 1.0e-12)
    t2 = np.cross(normal, t1)
    t2 = t2 / max(float(np.linalg.norm(t2)), 1.0e-12)
    return t1, t2


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
