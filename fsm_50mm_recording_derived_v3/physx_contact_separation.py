"""Strict decoding of PhysX signed contact separations.

The existing wheel/ground AABB diagnostic is useful geometry evidence, but it
is not a PhysX penetration measurement.  Isaac's rigid-contact view exposes
the signed separation for each buffered contact through ``get_contact_data``.
This module decodes that buffer without importing Isaac Sim so its identity,
shape, capacity, and finite-value rules can be tested in ordinary Python.

Negative separation means penetration.  A structurally valid zero-contact
pair is therefore valid evidence with zero penetration; a missing or malformed
view is UNKNOWN and must never be converted to zero.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


PHYSX_SEPARATION_SOURCE = (
    "omni.physics.tensors.RigidContactView.get_contact_data.distances"
)


class PhysxContactSeparationError(RuntimeError):
    """Base class for fail-closed signed-separation evidence errors."""


class PhysxContactSeparationLayoutError(PhysxContactSeparationError):
    """Raised when a contact buffer cannot be labelled unambiguously."""


class PhysxContactSeparationEvidenceError(PhysxContactSeparationError):
    """Raised when referenced signed-separation evidence is non-finite."""


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _exact_nonnegative_integers(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    raw = _to_numpy(value)
    if raw.ndim != 2 or tuple(int(part) for part in raw.shape) != expected_shape:
        raise PhysxContactSeparationLayoutError(
            f"{label} shape={raw.shape} does not equal {expected_shape}"
        )
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PhysxContactSeparationLayoutError(
            f"{label} is not numeric: {exc}"
        ) from exc
    if not bool(np.isfinite(numeric).all()):
        raise PhysxContactSeparationLayoutError(
            f"{label} contains non-finite values"
        )
    rounded = np.rint(numeric)
    if not bool(np.equal(numeric, rounded).all()):
        raise PhysxContactSeparationLayoutError(
            f"{label} contains non-integer values"
        )
    if bool((rounded < 0).any()):
        raise PhysxContactSeparationLayoutError(
            f"{label} contains negative values"
        )
    return rounded.astype(np.int64)


def _exact_absolute_paths(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        raise PhysxContactSeparationLayoutError(
            f"{label} must be an ordered path sequence"
        )
    try:
        raw_paths = list(value)
    except TypeError as exc:
        raise PhysxContactSeparationLayoutError(
            f"{label} must be an ordered path sequence"
        ) from exc
    paths = tuple(str(path).rstrip("/") for path in raw_paths)
    if not paths or any(not path.startswith("/") for path in paths):
        raise PhysxContactSeparationLayoutError(
            f"{label} must contain absolute prim paths"
        )
    if len(paths) != len(set(paths)):
        raise PhysxContactSeparationLayoutError(
            f"{label} contains duplicate prim paths"
        )
    return paths


def _exact_absolute_path_matrix(
    value: Any,
    *,
    label: str,
    expected_rows: int,
    expected_columns: int,
) -> tuple[tuple[str, ...], ...]:
    """Parse the native RigidContactView ``filter_paths`` matrix strictly.

    Isaac Sim 5.1 exposes ``sensor_paths`` as a one-dimensional sequence, but
    ``filter_paths`` as ``[sensor][filter]``.  Requiring the full matrix keeps
    every configured surface bound to the exact live sensor/filter identity;
    flattening would silently discard per-sensor ordering evidence.
    """

    if isinstance(value, (str, bytes)) or value is None:
        raise PhysxContactSeparationLayoutError(
            f"{label} must be an ordered sensor/filter path matrix"
        )
    try:
        raw_rows = list(value)
    except TypeError as exc:
        raise PhysxContactSeparationLayoutError(
            f"{label} must be an ordered sensor/filter path matrix"
        ) from exc
    if len(raw_rows) != int(expected_rows):
        raise PhysxContactSeparationLayoutError(
            f"{label} row count={len(raw_rows)} does not equal "
            f"sensor count={int(expected_rows)}"
        )
    parsed_rows: list[tuple[str, ...]] = []
    for row_index, raw_row in enumerate(raw_rows):
        row_label = f"{label}[{row_index}]"
        if isinstance(raw_row, (str, bytes)) or raw_row is None:
            raise PhysxContactSeparationLayoutError(
                f"{row_label} must be an ordered filter path sequence"
            )
        try:
            row_values = list(raw_row)
        except TypeError as exc:
            raise PhysxContactSeparationLayoutError(
                f"{row_label} must be an ordered filter path sequence"
            ) from exc
        if len(row_values) != int(expected_columns):
            raise PhysxContactSeparationLayoutError(
                f"{row_label} column count={len(row_values)} does not equal "
                f"filter_count={int(expected_columns)}"
            )
        paths = tuple(str(path).rstrip("/") for path in row_values)
        if any(not path.startswith("/") for path in paths):
            raise PhysxContactSeparationLayoutError(
                f"{row_label} must contain absolute prim paths"
            )
        if len(paths) != len(set(paths)):
            raise PhysxContactSeparationLayoutError(
                f"{row_label} contains duplicate prim paths"
            )
        parsed_rows.append(paths)
    return tuple(parsed_rows)


def _pair_id(
    *,
    env_id: int,
    body_prim_path: str,
    other_prim_path: str,
) -> str:
    return f"env={int(env_id)}|body={body_prim_path}|other={other_prim_path}"


def _identity_fields(
    *,
    env_id: int,
    body_class: str,
    body_name: str,
    body_prim_path: str,
    leg: str | None,
    filter_index: int,
    surface: str,
    other_prim_path: str,
) -> dict[str, Any]:
    return {
        "pair_id": _pair_id(
            env_id=env_id,
            body_prim_path=body_prim_path,
            other_prim_path=other_prim_path,
        ),
        "env_id": int(env_id),
        "body_class": str(body_class),
        "leg": None if leg is None else str(leg),
        "body_name": str(body_name),
        "body_prim_path": str(body_prim_path),
        "surface": str(surface),
        "other_prim_path": str(other_prim_path),
        "filter_index": int(filter_index),
    }


def unknown_contact_pair_rows(
    *,
    env_count: int,
    body_class: str,
    body_name: str,
    body_prim_path: str,
    filters: Sequence[tuple[str, str]],
    error: str,
    leg: str | None = None,
    capacity: int | None = None,
) -> list[dict[str, Any]]:
    """Return explicit UNKNOWN rows for every expected pair identity."""

    count = max(1, int(env_count))
    rows: list[dict[str, Any]] = []
    for env_id in range(count):
        for filter_index, (surface, other_prim_path) in enumerate(filters):
            rows.append(
                {
                    **_identity_fields(
                        env_id=env_id,
                        body_class=body_class,
                        body_name=body_name,
                        body_prim_path=body_prim_path,
                        leg=leg,
                        filter_index=filter_index,
                        surface=surface,
                        other_prim_path=other_prim_path,
                    ),
                    "contact_count": None,
                    "signed_separations_m": [],
                    "minimum_signed_separation_m": None,
                    "maximum_penetration_m": None,
                    "valid": False,
                    "status": "UNKNOWN",
                    "capacity": None if capacity is None else int(capacity),
                    "capacity_exhausted": None,
                    "source": PHYSX_SEPARATION_SOURCE,
                    "error": str(error or "PhysX separation evidence unavailable"),
                }
            )
    return rows


def decode_contact_pair_separations(
    *,
    dt_s: float,
    distances: Any,
    counts: Any,
    starts: Any,
    env_count: int,
    body_class: str,
    body_name: str,
    body_prim_path: str,
    filters: Sequence[tuple[str, str]],
    configured_filter_paths: Sequence[str],
    expected_sensor_paths: Sequence[str],
    view_sensor_paths: Any,
    view_filter_paths: Any,
    view_filter_count: Any,
    view_max_contact_data_count: Any,
    leg: str | None = None,
) -> list[dict[str, Any]]:
    """Decode one exact-body rigid-contact view into labelled pair rows.

    ``counts`` and ``starts`` must have shape ``(env_count, filter_count)``.
    The function rejects overlapping referenced intervals because they would
    make pair identity ambiguous.  It also marks a full contact buffer as
    UNKNOWN: a full buffer may have truncated contacts, so its maximum
    penetration cannot be used as proof of safety.
    """

    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    environments = int(env_count)
    if environments < 1:
        raise PhysxContactSeparationLayoutError(
            "env_count must be positive"
        )
    body_kind = str(body_class)
    if body_kind not in {"wheel", "nonwheel"}:
        raise PhysxContactSeparationLayoutError(
            f"unsupported body_class={body_class!r}"
        )
    body_path = str(body_prim_path).rstrip("/")
    if not body_path.startswith("/"):
        raise PhysxContactSeparationLayoutError(
            "body_prim_path must be absolute"
        )
    ordered_filters = tuple(
        (str(surface), str(path).rstrip("/")) for surface, path in filters
    )
    if not ordered_filters or any(
        not surface or not path.startswith("/")
        for surface, path in ordered_filters
    ):
        raise PhysxContactSeparationLayoutError(
            "filters must contain named absolute prim paths"
        )
    if len({path for _surface, path in ordered_filters}) != len(ordered_filters):
        raise PhysxContactSeparationLayoutError(
            "filter prim paths must be unique"
        )
    expected_paths = [path for _surface, path in ordered_filters]
    actual_paths = [str(path).rstrip("/") for path in configured_filter_paths]
    if actual_paths != expected_paths:
        raise PhysxContactSeparationLayoutError(
            "configured filter identity/order does not match expected filters: "
            f"configured={actual_paths}, expected={expected_paths}"
        )
    expected_sensors = _exact_absolute_paths(
        expected_sensor_paths, label="expected_sensor_paths"
    )
    live_sensors = _exact_absolute_paths(
        view_sensor_paths, label="RigidContactView.sensor_paths"
    )
    if live_sensors != expected_sensors:
        raise PhysxContactSeparationLayoutError(
            "RigidContactView sensor identity/order does not match expected "
            f"sensors: live={list(live_sensors)}, expected={list(expected_sensors)}"
        )
    if len(live_sensors) != environments:
        raise PhysxContactSeparationLayoutError(
            f"RigidContactView.sensor_paths count={len(live_sensors)} does not "
            f"equal env_count={environments}"
        )
    try:
        parsed_filter_count = int(view_filter_count)
    except (TypeError, ValueError) as exc:
        raise PhysxContactSeparationLayoutError(
            "RigidContactView.filter_count is unavailable"
        ) from exc
    if parsed_filter_count != len(ordered_filters):
        raise PhysxContactSeparationLayoutError(
            f"RigidContactView.filter_count={parsed_filter_count} does not equal "
            f"configured filters={len(ordered_filters)}"
        )
    live_filter_matrix = _exact_absolute_path_matrix(
        view_filter_paths,
        label="RigidContactView.filter_paths",
        expected_rows=len(live_sensors),
        expected_columns=parsed_filter_count,
    )
    expected_filter_row = tuple(expected_paths)
    for sensor_index, live_filter_row in enumerate(live_filter_matrix):
        if live_filter_row != expected_filter_row:
            raise PhysxContactSeparationLayoutError(
                "RigidContactView filter identity/order does not match expected "
                f"filters for sensor_index={sensor_index}: "
                f"live={list(live_filter_row)}, expected={expected_paths}"
            )
    try:
        declared_capacity = int(view_max_contact_data_count)
    except (TypeError, ValueError) as exc:
        raise PhysxContactSeparationLayoutError(
            "RigidContactView.max_contact_data_count is unavailable"
        ) from exc
    if declared_capacity < 1:
        raise PhysxContactSeparationLayoutError(
            "RigidContactView.max_contact_data_count must be positive"
        )

    signed = _to_numpy(distances)
    if signed.ndim != 2 or int(signed.shape[1]) != 1:
        raise PhysxContactSeparationLayoutError(
            f"distances shape={signed.shape}; expected (capacity, 1)"
        )
    capacity = int(signed.shape[0])
    if capacity < 1:
        raise PhysxContactSeparationLayoutError(
            "distance buffer capacity must be positive"
        )
    if capacity != declared_capacity:
        raise PhysxContactSeparationLayoutError(
            f"distance buffer capacity={capacity} does not equal "
            f"RigidContactView.max_contact_data_count={declared_capacity}"
        )
    expected_shape = (environments, len(ordered_filters))
    pair_counts = _exact_nonnegative_integers(
        counts, label="counts", expected_shape=expected_shape
    )
    pair_starts = _exact_nonnegative_integers(
        starts, label="starts", expected_shape=expected_shape
    )

    intervals: list[tuple[int, int, int, int]] = []
    total_referenced = 0
    for env_id in range(environments):
        for filter_index in range(len(ordered_filters)):
            start = int(pair_starts[env_id, filter_index])
            count = int(pair_counts[env_id, filter_index])
            stop = start + count
            if start > capacity or stop > capacity:
                raise PhysxContactSeparationLayoutError(
                    f"pair ({env_id}, {filter_index}) interval [{start}, {stop}) "
                    f"exceeds distance capacity={capacity}"
                )
            total_referenced += count
            if count:
                intervals.append((start, stop, env_id, filter_index))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise PhysxContactSeparationLayoutError(
                "contact intervals overlap: "
                f"({previous[2]}, {previous[3]})={previous[:2]} and "
                f"({current[2]}, {current[3]})={current[:2]}"
            )
    if total_referenced > capacity:
        raise PhysxContactSeparationLayoutError(
            f"referenced contact count={total_referenced} exceeds capacity={capacity}"
        )

    global_capacity_exhausted = bool(total_referenced >= capacity)
    rows: list[dict[str, Any]] = []
    for env_id in range(environments):
        for filter_index, (surface, other_prim_path) in enumerate(
            ordered_filters
        ):
            start = int(pair_starts[env_id, filter_index])
            count = int(pair_counts[env_id, filter_index])
            stop = start + count
            identity = _identity_fields(
                env_id=env_id,
                body_class=body_kind,
                body_name=str(body_name),
                body_prim_path=live_sensors[env_id],
                leg=leg,
                filter_index=filter_index,
                surface=surface,
                other_prim_path=other_prim_path,
            )
            capacity_exhausted = global_capacity_exhausted
            try:
                referenced = np.asarray(signed[start:stop, 0], dtype=float)
            except (TypeError, ValueError) as exc:
                raise PhysxContactSeparationEvidenceError(
                    f"referenced distances for {identity['pair_id']} are not numeric: {exc}"
                ) from exc
            if not bool(np.isfinite(referenced).all()):
                raise PhysxContactSeparationEvidenceError(
                    f"referenced distances for {identity['pair_id']} contain non-finite values"
                )
            values = [float(value) for value in referenced.tolist()]
            if capacity_exhausted:
                rows.append(
                    {
                        **identity,
                        "contact_count": count,
                        "signed_separations_m": values,
                        "minimum_signed_separation_m": (
                            min(values) if values else None
                        ),
                        "maximum_penetration_m": None,
                        "valid": False,
                        "status": "UNKNOWN",
                        "capacity": capacity,
                        "capacity_exhausted": True,
                        "source": PHYSX_SEPARATION_SOURCE,
                        "error": (
                            "PhysX contact buffer capacity exhausted; "
                            "maximum penetration may be truncated"
                        ),
                    }
                )
                continue
            minimum = min(values) if values else None
            rows.append(
                {
                    **identity,
                    "contact_count": count,
                    "signed_separations_m": values,
                    "minimum_signed_separation_m": minimum,
                    "maximum_penetration_m": (
                        0.0 if minimum is None else max(0.0, -float(minimum))
                    ),
                    "valid": True,
                    "status": "CONTACT" if count else "NO_CONTACT",
                    "capacity": capacity,
                    "capacity_exhausted": False,
                    "source": PHYSX_SEPARATION_SOURCE,
                    "error": "",
                }
            )
    return rows


def separation_evidence_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate exact pair rows without upgrading UNKNOWN to safe zero."""

    copied = [dict(row) for row in rows]
    pair_ids = [str(row.get("pair_id", "") or "") for row in copied]
    errors: list[str] = []
    if any(not pair_id for pair_id in pair_ids):
        errors.append("one or more PhysX separation rows lack pair_id")
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("PhysX separation pair identities are duplicated")
    if expected_pair_ids is not None:
        expected = [str(value) for value in expected_pair_ids]
        if set(pair_ids) != set(expected) or len(pair_ids) != len(expected):
            errors.append(
                "PhysX separation pair identity set does not match expected set"
            )
    invalid = [
        str(row.get("pair_id", "") or "<missing>")
        for row in copied
        if row.get("valid") is not True
    ]
    finite_penetrations: list[float] = []
    if not errors and not invalid:
        for row in copied:
            try:
                value = float(row.get("maximum_penetration_m"))
            except (TypeError, ValueError):
                errors.append(
                    f"{row.get('pair_id', '<missing>')} has no finite maximum penetration"
                )
                continue
            if not math.isfinite(value) or value < 0.0:
                errors.append(
                    f"{row.get('pair_id', '<missing>')} has invalid maximum penetration"
                )
            else:
                finite_penetrations.append(value)
    valid = bool(copied and not errors and not invalid)
    by_scope: dict[str, float] | None = None
    maximum: float | None = None
    if valid:
        by_scope = {}
        for row in copied:
            scope = f"{row.get('body_class', '')}_{row.get('surface', '')}"
            by_scope[scope] = max(
                by_scope.get(scope, 0.0),
                float(row["maximum_penetration_m"]),
            )
        maximum = max(finite_penetrations, default=0.0)
    return {
        "schema_version": "fsm50.physx_contact_separation.v1",
        "valid": valid,
        "status": "AVAILABLE" if valid else "UNKNOWN",
        "pair_count": len(copied),
        "pair_ids": pair_ids,
        "unknown_pair_ids": invalid,
        "maximum_physx_penetration_m": maximum,
        "maximum_by_scope_m": by_scope,
        "source": PHYSX_SEPARATION_SOURCE,
        "errors": errors,
    }
