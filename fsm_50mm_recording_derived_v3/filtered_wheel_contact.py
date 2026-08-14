"""Per-wheel, surface-filtered ContactSensor integration for the 50 mm FSM.

The module is intentionally safe to import before Isaac Sim starts.  IsaacLab
types are imported lazily by :func:`create_filtered_wheel_contact_sensor_bank`.
This lets configuration, layout, and decoding logic run in ordinary unit tests.

IsaacLab filtered contacts are one-to-many: one sensor body may be filtered
against several scene bodies, but a wildcard matching several sensor bodies is
not a supported many-to-many query.  Consequently this module creates four
ContactSensor instances, one for each wheel rigid body, and applies the same
ordered ground/obstacle filter list to every sensor.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .physx_contact_separation import (
    decode_contact_pair_separations,
    separation_evidence_summary,
    unknown_contact_pair_rows,
)


ROBOT_PRIM_PATH = "/World/WLRRobot"
GROUND_PRIM_PATH = "/World/defaultGroundPlane"
GROUND_CONTACT_PRIM_PATH = "/World/defaultGroundPlane/GroundPlane/CollisionPlane"
OBSTACLE_PRIM_PATH = "/World/Obstacle"

# These are rigid-body prim names, not the ankle joint names.  They are also
# present as resolved_body_path values in the project's real worker ground
# diagnostics.
LEG_TO_WHEEL_BODY: dict[str, str] = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}

FILTERED_SURFACES: tuple[tuple[str, str], ...] = (
    ("ground", GROUND_CONTACT_PRIM_PATH),
    ("obstacle", OBSTACLE_PRIM_PATH),
)


class FilteredContactLayoutError(RuntimeError):
    """Raised when runtime ContactSensor tensors cannot be labelled safely."""


@dataclass(frozen=True)
class WheelContactSensorSpec:
    """Identity and exact rigid-body prim path for one wheel sensor."""

    leg: str
    body_name: str
    prim_path: str


@dataclass(frozen=True)
class FilteredWheelContactObservation:
    """One wheel/filter-pair observation at one simulation sample."""

    leg: str
    wheel_body_name: str
    wheel_prim_path: str
    filter_index: int
    surface: str
    other_prim_path: str
    active: bool
    normal_force_w: tuple[float, float, float]
    normal_force_n: float
    upward_force_n: float
    friction_force_w: tuple[float, float, float]
    friction_force_n: float
    total_force_w: tuple[float, float, float]
    total_force_n: float
    contact_point_w: tuple[float, float, float]
    force_valid: bool
    contact_point_valid: bool
    source: str = "isaaclab.ContactSensor.force_matrix_w"

    def as_dict(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "wheel_body_name": self.wheel_body_name,
            "wheel_prim_path": self.wheel_prim_path,
            "filter_index": self.filter_index,
            "surface": self.surface,
            "other_prim_path": self.other_prim_path,
            "active": self.active,
            "normal_force_w": list(self.normal_force_w),
            "normal_force_n": self.normal_force_n,
            "upward_force_n": self.upward_force_n,
            "friction_force_w": list(self.friction_force_w),
            "friction_force_n": self.friction_force_n,
            "total_force_w": list(self.total_force_w),
            "total_force_n": self.total_force_n,
            "contact_point_w": list(self.contact_point_w),
            "force_valid": self.force_valid,
            "contact_point_valid": self.contact_point_valid,
            "source": self.source,
        }


@dataclass(frozen=True)
class FilteredWheelContactBankData:
    """Numpy snapshot compatible with the existing telemetry contact reader."""

    net_forces_w: np.ndarray
    force_matrix_w: np.ndarray
    contact_pos_w: np.ndarray
    friction_forces_w: np.ndarray


def wheel_contact_sensor_specs(
    robot_prim_path: str = ROBOT_PRIM_PATH,
) -> tuple[WheelContactSensorSpec, ...]:
    """Return the deterministic FL/FR/RL/RR single-body sensor layout."""

    root = str(robot_prim_path).rstrip("/")
    return tuple(
        WheelContactSensorSpec(
            leg=leg,
            body_name=body_name,
            prim_path=f"{root}/{body_name}",
        )
        for leg, body_name in LEG_TO_WHEEL_BODY.items()
    )


def contact_sensor_config_kwargs(
    spec: WheelContactSensorSpec,
    *,
    history_length: int = 3,
    force_threshold_n: float = 1.0,
    max_contact_data_count_per_prim: int = 16,
) -> dict[str, Any]:
    """Build one ContactSensorCfg payload without importing IsaacLab."""

    if int(max_contact_data_count_per_prim) < 1:
        raise ValueError("max_contact_data_count_per_prim must be positive")
    if int(history_length) < 0:
        raise ValueError("history_length cannot be negative")
    if not math.isfinite(float(force_threshold_n)) or float(force_threshold_n) < 0.0:
        raise ValueError("force_threshold_n must be finite and non-negative")
    return {
        "prim_path": spec.prim_path,
        "update_period": 0.0,
        "history_length": int(history_length),
        "debug_vis": False,
        "track_pose": True,
        "track_contact_points": True,
        "track_friction_forces": True,
        "track_air_time": True,
        "force_threshold": float(force_threshold_n),
        "filter_prim_paths_expr": [path for _name, path in FILTERED_SURFACES],
        "max_contact_data_count_per_prim": int(max_contact_data_count_per_prim),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _finite_vec3(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3:
        return (float("nan"),) * 3
    return tuple(float(item) for item in array[:3])


class FilteredWheelContactSensorBank:
    """Drop-in contact sensor facade over four filtered single-body sensors.

    The facade implements the subset used by ``telemetry.contact_metrics``:
    ``update()``, ``reset()``, ``body_names``, and ``data.net_forces_w``.  The
    richer :meth:`filtered_observations` method preserves the filter identity,
    true averaged contact point, and friction force for the FSM evidence path.
    """

    is_filtered_wheel_contact_bank = True

    def __init__(
        self,
        sensors: Mapping[str, Any],
        specs: Sequence[WheelContactSensorSpec],
        *,
        force_threshold_n: float,
        max_contact_data_count_per_prim: int = 16,
    ) -> None:
        self.specs = tuple(specs)
        self.sensors = {spec.leg: sensors[spec.leg] for spec in self.specs}
        self.force_threshold_n = float(force_threshold_n)
        self.filter_surfaces = FILTERED_SURFACES
        self.cfg = SimpleNamespace(
            force_threshold=self.force_threshold_n,
            filter_prim_paths_expr=[path for _name, path in FILTERED_SURFACES],
            max_contact_data_count_per_prim=int(
                max_contact_data_count_per_prim
            ),
        )
        self._cached_data: FilteredWheelContactBankData | None = None
        self._cached_separation_rows: list[dict[str, Any]] = []

    @property
    def body_names(self) -> list[str]:
        return [spec.body_name for spec in self.specs]

    @property
    def data(self) -> FilteredWheelContactBankData:
        if self._cached_data is None:
            self._cached_data = self._collect_data()
        return self._cached_data

    def update(self, dt: float, force_recompute: bool = False) -> None:
        for sensor in self.sensors.values():
            sensor.update(float(dt), force_recompute=bool(force_recompute))
        self._cached_data = self._collect_data()
        self._cached_separation_rows = self._collect_separation_rows(float(dt))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for sensor in self.sensors.values():
            sensor.reset(env_ids)
        self._cached_data = None
        self._cached_separation_rows = []

    def _collect_separation_rows(self, dt: float) -> list[dict[str, Any]]:
        """Capture signed PhysX separations without changing simulation state."""

        env_count = int(self.data.force_matrix_w.shape[0])
        rows: list[dict[str, Any]] = []
        for spec in self.specs:
            sensor = self.sensors[spec.leg]
            sensor_cfg = getattr(sensor, "cfg", None)
            configured_filters = list(
                getattr(sensor_cfg, "filter_prim_paths_expr", []) or []
            )
            view = getattr(sensor, "contact_physx_view", None)
            view_capacity: int | None = None
            try:
                if view is None or not hasattr(view, "get_contact_data"):
                    raise RuntimeError("ContactSensor.contact_physx_view unavailable")
                raw_view_capacity = getattr(view, "max_contact_data_count", None)
                try:
                    view_capacity = int(raw_view_capacity)
                except (TypeError, ValueError):
                    view_capacity = None
                payload = view.get_contact_data(float(dt))
                if not isinstance(payload, (tuple, list)) or len(payload) != 6:
                    raise RuntimeError(
                        "RigidContactView.get_contact_data did not return six buffers"
                    )
                rows.extend(
                    decode_contact_pair_separations(
                        dt_s=float(dt),
                        distances=payload[3],
                        counts=payload[4],
                        starts=payload[5],
                        env_count=env_count,
                        body_class="wheel",
                        body_name=spec.body_name,
                        body_prim_path=spec.prim_path,
                        filters=FILTERED_SURFACES,
                        configured_filter_paths=configured_filters,
                        expected_sensor_paths=[spec.prim_path],
                        view_sensor_paths=getattr(view, "sensor_paths", None),
                        view_filter_paths=getattr(view, "filter_paths", None),
                        view_filter_count=getattr(view, "filter_count", None),
                        view_max_contact_data_count=raw_view_capacity,
                        leg=spec.leg,
                    )
                )
            except Exception as exc:
                rows.extend(
                    unknown_contact_pair_rows(
                        env_count=env_count,
                        body_class="wheel",
                        body_name=spec.body_name,
                        body_prim_path=spec.prim_path,
                        filters=FILTERED_SURFACES,
                        leg=spec.leg,
                        capacity=view_capacity,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return rows

    def separation_observations(self, *, env_id: int = 0) -> list[dict[str, Any]]:
        """Return all eight signed-separation pairs for one environment."""

        env_count = int(self.data.force_matrix_w.shape[0])
        if not 0 <= int(env_id) < env_count:
            raise IndexError(f"env_id={env_id} outside [0, {env_count})")
        if not self._cached_separation_rows:
            rows: list[dict[str, Any]] = []
            for spec in self.specs:
                view = getattr(self.sensors[spec.leg], "contact_physx_view", None)
                try:
                    capacity = int(getattr(view, "max_contact_data_count", None))
                except (TypeError, ValueError):
                    capacity = None
                rows.extend(
                    unknown_contact_pair_rows(
                        env_count=env_count,
                        body_class="wheel",
                        body_name=spec.body_name,
                        body_prim_path=spec.prim_path,
                        filters=FILTERED_SURFACES,
                        leg=spec.leg,
                        capacity=capacity,
                        error="signed-separation view has not been sampled",
                    )
                )
        else:
            rows = self._cached_separation_rows
        return [dict(row) for row in rows if int(row["env_id"]) == int(env_id)]

    def separation_evidence(self, *, env_id: int = 0) -> dict[str, Any]:
        rows = self.separation_observations(env_id=env_id)
        expected = [
            f"env={int(env_id)}|body={spec.prim_path}|other={other_path}"
            for spec in self.specs
            for _surface, other_path in FILTERED_SURFACES
        ]
        return separation_evidence_summary(rows, expected_pair_ids=expected)

    def _collect_data(self) -> FilteredWheelContactBankData:
        snapshots = [self.sensors[spec.leg].data for spec in self.specs]
        net = self._combine(snapshots, "net_forces_w", expected_ndim=3)
        force_matrix = self._combine(snapshots, "force_matrix_w", expected_ndim=4)
        contact_pos = self._combine(snapshots, "contact_pos_w", expected_ndim=4)
        friction = self._combine(snapshots, "friction_forces_w", expected_ndim=4)
        filter_count = len(self.filter_surfaces)
        for name, array in (
            ("force_matrix_w", force_matrix),
            ("contact_pos_w", contact_pos),
            ("friction_forces_w", friction),
        ):
            if array.shape[2] != filter_count:
                raise FilteredContactLayoutError(
                    f"{name} filter_count={array.shape[2]} does not match "
                    f"configured ordered filters={filter_count}; refusing to mislabel contacts"
                )
        return FilteredWheelContactBankData(
            net_forces_w=net,
            force_matrix_w=force_matrix,
            contact_pos_w=contact_pos,
            friction_forces_w=friction,
        )

    @staticmethod
    def _combine(snapshots: Sequence[Any], field: str, *, expected_ndim: int) -> np.ndarray:
        arrays: list[np.ndarray] = []
        env_count: int | None = None
        trailing_shape: tuple[int, ...] | None = None
        for index, snapshot in enumerate(snapshots):
            array = _to_numpy(getattr(snapshot, field, None))
            if array.ndim != expected_ndim:
                raise FilteredContactLayoutError(
                    f"sensor[{index}].{field} expected {expected_ndim} dimensions, got {array.shape}"
                )
            if array.shape[1] != 1:
                raise FilteredContactLayoutError(
                    f"sensor[{index}].{field} resolved {array.shape[1]} bodies; exactly one is required"
                )
            if env_count is None:
                env_count = int(array.shape[0])
            elif int(array.shape[0]) != env_count:
                raise FilteredContactLayoutError(
                    f"sensor[{index}].{field} environment count differs from the first sensor"
                )
            if trailing_shape is None:
                trailing_shape = tuple(int(value) for value in array.shape[2:])
            elif tuple(int(value) for value in array.shape[2:]) != trailing_shape:
                raise FilteredContactLayoutError(
                    f"sensor[{index}].{field} trailing shape {array.shape[2:]} differs "
                    f"from the first sensor {trailing_shape}"
                )
            arrays.append(array)
        if not arrays:
            raise FilteredContactLayoutError("filtered wheel contact bank has no sensors")
        return np.concatenate(arrays, axis=1)

    def filtered_observations(
        self,
        *,
        env_id: int = 0,
        force_threshold_n: float | None = None,
    ) -> list[FilteredWheelContactObservation]:
        """Decode all eight wheel/filter pairs with explicit surface labels."""

        data = self.data
        if not 0 <= int(env_id) < data.force_matrix_w.shape[0]:
            raise IndexError(
                f"env_id={env_id} outside [0, {data.force_matrix_w.shape[0]})"
            )
        threshold = (
            self.force_threshold_n
            if force_threshold_n is None
            else float(force_threshold_n)
        )
        rows: list[FilteredWheelContactObservation] = []
        for body_index, spec in enumerate(self.specs):
            for filter_index, (surface, other_prim_path) in enumerate(
                self.filter_surfaces
            ):
                normal = np.asarray(
                    data.force_matrix_w[int(env_id), body_index, filter_index, :3],
                    dtype=float,
                )
                friction = np.asarray(
                    data.friction_forces_w[int(env_id), body_index, filter_index, :3],
                    dtype=float,
                )
                point = np.asarray(
                    data.contact_pos_w[int(env_id), body_index, filter_index, :3],
                    dtype=float,
                )
                force_valid = bool(np.isfinite(normal).all())
                friction_valid = bool(np.isfinite(friction).all())
                point_valid = bool(np.isfinite(point).all())
                normal_norm = float(np.linalg.norm(normal)) if force_valid else float("nan")
                friction_norm = (
                    float(np.linalg.norm(friction)) if friction_valid else float("nan")
                )
                total = normal + friction if force_valid and friction_valid else normal
                total_norm = (
                    float(np.linalg.norm(total)) if force_valid else float("nan")
                )
                active = bool(force_valid and normal_norm >= threshold)
                rows.append(
                    FilteredWheelContactObservation(
                        leg=spec.leg,
                        wheel_body_name=spec.body_name,
                        wheel_prim_path=spec.prim_path,
                        filter_index=filter_index,
                        surface=surface,
                        other_prim_path=other_prim_path,
                        active=active,
                        normal_force_w=_finite_vec3(normal),
                        normal_force_n=normal_norm,
                        upward_force_n=(
                            max(0.0, float(normal[2])) if force_valid else float("nan")
                        ),
                        friction_force_w=_finite_vec3(friction),
                        friction_force_n=friction_norm,
                        total_force_w=_finite_vec3(total),
                        total_force_n=total_norm,
                        contact_point_w=_finite_vec3(point),
                        force_valid=force_valid,
                        contact_point_valid=point_valid,
                    )
                )
        return rows


def create_filtered_wheel_contact_sensor_bank(
    *,
    robot_prim_path: str = ROBOT_PRIM_PATH,
    history_length: int = 3,
    force_threshold_n: float = 1.0,
    max_contact_data_count_per_prim: int = 16,
    sensor_cls: type[Any] | None = None,
    sensor_cfg_cls: type[Any] | None = None,
) -> FilteredWheelContactSensorBank:
    """Create four IsaacLab sensors; Isaac imports occur only inside this call.

    ``sensor_cls`` and ``sensor_cfg_cls`` are dependency-injection seams for
    no-Isaac tests.  Production callers leave them unset and invoke the factory
    only after ``AppLauncher`` has started Kit and the wheel prims exist.
    """

    if sensor_cls is None or sensor_cfg_cls is None:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore

        sensor_cls = ContactSensor if sensor_cls is None else sensor_cls
        sensor_cfg_cls = ContactSensorCfg if sensor_cfg_cls is None else sensor_cfg_cls
    specs = wheel_contact_sensor_specs(robot_prim_path)
    sensors: dict[str, Any] = {}
    for spec in specs:
        kwargs = contact_sensor_config_kwargs(
            spec,
            history_length=history_length,
            force_threshold_n=force_threshold_n,
            max_contact_data_count_per_prim=max_contact_data_count_per_prim,
        )
        sensors[spec.leg] = sensor_cls(sensor_cfg_cls(**kwargs))
    return FilteredWheelContactSensorBank(
        sensors,
        specs,
        force_threshold_n=force_threshold_n,
        max_contact_data_count_per_prim=max_contact_data_count_per_prim,
    )


def make_filtered_wheel_contact_sensor_factory(
    **kwargs: Any,
) -> Callable[[], tuple[FilteredWheelContactSensorBank | None, str]]:
    """Return the non-throwing zero-argument factory expected by the scene."""

    return functools.partial(create_filtered_wheel_contact_sensor_bank_result, **kwargs)


def create_filtered_wheel_contact_sensor_bank_result(
    **kwargs: Any,
) -> tuple[FilteredWheelContactSensorBank | None, str]:
    """Create the bank using the legacy ``(sensor, error)`` scene contract."""

    try:
        return create_filtered_wheel_contact_sensor_bank(**kwargs), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def configure_scene_for_filtered_wheel_contacts(
    scene_config: Any,
    **factory_kwargs: Any,
) -> Any:
    """Enable contact reporters and install the pre-reset per-wheel factory."""

    scene_config.telemetry_contact_sensors_enabled = True
    scene_config.contact_sensor_factory = make_filtered_wheel_contact_sensor_factory(
        **factory_kwargs
    )
    return scene_config


def filtered_contact_rows(
    sensor: Any,
    *,
    env_id: int = 0,
    force_threshold_n: float | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-ready filtered observations, failing on an unfiltered sensor."""

    if not bool(getattr(sensor, "is_filtered_wheel_contact_bank", False)):
        raise TypeError("sensor is not a FilteredWheelContactSensorBank")
    return [
        row.as_dict()
        for row in sensor.filtered_observations(
            env_id=env_id,
            force_threshold_n=force_threshold_n,
        )
    ]


__all__ = [
    "FILTERED_SURFACES",
    "GROUND_CONTACT_PRIM_PATH",
    "GROUND_PRIM_PATH",
    "LEG_TO_WHEEL_BODY",
    "OBSTACLE_PRIM_PATH",
    "ROBOT_PRIM_PATH",
    "FilteredContactLayoutError",
    "FilteredWheelContactBankData",
    "FilteredWheelContactObservation",
    "FilteredWheelContactSensorBank",
    "WheelContactSensorSpec",
    "configure_scene_for_filtered_wheel_contacts",
    "contact_sensor_config_kwargs",
    "create_filtered_wheel_contact_sensor_bank",
    "create_filtered_wheel_contact_sensor_bank_result",
    "filtered_contact_rows",
    "make_filtered_wheel_contact_sensor_factory",
    "wheel_contact_sensor_specs",
]
