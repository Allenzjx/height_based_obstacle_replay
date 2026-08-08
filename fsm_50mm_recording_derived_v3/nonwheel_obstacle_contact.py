"""Force-backed non-wheel robot contact evidence against the 50 mm obstacle.

This module deliberately does not infer collision from bounds or overlap.  It
creates one IsaacLab ``ContactSensor`` for every non-wheel rigid body below the
robot prim and filters every sensor exclusively against ``/World/Obstacle``.
The resulting PhysX force matrix is the source of collision evidence.

Isaac/Kit imports are lazy.  Production creation must happen after the robot
and obstacle prims have been authored, and before the first simulation reset.
The dependency-injection arguments are intentional: discovery, tensor layout,
finite-value rejection, and factory composition can be tested without starting
Kit.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROBOT_PRIM_PATH = "/World/WLRRobot"
OBSTACLE_PRIM_PATH = "/World/Obstacle"
WHEEL_BODY_NAMES: frozenset[str] = frozenset(
    {
        "front_left_wheel",
        "front_right_wheel",
        "rear_left_wheel",
        "rear_right_wheel",
    }
)

NONWHEEL_FORCE_SOURCE = "isaaclab.ContactSensor.force_matrix_w:nonwheel_to_obstacle"


class NonWheelContactDiscoveryError(RuntimeError):
    """Raised when the runtime USD rigid-body set cannot be resolved safely."""


class NonWheelContactLayoutError(RuntimeError):
    """Raised when ContactSensor tensors cannot be labelled without guessing."""


class NonWheelContactEvidenceError(RuntimeError):
    """Raised when force/contact evidence is non-finite or internally invalid."""


@dataclass(frozen=True)
class NonWheelRigidBodySpec:
    """Identity of one exact, non-wheel RigidBodyAPI prim."""

    body_name: str
    prim_path: str


@dataclass(frozen=True)
class NonWheelObstacleContactObservation:
    """One non-wheel body versus obstacle observation for one environment."""

    body_name: str
    body_prim_path: str
    other_prim_path: str
    filter_index: int
    active: bool
    normal_force_w: tuple[float, float, float]
    normal_force_n: float
    friction_force_w: tuple[float, float, float]
    friction_force_n: float
    total_force_w: tuple[float, float, float]
    total_force_n: float
    contact_point_w: tuple[float, float, float]
    force_valid: bool
    contact_point_valid: bool
    source: str = NONWHEEL_FORCE_SOURCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "body_name": self.body_name,
            "body_prim_path": self.body_prim_path,
            "other_prim_path": self.other_prim_path,
            "filter_index": self.filter_index,
            "active": self.active,
            "normal_force_w": list(self.normal_force_w),
            "normal_force_n": self.normal_force_n,
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
class NonWheelObstacleContactBankData:
    """Concatenated snapshot in deterministic body-path order."""

    net_forces_w: np.ndarray
    force_matrix_w: np.ndarray
    contact_pos_w: np.ndarray
    friction_forces_w: np.ndarray


def _prim_is_valid(prim: Any) -> bool:
    if prim is None:
        return False
    validator = getattr(prim, "IsValid", None)
    if callable(validator):
        try:
            return bool(validator())
        except Exception:
            return False
    try:
        return bool(prim)
    except Exception:
        return False


def _prim_path(prim: Any) -> str:
    path = prim.GetPath()
    value = getattr(path, "pathString", None)
    return str(value if value is not None else path)


def _default_stage() -> Any:
    import omni.usd  # type: ignore

    return omni.usd.get_context().get_stage()


def _default_usd_types() -> tuple[type[Any], Callable[[Any], Iterable[Any]]]:
    from pxr import Usd, UsdPhysics  # type: ignore

    return UsdPhysics.RigidBodyAPI, Usd.PrimRange


def _is_wheel_body_path(path: str, wheel_body_names: frozenset[str]) -> bool:
    """Return true for known wheel bodies and conservatively named wheel links."""

    leaf = str(path).rstrip("/").rsplit("/", 1)[-1].lower()
    # Exact known names cover the current WLR asset.  Rejecting any other
    # wheel-named rigid body prevents an asset revision from silently turning
    # wheel/obstacle contact into a chassis-collision failure.
    return leaf in wheel_body_names or "wheel" in leaf


def discover_nonwheel_rigid_body_specs(
    *,
    stage: Any | None = None,
    robot_prim_path: str = ROBOT_PRIM_PATH,
    obstacle_prim_path: str = OBSTACLE_PRIM_PATH,
    wheel_body_names: Sequence[str] = tuple(sorted(WHEEL_BODY_NAMES)),
    rigid_body_api_cls: type[Any] | None = None,
    prim_range_factory: Callable[[Any], Iterable[Any]] | None = None,
) -> tuple[NonWheelRigidBodySpec, ...]:
    """Discover every non-wheel RigidBodyAPI prim under the robot.

    The obstacle and robot roots must already exist.  An empty result, duplicate
    path, invalid path, or traversal outside the robot root raises rather than
    degrading to an unverified "no collision" result.
    """

    stage = _default_stage() if stage is None else stage
    if stage is None:
        raise NonWheelContactDiscoveryError("current USD stage is unavailable")

    root_path = str(robot_prim_path).rstrip("/")
    obstacle_path = str(obstacle_prim_path).rstrip("/")
    if not root_path.startswith("/") or not obstacle_path.startswith("/"):
        raise NonWheelContactDiscoveryError("robot and obstacle prim paths must be absolute")

    root = stage.GetPrimAtPath(root_path)
    if not _prim_is_valid(root):
        raise NonWheelContactDiscoveryError(f"robot prim does not exist: {root_path}")
    obstacle = stage.GetPrimAtPath(obstacle_path)
    if not _prim_is_valid(obstacle):
        raise NonWheelContactDiscoveryError(f"obstacle prim does not exist: {obstacle_path}")

    if rigid_body_api_cls is None or prim_range_factory is None:
        default_api, default_range = _default_usd_types()
        rigid_body_api_cls = default_api if rigid_body_api_cls is None else rigid_body_api_cls
        prim_range_factory = default_range if prim_range_factory is None else prim_range_factory

    wheel_names = frozenset(str(name).lower() for name in wheel_body_names)
    discovered: dict[str, NonWheelRigidBodySpec] = {}
    try:
        prims = prim_range_factory(root)
        for prim in prims:
            if not _prim_is_valid(prim):
                continue
            try:
                is_rigid = bool(prim.HasAPI(rigid_body_api_cls))
            except Exception as exc:
                raise NonWheelContactDiscoveryError(
                    f"RigidBodyAPI query failed for {_prim_path(prim)!r}: {exc}"
                ) from exc
            if not is_rigid:
                continue
            path = _prim_path(prim).rstrip("/")
            if path != root_path and not path.startswith(root_path + "/"):
                raise NonWheelContactDiscoveryError(
                    f"prim traversal escaped robot root: {path}"
                )
            if _is_wheel_body_path(path, wheel_names):
                continue
            body_name = path.rsplit("/", 1)[-1]
            if not body_name:
                raise NonWheelContactDiscoveryError(f"rigid body has invalid path: {path!r}")
            if path in discovered:
                raise NonWheelContactDiscoveryError(f"duplicate rigid-body path: {path}")
            discovered[path] = NonWheelRigidBodySpec(body_name=body_name, prim_path=path)
    except NonWheelContactDiscoveryError:
        raise
    except Exception as exc:
        raise NonWheelContactDiscoveryError(f"robot rigid-body traversal failed: {exc}") from exc

    specs = tuple(discovered[path] for path in sorted(discovered))
    if not specs:
        raise NonWheelContactDiscoveryError(
            f"no non-wheel RigidBodyAPI prims found under {root_path}"
        )
    return specs


def nonwheel_contact_sensor_config_kwargs(
    spec: NonWheelRigidBodySpec,
    *,
    obstacle_prim_path: str = OBSTACLE_PRIM_PATH,
    history_length: int = 3,
    force_threshold_n: float = 1.0,
    max_contact_data_count_per_prim: int = 16,
) -> dict[str, Any]:
    """Build one exact-body, obstacle-only ContactSensorCfg payload."""

    if int(history_length) < 0:
        raise ValueError("history_length cannot be negative")
    if int(max_contact_data_count_per_prim) < 1:
        raise ValueError("max_contact_data_count_per_prim must be positive")
    if not math.isfinite(float(force_threshold_n)) or float(force_threshold_n) < 0.0:
        raise ValueError("force_threshold_n must be finite and non-negative")
    obstacle_path = str(obstacle_prim_path).rstrip("/")
    if not obstacle_path.startswith("/"):
        raise ValueError("obstacle_prim_path must be absolute")
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
        "filter_prim_paths_expr": [obstacle_path],
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


def _vec3(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3:
        raise NonWheelContactLayoutError(f"expected vec3, got shape {array.shape}")
    return tuple(float(item) for item in array)


class NonWheelObstacleContactSensorBank:
    """Facade over one obstacle-filtered ContactSensor per non-wheel body."""

    is_nonwheel_obstacle_contact_bank = True

    def __init__(
        self,
        sensors: Mapping[str, Any],
        specs: Sequence[NonWheelRigidBodySpec],
        *,
        obstacle_prim_path: str,
        force_threshold_n: float,
    ) -> None:
        self.specs = tuple(specs)
        if not self.specs:
            raise NonWheelContactLayoutError("non-wheel contact bank has no body specs")
        self.sensors = {spec.prim_path: sensors[spec.prim_path] for spec in self.specs}
        self.obstacle_prim_path = str(obstacle_prim_path).rstrip("/")
        self.force_threshold_n = float(force_threshold_n)
        self.cfg = SimpleNamespace(
            force_threshold=self.force_threshold_n,
            filter_prim_paths_expr=[self.obstacle_prim_path],
        )
        self._cached_data: NonWheelObstacleContactBankData | None = None

    @property
    def body_names(self) -> list[str]:
        return [spec.body_name for spec in self.specs]

    @property
    def data(self) -> NonWheelObstacleContactBankData:
        if self._cached_data is None:
            self._cached_data = self._collect_data()
        return self._cached_data

    def update(self, dt: float, force_recompute: bool = False) -> None:
        if not math.isfinite(float(dt)) or float(dt) < 0.0:
            raise ValueError("dt must be finite and non-negative")
        for spec in self.specs:
            self.sensors[spec.prim_path].update(
                float(dt), force_recompute=bool(force_recompute)
            )
        self._cached_data = self._collect_data()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for spec in self.specs:
            self.sensors[spec.prim_path].reset(env_ids)
        self._cached_data = None

    @staticmethod
    def _validated_single_body_field(
        snapshot: Any,
        field: str,
        *,
        expected_ndim: int,
        filter_axis: bool,
        sensor_path: str,
    ) -> np.ndarray:
        array = _to_numpy(getattr(snapshot, field, None))
        if array.ndim != expected_ndim:
            raise NonWheelContactLayoutError(
                f"{sensor_path}.{field} expected {expected_ndim} dimensions, got {array.shape}"
            )
        if array.shape[1] != 1:
            raise NonWheelContactLayoutError(
                f"{sensor_path}.{field} resolved {array.shape[1]} bodies; exactly one is required"
            )
        if array.shape[-1] != 3:
            raise NonWheelContactLayoutError(
                f"{sensor_path}.{field} trailing vector size is {array.shape[-1]}, expected 3"
            )
        if filter_axis and array.shape[2] != 1:
            raise NonWheelContactLayoutError(
                f"{sensor_path}.{field} resolved {array.shape[2]} filters; exactly obstacle-only is required"
            )
        return array

    def _collect_data(self) -> NonWheelObstacleContactBankData:
        fields: dict[str, list[np.ndarray]] = {
            "net_forces_w": [],
            "force_matrix_w": [],
            "contact_pos_w": [],
            "friction_forces_w": [],
        }
        env_count: int | None = None
        for spec in self.specs:
            snapshot = self.sensors[spec.prim_path].data
            arrays = {
                "net_forces_w": self._validated_single_body_field(
                    snapshot,
                    "net_forces_w",
                    expected_ndim=3,
                    filter_axis=False,
                    sensor_path=spec.prim_path,
                ),
                "force_matrix_w": self._validated_single_body_field(
                    snapshot,
                    "force_matrix_w",
                    expected_ndim=4,
                    filter_axis=True,
                    sensor_path=spec.prim_path,
                ),
                "contact_pos_w": self._validated_single_body_field(
                    snapshot,
                    "contact_pos_w",
                    expected_ndim=4,
                    filter_axis=True,
                    sensor_path=spec.prim_path,
                ),
                "friction_forces_w": self._validated_single_body_field(
                    snapshot,
                    "friction_forces_w",
                    expected_ndim=4,
                    filter_axis=True,
                    sensor_path=spec.prim_path,
                ),
            }
            current_env_count = int(arrays["force_matrix_w"].shape[0])
            if current_env_count < 1:
                raise NonWheelContactLayoutError(
                    f"{spec.prim_path} has no sensor environments"
                )
            if env_count is None:
                env_count = current_env_count
            elif current_env_count != env_count:
                raise NonWheelContactLayoutError(
                    f"{spec.prim_path} environment count differs from preceding sensors"
                )
            for field, array in arrays.items():
                if int(array.shape[0]) != env_count:
                    raise NonWheelContactLayoutError(
                        f"{spec.prim_path}.{field} environment count is inconsistent"
                    )
                fields[field].append(array)

        combined = {
            field: np.concatenate(arrays, axis=1) for field, arrays in fields.items()
        }
        # Force tensors must always be finite, including the all-zero no-contact
        # case.  A NaN/Inf is unknown evidence and therefore a hard failure.
        for field in ("net_forces_w", "force_matrix_w", "friction_forces_w"):
            if not bool(np.isfinite(combined[field]).all()):
                raise NonWheelContactEvidenceError(
                    f"{field} contains non-finite values; collision status is unknown"
                )
        return NonWheelObstacleContactBankData(
            net_forces_w=combined["net_forces_w"],
            force_matrix_w=combined["force_matrix_w"],
            contact_pos_w=combined["contact_pos_w"],
            friction_forces_w=combined["friction_forces_w"],
        )

    def observations(
        self,
        *,
        env_id: int = 0,
        force_threshold_n: float | None = None,
    ) -> list[NonWheelObstacleContactObservation]:
        """Decode every body; malformed or non-finite evidence raises."""

        data = self.data
        if not 0 <= int(env_id) < int(data.force_matrix_w.shape[0]):
            raise IndexError(
                f"env_id={env_id} outside [0, {data.force_matrix_w.shape[0]})"
            )
        threshold = self.force_threshold_n if force_threshold_n is None else float(force_threshold_n)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("force_threshold_n must be finite and non-negative")

        rows: list[NonWheelObstacleContactObservation] = []
        for body_index, spec in enumerate(self.specs):
            normal = np.asarray(
                data.force_matrix_w[int(env_id), body_index, 0, :], dtype=float
            )
            friction = np.asarray(
                data.friction_forces_w[int(env_id), body_index, 0, :], dtype=float
            )
            point = np.asarray(
                data.contact_pos_w[int(env_id), body_index, 0, :], dtype=float
            )
            if not bool(np.isfinite(normal).all() and np.isfinite(friction).all()):
                raise NonWheelContactEvidenceError(
                    f"non-finite obstacle force for {spec.prim_path}"
                )
            normal_norm = float(np.linalg.norm(normal))
            friction_norm = float(np.linalg.norm(friction))
            total = normal + friction
            total_norm = float(np.linalg.norm(total))
            if not all(math.isfinite(value) for value in (normal_norm, friction_norm, total_norm)):
                raise NonWheelContactEvidenceError(
                    f"non-finite derived obstacle force for {spec.prim_path}"
                )
            # Match IsaacLab's own contact-time predicate (strictly greater
            # than force_threshold).  In particular, threshold=0 must not mark
            # an all-zero force row as a collision.
            active = bool(normal_norm > threshold)
            point_finite = bool(np.isfinite(point).all())
            point_all_nan = bool(np.isnan(point).all())
            if active and not point_finite:
                raise NonWheelContactEvidenceError(
                    f"active obstacle contact has no finite point for {spec.prim_path}"
                )
            if not point_finite and not point_all_nan:
                raise NonWheelContactEvidenceError(
                    f"partially non-finite contact point for {spec.prim_path}"
                )
            rows.append(
                NonWheelObstacleContactObservation(
                    body_name=spec.body_name,
                    body_prim_path=spec.prim_path,
                    other_prim_path=self.obstacle_prim_path,
                    filter_index=0,
                    active=active,
                    normal_force_w=_vec3(normal),
                    normal_force_n=normal_norm,
                    friction_force_w=_vec3(friction),
                    friction_force_n=friction_norm,
                    total_force_w=_vec3(total),
                    total_force_n=total_norm,
                    contact_point_w=(
                        _vec3(point) if point_finite else (float("nan"),) * 3
                    ),
                    force_valid=True,
                    contact_point_valid=point_finite,
                )
            )
        return rows


def create_nonwheel_obstacle_contact_sensor_bank(
    *,
    stage: Any | None = None,
    robot_prim_path: str = ROBOT_PRIM_PATH,
    obstacle_prim_path: str = OBSTACLE_PRIM_PATH,
    wheel_body_names: Sequence[str] = tuple(sorted(WHEEL_BODY_NAMES)),
    history_length: int = 3,
    force_threshold_n: float = 1.0,
    max_contact_data_count_per_prim: int = 16,
    sensor_cls: type[Any] | None = None,
    sensor_cfg_cls: type[Any] | None = None,
    rigid_body_api_cls: type[Any] | None = None,
    prim_range_factory: Callable[[Any], Iterable[Any]] | None = None,
) -> NonWheelObstacleContactSensorBank:
    """Create the pre-reset sensor bank from live USD rigid-body discovery."""

    specs = discover_nonwheel_rigid_body_specs(
        stage=stage,
        robot_prim_path=robot_prim_path,
        obstacle_prim_path=obstacle_prim_path,
        wheel_body_names=wheel_body_names,
        rigid_body_api_cls=rigid_body_api_cls,
        prim_range_factory=prim_range_factory,
    )
    if sensor_cls is None or sensor_cfg_cls is None:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore

        sensor_cls = ContactSensor if sensor_cls is None else sensor_cls
        sensor_cfg_cls = ContactSensorCfg if sensor_cfg_cls is None else sensor_cfg_cls

    sensors: dict[str, Any] = {}
    for spec in specs:
        kwargs = nonwheel_contact_sensor_config_kwargs(
            spec,
            obstacle_prim_path=obstacle_prim_path,
            history_length=history_length,
            force_threshold_n=force_threshold_n,
            max_contact_data_count_per_prim=max_contact_data_count_per_prim,
        )
        sensors[spec.prim_path] = sensor_cls(sensor_cfg_cls(**kwargs))
    return NonWheelObstacleContactSensorBank(
        sensors,
        specs,
        obstacle_prim_path=obstacle_prim_path,
        force_threshold_n=force_threshold_n,
    )


def create_nonwheel_obstacle_contact_sensor_bank_result(
    **kwargs: Any,
) -> tuple[NonWheelObstacleContactSensorBank | None, str]:
    """Non-throwing ``(sensor, error)`` result for ``SimSceneConfig`` factories."""

    try:
        return create_nonwheel_obstacle_contact_sensor_bank(**kwargs), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def make_nonwheel_obstacle_contact_sensor_factory(
    **kwargs: Any,
) -> Callable[[], tuple[NonWheelObstacleContactSensorBank | None, str]]:
    return functools.partial(create_nonwheel_obstacle_contact_sensor_bank_result, **kwargs)


def nonwheel_obstacle_contact_rows(
    sensor: Any,
    *,
    env_id: int = 0,
    force_threshold_n: float | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-ready force evidence, rejecting any unverified sensor type."""

    bank = getattr(sensor, "nonwheel_bank", sensor)
    if not bool(getattr(bank, "is_nonwheel_obstacle_contact_bank", False)):
        raise TypeError("sensor does not contain a NonWheelObstacleContactSensorBank")
    return [
        row.as_dict()
        for row in bank.observations(
            env_id=env_id, force_threshold_n=force_threshold_n
        )
    ]


class WheelAndNonWheelContactSensorBank:
    """One scene sensor facade preserving the existing filtered-wheel API."""

    is_filtered_wheel_contact_bank = True
    is_nonwheel_obstacle_contact_bank = True

    def __init__(self, wheel_bank: Any, nonwheel_bank: NonWheelObstacleContactSensorBank) -> None:
        if not bool(getattr(wheel_bank, "is_filtered_wheel_contact_bank", False)):
            raise TypeError("wheel_bank is not a filtered wheel contact bank")
        if not bool(getattr(nonwheel_bank, "is_nonwheel_obstacle_contact_bank", False)):
            raise TypeError("nonwheel_bank is not a non-wheel obstacle contact bank")
        self.wheel_bank = wheel_bank
        self.nonwheel_bank = nonwheel_bank

    @property
    def body_names(self) -> list[str]:
        # Existing generic and FSM wheel telemetry must continue to see only the
        # four wheel bodies.  Non-wheel rows have their own explicit accessor.
        return list(self.wheel_bank.body_names)

    @property
    def data(self) -> Any:
        return self.wheel_bank.data

    @property
    def cfg(self) -> Any:
        return self.wheel_bank.cfg

    def filtered_observations(self, **kwargs: Any) -> Any:
        return self.wheel_bank.filtered_observations(**kwargs)

    def nonwheel_obstacle_observations(self, **kwargs: Any) -> Any:
        return self.nonwheel_bank.observations(**kwargs)

    def observations(self, **kwargs: Any) -> Any:
        """Alias preserving the non-wheel bank observation contract."""

        return self.nonwheel_bank.observations(**kwargs)

    def update(self, dt: float, force_recompute: bool = False) -> None:
        self.wheel_bank.update(float(dt), force_recompute=bool(force_recompute))
        self.nonwheel_bank.update(float(dt), force_recompute=bool(force_recompute))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.wheel_bank.reset(env_ids)
        self.nonwheel_bank.reset(env_ids)


def create_wheel_and_nonwheel_contact_sensor_bank_result(
    *,
    wheel_factory: Callable[[], tuple[Any | None, str]],
    **nonwheel_kwargs: Any,
) -> tuple[WheelAndNonWheelContactSensorBank | None, str]:
    """Compose the existing wheel factory with live non-wheel discovery."""

    try:
        wheel_bank, wheel_error = wheel_factory()
        if wheel_bank is None or wheel_error:
            return None, f"wheel contact bank unavailable: {wheel_error or 'factory returned None'}"
        nonwheel_bank, nonwheel_error = create_nonwheel_obstacle_contact_sensor_bank_result(
            **nonwheel_kwargs
        )
        if nonwheel_bank is None or nonwheel_error:
            return None, (
                "non-wheel obstacle contact bank unavailable: "
                f"{nonwheel_error or 'factory returned None'}"
            )
        return WheelAndNonWheelContactSensorBank(wheel_bank, nonwheel_bank), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def make_wheel_and_nonwheel_contact_sensor_factory(
    *,
    wheel_factory: Callable[[], tuple[Any | None, str]],
    **nonwheel_kwargs: Any,
) -> Callable[[], tuple[WheelAndNonWheelContactSensorBank | None, str]]:
    """Return the zero-argument combined factory consumed by ``create_scene``."""

    return functools.partial(
        create_wheel_and_nonwheel_contact_sensor_bank_result,
        wheel_factory=wheel_factory,
        **nonwheel_kwargs,
    )


def configure_scene_for_wheel_and_nonwheel_contacts(
    scene_config: Any,
    *,
    wheel_factory: Callable[[], tuple[Any | None, str]],
    **nonwheel_kwargs: Any,
) -> Any:
    """Install one pre-reset factory that creates both strict contact banks."""

    scene_config.telemetry_contact_sensors_enabled = True
    scene_config.contact_sensor_factory = make_wheel_and_nonwheel_contact_sensor_factory(
        wheel_factory=wheel_factory,
        **nonwheel_kwargs,
    )
    return scene_config


__all__ = [
    "NONWHEEL_FORCE_SOURCE",
    "OBSTACLE_PRIM_PATH",
    "ROBOT_PRIM_PATH",
    "WHEEL_BODY_NAMES",
    "NonWheelContactDiscoveryError",
    "NonWheelContactEvidenceError",
    "NonWheelContactLayoutError",
    "NonWheelObstacleContactBankData",
    "NonWheelObstacleContactObservation",
    "NonWheelObstacleContactSensorBank",
    "NonWheelRigidBodySpec",
    "WheelAndNonWheelContactSensorBank",
    "configure_scene_for_wheel_and_nonwheel_contacts",
    "create_nonwheel_obstacle_contact_sensor_bank",
    "create_nonwheel_obstacle_contact_sensor_bank_result",
    "create_wheel_and_nonwheel_contact_sensor_bank_result",
    "discover_nonwheel_rigid_body_specs",
    "make_nonwheel_obstacle_contact_sensor_factory",
    "make_wheel_and_nonwheel_contact_sensor_factory",
    "nonwheel_contact_sensor_config_kwargs",
    "nonwheel_obstacle_contact_rows",
]
