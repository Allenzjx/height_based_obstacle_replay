from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from fsm_50mm_recording_derived_v3.fsm50_centroidal_support import (
    ANGULAR_MOMENTUM_FIELD_NAMES,
    CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA,
    COM_FIELD_NAMES,
    CentroidalAngularMomentumRateMeasurement,
    CentroidalSupportEvidence,
    EvidenceStatus,
    LEG_TO_WHEEL_BODY,
    LEGS,
    RAW_CONTACT_FORCE_AGGREGATE_ATOL_N,
    RAW_CONTACT_NORMAL_UNIT_TOLERANCE,
    RAW_CONTACT_POINT_AGGREGATE_ATOL_M,
    ContactWrenchFeasibility,
    SupportModel,
    SupportThresholds,
    TransferMethod,
    TransferSample,
    WheelContactMeasurement,
    assess_contact_wrench_feasibility,
    assess_primary_diagonal_support,
    assess_support_region,
    classify_com_transfer,
    measure_centroidal_angular_momentum_rate,
    measure_isaac_centroidal_angular_momentum_rate,
    measure_isaac_whole_body_com,
    measure_isaac_wheel_contacts,
    measure_whole_body_com,
    validate_wheel_contact_frame,
)


DT = 0.1
THRESHOLDS = SupportThresholds(
    minimum_normal_force_n=2.0,
    minimum_dwell_s=0.08,
    maximum_slip_speed_m_s=0.05,
    maximum_contact_drift_speed_m_s=0.05,
    coplanar_height_tolerance_m=0.005,
)


def _com(
    tick: int,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = 10.0,
):
    return measure_whole_body_com(
        body_names=("base",),
        body_masses_kg=(mass,),
        body_com_positions_w_m=(position,),
        body_com_velocities_w_m_s=(velocity,),
        body_com_accelerations_w_m_s2=(acceleration,),
        physics_tick=tick,
        sim_time_s=tick * DT,
        physics_dt_s=DT,
        field_physics_ticks={name: tick for name in COM_FIELD_NAMES},
        expected_body_names=("base",),
        source="test",
    )


def _angular_rate(
    tick: int,
    value: tuple[float, float, float],
    *,
    body_names: tuple[str, ...] = ("base",),
) -> CentroidalAngularMomentumRateMeasurement:
    return CentroidalAngularMomentumRateMeasurement(
        available=True,
        physics_tick=tick,
        sim_time_s=tick * DT,
        body_names=body_names,
        angular_momentum_rate_w_nm=value,
        source="test.independent_centroidal_ldot",
    )


def _proven_wrench(tick: int) -> ContactWrenchFeasibility:
    return ContactWrenchFeasibility(
        status=EvidenceStatus.PROVEN,
        proven_feasible=True,
        physics_tick=tick,
        witness_kind="test.independent_wrench",
        multiple_contact_heights=False,
        dynamic=True,
        angular_momentum_rate_w_nm=(0.0, 0.0, 0.0),
        angular_momentum_rate_source="test.independent_centroidal_ldot",
        force_residual_w_n=(0.0, 0.0, 0.0),
        force_residual_norm_n=0.0,
        moment_residual_w_nm=(0.0, 0.0, 0.0),
        moment_residual_norm_nm=0.0,
        maximum_friction_utilization=0.0,
        reasons=(),
    )


def _wheel(
    leg: str,
    tick: int,
    *,
    point: tuple[float, float, float] | None,
    normal_load: float,
    friction: tuple[float, float, float] = (0.0, 0.0, 0.0),
    active: bool = True,
    surface: str = "GROUND",
    height: float = 0.0,
    dwell: float | None = 0.2,
    dwell_verified: bool = True,
    slip: float | None = 0.0,
    drift: float | None = 0.0,
    mu: float = 0.8,
    patch: float | None = 0.1,
    moment: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    moment_model: str = "MEASURED",
) -> WheelContactMeasurement:
    return WheelContactMeasurement(
        leg=leg,
        wheel_body_name=LEG_TO_WHEEL_BODY[leg],
        physics_tick=tick,
        sim_time_s=tick * DT,
        surface_kind=surface,
        surface_height_m=height if surface in {"GROUND", "OBSTACLE_TOP"} else None,
        surface_normal_w=(0.0, 0.0, 1.0) if surface in {"GROUND", "OBSTACLE_TOP"} else None,
        active=active,
        contact_point_w_m=point,
        normal_force_w_n=(0.0, 0.0, normal_load),
        friction_force_w_n=friction,
        contact_moment_w_nm=moment,
        contact_moment_model=moment_model if moment is not None else "",
        dwell_s=dwell,
        surface_dwell_verified=dwell_verified,
        slip_speed_m_s=slip,
        contact_drift_speed_m_s=drift,
        friction_coefficient=mu,
        finite_patch_radius_m=patch,
        source="test",
    )


def _inactive(leg: str, tick: int) -> WheelContactMeasurement:
    return _wheel(
        leg,
        tick,
        point=None,
        normal_load=0.0,
        active=False,
        surface="AIR",
        dwell=None,
        dwell_verified=False,
        slip=None,
        drift=None,
        patch=None,
        moment=None,
    )


def _frame(tick: int, rows: list[WheelContactMeasurement]):
    return validate_wheel_contact_frame(
        rows,
        physics_tick=tick,
        sim_time_s=tick * DT,
        physics_dt_s=DT,
        thresholds=THRESHOLDS,
    )


class _FakeRawContactView:
    def __init__(
        self,
        *,
        contact_forces: tuple[tuple[float, float, float], ...],
        contact_points: tuple[tuple[float, float, float], ...],
        friction_forces: tuple[tuple[float, float, float], ...] = (),
        friction_points: tuple[tuple[float, float, float], ...] = (),
        capacity: int | None = None,
    ) -> None:
        if len(contact_forces) != len(contact_points):
            raise ValueError("test contact forces/points length mismatch")
        if len(friction_forces) != len(friction_points):
            raise ValueError("test friction forces/points length mismatch")
        used = max(len(contact_forces), len(friction_forces))
        self.max_contact_data_count = capacity if capacity is not None else max(used + 2, 4)
        self.sensor_count = 1
        self.filter_count = 2
        raw_capacity = self.max_contact_data_count
        self.contact_force_magnitudes = np.full((raw_capacity, 1), np.nan, dtype=float)
        self.contact_points = np.full((raw_capacity, 3), np.nan, dtype=float)
        self.contact_normals = np.full((raw_capacity, 3), np.nan, dtype=float)
        self.separation_distances = np.full((raw_capacity, 1), np.nan, dtype=float)
        for index, (force, point) in enumerate(zip(contact_forces, contact_points)):
            force_array = np.asarray(force, dtype=float)
            magnitude = float(np.linalg.norm(force_array))
            self.contact_force_magnitudes[index, 0] = magnitude
            self.contact_points[index] = point
            self.contact_normals[index] = (
                force_array / magnitude if magnitude else np.asarray((0.0, 0.0, 1.0))
            )
            self.separation_distances[index, 0] = 0.0
        self.contact_counts = np.asarray([[len(contact_forces), 0]], dtype=np.int64)
        self.contact_starts = np.asarray([[0, len(contact_forces)]], dtype=np.int64)

        self.friction_forces = np.full((raw_capacity, 3), np.nan, dtype=float)
        self.friction_points = np.full((raw_capacity, 3), np.nan, dtype=float)
        for index, (force, point) in enumerate(zip(friction_forces, friction_points)):
            self.friction_forces[index] = force
            self.friction_points[index] = point
        self.friction_counts = np.asarray([[len(friction_forces), 0]], dtype=np.int64)
        self.friction_starts = np.asarray([[0, len(friction_forces)]], dtype=np.int64)
        self.check_result = True
        self.contact_payload_override = None
        self.friction_payload_override = None
        self.after_friction_read = None
        self.contact_dt = None
        self.friction_dt = None

    def check(self) -> bool:
        return self.check_result

    def get_contact_data(self, *, dt: float):
        self.contact_dt = dt
        if self.contact_payload_override is not None:
            return self.contact_payload_override
        return (
            self.contact_force_magnitudes,
            self.contact_points,
            self.contact_normals,
            self.separation_distances,
            self.contact_counts,
            self.contact_starts,
        )

    def get_friction_data(self, *, dt: float):
        self.friction_dt = dt
        if self.after_friction_read is not None:
            self.after_friction_read()
        if self.friction_payload_override is not None:
            return self.friction_payload_override
        return (
            self.friction_forces,
            self.friction_points,
            self.friction_counts,
            self.friction_starts,
        )


def _lazy_contact_fixture(
    *,
    tick: int = 2,
    contact_forces_by_leg: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    contact_points_by_leg: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    friction_forces_by_leg: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    friction_points_by_leg: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    capacity_by_leg: dict[str, int] | None = None,
    total_mass_kg: float = 40.0 / 9.81,
):
    standard_points = {
        "FL": (-1.0, 1.0, 0.0),
        "FR": (1.0, 1.0, 0.0),
        "RL": (-1.0, -1.0, 0.0),
        "RR": (1.0, -1.0, 0.0),
    }
    if contact_forces_by_leg is None:
        contact_forces_by_leg = {
            leg: ((0.0, 0.0, 10.0),) for leg in LEGS
        }
    if contact_points_by_leg is None:
        contact_points_by_leg = {
            leg: (standard_points[leg],) for leg in LEGS
        }
    friction_forces_by_leg = friction_forces_by_leg or {leg: () for leg in LEGS}
    friction_points_by_leg = friction_points_by_leg or {leg: () for leg in LEGS}
    capacity_by_leg = capacity_by_leg or {}

    filter_surfaces = (("ground", "/Ground"), ("obstacle", "/Obstacle"))
    specs = tuple(
        SimpleNamespace(
            leg=leg,
            body_name=LEG_TO_WHEEL_BODY[leg],
            prim_path=f"/World/WLRRobot/{LEG_TO_WHEEL_BODY[leg]}",
        )
        for leg in LEGS
    )
    sensors = {}
    aggregate_points = {}
    for leg in LEGS:
        raw_forces = contact_forces_by_leg[leg]
        raw_points = contact_points_by_leg[leg]
        raw_friction = friction_forces_by_leg[leg]
        raw_friction_points = friction_points_by_leg[leg]
        view = _FakeRawContactView(
            contact_forces=raw_forces,
            contact_points=raw_points,
            friction_forces=raw_friction,
            friction_points=raw_friction_points,
            capacity=capacity_by_leg.get(leg),
        )
        force = np.zeros((1, 1, 2, 3), dtype=float)
        force[0, 0, 0] = np.sum(np.asarray(raw_forces, dtype=float), axis=0) if raw_forces else 0.0
        contact_pos = np.full((1, 1, 2, 3), np.nan, dtype=float)
        if raw_points:
            aggregate_points[leg] = tuple(
                float(value) for value in np.mean(np.asarray(raw_points, dtype=float), axis=0)
            )
            contact_pos[0, 0, 0] = aggregate_points[leg]
        else:
            aggregate_points[leg] = None
        friction = np.zeros((1, 1, 2, 3), dtype=float)
        friction[0, 0, 0] = (
            np.sum(np.asarray(raw_friction, dtype=float), axis=0)
            if raw_friction
            else 0.0
        )
        sensors[leg] = SimpleNamespace(
            _timestamp=np.asarray([tick * DT]),
            _timestamp_last_update=np.asarray([tick * DT]),
            cfg=SimpleNamespace(
                prim_path=next(spec.prim_path for spec in specs if spec.leg == leg),
                filter_prim_paths_expr=[path for _name, path in filter_surfaces],
            ),
            contact_physx_view=view,
            data=SimpleNamespace(
                force_matrix_w=force,
                contact_pos_w=contact_pos,
                friction_forces_w=friction,
            ),
        )
    bank = SimpleNamespace(
        specs=specs,
        sensors=sensors,
        filter_surfaces=filter_surfaces,
        force_threshold_n=1.0,
    )
    body_names = tuple(LEG_TO_WHEEL_BODY[leg] for leg in LEGS)
    body_positions = np.asarray(
        [[aggregate_points[leg] or (0.0, 0.0, 0.0) for leg in LEGS]],
        dtype=float,
    )
    robot_data = SimpleNamespace(
        _sim_timestamp=tick * DT,
        _body_com_pose_w=SimpleNamespace(timestamp=tick * DT),
        _body_com_vel_w=SimpleNamespace(timestamp=tick * DT),
        body_com_pos_w=body_positions,
        body_com_vel_w=np.zeros((1, 4, 6), dtype=float),
    )
    adapter = SimpleNamespace(
        sim_steps=tick,
        sim_time=tick * DT,
        sim=SimpleNamespace(get_physics_dt=lambda: DT),
        robot=SimpleNamespace(data=robot_data, body_names=body_names),
    )
    whole_body_com = measure_whole_body_com(
        body_names=body_names,
        body_masses_kg=tuple(total_mass_kg / len(LEGS) for _leg in LEGS),
        body_com_positions_w_m=tuple((0.0, 0.0, 1.0) for _leg in LEGS),
        body_com_velocities_w_m_s=tuple((0.0, 0.0, 0.0) for _leg in LEGS),
        body_com_accelerations_w_m_s2=tuple((0.0, 0.0, 0.0) for _leg in LEGS),
        physics_tick=tick,
        sim_time_s=tick * DT,
        physics_dt_s=DT,
        field_physics_ticks={name: tick for name in COM_FIELD_NAMES},
        expected_body_names=body_names,
        source="test.current_whole_body_com",
    )
    return adapter, bank, whole_body_com, aggregate_points


class WholeBodyCOMTests(unittest.TestCase):
    def test_mass_weighted_position_velocity_and_acceleration(self) -> None:
        result = measure_whole_body_com(
            body_names=("base", "wheel"),
            body_masses_kg=(3.0, 1.0),
            body_com_positions_w_m=((0.0, 0.0, 1.0), (4.0, 0.0, 1.0)),
            body_com_velocities_w_m_s=((0.0, 2.0, 0.0), (4.0, 2.0, 0.0)),
            body_com_accelerations_w_m_s2=((0.0, 0.0, 3.0), (4.0, 0.0, 3.0)),
            physics_tick=10,
            sim_time_s=1.0,
            physics_dt_s=0.1,
            field_physics_ticks={name: 10 for name in COM_FIELD_NAMES},
            expected_body_names=("base", "wheel"),
        )
        self.assertTrue(result.com_measurement_available)
        self.assertTrue(result.acceleration_available)
        self.assertEqual(4.0, result.total_mass_kg)
        self.assertEqual((1.0, 0.0, 1.0), result.position_w_m)
        self.assertEqual((1.0, 2.0, 0.0), result.velocity_w_m_s)
        self.assertEqual((1.0, 0.0, 3.0), result.acceleration_w_m_s2)

    def test_missing_stale_nonfinite_or_wrong_identity_fails_closed(self) -> None:
        base = dict(
            body_names=("base", "wheel"),
            body_masses_kg=(3.0, 1.0),
            body_com_positions_w_m=((0.0, 0.0, 1.0), (4.0, 0.0, 1.0)),
            body_com_velocities_w_m_s=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            body_com_accelerations_w_m_s2=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            physics_tick=10,
            sim_time_s=1.0,
            physics_dt_s=0.1,
            field_physics_ticks={name: 10 for name in COM_FIELD_NAMES},
            expected_body_names=("base", "wheel"),
        )
        cases = (
            {"body_com_accelerations_w_m_s2": None},
            {"body_masses_kg": (3.0, 0.0)},
            {"body_masses_kg": (3.0, float("nan"))},
            {"expected_body_names": ("wheel", "base")},
            {"body_names": ("base", "base")},
            {"field_physics_ticks": {**base["field_physics_ticks"], "velocities": 9}},
            {"sim_time_s": 1.1},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                result = measure_whole_body_com(**{**base, **changes})
                self.assertFalse(result.com_measurement_available)
                self.assertIsNone(result.position_w_m)
                self.assertTrue(result.errors)

    def test_lazy_isaac_adapter_validates_all_names_masses_and_buffer_ticks(self) -> None:
        timestamp = SimpleNamespace(timestamp=0.2)
        data = SimpleNamespace(
            body_names=["base", "wheel"],
            _sim_timestamp=0.2,
            _body_com_pose_w=timestamp,
            _body_com_vel_w=timestamp,
            _body_com_acc_w=timestamp,
            body_com_pos_w=np.asarray([[[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]]),
            body_com_lin_vel_w=np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            body_com_lin_acc_w=np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            root_pos_w=np.asarray([[999.0, 999.0, 999.0]]),
        )
        view = SimpleNamespace(
            shared_metatype=SimpleNamespace(link_names=["base", "wheel"]),
            get_masses=lambda: np.asarray([[3.0, 1.0]]),
        )
        robot = SimpleNamespace(
            data=data,
            body_names=["base", "wheel"],
            root_physx_view=view,
            num_bodies=2,
            num_instances=1,
        )
        adapter = SimpleNamespace(
            robot=robot,
            sim_steps=2,
            sim_time=0.2,
            sim=SimpleNamespace(get_physics_dt=lambda: 0.1),
        )
        result = measure_isaac_whole_body_com(adapter, expected_body_names=("base", "wheel"))
        self.assertTrue(result.available)
        self.assertEqual((0.5, 0.0, 1.0), result.position_w_m)
        data._body_com_acc_w = SimpleNamespace(timestamp=0.1)
        stale = measure_isaac_whole_body_com(adapter, expected_body_names=("base", "wheel"))
        self.assertFalse(stale.available)
        self.assertIsNone(stale.position_w_m)
        self.assertNotEqual((999.0, 999.0, 999.0), stale.position_w_m)


class CentroidalAngularMomentumRateTests(unittest.TestCase):
    def test_rotational_and_orbital_terms_use_world_rotated_inertia(self) -> None:
        com = measure_whole_body_com(
            body_names=("left", "right"),
            body_masses_kg=(1.0, 1.0),
            body_com_positions_w_m=((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            body_com_velocities_w_m_s=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            body_com_accelerations_w_m_s2=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            physics_tick=1,
            sim_time_s=DT,
            physics_dt_s=DT,
            field_physics_ticks={name: 1 for name in COM_FIELD_NAMES},
        )
        half = math.sqrt(0.5)
        result = measure_centroidal_angular_momentum_rate(
            whole_body_com=com,
            body_names=("left", "right"),
            body_masses_kg=(1.0, 1.0),
            body_com_positions_w_m=((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            body_com_linear_accelerations_w_m_s2=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            body_angular_velocities_w_rad_s=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            body_angular_accelerations_w_rad_s2=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            body_com_principal_quaternions_wxyz=((half, 0.0, 0.0, half), (1.0, 0.0, 0.0, 0.0)),
            body_inertias_com_principal_kg_m2=(
                ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ),
            physics_tick=1,
            sim_time_s=DT,
            physics_dt_s=DT,
            field_physics_ticks={name: 1 for name in ANGULAR_MOMENTUM_FIELD_NAMES},
        )
        self.assertTrue(result.available)
        # World-rotated first-body I*alpha=(2,0,0); orbital terms sum to (0,0,2).
        self.assertTrue(np.allclose((2.0, 0.0, 2.0), result.angular_momentum_rate_w_nm))

    def test_stale_or_nonphysical_inertia_fails_closed(self) -> None:
        base = dict(
            whole_body_com=_com(1),
            body_names=("base",),
            body_masses_kg=(10.0,),
            body_com_positions_w_m=((0.0, 0.0, 1.0),),
            body_com_linear_accelerations_w_m_s2=((0.0, 0.0, 0.0),),
            body_angular_velocities_w_rad_s=((0.0, 0.0, 0.0),),
            body_angular_accelerations_w_rad_s2=((0.0, 0.0, 0.0),),
            body_com_principal_quaternions_wxyz=((1.0, 0.0, 0.0, 0.0),),
            body_inertias_com_principal_kg_m2=(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),),
            physics_tick=1,
            sim_time_s=DT,
            physics_dt_s=DT,
            field_physics_ticks={name: 1 for name in ANGULAR_MOMENTUM_FIELD_NAMES},
        )
        stale_ticks = {**base["field_physics_ticks"], "inertias": 0}
        self.assertFalse(
            measure_centroidal_angular_momentum_rate(
                **{**base, "field_physics_ticks": stale_ticks}
            ).available
        )
        bad_inertia = (((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),)
        self.assertFalse(
            measure_centroidal_angular_momentum_rate(
                **{**base, "body_inertias_com_principal_kg_m2": bad_inertia}
            ).available
        )

    def test_lazy_adapter_uses_current_com_principal_orientation(self) -> None:
        half = math.sqrt(0.5)
        data = SimpleNamespace(
            body_names=("base",),
            _sim_timestamp=DT,
            _body_com_pose_w=SimpleNamespace(timestamp=DT),
            _body_com_vel_w=SimpleNamespace(timestamp=DT),
            _body_com_acc_w=SimpleNamespace(timestamp=DT),
            body_com_pose_w=np.asarray(
                [[[0.0, 0.0, 1.0, half, 0.0, 0.0, half]]], dtype=float
            ),
            body_com_vel_w=np.zeros((1, 1, 6), dtype=float),
            body_com_acc_w=np.asarray(
                [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]], dtype=float
            ),
        )
        root_view = SimpleNamespace(
            shared_metatype=SimpleNamespace(link_names=("base",)),
            get_masses=lambda: np.asarray([[10.0]], dtype=float),
            get_inertias=lambda: np.asarray(
                [[[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]]],
                dtype=float,
            ),
        )
        adapter = SimpleNamespace(
            robot=SimpleNamespace(
                data=data,
                body_names=("base",),
                root_physx_view=root_view,
                num_instances=1,
            ),
            sim_steps=1,
            sim_time=DT,
            sim=SimpleNamespace(get_physics_dt=lambda: DT),
        )
        result = measure_isaac_centroidal_angular_momentum_rate(
            adapter, _com(1), expected_body_names=("base",)
        )
        self.assertTrue(result.available, result.errors)
        self.assertTrue(
            np.allclose((2.0, 0.0, 0.0), result.angular_momentum_rate_w_nm)
        )


class ContactContractTests(unittest.TestCase):
    def test_exact_current_tick_four_wheel_contract_and_qualifiers(self) -> None:
        rows = [
            _wheel("FL", 1, point=(-1.0, 1.0, 0.0), normal_load=10.0),
            _wheel("FR", 1, point=(1.0, 1.0, 0.0), normal_load=10.0),
            _wheel("RL", 1, point=(-1.0, -1.0, 0.0), normal_load=10.0),
            _wheel("RR", 1, point=(1.0, -1.0, 0.0), normal_load=10.0),
        ]
        frame = _frame(1, rows)
        self.assertTrue(frame.available)
        self.assertEqual(LEGS, tuple(row.leg for row in frame.contacts))
        self.assertTrue(all(row.support_qualified for row in frame.contacts))
        self.assertTrue(all(row.friction_utilization == 0.0 for row in frame.contacts))

    def test_stale_missing_dwell_nonfinite_slip_and_inactive_force_fail_closed(self) -> None:
        base = _wheel("FL", 1, point=(0.0, 0.0, 0.0), normal_load=10.0)
        cases = (
            replace(base, physics_tick=0),
            replace(base, dwell_s=None, surface_dwell_verified=False),
            replace(base, slip_speed_m_s=float("nan")),
            replace(base, active=False),
        )
        for bad in cases:
            rows = [bad, _inactive("FR", 1), _inactive("RL", 1), _inactive("RR", 1)]
            frame = _frame(1, rows)
            self.assertFalse(frame.available)
            self.assertTrue(frame.errors)

    def test_large_finite_slip_or_friction_utilization_is_valid_but_not_support(self) -> None:
        slippery = _wheel(
            "FL",
            1,
            point=(0.0, 0.0, 0.0),
            normal_load=10.0,
            friction=(9.0, 0.0, 0.0),
            slip=0.2,
            mu=0.8,
        )
        frame = _frame(1, [slippery, _inactive("FR", 1), _inactive("RL", 1), _inactive("RR", 1)])
        self.assertTrue(frame.available)
        self.assertFalse(frame.by_leg()["FL"].support_qualified)
        self.assertGreater(frame.by_leg()["FL"].friction_utilization, 1.0)

    def test_front_face_or_wrong_height_cannot_qualify_as_obstacle_top(self) -> None:
        cases = (
            _wheel(
                "FL",
                1,
                point=(0.0, 0.0, 0.0),
                normal_load=10.0,
                surface="OBSTACLE_TOP",
                height=0.05,
            ),
            replace(
                _wheel(
                    "FL",
                    1,
                    point=(0.0, 0.0, 0.05),
                    normal_load=10.0,
                    surface="OBSTACLE_TOP",
                    height=0.05,
                ),
                normal_force_w_n=(10.0, 0.0, 0.0),
            ),
            _wheel(
                "FL",
                1,
                point=(0.0, 0.0, 0.05),
                normal_load=10.0,
                surface="FRONT_FACE",
                height=0.05,
            ),
        )
        for row in cases:
            with self.subTest(surface=row.surface_kind):
                frame = _frame(
                    1,
                    [row, _inactive("FR", 1), _inactive("RL", 1), _inactive("RR", 1)],
                )
                self.assertFalse(frame.available)
                self.assertFalse(frame.by_leg()["FL"].support_qualified)

    def test_surface_transition_does_not_inherit_aggregate_dwell(self) -> None:
        transitioned = _wheel(
            "FL",
            1,
            point=(0.0, 0.0, 0.05),
            normal_load=10.0,
            surface="OBSTACLE_TOP",
            height=0.05,
            dwell=1.0,
            dwell_verified=False,
        )
        frame = _frame(
            1,
            [transitioned, _inactive("FR", 1), _inactive("RL", 1), _inactive("RR", 1)],
        )
        self.assertTrue(frame.available)
        self.assertFalse(frame.by_leg()["FL"].support_qualified)


class SupportGeometryTests(unittest.TestCase):
    def _square(self, tick: int = 1, *, fourth_height: float = 0.0):
        return _frame(
            tick,
            [
                _wheel("FL", tick, point=(-1.0, 1.0, 0.0), normal_load=10.0),
                _wheel("FR", tick, point=(1.0, 1.0, 0.0), normal_load=10.0),
                _wheel("RL", tick, point=(-1.0, -1.0, 0.0), normal_load=10.0),
                _wheel(
                    "RR",
                    tick,
                    point=(1.0, -1.0, fourth_height),
                    normal_load=10.0,
                    surface="OBSTACLE_TOP" if fourth_height else "GROUND",
                    height=fourth_height,
                ),
            ],
        )

    def test_three_or_four_point_coplanar_hull_has_signed_margin(self) -> None:
        frame = self._square()
        inside = assess_support_region(_com(1), frame, thresholds=THRESHOLDS)
        self.assertEqual(EvidenceStatus.PROVEN, inside.status)
        self.assertEqual(SupportModel.STRICT_COPLANAR_CONVEX_HULL, inside.model)
        self.assertAlmostEqual(1.0, inside.signed_margin_m)
        outside = assess_support_region(_com(1, position=(2.0, 0.0, 1.0)), frame, thresholds=THRESHOLDS)
        self.assertAlmostEqual(-1.0, outside.signed_margin_m)

    def test_multi_height_refuses_planar_polygon(self) -> None:
        result = assess_support_region(_com(1), self._square(fourth_height=0.05), thresholds=THRESHOLDS)
        self.assertEqual(EvidenceStatus.NOT_PROVEN, result.status)
        self.assertEqual(SupportModel.MULTI_HEIGHT_OR_DYNAMIC_WRENCH_REQUIRED, result.model)
        self.assertEqual((), result.hull_xy_m)

    def test_diagonal_is_line_and_corridor_is_explicit_approximation(self) -> None:
        frame = _frame(
            1,
            [
                _wheel("FL", 1, point=(-1.0, -1.0, 0.0), normal_load=20.0, patch=0.1),
                _inactive("FR", 1),
                _inactive("RL", 1),
                _wheel("RR", 1, point=(1.0, 1.0, 0.0), normal_load=20.0, patch=0.1),
            ],
        )
        center = assess_support_region(_com(1), frame, thresholds=THRESHOLDS)
        self.assertEqual(SupportModel.DIAGONAL_LINE_SEGMENT, center.model)
        self.assertEqual("FL_RR", center.diagonal)
        self.assertAlmostEqual(0.5, center.line_parameter)
        self.assertAlmostEqual(0.0, center.line_distance_m)
        self.assertTrue(center.finite_patch_approximation)
        self.assertAlmostEqual(0.1, center.corridor_signed_margin_m)
        self.assertEqual((), center.hull_xy_m)
        outside = assess_support_region(_com(1, position=(0.0, 0.25, 1.0)), frame, thresholds=THRESHOLDS)
        self.assertLess(outside.corridor_signed_margin_m, 0.0)


class WrenchAndDiagonalTests(unittest.TestCase):
    def test_measured_static_wrench_is_a_conservative_sufficient_certificate(self) -> None:
        rows = [
            _wheel("FL", 1, point=(-1.0, 1.0, 0.0), normal_load=24.525),
            _wheel("FR", 1, point=(1.0, 1.0, 0.0), normal_load=24.525),
            _wheel("RL", 1, point=(-1.0, -1.0, 0.0), normal_load=24.525),
            _wheel("RR", 1, point=(1.0, -1.0, 0.0), normal_load=24.525),
        ]
        result = assess_contact_wrench_feasibility(
            _com(1),
            _frame(1, rows),
            angular_momentum_rate=_angular_rate(1, (0.0, 0.0, 0.0)),
            force_residual_tolerance_n=1.0e-9,
            moment_residual_tolerance_nm=1.0e-9,
        )
        self.assertEqual(EvidenceStatus.PROVEN, result.status)
        self.assertTrue(result.proven_feasible)
        self.assertAlmostEqual(0.0, result.force_residual_norm_n)
        self.assertAlmostEqual(0.0, result.moment_residual_norm_nm)
        self.assertFalse(result.dynamic)
        self.assertFalse(result.multiple_contact_heights)

    def test_dynamic_multi_height_wrench_and_friction_are_auditable(self) -> None:
        rows = [
            _wheel("FL", 1, point=(-1.0, 1.0, 0.0), normal_load=24.525, friction=(2.5, 0.0, 0.0)),
            _wheel("FR", 1, point=(1.0, 1.0, 0.0), normal_load=24.525, friction=(2.5, 0.0, 0.0)),
            _wheel(
                "RL", 1, point=(-1.0, -1.0, 0.05), normal_load=24.525, friction=(2.5, 0.0, 0.0), surface="OBSTACLE_TOP", height=0.05
            ),
            _wheel(
                "RR", 1, point=(1.0, -1.0, 0.05), normal_load=24.525, friction=(2.5, 0.0, 0.0), surface="OBSTACLE_TOP", height=0.05
            ),
        ]
        result = assess_contact_wrench_feasibility(
            _com(1, acceleration=(1.0, 0.0, 0.0)),
            _frame(1, rows),
            angular_momentum_rate=_angular_rate(1, (0.0, -9.75, 0.0)),
            force_residual_tolerance_n=1.0e-9,
            moment_residual_tolerance_nm=1.0e-9,
        )
        self.assertTrue(result.proven_feasible)
        self.assertTrue(result.dynamic)
        self.assertTrue(result.multiple_contact_heights)
        self.assertLess(result.maximum_friction_utilization, 1.0)

    def test_missing_moment_or_bad_friction_never_claims_feasible(self) -> None:
        balanced = [
            _wheel("FL", 1, point=(-1.0, 1.0, 0.0), normal_load=24.525),
            _wheel("FR", 1, point=(1.0, 1.0, 0.0), normal_load=24.525),
            _wheel("RL", 1, point=(-1.0, -1.0, 0.0), normal_load=24.525),
            _wheel("RR", 1, point=(1.0, -1.0, 0.0), normal_load=24.525),
        ]
        missing_ldot = assess_contact_wrench_feasibility(
            _com(1), _frame(1, balanced), angular_momentum_rate=None
        )
        self.assertEqual(EvidenceStatus.NOT_PROVEN, missing_ldot.status)
        self.assertEqual(
            ("NOT_PROVEN_MISSING_ANGULAR_MOMENTUM_RATE",),
            missing_ldot.reasons,
        )
        missing = [
            _wheel("FL", 1, point=(0.0, 0.0, 0.0), normal_load=98.1, moment=None),
            _inactive("FR", 1),
            _inactive("RL", 1),
            _inactive("RR", 1),
        ]
        unavailable = assess_contact_wrench_feasibility(
            _com(1),
            _frame(1, missing),
            angular_momentum_rate=_angular_rate(1, (0.0, 0.0, 0.0)),
        )
        self.assertEqual(EvidenceStatus.UNAVAILABLE, unavailable.status)
        outside_cone = [
            _wheel("FL", 1, point=(0.0, 0.0, 0.0), normal_load=98.1, friction=(90.0, 0.0, 0.0), mu=0.5),
            _inactive("FR", 1),
            _inactive("RL", 1),
            _inactive("RR", 1),
        ]
        not_proven = assess_contact_wrench_feasibility(
            _com(1, acceleration=(9.0, 0.0, 0.0)),
            _frame(1, outside_cone),
            angular_momentum_rate=_angular_rate(1, (0.0, -90.0, 0.0)),
            force_residual_tolerance_n=1.0e-9,
            moment_residual_tolerance_nm=1.0e-9,
        )
        self.assertEqual(EvidenceStatus.NOT_PROVEN, not_proven.status)
        self.assertFalse(not_proven.proven_feasible)

    def test_primary_diagonal_requires_all_load_dwell_unload_corridor_and_wrench_evidence(self) -> None:
        frame = _frame(
            1,
            [
                _wheel("FL", 1, point=(-1.0, -1.0, 0.0), normal_load=45.0, patch=0.2),
                _inactive("FR", 1),
                _wheel("RL", 1, point=(-0.5, 0.5, 0.0), normal_load=0.5, patch=0.2),
                _wheel("RR", 1, point=(1.0, 1.0, 0.0), normal_load=45.0, patch=0.2),
            ],
        )
        wrench = ContactWrenchFeasibility(
            status=EvidenceStatus.PROVEN,
            proven_feasible=True,
            physics_tick=1,
            witness_kind="test",
            multiple_contact_heights=False,
            dynamic=False,
            angular_momentum_rate_w_nm=(0.0, 0.0, 0.0),
            angular_momentum_rate_source="test.independent_centroidal_ldot",
            force_residual_w_n=(0.0, 0.0, 0.0),
            force_residual_norm_n=0.0,
            moment_residual_w_nm=(0.0, 0.0, 0.0),
            moment_residual_norm_nm=0.0,
            maximum_friction_utilization=0.0,
            reasons=(),
        )
        proven = assess_primary_diagonal_support(
            _com(1),
            frame,
            active_swing_leg="RL",
            wrench_feasibility=wrench,
            thresholds=THRESHOLDS,
        )
        self.assertEqual(EvidenceStatus.PROVEN, proven.status)
        self.assertEqual("PRIMARY_DIAGONAL_SUPPORT", proven.classification)
        self.assertEqual("FL_RR", proven.primary_diagonal)
        self.assertGreater(proven.load_ratio_fl_rr, 0.99)
        no_wrench = assess_primary_diagonal_support(
            _com(1),
            frame,
            active_swing_leg="RL",
            wrench_feasibility=None,
            thresholds=THRESHOLDS,
        )
        self.assertEqual(EvidenceStatus.NOT_PROVEN, no_wrench.status)


class CentroidalSupportEvidenceTests(unittest.TestCase):
    def _evidence(self) -> CentroidalSupportEvidence:
        com = _com(1)
        angular = _angular_rate(1, (0.0, 0.0, 0.0))
        frame = _frame(
            1,
            [
                _wheel("FL", 1, point=(-1.0, 1.0, 0.0), normal_load=24.525),
                _wheel("FR", 1, point=(1.0, 1.0, 0.0), normal_load=24.525),
                _wheel("RL", 1, point=(-1.0, -1.0, 0.0), normal_load=24.525),
                _wheel("RR", 1, point=(1.0, -1.0, 0.0), normal_load=24.525),
            ],
        )
        support = assess_support_region(com, frame, thresholds=THRESHOLDS)
        wrench = assess_contact_wrench_feasibility(
            com,
            frame,
            angular_momentum_rate=angular,
            force_residual_tolerance_n=1.0e-9,
            moment_residual_tolerance_nm=1.0e-9,
        )
        diagonal = assess_primary_diagonal_support(
            com,
            frame,
            active_swing_leg="RL",
            wrench_feasibility=wrench,
            thresholds=THRESHOLDS,
        )
        self.assertEqual(support, diagonal.support_region)
        return CentroidalSupportEvidence.create(
            sim_step=1,
            physics_time_s=DT,
            physics_dt_s=DT,
            whole_body_com=com,
            centroidal_angular_momentum_rate=angular,
            wheel_contacts=frame,
            support_region=support,
            contact_wrench_feasibility=wrench,
            diagonal_support=diagonal,
        )

    def test_exact_schema_canonical_sha_and_round_trip(self) -> None:
        evidence = self._evidence()
        mapping = evidence.to_mapping()
        self.assertEqual(CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA, mapping["schema_version"])
        expected = hashlib.sha256(
            json.dumps(
                mapping["payload"],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(expected, mapping["payload_sha256"])
        self.assertEqual(evidence, CentroidalSupportEvidence.from_mapping(mapping))

    def test_empty_or_invalid_active_leg_still_binds_envelope_support_and_wrench(self) -> None:
        base = self._evidence()
        for active_swing_leg in ("", "NOT_A_LEG"):
            with self.subTest(active_swing_leg=active_swing_leg):
                diagonal = assess_primary_diagonal_support(
                    base.whole_body_com,
                    base.wheel_contacts,
                    active_swing_leg=active_swing_leg,
                    wrench_feasibility=base.contact_wrench_feasibility,
                    thresholds=THRESHOLDS,
                )
                self.assertEqual(EvidenceStatus.UNAVAILABLE, diagonal.status)
                self.assertEqual(base.support_region, diagonal.support_region)
                self.assertEqual(
                    base.contact_wrench_feasibility.proven_feasible,
                    diagonal.wrench_feasibility_proven,
                )
                envelope = CentroidalSupportEvidence.create(
                    sim_step=base.sim_step,
                    physics_time_s=base.physics_time_s,
                    physics_dt_s=base.physics_dt_s,
                    whole_body_com=base.whole_body_com,
                    centroidal_angular_momentum_rate=base.centroidal_angular_momentum_rate,
                    wheel_contacts=base.wheel_contacts,
                    support_region=base.support_region,
                    contact_wrench_feasibility=base.contact_wrench_feasibility,
                    diagonal_support=diagonal,
                )
                self.assertEqual(diagonal, envelope.diagonal_support)

    def test_unavailable_contacts_still_bind_the_same_envelope_support_region(self) -> None:
        com = _com(1)
        angular = _angular_rate(1, (0.0, 0.0, 0.0))
        stale = replace(
            _wheel("FL", 1, point=(-1.0, 1.0, 0.0), normal_load=24.525),
            physics_tick=0,
        )
        frame = _frame(
            1,
            [stale, _inactive("FR", 1), _inactive("RL", 1), _inactive("RR", 1)],
        )
        self.assertFalse(frame.available)
        support = assess_support_region(com, frame, thresholds=THRESHOLDS)
        wrench = assess_contact_wrench_feasibility(
            com,
            frame,
            angular_momentum_rate=angular,
        )
        diagonal = assess_primary_diagonal_support(
            com,
            frame,
            active_swing_leg="RL",
            wrench_feasibility=wrench,
            thresholds=THRESHOLDS,
        )
        self.assertEqual(support, diagonal.support_region)
        envelope = CentroidalSupportEvidence.create(
            sim_step=1,
            physics_time_s=DT,
            physics_dt_s=DT,
            whole_body_com=com,
            centroidal_angular_momentum_rate=angular,
            wheel_contacts=frame,
            support_region=support,
            contact_wrench_feasibility=wrench,
            diagonal_support=diagonal,
        )
        self.assertEqual(diagonal, envelope.diagonal_support)

    def test_key_sha_tick_and_derived_contact_tampers_are_rejected(self) -> None:
        mapping = self._evidence().to_mapping()
        extra = copy.deepcopy(mapping)
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys are not exact"):
            CentroidalSupportEvidence.from_mapping(extra)

        stale = copy.deepcopy(mapping)
        stale["payload"]["sim_step"] = 2
        stale["payload"]["physics_time_s"] = 0.2
        stale["payload_sha256"] = hashlib.sha256(
            json.dumps(
                stale["payload"],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "not envelope-current"):
            CentroidalSupportEvidence.from_mapping(stale)

        derived = copy.deepcopy(mapping)
        derived["payload"]["wheel_contacts"]["contacts"][0]["support_qualified"] = False
        derived["payload_sha256"] = hashlib.sha256(
            json.dumps(
                derived["payload"],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "not reproducible"):
            CentroidalSupportEvidence.from_mapping(derived)


class TransferClassificationTests(unittest.TestCase):
    def _samples(self, *, change_joints: bool = True, acceleration_after_release: bool = True):
        samples: list[TransferSample] = []
        positions = [0.0, 0.0, 0.0, 0.0, 0.002, 0.004, 0.006, 0.008, 0.010]
        velocities = [0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02, 0.02]
        accelerations = [0.0, 0.0, 0.0, 0.0, 0.2 if acceleration_after_release else 0.0, 0.0, 0.0, 0.0, 0.0]
        support_loads = [100.0, 100.0, 104.0, 96.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        horizontal_forces = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        active_loads = [2.0, 2.0, 2.0, 1.8, 1.5, 1.2, 0.8, 0.5, 0.5]
        for tick in range(9):
            rows = [
                _wheel(
                    "FL",
                    tick,
                    point=(-1.0, -1.0, 0.0),
                    normal_load=support_loads[tick] / 2.0,
                    friction=(horizontal_forces[tick] / 2.0, 0.0, 0.0),
                    patch=0.2,
                ),
                _inactive("FR", tick),
                _wheel("RL", tick, point=(-0.5, 0.5, 0.0), normal_load=active_loads[tick], patch=0.2),
                _wheel(
                    "RR",
                    tick,
                    point=(1.0, 1.0, 0.0),
                    normal_load=support_loads[tick] / 2.0,
                    friction=(horizontal_forces[tick] / 2.0, 0.0, 0.0),
                    patch=0.2,
                ),
            ]
            angle = 0.02 * tick / 8.0 if change_joints else 0.0
            samples.append(
                TransferSample(
                    com=_com(
                        tick,
                        position=(positions[tick], 0.0, 1.0),
                        velocity=(velocities[tick], 0.0, 0.0),
                        acceleration=(accelerations[tick], 0.0, 0.0),
                    ),
                    contacts=_frame(tick, rows),
                    support_joint_angles_rad={"FL_knee": angle, "RR_knee": angle},
                    body_recoverable=True,
                    wrench_feasibility=_proven_wrench(tick),
                )
            )
        return samples

    def test_hybrid_requires_each_impulse_and_support_angle_chain_independently(self) -> None:
        result = classify_com_transfer(
            self._samples(),
            target_direction_w=(1.0, 0.0, 0.0),
            support_legs=("FL", "RR"),
            active_swing_leg="RL",
            support_joint_names=("FL_knee", "RR_knee"),
        )
        self.assertEqual(EvidenceStatus.PROVEN, result.status)
        self.assertEqual(TransferMethod.HYBRID_COM_TRANSFER, result.method)
        self.assertEqual((2, 3, 4, 5, 6, 7), (
            result.preload_tick,
            result.release_tick,
            result.acceleration_tick,
            result.velocity_tick,
            result.displacement_tick,
            result.settle_tick,
        ))
        self.assertAlmostEqual(0.02, result.predicted_delta_velocity_w_m_s[0])
        self.assertAlmostEqual(0.02, result.measured_delta_velocity_w_m_s[0])

    def test_each_method_is_never_inferred_from_the_other(self) -> None:
        impulse_only = classify_com_transfer(
            self._samples(change_joints=False),
            target_direction_w=(1.0, 0.0, 0.0),
            support_legs=("FL", "RR"),
            active_swing_leg="RL",
            support_joint_names=("FL_knee", "RR_knee"),
        )
        self.assertEqual(TransferMethod.IMPULSE_BASED_COM_TRANSFER, impulse_only.method)
        angle_only = classify_com_transfer(
            self._samples(acceleration_after_release=False),
            target_direction_w=(1.0, 0.0, 0.0),
            support_legs=("FL", "RR"),
            active_swing_leg="RL",
            support_joint_names=("FL_knee", "RR_knee"),
        )
        self.assertEqual(TransferMethod.SUPPORT_ANGLE_COM_TRANSFER, angle_only.method)
        missing_wrench = [replace(row, wrench_feasibility=None) for row in self._samples()]
        unproven = classify_com_transfer(
            missing_wrench,
            target_direction_w=(1.0, 0.0, 0.0),
            support_legs=("FL", "RR"),
            active_swing_leg="RL",
            support_joint_names=("FL_knee", "RR_knee"),
        )
        self.assertEqual(TransferMethod.NOT_YET_PROVEN, unproven.method)
        self.assertEqual(EvidenceStatus.NOT_PROVEN, unproven.status)

    def test_missing_tick_or_support_evidence_is_unavailable(self) -> None:
        samples = self._samples()
        samples[4] = replace(samples[4], com=replace(samples[4].com, physics_tick=99))
        result = classify_com_transfer(
            samples,
            target_direction_w=(1.0, 0.0, 0.0),
            support_legs=("FL", "RR"),
            active_swing_leg="RL",
            support_joint_names=("FL_knee", "RR_knee"),
        )
        self.assertEqual(EvidenceStatus.UNAVAILABLE, result.status)
        self.assertEqual(TransferMethod.NOT_YET_PROVEN, result.method)


class LazyContactAdapterTests(unittest.TestCase):
    def _measure(
        self,
        adapter,
        bank,
        whole_body_com,
        previous,
        *,
        surface_kind_by_leg=None,
        dwell_kind_by_leg=None,
        contact_moment_model="MEASURED",
    ):
        surfaces = surface_kind_by_leg or {leg: "GROUND" for leg in LEGS}
        dwell_kinds = dwell_kind_by_leg or surfaces
        return measure_isaac_wheel_contacts(
            adapter,
            bank,
            surface_kind_by_leg=surfaces,
            surface_height_m_by_leg={leg: 0.0 for leg in LEGS},
            friction_coefficient_by_leg={leg: 0.8 for leg in LEGS},
            surface_dwell_s_by_leg={leg: 0.2 for leg in LEGS},
            surface_dwell_kind_by_leg=dwell_kinds,
            previous_frame=previous,
            finite_patch_radius_m_by_leg={leg: 0.1 for leg in LEGS},
            whole_body_com=whole_body_com,
            contact_moment_model=contact_moment_model,
            thresholds=THRESHOLDS,
        )

    def test_lazy_adapter_measures_raw_current_wrench_slip_and_drift(self) -> None:
        tick = 2
        adapter, bank, whole_body_com, points = _lazy_contact_fixture(tick=tick)
        previous = _frame(
            tick - 1,
            [_wheel(leg, tick - 1, point=points[leg], normal_load=10.0) for leg in LEGS],
        )
        frame = self._measure(adapter, bank, whole_body_com, previous)
        self.assertTrue(frame.available)
        self.assertTrue(all(row.support_qualified for row in frame.contacts))
        self.assertTrue(all(row.measurement.slip_speed_m_s == 0.0 for row in frame.contacts))
        self.assertTrue(all(row.measurement.contact_drift_speed_m_s == 0.0 for row in frame.contacts))
        self.assertTrue(
            all(
                row.measurement.contact_moment_w_nm == (0.0, 0.0, 0.0)
                and row.measurement.contact_moment_model
                == "MEASURED"
                for row in frame.contacts
            )
        )
        self.assertTrue(
            all(
                bank.sensors[leg].contact_physx_view.contact_dt == DT
                and bank.sensors[leg].contact_physx_view.friction_dt == DT
                for leg in LEGS
            )
        )
        live_wrench = assess_contact_wrench_feasibility(
            whole_body_com,
            frame,
            angular_momentum_rate=_angular_rate(
                tick,
                (0.0, 0.0, 0.0),
                body_names=whole_body_com.body_names,
            ),
        )
        self.assertTrue(live_wrench.proven_feasible)
        self.assertEqual(
            "MEASURED_FORCE_AND_MOMENT_SUFFICIENT_CERTIFICATE",
            live_wrench.witness_kind,
        )

        switched = self._measure(
            adapter,
            bank,
            whole_body_com,
            previous,
            dwell_kind_by_leg={leg: "FRONT_FACE" for leg in LEGS},
        )
        self.assertTrue(switched.available)
        self.assertTrue(all(not row.support_qualified for row in switched.contacts))

    def test_explicit_point_zero_path_is_diagnostic_only_and_needs_no_raw_view(self) -> None:
        tick = 2
        adapter, bank, whole_body_com, points = _lazy_contact_fixture(tick=tick)
        previous = _frame(
            tick - 1,
            [_wheel(leg, tick - 1, point=points[leg], normal_load=10.0) for leg in LEGS],
        )
        for sensor in bank.sensors.values():
            del sensor.contact_physx_view
        frame = self._measure(
            adapter,
            bank,
            None,
            previous,
            contact_moment_model="POINT_CONTACT_ZERO_CONSERVATIVE",
        )
        self.assertTrue(frame.available)
        self.assertTrue(
            all(
                row.measurement.contact_moment_w_nm == (0.0, 0.0, 0.0)
                and row.measurement.contact_moment_model
                == "POINT_CONTACT_ZERO_CONSERVATIVE"
                for row in frame.contacts
            )
        )
        point_wrench = assess_contact_wrench_feasibility(
            whole_body_com,
            frame,
            angular_momentum_rate=_angular_rate(
                tick,
                (0.0, 0.0, 0.0),
                body_names=whole_body_com.body_names,
            ),
        )
        self.assertEqual(EvidenceStatus.NOT_PROVEN, point_wrench.status)
        self.assertFalse(point_wrench.proven_feasible)
        self.assertEqual(
            "POINT_CONTACT_ZERO_MOMENT_DIAGNOSTIC_ONLY",
            point_wrench.witness_kind,
        )
        self.assertIn(
            "NOT_PROVEN_POINT_CONTACT_ZERO_MOMENT_IS_NOT_MEASURED",
            point_wrench.reasons,
        )
        tampered_moment = replace(
            frame.contacts[0].measurement,
            contact_moment_w_nm=(0.0, 0.0, 0.01),
        )
        tampered_frame = replace(
            frame,
            contacts=(replace(frame.contacts[0], measurement=tampered_moment),)
            + frame.contacts[1:],
        )
        tampered_wrench = assess_contact_wrench_feasibility(
            whole_body_com,
            tampered_frame,
            angular_momentum_rate=_angular_rate(
                tick,
                (0.0, 0.0, 0.0),
                body_names=whole_body_com.body_names,
            ),
        )
        self.assertFalse(tampered_wrench.proven_feasible)
        self.assertEqual(EvidenceStatus.UNAVAILABLE, tampered_wrench.status)

    def test_asymmetric_raw_contact_distribution_preserves_required_moment(self) -> None:
        tick = 2
        forces = {
            "FL": ((0.0, 0.0, 29.43), (0.0, 0.0, 68.67)),
            "FR": (),
            "RL": (),
            "RR": (),
        }
        points = {
            "FL": ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            "FR": (),
            "RL": (),
            "RR": (),
        }
        adapter, bank, whole_body_com, aggregate_points = _lazy_contact_fixture(
            tick=tick,
            contact_forces_by_leg=forces,
            contact_points_by_leg=points,
            total_mass_kg=10.0,
        )
        previous = _frame(
            tick - 1,
            [
                _wheel(
                    "FL",
                    tick - 1,
                    point=aggregate_points["FL"],
                    normal_load=98.1,
                ),
                _inactive("FR", tick - 1),
                _inactive("RL", tick - 1),
                _inactive("RR", tick - 1),
            ],
        )
        surfaces = {"FL": "GROUND", "FR": "AIR", "RL": "AIR", "RR": "AIR"}
        measured_frame = self._measure(
            adapter,
            bank,
            whole_body_com,
            previous,
            surface_kind_by_leg=surfaces,
        )
        self.assertTrue(measured_frame.available)
        self.assertTrue(
            np.allclose(
                measured_frame.by_leg()["FL"].measurement.contact_moment_w_nm,
                (0.0, -39.24, 0.0),
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        angular_rate = _angular_rate(
            tick,
            (0.0, -39.24, 0.0),
            body_names=whole_body_com.body_names,
        )
        measured_wrench = assess_contact_wrench_feasibility(
            whole_body_com,
            measured_frame,
            angular_momentum_rate=angular_rate,
            force_residual_tolerance_n=1.0e-9,
            moment_residual_tolerance_nm=1.0e-9,
        )
        self.assertEqual(EvidenceStatus.PROVEN, measured_wrench.status)

        point_frame = self._measure(
            adapter,
            bank,
            None,
            previous,
            surface_kind_by_leg=surfaces,
            contact_moment_model="POINT_CONTACT_ZERO_CONSERVATIVE",
        )
        point_wrench = assess_contact_wrench_feasibility(
            whole_body_com,
            point_frame,
            angular_momentum_rate=angular_rate,
            force_residual_tolerance_n=1.0e-9,
            moment_residual_tolerance_nm=1.0e-9,
        )
        self.assertEqual(EvidenceStatus.NOT_PROVEN, point_wrench.status)
        self.assertAlmostEqual(39.24, point_wrench.moment_residual_norm_nm)
        self.assertIn(
            "NOT_PROVEN_POINT_CONTACT_ZERO_MOMENT_IS_NOT_MEASURED",
            point_wrench.reasons,
        )

    def test_raw_payload_layout_indices_capacity_and_tick_fail_closed(self) -> None:
        def standard_case():
            adapter, bank, com, points = _lazy_contact_fixture()
            previous = _frame(
                1,
                [_wheel(leg, 1, point=points[leg], normal_load=10.0) for leg in LEGS],
            )
            return adapter, bank, com, previous

        adapter, bank, com, previous = standard_case()
        bank.sensors["FL"].cfg.prim_path = "/World/WLRRobot/wrong_wheel"
        with self.assertRaisesRegex(ValueError, "prim identity"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        bank.sensors["FL"].cfg.filter_prim_paths_expr.reverse()
        with self.assertRaisesRegex(ValueError, "filter identity/order"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        bank.sensors["FL"].contact_physx_view.contact_payload_override = (np.zeros(1),)
        with self.assertRaisesRegex(ValueError, "get_contact_data payload is malformed"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        bank.sensors["FL"].contact_physx_view.contact_force_magnitudes = np.zeros(4)
        with self.assertRaisesRegex(ValueError, "contact force magnitudes shape"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        bank.sensors["FL"].contact_physx_view.contact_counts = np.asarray(
            [[0.5, 0.0]], dtype=float
        )
        with self.assertRaisesRegex(ValueError, "non-negative exact integers"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        view = bank.sensors["FL"].contact_physx_view
        view.contact_counts = np.asarray([[1, 1]], dtype=np.int64)
        view.contact_starts = np.asarray([[0, 0]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "slices overlap"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        sensor = bank.sensors["FL"]
        view = sensor.contact_physx_view
        view.contact_force_magnitudes[1, 0] = 2.0
        view.contact_points[1] = (-1.0, 1.0, 0.05)
        view.contact_normals[1] = (0.0, 0.0, 1.0)
        view.separation_distances[1, 0] = 0.0
        view.contact_counts = np.asarray([[1, 1]], dtype=np.int64)
        view.contact_starts = np.asarray([[0, 1]], dtype=np.int64)
        sensor.data.force_matrix_w[0, 0, 1] = (0.0, 0.0, 2.0)
        sensor.data.contact_pos_w[0, 0, 1] = (-1.0, 1.0, 0.05)
        with self.assertRaisesRegex(ValueError, "non-selected surface"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        view = bank.sensors["FL"].contact_physx_view
        view.contact_starts = np.asarray(
            [[view.max_contact_data_count, 0]],
            dtype=np.int64,
        )
        with self.assertRaisesRegex(ValueError, "slice exceeds capacity"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, points = _lazy_contact_fixture(
            capacity_by_leg={"FL": 1}
        )
        previous = _frame(
            1,
            [_wheel(leg, 1, point=points[leg], normal_load=10.0) for leg in LEGS],
        )
        with self.assertRaisesRegex(ValueError, "capacity is exhausted"):
            self._measure(adapter, bank, com, previous)

        no_contacts = {leg: () for leg in LEGS}
        friction_forces = {leg: () for leg in LEGS}
        friction_points = {leg: () for leg in LEGS}
        friction_forces["FL"] = ((1.0, 0.0, 0.0),)
        friction_points["FL"] = ((0.0, 0.0, 0.0),)
        adapter, bank, com, _points = _lazy_contact_fixture(
            contact_forces_by_leg=no_contacts,
            contact_points_by_leg=no_contacts,
            friction_forces_by_leg=friction_forces,
            friction_points_by_leg=friction_points,
            capacity_by_leg={"FL": 1},
        )
        previous = _frame(1, [_inactive(leg, 1) for leg in LEGS])
        with self.assertRaisesRegex(ValueError, "raw friction capacity is exhausted"):
            self._measure(adapter, bank, com, previous)

        adapter, bank, com, previous = standard_case()
        sensor = bank.sensors["FL"]
        sensor.contact_physx_view.after_friction_read = lambda: sensor._timestamp_last_update.__setitem__(
            0, 3 * DT
        )
        with self.assertRaisesRegex(ValueError, "changed tick during raw read"):
            self._measure(adapter, bank, com, previous)

    def test_raw_aggregate_and_normal_tolerances_are_explicit_and_fail_closed(self) -> None:
        adapter, bank, com, points = _lazy_contact_fixture()
        previous = _frame(
            1,
            [_wheel(leg, 1, point=points[leg], normal_load=10.0) for leg in LEGS],
        )
        sensor = bank.sensors["FL"]
        sensor.data.force_matrix_w[0, 0, 0, 2] += (
            0.5 * RAW_CONTACT_FORCE_AGGREGATE_ATOL_N
        )
        sensor.data.friction_forces_w[0, 0, 0, 0] += (
            0.5 * RAW_CONTACT_FORCE_AGGREGATE_ATOL_N
        )
        sensor.data.contact_pos_w[0, 0, 0, 0] += (
            0.5 * RAW_CONTACT_POINT_AGGREGATE_ATOL_M
        )
        sensor.contact_physx_view.contact_normals[0, 2] += (
            0.5 * RAW_CONTACT_NORMAL_UNIT_TOLERANCE
        )
        sensor.data.force_matrix_w[0, 0, 0, 2] += (
            10.0 * 0.5 * RAW_CONTACT_NORMAL_UNIT_TOLERANCE
        )
        inside = self._measure(adapter, bank, com, previous)
        self.assertTrue(inside.available)

        mismatch_cases = (
            (
                "normal force mismatch",
                "force_matrix_w",
                2.0 * RAW_CONTACT_FORCE_AGGREGATE_ATOL_N,
            ),
            (
                "friction force mismatch",
                "friction_forces_w",
                2.0 * RAW_CONTACT_FORCE_AGGREGATE_ATOL_N,
            ),
            (
                "contact point mismatch",
                "contact_pos_w",
                2.0 * RAW_CONTACT_POINT_AGGREGATE_ATOL_M,
            ),
        )
        for expected, field, delta in mismatch_cases:
            with self.subTest(field=field):
                adapter, bank, com, points = _lazy_contact_fixture()
                previous = _frame(
                    1,
                    [_wheel(leg, 1, point=points[leg], normal_load=10.0) for leg in LEGS],
                )
                array = getattr(bank.sensors["FL"].data, field)
                component = 2 if field == "force_matrix_w" else 0
                array[0, 0, 0, component] += delta
                with self.assertRaisesRegex(ValueError, expected):
                    self._measure(adapter, bank, com, previous)

        adapter, bank, com, points = _lazy_contact_fixture()
        previous = _frame(
            1,
            [_wheel(leg, 1, point=points[leg], normal_load=10.0) for leg in LEGS],
        )
        view = bank.sensors["FL"].contact_physx_view
        view.contact_normals[0, 2] += 2.0 * RAW_CONTACT_NORMAL_UNIT_TOLERANCE
        bank.sensors["FL"].data.force_matrix_w[0, 0, 0, 2] = (
            view.contact_force_magnitudes[0, 0] * view.contact_normals[0, 2]
        )
        with self.assertRaisesRegex(ValueError, "normal is not unit length"):
            self._measure(adapter, bank, com, previous)

    def test_measured_mode_requires_current_body_bound_com(self) -> None:
        adapter, bank, com, points = _lazy_contact_fixture()
        previous = _frame(
            1,
            [_wheel(leg, 1, point=points[leg], normal_load=10.0) for leg in LEGS],
        )
        with self.assertRaisesRegex(ValueError, "whole-body COM is required"):
            self._measure(adapter, bank, None, previous)
        with self.assertRaisesRegex(ValueError, "not current or body-identity-bound"):
            self._measure(adapter, bank, replace(com, physics_tick=1), previous)

    def test_unknown_surface_allows_two_reconciled_active_filters_but_is_not_support(self) -> None:
        adapter, bank, com, points = _lazy_contact_fixture()
        sensor = bank.sensors["FL"]
        view = sensor.contact_physx_view
        view.contact_force_magnitudes[1, 0] = 2.0
        view.contact_points[1] = (-1.0, 1.0, 0.05)
        view.contact_normals[1] = (0.0, 0.0, 1.0)
        view.separation_distances[1, 0] = 0.0
        view.contact_counts = np.asarray([[1, 1]], dtype=np.int64)
        view.contact_starts = np.asarray([[0, 1]], dtype=np.int64)
        sensor.data.force_matrix_w[0, 0, 1] = (0.0, 0.0, 2.0)
        sensor.data.contact_pos_w[0, 0, 1] = (-1.0, 1.0, 0.05)
        previous = _frame(
            1,
            [
                _inactive("FL", 1),
                *[
                    _wheel(leg, 1, point=points[leg], normal_load=10.0)
                    for leg in ("FR", "RL", "RR")
                ],
            ],
        )
        surfaces = {
            "FL": "UNKNOWN",
            "FR": "GROUND",
            "RL": "GROUND",
            "RR": "GROUND",
        }
        frame = self._measure(
            adapter,
            bank,
            com,
            previous,
            surface_kind_by_leg=surfaces,
        )
        self.assertTrue(frame.available)
        fl = frame.by_leg()["FL"]
        self.assertFalse(fl.measurement.active)
        self.assertFalse(fl.support_qualified)
        self.assertEqual((0.0, 0.0, 0.0), fl.measurement.normal_force_w_n)

    def test_force_exactly_equal_to_sensor_threshold_is_not_active(self) -> None:
        forces = {leg: ((0.0, 0.0, 1.0),) for leg in LEGS}
        adapter, bank, com, _points = _lazy_contact_fixture(
            contact_forces_by_leg=forces
        )
        previous = _frame(1, [_inactive(leg, 1) for leg in LEGS])
        air = {leg: "AIR" for leg in LEGS}
        frame = self._measure(
            adapter,
            bank,
            com,
            previous,
            surface_kind_by_leg=air,
        )
        self.assertTrue(
            all(not row.measurement.active for row in frame.contacts)
        )
        self.assertTrue(
            all(row.measurement.normal_force_w_n == (0.0, 0.0, 0.0) for row in frame.contacts)
        )
        self.assertTrue(frame.available)


if __name__ == "__main__":
    unittest.main()
