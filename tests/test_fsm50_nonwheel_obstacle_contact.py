from __future__ import annotations

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np

from fsm_50mm_recording_derived_v3.nonwheel_obstacle_contact import (
    OBSTACLE_PRIM_PATH,
    NonWheelContactDiscoveryError,
    NonWheelContactEvidenceError,
    NonWheelContactLayoutError,
    configure_scene_for_wheel_and_nonwheel_contacts,
    create_nonwheel_obstacle_contact_sensor_bank,
    create_nonwheel_obstacle_contact_sensor_bank_result,
    discover_nonwheel_rigid_body_specs,
    nonwheel_contact_sensor_config_kwargs,
    nonwheel_obstacle_contact_rows,
)


class _FakeRigidBodyAPI:
    pass


class _FakePath:
    def __init__(self, path: str) -> None:
        self.pathString = path

    def __str__(self) -> str:
        return self.pathString


class _FakePrim:
    def __init__(self, path: str, *, valid: bool = True, rigid: bool = False) -> None:
        self.path = path
        self.valid = valid
        self.rigid = rigid

    def IsValid(self) -> bool:
        return self.valid

    def GetPath(self) -> _FakePath:
        return _FakePath(self.path)

    def HasAPI(self, api: object) -> bool:
        if api is not _FakeRigidBodyAPI:
            raise AssertionError("unexpected API class")
        return self.rigid


class _FakeStage:
    def __init__(self, prims: list[_FakePrim]) -> None:
        self.prims = list(prims)
        self.by_path = {prim.path: prim for prim in prims}

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return self.by_path.get(path, _FakePrim(path, valid=False))


def _range_from_stage(stage: _FakeStage):
    def _range(root: _FakePrim):
        prefix = root.path.rstrip("/")
        return [
            prim
            for prim in stage.prims
            if prim.path == prefix or prim.path.startswith(prefix + "/")
        ]

    return _range


def _valid_stage() -> _FakeStage:
    return _FakeStage(
        [
            _FakePrim("/World/WLRRobot", rigid=True),
            _FakePrim("/World/WLRRobot/front_left_upper", rigid=True),
            _FakePrim("/World/WLRRobot/front_left_bot", rigid=True),
            _FakePrim("/World/WLRRobot/front_left_wheel", rigid=True),
            _FakePrim("/World/WLRRobot/front_right_upper", rigid=False),
            _FakePrim("/World/WLRRobot/spare_wheel_link", rigid=True),
            _FakePrim("/World/Obstacle", rigid=True),
        ]
    )


class _FakeCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSensor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.update_calls = []
        self.reset_calls = []
        body_name = str(cfg.prim_path).rsplit("/", 1)[-1]
        normal = {
            "WLRRobot": [0.0, 0.0, 0.0],
            "front_left_bot": [-3.0, 0.0, 4.0],
            "front_left_upper": [0.0, 0.0, 0.0],
        }[body_name]
        friction = [0.0, 2.0, 0.0] if body_name == "front_left_bot" else [0.0, 0.0, 0.0]
        force_matrix = np.asarray(normal, dtype=float).reshape(1, 1, 1, 3)
        friction_matrix = np.asarray(friction, dtype=float).reshape(1, 1, 1, 3)
        point = np.full((1, 1, 1, 3), np.nan, dtype=float)
        if np.linalg.norm(force_matrix) > 0.0:
            point[0, 0, 0] = [1.1, 0.05, 0.04]
        self.data = SimpleNamespace(
            net_forces_w=force_matrix.sum(axis=2),
            force_matrix_w=force_matrix,
            contact_pos_w=point,
            friction_forces_w=friction_matrix,
        )

    def update(self, dt, force_recompute=False):
        self.update_calls.append((float(dt), bool(force_recompute)))

    def reset(self, env_ids=None):
        self.reset_calls.append(env_ids)


class _TwoFilterSensor(_FakeSensor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.data.force_matrix_w = np.repeat(self.data.force_matrix_w, 2, axis=2)
        self.data.contact_pos_w = np.repeat(self.data.contact_pos_w, 2, axis=2)
        self.data.friction_forces_w = np.repeat(self.data.friction_forces_w, 2, axis=2)


class _NonFiniteForceSensor(_FakeSensor):
    def __init__(self, cfg):
        super().__init__(cfg)
        if str(cfg.prim_path).endswith("front_left_bot"):
            self.data.force_matrix_w[0, 0, 0, 0] = np.nan


class _ActiveWithoutPointSensor(_FakeSensor):
    def __init__(self, cfg):
        super().__init__(cfg)
        if str(cfg.prim_path).endswith("front_left_bot"):
            self.data.contact_pos_w[:] = np.nan


class _FakeWheelBank:
    is_filtered_wheel_contact_bank = True

    def __init__(self):
        self.body_names = ["front_left_wheel"]
        self.cfg = SimpleNamespace(force_threshold=1.0)
        self.data = SimpleNamespace(net_forces_w=np.zeros((1, 1, 3)))
        self.update_calls = []
        self.reset_calls = []

    def update(self, dt, force_recompute=False):
        self.update_calls.append((dt, force_recompute))

    def reset(self, env_ids=None):
        self.reset_calls.append(env_ids)

    def filtered_observations(self, **_kwargs):
        return ["wheel-row"]


def _create_bank(sensor_cls=_FakeSensor):
    stage = _valid_stage()
    return create_nonwheel_obstacle_contact_sensor_bank(
        stage=stage,
        sensor_cls=sensor_cls,
        sensor_cfg_cls=_FakeCfg,
        rigid_body_api_cls=_FakeRigidBodyAPI,
        prim_range_factory=_range_from_stage(stage),
        force_threshold_n=1.0,
    )


class NonWheelObstacleContactTests(unittest.TestCase):
    def test_import_is_safe_before_isaac_and_kit_start(self):
        code = """
import sys
before = set(sys.modules)
import fsm_50mm_recording_derived_v3.nonwheel_obstacle_contact
added = set(sys.modules) - before
blocked = sorted(name for name in added if name == 'isaaclab' or name.startswith('isaaclab.') or name == 'omni' or name.startswith('omni.') or name == 'pxr' or name.startswith('pxr.'))
assert not blocked, blocked
"""
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_discovers_every_nonwheel_rigid_body_in_sorted_order(self):
        stage = _valid_stage()
        specs = discover_nonwheel_rigid_body_specs(
            stage=stage,
            rigid_body_api_cls=_FakeRigidBodyAPI,
            prim_range_factory=_range_from_stage(stage),
        )
        self.assertEqual(
            [spec.prim_path for spec in specs],
            [
                "/World/WLRRobot",
                "/World/WLRRobot/front_left_bot",
                "/World/WLRRobot/front_left_upper",
            ],
        )
        self.assertNotIn("spare_wheel_link", [spec.body_name for spec in specs])

    def test_missing_obstacle_and_empty_body_set_fail_closed(self):
        no_obstacle = _FakeStage([_FakePrim("/World/WLRRobot", rigid=True)])
        with self.assertRaisesRegex(NonWheelContactDiscoveryError, "obstacle prim"):
            discover_nonwheel_rigid_body_specs(
                stage=no_obstacle,
                rigid_body_api_cls=_FakeRigidBodyAPI,
                prim_range_factory=_range_from_stage(no_obstacle),
            )

        no_nonwheel = _FakeStage(
            [
                _FakePrim("/World/WLRRobot", rigid=False),
                _FakePrim("/World/WLRRobot/front_left_wheel", rigid=True),
                _FakePrim("/World/Obstacle", rigid=True),
            ]
        )
        with self.assertRaisesRegex(NonWheelContactDiscoveryError, "no non-wheel"):
            discover_nonwheel_rigid_body_specs(
                stage=no_nonwheel,
                rigid_body_api_cls=_FakeRigidBodyAPI,
                prim_range_factory=_range_from_stage(no_nonwheel),
            )

    def test_sensor_cfg_is_exact_body_and_obstacle_only(self):
        stage = _valid_stage()
        spec = discover_nonwheel_rigid_body_specs(
            stage=stage,
            rigid_body_api_cls=_FakeRigidBodyAPI,
            prim_range_factory=_range_from_stage(stage),
        )[1]
        kwargs = nonwheel_contact_sensor_config_kwargs(spec)
        self.assertEqual(kwargs["prim_path"], spec.prim_path)
        self.assertEqual(kwargs["filter_prim_paths_expr"], [OBSTACLE_PRIM_PATH])
        self.assertTrue(kwargs["track_contact_points"])
        self.assertTrue(kwargs["track_friction_forces"])
        self.assertNotIn(".*", kwargs["prim_path"])

    def test_bank_reports_real_filtered_force_and_total_force(self):
        bank = _create_bank()
        bank.update(1.0 / 120.0, force_recompute=True)
        self.assertEqual(bank.data.force_matrix_w.shape, (1, 3, 1, 3))
        rows = nonwheel_obstacle_contact_rows(bank, force_threshold_n=1.0)
        self.assertEqual(len(rows), 3)
        collision = next(row for row in rows if row["body_name"] == "front_left_bot")
        self.assertTrue(collision["active"])
        self.assertEqual(collision["normal_force_n"], 5.0)
        self.assertEqual(collision["friction_force_n"], 2.0)
        self.assertAlmostEqual(collision["total_force_n"], np.sqrt(29.0))
        self.assertEqual(collision["body_prim_path"], "/World/WLRRobot/front_left_bot")
        self.assertEqual(collision["other_prim_path"], "/World/Obstacle")
        self.assertTrue(collision["force_valid"])
        self.assertTrue(collision["contact_point_valid"])
        zero_rows = nonwheel_obstacle_contact_rows(bank, force_threshold_n=0.0)
        zero_force = next(row for row in zero_rows if row["body_name"] == "WLRRobot")
        self.assertFalse(zero_force["active"])
        self.assertTrue(all(sensor.update_calls for sensor in bank.sensors.values()))
        bank.reset([0])
        self.assertTrue(all(sensor.reset_calls == [[0]] for sensor in bank.sensors.values()))

    def test_wrong_filter_layout_fails_closed(self):
        bank = _create_bank(_TwoFilterSensor)
        with self.assertRaisesRegex(NonWheelContactLayoutError, "exactly obstacle-only"):
            bank.update(1.0 / 120.0, force_recompute=True)

    def test_nonfinite_force_fails_closed(self):
        bank = _create_bank(_NonFiniteForceSensor)
        with self.assertRaisesRegex(NonWheelContactEvidenceError, "non-finite"):
            bank.update(1.0 / 120.0, force_recompute=True)

    def test_active_force_without_contact_point_fails_closed(self):
        bank = _create_bank(_ActiveWithoutPointSensor)
        bank.update(1.0 / 120.0, force_recompute=True)
        with self.assertRaisesRegex(NonWheelContactEvidenceError, "no finite point"):
            bank.observations()

    def test_result_factory_returns_error_instead_of_unverified_empty_bank(self):
        stage = _FakeStage([_FakePrim("/World/WLRRobot", rigid=True)])
        bank, error = create_nonwheel_obstacle_contact_sensor_bank_result(
            stage=stage,
            sensor_cls=_FakeSensor,
            sensor_cfg_cls=_FakeCfg,
            rigid_body_api_cls=_FakeRigidBodyAPI,
            prim_range_factory=_range_from_stage(stage),
        )
        self.assertIsNone(bank)
        self.assertIn("NonWheelContactDiscoveryError", error)

    def test_combined_factory_preserves_wheel_api_and_exposes_nonwheel_rows(self):
        wheel_bank = _FakeWheelBank()
        stage = _valid_stage()
        config = SimpleNamespace(
            telemetry_contact_sensors_enabled=False,
            contact_sensor_factory=None,
        )
        configure_scene_for_wheel_and_nonwheel_contacts(
            config,
            wheel_factory=lambda: (wheel_bank, ""),
            stage=stage,
            sensor_cls=_FakeSensor,
            sensor_cfg_cls=_FakeCfg,
            rigid_body_api_cls=_FakeRigidBodyAPI,
            prim_range_factory=_range_from_stage(stage),
        )
        self.assertTrue(config.telemetry_contact_sensors_enabled)
        combined, error = config.contact_sensor_factory()
        self.assertEqual(error, "")
        self.assertIsNotNone(combined)
        combined.update(1.0 / 120.0, force_recompute=True)
        self.assertEqual(combined.filtered_observations(), ["wheel-row"])
        self.assertEqual(len(combined.nonwheel_obstacle_observations()), 3)
        self.assertEqual(combined.body_names, ["front_left_wheel"])
        self.assertEqual(len(nonwheel_obstacle_contact_rows(combined)), 3)
        combined.reset([0])
        self.assertEqual(wheel_bank.reset_calls, [[0]])


if __name__ == "__main__":
    unittest.main()
