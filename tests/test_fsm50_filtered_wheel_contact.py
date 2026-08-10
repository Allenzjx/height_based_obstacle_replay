from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from fsm_50mm_recording_derived_v3.filtered_wheel_contact import (
    FILTERED_SURFACES,
    FilteredContactLayoutError,
    configure_scene_for_filtered_wheel_contacts,
    contact_sensor_config_kwargs,
    create_filtered_wheel_contact_sensor_bank,
    create_filtered_wheel_contact_sensor_bank_result,
    filtered_contact_rows,
    wheel_contact_sensor_specs,
)
from fsm_50mm_recording_derived_v3.fsm50_telemetry import (
    COMMON_WHEEL_FORCE_SOURCE,
    FILTERED_FORCE_SOURCE,
    FSM50TelemetryCollector,
    canonical_wheel_values,
)
from fsm_50mm_recording_derived_v3.support_classifier import ContactClass, WheelContact


class _FakeCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSensor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.update_calls = []
        self.reset_calls = []
        body = str(cfg.prim_path).rsplit("/", 1)[-1]
        normal_by_body = {
            "front_left_wheel": ([0.0, 0.0, 10.0], [0.0, 0.0, 0.0]),
            "front_right_wheel": ([0.0, 0.0, 0.0], [-5.0, 0.0, 1.0]),
            "rear_left_wheel": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "rear_right_wheel": ([0.0, 0.0, 3.0], [-2.0, 0.0, 0.0]),
        }
        ground, obstacle = normal_by_body[body]
        force_matrix = np.asarray([[ground, obstacle]], dtype=float).reshape(1, 1, 2, 3)
        points = np.full((1, 1, 2, 3), np.nan, dtype=float)
        for index, force in enumerate((ground, obstacle)):
            if np.linalg.norm(force) > 0.0:
                points[0, 0, index] = [0.50 + index * 0.02, 0.0, 0.0 + index * 0.05]
        friction = np.zeros((1, 1, 2, 3), dtype=float)
        if body == "front_right_wheel":
            friction[0, 0, 1] = [0.0, 2.0, 0.0]
        self.data = SimpleNamespace(
            net_forces_w=force_matrix.sum(axis=2),
            force_matrix_w=force_matrix,
            contact_pos_w=points,
            friction_forces_w=friction,
        )

    def update(self, dt, force_recompute=False):
        self.update_calls.append((float(dt), bool(force_recompute)))

    def reset(self, env_ids=None):
        self.reset_calls.append(env_ids)


class _BadLayoutSensor(_FakeSensor):
    def __init__(self, cfg):
        super().__init__(cfg)
        if str(cfg.prim_path).endswith("front_left_wheel"):
            self.data.force_matrix_w = self.data.force_matrix_w[:, :, :1, :]
            self.data.contact_pos_w = self.data.contact_pos_w[:, :, :1, :]
            self.data.friction_forces_w = self.data.friction_forces_w[:, :, :1, :]


class _FailingSensor:
    def __init__(self, _cfg):
        raise RuntimeError("synthetic sensor failure")


class FilteredWheelContactTests(unittest.TestCase):
    def test_module_import_does_not_import_isaac_or_omni(self):
        code = """
import sys
before = set(sys.modules)
import fsm_50mm_recording_derived_v3.filtered_wheel_contact
added = set(sys.modules) - before
blocked = sorted(name for name in added if name == 'isaaclab' or name.startswith('isaaclab.') or name == 'omni' or name.startswith('omni.'))
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

    def test_specs_are_exact_single_wheel_rigid_body_paths(self):
        specs = wheel_contact_sensor_specs()
        self.assertEqual([row.leg for row in specs], ["FL", "FR", "RL", "RR"])
        self.assertEqual(
            [row.prim_path for row in specs],
            [
                "/World/WLRRobot/front_left_wheel",
                "/World/WLRRobot/front_right_wheel",
                "/World/WLRRobot/rear_left_wheel",
                "/World/WLRRobot/rear_right_wheel",
            ],
        )

    def test_config_enables_ordered_filtered_pair_evidence(self):
        kwargs = contact_sensor_config_kwargs(wheel_contact_sensor_specs()[0])
        self.assertEqual(
            kwargs["filter_prim_paths_expr"],
            [path for _name, path in FILTERED_SURFACES],
        )
        self.assertTrue(kwargs["track_contact_points"])
        self.assertTrue(kwargs["track_friction_forces"])
        self.assertTrue(kwargs["track_air_time"])
        self.assertGreaterEqual(kwargs["max_contact_data_count_per_prim"], 4)

    def test_composite_is_generic_telemetry_compatible_and_surface_resolved(self):
        bank = create_filtered_wheel_contact_sensor_bank(
            sensor_cls=_FakeSensor,
            sensor_cfg_cls=_FakeCfg,
            force_threshold_n=2.0,
        )
        bank.update(1.0 / 120.0, force_recompute=True)
        self.assertEqual(
            bank.body_names,
            [
                "front_left_wheel",
                "front_right_wheel",
                "rear_left_wheel",
                "rear_right_wheel",
            ],
        )
        self.assertEqual(bank.data.net_forces_w.shape, (1, 4, 3))
        rows = filtered_contact_rows(bank, force_threshold_n=2.0)
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [(row["leg"], row["surface"]) for row in rows[:4]],
            [("FL", "ground"), ("FL", "obstacle"), ("FR", "ground"), ("FR", "obstacle")],
        )
        fl_ground = next(
            row for row in rows if row["leg"] == "FL" and row["surface"] == "ground"
        )
        self.assertTrue(fl_ground["active"])
        self.assertEqual(fl_ground["upward_force_n"], 10.0)
        self.assertEqual(fl_ground["other_prim_path"], "/World/defaultGroundPlane")
        fr_obstacle = next(
            row for row in rows if row["leg"] == "FR" and row["surface"] == "obstacle"
        )
        self.assertTrue(fr_obstacle["active"])
        self.assertAlmostEqual(fr_obstacle["normal_force_n"], np.sqrt(26.0))
        self.assertEqual(fr_obstacle["friction_force_n"], 2.0)
        self.assertAlmostEqual(fr_obstacle["total_force_n"], np.sqrt(30.0))
        rl_rows = [row for row in rows if row["leg"] == "RL"]
        self.assertTrue(all(not row["active"] for row in rl_rows))
        self.assertTrue(all(sensor.update_calls for sensor in bank.sensors.values()))

    def test_layout_mismatch_fails_instead_of_mislabeling_filters(self):
        bank = create_filtered_wheel_contact_sensor_bank(
            sensor_cls=_BadLayoutSensor,
            sensor_cfg_cls=_FakeCfg,
        )
        with self.assertRaises(FilteredContactLayoutError):
            bank.update(1.0 / 120.0, force_recompute=True)

    def test_scene_factory_uses_legacy_sensor_error_tuple(self):
        config = SimpleNamespace(
            telemetry_contact_sensors_enabled=False,
            contact_sensor_factory=None,
        )
        configure_scene_for_filtered_wheel_contacts(
            config,
            sensor_cls=_FakeSensor,
            sensor_cfg_cls=_FakeCfg,
        )
        self.assertTrue(config.telemetry_contact_sensors_enabled)
        sensor, error = config.contact_sensor_factory()
        self.assertEqual(error, "")
        self.assertIsNotNone(sensor)

        sensor, error = create_filtered_wheel_contact_sensor_bank_result(
            sensor_cls=_FailingSensor,
            sensor_cfg_cls=_FakeCfg,
        )
        self.assertIsNone(sensor)
        self.assertIn("synthetic sensor failure", error)

    def test_fsm_collector_prefers_filtered_rows_and_replaces_generic_contacts(self):
        bank = create_filtered_wheel_contact_sensor_bank(
            sensor_cls=_FakeSensor,
            sensor_cfg_cls=_FakeCfg,
            force_threshold_n=2.0,
        )
        collector = object.__new__(FSM50TelemetryCollector)
        collector.scene_handle = SimpleNamespace(contact_sensor=bank)
        collector.force_threshold_n = 2.0
        collector.filtered_contact_errors = []
        collector.contact_rows = [{"body_name": "generic-net-force-row"}]
        collector.source_version = "unit-test"

        rows, error = collector._filtered_contacts()
        self.assertEqual(error, "")
        self.assertEqual(len(rows), 8)
        forces = collector._filtered_force_by_leg(rows)
        self.assertEqual(forces["FL"]["upward_force_n"], 10.0)
        self.assertEqual(forces["FR"]["upward_force_n"], 1.0)

        base_row = {"time_s": 1.25, "contact_force_valid": False}
        evidence = collector._install_filtered_contact_sample(
            base_row=base_row,
            filtered_rows=rows,
            before_contacts=0,
        )
        self.assertEqual(len(collector.contact_rows), 8)
        self.assertEqual(
            {(row["leg"], row["surface"]) for row in collector.contact_rows},
            {
                (leg, surface)
                for leg in ("FL", "FR", "RL", "RR")
                for surface in ("ground", "obstacle")
            },
        )
        self.assertTrue(evidence["contact_force_valid"])
        self.assertEqual(evidence["contact_force_source"], FILTERED_FORCE_SOURCE)
        self.assertTrue(evidence["filtered_contact_layout_valid"])
        self.assertEqual(
            {row["source"] for row in collector.contact_rows},
            {FILTERED_FORCE_SOURCE},
        )
        self.assertTrue(
            all(row["time_s"] == 1.25 for row in collector.contact_rows)
        )

    def test_filtered_front_face_force_is_not_counted_as_vertical_support(self):
        rows = [
            {
                "leg": "FL",
                "active": True,
                "normal_force_n": 25.0,
                "upward_force_n": 0.0,
                "total_force_n": 25.0,
                "source": FILTERED_FORCE_SOURCE,
            }
        ]
        force = FSM50TelemetryCollector._filtered_force_by_leg(rows)["FL"]
        self.assertEqual(force["upward_force_n"], 0.0)
        self.assertEqual(force["total_force_n"], 25.0)
        self.assertEqual(force["source"], FILTERED_FORCE_SOURCE)

    def test_filtered_force_aggregation_preserves_inactive_finite_zero_rows(self):
        rows = [
            {
                "leg": leg,
                "surface": surface,
                "active": False,
                "force_valid": True,
                "upward_force_n": 0.0,
                "total_force_n": 0.0,
                "source": FILTERED_FORCE_SOURCE,
            }
            for leg in ("FL", "FR", "RL", "RR")
            for surface in ("ground", "obstacle")
        ]
        forces = FSM50TelemetryCollector._filtered_force_by_leg(rows)
        for leg in ("FL", "FR", "RL", "RR"):
            self.assertEqual(forces[leg]["upward_force_n"], 0.0)
            self.assertEqual(forces[leg]["total_force_n"], 0.0)

    @staticmethod
    def _aggregate_sensor(body_names, net_forces):
        return SimpleNamespace(
            body_names=list(body_names),
            data=SimpleNamespace(net_forces_w=np.asarray(net_forces, dtype=float)),
        )

    def test_common_net_force_readback_supports_formal_multi_body_sensor(self):
        sensor = self._aggregate_sensor(
            [
                "base_link",
                "front_right_wheel",
                "rear_left_wheel",
                "front_left_wheel",
                "rear_right_wheel",
                "camera_link",
            ],
            [[
                [100.0, 0.0, 0.0],
                [3.0, 4.0, 5.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 7.0],
                [0.0, 0.0, -2.0],
                [0.0, 0.0, 0.0],
            ]],
        )
        forces, evidence = FSM50TelemetryCollector._common_wheel_net_force_by_leg(sensor)
        self.assertTrue(evidence["wheel_net_force_layout_valid"])
        self.assertTrue(evidence["wheel_net_force_valid"])
        self.assertEqual(evidence["wheel_net_force_error"], "")
        self.assertEqual(
            evidence["wheel_contact_force_common_source"],
            COMMON_WHEEL_FORCE_SOURCE,
        )
        self.assertEqual(forces["FL"]["upward_force_n"], 7.0)
        self.assertEqual(forces["FL"]["total_force_n"], 7.0)
        self.assertEqual(forces["FR"]["upward_force_n"], 5.0)
        self.assertEqual(forces["FR"]["total_force_n"], np.sqrt(50.0))
        self.assertEqual(forces["RL"]["upward_force_n"], 0.0)
        self.assertEqual(forces["RL"]["total_force_n"], 0.0)
        self.assertEqual(forces["RR"]["upward_force_n"], 0.0)
        self.assertEqual(forces["RR"]["total_force_n"], 2.0)

    def test_common_net_force_readback_supports_filtered_four_wheel_bank(self):
        bank = create_filtered_wheel_contact_sensor_bank(
            sensor_cls=_FakeSensor,
            sensor_cfg_cls=_FakeCfg,
            force_threshold_n=2.0,
        )
        forces, evidence = FSM50TelemetryCollector._common_wheel_net_force_by_leg(bank)
        self.assertTrue(evidence["wheel_net_force_layout_valid"])
        self.assertTrue(evidence["wheel_net_force_valid"])
        self.assertEqual(forces["FL"]["upward_force_n"], 10.0)
        self.assertEqual(forces["FL"]["total_force_n"], 10.0)
        self.assertEqual(forces["FR"]["upward_force_n"], 1.0)
        self.assertEqual(forces["FR"]["total_force_n"], np.sqrt(26.0))
        self.assertEqual(forces["RL"]["upward_force_n"], 0.0)
        self.assertEqual(forces["RL"]["total_force_n"], 0.0)

    def test_common_net_force_airborne_zero_is_finite_sensor_evidence(self):
        names = [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ]
        sensor = self._aggregate_sensor(names, [np.zeros((4, 3), dtype=float)])
        forces, evidence = FSM50TelemetryCollector._common_wheel_net_force_by_leg(sensor)
        self.assertTrue(evidence["wheel_net_force_valid"])
        for leg in ("FL", "FR", "RL", "RR"):
            self.assertTrue(np.isfinite(forces[leg]["upward_force_n"]))
            self.assertTrue(np.isfinite(forces[leg]["total_force_n"]))
            self.assertEqual(forces[leg]["upward_force_n"], 0.0)
            self.assertEqual(forces[leg]["total_force_n"], 0.0)

    def test_common_net_force_readback_fails_closed_on_invalid_evidence(self):
        names = [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ]
        valid = np.zeros((1, 4, 3), dtype=float)
        cases = {
            "missing": self._aggregate_sensor(names[:-1], valid[:, :-1, :]),
            "duplicate": self._aggregate_sensor(
                [*names, "front_left_wheel"],
                np.zeros((1, 5, 3), dtype=float),
            ),
            "nonfinite": self._aggregate_sensor(
                names,
                np.where(
                    np.arange(valid.size).reshape(valid.shape) == 0,
                    np.nan,
                    valid,
                ),
            ),
            "wrong_rank": self._aggregate_sensor(names, np.zeros((4, 3), dtype=float)),
            "body_shape": self._aggregate_sensor(names, np.zeros((1, 3, 3), dtype=float)),
            "vector_shape": self._aggregate_sensor(names, np.zeros((1, 4, 4), dtype=float)),
            "empty_env": self._aggregate_sensor(names, np.zeros((0, 4, 3), dtype=float)),
        }
        for label, sensor in cases.items():
            with self.subTest(label=label):
                forces, evidence = (
                    FSM50TelemetryCollector._common_wheel_net_force_by_leg(sensor)
                )
                self.assertFalse(evidence["wheel_net_force_valid"])
                self.assertTrue(evidence["wheel_net_force_error"])
                for leg in ("FL", "FR", "RL", "RR"):
                    self.assertTrue(np.isnan(forces[leg]["upward_force_n"]))
                    self.assertTrue(np.isnan(forces[leg]["total_force_n"]))

    def test_physical_filtered_validity_requires_every_layout_and_force_sample(self):
        valid = {
            "filtered_contact_available": True,
            "filtered_contact_layout_valid": True,
            "filtered_contact_force_valid": True,
            "filtered_contact_geometry_valid": True,
        }
        invalid_force = {**valid, "filtered_contact_force_valid": False}
        available_only = {"filtered_contact_available": True}
        self.assertTrue(FSM50TelemetryCollector._filtered_samples_valid([valid]))
        self.assertFalse(
            FSM50TelemetryCollector._filtered_samples_valid([valid, invalid_force])
        )
        self.assertFalse(
            FSM50TelemetryCollector._filtered_samples_valid([available_only])
        )

    def test_support_geometry_uses_measured_filtered_contact_point(self):
        geometric = WheelContact(
            leg="FL",
            contact_class=ContactClass.TOP,
            active=True,
            center_w=(0.6, 0.2, 0.1),
            contact_point_w=(0.6, 0.2, 0.05),
            obstacle_relative=(0.1, 0.2, 0.0),
            upward_force_n=7.0,
            total_force_n=7.0,
            clearance_over_top_m=0.0,
            front_face_clearance_m=0.1,
            source="geometry",
            confidence="FORCE_AND_GEOMETRY",
        )
        rows = [
            {
                "leg": "FL",
                "surface": "obstacle",
                "active": True,
                "contact_point_valid": True,
                "contact_point_w": [0.57, 0.19, 0.05],
                "total_force_n": 7.0,
            }
        ]
        installed = FSM50TelemetryCollector._install_measured_contact_points(
            {"FL": geometric}, rows
        )
        self.assertEqual((0.57, 0.19, 0.05), installed["FL"].contact_point_w)
        self.assertIn("contact_pos_w", installed["FL"].source)

    def test_nonwheel_collision_sample_is_force_backed_and_fail_closed(self):
        collector = object.__new__(FSM50TelemetryCollector)
        collector.scene_handle = SimpleNamespace(contact_sensor=object())
        collector.source_version = "unit-test"
        collector.dangerous_nonwheel_contact_force_n = 5.0
        collector.nonwheel_obstacle_rows = []
        collector.dangerous_collision_rows = []
        collector.collision_evidence_errors = []
        sensor_rows = [
            {
                "body_name": "base_link",
                "active": True,
                "normal_force_n": 7.0,
                "total_force_n": 7.0,
                "force_valid": True,
                "contact_point_valid": True,
            }
        ]
        with patch(
            "fsm_50mm_recording_derived_v3.fsm50_telemetry.nonwheel_obstacle_contact_rows",
            return_value=sensor_rows,
        ):
            rows, valid, dangerous, error = collector._nonwheel_collision_sample(1.25)
        self.assertTrue(valid)
        self.assertTrue(dangerous)
        self.assertEqual("", error)
        self.assertEqual(1, len(rows))
        self.assertEqual(1, len(collector.dangerous_collision_rows))

        with patch(
            "fsm_50mm_recording_derived_v3.fsm50_telemetry.nonwheel_obstacle_contact_rows",
            side_effect=RuntimeError("missing collision bank"),
        ):
            _rows, valid, dangerous, error = collector._nonwheel_collision_sample(2.0)
        self.assertFalse(valid)
        self.assertFalse(dangerous)
        self.assertIn("missing collision bank", error)

    def test_canonical_wheel_values_apply_all_four_physical_forward_signs(self):
        raw = {leg: 1.0 for leg in ("FL", "FR", "RL", "RR")}
        forward = canonical_wheel_values(raw, wheel_direction=1.0)
        self.assertEqual(
            forward,
            {"FL": -1.0, "FR": 1.0, "RL": -1.0, "RR": 1.0},
        )
        reversed_direction = canonical_wheel_values(raw, wheel_direction=-1.0)
        self.assertEqual(
            reversed_direction,
            {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0},
        )
        with self.assertRaises(ValueError):
            canonical_wheel_values(raw, wheel_direction=0.0)


if __name__ == "__main__":
    unittest.main()
