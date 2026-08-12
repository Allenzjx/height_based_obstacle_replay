from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.grounding_diagnostics import (
    GroundingTraceWriter,
    analyze_joint_trace,
    enrich_grounding_tick,
    write_grounding_trace_csv,
)


def base_frame() -> dict:
    all_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    return {
        "local_tick": 1,
        "sim_step": 1,
        "sim_time_s": 1.0 / 120.0,
        "physics_dt_s": 1.0 / 120.0,
        "root_velocity": [0.0] * 6,
        "root_velocity_evidence_valid": True,
        "root_velocity_evidence_error": "",
        "servo_joint_speed": 0.01,
        "wheel_joint_speed": 0.02,
        "stable_count": 1,
        "strict_tick_stable": True,
        "rolling_window_metrics": {"capacity": 60, "size": 1},
        "joint_state_evidence_valid": True,
        "joint_state_evidence_error": "",
        "wheel_target_evidence_valid": True,
        "wheel_target_evidence_error": "",
        "joint_position_by_name": {name: 0.1 for name in all_names},
        "joint_velocity_by_name": {name: 0.01 for name in all_names},
        "joint_position_target_by_name": {name: 0.11 for name in all_names},
        "joint_target_minus_position_by_name": {name: 0.01 for name in all_names},
        "servo_command_target_by_name": {name: 0.11 for name in SERVO_JOINT_NAMES},
        "servo_command_to_readback_error_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "wheel_target_velocity_by_name": {
            name: 0.0 for name in WHEEL_JOINT_NAMES
        },
        "wheel_target_readback_velocity_by_name": {
            name: 0.0 for name in WHEEL_JOINT_NAMES
        },
        "wheel_target_command_to_readback_error_by_name": {
            name: 0.0 for name in WHEEL_JOINT_NAMES
        },
    }


def fake_runtime(*, missing_wheel: bool = False):
    body_names = [
        "front_left_wheel",
        "front_right_wheel",
        "rear_left_wheel",
        "rear_right_wheel",
        "base_link",
    ]
    if missing_wheel:
        body_names.pop(3)
    vectors = [[0.0, 0.0, 5.0] for _ in body_names]
    vectors[-1] = [0.0, 0.0, 2.0]
    sensor = SimpleNamespace(
        body_names=body_names,
        data=SimpleNamespace(net_forces_w=[vectors]),
        update=lambda _dt, force_recompute=True: None,
    )
    scene = SimpleNamespace(contact_sensor=sensor)
    adapter = SimpleNamespace(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_pose_w=[[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]]
            )
        ),
        validate_robot_ground_contact=lambda apply_correction=False: {
            "checked": True,
            "classification": "OK",
            "physical_ground_safe": True,
            "maximum_collision_penetration_m": 0.001,
            "missing_collision_wheels": [],
            "unresolved_collision_wheels": [],
            "wheels": [
                {
                    "wheel_name": name,
                    "joint_name": name,
                    "collision_penetration_m": 0.001,
                    "collision_ground_clearance_m": -0.001,
                    "bounds_valid": True,
                    "bounds_finite": True,
                    "bounds_source": "live_body_mesh_points:test",
                    "collision_resolution_state": "OK",
                }
                for name in WHEEL_JOINT_NAMES
            ],
        },
    )
    return adapter, scene


class GroundingDiagnosticsTest(unittest.TestCase):
    def test_enrichment_contains_required_pose_force_contact_and_penetration(self) -> None:
        adapter, scene = fake_runtime()

        row = enrich_grounding_tick(adapter, scene, base_frame())

        self.assertTrue(row["diagnostic_evidence_valid"])
        self.assertEqual(row["root_pose_w"], [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(row["roll_rad"], 0.0)
        self.assertAlmostEqual(row["pitch_rad"], 0.0)
        self.assertEqual(set(row["wheel_net_forces_w"]), {"FL", "FR", "RL", "RR"})
        self.assertEqual(row["nonwheel_contacts"][0]["body_name"], "base_link")
        self.assertAlmostEqual(
            row["penetration"]["maximum_collision_penetration_m"], 0.001
        )

    def test_missing_wheel_force_layout_fails_closed(self) -> None:
        adapter, scene = fake_runtime(missing_wheel=True)

        row = enrich_grounding_tick(adapter, scene, base_frame())

        self.assertFalse(row["diagnostic_evidence_valid"])
        self.assertFalse(row["wheel_force_evidence_valid"])
        self.assertIn("RR/rear_right_wheel", row["wheel_force_evidence_error"])

    def test_root_velocity_source_flag_fails_closed_even_for_finite_values(self) -> None:
        adapter, scene = fake_runtime()
        frame = base_frame()
        frame["root_velocity_evidence_valid"] = False
        frame["root_velocity_evidence_error"] = "root velocity source missing"

        row = enrich_grounding_tick(adapter, scene, frame)

        self.assertFalse(row["diagnostic_evidence_valid"])
        self.assertIn(
            "root velocity source missing", row["diagnostic_evidence_error"]
        )

    def test_wheel_target_source_flag_fails_closed_even_for_finite_values(self) -> None:
        adapter, scene = fake_runtime()
        frame = base_frame()
        frame["wheel_target_evidence_valid"] = False
        frame["wheel_target_evidence_error"] = "wheel drive readback mismatch"

        row = enrich_grounding_tick(adapter, scene, frame)

        self.assertFalse(row["diagnostic_evidence_valid"])
        self.assertIn(
            "wheel drive readback mismatch", row["diagnostic_evidence_error"]
        )

    def test_penetration_requires_exact_wheel_joint_identity(self) -> None:
        adapter, scene = fake_runtime()
        adapter.validate_robot_ground_contact = lambda apply_correction=False: {
            "checked": True,
            "classification": "OK",
            "physical_ground_safe": True,
            "maximum_collision_penetration_m": 0.0,
            "missing_collision_wheels": [],
            "unresolved_collision_wheels": [],
            "wheels": [
                {
                    "wheel_name": name,
                    "collision_penetration_m": 0.0,
                    "collision_ground_clearance_m": 0.0,
                    "bounds_valid": True,
                    "bounds_finite": True,
                    "bounds_source": "test",
                    "collision_resolution_state": "OK",
                }
                for name in ("a", "b", "c", "d")
            ],
        }

        row = enrich_grounding_tick(adapter, scene, base_frame())

        self.assertFalse(row["diagnostic_evidence_valid"])
        self.assertFalse(row["penetration"]["valid"])
        self.assertIn(
            "unexpected/empty wheel_name",
            row["penetration"]["error"],
        )

    def test_penetration_rejects_invalid_bounds_and_inconsistent_maximum(self) -> None:
        adapter, scene = fake_runtime()
        diagnostics = adapter.validate_robot_ground_contact()
        diagnostics["wheels"][0]["bounds_valid"] = False
        diagnostics["maximum_collision_penetration_m"] = 0.0
        adapter.validate_robot_ground_contact = (
            lambda apply_correction=False: diagnostics
        )

        row = enrich_grounding_tick(adapter, scene, base_frame())

        self.assertFalse(row["penetration"]["valid"])
        self.assertIn("bounds_valid is not true", row["penetration"]["error"])
        self.assertIn(
            "does not match wheel rows", row["penetration"]["error"]
        )

    def test_contact_force_rejects_more_than_one_environment(self) -> None:
        adapter, scene = fake_runtime()
        rows = scene.contact_sensor.data.net_forces_w[0]
        scene.contact_sensor.data.net_forces_w = [rows, rows]

        row = enrich_grounding_tick(adapter, scene, base_frame())

        self.assertFalse(row["wheel_force_evidence_valid"])
        self.assertIn(
            "exactly one environment row",
            row["wheel_force_evidence_error"],
        )

    def test_writer_jsonl_csv_and_joint_analysis_are_complete(self) -> None:
        adapter, scene = fake_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = GroundingTraceWriter(
                root / "grounding.jsonl",
                adapter=adapter,
                scene_handle=scene,
            )
            row = writer(base_frame())
            writer.close()
            write_grounding_trace_csv(root / "grounding.csv", writer.rows)
            analysis = analyze_joint_trace(
                writer.rows,
                servo_threshold_rad_s=0.02,
                wheel_threshold_rad_s=0.20,
            )

            parsed = [
                json.loads(line)
                for line in (root / "grounding.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(parsed, [row])
            self.assertTrue((root / "grounding.csv").is_file())
            self.assertTrue(analysis["all_ticks_diagnostic_evidence_valid"])
            self.assertEqual(analysis["tick_count"], 1)
            self.assertEqual(analysis["final_offending_servos"], [])
            self.assertAlmostEqual(
                analysis["joints"]["front_left_hip"][
                    "initial_command_to_readback_error_rad"
                ],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
