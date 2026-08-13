from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import unittest

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.motion_start_readiness import (
    evaluate_motion_start_ready,
    rest_qualification_summary,
)


def _ground(*, stable: bool = False) -> dict:
    return {
        "grounded_reference_valid": bool(stable),
        "grounded_reference_physics_valid": True,
        "grounded_reference_stable": bool(stable),
        "stable_frames": 10 if stable else 0,
        "stable_frames_required": 10,
        "servo_speed_threshold_rad_s": 0.02,
        "final_servo_joint_velocity_rad_s": 0.069,
        "ground_contact_resolved": True,
        "ground_support_clearance_evidence": {"valid": True, "error": ""},
        "grounded_reference_diagnostics": {
            "checked": True,
            "physical_ground_safe": True,
            "classification": "VISUAL_ONLY_INTERSECTION",
            "ground_reference_block_reason": "servo velocity above threshold"
            if not stable
            else "",
        },
    }


def _snapshot() -> dict:
    names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    joint_velocity = {name: 0.069 if name == "rear_left_hip" else 0.0 for name in names}
    return {
        "adapter_runtime_instance_id": "clean-adapter-1",
        "root_state_write_count": 0,
        "root_state_write_events": [],
        "sim_step": 180,
        "sim_time_s": 1.5,
        "command_state": {
            "servos": {name: 0.0 for name in SERVO_JOINT_NAMES},
            "wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES},
        },
        "diagnostic_evidence_valid": True,
        "diagnostic_evidence_error": "",
        "root_velocity_evidence_valid": True,
        "root_velocity_evidence_error": "",
        "root_velocity": [0.0, 0.0, 0.006, 0.004, 0.007, 0.001],
        "roll_rad": 0.001,
        "pitch_rad": -0.012,
        "obstacle_relative_pose": {
            "valid": True,
            "error": "",
            "wheel_obstacle_relative_pose": {
                leg: {
                    "center_w": [0.0, 0.0, 0.05],
                    "x_from_obstacle_front_m": -0.5,
                    "y_from_obstacle_center_m": 0.0,
                    "z_from_obstacle_top_m": 0.0,
                }
                for leg in ("FL", "FR", "RL", "RR")
            },
        },
        "joint_state_evidence_valid": True,
        "joint_state_evidence_error": "",
        "joint_position_by_name": {name: 0.0 for name in names},
        "joint_velocity_by_name": joint_velocity,
        "joint_position_target_by_name": {name: 0.0 for name in names},
        "servo_command_target_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "servo_command_to_readback_error_by_name": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "wheel_target_evidence_valid": True,
        "wheel_target_evidence_error": "",
        "wheel_target_velocity_by_name": {name: 0.0 for name in WHEEL_JOINT_NAMES},
        "wheel_target_readback_velocity_by_name": {
            name: 0.0 for name in WHEEL_JOINT_NAMES
        },
        "wheel_target_command_to_readback_error_by_name": {
            name: 0.0 for name in WHEEL_JOINT_NAMES
        },
        "wheel_force_evidence_valid": True,
        "wheel_force_evidence_error": "",
        "wheel_net_forces_w": {
            leg: [0.0, 0.0, 7.0] for leg in ("FL", "FR", "RL", "RR")
        },
        "nonwheel_contacts": [],
        "penetration": {
            "valid": True,
            "error": "",
            "physical_ground_safe": True,
            "maximum_collision_penetration_m": 0.00105,
        },
    }


def _plan_identity() -> dict:
    state = {
        "servos": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES},
    }
    state_sha = hashlib.sha256(
        json.dumps(
            state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "source_version": "v003_20260805_224517_157723_manual",
        "source_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "plan_id": "v003-trial-1",
        "request_id": "request-1",
        "worker_session_id": "session-1",
        "event_count": 160,
        "segment_count": 112,
        "source_initial_command_state": state,
        "source_initial_command_state_sha256": state_sha,
    }


class MotionStartReadinessTests(unittest.TestCase):
    def test_module_import_does_not_import_isaac_or_numpy(self):
        code = """
import sys
before = set(sys.modules)
import fsm_50mm_recording_derived_v3.motion_start_readiness
added = set(sys.modules) - before
blocked = sorted(name for name in added if name == 'isaaclab' or name.startswith('isaaclab.') or name == 'omni' or name.startswith('omni.') or name == 'numpy')
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

    def test_strict_rest_can_fail_while_motion_start_is_ready(self):
        result = evaluate_motion_start_ready(
            ground_reference=_ground(stable=False),
            snapshot=_snapshot(),
            production_runtime_ready=True,
            expected_sim_step=180,
            expected_adapter_runtime_instance_id="clean-adapter-1",
            plan_identity=_plan_identity(),
            root_seed_applied=False,
        )
        self.assertFalse(result["rest_qualification"]["passed"])
        self.assertEqual(result["rest_qualification"]["status"], "FAIL")
        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["observed_servo_speed_rad_s"], 0.069)
        self.assertFalse(result["strict_rest_is_required"])

    def test_rest_summary_never_promotes_failed_grounding(self):
        failed = rest_qualification_summary(_ground(stable=False))
        passed = rest_qualification_summary(_ground(stable=True))
        self.assertFalse(failed["passed"])
        self.assertTrue(passed["passed"])
        self.assertEqual(failed["servo_speed_threshold_rad_s"], 0.02)

    def test_missing_or_nan_joint_evidence_fails_closed(self):
        for label, mutate in (
            (
                "missing",
                lambda row: row["joint_position_by_name"].pop(SERVO_JOINT_NAMES[0]),
            ),
            (
                "nan",
                lambda row: row["joint_velocity_by_name"].__setitem__(
                    SERVO_JOINT_NAMES[0], math.nan
                ),
            ),
        ):
            with self.subTest(label=label):
                snapshot = _snapshot()
                mutate(snapshot)
                result = evaluate_motion_start_ready(
                    ground_reference=_ground(),
                    snapshot=snapshot,
                    production_runtime_ready=True,
                    expected_sim_step=180,
                    expected_adapter_runtime_instance_id="clean-adapter-1",
                    plan_identity=_plan_identity(),
                )
                self.assertFalse(result["ready"])
                self.assertIn(
                    "joint_and_physx_position_targets_valid", result["failed_checks"]
                )

    def test_physx_wheel_target_mismatch_fails_closed(self):
        snapshot = _snapshot()
        wheel = WHEEL_JOINT_NAMES[0]
        snapshot["wheel_target_readback_velocity_by_name"][wheel] = 1.0
        snapshot["wheel_target_command_to_readback_error_by_name"][wheel] = 1.0
        result = evaluate_motion_start_ready(
            ground_reference=_ground(),
            snapshot=snapshot,
            production_runtime_ready=True,
            expected_sim_step=180,
            expected_adapter_runtime_instance_id="clean-adapter-1",
            plan_identity=_plan_identity(),
        )
        self.assertFalse(result["ready"])
        self.assertIn("wheel_targets_zero_and_physx_verified", result["failed_checks"])

    def test_nonwheel_contact_penetration_or_stale_snapshot_blocks(self):
        cases = {}
        nonwheel = _snapshot()
        nonwheel["nonwheel_contacts"] = [{"body_name": "base_link", "active": True}]
        cases["nonwheel"] = nonwheel
        penetration = _snapshot()
        penetration["penetration"]["maximum_collision_penetration_m"] = 0.004
        cases["penetration"] = penetration
        stale = _snapshot()
        stale["sim_step"] = 179
        cases["stale"] = stale
        for label, snapshot in cases.items():
            with self.subTest(label=label):
                result = evaluate_motion_start_ready(
                    ground_reference=_ground(),
                    snapshot=snapshot,
                    production_runtime_ready=True,
                    expected_sim_step=180,
                    expected_adapter_runtime_instance_id="clean-adapter-1",
                    plan_identity=_plan_identity(),
                )
                self.assertFalse(result["ready"])

    def test_historical_root_seed_and_missing_support_block(self):
        snapshot = _snapshot()
        snapshot["wheel_net_forces_w"]["RL"] = [0.0, 0.0, 0.0]
        result = evaluate_motion_start_ready(
            ground_reference=_ground(),
            snapshot=snapshot,
            production_runtime_ready=True,
            expected_sim_step=180,
            expected_adapter_runtime_instance_id="clean-adapter-1",
            plan_identity=_plan_identity(),
            root_seed_applied=True,
        )
        self.assertFalse(result["ready"])
        self.assertIn("no_historical_root_seed", result["failed_checks"])
        self.assertNotIn("four_wheel_force_evidence_complete", result["failed_checks"])

    def test_root_write_or_unbound_plan_identity_blocks(self):
        snapshot = _snapshot()
        snapshot["root_state_write_count"] = 1
        snapshot["root_state_write_events"] = [
            {"operation": "respawn_robot", "sim_step": 180}
        ]
        result = evaluate_motion_start_ready(
            ground_reference=_ground(),
            snapshot=snapshot,
            production_runtime_ready=True,
            expected_sim_step=180,
            expected_adapter_runtime_instance_id="clean-adapter-1",
            plan_identity={},
        )
        self.assertFalse(result["ready"])
        self.assertIn("no_historical_root_seed", result["failed_checks"])
        self.assertIn(
            "plan_identity_bound_and_no_prior_dispatch", result["failed_checks"]
        )


if __name__ == "__main__":
    unittest.main()
