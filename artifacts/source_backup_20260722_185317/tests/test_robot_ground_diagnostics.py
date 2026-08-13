from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    COLLISION_PENETRATION,
    GROUND_OK,
    MISSING_WHEEL_COLLISION,
    VISUAL_ONLY_INTERSECTION,
    WheelGroundDiagnostics,
    build_robot_ground_diagnostics,
    can_change_camera_view,
    capture_physics_state_signature,
    compare_physics_signatures,
    compute_bounded_ground_correction,
)


class RobotGroundDiagnosticsTest(unittest.TestCase):
    def test_visual_only_intersection_does_not_allow_correction(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_prim_paths=["/collision"],
                    collision_api_present=True,
                    collision_enabled=True,
                    visual_aabb_min_z=-0.01,
                    collision_aabb_min_z=0.001,
                    visual_ground_clearance_m=-0.01,
                    collision_ground_clearance_m=0.001,
                )
            ],
            penetration_tolerance_m=0.003,
        )

        allowed, dz, reason = compute_bounded_ground_correction(diag)

        self.assertEqual(diag.classification, VISUAL_ONLY_INTERSECTION)
        self.assertFalse(allowed)
        self.assertEqual(dz, 0.0)
        self.assertIn("COLLISION_PENETRATION", reason)

    def test_collision_penetration_computes_bounded_positive_correction(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_prim_paths=["/collision"],
                    collision_api_present=True,
                    collision_enabled=True,
                    collision_aabb_min_z=-0.02,
                    collision_ground_clearance_m=-0.02,
                    collision_penetration_m=0.02,
                )
            ],
            penetration_tolerance_m=0.003,
        )

        allowed, dz, _reason = compute_bounded_ground_correction(diag, target_clearance_m=0.002, max_correction_m=0.01)

        self.assertEqual(diag.classification, COLLISION_PENETRATION)
        self.assertTrue(allowed)
        self.assertAlmostEqual(dz, 0.01)

    def test_missing_wheel_collision_is_explicit_failure(self) -> None:
        diag = build_robot_ground_diagnostics(
            [WheelGroundDiagnostics(wheel_name="front_left_ankle", visual_aabb_min_z=0.0, visual_ground_clearance_m=0.0)]
        )

        self.assertEqual(diag.classification, MISSING_WHEEL_COLLISION)
        self.assertEqual(diag.missing_collision_wheels, ["front_left_ankle"])

    def test_ok_classification_when_collision_clearance_is_non_negative(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_prim_paths=["/collision"],
                    collision_api_present=True,
                    collision_enabled=True,
                    visual_ground_clearance_m=0.001,
                    collision_ground_clearance_m=0.001,
                )
            ]
        )

        self.assertEqual(diag.classification, GROUND_OK)

    def test_physics_signature_detects_no_change_and_delta(self) -> None:
        before = capture_physics_state_signature(FakeAdapter(root_z=0.04, joint_pos=[0.0, 0.0]))
        after_same = capture_physics_state_signature(FakeAdapter(root_z=0.04, joint_pos=[0.0, 0.0]))
        after_changed = capture_physics_state_signature(FakeAdapter(root_z=0.04001, joint_pos=[0.0, 0.0]))

        same = compare_physics_signatures(before, after_same)
        changed = compare_physics_signatures(before, after_changed)

        self.assertTrue(same.passed)
        self.assertFalse(changed.passed)
        self.assertGreater(changed.root_pose_delta_m, 1.0e-6)

    def test_camera_view_idle_guard_blocks_motion_and_pending_work(self) -> None:
        guard = can_change_camera_view(
            flags={"recording_active": False, "playback_active": False, "pending_step": True},
            wheel_command_values={"front_left_ankle": 0.1},
        )

        self.assertFalse(guard.allowed)
        self.assertTrue(any("pending" in reason for reason in guard.reasons))
        self.assertTrue(any("wheel" in reason for reason in guard.reasons))


class FakeData:
    def __init__(self, root_z: float, joint_pos: list[float]) -> None:
        self.root_pose_w = [[0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0]]
        self.root_vel_w = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.joint_pos = [joint_pos]
        self.joint_vel = [[0.0 for _ in joint_pos]]


class FakeRobot:
    def __init__(self, root_z: float, joint_pos: list[float]) -> None:
        self.data = FakeData(root_z, joint_pos)


class FakeAdapter:
    def __init__(self, *, root_z: float, joint_pos: list[float]) -> None:
        self.robot = FakeRobot(root_z, joint_pos)
        self.sim_time = 0.0
        self.sim_steps = 0

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return {"servos": {}, "wheels": {}}


if __name__ == "__main__":
    unittest.main()
