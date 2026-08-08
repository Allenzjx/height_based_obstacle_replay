from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from fsm_50mm_recording_derived_v3.recording_alignment import (
    ENDPOINT_CANDIDATE,
    PENDING_REPLAY,
    boundary_snapshot,
    compare_endpoint_boundaries,
    enumerate_recording_directories,
    finite_vector,
    quaternion_distance_rad,
    quaternion_to_rpy_wxyz,
    summarize_fast_segments,
)


JOINT_NAMES = [
    "front_left_hip",
    "front_left_knee",
    "front_left_ankle",
]


def _state(
    *,
    root_x: float = 0.0,
    quaternion: list[float] | None = None,
    names: list[str] | None = None,
    position: list[float] | None = None,
    velocity: list[float] | None = None,
) -> dict:
    joint_names = list(names or JOINT_NAMES)
    joint_position = list(position or [0.1, -0.2, 0.3])
    joint_velocity = list(velocity or [0.01, -0.02, 0.03])
    return {
        "root_pose": [
            [
                root_x,
                0.0,
                0.1,
                *(quaternion or [1.0, 0.0, 0.0, 0.0]),
            ]
        ],
        "root_velocity": [[0.1, 0.0, 0.0, 0.0, 0.2, 0.0]],
        "joint_names": joint_names,
        "joint_pos": [joint_position],
        "joint_vel": [joint_velocity],
        "command_state": {
            "servos": {"front_left_hip": 0.0},
            "wheels": {"front_left_ankle": 0.0},
        },
        "sim_time": 1.0,
    }


class BoundaryMathTests(unittest.TestCase):
    def test_finite_vector_accepts_single_row_and_rejects_bad_values(self) -> None:
        self.assertEqual([1.0, 2.0], finite_vector([[1, 2]], width=2))
        self.assertIsNone(finite_vector([[1, 2]], width=3))
        self.assertIsNone(finite_vector([1, float("nan")]))
        self.assertIsNone(finite_vector("1,2"))

    def test_quaternion_rpy_and_distance_are_sign_invariant(self) -> None:
        yaw = math.pi / 2.0
        quaternion = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
        roll, pitch, measured_yaw = quaternion_to_rpy_wxyz(quaternion)
        self.assertAlmostEqual(0.0, roll, places=12)
        self.assertAlmostEqual(0.0, pitch, places=12)
        self.assertAlmostEqual(yaw, measured_yaw, places=12)
        self.assertAlmostEqual(
            0.0,
            quaternion_distance_rad(
                quaternion, [-value for value in quaternion]
            ),
            places=12,
        )

    def test_boundary_snapshot_does_not_fill_missing_legacy_fields(self) -> None:
        snapshot = boundary_snapshot(
            {
                "root_pose": [],
                "root_velocity": [],
                "joint_names": [],
                "joint_pos": [],
                "joint_vel": [],
            }
        )
        self.assertTrue(snapshot["status"].startswith("MISSING:"))
        self.assertIsNone(snapshot["root_position_m"])
        self.assertIsNone(snapshot["joint_position_rad"])

    def test_compatible_boundary_uses_joint_names_not_array_order(self) -> None:
        previous = _state()
        current = _state(
            names=[
                "front_left_ankle",
                "front_left_knee",
                "front_left_hip",
            ],
            position=[0.3, -0.2, 0.1],
            velocity=[0.03, -0.02, 0.01],
        )
        result = compare_endpoint_boundaries(previous, current)
        self.assertTrue(result["joint_schema_compatible"])
        self.assertEqual(
            "POSE_COMPATIBLE_ENDPOINTS",
            result["endpoint_compatibility"],
        )
        self.assertAlmostEqual(0.0, result["servo_position_gap_deg"])
        self.assertAlmostEqual(0.0, result["wheel_position_gap_rad"])
        self.assertAlmostEqual(0.0, result["joint_velocity_gap_rad_s"])

    def test_root_gap_over_restore_tolerance_is_not_compatible(self) -> None:
        result = compare_endpoint_boundaries(
            _state(root_x=0.0), _state(root_x=0.006)
        )
        self.assertEqual(
            "POSE_GAP_EXCEEDS_RESTORE_TOLERANCE",
            result["endpoint_compatibility"],
        )
        self.assertAlmostEqual(0.006, result["root_position_gap_m"])

    def test_missing_endpoint_remains_explicit(self) -> None:
        result = compare_endpoint_boundaries({}, _state())
        self.assertEqual("MISSING_ENDPOINT", result["endpoint_compatibility"])
        self.assertIsNone(result["root_position_gap_m"])


class FastSegmentSummaryTests(unittest.TestCase):
    def test_segment_range_commands_concurrency_and_wheel_integral(self) -> None:
        rows = [
            {
                "decoded_segment_index": 4,
                "command_start_s": 1.0,
                "command_end_s": 1.5,
                "commands": ["servo front_left_hip 10"],
                "servo_target_deg": {"front_left_hip": 10.0},
                "wheel_target_rad_s": {
                    "front_left_ankle": 0.0,
                    "front_right_ankle": 0.0,
                    "rear_left_ankle": 0.0,
                    "rear_right_ankle": 0.0,
                },
                "concurrent": False,
                "expected_wheel_displacement_rad": {},
            },
            {
                "decoded_segment_index": 5,
                "command_start_s": 1.5,
                "command_end_s": 2.0,
                "commands": ["wheel forward 0.2"],
                "servo_target_deg": {"front_left_hip": 10.0},
                "wheel_target_rad_s": {
                    "front_left_ankle": 0.2,
                    "front_right_ankle": 0.2,
                    "rear_left_ankle": 0.2,
                    "rear_right_ankle": 0.2,
                },
                "concurrent": True,
                "expected_wheel_displacement_rad": {
                    "front_left_ankle": 0.1,
                    "front_right_ankle": 0.1,
                    "rear_left_ankle": 0.1,
                    "rear_right_ankle": 0.1,
                },
            },
        ]
        summary = summarize_fast_segments(
            rows, previous_fast_end_s=0.75
        )
        self.assertEqual("4:5", summary["segment_range"])
        self.assertEqual(2, summary["segment_count"])
        self.assertEqual(
            ["servo front_left_hip 10", "wheel forward 0.2"],
            summary["commands"],
        )
        self.assertEqual("SERVO_WHEEL_COMMANDS", summary["command_shape"])
        self.assertEqual(1, summary["concurrent_segment_count"])
        self.assertAlmostEqual(0.25, summary["gap_from_previous_s"])
        self.assertAlmostEqual(
            0.1,
            summary["expected_wheel_displacement_rad"][
                "front_left_ankle"
            ],
        )

    def test_physical_directory_enumeration_ignores_manifest_only_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            versions = root / "versions"
            valid = versions / "v003"
            invalid = versions / "v004"
            valid.mkdir(parents=True)
            invalid.mkdir(parents=True)
            (valid / "accepted_steps.jsonl").write_text("{}\n")
            (valid / "metadata.json").write_text("{}\n")
            (invalid / "metadata.json").write_text("{}\n")
            self.assertEqual(
                [valid], enumerate_recording_directories(root)
            )

    def test_candidate_labels_cannot_imply_replay_success(self) -> None:
        self.assertEqual("ENDPOINT_CANDIDATE", ENDPOINT_CANDIDATE)
        self.assertEqual("PENDING_REPLAY", PENDING_REPLAY)


if __name__ == "__main__":
    unittest.main()
