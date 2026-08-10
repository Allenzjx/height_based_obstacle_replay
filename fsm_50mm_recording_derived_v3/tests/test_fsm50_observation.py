from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from fsm_50mm_recording_derived_v3.fsm50_observation import (
    ContactClass,
    CriticalTelemetryUnavailable,
    FSM50Observation,
    ObservationValidationError,
    PrimaryDiagonal,
    compute_world_com_target_direction,
)


def _canonical_mapping() -> dict[str, object]:
    """Return a valid mutable mapping accepted by from_mapping."""

    mapping = FSM50Observation.fake().to_mapping()
    corridor = dict(mapping.pop("two_leg_corridor"))
    mapping["two_leg_corridor_applicable"] = corridor["applicable"]
    mapping["two_leg_corridor_valid"] = corridor["valid"]
    if corridor["perpendicular_distance_m"] is not None:
        mapping["two_leg_corridor_distance_m"] = corridor[
            "perpendicular_distance_m"
        ]
    if corridor["segment_fraction"] is not None:
        mapping["two_leg_corridor_fraction"] = corridor["segment_fraction"]
    if corridor["within_longitudinal_bounds"] is not None:
        mapping["two_leg_corridor_within_longitudinal_bounds"] = corridor[
            "within_longitudinal_bounds"
        ]
    if corridor["within_corridor_width"] is not None:
        mapping["two_leg_corridor_within_width"] = corridor[
            "within_corridor_width"
        ]
    mapping.pop("control_ready")
    mapping.pop("issues")
    mapping.pop("com_target")
    return mapping


class ObservationConstructionTests(unittest.TestCase):
    def test_fake_is_complete_and_deeply_immutable(self) -> None:
        observation = FSM50Observation.fake()

        self.assertTrue(observation.control_ready)
        self.assertEqual((), observation.issues)
        self.assertEqual(PrimaryDiagonal.BALANCED, observation.primary_diagonal)
        self.assertEqual(ContactClass.GROUND, observation.wheel_contact_class["FL"])
        self.assertEqual({"FL", "FR", "RL", "RR"}, set(observation.wheel_center_w))
        self.assertEqual(observation, observation.require_control_ready())

        with self.assertRaises(FrozenInstanceError):
            observation.time_s = 1.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            observation.wheel_target_rad_s["FL"] = 1.0  # type: ignore[index]

    def test_mapping_copy_does_not_mutate_observation(self) -> None:
        observation = FSM50Observation.fake()
        exported = observation.to_mapping()
        exported["wheel_target_rad_s"]["FL"] = 9.0  # type: ignore[index]
        self.assertEqual(0.0, observation.wheel_target_rad_s["FL"])

    def test_current_telemetry_field_aliases_are_accepted(self) -> None:
        row = _canonical_mapping()
        root_position = row.pop("root_position_w")
        root_orientation = row.pop("root_orientation_wxyz")
        root_linear = row.pop("root_linear_velocity_w")
        root_angular = row.pop("root_angular_velocity_w")
        com_position = row.pop("com_position_w")
        com_velocity = row.pop("com_velocity_w")
        for key, value in zip(("base_x_m", "base_y_m", "base_z_m"), root_position):
            row[key] = value
        for key, value in zip(
            ("base_qw", "base_qx", "base_qy", "base_qz"), root_orientation
        ):
            row[key] = value
        for key, value in zip(
            ("base_vx_m_s", "base_vy_m_s", "base_vz_m_s"), root_linear
        ):
            row[key] = value
        for key, value in zip(
            ("base_wx_rad_s", "base_wy_rad_s", "base_wz_rad_s"), root_angular
        ):
            row[key] = value
        for key, value in zip(("com_x_m", "com_y_m", "com_z_m"), com_position):
            row[key] = value
        for key, value in zip(
            ("com_vx_m_s", "com_vy_m_s", "com_vz_m_s"), com_velocity
        ):
            row[key] = value

        renames = {
            "joint_position_rad": "measured_joint_position_rad",
            "joint_velocity_rad_s": "measured_joint_velocity_rad_s",
            "wheel_target_rad_s": "wheel_command_velocity_rad_s",
            "measured_wheel_velocity_rad_s": (
                "wheel_canonical_forward_velocity_rad_s"
            ),
            "integrated_wheel_rotation_rad": "wheel_integrated_rotation_rad",
            "integrated_wheel_travel_m": "wheel_integrated_travel_m",
            "wheel_center_w": "wheel_centers_w",
            "wheel_contact_point_w": "wheel_contact_points_w",
            "wheel_contact_class": "wheel_contact_classes",
            "contact_drift_m": "wheel_contact_drift_m",
            "support_polygon_margin_m": "wheel_support_polygon_margin_m",
            "support_polygon_valid": "wheel_support_polygon_valid",
            "nonwheel_contact_evidence_valid": "collision_evidence_valid",
            "nonwheel_obstacle_contact": "dangerous_collision",
        }
        for canonical, telemetry_name in renames.items():
            row[telemetry_name] = row.pop(canonical)

        # Exercise wheel-joint aliases rather than canonical leg keys.
        row["wheel_command_velocity_rad_s"] = {
            "front_left_ankle": 0.0,
            "front_right_ankle": 0.0,
            "rear_left_ankle": 0.0,
            "rear_right_ankle": 0.0,
        }
        row["actual_joint_target_rad"] = {
            **row.pop("servo_target_rad"),
            "front_left_ankle": 0.0,
            "front_right_ankle": 0.0,
            "rear_left_ankle": 0.0,
            "rear_right_ankle": 0.0,
        }
        ground = row.pop("filtered_ground_force_n")
        obstacle = row.pop("filtered_obstacle_force_n")
        row["wheel_surface_normal_force_n"] = {
            leg: {"ground": ground[leg], "obstacle": obstacle[leg]}
            for leg in ("FL", "FR", "RL", "RR")
        }
        shares = row.pop("diagonal_load_share")
        row["diagonal_load_share_fl_rr"] = shares["FL_RR"]
        row["diagonal_load_share_fr_rl"] = shares["FR_RL"]

        observation = FSM50Observation.from_mapping(row, strict=True)
        self.assertTrue(observation.control_ready)
        self.assertEqual((0.0, 0.0, 0.10), observation.root_position_w)
        self.assertNotIn("front_left_ankle", observation.servo_target_rad)
        self.assertEqual(set(("FL", "FR", "RL", "RR")), set(observation.wheel_target_rad_s))

    def test_quaternion_is_normalized(self) -> None:
        mapping = _canonical_mapping()
        mapping["root_orientation_wxyz"] = (2.0, 0.0, 0.0, 0.0)
        observation = FSM50Observation.from_mapping(mapping, strict=True)
        self.assertEqual((1.0, 0.0, 0.0, 0.0), observation.root_orientation_wxyz)


class FailClosedValidationTests(unittest.TestCase):
    def test_missing_critical_telemetry_is_reported_and_fails_closed(self) -> None:
        mapping = _canonical_mapping()
        mapping.pop("com_velocity_w")

        observation = FSM50Observation.from_mapping(mapping)
        self.assertFalse(observation.control_ready)
        self.assertIn("com_velocity_w", observation.missing_critical_fields)
        self.assertFalse(observation.guard_allows(True))
        with self.assertRaises(CriticalTelemetryUnavailable):
            observation.require_control_ready()
        with self.assertRaises(ObservationValidationError):
            FSM50Observation.from_mapping(mapping, strict=True)

    def test_nonfinite_shape_and_leg_key_errors_are_all_preserved(self) -> None:
        mapping = _canonical_mapping()
        mapping["root_position_w"] = (0.0, 1.0)
        mapping["com_velocity_w"] = (float("nan"), 0.0, 0.0)
        mapping["wheel_target_rad_s"] = {
            "FL": 0.0,
            "FR": 0.0,
            "RL": 0.0,
            "NOT_A_LEG": 0.0,
        }

        observation = FSM50Observation.from_mapping(mapping)
        issue_pairs = {(issue.field, issue.code) for issue in observation.issues}
        self.assertIn(("root_position_w", "SHAPE_OR_TYPE"), issue_pairs)
        self.assertIn(("com_velocity_w", "NONFINITE"), issue_pairs)
        self.assertIn(("wheel_target_rad_s.NOT_A_LEG", "LEG_KEY"), issue_pairs)
        self.assertIn(("wheel_target_rad_s", "LEG_KEYS"), issue_pairs)
        self.assertFalse(observation.control_ready)

    def test_invalid_nonwheel_evidence_is_not_interpreted_as_no_collision(self) -> None:
        mapping = _canonical_mapping()
        mapping["nonwheel_contact_evidence_valid"] = False
        mapping["nonwheel_obstacle_contact"] = False

        observation = FSM50Observation.from_mapping(mapping)
        self.assertFalse(observation.control_ready)
        self.assertFalse(observation.guard_allows(not observation.nonwheel_obstacle_contact))
        self.assertTrue(
            any(issue.code == "EVIDENCE_UNAVAILABLE" for issue in observation.issues)
        )

    def test_two_leg_corridor_requires_finite_geometry_when_applicable(self) -> None:
        mapping = _canonical_mapping()
        mapping["support_legs"] = ["FL", "RR"]
        mapping["two_leg_corridor_applicable"] = True
        mapping["two_leg_corridor_valid"] = False
        mapping.pop("two_leg_corridor_distance_m", None)
        mapping.pop("two_leg_corridor_fraction", None)

        observation = FSM50Observation.from_mapping(mapping)
        self.assertFalse(observation.control_ready)
        self.assertTrue(observation.two_leg_corridor.applicable)
        self.assertIn(
            "two_leg_corridor_distance_m", observation.missing_critical_fields
        )


class COMTargetDirectionTests(unittest.TestCase):
    def test_world_direction_uses_target_geometry_not_yaml_vector(self) -> None:
        mapping = _canonical_mapping()
        sqrt_half = math.sqrt(0.5)
        mapping["root_orientation_wxyz"] = (sqrt_half, 0.0, 0.0, sqrt_half)
        mapping["com_position_w"] = (0.0, 0.0, 0.08)
        contact_points = dict(mapping["wheel_contact_point_w"])
        contact_points["FR"] = (2.0, 0.0, 0.0)
        mapping["wheel_contact_point_w"] = contact_points
        mapping["target_com_leg"] = "FR"
        # This legacy hint would point world +Y if incorrectly rotated by yaw.
        mapping["target_com_direction"] = (1.0, 0.0)

        observation = FSM50Observation.from_mapping(mapping, strict=True)
        target = observation.com_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertAlmostEqual(1.0, target.direction_w[0], places=12)
        self.assertAlmostEqual(0.0, target.direction_w[1], places=12)
        self.assertAlmostEqual(0.0, target.direction_body[0], places=12)
        self.assertAlmostEqual(-1.0, target.direction_body[1], places=12)
        self.assertEqual("MEASURED_TARGET_CONTACT_GEOMETRY", target.source)

    def test_target_direction_can_be_resolved_for_fake_controller_state(self) -> None:
        observation = FSM50Observation.fake()
        target = observation.target_direction_for("front_left")
        expected_norm = math.hypot(0.30, 0.20)
        self.assertAlmostEqual(0.30 / expected_norm, target.direction_w[0])
        self.assertAlmostEqual(0.20 / expected_norm, target.direction_w[1])

    def test_zero_distance_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero horizontal distance"):
            compute_world_com_target_direction(
                root_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                com_position_w=(1.0, 2.0, 0.1),
                target_contact_w=(1.0, 2.0, 0.0),
                target_leg="FL",
            )


if __name__ == "__main__":
    unittest.main()
