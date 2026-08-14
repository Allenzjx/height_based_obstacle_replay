from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FSM_ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, FSM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from physx_contact_separation import (  # noqa: E402
    PHYSX_SEPARATION_SOURCE,
    PhysxContactSeparationEvidenceError,
    PhysxContactSeparationLayoutError,
    decode_contact_pair_separations,
    separation_evidence_summary,
    unknown_contact_pair_rows,
)


FILTERS = (
    ("ground", "/World/defaultGroundPlane/GroundPlane/CollisionPlane"),
    ("obstacle", "/World/Obstacle"),
)


def _decode(**overrides):
    args = {
        "dt_s": 1.0 / 120.0,
        "distances": np.asarray([[-0.001], [0.0002], [-0.004], [np.nan]]),
        "counts": np.asarray([[2, 1]]),
        "starts": np.asarray([[0, 2]]),
        "env_count": 1,
        "body_class": "wheel",
        "body_name": "front_left_wheel",
        "body_prim_path": "/World/WLRRobot/front_left_wheel",
        "filters": FILTERS,
        "configured_filter_paths": [path for _name, path in FILTERS],
        "expected_sensor_paths": ["/World/WLRRobot/front_left_wheel"],
        "view_sensor_paths": ["/World/WLRRobot/front_left_wheel"],
        "view_filter_paths": [[path for _name, path in FILTERS]],
        "view_filter_count": 2,
        "view_max_contact_data_count": 4,
        "leg": "FL",
    }
    args.update(overrides)
    if "view_max_contact_data_count" not in overrides:
        args["view_max_contact_data_count"] = int(args["distances"].shape[0])
    return decode_contact_pair_separations(**args)


class PhysxContactSeparationTests(unittest.TestCase):
    def test_signed_distances_preserve_filter_identity_and_sign(self) -> None:
        rows = _decode()
        self.assertEqual(len(rows), 2)
        ground, obstacle = rows
        self.assertEqual(ground["surface"], "ground")
        self.assertEqual(ground["contact_count"], 2)
        self.assertEqual(ground["signed_separations_m"], [-0.001, 0.0002])
        self.assertAlmostEqual(ground["minimum_signed_separation_m"], -0.001)
        self.assertAlmostEqual(ground["maximum_penetration_m"], 0.001)
        self.assertEqual(obstacle["surface"], "obstacle")
        self.assertAlmostEqual(obstacle["maximum_penetration_m"], 0.004)
        self.assertEqual(obstacle["source"], PHYSX_SEPARATION_SOURCE)
        self.assertTrue(all(row["valid"] for row in rows))

    def test_two_environments_decode_each_pair_once(self) -> None:
        rows = _decode(
            distances=np.asarray(
                [[-0.001], [0.001], [-0.002], [0.002], [99.0]]
            ),
            counts=np.asarray([[1, 1], [1, 1]]),
            starts=np.asarray([[0, 1], [2, 3]]),
            env_count=2,
            expected_sensor_paths=[
                "/World/envs/env_0/WLRRobot/front_left_wheel",
                "/World/envs/env_1/WLRRobot/front_left_wheel",
            ],
            view_sensor_paths=[
                "/World/envs/env_0/WLRRobot/front_left_wheel",
                "/World/envs/env_1/WLRRobot/front_left_wheel",
            ],
            view_filter_paths=[
                [path for _name, path in FILTERS],
                [path for _name, path in FILTERS],
            ],
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["env_id"], row["surface"]) for row in rows},
            {(0, "ground"), (0, "obstacle"), (1, "ground"), (1, "obstacle")},
        )
        self.assertEqual(len({row["pair_id"] for row in rows}), 4)

    def test_structurally_valid_zero_contact_is_valid_zero(self) -> None:
        rows = _decode(
            distances=np.asarray([[np.nan], [np.nan]]),
            counts=np.asarray([[0, 0]]),
            starts=np.asarray([[0, 0]]),
        )
        for row in rows:
            self.assertTrue(row["valid"])
            self.assertEqual(row["status"], "NO_CONTACT")
            self.assertEqual(row["signed_separations_m"], [])
            self.assertIsNone(row["minimum_signed_separation_m"])
            self.assertEqual(row["maximum_penetration_m"], 0.0)

    def test_only_referenced_distances_must_be_finite(self) -> None:
        rows = _decode()
        self.assertTrue(all(row["valid"] for row in rows))
        with self.assertRaises(PhysxContactSeparationEvidenceError):
            _decode(
                distances=np.asarray([[-0.001], [np.nan], [-0.004], [0.0]])
            )

    def test_positive_only_distance_is_not_penetration(self) -> None:
        rows = _decode(
            distances=np.asarray([[0.005], [0.002], [0.010], [9.0]]),
        )
        self.assertEqual(rows[0]["minimum_signed_separation_m"], 0.002)
        self.assertEqual(rows[0]["maximum_penetration_m"], 0.0)
        self.assertEqual(rows[1]["maximum_penetration_m"], 0.0)

    def test_layout_and_filter_identity_errors_fail_closed(self) -> None:
        bad_cases = (
            ("distance-rank", {"distances": np.zeros((4,))}),
            ("count-rank", {"counts": np.zeros((1, 2, 1))}),
            ("start-shape", {"starts": np.zeros((2, 2)), "env_count": 1}),
            (
                "filter-order",
                {"configured_filter_paths": list(reversed([p for _n, p in FILTERS]))},
            ),
            ("filter-count", {"view_filter_count": 1}),
            (
                "live-filter-order",
                {
                    "view_filter_paths": [
                        list(reversed([p for _n, p in FILTERS]))
                    ]
                },
            ),
            (
                "live-filter-flat-not-matrix",
                {"view_filter_paths": [p for _n, p in FILTERS]},
            ),
            ("live-filter-outer-count", {"view_filter_paths": []}),
            (
                "live-filter-missing-column",
                {"view_filter_paths": [[FILTERS[0][1]]]},
            ),
            (
                "live-filter-extra-column",
                {
                    "view_filter_paths": [
                        [p for _n, p in FILTERS] + ["/World/Extra"]
                    ]
                },
            ),
            (
                "live-filter-relative",
                {
                    "view_filter_paths": [
                        [FILTERS[0][1], "World/Obstacle"]
                    ]
                },
            ),
            (
                "live-filter-duplicate",
                {
                    "view_filter_paths": [
                        [FILTERS[0][1], FILTERS[0][1]]
                    ]
                },
            ),
            (
                "live-sensor-identity",
                {"view_sensor_paths": ["/World/WLRRobot/front_right_wheel"]},
            ),
            ("view-capacity", {"view_max_contact_data_count": 5}),
            ("fractional-count", {"counts": np.asarray([[0.5, 0]])}),
            ("negative-count", {"counts": np.asarray([[-1, 0]])}),
            ("out-of-bounds", {"starts": np.asarray([[0, 5]])}),
            ("overlap", {"starts": np.asarray([[0, 1]])}),
        )
        for label, overrides in bad_cases:
            with self.subTest(label=label):
                with self.assertRaises(PhysxContactSeparationLayoutError):
                    if label == "overlap":
                        # Both non-empty intervals begin in the first range.
                        _decode(
                            distances=np.zeros((5, 1)),
                            counts=np.asarray([[2, 2]]),
                            starts=np.asarray([[0, 1]]),
                        )
                    else:
                        _decode(**overrides)

    def test_capacity_exhaustion_is_unknown_not_safe(self) -> None:
        shared_capacity_not_full = _decode(
            counts=np.asarray([[3, 0]]),
            starts=np.asarray([[0, 3]]),
        )
        self.assertTrue(all(row["valid"] for row in shared_capacity_not_full))
        self.assertTrue(
            all(not row["capacity_exhausted"] for row in shared_capacity_not_full)
        )

        full_buffer = _decode(
            distances=np.asarray([[-0.001], [0.001], [-0.004]]),
        )
        self.assertTrue(all(not row["valid"] for row in full_buffer))
        self.assertTrue(all(row["status"] == "UNKNOWN" for row in full_buffer))
        self.assertTrue(all(row["maximum_penetration_m"] is None for row in full_buffer))

    def test_nonwheel_exact_obstacle_pair(self) -> None:
        rows = _decode(
            distances=np.asarray([[-0.002], [8.0]]),
            counts=np.asarray([[1]]),
            starts=np.asarray([[0]]),
            body_class="nonwheel",
            body_name="base_link",
            body_prim_path="/World/WLRRobot/base_link",
            filters=(("obstacle", "/World/Obstacle"),),
            configured_filter_paths=["/World/Obstacle"],
            expected_sensor_paths=["/World/WLRRobot/base_link"],
            view_sensor_paths=["/World/WLRRobot/base_link"],
            view_filter_paths=[["/World/Obstacle"]],
            view_filter_count=1,
            leg=None,
        )
        self.assertEqual(rows[0]["body_class"], "nonwheel")
        self.assertIsNone(rows[0]["leg"])
        self.assertEqual(rows[0]["maximum_penetration_m"], 0.002)

    def test_unknown_rows_never_claim_zero(self) -> None:
        rows = unknown_contact_pair_rows(
            env_count=1,
            body_class="wheel",
            body_name="front_left_wheel",
            body_prim_path="/World/WLRRobot/front_left_wheel",
            filters=FILTERS,
            leg="FL",
            error="contact view unavailable",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["valid"] is False for row in rows))
        self.assertTrue(all(row["maximum_penetration_m"] is None for row in rows))
        self.assertTrue(all(row["status"] == "UNKNOWN" for row in rows))

    def test_summary_requires_exact_unique_valid_rows(self) -> None:
        rows = _decode()
        expected = [row["pair_id"] for row in rows]
        summary = separation_evidence_summary(rows, expected_pair_ids=expected)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["maximum_physx_penetration_m"], 0.004)
        self.assertEqual(summary["maximum_by_scope_m"]["wheel_ground"], 0.001)
        self.assertEqual(summary["maximum_by_scope_m"]["wheel_obstacle"], 0.004)

        duplicated = separation_evidence_summary(
            [rows[0], rows[0]], expected_pair_ids=expected
        )
        self.assertFalse(duplicated["valid"])
        self.assertIsNone(duplicated["maximum_physx_penetration_m"])


if __name__ == "__main__":
    unittest.main()
