from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    live_collider_world_aabb_from_body_pose,
    world_aabb_from_body_local_points,
)


class FakeBound:
    def __init__(self, minimum: list[float], maximum: list[float]) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def IsEmpty(self) -> bool:
        return False

    def GetMin(self) -> list[float]:
        return self.minimum

    def GetMax(self) -> list[float]:
        return self.maximum


class FakeLocalBound:
    def ComputeAlignedBox(self) -> FakeBound:
        return FakeBound([-0.1, -0.1, -0.2], [0.1, 0.1, 0.2])


class FakeBBoxCache:
    def ComputeLocalBound(self, _prim: object) -> FakeLocalBound:
        return FakeLocalBound()


class FakePrim:
    def GetPath(self) -> str:
        return "/World/WLRRobot/front_left_wheel/collider"


class LiveColliderWorldBoundsTest(unittest.TestCase):
    def test_rotated_mesh_uses_real_points_instead_of_rotated_aabb_corners(self) -> None:
        half = 2.0 ** -0.5
        bound = world_aabb_from_body_local_points(
            [0.0, 0.0, 1.0],
            [half, 0.0, half, 0.0],
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, -0.01, 0.0]],
        )

        self.assertTrue(bound["valid"])
        self.assertAlmostEqual(bound["min"][2], 0.95)
        self.assertAlmostEqual(bound["max"][2], 1.05)

    def test_live_body_pose_offsets_local_collider_bound(self) -> None:
        adapter = SimpleNamespace(
            robot=SimpleNamespace(
                body_names=["front_left_wheel"],
                data=SimpleNamespace(
                    body_pos_w=[[[1.0, 2.0, 3.0]]],
                    body_quat_w=[[[1.0, 0.0, 0.0, 0.0]]],
                ),
            )
        )

        bound = live_collider_world_aabb_from_body_pose(
            adapter,
            "/World/WLRRobot/front_left_wheel",
            FakePrim(),
            FakeBBoxCache(),
        )

        self.assertTrue(bound["valid"])
        self.assertAlmostEqual(bound["min"][2], 2.8)
        self.assertIn("live_body_pose_local_bound", bound["source"])


if __name__ == "__main__":
    unittest.main()
