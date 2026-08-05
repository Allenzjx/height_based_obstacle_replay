from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import _score_wheel_body_candidate  # noqa: E402


class FakeAttr:
    def Get(self) -> bool:
        return True


class FakeCollisionAPI:
    def __init__(self, prim: "FakePrim") -> None:
        self.prim = prim

    def GetCollisionEnabledAttr(self) -> FakeAttr:
        return FakeAttr()


class FakeRigidBodyAPI:
    pass


class FakeUsdPhysics:
    CollisionAPI = FakeCollisionAPI
    RigidBodyAPI = FakeRigidBodyAPI


class FakePrim:
    def __init__(self, path: str, *, rigid: bool = False, collision: bool = False, children: list["FakePrim"] | None = None) -> None:
        self.path = path
        self.rigid = rigid
        self.collision = collision
        self.children = list(children or [])

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.path.rstrip("/").split("/")[-1]

    def IsValid(self) -> bool:
        return True

    def HasAPI(self, api: object) -> bool:
        if api is FakeRigidBodyAPI:
            return self.rigid
        if api is FakeCollisionAPI:
            return self.collision
        return False

    def GetChildren(self) -> list["FakePrim"]:
        return self.children

    def GetAttribute(self, _name: str) -> FakeAttr:
        return FakeAttr()


class WheelBodyResolutionScoringTest(unittest.TestCase):
    def test_body1_with_rigid_body_and_collider_beats_base_body0(self) -> None:
        collider = FakePrim("/World/WLRRobot/front_left_wheel/collider", collision=True)
        wheel = FakePrim("/World/WLRRobot/front_left_wheel", rigid=True, children=[collider])
        base = FakePrim("/World/WLRRobot/base_link", rigid=True)

        wheel_score = _score_wheel_body_candidate(
            wheel,
            joint_name="front_left_ankle",
            root_path="/World/WLRRobot",
            body0_path="/World/WLRRobot/base_link",
            body1_path="/World/WLRRobot/front_left_wheel",
            UsdPhysics=FakeUsdPhysics,
            PhysxSchema=None,
        )
        base_score = _score_wheel_body_candidate(
            base,
            joint_name="front_left_ankle",
            root_path="/World/WLRRobot",
            body0_path="/World/WLRRobot/base_link",
            body1_path="/World/WLRRobot/front_left_wheel",
            UsdPhysics=FakeUsdPhysics,
            PhysxSchema=None,
        )

        self.assertGreater(wheel_score["score"], base_score["score"])
        self.assertTrue(wheel_score["has_rigid_body_api"])
        self.assertIn("/World/WLRRobot/front_left_wheel/collider", wheel_score["enabled_collider_paths"])


if __name__ == "__main__":
    unittest.main()
