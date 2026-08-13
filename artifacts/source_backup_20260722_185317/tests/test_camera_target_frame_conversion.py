from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_onboard_camera import _world_point_to_parent_local  # noqa: E402


class FakeVec3d:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.values = (float(x), float(y), float(z))

    def __getitem__(self, index: int) -> float:
        return self.values[index]


class FakeInverse:
    translation = (0.0, 0.0, 0.0)
    yaw_rad = 0.0

    def Transform(self, vec: FakeVec3d) -> FakeVec3d:
        x = vec[0] - self.translation[0]
        y = vec[1] - self.translation[1]
        z = vec[2] - self.translation[2]
        c = math.cos(-self.yaw_rad)
        s = math.sin(-self.yaw_rad)
        return FakeVec3d(c * x - s * y, s * x + c * y, z)


class FakeMatrix:
    def GetInverse(self) -> FakeInverse:
        return FakeInverse()


class FakeXformCache:
    def GetLocalToWorldTransform(self, _prim: object) -> FakeMatrix:
        return FakeMatrix()


class FakeStage:
    def GetPrimAtPath(self, path: str) -> object:
        return {"path": path}


class CameraTargetFrameConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {name: sys.modules.get(name) for name in ("pxr", "pxr.Gf", "pxr.UsdGeom")}
        pxr = types.ModuleType("pxr")
        gf = types.ModuleType("pxr.Gf")
        usd_geom = types.ModuleType("pxr.UsdGeom")
        gf.Vec3d = FakeVec3d  # type: ignore[attr-defined]
        usd_geom.XformCache = FakeXformCache  # type: ignore[attr-defined]
        pxr.Gf = gf  # type: ignore[attr-defined]
        pxr.UsdGeom = usd_geom  # type: ignore[attr-defined]
        sys.modules["pxr"] = pxr
        sys.modules["pxr.Gf"] = gf
        sys.modules["pxr.UsdGeom"] = usd_geom

    def tearDown(self) -> None:
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_world_target_parent_translation(self) -> None:
        FakeInverse.translation = (1.0, 2.0, 0.5)
        FakeInverse.yaw_rad = 0.0

        local = _world_point_to_parent_local(FakeStage(), "/World/WLRRobot/base_link", (2.5, 2.0, 0.7))

        self.assertAlmostEqual(local[0], 1.5)
        self.assertAlmostEqual(local[1], 0.0)
        self.assertAlmostEqual(local[2], 0.2)

    def test_world_target_parent_yaw(self) -> None:
        FakeInverse.translation = (0.0, 0.0, 0.0)
        FakeInverse.yaw_rad = math.pi / 2.0

        local = _world_point_to_parent_local(FakeStage(), "/World/WLRRobot/base_link", (0.0, 1.0, 0.0))

        self.assertAlmostEqual(local[0], 1.0, places=6)
        self.assertAlmostEqual(local[1], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
