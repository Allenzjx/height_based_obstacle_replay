from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import GROUND_PRIM_PATH, resolve_ground_surface  # noqa: E402


class FakePrim:
    def __init__(self, path: str, *, world_translation: tuple[float, float, float], children: list["FakePrim"] | None = None) -> None:
        self.path = path
        self.world_translation = world_translation
        self.local_translation = world_translation
        self.children = list(children or [])

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return self.path

    def GetChildren(self) -> list["FakePrim"]:
        return list(self.children)

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class FakeStage:
    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def GetPrimAtPath(self, path: str) -> FakePrim | None:
        return self.prim if path == GROUND_PRIM_PATH else None


class GroundSurfaceResolutionTest(unittest.TestCase):
    def test_actual_ground_z_comes_from_prim_world_translation(self) -> None:
        collision = FakePrim(f"{GROUND_PRIM_PATH}/CollisionPlane", world_translation=(0.0, 0.0, 0.012))
        ground = FakePrim(GROUND_PRIM_PATH, world_translation=(0.0, 0.0, 0.012), children=[collision])
        scene = SimpleNamespace(stage=FakeStage(ground), config=SimpleNamespace(ground_z_m=0.0))

        info = resolve_ground_surface(scene)

        self.assertEqual(info.actual_ground_z_m, 0.012)
        self.assertEqual(info.configured_ground_z_m, 0.0)
        self.assertAlmostEqual(info.ground_z_delta_m or 0.0, 0.012)
        self.assertFalse(info.ground_resolution_ok)
        self.assertEqual(info.ground_prim_path, GROUND_PRIM_PATH)

    def test_matching_configured_and_actual_ground_z_is_ok(self) -> None:
        ground = FakePrim(GROUND_PRIM_PATH, world_translation=(0.0, 0.0, 0.0))
        scene = SimpleNamespace(stage=FakeStage(ground), config=SimpleNamespace(ground_z_m=0.0))

        info = resolve_ground_surface(scene)

        self.assertTrue(info.ground_resolution_ok)
        self.assertEqual(info.actual_ground_z_m, 0.0)


if __name__ == "__main__":
    unittest.main()
