from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    COLLIDER_CONFIRMED_MISSING,
    COLLIDER_RESOLUTION_FAILED,
    GROUND_OK,
    resolve_wheel_collision_geometry,
)


class Rel:
    def __init__(self, target: str) -> None:
        self.target = target

    def GetTargets(self) -> list[str]:
        return [self.target] if self.target else []


class Attr:
    def __init__(self, value: Any) -> None:
        self.value = value

    def Get(self) -> Any:
        return self.value


class Prim:
    def __init__(
        self,
        path: str,
        *,
        children: list["Prim"] | None = None,
        rels: dict[str, str] | None = None,
        collision: bool = False,
        enabled: bool = True,
        min_z: float = 0.01,
    ) -> None:
        self.path = path
        self.children = list(children or [])
        self.rels = dict(rels or {})
        self.collision = collision
        self.enabled = enabled
        self.min_z = min_z

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def GetChildren(self) -> list["Prim"]:
        return list(self.children)

    def GetFilteredChildren(self, _predicate: object) -> list["Prim"]:
        return list(self.children)

    def GetRelationship(self, name: str) -> Rel:
        return Rel(self.rels.get(name, ""))

    def HasAPI(self, _api: object) -> bool:
        return bool(self.collision)

    def GetAttribute(self, name: str) -> Attr | None:
        if name == "physics:collisionEnabled" and self.collision:
            return Attr(self.enabled)
        return None

    def GetAppliedSchemas(self) -> list[str]:
        return ["UsdPhysics.CollisionAPI"] if self.collision else []

    def GetTypeName(self) -> str:
        return "Mesh"


class Stage:
    def __init__(self, root: Prim) -> None:
        self.by_path: dict[str, Prim] = {}
        self._index(root)

    def _index(self, prim: Prim) -> None:
        self.by_path[prim.path] = prim
        for child in prim.children:
            self._index(child)

    def GetPrimAtPath(self, path: str) -> Prim | None:
        return self.by_path.get(path)


class BBox:
    def __init__(self, prim: Prim) -> None:
        self.prim = prim

    def ComputeAlignedBox(self) -> "BBox":
        return self

    def GetMin(self) -> tuple[float, float, float]:
        return (0.0, 0.0, self.prim.min_z)

    def GetMax(self) -> tuple[float, float, float]:
        return (0.1, 0.1, self.prim.min_z + 0.05)


class BBoxCache:
    def ComputeWorldBound(self, prim: Prim) -> BBox:
        return BBox(prim)


class WheelCollisionResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_pxr = sys.modules.get("pxr")
        pxr = types.ModuleType("pxr")
        pxr.Usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: 0), TraverseInstanceProxies=lambda: object())
        pxr.UsdGeom = SimpleNamespace(Tokens=SimpleNamespace(default_="default", render="render", proxy="proxy"), BBoxCache=lambda *_a, **_k: BBoxCache())
        pxr.UsdPhysics = SimpleNamespace(CollisionAPI=object)
        pxr.PhysxSchema = SimpleNamespace(PhysxCollisionAPI=object)
        sys.modules["pxr"] = pxr

    def tearDown(self) -> None:
        if self._saved_pxr is None:
            sys.modules.pop("pxr", None)
        else:
            sys.modules["pxr"] = self._saved_pxr

    def test_collider_on_child_body1_joint_name_mismatch(self) -> None:
        collider = Prim("/World/WLRRobot/front_left_wheel/collider", collision=True, min_z=0.004)
        body = Prim("/World/WLRRobot/front_left_wheel", children=[collider])
        joint = Prim(
            "/World/WLRRobot/joints/fl_drive",
            rels={"physics:body0": "/World/WLRRobot/base_link", "physics:body1": "/World/WLRRobot/front_left_wheel"},
        )
        root = Prim("/World/WLRRobot", children=[joint, body])
        scene = SimpleNamespace(stage=Stage(root), robot_prim_path="/World/WLRRobot", config=SimpleNamespace(ground_z_m=0.0))
        adapter = SimpleNamespace(wheel_name_to_id={"front_left_ankle": 0}, robot=SimpleNamespace(body_names=["front_left_wheel"]))

        row = resolve_wheel_collision_geometry(scene, adapter, "front_left_ankle", ground_z_m=0.0, bbox_cache=BBoxCache())

        self.assertEqual(row.collision_resolution_state, GROUND_OK)
        self.assertEqual(row.body1_path, "/World/WLRRobot/front_left_wheel")
        self.assertEqual(row.collision_prim_paths, ["/World/WLRRobot/front_left_wheel/collider"])
        self.assertAlmostEqual(row.clearance_m or 0.0, 0.004)

    def test_true_missing_is_confirmed_missing(self) -> None:
        body = Prim("/World/WLRRobot/front_left_wheel")
        joint = Prim("/World/WLRRobot/front_left_ankle", rels={"physics:body1": "/World/WLRRobot/front_left_wheel"})
        root = Prim("/World/WLRRobot", children=[joint, body])
        scene = SimpleNamespace(stage=Stage(root), robot_prim_path="/World/WLRRobot", config=SimpleNamespace(ground_z_m=0.0))
        adapter = SimpleNamespace(wheel_name_to_id={"front_left_ankle": 0}, robot=SimpleNamespace(body_names=[]))

        row = resolve_wheel_collision_geometry(scene, adapter, "front_left_ankle", ground_z_m=0.0, bbox_cache=BBoxCache())

        self.assertEqual(row.collision_resolution_state, COLLIDER_CONFIRMED_MISSING)

    def test_resolver_failure_is_not_confirmed_missing(self) -> None:
        root = Prim("/World/WLRRobot")
        scene = SimpleNamespace(stage=Stage(root), robot_prim_path="/World/WLRRobot", config=SimpleNamespace(ground_z_m=0.0))
        adapter = SimpleNamespace(wheel_name_to_id={}, robot=SimpleNamespace(body_names=[]))

        row = resolve_wheel_collision_geometry(scene, adapter, "front_left_ankle", ground_z_m=0.0, bbox_cache=BBoxCache())

        self.assertEqual(row.collision_resolution_state, COLLIDER_RESOLUTION_FAILED)
        self.assertEqual(row.collision_prim_paths, [])


if __name__ == "__main__":
    unittest.main()
