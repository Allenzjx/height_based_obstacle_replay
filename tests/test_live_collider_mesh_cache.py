from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    _cached_collider_mesh_body_local_aabb,
    live_collider_mesh_world_aabb_from_body_pose,
)


class FakePrim:
    def __init__(self, path: str, stage: object | None = None) -> None:
        self.path = path
        self.stage = stage

    def GetPath(self) -> str:
        return self.path

    def GetStage(self) -> object | None:
        return self.stage

    def IsValid(self) -> bool:
        return True


class FakeStage:
    def __init__(self, body: FakePrim) -> None:
        self.body = body

    def GetPrimAtPath(self, _path: str) -> FakePrim:
        return self.body


def geometry(*, valid: bool = True, cacheable: bool = True) -> dict[str, object]:
    return {
        "valid": valid,
        "empty": False,
        "min": [-0.1, 0.0, 0.0],
        "max": [0.1, 0.0, 0.0],
        "extent": [0.2, 0.0, 0.0],
        "finite": valid,
        "source": "live_body_mesh_points",
        "rejection_reason": "" if valid else "not ready",
        "mesh_point_count": 2,
        "mesh_prim_paths": ["/mesh"],
        "cacheable_static_mesh_points": cacheable,
        "_body_points": [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
    }


class LiveColliderMeshCacheTest(unittest.TestCase):
    def test_valid_static_geometry_is_cached_per_adapter_and_not_mutable(self) -> None:
        adapter = SimpleNamespace(_collider_body_local_geometry_cache={"stage": None, "entries": {}})
        stage = object()
        body = FakePrim("/body", stage)
        collider = FakePrim("/body/collider", stage)
        with patch("robot_ground_diagnostics.collider_mesh_body_local_aabb", return_value=geometry()) as compute:
            first = _cached_collider_mesh_body_local_aabb(adapter, stage, body, collider)
            first["min"][0] = 99.0
            second = _cached_collider_mesh_body_local_aabb(adapter, stage, body, collider)

        self.assertEqual(compute.call_count, 1)
        self.assertEqual(second["min"], [-0.1, 0.0, 0.0])
        self.assertIsInstance(second["_body_points"], tuple)

    def test_invalid_or_extent_fallback_is_not_cached_and_can_recover(self) -> None:
        adapter = SimpleNamespace(_collider_body_local_geometry_cache={"stage": None, "entries": {}})
        stage = object()
        body = FakePrim("/body", stage)
        collider = FakePrim("/body/collider", stage)
        with patch(
            "robot_ground_diagnostics.collider_mesh_body_local_aabb",
            side_effect=[geometry(valid=False), geometry(cacheable=False), geometry()],
        ) as compute:
            self.assertFalse(_cached_collider_mesh_body_local_aabb(adapter, stage, body, collider)["valid"])
            self.assertFalse(_cached_collider_mesh_body_local_aabb(adapter, stage, body, collider)["cacheable_static_mesh_points"])
            self.assertTrue(_cached_collider_mesh_body_local_aabb(adapter, stage, body, collider)["valid"])
            self.assertTrue(_cached_collider_mesh_body_local_aabb(adapter, stage, body, collider)["valid"])

        self.assertEqual(compute.call_count, 3)

    def test_cache_is_cleared_when_stage_identity_changes(self) -> None:
        adapter = SimpleNamespace(_collider_body_local_geometry_cache={"stage": None, "entries": {}})
        body = FakePrim("/body")
        collider = FakePrim("/body/collider")
        with patch("robot_ground_diagnostics.collider_mesh_body_local_aabb", return_value=geometry()) as compute:
            _cached_collider_mesh_body_local_aabb(adapter, object(), body, collider)
            _cached_collider_mesh_body_local_aabb(adapter, object(), body, collider)
        self.assertEqual(compute.call_count, 2)

    def test_two_adapters_do_not_share_cached_geometry(self) -> None:
        first_adapter = SimpleNamespace(_collider_body_local_geometry_cache={"stage": None, "entries": {}})
        second_adapter = SimpleNamespace(_collider_body_local_geometry_cache={"stage": None, "entries": {}})
        stage = object()
        body = FakePrim("/body", stage)
        collider = FakePrim("/body/collider", stage)
        with patch("robot_ground_diagnostics.collider_mesh_body_local_aabb", return_value=geometry()) as compute:
            _cached_collider_mesh_body_local_aabb(first_adapter, stage, body, collider)
            _cached_collider_mesh_body_local_aabb(second_adapter, stage, body, collider)
        self.assertEqual(compute.call_count, 2)

    def test_cached_points_still_follow_live_body_pose(self) -> None:
        body = FakePrim("/World/WLRRobot/front_left_wheel")
        stage = FakeStage(body)
        collider = FakePrim("/World/WLRRobot/front_left_wheel/collider", stage)
        adapter = SimpleNamespace(
            _collider_body_local_geometry_cache={"stage": None, "entries": {}},
            robot=SimpleNamespace(
                body_names=["front_left_wheel"],
                data=SimpleNamespace(
                    body_pos_w=[[[0.0, 0.0, 1.0]]],
                    body_quat_w=[[[1.0, 0.0, 0.0, 0.0]]],
                ),
            ),
        )
        with patch("robot_ground_diagnostics.collider_mesh_body_local_aabb", return_value=geometry()) as compute:
            first = live_collider_mesh_world_aabb_from_body_pose(adapter, body.path, collider)
            adapter.robot.data.body_pos_w[0][0][2] = 2.0
            second = live_collider_mesh_world_aabb_from_body_pose(adapter, body.path, collider)

        self.assertEqual(compute.call_count, 1)
        self.assertAlmostEqual(second["min"][2] - first["min"][2], 1.0)


if __name__ == "__main__":
    unittest.main()
