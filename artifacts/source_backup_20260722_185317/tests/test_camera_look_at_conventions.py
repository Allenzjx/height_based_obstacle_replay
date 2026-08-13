from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_onboard_camera import _rotate_direction, camera_look_at_quat_wxyz  # noqa: E402


def dot_to_x(vector: object) -> float:
    values = [float(v) for v in vector]
    return values[0]


class CameraLookAtConventionsTest(unittest.TestCase):
    def test_world_convention_uses_local_x_forward(self) -> None:
        quat = camera_look_at_quat_wxyz(camera_position=(0.0, 0.0, 0.0), target_position=(1.0, 0.0, 0.0), convention="world")
        self.assertAlmostEqual(dot_to_x(_rotate_direction((1.0, 0.0, 0.0), quat)), 1.0, places=6)

    def test_ros_convention_uses_local_z_forward(self) -> None:
        quat = camera_look_at_quat_wxyz(camera_position=(0.0, 0.0, 0.0), target_position=(1.0, 0.0, 0.0), convention="ros")
        self.assertAlmostEqual(dot_to_x(_rotate_direction((0.0, 0.0, 1.0), quat)), 1.0, places=6)

    def test_opengl_convention_uses_local_negative_z_forward(self) -> None:
        quat = camera_look_at_quat_wxyz(camera_position=(0.0, 0.0, 0.0), target_position=(1.0, 0.0, 0.0), convention="opengl")
        self.assertAlmostEqual(dot_to_x(_rotate_direction((0.0, 0.0, -1.0), quat)), 1.0, places=6)

    def test_old_world_plus_z_assumption_would_not_point_at_target(self) -> None:
        quat = camera_look_at_quat_wxyz(camera_position=(0.0, 0.0, 0.0), target_position=(1.0, 0.0, 0.0), convention="world")
        self.assertLess(dot_to_x(_rotate_direction((0.0, 0.0, 1.0), quat)), 0.5)


if __name__ == "__main__":
    unittest.main()
