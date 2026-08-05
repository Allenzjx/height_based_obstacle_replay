from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_ui_controller import HeightReplayController  # noqa: E402


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=10,
        robot_usd="C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd",
        save_usd="C:/robotics_sim/wlr_robot/usd/wlr_robot_height_replay_env.usd",
        spawn_z=0.04,
        obstacle_x=1.55,
        obstacle_width=None,
        obstacle_length=None,
        infer_obstacle_size=True,
        robot_width=0.80,
        robot_length=0.55,
        physics_dt=1.0 / 120.0,
        render_interval=2,
        device="cuda:0",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        apply_physx_joint_limits=False,
        ui_refresh_ms=100,
        sim_status_refresh_ms=250,
        full_refresh_ms=1000,
        no_continuous_sim_step=False,
        wheel_direction=1.0,
        servo_stiffness=600.0,
        servo_damping=60.0,
        wheel_damping=20.0,
        save_scene=False,
    )


class SimUiControllerTest(unittest.TestCase):
    def test_record_accept_save_current_height_no_sim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.start_sim_if_needed()
            controller.handle_command("step_record start")
            controller.handle_command("servo front_left_hip 12")
            controller.handle_command("wheel fl 0.5")
            controller.handle_command("step_record stop")
            controller.handle_command("step_record accept")
            self.assertEqual(controller.manager.count, 1)
            path = controller.save_steps_for_current_height()
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).is_file())
            self.assertIn("height_100mm", str(path))
            self.assertIn("versions", str(path))
            self.assertEqual(controller.store.status_rows()[-1]["version_count"], 1)
            self.assertTrue(controller.current_version_id.startswith("v001_"))


if __name__ == "__main__":
    unittest.main()
