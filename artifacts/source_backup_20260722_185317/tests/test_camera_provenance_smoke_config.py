from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_process_client import build_worker_command, build_worker_config  # noqa: E402
from sim_worker_process import _run_pure_camera_provenance_audit  # noqa: E402


class CameraProvenanceSmokeConfigTest(unittest.TestCase):
    def test_worker_config_and_cli_include_camera_provenance_smoke_flag(self) -> None:
        args = self._args()
        config = build_worker_config(args, host="127.0.0.1", port=4321)
        command = build_worker_command(args, host="127.0.0.1", port=4321)

        self.assertTrue(config["worker_smoke_camera_provenance"])
        self.assertTrue(config["camera_coverage_strict"])
        self.assertEqual(config["camera_aim_mode"], "look-at")
        self.assertIn("--worker-smoke-camera-provenance", command)
        self.assertIn("--camera-coverage-strict", command)
        self.assertIn("--camera-aim-mode", command)
        self.assertIn("look-at", command)

    def test_pure_provenance_audit_covers_metadata_depth_and_roi_checks(self) -> None:
        result = _run_pure_camera_provenance_audit()

        self.assertTrue(result["ok"])
        checks = result["checks"]
        self.assertTrue(checks["metadata_5cm_pointcloud_10cm_outputs_10cm"])
        self.assertTrue(checks["depth_none_invalid"])
        self.assertTrue(checks["generated_x_prior_rejects_moved_obstacle"])
        self.assertTrue(checks["external_auto_roi_accepts_moved_obstacle"])

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            height_cm=5,
            robot_usd="robot.usd",
            save_usd="scene.usd",
            spawn_z=0.04,
            obstacle_x=1.55,
            obstacle_width=None,
            obstacle_length=None,
            infer_obstacle_size=True,
            robot_width=0.8,
            robot_length=0.55,
            physics_dt=1.0 / 120.0,
            render_interval=2,
            wheel_direction=1.0,
            max_wheel_speed_rad_s=3.0,
            default_wheel_speed_rad_s=1.0,
            servo_stiffness=600.0,
            servo_damping=60.0,
            wheel_damping=20.0,
            device="cuda:0",
            sim_status_refresh_ms=250,
            sim_worker_log_lines=100,
            save_scene=True,
            onboard_camera=True,
            camera_parent_prim="",
            camera_width=424,
            camera_height=240,
            camera_update_period_s=0.1,
            camera_offset_x=0.35,
            camera_offset_y=0.0,
            camera_offset_z=0.18,
            camera_pitch_deg=14.0,
            camera_aim_mode="look-at",
            camera_target_x=1.55,
            camera_target_y=0.0,
            camera_target_z=0.02,
            camera_look_at_roll_deg=0.0,
            camera_coverage_strict=True,
            camera_focal_length=24.0,
            camera_horizontal_aperture=20.955,
            camera_near_clip_m=0.05,
            camera_far_clip_m=6.0,
            vision_confidence_threshold=0.75,
            vision_stable_frames=5,
            vision_window_size=7,
            vision_height_tolerance_cm=2.0,
            headless=False,
            livestream=0,
            enable_cameras=True,
            apply_safe_servo_joint_limits=True,
            apply_physx_joint_limits=False,
            no_continuous_sim_step=False,
            worker_smoke_negative_knee_test=False,
            worker_smoke_camera_detection=False,
            worker_smoke_camera_provenance=True,
            worker_smoke_camera_height_cm=5,
            worker_smoke_camera_validation_s=3.0,
            worker_smoke_camera_output="",
            worker_smoke_test_s=0.0,
            accept_isaac_eula=True,
            experience="",
        )


if __name__ == "__main__":
    unittest.main()
