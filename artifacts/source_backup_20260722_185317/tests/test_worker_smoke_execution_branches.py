from __future__ import annotations

import argparse
import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker_process  # noqa: E402
from sim_process_client import build_worker_command, build_worker_config  # noqa: E402


class WorkerSmokeExecutionBranchesTest(unittest.TestCase):
    def test_new_worker_smoke_flags_reach_config_and_command(self) -> None:
        args = argparse.Namespace(
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
            camera_width=424,
            camera_height=240,
            camera_update_period_s=0.1,
            camera_offset_x=0.35,
            camera_offset_y=0.0,
            camera_offset_z=0.18,
            camera_pitch_deg=14.0,
            camera_aim_mode="pitch",
            camera_target_x=1.55,
            camera_target_y=0.0,
            camera_target_z=0.02,
            camera_target_frame="world",
            camera_look_at_roll_deg=0.0,
            camera_coverage_strict=False,
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
            worker_smoke_camera_provenance=False,
            worker_smoke_camera_pose_ab=True,
            worker_smoke_camera_counterfactual=True,
            worker_smoke_camera_view_ground_contact=False,
            worker_smoke_ground_structure=True,
            worker_smoke_ground_calibration=True,
            worker_smoke_vision_playback=True,
            worker_smoke_camera_height_cm=5,
            worker_smoke_camera_validation_s=3.0,
            worker_smoke_camera_output="",
            worker_smoke_camera_counterfactual_output="counterfactual.json",
            worker_smoke_output="ground.json",
            worker_smoke_test_s=0.0,
            viewport_physics_guard=True,
            defer_first_visible_render=True,
            camera_view_active_fallback=False,
            camera_view_pending_timeout_s=10.0,
            camera_view_pending_max_retries=30,
            robot_ground_settle_s=0.75,
            robot_ground_settle_max_steps=180,
            robot_ground_stable_frames=10,
            robot_ground_vertical_speed_threshold_m_s=0.01,
            robot_ground_joint_speed_threshold_rad_s=0.02,
            robot_ground_servo_speed_threshold_rad_s=None,
            robot_ground_wheel_speed_threshold_rad_s=0.2,
            robot_ground_clearance_m=0.002,
            robot_ground_penetration_tolerance_m=0.003,
            robot_auto_ground_correction=False,
            robot_max_ground_correction_m=0.10,
            accept_isaac_eula=True,
            experience="",
        )

        config = build_worker_config(args, host="127.0.0.1", port=4321)
        command = build_worker_command(args, host="127.0.0.1", port=4321)

        self.assertTrue(config["worker_smoke_ground_structure"])
        self.assertTrue(config["worker_smoke_ground_calibration"])
        self.assertTrue(config["worker_smoke_vision_playback"])
        self.assertTrue(config["worker_smoke_camera_counterfactual"])
        self.assertTrue(config["defer_first_visible_render"])
        self.assertIn("--worker-smoke-ground-structure", command)
        self.assertIn("--worker-smoke-ground-calibration", command)
        self.assertIn("--worker-smoke-vision-playback", command)
        self.assertIn("--worker-smoke-camera-pose-ab", command)
        self.assertIn("--worker-smoke-camera-counterfactual", command)
        self.assertIn("--worker-smoke-camera-counterfactual-output", command)
        self.assertIn("--defer-first-visible-render", command)
        self.assertIn("--robot-ground-wheel-speed-threshold-rad-s", command)

    def test_worker_run_has_real_branches_for_smoke_flags(self) -> None:
        source = inspect.getsource(sim_worker_process.run_worker)

        self.assertIn("run_ground_structure_smoke", source)
        self.assertIn("run_ground_calibration_smoke", source)
        self.assertIn("run_vision_playback_smoke", source)
        self.assertIn("run_camera_pose_ab_smoke", source)
        self.assertIn("run_camera_counterfactual_smoke", source)


if __name__ == "__main__":
    unittest.main()
