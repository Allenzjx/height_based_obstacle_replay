"""Entry point for height-indexed obstacle replay UI and CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from command_model import DEFAULT_MAX_WHEEL_SPEED_RAD_S
from height_manifest import HEIGHT_ERROR_MESSAGE, normalize_height_cm, obstacle_height_m
from height_sequence_store import HeightSequenceStore
from sim_obstacle_scene import (
    DEFAULT_ROBOT_USD_PATH,
    DEFAULT_SCENE_SAVE_PATH,
    SimSceneConfig,
    create_scene,
    ensure_simulation_app,
    finalize_scene_after_grounding,
)
from sim_onboard_camera import camera_pitch_quat_wxyz
from sim_process_client import run_launch_preflight_for_args
from sim_robot_adapter import SimRobotAdapter
from sim_worker_runtime import create_adapter_config_from_args, initialize_adapter_ground_reference
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi
from telemetry import create_telemetry_collector
from telemetry.config import add_telemetry_args, load_telemetry_config
from vision_gui_e2e import VisionGuiE2ERunner


def config_from_args(args: argparse.Namespace, height_cm: int) -> SimSceneConfig:
    return SimSceneConfig(
        obstacle_height_m=obstacle_height_m(height_cm),
        robot_usd=Path(args.robot_usd),
        save_usd=Path(args.save_usd),
        spawn_z=float(args.spawn_z),
        obstacle_x=float(args.obstacle_x),
        obstacle_width=args.obstacle_width,
        obstacle_length=args.obstacle_length,
        infer_obstacle_size=bool(args.infer_obstacle_size),
        robot_width=float(args.robot_width),
        robot_length=float(args.robot_length),
        physics_dt=float(args.physics_dt),
        render_interval=int(args.render_interval),
        device=str(getattr(args, "device", "cuda:0")),
        max_wheel_speed=float(args.max_wheel_speed_rad_s),
        default_wheel_speed=float(args.default_wheel_speed_rad_s),
        wheel_direction=float(args.wheel_direction),
        servo_stiffness=float(args.servo_stiffness),
        servo_damping=float(args.servo_damping),
        wheel_damping=float(args.wheel_damping),
        save_scene=bool(args.save_scene),
        onboard_camera_enabled=bool(getattr(args, "onboard_camera", True)),
        camera_parent_prim=str(getattr(args, "camera_parent_prim", "") or ""),
        camera_width=int(getattr(args, "camera_width", 424)),
        camera_height=int(getattr(args, "camera_height", 240)),
        camera_update_period_s=float(getattr(args, "camera_update_period_s", 0.10)),
        camera_offset_pos=(
            float(getattr(args, "camera_offset_x", 0.35)),
            float(getattr(args, "camera_offset_y", 0.0)),
            float(getattr(args, "camera_offset_z", 0.18)),
        ),
        camera_offset_rot=camera_pitch_quat_wxyz(float(getattr(args, "camera_pitch_deg", 14.0))),
        camera_offset_convention="world",
        camera_aim_mode=str(getattr(args, "camera_aim_mode", "pitch") or "pitch"),
        camera_target_x=float(getattr(args, "camera_target_x", getattr(args, "obstacle_x", 1.55))),
        camera_target_y=float(getattr(args, "camera_target_y", 0.0)),
        camera_target_z=float(getattr(args, "camera_target_z", 0.02)),
        camera_target_frame=str(getattr(args, "camera_target_frame", "world") or "world"),
        camera_look_at_roll_deg=float(getattr(args, "camera_look_at_roll_deg", 0.0)),
        camera_focal_length=float(getattr(args, "camera_focal_length", 24.0)),
        camera_horizontal_aperture=float(getattr(args, "camera_horizontal_aperture", 20.955)),
        camera_near_clip_m=float(getattr(args, "camera_near_clip_m", 0.05)),
        camera_far_clip_m=float(getattr(args, "camera_far_clip_m", 6.0)),
        camera_coverage_strict=bool(getattr(args, "camera_coverage_strict", False)),
        telemetry_contact_sensors_enabled=bool(getattr(args, "telemetry_contact_sensors_enabled", False)),
        defer_first_visible_render=bool(getattr(args, "defer_first_visible_render", True)),
    )


def run_auto_play(args: argparse.Namespace) -> int:
    store = HeightSequenceStore()
    height_cm = normalize_height_cm(args.height_cm)
    if args.no_sim:
        steps = store.load_steps(height_cm)
        if not steps:
            print(f"No saved steps found for {height_cm}cm. Please record steps first.")
            return 1
        print(f"--no-sim: loaded {len(steps)} step(s) for {height_cm}cm; playback was not sent to Isaac Sim.")
        return 0

    simulation_app = ensure_simulation_app(args)
    scene_handle = None
    collector = None
    success = False
    try:
        scene_handle = create_scene(config_from_args(args, height_cm), simulation_app=simulation_app)
        adapter = SimRobotAdapter(scene_handle, create_adapter_config_from_args(args))
        initialize_adapter_ground_reference(adapter)
        finalize_scene_after_grounding(scene_handle)
        collector = create_telemetry_collector(args, scene_handle=scene_handle)
        if collector is not None:
            adapter.attach_telemetry(collector)
            collector.start_episode(
                adapter=adapter,
                scene_handle=scene_handle,
                obstacle_height_cm=height_cm,
                obstacle_height_m=obstacle_height_m(height_cm),
                sequence_label=f"{height_cm}cm accepted steps",
                source="auto_play",
            )
        steps = store.load_steps(height_cm)
        if not steps:
            print(f"No saved steps found for {height_cm}cm. Please record steps first.")
            return 1
        adapter.play_steps_blocking(
            steps,
            profile=args.profile,
            speed=effective_playback_speed(args),
            preserve_wheel_distance=bool(args.preserve_wheel_distance),
            label=f"{height_cm}cm accepted steps",
        )
        success = True
        return 0
    finally:
        if collector is not None:
            status = collector.finish_episode(success=success, reason="" if success else "auto-play did not complete")
            run_dir = status.get("run_dir")
            if run_dir:
                print(f"[INFO] Telemetry run saved to: {run_dir}")
        if scene_handle is not None:
            scene_handle.close()
        elif simulation_app is not None:
            simulation_app.close()


def run_ui(args: argparse.Namespace) -> int:
    controller = HeightReplayController(args)
    ui = RealRobotStyleHeightReplayUi(controller, smoke_test_ms=int(args.smoke_test_ms))
    e2e_runner: VisionGuiE2ERunner | None = None

    def start_sim() -> None:
        try:
            controller.start_sim_if_needed()
        except Exception as exc:
            controller._warn(f"[WARN] Isaac Sim startup failed; UI will remain open in sim-not-ready state: {exc}")

    ui.root.after(50, start_sim)
    if bool(getattr(args, "e2e_vision_gui_smoke", False)):
        e2e_runner = VisionGuiE2ERunner(ui, controller, args)
        ui.root.after(250, e2e_runner.start)
    ui.run()
    if e2e_runner is not None:
        return int(e2e_runner.exit_code)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Height-indexed WLR obstacle replay.")
    parser.add_argument("--height-cm", "--height_cm", dest="height_cm", type=int, default=0)
    parser.add_argument("--auto-play", "--auto_play", dest="auto_play", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--no-sim", "--no_sim", dest="no_sim", action="store_true", help="Open/test the UI without starting Isaac Sim.")
    parser.add_argument("--sim-launch-mode", "--sim_launch_mode", dest="sim_launch_mode", choices=("subprocess", "thread", "main", "disabled"), default="subprocess")
    parser.add_argument("--worker-python-mode", "--worker_python_mode", dest="worker_python_mode", choices=("current-python", "isaaclab-bat"), default=None)
    parser.add_argument(
        "--worker-launch-mode",
        "--worker_launch_mode",
        dest="worker_launch_mode",
        choices=("auto", "current-python", "isaaclab-bat", "explicit-python"),
        default="auto",
    )
    parser.add_argument("--worker-python-exe", "--worker_python_exe", dest="worker_python_exe", type=str, default="")
    parser.add_argument("--isaaclab-bat", "--isaaclab_bat", dest="isaaclab_bat", type=str, default="C:/robotics_sim/IsaacLab/isaaclab.bat")
    parser.add_argument("--sim-startup-timeout-s", "--sim_startup_timeout_s", dest="sim_startup_timeout_s", type=float, default=600.0)
    parser.add_argument("--sim-worker-status-timeout-s", "--sim_worker_status_timeout_s", dest="sim_worker_status_timeout_s", type=float, default=10.0)
    parser.add_argument("--sim-worker-log-lines", "--sim_worker_log_lines", dest="sim_worker_log_lines", type=int, default=200)
    parser.add_argument("--launch-preflight-only", "--launch_preflight_only", dest="launch_preflight_only", action="store_true")
    parser.add_argument("--preflight-timeout-s", "--preflight_timeout_s", dest="preflight_timeout_s", type=float, default=30.0)
    parser.add_argument("--accept-isaac-eula", "--accept_isaac_eula", dest="accept_isaac_eula", action="store_true", default=False)
    parser.add_argument("--no-accept-isaac-eula", "--no_accept_isaac_eula", dest="accept_isaac_eula", action="store_false")
    parser.add_argument("--smoke-test-ms", "--smoke_test_ms", dest="smoke_test_ms", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-accept-recording", "--smoke_accept_recording", dest="smoke_accept_recording", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("fast", "raw"), default="fast")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--robot-usd", "--robot_usd", dest="robot_usd", type=str, default=str(DEFAULT_ROBOT_USD_PATH))
    parser.add_argument("--save-usd", "--save_usd", dest="save_usd", type=str, default=str(DEFAULT_SCENE_SAVE_PATH))
    parser.add_argument("--save-scene", "--save_scene", dest="save_scene", action="store_true", default=True)
    parser.add_argument("--no-save-scene", dest="save_scene", action="store_false")
    parser.add_argument("--spawn-z", "--spawn_z", dest="spawn_z", type=float, default=0.04)
    parser.add_argument("--obstacle-x", "--obstacle_x", dest="obstacle_x", type=float, default=1.55)
    parser.add_argument("--obstacle-width", "--obstacle_width", dest="obstacle_width", type=float, default=None)
    parser.add_argument("--obstacle-length", "--obstacle_length", dest="obstacle_length", type=float, default=None)
    parser.add_argument("--infer-obstacle-size", "--infer_obstacle_size", dest="infer_obstacle_size", type=_bool_arg, nargs="?", const=True, default=True)
    parser.add_argument("--robot-width", "--robot_width", dest="robot_width", type=float, default=0.80)
    parser.add_argument("--robot-length", "--robot_length", dest="robot_length", type=float, default=0.55)
    parser.add_argument("--physics-dt", "--physics_dt", dest="physics_dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--render-interval", "--render_interval", dest="render_interval", type=int, default=2)
    parser.add_argument("--wheel-direction", "--wheel_direction", dest="wheel_direction", type=float, default=1.0)
    parser.add_argument(
        "--default-wheel-speed-rad-s",
        "--default-wheel-speed",
        "--default_wheel_speed",
        dest="default_wheel_speed_rad_s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-wheel-speed-rad-s",
        "--max-wheel-speed",
        "--max_wheel_speed",
        dest="max_wheel_speed_rad_s",
        type=float,
        default=DEFAULT_MAX_WHEEL_SPEED_RAD_S,
    )
    parser.add_argument("--global-motion-speed-scale", "--global_motion_speed_scale", dest="global_motion_speed_scale", type=float, default=1.0)
    parser.add_argument("--wheel-speed-scale", "--wheel_speed_scale", dest="wheel_speed_scale", type=float, default=1.0)
    parser.add_argument("--servo-command-scale", "--servo_command_scale", dest="servo_command_scale", type=float, default=1.0)
    parser.add_argument("--playback-speed-scale", "--playback_speed_scale", dest="playback_speed_scale", type=float, default=1.0)
    parser.add_argument("--preserve-wheel-distance", dest="preserve_wheel_distance", action="store_true", default=True)
    parser.add_argument("--no-preserve-wheel-distance", dest="preserve_wheel_distance", action="store_false")
    parser.add_argument("--apply-speed-scale-to-manual", dest="apply_speed_scale_to_manual", action="store_true", default=True)
    parser.add_argument("--no-apply-speed-scale-to-manual", dest="apply_speed_scale_to_manual", action="store_false")
    parser.add_argument("--apply-speed-scale-to-playback", dest="apply_speed_scale_to_playback", action="store_true", default=True)
    parser.add_argument("--no-apply-speed-scale-to-playback", dest="apply_speed_scale_to_playback", action="store_false")
    parser.add_argument("--apply-safe-servo-joint-limits", "--apply_safe_servo_joint_limits", dest="apply_safe_servo_joint_limits", action="store_true", default=True)
    parser.add_argument("--no-apply-safe-servo-joint-limits", dest="apply_safe_servo_joint_limits", action="store_false")
    parser.add_argument("--apply-physx-joint-limits", "--apply_physx_joint_limits", dest="apply_physx_joint_limits", action="store_true")
    parser.add_argument("--ui-refresh-ms", "--ui_refresh_ms", dest="ui_refresh_ms", type=int, default=100)
    parser.add_argument("--sim-status-refresh-ms", "--sim_status_refresh_ms", dest="sim_status_refresh_ms", type=int, default=250)
    parser.add_argument("--full-refresh-ms", "--full_refresh_ms", dest="full_refresh_ms", type=int, default=1000)
    parser.add_argument("--max-text-widget-chars", "--max_text_widget_chars", dest="max_text_widget_chars", type=int, default=200000)
    parser.add_argument("--disable-auto-sim-state-json", "--disable_auto_sim_state_json", dest="disable_auto_sim_state_json", action="store_true")
    parser.add_argument("--sim-state-json-on-demand", "--sim_state_json_on_demand", dest="sim_state_json_on_demand", action="store_true", default=True)
    parser.add_argument("--no-sim-state-json-on-demand", "--no_sim_state_json_on_demand", dest="sim_state_json_on_demand", action="store_false")
    parser.add_argument("--record-event-min-interval-ms", "--record_event_min_interval_ms", dest="record_event_min_interval_ms", type=float, default=50.0)
    parser.add_argument("--record-event-max-hz", "--record_event_max_hz", dest="record_event_max_hz", type=float, default=20.0)
    parser.add_argument("--record-coalesce-slider-events", "--record_coalesce_slider_events", dest="record_coalesce_slider_events", action="store_true", default=True)
    parser.add_argument("--no-record-coalesce-slider-events", "--no_record_coalesce_slider_events", dest="record_coalesce_slider_events", action="store_false")
    parser.add_argument("--record-max-events-per-step", "--record_max_events_per_step", dest="record_max_events_per_step", type=int, default=2000)
    parser.add_argument("--playback-pre-step-settle-s", "--playback_pre_step_settle_s", dest="playback_pre_step_settle_s", type=float, default=0.30)
    parser.add_argument("--respawn-play-settle-s", "--respawn_play_settle_s", dest="respawn_play_settle_s", type=float, default=0.30)
    parser.add_argument("--restore-step-start-state-before-selected-playback", dest="restore_step_start_state_before_selected_playback", action="store_true", default=True)
    parser.add_argument("--no-restore-step-start-state-before-selected-playback", dest="restore_step_start_state_before_selected_playback", action="store_false")
    parser.add_argument("--restore-full-sim-pose-if-available", dest="restore_full_sim_pose_if_available", action="store_true", default=True)
    parser.add_argument("--no-restore-full-sim-pose-if-available", dest="restore_full_sim_pose_if_available", action="store_false")
    parser.add_argument("--fallback-to-command-state-before", dest="fallback_to_command_state_before", action="store_true", default=True)
    parser.add_argument("--no-fallback-to-command-state-before", dest="fallback_to_command_state_before", action="store_false")
    parser.add_argument("--no-continuous-sim-step", "--no_continuous_sim_step", dest="no_continuous_sim_step", action="store_true")
    parser.add_argument("--servo-stiffness", "--servo_stiffness", dest="servo_stiffness", type=float, default=600.0)
    parser.add_argument("--servo-damping", "--servo_damping", dest="servo_damping", type=float, default=60.0)
    parser.add_argument("--wheel-damping", "--wheel_damping", dest="wheel_damping", type=float, default=20.0)
    parser.add_argument("--onboard-camera", "--onboard_camera", dest="onboard_camera", action="store_true", default=True)
    parser.add_argument("--no-onboard-camera", "--no_onboard_camera", dest="onboard_camera", action="store_false")
    parser.add_argument("--camera-parent-prim", "--camera_parent_prim", dest="camera_parent_prim", type=str, default="")
    parser.add_argument("--camera-width", "--camera_width", dest="camera_width", type=int, default=424)
    parser.add_argument("--camera-height", "--camera_height", dest="camera_height", type=int, default=240)
    parser.add_argument("--camera-update-period-s", "--camera_update_period_s", dest="camera_update_period_s", type=float, default=0.10)
    parser.add_argument("--camera-offset-x", "--camera_offset_x", dest="camera_offset_x", type=float, default=0.35)
    parser.add_argument("--camera-offset-y", "--camera_offset_y", dest="camera_offset_y", type=float, default=0.0)
    parser.add_argument("--camera-offset-z", "--camera_offset_z", dest="camera_offset_z", type=float, default=0.18)
    parser.add_argument("--camera-pitch-deg", "--camera_pitch_deg", dest="camera_pitch_deg", type=float, default=14.0)
    parser.add_argument("--camera-aim-mode", "--camera_aim_mode", dest="camera_aim_mode", choices=["pitch", "look-at"], default="pitch")
    parser.add_argument("--camera-target-x", "--camera_target_x", dest="camera_target_x", type=float, default=1.55)
    parser.add_argument("--camera-target-y", "--camera_target_y", dest="camera_target_y", type=float, default=0.0)
    parser.add_argument("--camera-target-z", "--camera_target_z", dest="camera_target_z", type=float, default=0.02)
    parser.add_argument("--camera-target-frame", "--camera_target_frame", dest="camera_target_frame", choices=["world", "parent"], default="world")
    parser.add_argument("--camera-look-at-roll-deg", "--camera_look_at_roll_deg", dest="camera_look_at_roll_deg", type=float, default=0.0)
    parser.add_argument("--camera-coverage-strict", "--camera_coverage_strict", dest="camera_coverage_strict", action="store_true", default=False)
    parser.add_argument("--camera-focal-length", "--camera_focal_length", dest="camera_focal_length", type=float, default=24.0)
    parser.add_argument("--camera-horizontal-aperture", "--camera_horizontal_aperture", dest="camera_horizontal_aperture", type=float, default=20.955)
    parser.add_argument("--camera-near-clip-m", "--camera_near_clip_m", dest="camera_near_clip_m", type=float, default=0.05)
    parser.add_argument("--camera-far-clip-m", "--camera_far_clip_m", dest="camera_far_clip_m", type=float, default=6.0)
    parser.add_argument("--vision-auto-replay", "--vision_auto_replay", dest="vision_auto_replay", action="store_true", default=False)
    parser.add_argument("--no-vision-auto-replay", "--no_vision_auto_replay", dest="vision_auto_replay", action="store_false")
    parser.add_argument("--vision-confidence-threshold", "--vision_confidence_threshold", dest="vision_confidence_threshold", type=float, default=0.75)
    parser.add_argument("--vision-stable-frames", "--vision_stable_frames", dest="vision_stable_frames", type=int, default=5)
    parser.add_argument("--vision-window-size", "--vision_window_size", dest="vision_window_size", type=int, default=7)
    parser.add_argument("--vision-height-tolerance-cm", "--vision_height_tolerance_cm", dest="vision_height_tolerance_cm", type=float, default=2.0)
    parser.add_argument("--vision-auto-replay-cooldown-s", "--vision_auto_replay_cooldown_s", dest="vision_auto_replay_cooldown_s", type=float, default=2.0)
    parser.add_argument("--vision-respawn-before-replay", "--vision_respawn_before_replay", dest="vision_respawn_before_replay", action="store_true", default=True)
    parser.add_argument("--no-vision-respawn-before-replay", "--no_vision_respawn_before_replay", dest="vision_respawn_before_replay", action="store_false")
    parser.add_argument("--viewport-physics-guard", "--viewport_physics_guard", dest="viewport_physics_guard", action="store_true", default=True)
    parser.add_argument("--no-viewport-physics-guard", "--no_viewport_physics_guard", dest="viewport_physics_guard", action="store_false")
    parser.add_argument("--defer-first-visible-render", "--defer_first_visible_render", dest="defer_first_visible_render", action="store_true", default=True)
    parser.add_argument("--no-defer-first-visible-render", "--no_defer_first_visible_render", dest="defer_first_visible_render", action="store_false")
    parser.add_argument("--camera-view-active-fallback", "--camera_view_active_fallback", dest="camera_view_active_fallback", action="store_true")
    parser.add_argument("--camera-view-pending-timeout-s", "--camera_view_pending_timeout_s", dest="camera_view_pending_timeout_s", type=float, default=10.0)
    parser.add_argument("--camera-view-pending-max-retries", "--camera_view_pending_max_retries", dest="camera_view_pending_max_retries", type=int, default=30)
    parser.add_argument("--e2e-vision-gui-smoke", "--e2e_vision_gui_smoke", dest="e2e_vision_gui_smoke", action="store_true")
    parser.add_argument("--e2e-height-cm", "--e2e_height_cm", dest="e2e_height_cm", type=int, default=5)
    parser.add_argument("--e2e-timeout-s", "--e2e_timeout_s", dest="e2e_timeout_s", type=float, default=300.0)
    parser.add_argument("--e2e-output", "--e2e_output", dest="e2e_output", type=str, default="")
    parser.add_argument("--e2e-open-camera-viewport", "--e2e_open_camera_viewport", dest="e2e_open_camera_viewport", action="store_true")
    parser.add_argument("--e2e-test-camera-fallback", "--e2e_test_camera_fallback", dest="e2e_test_camera_fallback", action="store_true")
    parser.add_argument("--e2e-playback-probe-s", "--e2e_playback_probe_s", dest="e2e_playback_probe_s", type=float, default=0.0)
    parser.add_argument("--e2e-keep-open-on-failure", "--e2e_keep_open_on_failure", dest="e2e_keep_open_on_failure", action="store_true")
    parser.add_argument("--e2e-save-screenshots", "--e2e_save_screenshots", dest="e2e_save_screenshots", action="store_true")
    parser.add_argument("--e2e-capture-startup-trace", "--e2e_capture_startup_trace", dest="e2e_capture_startup_trace", action="store_true")
    parser.add_argument("--e2e-camera-counterfactual", "--e2e_camera_counterfactual", dest="e2e_camera_counterfactual", action="store_true")
    parser.add_argument("--e2e-camera-pose-ab", "--e2e_camera_pose_ab", dest="e2e_camera_pose_ab", action="store_true")
    parser.add_argument("--e2e-ground-calibration", "--e2e_ground_calibration", dest="e2e_ground_calibration", action="store_true")
    parser.add_argument("--robot-ground-settle-s", "--robot_ground_settle_s", dest="robot_ground_settle_s", type=float, default=0.75)
    parser.add_argument("--robot-ground-settle-max-steps", "--robot_ground_settle_max_steps", dest="robot_ground_settle_max_steps", type=int, default=180)
    parser.add_argument("--robot-ground-stable-frames", "--robot_ground_stable_frames", dest="robot_ground_stable_frames", type=int, default=10)
    parser.add_argument("--robot-ground-vertical-speed-threshold-m-s", "--robot_ground_vertical_speed_threshold_m_s", dest="robot_ground_vertical_speed_threshold_m_s", type=float, default=0.01)
    parser.add_argument("--robot-ground-joint-speed-threshold-rad-s", "--robot_ground_joint_speed_threshold_rad_s", dest="robot_ground_joint_speed_threshold_rad_s", type=float, default=0.02)
    parser.add_argument("--robot-ground-servo-speed-threshold-rad-s", "--robot_ground_servo_speed_threshold_rad_s", dest="robot_ground_servo_speed_threshold_rad_s", type=float, default=None)
    parser.add_argument("--robot-ground-wheel-speed-threshold-rad-s", "--robot_ground_wheel_speed_threshold_rad_s", dest="robot_ground_wheel_speed_threshold_rad_s", type=float, default=0.20)
    parser.add_argument("--robot-ground-clearance-m", "--robot_ground_clearance_m", dest="robot_ground_clearance_m", type=float, default=0.002)
    parser.add_argument("--robot-ground-penetration-tolerance-m", "--robot_ground_penetration_tolerance_m", dest="robot_ground_penetration_tolerance_m", type=float, default=0.003)
    parser.add_argument("--robot-auto-ground-correction", "--robot_auto_ground_correction", dest="robot_auto_ground_correction", action="store_true")
    parser.add_argument("--robot-max-ground-correction-m", "--robot_max_ground_correction_m", dest="robot_max_ground_correction_m", type=float, default=0.10)
    parser.add_argument("--worker-smoke-negative-knee-test", "--worker_smoke_negative_knee_test", dest="worker_smoke_negative_knee_test", action="store_true")
    parser.add_argument("--worker-smoke-camera-detection", "--worker_smoke_camera_detection", dest="worker_smoke_camera_detection", action="store_true")
    parser.add_argument("--worker-smoke-camera-provenance", "--worker_smoke_camera_provenance", dest="worker_smoke_camera_provenance", action="store_true")
    parser.add_argument("--worker-smoke-camera-pose-ab", "--worker_smoke_camera_pose_ab", dest="worker_smoke_camera_pose_ab", action="store_true")
    parser.add_argument("--worker-smoke-camera-counterfactual", "--worker_smoke_camera_counterfactual", dest="worker_smoke_camera_counterfactual", action="store_true")
    parser.add_argument("--worker-smoke-camera-view-ground-contact", "--worker_smoke_camera_view_ground_contact", dest="worker_smoke_camera_view_ground_contact", action="store_true")
    parser.add_argument("--worker-smoke-ground-structure", "--worker_smoke_ground_structure", dest="worker_smoke_ground_structure", action="store_true")
    parser.add_argument("--worker-smoke-ground-calibration", "--worker_smoke_ground_calibration", dest="worker_smoke_ground_calibration", action="store_true")
    parser.add_argument("--worker-smoke-vision-playback", "--worker_smoke_vision_playback", dest="worker_smoke_vision_playback", action="store_true")
    parser.add_argument("--worker-smoke-output", "--worker_smoke_output", dest="worker_smoke_output", type=str, default="")
    parser.add_argument("--worker-smoke-camera-height-cm", "--worker_smoke_camera_height_cm", dest="worker_smoke_camera_height_cm", type=int, default=None)
    parser.add_argument("--worker-smoke-camera-validation-s", "--worker_smoke_camera_validation_s", dest="worker_smoke_camera_validation_s", type=float, default=10.0)
    parser.add_argument("--worker-smoke-camera-output", "--worker_smoke_camera_output", dest="worker_smoke_camera_output", type=str, default="")
    parser.add_argument("--worker-smoke-camera-counterfactual-output", "--worker_smoke_camera_counterfactual_output", dest="worker_smoke_camera_counterfactual_output", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--livestream", type=int, default=0)
    parser.add_argument("--experience", type=str, default="")
    add_telemetry_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_motion_args(args)
    if bool(getattr(args, "launch_preflight_only", False)):
        report = run_launch_preflight_for_args(args)
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return 0 if bool(report.get("preflight_ok", False)) else 1
    try:
        normalize_height_cm(args.height_cm)
    except Exception:
        print(HEIGHT_ERROR_MESSAGE)
        return 2
    if args.auto_play:
        return run_auto_play(args)
    return run_ui(args)


def normalize_motion_args(args: argparse.Namespace) -> None:
    if getattr(args, "worker_python_mode", None):
        args.worker_launch_mode = str(args.worker_python_mode)
    if str(getattr(args, "worker_launch_mode", "auto") or "").strip() == "explicit-python" and not str(getattr(args, "worker_python_exe", "") or "").strip():
        args.worker_launch_mode = "auto"
    if bool(getattr(args, "no_sim", False)) or str(getattr(args, "sim_launch_mode", "subprocess")) == "disabled":
        args.no_sim = True
        args.sim_launch_mode = "disabled"
    args.max_wheel_speed_rad_s = abs(float(args.max_wheel_speed_rad_s))
    if args.default_wheel_speed_rad_s is None:
        args.default_wheel_speed_rad_s = args.max_wheel_speed_rad_s * 0.25
    else:
        value = float(args.default_wheel_speed_rad_s)
        max_speed = args.max_wheel_speed_rad_s
        args.default_wheel_speed_rad_s = max(-max_speed, min(max_speed, value))
    args.global_motion_speed_scale = max(0.0, float(args.global_motion_speed_scale))
    args.wheel_speed_scale = max(0.0, float(args.wheel_speed_scale))
    args.servo_command_scale = max(0.0, float(args.servo_command_scale))
    args.playback_speed_scale = max(0.0, float(args.playback_speed_scale))
    args.max_text_widget_chars = max(1000, int(getattr(args, "max_text_widget_chars", 200000)))
    args.record_event_min_interval_ms = max(0.0, float(getattr(args, "record_event_min_interval_ms", 50.0)))
    args.record_event_max_hz = max(0.0, float(getattr(args, "record_event_max_hz", 20.0)))
    args.record_max_events_per_step = max(1, int(getattr(args, "record_max_events_per_step", 2000)))
    args.playback_pre_step_settle_s = max(0.0, float(getattr(args, "playback_pre_step_settle_s", 0.30)))
    args.respawn_play_settle_s = max(0.0, float(getattr(args, "respawn_play_settle_s", 0.30)))
    args.camera_width = max(16, int(getattr(args, "camera_width", 424)))
    args.camera_height = max(16, int(getattr(args, "camera_height", 240)))
    args.camera_update_period_s = max(0.01, float(getattr(args, "camera_update_period_s", 0.10)))
    args.camera_near_clip_m = max(0.001, float(getattr(args, "camera_near_clip_m", 0.05)))
    args.camera_far_clip_m = max(args.camera_near_clip_m + 0.01, float(getattr(args, "camera_far_clip_m", 6.0)))
    args.vision_confidence_threshold = max(0.0, min(1.0, float(getattr(args, "vision_confidence_threshold", 0.75))))
    args.vision_stable_frames = max(1, int(getattr(args, "vision_stable_frames", 5)))
    args.vision_window_size = max(args.vision_stable_frames, int(getattr(args, "vision_window_size", 7)))
    args.vision_height_tolerance_cm = max(0.1, float(getattr(args, "vision_height_tolerance_cm", 2.0)))
    args.vision_auto_replay_cooldown_s = max(0.0, float(getattr(args, "vision_auto_replay_cooldown_s", 2.0)))
    args.camera_view_pending_timeout_s = max(0.05, float(getattr(args, "camera_view_pending_timeout_s", 10.0)))
    args.camera_view_pending_max_retries = max(1, int(getattr(args, "camera_view_pending_max_retries", 30)))
    args.e2e_height_cm = normalize_height_cm(getattr(args, "e2e_height_cm", 5))
    args.e2e_timeout_s = max(1.0, float(getattr(args, "e2e_timeout_s", 300.0)))
    args.e2e_playback_probe_s = max(0.0, float(getattr(args, "e2e_playback_probe_s", 0.0)))
    args.robot_ground_settle_s = max(0.0, float(getattr(args, "robot_ground_settle_s", 0.75)))
    args.robot_ground_settle_max_steps = max(1, int(getattr(args, "robot_ground_settle_max_steps", 180)))
    args.robot_ground_stable_frames = max(1, int(getattr(args, "robot_ground_stable_frames", 10)))
    args.robot_ground_vertical_speed_threshold_m_s = max(0.0, float(getattr(args, "robot_ground_vertical_speed_threshold_m_s", 0.01)))
    args.robot_ground_joint_speed_threshold_rad_s = max(0.0, float(getattr(args, "robot_ground_joint_speed_threshold_rad_s", 0.02)))
    if getattr(args, "robot_ground_servo_speed_threshold_rad_s", None) is None:
        args.robot_ground_servo_speed_threshold_rad_s = args.robot_ground_joint_speed_threshold_rad_s
    args.robot_ground_servo_speed_threshold_rad_s = max(0.0, float(getattr(args, "robot_ground_servo_speed_threshold_rad_s", args.robot_ground_joint_speed_threshold_rad_s)))
    args.robot_ground_wheel_speed_threshold_rad_s = max(0.0, float(getattr(args, "robot_ground_wheel_speed_threshold_rad_s", 0.20)))
    args.robot_ground_clearance_m = max(0.0, float(getattr(args, "robot_ground_clearance_m", 0.002)))
    args.robot_ground_penetration_tolerance_m = max(0.0, float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003)))
    args.robot_max_ground_correction_m = max(0.0, float(getattr(args, "robot_max_ground_correction_m", 0.10)))
    args.worker_smoke_camera_validation_s = max(1.0, float(getattr(args, "worker_smoke_camera_validation_s", 10.0)))
    telemetry_config = load_telemetry_config(args)
    args.telemetry_runtime_config = telemetry_config
    args.telemetry_effective_enabled = bool(telemetry_config.telemetry.enabled)
    args.live_viz_effective_enabled = bool(telemetry_config.visualization.live_enabled)
    args.telemetry_report_effective_enabled = bool(telemetry_config.telemetry.report_on_finish)
    args.equilibrium_region_effective_enabled = bool(telemetry_config.stability.equilibrium_enabled)
    args.telemetry_contact_sensors_enabled = bool(telemetry_config.telemetry.enabled and telemetry_config.telemetry.enable_contact_sensor)
    if bool(getattr(args, "onboard_camera", True)):
        setattr(args, "enable_cameras", True)
    else:
        setattr(args, "enable_cameras", False)
    args.livestream = max(0, int(getattr(args, "livestream", 0) or 0))
    args.preflight_timeout_s = max(1.0, float(getattr(args, "preflight_timeout_s", 30.0)))
    args.sim_startup_timeout_s = max(1.0, float(getattr(args, "sim_startup_timeout_s", 600.0)))
    args.max_wheel_speed = args.max_wheel_speed_rad_s
    args.default_wheel_speed = args.default_wheel_speed_rad_s


def effective_playback_speed(args: argparse.Namespace) -> float:
    scale = 1.0
    if bool(getattr(args, "apply_speed_scale_to_playback", True)):
        scale = float(args.global_motion_speed_scale) * float(args.playback_speed_scale)
    return max(0.1, min(5.0, float(args.speed) * scale))


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    for action in parser._actions:
        if option in action.option_strings:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
