"""Entry point for height-indexed obstacle replay UI and CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from height_manifest import HEIGHT_ERROR_MESSAGE, legacy_cm_to_mm, normalize_height_mm, obstacle_height_m_mm
from height_version_store import DEFAULT_VERSION_ROOT, HeightVersionStore
from motion_speed import load_motion_reference
from sim_obstacle_scene import (
    DEFAULT_ROBOT_USD_PATH,
    DEFAULT_SCENE_SAVE_PATH,
    OBSTACLE_LENGTH_M,
    OBSTACLE_WIDTH_M,
    SimSceneConfig,
    create_scene,
    ensure_simulation_app,
    finalize_scene_after_grounding,
)
from sim_process_client import run_launch_preflight_for_args
from sim_robot_adapter import SimRobotAdapter
from sim_worker_runtime import create_adapter_config_from_args, initialize_adapter_ground_reference
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDING_ROOT = DEFAULT_VERSION_ROOT


def config_from_args(args: argparse.Namespace, height_mm: int) -> SimSceneConfig:
    return SimSceneConfig(
        obstacle_height_m=obstacle_height_m_mm(height_mm),
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
        telemetry_contact_sensors_enabled=False,
        defer_first_visible_render=bool(getattr(args, "defer_first_visible_render", True)),
    )


def run_auto_play(args: argparse.Namespace) -> int:
    store = HeightVersionStore(Path(args.store_root), robot_asset_path=args.robot_usd)
    height_mm = normalize_height_mm(args.height_mm)
    version_id = store.active_version_id(height_mm)
    if not version_id:
        versions = store.list_versions(height_mm, include_legacy=True)
        version_id = str(versions[0].get("version_id", "")) if versions else ""
    steps, _metadata = store.load_version(height_mm, version_id) if version_id else ([], {})
    if args.no_sim:
        if not steps:
            print(f"No saved version found for {height_mm} mm. Please record and save a version first.")
            return 1
        print(f"--no-sim: loaded {len(steps)} step(s) from {version_id} for {height_mm} mm; playback was not sent to Isaac Sim.")
        return 0

    simulation_app = ensure_simulation_app(args)
    scene_handle = None
    try:
        scene_handle = create_scene(config_from_args(args, height_mm), simulation_app=simulation_app)
        adapter = SimRobotAdapter(scene_handle, create_adapter_config_from_args(args))
        initialize_adapter_ground_reference(adapter)
        finalize_scene_after_grounding(scene_handle)
        if not steps:
            print(f"No saved version found for {height_mm} mm. Please record and save a version first.")
            return 1
        adapter.play_steps_blocking(
            steps,
            profile=args.profile,
            label=f"{height_mm} mm {version_id}",
        )
        return 0
    finally:
        if scene_handle is not None:
            scene_handle.close()
        elif simulation_app is not None:
            simulation_app.close()


def run_ui(args: argparse.Namespace) -> int:
    controller = HeightReplayController(args)
    ui = RealRobotStyleHeightReplayUi(controller, smoke_test_ms=int(args.smoke_test_ms))
    def start_sim() -> None:
        try:
            controller.start_sim_if_needed()
        except Exception as exc:
            controller._warn(f"[WARN] Isaac Sim startup failed; UI will remain open in sim-not-ready state: {exc}")

    ui.root.after(50, start_sim)
    ui.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Height-indexed WLR obstacle replay.")
    parser.add_argument("--height-mm", "--height_mm", dest="height_mm", type=int, default=50)
    parser.add_argument("--height-cm", "--height_cm", dest="height_cm", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--auto-play", "--auto_play", dest="auto_play", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--no-sim", "--no_sim", dest="no_sim", action="store_true", help="Open/test the UI without starting Isaac Sim.")
    parser.add_argument("--sim-launch-mode", "--sim_launch_mode", dest="sim_launch_mode", choices=("subprocess", "main", "disabled"), default="subprocess")
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
    parser.add_argument("--profile", choices=("fast", "raw"), default="raw")
    parser.add_argument("--store-root", "--store_root", dest="store_root", type=str, default=str(DEFAULT_RECORDING_ROOT))
    parser.add_argument("--robot-usd", "--robot_usd", dest="robot_usd", type=str, default=str(DEFAULT_ROBOT_USD_PATH))
    parser.add_argument("--save-usd", "--save_usd", dest="save_usd", type=str, default=str(DEFAULT_SCENE_SAVE_PATH))
    parser.add_argument("--save-scene", "--save_scene", dest="save_scene", action="store_true", default=True)
    parser.add_argument("--no-save-scene", dest="save_scene", action="store_false")
    parser.add_argument("--spawn-z", "--spawn_z", dest="spawn_z", type=float, default=0.04)
    parser.add_argument("--obstacle-x", "--obstacle_x", dest="obstacle_x", type=float, default=1.55)
    parser.add_argument("--obstacle-width", "--obstacle_width", dest="obstacle_width", type=float, default=OBSTACLE_WIDTH_M)
    parser.add_argument("--obstacle-length", "--obstacle_length", dest="obstacle_length", type=float, default=OBSTACLE_LENGTH_M)
    parser.add_argument("--infer-obstacle-size", "--infer_obstacle_size", dest="infer_obstacle_size", type=_bool_arg, nargs="?", const=True, default=False)
    parser.add_argument("--robot-width", "--robot_width", dest="robot_width", type=float, default=0.80)
    parser.add_argument("--robot-length", "--robot_length", dest="robot_length", type=float, default=0.55)
    parser.add_argument("--physics-dt", "--physics_dt", dest="physics_dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--render-interval", "--render_interval", dest="render_interval", type=int, default=8)
    parser.add_argument("--wheel-direction", "--wheel_direction", dest="wheel_direction", type=float, default=1.0)
    parser.add_argument(
        "--default-wheel-speed-rad-s",
        "--default-wheel-speed",
        "--default_wheel_speed",
        dest="default_wheel_speed_rad_s",
        type=float,
        default=load_motion_reference().wheel_reference_velocity_rad_s,
    )
    parser.add_argument(
        "--max-wheel-speed-rad-s",
        "--max-wheel-speed",
        "--max_wheel_speed",
        dest="max_wheel_speed_rad_s",
        type=float,
        default=load_motion_reference().wheel_velocity_limit_rad_s,
    )
    parser.add_argument("--apply-safe-servo-joint-limits", "--apply_safe_servo_joint_limits", dest="apply_safe_servo_joint_limits", action="store_true", default=True)
    parser.add_argument("--no-apply-safe-servo-joint-limits", dest="apply_safe_servo_joint_limits", action="store_false")
    parser.add_argument("--apply-physx-joint-limits", "--apply_physx_joint_limits", dest="apply_physx_joint_limits", action="store_true", default=True)
    parser.add_argument("--no-apply-physx-joint-limits", dest="apply_physx_joint_limits", action="store_false")
    parser.add_argument("--ui-refresh-ms", "--ui_refresh_ms", dest="ui_refresh_ms", type=int, default=100)
    parser.add_argument("--sim-status-refresh-ms", "--sim_status_refresh_ms", dest="sim_status_refresh_ms", type=int, default=125)
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
    parser.add_argument("--defer-first-visible-render", "--defer_first_visible_render", dest="defer_first_visible_render", action="store_true", default=True)
    parser.add_argument("--no-defer-first-visible-render", "--no_defer_first_visible_render", dest="defer_first_visible_render", action="store_false")
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
    parser.add_argument("--worker-smoke-ground-structure", "--worker_smoke_ground_structure", dest="worker_smoke_ground_structure", action="store_true")
    parser.add_argument("--worker-smoke-ground-calibration", "--worker_smoke_ground_calibration", dest="worker_smoke_ground_calibration", action="store_true")
    parser.add_argument("--worker-smoke-output", "--worker_smoke_output", dest="worker_smoke_output", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--livestream", type=int, default=0)
    parser.add_argument("--experience", type=str, default="")
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
        if args.height_cm is not None:
            args.height_mm = legacy_cm_to_mm(args.height_cm)
        args.height_mm = normalize_height_mm(args.height_mm)
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
    value = float(args.default_wheel_speed_rad_s)
    max_speed = args.max_wheel_speed_rad_s
    args.default_wheel_speed_rad_s = max(-max_speed, min(max_speed, value))
    args.max_text_widget_chars = max(1000, int(getattr(args, "max_text_widget_chars", 200000)))
    args.record_event_min_interval_ms = max(0.0, float(getattr(args, "record_event_min_interval_ms", 50.0)))
    args.record_event_max_hz = max(0.0, float(getattr(args, "record_event_max_hz", 20.0)))
    args.record_max_events_per_step = max(1, int(getattr(args, "record_max_events_per_step", 2000)))
    args.playback_pre_step_settle_s = max(0.0, float(getattr(args, "playback_pre_step_settle_s", 0.30)))
    args.respawn_play_settle_s = max(0.0, float(getattr(args, "respawn_play_settle_s", 0.30)))
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
    args.telemetry_effective_enabled = False
    args.live_viz_effective_enabled = False
    args.equilibrium_region_effective_enabled = False
    args.telemetry_contact_sensors_enabled = False
    args.livestream = max(0, int(getattr(args, "livestream", 0) or 0))
    args.preflight_timeout_s = max(1.0, float(getattr(args, "preflight_timeout_s", 30.0)))
    args.sim_startup_timeout_s = max(1.0, float(getattr(args, "sim_startup_timeout_s", 600.0)))
    args.max_wheel_speed = args.max_wheel_speed_rad_s
    args.default_wheel_speed = args.default_wheel_speed_rad_s


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
