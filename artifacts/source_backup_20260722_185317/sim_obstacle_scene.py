"""Isaac Sim scene creation for a height-controlled obstacle."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from command_model import DEFAULT_MAX_WHEEL_SPEED_RAD_S
from robot_ground_diagnostics import resolve_ground_surface
from sim_onboard_camera import camera_pitch_quat_wxyz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
ISAACLAB_ROOT = Path("C:/robotics_sim/IsaacLab")

DEFAULT_ROBOT_USD_PATH = PROJECT_ROOT / "usd" / "wlr_robot_drive_test.usd"
DEFAULT_SCENE_SAVE_PATH = PROJECT_ROOT / "usd" / "wlr_robot_height_replay_env.usd"

ROBOT_PRIM_PATH = "/World/WLRRobot"
GROUND_PRIM_PATH = "/World/defaultGroundPlane"
OBSTACLE_PRIM_PATH = "/World/Obstacle"

SERVO_TORQUE_LIMIT_NM = 2.7
WHEEL_MAX_SPEED_RPM = 20.0
WHEEL_MAX_SPEED_RAD_PER_SEC = DEFAULT_MAX_WHEEL_SPEED_RAD_S


@dataclass
class SimSceneConfig:
    obstacle_height_m: float
    robot_usd: str | Path = DEFAULT_ROBOT_USD_PATH
    save_usd: str | Path = DEFAULT_SCENE_SAVE_PATH
    spawn_z: float = 0.04
    obstacle_x: float = 1.55
    obstacle_width: float | None = None
    obstacle_length: float | None = None
    infer_obstacle_size: bool = True
    robot_width: float = 0.80
    robot_length: float = 0.55
    physics_dt: float = 1.0 / 120.0
    render_interval: int = 2
    device: str = "cuda:0"
    max_wheel_speed: float = DEFAULT_MAX_WHEEL_SPEED_RAD_S
    default_wheel_speed: float = DEFAULT_MAX_WHEEL_SPEED_RAD_S * 0.25
    wheel_direction: float = 1.0
    servo_stiffness: float = 600.0
    servo_damping: float = 60.0
    wheel_damping: float = 20.0
    save_scene: bool = True
    onboard_camera_enabled: bool = True
    camera_parent_prim: str = ""
    camera_width: int = 424
    camera_height: int = 240
    camera_update_period_s: float = 0.10
    camera_offset_pos: tuple[float, float, float] = (0.35, 0.0, 0.18)
    camera_offset_rot: tuple[float, float, float, float] = camera_pitch_quat_wxyz(14.0)
    camera_offset_convention: str = "world"
    camera_aim_mode: str = "pitch"
    camera_target_x: float = 1.55
    camera_target_y: float = 0.0
    camera_target_z: float = 0.02
    camera_target_frame: str = "world"
    camera_look_at_roll_deg: float = 0.0
    camera_focal_length: float = 24.0
    camera_horizontal_aperture: float = 20.955
    camera_near_clip_m: float = 0.05
    camera_far_clip_m: float = 6.0
    camera_coverage_strict: bool = False
    ground_z_m: float = 0.0
    telemetry_contact_sensors_enabled: bool = False
    defer_first_visible_render: bool = True
    default_camera_eye: tuple[float, float, float] = (1.45, -1.25, 0.80)
    default_camera_target: tuple[float, float, float] = (0.45, 0.0, 0.12)


@dataclass
class SimSceneHandle:
    sim: Any
    robot: Any
    config: SimSceneConfig
    simulation_app: Any | None = None
    obstacle_center: Any | None = None
    camera: Any | None = None
    camera_prim_path: str = ""
    camera_parent_prim: str = ""
    camera_error: str = ""
    contact_sensor: Any | None = None
    contact_sensor_error: str = ""
    ground_surface_info: dict[str, Any] | None = None
    default_camera_eye: tuple[float, float, float] = (1.45, -1.25, 0.80)
    default_camera_target: tuple[float, float, float] = (0.45, 0.0, 0.12)
    first_visible_render_completed: bool = False
    startup_pose_trace: list[dict[str, Any]] | None = None

    def physics_dt(self) -> float:
        return float(self.sim.get_physics_dt())

    def app_is_running(self) -> bool:
        if self.simulation_app is None:
            return True
        return bool(self.simulation_app.is_running())

    def close(self) -> None:
        if self.simulation_app is not None:
            self.simulation_app.close()


_SIMULATION_APP: Any | None = None


def add_isaaclab_paths(isaaclab_root: str | Path = ISAACLAB_ROOT) -> None:
    root = Path(isaaclab_root)
    source_dir = root / "source"
    if source_dir.exists():
        for extension_dir in source_dir.iterdir():
            if extension_dir.is_dir():
                extension_path = str(extension_dir)
                if extension_path not in sys.path:
                    sys.path.append(extension_path)
    root_path = str(root)
    if root.exists() and root_path not in sys.path:
        sys.path.append(root_path)


def add_app_launcher_args(parser: Any) -> bool:
    add_isaaclab_paths()
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except Exception:
        return False
    AppLauncher.add_app_launcher_args(parser)
    return True


def ensure_simulation_app(args: Any | None = None) -> Any:
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return _SIMULATION_APP
    add_isaaclab_paths()
    from isaaclab.app import AppLauncher  # type: ignore

    _SIMULATION_APP = AppLauncher(args).app
    return _SIMULATION_APP


def create_scene(
    config: SimSceneConfig,
    *,
    simulation_app: Any | None = None,
    phase_callback: Any | None = None,
) -> SimSceneHandle:
    """Create ground, WLR robot, and a height-controlled obstacle."""

    _validate_robot_usd(config.robot_usd)
    imports = _isaac_imports()
    sim_utils = imports["sim_utils"]
    SimulationContext = imports["SimulationContext"]
    Articulation = imports["Articulation"]

    _emit_phase(phase_callback, "creating_simulation_context")
    sim_cfg = sim_utils.SimulationCfg(
        dt=float(config.physics_dt),
        render_interval=int(config.render_interval),
        device=str(config.device),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(config.default_camera_eye), target=list(config.default_camera_target))
    _emit_phase(phase_callback, "simulation_context_created")

    _emit_phase(phase_callback, "creating_ground")
    ground_surface = create_ground_plane(sim_utils, ground_z_m=float(config.ground_z_m))
    if ground_surface.get("actual_ground_z_m") is not None:
        config.ground_z_m = float(ground_surface["actual_ground_z_m"])
    _emit_phase(phase_callback, "ground_created")
    _emit_phase(phase_callback, "creating_lighting")
    add_lighting(sim_utils)
    _emit_phase(phase_callback, "lighting_created")

    _emit_phase(phase_callback, "creating_robot")
    robot = Articulation(build_robot_cfg(config, imports))
    _emit_phase(phase_callback, "robot_created")
    _emit_phase(phase_callback, "raw_spawn_created")
    resolve_obstacle_dimensions(config, sim_utils)
    _emit_phase(phase_callback, "creating_obstacle")
    obstacle_center = create_obstacle(config, imports)
    _emit_phase(phase_callback, "obstacle_created")

    roots = detect_articulation_roots(ROBOT_PRIM_PATH, imports)
    print(f"[INFO] Detected articulation root path(s): {roots}")

    handle = SimSceneHandle(
        sim=sim,
        robot=robot,
        config=config,
        simulation_app=simulation_app,
        obstacle_center=obstacle_center,
        ground_surface_info=ground_surface,
        default_camera_eye=tuple(config.default_camera_eye),
        default_camera_target=tuple(config.default_camera_target),
    )
    if bool(config.telemetry_contact_sensors_enabled):
        _emit_phase(phase_callback, "creating_contact_sensor")
        contact_sensor, contact_error = create_robot_contact_sensor()
        handle.contact_sensor = contact_sensor
        handle.contact_sensor_error = contact_error
        if contact_error:
            _emit_phase(phase_callback, "contact_sensor_disabled_with_error", {"error": contact_error})
            print(f"[WARN] Telemetry ContactSensor unavailable: {contact_error}")
        else:
            _emit_phase(phase_callback, "contact_sensor_created")
            print("[INFO] Telemetry ContactSensor created for robot bodies.")
    if bool(config.onboard_camera_enabled):
        from sim_onboard_camera import create_onboard_camera, reset_camera

        _emit_phase(phase_callback, "resolving_camera_parent")
        _emit_phase(phase_callback, "creating_camera")
        camera, camera_path, parent_path, camera_error = create_onboard_camera(handle, robot_root=ROBOT_PRIM_PATH)
        handle.camera = camera
        handle.camera_prim_path = camera_path
        handle.camera_parent_prim = parent_path
        handle.camera_error = camera_error
        if camera_error:
            _emit_phase(phase_callback, "camera_disabled_with_error", {"error": camera_error})
            print(f"[WARN] Onboard camera unavailable: {camera_error}")
        else:
            _emit_phase(phase_callback, "camera_created", {"camera_prim_path": camera_path, "camera_parent_prim": parent_path})
            print(f"[INFO] Onboard RGB-D camera created at {camera_path}, parent={parent_path}.")

    _emit_phase(phase_callback, "sim_reset_started")
    sim.reset()
    robot.update(0.0)
    if handle.camera is not None:
        from sim_onboard_camera import reset_camera

        reset_camera(handle)
    _emit_phase(phase_callback, "sim_reset_completed")
    if bool(config.defer_first_visible_render):
        _emit_phase(phase_callback, "first_visible_render_deferred")
    else:
        finalize_scene_after_grounding(handle, phase_callback=phase_callback)
    if config.save_scene:
        save_scene(config.save_usd, sim_utils)
    return handle


def finalize_scene_after_grounding(scene_handle: SimSceneHandle, *, phase_callback: Any | None = None) -> None:
    if bool(getattr(scene_handle, "first_visible_render_completed", False)):
        return
    _emit_phase(phase_callback, "first_visible_render_started")
    if scene_handle.camera is not None:
        try:
            from sim_onboard_camera import reset_camera

            reset_camera(scene_handle)
        except Exception as exc:
            scene_handle.camera_error = f"camera reset failed before first visible render: {exc}"
    _warmup_viewport(scene_handle.sim, scene_handle.robot)
    scene_handle.sim.set_camera_view(
        eye=list(scene_handle.config.default_camera_eye),
        target=list(scene_handle.config.default_camera_target),
    )
    scene_handle.first_visible_render_completed = True
    _emit_phase(phase_callback, "first_visible_render_completed")


def update_obstacle_height(scene_handle: SimSceneHandle, obstacle_height_m: float) -> None:
    """Replace the existing obstacle prim with a new height in the active scene."""

    imports = _isaac_imports()
    sim_utils = imports["sim_utils"]
    stage = sim_utils.get_current_stage()
    try:
        if stage.GetPrimAtPath(OBSTACLE_PRIM_PATH).IsValid():
            stage.RemovePrim(OBSTACLE_PRIM_PATH)
    except Exception as exc:
        print(f"[WARN] Could not remove old obstacle prim: {exc}")
    scene_handle.config.obstacle_height_m = float(obstacle_height_m)
    resolve_obstacle_dimensions(scene_handle.config, sim_utils)
    scene_handle.obstacle_center = create_obstacle(scene_handle.config, imports)
    try:
        scene_handle.sim.render()
    except Exception:
        pass
    print(f"[INFO] Updated obstacle height to {float(obstacle_height_m):.3f} m.")


def _emit_phase(phase_callback: Any | None, name: str, details: dict[str, Any] | None = None) -> None:
    if phase_callback is None:
        return
    try:
        phase_callback(str(name), dict(details or {}))
    except Exception:
        pass


def _warmup_viewport(sim: Any, robot: Any, *, steps: int = 3) -> None:
    """Render a few bounded frames after reset so the first GUI viewport is not empty."""

    last_error: Exception | None = None
    for _index in range(max(1, int(steps))):
        try:
            robot.update(0.0)
        except Exception as exc:
            last_error = exc
            break
        try:
            if hasattr(sim, "render"):
                sim.render()
            elif hasattr(sim, "step"):
                sim.step(render=True)
        except Exception as exc:
            last_error = exc
            break
    if last_error is not None:
        raise RuntimeError(f"first render warmup failed: {last_error}") from last_error


def create_ground_plane(sim_utils: Any, *, ground_z_m: float) -> dict[str, Any]:
    ground_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.25,
        dynamic_friction=1.05,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    ground_cfg = sim_utils.GroundPlaneCfg(
        size=(6.0, 6.0),
        color=(0.08, 0.09, 0.10),
        physics_material=ground_material,
    )
    translation = (0.0, 0.0, float(ground_z_m))
    try:
        ground_cfg.func(GROUND_PRIM_PATH, ground_cfg, translation=translation)
    except TypeError:
        ground_cfg.func(GROUND_PRIM_PATH, ground_cfg)
        _set_prim_translation(sim_utils, GROUND_PRIM_PATH, translation)
    surface = resolve_ground_surface(
        SimpleNamespace(config=SimpleNamespace(ground_z_m=float(ground_z_m)), ground_prim_path=GROUND_PRIM_PATH)
    )
    if not surface.ground_resolution_ok:
        print(
            "[WARN] Ground plane actual Z differs from configured Z: "
            f"configured={surface.configured_ground_z_m:.6f} actual={surface.actual_ground_z_m} "
            f"delta={surface.ground_z_delta_m}"
        )
    print(
        f"[INFO] Ground plane created at {GROUND_PRIM_PATH}; "
        f"configured_z={float(ground_z_m):.6f} actual_z={surface.actual_ground_z_m}."
    )
    return surface.to_dict()


def _set_prim_translation(sim_utils: Any, prim_path: str, translation: tuple[float, float, float]) -> None:
    try:
        from pxr import Gf, UsdGeom  # type: ignore

        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        try:
            for op in xformable.GetOrderedXformOps():
                if "translate" in str(op.GetOpName()).lower():
                    translate_op = op
                    break
        except Exception:
            translate_op = None
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
    except Exception as exc:
        print(f"[WARN] Could not set ground translation to {translation}: {exc}")


def add_lighting(sim_utils: Any) -> None:
    dome_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.88, 0.95))
    dome_cfg.func("/World/Light/Dome", dome_cfg)
    distant_cfg = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.96, 0.90), angle=0.35)
    distant_cfg.func("/World/Light/Key", distant_cfg, translation=(1.5, -2.0, 4.0))


def resolve_obstacle_dimensions(config: SimSceneConfig, sim_utils: Any) -> None:
    fallback_width = 2.0 * float(config.robot_width)
    fallback_length = 3.0 * float(config.robot_length)
    inferred_width = fallback_width
    inferred_length = fallback_length
    if bool(config.infer_obstacle_size):
        inferred = infer_robot_xy_size(sim_utils)
        if inferred is not None:
            robot_length, robot_width = inferred
            inferred_length = 3.0 * robot_length
            inferred_width = 2.0 * robot_width
            print(
                "[INFO] Inferred robot bbox: "
                f"length={robot_length:.3f} m, width={robot_width:.3f} m; "
                f"obstacle length={inferred_length:.3f} m, width={inferred_width:.3f} m"
            )
    config.obstacle_width = float(config.obstacle_width) if config.obstacle_width is not None else inferred_width
    config.obstacle_length = float(config.obstacle_length) if config.obstacle_length is not None else inferred_length


def infer_robot_xy_size(sim_utils: Any) -> tuple[float, float] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore

        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not prim.IsValid():
            return None
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        size = aligned.GetSize()
        length = float(size[0])
        width = float(size[1])
        if not math.isfinite(length) or not math.isfinite(width) or length <= 0.05 or width <= 0.05:
            return None
        return length, width
    except Exception as exc:
        print(f"[WARN] Robot bbox inference failed: {exc}")
        return None


def create_obstacle(config: SimSceneConfig, imports: dict[str, Any]) -> Any | None:
    if float(config.obstacle_height_m) <= 0.0:
        print("[INFO] obstacle_height_m=0.000; no obstacle platform is spawned for 0cm.")
        return None

    torch = imports["torch"]
    sim_utils = imports["sim_utils"]
    center = torch.tensor(
        (float(config.obstacle_x), 0.0, float(config.obstacle_height_m) * 0.5),
        dtype=torch.float32,
    )
    obstacle_cfg = sim_utils.CuboidCfg(
        size=(float(config.obstacle_length), float(config.obstacle_width), float(config.obstacle_height_m)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.20,
            dynamic_friction=1.00,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.47, 0.33)),
        semantic_tags=[("class", "height_obstacle")],
    )
    obstacle_cfg.func(OBSTACLE_PRIM_PATH, obstacle_cfg, translation=tuple(center.tolist()))
    print(
        "[INFO] Obstacle platform: "
        f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), "
        f"size=({float(config.obstacle_length):.3f}, {float(config.obstacle_width):.3f}, "
        f"{float(config.obstacle_height_m):.3f})"
    )
    return center


def create_robot_contact_sensor() -> tuple[Any | None, str]:
    try:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore

        sensor_cfg = ContactSensorCfg(
            prim_path=f"{ROBOT_PRIM_PATH}/.*",
            history_length=3,
            track_air_time=True,
            debug_vis=False,
        )
        return ContactSensor(sensor_cfg), ""
    except Exception as exc:
        return None, str(exc)


def build_robot_cfg(config: SimSceneConfig, imports: dict[str, Any]) -> Any:
    from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES

    sim_utils = imports["sim_utils"]
    ArticulationCfg = imports["ArticulationCfg"]
    ImplicitActuatorCfg = imports["ImplicitActuatorCfg"]
    return ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_path_for_usd(config.robot_usd),
            activate_contact_sensors=bool(config.telemetry_contact_sensors_enabled),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(config.spawn_z)),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip_knee_position_servos": ImplicitActuatorCfg(
                joint_names_expr=[regex_union(SERVO_JOINT_NAMES)],
                effort_limit_sim=SERVO_TORQUE_LIMIT_NM,
                velocity_limit_sim=None,
                stiffness=float(config.servo_stiffness),
                damping=float(config.servo_damping),
                armature=0.005,
            ),
            "wheel_velocity_motors": ImplicitActuatorCfg(
                joint_names_expr=[regex_union(WHEEL_JOINT_NAMES)],
                effort_limit_sim=None,
                velocity_limit_sim=max(WHEEL_MAX_SPEED_RAD_PER_SEC, float(config.max_wheel_speed)),
                stiffness=0.0,
                damping=float(config.wheel_damping),
                armature=0.002,
            ),
        },
    )


def detect_articulation_roots(under_path: str, imports: dict[str, Any]) -> list[str]:
    sim_utils = imports["sim_utils"]
    UsdPhysics = imports["UsdPhysics"]
    stage = sim_utils.get_current_stage()
    roots = []
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if prim_path.startswith(under_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim_path)
    return roots


def save_scene(path: str | Path, sim_utils: Any) -> None:
    save_path = _path_for_usd(path)
    try:
        result = sim_utils.save_stage(save_path, save_and_reload_in_place=False)
    except Exception as exc:
        print(f"[WARN] Failed to save generated scene to {save_path}: {exc}")
        return
    if result:
        print(f"[INFO] Saved generated scene to: {save_path}")
    else:
        print(f"[WARN] Failed to save generated scene to: {save_path}")


def regex_union(names: list[str]) -> str:
    return "|".join(names)


def _path_for_usd(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def _validate_robot_usd(path: str | Path) -> None:
    robot_usd = Path(path)
    if not robot_usd.exists():
        raise FileNotFoundError(f"Robot USD was not found: {robot_usd}")


def _isaac_imports() -> dict[str, Any]:
    add_isaaclab_paths()
    import torch  # type: ignore
    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import Articulation  # type: ignore
    from isaaclab.assets import ArticulationCfg  # type: ignore
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.sim import SimulationContext  # type: ignore
    from pxr import UsdPhysics  # type: ignore

    return {
        "torch": torch,
        "sim_utils": sim_utils,
        "SimulationContext": SimulationContext,
        "Articulation": Articulation,
        "ArticulationCfg": ArticulationCfg,
        "ImplicitActuatorCfg": ImplicitActuatorCfg,
        "UsdPhysics": UsdPhysics,
    }
