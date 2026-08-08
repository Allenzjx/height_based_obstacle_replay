"""Isaac Sim scene creation for a height-controlled obstacle."""

from __future__ import annotations

import math
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from motion_speed import load_motion_reference
from robot_ground_diagnostics import (
    collider_mesh_body_local_aabb,
    live_collider_world_aabb_from_body_pose,
    resolve_ground_surface,
    safe_prim_world_aabb,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
ISAACLAB_ROOT = Path("C:/robotics_sim/IsaacLab")
ENVIRONMENT_CONFIG_PATH = MODULE_ROOT / "config" / "environment_reference.yaml"
ENVIRONMENT_REFERENCE = json.loads(ENVIRONMENT_CONFIG_PATH.read_text(encoding="utf-8"))

DEFAULT_ROBOT_USD_PATH = PROJECT_ROOT / "usd" / "wlr_robot_drive_test.usd"
DEFAULT_SCENE_SAVE_PATH = PROJECT_ROOT / "usd" / "wlr_robot_height_replay_env.usd"

ROBOT_PRIM_PATH = "/World/WLRRobot"
GROUND_PRIM_PATH = "/World/defaultGroundPlane"
OBSTACLE_PRIM_PATH = "/World/Obstacle"

SERVO_TORQUE_LIMIT_NM = 2.7
WHEEL_MAX_SPEED_RPM = 20.0
WHEEL_MAX_SPEED_RAD_PER_SEC = load_motion_reference().wheel_velocity_limit_rad_s
OBSTACLE_FRONT_FACE_X_M = float(ENVIRONMENT_REFERENCE["obstacle_front_face_x_m"])
OBSTACLE_LENGTH_M = float(ENVIRONMENT_REFERENCE["obstacle_length_m"])
OBSTACLE_WIDTH_M = float(ENVIRONMENT_REFERENCE["obstacle_width_m"])
ROBOT_COLLISION_WIDTH_M = float(ENVIRONMENT_REFERENCE["robot_collision_width_m"])


@dataclass
class SimSceneConfig:
    obstacle_height_m: float
    robot_usd: str | Path = DEFAULT_ROBOT_USD_PATH
    save_usd: str | Path = DEFAULT_SCENE_SAVE_PATH
    spawn_z: float = 0.04
    # ``obstacle_x`` remains the initial center for CLI compatibility.  The
    # resolved front face is captured once, then held fixed for every height.
    obstacle_x: float = 1.55
    obstacle_front_x: float | None = None
    obstacle_width: float | None = OBSTACLE_WIDTH_M
    obstacle_length: float | None = OBSTACLE_LENGTH_M
    infer_obstacle_size: bool = False
    robot_width: float = 0.80
    robot_length: float = 0.55
    physics_dt: float = 1.0 / 120.0
    render_interval: int = 8
    device: str = "cuda:0"
    max_wheel_speed: float = load_motion_reference().wheel_velocity_limit_rad_s
    default_wheel_speed: float = load_motion_reference().wheel_reference_velocity_rad_s
    wheel_direction: float = 1.0
    servo_stiffness: float = 600.0
    servo_damping: float = 60.0
    wheel_damping: float = 20.0
    save_scene: bool = True
    ground_z_m: float = 0.0
    telemetry_contact_sensors_enabled: bool = False
    contact_sensor_factory: Any | None = None
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
    # The Tk controller is intentionally a separate foreground window.  Kit's
    # default out-of-focus sleep otherwise drops the physics worker to roughly
    # 10 Hz whenever the operator uses Tk, making motion depend on UI focus.
    import carb  # type: ignore

    settings = carb.settings.get_settings()
    # The adapter performs explicit 1/120 s physics substeps and renders only
    # after render_interval substeps.  A second Kit main-loop limiter would
    # sleep again after that work and cut the measured RTF roughly in half.
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
    settings.set_int("/app/renderer/sleepMsOnFocus", 0)
    settings.set_int("/app/renderer/sleepMsOutOfFocus", 0)
    settings.set_bool("/app/renderer/skipWhileMinimized", False)
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
        contact_sensor_factory = config.contact_sensor_factory or create_robot_contact_sensor
        contact_sensor, contact_error = contact_sensor_factory()
        handle.contact_sensor = contact_sensor
        handle.contact_sensor_error = contact_error
        if contact_error:
            _emit_phase(phase_callback, "contact_sensor_disabled_with_error", {"error": contact_error})
            print(f"[WARN] Telemetry ContactSensor unavailable: {contact_error}")
        else:
            _emit_phase(phase_callback, "contact_sensor_created")
            print("[INFO] Telemetry ContactSensor created for robot bodies.")
    _emit_phase(phase_callback, "sim_reset_started")
    sim.reset()
    robot.update(0.0)
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
    _warmup_viewport(scene_handle.sim, scene_handle.robot)
    scene_handle.sim.set_camera_view(
        eye=list(scene_handle.config.default_camera_eye),
        target=list(scene_handle.config.default_camera_target),
    )
    scene_handle.first_visible_render_completed = True
    _emit_phase(phase_callback, "first_visible_render_completed")


def update_obstacle_height(scene_handle: SimSceneHandle, obstacle_height_m: float) -> dict[str, Any]:
    """Update the obstacle and return measured USD geometry, never a guessed success."""

    imports = _isaac_imports()
    sim_utils = imports["sim_utils"]
    stage = sim_utils.get_current_stage()
    scene_handle.config.obstacle_height_m = float(obstacle_height_m)
    prim = stage.GetPrimAtPath(OBSTACLE_PRIM_PATH)
    old_geometry = measure_obstacle_geometry(scene_handle)
    if not prim.IsValid():
        scene_handle.obstacle_center = create_obstacle(scene_handle.config, imports)
        measured = measure_obstacle_geometry(scene_handle)
        return {"old_geometry": old_geometry, "measured_geometry": measured, "update_mode": "created_missing_prim"}
    center = (
        float(scene_handle.config.obstacle_x),
        0.0,
        float(scene_handle.config.ground_z_m) + float(obstacle_height_m) * 0.5,
    )
    size = (
        float(scene_handle.config.obstacle_length),
        float(scene_handle.config.obstacle_width),
        float(obstacle_height_m),
    )
    _set_cuboid_size_and_translation(prim, size=size, translation=center)
    scene_handle.obstacle_center = imports["torch"].tensor(center, dtype=imports["torch"].float32)
    measured = measure_obstacle_geometry(scene_handle)
    expected_height = float(obstacle_height_m)
    expected_width = float(scene_handle.config.obstacle_width)
    in_place_ok = (
        bool(measured.get("prim_valid", False))
        and bool(measured.get("visual_valid", False))
        and bool(measured.get("collision_valid", False))
        and abs(float(measured.get("height_m", -1.0)) - expected_height) <= 0.001
        and abs(float(measured.get("width_m", -1.0)) - expected_width) <= 0.001
        and abs(float(measured.get("collision_height_m", -1.0)) - expected_height) <= 0.001
    )
    mode = "in_place"
    if not in_place_ok:
        if not stage.RemovePrim(OBSTACLE_PRIM_PATH):
            raise RuntimeError(f"Could not remove stale obstacle prim {OBSTACLE_PRIM_PATH}")
        if stage.GetPrimAtPath(OBSTACLE_PRIM_PATH).IsValid():
            raise RuntimeError(f"Obstacle prim remained valid after removal: {OBSTACLE_PRIM_PATH}")
        scene_handle.obstacle_center = create_obstacle(scene_handle.config, imports)
        measured = measure_obstacle_geometry(scene_handle)
        mode = "minimal_recreate_after_in_place_verification_failed"
    return {"old_geometry": old_geometry, "measured_geometry": measured, "update_mode": mode}


def measure_obstacle_geometry(scene_handle: SimSceneHandle) -> dict[str, Any]:
    """Measure only the small obstacle subtree, including visual and collision bounds."""

    result: dict[str, Any] = {
        "prim_path": OBSTACLE_PRIM_PATH,
        "prim_valid": False,
        "visual_valid": False,
        "collision_valid": False,
    }
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

        stage = getattr(scene_handle.sim, "stage", None)
        if stage is None:
            import omni.usd  # type: ignore

            stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(OBSTACLE_PRIM_PATH)
        result["prim_valid"] = bool(prim.IsValid())
        if not prim.IsValid():
            result["error"] = f"Obstacle prim is invalid: {OBSTACLE_PRIM_PATH}"
            return result
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )

        def bounds(row: Any) -> tuple[list[float], list[float]] | None:
            info = safe_prim_world_aabb(row, cache, source=str(row.GetPath()))
            if not bool(info.get("valid", False)):
                return None
            return list(info["min"]), list(info["max"])

        overall = bounds(prim)
        if overall is None:
            result["error"] = "Obstacle visual world AABB is unavailable."
            return result
        lo, hi = overall
        result.update(
            visual_valid=True,
            measured_bounds={"min": lo, "max": hi},
            visual_bounds={"min": lo, "max": hi},
            height_m=float(hi[2] - lo[2]),
            width_m=float(hi[1] - lo[1]),
            length_m=float(hi[0] - lo[0]),
            front_face_x_m=float(lo[0]),
            center_y_m=float((lo[1] + hi[1]) * 0.5),
            bottom_z_m=float(lo[2]),
            top_z_m=float(hi[2]),
        )
        collision_lo: list[float] | None = None
        collision_hi: list[float] | None = None
        collision_paths: list[str] = []
        for child in Usd.PrimRange(prim):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            row_bounds = bounds(child)
            if row_bounds is None:
                continue
            child_lo, child_hi = row_bounds
            collision_lo = child_lo if collision_lo is None else [min(collision_lo[i], child_lo[i]) for i in range(3)]
            collision_hi = child_hi if collision_hi is None else [max(collision_hi[i], child_hi[i]) for i in range(3)]
            collision_paths.append(str(child.GetPath()))
        if collision_lo is not None and collision_hi is not None:
            result.update(
                collision_valid=True,
                collision_prim_paths=collision_paths,
                collision_bounds={"min": collision_lo, "max": collision_hi},
                collision_height_m=float(collision_hi[2] - collision_lo[2]),
                collision_width_m=float(collision_hi[1] - collision_lo[1]),
            )
        else:
            result["error"] = "Obstacle collision geometry/API is unavailable."
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _set_cuboid_size_and_translation(
    prim: Any,
    *,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> None:
    from pxr import Gf, UsdGeom  # type: ignore

    cube = UsdGeom.Cube(prim)
    cube_size = float(cube.GetSizeAttr().Get() or 1.0) if cube else 1.0
    cube_size = cube_size if abs(cube_size) > 1.0e-12 else 1.0
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        name = str(op.GetOpName()).lower()
        if "translate" in name and translate_op is None:
            translate_op = op
        elif "scale" in name and scale_op is None:
            scale_op = op
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if scale_op is None:
        scale_op = xformable.AddScaleOp()
    translate_op.Set(Gf.Vec3d(*[float(value) for value in translation]))
    scale_op.Set(Gf.Vec3d(*[float(value) / cube_size for value in size]))
    print(f"[INFO] Updated obstacle height to {float(size[2]):.3f} m.")


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
    if bool(config.infer_obstacle_size) and config.obstacle_width is None and config.obstacle_length is None:
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
    if config.obstacle_front_x is None:
        config.obstacle_front_x = OBSTACLE_FRONT_FACE_X_M
    config.obstacle_x = float(config.obstacle_front_x) + 0.5 * float(config.obstacle_length)


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
        (float(config.obstacle_x), 0.0, float(config.ground_z_m) + float(config.obstacle_height_m) * 0.5),
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


def measure_scene_baseline(scene_handle: SimSceneHandle, adapter: Any) -> dict[str, Any]:
    """Read reset/obstacle geometry from the live USD stage for baseline QA."""

    result: dict[str, Any] = {
        "available": False,
        "height_m": float(scene_handle.config.obstacle_height_m),
        "ground_z_m": float(scene_handle.config.ground_z_m),
        "obstacle_front_face_x_m": scene_handle.config.obstacle_front_x,
    }
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

        stage = getattr(scene_handle.sim, "stage", None)
        if stage is None:
            import omni.usd  # type: ignore

            stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)

        def aligned_bounds(prim: Any) -> tuple[list[float], list[float]]:
            info = safe_prim_world_aabb(prim, cache, source=str(prim.GetPath()))
            if not bool(info.get("valid", False)):
                raise ValueError(str(info.get("rejection_reason", "invalid world bound")))
            return list(info["min"]), list(info["max"])

        def subtree(root: Any) -> list[Any]:
            try:
                children = list(root.GetFilteredChildren(Usd.TraverseInstanceProxies()))
            except Exception:
                children = list(root.GetChildren())
            return [root] + [item for child in children for item in subtree(child)]

        def rigid_body_ancestor(prim: Any) -> Any | None:
            current = prim
            while current is not None and current.IsValid():
                if current.HasAPI(UsdPhysics.RigidBodyAPI):
                    return current
                parent = current.GetParent()
                if parent is None or not parent.IsValid() or parent == current:
                    break
                current = parent
            return None

        obstacle = stage.GetPrimAtPath(OBSTACLE_PRIM_PATH)
        obstacle_min: list[float] | None = None
        obstacle_max: list[float] | None = None
        if obstacle.IsValid():
            obstacle_min, obstacle_max = aligned_bounds(obstacle)

        collision_min: list[float] | None = None
        collision_max: list[float] | None = None
        wheel_centers: list[dict[str, Any]] = []
        wheel_radii: list[float] = []
        robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        for prim in subtree(robot_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body_prim = rigid_body_ancestor(prim)
            bound_info = safe_prim_world_aabb(prim, cache, source=str(prim.GetPath()))
            if not bool(bound_info.get("valid", False)) and body_prim is not None:
                bound_info = live_collider_world_aabb_from_body_pose(
                    adapter,
                    str(body_prim.GetPath()),
                    prim,
                    cache,
                )
            if not bool(bound_info.get("valid", False)):
                continue
            lo = list(bound_info["min"])
            hi = list(bound_info["max"])
            collision_min = lo if collision_min is None else [min(collision_min[i], lo[i]) for i in range(3)]
            collision_max = hi if collision_max is None else [max(collision_max[i], hi[i]) for i in range(3)]
            key = str(prim.GetPath()).lower()
            if "wheel" in key or "ankle" in key:
                center = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
                local_info = collider_mesh_body_local_aabb(body_prim, prim) if body_prim is not None else {}
                local_extent = list(local_info.get("extent", []) or [])
                if len(local_extent) >= 3:
                    radius = 0.5 * max(float(local_extent[0]), float(local_extent[2]))
                    radius_source = str(local_info.get("source", "live_collision_local_mesh"))
                else:
                    size = [max(0.0, hi[i] - lo[i]) for i in range(3)]
                    radius = 0.5 * max(size[0], size[2])
                    radius_source = str(bound_info.get("source", "live_collision_world_aabb"))
                wheel_centers.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "center_m": center,
                        "bounds_min_m": lo,
                        "bounds_max_m": hi,
                        "radius_m": radius,
                        "radius_source": radius_source,
                    }
                )
                if radius > 0.0 and math.isfinite(radius):
                    wheel_radii.append(radius)

        if not wheel_centers:
            ground_rows = list(dict(getattr(adapter, "robot_ground_diagnostics", {}) or {}).get("wheels", []) or [])
            for row in ground_rows:
                lo = list(row.get("bounds_min", []) or [])
                hi = list(row.get("bounds_max", []) or [])
                center = list(row.get("body_position_w", []) or [])
                extent = list(row.get("bounds_extent", []) or [])
                if len(lo) < 3 or len(hi) < 3 or len(center) < 3 or len(extent) < 3:
                    continue
                radius = 0.5 * max(float(extent[0]), float(extent[2]))
                wheel_centers.append(
                    {
                        "prim_path": str(row.get("collision_prim_paths", [""])[0]),
                        "center_m": [float(value) for value in center[:3]],
                        "bounds_min_m": [float(value) for value in lo[:3]],
                        "bounds_max_m": [float(value) for value in hi[:3]],
                        "radius_m": radius,
                        "radius_source": str(row.get("data_source", "ground_diagnostic_live_bounds")),
                    }
                )
                wheel_radii.append(radius)
                collision_min = lo if collision_min is None else [min(collision_min[i], lo[i]) for i in range(3)]
                collision_max = hi if collision_max is None else [max(collision_max[i], hi[i]) for i in range(3)]

        sim_state = adapter.capture_sim_state()
        root_pose = sim_state.get("root_pose")
        while isinstance(root_pose, list) and len(root_pose) == 1 and isinstance(root_pose[0], list):
            root_pose = root_pose[0]
        root_pose = list(root_pose or [])
        root_x = float(root_pose[0]) if len(root_pose) >= 1 else 0.0
        root_y = float(root_pose[1]) if len(root_pose) >= 2 else 0.0
        yaw_deg = 0.0
        if len(root_pose) >= 7:
            w, x, y, z = (float(root_pose[3]), float(root_pose[4]), float(root_pose[5]), float(root_pose[6]))
            yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        front_x = float(obstacle_min[0]) if obstacle_min is not None else float(scene_handle.config.obstacle_front_x or 0.0)
        front_wheel_x = max([float(row["center_m"][0]) for row in wheel_centers] + [root_x])
        robot_collision_front_x = float(collision_max[0]) if collision_max is not None else root_x
        wheel_radius = sorted(wheel_radii)[len(wheel_radii) // 2] if wheel_radii else None
        result.update(
            available=True,
            robot_root_pose=root_pose,
            obstacle_bounds_min_m=obstacle_min,
            obstacle_bounds_max_m=obstacle_max,
            obstacle_center_m=(
                None
                if obstacle_min is None or obstacle_max is None
                else [(obstacle_min[i] + obstacle_max[i]) * 0.5 for i in range(3)]
            ),
            obstacle_front_face_x_m=front_x,
            obstacle_bottom_z_m=None if obstacle_min is None else float(obstacle_min[2]),
            obstacle_top_z_m=None if obstacle_max is None else float(obstacle_max[2]),
            obstacle_length_m=None if obstacle_min is None or obstacle_max is None else float(obstacle_max[0] - obstacle_min[0]),
            obstacle_width_m=None if obstacle_min is None or obstacle_max is None else float(obstacle_max[1] - obstacle_min[1]),
            wheel_collision_centers=wheel_centers,
            wheel_radius_m=wheel_radius,
            robot_collision_bounds_min_m=collision_min,
            robot_collision_bounds_max_m=collision_max,
            root_to_obstacle_front_m=front_x - root_x,
            front_wheel_center_to_obstacle_front_m=front_x - front_wheel_x,
            robot_collision_front_to_obstacle_front_m=front_x - robot_collision_front_x,
            lateral_alignment_m=0.0 - root_y,
            yaw_alignment_error_deg=yaw_deg,
            canonical_approach_distance="root_to_obstacle_front_m",
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


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
