"""Formal-parity Isaac Lab scene configuration for FSM-50 residual learning.

This module intentionally has no Isaac Lab imports at module import time.  Pure
contract tests may therefore load and validate the formal scene identity before
an :class:`isaaclab.app.AppLauncher` exists.  Isaac-backed configuration objects
are constructed only by :func:`build_isaaclab_scene_bundle` (or its smoke
wrapper), which callers must invoke after AppLauncher has started Kit.

The scene is a namespaced translation of ``sim_obstacle_scene.py``.  It reuses
the production USD, obstacle, ground, material, solver, and actuator semantics;
it does not reuse the legacy PPO task, its phase logic, or its asset authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "fsm50-formal-residual-scene-v1"

MODULE_ROOT = Path(__file__).resolve().parent
REPLAY_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = REPLAY_ROOT.parent

DEFAULT_ROBOT_USD_PATH = PROJECT_ROOT / "usd" / "wlr_robot_drive_test.usd"
ENVIRONMENT_REFERENCE_PATH = REPLAY_ROOT / "config" / "environment_reference.yaml"
FORMAL_SCENE_SOURCE_PATH = REPLAY_ROOT / "sim_obstacle_scene.py"

EXPECTED_ROBOT_USD_SHA256 = "e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892"
EXPECTED_ENVIRONMENT_REFERENCE_SHA256 = "822c8e47d75de92e2165a99875a2de484425b25cf15e7404481b60cf05f81604"
EXPECTED_FORMAL_SCENE_SOURCE_SHA256 = "32747b6ee542614529b8f308bf8de09325cea1c70871b967704cb8fede93a745"

ENV_REGEX_NS = "{ENV_REGEX_NS}"
ROBOT_PRIM_PATH = f"{ENV_REGEX_NS}/Robot"
OBSTACLE_PRIM_PATH = f"{ENV_REGEX_NS}/Obstacle"
GROUND_PRIM_PATH = "/World/defaultGroundPlane"
DOME_LIGHT_PRIM_PATH = "/World/Light/Dome"
KEY_LIGHT_PRIM_PATH = "/World/Light/Key"

SERVO_JOINT_NAMES = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_JOINT_NAMES = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)

PHYSICS_DT_S = 1.0 / 120.0
DIRECT_RL_DECIMATION = 1
# Rendering remains at the production outer-cycle cadence.  This is distinct
# from DirectRLEnv decimation: safety/actuation still executes at 120 Hz.
RENDER_INTERVAL_PHYSICS_STEPS = 8

ROBOT_SPAWN_POSITION_M = (0.0, 0.0, 0.04)
ROBOT_SPAWN_ROTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
ROBOT_MAX_DEPENETRATION_VELOCITY_M_S = 1.0
ROBOT_SOLVER_POSITION_ITERATIONS = 8
ROBOT_SOLVER_VELOCITY_ITERATIONS = 2
ROBOT_SELF_COLLISIONS_ENABLED = False
# Gate-D's formal/instrumented Macro scene enables PhysX contact reporting
# before reset.  Sensor-bank construction remains an observation-layer concern,
# but the spawned articulation must preserve that physical reporting capability.
ROBOT_CONTACT_SENSORS_ENABLED = True

SERVO_EFFORT_LIMIT_NM = 2.7
SERVO_STIFFNESS_NM_RAD = 600.0
SERVO_DAMPING_NM_S_RAD = 60.0
SERVO_ARMATURE = 0.005
WHEEL_VELOCITY_LIMIT_RAD_S = math.pi / 1.5
WHEEL_STIFFNESS = 0.0
WHEEL_DAMPING_NM_S_RAD = 20.0
WHEEL_ARMATURE = 0.002

OBSTACLE_HEIGHT_M = 0.05
OBSTACLE_FRONT_X_M = 0.5213121737735307
OBSTACLE_LENGTH_M = 2.057375557085507
OBSTACLE_WIDTH_M = 2.0
OBSTACLE_BOTTOM_Z_M = 0.0
OBSTACLE_CENTER_Y_M = 0.0
ROBOT_COLLISION_WIDTH_M = 0.44110036553592025
OBSTACLE_CONTACT_OFFSET_M = 0.005
OBSTACLE_REST_OFFSET_M = 0.0
OBSTACLE_COLOR_RGB = (0.55, 0.47, 0.33)
OBSTACLE_SEMANTIC_TAGS = (("class", "height_obstacle"),)

GROUND_SIZE_M = (6.0, 6.0)
GROUND_POSITION_M = (0.0, 0.0, 0.0)
GROUND_COLOR_RGB = (0.08, 0.09, 0.10)

DEFAULT_NUM_ENVS = 1
DEFAULT_ENV_SPACING_M = 3.5
REPLICATE_PHYSICS = True
CLONE_IN_FABRIC = False

DEFAULT_CAMERA_EYE_M = (1.45, -1.25, 0.80)
DEFAULT_CAMERA_TARGET_M = (0.45, 0.0, 0.12)
DOME_LIGHT_INTENSITY = 2500.0
DOME_LIGHT_COLOR_RGB = (0.85, 0.88, 0.95)
KEY_LIGHT_INTENSITY = 1800.0
KEY_LIGHT_COLOR_RGB = (1.0, 0.96, 0.90)
KEY_LIGHT_ANGLE = 0.35
KEY_LIGHT_POSITION_M = (1.5, -2.0, 4.0)


class FormalSceneContractError(ValueError):
    """The formal scene identity or one of its strict inputs is invalid."""


class IsaacLabSceneUnavailableError(RuntimeError):
    """Isaac Lab scene classes were requested before they were available."""


@dataclass(frozen=True)
class PhysicsMaterialSpec:
    static_friction: float
    dynamic_friction: float
    restitution: float
    friction_combine_mode: str = "max"
    restitution_combine_mode: str = "min"

    def as_isaac_kwargs(self) -> dict[str, Any]:
        return asdict(self)


GROUND_MATERIAL = PhysicsMaterialSpec(
    static_friction=1.25,
    dynamic_friction=1.05,
    restitution=0.0,
)
OBSTACLE_MATERIAL = PhysicsMaterialSpec(
    static_friction=1.20,
    dynamic_friction=1.00,
    restitution=0.0,
)


@dataclass(frozen=True)
class FormalResidualSceneSpec:
    """Isaac-free, immutable identity of the formal vector scene."""

    schema_version: str
    robot_usd_path: Path
    robot_usd_sha256: str
    environment_reference_path: Path
    environment_reference_sha256: str
    formal_scene_source_path: Path
    formal_scene_source_sha256: str
    environment_profile_id: str
    physics_dt_s: float
    direct_rl_decimation: int
    render_interval_physics_steps: int
    robot_prim_path: str
    obstacle_prim_path: str
    ground_prim_path: str
    obstacle_front_x_m: float
    obstacle_length_m: float
    obstacle_width_m: float
    obstacle_height_m: float
    obstacle_center_x_m: float
    obstacle_center_y_m: float
    obstacle_bottom_z_m: float
    robot_collision_width_m: float

    @property
    def obstacle_position_m(self) -> tuple[float, float, float]:
        return (
            self.obstacle_center_x_m,
            self.obstacle_center_y_m,
            self.obstacle_bottom_z_m + 0.5 * self.obstacle_height_m,
        )

    @property
    def obstacle_size_m(self) -> tuple[float, float, float]:
        return (self.obstacle_length_m, self.obstacle_width_m, self.obstacle_height_m)

    def validate(self, *, verify_files: bool = True) -> None:
        """Fail closed on identity drift, partial values, or non-finite geometry."""

        exact_values = {
            "schema_version": (self.schema_version, SCHEMA_VERSION),
            "robot_usd_path": (self.robot_usd_path.resolve(), DEFAULT_ROBOT_USD_PATH.resolve()),
            "robot_usd_sha256": (self.robot_usd_sha256, EXPECTED_ROBOT_USD_SHA256),
            "environment_reference_path": (
                self.environment_reference_path.resolve(),
                ENVIRONMENT_REFERENCE_PATH.resolve(),
            ),
            "environment_reference_sha256": (
                self.environment_reference_sha256,
                EXPECTED_ENVIRONMENT_REFERENCE_SHA256,
            ),
            "formal_scene_source_path": (
                self.formal_scene_source_path.resolve(),
                FORMAL_SCENE_SOURCE_PATH.resolve(),
            ),
            "formal_scene_source_sha256": (
                self.formal_scene_source_sha256,
                EXPECTED_FORMAL_SCENE_SOURCE_SHA256,
            ),
            "environment_profile_id": (
                self.environment_profile_id,
                "height-obstacle-environment-v2",
            ),
            "physics_dt_s": (self.physics_dt_s, PHYSICS_DT_S),
            "direct_rl_decimation": (self.direct_rl_decimation, DIRECT_RL_DECIMATION),
            "render_interval_physics_steps": (
                self.render_interval_physics_steps,
                RENDER_INTERVAL_PHYSICS_STEPS,
            ),
            "robot_prim_path": (self.robot_prim_path, ROBOT_PRIM_PATH),
            "obstacle_prim_path": (self.obstacle_prim_path, OBSTACLE_PRIM_PATH),
            "ground_prim_path": (self.ground_prim_path, GROUND_PRIM_PATH),
            "obstacle_front_x_m": (self.obstacle_front_x_m, OBSTACLE_FRONT_X_M),
            "obstacle_length_m": (self.obstacle_length_m, OBSTACLE_LENGTH_M),
            "obstacle_width_m": (self.obstacle_width_m, OBSTACLE_WIDTH_M),
            "obstacle_height_m": (self.obstacle_height_m, OBSTACLE_HEIGHT_M),
            "obstacle_center_y_m": (self.obstacle_center_y_m, OBSTACLE_CENTER_Y_M),
            "obstacle_bottom_z_m": (self.obstacle_bottom_z_m, OBSTACLE_BOTTOM_Z_M),
            "robot_collision_width_m": (
                self.robot_collision_width_m,
                ROBOT_COLLISION_WIDTH_M,
            ),
        }
        for name, (actual, expected) in exact_values.items():
            if actual != expected:
                raise FormalSceneContractError(f"{name} drifted: expected {expected!r}, got {actual!r}")

        for name in (
            "obstacle_front_x_m",
            "obstacle_length_m",
            "obstacle_width_m",
            "obstacle_height_m",
            "obstacle_center_x_m",
            "obstacle_center_y_m",
            "obstacle_bottom_z_m",
            "robot_collision_width_m",
            "physics_dt_s",
        ):
            _require_finite_number(name, getattr(self, name))
        for name in ("obstacle_length_m", "obstacle_width_m", "obstacle_height_m", "physics_dt_s"):
            if float(getattr(self, name)) <= 0.0:
                raise FormalSceneContractError(f"{name} must be positive")

        expected_center_x = self.obstacle_front_x_m + 0.5 * self.obstacle_length_m
        if self.obstacle_center_x_m != expected_center_x:
            raise FormalSceneContractError(
                "obstacle center/front/length identity is inconsistent: "
                f"center={self.obstacle_center_x_m!r}, expected={expected_center_x!r}"
            )
        if verify_files:
            _require_sha256(self.robot_usd_path, self.robot_usd_sha256, "robot USD")
            _require_sha256(
                self.environment_reference_path,
                self.environment_reference_sha256,
                "environment reference",
            )
            _require_sha256(
                self.formal_scene_source_path,
                self.formal_scene_source_sha256,
                "formal scene source",
            )

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("robot_usd_path", "environment_reference_path", "formal_scene_source_path"):
            payload[key] = str(payload[key])
        payload.update(
            obstacle_position_m=list(self.obstacle_position_m),
            obstacle_size_m=list(self.obstacle_size_m),
            servo_joint_names=list(SERVO_JOINT_NAMES),
            wheel_joint_names=list(WHEEL_JOINT_NAMES),
            robot_physics={
                "spawn_position_m": list(ROBOT_SPAWN_POSITION_M),
                "spawn_rotation_wxyz": list(ROBOT_SPAWN_ROTATION_WXYZ),
                "max_depenetration_velocity_m_s": ROBOT_MAX_DEPENETRATION_VELOCITY_M_S,
                "solver_position_iterations": ROBOT_SOLVER_POSITION_ITERATIONS,
                "solver_velocity_iterations": ROBOT_SOLVER_VELOCITY_ITERATIONS,
                "self_collisions_enabled": ROBOT_SELF_COLLISIONS_ENABLED,
                "contact_sensors_enabled": ROBOT_CONTACT_SENSORS_ENABLED,
            },
            servo_actuator={
                "effort_limit_nm": SERVO_EFFORT_LIMIT_NM,
                "velocity_limit_sim": None,
                "stiffness_nm_rad": SERVO_STIFFNESS_NM_RAD,
                "damping_nm_s_rad": SERVO_DAMPING_NM_S_RAD,
                "armature": SERVO_ARMATURE,
            },
            wheel_actuator={
                "effort_limit_sim": None,
                "velocity_limit_rad_s": WHEEL_VELOCITY_LIMIT_RAD_S,
                "stiffness": WHEEL_STIFFNESS,
                "damping_nm_s_rad": WHEEL_DAMPING_NM_S_RAD,
                "armature": WHEEL_ARMATURE,
            },
            obstacle_physics={
                "kinematic_enabled": True,
                "disable_gravity": True,
                "contact_offset_m": OBSTACLE_CONTACT_OFFSET_M,
                "rest_offset_m": OBSTACLE_REST_OFFSET_M,
                "material": asdict(OBSTACLE_MATERIAL),
                "color_rgb": list(OBSTACLE_COLOR_RGB),
                "semantic_tags": [list(row) for row in OBSTACLE_SEMANTIC_TAGS],
            },
            ground={
                "size_m": list(GROUND_SIZE_M),
                "position_m": list(GROUND_POSITION_M),
                "color_rgb": list(GROUND_COLOR_RGB),
                "material": asdict(GROUND_MATERIAL),
            },
            vector_scene={
                "default_num_envs": DEFAULT_NUM_ENVS,
                "default_env_spacing_m": DEFAULT_ENV_SPACING_M,
                "replicate_physics": REPLICATE_PHYSICS,
                "clone_in_fabric": CLONE_IN_FABRIC,
            },
            camera={
                "eye_m": list(DEFAULT_CAMERA_EYE_M),
                "target_m": list(DEFAULT_CAMERA_TARGET_M),
            },
            lighting={
                "dome_prim_path": DOME_LIGHT_PRIM_PATH,
                "dome_intensity": DOME_LIGHT_INTENSITY,
                "dome_color_rgb": list(DOME_LIGHT_COLOR_RGB),
                "key_prim_path": KEY_LIGHT_PRIM_PATH,
                "key_intensity": KEY_LIGHT_INTENSITY,
                "key_color_rgb": list(KEY_LIGHT_COLOR_RGB),
                "key_angle": KEY_LIGHT_ANGLE,
                "key_position_m": list(KEY_LIGHT_POSITION_M),
            },
        )
        return payload

    @property
    def manifest_sha256(self) -> str:
        encoded = json.dumps(
            self.to_manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IsaacLabSceneApi:
    """Late-bound Isaac Lab configuration classes (also injectable in tests)."""

    sim_utils: Any
    ArticulationCfg: Any
    RigidObjectCfg: Any
    ImplicitActuatorCfg: Any
    InteractiveSceneCfg: Any
    SimulationCfg: Any


@dataclass(frozen=True)
class GlobalSpawnRequest:
    prim_path: str
    spawn_cfg: Any
    translation: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class IsaacLabSceneBundle:
    """Config-only scene bundle consumed by a future DirectRLEnv ``_setup_scene``."""

    spec: FormalResidualSceneSpec
    direct_rl_decimation: int
    simulation_cfg: Any
    interactive_scene_cfg: Any
    robot_cfg: Any
    obstacle_cfg: Any
    ground_spawn: GlobalSpawnRequest
    dome_light_spawn: GlobalSpawnRequest
    key_light_spawn: GlobalSpawnRequest

    @property
    def global_collision_prim_paths(self) -> tuple[str, ...]:
        return (self.ground_spawn.prim_path,)

    def smoke_manifest(self) -> dict[str, Any]:
        """Return construction evidence without creating a stage or SimulationContext."""

        return {
            "schema_version": SCHEMA_VERSION,
            "scene_manifest_sha256": self.spec.manifest_sha256,
            "direct_rl_decimation": self.direct_rl_decimation,
            "robot_prim_path": self.spec.robot_prim_path,
            "obstacle_prim_path": self.spec.obstacle_prim_path,
            "global_collision_prim_paths": list(self.global_collision_prim_paths),
            "simulation_cfg_type": _qualified_type_name(self.simulation_cfg),
            "interactive_scene_cfg_type": _qualified_type_name(self.interactive_scene_cfg),
            "robot_cfg_type": _qualified_type_name(self.robot_cfg),
            "obstacle_cfg_type": _qualified_type_name(self.obstacle_cfg),
            "stage_created": False,
            "simulation_context_created": False,
        }


def load_formal_scene_spec(*, verify_files: bool = True) -> FormalResidualSceneSpec:
    """Load the sealed environment reference and return the exact 50 mm spec."""

    try:
        reference = json.loads(ENVIRONMENT_REFERENCE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FormalSceneContractError(f"environment reference is unreadable: {exc}") from exc
    required = {
        "profile_id",
        "obstacle_width_m",
        "obstacle_length_m",
        "obstacle_front_face_x_m",
        "obstacle_center_y_m",
        "obstacle_bottom_z_m",
        "robot_collision_width_m",
    }
    missing = sorted(required - set(reference))
    if missing:
        raise FormalSceneContractError(f"environment reference is missing keys: {missing}")

    values: dict[str, float] = {}
    for name in required - {"profile_id"}:
        values[name] = _require_finite_number(name, reference[name])
    center_x = values["obstacle_front_face_x_m"] + 0.5 * values["obstacle_length_m"]
    spec = FormalResidualSceneSpec(
        schema_version=SCHEMA_VERSION,
        robot_usd_path=DEFAULT_ROBOT_USD_PATH,
        robot_usd_sha256=EXPECTED_ROBOT_USD_SHA256,
        environment_reference_path=ENVIRONMENT_REFERENCE_PATH,
        environment_reference_sha256=EXPECTED_ENVIRONMENT_REFERENCE_SHA256,
        formal_scene_source_path=FORMAL_SCENE_SOURCE_PATH,
        formal_scene_source_sha256=EXPECTED_FORMAL_SCENE_SOURCE_SHA256,
        environment_profile_id=str(reference["profile_id"]),
        physics_dt_s=PHYSICS_DT_S,
        direct_rl_decimation=DIRECT_RL_DECIMATION,
        render_interval_physics_steps=RENDER_INTERVAL_PHYSICS_STEPS,
        robot_prim_path=ROBOT_PRIM_PATH,
        obstacle_prim_path=OBSTACLE_PRIM_PATH,
        ground_prim_path=GROUND_PRIM_PATH,
        obstacle_front_x_m=values["obstacle_front_face_x_m"],
        obstacle_length_m=values["obstacle_length_m"],
        obstacle_width_m=values["obstacle_width_m"],
        obstacle_height_m=OBSTACLE_HEIGHT_M,
        obstacle_center_x_m=center_x,
        obstacle_center_y_m=values["obstacle_center_y_m"],
        obstacle_bottom_z_m=values["obstacle_bottom_z_m"],
        robot_collision_width_m=values["robot_collision_width_m"],
    )
    spec.validate(verify_files=verify_files)
    return spec


def build_isaaclab_scene_bundle(
    *,
    num_envs: int = DEFAULT_NUM_ENVS,
    env_spacing_m: float = DEFAULT_ENV_SPACING_M,
    device: str = "cuda:0",
    api: IsaacLabSceneApi | Mapping[str, Any] | None = None,
    verify_files: bool = True,
) -> IsaacLabSceneBundle:
    """Build config objects after AppLauncher, without creating or stepping a sim.

    ``api`` is injectable so pure tests can verify every constructor argument.
    When omitted, Isaac Lab imports happen here and nowhere earlier in the
    module.  The caller remains responsible for assigning ``decimation=1`` to
    its DirectRLEnvCfg and for consuming this bundle from ``_setup_scene``.
    """

    num_envs = _require_positive_int("num_envs", num_envs)
    env_spacing_m = _require_positive_finite_number("env_spacing_m", env_spacing_m)
    if not isinstance(device, str) or not device.strip():
        raise FormalSceneContractError("device must be a non-empty string")
    resolved_api = _coerce_api(api) if api is not None else _load_isaaclab_api()
    spec = load_formal_scene_spec(verify_files=verify_files)
    sim_utils = resolved_api.sim_utils

    servo_expression = "|".join(SERVO_JOINT_NAMES)
    wheel_expression = "|".join(WHEEL_JOINT_NAMES)
    robot_cfg = resolved_api.ArticulationCfg(
        prim_path=spec.robot_prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_path_for_usd(spec.robot_usd_path),
            activate_contact_sensors=ROBOT_CONTACT_SENSORS_ENABLED,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=ROBOT_MAX_DEPENETRATION_VELOCITY_M_S,
                solver_position_iteration_count=ROBOT_SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=ROBOT_SOLVER_VELOCITY_ITERATIONS,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=ROBOT_SELF_COLLISIONS_ENABLED,
                solver_position_iteration_count=ROBOT_SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=ROBOT_SOLVER_VELOCITY_ITERATIONS,
            ),
        ),
        init_state=resolved_api.ArticulationCfg.InitialStateCfg(
            pos=ROBOT_SPAWN_POSITION_M,
            rot=ROBOT_SPAWN_ROTATION_WXYZ,
            joint_pos={},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip_knee_position_servos": resolved_api.ImplicitActuatorCfg(
                joint_names_expr=[servo_expression],
                effort_limit_sim=SERVO_EFFORT_LIMIT_NM,
                velocity_limit_sim=None,
                stiffness=SERVO_STIFFNESS_NM_RAD,
                damping=SERVO_DAMPING_NM_S_RAD,
                armature=SERVO_ARMATURE,
            ),
            "wheel_velocity_motors": resolved_api.ImplicitActuatorCfg(
                joint_names_expr=[wheel_expression],
                effort_limit_sim=None,
                velocity_limit_sim=WHEEL_VELOCITY_LIMIT_RAD_S,
                stiffness=WHEEL_STIFFNESS,
                damping=WHEEL_DAMPING_NM_S_RAD,
                armature=WHEEL_ARMATURE,
            ),
        },
    )
    obstacle_cfg = resolved_api.RigidObjectCfg(
        prim_path=spec.obstacle_prim_path,
        spawn=sim_utils.CuboidCfg(
            size=spec.obstacle_size_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=OBSTACLE_CONTACT_OFFSET_M,
                rest_offset=OBSTACLE_REST_OFFSET_M,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(**OBSTACLE_MATERIAL.as_isaac_kwargs()),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBSTACLE_COLOR_RGB),
            semantic_tags=list(OBSTACLE_SEMANTIC_TAGS),
        ),
        init_state=resolved_api.RigidObjectCfg.InitialStateCfg(pos=spec.obstacle_position_m),
        collision_group=0,
    )
    ground_cfg = sim_utils.GroundPlaneCfg(
        size=GROUND_SIZE_M,
        color=GROUND_COLOR_RGB,
        physics_material=sim_utils.RigidBodyMaterialCfg(**GROUND_MATERIAL.as_isaac_kwargs()),
    )
    dome_cfg = sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=DOME_LIGHT_COLOR_RGB)
    key_cfg = sim_utils.DistantLightCfg(
        intensity=KEY_LIGHT_INTENSITY,
        color=KEY_LIGHT_COLOR_RGB,
        angle=KEY_LIGHT_ANGLE,
    )
    simulation_cfg = resolved_api.SimulationCfg(
        dt=spec.physics_dt_s,
        render_interval=spec.render_interval_physics_steps,
        device=device.strip(),
    )
    scene_cfg = resolved_api.InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing_m,
        replicate_physics=REPLICATE_PHYSICS,
        clone_in_fabric=CLONE_IN_FABRIC,
    )
    return IsaacLabSceneBundle(
        spec=spec,
        direct_rl_decimation=spec.direct_rl_decimation,
        simulation_cfg=simulation_cfg,
        interactive_scene_cfg=scene_cfg,
        robot_cfg=robot_cfg,
        obstacle_cfg=obstacle_cfg,
        ground_spawn=GlobalSpawnRequest(spec.ground_prim_path, ground_cfg, GROUND_POSITION_M),
        dome_light_spawn=GlobalSpawnRequest(DOME_LIGHT_PRIM_PATH, dome_cfg),
        key_light_spawn=GlobalSpawnRequest(KEY_LIGHT_PRIM_PATH, key_cfg, KEY_LIGHT_POSITION_M),
    )


def spawn_global_scene_assets(bundle: IsaacLabSceneBundle) -> None:
    """Spawn the formal ground and lights into an already-running Isaac stage.

    This helper never starts AppLauncher or SimulationContext.  It exists so a
    DirectRLEnv ``_setup_scene`` has one narrow, testable global-spawn seam.
    """

    if not isinstance(bundle, IsaacLabSceneBundle):
        raise TypeError("bundle must be an IsaacLabSceneBundle")
    for request in (bundle.ground_spawn, bundle.dome_light_spawn, bundle.key_light_spawn):
        func = getattr(request.spawn_cfg, "func", None)
        if not callable(func):
            raise IsaacLabSceneUnavailableError(
                f"spawn config for {request.prim_path} has no callable func"
            )
        if request.translation is None:
            func(request.prim_path, request.spawn_cfg)
        else:
            func(request.prim_path, request.spawn_cfg, translation=request.translation)


def isaac_scene_construction_smoke(
    *,
    num_envs: int = 1,
    env_spacing_m: float = DEFAULT_ENV_SPACING_M,
    device: str = "cuda:0",
    api: IsaacLabSceneApi | Mapping[str, Any] | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Construct every Isaac cfg and return evidence, without spawning or stepping.

    With the real API, invoke this only after AppLauncher.  Supplying ``api`` is
    the supported pure-test path.
    """

    return build_isaaclab_scene_bundle(
        num_envs=num_envs,
        env_spacing_m=env_spacing_m,
        device=device,
        api=api,
        verify_files=verify_files,
    ).smoke_manifest()


def _load_isaaclab_api() -> IsaacLabSceneApi:
    try:
        import isaaclab.sim as sim_utils  # type: ignore
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg  # type: ignore
        from isaaclab.scene import InteractiveSceneCfg  # type: ignore
        from isaaclab.sim import SimulationCfg  # type: ignore
    except Exception as exc:
        raise IsaacLabSceneUnavailableError(
            "Isaac Lab scene configuration is unavailable. Start AppLauncher first, "
            "then call build_isaaclab_scene_bundle()."
        ) from exc
    return IsaacLabSceneApi(
        sim_utils=sim_utils,
        ArticulationCfg=ArticulationCfg,
        RigidObjectCfg=RigidObjectCfg,
        ImplicitActuatorCfg=ImplicitActuatorCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        SimulationCfg=SimulationCfg,
    )


def _coerce_api(api: IsaacLabSceneApi | Mapping[str, Any]) -> IsaacLabSceneApi:
    if isinstance(api, IsaacLabSceneApi):
        return api
    if not isinstance(api, Mapping):
        raise TypeError("api must be IsaacLabSceneApi or a mapping")
    required = {
        "sim_utils",
        "ArticulationCfg",
        "RigidObjectCfg",
        "ImplicitActuatorCfg",
        "InteractiveSceneCfg",
        "SimulationCfg",
    }
    missing = sorted(required - set(api))
    if missing:
        raise FormalSceneContractError(f"Isaac Lab API mapping is missing keys: {missing}")
    return IsaacLabSceneApi(**{name: api[name] for name in sorted(required)})


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FormalSceneContractError(f"{label} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise FormalSceneContractError(
            f"{label} SHA256 drifted: expected {expected}, got {actual} ({path})"
        )


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalSceneContractError(f"{name} must be an exact finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FormalSceneContractError(f"{name} must be finite")
    return result


def _require_positive_finite_number(name: str, value: Any) -> float:
    result = _require_finite_number(name, value)
    if result <= 0.0:
        raise FormalSceneContractError(f"{name} must be positive")
    return result


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FormalSceneContractError(f"{name} must be a positive integer")
    return value


def _path_for_usd(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def _qualified_type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


__all__ = [
    "CLONE_IN_FABRIC",
    "DEFAULT_ENV_SPACING_M",
    "DEFAULT_NUM_ENVS",
    "DIRECT_RL_DECIMATION",
    "ENV_REGEX_NS",
    "FormalResidualSceneSpec",
    "FormalSceneContractError",
    "GROUND_PRIM_PATH",
    "IsaacLabSceneApi",
    "IsaacLabSceneBundle",
    "IsaacLabSceneUnavailableError",
    "OBSTACLE_PRIM_PATH",
    "PHYSICS_DT_S",
    "RENDER_INTERVAL_PHYSICS_STEPS",
    "ROBOT_PRIM_PATH",
    "build_isaaclab_scene_bundle",
    "isaac_scene_construction_smoke",
    "load_formal_scene_spec",
    "spawn_global_scene_assets",
]
