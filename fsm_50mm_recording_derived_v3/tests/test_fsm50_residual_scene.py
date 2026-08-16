from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from command_model import SERVO_JOINT_NAMES as PRODUCTION_SERVO_JOINT_NAMES
from command_model import WHEEL_JOINT_NAMES as PRODUCTION_WHEEL_JOINT_NAMES
from motion_speed import load_motion_reference
from sim_obstacle_scene import (
    OBSTACLE_FRONT_FACE_X_M as PRODUCTION_OBSTACLE_FRONT_X_M,
    OBSTACLE_LENGTH_M as PRODUCTION_OBSTACLE_LENGTH_M,
    OBSTACLE_WIDTH_M as PRODUCTION_OBSTACLE_WIDTH_M,
    SimSceneConfig,
    resolve_obstacle_dimensions,
)
from fsm_50mm_recording_derived_v3 import fsm50_residual_scene as scene


REPLAY_ROOT = Path(__file__).resolve().parents[2]


class _SpawnRecorder:
    calls: list[tuple[str, object, dict[str, object]]] = []


class _CaptureCfg:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def func(self, prim_path: str, cfg: object, **kwargs: object) -> None:
        _SpawnRecorder.calls.append((prim_path, cfg, dict(kwargs)))


class _ArticulationCfg(_CaptureCfg):
    class InitialStateCfg(_CaptureCfg):
        pass


class _RigidObjectCfg(_CaptureCfg):
    class InitialStateCfg(_CaptureCfg):
        pass


def _fake_api() -> scene.IsaacLabSceneApi:
    sim_utils = SimpleNamespace(
        UsdFileCfg=_CaptureCfg,
        RigidBodyPropertiesCfg=_CaptureCfg,
        ArticulationRootPropertiesCfg=_CaptureCfg,
        CuboidCfg=_CaptureCfg,
        CollisionPropertiesCfg=_CaptureCfg,
        RigidBodyMaterialCfg=_CaptureCfg,
        PreviewSurfaceCfg=_CaptureCfg,
        GroundPlaneCfg=_CaptureCfg,
        DomeLightCfg=_CaptureCfg,
        DistantLightCfg=_CaptureCfg,
    )
    return scene.IsaacLabSceneApi(
        sim_utils=sim_utils,
        ArticulationCfg=_ArticulationCfg,
        RigidObjectCfg=_RigidObjectCfg,
        ImplicitActuatorCfg=_CaptureCfg,
        InteractiveSceneCfg=_CaptureCfg,
        SimulationCfg=_CaptureCfg,
    )


class FormalResidualSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        _SpawnRecorder.calls.clear()

    def test_module_import_is_isaac_free_without_app_launcher(self) -> None:
        code = (
            "import sys; "
            "before=set(sys.modules); "
            "import fsm_50mm_recording_derived_v3.fsm50_residual_scene as s; "
            "new=set(sys.modules)-before; "
            "assert not any(n == 'isaaclab' or n.startswith('isaaclab.') for n in new), sorted(new); "
            "assert s.DIRECT_RL_DECIMATION == 1"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPLAY_ROOT)
        completed = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=REPLAY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_sealed_sources_and_exact_50mm_geometry_are_bound(self) -> None:
        spec = scene.load_formal_scene_spec()
        self.assertEqual(spec.schema_version, "fsm50-formal-residual-scene-v1")
        self.assertEqual(
            spec.robot_usd_sha256,
            "e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892",
        )
        self.assertEqual(
            spec.environment_reference_sha256,
            "822c8e47d75de92e2165a99875a2de484425b25cf15e7404481b60cf05f81604",
        )
        self.assertEqual(
            spec.formal_scene_source_sha256,
            "32747b6ee542614529b8f308bf8de09325cea1c70871b967704cb8fede93a745",
        )
        self.assertEqual(spec.obstacle_front_x_m, 0.5213121737735307)
        self.assertEqual(spec.obstacle_length_m, 2.057375557085507)
        self.assertEqual(spec.obstacle_width_m, 2.0)
        self.assertEqual(spec.obstacle_height_m, 0.05)
        self.assertEqual(spec.obstacle_center_x_m, 1.5499999523162842)
        self.assertEqual(spec.obstacle_position_m, (1.5499999523162842, 0.0, 0.025))
        self.assertEqual(spec.obstacle_size_m, (2.057375557085507, 2.0, 0.05))
        self.assertEqual(spec.robot_collision_width_m, 0.44110036553592025)
        self.assertEqual(len(spec.manifest_sha256), 64)
        self.assertEqual(spec.manifest_sha256, scene.load_formal_scene_spec().manifest_sha256)
        manifest = spec.to_manifest()
        self.assertIs(manifest["robot_physics"]["contact_sensors_enabled"], True)
        self.assertEqual(manifest["robot_physics"]["solver_position_iterations"], 8)
        self.assertEqual(manifest["servo_actuator"]["effort_limit_nm"], 2.7)
        self.assertEqual(manifest["wheel_actuator"]["velocity_limit_rad_s"], math.pi / 1.5)
        self.assertEqual(manifest["obstacle_physics"]["material"]["static_friction"], 1.20)
        self.assertEqual(manifest["ground"]["material"]["static_friction"], 1.25)

    def test_formal_reference_file_is_exact_json_payload(self) -> None:
        payload = json.loads(scene.ENVIRONMENT_REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["profile_id"], "height-obstacle-environment-v2")
        self.assertEqual(payload["obstacle_front_face_x_m"], 0.5213121737735307)
        self.assertEqual(payload["obstacle_length_m"], 2.057375557085507)
        self.assertEqual(payload["obstacle_width_m"], 2.0)
        self.assertEqual(payload["obstacle_bottom_z_m"], 0.0)

    def test_scene_constants_match_current_production_semantics(self) -> None:
        config = SimSceneConfig(obstacle_height_m=0.05, save_scene=False)
        resolve_obstacle_dimensions(config, SimpleNamespace())
        self.assertEqual(scene.SERVO_JOINT_NAMES, tuple(PRODUCTION_SERVO_JOINT_NAMES))
        self.assertEqual(scene.WHEEL_JOINT_NAMES, tuple(PRODUCTION_WHEEL_JOINT_NAMES))
        self.assertEqual(scene.WHEEL_VELOCITY_LIMIT_RAD_S, load_motion_reference().wheel_velocity_limit_rad_s)
        self.assertEqual(scene.WHEEL_VELOCITY_LIMIT_RAD_S, math.pi / 1.5)
        self.assertEqual(config.physics_dt, scene.PHYSICS_DT_S)
        self.assertEqual(config.render_interval, scene.RENDER_INTERVAL_PHYSICS_STEPS)
        self.assertEqual(config.spawn_z, scene.ROBOT_SPAWN_POSITION_M[2])
        self.assertEqual(config.servo_stiffness, scene.SERVO_STIFFNESS_NM_RAD)
        self.assertEqual(config.servo_damping, scene.SERVO_DAMPING_NM_S_RAD)
        self.assertEqual(config.wheel_damping, scene.WHEEL_DAMPING_NM_S_RAD)
        self.assertEqual(config.obstacle_front_x, PRODUCTION_OBSTACLE_FRONT_X_M)
        self.assertEqual(config.obstacle_length, PRODUCTION_OBSTACLE_LENGTH_M)
        self.assertEqual(config.obstacle_width, PRODUCTION_OBSTACLE_WIDTH_M)
        self.assertEqual(config.obstacle_x, 1.5499999523162842)

    def test_namespaced_paths_and_distinct_control_render_cadences(self) -> None:
        spec = scene.load_formal_scene_spec()
        self.assertEqual(spec.robot_prim_path, "{ENV_REGEX_NS}/Robot")
        self.assertEqual(spec.obstacle_prim_path, "{ENV_REGEX_NS}/Obstacle")
        self.assertEqual(spec.ground_prim_path, "/World/defaultGroundPlane")
        self.assertEqual(spec.physics_dt_s, 1.0 / 120.0)
        self.assertEqual(spec.direct_rl_decimation, 1)
        self.assertEqual(spec.render_interval_physics_steps, 8)

    def test_fake_isaac_bundle_has_exact_sim_scene_robot_and_obstacle_cfg(self) -> None:
        bundle = scene.build_isaaclab_scene_bundle(
            num_envs=7,
            env_spacing_m=3.5,
            device="cuda:2",
            api=_fake_api(),
        )
        self.assertEqual(bundle.direct_rl_decimation, 1)
        self.assertEqual(bundle.simulation_cfg.dt, 1.0 / 120.0)
        self.assertEqual(bundle.simulation_cfg.render_interval, 8)
        self.assertEqual(bundle.simulation_cfg.device, "cuda:2")
        self.assertEqual(bundle.interactive_scene_cfg.num_envs, 7)
        self.assertEqual(bundle.interactive_scene_cfg.env_spacing, 3.5)
        self.assertIs(bundle.interactive_scene_cfg.replicate_physics, True)
        self.assertIs(bundle.interactive_scene_cfg.clone_in_fabric, False)

        robot = bundle.robot_cfg
        self.assertEqual(robot.prim_path, "{ENV_REGEX_NS}/Robot")
        self.assertEqual(
            robot.spawn.usd_path,
            "C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd",
        )
        self.assertIs(robot.spawn.activate_contact_sensors, True)
        self.assertIs(robot.spawn.rigid_props.disable_gravity, False)
        self.assertEqual(robot.spawn.rigid_props.max_depenetration_velocity, 1.0)
        self.assertEqual(robot.spawn.rigid_props.solver_position_iteration_count, 8)
        self.assertEqual(robot.spawn.rigid_props.solver_velocity_iteration_count, 2)
        self.assertIs(robot.spawn.articulation_props.enabled_self_collisions, False)
        self.assertEqual(robot.spawn.articulation_props.solver_position_iteration_count, 8)
        self.assertEqual(robot.spawn.articulation_props.solver_velocity_iteration_count, 2)
        self.assertEqual(robot.init_state.pos, (0.0, 0.0, 0.04))
        self.assertEqual(robot.init_state.rot, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(robot.init_state.joint_pos, {})
        self.assertEqual(robot.init_state.joint_vel, {".*": 0.0})

        servo = robot.actuators["hip_knee_position_servos"]
        wheel = robot.actuators["wheel_velocity_motors"]
        self.assertEqual(servo.joint_names_expr, ["|".join(PRODUCTION_SERVO_JOINT_NAMES)])
        self.assertEqual(servo.effort_limit_sim, 2.7)
        self.assertIsNone(servo.velocity_limit_sim)
        self.assertEqual((servo.stiffness, servo.damping, servo.armature), (600.0, 60.0, 0.005))
        self.assertEqual(wheel.joint_names_expr, ["|".join(PRODUCTION_WHEEL_JOINT_NAMES)])
        self.assertIsNone(wheel.effort_limit_sim)
        self.assertEqual(wheel.velocity_limit_sim, math.pi / 1.5)
        self.assertEqual((wheel.stiffness, wheel.damping, wheel.armature), (0.0, 20.0, 0.002))

        obstacle = bundle.obstacle_cfg
        self.assertEqual(obstacle.prim_path, "{ENV_REGEX_NS}/Obstacle")
        self.assertEqual(obstacle.spawn.size, (2.057375557085507, 2.0, 0.05))
        self.assertIs(obstacle.spawn.rigid_props.kinematic_enabled, True)
        self.assertIs(obstacle.spawn.rigid_props.disable_gravity, True)
        self.assertEqual(obstacle.spawn.collision_props.contact_offset, 0.005)
        self.assertEqual(obstacle.spawn.collision_props.rest_offset, 0.0)
        self.assertEqual(obstacle.spawn.physics_material.static_friction, 1.20)
        self.assertEqual(obstacle.spawn.physics_material.dynamic_friction, 1.00)
        self.assertEqual(obstacle.spawn.physics_material.restitution, 0.0)
        self.assertEqual(obstacle.spawn.physics_material.friction_combine_mode, "max")
        self.assertEqual(obstacle.spawn.physics_material.restitution_combine_mode, "min")
        self.assertEqual(obstacle.spawn.visual_material.diffuse_color, (0.55, 0.47, 0.33))
        self.assertEqual(obstacle.spawn.semantic_tags, [("class", "height_obstacle")])
        self.assertEqual(obstacle.init_state.pos, (1.5499999523162842, 0.0, 0.025))
        self.assertEqual(obstacle.collision_group, 0)

    def test_ground_and_lighting_match_formal_scene(self) -> None:
        bundle = scene.build_isaaclab_scene_bundle(api=_fake_api())
        ground = bundle.ground_spawn
        self.assertEqual(ground.prim_path, "/World/defaultGroundPlane")
        self.assertEqual(ground.translation, (0.0, 0.0, 0.0))
        self.assertEqual(ground.spawn_cfg.size, (6.0, 6.0))
        self.assertEqual(ground.spawn_cfg.color, (0.08, 0.09, 0.10))
        self.assertEqual(ground.spawn_cfg.physics_material.static_friction, 1.25)
        self.assertEqual(ground.spawn_cfg.physics_material.dynamic_friction, 1.05)
        self.assertEqual(ground.spawn_cfg.physics_material.restitution, 0.0)
        self.assertEqual(ground.spawn_cfg.physics_material.friction_combine_mode, "max")
        self.assertEqual(ground.spawn_cfg.physics_material.restitution_combine_mode, "min")
        self.assertEqual(bundle.global_collision_prim_paths, ("/World/defaultGroundPlane",))

        self.assertEqual(bundle.dome_light_spawn.prim_path, "/World/Light/Dome")
        self.assertIsNone(bundle.dome_light_spawn.translation)
        self.assertEqual(bundle.dome_light_spawn.spawn_cfg.intensity, 2500.0)
        self.assertEqual(bundle.dome_light_spawn.spawn_cfg.color, (0.85, 0.88, 0.95))
        self.assertEqual(bundle.key_light_spawn.prim_path, "/World/Light/Key")
        self.assertEqual(bundle.key_light_spawn.translation, (1.5, -2.0, 4.0))
        self.assertEqual(bundle.key_light_spawn.spawn_cfg.intensity, 1800.0)
        self.assertEqual(bundle.key_light_spawn.spawn_cfg.color, (1.0, 0.96, 0.90))
        self.assertEqual(bundle.key_light_spawn.spawn_cfg.angle, 0.35)

    def test_global_spawn_helper_is_explicit_and_exact_once(self) -> None:
        bundle = scene.build_isaaclab_scene_bundle(api=_fake_api())
        self.assertEqual(_SpawnRecorder.calls, [])
        scene.spawn_global_scene_assets(bundle)
        self.assertEqual(
            [(path, kwargs) for path, _cfg, kwargs in _SpawnRecorder.calls],
            [
                ("/World/defaultGroundPlane", {"translation": (0.0, 0.0, 0.0)}),
                ("/World/Light/Dome", {}),
                ("/World/Light/Key", {"translation": (1.5, -2.0, 4.0)}),
            ],
        )

    def test_construction_smoke_builds_cfg_only_and_never_a_stage(self) -> None:
        report = scene.isaac_scene_construction_smoke(num_envs=3, api=_fake_api())
        self.assertEqual(report["schema_version"], "fsm50-formal-residual-scene-v1")
        self.assertEqual(report["direct_rl_decimation"], 1)
        self.assertEqual(report["robot_prim_path"], "{ENV_REGEX_NS}/Robot")
        self.assertEqual(report["obstacle_prim_path"], "{ENV_REGEX_NS}/Obstacle")
        self.assertFalse(report["stage_created"])
        self.assertFalse(report["simulation_context_created"])
        self.assertEqual(_SpawnRecorder.calls, [])

    def test_spec_and_builder_fail_closed_on_drift_or_invalid_inputs(self) -> None:
        spec = scene.load_formal_scene_spec()
        bad_specs = (
            replace(spec, direct_rl_decimation=8),
            replace(spec, obstacle_center_x_m=spec.obstacle_center_x_m + 0.01),
            replace(spec, obstacle_length_m=spec.obstacle_length_m + 0.01),
            replace(spec, obstacle_width_m=float("nan")),
            replace(spec, robot_usd_sha256="0" * 64),
            replace(spec, robot_prim_path="/World/WLRRobot"),
        )
        for bad in bad_specs:
            with self.subTest(bad=bad):
                with self.assertRaises(scene.FormalSceneContractError):
                    bad.validate(verify_files=False)

        for invalid in (0, -1, True, 1.0):
            with self.subTest(num_envs=invalid):
                with self.assertRaises(scene.FormalSceneContractError):
                    scene.build_isaaclab_scene_bundle(num_envs=invalid, api=_fake_api())  # type: ignore[arg-type]
        for invalid in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(env_spacing_m=invalid):
                with self.assertRaises(scene.FormalSceneContractError):
                    scene.build_isaaclab_scene_bundle(env_spacing_m=invalid, api=_fake_api())
        for invalid in ("", "   ", None):
            with self.subTest(device=invalid):
                with self.assertRaises(scene.FormalSceneContractError):
                    scene.build_isaaclab_scene_bundle(device=invalid, api=_fake_api())  # type: ignore[arg-type]

    def test_api_mapping_must_be_complete(self) -> None:
        with self.assertRaisesRegex(scene.FormalSceneContractError, "missing keys"):
            scene.build_isaaclab_scene_bundle(api={"sim_utils": SimpleNamespace()})


if __name__ == "__main__":
    unittest.main()
