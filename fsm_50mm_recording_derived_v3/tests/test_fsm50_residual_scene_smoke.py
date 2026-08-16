from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from fsm_50mm_recording_derived_v3 import fsm50_residual_scene_smoke as smoke


REPLAY_ROOT = Path(__file__).resolve().parents[2]


def _rehash(envelope: dict[str, object]) -> None:
    payload = envelope["payload"]
    envelope["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _success_runtime(request: smoke.ValidatedSceneSmokeRequest) -> dict[str, object]:
    env_paths = [f"/World/envs/env_{index}" for index in range(request.num_envs)]
    return {
        "device": request.device,
        "num_envs": request.num_envs,
        "env_spacing_m": request.env_spacing_m,
        "physics_dt_s": 1.0 / 120.0,
        "direct_rl_decimation": 1,
        "render_interval_physics_steps": 8,
        "physics_steps_requested": request.physics_steps,
        "physics_steps_completed": request.physics_steps,
        "scene_env_prim_paths": env_paths,
        "scene_prim_validation": {
            "per_environment": {
                path: {"robot": True, "obstacle": True} for path in env_paths
            },
            "ground": True,
            "collision_filter_prim": True,
            "all_expected_prims_valid": True,
        },
        "collision_filter": {
            "clone_called": True,
            "copy_from_source": False,
            "filter_called": True,
            "global_collision_prim_paths": ["/World/defaultGroundPlane"],
        },
        "servo_joint_ids": list(range(8)),
        "servo_joint_names": list(smoke.SERVO_JOINT_NAMES),
        "wheel_joint_ids": list(range(8, 12)),
        "wheel_joint_names": list(smoke.WHEEL_JOINT_NAMES),
        "wheel_body_ids": list(range(4, 8)),
        "wheel_body_names": list(smoke.WHEEL_BODY_NAMES),
        "actuation": {
            "zero_servo_command_verified": True,
            "zero_wheel_command_verified": True,
            "standing_target_verified_all_steps": True,
            "target_echo_verified_all_steps": True,
            "servo_target_write_count": request.physics_steps,
            "wheel_target_write_count": request.physics_steps,
            "servo_reference_velocity_deg_s": 150.0,
            "max_delta_deg_per_physics_step": 1.25,
            "rate_limit_probe_requested_deg": 10.0,
            "rate_limit_probe_applied_deg": 1.25,
            "rate_limit_probe_applied_to_robot": False,
            "max_abs_servo_target_echo_error_rad": 0.0,
            "max_abs_wheel_target_echo_error_rad_s": 0.0,
            "final_max_abs_standing_position_error_rad": 0.001,
        },
        "finite_state": {
            "all_finite": True,
            "physics_frames_checked": request.physics_steps,
            "max_abs_by_tensor": {"joint_position_rad": 0.1},
        },
        "package_versions": {
            "isaaclab": "0.54.3",
            "isaacsim": "5.1.0.0",
            "torch": "2.7.0+cu128",
        },
        "process_id": os.getpid(),
    }


class SceneSmokeContractTests(unittest.TestCase):
    def test_import_is_isaac_free_and_does_not_self_launch(self) -> None:
        code = (
            "import sys; before=set(sys.modules); "
            "import fsm_50mm_recording_derived_v3.fsm50_residual_scene_smoke as s; "
            "new=set(sys.modules)-before; "
            "assert not any(n == 'isaaclab' or n.startswith('isaaclab.') for n in new), sorted(new); "
            "assert s.SMOKE_PHYSICS_STEPS == 24"
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

    def test_request_round_trip_is_exact_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result_path = Path(temp) / "result.json"
            envelope = smoke.build_smoke_request(
                request_id="scene-smoke-001",
                result_path=result_path,
                device="cuda:0",
            )
            request = smoke.validate_smoke_request(envelope)
            self.assertEqual(request.request_id, "scene-smoke-001")
            self.assertEqual(request.result_path, result_path.resolve())
            self.assertEqual(request.num_envs, 2)
            self.assertEqual(request.physics_steps, 24)
            self.assertEqual(request.env_spacing_m, 3.5)
            self.assertEqual(request.servo_reference_velocity_deg_s, 150.0)
            self.assertTrue(request.headless_required)
            self.assertEqual(request.expected_scene_source_sha256, smoke.FROZEN_SCENE_SOURCE_SHA256)
            self.assertEqual(request.expected_scene_manifest_sha256, smoke.FROZEN_SCENE_MANIFEST_SHA256)
            self.assertEqual(request.expected_smoke_source_sha256, hashlib.sha256(smoke.SMOKE_SOURCE_PATH.read_bytes()).hexdigest())

    def test_request_file_load_binds_exact_file_and_payload_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            result_path = root / "result.json"
            envelope = smoke.build_smoke_request(request_id="scene-smoke-load", result_path=result_path)
            smoke.write_json_atomic(request_path, envelope)
            loaded = smoke.load_smoke_request(request_path)
            self.assertEqual(loaded.request_path, request_path.resolve())
            self.assertEqual(loaded.request.result_path, result_path.resolve())
            self.assertEqual(loaded.request_file_sha256, hashlib.sha256(request_path.read_bytes()).hexdigest())
            self.assertEqual(loaded.request.payload_sha256, envelope["payload_sha256"])

    def test_request_rejects_tamper_extra_fields_and_relaxed_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = smoke.build_smoke_request(
                request_id="scene-smoke-negative", result_path=Path(temp) / "result.json"
            )
            bad_rows: list[dict[str, object]] = []
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["num_envs"] = 1  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["physics_steps"] = 16  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["headless_required"] = False  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["result_path"] = "relative.json"  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["expected_scene_source_sha256"] = "0" * 64  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["extra"] = True  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload"]["device"] = "cuda:-1"  # type: ignore[index]
            _rehash(corrupted)
            bad_rows.append(corrupted)
            corrupted = copy.deepcopy(base)
            corrupted["payload_sha256"] = "0" * 64
            bad_rows.append(corrupted)
            for index, row in enumerate(bad_rows):
                with self.subTest(index=index):
                    with self.assertRaises(smoke.SceneSmokeContractError):
                        smoke.validate_smoke_request(row, verify_files=False)

    def test_request_and_result_paths_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request_path = Path(temp) / "request.json"
            envelope = smoke.build_smoke_request(
                request_id="same-path-negative", result_path=request_path
            )
            smoke.write_json_atomic(request_path, envelope)
            with self.assertRaisesRegex(smoke.SceneSmokeContractError, "must differ"):
                smoke.load_smoke_request(request_path)

    def test_pass_result_schema_is_exact_and_request_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            request_envelope = smoke.build_smoke_request(
                request_id="scene-smoke-pass", result_path=root / "result.json"
            )
            smoke.write_json_atomic(request_path, request_envelope)
            loaded = smoke.load_smoke_request(request_path)
            runtime = _success_runtime(loaded.request)
            result = smoke.build_smoke_result(
                loaded,
                status="PASS",
                runtime=runtime,
                closure={
                    "simulation_context_cleared": True,
                    "application_close_requested": True,
                },
                completed_utc="2026-08-15T12:00:00+00:00",
            )
            payload = smoke.validate_smoke_result(result, loaded=loaded)
            self.assertEqual(payload["status"], "PASS")
            self.assertIsNone(payload["error"])
            self.assertEqual(payload["request_file_sha256"], loaded.request_file_sha256)
            self.assertEqual(payload["scene_manifest_sha256"], smoke.FROZEN_SCENE_MANIFEST_SHA256)

    def test_pass_result_rejects_false_finite_target_or_closure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            smoke.write_json_atomic(
                request_path,
                smoke.build_smoke_request(
                    request_id="scene-smoke-pass-negative", result_path=root / "result.json"
                ),
            )
            loaded = smoke.load_smoke_request(request_path)
            for mutation in ("finite", "target", "closure"):
                runtime = _success_runtime(loaded.request)
                closure = {
                    "simulation_context_cleared": True,
                    "application_close_requested": True,
                }
                if mutation == "finite":
                    runtime["finite_state"]["all_finite"] = False  # type: ignore[index]
                elif mutation == "target":
                    runtime["actuation"]["target_echo_verified_all_steps"] = False  # type: ignore[index]
                else:
                    closure["application_close_requested"] = False
                with self.subTest(mutation=mutation):
                    with self.assertRaises(smoke.SceneSmokeContractError):
                        smoke.build_smoke_result(
                            loaded,
                            status="PASS",
                            runtime=runtime,
                            closure=closure,
                            completed_utc="2026-08-15T12:00:00+00:00",
                        )

    def test_fail_result_requires_typed_sha_bound_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            smoke.write_json_atomic(
                request_path,
                smoke.build_smoke_request(
                    request_id="scene-smoke-fail", result_path=root / "result.json"
                ),
            )
            loaded = smoke.load_smoke_request(request_path)
            result = smoke.build_smoke_result(
                loaded,
                status="FAIL",
                runtime={},
                closure={
                    "simulation_context_cleared": False,
                    "application_close_requested": True,
                },
                error={
                    "type": "RuntimeError",
                    "message": "synthetic failure",
                    "traceback_sha256": hashlib.sha256(b"synthetic").hexdigest(),
                },
                completed_utc="2026-08-15T12:00:00Z",
            )
            payload = smoke.validate_smoke_result(result, loaded=loaded)
            self.assertEqual(payload["status"], "FAIL")
            self.assertEqual(payload["error"]["type"], "RuntimeError")

    def test_atomic_writer_refuses_overwrite_and_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "artifact.json"
            envelope = {"hello": "world"}
            smoke.write_json_atomic(target, envelope)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), envelope)
            with self.assertRaisesRegex(smoke.SceneSmokeContractError, "already exists"):
                smoke.write_json_atomic(target, envelope)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_rate_limit_probe_is_exact_150_deg_per_second_at_120hz(self) -> None:
        self.assertEqual(smoke.SERVO_MAX_DELTA_DEG_PER_STEP, 1.25)
        self.assertEqual(smoke.rate_limit_probe(), 1.25)
        self.assertEqual(smoke.rate_limit_probe(requested_deg=-10.0), -1.25)
        self.assertEqual(smoke.rate_limit_probe(current_deg=2.0, requested_deg=2.5), 2.5)
        for invalid in (0.0, -1.0, float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(smoke.SceneSmokeContractError):
                    smoke.rate_limit_probe(dt_s=invalid)  # type: ignore[arg-type]

    def test_exact_resolver_rejects_missing_reordered_or_duplicate_identity(self) -> None:
        expected = ("a", "b", "c")

        def good(names: list[str], preserve_order: bool = False):
            self.assertTrue(preserve_order)
            return [1, 2, 3], names

        self.assertEqual(smoke._resolve_exact(good, expected, "items"), ([1, 2, 3], ["a", "b", "c"]))
        bad = (
            lambda names, preserve_order=False: ([1, 2], names[:2]),
            lambda names, preserve_order=False: ([1, 2, 3], list(reversed(names))),
            lambda names, preserve_order=False: ([1, 1, 3], names),
            lambda names, preserve_order=False: ([1, -2, 3], names),
        )
        for index, resolver in enumerate(bad):
            with self.subTest(index=index):
                with self.assertRaises(smoke.SceneSmokeContractError):
                    smoke._resolve_exact(resolver, expected, "items")

    def test_manual_scene_asset_configs_expand_exact_env_regex_namespace(self) -> None:
        class FakeCfg:
            def __init__(self, prim_path: str):
                self.prim_path = prim_path

            def replace(self, **kwargs):
                return FakeCfg(kwargs["prim_path"])

        robot = smoke._expand_namespaced_asset_cfg(
            FakeCfg("{ENV_REGEX_NS}/Robot"), "/World/envs/env_.*", "robot"
        )
        obstacle = smoke._expand_namespaced_asset_cfg(
            FakeCfg("{ENV_REGEX_NS}/Obstacle"), "/World/envs/env_.*", "obstacle"
        )
        self.assertEqual(robot.prim_path, "/World/envs/env_.*/Robot")
        self.assertEqual(obstacle.prim_path, "/World/envs/env_.*/Obstacle")
        for bad_cfg, namespace in (
            (FakeCfg("/World/WLRRobot"), "/World/envs/env_.*"),
            (FakeCfg("{ENV_REGEX_NS}/Robot"), "/World/Wrong"),
        ):
            with self.subTest(path=bad_cfg.prim_path, namespace=namespace):
                with self.assertRaises(smoke.SceneSmokeContractError):
                    smoke._expand_namespaced_asset_cfg(bad_cfg, namespace, "asset")

    def test_main_cli_can_be_exercised_with_fake_app_launcher_module(self) -> None:
        captured: dict[str, object] = {}

        class FakeAppLauncher:
            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--headless", action="store_true")
                parser.add_argument("--device", default="cuda:0")

        fake_isaaclab = types.ModuleType("isaaclab")
        fake_app = types.ModuleType("isaaclab.app")
        fake_app.AppLauncher = FakeAppLauncher
        sentinel = object()

        def fake_run(args, loaded):
            captured["args"] = args
            captured["loaded"] = loaded
            return 0

        with mock.patch.dict(
            sys.modules,
            {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app},
            clear=False,
        ), mock.patch.object(smoke, "load_smoke_request", return_value=sentinel), mock.patch.object(
            smoke, "_run_with_app_launcher", side_effect=fake_run
        ):
            code = smoke.main(
                ["--request", "C:/sealed/request.json", "--headless", "--device", "cuda:0"]
            )
        self.assertEqual(code, 0)
        self.assertIs(captured["loaded"], sentinel)
        args = captured["args"]
        self.assertEqual(args.request, "C:/sealed/request.json")
        self.assertTrue(args.headless)
        self.assertEqual(args.device, "cuda:0")

    def test_fake_launcher_run_closes_app_and_atomically_writes_pass_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            result_path = root / "result.json"
            smoke.write_json_atomic(
                request_path,
                smoke.build_smoke_request(
                    request_id="fake-launch-pass", result_path=result_path, device="cuda:0"
                ),
            )
            loaded = smoke.load_smoke_request(request_path)
            fake_application = mock.Mock()

            class FakeAppLauncher:
                def __init__(self, args):
                    self.app = fake_application

            fake_isaaclab = types.ModuleType("isaaclab")
            fake_app_module = types.ModuleType("isaaclab.app")
            fake_app_module.AppLauncher = FakeAppLauncher
            args = Namespace(device="cuda:0", headless=True)
            with mock.patch.dict(
                sys.modules,
                {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app_module},
                clear=False,
            ), mock.patch.object(
                smoke,
                "_execute_real_isaac_smoke",
                return_value=(_success_runtime(loaded.request), True),
            ):
                exit_code = smoke._run_with_app_launcher(args, loaded)
            self.assertEqual(exit_code, 0)
            fake_application.close.assert_called_once_with()
            self.assertTrue(result_path.is_file())
            result = json.loads(result_path.read_text(encoding="utf-8"))
            payload = smoke.validate_smoke_result(result, loaded=loaded)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(
                payload["closure"],
                {
                    "simulation_context_cleared": True,
                    "application_close_requested": True,
                },
            )

    def test_result_is_fsynced_before_nonreturning_or_raising_app_close(self) -> None:
        class CloseSentinel(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            result_path = root / "result.json"
            smoke.write_json_atomic(
                request_path,
                smoke.build_smoke_request(
                    request_id="close-sentinel-pass",
                    result_path=result_path,
                    device="cuda:0",
                ),
            )
            loaded = smoke.load_smoke_request(request_path)

            class TerminatingApplication:
                def close(self):
                    self.result_existed_at_close = result_path.is_file()
                    if self.result_existed_at_close:
                        envelope = json.loads(result_path.read_text(encoding="utf-8"))
                        payload = smoke.validate_smoke_result(envelope, loaded=loaded)
                        self.close_contract_at_close = payload["closure"]
                    raise CloseSentinel("models SimulationApp.close() not returning")

            application = TerminatingApplication()

            class FakeAppLauncher:
                def __init__(self, args):
                    self.app = application

            fake_isaaclab = types.ModuleType("isaaclab")
            fake_app_module = types.ModuleType("isaaclab.app")
            fake_app_module.AppLauncher = FakeAppLauncher
            args = Namespace(device="cuda:0", headless=True)
            with mock.patch.dict(
                sys.modules,
                {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app_module},
                clear=False,
            ), mock.patch.object(
                smoke,
                "_execute_real_isaac_smoke",
                return_value=(_success_runtime(loaded.request), True),
            ):
                with self.assertRaisesRegex(CloseSentinel, "not returning"):
                    smoke._run_with_app_launcher(args, loaded)
            self.assertTrue(application.result_existed_at_close)
            self.assertEqual(
                application.close_contract_at_close,
                {
                    "simulation_context_cleared": True,
                    "application_close_requested": True,
                },
            )
            self.assertTrue(result_path.is_file())


if __name__ == "__main__":
    unittest.main()
