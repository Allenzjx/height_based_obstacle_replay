from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import pytest
import torch

from fsm_50mm_recording_derived_v3 import fsm50_residual_env as env_contract
from fsm_50mm_recording_derived_v3 import fsm50_residual_eval as contract
from fsm_50mm_recording_derived_v3 import play_fsm50_residual_ppo as play
from fsm_50mm_recording_derived_v3 import train_fsm50_residual_ppo as train


class FakeAppLauncher:
    events: list[str] = []

    @staticmethod
    def add_app_launcher_args(parser) -> None:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")

    def __init__(self, args) -> None:
        assert args.headless is True
        self.events.append("launcher")
        self.app = SimpleNamespace(close=lambda: self.events.append("close"))


def _authority(env_sha: str) -> dict[str, object]:
    return {
        "identity_sha256": "f" * 64,
        "code_sha256": {"fsm50_residual_env.py": env_sha},
        "scene_manifest_sha256": "1" * 64,
        "phase_bank_sha256": "2" * 64,
        "envelope_canonical_sha256": "3" * 64,
        "actor_observation_schema_sha256": "4" * 64,
        "residual_authority": {"stage": "R1"},
    }


def _checkpoint(
    tmp_path: Path,
    *,
    env_sha: str | None = None,
    config: contract.ResidualPPOStageConfig | None = None,
) -> contract.CheckpointAdmission:
    config = config or contract.load_stage_config()
    checkpoint_path = tmp_path / "fsm50_residual_ppo_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "fsm50_residual_ppo_checkpoint_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    admission = contract.TrainingAdmission(
        config=config,
        seed=config.seeds[0],
        environment={"source_closure_complete": True, "environment_lock_sha256": "e" * 64},
        r0_manifests=(),
        authority_identity=_authority(env_sha or env_contract.env_source_sha256()),
        admission_sha256="a" * 64,
    )
    return contract.CheckpointAdmission(
        manifest_path=manifest_path,
        manifest_file_sha256=contract.sha256_file(manifest_path),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=contract.sha256_file(checkpoint_path),
        payload={
            "checkpoint_module_state_sha256": {
                "policy": "1" * 64,
                "value": "2" * 64,
                "observation_preprocessor": "3" * 64,
                "value_preprocessor": "4" * 64,
            }
        },
        training_admission=admission,
    )


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--stage",
        "R1",
        "--checkpoint-manifest",
        str(tmp_path / "fsm50_residual_ppo_checkpoint_manifest.json"),
        "--output",
        str(tmp_path / play.EVALUATION_FILENAME),
        "--seed",
        str(contract.load_stage_config().seeds[0]),
        "--episodes-per-entry",
        "3",
        *extra,
    ]


def test_dry_run_is_pure_and_existing_output_unknown_args_r2_fail(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    FakeAppLauncher.events = []
    result = play.main(
        _argv(tmp_path, "--dry-run"),
        app_launcher_cls=FakeAppLauncher,
        checkpoint_loader=lambda _path: checkpoint,
        runtime_loader=lambda: pytest.fail("runtime imported during dry-run"),
    )
    assert result["status"] == "R1_CHECKPOINT_ADMISSION_VALID_DRY_RUN"
    assert result["output_created"] is False
    assert FakeAppLauncher.events == []
    assert not (tmp_path / play.EVALUATION_FILENAME).exists()

    with pytest.raises(SystemExit):
        play.parse_args(_argv(tmp_path, "--legacy-env", "x"), app_launcher_cls=FakeAppLauncher)
    (tmp_path / play.EVALUATION_FILENAME).write_text("occupied", encoding="utf-8")
    args = play.parse_args(_argv(tmp_path), app_launcher_cls=FakeAppLauncher)
    with pytest.raises(contract.ResidualPPOContractError, match="already exists"):
        play.preflight_evaluation(args, checkpoint_loader=lambda _path: checkpoint)
    (tmp_path / play.EVALUATION_FILENAME).unlink()
    r2 = _argv(tmp_path)
    r2[r2.index("R1")] = "R2"
    args = play.parse_args(r2, app_launcher_cls=FakeAppLauncher)
    with pytest.raises(contract.ResidualPPOContractError, match=contract.R2_UNAVAILABLE):
        play.preflight_evaluation(args, checkpoint_loader=lambda _path: pytest.fail("R2 loaded"))


def test_bounded_smoke_checkpoint_config_is_accepted_for_evaluation(tmp_path: Path) -> None:
    smoke = contract.load_stage_config(contract.DEFAULT_R1_SMOKE_CONFIG_PATH)
    checkpoint = _checkpoint(tmp_path, config=smoke)
    args = play.parse_args(_argv(tmp_path), app_launcher_cls=FakeAppLauncher)
    preflight = play.preflight_evaluation(
        args, checkpoint_loader=lambda _path: checkpoint
    )
    assert preflight.checkpoint.training_admission.config.to_identity() == smoke.to_identity()
    assert smoke.payload["trainer"]["timesteps"] == 8_544


def test_live_main_revalidates_checkpoint_before_lazy_runtime(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    events: list[str] = []

    class Launcher(FakeAppLauncher):
        def __init__(self, args) -> None:
            events.append("launcher")
            self.app = SimpleNamespace(close=lambda: events.append("close"))

    def loader(_path):
        events.append("checkpoint")
        return checkpoint

    def runtime_loader():
        events.append("runtime")
        return object()

    def fake_run(preflight, args, *, runtime):
        events.append("run")
        return {"status": "fake"}

    with patch.object(play, "run_live_evaluation", side_effect=fake_run):
        result = play.main(
            _argv(tmp_path),
            app_launcher_cls=Launcher,
            checkpoint_loader=loader,
            runtime_loader=runtime_loader,
        )
    assert result == {"status": "fake"}
    assert events == ["checkpoint", "launcher", "checkpoint", "runtime", "run", "close"]


def test_live_main_preserves_primary_exception_when_close_requests_normal_exit(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    class Launcher(FakeAppLauncher):
        def __init__(self, _args) -> None:
            self.app = SimpleNamespace(close=lambda: (_ for _ in ()).throw(SystemExit(0)))

    with patch.object(play, "run_live_evaluation", side_effect=RuntimeError("live failure")):
        with pytest.raises(RuntimeError, match="live failure"):
            play.main(
                _argv(tmp_path),
                app_launcher_cls=Launcher,
                checkpoint_loader=lambda _path: checkpoint,
                runtime_loader=lambda: object(),
            )


def test_live_main_returns_success_when_close_requests_normal_exit(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    expected = {"status": "fake"}

    class Launcher(FakeAppLauncher):
        def __init__(self, _args) -> None:
            self.app = SimpleNamespace(close=lambda: (_ for _ in ()).throw(SystemExit(0)))

    with patch.object(play, "run_live_evaluation", return_value=expected):
        assert play.main(
            _argv(tmp_path),
            app_launcher_cls=Launcher,
            checkpoint_loader=lambda _path: checkpoint,
            runtime_loader=lambda: object(),
        ) == expected


def _metrics(entry_id: str, authority: dict[str, object]) -> dict[str, object]:
    source, phase = entry_id.split(":", 1)
    values = {
        "schema_version": env_contract.EPISODE_METRICS_SCHEMA_VERSION,
        "env_source_sha256": authority["code_sha256"]["fsm50_residual_env.py"],
        "scene_manifest_sha256": authority["scene_manifest_sha256"],
        "bank_sha256": authority["phase_bank_sha256"],
        "entry_sha256": "5" * 64,
        "entry_id": entry_id,
        "source_version": source,
        "phase_state": phase,
        "profile_id": "profile",
        "profile_sha256": "6" * 64,
        "source_plan_sha256": "7" * 64,
        "residual_envelope_sha256": authority["envelope_canonical_sha256"],
        "actor_observation_schema_sha256": authority["actor_observation_schema_sha256"],
        "task_success": True,
        "hard_failure": False,
        "terminal_reason": "phase profile and physical guard complete",
        "body_crossed_front_face": True,
        "required_leg_lift_seen": True,
        "final_recoverable": True,
        "posture_complete": False,
        "completion_time_s": 1.0,
        "return": 2.0,
        "physics_length": 8,
        "outer_policy_length": 1,
        "peak_abs_roll_rad": 0.1,
        "peak_abs_pitch_rad": 0.1,
        "peak_root_angular_speed_rad_s": 0.1,
        "peak_abs_servo_error_deg": 0.5,
        "peak_residual_l2": 0.0,
        "peak_residual_slew_l2": 0.0,
        "source_consumption_count": 1,
        "residual_transform_count": 1,
        "physical_command_epoch": 1,
        "last_verified_physical_command_epoch": 1,
        "physical_batch_count": 1,
        "n_plus_one_verified_count": 1,
        "dynamic_wheel_stop_count": 0,
        "exact_eight_cadence_verified": True,
        "source_cursor_boundary_verified": True,
        "one_batch_per_tick_verified": True,
        "n_plus_one_verified": True,
        "terminal_latched": True,
        "terminal_ready": True,
    }
    assert set(values) == set(env_contract.EPISODE_METRICS_FIELDS)
    return {key: values[key] for key in env_contract.EPISODE_METRICS_FIELDS}


def test_episode_metrics_projection_uses_only_frozen_deployment_telemetry() -> None:
    authority = _authority(env_contract.env_source_sha256())
    entry = contract.load_stage_config().environment["entry_ids"][0]
    runtime = SimpleNamespace(env_module=env_contract)
    row = play.episode_metrics_to_row(
        runtime,
        _metrics(entry, authority),
        arm="ppo",
        episode_index=0,
        expected_entry_id=entry,
        authority_identity=authority,
    )
    assert row["phase_completed"] is True
    assert row["exact_eight_cadence_verified"] is True
    assert row["one_batch_verified"] is True
    assert row["n_plus_one_verified"] is True
    assert row["physical_command_epoch"] == row["last_verified_physical_command_epoch"]
    assert row["residual_transform_count"] == 1
    assert row["max_normalized_residual_l2"] == 0.0
    zero_transforms = _metrics(entry, authority)
    zero_transforms["residual_transform_count"] = 0
    with pytest.raises(
        contract.ResidualPPOContractError,
        match="residual_transform_count must be positive",
    ):
        play.episode_metrics_to_row(
            runtime,
            zero_transforms,
            arm="ppo",
            episode_index=0,
            expected_entry_id=entry,
            authority_identity=authority,
        )
    incomplete_epoch = _metrics(entry, authority)
    incomplete_epoch["n_plus_one_verified_count"] = 0
    with pytest.raises(
        contract.ResidualPPOContractError,
        match="physical command epoch was not fully N\\+1 verified",
    ):
        play.episode_metrics_to_row(
            runtime,
            incomplete_epoch,
            arm="ppo",
            episode_index=0,
            expected_entry_id=entry,
            authority_identity=authority,
        )
    stale = _metrics(entry, authority)
    stale["env_source_sha256"] = "0" * 64
    with pytest.raises(contract.ResidualPPOContractError, match="authority"):
        play.episode_metrics_to_row(
            runtime,
            stale,
            arm="ppo",
            episode_index=0,
            expected_entry_id=entry,
            authority_identity=authority,
        )


class FakeRawEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEvaluationEnv:
    def __init__(self, raw, entry_ids, runtime_env_module, authority) -> None:
        self.raw = raw
        self.entry_ids = tuple(entry_ids)
        self.num_envs = len(entry_ids)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (115,))
        self.state_space = gym.spaces.Box(-1.0, 1.0, (115,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,))
        self.device = "cpu"
        self.env_module = runtime_env_module
        self.authority = authority
        self.steps = 0

    def reset(self):
        return torch.zeros((self.num_envs, 115)), {}

    def state(self):
        return torch.zeros((self.num_envs, 115))

    def step(self, actions):
        # Sampled actions in FakePPO are ones; only exact mean_actions are zero.
        assert torch.equal(actions, torch.zeros_like(actions))
        self.steps += 1
        done = torch.ones((self.num_envs, 1), dtype=torch.bool)
        infos = {
            "fsm50": {
                "schema_version": self.env_module.EPISODE_METRICS_SCHEMA_VERSION,
                "schema_sha256": self.env_module.EPISODE_METRICS_SCHEMA_SHA256,
                "per_env": [_metrics(entry, self.authority) for entry in self.entry_ids],
            }
        }
        return torch.zeros((self.num_envs, 115)), torch.ones((self.num_envs, 1)), done, ~done, infos

    def close(self) -> None:
        self.raw.close()


class FakePolicy:
    def __init__(self, *_args) -> None:
        self.device = "cpu"


class FakeValue:
    def __init__(self, *_args) -> None:
        self.device = "cpu"


class FakePPO:
    def __init__(self, **_kwargs) -> None:
        self.loaded = False

    def load(self, _path) -> None:
        self.loaded = True

    def enable_training_mode(self, enabled) -> None:
        assert enabled is False and self.loaded

    def act(self, observations, states, *, timestep, timesteps):
        sampled = torch.ones((observations.shape[0], 12))
        mean = torch.zeros_like(sampled)
        return sampled, {"mean_actions": mean}


class FakeEvalEnvModule:
    GYM_ENV_ID = contract.EXPECTED_GYM_ID
    EXPECTED_RENDER_SUBSTEPS = 8
    CRITIC_STATE_DIM = 115
    EPISODE_METRICS_FIELDS = env_contract.EPISODE_METRICS_FIELDS
    EPISODE_METRICS_SCHEMA_VERSION = env_contract.EPISODE_METRICS_SCHEMA_VERSION
    EPISODE_METRICS_SCHEMA_SHA256 = env_contract.EPISODE_METRICS_SCHEMA_SHA256

    def __init__(self, env_sha, entry_ids) -> None:
        self.env_sha = env_sha
        self.entry_ids = entry_ids

    def env_source_sha256(self):
        return self.env_sha

    def register_gym_env(self, *, gym_module):
        return self.GYM_ENV_ID

    def build_env_cfg(self, **kwargs):
        assert tuple(kwargs["entry_ids"]) == tuple(self.entry_ids)
        return SimpleNamespace(seed=None, entry_ids=tuple(kwargs["entry_ids"]))


class FakeGym:
    spaces = gym.spaces

    def __init__(self) -> None:
        self.cfg = None
        self.raw = None

    def make(self, _env_id, *, cfg, render_mode):
        self.cfg = cfg
        self.raw = FakeRawEnv()
        return self.raw


def test_ppo_arm_dispatches_mean_actions_and_covers_all_twelve_entries(tmp_path: Path) -> None:
    env_sha = contract.EXPECTED_ENV_SOURCE_SHA256
    checkpoint = _checkpoint(tmp_path, env_sha=env_sha)
    config = checkpoint.training_admission.config
    entry_ids = tuple(config.environment["entry_ids"])
    env_module = FakeEvalEnvModule(env_sha, entry_ids)
    fake_gym = FakeGym()
    authority = checkpoint.training_admission.authority_identity
    runtime = train.LiveTrainingRuntime(
        gym=fake_gym,
        torch=torch,
        skrl=SimpleNamespace(config=SimpleNamespace(torch=SimpleNamespace(key=None))),
        env_module=env_module,
        skrl_vec_env_wrapper=lambda raw, ml_framework: FakeEvaluationEnv(
            raw, entry_ids, env_module, authority
        ),
        random_memory_cls=object,
        ppo_cls=FakePPO,
        sequential_trainer_cls=lambda **_kwargs: object(),
        running_standard_scaler_cls=object,
        policy_cls=FakePolicy,
        value_cls=FakeValue,
    )
    preflight = play.EvaluationPreflight(
        checkpoint=checkpoint,
        output_path=tmp_path / play.EVALUATION_FILENAME,
        seed=config.seeds[0],
        episodes_per_entry=3,
    )
    result = play.evaluate_arm(
        runtime,
        preflight,
        SimpleNamespace(device="cpu"),
        arm="ppo",
    )
    episodes = result["payload"]["episodes"]
    assert len(episodes) == 36
    assert {row["entry_id"] for row in episodes} == set(entry_ids)
    assert [row["episode_index"] for row in episodes] == list(range(36))
    assert fake_gym.raw.closed is True
