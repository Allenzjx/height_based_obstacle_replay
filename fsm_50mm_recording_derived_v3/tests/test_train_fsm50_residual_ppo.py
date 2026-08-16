from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import pytest
import torch

from fsm_50mm_recording_derived_v3 import fsm50_residual_eval as contract
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


def _authority(env_sha: str = contract.EXPECTED_ENV_SOURCE_SHA256) -> dict[str, object]:
    payload = {
        "graph_sha256": contract.EXPECTED_GRAPH_SHA256,
        "profile_library_sha256": contract.EXPECTED_PROFILE_LIBRARY_SHA256,
        "bundle_sha256_by_source": dict(contract.ALLOWED_BUNDLE_SHA256),
        "code_sha256": {"fsm50_residual_env.py": env_sha},
        "residual_authority": {
            "sole_composer": "fsm50_direct_command_residual.compose_direct_command_residual",
            "stage": contract.R1,
        },
    }
    return {**payload, "identity_sha256": contract.canonical_json_sha256(payload)}


def _admission(
    config: contract.ResidualPPOStageConfig | None = None,
) -> contract.TrainingAdmission:
    config = config or contract.load_stage_config()
    return contract.TrainingAdmission(
        config=config,
        seed=config.seeds[0],
        environment={"source_closure_complete": True, "environment_lock_sha256": "e" * 64},
        r0_manifests=(),
        authority_identity=_authority(),
        admission_sha256="f" * 64,
    )


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--stage",
        "R1",
        "--r0-manifest",
        str(tmp_path / "r0-a"),
        "--r0-manifest",
        str(tmp_path / "r0-b"),
        "--r0-manifest",
        str(tmp_path / "r0-c"),
        "--output-dir",
        str(tmp_path / "output"),
        "--seed",
        str(contract.load_stage_config().seeds[0]),
        *extra,
    ]


def test_dry_run_is_pure_and_does_not_construct_launcher_or_output(tmp_path: Path) -> None:
    FakeAppLauncher.events = []
    calls = []

    def admission_builder(**_kwargs):
        calls.append("admission")
        return _admission()

    result = train.main(
        _argv(tmp_path, "--dry-run"),
        app_launcher_cls=FakeAppLauncher,
        admission_builder=admission_builder,
        runtime_loader=lambda: pytest.fail("runtime imported during dry-run"),
    )
    assert result["status"] == "R1_ADMISSION_VALID_DRY_RUN"
    assert result["output_created"] is False
    assert calls == ["admission"]
    assert FakeAppLauncher.events == []
    assert not (tmp_path / "output").exists()


def test_bounded_smoke_config_preflight_preserves_exact_identity(tmp_path: Path) -> None:
    smoke = contract.load_stage_config(contract.DEFAULT_R1_SMOKE_CONFIG_PATH)
    args = train.parse_args(
        _argv(tmp_path, "--config", str(contract.DEFAULT_R1_SMOKE_CONFIG_PATH)),
        app_launcher_cls=FakeAppLauncher,
    )
    calls = []

    def admission_builder(**kwargs):
        calls.append(kwargs)
        return _admission(smoke)

    preflight = train.preflight_training(args, admission_builder=admission_builder)
    assert calls[0]["config"].to_identity() == smoke.to_identity()
    assert preflight.config.to_identity() == smoke.to_identity()
    assert preflight.admission.config.to_identity() == smoke.to_identity()
    assert train._trainer_cfg(smoke)["timesteps"] == 8_544
    assert not preflight.output_dir.exists()


def test_live_main_revalidates_before_lazy_runtime_import(tmp_path: Path) -> None:
    events: list[str] = []

    class Launcher(FakeAppLauncher):
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            FakeAppLauncher.add_app_launcher_args(parser)

        def __init__(self, args) -> None:
            assert args.headless is True
            events.append("launcher")
            self.app = SimpleNamespace(close=lambda: events.append("close"))

    def admission_builder(**_kwargs):
        events.append("admission")
        return _admission()

    def runtime_loader():
        events.append("runtime")
        return object()

    def fake_run(preflight, args, *, runtime):
        assert runtime is not None and args.headless is True
        events.append("run")
        return {"status": "fake"}

    with patch.object(train, "run_live_training", side_effect=fake_run):
        result = train.main(
            _argv(tmp_path),
            app_launcher_cls=Launcher,
            admission_builder=admission_builder,
            runtime_loader=runtime_loader,
        )
    assert result == {"status": "fake"}
    assert events == ["admission", "launcher", "admission", "runtime", "run", "close"]


def test_unknown_args_existing_output_and_r2_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        train.parse_args(_argv(tmp_path, "--checkpoint", "legacy.pt"), app_launcher_cls=FakeAppLauncher)
    (tmp_path / "output").mkdir()
    args = train.parse_args(_argv(tmp_path), app_launcher_cls=FakeAppLauncher)
    with pytest.raises(contract.ResidualPPOContractError, match="already exists"):
        train.preflight_training(args, admission_builder=lambda **_kwargs: _admission())
    r2 = _argv(tmp_path / "other")
    (tmp_path / "other").mkdir()
    r2[r2.index("R1")] = "R2"
    args = train.parse_args(r2, app_launcher_cls=FakeAppLauncher)
    with pytest.raises(contract.ResidualPPOContractError, match=contract.R2_UNAVAILABLE):
        train.preflight_training(args, admission_builder=lambda **_kwargs: pytest.fail("admitted R2"))

    safe_root = tmp_path / "strict-launcher"
    safe_root.mkdir()
    args = train.parse_args(_argv(safe_root), app_launcher_cls=FakeAppLauncher)
    args.device = "cpu"
    with pytest.raises(contract.ResidualPPOContractError, match="cuda:0"):
        train.preflight_training(args, admission_builder=lambda **_kwargs: pytest.fail("cpu admitted"))
    args.device = "cuda:0"
    args.enable_cameras = True
    with pytest.raises(contract.ResidualPPOContractError, match="noncanonical"):
        train.preflight_training(args, admission_builder=lambda **_kwargs: pytest.fail("camera admitted"))


def test_real_app_launcher_parser_preparse_keeps_gate_flags_strict(monkeypatch) -> None:
    # Regression for AppLauncher.add_app_launcher_args calling
    # parse_known_args(sys.argv) before it appends its own options.
    monkeypatch.setattr("sys.argv", ["gate-e-parser-test"])
    train_parser = train.build_parser()
    play_module = __import__(
        "fsm_50mm_recording_derived_v3.play_fsm50_residual_ppo",
        fromlist=["build_parser"],
    )
    play_parser = play_module.build_parser()
    train_required = {action.dest for action in train_parser._actions if action.required}
    play_required = {action.dest for action in play_parser._actions if action.required}
    assert {"stage", "r0_manifest", "output_dir", "seed"}.issubset(train_required)
    assert {"stage", "checkpoint_manifest", "output", "seed"}.issubset(play_required)


class FakeRawEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeWrappedEnv:
    def __init__(self, raw: FakeRawEnv) -> None:
        self.raw = raw
        self.num_envs = 9
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (115,))
        self.state_space = gym.spaces.Box(-1.0, 1.0, (115,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,))
        self.device = "cpu"

    def close(self) -> None:
        self.raw.close()


class FakeEnvModule:
    GYM_ENV_ID = contract.EXPECTED_GYM_ID
    EXPECTED_RENDER_SUBSTEPS = 8
    CRITIC_STATE_DIM = 115
    EPISODE_METRICS_SCHEMA_SHA256 = contract.EXPECTED_ENV_METRICS_SCHEMA_SHA256

    def __init__(self, env_sha: str) -> None:
        self._env_sha = env_sha

    def env_source_sha256(self) -> str:
        return self._env_sha

    def register_gym_env(self, *, gym_module) -> str:
        return self.GYM_ENV_ID

    def build_env_cfg(self, **kwargs):
        assert kwargs["num_envs"] == 9
        assert len(kwargs["entry_ids"]) == 9
        return SimpleNamespace(seed=None)


class FakeGym:
    spaces = gym.spaces

    def __init__(self) -> None:
        self.raw = FakeRawEnv()

    def make(self, env_id, **_kwargs):
        assert env_id == contract.EXPECTED_GYM_ID
        return self.raw


class FakePolicy:
    def __init__(self, _observation_space, _action_space, device) -> None:
        self.device = device

    def compute(self, inputs, role):
        assert role == "policy"
        return torch.zeros((inputs["observations"].shape[0], 12)), {}


class FakeValue:
    def __init__(self, _state_space, _action_space, device) -> None:
        self.device = device


class FakePPO:
    def __init__(self, *, models, memory, **kwargs) -> None:
        assert memory is not None
        self.models = models
        self.kwargs = kwargs

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": {"weight": torch.zeros(1)},
                "value": {"weight": torch.ones(1)},
                "observation_preprocessor": {
                    "mean": torch.zeros(115),
                    "count": torch.tensor(1.0),
                },
                "value_preprocessor": {"mean": torch.zeros(1), "count": torch.tensor(1.0)},
            },
            path,
        )


class FakeTrainer:
    trained = False

    def __init__(self, *, env, agents, cfg) -> None:
        assert env.num_envs == 9
        assert cfg["stochastic_evaluation"] is False
        self.agent = agents

    def train(self) -> None:
        self.__class__.trained = True


def test_fake_skrl_env_live_path_writes_bound_checkpoint_manifests(tmp_path: Path) -> None:
    env_sha = contract.EXPECTED_ENV_SOURCE_SHA256
    fake_gym = FakeGym()
    env_module = FakeEnvModule(env_sha)
    skrl_config = SimpleNamespace(torch=SimpleNamespace(key=None))
    runtime = train.LiveTrainingRuntime(
        gym=fake_gym,
        torch=torch,
        skrl=SimpleNamespace(config=skrl_config),
        env_module=env_module,
        skrl_vec_env_wrapper=lambda raw, ml_framework: FakeWrappedEnv(raw),
        random_memory_cls=lambda **kwargs: SimpleNamespace(**kwargs),
        ppo_cls=FakePPO,
        sequential_trainer_cls=FakeTrainer,
        running_standard_scaler_cls=object,
        policy_cls=FakePolicy,
        value_cls=FakeValue,
    )
    config = contract.load_stage_config()
    admission = contract.TrainingAdmission(
        config=config,
        seed=config.seeds[0],
        environment={"source_closure_complete": True, "environment_lock_sha256": "e" * 64},
        r0_manifests=(),
        authority_identity=_authority(env_sha),
        admission_sha256="f" * 64,
    )
    output = tmp_path / "live-output"
    preflight = train.TrainingPreflight(config=config, admission=admission, output_dir=output)
    args = SimpleNamespace(device="cuda:0")
    result = train.run_live_training(preflight, args, runtime=runtime)
    assert result["status"] == "R1_TRAINING_COMPLETE_EVALUATION_REQUIRED"
    assert result["physical_promotion_claimed"] is False
    assert FakeTrainer.trained is True
    assert fake_gym.raw.closed is True
    checkpoint = output / train.CHECKPOINT_FILENAME
    checkpoint_manifest = output / train.CHECKPOINT_MANIFEST_FILENAME
    training_manifest = output / train.TRAINING_MANIFEST_FILENAME
    assert checkpoint.is_file() and checkpoint_manifest.is_file() and training_manifest.is_file()
    assert contract.strict_json_load(checkpoint_manifest)["payload"]["checkpoint_sha256"] == contract.sha256_file(checkpoint)
    payload = contract.strict_json_load(training_manifest)["payload"]
    assert payload["train_entry_ids"] and len(payload["train_entry_ids"]) == 9
    assert len(payload["eval_zero_authority_entry_ids"]) == 3
    assert payload["physical_promotion_claimed"] is False


def test_installed_skrl_2_api_agent_trainer_save_load_and_mean_action(tmp_path: Path) -> None:
    """Static SKRL smoke: real learning API, fake env, no AppLauncher/Isaac."""

    import skrl
    from skrl.agents.torch.ppo import PPO
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.trainers.torch import SequentialTrainer

    from fsm_50mm_recording_derived_v3.fsm50_residual_models import (
        Fsm50ResidualPolicy,
        Fsm50ResidualValue,
    )

    class StaticEnv:
        num_envs = 9
        device = "cpu"
        observation_space = gym.spaces.Box(-float("inf"), float("inf"), (115,))
        state_space = gym.spaces.Box(-float("inf"), float("inf"), (115,))
        action_space = gym.spaces.Box(-1.0, 1.0, (12,))

        def reset(self):
            return torch.zeros(9, 115), {}

        def state(self):
            return torch.zeros(9, 115)

        def step(self, actions):
            assert actions.shape == (9, 12)
            zeros = torch.zeros(9, 1, dtype=torch.bool)
            return torch.zeros(9, 115), torch.zeros(9, 1), zeros, zeros, {}

        def render(self):
            return None

        def close(self):
            return None

    assert skrl.__version__ == "2.0.0"
    runtime = train.LiveTrainingRuntime(
        gym=gym,
        torch=torch,
        skrl=skrl,
        env_module=None,
        skrl_vec_env_wrapper=None,
        random_memory_cls=RandomMemory,
        ppo_cls=PPO,
        sequential_trainer_cls=SequentialTrainer,
        running_standard_scaler_cls=RunningStandardScaler,
        policy_cls=Fsm50ResidualPolicy,
        value_cls=Fsm50ResidualValue,
    )
    config = contract.load_stage_config()
    env = StaticEnv()
    memory = RandomMemory(memory_size=32, num_envs=9, device="cpu")
    agent, models = train.build_skrl_agent(
        runtime,
        config,
        env,
        experiment_dir=tmp_path,
        memory=memory,
        training=False,
    )
    train.verify_zero_mean_initialization(runtime, models["policy"], num_envs=9)
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={
            "timesteps": 1,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
            "stochastic_evaluation": False,
        },
    )
    trainer.train()
    checkpoint = tmp_path / "static-skrl.pt"
    agent.save(str(checkpoint))
    modules = train.checkpoint_module_state_sha256(checkpoint, torch_module=torch)
    assert set(modules) == {
        "policy",
        "value",
        "observation_preprocessor",
        "value_preprocessor",
    }
    eval_agent, _ = train.build_skrl_agent(
        runtime,
        config,
        env,
        experiment_dir=tmp_path,
        memory=None,
        training=False,
    )
    SequentialTrainer(
        env=env,
        agents=eval_agent,
        cfg={
            "timesteps": 0,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
            "stochastic_evaluation": False,
        },
    )
    eval_agent.load(str(checkpoint))
    mean = contract.deterministic_mean_action(
        eval_agent, torch.zeros(9, 115), torch.zeros(9, 115)
    )
    assert mean.shape == (9, 12)
    assert torch.equal(mean, torch.zeros_like(mean))
