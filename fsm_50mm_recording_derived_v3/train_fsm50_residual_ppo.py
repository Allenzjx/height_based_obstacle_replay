"""Strict Gate-E R1 PPO training entry point.

The import surface of this module is deliberately Isaac-, Torch-, Gym- and
SKRL-free.  A canonical reviewed ZERO-R0 admission is checked before
``AppLauncher`` is constructed, checked again after the app exists, and only
then are the live simulation and learning packages imported.  R2 is an
explicit fail-closed placeholder until a clean-S0 full-episode environment is
sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .fsm50_residual_eval import (
    DEFAULT_R1_CONFIG_PATH,
    EXPECTED_ACTION_DIM,
    EXPECTED_ENV_METRICS_SCHEMA_SHA256,
    EXPECTED_ENV_SOURCE_SHA256,
    EXPECTED_GYM_ID,
    EXPECTED_OBSERVATION_DIM,
    EXPECTED_SUBSTEPS,
    R1,
    R2,
    R2_UNAVAILABLE,
    TRAINING_MANIFEST_SCHEMA,
    ResidualPPOContractError,
    ResidualPPOStageConfig,
    TrainingAdmission,
    atomic_write_json,
    build_checkpoint_manifest,
    build_training_admission,
    canonical_json_sha256,
    load_stage_config,
    sha256_file,
)


CHECKPOINT_FILENAME = "fsm50_residual_ppo_checkpoint.pt"
CHECKPOINT_MANIFEST_FILENAME = "fsm50_residual_ppo_checkpoint_manifest.json"
TRAINING_MANIFEST_FILENAME = "fsm50_residual_ppo_training_manifest.json"
SKRL_VERSION = "2.0.0"


@dataclass(frozen=True)
class TrainingPreflight:
    """All pure, pre-AppLauncher training inputs."""

    config: ResidualPPOStageConfig
    admission: TrainingAdmission
    output_dir: Path


@dataclass(frozen=True)
class LiveTrainingRuntime:
    """Late-imported production dependencies (also a narrow fake-test seam)."""

    gym: Any
    torch: Any
    skrl: Any
    env_module: Any
    skrl_vec_env_wrapper: Any
    random_memory_cls: Any
    ppo_cls: Any
    sequential_trainer_cls: Any
    running_standard_scaler_cls: Any
    policy_cls: Any
    value_cls: Any


def _load_app_launcher_class() -> Any:
    from isaaclab.app import AppLauncher  # type: ignore

    return AppLauncher


def build_parser(*, app_launcher_cls: Any | None = None) -> argparse.ArgumentParser:
    """Build the exact CLI without importing Isaac unless explicitly needed."""

    launcher_cls = app_launcher_cls or _load_app_launcher_class()
    parser = argparse.ArgumentParser(
        description="Train the sealed Gate-E R1 phase-local residual PPO policy."
    )
    # AppLauncher.add_app_launcher_args() internally calls parse_known_args on
    # process sys.argv.  Add our options as temporarily optional so the public
    # parser can also be built programmatically, then restore exact required
    # semantics after AppLauncher's collision check.
    stage_action = parser.add_argument("--stage", choices=(R1, R2))
    parser.add_argument("--config", default=str(DEFAULT_R1_CONFIG_PATH))
    r0_action = parser.add_argument("--r0-manifest", action="append")
    output_action = parser.add_argument("--output-dir")
    seed_action = parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate immutable admission only; do not launch AppLauncher or create output.",
    )
    launcher_cls.add_app_launcher_args(parser)
    for action in (stage_action, r0_action, output_action, seed_action):
        action.required = True
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    app_launcher_cls: Any | None = None,
) -> argparse.Namespace:
    """Parse strictly; unknown/legacy/checkpoint-resume arguments are rejected."""

    return build_parser(app_launcher_cls=app_launcher_cls).parse_args(argv)


def _resolve_new_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise ResidualPPOContractError(f"training output already exists: {output}")
    if not output.parent.is_dir():
        raise ResidualPPOContractError(f"training output parent is missing: {output.parent}")
    return output


def validate_app_launcher_args(
    args: argparse.Namespace,
    config: ResidualPPOStageConfig,
) -> None:
    """Reject launcher options that could select a different live runtime."""

    if str(getattr(args, "device", "")) != str(config.environment["device"]):
        raise ResidualPPOContractError("Gate-E R1 requires the frozen cuda:0 device")
    if int(getattr(args, "livestream", -1)) not in (-1, 0):
        raise ResidualPPOContractError("Gate-E R1 forbids livestream launch modes")
    forbidden = {
        "enable_cameras": bool(getattr(args, "enable_cameras", False)),
        "xr": bool(getattr(args, "xr", False)),
        "cpu": bool(getattr(args, "cpu", False)),
        "anim_recording_enabled": bool(getattr(args, "anim_recording_enabled", False)),
        "custom_experience": bool(str(getattr(args, "experience", "") or "")),
        "custom_rendering_mode": getattr(args, "rendering_mode", None) is not None,
        "custom_kit_args": bool(str(getattr(args, "kit_args", "") or "")),
    }
    selected = sorted(name for name, enabled in forbidden.items() if enabled)
    if selected:
        raise ResidualPPOContractError(
            f"Gate-E R1 forbids noncanonical AppLauncher options: {selected!r}"
        )


def prepare_app_launcher_args(args: argparse.Namespace) -> None:
    """Seal environment-variable-sensitive AppLauncher modes before launch."""

    args.headless = True
    args.livestream = 0
    args.enable_cameras = False
    args.xr = False
    args.cpu = False
    args.experience = ""
    args.rendering_mode = None
    args.kit_args = ""
    args.anim_recording_enabled = False


def preflight_training(
    args: argparse.Namespace,
    *,
    admission_builder: Callable[..., TrainingAdmission] = build_training_admission,
) -> TrainingPreflight:
    """Complete every pure authority check before AppLauncher construction."""

    if args.stage == R2:
        raise ResidualPPOContractError(R2_UNAVAILABLE)
    if args.stage != R1:
        raise ResidualPPOContractError(f"unsupported residual stage: {args.stage}")
    manifests = tuple(args.r0_manifest or ())
    if len(manifests) != 3 or len({str(Path(item).resolve()) for item in manifests}) != 3:
        raise ResidualPPOContractError(
            "R1 requires exactly three distinct reviewed ZERO-R0 manifests"
        )
    output = _resolve_new_output_dir(args.output_dir)
    config = load_stage_config(args.config, expected_stage=R1)
    validate_app_launcher_args(args, config)
    admission = admission_builder(
        config=config,
        seed=args.seed,
        r0_manifest_paths=manifests,
    )
    if admission.config.config_sha256 != config.config_sha256:
        raise ResidualPPOContractError("training admission returned a different stage config")
    return TrainingPreflight(config=config, admission=admission, output_dir=output)


def load_live_training_runtime() -> LiveTrainingRuntime:
    """Import Isaac/Gym/Torch/SKRL only after an AppLauncher exists."""

    import gymnasium as gym  # type: ignore
    import skrl  # type: ignore
    import torch  # type: ignore
    from isaaclab_rl.skrl import SkrlVecEnvWrapper  # type: ignore
    from skrl.agents.torch.ppo import PPO  # type: ignore
    from skrl.memories.torch import RandomMemory  # type: ignore
    from skrl.resources.preprocessors.torch import RunningStandardScaler  # type: ignore
    from skrl.trainers.torch import SequentialTrainer  # type: ignore

    from . import fsm50_residual_env as env_module
    from .fsm50_residual_models import Fsm50ResidualPolicy, Fsm50ResidualValue

    if str(skrl.__version__) != SKRL_VERSION:
        raise ResidualPPOContractError(
            f"Gate-E requires exact SKRL {SKRL_VERSION}, got {skrl.__version__}"
        )
    return LiveTrainingRuntime(
        gym=gym,
        torch=torch,
        skrl=skrl,
        env_module=env_module,
        skrl_vec_env_wrapper=SkrlVecEnvWrapper,
        random_memory_cls=RandomMemory,
        ppo_cls=PPO,
        sequential_trainer_cls=SequentialTrainer,
        running_standard_scaler_cls=RunningStandardScaler,
        policy_cls=Fsm50ResidualPolicy,
        value_cls=Fsm50ResidualValue,
    )


def _flat_dim(runtime: LiveTrainingRuntime, space: Any, label: str) -> int:
    try:
        value = int(runtime.gym.spaces.flatdim(space))
    except Exception as exc:
        raise ResidualPPOContractError(f"cannot flatten SKRL {label} space: {exc}") from exc
    if value <= 0:
        raise ResidualPPOContractError(f"SKRL {label} space is empty")
    return value


def validate_wrapped_environment(
    runtime: LiveTrainingRuntime,
    env: Any,
    *,
    expected_num_envs: int,
    authority_identity: Mapping[str, Any],
) -> None:
    """Require the exact deployable actor, critic, action and cadence ABI."""

    if int(getattr(env, "num_envs", -1)) != expected_num_envs:
        raise ResidualPPOContractError("wrapped environment num_envs drifted")
    if _flat_dim(runtime, env.observation_space, "observation") != EXPECTED_OBSERVATION_DIM:
        raise ResidualPPOContractError("wrapped actor observation is not exact 115-D")
    if env.state_space is None or _flat_dim(runtime, env.state_space, "state") != EXPECTED_OBSERVATION_DIM:
        raise ResidualPPOContractError(
            "wrapped critic state is unavailable or not the exact 115-D actor telemetry"
        )
    if _flat_dim(runtime, env.action_space, "action") != EXPECTED_ACTION_DIM:
        raise ResidualPPOContractError("wrapped residual action is not exact canonical 12-D")
    env_module = runtime.env_module
    if (
        env_module.GYM_ENV_ID != EXPECTED_GYM_ID
        or int(env_module.EXPECTED_RENDER_SUBSTEPS) != EXPECTED_SUBSTEPS
        or int(env_module.CRITIC_STATE_DIM) != EXPECTED_OBSERVATION_DIM
        or env_module.EPISODE_METRICS_SCHEMA_SHA256 != EXPECTED_ENV_METRICS_SCHEMA_SHA256
        or env_module.env_source_sha256() != EXPECTED_ENV_SOURCE_SHA256
        or env_module.env_source_sha256()
        != authority_identity["code_sha256"]["fsm50_residual_env.py"]
    ):
        raise ResidualPPOContractError("live R1 environment identity/cadence drifted")


def _ppo_cfg(
    runtime: LiveTrainingRuntime,
    config: ResidualPPOStageConfig,
    env: Any,
    *,
    experiment_dir: Path,
    training: bool,
) -> dict[str, Any]:
    ppo = dict(config.payload["ppo"])
    normalization = config.payload["normalization"]
    trainer = config.payload["trainer"]
    return {
        **ppo,
        "observation_preprocessor": runtime.running_standard_scaler_cls,
        "observation_preprocessor_kwargs": {
            "size": env.observation_space,
            "epsilon": normalization["epsilon"],
            "clip_threshold": normalization["clip_threshold"],
            "device": env.device,
        },
        "value_preprocessor": runtime.running_standard_scaler_cls,
        "value_preprocessor_kwargs": {
            "size": 1,
            "epsilon": normalization["epsilon"],
            "clip_threshold": normalization["clip_threshold"],
            "device": env.device,
        },
        "experiment": {
            "directory": str(experiment_dir),
            "experiment_name": "skrl_internal",
            "write_interval": trainer["write_interval"] if training else 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }


def build_skrl_agent(
    runtime: LiveTrainingRuntime,
    config: ResidualPPOStageConfig,
    env: Any,
    *,
    experiment_dir: Path,
    memory: Any | None,
    training: bool,
) -> tuple[Any, Mapping[str, Any]]:
    """Build the exact SKRL 2.0 PPO actor/critic and normalizers."""

    policy = runtime.policy_cls(env.observation_space, env.action_space, env.device)
    value = runtime.value_cls(env.state_space, env.action_space, env.device)
    models = {"policy": policy, "value": value}
    agent = runtime.ppo_cls(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        cfg=_ppo_cfg(
            runtime,
            config,
            env,
            experiment_dir=experiment_dir,
            training=training,
        ),
    )
    return agent, models


def verify_zero_mean_initialization(
    runtime: LiveTrainingRuntime,
    policy: Any,
    *,
    num_envs: int,
) -> None:
    """Prove the deterministic actor begins at the exact nominal command."""

    zeros = runtime.torch.zeros(
        (num_envs, EXPECTED_OBSERVATION_DIM), dtype=runtime.torch.float32, device=policy.device
    )
    with runtime.torch.no_grad():
        mean, _metadata = policy.compute({"observations": zeros}, role="policy")
    if tuple(mean.shape) != (num_envs, EXPECTED_ACTION_DIM):
        raise ResidualPPOContractError("initial actor mean shape is not [num_envs,12]")
    if not bool(runtime.torch.equal(mean, runtime.torch.zeros_like(mean))):
        raise ResidualPPOContractError("R1 actor mean head is not initialized to exact zero")


def _hash_state_value(hasher: Any, value: Any, torch_module: Any) -> None:
    if isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_state_value(hasher, key, torch_module)
            _hash_state_value(hasher, value[key], torch_module)
        hasher.update(b"}")
        return
    if isinstance(value, (tuple, list)):
        hasher.update(b"tuple[" if isinstance(value, tuple) else b"list[")
        for item in value:
            _hash_state_value(hasher, item, torch_module)
        hasher.update(b"]")
        return
    if isinstance(value, torch_module.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor:")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        # ``view(dtype)`` rejects zero-dimensional tensors unless they are
        # flattened first; SKRL normalizers contain scalar count state.
        hasher.update(
            tensor.reshape(-1).view(torch_module.uint8).numpy().tobytes(order="C")
        )
        return
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise ResidualPPOContractError("checkpoint state contains non-finite scalar metadata")
        hasher.update(type(value).__name__.encode("ascii"))
        hasher.update(repr(value).encode("utf-8"))
        return
    raise ResidualPPOContractError(
        f"unsupported checkpoint state value: {type(value).__module__}.{type(value).__name__}"
    )


def checkpoint_module_state_sha256(
    checkpoint_path: str | Path,
    *,
    torch_module: Any,
) -> dict[str, str]:
    """Hash required learned/normalization state independent of zip metadata."""

    try:
        state = torch_module.load(
            str(Path(checkpoint_path).resolve(strict=True)),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ResidualPPOContractError(f"cannot decode the newly saved SKRL checkpoint: {exc}") from exc
    if not isinstance(state, Mapping):
        raise ResidualPPOContractError("SKRL checkpoint root is not a mapping")
    required = ("policy", "value", "observation_preprocessor", "value_preprocessor")
    if any(name not in state for name in required):
        raise ResidualPPOContractError("SKRL checkpoint omits required model/normalization state")
    result: dict[str, str] = {}
    for name in required:
        digest = hashlib.sha256()
        _hash_state_value(digest, state[name], torch_module)
        result[name] = digest.hexdigest()
    return result


def _trainer_cfg(config: ResidualPPOStageConfig) -> dict[str, Any]:
    trainer = config.payload["trainer"]
    return {
        "timesteps": trainer["timesteps"],
        "headless": trainer["headless"],
        "render_interval": 1,
        "disable_progressbar": trainer["disable_progressbar"],
        "close_environment_at_exit": False,
        "environment_info": "fsm50",
        "stochastic_evaluation": False,
    }


def run_live_training(
    preflight: TrainingPreflight,
    args: argparse.Namespace,
    *,
    runtime: LiveTrainingRuntime,
) -> dict[str, Any]:
    """Execute the live R1 run after the post-AppLauncher revalidation."""

    output = _resolve_new_output_dir(preflight.output_dir)
    config = preflight.config
    admission = preflight.admission
    entry_roles = config.environment["entry_roles"]
    train_entry_ids = tuple(
        entry_id
        for entry_id in config.environment["entry_ids"]
        if entry_roles[entry_id] == "TRAIN_NONZERO_AUTHORITY"
    )
    if len(train_entry_ids) != int(config.environment["num_envs"]):
        raise ResidualPPOContractError("R1 train entry role/count drifted")
    runtime.env_module.register_gym_env(gym_module=runtime.gym)
    env_cfg = runtime.env_module.build_env_cfg(
        num_envs=len(train_entry_ids),
        device=str(getattr(args, "device", None) or "cuda:0"),
        entry_ids=train_entry_ids,
        episode_length_s=float(config.environment["episode_length_s"]),
    )
    env_cfg.seed = admission.seed
    runtime.torch.manual_seed(admission.seed)
    if hasattr(runtime.skrl, "config") and hasattr(runtime.skrl.config, "torch"):
        runtime.skrl.config.torch.key = admission.seed

    raw_env = None
    env = None
    try:
        raw_env = runtime.gym.make(EXPECTED_GYM_ID, cfg=env_cfg, render_mode=None)
        env = runtime.skrl_vec_env_wrapper(raw_env, ml_framework="torch")
        validate_wrapped_environment(
            runtime,
            env,
            expected_num_envs=len(train_entry_ids),
            authority_identity=admission.authority_identity,
        )
        output.mkdir(parents=False, exist_ok=False)
        memory = runtime.random_memory_cls(
            memory_size=int(config.payload["ppo"]["rollouts"]),
            num_envs=len(train_entry_ids),
            device=env.device,
        )
        agent, models = build_skrl_agent(
            runtime,
            config,
            env,
            experiment_dir=output,
            memory=memory,
            training=True,
        )
        verify_zero_mean_initialization(runtime, models["policy"], num_envs=len(train_entry_ids))
        trainer = runtime.sequential_trainer_cls(
            env=env,
            agents=agent,
            cfg=_trainer_cfg(config),
        )
        trainer.train()

        checkpoint = output / CHECKPOINT_FILENAME
        temporary_checkpoint = output / f".{CHECKPOINT_FILENAME}.tmp"
        if checkpoint.exists() or temporary_checkpoint.exists():
            raise ResidualPPOContractError("checkpoint target unexpectedly exists")
        agent.save(str(temporary_checkpoint))
        if not temporary_checkpoint.is_file():
            raise ResidualPPOContractError("SKRL did not create the requested checkpoint")
        os.replace(temporary_checkpoint, checkpoint)
        modules = checkpoint_module_state_sha256(checkpoint, torch_module=runtime.torch)
        checkpoint_manifest = build_checkpoint_manifest(
            checkpoint_path=checkpoint,
            admission=admission,
            checkpoint_module_state_sha256=modules,
        )
        checkpoint_manifest_path = output / CHECKPOINT_MANIFEST_FILENAME
        atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)
        training_payload = {
            "stage": R1,
            "live_isaac_execution": True,
            "algorithm": "PPO",
            "framework": "torch",
            "skrl_version": SKRL_VERSION,
            "ppo_training_performed": True,
            "physical_promotion_claimed": False,
            "source_split": dict(config.source_split),
            "train_entry_ids": list(train_entry_ids),
            "eval_zero_authority_entry_ids": [
                entry_id
                for entry_id in config.environment["entry_ids"]
                if entry_roles[entry_id] == "EVAL_ZERO_AUTHORITY_ONLY"
            ],
            "seed": admission.seed,
            "training_admission_sha256": admission.admission_sha256,
            "authority_identity_sha256": admission.authority_identity["identity_sha256"],
            "stage_config": config.to_identity(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "checkpoint_module_state_sha256": modules,
            "exact_substeps_per_action": EXPECTED_SUBSTEPS,
            "actor_observation_dim": EXPECTED_OBSERVATION_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "deterministic_evaluation_action": "mean_actions",
        }
        training_manifest = {
            "schema_version": TRAINING_MANIFEST_SCHEMA,
            "payload_sha256": canonical_json_sha256(training_payload),
            "payload": training_payload,
        }
        training_manifest_path = output / TRAINING_MANIFEST_FILENAME
        atomic_write_json(training_manifest_path, training_manifest)
        return {
            "status": "R1_TRAINING_COMPLETE_EVALUATION_REQUIRED",
            "training_manifest_path": str(training_manifest_path),
            "training_manifest_sha256": sha256_file(training_manifest_path),
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "physical_promotion_claimed": False,
        }
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()


def main(
    argv: Sequence[str] | None = None,
    *,
    app_launcher_cls: Any | None = None,
    admission_builder: Callable[..., TrainingAdmission] = build_training_admission,
    runtime_loader: Callable[[], LiveTrainingRuntime] = load_live_training_runtime,
) -> dict[str, Any]:
    """Strict CLI main.  ``--dry-run`` never constructs AppLauncher."""

    launcher_cls = app_launcher_cls or _load_app_launcher_class()
    args = parse_args(argv, app_launcher_cls=launcher_cls)
    preflight = preflight_training(args, admission_builder=admission_builder)
    if args.dry_run:
        return {
            "status": "R1_ADMISSION_VALID_DRY_RUN",
            "stage_config_sha256": preflight.config.config_sha256,
            "training_admission_sha256": preflight.admission.admission_sha256,
            "output_created": False,
        }

    # R1 is headless by contract.  AppLauncher is the first live dependency.
    prepare_app_launcher_args(args)
    launcher = launcher_cls(args)
    try:
        # Close the TOCTOU window before importing Gym/Torch/SKRL/Isaac env code.
        revalidated = preflight_training(args, admission_builder=admission_builder)
        if (
            revalidated.output_dir != preflight.output_dir
            or revalidated.admission.admission_sha256 != preflight.admission.admission_sha256
        ):
            raise ResidualPPOContractError("training authority changed across AppLauncher boundary")
        runtime = runtime_loader()
        return run_live_training(revalidated, args, runtime=runtime)
    finally:
        launcher.app.close()


def _cli() -> int:
    try:
        result = main()
    except ResidualPPOContractError as exc:
        print(f"Gate-E R1 training rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_MANIFEST_FILENAME",
    "LiveTrainingRuntime",
    "TRAINING_MANIFEST_FILENAME",
    "TrainingPreflight",
    "build_parser",
    "build_skrl_agent",
    "checkpoint_module_state_sha256",
    "load_live_training_runtime",
    "main",
    "parse_args",
    "preflight_training",
    "run_live_training",
    "prepare_app_launcher_args",
    "validate_app_launcher_args",
    "validate_wrapped_environment",
    "verify_zero_mean_initialization",
]
