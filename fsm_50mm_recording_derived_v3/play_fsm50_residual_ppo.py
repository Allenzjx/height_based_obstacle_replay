"""Deterministic Gate-E R1 PPO evaluation against nominal and ZERO arms.

All checkpoint, reviewed ZERO-R0, environment-lock, package and code identities
are admitted before :class:`AppLauncher` exists and revalidated immediately
after launch.  PPO decisions use SKRL's ``mean_actions`` metadata; sampled
Gaussian actions are never dispatched during evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .fsm50_residual_eval import (
    EXPECTED_GYM_ID,
    R1,
    R2,
    R2_UNAVAILABLE,
    CheckpointAdmission,
    ResidualPPOContractError,
    atomic_write_json,
    build_arm_result,
    compare_evaluation_arms,
    deterministic_mean_action,
    load_checkpoint_manifest,
    sha256_file,
)
from .train_fsm50_residual_ppo import (
    LiveTrainingRuntime,
    build_skrl_agent,
    checkpoint_module_state_sha256,
    load_live_training_runtime,
    prepare_app_launcher_args,
    validate_app_launcher_args,
    validate_wrapped_environment,
)


EVALUATION_FILENAME = "fsm50_residual_ppo_evaluation.json"


@dataclass(frozen=True)
class EvaluationPreflight:
    """Pure pre-AppLauncher evaluation identity."""

    checkpoint: CheckpointAdmission
    output_path: Path
    seed: int
    episodes_per_entry: int


def _load_app_launcher_class() -> Any:
    from isaaclab.app import AppLauncher  # type: ignore

    return AppLauncher


def build_parser(*, app_launcher_cls: Any | None = None) -> argparse.ArgumentParser:
    launcher_cls = app_launcher_cls or _load_app_launcher_class()
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate sealed Gate-E R1 nominal/ZERO/PPO arms."
    )
    # AppLauncher performs an internal pre-parse before adding its flags.  Keep
    # our options temporarily optional for that pass, then restore strict CLI
    # requirements on the returned parser.
    stage_action = parser.add_argument("--stage", choices=(R1, R2))
    checkpoint_action = parser.add_argument("--checkpoint-manifest")
    output_action = parser.add_argument("--output")
    seed_action = parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes-per-entry", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint/admission only; do not launch AppLauncher or create output.",
    )
    launcher_cls.add_app_launcher_args(parser)
    for action in (stage_action, checkpoint_action, output_action, seed_action):
        action.required = True
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    app_launcher_cls: Any | None = None,
) -> argparse.Namespace:
    return build_parser(app_launcher_cls=app_launcher_cls).parse_args(argv)


def _resolve_new_evaluation_path(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.name != EVALUATION_FILENAME:
        raise ResidualPPOContractError(
            f"evaluation output filename must be {EVALUATION_FILENAME}"
        )
    if output.exists():
        raise ResidualPPOContractError(f"evaluation output already exists: {output}")
    if not output.parent.is_dir():
        raise ResidualPPOContractError(f"evaluation output parent is missing: {output.parent}")
    return output


def preflight_evaluation(
    args: argparse.Namespace,
    *,
    checkpoint_loader: Callable[..., CheckpointAdmission] = load_checkpoint_manifest,
) -> EvaluationPreflight:
    if args.stage == R2:
        raise ResidualPPOContractError(R2_UNAVAILABLE)
    if args.stage != R1:
        raise ResidualPPOContractError(f"unsupported residual stage: {args.stage}")
    output = _resolve_new_evaluation_path(args.output)
    checkpoint = checkpoint_loader(args.checkpoint_manifest)
    config = checkpoint.training_admission.config
    validate_app_launcher_args(args, config)
    if type(args.seed) is not int or args.seed not in config.seeds:
        raise ResidualPPOContractError("evaluation seed is not in the frozen R1 allowlist")
    if type(args.episodes_per_entry) is not int or args.episodes_per_entry <= 0:
        raise ResidualPPOContractError("episodes-per-entry must be a positive integer")
    episode_count = args.episodes_per_entry * len(config.environment["entry_ids"])
    if episode_count < int(config.payload["evaluation"]["minimum_episode_count_per_arm"]):
        raise ResidualPPOContractError("evaluation episode count is below the frozen R1 minimum")
    return EvaluationPreflight(
        checkpoint=checkpoint,
        output_path=output,
        seed=args.seed,
        episodes_per_entry=args.episodes_per_entry,
    )


def _flat_bools(value: Any, *, expected: int, label: str) -> list[bool]:
    try:
        if hasattr(value, "detach"):
            values = value.detach().cpu().reshape(-1).tolist()
        else:
            values = list(value)
    except Exception as exc:
        raise ResidualPPOContractError(f"cannot decode {label}: {exc}") from exc
    if len(values) != expected or any(type(item) not in (bool, int) for item in values):
        raise ResidualPPOContractError(f"{label} does not contain one flag per environment")
    return [bool(item) for item in values]


def _exact_fsm50_metrics(
    runtime: LiveTrainingRuntime,
    infos: Any,
    *,
    expected_num_envs: int,
) -> list[Mapping[str, Any] | None]:
    if not isinstance(infos, Mapping) or not isinstance(infos.get("fsm50"), Mapping):
        raise ResidualPPOContractError("environment info lacks the exact fsm50 metrics namespace")
    root = infos["fsm50"]
    if set(root) != {"schema_version", "schema_sha256", "per_env"}:
        raise ResidualPPOContractError("fsm50 metrics namespace keys drifted")
    env_module = runtime.env_module
    if (
        root.get("schema_version") != env_module.EPISODE_METRICS_SCHEMA_VERSION
        or root.get("schema_sha256") != env_module.EPISODE_METRICS_SCHEMA_SHA256
    ):
        raise ResidualPPOContractError("fsm50 metrics schema identity drifted")
    per_env = root.get("per_env")
    if not isinstance(per_env, (list, tuple)) or len(per_env) != expected_num_envs:
        raise ResidualPPOContractError("fsm50 per-env metrics shape drifted")
    return list(per_env)


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise ResidualPPOContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ResidualPPOContractError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def episode_metrics_to_row(
    runtime: LiveTrainingRuntime,
    metrics: Mapping[str, Any],
    *,
    arm: str,
    episode_index: int,
    expected_entry_id: str,
    authority_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only frozen deployment telemetry into the comparison schema."""

    env_module = runtime.env_module
    if tuple(metrics) != tuple(env_module.EPISODE_METRICS_FIELDS):
        raise ResidualPPOContractError("episode metrics keys/order drifted")
    source = expected_entry_id.split(":", 1)[0]
    if (
        metrics.get("schema_version") != env_module.EPISODE_METRICS_SCHEMA_VERSION
        or metrics.get("env_source_sha256")
        != authority_identity["code_sha256"]["fsm50_residual_env.py"]
        or metrics.get("scene_manifest_sha256") != authority_identity["scene_manifest_sha256"]
        or metrics.get("bank_sha256") != authority_identity["phase_bank_sha256"]
        or metrics.get("residual_envelope_sha256")
        != authority_identity["envelope_canonical_sha256"]
        or metrics.get("actor_observation_schema_sha256")
        != authority_identity["actor_observation_schema_sha256"]
        or metrics.get("entry_id") != expected_entry_id
        or metrics.get("source_version") != source
    ):
        raise ResidualPPOContractError("episode metrics authority/source identity drifted")
    for key in (
        "task_success",
        "hard_failure",
        "body_crossed_front_face",
        "required_leg_lift_seen",
        "final_recoverable",
        "exact_eight_cadence_verified",
        "one_batch_per_tick_verified",
        "n_plus_one_verified",
        "terminal_latched",
        "terminal_ready",
    ):
        if type(metrics.get(key)) is not bool:
            raise ResidualPPOContractError(f"episode metric {key} must be exact bool")
    if metrics["terminal_latched"] is not True or metrics["terminal_ready"] is not True:
        raise ResidualPPOContractError("episode metrics were exposed before terminal safe-stop readiness")
    terminal_reason = metrics.get("terminal_reason")
    if not isinstance(terminal_reason, str) or not terminal_reason:
        raise ResidualPPOContractError("episode terminal reason is empty")
    episode_length = metrics.get("outer_policy_length")
    if type(episode_length) is not int or episode_length <= 0:
        raise ResidualPPOContractError("episode outer_policy_length must be a positive int")
    hard_failure = bool(metrics["hard_failure"])
    residual_transform_count = metrics.get("residual_transform_count")
    physical_batch_count = metrics.get("physical_batch_count")
    physical_command_epoch = metrics.get("physical_command_epoch")
    last_verified_epoch = metrics.get("last_verified_physical_command_epoch")
    n_plus_one_count = metrics.get("n_plus_one_verified_count")
    for key, value in (
        ("residual_transform_count", residual_transform_count),
        ("physical_batch_count", physical_batch_count),
        ("physical_command_epoch", physical_command_epoch),
        ("last_verified_physical_command_epoch", last_verified_epoch),
        ("n_plus_one_verified_count", n_plus_one_count),
    ):
        if type(value) is not int or value < 0:
            raise ResidualPPOContractError(f"episode metric {key} must be nonnegative int")
    if residual_transform_count <= 0:
        raise ResidualPPOContractError(
            "episode residual_transform_count must be positive"
        )
    if (
        physical_batch_count != physical_command_epoch
        or last_verified_epoch != physical_command_epoch
        or n_plus_one_count != physical_batch_count
    ):
        raise ResidualPPOContractError("episode physical command epoch was not fully N+1 verified")
    return {
        "arm": arm,
        "episode_index": episode_index,
        "entry_id": expected_entry_id,
        "source_version": source,
        "phase_state": str(metrics["phase_state"]),
        "episode_return": _finite_number(metrics["return"], "episode return"),
        "episode_length": episode_length,
        "phase_completed": bool(metrics["task_success"]),
        "task_success": bool(metrics["task_success"]),
        "hard_failure": hard_failure,
        "hard_failure_reason": terminal_reason if hard_failure else "",
        "body_crossed_front_face": bool(metrics["body_crossed_front_face"]),
        "required_leg_lift_completed": bool(metrics["required_leg_lift_seen"]),
        "final_recoverable": bool(metrics["final_recoverable"]),
        "max_abs_roll_rad": _finite_number(
            metrics["peak_abs_roll_rad"], "peak abs roll", nonnegative=True
        ),
        "max_abs_pitch_rad": _finite_number(
            metrics["peak_abs_pitch_rad"], "peak abs pitch", nonnegative=True
        ),
        "max_root_angular_speed_rad_s": _finite_number(
            metrics["peak_root_angular_speed_rad_s"],
            "peak root angular speed",
            nonnegative=True,
        ),
        "max_servo_error_deg": _finite_number(
            metrics["peak_abs_servo_error_deg"], "peak servo error", nonnegative=True
        ),
        "max_normalized_residual_l2": _finite_number(
            metrics["peak_residual_l2"], "peak residual L2", nonnegative=True
        ),
        "max_normalized_residual_slew_l2": _finite_number(
            metrics["peak_residual_slew_l2"], "peak residual slew L2", nonnegative=True
        ),
        "residual_transform_count": residual_transform_count,
        "physical_batch_count": physical_batch_count,
        "physical_command_epoch": physical_command_epoch,
        "last_verified_physical_command_epoch": last_verified_epoch,
        "n_plus_one_verified_count": n_plus_one_count,
        "exact_eight_cadence_verified": bool(metrics["exact_eight_cadence_verified"]),
        "one_batch_verified": bool(metrics["one_batch_per_tick_verified"]),
        "n_plus_one_verified": bool(metrics["n_plus_one_verified"]),
    }


def _make_raw_and_wrapped_env(
    runtime: LiveTrainingRuntime,
    preflight: EvaluationPreflight,
    args: argparse.Namespace,
) -> tuple[Any, Any, tuple[str, ...]]:
    config = preflight.checkpoint.training_admission.config
    entry_ids = tuple(config.environment["entry_ids"])
    runtime.env_module.register_gym_env(gym_module=runtime.gym)
    env_cfg = runtime.env_module.build_env_cfg(
        num_envs=len(entry_ids),
        device=str(getattr(args, "device", None) or "cuda:0"),
        entry_ids=entry_ids,
        episode_length_s=float(config.environment["episode_length_s"]),
    )
    env_cfg.seed = preflight.seed
    raw_env = runtime.gym.make(EXPECTED_GYM_ID, cfg=env_cfg, render_mode=None)
    env = runtime.skrl_vec_env_wrapper(raw_env, ml_framework="torch")
    validate_wrapped_environment(
        runtime,
        env,
        expected_num_envs=len(entry_ids),
        authority_identity=preflight.checkpoint.training_admission.authority_identity,
    )
    return raw_env, env, entry_ids


def evaluate_arm(
    runtime: LiveTrainingRuntime,
    preflight: EvaluationPreflight,
    args: argparse.Namespace,
    *,
    arm: str,
) -> dict[str, Any]:
    if arm not in {"nominal", "zero", "ppo"}:
        raise ResidualPPOContractError(f"unknown evaluation arm: {arm}")
    checkpoint = preflight.checkpoint
    config = checkpoint.training_admission.config
    if hasattr(runtime.skrl, "config") and hasattr(runtime.skrl.config, "torch"):
        runtime.skrl.config.torch.key = preflight.seed
    runtime.torch.manual_seed(preflight.seed)
    raw_env = None
    env = None
    try:
        raw_env, env, entry_ids = _make_raw_and_wrapped_env(runtime, preflight, args)
        agent = None
        if arm == "ppo":
            agent, _models = build_skrl_agent(
                runtime,
                config,
                env,
                experiment_dir=preflight.output_path.parent,
                memory=None,
                training=False,
            )
            # SequentialTrainer is the installed SKRL 2.0 initialization seam;
            # evaluation itself remains an explicit deterministic mean loop.
            runtime.sequential_trainer_cls(
                env=env,
                agents=agent,
                cfg={
                    "timesteps": 0,
                    "headless": True,
                    "render_interval": 1,
                    "disable_progressbar": True,
                    "close_environment_at_exit": False,
                    "environment_info": "fsm50",
                    "stochastic_evaluation": False,
                },
            )
            agent.load(str(checkpoint.checkpoint_path))
            agent.enable_training_mode(False)

        observations, _reset_info = env.reset()
        states = env.state()
        num_envs = len(entry_ids)
        counts = [0] * num_envs
        completed: dict[tuple[int, int], dict[str, Any]] = {}
        maximum_steps = int(
            math.ceil(float(config.environment["episode_length_s"]) * 15.0 + 2.0)
            * preflight.episodes_per_entry
            + 16
        )
        for _outer_step in range(maximum_steps):
            if all(count >= preflight.episodes_per_entry for count in counts):
                break
            if arm == "ppo":
                actions = deterministic_mean_action(agent, observations, states)
            else:
                # Both reference arms are truthfully ZERO action.  "nominal"
                # names the nominal-command reference, not a hidden bypass.
                actions = runtime.torch.zeros(
                    (num_envs, 12), dtype=runtime.torch.float32, device=env.device
                )
            observations, _rewards, terminated, truncated, infos = env.step(actions)
            states = env.state()
            done = [
                left or right
                for left, right in zip(
                    _flat_bools(terminated, expected=num_envs, label="terminated"),
                    _flat_bools(truncated, expected=num_envs, label="truncated"),
                )
            ]
            metrics_by_env = _exact_fsm50_metrics(
                runtime, infos, expected_num_envs=num_envs
            )
            for index, is_done in enumerate(done):
                if not is_done or counts[index] >= preflight.episodes_per_entry:
                    continue
                metrics = metrics_by_env[index]
                if not isinstance(metrics, Mapping):
                    raise ResidualPPOContractError("terminal environment has no episode metrics")
                ordinal = counts[index]
                completed[(ordinal, index)] = episode_metrics_to_row(
                    runtime,
                    metrics,
                    arm=arm,
                    episode_index=-1,
                    expected_entry_id=entry_ids[index],
                    authority_identity=checkpoint.training_admission.authority_identity,
                )
                counts[index] += 1
        if any(count != preflight.episodes_per_entry for count in counts):
            raise ResidualPPOContractError(
                f"{arm} arm did not finish the exact per-entry episode count: {counts!r}"
            )
        episodes: list[dict[str, Any]] = []
        for episode_index, key in enumerate(sorted(completed)):
            row = dict(completed[key])
            row["episode_index"] = episode_index
            episodes.append(row)
        return build_arm_result(
            arm=arm,
            episodes=episodes,
            stage_config_sha256=config.config_sha256,
            seed=preflight.seed,
            checkpoint_manifest_sha256=(
                checkpoint.manifest_file_sha256 if arm == "ppo" else ""
            ),
        )
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()


def run_live_evaluation(
    preflight: EvaluationPreflight,
    args: argparse.Namespace,
    *,
    runtime: LiveTrainingRuntime,
) -> dict[str, Any]:
    expected_modules = preflight.checkpoint.payload["checkpoint_module_state_sha256"]
    actual_modules = checkpoint_module_state_sha256(
        preflight.checkpoint.checkpoint_path,
        torch_module=runtime.torch,
    )
    if actual_modules != expected_modules:
        raise ResidualPPOContractError("checkpoint module/normalization state SHA drifted")
    results = {
        arm: evaluate_arm(runtime, preflight, args, arm=arm)
        for arm in ("nominal", "zero", "ppo")
    }
    comparison = compare_evaluation_arms(
        nominal=results["nominal"],
        zero=results["zero"],
        ppo=results["ppo"],
        checkpoint=preflight.checkpoint,
    )
    _resolve_new_evaluation_path(preflight.output_path)
    atomic_write_json(preflight.output_path, comparison)
    return {
        "status": (
            "R1_EVALUATION_CONTRACT_PASS"
            if comparison["payload"]["evaluation_contract_passed"]
            else "R1_EVALUATION_CONTRACT_FAIL"
        ),
        "evaluation_path": str(preflight.output_path),
        "evaluation_sha256": sha256_file(preflight.output_path),
        "physical_promotion_claimed": False,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    app_launcher_cls: Any | None = None,
    checkpoint_loader: Callable[..., CheckpointAdmission] = load_checkpoint_manifest,
    runtime_loader: Callable[[], LiveTrainingRuntime] = load_live_training_runtime,
) -> dict[str, Any]:
    launcher_cls = app_launcher_cls or _load_app_launcher_class()
    args = parse_args(argv, app_launcher_cls=launcher_cls)
    preflight = preflight_evaluation(args, checkpoint_loader=checkpoint_loader)
    if args.dry_run:
        return {
            "status": "R1_CHECKPOINT_ADMISSION_VALID_DRY_RUN",
            "checkpoint_manifest_sha256": preflight.checkpoint.manifest_file_sha256,
            "checkpoint_sha256": preflight.checkpoint.checkpoint_sha256,
            "output_created": False,
        }
    prepare_app_launcher_args(args)
    launcher = launcher_cls(args)
    primary_exception: BaseException | None = None
    close_exception: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        revalidated = preflight_evaluation(args, checkpoint_loader=checkpoint_loader)
        if (
            revalidated.output_path != preflight.output_path
            or revalidated.checkpoint.manifest_file_sha256
            != preflight.checkpoint.manifest_file_sha256
            or revalidated.checkpoint.checkpoint_sha256 != preflight.checkpoint.checkpoint_sha256
        ):
            raise ResidualPPOContractError("checkpoint authority changed across AppLauncher boundary")
        runtime = runtime_loader()
        result = run_live_evaluation(revalidated, args, runtime=runtime)
    except BaseException as exc:
        primary_exception = exc
    finally:
        try:
            launcher.app.close()
        except BaseException as exc:
            # Kit can request a normal process exit from close().  It must not
            # hide a live evaluation exception or prevent a successful result
            # from reaching _cli for its final JSON emission.
            if not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
                close_exception = exc
    if primary_exception is not None:
        raise primary_exception
    if close_exception is not None:
        raise close_exception
    if result is None:
        raise RuntimeError("live evaluation returned no result")
    return result


def _cli() -> int:
    try:
        result = main()
    except ResidualPPOContractError as exc:
        print(f"Gate-E R1 evaluation rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "EVALUATION_FILENAME",
    "EvaluationPreflight",
    "build_parser",
    "episode_metrics_to_row",
    "evaluate_arm",
    "main",
    "parse_args",
    "preflight_evaluation",
    "run_live_evaluation",
]
