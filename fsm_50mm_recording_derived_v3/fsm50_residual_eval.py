"""Fail-closed Gate-E PPO admission, checkpoint and evaluation contracts.

This module is intentionally Isaac-free. Live entry points must validate these
contracts before constructing :class:`isaaclab.app.AppLauncher`, and may import
Gym, Isaac Lab, Torch and SKRL only after the app exists. No function in this
module grants physical authority or treats a pending ZERO-R0 run as reviewed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_R1_CONFIG_PATH = MODULE_ROOT / "configs" / "fsm50_residual_ppo_r1.json"
DEFAULT_R1_SMOKE_CONFIG_PATH = (
    MODULE_ROOT / "configs" / "fsm50_residual_ppo_r1_smoke.json"
)
DEFAULT_ENVIRONMENT_LOCK_PATH = MODULE_ROOT / "reports" / "environment_lock_50mm.json"

STAGE_CONFIG_SCHEMA = "fsm50.residual_ppo_stage_config.v1"
AUTHORITY_IDENTITY_SCHEMA = "fsm50.residual_ppo_authority_identity.v1"
CHECKPOINT_MANIFEST_SCHEMA = "fsm50.residual_ppo_checkpoint_manifest.v1"
TRAINING_MANIFEST_SCHEMA = "fsm50.residual_ppo_training_manifest.v1"
EPISODE_METRICS_SCHEMA = "fsm50.residual_ppo_episode_metrics.v1"
ARM_RESULT_SCHEMA = "fsm50.residual_ppo_arm_result.v1"
EVALUATION_MANIFEST_SCHEMA = "fsm50.residual_ppo_evaluation.v1"

R1 = "R1"
R2 = "R2"
R2_UNAVAILABLE = "UNAVAILABLE_PENDING_FULL_EPISODE_SEAM"
POLICY_KIND = "PPO"
DETERMINISTIC_ACTION = "mean_actions"

V003 = "v003_20260805_224517_157723_manual"
V008 = "v008_20260806_211408_578700_manual"
V009 = "v009_20260806_215232_433234_manual"
V010_FAILED = "v010_20260806_220745_363972_manual"
TRAIN_SOURCES = (V003, V008, V009)
HELD_OUT_SOURCES = (V010_FAILED,)
ALLOWED_BUNDLE_SHA256 = {
    V003: "5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78",
    V008: "0a47cee05cbb41a3a09b23836386bddcf9ae1ab2aca5bbe451ee20a0cbaa8c41",
    V009: "0e5686354eed7d89fb57d3aba722602eee7115b44749d531b6634dd0420ee919",
}
EXPECTED_GRAPH_SHA256 = "ffa5acfbf64b65c22eee54709a2afae5a56fa0b9345d8db84eb86acac58447c5"
EXPECTED_PROFILE_LIBRARY_SHA256 = (
    "762665911785eb755b7000fbb2cb450a52f5579f3945d8f214129f41eaa50066"
)
EXPECTED_ENVELOPE_CANONICAL_SHA256 = (
    "fa5002690737d94fab7304f40044293da46de34cd197af19f3f1140047ae7fbe"
)
EXPECTED_R1_CONFIG_CANONICAL_SHA256 = (
    "9df218c850a5c19862bca7197da3a200d3b5ec11f75eaba4b09b0e530e773e25"
)
EXPECTED_R1_SMOKE_CONFIG_CANONICAL_SHA256 = (
    "f84b927e181f532705e426516a836c1a304d64f0fce0bf1a61994bf33656b046"
)
EXPECTED_PHASE_BANK_SHA256 = (
    "46cfd697d130225478e87a7bf890285e682ca455b172d424ef6c04f3c33d8177"
)
EXPECTED_ENV_SOURCE_SHA256 = (
    "3dde9bd36b1bf584cb7cbc62560a894cc9bd9b93ad9f839562a7470b8671da44"
)
EXPECTED_ENV_METRICS_SCHEMA_SHA256 = (
    "1968b9c6f93e1de8e5d6cc7a9453aebb7d74853194991bc7509051608ee59151"
)
EXPECTED_REVIEWED_R0_RUNNER_SHA256 = (
    "b253d22c9647537a0c9c31e3bc27ac436a76e65fe52f145c36d97d8b4daf47db"
)
EXPECTED_OBSERVATION_SCHEMA_SHA256 = (
    "6552a6cad5358f36ef4d60b0291658951c4a3901203e3943fd8770e5605bdb83"
)
EXPECTED_OBSERVATION_DIM = 115
EXPECTED_ACTION_DIM = 12
EXPECTED_SUBSTEPS = 8
EXPECTED_GYM_ID = "Isaac-Fsm50-Residual-PhaseLocal-Direct-v0"

_HEX = frozenset("0123456789abcdef")
_STAGE_ROOT_KEYS = frozenset({"schema_version", "config_sha256", "payload"})
_STAGE_PAYLOAD_KEYS = frozenset(
    {
        "stage",
        "availability",
        "algorithm",
        "framework",
        "environment",
        "source_split",
        "seed_allowlist",
        "model",
        "normalization",
        "ppo",
        "trainer",
        "evaluation",
        "required_package_versions",
        "legacy_inputs",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "mode",
        "device",
        "gym_id",
        "num_envs",
        "entry_ids",
        "entry_roles",
        "episode_length_s",
        "physics_hz",
        "policy_hz",
        "exact_substeps_per_action",
        "actor_observation_dim",
        "action_dim",
    }
)
_SOURCE_SPLIT_KEYS = frozenset({"train", "held_out", "held_out_execution"})
_MODEL_KEYS = frozenset(
    {
        "actor_hidden",
        "critic_hidden",
        "initial_log_std",
        "minimum_log_std",
        "maximum_log_std",
        "zero_mean_head_required",
    }
)
_NORMALIZATION_KEYS = frozenset(
    {"observation", "value", "epsilon", "clip_threshold", "state_embedded_in_checkpoint"}
)
_PPO_KEYS = frozenset(
    {
        "rollouts",
        "learning_epochs",
        "mini_batches",
        "discount_factor",
        "gae_lambda",
        "learning_rate",
        "random_timesteps",
        "learning_starts",
        "grad_norm_clip",
        "ratio_clip",
        "value_clip",
        "entropy_loss_scale",
        "value_loss_scale",
        "kl_threshold",
        "time_limit_bootstrap",
        "mixed_precision",
    }
)
_TRAINER_KEYS = frozenset(
    {
        "timesteps",
        "headless",
        "disable_progressbar",
        "close_environment_at_exit",
        "write_interval",
        "checkpoint_interval",
    }
)
_EVALUATION_KEYS = frozenset(
    {
        "deterministic_action",
        "minimum_episode_count_per_arm",
        "maximum_hard_failure_rate",
        "maximum_phase_completion_drop_vs_zero",
        "require_exact_eight_cadence",
        "require_residual_transform_per_episode",
        "require_verified_physical_epoch",
        "require_one_batch",
        "require_n_plus_one",
    }
)
_LEGACY_KEYS = frozenset(
    {
        "allow_legacy_environment",
        "allow_legacy_checkpoint",
        "allow_checkpoint_resume",
        "allow_v010_execution",
    }
)
class ResidualPPOContractError(RuntimeError):
    """A Gate-E training/evaluation admission or immutable identity failed."""


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ResidualPPOContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ResidualPPOContractError(f"non-finite JSON number is forbidden: {value}")


def strict_json_load(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise ResidualPPOContractError(f"required JSON file is missing: {candidate}")
    try:
        value = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ResidualPPOContractError:
        raise
    except Exception as exc:
        raise ResidualPPOContractError(f"invalid JSON file {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResidualPPOContractError(f"JSON root must be an object: {candidate}")
    return value


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidualPPOContractError(f"value is not strict JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise ResidualPPOContractError(f"required file is missing: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ResidualPPOContractError(
            f"{label} keys mismatch: missing={sorted(set(expected) - actual)!r} "
            f"unexpected={sorted(actual - set(expected))!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualPPOContractError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResidualPPOContractError(f"{label} must be an array")
    return value


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value) or value != value.strip():
        raise ResidualPPOContractError(f"{label} must be exact text")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or text != text.lower() or any(ch not in _HEX for ch in text):
        raise ResidualPPOContractError(f"{label} must be a lowercase SHA-256")
    return text


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResidualPPOContractError(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ResidualPPOContractError(f"{label} must be finite")
    return float(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class ResidualPPOStageConfig:
    path: Path
    file_sha256: str
    config_sha256: str
    payload: Mapping[str, Any]

    @property
    def stage(self) -> str:
        return str(self.payload["stage"])

    @property
    def environment(self) -> Mapping[str, Any]:
        return _mapping(self.payload["environment"], "stage environment")

    @property
    def source_split(self) -> Mapping[str, Any]:
        return _mapping(self.payload["source_split"], "stage source_split")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.payload["seed_allowlist"])

    def to_identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "config_sha256": self.config_sha256,
            "stage": self.stage,
        }


def load_stage_config(
    path: str | Path = DEFAULT_R1_CONFIG_PATH,
    *,
    expected_stage: str = R1,
    require_canonical_path: bool = True,
) -> ResidualPPOStageConfig:
    candidate = Path(path).resolve(strict=True)
    if expected_stage == R2:
        raise ResidualPPOContractError(R2_UNAVAILABLE)
    if expected_stage != R1:
        raise ResidualPPOContractError(f"unsupported residual stage: {expected_stage}")
    canonical_sha_by_path = {
        DEFAULT_R1_CONFIG_PATH.resolve(strict=True): (
            EXPECTED_R1_CONFIG_CANONICAL_SHA256
        ),
        DEFAULT_R1_SMOKE_CONFIG_PATH.resolve(strict=True): (
            EXPECTED_R1_SMOKE_CONFIG_CANONICAL_SHA256
        ),
    }
    expected_canonical_sha = canonical_sha_by_path.get(candidate)
    if require_canonical_path and expected_canonical_sha is None:
        raise ResidualPPOContractError(
            "only the frozen full/smoke R1 PPO configs are admitted"
        )
    root = strict_json_load(candidate)
    _exact_keys(root, _STAGE_ROOT_KEYS, "stage config")
    if root.get("schema_version") != STAGE_CONFIG_SCHEMA:
        raise ResidualPPOContractError("stage config schema mismatch")
    payload = _mapping(root.get("payload"), "stage config payload")
    _exact_keys(payload, _STAGE_PAYLOAD_KEYS, "stage config payload")
    claimed = _sha(root.get("config_sha256"), "stage config_sha256")
    if canonical_json_sha256(payload) != claimed:
        raise ResidualPPOContractError("stage config canonical SHA mismatch")
    if expected_canonical_sha is not None:
        if claimed != expected_canonical_sha:
            raise ResidualPPOContractError(
                "stage config differs from its frozen path-bound R1 payload"
            )
    elif claimed not in frozenset(canonical_sha_by_path.values()):
        raise ResidualPPOContractError(
            "stage config is not a frozen full/smoke R1 payload"
        )
    _validate_r1_stage_payload(payload, expected_stage=expected_stage)
    return ResidualPPOStageConfig(
        path=candidate,
        file_sha256=sha256_file(candidate),
        config_sha256=claimed,
        payload=dict(payload),
    )


def _validate_r1_stage_payload(payload: Mapping[str, Any], *, expected_stage: str) -> None:
    if (
        payload.get("stage") != expected_stage
        or payload.get("availability") != "AVAILABLE_PHASE_LOCAL_ONLY"
        or payload.get("algorithm") != "PPO"
        or payload.get("framework") != "torch"
    ):
        raise ResidualPPOContractError("R1 stage identity is not exact")
    environment = _mapping(payload.get("environment"), "environment")
    _exact_keys(environment, _ENVIRONMENT_KEYS, "environment")
    entry_ids = _list(environment.get("entry_ids"), "environment.entry_ids")
    if (
        environment.get("mode") != "PHASE_LOCAL"
        or environment.get("device") != "cuda:0"
        or environment.get("gym_id") != EXPECTED_GYM_ID
        or environment.get("actor_observation_dim") != EXPECTED_OBSERVATION_DIM
        or environment.get("action_dim") != EXPECTED_ACTION_DIM
        or environment.get("physics_hz") != 120
        or environment.get("policy_hz") != 15
        or environment.get("exact_substeps_per_action") != EXPECTED_SUBSTEPS
        or len(entry_ids) != len(set(entry_ids))
        or not entry_ids
        or _finite(environment.get("episode_length_s"), "episode_length_s") <= 0
    ):
        raise ResidualPPOContractError("R1 environment/cadence identity is invalid")
    roles = _mapping(environment.get("entry_roles"), "environment.entry_roles")
    if set(roles) != set(entry_ids):
        raise ResidualPPOContractError("R1 entry role mapping does not cover the exact bank selection")
    train_entries = [entry_id for entry_id in entry_ids if roles[entry_id] == "TRAIN_NONZERO_AUTHORITY"]
    zero_entries = [entry_id for entry_id in entry_ids if roles[entry_id] == "EVAL_ZERO_AUTHORITY_ONLY"]
    expected_zero_entries = {
        f"{V003}:S5_PRE_RR_COM_SHIFT",
        f"{V003}:S7_PRE_RL_SUPPORT_SETUP",
        f"{V008}:S7_PRE_RL_SUPPORT_SETUP",
    }
    if (
        len(entry_ids) != 12
        or len(train_entries) != 9
        or set(zero_entries) != expected_zero_entries
        or environment.get("num_envs") != len(train_entries)
        or any(
            role not in {"TRAIN_NONZERO_AUTHORITY", "EVAL_ZERO_AUTHORITY_ONLY"}
            for role in roles.values()
        )
    ):
        raise ResidualPPOContractError("R1 entry roles do not preserve all 12 bank entries")
    source_split = _mapping(payload.get("source_split"), "source_split")
    _exact_keys(source_split, _SOURCE_SPLIT_KEYS, "source_split")
    if (
        tuple(source_split.get("train", ())) != TRAIN_SOURCES
        or tuple(source_split.get("held_out", ())) != HELD_OUT_SOURCES
        or source_split.get("held_out_execution")
        != "FORBIDDEN_FAILED_PROFILE_IDENTITY_ONLY"
    ):
        raise ResidualPPOContractError("source split is not the frozen v003/v008/v009 + v010 holdout")
    for entry_id in entry_ids:
        entry = _text(entry_id, "entry_id")
        source = entry.split(":", 1)[0]
        if source not in TRAIN_SOURCES or source == V010_FAILED:
            raise ResidualPPOContractError("R1 entry source is not authorized")
    seeds = _list(payload.get("seed_allowlist"), "seed_allowlist")
    if len(seeds) < 3 or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ResidualPPOContractError("R1 seed allowlist must contain at least three unique seeds")
    model = _mapping(payload.get("model"), "model")
    _exact_keys(model, _MODEL_KEYS, "model")
    if (
        model.get("actor_hidden") != [256, 256, 128]
        or model.get("critic_hidden") != [256, 256, 128]
        or model.get("initial_log_std") != -4.0
        or model.get("minimum_log_std") != -5.0
        or model.get("maximum_log_std") != -2.5
        or model.get("zero_mean_head_required") is not True
    ):
        raise ResidualPPOContractError("R1 actor/critic model identity drifted")
    normalization = _mapping(payload.get("normalization"), "normalization")
    _exact_keys(normalization, _NORMALIZATION_KEYS, "normalization")
    if (
        normalization.get("observation") != "RunningStandardScaler"
        or normalization.get("value") != "RunningStandardScaler"
        or normalization.get("state_embedded_in_checkpoint") is not True
        or _finite(normalization.get("epsilon"), "normalization.epsilon") != 1e-8
        or _finite(normalization.get("clip_threshold"), "normalization.clip_threshold") != 5.0
    ):
        raise ResidualPPOContractError("normalization identity drifted")
    ppo = _mapping(payload.get("ppo"), "ppo")
    _exact_keys(ppo, _PPO_KEYS, "ppo")
    for key in ("rollouts", "learning_epochs", "mini_batches"):
        _positive_int(ppo.get(key), f"ppo.{key}")
    if ppo.get("random_timesteps") != 0 or ppo.get("learning_starts") != 0:
        raise ResidualPPOContractError("R1 forbids random warmup and delayed learning")
    if ppo.get("time_limit_bootstrap") is not False or ppo.get("mixed_precision") is not False:
        raise ResidualPPOContractError("R1 PPO safety settings drifted")
    trainer = _mapping(payload.get("trainer"), "trainer")
    _exact_keys(trainer, _TRAINER_KEYS, "trainer")
    _positive_int(trainer.get("timesteps"), "trainer.timesteps")
    if trainer.get("headless") is not True or trainer.get("close_environment_at_exit") is not False:
        raise ResidualPPOContractError("R1 trainer lifecycle settings drifted")
    evaluation = _mapping(payload.get("evaluation"), "evaluation")
    _exact_keys(evaluation, _EVALUATION_KEYS, "evaluation")
    if (
        evaluation.get("deterministic_action") != DETERMINISTIC_ACTION
        or evaluation.get("maximum_hard_failure_rate") != 0.0
        or evaluation.get("require_exact_eight_cadence") is not True
        or evaluation.get("require_residual_transform_per_episode") is not True
        or evaluation.get("require_verified_physical_epoch") is not True
        or evaluation.get("require_one_batch") is not True
        or evaluation.get("require_n_plus_one") is not True
    ):
        raise ResidualPPOContractError("R1 deterministic evaluation contract drifted")
    packages = _mapping(payload.get("required_package_versions"), "required_package_versions")
    if set(packages) != {
        "skrl",
        "torch",
        "gymnasium",
        "numpy",
        "isaacsim",
        "isaaclab",
        "isaaclab_rl",
        "isaaclab_tasks",
        "isaaclab_assets",
    }:
        raise ResidualPPOContractError("required package set is not exact")
    for name, version in packages.items():
        _text(version, f"required package {name}")
    if packages.get("skrl") != "2.0.0":
        raise ResidualPPOContractError("Gate-E requires the installed SKRL 2.0.0 API")
    legacy = _mapping(payload.get("legacy_inputs"), "legacy_inputs")
    _exact_keys(legacy, _LEGACY_KEYS, "legacy_inputs")
    if any(value is not False for value in legacy.values()):
        raise ResidualPPOContractError("legacy environments/checkpoints or v010 execution are forbidden")


def current_package_identity(required_versions: Mapping[str, Any]) -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in sorted(required_versions):
        expected = _text(required_versions[name], f"required package version {name}")
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ResidualPPOContractError(f"required package is not installed: {name}") from exc
        if distribution.version != expected:
            raise ResidualPPOContractError(
                f"package version mismatch for {name}: expected {expected}, got {distribution.version}"
            )
        metadata_text = distribution.read_text("METADATA")
        record_text = distribution.read_text("RECORD")
        if not metadata_text or not record_text:
            raise ResidualPPOContractError(f"package metadata/RECORD is unavailable: {name}")
        packages[name] = {
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(metadata_text.encode("utf-8")).hexdigest(),
            "record_sha256": hashlib.sha256(record_text.encode("utf-8")).hexdigest(),
        }
    return {
        "packages": packages,
        "identity_sha256": canonical_json_sha256(packages),
    }


def validate_current_environment_lock() -> dict[str, Any]:
    try:
        from .fsm50_macro_runner import _validate_canonical_environment_lock

        value = dict(
            _validate_canonical_environment_lock(lock_path=DEFAULT_ENVIRONMENT_LOCK_PATH)
        )
    except Exception as exc:
        raise ResidualPPOContractError(
            f"canonical environment lock is not current: {type(exc).__name__}: {exc}"
        ) from exc
    if value.get("source_closure_complete") is not True:
        raise ResidualPPOContractError("canonical environment source closure is incomplete")
    lock_path = Path(_text(value.get("environment_lock_path"), "environment_lock_path")).resolve()
    if lock_path != DEFAULT_ENVIRONMENT_LOCK_PATH.resolve():
        raise ResidualPPOContractError("environment lock path is noncanonical")
    if _sha(value.get("environment_lock_sha256"), "environment_lock_sha256") != sha256_file(lock_path):
        raise ResidualPPOContractError("environment lock bytes changed after validation")
    return value


def build_current_authority_identity(
    config: ResidualPPOStageConfig,
    *,
    package_identity_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]] = current_package_identity,
) -> dict[str, Any]:
    """Bind every pure code/config/package input before AppLauncher exists."""

    if config.stage != R1:
        raise ResidualPPOContractError(R2_UNAVAILABLE)
    from .fsm50_phase_entry_bank import DEFAULT_CONFIG_PATH as BANK_CONFIG_PATH
    from .fsm50_phase_entry_bank import load_phase_entry_bank
    from .fsm50_residual_envelope import DEFAULT_CONFIG_PATH as ENVELOPE_CONFIG_PATH
    from .fsm50_residual_envelope import load_residual_envelope
    from .fsm50_residual_env import (
        CRITIC_STATE_DIM,
        EPISODE_METRICS_SCHEMA_SHA256,
        env_source_sha256,
    )
    from .fsm50_residual_observation import (
        ACTOR_OBSERVATION_DIM,
        ACTOR_OBSERVATION_SCHEMA_SHA256,
    )
    from .fsm50_residual_reward import REWARD_SCHEMA_VERSION
    from .fsm50_residual_scene import load_formal_scene_spec

    bank = load_phase_entry_bank(verify_artifacts=True)
    bank_entry_ids = {str(entry["entry_id"]) for entry in bank.entries}
    requested_entry_ids = set(config.environment["entry_ids"])
    if not requested_entry_ids.issubset(bank_entry_ids):
        raise ResidualPPOContractError("R1 config references an entry outside the sealed phase bank")
    if bank.bank_sha256 != EXPECTED_PHASE_BANK_SHA256:
        raise ResidualPPOContractError("R1 phase-entry bank identity drifted")
    envelope = load_residual_envelope(verify_evidence=True)
    if envelope.canonical_sha256 != EXPECTED_ENVELOPE_CANONICAL_SHA256 or not envelope.evidence_verified:
        raise ResidualPPOContractError("residual envelope is not the reviewed canonical authority")
    if (
        ACTOR_OBSERVATION_DIM != EXPECTED_OBSERVATION_DIM
        or ACTOR_OBSERVATION_SCHEMA_SHA256 != EXPECTED_OBSERVATION_SCHEMA_SHA256
    ):
        raise ResidualPPOContractError("deployable actor observation schema drifted")
    if (
        CRITIC_STATE_DIM != EXPECTED_OBSERVATION_DIM
        or EPISODE_METRICS_SCHEMA_SHA256 != EXPECTED_ENV_METRICS_SCHEMA_SHA256
        or env_source_sha256() != EXPECTED_ENV_SOURCE_SHA256
    ):
        raise ResidualPPOContractError("revised exact-eight R1 environment ABI drifted")
    scene = load_formal_scene_spec(verify_files=True)

    required_code = {
        "train_fsm50_residual_ppo.py": MODULE_ROOT / "train_fsm50_residual_ppo.py",
        "play_fsm50_residual_ppo.py": MODULE_ROOT / "play_fsm50_residual_ppo.py",
        "fsm50_residual_eval.py": Path(__file__).resolve(),
        "fsm50_residual_env.py": MODULE_ROOT / "fsm50_residual_env.py",
        "fsm50_residual_scene.py": MODULE_ROOT / "fsm50_residual_scene.py",
        "fsm50_phase_entry_bank.py": MODULE_ROOT / "fsm50_phase_entry_bank.py",
        "fsm50_residual_envelope.py": MODULE_ROOT / "fsm50_residual_envelope.py",
        "fsm50_residual_observation.py": MODULE_ROOT / "fsm50_residual_observation.py",
        "fsm50_residual_reward.py": MODULE_ROOT / "fsm50_residual_reward.py",
        "fsm50_residual_models.py": MODULE_ROOT / "fsm50_residual_models.py",
        "fsm50_direct_command_residual.py": MODULE_ROOT / "fsm50_direct_command_residual.py",
        "fsm50_residual_outer_cycle.py": MODULE_ROOT / "fsm50_residual_outer_cycle.py",
        "fsm50_residual_runner.py": MODULE_ROOT / "fsm50_residual_runner.py",
    }
    code_sha256 = {name: sha256_file(path) for name, path in sorted(required_code.items())}
    if (
        code_sha256["fsm50_residual_env.py"] != EXPECTED_ENV_SOURCE_SHA256
        or code_sha256["fsm50_residual_runner.py"] != EXPECTED_REVIEWED_R0_RUNNER_SHA256
    ):
        raise ResidualPPOContractError("frozen environment or reviewed-R0 validator source drifted")
    packages = dict(
        package_identity_builder(
            _mapping(config.payload["required_package_versions"], "required_package_versions")
        )
    )
    if _sha(packages.get("identity_sha256"), "package identity_sha256") != canonical_json_sha256(
        _mapping(packages.get("packages"), "package identity packages")
    ):
        raise ResidualPPOContractError("package identity canonical SHA mismatch")
    payload = {
        "schema_version": AUTHORITY_IDENTITY_SCHEMA,
        "stage_config": config.to_identity(),
        "scene_manifest_sha256": scene.manifest_sha256,
        "scene_manifest": scene.to_manifest(),
        "phase_bank_path": str(Path(BANK_CONFIG_PATH).resolve()),
        "phase_bank_file_sha256": sha256_file(BANK_CONFIG_PATH),
        "phase_bank_sha256": bank.bank_sha256,
        "envelope_config_path": str(Path(ENVELOPE_CONFIG_PATH).resolve()),
        "envelope_config_file_sha256": sha256_file(ENVELOPE_CONFIG_PATH),
        "envelope_canonical_sha256": envelope.canonical_sha256,
        "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
        "critic_state_dim": CRITIC_STATE_DIM,
        "episode_metrics_schema_sha256": EPISODE_METRICS_SCHEMA_SHA256,
        "action_dim": EXPECTED_ACTION_DIM,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "reward_code_sha256": code_sha256["fsm50_residual_reward.py"],
        "model_code_sha256": code_sha256["fsm50_residual_models.py"],
        "environment_code_sha256": code_sha256["fsm50_residual_env.py"],
        "direct_command_core_sha256": code_sha256["fsm50_direct_command_residual.py"],
        "reviewed_r0_validator_code_sha256": code_sha256["fsm50_residual_runner.py"],
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "profile_library_sha256": EXPECTED_PROFILE_LIBRARY_SHA256,
        "bundle_sha256_by_source": dict(ALLOWED_BUNDLE_SHA256),
        "code_sha256": code_sha256,
        "package_identity": packages,
        "residual_authority": {
            "sole_composer": "fsm50_direct_command_residual.compose_direct_command_residual",
            "stage": R1,
            "normalized_action_order": "canonical_servo8_then_wheel4",
            "legacy_checkpoint_allowed": False,
            "v010_execution_allowed": False,
        },
    }
    return {**payload, "identity_sha256": canonical_json_sha256(payload)}


@dataclass(frozen=True)
class ReviewedR0Manifest:
    path: Path
    file_sha256: str
    source_version: str
    request_id: str
    request_identity_sha256: str
    bundle_sha256: str
    environment: Mapping[str, Any]
    validation: Mapping[str, Any]

    def to_identity(self) -> dict[str, Any]:
        # The runner-owned validator is the sole interpretation of reviewed
        # ZERO-R0 evidence.  Persist its complete exact normalized projection,
        # not a locally reconstructed subset of mutable manifest flags.
        return dict(self.validation)


def validate_reviewed_r0_manifest(
    path: str | Path,
    *,
    current_environment: Mapping[str, Any],
    runner_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> ReviewedR0Manifest:
    """Delegate reviewed ZERO-R0 authority to the runner's public validator."""

    manifest_path = Path(path).resolve(strict=True)
    if manifest_path.name != "fsm50_residual_r0_manifest.json":
        raise ResidualPPOContractError("R0 manifest filename is noncanonical")
    if runner_validator is None:
        from .fsm50_residual_runner import (
            REVIEW_VALIDATION_KEYS,
            REVIEW_VALIDATION_SCHEMA,
            validate_reviewed_r0_manifest as runner_validator,
        )
    else:
        from .fsm50_residual_runner import REVIEW_VALIDATION_KEYS, REVIEW_VALIDATION_SCHEMA

    environment = dict(current_environment)

    def _already_validated_environment() -> Mapping[str, Any]:
        return dict(environment)

    try:
        validation = dict(
            runner_validator(
                manifest_path,
                environment_lock_validator=_already_validated_environment,
            )
        )
    except Exception as exc:
        raise ResidualPPOContractError(
            f"runner rejected reviewed ZERO-R0 manifest: {type(exc).__name__}: {exc}"
        ) from exc
    if set(validation) != set(REVIEW_VALIDATION_KEYS):
        raise ResidualPPOContractError("runner review validation normalized keys are not exact")
    if validation.get("schema_version") != REVIEW_VALIDATION_SCHEMA:
        raise ResidualPPOContractError("runner review validation schema mismatch")
    source = _text(validation.get("source_version"), "source_version")
    bundle_sha = _sha(validation.get("bundle_sha256"), "bundle_sha256")
    current_lock_sha = _sha(
        environment.get("environment_lock_sha256"), "current environment_lock_sha256"
    )
    if (
        validation.get("manifest_path") != str(manifest_path)
        or validation.get("manifest_sha256") != sha256_file(manifest_path)
        or validation.get("reviewed_success") is not True
        or validation.get("shutdown_verified") is not True
        or validation.get("semantic_projection_equal") is not True
        or validation.get("macro_fsm_complete") is not True
        or source not in TRAIN_SOURCES
        or bundle_sha != ALLOWED_BUNDLE_SHA256[source]
        or validation.get("graph_sha256") != EXPECTED_GRAPH_SHA256
        or validation.get("profile_library_sha256") != EXPECTED_PROFILE_LIBRARY_SHA256
        or validation.get("canonical_environment_lock_sha256") != current_lock_sha
    ):
        raise ResidualPPOContractError(
            "runner review validation is not current reviewed v003/v008/v009 authority"
        )
    for key, value in validation.items():
        if key.endswith("sha256"):
            _sha(value, f"runner review validation {key}")
    return ReviewedR0Manifest(
        path=manifest_path,
        file_sha256=sha256_file(manifest_path),
        source_version=source,
        request_id=_text(validation.get("request_id"), "request_id"),
        request_identity_sha256=_sha(
            validation.get("request_identity_sha256"), "request_identity_sha256"
        ),
        bundle_sha256=bundle_sha,
        environment=dict(environment),
        validation=validation,
    )


@dataclass(frozen=True)
class TrainingAdmission:
    config: ResidualPPOStageConfig
    seed: int
    environment: Mapping[str, Any]
    r0_manifests: tuple[ReviewedR0Manifest, ...]
    authority_identity: Mapping[str, Any]
    admission_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "stage_config": self.config.to_identity(),
            "seed": self.seed,
            "canonical_environment_lock": dict(self.environment),
            "reviewed_r0_manifests": [item.to_identity() for item in self.r0_manifests],
            "authority_identity": dict(self.authority_identity),
        }
        return {**payload, "admission_sha256": self.admission_sha256}


def build_training_admission(
    *,
    config: ResidualPPOStageConfig,
    seed: int,
    r0_manifest_paths: Sequence[str | Path],
    environment_validator: Callable[[], Mapping[str, Any]] = validate_current_environment_lock,
    authority_builder: Callable[[ResidualPPOStageConfig], Mapping[str, Any]] = build_current_authority_identity,
    reviewed_manifest_validator: Callable[..., ReviewedR0Manifest] = validate_reviewed_r0_manifest,
) -> TrainingAdmission:
    if config.stage != R1:
        raise ResidualPPOContractError(R2_UNAVAILABLE)
    if type(seed) is not int or seed not in config.seeds:
        raise ResidualPPOContractError("training seed is not in the frozen R1 allowlist")
    environment = dict(environment_validator())
    if environment.get("source_closure_complete") is not True:
        raise ResidualPPOContractError("current environment lock closure is incomplete")
    reviewed = tuple(
        reviewed_manifest_validator(path, current_environment=environment)
        for path in r0_manifest_paths
    )
    if tuple(sorted(item.source_version for item in reviewed)) != tuple(sorted(TRAIN_SOURCES)):
        raise ResidualPPOContractError("R1 requires one reviewed live ZERO-R0 manifest per train source")
    if len({item.request_id for item in reviewed}) != len(TRAIN_SOURCES):
        raise ResidualPPOContractError("reviewed ZERO-R0 request identities are not unique")
    if len(
        {
            item.validation["gate_e_code_identity_sha256"]
            for item in reviewed
        }
    ) != 1:
        raise ResidualPPOContractError("reviewed ZERO-R0 runs do not share one current Gate-E code identity")
    authority = dict(authority_builder(config))
    claimed = _sha(authority.get("identity_sha256"), "authority identity_sha256")
    if claimed != canonical_json_sha256(
        {key: value for key, value in authority.items() if key != "identity_sha256"}
    ):
        raise ResidualPPOContractError("authority identity SHA is not canonical")
    if (
        authority.get("graph_sha256") != EXPECTED_GRAPH_SHA256
        or authority.get("profile_library_sha256") != EXPECTED_PROFILE_LIBRARY_SHA256
        or authority.get("bundle_sha256_by_source") != ALLOWED_BUNDLE_SHA256
    ):
        raise ResidualPPOContractError("authority graph/library/bundle binding is stale")
    payload = {
        "stage_config": config.to_identity(),
        "seed": seed,
        "canonical_environment_lock": environment,
        "reviewed_r0_manifests": [item.to_identity() for item in reviewed],
        "authority_identity": authority,
    }
    return TrainingAdmission(
        config=config,
        seed=seed,
        environment=environment,
        r0_manifests=reviewed,
        authority_identity=authority,
        admission_sha256=canonical_json_sha256(payload),
    )


_CHECKPOINT_PAYLOAD_KEYS = frozenset(
    {
        "created_utc",
        "stage",
        "policy_kind",
        "ppo_training_performed",
        "legacy_checkpoint",
        "checkpoint_path",
        "checkpoint_sha256",
        "seed",
        "source_split",
        "stage_config",
        "canonical_environment_lock",
        "reviewed_r0_manifests",
        "authority_identity",
        "normalization",
        "checkpoint_module_state_sha256",
        "actor_observation_dim",
        "action_dim",
        "deterministic_evaluation_action",
        "residual_authority",
        "training_admission_sha256",
    }
)


def build_checkpoint_manifest(
    *,
    checkpoint_path: str | Path,
    admission: TrainingAdmission,
    checkpoint_module_state_sha256: Mapping[str, str],
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    modules = dict(checkpoint_module_state_sha256)
    required_modules = {
        "policy",
        "value",
        "observation_preprocessor",
        "value_preprocessor",
    }
    if set(modules) != required_modules:
        raise ResidualPPOContractError("checkpoint module-state SHA set is not exact")
    for name, digest in modules.items():
        _sha(digest, f"checkpoint module {name}")
    payload = {
        "created_utc": _utc_now(),
        "stage": R1,
        "policy_kind": POLICY_KIND,
        "ppo_training_performed": True,
        "legacy_checkpoint": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": admission.seed,
        "source_split": dict(admission.config.source_split),
        "stage_config": admission.config.to_identity(),
        "canonical_environment_lock": dict(admission.environment),
        "reviewed_r0_manifests": [item.to_identity() for item in admission.r0_manifests],
        "authority_identity": dict(admission.authority_identity),
        "normalization": dict(admission.config.payload["normalization"]),
        "checkpoint_module_state_sha256": modules,
        "actor_observation_dim": EXPECTED_OBSERVATION_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "deterministic_evaluation_action": DETERMINISTIC_ACTION,
        "residual_authority": dict(admission.authority_identity["residual_authority"]),
        "training_admission_sha256": admission.admission_sha256,
    }
    _exact_keys(payload, _CHECKPOINT_PAYLOAD_KEYS, "checkpoint manifest payload")
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "payload_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


@dataclass(frozen=True)
class CheckpointAdmission:
    manifest_path: Path
    manifest_file_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    payload: Mapping[str, Any]
    training_admission: TrainingAdmission


def load_checkpoint_manifest(
    path: str | Path,
    *,
    environment_validator: Callable[[], Mapping[str, Any]] = validate_current_environment_lock,
    authority_builder: Callable[[ResidualPPOStageConfig], Mapping[str, Any]] = build_current_authority_identity,
    reviewed_manifest_validator: Callable[..., ReviewedR0Manifest] = validate_reviewed_r0_manifest,
) -> CheckpointAdmission:
    manifest_path = Path(path).resolve(strict=True)
    if manifest_path.name != "fsm50_residual_ppo_checkpoint_manifest.json":
        raise ResidualPPOContractError("checkpoint manifest filename is noncanonical")
    root = strict_json_load(manifest_path)
    _exact_keys(root, frozenset({"schema_version", "payload_sha256", "payload"}), "checkpoint manifest")
    if root.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
        raise ResidualPPOContractError("checkpoint manifest schema mismatch")
    payload = _mapping(root.get("payload"), "checkpoint manifest payload")
    _exact_keys(payload, _CHECKPOINT_PAYLOAD_KEYS, "checkpoint manifest payload")
    if canonical_json_sha256(payload) != _sha(root.get("payload_sha256"), "checkpoint payload_sha256"):
        raise ResidualPPOContractError("checkpoint manifest canonical SHA mismatch")
    if (
        payload.get("stage") != R1
        or payload.get("policy_kind") != POLICY_KIND
        or payload.get("ppo_training_performed") is not True
        or payload.get("legacy_checkpoint") is not False
        or payload.get("deterministic_evaluation_action") != DETERMINISTIC_ACTION
        or payload.get("actor_observation_dim") != EXPECTED_OBSERVATION_DIM
        or payload.get("action_dim") != EXPECTED_ACTION_DIM
    ):
        raise ResidualPPOContractError("checkpoint is not a current nonlegacy R1 PPO policy")
    config_identity = _mapping(payload.get("stage_config"), "checkpoint stage_config")
    config_path = Path(_text(config_identity.get("path"), "stage config path")).resolve()
    config = load_stage_config(config_path, expected_stage=R1)
    if config.to_identity() != dict(config_identity):
        raise ResidualPPOContractError("checkpoint stage config is stale")
    admission = build_training_admission(
        config=config,
        seed=payload.get("seed"),
        r0_manifest_paths=[
            _mapping(item, "reviewed R0 identity").get("manifest_path")
            for item in _list(payload.get("reviewed_r0_manifests"), "reviewed R0 manifests")
        ],
        environment_validator=environment_validator,
        authority_builder=authority_builder,
        reviewed_manifest_validator=reviewed_manifest_validator,
    )
    if (
        admission.admission_sha256 != payload.get("training_admission_sha256")
        or admission.config.source_split != payload.get("source_split")
        or admission.config.payload["normalization"] != payload.get("normalization")
        or admission.environment != payload.get("canonical_environment_lock")
        or admission.authority_identity != payload.get("authority_identity")
        or [item.to_identity() for item in admission.r0_manifests]
        != payload.get("reviewed_r0_manifests")
    ):
        raise ResidualPPOContractError("checkpoint admission identity is stale")
    checkpoint_path = Path(_text(payload.get("checkpoint_path"), "checkpoint_path")).resolve()
    try:
        checkpoint_path.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ResidualPPOContractError("checkpoint escapes its immutable manifest directory") from exc
    checkpoint_sha = _sha(payload.get("checkpoint_sha256"), "checkpoint_sha256")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ResidualPPOContractError("checkpoint bytes are stale or tampered")
    modules = _mapping(payload.get("checkpoint_module_state_sha256"), "checkpoint module states")
    if set(modules) != {
        "policy",
        "value",
        "observation_preprocessor",
        "value_preprocessor",
    }:
        raise ResidualPPOContractError("checkpoint module-state SHA set is invalid")
    for name, digest in modules.items():
        _sha(digest, f"checkpoint module {name}")
    return CheckpointAdmission(
        manifest_path=manifest_path,
        manifest_file_sha256=sha256_file(manifest_path),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        payload=dict(payload),
        training_admission=admission,
    )


_EPISODE_KEYS = frozenset(
    {
        "arm",
        "episode_index",
        "entry_id",
        "source_version",
        "phase_state",
        "episode_return",
        "episode_length",
        "phase_completed",
        "task_success",
        "hard_failure",
        "hard_failure_reason",
        "body_crossed_front_face",
        "required_leg_lift_completed",
        "final_recoverable",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "max_root_angular_speed_rad_s",
        "max_servo_error_deg",
        "max_normalized_residual_l2",
        "max_normalized_residual_slew_l2",
        "residual_transform_count",
        "physical_batch_count",
        "physical_command_epoch",
        "last_verified_physical_command_epoch",
        "n_plus_one_verified_count",
        "exact_eight_cadence_verified",
        "one_batch_verified",
        "n_plus_one_verified",
    }
)


def validate_episode_metrics(value: Mapping[str, Any], *, expected_arm: str) -> dict[str, Any]:
    row = dict(value)
    _exact_keys(row, _EPISODE_KEYS, "episode metrics")
    if row.get("arm") != expected_arm or expected_arm not in {"nominal", "zero", "ppo"}:
        raise ResidualPPOContractError("episode evaluation arm mismatch")
    if type(row.get("episode_index")) is not int or row["episode_index"] < 0:
        raise ResidualPPOContractError("episode index is invalid")
    entry_id = _text(row.get("entry_id"), "episode entry_id")
    source = _text(row.get("source_version"), "episode source_version")
    if source not in TRAIN_SOURCES or not entry_id.startswith(source + ":"):
        raise ResidualPPOContractError("episode source/entry identity is unauthorized")
    _text(row.get("phase_state"), "episode phase_state")
    _finite(row.get("episode_return"), "episode return")
    _positive_int(row.get("episode_length"), "episode length")
    for key in (
        "phase_completed",
        "task_success",
        "hard_failure",
        "body_crossed_front_face",
        "required_leg_lift_completed",
        "final_recoverable",
        "exact_eight_cadence_verified",
        "one_batch_verified",
        "n_plus_one_verified",
    ):
        if type(row.get(key)) is not bool:
            raise ResidualPPOContractError(f"episode {key} must be bool")
    if type(row.get("hard_failure_reason")) is not str:
        raise ResidualPPOContractError("hard_failure_reason must be text")
    if bool(row["hard_failure"]) != bool(row["hard_failure_reason"]):
        raise ResidualPPOContractError("hard failure flag/reason mismatch")
    for key in (
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "max_root_angular_speed_rad_s",
        "max_servo_error_deg",
        "max_normalized_residual_l2",
        "max_normalized_residual_slew_l2",
    ):
        if _finite(row.get(key), f"episode {key}") < 0.0:
            raise ResidualPPOContractError(f"episode {key} must be nonnegative")
    for key in (
        "residual_transform_count",
        "physical_batch_count",
        "physical_command_epoch",
        "last_verified_physical_command_epoch",
        "n_plus_one_verified_count",
    ):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ResidualPPOContractError(f"episode {key} must be a nonnegative int")
    if (
        row["residual_transform_count"] <= 0
        or row["physical_batch_count"] != row["physical_command_epoch"]
        or row["last_verified_physical_command_epoch"] != row["physical_command_epoch"]
        or row["n_plus_one_verified_count"] != row["physical_batch_count"]
    ):
        raise ResidualPPOContractError("episode residual-transform/physical-epoch ledger is incomplete")
    return row


def build_arm_result(
    *,
    arm: str,
    episodes: Sequence[Mapping[str, Any]],
    stage_config_sha256: str,
    seed: int,
    checkpoint_manifest_sha256: str = "",
) -> dict[str, Any]:
    rows = [validate_episode_metrics(row, expected_arm=arm) for row in episodes]
    if not rows:
        raise ResidualPPOContractError("evaluation arm has no completed episodes")
    if type(seed) is not int or seed < 0:
        raise ResidualPPOContractError("evaluation arm seed must be a nonnegative int")
    if [row["episode_index"] for row in rows] != list(range(len(rows))):
        raise ResidualPPOContractError("episode indices are not contiguous")
    if checkpoint_manifest_sha256:
        _sha(checkpoint_manifest_sha256, "checkpoint manifest SHA")
    aggregate = {
        "episode_count": len(rows),
        "mean_return": sum(float(row["episode_return"]) for row in rows) / len(rows),
        "phase_completion_rate": sum(bool(row["phase_completed"]) for row in rows) / len(rows),
        "task_success_rate": sum(bool(row["task_success"]) for row in rows) / len(rows),
        "hard_failure_rate": sum(bool(row["hard_failure"]) for row in rows) / len(rows),
        "maximum_abs_roll_rad": max(float(row["max_abs_roll_rad"]) for row in rows),
        "maximum_abs_pitch_rad": max(float(row["max_abs_pitch_rad"]) for row in rows),
        "maximum_root_angular_speed_rad_s": max(
            float(row["max_root_angular_speed_rad_s"]) for row in rows
        ),
        "maximum_servo_error_deg": max(float(row["max_servo_error_deg"]) for row in rows),
        "maximum_normalized_residual_l2": max(
            float(row["max_normalized_residual_l2"]) for row in rows
        ),
        "maximum_normalized_residual_slew_l2": max(
            float(row["max_normalized_residual_slew_l2"]) for row in rows
        ),
        "total_residual_transform_count": sum(
            int(row["residual_transform_count"]) for row in rows
        ),
        "minimum_residual_transform_count": min(
            int(row["residual_transform_count"]) for row in rows
        ),
        "all_physical_epoch_verified": all(
            row["physical_batch_count"] == row["physical_command_epoch"]
            == row["last_verified_physical_command_epoch"]
            == row["n_plus_one_verified_count"]
            for row in rows
        ),
        "all_exact_eight_cadence_verified": all(
            row["exact_eight_cadence_verified"] for row in rows
        ),
        "all_one_batch_verified": all(row["one_batch_verified"] for row in rows),
        "all_n_plus_one_verified": all(row["n_plus_one_verified"] for row in rows),
    }
    payload = {
        "arm": arm,
        "stage_config_sha256": _sha(stage_config_sha256, "stage_config_sha256"),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "seed": seed,
        "episodes": rows,
        "aggregate": aggregate,
    }
    return {
        "schema_version": ARM_RESULT_SCHEMA,
        "payload_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


def compare_evaluation_arms(
    *,
    nominal: Mapping[str, Any],
    zero: Mapping[str, Any],
    ppo: Mapping[str, Any],
    checkpoint: CheckpointAdmission,
) -> dict[str, Any]:
    arms: dict[str, Mapping[str, Any]] = {}
    for name, value in (("nominal", nominal), ("zero", zero), ("ppo", ppo)):
        root = _mapping(value, f"{name} arm result")
        if root.get("schema_version") != ARM_RESULT_SCHEMA:
            raise ResidualPPOContractError(f"{name} arm result schema mismatch")
        payload = _mapping(root.get("payload"), f"{name} arm payload")
        if canonical_json_sha256(payload) != _sha(root.get("payload_sha256"), f"{name} payload SHA"):
            raise ResidualPPOContractError(f"{name} arm result was tampered")
        if payload.get("arm") != name:
            raise ResidualPPOContractError(f"{name} arm label mismatch")
        rebuilt = build_arm_result(
            arm=name,
            episodes=_list(payload.get("episodes"), f"{name} arm episodes"),
            stage_config_sha256=_text(
                payload.get("stage_config_sha256"), f"{name} stage config SHA"
            ),
            seed=payload.get("seed"),
            checkpoint_manifest_sha256=_text(
                payload.get("checkpoint_manifest_sha256"),
                f"{name} checkpoint manifest SHA",
                allow_empty=True,
            ),
        )
        if rebuilt["payload"] != dict(payload):
            raise ResidualPPOContractError(f"{name} arm aggregate/episode projection is not canonical")
        if payload.get("stage_config_sha256") != checkpoint.training_admission.config.config_sha256:
            raise ResidualPPOContractError(f"{name} arm stage config identity is stale")
        arms[name] = payload
    baseline_entries = [row["entry_id"] for row in arms["nominal"]["episodes"]]
    for name in ("zero", "ppo"):
        if (
            arms[name]["seed"] != arms["nominal"]["seed"]
            or [row["entry_id"] for row in arms[name]["episodes"]] != baseline_entries
            or len(arms[name]["episodes"]) != len(baseline_entries)
        ):
            raise ResidualPPOContractError("evaluation arms do not share the exact seed/episode split")
    if arms["nominal"]["seed"] not in checkpoint.training_admission.config.seeds:
        raise ResidualPPOContractError("evaluation arm seed is outside the frozen R1 allowlist")
    if arms["ppo"].get("checkpoint_manifest_sha256") != checkpoint.manifest_file_sha256:
        raise ResidualPPOContractError("PPO arm is not bound to the admitted checkpoint manifest")
    if (
        arms["nominal"].get("checkpoint_manifest_sha256")
        or arms["zero"].get("checkpoint_manifest_sha256")
    ):
        raise ResidualPPOContractError("reference arms must not claim a PPO checkpoint")
    evaluation_cfg = checkpoint.training_admission.config.payload["evaluation"]
    aggregate = {name: dict(arms[name]["aggregate"]) for name in arms}
    ppo_aggregate = aggregate["ppo"]
    zero_aggregate = aggregate["zero"]
    reasons: list[str] = []
    for name, arm_aggregate in aggregate.items():
        if arm_aggregate["episode_count"] < evaluation_cfg["minimum_episode_count_per_arm"]:
            reasons.append(f"insufficient {name} episode count")
        if arm_aggregate["hard_failure_rate"] > evaluation_cfg["maximum_hard_failure_rate"]:
            reasons.append(f"{name} hard-failure rate exceeds zero tolerance")
        for key in (
            "all_exact_eight_cadence_verified",
            "all_one_batch_verified",
            "all_n_plus_one_verified",
            "all_physical_epoch_verified",
        ):
            if arm_aggregate[key] is not True:
                reasons.append(f"{name} safety metric failed: {key}")
    if (
        ppo_aggregate["phase_completion_rate"]
        < zero_aggregate["phase_completion_rate"]
        - evaluation_cfg["maximum_phase_completion_drop_vs_zero"]
    ):
        reasons.append("PPO phase completion regressed versus ZERO")
    nominal_zero_equal = all(
        {key: value for key, value in nominal_row.items() if key != "arm"}
        == {key: value for key, value in zero_row.items() if key != "arm"}
        for nominal_row, zero_row in zip(
            arms["nominal"]["episodes"], arms["zero"]["episodes"]
        )
    )
    if not nominal_zero_equal:
        reasons.append("nominal and ZERO reference arms are not semantically deterministic")
    payload = {
        "created_utc": _utc_now(),
        "checkpoint_manifest_path": str(checkpoint.manifest_path),
        "checkpoint_manifest_sha256": checkpoint.manifest_file_sha256,
        "deterministic_evaluation_action": DETERMINISTIC_ACTION,
        "nominal_arm_semantics": "ZERO_ACTION_NOMINAL_REFERENCE",
        "nominal_zero_semantic_equal": nominal_zero_equal,
        "arms": {name: dict(value) for name, value in arms.items()},
        "aggregate": aggregate,
        "deltas_vs_zero": {
            "ppo_mean_return": ppo_aggregate["mean_return"] - zero_aggregate["mean_return"],
            "ppo_phase_completion_rate": (
                ppo_aggregate["phase_completion_rate"]
                - zero_aggregate["phase_completion_rate"]
            ),
            "ppo_hard_failure_rate": (
                ppo_aggregate["hard_failure_rate"] - zero_aggregate["hard_failure_rate"]
            ),
        },
        "evaluation_contract_passed": not reasons,
        "failure_reasons": reasons,
        "physical_promotion_claimed": False,
    }
    return {
        "schema_version": EVALUATION_MANIFEST_SCHEMA,
        "payload_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


def deterministic_mean_action(agent: Any, observations: Any, states: Any) -> Any:
    """Return SKRL's policy mean, never the sampled Gaussian action."""

    output = agent.act(observations, states, timestep=0, timesteps=0)
    if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], Mapping):
        raise ResidualPPOContractError("SKRL agent.act returned an unsupported shape")
    sampled, metadata = output[0], output[1]
    if "mean_actions" not in metadata:
        raise ResidualPPOContractError("deterministic evaluation requires SKRL mean_actions")
    mean = metadata["mean_actions"]
    if mean is sampled:
        return mean
    return mean


__all__ = [
    "ARM_RESULT_SCHEMA",
    "AUTHORITY_IDENTITY_SCHEMA",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CheckpointAdmission",
    "DEFAULT_R1_CONFIG_PATH",
    "DEFAULT_R1_SMOKE_CONFIG_PATH",
    "DETERMINISTIC_ACTION",
    "EVALUATION_MANIFEST_SCHEMA",
    "EPISODE_METRICS_SCHEMA",
    "EXPECTED_ACTION_DIM",
    "EXPECTED_ENV_METRICS_SCHEMA_SHA256",
    "EXPECTED_ENV_SOURCE_SHA256",
    "EXPECTED_GYM_ID",
    "EXPECTED_OBSERVATION_DIM",
    "EXPECTED_PHASE_BANK_SHA256",
    "EXPECTED_R1_CONFIG_CANONICAL_SHA256",
    "EXPECTED_R1_SMOKE_CONFIG_CANONICAL_SHA256",
    "EXPECTED_REVIEWED_R0_RUNNER_SHA256",
    "EXPECTED_SUBSTEPS",
    "R1",
    "R2",
    "R2_UNAVAILABLE",
    "ResidualPPOContractError",
    "ResidualPPOStageConfig",
    "ReviewedR0Manifest",
    "TrainingAdmission",
    "atomic_write_json",
    "build_arm_result",
    "build_checkpoint_manifest",
    "build_current_authority_identity",
    "build_training_admission",
    "canonical_json_sha256",
    "compare_evaluation_arms",
    "current_package_identity",
    "deterministic_mean_action",
    "load_checkpoint_manifest",
    "load_stage_config",
    "sha256_file",
    "strict_json_load",
    "validate_current_environment_lock",
    "validate_episode_metrics",
    "validate_reviewed_r0_manifest",
]
