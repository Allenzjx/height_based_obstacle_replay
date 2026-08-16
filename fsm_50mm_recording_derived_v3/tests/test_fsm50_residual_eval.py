from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3 import fsm50_residual_eval as contract
from fsm_50mm_recording_derived_v3.fsm50_residual_runner import (
    REVIEW_VALIDATION_KEYS,
    REVIEW_VALIDATION_SCHEMA,
)


SHA = "a" * 64


def _environment() -> dict[str, object]:
    return {
        "source_closure_complete": True,
        "environment_lock_sha256": "e" * 64,
    }


def _authority() -> dict[str, object]:
    payload = {
        "graph_sha256": contract.EXPECTED_GRAPH_SHA256,
        "profile_library_sha256": contract.EXPECTED_PROFILE_LIBRARY_SHA256,
        "bundle_sha256_by_source": dict(contract.ALLOWED_BUNDLE_SHA256),
        "residual_authority": {
            "sole_composer": "fsm50_direct_command_residual.compose_direct_command_residual",
            "stage": contract.R1,
        },
    }
    return {**payload, "identity_sha256": contract.canonical_json_sha256(payload)}


def _review_validation(path: Path, source: str) -> dict[str, object]:
    result = {key: SHA for key in REVIEW_VALIDATION_KEYS}
    result.update(
        {
            "schema_version": REVIEW_VALIDATION_SCHEMA,
            "manifest_path": str(path.resolve()),
            "manifest_sha256": contract.sha256_file(path),
            "source_version": source,
            "request_id": f"request-{source}",
            "request_identity_sha256": "1" * 64,
            "base_request_id": f"base-{source}",
            "bundle_sha256": contract.ALLOWED_BUNDLE_SHA256[source],
            "graph_sha256": contract.EXPECTED_GRAPH_SHA256,
            "profile_library_sha256": contract.EXPECTED_PROFILE_LIBRARY_SHA256,
            "canonical_environment_lock_sha256": "e" * 64,
            "manual_review_status": "ZERO_R0_REVIEWED_TASK_SUCCESS_POSTURE_INCOMPLETE",
            "manual_review_classification": "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
            "reviewed_utc": "2026-08-15T12:00:00Z",
            "shutdown_verified": True,
            "semantic_projection_equal": True,
            "macro_fsm_complete": True,
            "reviewed_success": True,
        }
    )
    assert set(result) == set(REVIEW_VALIDATION_KEYS)
    return result


def _reviewed(path: Path, source: str, environment: dict[str, object]) -> contract.ReviewedR0Manifest:
    validation = _review_validation(path, source)
    return contract.ReviewedR0Manifest(
        path=path.resolve(),
        file_sha256=contract.sha256_file(path),
        source_version=source,
        request_id=str(validation["request_id"]),
        request_identity_sha256=str(validation["request_identity_sha256"]),
        bundle_sha256=str(validation["bundle_sha256"]),
        environment=environment,
        validation=validation,
    )


def _manifest_files(tmp_path: Path) -> list[Path]:
    paths = []
    for source in contract.TRAIN_SOURCES:
        run = tmp_path / source
        run.mkdir(parents=True)
        manifest = run / "fsm50_residual_r0_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        paths.append(manifest)
    return paths


def test_stage_config_accounts_for_all_twelve_entries_and_roles() -> None:
    config = contract.load_stage_config()
    entries = list(config.environment["entry_ids"])
    roles = dict(config.environment["entry_roles"])
    assert len(entries) == 12
    assert set(roles) == set(entries)
    assert sum(role == "TRAIN_NONZERO_AUTHORITY" for role in roles.values()) == 9
    assert sum(role == "EVAL_ZERO_AUTHORITY_ONLY" for role in roles.values()) == 3
    assert config.environment["device"] == "cuda:0"
    assert any(entry.startswith(contract.V009) and ":S7_" in entry for entry in entries)
    assert config.config_sha256 == contract.canonical_json_sha256(config.payload)
    assert config.payload["evaluation"]["require_residual_transform_per_episode"] is True
    assert config.payload["evaluation"]["require_verified_physical_epoch"] is True
    assert config.payload["trainer"]["timesteps"] == 2_000_000
    assert config.file_sha256 == "a4c2fb2363f86af3d61e0ca7e5f4f5edd43f0bc871af73f7455f5e80f4f373e3"


def test_exact_bounded_smoke_config_is_allowlisted_and_path_bound(tmp_path: Path) -> None:
    full = contract.load_stage_config(contract.DEFAULT_R1_CONFIG_PATH)
    smoke = contract.load_stage_config(contract.DEFAULT_R1_SMOKE_CONFIG_PATH)
    assert smoke.path == contract.DEFAULT_R1_SMOKE_CONFIG_PATH.resolve()
    assert smoke.config_sha256 == contract.EXPECTED_R1_SMOKE_CONFIG_CANONICAL_SHA256
    assert smoke.config_sha256 == contract.canonical_json_sha256(smoke.payload)
    assert smoke.payload["trainer"]["timesteps"] == 8_544
    assert smoke.payload["trainer"]["timesteps"] % smoke.payload["ppo"]["rollouts"] == 0
    assert smoke.payload["trainer"]["timesteps"] * smoke.environment["num_envs"] == 76_896

    expected_smoke_payload = json.loads(json.dumps(full.payload))
    expected_smoke_payload["trainer"]["timesteps"] = 8_544
    assert smoke.payload == expected_smoke_payload

    copied = tmp_path / "copied-smoke.json"
    copied.write_bytes(contract.DEFAULT_R1_SMOKE_CONFIG_PATH.read_bytes())
    with pytest.raises(contract.ResidualPPOContractError, match="full/smoke"):
        contract.load_stage_config(copied)

    document = json.loads(contract.DEFAULT_R1_SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
    document["payload"]["trainer"]["timesteps"] = 8_512
    document["config_sha256"] = contract.canonical_json_sha256(document["payload"])
    unreviewed = tmp_path / "unreviewed-smoke.json"
    unreviewed.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(contract.ResidualPPOContractError, match="not a frozen"):
        contract.load_stage_config(unreviewed, require_canonical_path=False)


def test_stage_config_tamper_and_r2_fail_closed(tmp_path: Path) -> None:
    document = json.loads(contract.DEFAULT_R1_CONFIG_PATH.read_text(encoding="utf-8"))
    document["payload"]["environment"]["action_dim"] = 13
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(contract.ResidualPPOContractError, match="canonical SHA"):
        contract.load_stage_config(tampered, require_canonical_path=False)
    with pytest.raises(contract.ResidualPPOContractError, match=contract.R2_UNAVAILABLE):
        contract.load_stage_config(expected_stage=contract.R2)


def test_authority_identity_distinguishes_full_and_bounded_smoke_configs() -> None:
    def packages(required_versions):
        payload = {
            name: {"version": version}
            for name, version in sorted(required_versions.items())
        }
        return {
            "packages": payload,
            "identity_sha256": contract.canonical_json_sha256(payload),
        }

    full = contract.load_stage_config(contract.DEFAULT_R1_CONFIG_PATH)
    smoke = contract.load_stage_config(contract.DEFAULT_R1_SMOKE_CONFIG_PATH)
    full_authority = contract.build_current_authority_identity(
        full, package_identity_builder=packages
    )
    smoke_authority = contract.build_current_authority_identity(
        smoke, package_identity_builder=packages
    )
    assert full_authority["stage_config"] == full.to_identity()
    assert smoke_authority["stage_config"] == smoke.to_identity()
    assert full_authority["identity_sha256"] != smoke_authority["identity_sha256"]
    assert full_authority["code_sha256"] == smoke_authority["code_sha256"]


def test_review_admission_delegates_to_runner_public_validator(tmp_path: Path) -> None:
    manifest = tmp_path / "fsm50_residual_r0_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    environment = _environment()
    calls = []

    def runner_validator(path, *, environment_lock_validator):
        calls.append((Path(path).resolve(), dict(environment_lock_validator())))
        return _review_validation(manifest, contract.V003)

    reviewed = contract.validate_reviewed_r0_manifest(
        manifest,
        current_environment=environment,
        runner_validator=runner_validator,
    )
    assert calls == [(manifest.resolve(), environment)]
    assert reviewed.to_identity() == _review_validation(manifest, contract.V003)
    assert set(reviewed.to_identity()) == set(REVIEW_VALIDATION_KEYS)


def test_review_admission_rejects_runner_failure_or_nonexact_projection(tmp_path: Path) -> None:
    manifest = tmp_path / "fsm50_residual_r0_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    def rejected(*_args, **_kwargs):
        raise RuntimeError("pending review")

    with pytest.raises(contract.ResidualPPOContractError, match="runner rejected"):
        contract.validate_reviewed_r0_manifest(
            manifest,
            current_environment=_environment(),
            runner_validator=rejected,
        )

    def nonexact(*_args, **_kwargs):
        value = _review_validation(manifest, contract.V003)
        value["untrusted_extra"] = True
        return value

    with pytest.raises(contract.ResidualPPOContractError, match="normalized keys"):
        contract.validate_reviewed_r0_manifest(
            manifest,
            current_environment=_environment(),
            runner_validator=nonexact,
        )


def test_training_admission_requires_one_unique_review_per_train_source(tmp_path: Path) -> None:
    config = contract.load_stage_config()
    environment = _environment()
    paths = _manifest_files(tmp_path)
    by_path = {path.resolve(): source for path, source in zip(paths, contract.TRAIN_SOURCES)}

    def review(path, *, current_environment):
        source = by_path[Path(path).resolve()]
        return _reviewed(Path(path), source, dict(current_environment))

    admission = contract.build_training_admission(
        config=config,
        seed=config.seeds[0],
        r0_manifest_paths=paths,
        environment_validator=lambda: environment,
        authority_builder=lambda _config: _authority(),
        reviewed_manifest_validator=review,
    )
    assert [item.source_version for item in admission.r0_manifests] == list(contract.TRAIN_SOURCES)
    assert all(set(item.to_identity()) == set(REVIEW_VALIDATION_KEYS) for item in admission.r0_manifests)

    with pytest.raises(contract.ResidualPPOContractError, match="one reviewed"):
        contract.build_training_admission(
            config=config,
            seed=config.seeds[0],
            r0_manifest_paths=[paths[0], paths[0], paths[1]],
            environment_validator=lambda: environment,
            authority_builder=lambda _config: _authority(),
            reviewed_manifest_validator=review,
        )


@pytest.mark.parametrize(
    "config_path",
    [contract.DEFAULT_R1_CONFIG_PATH, contract.DEFAULT_R1_SMOKE_CONFIG_PATH],
    ids=["full", "bounded-smoke"],
)
def test_checkpoint_manifest_rejects_tampered_checkpoint_bytes(
    tmp_path: Path, config_path: Path
) -> None:
    config = contract.load_stage_config(config_path)
    environment = _environment()
    r0_paths = _manifest_files(tmp_path / "r0")
    by_path = {path.resolve(): source for path, source in zip(r0_paths, contract.TRAIN_SOURCES)}

    def review(path, *, current_environment):
        return _reviewed(Path(path), by_path[Path(path).resolve()], dict(current_environment))

    admission = contract.build_training_admission(
        config=config,
        seed=config.seeds[0],
        r0_manifest_paths=r0_paths,
        environment_validator=lambda: environment,
        authority_builder=lambda _config: _authority(),
        reviewed_manifest_validator=review,
    )
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / "fsm50_residual_ppo_checkpoint.pt"
    checkpoint.write_bytes(b"sealed-checkpoint")
    manifest = contract.build_checkpoint_manifest(
        checkpoint_path=checkpoint,
        admission=admission,
        checkpoint_module_state_sha256={
            "policy": "1" * 64,
            "value": "2" * 64,
            "observation_preprocessor": "3" * 64,
            "value_preprocessor": "4" * 64,
        },
    )
    manifest_path = output / "fsm50_residual_ppo_checkpoint_manifest.json"
    contract.atomic_write_json(manifest_path, manifest)
    loaded = contract.load_checkpoint_manifest(
        manifest_path,
        environment_validator=lambda: environment,
        authority_builder=lambda _config: _authority(),
        reviewed_manifest_validator=review,
    )
    assert loaded.checkpoint_sha256 == contract.sha256_file(checkpoint)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(contract.ResidualPPOContractError, match="tampered"):
        contract.load_checkpoint_manifest(
            manifest_path,
            environment_validator=lambda: environment,
            authority_builder=lambda _config: _authority(),
            reviewed_manifest_validator=review,
        )


def test_deterministic_action_uses_mean_not_sample() -> None:
    sampled = object()
    mean = object()

    class Agent:
        def act(self, observations, states, *, timestep, timesteps):
            assert timestep == 0 and timesteps == 0
            return sampled, {"mean_actions": mean}

    assert contract.deterministic_mean_action(Agent(), object(), object()) is mean


def _episode(arm: str, index: int, entry: str, *, hard: bool = False) -> dict[str, object]:
    return {
        "arm": arm,
        "episode_index": index,
        "entry_id": entry,
        "source_version": entry.split(":", 1)[0],
        "phase_state": entry.split(":", 1)[1],
        "episode_return": 1.0,
        "episode_length": 8,
        "phase_completed": not hard,
        "task_success": not hard,
        "hard_failure": hard,
        "hard_failure_reason": "failure" if hard else "",
        "body_crossed_front_face": True,
        "required_leg_lift_completed": True,
        "final_recoverable": not hard,
        "max_abs_roll_rad": 0.1,
        "max_abs_pitch_rad": 0.1,
        "max_root_angular_speed_rad_s": 0.1,
        "max_servo_error_deg": 1.0,
        "max_normalized_residual_l2": 0.0,
        "max_normalized_residual_slew_l2": 0.0,
        "residual_transform_count": 1,
        "physical_batch_count": 1,
        "physical_command_epoch": 1,
        "last_verified_physical_command_epoch": 1,
        "n_plus_one_verified_count": 1,
        "exact_eight_cadence_verified": True,
        "one_batch_verified": True,
        "n_plus_one_verified": True,
    }


def test_arm_comparison_fails_closed_on_ppo_hard_failure(tmp_path: Path) -> None:
    # The comparison logic only needs the admitted config and checkpoint identity.
    config = contract.load_stage_config()
    checkpoint = type("Checkpoint", (), {})()
    checkpoint.manifest_path = tmp_path / "fsm50_residual_ppo_checkpoint_manifest.json"
    checkpoint.manifest_file_sha256 = "9" * 64
    checkpoint.training_admission = type("Admission", (), {"config": config})()
    entries = list(config.environment["entry_ids"])
    nominal_rows = [_episode("nominal", i, entry) for i, entry in enumerate(entries * 3)]
    zero_rows = [_episode("zero", i, entry) for i, entry in enumerate(entries * 3)]
    ppo_rows = [_episode("ppo", i, entry, hard=i == 0) for i, entry in enumerate(entries * 3)]
    nominal = contract.build_arm_result(
        arm="nominal", episodes=nominal_rows, stage_config_sha256=config.config_sha256, seed=config.seeds[0]
    )
    zero = contract.build_arm_result(
        arm="zero", episodes=zero_rows, stage_config_sha256=config.config_sha256, seed=config.seeds[0]
    )
    ppo = contract.build_arm_result(
        arm="ppo",
        episodes=ppo_rows,
        stage_config_sha256=config.config_sha256,
        seed=config.seeds[0],
        checkpoint_manifest_sha256=checkpoint.manifest_file_sha256,
    )
    result = contract.compare_evaluation_arms(
        nominal=nominal, zero=zero, ppo=ppo, checkpoint=checkpoint
    )
    assert result["payload"]["evaluation_contract_passed"] is False
    assert result["payload"]["physical_promotion_claimed"] is False
    assert any("hard-failure" in reason for reason in result["payload"]["failure_reasons"])
