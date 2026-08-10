from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3.fsm50_state_restore import (
    A0_STATE_ID,
    RestoreProvenanceError,
    TRUSTED_SIM_STATE_SCHEMA,
    VERIFIED_PREFIX_REPLAY_SCHEMA,
    sha256_file,
    validate_state_restore,
)


ENVIRONMENT_SHA256 = "e" * 64
TARGET_STATE = "A7_LIFT_FR"
STATE_ORDER = [A0_STATE_ID, "A1_COM_TO_RL", "A2_PRELOAD_FR", TARGET_STATE, "F5_SUCCESS"]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _complete_sim_state() -> dict[str, object]:
    return {
        "capture_source": "FakeIsaacAdapter",
        "pose_restore_eligible": True,
        "root_pose": [[0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]],
        "root_velocity": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "joint_names": ["front_left_hip", "front_left_wheel"],
        "joint_pos": [[0.2, 1.5]],
        "joint_vel": [[0.0, 0.1]],
        "command_state": {
            "servos": {"front_left_hip": 11.0},
            "wheels": {"front_left_wheel": 0.5},
        },
    }


def _source_artifact(
    tmp_path: Path,
    *,
    state_id: str = TARGET_STATE,
    environment_sha256: str = ENVIRONMENT_SHA256,
    marker: str = ".finalized",
    lifecycle_failed: bool = False,
    captured_state_ids: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    artifact_root = tmp_path / "artifact"
    run_dir = artifact_root / "source_run"
    run_dir.mkdir(parents=True)
    result_path = run_dir / "result.json"
    result = {
            "schema_version": "fsm50.state_test_result.v1",
            "state_id": state_id,
            "environment_fingerprint_sha256": environment_sha256,
            "run_dir": str(run_dir.resolve()),
            "artifact_root": str(artifact_root.resolve()),
            "artifact_valid": not lifecycle_failed,
            "lifecycle": {
                "finalized": not lifecycle_failed,
                "failed": lifecycle_failed,
            },
        }
    if captured_state_ids is not None:
        result.pop("state_id")
        result["captured_state_ids"] = captured_state_ids
    _write_json(result_path, result)
    (artifact_root / marker).write_text(marker.removeprefix(".") + "\n", encoding="utf-8")
    return artifact_root, run_dir, result_path


def _direct_evidence(
    run_dir: Path,
    result_path: Path,
    *,
    state_id: str = TARGET_STATE,
    environment_sha256: str = ENVIRONMENT_SHA256,
    result_sha256: str | None = None,
    sim_state: dict[str, object] | None = None,
) -> Path:
    path = run_dir / "sim_state_before.json"
    _write_json(
        path,
        {
            "schema_version": TRUSTED_SIM_STATE_SCHEMA,
            "state_id": state_id,
            "environment_fingerprint_sha256": environment_sha256,
            "source_run_directory": str(run_dir.resolve()),
            "source_result_path": "result.json",
            "source_result_sha256": result_sha256 or sha256_file(result_path),
            "sim_state_before": _complete_sim_state() if sim_state is None else sim_state,
        },
    )
    return path


def _prefix_manifest(
    run_dir: Path,
    result_path: Path,
    *,
    completed_prefix: list[str] | None = None,
    verified: bool = True,
    prefix_complete: bool = True,
) -> Path:
    path = run_dir / "verified_prefix_replay_manifest.json"
    _write_json(
        path,
        {
            "schema_version": VERIFIED_PREFIX_REPLAY_SCHEMA,
            "verified": verified,
            "prefix_complete": prefix_complete,
            "target_state_id": TARGET_STATE,
            "environment_fingerprint_sha256": ENVIRONMENT_SHA256,
            "source_run_directory": str(run_dir.resolve()),
            "source_result_path": "result.json",
            "source_result_sha256": sha256_file(result_path),
            "completed_prefix": (
                STATE_ORDER[: STATE_ORDER.index(TARGET_STATE)]
                if completed_prefix is None
                else completed_prefix
            ),
        },
    )
    return path


def test_a0_clean_reset_needs_no_restore_evidence() -> None:
    bundle = validate_state_restore(target_state_id=A0_STATE_ID)
    assert bundle == {
        "restore_provenance": {
            "method": "CLEAN_A0_RESET",
            "validated": True,
            "state_id": A0_STATE_ID,
        },
        "sim_state": None,
    }


def test_non_a0_without_exactly_one_evidence_mode_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RestoreProvenanceError, match="non-A0"):
        validate_state_restore(target_state_id=TARGET_STATE)
    with pytest.raises(RestoreProvenanceError, match="exactly one"):
        validate_state_restore(
            target_state_id=TARGET_STATE,
            environment_fingerprint_sha256=ENVIRONMENT_SHA256,
            sim_state_before_path=tmp_path / "state.json",
            prefix_replay_manifest_path=tmp_path / "prefix.json",
        )


def test_trusted_sim_state_returns_controller_provenance_and_normalized_state(
    tmp_path: Path,
) -> None:
    _artifact_root, run_dir, result_path = _source_artifact(tmp_path)
    evidence_path = _direct_evidence(run_dir, result_path)
    bundle = validate_state_restore(
        target_state_id=TARGET_STATE,
        environment_fingerprint_sha256=ENVIRONMENT_SHA256,
        sim_state_before_path=evidence_path,
        sim_state_before_sha256=sha256_file(evidence_path),
        expected_joint_names=["front_left_wheel", "front_left_hip"],
    )
    provenance = bundle["restore_provenance"]
    assert provenance["method"] == "TRUSTED_SIM_STATE_RESTORE"
    assert provenance["validated"] is True
    assert provenance["state_id"] == TARGET_STATE
    assert provenance["source_run_directory"] == str(run_dir.resolve())
    assert provenance["source_sha256"] == sha256_file(evidence_path)
    assert bundle["sim_state"]["root_pose"] == [
        [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    ]


def test_full_controller_result_can_authorize_one_of_many_captured_states(
    tmp_path: Path,
) -> None:
    _artifact_root, run_dir, result_path = _source_artifact(
        tmp_path,
        captured_state_ids=[A0_STATE_ID, "A1_COM_TO_RL", TARGET_STATE],
    )
    evidence_path = _direct_evidence(run_dir, result_path)
    bundle = validate_state_restore(
        target_state_id=TARGET_STATE,
        environment_fingerprint_sha256=ENVIRONMENT_SHA256,
        sim_state_before_path=evidence_path,
        sim_state_before_sha256=sha256_file(evidence_path),
    )
    assert bundle["restore_provenance"]["state_id"] == TARGET_STATE


def test_full_controller_result_rejects_uncaptured_target(tmp_path: Path) -> None:
    _artifact_root, run_dir, result_path = _source_artifact(
        tmp_path,
        captured_state_ids=[A0_STATE_ID, "A1_COM_TO_RL"],
    )
    evidence_path = _direct_evidence(run_dir, result_path)
    with pytest.raises(RestoreProvenanceError, match="captured_state_ids"):
        validate_state_restore(
            target_state_id=TARGET_STATE,
            environment_fingerprint_sha256=ENVIRONMENT_SHA256,
            sim_state_before_path=evidence_path,
            sim_state_before_sha256=sha256_file(evidence_path),
        )


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "failed",
        "failed_lifecycle",
        "missing_result",
        "evidence_hash",
        "source_result_hash",
        "state_id",
        "result_state_id",
        "environment",
        "result_environment",
        "missing_joint_velocity",
        "incomplete_command_state",
    ],
)
def test_trusted_sim_state_rejects_incomplete_or_untrusted_evidence(
    tmp_path: Path,
    case: str,
) -> None:
    marker = ".partial" if case == "partial" else ".failed" if case == "failed" else ".finalized"
    _artifact_root, run_dir, result_path = _source_artifact(
        tmp_path,
        state_id="WRONG_STATE" if case == "result_state_id" else TARGET_STATE,
        environment_sha256=(
            "c" * 64 if case == "result_environment" else ENVIRONMENT_SHA256
        ),
        marker=marker,
        lifecycle_failed=case == "failed_lifecycle",
    )
    sim_state = _complete_sim_state()
    if case == "missing_joint_velocity":
        sim_state.pop("joint_vel")
    if case == "incomplete_command_state":
        sim_state["command_state"] = {
            "servos": {"front_left_hip": 11.0},
            "wheels": {"unrelated_wheel": 0.5},
        }
    evidence_path = _direct_evidence(
        run_dir,
        result_path,
        state_id="WRONG_STATE" if case == "state_id" else TARGET_STATE,
        environment_sha256="d" * 64 if case == "environment" else ENVIRONMENT_SHA256,
        result_sha256="a" * 64 if case == "source_result_hash" else None,
        sim_state=sim_state,
    )
    if case == "missing_result":
        result_path.unlink()
    expected_evidence_sha256 = (
        "b" * 64 if case == "evidence_hash" else sha256_file(evidence_path)
    )
    with pytest.raises(RestoreProvenanceError):
        validate_state_restore(
            target_state_id=TARGET_STATE,
            environment_fingerprint_sha256=ENVIRONMENT_SHA256,
            sim_state_before_path=evidence_path,
            sim_state_before_sha256=expected_evidence_sha256,
        )


def test_verified_prefix_manifest_returns_no_sim_state(tmp_path: Path) -> None:
    _artifact_root, run_dir, result_path = _source_artifact(tmp_path)
    manifest_path = _prefix_manifest(run_dir, result_path)
    bundle = validate_state_restore(
        target_state_id=TARGET_STATE,
        environment_fingerprint_sha256=ENVIRONMENT_SHA256,
        prefix_replay_manifest_path=manifest_path,
        prefix_replay_manifest_sha256=sha256_file(manifest_path),
        state_order=STATE_ORDER,
    )
    assert bundle["sim_state"] is None
    provenance = bundle["restore_provenance"]
    assert provenance["method"] == "VERIFIED_PREFIX_REPLAY"
    assert provenance["validated"] is True
    assert provenance["state_id"] == TARGET_STATE
    assert provenance["completed_prefix"] == STATE_ORDER[:3]


@pytest.mark.parametrize("case", ["manifest_hash", "unverified", "incomplete_prefix"])
def test_prefix_manifest_hash_verification_and_exact_completed_prefix(
    tmp_path: Path,
    case: str,
) -> None:
    _artifact_root, run_dir, result_path = _source_artifact(tmp_path)
    manifest_path = _prefix_manifest(
        run_dir,
        result_path,
        completed_prefix=[A0_STATE_ID] if case == "incomplete_prefix" else None,
        verified=case != "unverified",
    )
    expected_manifest_sha256 = (
        "c" * 64 if case == "manifest_hash" else sha256_file(manifest_path)
    )
    with pytest.raises(RestoreProvenanceError):
        validate_state_restore(
            target_state_id=TARGET_STATE,
            environment_fingerprint_sha256=ENVIRONMENT_SHA256,
            prefix_replay_manifest_path=manifest_path,
            prefix_replay_manifest_sha256=expected_manifest_sha256,
            state_order=STATE_ORDER,
        )
