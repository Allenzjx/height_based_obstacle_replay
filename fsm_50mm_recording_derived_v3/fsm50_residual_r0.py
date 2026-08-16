"""R0 shadow validation for the exact-zero residual command path.

This is deliberately labelled *shadow*: it binds the new residual composer to
the reviewed Gate-C/Gate-D source-action streams without claiming a new Isaac
execution.  A later live R0 must still run the ZERO policy through the worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from command_model import DEFAULT_MAX_WHEEL_SPEED_RAD_S

from .fsm50_direct_command_residual import (
    RESIDUAL_ACTION_DIM,
    ZERO_RESIDUAL_ACTION,
    ResidualPhaseContract,
    ResidualTransformInput,
    ZeroResidualPolicy,
    canonical_mapping_sha256,
    compose_direct_command_residual,
)


R0_SHADOW_SCHEMA = "fsm50.residual_r0_shadow_result.v1"
REVIEWED_SUCCESS = {
    "MACRO_FSM_TASK_SUCCESS",
    "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
}


@dataclass(frozen=True)
class ReviewedMacroRun:
    source_version: str
    bundle_sha256: str
    run_dir: Path
    expected_actions: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _identity_contract(row: Mapping[str, Any]) -> ResidualPhaseContract:
    return ResidualPhaseContract(
        source_version=str(row["profile_source_version"]),
        profile_strategy=str(row["profile_strategy"]),
        macro_state=str(row["macro_state"]),
        subphase=str(row["subphase"]),
        enabled_mask=(False,) * RESIDUAL_ACTION_DIM,
        residual_min_command_units=(0.0,) * RESIDUAL_ACTION_DIM,
        residual_max_command_units=(0.0,) * RESIDUAL_ACTION_DIM,
        maximum_rate_command_units_per_s=(0.0,) * RESIDUAL_ACTION_DIM,
    )


def validate_reviewed_macro_run_zero_identity(spec: ReviewedMacroRun) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = spec.run_dir.resolve()
    manifest_path = run_dir / "macro_fsm_runner_manifest.json"
    verdict_path = run_dir / "manual_macro_video_verdict.json"
    source_path = run_dir / "macro_source_action_consumption.jsonl"
    for path in (manifest_path, verdict_path, source_path):
        if not path.is_file():
            raise RuntimeError(f"missing R0 input artifact: {path}")

    manifest = _load_json(manifest_path)
    verdict = _load_json(verdict_path)
    if manifest.get("source_version") != spec.source_version:
        raise RuntimeError("manifest source_version mismatch")
    if manifest.get("bundle_sha256") != spec.bundle_sha256:
        raise RuntimeError("manifest bundle_sha256 mismatch")
    if manifest.get("controller_complete") is not True or manifest.get("shutdown_verified") is not True:
        raise RuntimeError("R0 input run is not a completed, cleanly closed Macro run")
    if manifest.get("manual_review_status") not in REVIEWED_SUCCESS:
        raise RuntimeError("R0 input run is not a reviewed task success")
    if verdict.get("review_complete") is not True or verdict.get("verdict", {}).get("task_completed") is not True:
        raise RuntimeError("R0 input manual verdict is not a completed task success")
    if _sha256_file(verdict_path) != manifest.get("manual_verdict_sha256"):
        raise RuntimeError("manual verdict SHA does not match its manifest binding")

    rows = _load_jsonl(source_path)
    if len(rows) != spec.expected_actions:
        raise RuntimeError(f"source action count {len(rows)} != {spec.expected_actions}")
    policy = ZeroResidualPolicy()
    evidence: list[dict[str, Any]] = []
    nominal_stream: list[dict[str, Any]] = []
    applied_stream: list[dict[str, Any]] = []
    previous_probe = tuple(0.25 if index < 8 else 0.05 for index in range(12))
    for index, row in enumerate(rows):
        if row.get("source_action_index") != index:
            raise RuntimeError(f"source action index discontinuity at {index}")
        if row.get("profile_source_version") != spec.source_version:
            raise RuntimeError(f"source action {index} source mismatch")
        contract = _identity_contract(row)
        nominal_servo = row.get("servo_targets_deg")
        nominal_wheel = row.get("wheel_targets_rad_s")
        transform_input = ResidualTransformInput(
            source_version=spec.source_version,
            profile_strategy=str(row["profile_strategy"]),
            macro_state=str(row["macro_state"]),
            subphase=str(row["subphase"]),
            nominal_servo_targets_deg=nominal_servo,
            nominal_wheel_targets_rad_s=nominal_wheel,
            normalized_action=policy.act({}),
            previous_applied_residual=previous_probe,
            decision_dt_s=1.0 / 15.0,
            maximum_wheel_speed_rad_s=DEFAULT_MAX_WHEEL_SPEED_RAD_S,
        )
        result = compose_direct_command_residual(transform_input, contract)
        if not result.zero_identity:
            raise RuntimeError(f"source action {index} did not take the exact-zero branch")
        if result.applied_residual != ZERO_RESIDUAL_ACTION:
            raise RuntimeError(f"source action {index} emitted a residual tail")
        if result.applied_servo_targets_deg != nominal_servo:
            raise RuntimeError(f"source action {index} changed a nominal servo target")
        if result.applied_wheel_targets_rad_s != nominal_wheel:
            raise RuntimeError(f"source action {index} changed a nominal wheel target")
        stream_row = {
            "source_action_index": index,
            "source_action_identity": row["command_provenance"]["source_action_identity"],
            "macro_state": row["macro_state"],
            "subphase": row["subphase"],
            "servo_targets_deg": nominal_servo,
            "wheel_targets_rad_s": nominal_wheel,
        }
        applied_row = dict(stream_row)
        applied_row["servo_targets_deg"] = result.applied_servo_targets_deg
        applied_row["wheel_targets_rad_s"] = result.applied_wheel_targets_rad_s
        nominal_stream.append(stream_row)
        applied_stream.append(applied_row)
        evidence.append(
            {
                "schema_version": "fsm50.residual_r0_shadow_action.v1",
                "source_action_index": index,
                "source_action_identity": stream_row["source_action_identity"],
                "macro_state": row["macro_state"],
                "subphase": row["subphase"],
                "zero_identity": True,
                "contract_sha256": contract.sha256,
                "transform_result_sha256": result.sha256,
                "nominal_targets_sha256": canonical_mapping_sha256(
                    {"servos": nominal_servo, "wheels": nominal_wheel}
                ),
                "applied_targets_sha256": canonical_mapping_sha256(
                    {
                        "servos": result.applied_servo_targets_deg,
                        "wheels": result.applied_wheel_targets_rad_s,
                    }
                ),
                "residual_dispatch_required": False,
            }
        )

    nominal_sha = canonical_mapping_sha256({"rows": nominal_stream})
    applied_sha = canonical_mapping_sha256({"rows": applied_stream})
    if nominal_sha != applied_sha or nominal_stream != applied_stream:
        raise RuntimeError("zero-residual applied command stream differs from nominal")
    result = {
        "source_version": spec.source_version,
        "bundle_sha256": spec.bundle_sha256,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manual_verdict_path": str(verdict_path),
        "manual_verdict_sha256": _sha256_file(verdict_path),
        "source_action_path": str(source_path),
        "source_action_sha256": _sha256_file(source_path),
        "source_action_count": len(rows),
        "zero_identity_count": len(rows),
        "residual_dispatch_count": 0,
        "nominal_command_stream_sha256": nominal_sha,
        "applied_command_stream_sha256": applied_sha,
        "reviewed_classification": manifest["manual_review_status"],
        "passed": True,
    }
    return result, evidence


def default_reviewed_runs(project_root: Path) -> tuple[ReviewedMacroRun, ...]:
    module = project_root / "fsm_50mm_recording_derived_v3"
    return (
        ReviewedMacroRun(
            "v003_20260805_224517_157723_manual",
            "5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78",
            module / "runs" / "v003_macro_fsm_completion_aware_coalesced_r4"
            / "v003_20260805_224517_157723_manual" / "baseline"
            / "20260815T105857_601132Z_baseline_00_5742cda6e438",
            112,
        ),
        ReviewedMacroRun(
            "v008_20260806_211408_578700_manual",
            "0a47cee05cbb41a3a09b23836386bddcf9ae1ab2aca5bbe451ee20a0cbaa8c41",
            module / "runs" / "cross_version_macro_fsm_completion_aware_coalesced_r4"
            / "v008_20260806_211408_578700_manual" / "trials"
            / "20260815T113449_254770Z_cross_version_00_0a47cee05cbb",
            119,
        ),
        ReviewedMacroRun(
            "v009_20260806_215232_433234_manual",
            "0e5686354eed7d89fb57d3aba722602eee7115b44749d531b6634dd0420ee919",
            module / "runs" / "cross_version_macro_fsm_completion_aware_coalesced_r4"
            / "v009_20260806_215232_433234_manual" / "trials"
            / "20260815T114309_706871Z_cross_version_00_0e5686354eed",
            132,
        ),
    )


def run_shadow_r0(project_root: Path, output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}_r0_shadow_zero"
    run_dir.mkdir(parents=False, exist_ok=False)
    summaries: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    for spec in default_reviewed_runs(project_root.resolve()):
        summary, evidence = validate_reviewed_macro_run_zero_identity(spec)
        summaries.append(summary)
        all_evidence.extend(evidence)
    ledger_path = run_dir / "r0_zero_identity_actions.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in all_evidence),
        encoding="utf-8",
    )
    result = {
        "schema_version": R0_SHADOW_SCHEMA,
        "classification": "R0_SHADOW_ZERO_IDENTITY_PASS",
        "live_isaac_execution": False,
        "notes": (
            "Offline SHA-bound shadow validation only; a live worker ZERO-policy run is still required "
            "before nonzero residual deployment or PPO promotion."
        ),
        "policy": ZeroResidualPolicy().to_mapping(),
        "policy_sha256": ZeroResidualPolicy().policy_sha256,
        "reviewed_runs": summaries,
        "source_action_count": sum(item["source_action_count"] for item in summaries),
        "zero_identity_count": sum(item["zero_identity_count"] for item in summaries),
        "residual_dispatch_count": 0,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256_file(ledger_path),
        "passed": True,
    }
    result_path = run_dir / "r0_zero_identity_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "fsm_residual_ppo" / "r0_shadow",
    )
    args = parser.parse_args(argv)
    result_path = run_shadow_r0(args.project_root, args.output_root)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "R0_SHADOW_SCHEMA",
    "ReviewedMacroRun",
    "default_reviewed_runs",
    "run_shadow_r0",
    "validate_reviewed_macro_run_zero_identity",
]
