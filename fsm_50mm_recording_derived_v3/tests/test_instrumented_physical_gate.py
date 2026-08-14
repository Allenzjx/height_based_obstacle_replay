from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from fsm_50mm_recording_derived_v3.environment_ab_artifacts import ReplayArtifact
from fsm_50mm_recording_derived_v3 import instrumented_physical_gate as gate


SOURCE_VERSION = gate.EXPECTED_SOURCE_VERSION
ACCEPTED_SHA = "a" * 64
PLAN_SHA = "b" * 64
SOURCE_FILES_SHA = "c" * 64
SOURCE_HEAD = "d" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shutdown_payload(*, status: str = "FAST_EXIT_VERIFIED") -> dict:
    if status == "FAST_EXIT_VERIFIED":
        mode = "fast"
        state = "FAST_CLOSE_RETURNED"
        kwargs = {"wait_for_replicator": False, "skip_cleanup": True}
    elif status == "GRACEFUL_EXIT":
        mode = "graceful"
        state = "GRACEFUL_CLOSE_RETURNED"
        kwargs = {"wait_for_replicator": False, "skip_cleanup": False}
    else:
        mode = "graceful"
        state = "CLOSE_RETURNED"
        kwargs = {}
    return {
        "schema_version": "fsm50.shutdown_outcome.v1",
        "status": status,
        "preclose_observed": True,
        "preclose_verification": {"ok": True},
        "process_returned_normally": True,
        "intended_returncode": 0,
        "child_returncode": 0,
        "close_error": "",
        "shutdown_mode": mode,
        "handshake_state": state,
        "close_kwargs": kwargs,
        "runtime_version": "5.1.0",
    }


def _result(*, trial_id: int, runtime_id: str, nonce: str) -> dict:
    token = "e" * 64
    criteria = {
        name: {
            "name": name,
            "scope": gate.PHYSICAL_CRITERION_SCOPES[name],
            "availability": "AVAILABLE",
            "passed": True,
            "reason": "",
        }
        for name in sorted(gate.REQUIRED_PHYSICAL_CRITERIA)
    }
    physical = {
        "schema_version": "fsm50.physical_evidence.v2",
        "source_version": SOURCE_VERSION,
        "contact_mode": "instrumented",
        "sample_count": 2,
        "evidence_complete": True,
        "not_evaluable_reasons": [],
        "criteria": criteria,
        "criterion_records": list(criteria.values()),
        "strict_criteria": {name: True for name in criteria},
        "role_capture_verdict": {"status": "PASS", "passed": True},
        "full_physical_verdict": "PASS",
        "role_capture_verdict_reasons": [],
        "full_physical_verdict_reasons": [],
        "physical_success": True,
    }
    wheel_integral = {
        "schema_version": 1,
        "source_version": SOURCE_VERSION,
        "structural_errors": [],
        "target_integral_verdict": "PASS",
        "target_not_evaluable_reasons": [],
        "measured_tracking_verdict": "NOT_EVALUABLE",
        "overall_verdict": "NOT_EVALUABLE",
        "physical_success": False,
    }
    return {
        "schema_version": "fsm50.recording_replay_result.v1",
        "run_nonce": nonce,
        "source_version": SOURCE_VERSION,
        "trial_id": trial_id,
        "contact_mode": "instrumented",
        "environment_equivalence_role": "",
        "environment_equivalence_diagnostic": False,
        "environment_equivalence_diagnostic_complete": False,
        "diagnostic_role": "",
        "ordinary_ui_diagnostic": False,
        "ordinary_ui_diagnostic_complete": False,
        "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
        "gate1_eligible": True,
        "gate1_physical_qualification_eligible": True,
        "environment_equivalence_eligible": False,
        "physical_qualification_eligible": True,
        "fresh_process_clean_reset": True,
        "artifact_valid": True,
        "scheduler_complete": True,
        "scheduler_stop_reason": "complete",
        "classification": "FULL_SUCCESS",
        "motion_start_ready": True,
        "dispatch_complete": True,
        "wheel_integral_evidence": wheel_integral,
        "wheel_target_integral_verdict": "PASS",
        "wheel_target_integral_complete": True,
        "physical_success": True,
        "strict_full_success": True,
        "timed_out": False,
        "simulation_app_stopped": False,
        "lifecycle": {
            "finalized": True,
            "failed": False,
            "strict_success": True,
        },
        "respawn": {
            "adapter_runtime_instance_id": runtime_id,
            "ok": True,
            "respawned": False,
            "root_pose_written": False,
            "root_state_write_count": 0,
        },
        "motion_start_readiness": {
            "adapter_runtime_instance_id": runtime_id,
            "ready": True,
            "status": "PASS",
            "writes_robot_state": False,
            "root_state_write_count": 0,
        },
        "motion_start_pre_first_dispatch": {
            "adapter_runtime_instance_id": runtime_id,
            "ready": True,
            "status": "PASS",
            "writes_robot_state": False,
            "root_state_write_count": 0,
            "readiness_token_sha256": token,
            "final": {"adapter_runtime_instance_id": runtime_id},
            "token_payload": {
                "adapter_runtime_instance_id": runtime_id,
                "source_version": SOURCE_VERSION,
                "trial_id": trial_id,
                "root_state_write_count": 0,
            },
        },
        "dispatch_ledger": {
            "schema_version": "fsm50.source_dispatch_ledger.v1",
            "source_version": SOURCE_VERSION,
            "complete": True,
            "errors": [],
            "one_motion_batch_per_physics_tick": True,
            "motion_start_readiness_token": token,
        },
        "physical_evidence": physical,
    }


def _artifact(
    root: Path,
    *,
    trial_id: int,
    runtime_id: str,
    nonce: str,
    shutdown_status: str = "FAST_EXIT_VERIFIED",
) -> ReplayArtifact:
    batch_root = root / f"batch_{trial_id}"
    artifact_root = batch_root / SOURCE_VERSION / f"artifact_{trial_id}"
    run_dir = artifact_root / f"run_{trial_id}"
    run_dir.mkdir(parents=True)
    result = _result(trial_id=trial_id, runtime_id=runtime_id, nonce=nonce)
    result["run_dir"] = str(run_dir.resolve())
    result["artifact_root"] = str(artifact_root.resolve())
    _write_json(run_dir / "result.json", result)
    wheel_path = run_dir / gate.WHEEL_INTEGRAL_EVIDENCE_FILENAME
    _write_json(wheel_path, result["wheel_integral_evidence"])
    (run_dir / "checksums.sha256").write_text(
        f"{_sha(wheel_path)}  {gate.WHEEL_INTEGRAL_EVIDENCE_FILENAME}\n",
        encoding="utf-8",
    )
    _write_json(
        batch_root / "batch_request.json",
        {
            "trial_id": trial_id,
            "environment_equivalence_role": "",
            "diagnostic_role": "",
            "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
            "gate1_eligible": True,
            "gate1_physical_qualification_eligible": True,
            "environment_equivalence_eligible": False,
            "args": {
                "environment_equivalence_role": "",
                "diagnostic_role": "",
                "contact_mode": "instrumented",
            },
        },
    )
    _write_json(
        batch_root / "shutdown_outcome.json",
        _shutdown_payload(status=shutdown_status),
    )
    runtime = {
        "contact_mode": "instrumented",
        "environment_equivalence_role": "",
        "diagnostic_role": "",
        "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
        "gate1_eligible": True,
        "gate1_physical_qualification_eligible": True,
        "environment_equivalence_eligible": False,
        "physics_dt_s": 1.0 / 120.0,
        "render_interval": 8,
        "runtime": {"packages": {"isaacsim": "5.1.0"}},
        "scene_config": {
            "device": "cuda:0",
            "physics_dt": 1.0 / 120.0,
            "servo_stiffness": 600.0,
            "servo_damping": 60.0,
            "wheel_damping": 20.0,
            "telemetry_contact_sensors_enabled": True,
        },
        "environment_equivalence": {
            "schema_version": "fsm50.runtime_environment_equivalence.v1",
            "ok": True,
            "locked_source_hashes": {"ok": True},
        },
        "live_obstacle_geometry": {
            "height_m": 0.05,
            "front_face_x_m": 0.5213121737735307,
        },
        "motion_reference": {
            "servo_velocity_deg_s": 150.0,
            "wheel_velocity_rad_s": 0.5235987755982988,
        },
    }
    provenance = {
        "source_version": SOURCE_VERSION,
        "accepted_steps_sha256": ACCEPTED_SHA,
        "plan_sha256": PLAN_SHA,
        "contact_mode": "instrumented",
        "source_git_head": SOURCE_HEAD,
        "source_files_sha256": SOURCE_FILES_SHA,
        "physics_dt_s": 1.0 / 120.0,
        "render_interval": 8,
        "runtime_versions": runtime["runtime"],
        "batch_root": str(batch_root.resolve()),
        "batch_shutdown_closure": {
            "status": shutdown_status,
            "phase": "SHUTDOWN_COMPLETE",
            "closure_sha256": (str(trial_id) * 64)[:64],
            "live_checksum_entry_count": 20,
            "preclose_checksum_entry_count": 18,
            "preclose_evidence_files": {
                "result.json": {"sha256": "5" * 64, "size_bytes": 1}
            },
        },
        "source_closure": {
            "git_head": SOURCE_HEAD,
            "current_git_head": SOURCE_HEAD,
            "files_sha256": SOURCE_FILES_SHA,
            "source_freeze_pre_sha256": "6" * 64,
            "source_freeze_post_sha256": "7" * 64,
            "source_integrity_sha256": "8" * 64,
        },
        "telemetry_finalization": {"marker_sha256": "f" * 64},
        "viewport_video": {
            "manifest_sha256": ("1" * 63) + str(trial_id),
            "video_sha256": ("2" * 63) + str(trial_id),
            "frame_count": 10,
        },
        "artifact_hashes": {"fsm50_telemetry.jsonl": "3" * 64},
        "checksums_sha256": _sha(run_dir / "checksums.sha256"),
    }
    return ReplayArtifact(
        role=f"I{trial_id}",
        artifact_root=artifact_root.resolve(),
        run_dir=run_dir.resolve(),
        result=result,
        runtime_environment=runtime,
        visual_manifest={},
        fast_plan={},
        telemetry_rows=({"sample_index": 0}, {"sample_index": 1}),
        trajectory_metrics={},
        provenance=provenance,
    )


def _report_payload(artifacts: list[ReplayArtifact], *, b_index: int = 0) -> dict:
    b = artifacts[b_index]
    provenance = {
        key: b.provenance[key]
        for key in (
            "source_version",
            "accepted_steps_sha256",
            "plan_sha256",
            "contact_mode",
            "source_git_head",
            "source_files_sha256",
            "physics_dt_s",
            "render_interval",
            "runtime_versions",
        )
    }
    return {
        "schema_version": "fsm50.environment_equivalence_report.v1",
        "status": "PASS",
        "environment_equivalent": True,
        "static_fingerprint": {
            "schema_version": "fsm50.environment_static_fingerprint.v1",
            "robot_usd_sha256": "9" * 64,
        },
        "instrumentation_comparison": {"ok": True},
        "trajectory_comparison": {"ok": True},
        "runtime_readback": {
            "ok": True,
            "readback_complete": True,
            "runs": {
                "B": {
                    "artifact_root": str(b.artifact_root),
                    "run_dir": str(b.run_dir),
                    "provenance": provenance,
                }
            },
        },
        "extra": {"artifact_conversion": {"ok": True}},
    }


@pytest.fixture
def evidence(tmp_path: Path):
    artifacts = [
        _artifact(
            tmp_path,
            trial_id=index,
            runtime_id=f"runtime-instance-{index}",
            nonce=f"nonce-{index}",
            shutdown_status="GRACEFUL_EXIT" if index == 2 else "FAST_EXIT_VERIFIED",
        )
        for index in (1, 2, 3)
    ]
    report_path = tmp_path / "explicit_environment_report.json"
    _write_json(report_path, _report_payload(artifacts))
    by_path = {artifact.run_dir: artifact for artifact in artifacts}

    def loader(path: str | Path, *, role: str):
        del role
        return by_path[Path(path).resolve()]

    return artifacts, report_path, loader


def _build(evidence):
    artifacts, report_path, loader = evidence
    with mock.patch.object(gate, "load_completed_replay_artifact", side_effect=loader):
        return gate.build_instrumented_physical_gate(
            artifact_paths=[artifact.run_dir for artifact in artifacts],
            environment_report_path=report_path,
            environment_report_sha256=_sha(report_path),
        )


def _rewrite_result(artifact: ReplayArtifact) -> None:
    _write_json(artifact.run_dir / "result.json", artifact.result)


def test_three_explicit_instrumented_passes_build_read_only_gate(evidence) -> None:
    artifacts, report_path, _loader = evidence
    observed_paths = sorted(
        path for path in report_path.parent.rglob("*") if path.is_file()
    )
    before = {path: (path.stat().st_mtime_ns, _sha(path)) for path in observed_paths}

    payload = _build(evidence)

    assert payload["schema_version"] == gate.SCHEMA_VERSION
    assert payload["status"] == "PASS"
    assert payload["gate_passed"] is True
    assert payload["admitted_run_count"] == 3
    assert payload["environment_report"]["b_counted_run_ordinals"] == [1]
    assert all(
        row["all_distinct"] is True
        for row in payload["distinctness"].values()
    )
    assert {row["shutdown_status"] for row in payload["runs"]} == {
        "GRACEFUL_EXIT",
        "FAST_EXIT_VERIFIED",
    }
    assert all(
        row["wheel_integral"]["loader_checksum_covered"] is True
        and row["wheel_integral"]["target_integral_verdict"] == "PASS"
        for row in payload["runs"]
    )
    assert before == {
        path: (path.stat().st_mtime_ns, _sha(path)) for path in observed_paths
    }


def test_validate_reloads_evidence_and_rejects_gate_tamper(evidence) -> None:
    artifacts, report_path, loader = evidence
    payload = _build(evidence)
    with mock.patch.object(gate, "load_completed_replay_artifact", side_effect=loader):
        validated = gate.validate_instrumented_physical_gate(
            payload,
            artifact_paths=[artifact.run_dir for artifact in artifacts],
            environment_report_path=report_path,
            environment_report_sha256=_sha(report_path),
        )
    assert validated == payload

    tampered = copy.deepcopy(payload)
    tampered["runs"][0]["trial_id"] = 99
    with mock.patch.object(gate, "load_completed_replay_artifact", side_effect=loader):
        with pytest.raises(
            gate.InstrumentedPhysicalGateError,
            match="payload differs from current artifact bytes",
        ):
            gate.validate_instrumented_physical_gate(
                tampered,
                artifact_paths=[artifact.run_dir for artifact in artifacts],
                environment_report_path=report_path,
                environment_report_sha256=_sha(report_path),
            )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda result: result.__setitem__("fresh_process_clean_reset", False), "clean_reset"),
        (lambda result: result.__setitem__("artifact_valid", False), "artifact_valid"),
        (lambda result: result.__setitem__("scheduler_complete", False), "scheduler_complete"),
        (lambda result: result.__setitem__("motion_start_ready", False), "motion_start_ready"),
        (lambda result: result.__setitem__("dispatch_complete", False), "dispatch_complete"),
        (lambda result: result.__setitem__("physical_success", False), "physical_success"),
        (lambda result: result.__setitem__("strict_full_success", False), "strict_full_success"),
        (
            lambda result: result["physical_evidence"].__setitem__(
                "role_capture_verdict", "NE"
            ),
            "verdict is NE",
        ),
        (
            lambda result: result["physical_evidence"].__setitem__(
                "full_physical_verdict", "FAIL"
            ),
            "verdict is FAIL",
        ),
        (
            lambda result: result["physical_evidence"].__setitem__(
                "full_physical_verdict", "NOT_EVALUABLE"
            ),
            "verdict is NOT_EVALUABLE",
        ),
        (
            lambda result: (
                result["physical_evidence"]["criteria"]["collision_safe"].update(
                    {"availability": "MISSING", "passed": None}
                ),
                result["physical_evidence"]["strict_criteria"].__setitem__(
                    "collision_safe", None
                ),
            ),
            "availability is not AVAILABLE",
        ),
        (
            lambda result: result["physical_evidence"].__setitem__(
                "not_evaluable_reasons", ["missing separation evidence"]
            ),
            "not_evaluable_reasons",
        ),
        (
            lambda result: result["respawn"].__setitem__("root_pose_written", True),
            "root_pose_written",
        ),
    ),
)
def test_missing_ne_fail_and_false_contracts_are_rejected(
    evidence, mutation, message: str
) -> None:
    artifacts, _report_path, _loader = evidence
    mutation(artifacts[0].result)
    _rewrite_result(artifacts[0])
    with pytest.raises(gate.InstrumentedPhysicalGateError, match=message):
        _build(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result.__setitem__(
                "wheel_target_integral_complete", False
            ),
            "wheel_target_integral_complete is not true",
        ),
        (
            lambda result: result.__setitem__(
                "wheel_target_integral_verdict", "FAIL"
            ),
            "wheel_target_integral_verdict is not PASS",
        ),
        (
            lambda result: result["wheel_integral_evidence"].__setitem__(
                "target_integral_verdict", "NOT_EVALUABLE"
            ),
            "target_integral_verdict is not PASS",
        ),
        (
            lambda result: result["wheel_integral_evidence"].__setitem__(
                "structural_errors", ["missing ACK epoch"]
            ),
            "structural_errors is missing/non-empty",
        ),
        (
            lambda result: result["wheel_integral_evidence"].__setitem__(
                "target_not_evaluable_reasons", ["missing physics tick"]
            ),
            "target_not_evaluable_reasons is missing/non-empty",
        ),
    ),
)
def test_wheel_integral_pass_fields_are_hard_gate_requirements(
    evidence, mutation, message: str
) -> None:
    artifacts, _report_path, _loader = evidence
    mutation(artifacts[0].result)
    _rewrite_result(artifacts[0])
    with pytest.raises(gate.InstrumentedPhysicalGateError, match=message):
        _build(evidence)


def test_wheel_integral_file_must_exactly_equal_result_embedding(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    path = artifacts[0].run_dir / gate.WHEEL_INTEGRAL_EVIDENCE_FILENAME
    durable = copy.deepcopy(artifacts[0].result["wheel_integral_evidence"])
    durable["tampered_only_on_disk"] = True
    _write_json(path, durable)
    (artifacts[0].run_dir / "checksums.sha256").write_text(
        f"{_sha(path)}  {gate.WHEEL_INTEGRAL_EVIDENCE_FILENAME}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="differs from result.wheel_integral_evidence",
    ):
        _build(evidence)


def test_wheel_integral_file_requires_loader_checksum_coverage(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    unrelated = artifacts[0].run_dir / "unrelated.json"
    _write_json(unrelated, {"ok": True})
    (artifacts[0].run_dir / "checksums.sha256").write_text(
        f"{_sha(unrelated)}  unrelated.json\n",
        encoding="utf-8",
    )
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="required file is not covered",
    ):
        _build(evidence)


def test_requires_exact_v003_and_instrumented_mode(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    artifacts[0].result["source_version"] = "v004_not_v003"
    _rewrite_result(artifacts[0])
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="exact v003"):
        _build(evidence)

    artifacts[0].result["source_version"] = SOURCE_VERSION
    artifacts[0].result["contact_mode"] = "formal"
    _rewrite_result(artifacts[0])
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="not instrumented"):
        _build(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("environment_equivalence_role", "B"),
        ("environment_equivalence_diagnostic", True),
        ("environment_equivalence_diagnostic_complete", True),
        ("diagnostic_role", "U"),
        ("ordinary_ui_diagnostic", True),
        ("ordinary_ui_diagnostic_complete", True),
        ("qualification_scope", "TRAJECTORY_COMPARISON"),
        ("gate1_eligible", False),
        ("gate1_physical_qualification_eligible", False),
        ("environment_equivalence_eligible", True),
        ("physical_qualification_eligible", False),
    ),
)
def test_diagnostic_results_cannot_be_promoted_to_gate1(
    evidence, field: str, value
) -> None:
    artifacts, _report_path, _loader = evidence
    artifacts[0].result[field] = value
    _rewrite_result(artifacts[0])
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match=rf"result {field} is not the exact Gate-1 eligibility value",
    ):
        _build(evidence)


def test_diagnostic_runtime_cannot_be_promoted_to_gate1(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    artifacts[0].runtime_environment["environment_equivalence_role"] = "B"
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="runtime environment_equivalence_role is not the exact Gate-1",
    ):
        _build(evidence)


def test_diagnostic_batch_request_cannot_be_promoted_to_gate1(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    batch_path = Path(artifacts[0].provenance["batch_root"]) / "batch_request.json"
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    payload["environment_equivalence_role"] = "B"
    payload["qualification_scope"] = "TRAJECTORY_COMPARISON"
    payload["gate1_physical_qualification_eligible"] = False
    payload["environment_equivalence_eligible"] = True
    payload["args"]["environment_equivalence_role"] = "B"
    _write_json(batch_path, payload)
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="batch request environment_equivalence_role is not the exact Gate-1",
    ):
        _build(evidence)


def test_explicit_paths_only_and_duplicates_rejected(evidence, tmp_path: Path) -> None:
    artifacts, report_path, loader = evidence
    with mock.patch.object(gate, "load_completed_replay_artifact", side_effect=loader):
        with pytest.raises(gate.InstrumentedPhysicalGateError, match="exactly 3"):
            gate.build_instrumented_physical_gate(
                artifact_paths=[artifacts[0].run_dir, artifacts[1].run_dir],
                environment_report_path=report_path,
                environment_report_sha256=_sha(report_path),
            )
        with pytest.raises(gate.InstrumentedPhysicalGateError, match="duplicated"):
            gate.build_instrumented_physical_gate(
                artifact_paths=[artifacts[0].run_dir] * 3,
                environment_report_path=report_path,
                environment_report_sha256=_sha(report_path),
            )
        discovery_root = tmp_path / "latest"
        (discovery_root / "nested").mkdir(parents=True)
        with pytest.raises(
            gate.InstrumentedPhysicalGateError,
            match="recursive/latest discovery is forbidden",
        ):
            gate.build_instrumented_physical_gate(
                artifact_paths=[discovery_root, artifacts[1].run_dir, artifacts[2].run_dir],
                environment_report_path=report_path,
                environment_report_sha256=_sha(report_path),
            )


def test_cross_run_plan_environment_physics_and_runtime_identity_fail_closed(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    artifacts[1].provenance["plan_sha256"] = "8" * 64
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="plan_sha256 differs"):
        _build(evidence)

    artifacts[1].provenance["plan_sha256"] = PLAN_SHA
    artifacts[1].runtime_environment["scene_config"]["physics_dt"] = 0.01
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match=(
            "environment_identity_sha256 differs|physics_identity_sha256 differs|"
            "physics dt is not consistently"
        ),
    ):
        _build(evidence)

    artifacts[1].runtime_environment["scene_config"]["physics_dt"] = 1.0 / 120.0
    artifacts[1].result["motion_start_readiness"][
        "adapter_runtime_instance_id"
    ] = artifacts[0].result["motion_start_readiness"]["adapter_runtime_instance_id"]
    _rewrite_result(artifacts[1])
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="conflicting.*runtime"):
        _build(evidence)


def test_trial_and_result_hash_distinctness_are_enforced(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    duplicated = copy.deepcopy(artifacts[0].result)
    artifacts[1].result.clear()
    artifacts[1].result.update(duplicated)
    _rewrite_result(artifacts[1])
    artifacts[1].provenance["batch_root"] = artifacts[0].provenance["batch_root"]
    artifacts[1].provenance["batch_shutdown_closure"]["status"] = (
        artifacts[0].provenance["batch_shutdown_closure"]["status"]
    )
    with pytest.raises(gate.InstrumentedPhysicalGateError) as exc_info:
        _build(evidence)
    message = str(exc_info.value)
    assert "result_sha256" in message or "trial_id" in message or "batch" in message


def test_environment_report_sha_status_and_b_binding_are_strict(evidence) -> None:
    artifacts, report_path, loader = evidence
    original_sha = _sha(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "PENDING_RUNTIME_A_B"
    _write_json(report_path, report)
    with mock.patch.object(gate, "load_completed_replay_artifact", side_effect=loader):
        with pytest.raises(gate.InstrumentedPhysicalGateError, match="SHA-256 differs"):
            gate.build_instrumented_physical_gate(
                artifact_paths=[artifact.run_dir for artifact in artifacts],
                environment_report_path=report_path,
                environment_report_sha256=original_sha,
            )
        with pytest.raises(gate.InstrumentedPhysicalGateError, match="status is not PASS"):
            gate.build_instrumented_physical_gate(
                artifact_paths=[artifact.run_dir for artifact in artifacts],
                environment_report_path=report_path,
                environment_report_sha256=_sha(report_path),
            )

    report = _report_payload(artifacts)
    report["runtime_readback"]["runs"]["B"]["artifact_root"] = str(
        artifacts[1].artifact_root
    )
    _write_json(report_path, report)
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="binding is inconsistent"):
        _build(evidence)


def test_environment_report_plan_identity_is_bound_to_all_three_runs(evidence) -> None:
    artifacts, report_path, _loader = evidence
    report = _report_payload(artifacts)
    report["runtime_readback"]["runs"]["B"]["provenance"]["plan_sha256"] = "0" * 64
    _write_json(report_path, report)
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="environment report B plan_sha256 differs",
    ):
        _build(evidence)


def test_preclose_checksum_and_source_closure_proofs_are_required(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    closure = artifacts[0].provenance["batch_shutdown_closure"]
    original_preclose = closure["preclose_evidence_files"]
    closure["preclose_evidence_files"] = {}
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="preclose evidence-file closure is empty",
    ):
        _build(evidence)

    closure["preclose_evidence_files"] = original_preclose
    artifacts[0].provenance["source_closure"]["files_sha256"] = "0" * 64
    with pytest.raises(
        gate.InstrumentedPhysicalGateError,
        match="source closure file-map identity mismatch",
    ):
        _build(evidence)


def test_report_b_may_be_omitted_from_three_but_cannot_be_duplicated(evidence, tmp_path: Path) -> None:
    artifacts, report_path, _loader = evidence
    report = _report_payload(artifacts)
    report["runtime_readback"]["runs"]["B"]["artifact_root"] = str(
        (tmp_path / "prior_b_artifact").resolve()
    )
    report["runtime_readback"]["runs"]["B"]["run_dir"] = str(
        (tmp_path / "prior_b_artifact" / "prior_b_run").resolve()
    )
    _write_json(report_path, report)
    payload = _build(evidence)
    assert payload["environment_report"]["b_counted_run_ordinals"] == []
    assert payload["environment_report"]["b_counted_at_most_once"] is True


def test_legacy_or_nonzero_shutdown_is_rejected(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    batch = Path(artifacts[0].provenance["batch_root"])
    outcome = _shutdown_payload(status="NORMAL_EXIT")
    outcome["handshake_state"] = "CLOSE_RETURNED"
    _write_json(batch / "shutdown_outcome.json", outcome)
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="NORMAL_EXIT|not allowed"):
        _build(evidence)

    outcome = _shutdown_payload(status="GRACEFUL_EXIT")
    outcome["intended_returncode"] = 1
    outcome["child_returncode"] = 1
    _write_json(batch / "shutdown_outcome.json", outcome)
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="zero process return"):
        _build(evidence)


def test_result_file_tamper_after_loader_snapshot_is_rejected(evidence) -> None:
    artifacts, _report_path, _loader = evidence
    durable = json.loads(
        (artifacts[0].run_dir / "result.json").read_text(encoding="utf-8")
    )
    durable["post_validation_tamper"] = True
    _write_json(artifacts[0].run_dir / "result.json", durable)
    with pytest.raises(gate.InstrumentedPhysicalGateError, match="changed or differs"):
        _build(evidence)
