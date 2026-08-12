from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3 import run_fsm50 as runner
from fsm_50mm_recording_derived_v3 import fsm50_isaac_runtime as isaac_runtime
from fsm_50mm_recording_derived_v3.environment_ab_artifacts import (
    generate_environment_equivalence_report,
)
from fsm_50mm_recording_derived_v3.fsm50_isaac_runtime import (
    ViewportVideoRecorder,
    _finalize_episode_result,
    _load_environment_gate,
    run_fsm_locked,
)
from fsm_50mm_recording_derived_v3.tests.test_environment_ab_artifacts import (
    _make_run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_viewport_capture_delegates_unique_frame_numbering_to_isaac() -> None:
    source = inspect.getsource(ViewportVideoRecorder.start)

    assert 'movie_capture.baseFilename = ""' in source
    assert 'movie_capture.baseFilename = "fsm50_viewport_frame.png"' not in source


def _required_run_files(run_dir: Path) -> None:
    for name in (
        "physical_evidence.json",
        "fsm50_telemetry.csv",
        "fsm50_telemetry.jsonl",
        "state_timeline.csv",
        "fsm_controller_timeline.json",
        "fsm_controller_timeline.csv",
        "runtime_environment.json",
        "fsm50_equivalent_visualization.png",
        "fsm50_equivalent_visualization.html",
    ):
        (run_dir / name).write_bytes(b"evidence\n")
    (run_dir / "visual_recording_manifest.json").write_text(
        json.dumps(
            {
                "kind": "fsm50_equivalent_telemetry_visualization",
                "not_camera_only": True,
                "visualization": {"ok": True},
                "artifact_valid": False,
            }
        ),
        encoding="utf-8",
    )


def _result(artifact_root: Path, run_dir: Path) -> dict[str, object]:
    return {
        "schema_version": "fsm50.controller_run_result.v1",
        "mode": "test-state",
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "classification": "PHYSICAL_EVIDENCE_INCOMPLETE",
        "strict_success_before_video": False,
        "strict_success": False,
        "artifact_valid": False,
        "visualization": {"ok": True},
        "captured_state_ids": ["A0_RESET_AND_SETTLE"],
        "lifecycle": {"finalized": False, "failed": False},
    }


def _environment_gate_fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    role_options = {
        "A1": {"offset": 0.0, "force_offset": 0.0},
        "A2": {"offset": 0.001, "force_offset": 0.1},
        "B": {
            "offset": 0.0005,
            "force_offset": 0.05,
            "instrumented": True,
        },
    }
    runs: dict[str, tuple[Path, Path]] = {}
    for role, options in role_options.items():
        runs[role] = _make_run(root, role, **options)

    report_path = root / "ENVIRONMENT_EQUIVALENCE_REPORT.json"
    report = generate_environment_equivalence_report(
        a1_run=runs["A1"][0],
        a2_run=runs["A2"][0],
        b_run=runs["B"][0],
        output_path=report_path,
        fingerprint={"schema_version": "test.static.fingerprint"},
    )
    assert report["status"] == "PASS"
    return report_path, report, {
        **runs,
        "batch": (runs["A1"][0].parent.parent, runs["A1"][0].parent.parent),
        "source": (
            root / "source_closure" / "formal.py",
            root / "source_closure" / "formal.py",
        ),
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_complete_nonpassing_episode_is_a_finalized_evidence_artifact(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    run_dir = artifact_root / "run"
    run_dir.mkdir(parents=True)
    (artifact_root / ".partial").write_text("running\n", encoding="utf-8")
    (artifact_root / "artifact_pointer.json").write_text(
        json.dumps({"run_dir": str(run_dir)}), encoding="utf-8"
    )
    _required_run_files(run_dir)
    source_video = tmp_path / "batch.mp4"
    source_video.write_bytes(b"real-viewport-frame-stream")
    result = _result(artifact_root, run_dir)

    _finalize_episode_result(
        result,
        video={
            "valid": True,
            "actual_viewport_video": True,
            "video_path": str(source_video),
            "video_sha256": _sha256(source_video),
        },
        source_integrity={"equal": True},
        runner=runner,
    )

    written = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert written["artifact_valid"] is True
    assert written["strict_success"] is False
    assert written["classification"] == "PHYSICAL_EVIDENCE_INCOMPLETE"
    assert written["lifecycle"]["finalized"] is True
    assert (artifact_root / ".finalized").is_file()
    assert not (artifact_root / ".partial").exists()
    assert (run_dir / "actual_viewport_video.mp4").is_file()
    manifest = json.loads(
        (run_dir / "visual_recording_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["actual_viewport_video"] is True
    assert manifest["not_camera_video"] is False
    assert "actual_viewport_video.mp4" in (run_dir / "checksums.sha256").read_text(
        encoding="utf-8"
    )


def test_missing_video_fails_artifact_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    run_dir = artifact_root / "run"
    run_dir.mkdir(parents=True)
    (artifact_root / ".partial").write_text("running\n", encoding="utf-8")
    _required_run_files(run_dir)
    result = _result(artifact_root, run_dir)

    _finalize_episode_result(
        result,
        video={"valid": False, "error": "capture unavailable"},
        source_integrity={"equal": True},
        runner=runner,
    )

    assert result["artifact_valid"] is False
    assert result["classification"] == "ARTIFACT_INVALID"
    assert result["lifecycle"]["failed"] is True
    assert (artifact_root / ".failed").is_file()


def test_environment_gate_reloads_current_artifacts_and_accepts_exact_report(
    tmp_path: Path,
) -> None:
    report_path, report, _runs = _environment_gate_fixture(tmp_path / "valid")

    loaded = _load_environment_gate(report_path)

    assert loaded == report
    assert loaded["runtime_readback"]["ok"] is True
    assert loaded["runtime_readback"]["readback_complete"] is True
    assert set(loaded["runtime_readback"]["runs"]) == {"A1", "A2", "B"}


def test_environment_gate_rejects_missing_false_and_pending_checks(
    tmp_path: Path,
) -> None:
    report_path, report, _runs = _environment_gate_fixture(tmp_path / "checks")
    cases = {
        "unsupported_schema": lambda row: row.__setitem__(
            "schema_version", "fsm50.environment_equivalence_report.v0"
        ),
        "missing_created_utc": lambda row: row.pop("created_utc"),
        "missing_static_fingerprint": lambda row: row.pop("static_fingerprint"),
        "pending_status": lambda row: row.__setitem__(
            "status", "PENDING_RUNTIME_A_B"
        ),
        "noncanonical_pass": lambda row: row.__setitem__("status", "pass"),
        "pending_static_value": lambda row: row["static_fingerprint"].__setitem__(
            "runtime", "unknown_pending_runtime_readback"
        ),
        "not_equivalent": lambda row: row.__setitem__(
            "environment_equivalent", False
        ),
        "instrumentation_not_ok": lambda row: row[
            "instrumentation_comparison"
        ].__setitem__("ok", False),
        "instrumentation_schema": lambda row: row[
            "instrumentation_comparison"
        ].__setitem__("schema_version", "pending.schema"),
        "trajectory_missing_ok": lambda row: row[
            "trajectory_comparison"
        ].pop("ok"),
        "runtime_not_ok": lambda row: row["runtime_readback"].__setitem__(
            "ok", False
        ),
        "readback_incomplete": lambda row: row["runtime_readback"].__setitem__(
            "readback_complete", False
        ),
        "conversion_not_ok": lambda row: row["extra"][
            "artifact_conversion"
        ].__setitem__("ok", False),
        "pending_dynamic_value": lambda row: row["runtime_readback"].__setitem__(
            "status", "PENDING_ARTIFACT_READBACK"
        ),
        "missing_b": lambda row: row["runtime_readback"]["runs"].pop("B"),
    }
    for case, mutate in cases.items():
        candidate = json.loads(json.dumps(report))
        mutate(candidate)
        candidate_path = report_path.with_name(f"{case}.json")
        _write_report(candidate_path, candidate)
        with pytest.raises(RuntimeError):
            _load_environment_gate(candidate_path)


@pytest.mark.parametrize(
    "field_path",
    (
        ("checksums_sha256",),
        ("metrics_sha256",),
        ("source_closure", "files_sha256"),
        ("batch_shutdown_closure", "closure_sha256"),
        ("artifact_hashes", "fsm50_telemetry.jsonl"),
        ("viewport_video", "video_sha256"),
    ),
)
def test_environment_gate_rejects_embedded_provenance_or_hash_tamper(
    tmp_path: Path,
    field_path: tuple[str, ...],
) -> None:
    report_path, report, _runs = _environment_gate_fixture(tmp_path / "report_tamper")
    candidate = json.loads(json.dumps(report))
    target = candidate["runtime_readback"]["runs"]["A1"]["provenance"]
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = "0" * 64
    _write_report(report_path, candidate)

    with pytest.raises(RuntimeError, match="provenance differs"):
        _load_environment_gate(report_path)


@pytest.mark.parametrize("tamper", ("telemetry", "source_closure"))
def test_environment_gate_rejects_disk_artifact_tamper_after_report(
    tmp_path: Path,
    tamper: str,
) -> None:
    report_path, _report, runs = _environment_gate_fixture(
        tmp_path / f"disk_{tamper}"
    )
    b_root, b_run = runs["B"]
    if tamper == "telemetry":
        with (b_run / "fsm50_telemetry.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("{}\n")
    else:
        source_pre = b_root / "source_freeze_pre.json"
        source_payload = json.loads(source_pre.read_text(encoding="utf-8"))
        source_payload["git"]["head"] = "f" * 40
        _write_report(source_pre, source_payload)

    with pytest.raises(RuntimeError, match="artifact revalidation failed"):
        _load_environment_gate(report_path)


def test_environment_gate_rejects_current_source_file_tamper(
    tmp_path: Path,
) -> None:
    report_path, _report, runs = _environment_gate_fixture(
        tmp_path / "live_source_tamper"
    )
    live_source = runs["source"][0]
    live_source.write_text("GATE_SOURCE_VERSION = 2\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="source closure file changed|current source file differs",
    ):
        _load_environment_gate(report_path)


def test_environment_gate_rejects_current_git_head_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _report, _runs = _environment_gate_fixture(
        tmp_path / "git_head_mismatch"
    )
    monkeypatch.setattr(isaac_runtime, "_current_git_head", lambda: "0" * 40)

    with pytest.raises(RuntimeError, match="current git HEAD differs"):
        _load_environment_gate(report_path)


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_batch_marker",
        "partial_batch_marker",
        "shutdown_not_normal",
        "batch_not_finalized",
        "wrong_shutdown_phase",
        "missing_preclose",
        "preclose_error",
        "preclose_snapshot",
        "closure_checksum",
    ),
)
def test_environment_gate_rejects_incomplete_batch_shutdown_closure(
    tmp_path: Path,
    tamper: str,
) -> None:
    report_path, _report, runs = _environment_gate_fixture(
        tmp_path / f"closure_{tamper}"
    )
    batch_root = runs["batch"][0]
    if tamper == "missing_batch_marker":
        (batch_root / ".finalized").unlink()
    elif tamper == "partial_batch_marker":
        (batch_root / ".partial").write_text("running\n", encoding="utf-8")
    elif tamper == "shutdown_not_normal":
        path = batch_root / "shutdown_outcome.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "SIMULATION_CLOSE_TIMEOUT"
        _write_report(path, payload)
    elif tamper == "batch_not_finalized":
        path = batch_root / "batch_finalization.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["finalized"] = False
        payload["failed"] = True
        _write_report(path, payload)
    elif tamper == "wrong_shutdown_phase":
        path = batch_root / "batch_finalization.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["phase"] = "PRECLOSE_FINALIZED"
        _write_report(path, payload)
    elif tamper == "missing_preclose":
        (batch_root / "preclose_complete.json").unlink()
    elif tamper == "preclose_error":
        path = batch_root / "preclose_complete.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evidence"]["manifest_error"] = "capture incomplete"
        _write_report(path, payload)
    elif tamper == "preclose_snapshot":
        path = batch_root / "batch_finalization.preclose.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif tamper == "closure_checksum":
        path = batch_root / "checksums.sha256"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                _sha256(batch_root / "shutdown_outcome.json"),
                "0" * 64,
                1,
            ),
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError):
        _load_environment_gate(report_path)


def test_environment_gate_is_evaluated_before_simulation_app_start() -> None:
    source = inspect.getsource(run_fsm_locked)
    assert source.index("_load_environment_gate(environment_report_path)") < source.index(
        "ensure_simulation_app(args)"
    )
