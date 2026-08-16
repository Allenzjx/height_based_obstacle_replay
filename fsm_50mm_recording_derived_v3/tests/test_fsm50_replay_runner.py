from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fsm_50mm_recording_derived_v3 import fsm50_replay_runner as runner
from fsm_50mm_recording_derived_v3.fsm50_task_success import (
    NOT_EVALUATED,
    POSTURE_NOT_APPLICABLE,
    REPLAY_TASK_NOT_EVALUATED,
    TABLE_COLUMNS,
    TaskSuccessAssessment,
)


def _write_recording(root: Path, version: str, *, step_count: int = 1) -> runner.RecordingVersion:
    directory = root / "versions" / version
    directory.mkdir(parents=True)
    steps_path = directory / "accepted_steps.jsonl"
    steps_path.write_text(
        json.dumps({"index": 1, "events": [{"time": 0.0, "command": "wheel stop"}]})
        + "\n",
        encoding="utf-8",
    )
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "version_id": version,
                "step_count": step_count,
                "accepted_steps_sha256": runner.sha256_file(steps_path),
            }
        ),
        encoding="utf-8",
    )
    return runner.RecordingVersion(
        version_id=version,
        directory=directory.resolve(),
        accepted_steps_path=steps_path.resolve(),
        metadata_path=metadata_path.resolve(),
    )


def _plan(*, profile: str = "motion_only", digest: str = "a" * 64):
    return SimpleNamespace(
        profile=profile,
        plan_sha256=digest,
        events=[SimpleNamespace()],
        segments=[SimpleNamespace()],
        final_time_s=2.0,
    )


def _assessment(version: str, *, reason: str = "MANUAL_VIDEO_REVIEW_PENDING") -> TaskSuccessAssessment:
    return TaskSuccessAssessment(
        version=version,
        step_count=1,
        fast_segment_count=1,
        evaluation_status=NOT_EVALUATED,
        task_result=REPLAY_TASK_NOT_EVALUATED,
        posture_result=POSTURE_NOT_APPLICABLE,
        body_crossed_front_face=True,
        required_leg_lift_completed=True,
        final_recoverable=True,
        final_wheel_classes={"FL": "AIR", "FR": "TOP", "RL": "TOP", "RR": "TOP"},
        peak_roll=0.1,
        peak_pitch=0.2,
        video_path="",
        first_actual_failure_phase="",
        hard_failure_reasons=(),
        not_evaluated_reasons=(reason,),
        secondary_diagnostics=(),
        notes=(),
        classification_reasons=("manual review pending",),
    )


def _run_args(tmp_path: Path, recording_root: Path, **overrides) -> argparse.Namespace:
    values = {
        "versions": ["all"],
        "recording_root": recording_root,
        "output_root": tmp_path / "runs" / "50mm_fast_replay",
        "report_root": tmp_path / "reports",
        "resume": False,
        "continue_on_error": True,
        "telemetry_hz": 15.0,
        "post_run_settle_s": 0.5,
        "task_timeout_s": 0.0,
        "terminal_timeout_s": 7200.0,
        "operation_timeout_s": 30.0,
        "sim_startup_timeout_s": 600.0,
        "sim_worker_status_timeout_s": 10.0,
        "sim_worker_log_lines": 200,
        "worker_launch_mode": "auto",
        "worker_python_exe": "",
        "isaaclab_bat": "C:/robotics_sim/IsaacLab/isaaclab.bat",
        "device": "cuda:0",
        "livestream": 0,
        "experience": "",
        "accept_isaac_eula": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dynamic_enumeration_and_selector_order(tmp_path: Path) -> None:
    root = tmp_path / "height_050mm"
    v12 = _write_recording(root, "v012_late")
    v3 = _write_recording(root, "v003_first")
    v5 = _write_recording(root, "v005_middle")
    (root / "versions" / "notes").mkdir()

    versions = runner.enumerate_recording_versions(root)
    assert [row.version_id for row in versions] == [
        v3.version_id,
        v5.version_id,
        v12.version_id,
    ]
    assert runner.select_recording_versions(versions, ["v012", "v003"]) == [v12, v3]
    assert runner.select_recording_versions(versions, ["all"]) == [v3, v5, v12]
    with pytest.raises(ValueError, match="cannot be combined"):
        runner.select_recording_versions(versions, ["all", "v003"])
    with pytest.raises(ValueError, match="matched 0"):
        runner.select_recording_versions(versions, ["v999"])


def test_current_corpus_is_not_hardcoded_and_production_plan_is_motion_only() -> None:
    versions = runner.enumerate_recording_versions(runner.DEFAULT_RECORDING_ROOT)
    assert [row.short_id for row in versions] == [
        "v003",
        "v005",
        "v006",
        "v007",
        "v008",
        "v009",
        "v010",
        "v011",
        "v012",
    ]
    v003 = runner.select_recording_versions(versions, ["v003"])[0]
    prepared = runner.prepare_replay(v003)
    assert prepared.plan.profile == "motion_only"
    assert len(prepared.plan.segments) == 112
    assert len(prepared.steps) == 24


def test_prepare_replay_calls_production_signature_and_rejects_wrong_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "height_050mm"
    recording = _write_recording(root, "v003_fixture")
    calls: list[tuple[object, dict[str, object]]] = []

    def build(steps, **kwargs):
        calls.append((steps, kwargs))
        return _plan()

    prepared = runner.prepare_replay(
        recording,
        plan_builder=build,
        max_wheel_speed_rad_s=2.0,
    )
    assert prepared.plan.profile == "motion_only"
    assert calls[0][1]["profile"] == "fast"
    assert calls[0][1]["max_wheel_speed"] == 2.0
    assert calls[0][1]["sequence_total_steps"] == 1

    with pytest.raises(runner.RunnerContractError, match="motion_only"):
        runner.prepare_replay(
            recording,
            plan_builder=lambda *_args, **_kwargs: _plan(profile="fast"),
            max_wheel_speed_rad_s=2.0,
        )


def test_request_schema_and_output_directory_are_exact(tmp_path: Path) -> None:
    root = tmp_path / "height_050mm"
    recording = _write_recording(root, "v003_fixture")
    prepared = runner.prepare_replay(
        recording,
        plan_builder=lambda *_args, **_kwargs: _plan(),
        max_wheel_speed_rad_s=2.0,
    )
    paths = runner.allocate_run_paths(tmp_path / "runs" / "50mm_fast_replay", recording.version_id)
    request = runner.build_worker_task_request(
        prepared,
        paths,
        request_id="request",
        plan_id="plan",
    )
    assert paths.run_dir.parent == (
        tmp_path / "runs" / "50mm_fast_replay" / recording.version_id
    ).resolve()
    assert set(request) == {
        "schema_version",
        "enabled",
        "execution_mode",
        "request_id",
        "plan_id",
        "plan_sha256",
        "plan_event_count",
        "plan_segment_count",
        "source_version",
        "height_mm",
        "step_count",
        "run_dir",
        "accepted_steps_path",
        "accepted_steps_sha256",
        "telemetry_hz",
        "video_fps",
        "capture_video",
        "post_run_settle_s",
        "timeout_s",
        "filtered_contact_bank_enabled",
    }
    assert request["schema_version"] == runner.REQUEST_SCHEMA
    assert request["execution_mode"] == "normal_development"
    assert request["filtered_contact_bank_enabled"] is False
    assert request["telemetry_hz"] == 15.0
    assert request["video_fps"] == 15.0


def test_tables_are_stable_atomic_and_include_on_disk_inventory(tmp_path: Path) -> None:
    root = tmp_path / "height_050mm"
    v5 = _write_recording(root, "v005_fixture", step_count=2)
    v3 = _write_recording(root, "v003_fixture", step_count=1)
    update = _assessment(v3.version_id).to_table_row()
    csv_path, md_path = runner.update_task_success_tables(
        report_root=tmp_path / "reports",
        recordings=[v5, v3],
        updates=[update],
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == TABLE_COLUMNS
    assert [row["version"] for row in rows] == [v3.version_id, v5.version_id]
    assert rows[0]["not_evaluated_reasons"] == "MANUAL_VIDEO_REVIEW_PENDING"
    assert rows[1]["not_evaluated_reasons"] == "NOT_RUN"
    assert "Task completion and final-posture quality are separate" in md_path.read_text(
        encoding="utf-8"
    )
    assert not list((tmp_path / "reports").glob("*.tmp"))


def _bound_review_fixture(tmp_path: Path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    task_inputs = run_dir / runner.CLASSIFIER_INPUTS_NAME
    task_inputs.write_text("{}\n", encoding="utf-8")
    video = run_dir / "actual_viewport_video.mp4"
    video.write_bytes(b"video-frames")
    document = runner._video_verdict_template(
        source_version="v003_fixture",
        request_id="request",
        plan_id="plan",
        run_dir=run_dir,
        task_inputs_path=task_inputs,
        video_path=video,
    )
    return run_dir, task_inputs, video, document


def test_manual_video_verdict_is_strict_and_sha_bound(tmp_path: Path) -> None:
    run_dir, task_inputs, video, document = _bound_review_fixture(tmp_path)
    template = run_dir / runner.VIDEO_VERDICT_TEMPLATE_NAME
    runner.atomic_write_json(template, document)
    _raw, verdict = runner._load_bound_video_verdict(
        template,
        source_version="v003_fixture",
        request_id="request",
        plan_id="plan",
        run_dir=run_dir,
        task_inputs_path=task_inputs,
        video_path=video,
        require_complete=False,
    )
    assert verdict is None
    with pytest.raises(runner.RunnerContractError, match="incomplete template"):
        runner._load_bound_video_verdict(
            template,
            source_version="v003_fixture",
            request_id="request",
            plan_id="plan",
            run_dir=run_dir,
            task_inputs_path=task_inputs,
            video_path=video,
        )

    document["review_complete"] = True
    document["reviewed_utc"] = "2026-08-14T19:00:00+00:00"
    document["verdict"]["task_completed"] = True
    verdict_path = run_dir / "review.json"
    runner.atomic_write_json(verdict_path, document)
    loaded, verdict = runner._load_bound_video_verdict(
        verdict_path,
        source_version="v003_fixture",
        request_id="request",
        plan_id="plan",
        run_dir=run_dir,
        task_inputs_path=task_inputs,
        video_path=video,
    )
    assert loaded == document
    assert verdict is not None and verdict.task_completed is True

    video.write_bytes(b"changed")
    with pytest.raises(runner.RunnerContractError, match="video_sha256"):
        runner._load_bound_video_verdict(
            verdict_path,
            source_version="v003_fixture",
            request_id="request",
            plan_id="plan",
            run_dir=run_dir,
            task_inputs_path=task_inputs,
            video_path=video,
        )


def _ready_status(request: dict[str, object]) -> dict[str, object]:
    preflight = {
        key: request[key]
        for key in (
            "schema_version",
            "request_id",
            "plan_id",
            "source_version",
            "height_mm",
            "step_count",
            "run_dir",
            "telemetry_hz",
            "video_fps",
            "filtered_contact_bank_enabled",
        )
    }
    preflight.update(
        enabled=True,
        execution_mode="normal_development",
        preflight_ok=True,
    )
    return {
        "ready": True,
        "worker_pid": 321,
        "worker_session_id": "worker-session",
        "adapter_runtime_instance_id": "adapter-id",
        "root_state_write_count": 0,
        "task_replay_preflight_ready": True,
        "task_replay_request_id": request["request_id"],
        "worker_task_replay_preflight": preflight,
        "worker_task_replay_session": {
            "enabled": True,
            "execution_mode": "normal_development",
            "request_id": request["request_id"],
            "source_version": request["source_version"],
            "state": "ready_for_plan",
            "filtered_contact_bank_enabled": False,
        },
        "worker_artifact_session": {"enabled": False},
        "worker_artifact_preflight": {"enabled": False},
    }


def test_startup_binding_rejects_missing_task_request_passthrough(tmp_path: Path) -> None:
    request = {
        "schema_version": runner.REQUEST_SCHEMA,
        "request_id": "request",
        "plan_id": "plan",
        "source_version": "v003_fixture",
        "height_mm": 50,
        "step_count": 1,
        "run_dir": str(tmp_path.resolve()),
        "telemetry_hz": 15.0,
        "video_fps": 15.0,
        "filtered_contact_bank_enabled": False,
    }
    status = _ready_status(request)
    assert runner.validate_task_worker_binding(status, request=request)["worker_pid"] == 321
    status["task_replay_preflight_ready"] = False
    with pytest.raises(runner.RunnerContractError, match="preflight is not ready"):
        runner.validate_task_worker_binding(status, request=request)


def _terminal(run_dir: Path, *, complete: bool = True) -> dict[str, object]:
    for name in ("task_inputs.json", "worker_task_replay_result.json", "video.mp4"):
        (run_dir / name).write_bytes(b"{}" if name.endswith("json") else b"video")
    return {
        "type": "task_replay_complete" if complete else "task_replay_failed",
        "operation": "task_replay",
        "phase": "TASK_REPLAY_COMPLETE" if complete else "TASK_REPLAY_FAILED",
        "accepted": complete,
        "task_replay_complete": complete,
        "request_id": "request",
        "plan_id": "plan",
        "source_version": "v003_fixture",
        "run_dir": str(run_dir.resolve()),
        "task_inputs_path": str((run_dir / "task_inputs.json").resolve()),
        "worker_result_path": str((run_dir / "worker_task_replay_result.json").resolve()),
        "video_path": str((run_dir / "video.mp4").resolve()),
        "video_writer_quiesced": True,
        "error": "" if complete else "scheduler failed",
    }


def test_raw_terminal_and_separate_operation_ack_are_exact(tmp_path: Path) -> None:
    terminal = _terminal(tmp_path)
    validated = runner.validate_task_terminal(
        terminal,
        request_id="request",
        plan_id="plan",
        source_version="v003_fixture",
        run_dir=tmp_path,
    )
    assert validated["type"] == "task_replay_complete"
    binding = {
        "worker_pid": 321,
        "worker_session_id": "worker-session",
        "adapter_runtime_instance_id": "adapter-id",
    }
    ack = {
        **terminal,
        "type": "operation_ack",
        "worker_pid": 321,
        "worker_session_id": "worker-session",
        "adapter_runtime_instance_id": "adapter-id",
        "artifact_request_id": "",
        "root_state_write_count": 0,
    }
    runner.validate_task_operation_ack(
        ack,
        terminal=terminal,
        request_id="request",
        plan_id="plan",
        source_version="v003_fixture",
        run_dir=tmp_path,
        worker_binding=binding,
    )
    ack["plan_id"] = "wrong"
    with pytest.raises(runner.RunnerContractError, match="plan_id mismatch"):
        runner.validate_task_operation_ack(
            ack,
            terminal=terminal,
            request_id="request",
            plan_id="plan",
            source_version="v003_fixture",
            run_dir=tmp_path,
            worker_binding=binding,
        )


def test_wait_for_terminal_never_promotes_operation_ack(tmp_path: Path) -> None:
    raw = _terminal(tmp_path)

    class Process:
        def poll(self):
            return None

    class Client:
        latest_task_replay_terminal = raw
        latest_task_replay_ack = {**raw, "type": "operation_ack"}
        process = Process()

        def poll(self):
            return []

        def status(self):
            return {
                "last_task_replay_terminal": raw,
                "last_task_replay_ack": self.latest_task_replay_ack,
            }

    result = runner.wait_for_task_terminal(
        Client(), request_id="request", timeout_s=0.1, poll_interval_s=0.001
    )
    assert result["type"] == "task_replay_complete"


def _shutdown_outcome(*, include_returned: bool = False) -> dict[str, object]:
    identity = {
        "worker_pid": 321,
        "worker_session_id": "worker-session",
        "adapter_runtime_instance_id": "adapter-id",
        "artifact_request_id": "",
        "task_replay_request_id": "task-request",
        "root_state_write_count": 0,
    }
    common = {
        "request_id": "shutdown-request",
        "mode": "fast",
        "accepted": True,
        "error": "",
        **identity,
        "close_kwargs": {"wait_for_replicator": False, "skip_cleanup": True},
        "runtime_version": "5.1.0",
    }
    outcome: dict[str, object] = {
        "pid": 321,
        "returncode": 0,
        "forced_termination": False,
        "normal_exit": True,
        "timed_out": False,
        "requested_mode": "fast",
        "shutdown_ack": {**common, "type": "operation_ack", "operation": "shutdown"},
        "close_requested": {**common, "type": "close_requested"},
        "close_requested_receipt": {
            **common,
            "type": "close_receipt",
            "close_event_type": "close_requested",
            "received": True,
        },
        "close_returned": {},
        "close_returned_receipt": {},
    }
    if include_returned:
        outcome["close_returned"] = {**common, "type": "close_returned"}
        outcome["close_returned_receipt"] = {
            **common,
            "type": "close_receipt",
            "close_event_type": "close_returned",
            "received": True,
        }
    return outcome


@pytest.mark.parametrize("include_returned", [False, True])
def test_fast_shutdown_requires_receipted_task_identity_and_rc0(include_returned: bool) -> None:
    outcome = _shutdown_outcome(include_returned=include_returned)
    runner.validate_fast_shutdown(
        outcome,
        request_id="shutdown-request",
        owned_worker_pid=321,
        task_request_id="task-request",
        worker_session_id="worker-session",
        adapter_runtime_instance_id="adapter-id",
    )
    outcome["close_requested_receipt"] = {}
    with pytest.raises(runner.RunnerContractError, match="receipt"):
        runner.validate_fast_shutdown(
            outcome,
            request_id="shutdown-request",
            owned_worker_pid=321,
            task_request_id="task-request",
            worker_session_id="worker-session",
            adapter_runtime_instance_id="adapter-id",
        )


def test_second_simulator_preflight_is_measured_and_bound_to_classifier_inputs() -> None:
    snapshot = [{"pid": 10, "name": "python.exe", "command_line": "pytest"}]
    evidence = runner._assert_no_simulator_process(lambda: snapshot, lambda _rows: [])
    assert evidence["checked"] is True
    assert evidence["second_simulator_process_detected"] is False
    assert len(evidence["process_snapshot_sha256"]) == 64
    inputs = runner.build_classifier_inputs(
        {
            "completed_result": {
                "source_version": "v003_fixture",
                "single_simulator_preflight_available": False,
                "second_simulator_process_detected": None,
            },
            "physical_evidence": {},
            "final_telemetry_row": {},
        },
        process_preflight=evidence,
    )
    assert inputs["completed_result"]["single_simulator_preflight_available"] is True
    assert inputs["completed_result"]["second_simulator_process_detected"] is False
    assert inputs["completed_result"]["second_simulator_process_preflight"] == evidence
    with pytest.raises(runner.RunnerContractError, match="SECOND_SIMULATOR_PROCESS"):
        runner._assert_no_simulator_process(lambda: snapshot, lambda rows: list(rows))


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("checked", 1, "measured process preflight"),
        ("checked_utc", 1, "checked_utc"),
        ("snapshot_process_count", True, "snapshot_process_count"),
        ("process_snapshot_sha256", "A" * 64, "SHA-256"),
        ("conflict_count", False, "conflict_count"),
        ("conflicts", (), "list of objects"),
        ("second_simulator_process_detected", 0, "boolean conflict verdict"),
    ],
)
def test_classifier_preflight_contract_is_type_strict(
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    evidence = runner._assert_no_simulator_process(lambda: [], lambda _rows: [])
    evidence[field] = invalid_value
    with pytest.raises(runner.RunnerContractError, match=message):
        runner.build_classifier_inputs(
            {
                "completed_result": {},
                "physical_evidence": {},
                "final_telemetry_row": {},
            },
            process_preflight=evidence,
        )


def test_classifier_preflight_rejects_inconsistent_structured_verdict() -> None:
    evidence = runner._assert_no_simulator_process(lambda: [], lambda _rows: [])
    evidence["second_simulator_process_detected"] = True
    with pytest.raises(runner.RunnerContractError, match="verdict is inconsistent"):
        runner.build_classifier_inputs(
            {
                "completed_result": {},
                "physical_evidence": {},
                "final_telemetry_row": {},
            },
            process_preflight=evidence,
        )


def test_runner_manifest_persists_exact_validated_worker_protocol_payloads(
    tmp_path: Path,
) -> None:
    recording = _write_recording(tmp_path / "recordings", "v003_fixture")
    prepared = runner.PreparedReplay(
        recording=recording,
        steps=[{"index": 1}],
        accepted_steps_sha256=runner.sha256_file(recording.accepted_steps_path),
        plan=_plan(),
    )
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    paths = runner.RunPaths(
        run_dir=run_dir,
        request_path=run_dir / "worker_task_replay_request.json",
        runner_result_path=run_dir / runner.RUNNER_RESULT_NAME,
    )
    request = runner.build_worker_task_request(
        prepared,
        paths,
        request_id="request",
        plan_id="plan",
    )
    runner.atomic_write_json(paths.request_path, request)
    ready = runner.validate_task_worker_binding(
        _ready_status(request), request=request
    )
    start_ack = runner._validate_operation_ack(
        {
            "type": "operation_ack",
            "operation": "start_playback_plan",
            "request_id": "request",
            "accepted": True,
            "error": "",
            "nested_wire_payload": {"indices": [1, 2, 3]},
        },
        operation="start_playback_plan",
        request_id="request",
    )
    terminal = runner.validate_task_terminal(
        _terminal(run_dir),
        request_id="request",
        plan_id="plan",
        source_version="v003_fixture",
        run_dir=run_dir,
    )
    task_ack = runner.validate_task_operation_ack(
        {
            **terminal,
            "type": "operation_ack",
            "worker_pid": 321,
            "worker_session_id": "worker-session",
            "adapter_runtime_instance_id": "adapter-id",
            "artifact_request_id": "",
            "root_state_write_count": 0,
        },
        terminal=terminal,
        request_id="request",
        plan_id="plan",
        source_version="v003_fixture",
        run_dir=run_dir,
        worker_binding=ready,
    )
    manifest = runner._run_manifest(
        prepared=prepared,
        paths=paths,
        request=request,
        terminal=terminal,
        assessment=_assessment(recording.version_id),
        shutdown_outcome={},
        shutdown_verified=False,
        error="",
        worker_ready_status=ready,
        worker_playback_start_ack=start_ack,
        worker_task_replay_terminal=terminal,
        worker_task_replay_ack=task_ack,
    )
    runner.atomic_write_json(paths.runner_result_path, manifest)
    persisted = runner._strict_json_load(paths.runner_result_path)
    assert persisted["worker_ready_status"] == ready
    assert persisted["worker_playback_start_ack"] == start_ack
    assert persisted["worker_task_replay_terminal"] == terminal
    assert persisted["worker_task_replay_ack"] == task_ack


def test_run_selected_holds_one_worker_at_a_time_and_writes_no_empty_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "height_050mm"
    _write_recording(root, "v003_fixture")
    _write_recording(root, "v005_fixture")
    args = _run_args(tmp_path, root)
    lock_events: list[str] = []
    first_call = True

    class Lock:
        acquired = False

        def acquire(self):
            self.acquired = True
            lock_events.append("acquire")

        def release(self):
            self.acquired = False
            lock_events.append("release")

    def fake_run_one(recording, **kwargs):
        nonlocal first_call
        if first_call:
            assert not (Path(args.report_root) / runner.CSV_REPORT_NAME).exists()
            first_call = False
        assert kwargs["process_preflight"]["checked"] is True
        run_dir = Path(args.output_root) / recording.version_id / "fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / runner.RUNNER_RESULT_NAME
        result_path.write_text("{}", encoding="utf-8")
        return runner.ReplayAttempt(
            recording=recording,
            run_dir=run_dir,
            assessment=_assessment(recording.version_id),
            runner_result_path=result_path,
            shutdown_verified=True,
        )

    monkeypatch.setattr(runner, "run_one_replay", fake_run_one)
    snapshots: list[int] = []

    def snapshot():
        snapshots.append(1)
        return []

    attempts = runner.run_selected(
        args,
        lock_factory=Lock,
        process_snapshot_fn=snapshot,
        conflict_detector=lambda _rows: [],
    )
    assert [row.recording.short_id for row in attempts] == ["v003", "v005"]
    assert lock_events == ["acquire", "release", "acquire", "release"]
    assert len(snapshots) == 4  # immediately before and after each worker
    assert (Path(args.report_root) / runner.CSV_REPORT_NAME).is_file()


def test_resume_uses_bound_terminal_artifacts_not_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording_root = tmp_path / "height_050mm"
    recording = _write_recording(recording_root, "v003_fixture")
    output_root = tmp_path / "runs" / "50mm_fast_replay"
    run_dir = output_root / recording.version_id / "run-1"
    run_dir.mkdir(parents=True)
    request_path = run_dir / "worker_task_replay_request.json"
    task_inputs_path = run_dir / "task_inputs.json"
    classifier_inputs_path = run_dir / runner.CLASSIFIER_INPUTS_NAME
    worker_result_path = run_dir / "worker_task_replay_result.json"
    video_path = run_dir / "actual_viewport_video.mp4"
    video_path.write_bytes(b"video")
    source_sha = runner.sha256_file(recording.accepted_steps_path)
    plan_sha = "a" * 64
    request = {
        "schema_version": runner.REQUEST_SCHEMA,
        "source_version": recording.version_id,
        "accepted_steps_path": str(recording.accepted_steps_path),
        "accepted_steps_sha256": source_sha,
        "plan_sha256": plan_sha,
        "request_id": "request",
        "plan_id": "plan",
        "run_dir": str(run_dir),
        "plan_event_count": 1,
        "plan_segment_count": 1,
    }
    runner.atomic_write_json(request_path, request)
    runner.atomic_write_json(task_inputs_path, {"worker": True})
    classifier_inputs = {
        "completed_result": {
            "source_version": recording.version_id,
            "plan_event_count": 1,
            "plan_segment_count": 1,
            "second_simulator_process_detected": False,
        },
        "physical_evidence": {},
        "final_telemetry_row": {},
    }
    runner.atomic_write_json(classifier_inputs_path, classifier_inputs)
    runner.atomic_write_json(
        worker_result_path,
        {
            "source_version": recording.version_id,
            "request_id": "request",
            "plan_id": "plan",
            "run_dir": str(run_dir),
            "task_inputs_path": str(task_inputs_path),
            "video_writer_quiesced": True,
        },
    )
    assessment = _assessment(recording.version_id)
    assessment = TaskSuccessAssessment(
        **{**assessment.__dict__, "video_path": str(video_path)}
    )
    manifest = {
        "schema_version": runner.RUNNER_RESULT_SCHEMA,
        "source_version": recording.version_id,
        "accepted_steps_sha256": source_sha,
        "plan_sha256": plan_sha,
        "request_id": "request",
        "plan_id": "plan",
        "run_dir": str(run_dir),
        "request_path": str(request_path),
        "request_sha256": runner.sha256_file(request_path),
        "task_inputs_path": str(task_inputs_path),
        "task_inputs_sha256": runner.sha256_file(task_inputs_path),
        "classifier_inputs_path": str(classifier_inputs_path),
        "classifier_inputs_sha256": runner.sha256_file(classifier_inputs_path),
        "worker_result_path": str(worker_result_path),
        "worker_result_sha256": runner.sha256_file(worker_result_path),
        "video_path": str(video_path),
        "video_sha256": runner.sha256_file(video_path),
        "terminal_phase": "TASK_REPLAY_COMPLETE",
        "shutdown_verified": True,
        "assessment": assessment.to_table_row(),
    }
    manifest_path = run_dir / runner.RUNNER_RESULT_NAME
    runner.atomic_write_json(manifest_path, manifest)
    monkeypatch.setattr(
        runner,
        "classify_run_inputs",
        lambda **_kwargs: assessment,
    )
    assert runner._latest_resumable_manifest(
        recording,
        output_root,
        expected_plan_sha256=plan_sha,
    ) == manifest

    video_path.write_bytes(b"tampered")
    assert runner._latest_resumable_manifest(
        recording,
        output_root,
        expected_plan_sha256=plan_sha,
    ) is None


def test_cli_budget_and_production_worker_defaults(tmp_path: Path) -> None:
    args = runner.build_parser().parse_args(["run"])
    assert args.terminal_timeout_s == 7200.0
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    worker = runner._build_production_worker_args(args, request_path)
    assert worker.profile == "fast"
    assert worker.fsm50_task_request_path == str(request_path.resolve())
    assert worker.fsm50_gate_request_path == ""
    assert worker.physics_dt == pytest.approx(1.0 / 120.0)
    assert worker.render_interval == 8
    assert worker.servo_stiffness == 600.0
    assert worker.servo_damping == 60.0
    assert worker.wheel_direction == 1.0
    assert worker.headless is False
