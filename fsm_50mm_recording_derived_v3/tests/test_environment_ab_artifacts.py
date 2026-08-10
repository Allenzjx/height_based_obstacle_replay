from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from command_model import SERVO_JOINT_NAMES
from fsm_50mm_recording_derived_v3.environment_ab_artifacts import (
    ArtifactValidationError,
    compare_environment_run_artifacts,
    extract_trajectory_metrics,
    generate_environment_equivalence_report,
    load_completed_replay_artifact,
    validate_artifact_triplet,
)


PLAN_SHA = "b" * 64
SOURCE_VERSION = "v012_20260806_231025_027004_manual"
LEGS = ("FL", "FR", "RL", "RR")
DEFAULT_RECORDING_CONTENT = '{"index":1,"events":[]}\n'


def _write_json(path: Path, payload, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=allow_nan) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _rewrite_checksums(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "checksums.sha256",
            ".partial",
            ".complete",
            ".finalized",
            ".failed",
        }:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(run_dir).as_posix()}")
    (run_dir / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _current_git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip().lower()


def _evidence_files(paths) -> dict[str, dict[str, object]]:
    return {
        str(path.resolve()): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }


def _rewrite_preclose_checksums(batch_root: Path) -> None:
    excluded = {
        "checksums.sha256",
        "checksums.preclose.sha256",
        "batch_results.preclose.json",
        "batch_finalization.preclose.json",
        "preclose_complete.json",
        "shutdown_outcome.json",
        ".partial",
        ".complete",
        ".finalized",
        ".failed",
    }
    rows = []
    for path in sorted(batch_root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        candidate = path
        if path == batch_root / "batch_results.json":
            candidate = batch_root / "batch_results.preclose.json"
        elif path == batch_root / "batch_finalization.json":
            candidate = batch_root / "batch_finalization.preclose.json"
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(batch_root).as_posix()}")
    (batch_root / "checksums.preclose.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _reseal_batch(artifact_root: Path, run_dir: Path, *, sync_result: bool = False) -> None:
    batch_root = artifact_root.parent.parent
    if sync_result:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        _write_json(batch_root / "batch_results.json", [result])
        _write_json(batch_root / "batch_results.preclose.json", [result])
    _rewrite_preclose_checksums(batch_root)
    preclose_finalization = json.loads(
        (batch_root / "batch_finalization.preclose.json").read_text(encoding="utf-8")
    )
    source_integrity = json.loads(
        (batch_root / "source_integrity.json").read_text(encoding="utf-8")
    )
    stable_paths = [
        batch_root / "batch_results.preclose.json",
        batch_root / "batch_finalization.preclose.json",
        batch_root / "checksums.preclose.sha256",
        batch_root / "source_integrity.json",
        batch_root / "batch_request.json",
        run_dir / "result.json",
        run_dir / "checksums.sha256",
    ]
    marker_path = batch_root / "preclose_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["evidence"] = {
        "physics_result_count": 1,
        "source_integrity": source_integrity,
        "batch_finalization": preclose_finalization,
        "evidence_files": _evidence_files(stable_paths),
    }
    _write_json(marker_path, marker)
    _rewrite_checksums(batch_root)


def _geometry(offset: float = 0.0):
    lo = [0.5213121737735307 + offset, -1.0, 0.0]
    hi = [2.5786877308590377 + offset, 1.0, 0.05]
    return {
        "prim_path": "/World/Obstacle",
        "prim_valid": True,
        "visual_valid": True,
        "collision_valid": True,
        "measured_bounds": {"min": lo, "max": hi},
        "visual_bounds": {"min": lo, "max": hi},
        "collision_bounds": {"min": lo, "max": hi},
        "height_m": 0.05,
        "width_m": 2.0,
        "length_m": 2.057375557085507,
        "front_face_x_m": lo[0],
        "center_y_m": 0.0,
        "bottom_z_m": 0.0,
        "top_z_m": 0.05,
        "collision_height_m": 0.05,
        "collision_width_m": 2.0,
    }


def _telemetry_rows(
    *,
    source_version: str,
    offset: float,
    force_offset: float,
    contact: str,
    instrumented: bool,
    grid_tick_delta: int = 0,
    time_origin_s: float = 10.0,
):
    rows = []
    dt = 1.0 / 120.0
    ticks = [0, 12 + grid_tick_delta, 24]
    for index, tick in enumerate(ticks):
        positions = {
            name: float(index) * 0.01 + offset
            for name in SERVO_JOINT_NAMES
        }
        rotations = {
            leg: float(index) * (leg_index + 1) * 0.1 + offset
            for leg_index, leg in enumerate(LEGS)
        }
        travels = {
            leg: rotations[leg] * 0.05
            for leg in LEGS
        }
        row = {
                "sample_index": index,
                "time_s": time_origin_s + tick * dt,
                "sim_step": 100 + tick,
                "source_version": source_version,
                "base_x_m": 0.1 * index + offset,
                "base_y_m": offset,
                "base_z_m": 0.10 + offset,
                "base_qw": 1.0,
                "base_qx": 0.0,
                "base_qy": 0.0,
                "base_qz": 0.0,
                "measured_joint_position_rad": positions,
                "wheel_integrated_rotation_rad": rotations,
                "wheel_integrated_travel_m": travels,
                "wheel_contact_classes": {leg: contact for leg in LEGS},
                "wheel_contact_force_up_n": {
                    leg: 10.0 + index + force_offset for leg in LEGS
                },
                "wheel_contact_force_total_n": {
                    leg: 11.0 + index + force_offset for leg in LEGS
                },
                "wheel_net_force_layout_valid": True,
                "wheel_net_force_valid": True,
                "wheel_net_force_error": "",
                "wheel_contact_force_common_source": (
                    "isaaclab.ContactSensor.net_forces_w"
                ),
            }
        if instrumented:
            row.update(
                {
                    "filtered_contact_layout_valid": True,
                    "filtered_contact_force_valid": True,
                    "filtered_contact_geometry_valid": True,
                    "filtered_contact_available": True,
                    "filtered_contact_error": "",
                    "wheel_filtered_contacts": [
                        {"leg": leg, "surface": surface, "force_valid": True}
                        for leg in LEGS
                        for surface in ("ground", "obstacle")
                    ],
                    "nonwheel_obstacle_contacts": [
                        {
                            "body": "base_link",
                            "active": False,
                            "force_valid": True,
                            "contact_point_valid": True,
                        }
                    ],
                    "collision_evidence_source": "tests.nonwheel_contact_bank",
                    "collision_evidence_valid": True,
                    "collision_evidence_error": "",
                }
            )
        rows.append(row)
    return rows


def _make_run(
    root: Path,
    role: str,
    *,
    offset: float = 0.0,
    force_offset: float = 0.0,
    contact: str = "GROUND",
    recording_content: str = DEFAULT_RECORDING_CONTENT,
    plan_sha: str = PLAN_SHA,
    source_version: str = SOURCE_VERSION,
    device: str = "cuda:0",
    grid_tick_delta: int = 0,
    time_origin_s: float = 10.0,
    instrumented: bool = False,
    source_git_head: str | None = None,
    source_file_variant: str = "formal",
):
    batch_root = root / f"{role}_batch"
    version_root = batch_root / f"{role}_version"
    artifact_root = version_root / f"{role}_artifact"
    run_dir = artifact_root / f"{role}_run"
    run_dir.mkdir(parents=True)
    contact_mode = "instrumented" if instrumented else "formal"
    (artifact_root / ".finalized").write_text("finalized\n", encoding="utf-8")
    _write_json(artifact_root / "artifact_pointer.json", {"run_dir": str(run_dir.resolve())})

    source_path = root / "source_closure" / f"{source_file_variant}.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        source_path.write_text(
            f"SOURCE_VARIANT = {source_file_variant!r}\n",
            encoding="utf-8",
        )
    frozen_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    frozen_head = source_git_head or _current_git_head()
    frozen_files = {
        str(source_path.resolve()): {
            "sha256": frozen_source_sha,
            "size_bytes": source_path.stat().st_size,
        }
    }
    source_pre = {
        "created_utc": f"2026-08-09T00:00:00+00:00-{role}",
        "git": {"head": frozen_head, "branch": "main", "status_porcelain": ""},
        "files": frozen_files,
    }
    source_post = {
        **source_pre,
        "created_utc": f"2026-08-09T00:00:01+00:00-{role}",
    }
    source_comparison = {
        "equal": True,
        "changed": [],
        "missing": [],
        "added": [],
        "before_created_utc": source_pre["created_utc"],
        "after_created_utc": source_post["created_utc"],
    }
    _write_json(artifact_root / "source_freeze_pre.json", source_pre)
    _write_json(artifact_root / "source_freeze.json", source_pre)
    _write_json(artifact_root / "source_freeze_post.json", source_post)
    _write_json(artifact_root / "source_integrity.json", source_comparison)

    accepted_steps_path = run_dir / "input" / "accepted_steps.jsonl"
    accepted_steps_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_steps_path.write_text(recording_content, encoding="utf-8")
    recording_sha = hashlib.sha256(accepted_steps_path.read_bytes()).hexdigest()

    rows = _telemetry_rows(
        source_version=source_version,
        offset=offset,
        force_offset=force_offset,
        contact=contact,
        instrumented=instrumented,
        grid_tick_delta=grid_tick_delta,
        time_origin_s=time_origin_s,
    )
    _write_jsonl(run_dir / "fsm50_telemetry.jsonl", rows)
    physical = {
        "source_version": source_version,
        "sample_count": len(rows),
        "evidence_complete": True,
        "physical_success": False,
    }
    _write_json(run_dir / "physical_evidence.json", physical)

    video_path = run_dir / "fsm50_viewport.mp4"
    video_path.write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41"
    )
    viewport_manifest_path = run_dir / "viewport_video_manifest.json"
    viewport_manifest = {
        "schema_version": "fsm50.recording_viewport_video.v1",
        "contact_mode": contact_mode,
        "capture_requested": True,
        "diagnostic_only": False,
        "valid": True,
        "artifact_valid": True,
        "actual_viewport_video": True,
        "not_camera_video": False,
        "source": "actual_active_isaac_gui_viewport_render_product",
        "render_product_path": "/Render/Viewport",
        "frame_count": 3,
        "fps": 15.0,
        "video_path": str(video_path.resolve()),
        "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "error": "",
    }
    _write_json(viewport_manifest_path, viewport_manifest)
    video = {
        **viewport_manifest,
        "manifest_path": str(viewport_manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(viewport_manifest_path.read_bytes()).hexdigest(),
    }
    common_video_fields = {
        "contact_mode": contact_mode,
        "actual_viewport_video": True,
        "video_path": video["video_path"],
        "video_sha256": video["video_sha256"],
        "viewport_video_manifest_path": video["manifest_path"],
        "viewport_video_manifest_sha256": video["manifest_sha256"],
        "video": video,
    }
    result = {
        "schema_version": "fsm50.recording_replay_result.v1",
        "source_version": source_version,
        "accepted_steps_sha256": recording_sha,
        "expected_preflight_steps_sha256": recording_sha,
        "requested_profile": "fast",
        "canonical_profile": "motion_only",
        "plan_sha256": plan_sha,
        "plan_event_count": 3,
        "plan_segment_count": 2,
        "plan_final_time_s": 1.0,
        "run_dir": str(run_dir.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "scheduler_complete": True,
        "scheduler_stop_reason": "complete",
        "timed_out": False,
        "simulation_app_stopped": False,
        "classification": "PHYSICAL_FAILURE",
        "physical_evidence": physical,
        "artifact_valid": True,
        "strict_full_success": False,
        "source_integrity": {
            "ok": True,
            "scope": "recording_version",
            "comparison": source_comparison,
        },
        "visualization": {"ok": True},
        "lifecycle": {"finalized": True, "failed": False, "strict_success": False},
        **common_video_fields,
    }
    _write_json(run_dir / "result.json", result)
    _write_json(
        run_dir / "failure_diagnostics.json",
        {"classification": "PHYSICAL_FAILURE", "artifact_valid": True},
    )
    scene_config = {
        "obstacle_height_m": 0.05,
        "physics_dt": 1.0 / 120.0,
        "render_interval": 8,
        "device": device,
        "servo_stiffness": 600.0,
        "servo_damping": 60.0,
        "wheel_damping": 20.0,
        "telemetry_contact_sensors_enabled": True,
        "contact_sensor_factory": "tests.custom_contact_factory" if instrumented else None,
    }
    runtime = {
        "runtime": {
            "isaacsim": "4.5.0-test",
            "isaaclab": "0.0-test",
            "torch": "2.7-test",
        },
        "scene_config": scene_config,
        "live_obstacle_geometry": _geometry(offset),
        "environment_equivalence": {"ok": True},
        "physics_dt_s": 1.0 / 120.0,
        "render_interval": 8,
        "contact_sensor_type": "CustomContactBank" if instrumented else "ProductionContactBank",
        "contact_sensor_error": "",
        **common_video_fields,
    }
    _write_json(run_dir / "runtime_environment.json", runtime)
    _write_json(
        run_dir / "visual_recording_manifest.json",
        {
            "schema_version": "fsm50.recording_visual_evidence.v2",
            "kind": "actual_active_gui_viewport_video_with_telemetry_visualization",
            **common_video_fields,
            "not_camera_video": False,
            "artifact_valid": True,
            "telemetry_visualization": {"ok": True},
            "basis": ["actual active Isaac GUI viewport frames", "fsm50_telemetry.csv"],
        },
    )
    _write_json(
        run_dir / "input" / "fast_plan" / f"{source_version}_fast_plan.json",
        {
            "schema_version": "fsm50.recording_fast_plan.v1",
            "source_version": source_version,
            "profile_requested": "fast",
            "profile_normalized": "motion_only",
            "plan_sha256": plan_sha,
            "event_count": 3,
            "segment_count": 2,
            "final_time_s": 1.0,
            "segments": [],
        },
    )
    _rewrite_checksums(run_dir)

    _write_json(
        batch_root / "batch_request.json",
        {"created_utc": "2026-08-09T00:00:00+00:00", "versions": [source_version]},
    )
    batch_source_integrity = {
        "equal": True,
        "changed": [],
        "missing": [],
        "added": [],
        "before_created_utc": "2026-08-09T00:00:00+00:00",
        "after_created_utc": "2026-08-09T00:00:01+00:00",
    }
    _write_json(batch_root / "source_integrity.json", batch_source_integrity)
    _write_json(batch_root / "batch_results.json", [result])
    preclose_finalization = {
        "artifact_root": str(batch_root.resolve()),
        "finalized": True,
        "failed": False,
        "strict_success": False,
        "batch_error": "",
        "close_error": "PENDING_SIMULATION_CLOSE",
        "phase": "PRECLOSE_FINALIZED",
        "source_integrity": batch_source_integrity,
        "finalization_errors": [],
    }
    _write_json(batch_root / "batch_finalization.json", preclose_finalization)
    _rewrite_checksums(batch_root)
    (batch_root / "batch_results.preclose.json").write_bytes(
        (batch_root / "batch_results.json").read_bytes()
    )
    (batch_root / "batch_finalization.preclose.json").write_bytes(
        (batch_root / "batch_finalization.json").read_bytes()
    )
    (batch_root / "checksums.preclose.sha256").write_bytes(
        (batch_root / "checksums.sha256").read_bytes()
    )
    stable_preclose_paths = [
        batch_root / "batch_results.preclose.json",
        batch_root / "batch_finalization.preclose.json",
        batch_root / "checksums.preclose.sha256",
        batch_root / "source_integrity.json",
        batch_root / "batch_request.json",
        run_dir / "result.json",
        run_dir / "checksums.sha256",
    ]
    preclose_marker = {
        "schema_version": "fsm50.preclose_complete.v1",
        "created_utc": "2026-08-09T00:00:02+00:00",
        "token": "test-token",
        "parent_pid": 100,
        "child_pid": 101,
        "batch_root": str(batch_root.resolve()),
        "evidence": {
            "physics_result_count": 1,
            "source_integrity": batch_source_integrity,
            "batch_finalization": preclose_finalization,
            "evidence_files": _evidence_files(stable_preclose_paths),
        },
    }
    _write_json(batch_root / "preclose_complete.json", preclose_marker)
    (batch_root / ".finalized").write_text("finalized\n", encoding="utf-8")
    shutdown_outcome = {
        "schema_version": "fsm50.shutdown_outcome.v1",
        "created_utc": "2026-08-09T00:00:03+00:00",
        "status": "NORMAL_EXIT",
        "parent_pid": 100,
        "child_pid": 101,
        "child_returncode": 0,
        "preclose_observed": True,
        "handshake_state": "CLOSE_RETURNED",
    }
    _write_json(batch_root / "shutdown_outcome.json", shutdown_outcome)
    finalization = {
        **preclose_finalization,
        "close_error": "",
        "phase": "SHUTDOWN_COMPLETE",
        "shutdown_outcome": shutdown_outcome,
    }
    _write_json(batch_root / "batch_finalization.json", finalization)
    _rewrite_checksums(batch_root)
    return artifact_root, run_dir


def test_completed_artifacts_extract_eight_metrics_and_write_pass_report(tmp_path):
    a1_root, _ = _make_run(tmp_path, "A1", offset=0.0, force_offset=0.0)
    a2_root, _ = _make_run(tmp_path, "A2", offset=0.001, force_offset=0.1)
    b_root, _ = _make_run(
        tmp_path,
        "B",
        offset=0.002,
        force_offset=0.2,
        instrumented=True,
    )

    a1 = load_completed_replay_artifact(a1_root, role="A1")
    assert set(a1.trajectory_metrics) == {
        "root_trajectory",
        "joint_trajectory",
        "wheel_rotation",
        "wheel_travel",
        "final_pose",
        "obstacle_geometry",
        "contact_class",
        "contact_force",
    }
    assert a1.trajectory_metrics["root_trajectory"][0] == [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
    assert a1.trajectory_metrics["wheel_rotation"]["FL"] == [0.0, 0.1, 0.2]
    assert a1.trajectory_metrics["contact_class"]["RR"] == ["GROUND"] * 3
    assert a1.trajectory_metrics["contact_force"]["FL"]["upward_n"] == [10.0, 11.0, 12.0]
    assert a1.provenance["contact_mode"] == "formal"
    assert a1.provenance["batch_shutdown_status"] == "NORMAL_EXIT"
    assert a1.provenance["batch_finalization_phase"] == "SHUTDOWN_COMPLETE"
    assert Path(a1.provenance["batch_root"]).is_dir()
    assert len(a1.provenance["batch_shutdown_closure"]["closure_sha256"]) == 64

    a2 = load_completed_replay_artifact(a2_root, role="A2")
    b = load_completed_replay_artifact(b_root, role="B")
    triplet = validate_artifact_triplet(a1, a2, b)
    assert triplet["ok"] is True
    assert triplet["allowed_B_differences"]
    assert triplet["contact_modes"] == {
        "A1": "formal",
        "A2": "formal",
        "B": "instrumented",
    }

    comparison = compare_environment_run_artifacts(
        baseline_a1=a1_root,
        baseline_a2=a2_root,
        instrumented_b=b_root,
    )
    assert comparison["ok"] is True
    assert set(comparison) == {
        "schema_version",
        "ok",
        "fail_closed",
        "instrumentation_comparison",
        "trajectory_comparison",
        "runtime_readback",
        "artifact_conversion",
    }

    report_path = tmp_path / "ENVIRONMENT_EQUIVALENCE_REPORT.json"
    report = generate_environment_equivalence_report(
        a1_run=a1_root,
        a2_run=a2_root,
        b_run=b_root,
        output_path=report_path,
        fingerprint={"schema_version": "test.static.fingerprint", "environment_equivalent": False},
    )
    assert report_path.is_file()
    assert report["status"] == "PASS"
    assert report["environment_equivalent"] is True
    assert report["trajectory_comparison"]["ok"] is True
    assert report["runtime_readback"]["readback_complete"] is True
    assert set(report["extra"]["artifact_conversion"]["metric_names"]) == set(a1.trajectory_metrics)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recording_content", '{"index":2,"events":[]}\n', "accepted_steps_sha256"),
        ("plan_sha", "d" * 64, "plan_sha256"),
        ("source_version", "v011_other", "source_version"),
        ("device", "cpu", "device"),
        ("grid_tick_delta", 1, "sample grid"),
        ("time_origin_s", 20.0, "start_time_s"),
        ("source_file_variant", "variant", "source_files_sha256"),
    ],
)
def test_triplet_rejects_provenance_and_sample_grid_drift(tmp_path, field, value, message):
    a1_root, _ = _make_run(tmp_path, "A1")
    a2_root, _ = _make_run(tmp_path, "A2", offset=0.001, force_offset=0.1)
    kwargs = {field: value}
    b_root, _ = _make_run(
        tmp_path,
        "B",
        offset=0.002,
        force_offset=0.2,
        instrumented=True,
        **kwargs,
    )
    result = validate_artifact_triplet(
        load_completed_replay_artifact(a1_root, role="A1"),
        load_completed_replay_artifact(a2_root, role="A2"),
        load_completed_replay_artifact(b_root, role="B"),
    )
    assert result["ok"] is False
    assert message.lower() in " ".join(result["failures"]).lower()


def test_partial_camera_only_missing_and_nonfinite_artifacts_fail_closed(tmp_path):
    partial_root, _ = _make_run(tmp_path, "partial")
    (partial_root / ".partial").write_text("running\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="partial"):
        load_completed_replay_artifact(partial_root, role="partial")

    legacy_root, legacy_run = _make_run(tmp_path, "legacy")
    _write_json(
        legacy_run / "visual_recording_manifest.json",
        {
            "kind": "fsm50_equivalent_telemetry_visualization",
            "not_camera_video": True,
            "artifact_valid": True,
            "basis": ["fsm50_telemetry.jsonl"],
        },
    )
    _rewrite_checksums(legacy_run)
    _reseal_batch(legacy_root, legacy_run)
    with pytest.raises(ArtifactValidationError, match="legacy telemetry-only"):
        load_completed_replay_artifact(legacy_root, role="legacy")

    camera_root, camera_run = _make_run(tmp_path, "camera")
    (camera_run / "viewport_video_manifest.json").unlink()
    _rewrite_checksums(camera_run)
    _reseal_batch(camera_root, camera_run)
    with pytest.raises(ArtifactValidationError, match="actual viewport|not video evidence"):
        load_completed_replay_artifact(camera_root, role="camera")

    fake_video_root, fake_video_run = _make_run(tmp_path, "fake_video")
    fake_video_path = fake_video_run / "fsm50_viewport.mp4"
    fake_video_path.write_bytes(b"this is a telemetry plot, not an mp4")
    viewport_manifest_path = fake_video_run / "viewport_video_manifest.json"
    viewport_manifest = json.loads(viewport_manifest_path.read_text(encoding="utf-8"))
    viewport_manifest["video_sha256"] = hashlib.sha256(fake_video_path.read_bytes()).hexdigest()
    _write_json(viewport_manifest_path, viewport_manifest)
    _rewrite_checksums(fake_video_run)
    _reseal_batch(fake_video_root, fake_video_run)
    with pytest.raises(ArtifactValidationError, match="ftyp"):
        load_completed_replay_artifact(fake_video_root, role="fake_video")

    missing_root, missing_run = _make_run(tmp_path, "missing")
    (missing_run / "fsm50_telemetry.jsonl").unlink()
    _rewrite_checksums(missing_run)
    _reseal_batch(missing_root, missing_run)
    with pytest.raises(ArtifactValidationError, match="missing"):
        load_completed_replay_artifact(missing_root, role="missing")

    nonfinite_root, nonfinite_run = _make_run(tmp_path, "nonfinite")
    telemetry_path = nonfinite_run / "fsm50_telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["base_x_m"] = float("nan")
    _write_jsonl(telemetry_path, rows, allow_nan=True)
    _rewrite_checksums(nonfinite_run)
    _reseal_batch(nonfinite_root, nonfinite_run)
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        load_completed_replay_artifact(nonfinite_root, role="nonfinite")


def test_checksum_tamper_and_incomplete_lifecycle_fail_closed(tmp_path):
    tamper_root, tamper_run = _make_run(tmp_path, "tamper")
    with (tamper_run / "fsm50_telemetry.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        load_completed_replay_artifact(tamper_root, role="tamper")

    incomplete_root, incomplete_run = _make_run(tmp_path, "incomplete")
    result_path = incomplete_run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["lifecycle"] = {"finalized": False, "failed": False}
    _write_json(result_path, result)
    _rewrite_checksums(incomplete_run)
    _reseal_batch(incomplete_root, incomplete_run, sync_result=True)
    with pytest.raises(ArtifactValidationError, match="lifecycle"):
        load_completed_replay_artifact(incomplete_root, role="incomplete")


def test_shutdown_closure_missing_timeout_failed_and_tamper_fail_closed(tmp_path):
    missing_root, _ = _make_run(tmp_path / "missing", "missing_shutdown")
    missing_batch = missing_root.parent.parent
    (missing_batch / "shutdown_outcome.json").unlink()
    with pytest.raises(ArtifactValidationError, match="missing:.*shutdown_outcome"):
        load_completed_replay_artifact(missing_root, role="missing_shutdown")

    timeout_root, _ = _make_run(tmp_path / "timeout", "timeout")
    timeout_batch = timeout_root.parent.parent
    shutdown_path = timeout_batch / "shutdown_outcome.json"
    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
    shutdown["status"] = "SIMULATION_CLOSE_TIMEOUT"
    _write_json(shutdown_path, shutdown)
    with pytest.raises(ArtifactValidationError, match="not NORMAL_EXIT"):
        load_completed_replay_artifact(timeout_root, role="timeout")

    failed_root, _ = _make_run(tmp_path / "failed", "failed")
    failed_batch = failed_root.parent.parent
    (failed_batch / ".finalized").unlink()
    (failed_batch / ".failed").write_text("failed\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="finalized|failed marker"):
        load_completed_replay_artifact(failed_root, role="failed")

    tamper_root, _ = _make_run(tmp_path / "tamper", "batch_tamper")
    tamper_batch = tamper_root.parent.parent
    with (tamper_batch / "batch_request.json").open("a", encoding="utf-8") as stream:
        stream.write(" \n")
    with pytest.raises(ArtifactValidationError, match="checksum mismatch|evidence hash"):
        load_completed_replay_artifact(tamper_root, role="batch_tamper")

    snapshot_root, _ = _make_run(tmp_path / "snapshot", "snapshot_tamper")
    snapshot_batch = snapshot_root.parent.parent
    with (snapshot_batch / "batch_finalization.preclose.json").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(" \n")
    # Refresh only the final live checksum: immutable preclose evidence must
    # still detect that its snapshotted file changed.
    _rewrite_checksums(snapshot_batch)
    with pytest.raises(ArtifactValidationError, match="evidence hash/size mismatch"):
        load_completed_replay_artifact(snapshot_root, role="snapshot_tamper")


def test_current_source_closure_and_git_head_are_revalidated_for_report(tmp_path):
    report_root = tmp_path / "report"
    a1_root, _ = _make_run(report_root, "A1")
    a2_root, _ = _make_run(report_root, "A2", offset=0.001, force_offset=0.1)
    b_root, _ = _make_run(
        report_root,
        "B",
        offset=0.002,
        force_offset=0.2,
        instrumented=True,
    )
    source_path = report_root / "source_closure" / "formal.py"
    source_path.write_text("SOURCE_VARIANT = 'tampered-after-run'\n", encoding="utf-8")
    report = generate_environment_equivalence_report(
        a1_run=a1_root,
        a2_run=a2_root,
        b_run=b_root,
        output_path=report_root / "ENVIRONMENT_EQUIVALENCE_REPORT.json",
        fingerprint={"schema_version": "test.static.fingerprint"},
    )
    assert report["status"] == "FAIL"
    assert report["environment_equivalent"] is False
    assert report["runtime_readback"]["readback_complete"] is False
    assert "changed after replay" in " ".join(report["runtime_readback"]["failures"])

    missing_root, _ = _make_run(tmp_path / "source_missing", "source_missing")
    (tmp_path / "source_missing" / "source_closure" / "formal.py").unlink()
    with pytest.raises(ArtifactValidationError, match="source closure file is missing"):
        load_completed_replay_artifact(missing_root, role="source_missing")

    head_root, _ = _make_run(
        tmp_path / "head",
        "head",
        source_git_head="d" * 40,
    )
    with pytest.raises(ArtifactValidationError, match="current git HEAD.*frozen"):
        load_completed_replay_artifact(head_root, role="head")


def test_large_b_contact_drift_writes_fail_not_pass(tmp_path):
    a1_root, _ = _make_run(tmp_path, "A1")
    a2_root, _ = _make_run(tmp_path, "A2", offset=0.001, force_offset=0.1)
    b_root, _ = _make_run(
        tmp_path,
        "B",
        offset=0.002,
        force_offset=0.2,
        contact="TOP",
        instrumented=True,
    )
    report = generate_environment_equivalence_report(
        a1_run=a1_root,
        a2_run=a2_root,
        b_run=b_root,
        output_path=tmp_path / "ENVIRONMENT_EQUIVALENCE_REPORT.json",
        fingerprint={"schema_version": "test.static.fingerprint"},
    )
    assert report["status"] == "FAIL"
    assert report["environment_equivalent"] is False
    assert "contact_class" in report["trajectory_comparison"]["failed_metrics"]


def test_invalid_input_still_writes_a_fail_report(tmp_path):
    a1_root, _ = _make_run(tmp_path, "A1")
    a2_root, _ = _make_run(tmp_path, "A2")
    bad_root, bad_run = _make_run(tmp_path, "B", instrumented=True)
    (bad_run / "fsm50_telemetry.jsonl").unlink()
    report_path = tmp_path / "ENVIRONMENT_EQUIVALENCE_REPORT.json"
    report = generate_environment_equivalence_report(
        a1_run=a1_root,
        a2_run=a2_root,
        b_run=bad_root,
        output_path=report_path,
        fingerprint={"schema_version": "test.static.fingerprint"},
    )
    assert report_path.is_file()
    assert report["status"] == "FAIL"
    assert report["environment_equivalent"] is False
    assert report["runtime_readback"]["readback_complete"] is False
    assert report["extra"]["artifact_conversion"]["fail_closed"] is True


def test_extract_metrics_rejects_missing_required_force_even_when_other_values_exist():
    rows = _telemetry_rows(
        source_version=SOURCE_VERSION,
        offset=0.0,
        force_offset=0.0,
        contact="AIR",
        instrumented=False,
    )
    del rows[0]["wheel_contact_force_up_n"]["FL"]
    runtime = {"live_obstacle_geometry": _geometry()}
    with pytest.raises(ArtifactValidationError, match="wheel_contact_force_up_n.*FL"):
        extract_trajectory_metrics(
            rows,
            runtime,
            contact_mode="formal",
            label="force_missing",
        )

    wrong_source_rows = _telemetry_rows(
        source_version=SOURCE_VERSION,
        offset=0.0,
        force_offset=0.0,
        contact="AIR",
        instrumented=False,
    )
    wrong_source_rows[0]["wheel_contact_force_common_source"] = "filtered-only-source"
    with pytest.raises(ArtifactValidationError, match="wheel_contact_force_common_source"):
        extract_trajectory_metrics(
            wrong_source_rows,
            runtime,
            contact_mode="formal",
            label="wrong_force_source",
        )


def test_formal_does_not_require_filtered_fields_but_instrumented_does(tmp_path):
    formal_root, formal_run = _make_run(tmp_path, "A1")
    formal = load_completed_replay_artifact(formal_root, role="A1")
    formal_rows = [
        json.loads(line)
        for line in (formal_run / "fsm50_telemetry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("filtered_contact_layout_valid" not in row for row in formal_rows)
    assert formal.provenance["contact_mode"] == "formal"
    formal_rows[0]["wheel_net_force_valid"] = False
    formal_rows[0]["wheel_net_force_error"] = "net_forces_w unavailable"
    formal_telemetry = formal_run / "fsm50_telemetry.jsonl"
    _write_jsonl(formal_telemetry, formal_rows)
    _rewrite_checksums(formal_run)
    _reseal_batch(formal_root, formal_run)
    with pytest.raises(ArtifactValidationError, match="wheel_net_force_valid"):
        load_completed_replay_artifact(formal_root, role="A1")

    instrumented_root, instrumented_run = _make_run(tmp_path, "B", instrumented=True)
    instrumented = load_completed_replay_artifact(instrumented_root, role="B")
    assert instrumented.provenance["contact_mode"] == "instrumented"

    telemetry_path = instrumented_run / "fsm50_telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    del rows[1]["nonwheel_obstacle_contacts"]
    _write_jsonl(telemetry_path, rows)
    _rewrite_checksums(instrumented_run)
    _reseal_batch(instrumented_root, instrumented_run)
    with pytest.raises(ArtifactValidationError, match="nonwheel_obstacle_contacts"):
        load_completed_replay_artifact(instrumented_root, role="B")
