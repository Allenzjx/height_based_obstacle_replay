from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from command_model import SERVO_JOINT_NAMES
from fsm_50mm_recording_derived_v3.environment_ab_artifacts import (
    ArtifactValidationError,
    _jsonl,
    compare_environment_run_artifacts,
    compare_sealed_trajectory_diagnostics,
    extract_trajectory_metrics,
    generate_environment_equivalence_report,
    load_completed_replay_artifact,
    load_sealed_trajectory_diagnostic_artifact,
    validate_artifact_triplet,
)


PLAN_SHA = "b" * 64
SOURCE_VERSION = "v012_20260806_231025_027004_manual"
LEGS = ("FL", "FR", "RL", "RR")
DEFAULT_RECORDING_CONTENT = '{"index":1,"events":[]}\n'
WORKER_CONTROL_FILENAMES = (
    "worker_artifact_request.json",
    "worker_startup_binding.json",
    "worker_preplay_stop_ack.json",
    "worker_playback_start_ack.json",
    "worker_artifact_complete_ack.json",
)


@pytest.mark.parametrize(
    ("bad_row", "message"),
    (
        ('{"unused":NaN}', "non-finite JSON constant NaN"),
        ('{"value":1,"value":2}', "duplicate JSON object key 'value'"),
    ),
)
def test_jsonl_strict_decoder_rejects_every_row_with_location(
    tmp_path: Path,
    bad_row: str,
    message: str,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text('{"ok":true}\n' + bad_row + "\n", encoding="utf-8")

    with pytest.raises(
        ArtifactValidationError,
        match=rf":2:.*{message}",
    ):
        _jsonl(path)


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
        relative_parts = path.relative_to(run_dir).parts
        if not path.is_file() or path.name in {
            "checksums.sha256",
            ".partial",
            ".complete",
            ".finalized",
            ".failed",
        }:
            continue
        if ".telemetry_journal" in relative_parts or (
            path.name.startswith(".") and ".tmp" in path.name
        ):
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


def _write_telemetry_finalization(
    run_dir: Path,
    record_counts: dict[str, int],
    *,
    fsm50_count: int,
) -> dict[str, object]:
    canonical_files = {
        name: {
            "record_count": int(count),
            "size_bytes": (run_dir / name).stat().st_size,
            "sha256": hashlib.sha256((run_dir / name).read_bytes()).hexdigest(),
        }
        for name, count in sorted(record_counts.items())
    }
    marker = {
        "schema_version": "telemetry.canonical_finalization.v1",
        "created_wall_time_s": 1.0,
        "canonical_export_attempted": True,
        "canonical_complete": True,
        "stream_counts": {"fsm50_telemetry": int(fsm50_count)},
        "canonical_files": canonical_files,
        "journal": {
            "committed_checkpoint_count": 1,
            "final_cursors": {"fsm50_telemetry": int(fsm50_count)},
            "removed_after_success": True,
            "errors": [],
        },
        "errors": [],
    }
    marker_path = run_dir / "telemetry_finalization.json"
    _write_json(marker_path, marker)
    return {
        **marker,
        "marker_path": str(marker_path.resolve()),
        "marker_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
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


def _refresh_worker_complete_ack(artifact_root: Path, run_dir: Path) -> None:
    batch_root = artifact_root.parent.parent
    ack_path = batch_root / "worker_artifact_complete_ack.json"
    if not ack_path.is_file():
        return
    result_path = run_dir / "result.json"
    ack = json.loads(ack_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    digest_paths = {
        "result_sha256": result_path,
        "artifact_pointer_sha256": artifact_root / "artifact_pointer.json",
        "checksums_sha256": run_dir / "checksums.sha256",
        "telemetry_finalization_sha256": run_dir / "telemetry_finalization.json",
        "visual_manifest_sha256": run_dir / "visual_recording_manifest.json",
    }
    ack.update(
        {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in digest_paths.items()
            if path.is_file()
        }
    )
    for key in (
        "artifact_valid",
        "classification",
        "scheduler_complete",
        "physical_success",
        "strict_full_success",
    ):
        ack[key] = result.get(key)
    _write_json(ack_path, ack)


def _reseal_batch(
    artifact_root: Path,
    run_dir: Path,
    *,
    sync_result: bool = False,
    refresh_telemetry: bool = True,
) -> None:
    batch_root = artifact_root.parent.parent
    if refresh_telemetry and _refresh_telemetry_finalization(run_dir):
        sync_result = True
    if sync_result:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        _write_json(batch_root / "batch_results.json", [result])
        _write_json(batch_root / "batch_results.preclose.json", [result])
    _refresh_worker_complete_ack(artifact_root, run_dir)
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
    stable_paths.extend(
        batch_root / name
        for name in WORKER_CONTROL_FILENAMES
        if (batch_root / name).is_file()
    )
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


def _refresh_telemetry_finalization(run_dir: Path) -> bool:
    marker_path = run_dir / "telemetry_finalization.json"
    result_path = run_dir / "result.json"
    if not marker_path.is_file() or not result_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    changed = False
    for name, row in dict(marker.get("canonical_files", {}) or {}).items():
        candidate = run_dir / name
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if row.get("size_bytes") != size or row.get("sha256") != digest:
            row["size_bytes"] = size
            row["sha256"] = digest
            changed = True
    if not changed:
        return False
    _write_json(marker_path, marker)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["telemetry_finalization"] = {
        **marker,
        "marker_path": str(marker_path.resolve()),
        "marker_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
    }
    _write_json(result_path, result)
    _rewrite_checksums(run_dir)
    return True


def _replace_telemetry_finalization(
    run_dir: Path,
    marker: dict,
) -> None:
    marker_path = run_dir / "telemetry_finalization.json"
    _write_json(marker_path, marker)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["telemetry_finalization"] = {
        **marker,
        "marker_path": str(marker_path.resolve()),
        "marker_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
    }
    _write_json(result_path, result)
    _rewrite_checksums(run_dir)


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
                    "filtered_contact_consistency_valid": True,
                    "filtered_contact_consistency_error": "",
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
    worker_owned: bool = True,
    diagnostic_role: str | None = None,
):
    if diagnostic_role is None:
        diagnostic_role = role if role in {"A1", "A2", "B"} else ""
    if worker_owned and not diagnostic_role:
        instrumented = True
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
    trial_id = 1
    worker_pid = 12001
    worker_session_id = f"worker-session-{role}"
    adapter_runtime_instance_id = f"adapter-instance-{role}"
    request_id = f"request-{role}"
    plan_id = f"plan-{role}"
    worker_request_path = batch_root / "worker_artifact_request.json"
    worker_request_sha = ""
    if worker_owned:
        worker_request = {
            "schema_version": "fsm50.worker_recording_gate_request.v1",
            "enabled": True,
            "artifact_owner": "sim_worker_process",
            "request_id": request_id,
            "plan_id": plan_id,
            "plan_sha256": plan_sha,
            "expected_plan_sha256": plan_sha,
            "plan_event_count": 3,
            "plan_segment_count": 2,
            "source_version": source_version,
            "trial_id": trial_id,
            "contact_mode": contact_mode,
            "height_mm": 50,
            "artifact_root": str(artifact_root.resolve()),
            "accepted_steps_path": str(accepted_steps_path.resolve()),
            "accepted_steps_sha256": recording_sha,
            "expected_accepted_steps_sha256": recording_sha,
            "expected_root_state_write_count": 0,
        }
        worker_request["environment_equivalence_role"] = diagnostic_role
        _write_json(worker_request_path, worker_request)
        worker_request_sha = hashlib.sha256(worker_request_path.read_bytes()).hexdigest()
        (run_dir / "input" / "worker_artifact_request.json").write_bytes(
            worker_request_path.read_bytes()
        )

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
    (run_dir / "telemetry_samples.csv").write_text(
        "time_s\n" + "".join(f"{row['time_s']}\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "body_com_timeseries.csv").write_text("time_s\n", encoding="utf-8")
    (run_dir / "joint_timeseries.csv").write_text("time_s\n", encoding="utf-8")
    (run_dir / "contacts.csv").write_text("time_s\n", encoding="utf-8")
    _write_jsonl(run_dir / "events.jsonl", [{"event_type": "finished"}])
    _write_json(run_dir / "stability_summary.json", {"sample_count": len(rows)})
    (run_dir / "fsm50_telemetry.csv").write_text(
        "time_s\n" + "".join(f"{row['time_s']}\n" for row in rows),
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "wheel_filtered_contacts.jsonl", [])
    (run_dir / "nonwheel_obstacle_contacts.csv").write_text(
        "time_s\n", encoding="utf-8"
    )
    _write_jsonl(run_dir / "nonwheel_obstacle_contacts.jsonl", [])
    (run_dir / "state_timeline.csv").write_text(
        "start_time_s,end_time_s\n0.0,0.0\n", encoding="utf-8"
    )
    telemetry_finalization = _write_telemetry_finalization(
        run_dir,
        {
            "telemetry_samples.csv": len(rows),
            "body_com_timeseries.csv": 0,
            "joint_timeseries.csv": 0,
            "contacts.csv": 0,
            "events.jsonl": 1,
            "stability_summary.json": 1,
            "fsm50_telemetry.csv": len(rows),
            "fsm50_telemetry.jsonl": len(rows),
            "wheel_filtered_contacts.jsonl": 0,
            "nonwheel_obstacle_contacts.csv": 0,
            "nonwheel_obstacle_contacts.jsonl": 0,
            "state_timeline.csv": 1,
            "physical_evidence.json": 1,
        },
        fsm50_count=len(rows),
    )

    video_path = run_dir / "actual_viewport_video.mp4"
    video_path.write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41"
    )
    render_product_path = "/Render/Viewport"
    viewport_identity = 12345
    ledger_path = run_dir / "viewport_frame_ledger.jsonl"
    ledger_rows = [
        {
            "render_sequence": index,
            "sim_step": 8 * (index + 1),
            "sim_time_s": 8 * (index + 1) / 120.0,
            "encoded_frame_index": index,
            "width": 2,
            "height": 1,
            "rgba_buffer_size": 8,
            "byte_format": "TextureFormat.RGBA8_UNORM",
            "capture_backend": (
                "active_viewport_ldr_byte_buffer_to_omni_videoencoding"
            ),
            "render_product_path": render_product_path,
            "viewport_identity": viewport_identity,
        }
        for index in range(3)
    ]
    _write_jsonl(ledger_path, ledger_rows)
    first_frame_path = run_dir / "viewport_first_frame.png"
    last_frame_path = run_dir / "viewport_last_frame.png"
    first_frame_path.write_bytes(b"\x89PNG\r\n\x1a\nfirst-frame")
    last_frame_path.write_bytes(b"\x89PNG\r\n\x1a\nlast-frame")
    direct_viewport = {
        "valid": True,
        "artifact_valid": True,
        "actual_viewport_video": True,
        "not_camera_video": False,
        "capture_backend": (
            "active_viewport_ldr_byte_buffer_to_omni_videoencoding"
        ),
        "source": "actual_active_isaac_gui_viewport_render_product",
        "render_product_path": render_product_path,
        "viewport_identity": viewport_identity,
        "viewport_identity_check_count": 7,
        "render_product_unchanged": True,
        "active_render_product_identity_proven": True,
        "capture_graph_created": False,
        "extra_app_update_count": 0,
        "extra_render_count": 0,
        "render_observer_only": True,
        "maximum_pending_captures": 1,
        "fps": 15.0,
        "frame_count": 3,
        "frame_ledger_complete": True,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "video_path": str(video_path.resolve()),
        "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "video_size": video_path.stat().st_size,
        "first_frame_path": str(first_frame_path.resolve()),
        "first_frame_sha256": hashlib.sha256(
            first_frame_path.read_bytes()
        ).hexdigest(),
        "last_frame_path": str(last_frame_path.resolve()),
        "last_frame_sha256": hashlib.sha256(
            last_frame_path.read_bytes()
        ).hexdigest(),
        "full_decode": {
            "valid": True,
            "decoded_frame_count": 3,
            "decoded_width": 2,
            "decoded_height": 1,
            "decoded_channels": 3,
        },
        "full_decode_all_frames": True,
        "error": "",
        "checkpoint_error": "",
    }
    buffer_manifest_path = run_dir / "viewport_buffer_video_manifest.json"
    _write_json(
        buffer_manifest_path,
        {
            "schema_version": "fsm50.active_viewport_buffer_video.v1",
            **direct_viewport,
        },
    )
    viewport_manifest_path = run_dir / "viewport_video_manifest.json"
    viewport_manifest = {
        **direct_viewport,
        "schema_version": "fsm50.recording_viewport_video.v1",
        "contact_mode": contact_mode,
        "capture_requested": True,
        "diagnostic_only": False,
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
    dispatch_ledger = None
    if diagnostic_role:
        readiness_common = {
            "schema_version": "fsm50.motion_start_readiness_window.v1",
            "ready": True,
            "status": "PASS",
            "gate": "MOTION_START_READY",
            "window_failed_checks": [],
            "root_state_write_count": 0,
            "writes_robot_state": False,
            "command_dispatch_idle": True,
            "adapter_runtime_instance_id": adapter_runtime_instance_id,
        }
        _write_json(run_dir / "motion_start_readiness.json", readiness_common)
        _write_json(
            run_dir / "motion_start_pre_first_dispatch.json",
            {
                **readiness_common,
                "readiness_token_bound": True,
                "source_command_dispatch_count": 0,
                "boundary_batch_is_not_source_command": True,
            },
        )
        dispatch_rows = [
            {"global_command_index": index, "dispatch_command": True}
            for index in range(1, 4)
        ]
        dispatch_batches = [
            {
                "segment_index": index,
                "dispatch_kind": "source_segment_start",
                "ack_present": True,
                "ack_valid": True,
            }
            for index in range(2)
        ]
        dispatch_common = {
            "schema_version": "fsm50.source_dispatch_ledger.v1",
            "complete": True,
            "errors": [],
            "source_version": source_version,
            "plan_event_count": 3,
            "retained_plan_event_count": 3,
            "live_timing_command_count": 3,
            "plan_segment_count": 2,
            "primary_motion_batch_count": 2,
            "runtime_generated_motion_batch_count": 0,
            "one_motion_batch_per_physics_tick": True,
            "motion_start_readiness_token_count": 1,
            "playback_start_boundary_count": 1,
            "final_safety_stop_count": 1,
        }
        dispatch_json_path = run_dir / "V003_DISPATCH_TRACE.json"
        dispatch_csv_path = run_dir / "V003_DISPATCH_TRACE.csv"
        _write_json(
            dispatch_json_path,
            {
                **dispatch_common,
                "rows": dispatch_rows,
                "motion_batches": dispatch_batches,
            },
        )
        dispatch_csv_path.write_text(
            "global_command_index,dispatch_command\n1,true\n2,true\n3,true\n",
            encoding="utf-8",
        )
        _write_json(
            run_dir / "production_dispatch_timing.json",
            {"commands": dispatch_rows},
        )
        dispatch_ledger = {
            **dispatch_common,
            "json_path": str(dispatch_json_path.resolve()),
            "csv_path": str(dispatch_csv_path.resolve()),
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
        "telemetry_finalization": telemetry_finalization,
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
    if worker_owned:
        result.update(
            {
                "artifact_owner": "sim_worker_process",
                "execution_path": "sim_worker_process_ipc",
                "artifact_request_sha256": worker_request_sha,
                "request_id": request_id,
                "plan_id": plan_id,
                "trial_id": trial_id,
                "contact_mode": contact_mode,
                "worker_pid": worker_pid,
                "worker_session_id": worker_session_id,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "root_state_write_count": 0,
                "physical_success": False,
                "respawn": {
                    "ok": True,
                    "respawned": False,
                    "root_pose_written": False,
                    "adapter_runtime_instance_id": adapter_runtime_instance_id,
                    "root_state_write_count": 0,
                },
                "required_evidence_paths": [
                    str((run_dir / name).resolve())
                    for name in (
                        "result.json",
                        "failure_diagnostics.json",
                        "physical_evidence.json",
                        "telemetry_finalization.json",
                        "fsm50_telemetry.jsonl",
                        "runtime_environment.json",
                        "visual_recording_manifest.json",
                        "input/worker_artifact_request.json",
                        "checksums.sha256",
                    )
                ],
            }
        )
        result.update(
            {
                "environment_equivalence_role": diagnostic_role,
                "environment_equivalence_diagnostic": bool(diagnostic_role),
                "environment_equivalence_diagnostic_complete": bool(
                    diagnostic_role
                ),
                "qualification_scope": (
                    "TRAJECTORY_COMPARISON"
                    if diagnostic_role
                    else "GATE1_PHYSICAL_QUALIFICATION"
                ),
            }
        )
        if diagnostic_role:
            result.update(
                {"dispatch_complete": True, "dispatch_ledger": dispatch_ledger}
            )
            result["required_evidence_paths"].extend(
                str((run_dir / name).resolve())
                for name in (
                    "motion_start_readiness.json",
                    "motion_start_pre_first_dispatch.json",
                    "V003_DISPATCH_TRACE.csv",
                    "V003_DISPATCH_TRACE.json",
                    "production_dispatch_timing.json",
                    "worker_recording_session.json",
                )
            )
    _write_json(run_dir / "result.json", result)
    failure_diagnostics = {
        "classification": "PHYSICAL_FAILURE",
        "artifact_valid": True,
    }
    if diagnostic_role:
        failure_diagnostics.update(
            {
                "contact_mode": contact_mode,
                "environment_equivalence_role": diagnostic_role,
                "environment_equivalence_diagnostic_complete": True,
            }
        )
    _write_json(
        run_dir / "failure_diagnostics.json",
        failure_diagnostics,
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
    if diagnostic_role:
        runtime["environment_equivalence_role"] = diagnostic_role
    _write_json(run_dir / "runtime_environment.json", runtime)
    if diagnostic_role:
        _write_json(
            run_dir / "worker_recording_session.json",
            {
                "finalization_complete": True,
                "artifact_valid": True,
                "scheduler_complete": True,
                "contact_mode": contact_mode,
                "environment_equivalence_role": diagnostic_role,
                "environment_equivalence_diagnostic_complete": True,
            },
        )
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

    batch_request = {
        "created_utc": "2026-08-09T00:00:00+00:00",
        "versions": [source_version],
    }
    if worker_owned:
        batch_request.update(
            {
                "schema_version": "fsm50.formal_worker_recording_batch.v1",
                "trial_id": trial_id,
                "artifact_request_path": str(worker_request_path.resolve()),
                "artifact_request_sha256": worker_request_sha,
                "plan_id": plan_id,
                "plan_sha256": plan_sha,
                "args": {"contact_mode": contact_mode},
                "prelaunch_environment_validation": {"ok": True},
            }
        )
        batch_request.update(
            {
                "environment_equivalence_role": diagnostic_role,
                "qualification_scope": (
                    "TRAJECTORY_COMPARISON"
                    if diagnostic_role
                    else "GATE1_PHYSICAL_QUALIFICATION"
                ),
            }
        )
        batch_request["args"]["environment_equivalence_role"] = diagnostic_role
    _write_json(batch_root / "batch_request.json", batch_request)
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
    if worker_owned:
        startup_status = {
            "ready": True,
            "artifact_preflight_ready": True,
            "worker_pid": worker_pid,
            "worker_session_id": worker_session_id,
            "worker_artifact_session": {
                "request_id": request_id,
                "source_version": source_version,
                "trial_id": trial_id,
                "contact_mode": contact_mode,
                "artifact_root": str(artifact_root.resolve()),
                "worker_session_id": worker_session_id,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "root_state_write_count": 0,
                "motion_start_ready": True,
                "readiness_frame_count": 10,
                "readiness_frame_count_required": 10,
                "readiness_sample_stride_physics_ticks": 8,
                "environment_equivalence": {"ok": True, "failed_checks": []},
                "contact_sensor_type": "TestContactBank",
                "contact_sensor_error": "",
            },
            "worker_artifact_preflight": {
                "artifact_request_sha256": worker_request_sha,
                "expected_plan_sha256": plan_sha,
            },
        }
        startup_status["worker_artifact_session"][
            "environment_equivalence_role"
        ] = diagnostic_role
        startup_status["worker_artifact_preflight"][
            "environment_equivalence_role"
        ] = diagnostic_role
        _write_json(
            batch_root / "worker_startup_binding.json",
            {
                "worker_pid": worker_pid,
                "worker_session_id": worker_session_id,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "artifact_request_id": request_id,
                "artifact_request_sha256": worker_request_sha,
                "status": startup_status,
            },
        )
        _write_json(
            batch_root / "worker_preplay_stop_ack.json",
            {
                "type": "stop_ack",
                "command_id": f"stop-{role}",
                "reason": "playback_start_boundary",
                "worker_pid": worker_pid,
                "worker_session_id": worker_session_id,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "artifact_request_id": request_id,
                "root_state_write_count": 0,
                "received_wall_time": 100.0,
                "target_applied_wall_time": 101.0,
                "target_applied_sim_time": 2.0,
                "zero_target_applied": True,
                "error": "",
            },
        )
        _write_json(
            batch_root / "worker_playback_start_ack.json",
            {
                "type": "operation_ack",
                "operation": "start_playback_plan",
                "request_id": request_id,
                "accepted": True,
                "rejection_reason": "",
                "error": "",
                "plan_id": plan_id,
                "plan_sha256": plan_sha,
                "event_count": 3,
                "segment_count": 2,
                "worker_pid": worker_pid,
                "worker_session_id": worker_session_id,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "artifact_request_id": request_id,
                "root_state_write_count": 0,
                "accepted_wall_time": 102.0,
                "motion_start_ready": True,
            },
        )
        result_path = run_dir / "result.json"
        pointer_path = artifact_root / "artifact_pointer.json"
        run_checksums_path = run_dir / "checksums.sha256"
        telemetry_finalization_path = run_dir / "telemetry_finalization.json"
        visual_manifest_path = run_dir / "visual_recording_manifest.json"
        _write_json(
            batch_root / "worker_artifact_complete_ack.json",
            {
                "type": "operation_ack",
                "operation": "recording_artifact",
                "phase": "ARTIFACT_COMPLETE",
                "artifact_owner": "sim_worker_process",
                "accepted": True,
                "artifact_complete": True,
                "request_id": request_id,
                "artifact_request_id": request_id,
                "plan_id": plan_id,
                "plan_sha256": plan_sha,
                "artifact_request_sha256": worker_request_sha,
                "worker_pid": worker_pid,
                "worker_session_id": worker_session_id,
                "source_version": source_version,
                "trial_id": trial_id,
                "contact_mode": contact_mode,
                "accepted_steps_sha256": recording_sha,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "root_state_write_count": 0,
                "artifact_root": str(artifact_root.resolve()),
                "run_dir": str(run_dir.resolve()),
                "result_path": str(result_path.resolve()),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "artifact_pointer_path": str(pointer_path.resolve()),
                "artifact_pointer_sha256": hashlib.sha256(pointer_path.read_bytes()).hexdigest(),
                "checksums_path": str(run_checksums_path.resolve()),
                "checksums_sha256": hashlib.sha256(run_checksums_path.read_bytes()).hexdigest(),
                "telemetry_finalization_path": str(telemetry_finalization_path.resolve()),
                "telemetry_finalization_sha256": hashlib.sha256(
                    telemetry_finalization_path.read_bytes()
                ).hexdigest(),
                "visual_manifest_path": str(visual_manifest_path.resolve()),
                "visual_manifest_sha256": hashlib.sha256(
                    visual_manifest_path.read_bytes()
                ).hexdigest(),
                "artifact_valid": result["artifact_valid"],
                "classification": result["classification"],
                "scheduler_complete": result["scheduler_complete"],
                "physical_success": result["physical_success"],
                "strict_full_success": result["strict_full_success"],
                "error": "",
            },
        )
        complete_path = batch_root / "worker_artifact_complete_ack.json"
        complete_ack = json.loads(complete_path.read_text(encoding="utf-8"))
        complete_ack.update(
            {
                "environment_equivalence_role": diagnostic_role,
                "environment_equivalence_diagnostic_complete": bool(
                    diagnostic_role
                ),
            }
        )
        _write_json(complete_path, complete_ack)
    preclose_finalization = {
        "artifact_root": str(batch_root.resolve()),
        "finalized": True,
        "failed": False,
        "strict_success": False,
        "environment_equivalence_role": diagnostic_role if worker_owned else "",
        "environment_equivalence_diagnostic_complete": bool(
            worker_owned and diagnostic_role
        ),
        "command_success": False,
        "qualification_scope": (
            "TRAJECTORY_COMPARISON"
            if worker_owned and diagnostic_role
            else "GATE1_PHYSICAL_QUALIFICATION"
        ),
        "batch_error": "",
        "close_error": (
            "PENDING_FORMAL_WORKER_CLOSE"
            if worker_owned
            else "PENDING_SIMULATION_CLOSE"
        ),
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
    if worker_owned:
        stable_preclose_paths.extend(
            batch_root / name for name in WORKER_CONTROL_FILENAMES
        )
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
    if worker_owned:
        shutdown_request_id = f"shutdown-{role}"
        fast_close_kwargs = {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        }
        runtime_version = "5.1.0-test"
        raw_common = {
            "mode": "fast",
            "accepted": True,
            "error": "",
            "request_id": shutdown_request_id,
            "worker_pid": worker_pid,
            "worker_session_id": worker_session_id,
            "adapter_runtime_instance_id": adapter_runtime_instance_id,
            "artifact_request_id": request_id,
            "root_state_write_count": 0,
            "close_kwargs": fast_close_kwargs,
            "runtime_version": runtime_version,
        }
        shutdown_outcome = {
            "schema_version": "fsm50.shutdown_outcome.v1",
            "created_utc": "2026-08-09T00:00:03+00:00",
            "status": "FAST_EXIT_VERIFIED",
            "parent_pid": 100,
            "child_pid": 101,
            "child_returncode": 0,
            "preclose_observed": True,
            "preclose_verification": {
                "ok": True,
                "formal_worker_identity": {
                    "worker_pid": worker_pid,
                    "worker_session_id": worker_session_id,
                    "adapter_runtime_instance_id": adapter_runtime_instance_id,
                    "artifact_request_id": request_id,
                    "artifact_request_sha256": worker_request_sha,
                },
            },
            "handshake_state": "FAST_WORKER_PROCESS_RETURNED",
            "shutdown_mode": "fast",
            "close_kwargs": fast_close_kwargs,
            "intended_returncode": 0,
            "process_returned_normally": True,
            "runtime_version": runtime_version,
            "close_error": "",
            "formal_worker_pid": worker_pid,
            "formal_worker_session_id": worker_session_id,
            "adapter_runtime_instance_id": adapter_runtime_instance_id,
            "artifact_request_id": request_id,
            "artifact_request_sha256": worker_request_sha,
            "worker_returncode": 0,
            "worker_process_returned_normally": True,
            "worker_shutdown_accepted": True,
            "worker_close_requested": True,
            "worker_close_returned": False,
            "worker_forced_termination": False,
            "worker_shutdown_request_id": shutdown_request_id,
            "worker_shutdown_ack": {
                "type": "operation_ack",
                "operation": "shutdown",
                **raw_common,
            },
            "worker_close_requested_ack": {
                "type": "close_requested",
                **raw_common,
            },
            "worker_close_returned_ack": {},
        }
    else:
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


def _make_diagnostic_run(root: Path, role: str, **kwargs):
    return _make_run(
        root,
        role,
        diagnostic_role=role,
        instrumented=role == "B",
        **kwargs,
    )


def test_sealed_trajectory_diagnostic_loads_physical_failure_without_claim(
    tmp_path: Path,
) -> None:
    artifact_root, _run_dir = _make_diagnostic_run(tmp_path, "A1")

    artifact = load_sealed_trajectory_diagnostic_artifact(
        artifact_root, role="A1"
    )

    assert artifact.result["physical_success"] is False
    assert artifact.result["strict_full_success"] is False
    assert artifact.provenance["diagnostic_scope"] == "TRAJECTORY_DIAGNOSTIC_ONLY"
    assert artifact.provenance["physical_claim"] == "NO_PHYSICAL_CLAIM"
    assert artifact.provenance["qualification_scope"] == "TRAJECTORY_COMPARISON"
    assert artifact.provenance["dispatch_evidence"]["dispatch_complete"] is True
    assert artifact.provenance["observed_physical_outcome"] == {
        "classification": "PHYSICAL_FAILURE",
        "physical_success": False,
        "strict_full_success": False,
        "used_for_diagnostic_admission": False,
    }


def test_sealed_trajectory_diagnostic_triplet_compares_self_error_without_physical_claim(
    tmp_path: Path,
) -> None:
    a1_root, _ = _make_diagnostic_run(tmp_path, "A1")
    a2_root, _ = _make_diagnostic_run(tmp_path, "A2")
    b_root, _ = _make_diagnostic_run(tmp_path, "B")

    report = compare_sealed_trajectory_diagnostics(
        baseline_a1=a1_root,
        baseline_a2=a2_root,
        instrumented_b=b_root,
    )

    assert report["ok"] is True
    assert report["diagnostic_scope"] == "TRAJECTORY_DIAGNOSTIC_ONLY"
    assert report["physical_claim"] == "NO_PHYSICAL_CLAIM"
    assert report["qualification_eligible"] is False
    assert report["artifact_admission"]["ok"] is True
    assert report["sensor_independent_trajectory_ok"] is True
    assert report["observation_metrics_ok"] is True
    assert all(
        row["observed_physical_outcome"]["physical_success"] is False
        for row in report["artifact_admission"]["roles"].values()
    )
    contact = report["trajectory_comparison"]["metrics"]["contact_class"]
    assert "observation-semantics-sensitive" in contact["interpretation"]
    assert "not a dynamics or physical-success claim" in contact["interpretation"]
    force = report["trajectory_comparison"]["metrics"]["contact_force"]
    assert "observation-semantics-sensitive" in force["interpretation"]
    assert report["trajectory_comparison"]["metric_groups"] == {
        "sensor_independent_trajectory": [
            "root_trajectory",
            "joint_trajectory",
            "wheel_rotation",
            "wheel_travel",
            "final_pose",
            "obstacle_geometry",
        ],
        "observation_sensitive": ["contact_class", "contact_force"],
    }


def test_sealed_trajectory_diagnostic_separates_common_trajectory_from_contact_semantics(
    tmp_path: Path,
) -> None:
    a1_root, _ = _make_diagnostic_run(tmp_path, "A1")
    a2_root, _ = _make_diagnostic_run(tmp_path, "A2")
    b_root, _ = _make_diagnostic_run(tmp_path, "B", contact="TOP")

    report = compare_sealed_trajectory_diagnostics(
        baseline_a1=a1_root,
        baseline_a2=a2_root,
        instrumented_b=b_root,
    )

    assert report["ok"] is False
    assert report["artifact_admission"]["ok"] is True
    assert report["sensor_independent_trajectory_ok"] is True
    assert report["observation_metrics_ok"] is False
    assert report["trajectory_comparison"]["failed_metrics"] == ["contact_class"]


def test_sealed_trajectory_diagnostic_rejects_gate1_artifact_without_role(
    tmp_path: Path,
) -> None:
    artifact_root, _ = _make_run(
        tmp_path,
        "Gate1",
        instrumented=True,
        diagnostic_role="",
    )
    load_completed_replay_artifact(artifact_root, role="Gate1")

    with pytest.raises(
        ArtifactValidationError,
        match="environment_equivalence_role",
    ):
        load_sealed_trajectory_diagnostic_artifact(artifact_root, role="B")


@pytest.mark.parametrize(
    ("target", "mutator", "message"),
    (
        (
            "result",
                lambda payload: payload.update(
                    {"environment_equivalence_diagnostic_complete": False}
                ),
                "artifact-complete ACK identity mismatch",
        ),
        (
            "batch_request",
                lambda payload: payload.update({"qualification_scope": "GATE1_PHYSICAL_QUALIFICATION"}),
                "batch_request diagnostic role/scope",
        ),
        (
            "startup_preflight",
                lambda payload: payload.update({"environment_equivalence_role": "A2"}),
                "startup diagnostic role",
        ),
        (
            "dispatch",
            lambda payload: payload.update({"complete": False}),
            "dispatch ledger",
        ),
    ),
)
def test_sealed_trajectory_diagnostic_role_scope_and_dispatch_tamper_fail_closed(
    tmp_path: Path,
    target: str,
    mutator,
    message: str,
) -> None:
    artifact_root, run_dir = _make_diagnostic_run(tmp_path, "A1")
    batch_root = artifact_root.parent.parent
    paths = {
        "result": run_dir / "result.json",
        "batch_request": batch_root / "batch_request.json",
        "startup_preflight": batch_root / "worker_startup_binding.json",
        "dispatch": run_dir / "V003_DISPATCH_TRACE.json",
    }
    path = paths[target]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if target == "startup_preflight":
        payload = payload["status"]["worker_artifact_preflight"]
        mutator(payload)
        startup = json.loads(path.read_text(encoding="utf-8"))
        startup["status"]["worker_artifact_preflight"] = payload
        payload = startup
    else:
        mutator(payload)
    _write_json(path, payload)
    sync_result = target == "result"
    if target == "dispatch":
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["dispatch_ledger"]["complete"] = False
        _write_json(result_path, result)
        sync_result = True
    if sync_result:
        _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        sync_result=sync_result,
    )

    with pytest.raises(ArtifactValidationError, match=message):
        load_sealed_trajectory_diagnostic_artifact(artifact_root, role="A1")


def test_sealed_trajectory_diagnostic_comparator_failure_retains_no_claim_markers(
    tmp_path: Path,
) -> None:
    a1_root, _ = _make_diagnostic_run(tmp_path, "A1")
    a2_root, _ = _make_diagnostic_run(tmp_path, "A2")
    b_root, b_run = _make_diagnostic_run(tmp_path, "B")
    (b_run / "actual_viewport_video.mp4").write_bytes(b"tampered")

    report = compare_sealed_trajectory_diagnostics(
        baseline_a1=a1_root,
        baseline_a2=a2_root,
        instrumented_b=b_root,
    )

    assert report["ok"] is False
    assert report["fail_closed"] is True
    assert report["diagnostic_scope"] == "TRAJECTORY_DIAGNOSTIC_ONLY"
    assert report["physical_claim"] == "NO_PHYSICAL_CLAIM"
    assert report["qualification_eligible"] is False
    assert report["artifact_admission"]["ok"] is False


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
    assert a1.provenance["batch_shutdown_status"] == "FAST_EXIT_VERIFIED"
    assert a1.provenance["batch_finalization_phase"] == "SHUTDOWN_COMPLETE"
    assert Path(a1.provenance["batch_root"]).is_dir()
    assert len(a1.provenance["batch_shutdown_closure"]["closure_sha256"]) == 64
    viewport = a1.provenance["viewport_video"]
    assert viewport["capture_backend"] == (
        "active_viewport_ldr_byte_buffer_to_omni_videoencoding"
    )
    assert viewport["render_product_unchanged"] is True
    assert viewport["active_render_product_identity_proven"] is True
    assert viewport["capture_graph_created"] is False
    assert viewport["full_decode_all_frames"] is True
    assert viewport["full_decode"]["decoded_frame_count"] == viewport["frame_count"]
    for name in (
        "viewport_buffer_video_manifest.json",
        "viewport_frame_ledger.jsonl",
        "viewport_first_frame.png",
        "viewport_last_frame.png",
    ):
        assert name in a1.provenance["artifact_hashes"]

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
    fake_video_path = fake_video_run / "actual_viewport_video.mp4"
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
    _reseal_batch(missing_root, missing_run, refresh_telemetry=False)
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
    _reseal_batch(
        incomplete_root,
        incomplete_run,
        sync_result=True,
        refresh_telemetry=False,
    )
    with pytest.raises(ArtifactValidationError, match="lifecycle"):
        load_completed_replay_artifact(incomplete_root, role="incomplete")


@pytest.mark.parametrize(
    ("filename", "message"),
    (
        (
            "viewport_buffer_video_manifest.json",
            "required direct viewport buffer manifest is missing",
        ),
        ("viewport_frame_ledger.jsonl", "viewport ledger is missing or empty"),
        ("viewport_first_frame.png", "viewport first_frame is missing or empty"),
        ("viewport_last_frame.png", "viewport last_frame is missing or empty"),
    ),
)
def test_direct_viewport_required_file_deletion_fails_closed(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path / filename, "A1")
    (run_dir / filename).unlink()
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        refresh_telemetry=False,
    )

    with pytest.raises(ArtifactValidationError, match=message):
        load_completed_replay_artifact(artifact_root, role="A1")


@pytest.mark.parametrize(
    ("filename", "message"),
    (
        ("viewport_buffer_video_manifest.json", "required file is not covered"),
        ("viewport_frame_ledger.jsonl", "required file is not covered"),
        ("viewport_first_frame.png", "required file is not covered"),
        ("viewport_last_frame.png", "required file is not covered"),
    ),
)
def test_direct_viewport_files_are_required_checksum_entries(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path / filename, "A1")
    checksum_path = run_dir / "checksums.sha256"
    retained = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if not line.endswith(f"  {filename}")
    ]
    checksum_path.write_text("\n".join(retained) + "\n", encoding="utf-8")
    _reseal_batch(
        artifact_root,
        run_dir,
        refresh_telemetry=False,
    )

    with pytest.raises(ArtifactValidationError, match=message):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_direct_viewport_buffer_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    buffer_manifest_path = run_dir / "viewport_buffer_video_manifest.json"
    buffer_manifest = json.loads(buffer_manifest_path.read_text(encoding="utf-8"))
    buffer_manifest["capture_backend"] = "legacy_png_movie_capture"
    _write_json(buffer_manifest_path, buffer_manifest)
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        refresh_telemetry=False,
    )

    with pytest.raises(ArtifactValidationError, match="capture_backend"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_direct_viewport_ledger_semantic_tamper_fails_closed(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    ledger_path = run_dir / "viewport_frame_ledger.jsonl"
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["encoded_frame_index"] = 0
    _write_jsonl(ledger_path, rows)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    for name in (
        "viewport_buffer_video_manifest.json",
        "viewport_video_manifest.json",
    ):
        path = run_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ledger_sha256"] = ledger_sha
        _write_json(path, payload)
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        refresh_telemetry=False,
    )

    with pytest.raises(ArtifactValidationError, match="encoded_frame_index"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_direct_viewport_png_hash_tamper_fails_closed(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    (run_dir / "viewport_first_frame.png").write_bytes(
        b"\x89PNG\r\n\x1a\ntampered-frame"
    )
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        refresh_telemetry=False,
    )

    with pytest.raises(ArtifactValidationError, match="first_frame SHA-256 mismatch"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_telemetry_finalization_missing_incomplete_hash_and_count_fail_closed(
    tmp_path,
):
    missing_root, missing_run = _make_run(tmp_path / "missing", "missing_marker")
    (missing_run / "telemetry_finalization.json").unlink()
    _rewrite_checksums(missing_run)
    _reseal_batch(missing_root, missing_run)
    with pytest.raises(ArtifactValidationError, match="required artifact is missing"):
        load_completed_replay_artifact(missing_root, role="missing_marker")

    incomplete_root, incomplete_run = _make_run(
        tmp_path / "incomplete", "incomplete_marker"
    )
    marker_path = incomplete_run / "telemetry_finalization.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["canonical_complete"] = False
    _replace_telemetry_finalization(incomplete_run, marker)
    _reseal_batch(incomplete_root, incomplete_run, sync_result=True)
    with pytest.raises(ArtifactValidationError, match="canonical_complete"):
        load_completed_replay_artifact(incomplete_root, role="incomplete_marker")

    hash_root, hash_run = _make_run(tmp_path / "hash", "bad_marker_hash")
    marker_path = hash_run / "telemetry_finalization.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["canonical_files"]["fsm50_telemetry.jsonl"]["sha256"] = "0" * 64
    _replace_telemetry_finalization(hash_run, marker)
    _reseal_batch(
        hash_root,
        hash_run,
        sync_result=True,
        refresh_telemetry=False,
    )
    with pytest.raises(ArtifactValidationError, match="hash/size mismatch"):
        load_completed_replay_artifact(hash_root, role="bad_marker_hash")

    count_root, count_run = _make_run(tmp_path / "count", "bad_marker_count")
    marker_path = count_run / "telemetry_finalization.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["canonical_files"]["fsm50_telemetry.jsonl"]["record_count"] += 1
    _replace_telemetry_finalization(count_run, marker)
    _reseal_batch(
        count_root,
        count_run,
        sync_result=True,
        refresh_telemetry=False,
    )
    with pytest.raises(ArtifactValidationError, match="record_count differs"):
        load_completed_replay_artifact(count_root, role="bad_marker_count")

    retained_root, retained_run = _make_run(
        tmp_path / "retained", "retained_journal"
    )
    (retained_run / ".telemetry_journal").mkdir()
    with pytest.raises(ArtifactValidationError, match="retains .telemetry_journal"):
        load_completed_replay_artifact(retained_root, role="retained_journal")

    flag_root, flag_run = _make_run(tmp_path / "flag", "bad_removed_flag")
    marker_path = flag_run / "telemetry_finalization.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["journal"]["removed_after_success"] = False
    _replace_telemetry_finalization(flag_run, marker)
    _reseal_batch(flag_root, flag_run, sync_result=True)
    with pytest.raises(ArtifactValidationError, match="journal was not removed"):
        load_completed_replay_artifact(flag_root, role="bad_removed_flag")


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
    with pytest.raises(
        ArtifactValidationError,
        match="not a verified graceful/fast exit",
    ):
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


def test_formal_worker_closure_binds_request_runtime_and_double_process_exit(
    tmp_path: Path,
) -> None:
    artifact_root, _run_dir = _make_run(
        tmp_path,
        "B",
        instrumented=True,
    )

    artifact = load_completed_replay_artifact(artifact_root, role="B")

    closure = artifact.provenance["formal_worker_closure"]
    assert artifact.provenance["artifact_owner"] == "sim_worker_process"
    assert artifact.provenance["execution_path"] == "sim_worker_process_ipc"
    assert closure["worker_pid"] == 12001
    assert closure["worker_session_id"] == "worker-session-B"
    assert closure["adapter_runtime_instance_id"] == "adapter-instance-B"
    assert closure["preplay_stop_applied_wall_time"] < closure[
        "playback_start_accepted_wall_time"
    ]
    assert artifact.provenance["batch_shutdown_status"] == "FAST_EXIT_VERIFIED"
    assert artifact.provenance["batch_shutdown_closure"][
        "shutdown_contract_kind"
    ] == "isaac_5_1_worker_fast_exit"


@pytest.mark.parametrize("filename", WORKER_CONTROL_FILENAMES)
def test_formal_worker_control_file_missing_fails_closed(
    tmp_path: Path,
    filename: str,
) -> None:
    artifact_root, _run_dir = _make_run(
        tmp_path / filename,
        "B",
        instrumented=True,
    )
    (artifact_root.parent.parent / filename).unlink()

    with pytest.raises(
        ArtifactValidationError,
        match="required formal-worker shutdown-closure artifact is missing",
    ):
        load_completed_replay_artifact(artifact_root, role="B")


def test_formal_worker_identity_cannot_be_downgraded_to_legacy(
    tmp_path: Path,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("artifact_owner")
    result.pop("execution_path")
    _write_json(result_path, result)
    _rewrite_checksums(run_dir)
    _reseal_batch(artifact_root, run_dir, sync_result=True, refresh_telemetry=False)

    with pytest.raises(ArtifactValidationError, match="cannot be downgraded"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_worker_session_source_integrity_scope_fails_closed(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["source_integrity"]["scope"] = "worker_recording_session"
    _write_json(result_path, result)
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        sync_result=True,
        refresh_telemetry=False,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="result.source_integrity does not match",
    ):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_formal_worker_stop_start_and_complete_identity_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    late_stop_root, late_stop_run = _make_run(tmp_path / "late", "A1")
    late_stop_batch = late_stop_root.parent.parent
    stop_path = late_stop_batch / "worker_preplay_stop_ack.json"
    stop_ack = json.loads(stop_path.read_text(encoding="utf-8"))
    stop_ack["target_applied_wall_time"] = 103.0
    _write_json(stop_path, stop_ack)
    _reseal_batch(late_stop_root, late_stop_run, refresh_telemetry=False)
    with pytest.raises(ArtifactValidationError, match="occurred after playback-start"):
        load_completed_replay_artifact(late_stop_root, role="A1")

    start_root, start_run = _make_run(tmp_path / "start", "A1")
    start_path = start_root.parent.parent / "worker_playback_start_ack.json"
    start_ack = json.loads(start_path.read_text(encoding="utf-8"))
    start_ack["motion_start_ready"] = False
    _write_json(start_path, start_ack)
    _reseal_batch(start_root, start_run, refresh_telemetry=False)
    with pytest.raises(ArtifactValidationError, match="playback-start ACK"):
        load_completed_replay_artifact(start_root, role="A1")

    complete_root, complete_run = _make_run(tmp_path / "complete", "A1")
    complete_path = complete_root.parent.parent / "worker_artifact_complete_ack.json"
    complete_ack = json.loads(complete_path.read_text(encoding="utf-8"))
    complete_ack["worker_pid"] += 1
    _write_json(complete_path, complete_ack)
    _reseal_batch(complete_root, complete_run, refresh_telemetry=False)
    with pytest.raises(ArtifactValidationError, match="artifact-complete ACK identity"):
        load_completed_replay_artifact(complete_root, role="A1")


def test_formal_worker_batch_results_must_be_exact_singleton(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    batch_root = artifact_root.parent.parent
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    _write_json(batch_root / "batch_results.json", [result, result])
    _write_json(batch_root / "batch_results.preclose.json", [result, result])
    _reseal_batch(artifact_root, run_dir, refresh_telemetry=False)

    with pytest.raises(ArtifactValidationError, match="equal.*durable worker result"):
        load_completed_replay_artifact(artifact_root, role="A1")


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("artifact_valid", 1),
        ("plan_event_count", 3.0),
    ),
)
def test_formal_worker_batch_result_equality_is_json_type_strict(
    tmp_path: Path,
    field: str,
    mutated: object,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path / field, "A1")
    batch_root = artifact_root.parent.parent
    durable = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    mutated_result = dict(durable)
    mutated_result[field] = mutated
    _write_json(batch_root / "batch_results.json", [mutated_result])
    _write_json(batch_root / "batch_results.preclose.json", [mutated_result])
    _reseal_batch(artifact_root, run_dir, refresh_telemetry=False)

    with pytest.raises(ArtifactValidationError, match="equal.*durable worker result"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_formal_worker_pid_must_be_an_exact_json_integer(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["worker_pid"] = 12001.0
    _write_json(result_path, result)
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        sync_result=True,
        refresh_telemetry=False,
    )

    with pytest.raises(ArtifactValidationError, match="exact JSON integer"):
        load_completed_replay_artifact(artifact_root, role="A1")


@pytest.mark.parametrize("mutation", ("missing", True, 0))
def test_formal_worker_result_requires_explicit_unstopped_app_state(
    tmp_path: Path,
    mutation: object,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path / str(mutation), "A1")
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        result.pop("simulation_app_stopped")
    else:
        result["simulation_app_stopped"] = mutation
    _write_json(result_path, result)
    _rewrite_checksums(run_dir)
    _reseal_batch(
        artifact_root,
        run_dir,
        sync_result=True,
        refresh_telemetry=False,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="result simulation_app_stopped is not false",
    ):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_formal_worker_shutdown_required_raw_ack_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    batch_root = artifact_root.parent.parent
    shutdown_path = batch_root / "shutdown_outcome.json"
    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
    shutdown["worker_close_requested_ack"]["adapter_runtime_instance_id"] = (
        "different-adapter"
    )
    _write_json(shutdown_path, shutdown)
    finalization_path = batch_root / "batch_finalization.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["shutdown_outcome"] = shutdown
    _write_json(finalization_path, finalization)
    _rewrite_checksums(batch_root)

    with pytest.raises(ArtifactValidationError, match="close_requested.*mismatch"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_formal_worker_real_close_returned_ack_is_optional_but_exact(
    tmp_path: Path,
) -> None:
    artifact_root, run_dir = _make_run(tmp_path, "A1")
    batch_root = artifact_root.parent.parent
    shutdown_path = batch_root / "shutdown_outcome.json"
    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
    common = {
        key: shutdown["worker_close_requested_ack"][key]
        for key in (
            "mode",
            "accepted",
            "error",
            "request_id",
            "worker_pid",
            "worker_session_id",
            "adapter_runtime_instance_id",
            "artifact_request_id",
            "root_state_write_count",
            "close_kwargs",
            "runtime_version",
        )
    }
    shutdown["worker_close_returned"] = True
    shutdown["worker_close_returned_ack"] = {
        "type": "close_returned",
        **common,
    }
    _write_json(shutdown_path, shutdown)
    finalization_path = batch_root / "batch_finalization.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["shutdown_outcome"] = shutdown
    _write_json(finalization_path, finalization)
    _rewrite_checksums(batch_root)

    load_completed_replay_artifact(artifact_root, role="A1")

    shutdown["worker_close_returned_ack"]["accepted"] = 1
    _write_json(shutdown_path, shutdown)
    finalization["shutdown_outcome"] = shutdown
    _write_json(finalization_path, finalization)
    _rewrite_checksums(batch_root)
    with pytest.raises(ArtifactValidationError, match="close_returned"):
        load_completed_replay_artifact(artifact_root, role="A1")


def test_legacy_direct_artifact_is_diagnostic_only_and_cannot_fill_role_b(
    tmp_path: Path,
) -> None:
    artifact_root, _run_dir = _make_run(
        tmp_path,
        "legacy",
        instrumented=True,
        worker_owned=False,
    )

    legacy = load_completed_replay_artifact(artifact_root, role="legacy")
    assert legacy.provenance["artifact_owner"] == "legacy_direct"
    assert legacy.provenance["formal_worker_closure"] is None
    assert legacy.provenance["batch_shutdown_status"] == "NORMAL_EXIT"
    with pytest.raises(ArtifactValidationError, match="diagnostic-only"):
        load_completed_replay_artifact(artifact_root, role="B")


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
