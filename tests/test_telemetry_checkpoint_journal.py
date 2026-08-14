from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fsm_50mm_recording_derived_v3.fsm50_telemetry import (
    FSM50TelemetryCollector,
)
from fsm_50mm_recording_derived_v3.support_classifier import ObstacleGeometry
from telemetry.collector import TelemetryCollector
from telemetry.config import RuntimeTelemetryConfig


def _config(root: Path) -> RuntimeTelemetryConfig:
    config = RuntimeTelemetryConfig()
    config.telemetry.output_root = str(root)
    config.telemetry.flush_interval_s = 0.1
    config.telemetry.save_csv = True
    config.telemetry.save_npz = False
    config.telemetry.save_contacts = True
    config.telemetry.save_events = True
    config.visualization.live_enabled = False
    return config


def _strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(
            f"non-finite JSON constant {value}"
        ),
    )


def _strict_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(
            line,
            parse_constant=lambda value: pytest.fail(
                f"non-finite JSON constant {value}"
            ),
        )
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_checkpoint_writes_only_new_delta_and_final_exports_once(
    tmp_path: Path,
) -> None:
    collector = TelemetryCollector(
        _config(tmp_path),
        args=SimpleNamespace(headless=True),
    )
    run_dir = collector.start_episode(sequence_label="journal")
    assert run_dir is not None
    collector.rows.extend(
        [
            {"time_s": 0.0, "value": 1.0},
            {"time_s": 0.1, "value": float("nan")},
        ]
    )

    first = collector.checkpoint()
    collector.rows.append({"time_s": 0.2, "value": 3.0})
    second = collector.checkpoint()

    assert first["streams"]["telemetry_samples"]["row_start"] == 0
    assert first["streams"]["telemetry_samples"]["row_count"] == 2
    assert second["streams"]["telemetry_samples"]["row_start"] == 2
    assert second["streams"]["telemetry_samples"]["row_count"] == 1
    first_rows = _strict_jsonl(
        Path(first["commit_path"]).parent / "telemetry_samples.jsonl"
    )
    second_rows = _strict_jsonl(
        Path(second["commit_path"]).parent / "telemetry_samples.jsonl"
    )
    assert [*first_rows, *second_rows] == [
        {"time_s": 0.0, "value": 1.0},
        {"time_s": 0.1, "value": None},
        {"time_s": 0.2, "value": 3.0},
    ]
    assert not (run_dir / "telemetry_samples.csv").exists()
    assert not (run_dir / "stability_summary.json").exists()

    collector.finish_episode(success=False, reason="diagnostic failure")
    marker_path = run_dir / "telemetry_finalization.json"
    marker_before = marker_path.read_bytes()
    marker = _strict_json(marker_path)
    assert marker["canonical_complete"] is True
    assert marker["stream_counts"]["telemetry_samples"] == 3
    assert marker["journal"]["removed_after_success"] is True
    assert not (run_dir / ".telemetry_journal").exists()
    for name, evidence in marker["canonical_files"].items():
        path = run_dir / name
        assert path.is_file()
        assert path.stat().st_size == evidence["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]

    collector.flush()
    assert marker_path.read_bytes() == marker_before


def test_failed_canonical_export_preserves_committed_journal_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = TelemetryCollector(
        _config(tmp_path),
        args=SimpleNamespace(headless=True),
    )
    run_dir = collector.start_episode(sequence_label="failed-export")
    assert run_dir is not None
    collector.rows.append({"time_s": 0.0, "value": 1.0})
    collector.checkpoint()
    attempts = 0

    def fail_export() -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("synthetic canonical failure")

    monkeypatch.setattr(collector, "_write_canonical_artifacts", fail_export)
    with pytest.raises(RuntimeError, match="synthetic canonical failure"):
        collector.finish_episode(success=False, reason="runner failed")

    marker = _strict_json(run_dir / "telemetry_finalization.json")
    assert marker["canonical_complete"] is False
    assert marker["journal"]["removed_after_success"] is False
    assert (run_dir / ".telemetry_journal").is_dir()
    journal_manifest_path = (
        run_dir / ".telemetry_journal" / "journal_checksums.json"
    )
    journal_manifest = _strict_json(journal_manifest_path)
    assert journal_manifest["committed_checkpoint_count"] >= 1
    for relative, evidence in journal_manifest["files"].items():
        path = run_dir / ".telemetry_journal" / relative
        assert path.stat().st_size == evidence["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]

    with pytest.raises(RuntimeError, match="synthetic canonical failure"):
        collector.flush()
    assert attempts == 1


def test_final_checkpoint_error_blocks_canonical_complete_and_retains_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = TelemetryCollector(
        _config(tmp_path),
        args=SimpleNamespace(headless=True),
    )
    run_dir = collector.start_episode(sequence_label="failed-final-checkpoint")
    assert run_dir is not None
    collector.rows.append({"time_s": 0.0, "value": 1.0})
    collector.checkpoint()
    collector.rows.append({"time_s": 0.1, "value": 2.0})

    def fail_final_checkpoint() -> dict[str, object]:
        raise RuntimeError("synthetic final checkpoint failure")

    monkeypatch.setattr(collector, "checkpoint", fail_final_checkpoint)
    with pytest.raises(RuntimeError, match="canonical telemetry finalization"):
        collector.flush()

    marker = _strict_json(run_dir / "telemetry_finalization.json")
    assert marker["canonical_complete"] is False
    assert marker["journal"]["removed_after_success"] is False
    assert any("final checkpoint" in error for error in marker["errors"])
    journal_manifest = _strict_json(
        run_dir / ".telemetry_journal" / "journal_checksums.json"
    )
    assert journal_manifest["committed_checkpoint_count"] == 1


def test_fsm_checkpoint_occurs_after_same_frame_extended_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = FSM50TelemetryCollector(
        _config(tmp_path),
        args=SimpleNamespace(headless=True),
        scene_handle=None,
        obstacle=ObstacleGeometry(
            front_face_x_m=0.5,
            top_z_m=0.05,
            bottom_z_m=0.0,
            rear_face_x_m=2.5,
            width_m=2.0,
        ),
        wheel_radius_m=0.05,
        source_version="v003-test",
        contact_mode="instrumented",
    )
    run_dir = collector.start_episode(sequence_label="fsm-order")
    assert run_dir is not None
    assert collector.defer_periodic_checkpoint is True

    def base_on_step(self, adapter, _dt_s):
        row = {"time_s": float(adapter.sim_time), "source_version": "v003-test"}
        self.rows.append(row)
        self.last_row = row
        if not self.defer_periodic_checkpoint:
            self._maybe_checkpoint(adapter.sim_time)

    monkeypatch.setattr(TelemetryCollector, "on_step", base_on_step)
    monkeypatch.setattr(
        collector,
        "_extend_row",
        lambda _adapter, base_row, _contacts, dt_s: {
            **base_row,
            "wheel_filtered_contacts": [],
            "primary_diagonal": "NONE",
            "support_legs": [],
            "light_support_legs": [],
            "diagonal_load_fl_rr_n": 0.0,
            "diagonal_load_fr_rl_n": 0.0,
            "two_leg_corridor_distance_m": None,
            "wheel_contact_classes": {},
        },
    )
    monkeypatch.setattr(
        collector,
        "_install_filtered_contact_sample",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(collector, "_record_timeline", lambda _row: None)

    collector.on_step(SimpleNamespace(sim_time=0.1), 1.0 / 120.0)

    commits = list(
        (run_dir / ".telemetry_journal").glob("checkpoint-*/commit.json")
    )
    assert len(commits) == 1
    commit = _strict_json(commits[0])
    assert commit["streams"]["telemetry_samples"]["row_count"] == 1
    assert commit["streams"]["fsm50_telemetry"]["row_count"] == 1
    assert "state_timeline" not in commit["streams"]
    assert len(collector.rows) == len(collector.fsm50_rows) == 1
    assert not (run_dir / "fsm50_telemetry.jsonl").exists()
