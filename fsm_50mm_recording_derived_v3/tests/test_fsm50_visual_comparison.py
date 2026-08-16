import hashlib
import json
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3.fsm50_visual_comparison import (
    BindingError,
    PHASES,
    bind_run,
    phase_spans,
    recording_phase,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_fixture(tmp_path: Path, *, corrupt_index: bool = False) -> Path:
    run = tmp_path / "run"; run.mkdir()
    video = run / "actual_viewport_video.mp4"; video.write_bytes(b"sealed-input")
    ledger = run / "viewport_frame_ledger.jsonl"
    ledger.write_text("\n".join(json.dumps({"encoded_frame_index": 2 if corrupt_index and index == 1 else index, "sim_time_s": 1.0 + index}) for index in range(2)) + "\n")
    telemetry = run / "minimal_telemetry.jsonl"
    telemetry.write_text("\n".join(json.dumps({"sim_time_s": 1.0 + index, "segment_cursor": index * 7}) for index in range(2)) + "\n")
    manifest = {"valid": True, "artifact_valid": True, "actual_viewport_video": True, "frame_ledger_complete": True, "full_decode_all_frames": True,
                "video_sha256": _sha(video), "video_size": video.stat().st_size, "ledger_sha256": _sha(ledger), "frame_count": 2, "fps": 15.0,
                "full_decode": {"valid": True, "decoded_frame_count": 2, "decoded_width": 1280, "decoded_height": 720}}
    (run / "viewport_buffer_video_manifest.json").write_text(json.dumps(manifest))
    return run


def test_recording_segment_cursor_maps_to_sealed_canonical_owners():
    assert recording_phase({"segment_cursor": 0}) == PHASES[0]
    assert recording_phase({"segment_cursor": 23}) == PHASES[1]
    assert recording_phase({"segment_cursor": 104}) == PHASES[-1]
    assert recording_phase({"segment_cursor": 112}) is None


def test_bind_run_rejects_non_contiguous_ledger_indices(tmp_path: Path):
    with pytest.raises(BindingError, match="contiguous"):
        bind_run(_bound_fixture(tmp_path, corrupt_index=True), "recording")


def test_phase_spans_rejects_reentry():
    with pytest.raises(BindingError, match="non-contiguous"):
        phase_spans((PHASES[0], PHASES[1], PHASES[0]))
