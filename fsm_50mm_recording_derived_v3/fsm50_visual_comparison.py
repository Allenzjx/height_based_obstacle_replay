"""SHA-bound, offline visual comparison for recording and Macro-FSM runs.

This module consumes already captured viewport MP4s only.  It never imports
Isaac, opens a simulator, or writes beneath either source run directory.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


PHASES = (
    "S1_APPROACH_AND_PRE_FR_SHIFT", "S2_FR_TRAVERSE", "S3_FL_TRAVERSE",
    "S6_RR_TRAVERSE", "S8_RL_COM_SHIFT_AND_TRAVERSE", "S9_FINAL_ADVANCE",
    "S10_POSTURE_RECOVERY",
)
SEGMENT_PHASES = ((0, 6, PHASES[0]), (7, 23, PHASES[1]), (24, 40, PHASES[2]),
                  (41, 56, PHASES[3]), (57, 101, PHASES[4]),
                  (102, 103, PHASES[5]), (104, 111, PHASES[6]))
DEFAULT_FFMPEG = Path(r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS (2)\FloXpress\bin\ffmpeg.exe")


class BindingError(ValueError):
    """A source artifact is not the manifest-bound input the tool requires."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise BindingError(f"invalid JSONL: {path}: {exc}") from exc
    if not rows:
        raise BindingError(f"empty JSONL: {path}")
    return rows


def recording_phase(row: dict[str, Any]) -> str | None:
    """Map sealed v003 segment cursor to its source-owned Macro-FSM phase."""
    cursor = row.get("segment_cursor")
    if not isinstance(cursor, int):
        return None
    for start, end, phase in SEGMENT_PHASES:
        if start <= cursor <= end:
            return phase
    return None


def fsm_phase(row: dict[str, Any]) -> str | None:
    state = row.get("macro_state")
    return state if state in PHASES else None


@dataclass(frozen=True)
class BoundRun:
    kind: str
    run_dir: Path
    video: Path
    video_sha256: str
    fps: float
    width: int
    height: int
    frame_times_s: tuple[float, ...]
    phases: tuple[str | None, ...]
    telemetry_path: Path
    telemetry_sha256: str
    ledger_path: Path
    ledger_sha256: str
    manifest_path: Path

    @property
    def frame_count(self) -> int:
        return len(self.frame_times_s)


def _phase_per_frame(rows: list[dict[str, Any]], frame_times: Iterable[float], phase_fn: Any) -> tuple[str | None, ...]:
    times = [row.get("sim_time_s") for row in rows]
    if any(not isinstance(value, (int, float)) for value in times) or any(b < a for a, b in zip(times, times[1:])):
        raise BindingError("telemetry sim_time_s is not numeric and nondecreasing")
    result: list[str | None] = []
    for frame_time in frame_times:
        index = max(0, bisect.bisect_right(times, frame_time) - 1)
        result.append(phase_fn(rows[index]))
    return tuple(result)


def bind_run(run_dir: Path, kind: str) -> BoundRun:
    """Strictly bind MP4, manifest, ledger, and the appropriate telemetry."""
    if kind not in {"recording", "fsm"}:
        raise ValueError("kind must be recording or fsm")
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "viewport_buffer_video_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BindingError(f"invalid manifest: {manifest_path}: {exc}") from exc
    required_true = ("valid", "artifact_valid", "actual_viewport_video", "frame_ledger_complete", "full_decode_all_frames")
    if any(manifest.get(name) is not True for name in required_true):
        raise BindingError("manifest does not certify a complete actual-viewport video")
    video = run_dir / "actual_viewport_video.mp4"
    ledger = run_dir / "viewport_frame_ledger.jsonl"
    telemetry = run_dir / ("minimal_telemetry.jsonl" if kind == "recording" else "minimal_macro_telemetry.jsonl")
    for path in (video, ledger, telemetry):
        if not path.is_file():
            raise BindingError(f"missing required artifact: {path}")
    if manifest.get("video_sha256") != sha256_file(video):
        raise BindingError("MP4 SHA-256 differs from manifest")
    if manifest.get("video_size") != video.stat().st_size:
        raise BindingError("MP4 size differs from manifest")
    if manifest.get("ledger_sha256") != sha256_file(ledger):
        raise BindingError("ledger SHA-256 differs from manifest")
    ledger_rows = _jsonl(ledger)
    expected_count = manifest.get("frame_count")
    if not isinstance(expected_count, int) or len(ledger_rows) != expected_count:
        raise BindingError("manifest frame_count and ledger row count differ")
    indexes = [row.get("encoded_frame_index") for row in ledger_rows]
    if indexes != list(range(expected_count)):
        raise BindingError("ledger encoded_frame_index is not a contiguous zero-based sequence")
    frame_times = [row.get("sim_time_s") for row in ledger_rows]
    if any(not isinstance(value, (int, float)) for value in frame_times) or any(b <= a for a, b in zip(frame_times, frame_times[1:])):
        raise BindingError("ledger sim_time_s is not strictly increasing")
    full = manifest.get("full_decode", {})
    if not full.get("valid") or full.get("decoded_frame_count") != expected_count:
        raise BindingError("manifest full-decode certificate is invalid")
    fps = manifest.get("fps")
    width, height = full.get("decoded_width"), full.get("decoded_height")
    if not isinstance(fps, (int, float)) or fps <= 0 or not isinstance(width, int) or not isinstance(height, int):
        raise BindingError("manifest lacks usable fps or decoded dimensions")
    rows = _jsonl(telemetry)
    phase_fn = recording_phase if kind == "recording" else fsm_phase
    return BoundRun(kind, run_dir, video, manifest["video_sha256"], float(fps), width, height,
                    tuple(float(value) for value in frame_times), _phase_per_frame(rows, frame_times, phase_fn),
                    telemetry, sha256_file(telemetry), ledger, manifest["ledger_sha256"], manifest_path)


def phase_spans(phases: tuple[str | None, ...]) -> dict[str, tuple[int, int]]:
    """Return first/last-exclusive frame spans; each phase must be contiguous."""
    result: dict[str, tuple[int, int]] = {}
    for phase in PHASES:
        positions = [index for index, value in enumerate(phases) if value == phase]
        if not positions:
            continue
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise BindingError(f"non-contiguous phase in ledger-matched telemetry: {phase}")
        result[phase] = (positions[0], positions[-1] + 1)
    return result


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _panel(frame: bytes, width: int, height: int, title: str, detail: str) -> Image.Image:
    image = Image.frombytes("RGB", (width, height), frame).resize((640, 360), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 404), "black")
    canvas.paste(image, (0, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="white", font=_font())
    draw.text((8, 23), detail, fill=(190, 220, 255), font=_font())
    return canvas


class _Reader:
    def __init__(self, ffmpeg: Path, bound: BoundRun):
        self.bound, self.index = bound, -1
        self.size = bound.width * bound.height * 3
        self.process = subprocess.Popen([str(ffmpeg), "-v", "error", "-i", str(bound.video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.current: bytes | None = None

    def get(self, target: int) -> bytes:
        if target < self.index:
            raise ValueError("comparison mappings must be monotonic")
        while self.index < target:
            data = self.process.stdout.read(self.size) if self.process.stdout else b""
            if len(data) != self.size:
                raise BindingError(f"decoder ended at frame {self.index + 1}, expected {target}")
            self.current = data
            self.index += 1
        assert self.current is not None
        return self.current

    def close(self) -> None:
        if self.process.stdout:
            self.process.stdout.close()
        stderr = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
        code = self.process.wait()
        # The comparison consumes a monotonic subset of the source frames.
        # Closing a raw-video pipe before ffmpeg has decoded an unneeded tail
        # intentionally produces its normal broken-pipe exit on Windows.
        if code and "Broken pipe" not in stderr:
            raise BindingError(f"ffmpeg decoder failed ({code}): {stderr[-500:]}")


class _Writer:
    def __init__(self, ffmpeg: Path, path: Path, fps: float, width: int, height: int):
        self.path, self.size = path, width * height * 3
        # MPEG-4 Part 2 is deliberately used for portability: the available
        # local, offline ffmpeg is a slim build without libx264.
        self.process = subprocess.Popen([str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, image: Image.Image) -> None:
        data = image.tobytes()
        if len(data) != self.size:
            raise BindingError("incorrect generated comparison frame dimensions")
        assert self.process.stdin is not None
        self.process.stdin.write(data)

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
        code = self.process.wait()
        if code:
            raise BindingError(f"ffmpeg encoder failed ({code}): {stderr[-500:]}")


def _lookup(times: tuple[float, ...], target: float) -> int:
    return min(len(times) - 1, max(0, bisect.bisect_right(times, target) - 1))


def _compose(rec: _Reader, fsm: _Reader, ri: int, fi: int, label: str, ordinal: int, total: int) -> Image.Image:
    left = _panel(rec.get(ri), rec.bound.width, rec.bound.height, "v003 production Fast Replay", f"t={rec.bound.frame_times_s[ri]:.3f}s  {rec.bound.phases[ri] or 'unmapped'}")
    right = _panel(fsm.get(fi), fsm.bound.width, fsm.bound.height, "current coalesced-r4 nominal (5742cda6)", f"t={fsm.bound.frame_times_s[fi]:.3f}s  {fsm.bound.phases[fi] or 'terminal/unmapped'}")
    image = Image.new("RGB", (1280, 434), "black")
    image.paste(left, (0, 30)); image.paste(right, (640, 30))
    ImageDraw.Draw(image).text((8, 8), f"{label} | frame {ordinal + 1}/{total}", fill=(255, 230, 120), font=_font())
    return image


def render_realtime(recording: BoundRun, fsm: BoundRun, ffmpeg: Path, output: Path) -> int:
    fps = recording.fps
    if abs(fps - fsm.fps) > 1e-9:
        raise BindingError("realtime comparison requires identical manifest fps")
    start = max(recording.frame_times_s[0], fsm.frame_times_s[0])
    end = max(recording.frame_times_s[-1], fsm.frame_times_s[-1])
    count = int(round((end - start) * fps)) + 1
    rec, macro, writer = _Reader(ffmpeg, recording), _Reader(ffmpeg, fsm), _Writer(ffmpeg, output, fps, 1280, 434)
    try:
        for ordinal in range(count):
            time_s = start + ordinal / fps
            writer.write(_compose(rec, macro, _lookup(recording.frame_times_s, time_s), _lookup(fsm.frame_times_s, time_s), "realtime (same simulation clock)", ordinal, count))
    finally:
        writer.close(); rec.close(); macro.close()
    return count


def render_phase_aligned(recording: BoundRun, fsm: BoundRun, ffmpeg: Path, output: Path) -> tuple[int, list[dict[str, Any]]]:
    rec_spans, fsm_spans = phase_spans(recording.phases), phase_spans(fsm.phases)
    missing = [phase for phase in PHASES if phase not in rec_spans or phase not in fsm_spans]
    if missing:
        raise BindingError(f"cannot phase-align absent phase(s): {missing}")
    jobs = [(phase, rec_spans[phase], fsm_spans[phase]) for phase in PHASES]
    total = sum(max(re - rs, fe - fs) for _, (rs, re), (fs, fe) in jobs)
    rec, macro, writer, ordinal = _Reader(ffmpeg, recording), _Reader(ffmpeg, fsm), _Writer(ffmpeg, output, recording.fps, 1280, 434), 0
    summary: list[dict[str, Any]] = []
    try:
        for phase, (rs, re), (fs, fe) in jobs:
            count = max(re - rs, fe - fs)
            summary.append({"phase": phase, "recording_frame_span": [rs, re], "fsm_frame_span": [fs, fe], "aligned_frames": count})
            for offset in range(count):
                ri = rs + min(re - rs - 1, int(offset * (re - rs) / count))
                fi = fs + min(fe - fs - 1, int(offset * (fe - fs) / count))
                writer.write(_compose(rec, macro, ri, fi, f"phase-aligned: {phase}", ordinal, total)); ordinal += 1
    finally:
        writer.close(); rec.close(); macro.close()
    return total, summary


def render_keyframes(recording: BoundRun, fsm: BoundRun, ffmpeg: Path, output: Path) -> None:
    rec_spans, fsm_spans = phase_spans(recording.phases), phase_spans(fsm.phases)
    targets = []
    for phase in PHASES:
        rs, re = rec_spans[phase]; fs, fe = fsm_spans[phase]
        targets.append((phase, (rs + re - 1) // 2, (fs + fe - 1) // 2))
    rec, macro = _Reader(ffmpeg, recording), _Reader(ffmpeg, fsm)
    try:
        canvas = Image.new("RGB", (1280, ((len(targets) + 1) // 2) * 217), "black")
        for index, (phase, ri, fi) in enumerate(targets):
            tile = _compose(rec, macro, ri, fi, f"phase midpoint: {phase}", index, len(targets)).resize((640, 217), Image.Resampling.BILINEAR)
            x, y = (index % 2) * 640, (index // 2) * 217
            canvas.paste(tile, (x, y))
        canvas.save(output)
    finally:
        rec.close(); macro.close()


def _metadata(bound: BoundRun) -> dict[str, Any]:
    return {"run_dir": str(bound.run_dir), "video": str(bound.video), "video_sha256": bound.video_sha256,
            "fps": bound.fps, "frame_count": bound.frame_count, "dimensions": [bound.width, bound.height],
            "manifest": str(bound.manifest_path), "ledger": str(bound.ledger_path), "ledger_sha256": bound.ledger_sha256,
            "telemetry": str(bound.telemetry_path), "telemetry_sha256": bound.telemetry_sha256,
            "phase_spans": {key: list(value) for key, value in phase_spans(bound.phases).items()}}


def write_report(recording: BoundRun, fsm: BoundRun, phase_summary: list[dict[str, Any]], output_dir: Path, ffmpeg: Path) -> tuple[Path, Path]:
    payload = {"schema_version": "fsm50.visual_comparison.v1", "offline_only": True, "tool_sha256": sha256_file(Path(__file__)),
               "ffmpeg": str(ffmpeg), "recording": _metadata(recording), "current_fsm": _metadata(fsm),
               "phase_alignment": phase_summary,
               "limitations": ["Visual comparison does not establish a true whole-body COM measurement.", "Visual comparison does not establish filtered/force-based diagonal support evidence.", "The current baseline predates the proposed feedback-landing S10 redesign; it is not proof of that new logic."]}
    json_path = output_dir.parent / "V003_RECORDING_VS_CURRENT_FSM.json"
    md_path = output_dir.parent / "V003_RECORDING_VS_CURRENT_FSM.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text("# v003 Recording vs Current FSM Visual Comparison\n\n"
                       "Offline comparison only: no simulator was launched and source run artifacts were not modified.\n\n"
                       f"- Recording MP4: `{recording.video_sha256}`; {recording.fps:g} fps; {recording.frame_count} frames.\n"
                       f"- Current coalesced-r4 baseline MP4: `{fsm.video_sha256}`; {fsm.fps:g} fps; {fsm.frame_count} frames; bundle `5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78`.\n"
                       f"- Generated outputs: `{output_dir / 'v003_recording_vs_fsm_realtime.mp4'}`, `{output_dir / 'v003_recording_vs_fsm_phase_aligned.mp4'}`, and `{output_dir / 'v003_recording_vs_fsm_keyframes.png'}`.\n\n"
                       "## Evidence boundary\n\n"
                       "The videos are manifest/ledger/SHA bound and phase labels are derived from recording `segment_cursor` ownership and current FSM `macro_state`. They are visual evidence only. They do **not** prove a true COM measurement, filtered or force-based diagonal-support evidence, or the proposed new feedback-landing S10 logic. Those require live runtime instrumentation and validation.\n\n"
                       "## Phase alignment\n\n"
                       + "\n".join(f"- `{item['phase']}`: recording frames {item['recording_frame_span']}; FSM frames {item['fsm_frame_span']}; aligned {item['aligned_frames']}." for item in phase_summary) + "\n",
                       encoding="utf-8")
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create SHA-bound offline v003/current-FSM visual comparisons.")
    parser.add_argument("--recording-run", type=Path, required=True)
    parser.add_argument("--fsm-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    args = parser.parse_args(argv)
    if not args.ffmpeg.is_file():
        raise BindingError(f"ffmpeg executable not found: {args.ffmpeg}")
    output_dir = args.output_dir.resolve()
    if "updated" in {part.lower() for part in output_dir.parts}:
        raise BindingError("refusing an output path containing 'updated'")
    output_dir.mkdir(parents=True, exist_ok=True)
    recording, fsm = bind_run(args.recording_run, "recording"), bind_run(args.fsm_run, "fsm")
    realtime = output_dir / "v003_recording_vs_fsm_realtime.mp4"
    aligned = output_dir / "v003_recording_vs_fsm_phase_aligned.mp4"
    keyframes = output_dir / "v003_recording_vs_fsm_keyframes.png"
    render_realtime(recording, fsm, args.ffmpeg, realtime)
    _, phase_summary = render_phase_aligned(recording, fsm, args.ffmpeg, aligned)
    render_keyframes(recording, fsm, args.ffmpeg, keyframes)
    md_path, json_path = write_report(recording, fsm, phase_summary, output_dir, args.ffmpeg)
    print(json.dumps({"realtime": str(realtime), "phase_aligned": str(aligned), "keyframes": str(keyframes), "report_md": str(md_path), "report_json": str(json_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindingError as exc:
        print(f"visual comparison binding error: {exc}", file=sys.stderr)
        raise SystemExit(2)
