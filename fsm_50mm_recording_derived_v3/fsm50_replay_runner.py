"""Thin Gate-A runner for the production 50 mm Fast Replay pipeline.

The runner owns orchestration and reports only.  Motion is still compiled by
``playback.plan_from_steps`` and executed by the existing
``SimTimePlaybackService -> SimRobotAdapter`` worker path.  The worker-side
normal-development observer writes minimal telemetry and an active-viewport
video; this module never implements scheduler or actuator semantics.

One invocation may select v003, explicit versions, or every on-disk version.
Workers are launched strictly serially.  A completed run is not eligible for a
task-success verdict until either the worker provides fully decoded video
evidence or a SHA-bound ``manual_video_verdict.json`` is reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from sequence_model import load_steps_jsonl

from .fsm50_task_success import (
    EVALUATED,
    NOT_EVALUATED,
    POSTURE_NOT_APPLICABLE,
    REPLAY_TASK_FAIL,
    REPLAY_TASK_NOT_EVALUATED,
    REPLAY_TASK_SUCCESS,
    REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE,
    TABLE_COLUMNS,
    ManualVideoVerdict,
    TaskSuccessAssessment,
    classify_replay_task,
)
from .worker_task_replay_session import (
    DEFAULT_POST_SETTLE_S,
    DEFAULT_TELEMETRY_HZ,
    DEFAULT_VIDEO_FPS,
    REQUEST_SCHEMA,
)


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_RECORDING_ROOT = (
    PROJECT_ROOT / "saved_height_steps_fsm_reference_v2" / "height_050mm"
)
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "runs" / "50mm_fast_replay"
DEFAULT_REPORT_ROOT = MODULE_ROOT / "reports"
CSV_REPORT_NAME = "50MM_REPLAY_TASK_SUCCESS_TABLE.csv"
MARKDOWN_REPORT_NAME = "50MM_REPLAY_TASK_SUCCESS_TABLE.md"
RUNNER_RESULT_NAME = "fsm50_task_replay_runner_result.json"
TASK_ASSESSMENT_NAME = "task_success_assessment.json"
CLASSIFIER_INPUTS_NAME = "classifier_inputs.json"
VIDEO_VERDICT_NAME = "manual_video_verdict.json"
VIDEO_VERDICT_TEMPLATE_NAME = "manual_video_verdict.template.json"
VIDEO_VERDICT_SCHEMA = "fsm50.manual_video_verdict.v1"
RUNNER_RESULT_SCHEMA = "fsm50.task_replay_runner_result.v1"

_VERSION_RE = re.compile(r"^v(?P<number>\d{3})(?:_|$)", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUCCESS_RESULTS = {
    REPLAY_TASK_SUCCESS,
    REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE,
}
_VIDEO_BOOL_FIELDS = (
    "task_completed",
    "body_crossed_front_face",
    "required_leg_lift_completed",
    "final_recoverable",
    "posture_incomplete",
    "robot_fell",
    "body_stuck",
    "wheel_drive_up_without_required_lift",
    "dangerous_body_collision",
    "joint_limit_violation",
    "severe_penetration",
    "irrecoverable",
)
_VIDEO_VERDICT_FIELDS = frozenset(
    (*_VIDEO_BOOL_FIELDS, "first_actual_failure_phase", "notes")
)
_VIDEO_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "review_complete",
        "source_version",
        "request_id",
        "plan_id",
        "run_dir",
        "task_inputs_path",
        "task_inputs_sha256",
        "video_path",
        "video_sha256",
        "reviewed_utc",
        "verdict",
    }
)


class RunnerContractError(RuntimeError):
    """A production-runner boundary was not satisfied."""


class SimulatorProcessConflictError(RunnerContractError):
    """A measured process preflight found an existing Isaac/Kit worker."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


@dataclass(frozen=True)
class RecordingVersion:
    version_id: str
    directory: Path
    accepted_steps_path: Path
    metadata_path: Path

    @property
    def short_id(self) -> str:
        match = _VERSION_RE.match(self.version_id)
        return match.group(0).rstrip("_") if match else self.version_id


@dataclass(frozen=True)
class PreparedReplay:
    recording: RecordingVersion
    steps: list[dict[str, Any]]
    accepted_steps_sha256: str
    plan: Any


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    request_path: Path
    runner_result_path: Path


@dataclass(frozen=True)
class ReplayAttempt:
    recording: RecordingVersion
    run_dir: Path
    assessment: TaskSuccessAssessment
    runner_result_path: Path
    shutdown_verified: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_load(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {source}")
            result[key] = value
        return result

    value = json.loads(
        source.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("runner artifacts cannot contain NaN or Infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _jsonable(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def version_sort_key(value: str | RecordingVersion) -> tuple[int, str]:
    text = value.version_id if isinstance(value, RecordingVersion) else str(value)
    match = _VERSION_RE.match(text)
    return (int(match.group("number")) if match else sys.maxsize, text.lower())


def enumerate_recording_versions(
    recording_root: str | Path = DEFAULT_RECORDING_ROOT,
) -> list[RecordingVersion]:
    """Enumerate physical version directories without assuming their count."""

    root = Path(recording_root).resolve()
    versions_root = root / "versions"
    if not versions_root.is_dir():
        raise FileNotFoundError(versions_root)
    result = [
        RecordingVersion(
            version_id=directory.name,
            directory=directory.resolve(),
            accepted_steps_path=(directory / "accepted_steps.jsonl").resolve(),
            metadata_path=(directory / "metadata.json").resolve(),
        )
        for directory in versions_root.iterdir()
        if directory.is_dir() and _VERSION_RE.match(directory.name)
    ]
    result.sort(key=version_sort_key)
    if not result:
        raise RunnerContractError(f"no recording versions found under {versions_root}")
    return result


def _selector_tokens(selectors: Iterable[str]) -> list[str]:
    return [
        token.strip()
        for selector in selectors
        for token in str(selector).split(",")
        if token.strip()
    ]


def select_recording_versions(
    versions: Sequence[RecordingVersion], selectors: Iterable[str]
) -> list[RecordingVersion]:
    """Resolve ``v003``/full-id/``all`` selectors, preserving explicit order."""

    tokens = _selector_tokens(selectors)
    if not tokens:
        tokens = ["v003"]
    if any(token.lower() == "all" for token in tokens):
        if len(tokens) != 1:
            raise ValueError("version selector 'all' cannot be combined with other selectors")
        return sorted(versions, key=version_sort_key)

    selected: list[RecordingVersion] = []
    for token in tokens:
        lowered = token.lower()
        matches = [
            item
            for item in versions
            if item.version_id.lower() == lowered
            or item.version_id.lower().startswith(lowered + "_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"version selector {token!r} matched {len(matches)} directories"
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def _load_recording_metadata(recording: RecordingVersion) -> dict[str, Any]:
    if not recording.accepted_steps_path.is_file():
        raise FileNotFoundError(recording.accepted_steps_path)
    if not recording.metadata_path.is_file():
        raise FileNotFoundError(recording.metadata_path)
    metadata = _strict_json_load(recording.metadata_path)
    claimed_id = str(metadata.get("version_id", "") or "")
    if claimed_id and claimed_id != recording.version_id:
        raise RunnerContractError(
            f"metadata version_id mismatch for {recording.version_id}: {claimed_id!r}"
        )
    digest = sha256_file(recording.accepted_steps_path)
    expected = str(metadata.get("accepted_steps_sha256", "") or "").lower()
    if not _SHA256_RE.fullmatch(expected) or expected != digest:
        raise RunnerContractError(
            f"accepted_steps SHA-256 mismatch for {recording.version_id}"
        )
    return metadata


def prepare_replay(
    recording: RecordingVersion,
    *,
    plan_builder: Callable[..., Any] | None = None,
    max_wheel_speed_rad_s: float | None = None,
) -> PreparedReplay:
    """Load source commands and call the production Fast compiler once."""

    _load_recording_metadata(recording)
    steps = load_steps_jsonl(recording.accepted_steps_path)
    if not steps:
        raise RunnerContractError(f"recording {recording.version_id} has no steps")
    if plan_builder is None:
        from playback import plan_from_steps

        plan_builder = plan_from_steps
    if max_wheel_speed_rad_s is None:
        from motion_speed import load_motion_reference

        max_wheel_speed_rad_s = float(
            load_motion_reference().wheel_velocity_limit_rad_s
        )
    plan = plan_builder(
        steps,
        profile="fast",
        max_wheel_speed=float(max_wheel_speed_rad_s),
        label=f"50 mm {recording.version_id} normal development Fast Replay",
        sequence_total_steps=len(steps),
    )
    # ``fast`` is the UI request spelling.  The production compiler's
    # canonical plan-object profile is ``motion_only``.
    if str(getattr(plan, "profile", "") or "") != "motion_only":
        raise RunnerContractError(
            "production compiler did not return canonical Fast profile motion_only"
        )
    digest = str(getattr(plan, "plan_sha256", "") or "").lower()
    if not _SHA256_RE.fullmatch(digest):
        raise RunnerContractError("production Fast plan has no valid SHA-256")
    if not list(getattr(plan, "events", []) or []):
        raise RunnerContractError("production Fast plan contains no events")
    if not list(getattr(plan, "segments", []) or []):
        raise RunnerContractError("production Fast plan contains no segments")
    return PreparedReplay(
        recording=recording,
        steps=steps,
        accepted_steps_sha256=sha256_file(recording.accepted_steps_path),
        plan=plan,
    )


def allocate_run_paths(output_root: str | Path, version_id: str) -> RunPaths:
    version_root = Path(output_root).resolve() / version_id
    version_root.mkdir(parents=True, exist_ok=True)
    run_dir = version_root / f"{_utc_stamp()}_production_fast_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        request_path=run_dir / "worker_task_replay_request.json",
        runner_result_path=run_dir / RUNNER_RESULT_NAME,
    )


def build_worker_task_request(
    prepared: PreparedReplay,
    paths: RunPaths,
    *,
    request_id: str,
    plan_id: str,
    telemetry_hz: float = DEFAULT_TELEMETRY_HZ,
    video_fps: float = DEFAULT_VIDEO_FPS,
    post_run_settle_s: float = DEFAULT_POST_SETTLE_S,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    plan = prepared.plan
    if timeout_s is None:
        timeout_s = float(getattr(plan, "final_time_s", 0.0) or 0.0) + 30.0
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": str(request_id),
        "plan_id": str(plan_id),
        "plan_sha256": str(plan.plan_sha256).lower(),
        "plan_event_count": len(list(plan.events)),
        "plan_segment_count": len(list(plan.segments)),
        "source_version": prepared.recording.version_id,
        "height_mm": 50,
        "step_count": len(prepared.steps),
        "run_dir": str(paths.run_dir.resolve()),
        "accepted_steps_path": str(
            prepared.recording.accepted_steps_path.resolve()
        ),
        "accepted_steps_sha256": prepared.accepted_steps_sha256,
        "telemetry_hz": float(telemetry_hz),
        "video_fps": float(video_fps),
        "capture_video": True,
        "post_run_settle_s": float(post_run_settle_s),
        "timeout_s": float(timeout_s),
        "filtered_contact_bank_enabled": False,
    }
    # The worker performs the authoritative exact-key validation.  Catch
    # obvious frontend errors before a costly Isaac launch.
    if not 10.0 <= float(telemetry_hz) <= 30.0:
        raise ValueError("telemetry_hz must be in the minimal 10..30 Hz range")
    if abs(float(video_fps) - 15.0) > 1.0e-9:
        raise ValueError("normal task replay requires 15 fps viewport video")
    if float(post_run_settle_s) <= 0.0 or float(timeout_s) <= 0.0:
        raise ValueError("settle and task timeout must be positive")
    return payload


def _not_evaluated_assessment(
    *,
    version: str,
    step_count: int = 0,
    segment_count: int = 0,
    reason: str,
    video_path: str = "",
) -> TaskSuccessAssessment:
    return TaskSuccessAssessment(
        version=version,
        step_count=max(0, int(step_count)),
        fast_segment_count=max(0, int(segment_count)),
        evaluation_status=NOT_EVALUATED,
        task_result=REPLAY_TASK_NOT_EVALUATED,
        posture_result=POSTURE_NOT_APPLICABLE,
        body_crossed_front_face=False,
        required_leg_lift_completed=False,
        final_recoverable=False,
        final_wheel_classes={},
        peak_roll=None,
        peak_pitch=None,
        video_path=str(video_path),
        first_actual_failure_phase="",
        hard_failure_reasons=(),
        not_evaluated_reasons=(str(reason),),
        secondary_diagnostics=(),
        notes=(),
        classification_reasons=(
            "task was not evaluated because the production run or required evidence was incomplete",
        ),
    )


def _defer_success_until_manual_review(
    assessment: TaskSuccessAssessment,
) -> TaskSuccessAssessment:
    """Keep measured task facts while withholding an unreviewed success label."""

    if assessment.task_result not in _SUCCESS_RESULTS:
        return assessment
    return TaskSuccessAssessment(
        version=assessment.version,
        step_count=assessment.step_count,
        fast_segment_count=assessment.fast_segment_count,
        evaluation_status=NOT_EVALUATED,
        task_result=REPLAY_TASK_NOT_EVALUATED,
        posture_result=POSTURE_NOT_APPLICABLE,
        body_crossed_front_face=assessment.body_crossed_front_face,
        required_leg_lift_completed=assessment.required_leg_lift_completed,
        final_recoverable=assessment.final_recoverable,
        final_wheel_classes=dict(assessment.final_wheel_classes),
        peak_roll=assessment.peak_roll,
        peak_pitch=assessment.peak_pitch,
        video_path=assessment.video_path,
        first_actual_failure_phase="",
        hard_failure_reasons=(),
        not_evaluated_reasons=(
            *assessment.not_evaluated_reasons,
            "MANUAL_VIDEO_REVIEW_PENDING",
        ),
        secondary_diagnostics=assessment.secondary_diagnostics,
        notes=assessment.notes,
        classification_reasons=(
            "production replay artifacts are sealed; task success is withheld until a SHA-bound manual viewport review",
        ),
    )


def _pending_table_row(recording: RecordingVersion) -> dict[str, object]:
    step_count = 0
    try:
        metadata = _strict_json_load(recording.metadata_path)
        raw_count = metadata.get("step_count", 0)
        if type(raw_count) is int and raw_count >= 0:
            step_count = raw_count
    except Exception:
        pass
    return _not_evaluated_assessment(
        version=recording.version_id,
        step_count=step_count,
        reason="NOT_RUN",
    ).to_table_row()


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        text = "true" if value else "false"
    else:
        text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(TABLE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in TABLE_COLUMNS})
    return stream.getvalue()


def _markdown_text(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# 50 mm Fast Replay Task Success Table",
        "",
        (
            "Task completion and final-posture quality are separate. Strict rest, "
            "contact drift, and final all-TOP state remain secondary diagnostics."
        ),
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in TABLE_COLUMNS)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_existing_table(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
            raise RunnerContractError(
                f"existing task table has an incompatible header: {path}"
            )
        rows: dict[str, dict[str, object]] = {}
        for row in reader:
            version = str(row.get("version", "") or "")
            if version:
                rows[version] = {column: row.get(column, "") for column in TABLE_COLUMNS}
        return rows


def update_task_success_tables(
    *,
    report_root: str | Path,
    recordings: Sequence[RecordingVersion],
    updates: Iterable[Mapping[str, object]] = (),
) -> tuple[Path, Path]:
    """Merge rows by version and atomically replace both report files."""

    root = Path(report_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / CSV_REPORT_NAME
    markdown_path = root / MARKDOWN_REPORT_NAME
    existing = _read_existing_table(csv_path)
    on_disk = {item.version_id: item for item in recordings}
    merged: dict[str, dict[str, object]] = {
        version: dict(existing.get(version) or _pending_table_row(recording))
        for version, recording in on_disk.items()
    }
    for raw in updates:
        row = {column: raw.get(column, "") for column in TABLE_COLUMNS}
        version = str(row["version"] or "")
        if version not in on_disk:
            raise RunnerContractError(
                f"refusing a report row for a non-existent recording: {version!r}"
            )
        merged[version] = row
    ordered = [merged[key] for key in sorted(merged, key=version_sort_key)]
    # Each replacement is atomic and both complete byte streams are prepared
    # before the first destination changes.
    csv_payload = "\ufeff" + _csv_text(ordered)
    md_payload = _markdown_text(ordered)
    _atomic_write_text(csv_path, csv_payload, encoding="utf-8")
    _atomic_write_text(markdown_path, md_payload, encoding="utf-8")
    return csv_path, markdown_path


def _path_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _video_verdict_template(
    *,
    source_version: str,
    request_id: str,
    plan_id: str,
    run_dir: Path,
    task_inputs_path: Path,
    video_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": VIDEO_VERDICT_SCHEMA,
        "review_complete": False,
        "source_version": source_version,
        "request_id": request_id,
        "plan_id": plan_id,
        "run_dir": str(run_dir.resolve()),
        "task_inputs_path": str(task_inputs_path.resolve()),
        "task_inputs_sha256": sha256_file(task_inputs_path),
        "video_path": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "reviewed_utc": "",
        "verdict": {
            **{name: None for name in _VIDEO_BOOL_FIELDS},
            "first_actual_failure_phase": "",
            "notes": [],
        },
    }


def _load_bound_video_verdict(
    verdict_path: str | Path,
    *,
    source_version: str,
    request_id: str,
    plan_id: str,
    run_dir: Path,
    task_inputs_path: Path,
    video_path: Path,
    require_complete: bool = True,
) -> tuple[dict[str, Any], ManualVideoVerdict | None]:
    document = _strict_json_load(verdict_path)
    if set(document) != _VIDEO_DOCUMENT_FIELDS:
        missing = sorted(_VIDEO_DOCUMENT_FIELDS - set(document))
        unexpected = sorted(set(document) - _VIDEO_DOCUMENT_FIELDS)
        raise RunnerContractError(
            f"manual video verdict keys mismatch; missing={missing} unexpected={unexpected}"
        )
    expected_text = {
        "schema_version": VIDEO_VERDICT_SCHEMA,
        "source_version": source_version,
        "request_id": request_id,
        "plan_id": plan_id,
        "run_dir": str(run_dir.resolve()),
        "task_inputs_path": str(task_inputs_path.resolve()),
        "video_path": str(video_path.resolve()),
    }
    for key, expected in expected_text.items():
        if document.get(key) != expected:
            raise RunnerContractError(
                f"manual video verdict {key} mismatch: expected={expected!r} "
                f"actual={document.get(key)!r}"
            )
    if not _path_within(video_path, run_dir) or not video_path.is_file():
        raise RunnerContractError("review video is missing or escapes its run directory")
    if not task_inputs_path.is_file() or not _path_within(task_inputs_path, run_dir):
        raise RunnerContractError("task_inputs is missing or escapes its run directory")
    digest_expectations = {
        "task_inputs_sha256": sha256_file(task_inputs_path),
        "video_sha256": sha256_file(video_path),
    }
    for key, expected in digest_expectations.items():
        actual = document.get(key)
        if type(actual) is not str or actual.lower() != expected:
            raise RunnerContractError(f"manual video verdict {key} is stale or invalid")

    complete = document.get("review_complete")
    if type(complete) is not bool:
        raise RunnerContractError("manual video verdict review_complete must be boolean")
    if require_complete and complete is not True:
        raise RunnerContractError("manual video verdict is still an incomplete template")
    reviewed_utc = document.get("reviewed_utc")
    if type(reviewed_utc) is not str:
        raise RunnerContractError("manual video verdict reviewed_utc must be a string")
    if complete and not reviewed_utc.strip():
        raise RunnerContractError("completed manual video verdict needs reviewed_utc")
    if complete:
        try:
            parsed_reviewed = datetime.fromisoformat(reviewed_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RunnerContractError(
                "manual video verdict reviewed_utc is not ISO-8601"
            ) from exc
        if parsed_reviewed.tzinfo is None:
            raise RunnerContractError(
                "manual video verdict reviewed_utc must include a timezone"
            )

    verdict = document.get("verdict")
    if not isinstance(verdict, Mapping) or set(verdict) != _VIDEO_VERDICT_FIELDS:
        raise RunnerContractError("manual video verdict payload has an incompatible schema")
    kwargs: dict[str, Any] = {}
    for name in _VIDEO_BOOL_FIELDS:
        value = verdict[name]
        if value is not None and type(value) is not bool:
            raise RunnerContractError(f"manual video verdict {name} must be bool or null")
        kwargs[name] = value
    phase = verdict["first_actual_failure_phase"]
    notes = verdict["notes"]
    if type(phase) is not str:
        raise RunnerContractError("first_actual_failure_phase must be a string")
    if not isinstance(notes, list) or any(type(note) is not str for note in notes):
        raise RunnerContractError("manual video verdict notes must be a string list")
    kwargs["first_actual_failure_phase"] = phase
    kwargs["notes"] = tuple(notes)
    return document, ManualVideoVerdict(**kwargs) if complete else None


def _task_inputs_from_terminal(
    terminal: Mapping[str, Any], run_dir: Path
) -> tuple[Path, dict[str, Any], Path]:
    task_inputs_path = Path(str(terminal.get("task_inputs_path", "") or "")).resolve()
    if not task_inputs_path.is_file() or not _path_within(task_inputs_path, run_dir):
        raise RunnerContractError("worker terminal has no contained task_inputs artifact")
    inputs = _strict_json_load(task_inputs_path)
    required = {"completed_result", "physical_evidence", "final_telemetry_row"}
    if not required.issubset(inputs):
        raise RunnerContractError("task_inputs artifact is missing classifier inputs")
    completed = inputs["completed_result"]
    if not isinstance(completed, Mapping):
        raise RunnerContractError("completed_result is not an object")
    terminal_video = str(terminal.get("video_path", "") or "")
    input_video = str(completed.get("video_path", "") or "")
    if terminal_video and input_video and Path(terminal_video).resolve() != Path(input_video).resolve():
        raise RunnerContractError("worker terminal and task_inputs disagree on video_path")
    video_path = Path(terminal_video or input_video).resolve()
    if not video_path.is_file() or not _path_within(video_path, run_dir):
        raise RunnerContractError("worker viewport video is missing or outside run_dir")
    return task_inputs_path, inputs, video_path


def classify_run_inputs(
    *,
    task_inputs: Mapping[str, Any],
    manual_video_verdict: ManualVideoVerdict | None,
    second_simulator_process_detected: bool | None = None,
) -> TaskSuccessAssessment:
    completed_result = dict(task_inputs["completed_result"])
    if second_simulator_process_detected is not None:
        if type(second_simulator_process_detected) is not bool:
            raise TypeError("second_simulator_process_detected must be bool or None")
        completed_result["second_simulator_process_detected"] = (
            second_simulator_process_detected
        )
    return classify_replay_task(
        completed_result=completed_result,
        physical_evidence=task_inputs["physical_evidence"],
        final_telemetry_row=task_inputs["final_telemetry_row"],
        video_verdict=manual_video_verdict,
    )


def build_classifier_inputs(
    task_inputs: Mapping[str, Any],
    *,
    process_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the measured second-Isaac preflight with worker-owned inputs."""

    if not isinstance(process_preflight, Mapping):
        raise RunnerContractError("process preflight must be an object")
    if process_preflight.get("schema_version") != (
        "fsm50.second_simulator_preflight.v1"
    ):
        raise RunnerContractError("process preflight schema is not supported")
    if process_preflight.get("checked") is not True:
        raise RunnerContractError("classifier inputs require a measured process preflight")
    checked_utc = process_preflight.get("checked_utc")
    if type(checked_utc) is not str or not checked_utc.strip():
        raise RunnerContractError("process preflight has no exact checked_utc")
    snapshot_count = process_preflight.get("snapshot_process_count")
    if type(snapshot_count) is not int or snapshot_count < 0:
        raise RunnerContractError(
            "process preflight snapshot_process_count must be a nonnegative int"
        )
    snapshot_sha256 = process_preflight.get("process_snapshot_sha256")
    if type(snapshot_sha256) is not str or _SHA256_RE.fullmatch(
        snapshot_sha256
    ) is None:
        raise RunnerContractError("process preflight snapshot SHA-256 is invalid")
    conflict_count = process_preflight.get("conflict_count")
    if type(conflict_count) is not int or conflict_count < 0:
        raise RunnerContractError(
            "process preflight conflict_count must be a nonnegative int"
        )
    conflicts = process_preflight.get("conflicts")
    if type(conflicts) is not list or any(type(row) is not dict for row in conflicts):
        raise RunnerContractError("process preflight conflicts must be a list of objects")
    if conflict_count != len(conflicts) or conflict_count > snapshot_count:
        raise RunnerContractError("process preflight conflict count is inconsistent")
    detected = process_preflight.get("second_simulator_process_detected")
    if type(detected) is not bool:
        raise RunnerContractError("process preflight has no boolean conflict verdict")
    if detected is not bool(conflict_count):
        raise RunnerContractError("process preflight conflict verdict is inconsistent")
    try:
        measured_preflight = json.loads(
            json.dumps(process_preflight, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise RunnerContractError(
            f"process preflight is not exact finite JSON: {exc}"
        ) from exc
    if not isinstance(measured_preflight, dict):
        raise RunnerContractError("process preflight must round-trip as an object")
    payload = json.loads(json.dumps(task_inputs, allow_nan=False))
    completed = payload.get("completed_result")
    if not isinstance(completed, dict):
        raise RunnerContractError("classifier inputs have no completed_result object")
    completed["single_simulator_preflight_available"] = True
    completed["second_simulator_process_detected"] = detected
    completed["second_simulator_process_preflight"] = measured_preflight
    payload["runner_classifier_input_schema"] = "fsm50.runner_classifier_inputs.v1"
    return payload


def _validate_operation_ack(
    ack: Mapping[str, Any],
    *,
    operation: str,
    request_id: str,
) -> dict[str, Any]:
    result = dict(ack)
    if (
        result.get("type") != "operation_ack"
        or result.get("operation") != operation
        or result.get("request_id") != request_id
        or result.get("accepted") is not True
        or str(result.get("error", "") or "")
    ):
        raise RunnerContractError(
            f"production worker rejected {operation}: {result!r}"
        )
    return result


def wait_for_worker_ready(
    client: Any, *, timeout_s: float, poll_interval_s: float = 0.02
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        client.poll()
        last = dict(client.status() or {})
        if last.get("ready") is True:
            worker_session_id = str(last.get("worker_session_id", "") or "")
            if not worker_session_id:
                raise RunnerContractError("ready worker has no worker_session_id")
            return last
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise RunnerContractError(
                f"worker exited before ready: returncode={process.poll()} status={last!r}"
            )
        if str(last.get("error", "") or ""):
            raise RunnerContractError(f"worker startup failed: {last['error']}")
        time.sleep(max(0.001, float(poll_interval_s)))
    raise RunnerContractError(f"worker readiness timed out: {last!r}")


def validate_task_worker_binding(
    status: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the request reached the normal task hook before dispatch."""

    result = dict(status)
    preflight = result.get("worker_task_replay_preflight")
    task_session = result.get("worker_task_replay_session")
    artifact_session = result.get("worker_artifact_session")
    artifact_preflight = result.get("worker_artifact_preflight")
    if result.get("task_replay_preflight_ready") is not True:
        raise RunnerContractError("worker task replay preflight is not ready")
    if result.get("task_replay_request_id") != request.get("request_id"):
        raise RunnerContractError("worker task replay request_id was not bound")
    if not isinstance(preflight, Mapping) or not isinstance(task_session, Mapping):
        raise RunnerContractError("worker has no task replay preflight/session status")
    for key, expected in {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": request.get("request_id"),
        "plan_id": request.get("plan_id"),
        "source_version": request.get("source_version"),
        "height_mm": request.get("height_mm"),
        "step_count": request.get("step_count"),
        "run_dir": request.get("run_dir"),
        "telemetry_hz": request.get("telemetry_hz"),
        "video_fps": request.get("video_fps"),
        "filtered_contact_bank_enabled": False,
        "preflight_ok": True,
    }.items():
        if preflight.get(key) != expected:
            raise RunnerContractError(
                f"worker task preflight {key} mismatch: expected={expected!r} "
                f"actual={preflight.get(key)!r}"
            )
    worker_pid = result.get("worker_pid")
    worker_session_id = result.get("worker_session_id")
    adapter_id = result.get("adapter_runtime_instance_id")
    root_writes = result.get("root_state_write_count")
    if type(worker_pid) is not int or worker_pid <= 0:
        raise RunnerContractError("ready task worker has no exact positive PID")
    if type(worker_session_id) is not str or not worker_session_id:
        raise RunnerContractError("ready task worker has no session identity")
    if type(adapter_id) is not str or not adapter_id:
        raise RunnerContractError("ready task worker has no adapter identity")
    if type(root_writes) is not int or root_writes != 0:
        raise RunnerContractError("normal task worker unexpectedly wrote root state")
    if (
        task_session.get("enabled") is not True
        or task_session.get("execution_mode") != "normal_development"
        or task_session.get("request_id") != request.get("request_id")
        or task_session.get("source_version") != request.get("source_version")
        or task_session.get("state") != "ready_for_plan"
        or task_session.get("filtered_contact_bank_enabled") is not False
    ):
        raise RunnerContractError("worker task session identity/state is invalid")
    for label, strict_status in (
        ("worker_artifact_session", artifact_session),
        ("worker_artifact_preflight", artifact_preflight),
    ):
        if not isinstance(strict_status, Mapping) or strict_status.get("enabled") is not False:
            raise RunnerContractError(
                f"strict artifact pipeline must be disabled: {label}={strict_status!r}"
            )
    return result


def wait_for_operation_ack(
    client: Any,
    *,
    operation: str,
    request_id: str,
    timeout_s: float,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() <= deadline:
        client.poll()
        status = dict(client.status() or {})
        history = list(status.get("operation_ack_history", []) or [])
        latest = status.get("last_operation_ack")
        if isinstance(latest, Mapping):
            history.append(dict(latest))
        for raw in reversed(history):
            ack = dict(raw or {})
            if ack.get("operation") == operation and ack.get("request_id") == request_id:
                return ack
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise RunnerContractError(
                f"worker exited while waiting for {operation}: returncode={process.poll()}"
            )
        time.sleep(max(0.001, float(poll_interval_s)))
    raise RunnerContractError(f"timed out waiting for {operation} ACK")


def wait_for_task_terminal(
    client: Any,
    *,
    request_id: str,
    timeout_s: float,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() <= deadline:
        client.poll()
        terminal = _latest_raw_task_terminal(client, request_id=request_id)
        if terminal is not None:
            return terminal
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise RunnerContractError(
                "worker exited before the request-matched task terminal ACK"
            )
        time.sleep(max(0.001, float(poll_interval_s)))
    raise RunnerContractError("timed out waiting for task replay terminal ACK")


def _latest_raw_task_terminal(
    client: Any,
    *,
    request_id: str,
) -> dict[str, Any] | None:
    """Return only a raw request-matched terminal, never its operation ACK."""

    status = dict(client.status() or {})
    candidates = (
        getattr(client, "latest_task_replay_terminal", None),
        status.get("last_task_replay_terminal"),
    )
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        candidate = dict(raw)
        if (
            candidate.get("type") in {"task_replay_complete", "task_replay_failed"}
            and candidate.get("operation") == "task_replay"
            and candidate.get("request_id") == request_id
            and candidate.get("phase")
            in {"TASK_REPLAY_COMPLETE", "TASK_REPLAY_FAILED"}
        ):
            return candidate
    return None


def validate_task_terminal(
    terminal: Mapping[str, Any],
    *,
    request_id: str,
    plan_id: str,
    source_version: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Validate the exact worker/run identity before reading any artifact."""

    result = dict(terminal)
    phase = result.get("phase")
    complete = phase == "TASK_REPLAY_COMPLETE"
    failed = phase == "TASK_REPLAY_FAILED"
    expected_type = "task_replay_complete" if complete else "task_replay_failed"
    expected_accepted = True if complete else False
    expected_complete = True if complete else False
    for key, expected in {
        "type": expected_type,
        "operation": "task_replay",
        "request_id": request_id,
        "plan_id": plan_id,
        "source_version": source_version,
        "run_dir": str(run_dir.resolve()),
        "accepted": expected_accepted,
        "task_replay_complete": expected_complete,
        "video_writer_quiesced": True,
    }.items():
        if result.get(key) != expected:
            raise RunnerContractError(
                f"task terminal {key} mismatch: expected={expected!r} "
                f"actual={result.get(key)!r}"
            )
    if not (complete or failed):
        raise RunnerContractError(f"unsupported task terminal phase: {phase!r}")
    for key in ("task_inputs_path", "worker_result_path", "video_path"):
        path = Path(str(result.get(key, "") or "")).resolve()
        if not path.is_file() or not _path_within(path, run_dir):
            raise RunnerContractError(
                f"task terminal {key} is missing or escapes run_dir: {path}"
            )
    return result


def validate_task_operation_ack(
    ack: Mapping[str, Any],
    *,
    terminal: Mapping[str, Any],
    request_id: str,
    plan_id: str,
    source_version: str,
    run_dir: Path,
    worker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(ack)
    complete = terminal.get("phase") == "TASK_REPLAY_COMPLETE"
    for key, expected in {
        "type": "operation_ack",
        "operation": "task_replay",
        "phase": terminal.get("phase"),
        "request_id": request_id,
        "plan_id": plan_id,
        "source_version": source_version,
        "run_dir": str(run_dir.resolve()),
        "accepted": bool(complete),
        "task_replay_complete": bool(complete),
        "video_writer_quiesced": True,
        "worker_pid": worker_binding.get("worker_pid"),
        "worker_session_id": worker_binding.get("worker_session_id"),
        "adapter_runtime_instance_id": worker_binding.get(
            "adapter_runtime_instance_id"
        ),
        "artifact_request_id": "",
        "root_state_write_count": 0,
    }.items():
        if result.get(key) != expected:
            raise RunnerContractError(
                f"task operation ACK {key} mismatch: expected={expected!r} "
                f"actual={result.get(key)!r}"
            )
    return result


def validate_fast_shutdown(
    outcome: Mapping[str, Any],
    *,
    request_id: str,
    owned_worker_pid: int | None = None,
    task_request_id: str,
    worker_session_id: str,
    adapter_runtime_instance_id: str,
) -> dict[str, Any]:
    result = dict(outcome)
    problems: list[str] = []
    if type(owned_worker_pid) is not int or owned_worker_pid <= 0:
        problems.append("owned worker PID is invalid")
    if not worker_session_id:
        problems.append("worker session identity is missing")
    if not adapter_runtime_instance_id:
        problems.append("adapter runtime identity is missing")
    if not task_request_id:
        problems.append("task replay request identity is missing")
    if result.get("returncode") != 0:
        problems.append(f"returncode={result.get('returncode')!r}")
    if result.get("forced_termination") is not False:
        problems.append("forced termination")
    if result.get("normal_exit") is not True or result.get("timed_out") is not False:
        problems.append("worker did not exit normally")
    if result.get("requested_mode") != "fast":
        problems.append("shutdown was not fast")
    if owned_worker_pid is not None and result.get("pid") != owned_worker_pid:
        problems.append("shutdown PID differs from the owned worker")
    close_kwargs = {
        "wait_for_replicator": False,
        "skip_cleanup": True,
    }
    identity = {
        "worker_pid": owned_worker_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_runtime_instance_id,
        "artifact_request_id": "",
        "task_replay_request_id": task_request_id,
        "root_state_write_count": 0,
    }
    shutdown_ack = result.get("shutdown_ack")
    if not isinstance(shutdown_ack, Mapping) or (
        shutdown_ack.get("type") != "operation_ack"
        or shutdown_ack.get("operation") != "shutdown"
        or shutdown_ack.get("request_id") != request_id
        or shutdown_ack.get("mode") != "fast"
        or shutdown_ack.get("accepted") is not True
        or shutdown_ack.get("close_kwargs") != close_kwargs
        or str(shutdown_ack.get("error", "") or "")
    ):
        problems.append("shutdown operation ACK invalid")
    else:
        for key, expected in identity.items():
            if shutdown_ack.get(key) != expected:
                problems.append(f"shutdown operation ACK {key} mismatch")
    runtime_version = (
        str(shutdown_ack.get("runtime_version", "") or "")
        if isinstance(shutdown_ack, Mapping)
        else ""
    )
    if not runtime_version.startswith("5.1."):
        problems.append("shutdown runtime version is not supported Isaac 5.1")
    for name in ("close_requested",):
        ack = result.get(name)
        if not isinstance(ack, Mapping):
            problems.append(f"{name} missing")
            continue
        if (
            ack.get("type") != name
            or ack.get("mode") != "fast"
            or ack.get("request_id") != request_id
            or ack.get("accepted") is not True
            or ack.get("close_kwargs") != close_kwargs
            or ack.get("runtime_version") != runtime_version
            or str(ack.get("error", "") or "")
        ):
            problems.append(f"{name} invalid")
        else:
            for key, expected in identity.items():
                if ack.get(key) != expected:
                    problems.append(f"{name} {key} mismatch")
    close_returned = result.get("close_returned")
    if close_returned:
        if not isinstance(close_returned, Mapping) or (
            close_returned.get("type") != "close_returned"
            or close_returned.get("mode") != "fast"
            or close_returned.get("request_id") != request_id
            or close_returned.get("accepted") is not True
            or close_returned.get("close_kwargs") != close_kwargs
            or close_returned.get("runtime_version") != runtime_version
            or str(close_returned.get("error", "") or "")
        ):
            problems.append("close_returned invalid")
        else:
            for key, expected in identity.items():
                if close_returned.get(key) != expected:
                    problems.append(f"close_returned {key} mismatch")
    receipt = result.get("close_requested_receipt")
    if not isinstance(receipt, Mapping) or (
        receipt.get("type") != "close_receipt"
        or receipt.get("close_event_type") != "close_requested"
        or receipt.get("request_id") != request_id
        or receipt.get("mode") != "fast"
        or receipt.get("received") is not True
        or receipt.get("accepted") is not True
        or receipt.get("error") != ""
        or receipt.get("close_kwargs") != close_kwargs
        or receipt.get("runtime_version") != runtime_version
    ):
        problems.append("close_requested receipt invalid")
    else:
        for key, expected in identity.items():
            if receipt.get(key) != expected:
                problems.append(f"close_requested receipt {key} mismatch")
    returned_receipt = result.get("close_returned_receipt")
    if close_returned:
        if not isinstance(returned_receipt, Mapping) or (
            returned_receipt.get("type") != "close_receipt"
            or returned_receipt.get("close_event_type") != "close_returned"
            or returned_receipt.get("request_id") != request_id
            or returned_receipt.get("mode") != "fast"
            or returned_receipt.get("received") is not True
            or returned_receipt.get("accepted") is not True
            or returned_receipt.get("error") != ""
            or returned_receipt.get("close_kwargs") != close_kwargs
            or returned_receipt.get("runtime_version") != runtime_version
        ):
            problems.append("close_returned receipt invalid")
        else:
            for key, expected in identity.items():
                if returned_receipt.get(key) != expected:
                    problems.append(f"close_returned receipt {key} mismatch")
    if problems:
        raise RunnerContractError("fast worker close was not verified: " + "; ".join(problems))
    return result


def _build_production_worker_args(cli_args: argparse.Namespace, request_path: Path) -> argparse.Namespace:
    """Start from the official UI defaults; overlay only non-physics launch knobs."""

    from height_replay_ui import build_parser as build_ui_parser
    from height_replay_ui import normalize_motion_args

    worker_args = build_ui_parser().parse_args([])
    worker_args.height_mm = 50
    worker_args.height_cm = None
    worker_args.profile = "fast"
    worker_args.render_interval = 8  # official GUI 120 Hz / 8 = 15 fps
    worker_args.headless = False
    worker_args.no_sim = False
    worker_args.sim_launch_mode = "subprocess"
    worker_args.fsm50_gate_request_path = ""
    worker_args.fsm50_task_request_path = str(request_path.resolve())
    for name in (
        "device",
        "worker_launch_mode",
        "worker_python_exe",
        "isaaclab_bat",
        "sim_startup_timeout_s",
        "sim_worker_status_timeout_s",
        "sim_worker_log_lines",
        "accept_isaac_eula",
        "livestream",
        "experience",
    ):
        if hasattr(cli_args, name):
            setattr(worker_args, name, getattr(cli_args, name))
    normalize_motion_args(worker_args)
    return worker_args


def _run_manifest(
    *,
    prepared: PreparedReplay,
    paths: RunPaths,
    request: Mapping[str, Any],
    terminal: Mapping[str, Any] | None,
    assessment: TaskSuccessAssessment,
    shutdown_outcome: Mapping[str, Any],
    shutdown_verified: bool,
    error: str,
    classifier_inputs_path: Path | None = None,
    worker_ready_status: Mapping[str, Any] | None = None,
    worker_playback_start_ack: Mapping[str, Any] | None = None,
    worker_task_replay_terminal: Mapping[str, Any] | None = None,
    worker_task_replay_ack: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    terminal = dict(terminal or {})
    return {
        "schema_version": RUNNER_RESULT_SCHEMA,
        "created_utc": _utc_now(),
        "execution_path": "production_ui_worker_fast_replay",
        "source_version": prepared.recording.version_id,
        "accepted_steps_path": str(prepared.recording.accepted_steps_path),
        "accepted_steps_sha256": prepared.accepted_steps_sha256,
        "plan_sha256": str(prepared.plan.plan_sha256).lower(),
        "request_id": str(request.get("request_id", "") or ""),
        "plan_id": str(request.get("plan_id", "") or ""),
        "run_dir": str(paths.run_dir),
        "request_path": str(paths.request_path),
        "request_sha256": sha256_file(paths.request_path),
        "task_inputs_path": str(terminal.get("task_inputs_path", "") or ""),
        "task_inputs_sha256": (
            sha256_file(Path(str(terminal.get("task_inputs_path", "") or "")))
            if str(terminal.get("task_inputs_path", "") or "")
            and Path(str(terminal.get("task_inputs_path", "") or "")).is_file()
            else ""
        ),
        "classifier_inputs_path": (
            str(classifier_inputs_path.resolve())
            if classifier_inputs_path is not None
            else ""
        ),
        "classifier_inputs_sha256": (
            sha256_file(classifier_inputs_path)
            if classifier_inputs_path is not None and classifier_inputs_path.is_file()
            else ""
        ),
        "worker_result_path": str(terminal.get("worker_result_path", "") or ""),
        "worker_result_sha256": (
            sha256_file(Path(str(terminal.get("worker_result_path", "") or "")))
            if str(terminal.get("worker_result_path", "") or "")
            and Path(str(terminal.get("worker_result_path", "") or "")).is_file()
            else ""
        ),
        "video_path": str(terminal.get("video_path", "") or assessment.video_path),
        "video_sha256": (
            sha256_file(Path(str(terminal.get("video_path", "") or assessment.video_path)))
            if str(terminal.get("video_path", "") or assessment.video_path)
            and Path(str(terminal.get("video_path", "") or assessment.video_path)).is_file()
            else ""
        ),
        "terminal_phase": str(terminal.get("phase", "") or ""),
        "video_writer_quiesced": terminal.get("video_writer_quiesced"),
        # Deep-frozen immediately after their complete live contracts pass,
        # so the ready/start/terminal/ACK chain survives client shutdown.
        "worker_ready_status": dict(worker_ready_status or {}),
        "worker_playback_start_ack": dict(worker_playback_start_ack or {}),
        "worker_task_replay_terminal": dict(worker_task_replay_terminal or {}),
        "worker_task_replay_ack": dict(worker_task_replay_ack or {}),
        "shutdown_verified": bool(shutdown_verified),
        "shutdown_outcome": dict(shutdown_outcome),
        "assessment": assessment.to_table_row(),
        "error": str(error or ""),
    }


def _latest_resumable_manifest(
    recording: RecordingVersion,
    output_root: Path,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any] | None:
    version_root = output_root.resolve() / recording.version_id
    if not version_root.is_dir():
        return None
    expected_source_sha = (
        sha256_file(recording.accepted_steps_path)
        if recording.accepted_steps_path.is_file()
        else ""
    )
    candidates = sorted(
        version_root.glob(f"*/{RUNNER_RESULT_NAME}"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            result = _strict_json_load(path)
            row = result.get("assessment")
            run_dir = path.parent.resolve()
            request_path = Path(str(result.get("request_path", "") or "")).resolve()
            task_inputs_path = Path(str(result.get("task_inputs_path", "") or "")).resolve()
            classifier_inputs_path = Path(
                str(result.get("classifier_inputs_path", "") or "")
            ).resolve()
            worker_result_path = Path(
                str(result.get("worker_result_path", "") or "")
            ).resolve()
            video_path = Path(str(result.get("video_path", "") or "")).resolve()
            if (
                result.get("schema_version") != RUNNER_RESULT_SCHEMA
                or result.get("source_version") != recording.version_id
                or result.get("accepted_steps_sha256") != expected_source_sha
                or result.get("plan_sha256") != expected_plan_sha256
                or result.get("shutdown_verified") is not True
                or not isinstance(row, Mapping)
                or row.get("task_result")
                not in {*_SUCCESS_RESULTS, REPLAY_TASK_FAIL, REPLAY_TASK_NOT_EVALUATED}
                or result.get("terminal_phase")
                not in {"TASK_REPLAY_COMPLETE", "TASK_REPLAY_FAILED"}
                or not task_inputs_path.is_file()
                or not classifier_inputs_path.is_file()
                or not worker_result_path.is_file()
                or not video_path.is_file()
                or not _path_within(path, version_root)
                or not _path_within(request_path, run_dir)
                or not _path_within(task_inputs_path, run_dir)
                or not _path_within(classifier_inputs_path, run_dir)
                or not _path_within(worker_result_path, run_dir)
                or not _path_within(video_path, run_dir)
            ):
                continue
            for path_key, sha_key, artifact_path in (
                ("request_path", "request_sha256", request_path),
                ("task_inputs_path", "task_inputs_sha256", task_inputs_path),
                (
                    "classifier_inputs_path",
                    "classifier_inputs_sha256",
                    classifier_inputs_path,
                ),
                ("worker_result_path", "worker_result_sha256", worker_result_path),
                ("video_path", "video_sha256", video_path),
            ):
                if str(result.get(path_key, "") or "") != str(artifact_path):
                    raise RunnerContractError(f"resume {path_key} is not canonical")
                if str(result.get(sha_key, "") or "").lower() != sha256_file(
                    artifact_path
                ):
                    raise RunnerContractError(f"resume {sha_key} is stale")
            request = _strict_json_load(request_path)
            for key, expected in {
                "schema_version": REQUEST_SCHEMA,
                "source_version": recording.version_id,
                "accepted_steps_path": str(recording.accepted_steps_path.resolve()),
                "accepted_steps_sha256": expected_source_sha,
                "plan_sha256": expected_plan_sha256,
                "request_id": result.get("request_id"),
                "plan_id": result.get("plan_id"),
                "run_dir": str(run_dir),
            }.items():
                if request.get(key) != expected:
                    raise RunnerContractError(f"resume request {key} mismatch")
            worker_result = _strict_json_load(worker_result_path)
            for key, expected in {
                "source_version": recording.version_id,
                "request_id": result.get("request_id"),
                "plan_id": result.get("plan_id"),
                "run_dir": str(run_dir),
                "task_inputs_path": str(task_inputs_path),
                "video_writer_quiesced": True,
            }.items():
                if worker_result.get(key) != expected:
                    raise RunnerContractError(f"resume worker result {key} mismatch")
            classifier_inputs = _strict_json_load(classifier_inputs_path)
            completed = classifier_inputs.get("completed_result")
            if not isinstance(completed, Mapping):
                raise RunnerContractError("resume classifier inputs are malformed")
            for key, expected in {
                "source_version": recording.version_id,
                "plan_event_count": request.get("plan_event_count"),
                "plan_segment_count": request.get("plan_segment_count"),
                "second_simulator_process_detected": False,
            }.items():
                if completed.get(key) != expected:
                    raise RunnerContractError(f"resume classifier input {key} mismatch")
            verdict: ManualVideoVerdict | None = None
            verdict_path_text = str(result.get("manual_video_verdict_path", "") or "")
            if verdict_path_text:
                verdict_path = Path(verdict_path_text).resolve()
                if (
                    not verdict_path.is_file()
                    or not _path_within(verdict_path, run_dir)
                    or str(result.get("manual_video_verdict_sha256", "") or "")
                    != sha256_file(verdict_path)
                ):
                    raise RunnerContractError("resume manual verdict is stale")
                _document, verdict = _load_bound_video_verdict(
                    verdict_path,
                    source_version=recording.version_id,
                    request_id=str(result.get("request_id", "") or ""),
                    plan_id=str(result.get("plan_id", "") or ""),
                    run_dir=run_dir,
                    task_inputs_path=classifier_inputs_path,
                    video_path=video_path,
                    require_complete=True,
                )
            current = classify_run_inputs(
                task_inputs=classifier_inputs,
                manual_video_verdict=verdict,
                second_simulator_process_detected=False,
            )
            if verdict is None:
                current = _defer_success_until_manual_review(current)
            current = current.to_table_row()
            if dict(row) != current:
                raise RunnerContractError("resume assessment no longer matches its inputs")
            return result
        except Exception:
            continue
    return None


def _attempt_from_infrastructure_failure(
    *,
    recording: RecordingVersion,
    prepared: PreparedReplay | None,
    paths: RunPaths,
    reason: str,
    process_preflight: Mapping[str, Any] | None = None,
) -> ReplayAttempt:
    assessment = _not_evaluated_assessment(
        version=recording.version_id,
        step_count=len(prepared.steps) if prepared is not None else 0,
        segment_count=(
            len(list(prepared.plan.segments)) if prepared is not None else 0
        ),
        reason=reason,
    )
    atomic_write_json(paths.run_dir / TASK_ASSESSMENT_NAME, asdict(assessment))
    classifier_inputs_path: Path | None = None
    if isinstance(process_preflight, Mapping):
        detected = process_preflight.get("second_simulator_process_detected")
        if process_preflight.get("checked") is True and type(detected) is bool:
            classifier_inputs_path = paths.run_dir / CLASSIFIER_INPUTS_NAME
            atomic_write_json(
                classifier_inputs_path,
                {
                    "runner_classifier_input_schema": (
                        "fsm50.runner_classifier_inputs.v1"
                    ),
                    "completed_result": {
                        "source_version": recording.version_id,
                        "replay_task_complete": False,
                        "artifact_valid": False,
                        "runner_infrastructure_failure": True,
                        "second_simulator_process_detected": detected,
                        "second_simulator_process_preflight": dict(
                            process_preflight
                        ),
                    },
                    "physical_evidence": {},
                    "final_telemetry_row": {},
                },
            )
    payload = {
        "schema_version": RUNNER_RESULT_SCHEMA,
        "created_utc": _utc_now(),
        "execution_path": "production_ui_worker_fast_replay",
        "source_version": recording.version_id,
        "accepted_steps_path": str(recording.accepted_steps_path),
        "accepted_steps_sha256": (
            sha256_file(recording.accepted_steps_path)
            if recording.accepted_steps_path.is_file()
            else ""
        ),
        "plan_sha256": (
            str(prepared.plan.plan_sha256).lower() if prepared is not None else ""
        ),
        "request_id": "",
        "plan_id": "",
        "run_dir": str(paths.run_dir),
        "task_inputs_path": "",
        "classifier_inputs_path": (
            str(classifier_inputs_path) if classifier_inputs_path is not None else ""
        ),
        "classifier_inputs_sha256": (
            sha256_file(classifier_inputs_path)
            if classifier_inputs_path is not None
            else ""
        ),
        "worker_result_path": "",
        "video_path": "",
        "terminal_phase": "",
        "video_writer_quiesced": False,
        "worker_ready_status": {},
        "worker_playback_start_ack": {},
        "worker_task_replay_terminal": {},
        "worker_task_replay_ack": {},
        "shutdown_verified": False,
        "shutdown_outcome": {},
        "assessment": assessment.to_table_row(),
        "error": reason,
    }
    atomic_write_json(paths.runner_result_path, payload)
    return ReplayAttempt(
        recording=recording,
        run_dir=paths.run_dir,
        assessment=assessment,
        runner_result_path=paths.runner_result_path,
        shutdown_verified=False,
    )


def run_one_replay(
    recording: RecordingVersion,
    *,
    cli_args: argparse.Namespace,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    plan_builder: Callable[..., Any] | None = None,
    process_preflight: Mapping[str, Any],
) -> ReplayAttempt:
    """Launch exactly one production worker and classify its durable inputs."""

    if (
        process_preflight.get("checked") is not True
        or process_preflight.get("second_simulator_process_detected") is not False
        or process_preflight.get("conflict_count") != 0
    ):
        raise RunnerContractError(
            "run_one_replay requires a measured no-conflict process preflight"
        )
    prepared = prepare_replay(recording, plan_builder=plan_builder)
    paths = allocate_run_paths(cli_args.output_root, recording.version_id)
    request_id = uuid.uuid4().hex
    plan_id = f"fsm50-task-{recording.short_id}-{uuid.uuid4().hex[:12]}"
    request = build_worker_task_request(
        prepared,
        paths,
        request_id=request_id,
        plan_id=plan_id,
        telemetry_hz=float(cli_args.telemetry_hz),
        video_fps=DEFAULT_VIDEO_FPS,
        post_run_settle_s=float(cli_args.post_run_settle_s),
        timeout_s=float(getattr(cli_args, "task_timeout_s", 0.0) or 0.0) or None,
    )
    atomic_write_json(paths.request_path, request)
    if client_factory is None:
        from sim_process_client import SimProcessClient

        client_factory = SimProcessClient
    worker_args = _build_production_worker_args(cli_args, paths.request_path)
    client = client_factory(worker_args)
    terminal: dict[str, Any] | None = None
    shutdown_outcome: dict[str, Any] = {}
    shutdown_verified = False
    task_inputs: dict[str, Any] | None = None
    classifier_inputs_path: Path | None = None
    assessment: TaskSuccessAssessment | None = None
    error = ""
    worker_binding: dict[str, Any] = {}
    worker_ready_status: dict[str, Any] = {}
    worker_playback_start_ack: dict[str, Any] = {}
    worker_task_replay_terminal: dict[str, Any] = {}
    worker_task_replay_ack: dict[str, Any] = {}
    try:
        client.start()
        ready = wait_for_worker_ready(
            client, timeout_s=float(cli_args.sim_startup_timeout_s)
        )
        ready = validate_task_worker_binding(ready, request=request)
        worker_binding = {
            "worker_pid": ready.get("worker_pid"),
            "worker_session_id": ready.get("worker_session_id"),
            "adapter_runtime_instance_id": ready.get(
                "adapter_runtime_instance_id"
            ),
            "root_state_write_count": ready.get("root_state_write_count"),
        }
        if worker_binding["worker_pid"] != getattr(client, "pid", None):
            raise RunnerContractError("ready status PID differs from owned client PID")
        worker_ready_status = json.loads(json.dumps(ready, allow_nan=False))
        worker_session_id = str(ready["worker_session_id"])
        from playback import playback_plan_to_payload

        client.start_playback_plan(
            playback_plan_to_payload(prepared.plan),
            start_delay_sim_s=0.0,
            plan_id=plan_id,
            request_id=request_id,
            plan_sha256=str(prepared.plan.plan_sha256),
            worker_session_id=worker_session_id,
        )
        start_ack = wait_for_operation_ack(
            client,
            operation="start_playback_plan",
            request_id=request_id,
            timeout_s=float(cli_args.operation_timeout_s),
        )
        _validate_operation_ack(
            start_ack, operation="start_playback_plan", request_id=request_id
        )
        for key, expected in {
            "plan_id": plan_id,
            "plan_sha256": str(prepared.plan.plan_sha256),
            "profile": "motion_only",
            "event_count": len(list(prepared.plan.events)),
            "segment_count": len(list(prepared.plan.segments)),
            "input_step_count": len(prepared.steps),
            "worker_pid": worker_binding["worker_pid"],
            "worker_session_id": worker_session_id,
            "adapter_runtime_instance_id": worker_binding[
                "adapter_runtime_instance_id"
            ],
            "artifact_request_id": "",
            "root_state_write_count": 0,
            "motion_start_ready": True,
        }.items():
            if start_ack.get(key) != expected:
                raise RunnerContractError(
                    f"playback start ACK {key} mismatch: {start_ack.get(key)!r}"
                )
        worker_playback_start_ack = json.loads(
            json.dumps(start_ack, allow_nan=False)
        )
        terminal = wait_for_task_terminal(
            client,
            request_id=request_id,
            timeout_s=float(cli_args.terminal_timeout_s),
        )
        terminal = validate_task_terminal(
            terminal,
            request_id=request_id,
            plan_id=plan_id,
            source_version=recording.version_id,
            run_dir=paths.run_dir,
        )
        worker_task_replay_terminal = json.loads(
            json.dumps(terminal, allow_nan=False)
        )
        task_ack = wait_for_operation_ack(
            client,
            operation="task_replay",
            request_id=request_id,
            timeout_s=float(cli_args.operation_timeout_s),
        )
        task_ack = validate_task_operation_ack(
            task_ack,
            terminal=terminal,
            request_id=request_id,
            plan_id=plan_id,
            source_version=recording.version_id,
            run_dir=paths.run_dir,
            worker_binding=worker_binding,
        )
        worker_task_replay_ack = json.loads(json.dumps(task_ack, allow_nan=False))
        task_inputs_path, task_inputs, video_path = _task_inputs_from_terminal(
            terminal, paths.run_dir
        )
        task_inputs = build_classifier_inputs(
            task_inputs,
            process_preflight=process_preflight,
        )
        classifier_inputs_path = paths.run_dir / CLASSIFIER_INPUTS_NAME
        atomic_write_json(classifier_inputs_path, task_inputs)
        template = _video_verdict_template(
            source_version=recording.version_id,
            request_id=request_id,
            plan_id=plan_id,
            run_dir=paths.run_dir,
            task_inputs_path=classifier_inputs_path,
            video_path=video_path,
        )
        atomic_write_json(paths.run_dir / VIDEO_VERDICT_TEMPLATE_NAME, template)
        assessment = classify_run_inputs(
            task_inputs=task_inputs,
            manual_video_verdict=None,
            second_simulator_process_detected=False,
        )
        assessment = _defer_success_until_manual_review(assessment)
        atomic_write_json(paths.run_dir / TASK_ASSESSMENT_NAME, asdict(assessment))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        process = getattr(client, "process", None)
        if process is not None:
            try:
                # An exception can race the durable terminal delivery.  Recover
                # only the raw task terminal (never the convenience operation
                # ACK), then re-run the full identity/path/quiescence contract.
                recovered = _latest_raw_task_terminal(
                    client,
                    request_id=request_id,
                )
                candidate = recovered if recovered is not None else terminal
                terminal_finalized = False
                if candidate is not None:
                    try:
                        terminal = validate_task_terminal(
                            candidate,
                            request_id=request_id,
                            plan_id=plan_id,
                            source_version=recording.version_id,
                            run_dir=paths.run_dir,
                        )
                        worker_task_replay_terminal = json.loads(
                            json.dumps(terminal, allow_nan=False)
                        )
                        terminal_finalized = True
                    except Exception as terminal_exc:
                        terminal_error = (
                            f"terminal recovery rejected: {type(terminal_exc).__name__}: "
                            f"{terminal_exc}"
                        )
                        error = f"{error}; {terminal_error}" if error else terminal_error
                shutdown_request_id = uuid.uuid4().hex
                owned_worker_pid = getattr(client, "pid", None)
                shutdown_outcome = dict(
                    client.shutdown(
                        # A task worker rejects normal close.  If no sealed
                        # terminal can be proved, this bounded fast request is
                        # rejected and the client owns the process-tree cleanup.
                        mode="fast",
                        timeout_s=60.0,
                        request_id=shutdown_request_id,
                    )
                    or {}
                )
                if terminal_finalized:
                    validate_fast_shutdown(
                        shutdown_outcome,
                        request_id=shutdown_request_id,
                        owned_worker_pid=(
                            int(owned_worker_pid)
                            if type(owned_worker_pid) is int
                            else None
                        ),
                        task_request_id=request_id,
                        worker_session_id=str(
                            worker_binding.get("worker_session_id", "") or ""
                        ),
                        adapter_runtime_instance_id=str(
                            worker_binding.get(
                                "adapter_runtime_instance_id", ""
                            )
                            or ""
                        ),
                    )
                    if (
                        process is None
                        or process.poll() != 0
                        or process.pid != owned_worker_pid
                    ):
                        raise RunnerContractError(
                            "owned worker PID did not disappear with returncode 0"
                        )
                    shutdown_verified = True
                else:
                    shutdown_verified = False
            except Exception as exc:
                close_error = f"{type(exc).__name__}: {exc}"
                error = f"{error}; {close_error}" if error else close_error
                shutdown_verified = False
        try:
            client.close()
        except Exception:
            pass

    if assessment is None:
        assessment = _not_evaluated_assessment(
            version=recording.version_id,
            step_count=len(prepared.steps),
            segment_count=len(list(prepared.plan.segments)),
            reason="PRODUCTION_RUN_INCOMPLETE: " + (error or "unknown worker failure"),
            video_path=str((terminal or {}).get("video_path", "") or ""),
        )
        atomic_write_json(paths.run_dir / TASK_ASSESSMENT_NAME, asdict(assessment))
    if not shutdown_verified:
        assessment = _not_evaluated_assessment(
            version=recording.version_id,
            step_count=assessment.step_count,
            segment_count=assessment.fast_segment_count,
            reason="WORKER_SHUTDOWN_NOT_VERIFIED: " + (error or "close contract failed"),
            video_path=assessment.video_path,
        )
        atomic_write_json(paths.run_dir / TASK_ASSESSMENT_NAME, asdict(assessment))
    runner_result = _run_manifest(
        prepared=prepared,
        paths=paths,
        request=request,
        terminal=terminal,
        assessment=assessment,
        shutdown_outcome=shutdown_outcome,
        shutdown_verified=shutdown_verified,
        error=error,
        classifier_inputs_path=classifier_inputs_path,
        worker_ready_status=worker_ready_status,
        worker_playback_start_ack=worker_playback_start_ack,
        worker_task_replay_terminal=worker_task_replay_terminal,
        worker_task_replay_ack=worker_task_replay_ack,
    )
    atomic_write_json(paths.runner_result_path, runner_result)
    return ReplayAttempt(
        recording=recording,
        run_dir=paths.run_dir,
        assessment=assessment,
        runner_result_path=paths.runner_result_path,
        shutdown_verified=shutdown_verified,
    )


def review_run(
    *,
    run_dir: str | Path,
    verdict_path: str | Path,
    recording_root: str | Path = DEFAULT_RECORDING_ROOT,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
) -> TaskSuccessAssessment:
    """Validate a SHA-bound manual review and atomically reclassify one run."""

    run = Path(run_dir).resolve()
    manifest_path = run / RUNNER_RESULT_NAME
    manifest = _strict_json_load(manifest_path)
    if manifest.get("schema_version") != RUNNER_RESULT_SCHEMA:
        raise RunnerContractError("run has no compatible frontend manifest")
    if manifest.get("shutdown_verified") is not True:
        raise RunnerContractError("run shutdown is not verified; review cannot promote it")
    source_version = str(manifest.get("source_version", "") or "")
    request_id = str(manifest.get("request_id", "") or "")
    plan_id = str(manifest.get("plan_id", "") or "")
    task_inputs_path = Path(
        str(manifest.get("classifier_inputs_path", "") or "")
    ).resolve()
    video_path = Path(str(manifest.get("video_path", "") or "")).resolve()
    recordings = enumerate_recording_versions(recording_root)
    matches = [item for item in recordings if item.version_id == source_version]
    if len(matches) != 1:
        raise RunnerContractError("reviewed run no longer has exactly one source recording")
    source_recording = matches[0]
    current_source_sha = sha256_file(source_recording.accepted_steps_path)
    if manifest.get("accepted_steps_sha256") != current_source_sha:
        raise RunnerContractError("reviewed run source recording SHA-256 is stale")
    current_plan = prepare_replay(source_recording)
    if manifest.get("plan_sha256") != str(current_plan.plan.plan_sha256).lower():
        raise RunnerContractError("reviewed run plan no longer matches production compiler")
    if (
        str(manifest.get("classifier_inputs_sha256", "") or "")
        != sha256_file(task_inputs_path)
        or str(manifest.get("video_sha256", "") or "") != sha256_file(video_path)
    ):
        raise RunnerContractError("reviewed run classifier/video artifacts are stale")
    document, verdict = _load_bound_video_verdict(
        verdict_path,
        source_version=source_version,
        request_id=request_id,
        plan_id=plan_id,
        run_dir=run,
        task_inputs_path=task_inputs_path,
        video_path=video_path,
        require_complete=True,
    )
    assert verdict is not None
    task_inputs = _strict_json_load(task_inputs_path)
    assessment = classify_run_inputs(
        task_inputs=task_inputs,
        manual_video_verdict=verdict,
        second_simulator_process_detected=False,
    )
    if assessment.version != source_version:
        raise RunnerContractError("classifier source_version differs from reviewed run")
    # Persist the validated, bound review inside the run before promotion.
    atomic_write_json(run / VIDEO_VERDICT_NAME, document)
    if _strict_json_load(run / VIDEO_VERDICT_NAME) != document:
        raise RunnerContractError("manual video verdict strict reread mismatch")
    atomic_write_json(run / TASK_ASSESSMENT_NAME, asdict(assessment))
    manifest["assessment"] = assessment.to_table_row()
    manifest["manual_video_verdict_path"] = str(run / VIDEO_VERDICT_NAME)
    manifest["manual_video_verdict_sha256"] = sha256_file(run / VIDEO_VERDICT_NAME)
    manifest["reviewed_utc"] = str(document["reviewed_utc"])
    atomic_write_json(manifest_path, manifest)
    update_task_success_tables(
        report_root=report_root,
        recordings=recordings,
        updates=[assessment.to_table_row()],
    )
    return assessment


def _process_guard_dependencies() -> tuple[Any, Callable[[], list[dict[str, Any]]], Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]]]:
    # Reuse the already audited singleton/process detector; do not maintain a
    # second, subtly different Isaac process recognizer in this thin runner.
    from .run_fsm50 import (
        ReplaySingletonLock,
        _existing_simulator_processes,
        _os_process_snapshot,
    )

    return ReplaySingletonLock, _os_process_snapshot, _existing_simulator_processes


def _assert_no_simulator_process(
    snapshot_fn: Callable[[], list[dict[str, Any]]],
    conflict_fn: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]],
) -> dict[str, Any]:
    checked_utc = _utc_now()
    snapshot = snapshot_fn()
    conflicts = conflict_fn(snapshot)
    compact = [
        {
            "pid": row.get("pid"),
            "name": row.get("name"),
            "command_line": row.get("command_line"),
        }
        for row in conflicts
    ]
    canonical_snapshot = json.dumps(
        _jsonable(snapshot),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    evidence = {
        "schema_version": "fsm50.second_simulator_preflight.v1",
        "checked": True,
        "checked_utc": checked_utc,
        "snapshot_process_count": len(snapshot),
        "process_snapshot_sha256": hashlib.sha256(
            canonical_snapshot.encode("utf-8")
        ).hexdigest(),
        "conflict_count": len(compact),
        "conflicts": compact,
        "second_simulator_process_detected": bool(compact),
    }
    if compact:
        raise SimulatorProcessConflictError(
            "SECOND_SIMULATOR_PROCESS: "
            + json.dumps(compact, ensure_ascii=False),
            evidence=evidence,
        )
    return evidence


def run_selected(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    plan_builder: Callable[..., Any] | None = None,
    lock_factory: Callable[[], Any] | None = None,
    process_snapshot_fn: Callable[[], list[dict[str, Any]]] | None = None,
    conflict_detector: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> list[ReplayAttempt]:
    recordings = enumerate_recording_versions(args.recording_root)
    selected = select_recording_versions(recordings, args.versions)
    if lock_factory is None or process_snapshot_fn is None or conflict_detector is None:
        default_lock, default_snapshot, default_conflicts = _process_guard_dependencies()
        lock_factory = lock_factory or default_lock
        process_snapshot_fn = process_snapshot_fn or default_snapshot
        conflict_detector = conflict_detector or default_conflicts

    attempts: list[ReplayAttempt] = []
    resumable: dict[str, dict[str, Any]] = {}
    if bool(args.resume):
        for recording in selected:
            prepared_for_resume = prepare_replay(
                recording,
                plan_builder=plan_builder,
            )
            result = _latest_resumable_manifest(
                recording,
                Path(args.output_root),
                expected_plan_sha256=str(prepared_for_resume.plan.plan_sha256).lower(),
            )
            if result is not None:
                resumable[recording.version_id] = result
    if resumable:
        update_task_success_tables(
            report_root=args.report_root,
            recordings=recordings,
            updates=[
                dict(result["assessment"])
                for result in resumable.values()
            ],
        )
    for recording in selected:
        if recording.version_id in resumable:
            continue
        paths: RunPaths | None = None
        prepared: PreparedReplay | None = None
        process_preflight: Mapping[str, Any] | None = None
        lock = lock_factory()
        try:
            lock.acquire()
            # Required before every launch, not merely once per batch.
            process_preflight = _assert_no_simulator_process(
                process_snapshot_fn, conflict_detector
            )
            attempt = run_one_replay(
                recording,
                cli_args=args,
                client_factory=client_factory,
                plan_builder=plan_builder,
                process_preflight=process_preflight,
            )
        except Exception as exc:
            if paths is None:
                paths = allocate_run_paths(args.output_root, recording.version_id)
            reason = f"{type(exc).__name__}: {exc}"
            failure_preflight = (
                exc.evidence
                if isinstance(exc, SimulatorProcessConflictError)
                else process_preflight
            )
            attempt = _attempt_from_infrastructure_failure(
                recording=recording,
                prepared=prepared,
                paths=paths,
                reason=reason,
                process_preflight=failure_preflight,
            )
        finally:
            lock.release()
        if bool(getattr(lock, "acquired", False)):
            raise RunnerContractError("ReplaySingletonLock did not release")
        attempts.append(attempt)
        update_task_success_tables(
            report_root=args.report_root,
            recordings=recordings,
            updates=[attempt.assessment.to_table_row()],
        )
        # A new worker is legal only after the prior PID returned zero and
        # disappeared, and only after the singleton was released above.
        if not attempt.shutdown_verified:
            break
        _assert_no_simulator_process(process_snapshot_fn, conflict_detector)
        if not bool(args.continue_on_error) and attempt.assessment.task_result not in _SUCCESS_RESULTS:
            break
    return attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/review 50 mm recordings through the production UI worker Fast path."
    )
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="Capture one or more production Fast Replays.")
    run.add_argument("--versions", nargs="+", default=["v003"])
    run.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--continue-on-error", action="store_true", default=True)
    run.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    run.add_argument("--telemetry-hz", type=float, default=DEFAULT_TELEMETRY_HZ)
    run.add_argument("--post-run-settle-s", type=float, default=DEFAULT_POST_SETTLE_S)
    run.add_argument("--task-timeout-s", type=float, default=0.0)
    # Historical GUI+viewport captures can run close to an hour of wall time
    # even though the production plan is about 80 simulation seconds.
    run.add_argument("--terminal-timeout-s", type=float, default=7200.0)
    run.add_argument("--operation-timeout-s", type=float, default=30.0)
    run.add_argument("--sim-startup-timeout-s", type=float, default=600.0)
    run.add_argument("--sim-worker-status-timeout-s", type=float, default=10.0)
    run.add_argument("--sim-worker-log-lines", type=int, default=200)
    run.add_argument(
        "--worker-launch-mode",
        choices=("auto", "current-python", "isaaclab-bat", "explicit-python"),
        default="auto",
    )
    run.add_argument("--worker-python-exe", default="")
    run.add_argument("--isaaclab-bat", default="C:/robotics_sim/IsaacLab/isaaclab.bat")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--livestream", type=int, default=0)
    run.add_argument("--experience", default="")
    run.add_argument("--accept-isaac-eula", action="store_true", default=False)

    review = subparsers.add_parser(
        "review", help="Apply a strict SHA-bound manual viewport-video verdict."
    )
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--verdict", type=Path, required=True)
    review.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    review.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {None, "run"}:
        if args.command is None:
            # ``run`` is explicit to prevent a typo from unexpectedly starting
            # Isaac.  argparse's error is clearer than silently launching.
            build_parser().error("a command is required: run or review")
        attempts = run_selected(args)
        summary = {
            "selected": [attempt.recording.version_id for attempt in attempts],
            "results": [attempt.assessment.to_table_row() for attempt in attempts],
            "run_dirs": [str(attempt.run_dir) for attempt in attempts],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if attempts and all(
            attempt.assessment.task_result in _SUCCESS_RESULTS
            for attempt in attempts
        ) else 1
    if args.command == "review":
        assessment = review_run(
            run_dir=args.run_dir,
            verdict_path=args.verdict,
            recording_root=args.recording_root,
            report_root=args.report_root,
        )
        print(json.dumps(assessment.to_table_row(), ensure_ascii=False, indent=2))
        return 0 if assessment.task_result in _SUCCESS_RESULTS else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_REPORT_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RECORDING_ROOT",
    "DEFAULT_REPORT_ROOT",
    "MARKDOWN_REPORT_NAME",
    "RUNNER_RESULT_NAME",
    "RUNNER_RESULT_SCHEMA",
    "RunnerContractError",
    "VIDEO_VERDICT_NAME",
    "VIDEO_VERDICT_SCHEMA",
    "VIDEO_VERDICT_TEMPLATE_NAME",
    "RecordingVersion",
    "ReplayAttempt",
    "RunPaths",
    "allocate_run_paths",
    "atomic_write_json",
    "build_parser",
    "build_worker_task_request",
    "classify_run_inputs",
    "enumerate_recording_versions",
    "main",
    "prepare_replay",
    "review_run",
    "run_one_replay",
    "run_selected",
    "select_recording_versions",
    "sha256_file",
    "update_task_success_tables",
    "validate_fast_shutdown",
    "version_sort_key",
]
