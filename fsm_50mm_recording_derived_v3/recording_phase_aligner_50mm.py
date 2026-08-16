"""Evidence-bound, Isaac-free phase alignment for successful 50 mm replays.

This module consumes only finalized artifacts produced by the production
``height_replay_ui -> worker -> PlaybackController -> SimRobotAdapter`` path.
It deliberately refuses to emit reports until every physically present 50 mm
recording has an ``EVALUATED`` Gate-A result.  Failed replays are retained as
explicit exclusions, while only task-success runs contribute motion profiles.

The normal Gate-A telemetry has geometry-only wheel classes and no measured
COM or contact load.  Consequently:

* ``candidate_support_legs`` are geometry support candidates, not load claims;
* COM direction is a hypothesis accompanied by measured base translation;
* functional phase windows may overlap when one command serves two purposes;
* strategy clustering is categorical and never averages unlike actions.

No Isaac module is imported or launched here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .recording_alignment import (
    enumerate_recording_directories,
    load_max_wheel_speed,
    load_steps_jsonl,
)
from .recording_fast_plan import fast_plan_rows


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_RECORDING_ROOT = (
    PROJECT_ROOT / "saved_height_steps_fsm_reference_v2" / "height_050mm"
)
DEFAULT_RUN_ROOT = PACKAGE_ROOT / "runs" / "50mm_fast_replay"
DEFAULT_REPORT_ROOT = PACKAGE_ROOT / "reports"

SUCCESS_RESULTS = frozenset(
    {"REPLAY_TASK_SUCCESS", "REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE"}
)
LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_JOINT_PREFIX = {
    "FR": "front_right_",
    "FL": "front_left_",
    "RR": "rear_right_",
    "RL": "rear_left_",
}
SUPPORT_CLASSES = frozenset({"GROUND", "TOP"})

PHASE_ORDER = (
    "INITIAL_APPROACH",
    "PRE_FR_COM_SHIFT",
    "FR_UNLOAD_AND_LIFT",
    "FR_FACE_CROSS",
    "FR_TOP_PLACE",
    "FL_UNLOAD_AND_LIFT",
    "FL_FACE_CROSS",
    "FL_TOP_PLACE",
    "FRONT_PAIR_ADVANCE",
    "PRE_RR_COM_SHIFT",
    "RR_UNLOAD_AND_LIFT",
    "RR_FACE_CROSS",
    "RR_TOP_PLACE",
    "PRE_RL_SUPPORT_SETUP",
    "PRE_RL_COM_SHIFT",
    "RL_UNLOAD_AND_LIFT",
    "RL_FACE_CROSS",
    "RL_TOP_PLACE",
    "FINAL_ADVANCE",
    "FINAL_POSTURE_RECOVERY",
)

ACTIVE_LEG = {
    phase: next((leg for leg in LEG_ORDER if phase.startswith(f"{leg}_")), "")
    for phase in PHASE_ORDER
}
ACTIVE_LEG.update(
    {
        "PRE_FR_COM_SHIFT": "FR",
        "PRE_RR_COM_SHIFT": "RR",
        "PRE_RL_COM_SHIFT": "RL",
    }
)

COM_CANDIDATE = {
    "INITIAL_APPROACH": "FORWARD",
    "PRE_FR_COM_SHIFT": "TOWARD_RL_CANDIDATE",
    "FRONT_PAIR_ADVANCE": "FORWARD",
    "PRE_RR_COM_SHIFT": "TOWARD_FL_OR_FRONT_SUPPORT_CANDIDATE",
    "PRE_RL_SUPPORT_SETUP": "SUPPORT_WORKSPACE_CANDIDATE",
    "PRE_RL_COM_SHIFT": "TOWARD_FR_OR_DIAGONAL_SUPPORT_CANDIDATE",
    "FINAL_ADVANCE": "FORWARD",
    "FINAL_POSTURE_RECOVERY": "HOME_LIKE_RECOVERY_CANDIDATE",
}

ENTRY_COMPLETION = {
    "INITIAL_APPROACH": (
        "first pre-FR nonzero wheel command",
        "end of the selected pre-FR wheel-command run",
    ),
    "PRE_FR_COM_SHIFT": (
        "production replay begins",
        "first pre-FR wheel-command run begins",
    ),
    "FRONT_PAIR_ADVANCE": (
        "FL front-face crossing has occurred",
        "selected wheel-command run before RR final unload ends",
    ),
    "PRE_RR_COM_SHIFT": (
        "FL reaches stable geometric TOP after crossing",
        "RR final geometry-support departure before crossing",
    ),
    "PRE_RL_SUPPORT_SETUP": (
        "RR reaches stable geometric TOP after crossing",
        "first RL servo-target segment in the pre-lift window",
    ),
    "PRE_RL_COM_SHIFT": (
        "first RL servo-target segment in the pre-lift window",
        "RL final geometry-support departure before crossing",
    ),
    "FINAL_ADVANCE": (
        "RL front-face crossing has occurred",
        "selected post-RL wheel-command run ends",
    ),
    "FINAL_POSTURE_RECOVERY": (
        "final advance ends or RL reaches stable geometric TOP",
        "production scheduler completes",
    ),
}
for _leg in LEG_ORDER:
    ENTRY_COMPLETION[f"{_leg}_UNLOAD_AND_LIFT"] = (
        f"{_leg} final geometry-support departure before crossing",
        f"{_leg} first sufficient top-height clearance, else pre-crossing clearance peak",
    )
    ENTRY_COMPLETION[f"{_leg}_FACE_CROSS"] = (
        f"{_leg} lift-clearance landmark",
        f"{_leg} wheel center crosses obstacle front-face plane",
    )
    ENTRY_COMPLETION[f"{_leg}_TOP_PLACE"] = (
        f"{_leg} front-face crossing",
        f"{_leg} first geometric TOP after crossing with nonnegative face clearance",
    )

CSV_COLUMNS = (
    "version",
    "task_result",
    "strategy_profile",
    "phase",
    "phase_status",
    "evidence_basis",
    "functional_windows_may_overlap",
    "run_dir",
    "accepted_steps_sha256",
    "plan_sha256",
    "task_inputs_sha256",
    "telemetry_sha256",
    "video_sha256",
    "replay_start_s",
    "replay_end_s",
    "duration_s",
    "source_step_range",
    "fast_segment_range",
    "active_leg",
    "candidate_support_legs",
    "geometry_support_fraction_json",
    "support_evidence_status",
    "servo_commands_json",
    "servo_target_range_deg_json",
    "wheel_commands_json",
    "wheel_target_range_rad_s_json",
    "servo_wheel_concurrent",
    "concurrent_segment_count",
    "candidate_com_target_direction",
    "observed_base_delta_m_json",
    "com_evidence_status",
    "entry_event",
    "completion_event",
    "active_wheel_clearance_range_m_json",
    "peak_abs_roll_rad",
    "peak_abs_pitch_rad",
    "suitable_for_first_fsm",
    "notes",
)


class EvidenceError(RuntimeError):
    """A finalized artifact contradicts or cannot support the analysis."""


class GateAIncompleteError(EvidenceError):
    """At least one physical recording lacks a finalized Gate-A verdict."""


@dataclass(frozen=True)
class LoadedRun:
    version: str
    recording_dir: Path
    run_dir: Path
    result: dict[str, Any]
    task_result: str
    plan_rows: tuple[dict[str, Any], ...]
    telemetry: tuple[dict[str, Any], ...]
    task_inputs: dict[str, Any]
    accepted_steps_sha256: str
    plan_sha256: str
    task_inputs_sha256: str
    telemetry_sha256: str
    video_sha256: str

    @property
    def successful(self) -> bool:
        return self.task_result in SUCCESS_RESULTS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _json_cell(value: Any) -> str:
    return _canonical_json(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite")
    return result


def _same_path(left: Any, right: Path) -> bool:
    try:
        return os.path.normcase(str(Path(str(left)).resolve())) == os.path.normcase(
            str(right.resolve())
        )
    except (OSError, TypeError, ValueError):
        return False


def _require_hash(path: Path, expected: Any, label: str) -> str:
    expected_text = str(expected or "").lower()
    actual = _sha256_file(path)
    if len(expected_text) != 64 or actual != expected_text:
        raise EvidenceError(
            f"{label} SHA mismatch: expected={expected_text!r} actual={actual}"
        )
    return actual


def _latest_run_dir(version: str, run_root: Path) -> Path:
    version_root = Path(run_root) / version
    if not version_root.is_dir():
        raise GateAIncompleteError(f"Gate A has no run directory for {version}")
    candidates = sorted(
        (path for path in version_root.iterdir() if path.is_dir()), reverse=True
    )
    if not candidates:
        raise GateAIncompleteError(f"Gate A has no run for {version}")
    latest = candidates[0]
    if not (latest / "fsm50_task_replay_runner_result.json").is_file():
        raise GateAIncompleteError(
            f"latest Gate-A run is not finalized for {version}: {latest}"
        )
    return latest


def _load_telemetry(
    path: Path,
    *,
    version: str,
    segment_count: int,
    task_final_row: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceError(f"cannot read telemetry {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"invalid telemetry JSON at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise EvidenceError(f"telemetry row is not an object at {path}:{number}")
        if row.get("schema_version") != "fsm50.minimal_task_telemetry.v1":
            raise EvidenceError(f"unexpected telemetry schema at {path}:{number}")
        if row.get("source_version") != version:
            raise EvidenceError(f"telemetry source mismatch at {path}:{number}")
        if not bool(row.get("robot_state_finite", False)):
            raise EvidenceError(f"non-finite robot state at {path}:{number}")
        for key in ("sim_time_s", "base_roll_rad", "base_pitch_rad"):
            _finite(row.get(key), f"telemetry[{number}].{key}")
        base = row.get("base_position_m")
        if not isinstance(base, dict):
            raise EvidenceError(f"missing base_position_m at {path}:{number}")
        for axis in ("x", "y", "z"):
            _finite(base.get(axis), f"telemetry[{number}].base.{axis}")
        classes = row.get("wheel_contact_classes")
        face = row.get("wheel_front_face_clearance_m")
        top = row.get("wheel_top_clearance_m")
        centers = row.get("wheel_center_w_m")
        if not all(isinstance(value, dict) for value in (classes, face, top, centers)):
            raise EvidenceError(f"missing wheel geometry at {path}:{number}")
        for leg in LEG_ORDER:
            if not str(classes.get(leg, "")):
                raise EvidenceError(f"missing {leg} wheel class at {path}:{number}")
            _finite(face.get(leg), f"telemetry[{number}].{leg}.face")
            _finite(top.get(leg), f"telemetry[{number}].{leg}.top")
            center = centers.get(leg)
            if not isinstance(center, list) or len(center) != 3:
                raise EvidenceError(f"invalid {leg} wheel center at {path}:{number}")
            for index, value in enumerate(center):
                _finite(value, f"telemetry[{number}].{leg}.center[{index}]")
        rows.append(row)
    if len(rows) < 2:
        raise EvidenceError(f"telemetry has fewer than two rows: {path}")

    previous_time = -math.inf
    previous_step = -1
    previous_segment = -1
    previous_event = -1
    for number, row in enumerate(rows, start=1):
        sim_time = float(row["sim_time_s"])
        sim_step = int(row.get("sim_step", -1))
        segment = int(row.get("segment_cursor", -1))
        event = int(row.get("event_cursor", -1))
        if sim_time < previous_time or sim_step < previous_step:
            raise EvidenceError(f"telemetry time/step regressed at row {number}")
        if segment < previous_segment or event < previous_event:
            raise EvidenceError(f"telemetry cursor regressed at row {number}")
        if not 0 <= segment <= segment_count:
            raise EvidenceError(f"telemetry segment cursor out of range at row {number}")
        previous_time, previous_step = sim_time, sim_step
        previous_segment, previous_event = segment, event

    final = rows[-1]
    if final.get("scheduler_state") != "COMPLETED":
        raise EvidenceError(f"successful telemetry does not end COMPLETED: {path}")
    if int(final.get("segment_cursor", -1)) != int(segment_count):
        raise EvidenceError(f"successful telemetry did not consume every segment: {path}")
    for key in ("source_version", "sim_step", "segment_cursor", "event_cursor"):
        if final.get(key) != task_final_row.get(key):
            raise EvidenceError(f"telemetry/task_inputs final-row mismatch for {key}")
    return tuple(rows)


def _load_one_run(
    recording_dir: Path,
    *,
    run_root: Path,
    max_wheel_speed: float,
) -> LoadedRun:
    version = recording_dir.name
    run_dir = _latest_run_dir(version, run_root)
    result_path = run_dir / "fsm50_task_replay_runner_result.json"
    result = _read_json(result_path)
    if result.get("schema_version") != "fsm50.task_replay_runner_result.v1":
        raise EvidenceError(f"unexpected runner-result schema for {version}")
    if result.get("source_version") != version:
        raise EvidenceError(f"runner-result source mismatch for {version}")
    assessment = result.get("assessment")
    if not isinstance(assessment, dict):
        raise EvidenceError(f"missing assessment for {version}")
    if assessment.get("evaluation_status") != "EVALUATED":
        raise GateAIncompleteError(
            f"Gate A is not EVALUATED for {version}: "
            f"{assessment.get('evaluation_status')!r}"
        )
    task_result = str(assessment.get("task_result", ""))
    if task_result not in SUCCESS_RESULTS | {"REPLAY_TASK_FAIL"}:
        raise EvidenceError(f"unexpected Gate-A task result for {version}: {task_result}")
    if not bool(result.get("shutdown_verified", False)):
        raise EvidenceError(f"Gate-A shutdown is not verified for {version}")

    accepted_path = recording_dir / "accepted_steps.jsonl"
    accepted_sha = _require_hash(
        accepted_path, result.get("accepted_steps_sha256"), f"{version} accepted_steps"
    )
    if not _same_path(result.get("accepted_steps_path"), accepted_path):
        raise EvidenceError(f"accepted_steps path mismatch for {version}")
    steps = load_steps_jsonl(accepted_path)
    plan, plan_rows = fast_plan_rows(
        source_version=version,
        steps=steps,
        max_wheel_speed=max_wheel_speed,
    )
    plan_sha = str(plan.plan_sha256).lower()
    if plan_sha != str(result.get("plan_sha256", "")).lower():
        raise EvidenceError(f"production Fast plan SHA mismatch for {version}")

    if task_result == "REPLAY_TASK_FAIL":
        return LoadedRun(
            version=version,
            recording_dir=recording_dir,
            run_dir=run_dir,
            result=result,
            task_result=task_result,
            plan_rows=tuple(plan_rows),
            telemetry=(),
            task_inputs={},
            accepted_steps_sha256=accepted_sha,
            plan_sha256=plan_sha,
            task_inputs_sha256=str(result.get("task_inputs_sha256", "")),
            telemetry_sha256=(
                _sha256_file(run_dir / "minimal_telemetry.jsonl")
                if (run_dir / "minimal_telemetry.jsonl").is_file()
                else ""
            ),
            video_sha256=str(result.get("video_sha256", "")),
        )

    if result.get("terminal_phase") != "TASK_REPLAY_COMPLETE":
        raise EvidenceError(f"successful Gate-A run has wrong terminal phase: {version}")
    task_path = run_dir / "task_inputs.json"
    classifier_path = run_dir / "classifier_inputs.json"
    manual_path = run_dir / "manual_video_verdict.json"
    video_path = run_dir / "actual_viewport_video.mp4"
    telemetry_path = run_dir / "minimal_telemetry.jsonl"
    for field, expected in (
        ("task_inputs_path", task_path),
        ("classifier_inputs_path", classifier_path),
        ("manual_video_verdict_path", manual_path),
        ("video_path", video_path),
    ):
        if not _same_path(result.get(field), expected):
            raise EvidenceError(f"{field} mismatch for {version}")
    task_sha = _require_hash(
        task_path, result.get("task_inputs_sha256"), f"{version} task_inputs"
    )
    classifier_sha = _require_hash(
        classifier_path,
        result.get("classifier_inputs_sha256"),
        f"{version} classifier_inputs",
    )
    manual_sha = _require_hash(
        manual_path,
        result.get("manual_video_verdict_sha256"),
        f"{version} manual verdict",
    )
    video_sha = _require_hash(video_path, result.get("video_sha256"), f"{version} video")
    task_inputs = _read_json(task_path)
    manual = _read_json(manual_path)
    if task_inputs.get("schema_version") != "fsm50.replay_task_inputs.v1":
        raise EvidenceError(f"unexpected task-input schema for {version}")
    if not bool(manual.get("review_complete", False)):
        raise GateAIncompleteError(f"manual video review is incomplete for {version}")
    for key, expected in (
        ("source_version", version),
        # Manual review is intentionally bound to the immutable classifier
        # snapshot, while task_inputs.json is finalized afterward with the
        # resulting manual verdict.
        ("task_inputs_sha256", classifier_sha),
        ("video_sha256", video_sha),
    ):
        if manual.get(key) != expected:
            raise EvidenceError(f"manual-verdict binding mismatch for {version}.{key}")
    if not _same_path(manual.get("task_inputs_path"), classifier_path):
        raise EvidenceError(f"manual classifier-input path mismatch for {version}")
    if not manual_sha:
        raise EvidenceError(f"manual verdict digest missing for {version}")
    completed = task_inputs.get("completed_result")
    physical = task_inputs.get("physical_evidence")
    final_row = task_inputs.get("final_telemetry_row")
    if not all(isinstance(value, dict) for value in (completed, physical, final_row)):
        raise EvidenceError(f"incomplete task evidence for {version}")
    if not bool(completed.get("dispatch_complete", False)):
        raise EvidenceError(f"successful Gate-A dispatch is incomplete for {version}")
    if not bool(completed.get("scheduler_complete", False)):
        raise EvidenceError(f"successful Gate-A scheduler is incomplete for {version}")
    if not bool(completed.get("video_full_decode_valid", False)):
        raise EvidenceError(f"successful Gate-A video is not fully decoded for {version}")
    if not bool(physical.get("body_crossed_front_face", False)):
        raise EvidenceError(f"successful Gate-A body crossing is absent for {version}")
    traversal = physical.get("traversal", {}).get("legs", {})
    if not isinstance(traversal, dict) or any(leg not in traversal for leg in LEG_ORDER):
        raise EvidenceError(f"four-leg traversal evidence is absent for {version}")
    telemetry = _load_telemetry(
        telemetry_path,
        version=version,
        segment_count=len(plan_rows),
        task_final_row=final_row,
    )
    return LoadedRun(
        version=version,
        recording_dir=recording_dir,
        run_dir=run_dir,
        result=result,
        task_result=task_result,
        plan_rows=tuple(plan_rows),
        telemetry=telemetry,
        task_inputs=task_inputs,
        accepted_steps_sha256=accepted_sha,
        plan_sha256=plan_sha,
        task_inputs_sha256=task_sha,
        telemetry_sha256=_sha256_file(telemetry_path),
        video_sha256=video_sha,
    )


def load_gate_a_corpus(
    *,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> tuple[list[LoadedRun], list[LoadedRun]]:
    """Load all physical versions; return successful runs and exclusions.

    The function finishes validation for the entire corpus before returning,
    so callers cannot accidentally synthesize phases from a partial Gate A.
    """

    directories = enumerate_recording_directories(recording_root)
    if not directories:
        raise GateAIncompleteError("no physical 50 mm recordings were found")
    max_wheel_speed = load_max_wheel_speed()
    loaded = [
        _load_one_run(
            directory,
            run_root=run_root,
            max_wheel_speed=max_wheel_speed,
        )
        for directory in directories
    ]
    successful = [run for run in loaded if run.successful]
    excluded = [run for run in loaded if not run.successful]
    if not successful:
        raise EvidenceError("Gate A has no successful recording to align")
    return successful, excluded


def _row_elapsed(rows: Sequence[Mapping[str, Any]], index: int) -> float:
    return float(rows[index]["sim_time_s"]) - float(rows[0]["sim_time_s"])


def _nearest_elapsed_index(
    rows: Sequence[Mapping[str, Any]], elapsed_s: float
) -> int:
    return min(
        range(len(rows)),
        key=lambda index: abs(_row_elapsed(rows, index) - float(elapsed_s)),
    )


def _crossing_index(
    rows: Sequence[Mapping[str, Any]], leg: str, elapsed_s: float
) -> int:
    anchor = _nearest_elapsed_index(rows, elapsed_s)
    candidates = [
        index
        for index in range(max(0, anchor - 2), min(len(rows), anchor + 6))
        if float(rows[index]["wheel_front_face_clearance_m"][leg]) >= -0.002
    ]
    if candidates:
        return min(candidates, key=lambda index: abs(index - anchor))
    for index in range(anchor, len(rows)):
        if float(rows[index]["wheel_front_face_clearance_m"][leg]) >= 0.0:
            return index
    raise EvidenceError(f"{leg} front-face crossing cannot be located in telemetry")


def _stable_top_index(
    rows: Sequence[Mapping[str, Any]], leg: str, crossing_index: int
) -> int:
    for index in range(crossing_index, len(rows)):
        if (
            rows[index]["wheel_contact_classes"][leg] == "TOP"
            and float(rows[index]["wheel_front_face_clearance_m"][leg]) >= -0.002
        ):
            return index
    raise EvidenceError(f"{leg} has no geometric TOP observation after crossing")


def _final_support_departure(
    rows: Sequence[Mapping[str, Any]],
    leg: str,
    *,
    lower_bound: int,
    crossing_index: int,
) -> int:
    classes = [str(row["wheel_contact_classes"][leg]) for row in rows]
    for index in range(crossing_index, max(lower_bound, 1), -1):
        if classes[index] not in SUPPORT_CLASSES and classes[index - 1] in SUPPORT_CLASSES:
            return index
    non_support = [
        index
        for index in range(lower_bound, crossing_index + 1)
        if classes[index] not in SUPPORT_CLASSES
    ]
    if non_support:
        return non_support[0]
    raise EvidenceError(f"{leg} has no geometry-support departure before crossing")


def _lift_clearance_index(
    rows: Sequence[Mapping[str, Any]],
    leg: str,
    *,
    departure_index: int,
    crossing_index: int,
) -> int:
    for index in range(departure_index, crossing_index + 1):
        if float(rows[index]["wheel_top_clearance_m"][leg]) >= 0.0:
            return index
    return max(
        range(departure_index, crossing_index + 1),
        key=lambda index: float(rows[index]["wheel_top_clearance_m"][leg]),
    )


def derive_leg_landmarks(run: LoadedRun) -> dict[str, dict[str, int]]:
    rows = run.telemetry
    traversal = run.task_inputs["physical_evidence"]["traversal"]["legs"]
    crossings: dict[str, int] = {}
    for leg in LEG_ORDER:
        crossing_time = _finite(
            traversal[leg].get("front_face_crossing_s"),
            f"{run.version}.{leg}.front_face_crossing_s",
        )
        crossings[leg] = _crossing_index(rows, leg, crossing_time)
    chronological = sorted(LEG_ORDER, key=lambda leg: crossings[leg])
    lower_by_leg: dict[str, int] = {}
    previous_crossing = 0
    for leg in chronological:
        lower_by_leg[leg] = previous_crossing
        previous_crossing = crossings[leg]

    result: dict[str, dict[str, int]] = {}
    for leg in LEG_ORDER:
        crossing = crossings[leg]
        departure = _final_support_departure(
            rows,
            leg,
            lower_bound=lower_by_leg[leg],
            crossing_index=crossing,
        )
        result[leg] = {
            "departure": departure,
            "clearance": _lift_clearance_index(
                rows,
                leg,
                departure_index=departure,
                crossing_index=crossing,
            ),
            "crossing": crossing,
            "top": _stable_top_index(rows, leg, crossing),
        }
    return result


def _segment_for_row(run: LoadedRun, row_index: int) -> int:
    cursor = int(run.telemetry[row_index].get("segment_cursor", 0))
    return max(0, min(cursor, len(run.plan_rows) - 1))


def _nonzero_wheel(segment: Mapping[str, Any]) -> bool:
    return any(
        abs(float(value)) > 1.0e-12
        for value in dict(segment.get("wheel_target_rad_s", {})).values()
    )


def _telemetry_groups(
    run: LoadedRun,
    predicate: Callable[[Mapping[str, Any]], bool],
    *,
    lower: int,
    upper: int,
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(max(0, lower), min(len(run.telemetry) - 1, upper) + 1):
        segment = run.plan_rows[_segment_for_row(run, index)]
        active = bool(predicate(segment))
        if active and start is None:
            start = index
        elif not active and start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, min(len(run.telemetry) - 1, upper)))
    return groups


def _longest_group(groups: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not groups:
        return None
    return max(groups, key=lambda pair: (pair[1] - pair[0], -pair[0]))


def _first_leg_servo_index(
    run: LoadedRun,
    leg: str,
    *,
    lower: int,
    upper: int,
) -> int | None:
    prefix = LEG_JOINT_PREFIX[leg]
    for index in range(lower, upper + 1):
        segment = run.plan_rows[_segment_for_row(run, index)]
        if any(str(name).startswith(prefix) for name in segment["servo_target_deg"]):
            return index
    return None


def _phase_specs(
    run: LoadedRun, landmarks: Mapping[str, Mapping[str, int]]
) -> dict[str, tuple[int | None, int | None, str]]:
    last = len(run.telemetry) - 1
    specs: dict[str, tuple[int | None, int | None, str]] = {}
    fr_cross = landmarks["FR"]["crossing"]
    early_wheels = _telemetry_groups(
        run, _nonzero_wheel, lower=0, upper=fr_cross
    )
    approach = early_wheels[0] if early_wheels else None
    if approach is None:
        specs["INITIAL_APPROACH"] = (None, None, "NOT_DISTINCTLY_OBSERVED")
        specs["PRE_FR_COM_SHIFT"] = (
            0,
            landmarks["FR"]["departure"],
            "COMMAND_AND_BASE_PROXY",
        )
    else:
        specs["INITIAL_APPROACH"] = (*approach, "COMMAND_AND_BASE_PROXY")
        specs["PRE_FR_COM_SHIFT"] = (
            0,
            max(0, approach[0] - 1),
            "COMMAND_AND_BASE_PROXY",
        )

    for leg in LEG_ORDER:
        mark = landmarks[leg]
        specs[f"{leg}_UNLOAD_AND_LIFT"] = (
            mark["departure"],
            mark["clearance"],
            "GEOMETRY_EVENT_ANCHORED",
        )
        specs[f"{leg}_FACE_CROSS"] = (
            mark["clearance"],
            mark["crossing"],
            "GEOMETRY_EVENT_ANCHORED",
        )
        specs[f"{leg}_TOP_PLACE"] = (
            mark["crossing"],
            mark["top"],
            "GEOMETRY_EVENT_ANCHORED",
        )

    front_groups = _telemetry_groups(
        run,
        _nonzero_wheel,
        lower=landmarks["FL"]["crossing"],
        upper=landmarks["RR"]["departure"],
    )
    front_advance = _longest_group(front_groups)
    specs["FRONT_PAIR_ADVANCE"] = (
        (*front_advance, "COMMAND_AND_BASE_PROXY")
        if front_advance
        else (None, None, "NOT_DISTINCTLY_OBSERVED")
    )
    specs["PRE_RR_COM_SHIFT"] = (
        landmarks["FL"]["top"],
        landmarks["RR"]["departure"],
        "COMMAND_AND_BASE_PROXY",
    )

    setup_start = landmarks["RR"]["top"]
    rl_departure = landmarks["RL"]["departure"]
    rl_servo = _first_leg_servo_index(
        run, "RL", lower=setup_start, upper=rl_departure
    )
    if rl_servo is None:
        specs["PRE_RL_SUPPORT_SETUP"] = (
            setup_start,
            rl_departure,
            "COMMAND_AND_BASE_PROXY",
        )
        specs["PRE_RL_COM_SHIFT"] = (None, None, "NOT_DISTINCTLY_OBSERVED")
    else:
        specs["PRE_RL_SUPPORT_SETUP"] = (
            (setup_start, rl_servo - 1, "COMMAND_AND_BASE_PROXY")
            if rl_servo > setup_start
            else (None, None, "NOT_DISTINCTLY_OBSERVED")
        )
        specs["PRE_RL_COM_SHIFT"] = (
            rl_servo,
            rl_departure,
            "COMMAND_AND_BASE_PROXY",
        )

    final_groups = _telemetry_groups(
        run,
        _nonzero_wheel,
        lower=landmarks["RL"]["crossing"],
        upper=last,
    )
    final_advance = _longest_group(final_groups)
    specs["FINAL_ADVANCE"] = (
        (*final_advance, "COMMAND_AND_BASE_PROXY")
        if final_advance
        else (None, None, "NOT_DISTINCTLY_OBSERVED")
    )
    recovery_start = max(
        landmarks["RL"]["top"],
        final_advance[1] if final_advance else landmarks["RL"]["top"],
    )
    specs["FINAL_POSTURE_RECOVERY"] = (
        recovery_start,
        last,
        "COMMAND_AND_BASE_PROXY",
    )
    return specs


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _target_ranges(
    segments: Sequence[Mapping[str, Any]], key: str, *, nonzero_only: bool = False
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for segment in segments:
        for name, value in dict(segment.get(key, {})).items():
            number = float(value)
            if nonzero_only and abs(number) <= 1.0e-12:
                continue
            values.setdefault(str(name), []).append(number)
    return {
        name: [min(numbers), max(numbers)] for name, numbers in sorted(values.items())
    }


def _observed_base_direction(delta: Mapping[str, float]) -> str:
    labels: list[str] = []
    if abs(delta["x"]) >= 1.0e-4:
        labels.append("+X" if delta["x"] > 0 else "-X")
    if abs(delta["y"]) >= 1.0e-4:
        labels.append("+Y" if delta["y"] > 0 else "-Y")
    return "+".join(labels) if labels else "NEGLIGIBLE_XY"


def _phase_evidence_basis(phase: str, status: str) -> str:
    if status == "NOT_DISTINCTLY_OBSERVED":
        return status
    if phase.endswith("UNLOAD_AND_LIFT"):
        return "FINAL_GEOMETRY_SUPPORT_DEPARTURE_AND_CLEARANCE_LANDMARK"
    if phase.endswith("FACE_CROSS"):
        return "WHEEL_CLEARANCE_AND_FRONT_FACE_GEOMETRY"
    if phase.endswith("TOP_PLACE"):
        return "POST_CROSSING_GEOMETRIC_TOP_WITH_FACE_CLEARANCE"
    return "PRODUCTION_COMMAND_PATTERN_PLUS_MEASURED_BASE_MOTION_PROXY"


def _aggregate_phase(
    run: LoadedRun,
    phase: str,
    spec: tuple[int | None, int | None, str],
) -> dict[str, Any]:
    start, end, status = spec
    active_leg = ACTIVE_LEG.get(phase, "")
    entry, completion = ENTRY_COMPLETION[phase]
    base = {
        "version": run.version,
        "task_result": run.task_result,
        "strategy_profile": "",
        "phase": phase,
        "phase_status": status,
        "evidence_basis": _phase_evidence_basis(phase, status),
        "functional_windows_may_overlap": True,
        "run_dir": str(run.run_dir.resolve()),
        "accepted_steps_sha256": run.accepted_steps_sha256,
        "plan_sha256": run.plan_sha256,
        "task_inputs_sha256": run.task_inputs_sha256,
        "telemetry_sha256": run.telemetry_sha256,
        "video_sha256": run.video_sha256,
        "active_leg": active_leg,
        "candidate_com_target_direction": COM_CANDIDATE.get(
            phase, "TOWARD_GEOMETRY_SUPPORT_CANDIDATES"
        ),
        "com_evidence_status": "BASE_TRANSLATION_PROXY_ONLY_NO_COM_TELEMETRY",
        "support_evidence_status": "GEOMETRY_ONLY_NO_CONTACT_LOAD",
        "entry_event": entry,
        "completion_event": completion,
    }
    if start is None or end is None or status == "NOT_DISTINCTLY_OBSERVED":
        return {
            **base,
            "replay_start_s": None,
            "replay_end_s": None,
            "duration_s": None,
            "source_step_range": "",
            "fast_segment_range": "",
            "candidate_support_legs": [],
            "geometry_support_fraction": {},
            "servo_commands": [],
            "servo_target_range_deg": {},
            "wheel_commands": [],
            "wheel_target_range_rad_s": {},
            "servo_wheel_concurrent": False,
            "concurrent_segment_count": 0,
            "observed_base_delta_m": {},
            "observed_base_direction": "UNAVAILABLE",
            "active_wheel_clearance_range_m": {},
            "peak_abs_roll_rad": None,
            "peak_abs_pitch_rad": None,
            "suitable_for_first_fsm": "NO_DISTINCT_WINDOW",
            "notes": "No distinct evidence window; no phase behavior was invented.",
            "structure": {},
        }
    if end < start:
        start, end = end, start
    telemetry = run.telemetry[start : end + 1]
    first_segment = _segment_for_row(run, start)
    last_segment = _segment_for_row(run, end)
    if last_segment < first_segment:
        first_segment, last_segment = last_segment, first_segment
    segments = run.plan_rows[first_segment : last_segment + 1]
    source_steps = [int(segment["source_step_index"]) for segment in segments]
    servo_commands = [
        {
            "segment": int(segment["decoded_segment_index"]),
            "targets_deg": dict(segment["servo_target_deg"]),
        }
        for segment in segments
        if segment["servo_target_deg"]
    ]
    wheel_commands = [
        {
            "segment": int(segment["decoded_segment_index"]),
            "targets_rad_s": {
                name: float(value)
                for name, value in segment["wheel_target_rad_s"].items()
                if abs(float(value)) > 1.0e-12
            },
        }
        for segment in segments
        if _nonzero_wheel(segment)
    ]
    support_fraction: dict[str, float] = {}
    for leg in LEG_ORDER:
        if leg == active_leg:
            continue
        support_fraction[leg] = sum(
            row["wheel_contact_classes"][leg] in SUPPORT_CLASSES for row in telemetry
        ) / len(telemetry)
    candidate_supports = [
        leg for leg, fraction in support_fraction.items() if fraction >= 0.5
    ]
    candidate_com_direction = COM_CANDIDATE.get(
        phase,
        "TOWARD_GEOMETRY_SUPPORT_CANDIDATES:"
        + (",".join(candidate_supports) or "UNRESOLVED"),
    )
    start_base = telemetry[0]["base_position_m"]
    end_base = telemetry[-1]["base_position_m"]
    delta = {
        axis: float(end_base[axis]) - float(start_base[axis])
        for axis in ("x", "y", "z")
    }
    clearance: dict[str, list[float]] = {}
    if active_leg:
        for key, telemetry_key in (
            ("front_face", "wheel_front_face_clearance_m"),
            ("top", "wheel_top_clearance_m"),
            ("wheel_center_z", "wheel_center_w_m"),
        ):
            if telemetry_key == "wheel_center_w_m":
                numbers = [float(row[telemetry_key][active_leg][2]) for row in telemetry]
            else:
                numbers = [float(row[telemetry_key][active_leg]) for row in telemetry]
            clearance[key] = [min(numbers), max(numbers)]
    concurrent_count = sum(bool(segment.get("concurrent", False)) for segment in segments)
    servo_sequence = _ordered_unique(
        str(name)
        for segment in segments
        for name in segment["servo_target_deg"].keys()
    )
    wheel_sequence = _ordered_unique(
        str(name)
        for segment in segments
        for name, value in segment["wheel_target_rad_s"].items()
        if abs(float(value)) > 1.0e-12
    )
    first_fsm = (
        "YES_EVENT_GUARD_CANDIDATE"
        if status == "GEOMETRY_EVENT_ANCHORED"
        else "CANDIDATE_REQUIRES_FEEDBACK_GUARD"
    )
    return {
        **base,
        "candidate_com_target_direction": candidate_com_direction,
        "replay_start_s": _row_elapsed(run.telemetry, start),
        "replay_end_s": _row_elapsed(run.telemetry, end),
        "duration_s": _row_elapsed(run.telemetry, end)
        - _row_elapsed(run.telemetry, start),
        "source_step_range": f"{min(source_steps)}:{max(source_steps)}",
        "fast_segment_range": f"{first_segment}:{last_segment}",
        "candidate_support_legs": candidate_supports,
        "geometry_support_fraction": support_fraction,
        "servo_commands": servo_commands,
        "servo_target_range_deg": _target_ranges(segments, "servo_target_deg"),
        "wheel_commands": wheel_commands,
        "wheel_target_range_rad_s": _target_ranges(
            segments, "wheel_target_rad_s", nonzero_only=True
        ),
        "servo_wheel_concurrent": bool(concurrent_count),
        "concurrent_segment_count": concurrent_count,
        "observed_base_delta_m": delta,
        "observed_base_direction": _observed_base_direction(delta),
        "active_wheel_clearance_range_m": clearance,
        "peak_abs_roll_rad": max(abs(float(row["base_roll_rad"])) for row in telemetry),
        "peak_abs_pitch_rad": max(abs(float(row["base_pitch_rad"])) for row in telemetry),
        "suitable_for_first_fsm": first_fsm,
        "notes": (
            f"Observed base direction {_observed_base_direction(delta)}; "
            "this is not a measured COM direction."
        ),
        "structure": {
            "servo_joint_sequence": servo_sequence,
            "wheel_assist_joint_sequence": wheel_sequence,
            "servo_wheel_concurrent": bool(concurrent_count),
            "phase_status": status,
        },
    }


def align_successful_run(run: LoadedRun) -> list[dict[str, Any]]:
    landmarks = derive_leg_landmarks(run)
    specs = _phase_specs(run, landmarks)
    return [_aggregate_phase(run, phase, specs[phase]) for phase in PHASE_ORDER]


def _traversal_order(run: LoadedRun) -> list[str]:
    traversal = run.task_inputs["physical_evidence"]["traversal"]["legs"]
    return sorted(
        LEG_ORDER,
        key=lambda leg: float(traversal[leg]["front_face_crossing_s"]),
    )


def assign_strategy_profiles(
    runs: Sequence[LoadedRun], rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, str]]:
    by_version: dict[str, list[dict[str, Any]]] = {
        run.version: [] for run in runs
    }
    for row in rows:
        by_version[row["version"]].append(row)
    fingerprints: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for run in runs:
        payload = {
            "traversal_order": _traversal_order(run),
            "phases": [
                {
                    "phase": row["phase"],
                    **row["structure"],
                }
                for row in sorted(
                    by_version[run.version],
                    key=lambda item: PHASE_ORDER.index(item["phase"]),
                )
            ],
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        fingerprints[run.version] = digest
        payloads[digest] = payload
    groups: dict[str, list[str]] = {}
    for version, digest in fingerprints.items():
        groups.setdefault(digest, []).append(version)
    primary_digest = next(
        (
            digest
            for version, digest in fingerprints.items()
            if version.startswith("v003_")
        ),
        sorted(groups, key=lambda digest: (-len(groups[digest]), digest))[0],
    )
    role_by_digest = {primary_digest: "PRIMARY_PROFILE"}
    alternates = [
        digest
        for digest in sorted(groups, key=lambda item: (-len(groups[item]), item))
        if digest != primary_digest
    ]
    for index, digest in enumerate(alternates, start=1):
        role_by_digest[digest] = f"ALTERNATE_PROFILE_{index}"
    version_roles = {
        version: role_by_digest[digest] for version, digest in fingerprints.items()
    }
    clusters = [
        {
            "role": role_by_digest[digest],
            "fingerprint_sha256": digest,
            "versions": sorted(groups[digest]),
            "structure": payloads[digest],
        }
        for digest in sorted(groups, key=lambda item: role_by_digest[item])
    ]

    recovery_groups: dict[str, list[str]] = {}
    for version, version_rows in by_version.items():
        recovery = next(
            row for row in version_rows if row["phase"] == "FINAL_POSTURE_RECOVERY"
        )
        digest = hashlib.sha256(
            _canonical_json(recovery["structure"]).encode("utf-8")
        ).hexdigest()
        recovery_groups.setdefault(digest, []).append(version)
    recovery_roles: dict[str, str] = {}
    for index, digest in enumerate(sorted(recovery_groups), start=1):
        role = f"RECOVERY_PROFILE_{index}"
        for version in recovery_groups[digest]:
            recovery_roles[version] = role
    return version_roles, clusters, recovery_roles


def analyze_gate_b(
    *,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    successful, excluded = load_gate_a_corpus(
        recording_root=recording_root, run_root=run_root
    )
    rows = [row for run in successful for row in align_successful_run(run)]
    version_roles, clusters, recovery_roles = assign_strategy_profiles(successful, rows)
    for row in rows:
        row["strategy_profile"] = (
            recovery_roles[row["version"]]
            if row["phase"] == "FINAL_POSTURE_RECOVERY"
            else version_roles[row["version"]]
        )
    return {
        "schema_version": "fsm50.gate_b_phase_alignment.v1",
        "successful_runs": successful,
        "excluded_runs": excluded,
        "rows": rows,
        "clusters": clusters,
        "version_roles": version_roles,
        "recovery_roles": recovery_roles,
    }


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    json_fields = {
        "candidate_support_legs": row["candidate_support_legs"],
        "geometry_support_fraction_json": row["geometry_support_fraction"],
        "servo_commands_json": row["servo_commands"],
        "servo_target_range_deg_json": row["servo_target_range_deg"],
        "wheel_commands_json": row["wheel_commands"],
        "wheel_target_range_rad_s_json": row["wheel_target_range_rad_s"],
        "observed_base_delta_m_json": row["observed_base_delta_m"],
        "active_wheel_clearance_range_m_json": row[
            "active_wheel_clearance_range_m"
        ],
    }
    result: dict[str, Any] = {}
    for column in CSV_COLUMNS:
        if column in json_fields:
            result[column] = _json_cell(json_fields[column])
        else:
            value = row.get(column, "")
            result[column] = "" if value is None else value
    return result


def _format_range(values: Sequence[float | None], digits: int = 4) -> str:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return "unavailable"
    return f"{min(finite):.{digits}f}..{max(finite):.{digits}f}"


def _phase_parameter_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    durations = _format_range([row.get("duration_s") for row in rows], 3)
    servo: dict[str, list[float]] = {}
    wheel: dict[str, list[float]] = {}
    for row in rows:
        for name, bounds in row["servo_target_range_deg"].items():
            servo.setdefault(name, []).extend(float(value) for value in bounds)
        for name, bounds in row["wheel_target_range_rad_s"].items():
            wheel.setdefault(name, []).extend(float(value) for value in bounds)
    servo_text = ", ".join(
        f"{name}={min(values):.2f}..{max(values):.2f}deg"
        for name, values in sorted(servo.items())
    ) or "none"
    wheel_text = ", ".join(
        f"{name}={min(values):.3f}..{max(values):.3f}rad/s"
        for name, values in sorted(wheel.items())
    ) or "none"
    return f"duration={durations}s; servo: {servo_text}; wheel: {wheel_text}"


def _common_phases_markdown(analysis: Mapping[str, Any]) -> str:
    runs: Sequence[LoadedRun] = analysis["successful_runs"]
    excluded: Sequence[LoadedRun] = analysis["excluded_runs"]
    rows: Sequence[dict[str, Any]] = analysis["rows"]
    lines = [
        "# 50 mm Common Physical Phases",
        "",
        "This report is generated only from finalized production Fast Replay artifacts. "
        "Functional windows may overlap because a concurrent command can serve more than "
        "one physical purpose.",
        "",
        "COM is not present in normal Gate-A telemetry. Every COM-direction entry below is "
        "therefore a candidate interpretation paired with measured base translation, never "
        "a measured COM claim. Support sets are geometry-only candidates because contact "
        "load telemetry was intentionally disabled in normal development mode.",
        "",
        f"Successful synthesis set ({len(runs)}): "
        + ", ".join(f"`{run.version}`" for run in runs)
        + ".",
        "",
        f"Excluded Gate-A failures ({len(excluded)}): "
        + ", ".join(f"`{run.version}`" for run in excluded)
        + ". Failed runs are provenance only and never contribute profile parameters.",
        "",
        "## Aligned phases",
        "",
        "| Phase | Included versions | Step / segment ranges | Active leg | Geometry support candidates | Servo / wheel concurrency | COM candidate and base proxy | Entry -> completion | Parameter range | Strategy | First FSM |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for phase in PHASE_ORDER:
        phase_rows = [
            row
            for row in rows
            if row["phase"] == phase
            and row["phase_status"] != "NOT_DISTINCTLY_OBSERVED"
        ]
        all_rows = [row for row in rows if row["phase"] == phase]
        included = ", ".join(row["version"].split("_")[0] for row in phase_rows) or "none"
        ranges = "; ".join(
            f"{row['version'].split('_')[0]} step {row['source_step_range']} / seg {row['fast_segment_range']}"
            for row in phase_rows
        ) or "unresolved"
        supports = sorted(
            {leg for row in phase_rows for leg in row["candidate_support_legs"]}
        )
        concurrency = sum(bool(row["servo_wheel_concurrent"]) for row in phase_rows)
        com_parts = [
            f"{row['version'].split('_')[0]} {row['observed_base_direction']} "
            f"dx={row['observed_base_delta_m'].get('x', 0.0):.4f},dy={row['observed_base_delta_m'].get('y', 0.0):.4f}m"
            for row in phase_rows
        ]
        structures = {
            hashlib.sha256(_canonical_json(row["structure"]).encode("utf-8")).hexdigest()[:8]
            for row in phase_rows
        }
        strategy = (
            "common structure"
            if len(structures) <= 1
            else f"{len(structures)} categorical alternatives; do not average"
        )
        first_fsm = (
            "yes, event guard"
            if phase_rows
            and all(row["phase_status"] == "GEOMETRY_EVENT_ANCHORED" for row in phase_rows)
            else "candidate; add feedback guard"
            if phase_rows
            else "no distinct window"
        )
        entry, completion = ENTRY_COMPLETION[phase]
        lines.append(
            "| "
            + " | ".join(
                [
                    phase,
                    included,
                    ranges,
                    ACTIVE_LEG.get(phase, "") or "multi-leg",
                    ",".join(supports) or "unresolved",
                    f"{concurrency}/{len(phase_rows)} versions",
                    "/".join(
                        sorted(
                            {
                                row["candidate_com_target_direction"]
                                for row in phase_rows
                            }
                        )
                    )
                    + "; "
                    + "; ".join(com_parts),
                    f"{entry} -> {completion}",
                    _phase_parameter_summary(phase_rows),
                    strategy,
                    first_fsm,
                ]
            )
            + " |"
        )
        if len(phase_rows) != len(all_rows):
            missing = [
                row["version"].split("_")[0]
                for row in all_rows
                if row["phase_status"] == "NOT_DISTINCTLY_OBSERVED"
            ]
            lines.append(
                f"| {phase} note | distinct window absent in {', '.join(missing)}; "
                "no commands or parameters were invented | | | | | | | | | |"
            )
    lines.extend(
        [
            "",
            "## Evidence interpretation",
            "",
            "Traversal rows are anchored to wheel geometry: final support departure before "
            "crossing, a clearance landmark, front-face crossing, and the first stable "
            "geometric TOP observation after crossing. Pre-shift, advance, and recovery rows "
            "are command/base-motion proxies and require feedback guards before use in an FSM.",
            "",
            "No failed replay contributes a target, duration range, support candidate, or "
            "strategy cluster. No numerical average is taken across categorical strategies.",
            "",
        ]
    )
    return "\n".join(lines)


def _clusters_markdown(analysis: Mapping[str, Any]) -> str:
    runs: Sequence[LoadedRun] = analysis["successful_runs"]
    excluded: Sequence[LoadedRun] = analysis["excluded_runs"]
    clusters: Sequence[Mapping[str, Any]] = analysis["clusters"]
    recovery_roles: Mapping[str, str] = analysis["recovery_roles"]
    lines = [
        "# 50 mm Recording Strategy Clusters",
        "",
        "Clusters use exact categorical structure: physical leg-crossing order, ordered "
        "servo-joint participation, wheel-assist participation, concurrency, and whether a "
        "phase has a distinct evidence window. Numeric targets and durations are not part of "
        "the structural fingerprint and are never averaged between clusters.",
        "",
        "## Successful Gate-A inputs",
        "",
        "| Version | Task result | Strategy profile | Recovery profile | Plan SHA-256 | Telemetry SHA-256 | Video SHA-256 |",
        "|---|---|---|---|---|---|---|",
    ]
    roles = analysis["version_roles"]
    for run in runs:
        lines.append(
            f"| {run.version} | {run.task_result} | {roles[run.version]} | "
            f"{recovery_roles[run.version]} | `{run.plan_sha256}` | "
            f"`{run.telemetry_sha256}` | `{run.video_sha256}` |"
        )
    lines.extend(
        [
            "",
            "## Explicit Gate-A exclusions",
            "",
            "| Version | Result | First actual failure phase | Run | Plan SHA-256 |",
            "|---|---|---|---|---|",
        ]
    )
    for run in excluded:
        assessment = run.result["assessment"]
        lines.append(
            f"| {run.version} | {run.task_result} | "
            f"{assessment.get('first_actual_failure_phase', '') or 'unspecified'} | "
            f"`{run.run_dir.resolve()}` | `{run.plan_sha256}` |"
        )
    lines.extend(["", "## Structural strategy clusters", ""])
    for cluster in clusters:
        structure = cluster["structure"]
        lines.extend(
            [
                f"### {cluster['role']}",
                "",
                f"- Versions: {', '.join(cluster['versions'])}",
                f"- Fingerprint: `{cluster['fingerprint_sha256']}`",
                f"- Crossing order: {' -> '.join(structure['traversal_order'])}",
                "- Phase structures:",
                "",
            ]
        )
        for phase in structure["phases"]:
            lines.append(
                f"  - {phase['phase']}: servo={phase.get('servo_joint_sequence', [])}; "
                f"wheel={phase.get('wheel_assist_joint_sequence', [])}; "
                f"concurrent={phase.get('servo_wheel_concurrent', False)}; "
                f"status={phase.get('phase_status', '')}"
            )
        lines.append("")
    lines.extend(
        [
            "## Profile-selection rule",
            "",
            "The cluster containing v003 is `PRIMARY_PROFILE`. Other successful structural "
            "clusters are `ALTERNATE_PROFILE_n`. Final-posture command structures are kept "
            "separately as `RECOVERY_PROFILE_n`; all current successful Gate-A outcomes are "
            "posture-incomplete, so these are demonstrated recovery attempts, not proof of "
            "posture-complete recovery.",
            "",
            "A first Macro FSM should select one complete profile per phase. It must not "
            "average targets across structural clusters. Alternate selection needs a bounded "
            "feedback guard tied to unload, clearance, crossing, or top-placement evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _alignment_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_csv_row(row))
    return stream.getvalue()


def emit_reports(
    analysis: Mapping[str, Any], report_root: Path = DEFAULT_REPORT_ROOT
) -> dict[str, Path]:
    """Stage all requested reports, then replace their final paths.

    Analysis and corpus validation have already completed before this function
    is entered. A Gate-A incomplete error therefore creates no report file.
    """

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "50MM_COMMON_PHASE_ALIGNMENT.csv": _alignment_csv(analysis["rows"]),
        "50MM_COMMON_PHASES.md": _common_phases_markdown(analysis),
        "50MM_VERSION_STRATEGY_CLUSTERS.md": _clusters_markdown(analysis),
    }
    with tempfile.TemporaryDirectory(prefix=".gate_b_phase_", dir=report_root) as temp:
        staging = Path(temp)
        for name, text in payloads.items():
            (staging / name).write_text(text, encoding="utf-8", newline="")
        for name in payloads:
            os.replace(staging / name, report_root / name)
    return {name: (report_root / name).resolve() for name in payloads}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align finalized successful 50 mm production Fast Replay artifacts."
    )
    parser.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--emit-reports",
        action="store_true",
        help="write the three requested Gate-B reports after full validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_gate_b(
        recording_root=args.recording_root,
        run_root=args.run_root,
    )
    paths = emit_reports(analysis, args.report_root) if args.emit_reports else {}
    print(
        json.dumps(
            {
                "schema_version": analysis["schema_version"],
                "gate_a_version_count": len(analysis["successful_runs"])
                + len(analysis["excluded_runs"]),
                "successful_versions": [
                    run.version for run in analysis["successful_runs"]
                ],
                "excluded_failed_versions": [
                    run.version for run in analysis["excluded_runs"]
                ],
                "phase_row_count": len(analysis["rows"]),
                "strategy_cluster_count": len(analysis["clusters"]),
                "reports_emitted": {
                    name: str(path) for name, path in paths.items()
                },
                "evidence_limits": [
                    "COM_DIRECTION_IS_BASE_TRANSLATION_PROXY_ONLY",
                    "SUPPORT_IS_GEOMETRY_ONLY_NO_CONTACT_LOAD",
                    "FUNCTIONAL_WINDOWS_MAY_OVERLAP",
                    "FAILED_RUNS_EXCLUDED_FROM_PROFILE_SYNTHESIS",
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
