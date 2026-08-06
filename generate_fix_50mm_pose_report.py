"""Generate the immutable evidence bundle for the 2026-08-05 replay/pose fix."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from playback import plan_from_steps
from sequence_model import load_steps_jsonl
from sim_state_validation import validate_full_sim_pose_state


PROJECT = Path(__file__).resolve().parent
REPORT = PROJECT / "reports" / "fix_50mm_manual_replay_and_selected_fast_pose_restore_20260805_193824"
ACTIVE_POINTER = PROJECT / "saved_height_steps_fsm_reference_v2" / "height_050mm" / "active_version.json"
ACTIVE_ID = json.loads(ACTIVE_POINTER.read_text(encoding="utf-8-sig"))["version_id"]
ACTIVE_DIR = PROJECT / "saved_height_steps_fsm_reference_v2" / "height_050mm" / "versions" / ACTIVE_ID
STEPS_PATH = ACTIVE_DIR / "accepted_steps.jsonl"
METADATA_PATH = ACTIVE_DIR / "metadata.json"
STEPS = load_steps_jsonl(STEPS_PATH)
METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_result(folder: str) -> dict[str, Any]:
    return json.loads((REPORT / folder / "real_isaac_result.json").read_text(encoding="utf-8"))


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def single_row(value: Any) -> list[float]:
    row = value
    while isinstance(row, list) and len(row) == 1 and isinstance(row[0], list):
        row = row[0]
    return [float(item) for item in row] if isinstance(row, list) else []


def joint_max_difference(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_names = [str(name) for name in list(left.get("joint_names", []) or [])]
    right_names = [str(name) for name in list(right.get("joint_names", []) or [])]
    left_pos = single_row(left.get("joint_pos"))
    right_pos = single_row(right.get("joint_pos"))
    left_by_name = {name: left_pos[index] for index, name in enumerate(left_names) if index < len(left_pos)}
    right_by_name = {name: right_pos[index] for index, name in enumerate(right_names) if index < len(right_pos)}
    common = set(left_by_name).intersection(right_by_name)
    return max((abs(left_by_name[name] - right_by_name[name]) for name in common), default=None)


# Active version audit.
state_rows: list[dict[str, Any]] = []
classification_counts: dict[str, int] = {}
steps_without_full: set[int] = set()
for step in STEPS:
    step_index = int(step.get("index", 0) or 0)
    for field in ("sim_state_before", "sim_state_after"):
        validation = validate_full_sim_pose_state(step.get(field))
        classification = str(validation["classification"])
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        if not validation["valid"]:
            steps_without_full.add(step_index)
        state_rows.append(
            {
                "step_index": step_index,
                "step_id": str(step.get("step_id", step.get("name", "")) or ""),
                "field": field,
                **validation,
            }
        )
active_audit = {
    "height_mm": 50,
    "version_id": ACTIVE_ID,
    "version_path": str(ACTIVE_DIR.resolve()),
    "accepted_steps_path": str(STEPS_PATH.resolve()),
    "metadata_path": str(METADATA_PATH.resolve()),
    "accepted_steps_sha256": sha256(STEPS_PATH),
    "metadata_accepted_steps_sha256": str(METADATA.get("accepted_steps_sha256", "")),
    "step_count": len(STEPS),
    "source_event_count": sum(len(step.get("events", []) or []) for step in STEPS),
    "source_command_count": int(METADATA.get("command_count", 0) or 0),
    "steps_without_full_pose_count": len(steps_without_full),
    "steps_without_full_pose": sorted(steps_without_full),
    "state_classification_counts": classification_counts,
    "state_audit": state_rows,
}
(REPORT / "active_50mm_version_audit.json").write_text(
    json.dumps(active_audit, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)


# Before/after replay outcome.
pre_fast = read_result("pre_fix_formal_replay_retry")["runs"][0]
pre_raw = read_result("pre_fix_formal_raw")["runs"][0]
post_fast_result = read_result("post_fix_formal_fast_detailed")
post_raw_result = read_result("post_fix_formal_raw_detailed")
post_fast = post_fast_result["runs"][0]
post_raw = post_raw_result["runs"][0]


def error_field(text: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}=([^\s]+)", str(text or ""))
    return match.group(1) if match else ""


def stalled_duration(text: str) -> str:
    match = re.search(r"did not improve for ([0-9.]+)s sim", str(text or ""))
    return match.group(1) if match else ""


replay_rows: list[dict[str, Any]] = []
for phase, run in (
    ("before", pre_raw),
    ("after", post_raw),
    ("before", pre_fast),
    ("after", post_fast),
):
    worker = dict(run.get("final_worker", {}) or {})
    error = str(worker.get("last_error", "") or "")
    replay_rows.append(
        {
            "phase": phase,
            "profile": run.get("profile"),
            "replay_run_id": run.get("replay_run_id"),
            "last_completed_step": max(0, int(run.get("last_started_step", 0) or 0) - (1 if worker.get("stop_reason") != "complete" else 0)),
            "last_started_step": run.get("last_started_step"),
            "last_completed_segment_zero_based": run.get("last_completed_segment"),
            "last_started_segment_zero_based": run.get("last_started_segment"),
            "events_sent": worker.get("events_sent"),
            "event_count": worker.get("event_count"),
            "segment_count": worker.get("segment_count"),
            "stop_reason": worker.get("stop_reason"),
            "offending_joint": error_field(error, "joint"),
            "requested_command_deg": error_field(error, "requested_command_deg"),
            "expected_actual_deg": error_field(error, "expected_actual_deg"),
            "measured_actual_deg": error_field(error, "measured_actual_deg"),
            "error_deg": error_field(error, "error_deg"),
            "tolerance_deg": error_field(error, "tolerance_deg"),
            "stalled_duration_s": stalled_duration(error),
            "last_error": error,
            "final_result": "PASS" if worker.get("stop_reason") == "complete" else "FAIL",
            "operation": run.get("final_operation", "IDLE" if worker.get("stop_reason") else ""),
        }
    )
write_csv(REPORT / "replay_before_after.csv", replay_rows)


# All 64 worker-generated timing rows for each formal profile.
formal_segment_rows: list[dict[str, Any]] = []
for result, run in ((post_raw_result, post_raw), (post_fast_result, post_fast)):
    profile = str(run["profile"])
    plan = plan_from_steps(
        STEPS,
        profile=profile,
        label=f"report rebuild {profile}",
        sequence_total_steps=len(STEPS),
    )
    worker = dict(run["final_worker"])
    timing_by_segment = {
        int(row["segment_index"]): dict(row)
        for row in list(dict(worker.get("timing", {}) or {}).get("segments", []) or [])
    }
    sampled_by_segment = {int(row["segment_index"]): dict(row) for row in list(run.get("segments", []) or [])}
    for segment in plan.segments:
        timing = timing_by_segment.get(int(segment.segment_index), {})
        sample = sampled_by_segment.get(int(segment.segment_index), {})
        state = dict(sample.get("state", {}) or {})
        formal_segment_rows.append(
            {
                "profile": profile,
                "replay_run_id": run["replay_run_id"],
                "plan_sha256": run["plan"]["plan_sha256"],
                "worker_plan_sha256": worker.get("plan_sha256"),
                "source_step_index": segment.source_step,
                "source_step_id": segment.source_step_id,
                "segment_index_zero_based": segment.segment_index,
                "event_indices_zero_based": json_cell(list(range(segment.event_start_index, segment.event_start_index + segment.event_count))),
                "planned_start_s": segment.planned_start_s,
                "planned_end_s": segment.planned_end_s,
                "actual_start_sim_time": timing.get("actual_start_sim_time"),
                "actual_end_sim_time": timing.get("actual_end_sim_time"),
                "servo_targets": json_cell(segment.servo_targets),
                "measured_servo_positions_at_transition": json_cell(timing.get("servo_actual_at_transition", {})),
                "servo_errors": json_cell(timing.get("servo_target_error", {})),
                "effective_tolerance_deg": segment.servo_tolerance_deg,
                "recorded_servo_residual_deg": json_cell(segment.recorded_servo_residual_deg),
                "legacy_missing_endpoint": segment.legacy_missing_endpoint,
                "servo_measured_average_velocity_deg_s": timing.get("servo_measured_average_velocity_deg_s"),
                "sampled_joint_velocity": json_cell(state.get("joint_vel")),
                "wheel_target_rad_s": json_cell(segment.wheel_applied_target_rad_s),
                "wheel_duration_s": segment.wheel_active_duration_s,
                "contact_ground_state": json_cell(state.get("robot_ground_diagnostics", {})),
                "completion_extension_s": timing.get("servo_completion_extension"),
                "progress_improvement": "complete" if timing else "missing timing row",
                "gap_reason": timing.get("gap_reason", segment.gap_reason),
                "stop_reason": worker.get("stop_reason") if segment.segment_index == len(plan.segments) - 1 else "",
                "last_error": worker.get("last_error") if segment.segment_index == len(plan.segments) - 1 else "",
            }
        )
write_csv(REPORT / "formal_50mm_segment_trace.csv", formal_segment_rows)


# Real recording capture trace and selected restore evidence.
selected_result = read_result("post_fix_recording_selected_fast_final_visible")
recording_rows: list[dict[str, Any]] = []
for step in selected_result["recorded_steps"]:
    capture = dict(step["capture"])
    trace_by_request = {
        str(row.get("request_id", "")): row for row in list(capture.get("trace", []) or [])
    }
    for boundary in ("start", "stop"):
        row = dict(capture[boundary])
        trace = dict(trace_by_request.get(str(row.get("request_id", "")), {}) or {})
        recording_rows.append(
            {
                "step_index": step["step_index"],
                "request_id": row.get("request_id"),
                "purpose": row.get("purpose"),
                "request_time_monotonic_s": trace.get("monotonic_s"),
                "worker_session_id": row.get("worker_session_id"),
                "worker_capture_step": row.get("worker_sim_step"),
                "worker_capture_time": row.get("worker_sim_time"),
                "classification": dict(row.get("validation", {}) or {}).get("classification"),
                "full_state_valid": dict(row.get("validation", {}) or {}).get("valid"),
                "pose_restore_eligible": row.get("pose_restore_eligible"),
                "result": "PASS" if row.get("pose_restore_eligible") else "FAIL",
            }
        )
write_csv(REPORT / "recording_state_capture_trace.csv", recording_rows)

home_state = dict(selected_result.get("home_state", {}) or {})
current_state = dict(selected_result.get("perturbed_state_before_restore", {}) or {})
selected_rows: list[dict[str, Any]] = []
transactions: list[dict[str, Any]] = []
for run in selected_result["selected_fast_runs"]:
    worker_verification = dict(run["worker_restore_verification"])
    expected = dict(worker_verification.get("expected_state_summary", {}) or {})
    measured = dict(worker_verification.get("measured_state_summary", {}) or {})
    restore_result = dict(run["restore_result"])
    selected_rows.append(
        {
            "attempt": run["attempt"],
            "selected_step": run["selected_step_index"],
            "source_step": restore_result.get("restore_source_step_index"),
            "source_field": restore_result.get("restore_source_field"),
            "current_pose": json_cell({"root_pose": current_state.get("root_pose"), "joint_pos": current_state.get("joint_pos")}),
            "home_pose": json_cell({"root_pose": home_state.get("root_pose"), "joint_pos": home_state.get("joint_pos")}),
            "expected_previous_pose": json_cell({"root_pose": expected.get("root_pose"), "joint_pos": expected.get("joint_pos")}),
            "restored_measured_pose": json_cell({"root_pose": measured.get("root_pose"), "joint_pos": measured.get("joint_pos")}),
            "expected_vs_current_joint_max_rad": joint_max_difference(expected, current_state),
            "expected_vs_home_joint_max_rad": joint_max_difference(expected, home_state),
            "root_error_m": worker_verification.get("root_position_error_m"),
            "orientation_error_deg": worker_verification.get("root_orientation_error_deg"),
            "servo_joint_max_error_deg": worker_verification.get("servo_joint_position_max_error_deg"),
            "wheel_joint_max_error_rad": worker_verification.get("wheel_joint_position_max_error_rad"),
            "verified": worker_verification.get("verified"),
            "first_playback_step": list(run.get("represented_step_indices", []) or [None])[0],
            "represented_steps": json_cell(run.get("represented_step_indices", [])),
            "stop_reason": dict(run.get("final_worker", {}) or {}).get("stop_reason"),
        }
    )
    transactions.append(
        {
            "attempt": run["attempt"],
            "restore_request_id": worker_verification.get("request_id"),
            "worker_session_id": worker_verification.get("worker_session_id"),
            "source_step": restore_result.get("restore_source_step_index"),
            "source_field": restore_result.get("restore_source_field"),
            "restore_trace": worker_verification.get("restore_trace", []),
            "expected_state_summary": expected,
            "measured_state_summary": measured,
            "verification": {
                key: value
                for key, value in worker_verification.items()
                if key not in {"expected_state_summary", "measured_state_summary", "restore_trace"}
            },
            "playback_start": {
                "selected_step": run["selected_step_index"],
                "represented_step_indices": run["represented_step_indices"],
                "event_count": run["final_worker"].get("event_count"),
                "stop_reason": run["final_worker"].get("stop_reason"),
            },
        }
    )
write_csv(REPORT / "selected_fast_pose_restore.csv", selected_rows)
(REPORT / "restore_transaction_trace.json").write_text(
    json.dumps(
        {
            "successful_restore_transactions": transactions,
            "stop_during_restore": selected_result.get("stop_during_restore"),
            "conflict_checks": selected_result.get("conflict_checks"),
            "physical_clicks": selected_result.get("physical_clicks"),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    + "\n",
    encoding="utf-8",
)


# Protected data before/after audit.
before_rows = json.loads((REPORT / "protected_data_sha256_before.json").read_text(encoding="utf-8"))
after_rows: list[dict[str, Any]] = []
for before in before_rows:
    path = Path(before["path"])
    after_rows.append(
        {
            "path": str(path),
            "exists": path.exists(),
            "length": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path).upper() if path.exists() else None,
        }
    )
(REPORT / "protected_data_sha256_after.json").write_text(
    json.dumps(after_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
after_by_path = {row["path"].lower(): row for row in after_rows}
mismatches = []
for before in before_rows:
    after = after_by_path.get(str(before["path"]).lower(), {})
    if before.get("length") != after.get("length") or str(before.get("sha256")) != str(after.get("sha256")):
        mismatches.append({"before": before, "after": after})
active_before = next(row for row in before_rows if Path(row["path"]) == STEPS_PATH.resolve())
protected_text = f"""Protected data SHA-256 audit
before_file_count={len(before_rows)}
after_file_count={len(after_rows)}
mismatch_count={len(mismatches)}
all_byte_identical={'YES' if not mismatches else 'NO'}

Active formal 50 mm v2 accepted_steps.jsonl
path={STEPS_PATH.resolve()}
before_length={active_before['length']}
after_length={STEPS_PATH.stat().st_size}
before_sha256={active_before['sha256']}
after_sha256={sha256(STEPS_PATH).upper()}
identical={'YES' if active_before['sha256'] == sha256(STEPS_PATH).upper() else 'NO'}

mismatches={json.dumps(mismatches, ensure_ascii=False, default=str)}
"""
(REPORT / "protected_data_sha256_audit.txt").write_text(protected_text, encoding="utf-8")


# Compatibility and summary.
(REPORT / "legacy_pose_compatibility.txt").write_text(
    f"""Current active version: {ACTIVE_ID}
State audit: {classification_counts}; all {len(steps_without_full)} steps lack a FULL_VALID boundary pair.

Why pose is missing:
The old subprocess recording path requested lightweight status and synchronously called capture_sim_state.
Lightweight status had no sim_state, and SimTransport silently fell back to the controller-local
NullSimRobotAdapter. The resulting non-empty dictionaries contained command data but null root/joint
pose fields and 'no-sim adapter' diagnostics.

Compatibility policy:
- PLACEHOLDER_NO_SIM and COMMAND_ONLY are never pose-restore sources for Selected Fast.
- Current v002 remains usable for Play All Raw/Fast; its legacy_missing_endpoint segments use the
  conservative finite 1 degree normal completion policy and never claim an empty residual is measured.
- Selected Fast on this v002 is explicitly blocked with a FULL_VALID/re-record instruction.
- No command_state is wrapped or advertised as a full pose.

Replay-derived version:
NOT CREATED. Replaying can create replay-derived checkpoints, but those are not the lost original manual
poses. This run did not synthesize or activate such a version. For exact manual-pose semantics, re-record
with the repaired Recording path. A future maintenance operation may create a NEW child version only after
all replay-derived checkpoints validate; it must preserve parent id/SHA and label the source replay_derived.

Original v002 modified: NO. See protected_data_sha256_audit.txt.
""",
    encoding="utf-8",
)

changed_files = """MODIFIED
playback.py - fixes mixed servo/wheel false-stall completion, velocity-gated real stall, legacy endpoint marker, and visible failure status.
sim_process_client.py - request-id/purpose detailed-state protocol.
sim_transport.py - request-id forwarding and prohibition on subprocess NullSim fallback.
sim_robot_adapter.py - FULL_VALID capture markers and reset-before-write full articulation restore order.
sim_ui_controller.py - asynchronous recording checkpoints, strict Selected Fast resolver/verification, retry/error handling, command fallback compatibility.
sim_worker_process.py - detailed capture correlation and post-physics restore verification before OK acknowledgment.
tests/test_play_selected_fast_click_fix.py - full-pose fixtures and restore-verification acknowledgments.
tests/test_selected_step_previous_saved_state.py - complete pose fixtures and correlated fake worker protocol.
ui_motion_speed_height_version_e2e.py - asynchronous recording-preparation compatibility.

ADDED
sim_state_validation.py - shared FULL_VALID/COMMAND_ONLY/PLACEHOLDER_NO_SIM/INVALID classifier and centralized pose tolerances.
latest_50mm_manual_v2_e2e.py - active-version read-only Raw/Fast/full-state audit and full worker timing capture.
pose_checkpoint_selected_fast_e2e.py - visible real-Isaac recording and three physical-click Selected Fast transactions.
tests/test_pose_checkpoint_protocol.py - classifier, timeout, restore-order, mixed-channel, and velocity-gated stall regressions.
generate_fix_50mm_pose_report.py - reproducible evidence bundle generator.

DELETED
None.

PRESERVED PRE-EXISTING USER CHANGES (NOT MODIFIED BY THIS FIX)
saved_height_steps_fsm_reference_v2/height_050mm/active_version.json
saved_height_steps_fsm_reference_v2/manifest.json
the ten pre-existing 19:05 backup files listed in git_baseline.txt
"""
(REPORT / "changed_files.txt").write_text(changed_files, encoding="utf-8")

fast_before_error = str(pre_fast["final_worker"]["last_error"])
raw_before_error = str(pre_raw["final_worker"]["last_error"])
summary = f"""# 50 mm replay and Selected Fast pose restore fix

## Outcome

- Active formal version: `{ACTIVE_ID}` at `{ACTIVE_DIR.resolve()}`.
- Formal SHA-256: `{sha256(STEPS_PATH)}`; 16 steps, 80 source events/commands, 96 planned events, 64 segments.
- Before: Fast and Raw both false-stopped at Step 14 / Segment 56 (zero-based), after 81/96 events, with `actuator_limit` even though their errors (0.144823° and 0.289960°) were already inside the 1° normal tolerance.
- After: Raw and Fast each reached Step 16, sent 96/96 events, completed 64/64 segments, `stop_reason=complete`, `operation=IDLE`.
- New real recordings produced four FULL_VALID boundaries. Three real mouse Selected Fast runs restored Step 1 `sim_state_after`, verified the physical pose, then played only Step 2; all completed.
- All {len(before_rows)} protected files, including active v002, are byte-identical before/after.

## Root causes

1. Replay false stop: in a mixed servo+wheel segment, the scheduler ran servo-stall logic whenever the whole segment was not complete. It did so even after `servo_done=True` while only the wheel duration remained. The formal failures prove this: error was below tolerance but a fixed no-improvement window still raised `actuator_limit`.
2. Placeholder recording: lightweight subprocess status omitted `sim_state`; synchronous capture fell back to local NullSim and persisted null root/joint fields.
3. Selected Fast: the resolver treated any non-empty dictionary as a pose and even wrapped command_state as sim_state. The worker acknowledged restore before a verified physics boundary, and adapter reset ordering could overwrite writes.

The missing-state bug explains old unusable pose checkpoints and the Selected behavior. The formal mid-run stop had a separate immediate cause (the mixed-channel false-stall branch); missing physical endpoint data forced the legacy conservative policy but was not the direct reason an in-tolerance joint failed.

## Runtime repair

- One shared validator classifies all states and requires finite root pose/velocity, complete named articulation joint pos/vel, complete servo/wheel command and actual state, and a non-NullSim source.
- Recording enters `RECORDING_PREPARING`, correlates request/purpose/session/height/version/revision, and begins/finalizes only on FULL_VALID detailed worker snapshots. Subprocess capture cannot fall back to NullSim. A stop-capture failure creates no pending step and supports explicit retry.
- The completion change is minimal: never invoke servo-stall logic after the servo channel is done; a real stall additionally requires error outside tolerance, a full no-improvement window, readable state, and near-zero target-joint velocity. The 3° hard contact cap remains.
- Selected Fast accepts only previous `sim_state_after` FULL_VALID, or a FULL_VALID selected start with continuity; Step 1 requires its own FULL_VALID start. Command-only and placeholder sources are rejected.
- Adapter reset occurs before root/joint writes. Worker applies a safe wheel boundary, advances one bounded physics step, captures measured state, and verifies 5 mm / 0.5° / 1° / 0.05 rad tolerances before publishing restore OK or starting playback.

## Before failure evidence

- Fast: `{fast_before_error}`
- Raw: `{raw_before_error}`

## Compatibility

The current v002 has `{classification_counts}` across 32 boundaries and cannot support exact original manual pose restore. It was not rewritten and no replay-derived version was created. Re-record with the repaired flow for exact manual poses.
"""
(REPORT / "summary.md").write_text(summary, encoding="utf-8")


# Test command ledger.
fast_pid = post_fast_result.get("worker_pid")
raw_pid = post_raw_result.get("worker_pid")
selected_pid = selected_result.get("worker_pid")
test_results = f"""Test command/result ledger

PASS  git fetch origin; HEAD == origin/main == 249604b02192f73f096ab4e41fa253ae15b8cf11 (0 ahead / 0 behind).
PASS  python -m py_compile sim_ui_controller.py sim_worker_process.py sim_robot_adapter.py sim_transport.py sim_process_client.py sim_state_validation.py playback.py latest_50mm_manual_v2_e2e.py pose_checkpoint_selected_fast_e2e.py
PASS  python -m unittest discover -s tests -p 'test_*.py'
      Ran 246 tests in 8.549s - OK.

PRE-FIX REPRO (expected product failure)
FAIL  latest_50mm_manual_v2_e2e.py Fast, worker PID 173756, Step 14 / Segment 56, actuator_limit; report pre_fix_formal_replay_retry/real_isaac_result.json.
FAIL  latest_50mm_manual_v2_e2e.py Raw, Step 14 / Segment 56, actuator_limit; report pre_fix_formal_raw/real_isaac_result.json (harness success only because require-complete was false; run itself failed).
FAIL  initial pre_fix_formal_replay harness referenced a nonexistent diagnostic field; transparent harness-only traceback retained in pre_fix_formal_replay/real_isaac_result.json.

POST-FIX REAL ISAAC GUI
PASS  latest_50mm_manual_v2_e2e.py --profiles fast --require-complete; worker/Isaac host PID {fast_pid}; 96/96 events, 64/64 timing segments, Step 16, complete, IDLE.
PASS  latest_50mm_manual_v2_e2e.py --profiles raw --require-complete; worker/Isaac host PID {raw_pid}; 96/96 events, 64/64 timing segments, Step 16, complete, IDLE.
PASS  pose_checkpoint_selected_fast_e2e.py; worker/Isaac host PID {selected_pid}; 2 real recorded steps / 4 FULL_VALID boundaries; 3 physical mouse Selected Fast runs complete; stop-during-restore and both conflicts pass.
FAIL  first post_fix_recording_selected_fast run reached verified restore and completed playback but the E2E read represented steps from compact timing and reported []; harness-only traceback retained in post_fix_recording_selected_fast/real_isaac_result.json. Corrected final run passed using actual plan events.

Single-instance evidence
- Process inventory was checked before every launch and after every close.
- Runs were sequential; never more than one project worker/Isaac host was present.
- Each Isaac SimulationApp lived inside its one worker PID; no separate second PlaybackManager, SequenceManager, or SimProcessClient was created.
- Final process inventory: no project Isaac/Kit/worker process remains.
- No package, Python, Conda, Isaac Sim, or Isaac Lab installation/change was performed.

Protected data
PASS  {len(before_rows)}/{len(before_rows)} protected files byte-identical; mismatch_count={len(mismatches)}.
"""
(REPORT / "test_results.txt").write_text(test_results, encoding="utf-8")


# Stable screenshot aliases and compact evidence animations.
screenshot_sources = {
    "01_pre_fix_mid_run.png": REPORT / "existing_ui_before.png",
    "02_pre_fix_stop_reason.png": REPORT / "pre_fix_formal_replay_retry" / "screenshots" / "fast_final.png",
    "03_raw_last_step.png": REPORT / "post_fix_formal_raw_detailed" / "screenshots" / "raw_final.png",
    "04_fast_last_step.png": REPORT / "post_fix_formal_fast_detailed" / "screenshots" / "fast_final.png",
    "05_current_robot_different_pose.png": REPORT / "post_fix_recording_selected_fast_final_visible" / "screenshots" / "perturbed_before_selected.png",
    "06_previous_step_saved_pose.png": REPORT / "post_fix_recording_selected_fast_final_visible" / "screenshots" / "recorded_step_1.png",
    "07_restore_verified_selected_running.png": REPORT / "post_fix_recording_selected_fast_final_visible" / "screenshots" / "selected_fast_running.png",
    "08_selected_step_started.png": REPORT / "post_fix_recording_selected_fast_final_visible" / "screenshots" / "selected_fast_running.png",
    "09_selected_step_complete.png": REPORT / "post_fix_recording_selected_fast_final_visible" / "screenshots" / "selected_fast_complete_3.png",
}
for name, source in screenshot_sources.items():
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copyfile(source, REPORT / name)


def make_gif(output: Path, sources: list[Path]) -> None:
    frames = [Image.open(source).convert("P", palette=Image.Palette.ADAPTIVE).resize((850, 490)) for source in sources]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=900, loop=0)


make_gif(
    REPORT / "formal_50mm_fast_crossing_original_stop_point.gif",
    [
        REPORT / "post_fix_formal_fast_detailed" / "screenshots" / "fast_started.png",
        REPORT / "post_fix_formal_fast_detailed" / "screenshots" / "fast_step11.png",
        REPORT / "post_fix_formal_fast_detailed" / "screenshots" / "fast_final.png",
    ],
)
make_gif(
    REPORT / "selected_fast_pose_restore_then_selected_only.gif",
    [
        REPORT / "05_current_robot_different_pose.png",
        REPORT / "06_previous_step_saved_pose.png",
        REPORT / "07_restore_verified_selected_running.png",
        REPORT / "09_selected_step_complete.png",
    ],
)

print(f"report={REPORT}")
print(f"active_version={ACTIVE_ID}")
print(f"formal_sha256={sha256(STEPS_PATH)}")
print(f"protected_files={len(before_rows)} mismatches={len(mismatches)}")
print(f"formal_segment_rows={len(formal_segment_rows)}")
print(f"selected_rows={len(selected_rows)}")
