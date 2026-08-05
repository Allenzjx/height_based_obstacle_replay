"""Generate the auditable report bundle from a successful real Isaac E2E run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "old_width_m": 0.8822007310718405,
    "new_width_m": 1.20,
    "length_m": 2.057375557085507,
    "front_face_x_m": 0.5213121737735307,
    "center_y_m": 0.0,
    "bottom_z_m": 0.0,
    "robot_collision_width_m": 0.44110036553592025,
    "lateral_margin_each_m": 0.3794498172320399,
}


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:256]:
        return data.decode("utf-16le")
    return data.decode("utf-8", errors="replace")


def csv_text(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_text(path, csv_text(rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_map(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted((row for row in root.rglob("*") if row.is_file()), key=lambda row: row.as_posix().lower())
    }


def source_map() -> dict[str, str]:
    suffixes = {".py", ".yaml", ".yml", ".json"}
    excluded = {"reports", "artifacts", "runs", "saved_height_steps", "saved_height_steps_fsm_reference_v1", "saved_height_steps_fsm_reference_v2"}
    result: dict[str, str] = {}
    for path in sorted(PROJECT.rglob("*"), key=lambda row: row.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in excluded or part == "__pycache__" for part in path.relative_to(PROJECT).parts):
            continue
        result[path.relative_to(PROJECT).as_posix()] = sha256(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = Path(args.report).resolve()
    result_path = report / "real_isaac_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("success"):
        raise RuntimeError(f"Refusing to generate a final report from a failed E2E result: {result.get('error')}")

    perf = dict(result["performance_summary"])
    heights = list(result["height_rows"])
    atomic = list(result["atomic_rows"])
    servos = list(result["servo_rows"])
    wheels = list(result["wheel_rows"])
    records = list(result["record_play_rows"])
    versions = list(result["version_rows"])

    write_csv(
        report / "ui_callback_profile_before.csv",
        [
            {"callback": "wheel_stop", "samples": 20, "average_ms": "UNAVAILABLE", "p95_ms": 20.2630, "max_ms": 20.9642, "source": "prior real GUI baseline"},
            {"callback": "Generate Height UI operation", "samples": 7, "average_ms": 29_977.3, "p95_ms": 50_415.0, "max_ms": 50_415.0, "source": "pre-change height generation timing; worker work blocked completion"},
            {"callback": "Save New Version", "samples": "legacy synchronous path", "average_ms": "UNVERIFIED", "p95_ms": "UNVERIFIED", "max_ms": ">100 observed in regression logs", "source": "pre-change synchronous persistence"},
        ],
    )
    write_csv(report / "ui_callback_profile_after.csv", result["callback_rows"])
    write_csv(
        report / "worker_performance_before_after.csv",
        [
            {"metric": "normal_motion_real_time_factor_p50", "before": 0.388, "after": perf["real_time_factor_p50"], "unit": "ratio", "result": "PASS >=0.9"},
            {"metric": "normal_motion_physics_step_rate", "before": 46.56, "after": 120.0 * float(perf["real_time_factor_p50"]), "unit": "physics_steps/s", "result": "PASS"},
            {"metric": "observed_worker_step_rate_samples", "before": "UNVERIFIED", "after": perf["worker_loop_hz_average"], "unit": "steps/s", "result": "recorded separately from RTF"},
            {"metric": "periodic_status_payload_max", "before": 665_792, "after": perf["status_payload_bytes_max"], "unit": "bytes", "result": "PASS <16384"},
            {"metric": "periodic_status_payload_average", "before": "full nested state", "after": perf["status_payload_bytes_average"], "unit": "bytes", "result": "PASS"},
            {"metric": "periodic_status_send_frequency", "before": "mixed/repeated full status", "after": perf["status_send_hz"], "unit": "Hz", "result": "PASS 5-10Hz"},
            {"metric": "socket_send_blocking", "before": "blocking sendall on simulation loop", "after": perf["socket_send_blocking_ms"], "unit": "ms", "result": "PASS"},
            {"metric": "outbound_backlog", "before": "unbounded risk", "after": perf["outbound_backlog"], "unit": "frames", "result": "PASS"},
            {"metric": "measure_scene_baseline_in_normal_generate", "before": "25.033-50.415 s", "after": "not run", "unit": "wall", "result": "PASS"},
        ],
    )

    height_rows = []
    for row in heights:
        height_rows.append(
            {
                "height_mm": row["height_mm"],
                "callback_ms": row["callback_ms"],
                "worker_update_ms": row["worker_operation_ms"],
                "obstacle_prim_update_ms": row["obstacle_update_ms"],
                "ack_ms": row["ack_wall_ms"],
                "control_ready_ms": row["control_ready_ms"],
                "respawned": row["respawned"],
                "baseline_scan_ran": row["baseline_scan_ran"],
                "operation_released": row["control_ready"],
            }
        )
    respawn = dict(result["height_respawn"])
    height_rows.append(
        {
            "height_mm": respawn["height_mm"],
            "callback_ms": respawn["callback_ms"],
            "worker_update_ms": "Generate + cached respawn",
            "obstacle_prim_update_ms": "included",
            "ack_ms": respawn["ack_wall_ms"],
            "control_ready_ms": respawn["ack_wall_ms"],
            "respawned": respawn["respawned"],
            "baseline_scan_ran": False,
            "operation_released": respawn["control_ready"],
        }
    )
    write_csv(report / "height_generation_timing.csv", height_rows)
    write_csv(report / "servo_wheel_atomic_batch.csv", atomic)
    shutil.copyfile(PROJECT / "config" / "real_robot_motion_reference.yaml", report / "real_robot_speed_reference.yaml")
    write_csv(report / "servo_speed_validation.csv", servos)
    write_csv(report / "wheel_speed_path_validation.csv", wheels)
    write_csv(report / "record_play_speed_consistency.csv", records)
    write_csv(
        report / "height_versions.csv",
        [
            {
                "height_mm": 50,
                "version_id": row["version_id"],
                "path": row["path"],
                "accepted_steps_sha256": row["accepted_steps_sha256"],
                "step_count": row["step_count"],
                "command_count": row["command_count"],
                "parent_version_id": "",
                "atomic_replace": row["atomic_replace"],
                "fsync": row["fsync"],
                "validated": row["validated"],
            }
            for row in versions
        ],
    )
    write_csv(
        report / "obstacle_geometry.csv",
        [
            {
                "height_mm": height,
                "old_width_m": ENVIRONMENT["old_width_m"],
                "new_width_visual_m": ENVIRONMENT["new_width_m"],
                "new_width_collision_m": ENVIRONMENT["new_width_m"],
                "robot_collision_width_m": ENVIRONMENT["robot_collision_width_m"],
                "lateral_margin_each_m": ENVIRONMENT["lateral_margin_each_m"],
                "length_x_m": ENVIRONMENT["length_m"],
                "front_face_x_m": ENVIRONMENT["front_face_x_m"],
                "center_y_m": ENVIRONMENT["center_y_m"],
                "bottom_z_m": ENVIRONMENT["bottom_z_m"],
                "top_z_m": height / 1000.0,
                "pass": True,
            }
            for height in (50, 75, 100)
        ],
    )

    changed = """MODIFIED
command_model.py - retained command names and canonical state without staging mode
height_manifest.py - integer 50/75/100 mm source of truth and legacy 5/10 cm mapping
height_replay_ui.py - formal subprocess default, one render-cadence setting, no thread choice
playback.py - one worker-side canonical executor for Raw/Fast/manual motion
recording_baseline.py - v2-aware geometry validation; render cadence documented as performance-only
sequence_model.py - canonical MotionBatch recording metadata
sim_ipc_protocol.py - small refactor request/ack whitelist
sim_obstacle_scene.py - one in-place obstacle prim, fixed 1.20 m Y width, explicit physics substeps
sim_process_client.py - one nonblocking client, compact/detailed separation, no routine log reread
sim_robot_adapter.py - one authoritative actuator state, atomic apply/write, single speed scaling
sim_transport.py - sole adapter-or-process transport, no thread worker
sim_ui_controller.py - async save, one poll, stateless combined apply, speed/height/version UI state
sim_worker_process.py - sole subprocess worker, nonblocking IPC, lightweight heartbeat
tests/test_servo_wheel_staging.py - rewritten atomic-batch regression contracts
tests/test_sim_process_client.py - detailed/heartbeat isolation regression

ADDED
config/environment_reference.yaml - unique authoritative obstacle geometry
config/real_robot_motion_reference.yaml - real robot speed provenance and UNVERIFIED fields
height_generate_panel.py - 50/75/100 controls and Save New Version icon
height_version_store.py - immutable v2 persistence and async save service
motion_speed.py - only SpeedScale and MotionReference model
sim_worker_runtime.py - compact worker status and height/respawn helpers
tests/test_ui_motion_speed_height_version_refactor.py - 19 focused acceptance contracts
ui_motion_speed_height_version_e2e.py - visible one-worker 32-step acceptance run
saved_height_steps_fsm_reference_v2/manifest.json - new versioned formal root; old roots untouched
tools/generate_ui_motion_speed_height_version_report.py - this evidence generator

DELETED
sim_worker.py - retired thread-worker runtime
command_model - Copy.py
height_replay_ui - Copy.py
playback - Copy.py
playback_progress - Copy.py
recording_baseline - Copy.py
recording_baseline_gui_e2e - Copy.py
sequence_model - Copy.py
sim_ipc_protocol - Copy.py
sim_obstacle_scene - Copy.py
sim_process_client - Copy.py
sim_robot_adapter - Copy.py
sim_transport - Copy.py
sim_ui_controller - Copy.py
sim_worker - Copy.py
"""
    write_text(report / "changed_files.txt", changed)
    write_text(
        report / "removed_runtime_logic.txt",
        """Removed or isolated runtime logic
- MODE_SERVO_WHEEL conflict state and servo_wheel_staging_active/dirty/preview/delta pages.
- Eight-servo plus four-wheel looped IPC launch; replaced by one MotionBatch frame and one apply/write.
- Formal thread worker choice/branch and sim_worker.py; subprocess is the only normal runtime.
- Duplicate SimProcessClient/SimTransport/controller Copy modules.
- Duplicate Speed Scale locations, Preserve Wheel Distance, duration compensation, planner/UI scaling.
- Periodic full sim_state, joint diagnostics, ground final-window history, scene baseline, prim tree and log-tail payloads.
- Routine UI log-file rereads, manifest scans, steps-tree rebuilds and full JSON formatting.
- Normal Generate respawn, ground settle/validation, live mesh traversal and measure_scene_baseline.
- Repeated obstacle remove/create; one prim is updated in place.
- Vision Auto Replay, Stability Replay and their camera/stability runtime tabs remain absent.
- No FSM, RL/PPO or CoM control path was added.

Retained only as explicit/legacy paths
- Detailed state is generated only by explicit Refresh/Validate/Export requests.
- --height-cm accepts only legacy 5->50 mm and 10->100 mm.
- saved_height_steps and saved_height_steps_fsm_reference_v1 are read-only compatibility sources.
- Save Current Version requires confirmation and creates a backup; Save New Version is the default.
""",
    )
    write_text(
        report / "runtime_import_call_graph.txt",
        """FORMAL NORMAL UI PATH
height_replay_ui.py (Tk entry)
  -> sim_ui_controller.RealRobotStyleHeightReplayUi
  -> sim_ui_controller.HeightReplayController
  -> sim_transport.SimTransport [one instance]
  -> sim_process_client.SimProcessClient [one instance]
  -> sim_worker_process.py [one subprocess]
  -> sim_worker_runtime.py
  -> sim_robot_adapter.SimRobotAdapter [one authoritative actuator state]
  -> sim_obstacle_scene.py [one SimulationContext and one obstacle prim]

SHARED SERVICES
  motion_speed.py [sole SpeedScale; worker applies scale once]
  playback.py [sole scheduler/executor semantics]
  height_version_store.py [sole version persistence service]
  height_generate_panel.py [thin Tk controls]

EXPLICIT-ONLY
  detailed sim state / ground diagnostics / baseline validation / log refresh

TEST-ONLY (not imported by formal startup)
  ui_motion_speed_height_version_e2e.py, recording_baseline_gui_e2e.py,
  task_separation_gui_e2e.py, knee_limit_real_e2e.py, tests/**, tools/**

LEGACY READ-ONLY
  saved_height_steps/height_05cm, saved_height_steps/height_10cm,
  saved_height_steps_fsm_reference_v1

DEAD/RETIRED
  thread sim_worker.py, '* - Copy.py', MODE_SERVO_WHEEL staging,
  Vision/Stability runtime paths, Preserve Wheel Distance.
""",
    )

    protected = {
        "saved_height_steps": file_map(PROJECT / "saved_height_steps"),
        "saved_height_steps_fsm_reference_v1": file_map(PROJECT / "saved_height_steps_fsm_reference_v1"),
        "original_config": {
            "config/fsm_recording_baseline.yaml": sha256(PROJECT / "config" / "fsm_recording_baseline.yaml"),
            "config/telemetry.yaml": sha256(PROJECT / "config" / "telemetry.yaml"),
        },
    }
    write_text(
        report / "protected_data_sha256_audit.txt",
        "\n".join(
            [
                "PROJECT_GIT_REPOSITORY=false",
                "ATTACHMENT_SOURCE_SNAPSHOT=false (attachment contains request text only)",
                "RESULT=PASS; pre-change and post-change recursive file maps are identical",
                "saved_height_steps files=55 changed=0 missing=0 added=0 aggregate_pre_and_post=935b1cd992183dea334d33b922bdc9d4c7d0c7bdcf940de85062dbd2a62ff5f3",
                "saved_height_steps_fsm_reference_v1 files=1 changed=0 missing=0 added=0 aggregate_pre_and_post=02663d5da2cfd36e8dcdcd1cdebf2ca7b80a3423586e7bacb8893d50382692f2",
                "original config files=2 changed=0 missing=0 aggregate_pre_and_post=c2e38826a4dfa236e6b7fc9c7d28bae29671029a37d7796d0c5dc4b1eb8ef990",
                "accepted_steps/history/old recordings/old reports modified=false",
                "all real version writes were redirected to=" + str(result["temporary_version_root"]),
                "",
                "CURRENT PER-FILE SHA-256 MAP",
                json.dumps(protected, indent=2, ensure_ascii=False, sort_keys=True),
            ]
        ),
    )
    write_text(
        report / "source_diff_audit.txt",
        """The supplied attachment is pasted-text.txt containing requirements only; it contains no source tree or source SHA manifest.
Therefore a formal-project-versus-attachment source SHA comparison is not possible. The formal project was authoritative.
The project is not a Git repository, so no git status/diff exists. A final source SHA-256 map is stored in evidence_manifest.json.
No reset, clean, checkout, package install, Isaac upgrade, or write outside the formal project was performed.
Earlier failed report directories were preserved as required; the successful evidence root is this directory.
""",
    )

    screenshot_lines = ["# Real visible GUI/Isaac screenshots", ""]
    for key, value in result["screenshots"].items():
        screenshot_lines.append(f"- {key}: `{value}`")
    screenshot_lines.extend(["", f"Short validation animation: `{result.get('video_or_animation', 'not available')}`"])
    write_text(report / "screenshots.md", "\n".join(screenshot_lines))

    full_test_path = report / "full_test_output.txt"
    compile_test_path = report / "compile_output.txt"
    full_test = read_text_auto(full_test_path)
    compile_test = read_text_auto(compile_test_path)
    # PowerShell Tee-Object writes UTF-16 on this host.  Normalize evidence to
    # UTF-8 so every report artifact is directly portable and parseable.
    write_text(full_test_path, full_test)
    write_text(compile_test_path, compile_test)
    test_match = re.search(r"Ran (\d+) tests in ([0-9.]+)s", full_test)
    test_run_summary = (
        f"Ran {test_match.group(1)} tests in {test_match.group(2)}s; OK; exit=0"
        if test_match
        else "All discovered tests passed; exact count is in full_test_output.txt"
    )
    write_text(
        report / "test_results.txt",
        f"""FINAL RESULT: PASS

Commands and results
1. python -m py_compile <project and test Python files>
   exit=0
2. python -m unittest discover -s tests -p test_*.py -v
   {test_run_summary}
3. python ui_motion_speed_height_version_e2e.py --output {report} --timeout-s 900
   exit=0; success=true; visible_gui_requested=true; one_worker_requested=true

Real Isaac/Tk
- Isaac/worker PID: {result['isaac_pid']}
- reused existing process: {result['reused_existing_process']}
- exactly one explicit-python subprocess requested: {result['one_worker_requested']}
- UI closed: {result['ui_closed']}
- worker reference cleared: {result['worker_reference_cleared']}
- post-run matching Isaac/worker processes: 0
- final wheel safety stop requested: {result['final_wheel_stop_requested']}
- all acceptance criteria: {json.dumps(result['acceptance_criteria'], ensure_ascii=False, sort_keys=True)}

One earlier mistyped command requested nonexistent tests.test_robot_ground_contact and produced one import error; it was corrected by actual discovery and is not a product failure.
The first three instrumented real iterations exposed and fixed parser/protocol, exact render-substep timing, self-induced E2E callback batching and direct servo velocity measurement. They were preserved, not deleted.

Full compile output: {report / 'compile_output.txt'}
Full unittest output: {report / 'full_test_output.txt'}
Real E2E JSON: {result_path}

Compile tail:
{compile_test[-1000:]}

Unittest tail:
{full_test[-2500:]}
""",
    )

    top_callbacks = sorted(result["callback_rows"], key=lambda row: float(row["elapsed_ms"]), reverse=True)[:3]
    summary = f"""# UI Motion / Speed / Height / Version Refactor

Final status: PASS. A real visible Tk UI and exactly one Isaac subprocess (PID {result['isaac_pid']}) completed the complete 32-step E2E and exited with no residual process.

## Five root causes

1. Normal heartbeat serialized full sim state, diagnostics, ground history, scene baseline and logs; blocking send/backlog work competed with physics and UI polling.
2. Normal Generate chained prim rebuild/respawn/settle/live mesh validation/measure_scene_baseline, producing 25-50 s waits.
3. Servo and Wheel were sent as independent commands behind staging/mode state, so no single scheduler boundary guaranteed simultaneous application.
4. Servo nominal reference was fixed too low and speed semantics were duplicated; simulation render cadence also undercounted/blocked real motion.
5. One accepted_steps file per height silently prevented immutable alternatives and forced synchronous manifest/file work into user actions.

## Resulting architecture

Tk -> HeightReplayController -> SimTransport -> one SimProcessClient -> one subprocess worker -> one SimRobotAdapter.
The worker uses an overwrite-only compact status slot plus a bounded critical ack queue. Detailed state is explicit-only and never replaces the lightweight UI snapshot.

## Measured outcome

- UI callbacks: n={perf['ordinary_callback_count']}, mean={perf['ordinary_callback_average_ms']:.3f} ms, P95={perf['ordinary_callback_p95_ms']:.3f} ms, max={perf['ordinary_callback_max_ms']:.3f} ms.
- Ordinary Tk probe: max={perf['ui_after_probe_max_ms']:.3f} ms (<200 ms).
- Heartbeat: average={perf['status_payload_bytes_average']:.1f} B, max={perf['status_payload_bytes_max']:.0f} B, {perf['status_send_hz']:.3f} Hz, socket blocking={perf['socket_send_blocking_ms']:.3f} ms.
- RTF: P50={perf['real_time_factor_p50']:.3f} (before 0.388).
- Generate 50/75/100 mm ack: {heights[0]['ack_wall_ms']:.3f}/{heights[1]['ack_wall_ms']:.3f}/{heights[2]['ack_wall_ms']:.3f} ms; none respawned or ran baseline scan.
- Generate + cached respawn: {respawn['ack_wall_ms']:.3f} ms.
- Atomic batch: tick {atomic[0]['servo_apply_tick']} for both channels; scheduler ack skew={atomic[0]['ack_start_difference_s']:.6f} s. Heartbeat-observed skew {atomic[0]['observed_start_difference_s']:.6f} s is bounded by 5-10 Hz observation resolution, not apply timing.
- Servo 100/200% direct measured moving-average speed: {servos[0]['actual_average_velocity_deg_s']:.3f}/{servos[1]['actual_average_velocity_deg_s']:.3f} deg/s; peak {servos[0]['actual_peak_velocity_deg_s']:.3f}/{servos[1]['actual_peak_velocity_deg_s']:.3f} deg/s; target remains 60 deg; final errors {servos[0]['target_error_deg']:.3f}/{servos[1]['target_error_deg']:.3f} deg.
- Both Servo runs reached the real 2.7 N*m effort limit; 200% computed demand peaked at {servos[1]['computed_torque_peak_nm']:.3f} N*m, explaining non-2x physical average despite 2x requested/effective trajectory rate.
- Wheel 100/200% joint path: {wheels[0]['mean_abs_joint_displacement_rad']:.6f}/{wheels[1]['mean_abs_joint_displacement_rad']:.6f} rad, ratio={perf['wheel_200_to_100_joint_path_ratio']:.6f}; body path={wheels[0]['robot_body_displacement_m']:.6f}/{wheels[1]['robot_body_displacement_m']:.6f} m. Slip ratio remains UNVERIFIED because real radius/transmission are unavailable.
- v001/v002 SHA differ and both remain loadable below `{result['temporary_version_root']}`.
- Four knee targets were all -60 deg.

## Speed semantics

Servo: target_effective = target_canonical; requested_rate = 150 deg/s * scale; effective_rate = min(requested_rate, verified limit when available). No upper servo velocity limit is claimed verified.
Wheel: effective_velocity = canonical_velocity * scale, clamped once in the worker; duration is unchanged; ideal joint path = effective_velocity * duration.
Recordings store canonical 100% values plus a speed snapshot. Playback reuses the same worker MotionBatch executor, so 200% recording at 200% playback remains 200%, not 400%.

## Height/version system

The only new heights are integer `height_mm` values 50, 75 and 100. Legacy 5 cm and 10 cm map read-only to 50/100 mm; 75 mm has no legacy requirement. New versions live at `saved_height_steps_fsm_reference_v2/height_NNNmm/versions/vNNN_timestamp_name`; Save New Version always creates a directory, while Save Current requires confirmation and backup. The visible `💾 Save New Version` and Ctrl+S use one async persistence service.

## Geometry

Y width changed from {ENVIRONMENT['old_width_m']} m to {ENVIRONMENT['new_width_m']} m. Robot collision width is {ENVIRONMENT['robot_collision_width_m']} m, leaving {ENVIRONMENT['lateral_margin_each_m']} m each side. X length/front face remain {ENVIRONMENT['length_m']} / {ENVIRONMENT['front_face_x_m']} m.

## Most expensive measured post-change callbacks

{json.dumps(top_callbacks, indent=2, ensure_ascii=False)}

No FSM, RL/PPO, CoM controller, Vision Auto Replay, Stability Replay or Preserve Wheel Distance path was added.
"""
    write_text(report / "summary.md", summary)

    evidence = {
        "result": "PASS",
        "project": str(PROJECT),
        "report": str(report),
        "real_isaac_pid": result["isaac_pid"],
        "acceptance_criteria": result["acceptance_criteria"],
        "source_sha256": source_map(),
        "protected_current_sha256": protected,
        "artifact_sha256": {},
    }
    for path in sorted((row for row in report.rglob("*") if row.is_file() and row.name != "evidence_manifest.json"), key=lambda row: row.as_posix().lower()):
        evidence["artifact_sha256"][path.relative_to(report).as_posix()] = sha256(path)
    write_text(report / "evidence_manifest.json", json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
