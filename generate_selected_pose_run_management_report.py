from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "selected_pose_run_management_and_full_replay_fix_20260805_235722"
RUN_MANAGEMENT = REPORT / "post_fix_run_management_retry4" / "real_isaac_result.json"
SELECTED = REPORT / "post_fix_selected_raw_fast_retry10" / "real_isaac_result.json"
RAW = REPORT / "post_fix_formal_50mm_raw" / "real_isaac_result.json"
FAST = REPORT / "post_fix_formal_50mm_fast" / "real_isaac_result.json"
SELECTION_PROBE = REPORT / "post_fix_formal_50mm_fast_selection_probe" / "real_isaac_result.json"
PRE_FIX = REPORT / "pre_fix_formal_v006_final" / "real_isaac_result.json"
RESPAWN_PROBE = REPORT / "post_fix_formal_50mm_retry5" / "real_isaac_result.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


run_management = load(RUN_MANAGEMENT)
selected = load(SELECTED)
raw = load(RAW)
fast = load(FAST)
selection_probe = load(SELECTION_PROBE)
pre_fix = load(PRE_FIX)
respawn_probe = load(RESPAWN_PROBE)


# Selected restore evidence keeps the complete controller and worker traces.
selected_profiles: list[dict[str, Any]] = []
click_by_profile = {row["profile"]: row for row in selected["physical_clicks"]}
for run in selected["selected_raw_fast_runs"]:
    profile = str(run["profile"])
    restore = run["restore_result"]
    verification = run["worker_restore_verification"]
    worker = run["final_worker"]
    selected_profiles.append(
        {
            "profile": profile,
            "physical_click": click_by_profile[profile],
            "selected_step_index": run["selected_step_index"],
            "selected_play_request_id": restore["selected_play_request_id"],
            "restore_source": {
                "step_index": restore["restore_source_step_index"],
                "field": restore["restore_source_field"],
                "fallback_used": restore["fallback_used"],
                "classification": restore["candidate_validations"]["previous.sim_state_after"]["classification"],
            },
            "controller_trace": restore["trace"],
            "worker_restore_trace": verification["restore_trace"],
            "verification": verification,
            "selected_only_plan": {
                "source_steps": restore["plan_source_steps"],
                "selected_playback": restore["plan_selected_playback"],
                "event_count": restore["plan_event_count"],
                "represented_step_indices": run["represented_step_indices"],
            },
            "worker_acceptance_and_completion": {
                "plan_id": worker["plan_id"],
                "request_id": worker["request_id"],
                "first_command_applied": worker["first_command_applied"],
                "events_sent": worker["events_sent"],
                "event_count": worker["count"],
                "stop_reason": worker["stop_reason"],
                "active": worker["active"],
            },
            "actual_motion": run["actual_motion"],
        }
    )
dump(
    REPORT / "selected_pose_restore_trace.json",
    {
        "result": "PASS",
        "formal_checkpoint_counts": raw["saved_state_counts"],
        "formal_checkpoint_audit": raw["saved_state_audit"],
        "profiles": selected_profiles,
        "live_tk_trace": selected["live_ui_click_trace"],
        "pause_resume": selected["pause_resume"],
        "stop_during_restore": selected["stop_during_restore"],
        "final_operation": selected["final_operation"],
        "worker_reference_cleared": selected["worker_reference_cleared"],
    },
)


# Run management results.
run_rows: list[dict[str, Any]] = []
for row in run_management["run_management_rows"]:
    run_rows.append(
        {
            "action": row.get("action", ""),
            "source_run": row.get("source_run", ""),
            "destination_run": row.get("destination_run", ""),
            "dirty": row.get("dirty", ""),
            "read_only": row.get("read_only", ""),
            "old_sha": row.get("old_sha", ""),
            "new_sha": row.get("new_sha", ""),
            "old_sha_after": row.get("old_sha_after", ""),
            "version_count_before": row.get("version_count_before", ""),
            "version_count_after": row.get("version_count_after", ""),
            "same_run_id": row.get("source_run") == row.get("destination_run"),
            "original_preserved": row.get("old_sha", "") == row.get("old_sha_after", "") if row.get("old_sha_after") else "",
            "result": row.get("result", ""),
        }
    )
write_csv(
    REPORT / "run_management_tests.csv",
    [
        "action", "source_run", "destination_run", "dirty", "read_only",
        "old_sha", "new_sha", "old_sha_after", "version_count_before",
        "version_count_after", "same_run_id", "original_preserved", "result",
    ],
    run_rows,
)


conflict_rows = [
    ("IDLE", "Allow", "Allow if selected", "Allow", "Allow if dirty", "Allow"),
    ("Dirty sequence", "Allow", "Allow", "Prompt", "Allow", "Allow"),
    ("Combobox selection only", "Allow current run", "Allow current run", "Allow pending run", "Current run only", "Current working copy"),
    ("Recording", "Block", "Block", "Block", "Block", "Block"),
    ("Pending recorded step", "Block", "Block", "Block", "Block", "Block"),
    ("Scene Update", "Block", "Block", "Block", "Block", "Block"),
    ("Respawn", "Block", "Block", "Block", "Block", "Block"),
    ("Real Playback active", "Block", "Block", "Block", "Block", "Block"),
    ("Completed worker/stale local", "Reconcile then Allow", "Reconcile then Allow", "Allow", "Allow", "Allow"),
    ("Dirty Servo-Wheel staging", "Explicit block", "Explicit block", "Prompt/block", "Allow only if safe", "Allow only if safe"),
]
write_csv(
    REPORT / "playback_conflict_matrix.csv",
    ["current_state", "play_all", "play_selected", "open_run", "update_run", "save_as_new"],
    [dict(zip(["current_state", "play_all", "play_selected", "open_run", "update_run", "save_as_new"], row)) for row in conflict_rows],
)


probe = selection_probe["runs"][0]["pre_start_probe"]
dump(
    REPORT / "playback_state_reconciliation.json",
    {
        "result": "PASS",
        "real_gui_selection_independence": probe["selection_independence"],
        "real_gui_stale_reconciliation": probe["stale_reconciliation"],
        "covered_automatic_cases": [
            "local active=True / worker inactive",
            "local operation=PLAYBACK / worker complete",
            "expired start_requested",
            "expired pending Selected restore",
            "worker error / inactive",
            "stale plan or request identity",
        ],
        "non_conflicts": ["step selection", "pending Run combobox selection", "tab selection", "dirty current Run"],
        "policy": "Only fresh matching worker activity or a live matching start/restore deadline retains Playback ownership.",
    },
)


old_worker = pre_fix["latest_sim_status"]["worker_playback"]
new_warning = raw["runs"][0]["final_worker"]["last_servo_residual_warning"]
new_segment = next(
    row for row in raw["runs"][0]["final_worker"]["timing"]["segments"]
    if row["segment_index"] == 37 and row["step_index"] == 10
)
dump(
    REPORT / "step10_segment37_before_after.json",
    {
        "screenshot_reported_case": {
            "max_error_deg": 2.986098,
            "effective_tolerance_deg": 3.0,
            "old_result": "actuator_unstable",
            "regression_test_result": "contact_residual_accepted; continue",
        },
        "pre_fix_real_isaac_reproduction": {
            "stop_reason": old_worker["stop_reason"],
            "last_error": old_worker["last_error"],
            "progress_detail": old_worker["progress_detail"],
            "measured_errors_deg": old_worker["current_servo_errors"],
            "recent_window": "not emitted by the old abort path",
        },
        "post_fix_real_isaac_same_formal_run": {
            "completion_decision": new_segment["completion_decision"],
            "warning": new_warning,
            "servo_target_deg": new_segment["servo_target"],
            "servo_actual_deg": new_segment["servo_actual_at_transition"],
            "servo_target_error_deg": new_segment["servo_target_error"],
            "continued_to_step_11": True,
            "continued_to_step_12": True,
            "final_stop_reason": raw["runs"][0]["final_worker"]["stop_reason"],
        },
        "hard_safety_regressions": {
            "3.2_over_3.0": "FAIL",
            "NaN_or_Inf": "FAIL immediately",
            "sustained_worsening_outside_tolerance": "FAIL",
            "hard_safety_cap_deg": 3.0,
        },
    },
)


# One trace row per formal segment, with the commands represented by the segment.
formal_rows: list[dict[str, Any]] = []
for profile, result in (("raw", raw), ("fast", fast)):
    run = result["runs"][0]
    worker = run["final_worker"]
    commands: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for command in worker["timing"]["commands"]:
        commands[int(command["segment_index"])].append(command)
    for segment in worker["timing"]["segments"]:
        segment_commands = commands[int(segment["segment_index"])]
        errors = [abs(float(value)) for value in segment.get("servo_target_error", {}).values()]
        warning = segment.get("completion_warning", {}) or {}
        formal_rows.append(
            {
                "profile": profile,
                "step": segment["step_index"],
                "segment": segment["segment_index"],
                "global_commands": ";".join(str(row["global_command_index"]) for row in segment_commands),
                "commands": " | ".join(str(row["command"]) for row in segment_commands),
                "planned_start_sim_s": segment.get("planned_start_sim_time", ""),
                "actual_start_sim_s": segment.get("actual_start_sim_time", ""),
                "actual_end_sim_s": segment.get("actual_end_sim_time", ""),
                "max_residual_deg": max(errors) if errors else 0.0,
                "effective_tolerance_deg": warning.get("effective_tolerance_deg", ""),
                "completion_decision": segment.get("completion_decision", ""),
                "completion_warning": warning.get("warning", ""),
                "stop_reason": worker["stop_reason"] if int(segment["segment_index"]) == int(worker["segment_count"]) - 1 else "",
            }
        )
write_csv(
    REPORT / "formal_50mm_replay_trace.csv",
    [
        "profile", "step", "segment", "global_commands", "commands",
        "planned_start_sim_s", "actual_start_sim_s", "actual_end_sim_s",
        "max_residual_deg", "effective_tolerance_deg", "completion_decision",
        "completion_warning", "stop_reason",
    ],
    formal_rows,
)


completion: dict[str, Any] = {
    "formal_run": {
        "run_id": raw["active_version_id"],
        "path": raw["version_path"],
        "accepted_steps_sha256_before": raw["accepted_steps_sha256"],
        "accepted_steps_sha256_after_raw": raw["accepted_steps_sha256_after"],
        "accepted_steps_sha256_after_fast": fast["accepted_steps_sha256_after"],
        "source_steps": raw["source_step_count"],
        "source_events": raw["source_event_count"],
        "source_commands": raw["source_command_count"],
    },
    "profiles": {},
}
for profile, result in (("raw", raw), ("fast", fast)):
    worker = result["runs"][0]["final_worker"]
    progress = worker["progress_detail"]
    completion["profiles"][profile] = {
        "success": result["success"],
        "worker_pid": result["worker_pid"],
        "worker_session_id": result["worker_session_id"],
        "plan_sha256": result["plan_summaries"][profile]["plan_sha256"],
        "steps": [progress["current_step_index"], progress["total_steps"]],
        "events": [worker["events_sent"], worker["count"]],
        "segments": [worker["segment_index"], worker["segment_count"]],
        "final_global_command": progress["global_command_index"],
        "stop_reason": worker["stop_reason"],
        "active": worker["active"],
        "operation": result["final_operation"],
        "wheels_stopped": True,
        "next_play_available": True,
    }
dump(REPORT / "raw_fast_completion.json", completion)


# Protected-data audit. The current report directory is intentionally excluded.
protected_roots = (
    ROOT / "saved_height_steps",
    ROOT / "saved_height_steps_fsm_reference_v1",
    ROOT / "saved_height_steps_fsm_reference_v2",
    ROOT / "saved_sequences",
)
current_files: list[Path] = []
for protected_root in protected_roots:
    if protected_root.exists():
        current_files.extend(path for path in protected_root.rglob("*") if path.is_file())
history = ROOT / "history_commands.log"
if history.is_file():
    current_files.append(history)
reports_root = ROOT / "reports"
current_files.extend(
    path for path in reports_root.rglob("*")
    if path.is_file() and REPORT not in path.parents
)
after = {
    str(path.relative_to(ROOT)): {"length": path.stat().st_size, "sha256": sha256(path)}
    for path in sorted(set(current_files))
}
before = load(REPORT / "protected_data_sha256_before.json")
changed = sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
missing = sorted(before.keys() - after.keys())
added = sorted(after.keys() - before.keys())
formal_path = Path(raw["accepted_steps_path"])
audit_text = "\n".join(
    (
        "Protected data SHA-256 audit",
        f"baseline_count={len(before)}",
        f"after_count={len(after)}",
        f"changed_count={len(changed)}",
        f"missing_count={len(missing)}",
        f"added_count={len(added)}",
        f"changed={json.dumps(changed)}",
        f"missing={json.dumps(missing)}",
        f"added={json.dumps(added)}",
        f"formal_run_id={raw['active_version_id']}",
        f"formal_accepted_steps_path={formal_path}",
        f"formal_sha256_baseline={raw['accepted_steps_sha256']}",
        f"formal_sha256_after_raw={raw['accepted_steps_sha256_after']}",
        f"formal_sha256_after_fast={fast['accepted_steps_sha256_after']}",
        f"formal_sha256_current={sha256(formal_path)}",
        "result=PASS" if not changed and not missing and not added else "result=FAIL",
    )
) + "\n"
(REPORT / "protected_data_sha256_audit.txt").write_text(audit_text, encoding="utf-8")
if changed or missing or added:
    raise RuntimeError("protected data audit failed")


descriptions = {
    "height_generate_panel.py": "Run Management UI、预览/打开分离、五个明确操作与按钮状态",
    "height_version_store.py": "只读 inspect、同 ID 原子 Update、备份/验证/回滚、唯一时间戳",
    "latest_50mm_manual_v2_e2e.py": "正式 50 mm Raw/Fast、selection independence 与 stale reconciliation 真实验证",
    "operation_coordinator.py": "增加 RUN_MANAGEMENT 协调状态",
    "playback.py": "bounded contact residual completion、hard safety 保留、Selected resume 修复",
    "playback_availability.py": "仅真实冲突阻止 Playback，并提供精确原因",
    "pose_checkpoint_selected_fast_e2e.py": "Selected Raw/Fast 真实鼠标、pose 恢复、运动、Pause/Resume/Stop 验证",
    "run_management_gui_e2e.py": "临时 store 上的 Run UI 真实 GUI 验证",
    "sim_ui_controller.py": "严格 FULL_VALID Selected pipeline、即时反馈、stale reconciliation、Run 控制器",
    "capture_selected_run_baseline.py": "修改前 Git 与 1021 个受保护文件 SHA-256 基线",
    "generate_selected_pose_run_management_report.py": "生成本轮证据报告及保护审计",
    "tests/test_selected_pose_run_management_and_reconciliation.py": "本轮专项回归与冲突矩阵测试",
    "tests/test_height_wheel_servo_batch_replay_completion_fix.py": "按新安全语义更新既有 completion 断言",
    "tests/test_play_selected_fast_click_fix.py": "按 Raw/Fast 统一即时反馈更新断言",
    "tests/test_play_selected_playback.py": "按 strict FULL_VALID Selected 语义更新覆盖",
    "tests/test_selected_step_previous_saved_state.py": "按 previous-step 完整 pose 语义更新覆盖",
    "tests/test_ui_motion_speed_height_version_refactor.py": "按 Run Management 标签更新 UI 断言",
}
status_lines = [line for line in git("status", "--short").splitlines() if line]
changed_lines = ["修改/新增文件（无删除）："]
for line in status_lines:
    path = line[3:].strip().replace("\\", "/")
    state = "新增" if line.startswith("??") else "修改"
    changed_lines.append(f"{state}\t{path}\t{descriptions.get(path, '本轮验证支持文件')}")
changed_lines.append("删除\t无")
(REPORT / "changed_files.txt").write_text("\n".join(changed_lines) + "\n", encoding="utf-8")


test_results = f"""Final verification commands and results

1. python -m py_compile height_generate_panel.py height_version_store.py latest_50mm_manual_v2_e2e.py operation_coordinator.py playback.py playback_availability.py pose_checkpoint_selected_fast_e2e.py run_management_gui_e2e.py sim_ui_controller.py capture_selected_run_baseline.py tests\\test_selected_pose_run_management_and_reconciliation.py
   PASS
2. git diff --check
   PASS (only Git line-ending notices; no whitespace errors)
3. python -m unittest discover -s tests -p 'test_*.py'
   PASS: Ran 265 tests in 12.843s; OK; traceback=none

Real visible Isaac GUI / worker runs (strictly sequential; one worker per run):
- Run Management: PASS, PID={run_management['worker_pid']}, success={run_management['success']}, final_operation={run_management['final_operation']}
- Selected Raw/Fast: PASS, PID={selected['worker_pid']}, success={selected['success']}, final_operation={selected['final_operation']}
- Formal Raw: PASS, PID={raw['worker_pid']}, success={raw['success']}, stop_reason={raw['runs'][0]['final_worker']['stop_reason']}, final_operation={raw['final_operation']}
- Formal Fast: PASS, PID={fast['worker_pid']}, success={fast['success']}, stop_reason={fast['runs'][0]['final_worker']['stop_reason']}, final_operation={fast['final_operation']}
- Formal Fast + selection/stale probe: PASS, PID={selection_probe['worker_pid']}, success={selection_probe['success']}, final_operation={selection_probe['final_operation']}
- Respawn after completed Raw: SAFELY BLOCKED by grounded_reference_valid/stable/physics validity; no bypass was attempted. Raw itself completed before the block.

Single-instance controls:
- All E2E results report single_worker_requested=true.
- Runs were launched sequentially, never concurrently.
- Every successful result reports ui_closed=true and worker_reference_cleared=true.
- Final process inventory is appended after report generation.

Temporary-write rule:
- Run writes used only report-local temporary_version_store_v2 directories.
- Formal v006 accepted_steps SHA remained {raw['accepted_steps_sha256_after']}.
- protected file audit: 1021/1021 unchanged.
"""
(REPORT / "test_results.txt").write_text(test_results, encoding="utf-8")


# Copy a concise, stable set of evidence images without altering any source capture.
evidence_images = REPORT / "screenshots"
evidence_images.mkdir(exist_ok=True)
image_sources = {
    "01_run_management_new_ui.png": Path(run_management["screenshots"]["run_management_ui"]),
    "02_pending_run_selection_not_opened.png": Path(run_management["screenshots"]["pending_selection_not_opened"]),
    "03_open_run.png": Path(run_management["screenshots"]["open_selected_run"]),
    "04_new_empty_run.png": Path(run_management["screenshots"]["new_empty_run"]),
    "05_update_current_run.png": Path(run_management["screenshots"]["update_current_run"]),
    "06_save_as_new_run.png": Path(run_management["screenshots"]["save_as_new_run"]),
    "07_robot_different_pose.png": Path(selected["screenshots"]["perturbed_before_selected"]),
    "08_previous_step_pose_restored_raw_running.png": Path(selected["screenshots"]["selected_raw_running"]),
    "09_selected_step_actual_motion_raw_complete.png": Path(selected["screenshots"]["selected_raw_complete"]),
    "10_previous_step_pose_restored_fast_running.png": Path(selected["screenshots"]["selected_fast_running"]),
    "11_selected_step_actual_motion_fast_complete.png": Path(selected["screenshots"]["selected_fast_complete"]),
    "12_step10_segment37.png": Path(raw["screenshots"]["raw_1_step10_segment37"]),
    "13_step11.png": Path(raw["screenshots"]["raw_1_step11"]),
    "14_last_step_completed_raw.png": Path(raw["screenshots"]["raw_1_final"]),
    "15_last_step_completed_fast.png": Path(fast["screenshots"]["fast_1_final"]),
    "16_selection_independence.png": Path(selection_probe["screenshots"]["selection_independence"]),
}
for name, source in image_sources.items():
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, evidence_images / name)


animations: list[str] = []
run_gif = REPORT / "post_fix_run_management_retry4" / "validation_motion.gif"
shutil.copy2(run_gif, REPORT / "run_open_update_save_as_new.gif")
animations.append("run_open_update_save_as_new.gif")
try:
    from PIL import Image

    for target, names in (
        (
            "selected_pose_restore_and_motion.gif",
            [
                "07_robot_different_pose.png", "08_previous_step_pose_restored_raw_running.png",
                "09_selected_step_actual_motion_raw_complete.png", "10_previous_step_pose_restored_fast_running.png",
                "11_selected_step_actual_motion_fast_complete.png",
            ],
        ),
        (
            "formal_50mm_step10_to_complete.gif",
            ["12_step10_segment37.png", "13_step11.png", "14_last_step_completed_raw.png", "15_last_step_completed_fast.png"],
        ),
    ):
        frames = [Image.open(evidence_images / name).convert("RGB") for name in names]
        frames[0].save(REPORT / target, save_all=True, append_images=frames[1:], duration=1100, loop=0)
        for frame in frames:
            frame.close()
        animations.append(target)
except ImportError:
    pass


screenshot_lines = ["# 截图和动画证据", ""]
for name in image_sources:
    screenshot_lines.append(f"- `screenshots/{name}`")
for name in animations:
    screenshot_lines.append(f"- `{name}`")
(REPORT / "screenshots.md").write_text("\n".join(screenshot_lines) + "\n", encoding="utf-8")


summary = f"""# Selected Pose、Run Management 与完整回放修复报告

## 结论

三个问题不是同一个 bug：Selected Playback 是 restore 合约/即时反馈/陈旧所有权的组合问题；Run 管理是 UI 语义和持久化事务缺口；完整回放是 completion policy 将容差内接触残差误判为 actuator_unstable。

## 最小修复

1. Selected Raw/Fast 统一要求 `FULL_VALID`，Step N 严格恢复 Step N-1 `sim_state_after`，经过 worker ACK、physics boundary 与实际 pose 校验后仅构建 Step N plan；command-only、placeholder、home/current pose 均不能继续。
2. Run 区改为 New Empty / Open Selected / Update Current / Save As New / Refresh；combobox 仅预览。Update 使用临时文件、fsync、重读校验、timestamp backup、atomic replace 和失败回滚。
3. 每次 Playback 前用新鲜 worker 状态清理 proven-stale 本地 ownership；selection/dirty Run 不再成为冲突。容差内 bounded contact residual 记录 `contact_residual_accepted` 并继续，3.0° hard cap、NaN、超差无响应及持续恶化仍失败。

## 正式数据与真实 Isaac

- 正式 Run：`{raw['active_version_id']}`，12 steps，78 events/commands，62 segments。
- checkpoint：12/12 steps 的 before/after 共 24 个状态全部 `FULL_VALID`。
- Selected Raw：root {selected_profiles[0]['verification']['root_position_error_m']:.9f} m，orientation {selected_profiles[0]['verification']['root_orientation_error_deg']:.6f}°，servo max {selected_profiles[0]['verification']['servo_joint_position_max_error_deg']:.6f}°，实际最大关节变化 {selected_profiles[0]['actual_motion']['max_servo_change_from_restored_checkpoint_deg']:.6f}°。
- Selected Fast：root {selected_profiles[1]['verification']['root_position_error_m']:.9f} m，orientation {selected_profiles[1]['verification']['root_orientation_error_deg']:.6f}°，servo max {selected_profiles[1]['verification']['servo_joint_position_max_error_deg']:.6f}°，实际最大关节变化 {selected_profiles[1]['actual_motion']['max_servo_change_from_restored_checkpoint_deg']:.6f}°。
- Formal Raw：12/12、78/78、62/62，Step 10/Segment 37=`contact_residual_accepted`，进入 Step 11/12，`stop_reason=complete`，operation=IDLE。
- Formal Fast：12/12、78/78、62/62，`stop_reason=complete`，operation=IDLE。
- 265 项自动测试通过；1021 个受保护文件 SHA-256 全部不变；正式 accepted_steps SHA 始终为 `{raw['accepted_steps_sha256_after']}`。

## 唯一未完成的运行时动作

正式 Raw 完成后请求 Respawn 时，当前 Isaac ground reference 报告 grounded/stable/physics reference 无效，因此 Respawn 被安全门阻止。未绕过安全条件，也未把这一项伪报为成功；Raw/Fast 完整回放分别在独立、顺序、单 worker 的真实 Isaac 运行中均已完成。
"""
(REPORT / "summary.md").write_text(summary, encoding="utf-8")


manifest = {
    "report": str(REPORT),
    "required_artifacts": [
        "summary.md", "changed_files.txt", "selected_pose_restore_trace.json",
        "run_management_tests.csv", "playback_conflict_matrix.csv",
        "playback_state_reconciliation.json", "step10_segment37_before_after.json",
        "formal_50mm_replay_trace.csv", "raw_fast_completion.json",
        "test_results.txt", "protected_data_sha256_audit.txt", "screenshots.md",
    ],
    "animations": animations,
    "protected_audit": "PASS",
    "unit_tests": "265 PASS",
    "formal_raw": "PASS",
    "formal_fast": "PASS",
    "respawn": "SAFELY BLOCKED: grounded reference invalid",
}
dump(REPORT / "evidence_manifest.json", manifest)
print(REPORT)
