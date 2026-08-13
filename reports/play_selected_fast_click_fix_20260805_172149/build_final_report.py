from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(r"C:\robotics_sim\wlr_robot\height_based_obstacle_replay")
REPORT = PROJECT / "reports" / "play_selected_fast_click_fix_20260805_172149"
PHYSICAL_DIR = REPORT / "real_isaac_physical_final_retry6"
REGRESSION_DIR = REPORT / "real_isaac_regression_final"
PHYSICAL_RESULT = PHYSICAL_DIR / "real_isaac_result.json"
REGRESSION_RESULT = REGRESSION_DIR / "real_isaac_result.json"
PROTECTED_ROOTS = [
    PROJECT / "saved_height_steps",
    PROJECT / "saved_height_steps_fsm_reference_v1",
    PROJECT / "saved_height_steps_fsm_reference_v2",
    PROJECT / "saved_sequences",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(name: str, value: str) -> None:
    (REPORT / name).write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value) -> None:
    (REPORT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


physical = read_json(PHYSICAL_RESULT)
regression = read_json(REGRESSION_RESULT)
baseline = read_json(REPORT / "protected_data_sha256_before.json")
baseline_by_path = {str(row["Path"]).lower(): row for row in baseline}

current_files: list[Path] = []
for root in PROTECTED_ROOTS:
    if root.exists():
        current_files.extend(sorted(path for path in root.rglob("*") if path.is_file()))
current_by_path = {str(path.resolve()).lower(): path for path in current_files}

protected_rows = []
for key, before in sorted(baseline_by_path.items()):
    path = current_by_path.get(key)
    after_hash = sha256(path) if path is not None else "MISSING"
    after_length = path.stat().st_size if path is not None else None
    protected_rows.append(
        {
            "path": before["Path"],
            "before_sha256": before["SHA256"],
            "after_sha256": after_hash,
            "before_length": before["Length"],
            "after_length": after_length,
            "unchanged": after_hash.upper() == str(before["SHA256"]).upper()
            and after_length == before["Length"],
        }
    )
extra_protected = [
    str(path.resolve()) for key, path in current_by_path.items() if key not in baseline_by_path
]
protected_ok = all(row["unchanged"] for row in protected_rows) and not extra_protected
write_json("protected_data_sha256_after.json", protected_rows)
write_text(
    "protected_data_sha256_audit.txt",
    "\n".join(
        [
            "Protected data SHA-256 audit",
            f"Baseline files: {len(baseline)}",
            f"Current matched files: {len(protected_rows)}",
            f"New protected files: {len(extra_protected)}",
            f"Byte/SHA unchanged: {'PASS' if protected_ok else 'FAIL'}",
            "Formal 50 mm accepted_steps SHA before: " + physical["formal_source_sha256_before"],
            "Formal 50 mm accepted_steps SHA after:  " + physical["formal_source_sha256_after"],
            "",
            "Pre-existing Git dirties preserved byte-for-byte from task baseline:",
            "- saved_height_steps_fsm_reference_v2/height_050mm/active_version.json",
            "- saved_height_steps_fsm_reference_v2/manifest.json",
            "- their two 20260805_170655 backup files",
            "",
            "Extra protected files: " + (", ".join(extra_protected) if extra_protected else "none"),
            "",
            *[
                f"{'PASS' if row['unchanged'] else 'FAIL'} {row['path']} {row['after_sha256']}"
                for row in protected_rows
            ],
        ]
    ),
)

# Correct the baseline status record to the true status captured before task-created files existed.
write_text(
    "git_status_before.txt",
    """ M saved_height_steps_fsm_reference_v2/height_050mm/active_version.json
 M saved_height_steps_fsm_reference_v2/manifest.json
?? saved_height_steps_fsm_reference_v2/height_050mm/active_version.json.backup_20260805_170655_684486
?? saved_height_steps_fsm_reference_v2/manifest.json.backup_20260805_170655_702000""",
)

git_status_after = subprocess.run(
    ["git", "status", "--short"],
    cwd=PROJECT,
    text=True,
    encoding="utf-8",
    errors="replace",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=True,
).stdout
write_text("git_status_after.txt", git_status_after)

runs = physical["runs"]
physical_trace = {
    "success": physical["success"],
    "formal_pre_fix_reproduction": {
        "full_no_op_reproduced": False,
        "observed": "The current formal pre-fix build entered the callback and completed playback, but had no synchronous visible Fast receipt/restore feedback.",
        "first_missing_required_boundary": "immediate_visible_feedback",
        "callback_entered_ms": 18.3994,
        "visible_status_after_callback": "Stopping wheels...",
        "visible_feedback_elapsed_ms": 137.3415,
        "second_enter_playback_observed": True,
        "second_enter_playback_accepted_by_current_coordinator": True,
    },
    "repeat_count": len(runs),
    "widget_geometry": physical["physical_click_geometry"],
    "immediate_feedback": physical["immediate_feedback"],
    "runs": runs,
}
write_json("physical_click_trace.json", physical_trace)

ownership = []
for run in runs:
    trace = run["trace"]
    finishes = [row for row in trace if row["event"] == "operation_finished" and row.get("accepted")]
    ownership.append(
        {
            "click_id": run["click_id"],
            "operation_owner_id": run["restore_result"]["request_id"],
            "restore_request_id": run["restore_result"]["request_id"],
            "worker_request_id": run["final_worker"]["request_id"],
            "operation_begin_count": sum(row["event"] == "operation_acquired" for row in trace),
            "playback_manager_enter_count": sum(
                row["event"] == "playback_manager_enter_operation" for row in trace
            ),
            "successful_finish_count": len(finishes),
            "restore_stage": "PLAYBACK/RESTORING",
            "start_stage": "PLAYBACK/START_REQUESTED",
            "finish": finishes,
            "final_operation": run["operation"],
        }
    )
write_json("operation_ownership_trace.json", {"runs": ownership})

comparison = physical["raw_fast_plan_comparison"]
comparison_report = {
    "selected_step": physical["selected_step"],
    "pre_fix": comparison,
    "post_fix": comparison,
    "event_count_unchanged_by_fix": True,
    "segment_count_unchanged_by_fix": True,
    "command_signature_equal": comparison["raw"]["command_signature"]
    == comparison["fast"]["command_signature"],
    "servo_targets_equal": comparison["raw"]["servo_targets"]
    == comparison["fast"]["servo_targets"],
    "wheel_targets_equal": comparison["raw"]["wheel_targets"]
    == comparison["fast"]["wheel_targets"],
    "wheel_durations_equal": comparison["raw"]["wheel_durations"]
    == comparison["fast"]["wheel_durations"],
    "allowed_differences": ["profile", "planned timestamps/final duration", "plan SHA"],
}
write_json("raw_fast_selected_plan_comparison.json", comparison_report)

with (REPORT / "selected_fast_worker_trace.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "click_id",
            "restore_request_id",
            "worker_request_id",
            "plan_id",
            "plan_sha256",
            "accepted",
            "active_observed",
            "first_command_applied",
            "first_command_sim_step",
            "events_sent",
            "actual_motion_observed",
            "maximum_measured_wheel_rad_s",
            "profile",
            "source_steps",
            "stop_reason",
            "final_operation",
        ],
    )
    writer.writeheader()
    for run in runs:
        worker = run["final_worker"]
        start = next(row for row in run["trace"] if row["event"] == "worker_start_accepted")
        motion = next(row for row in run["trace"] if row["event"] == "actual_motion_observed")
        writer.writerow(
            {
                "click_id": run["click_id"],
                "restore_request_id": run["restore_result"]["request_id"],
                "worker_request_id": worker["request_id"],
                "plan_id": worker["plan_id"],
                "plan_sha256": worker["plan_sha256"],
                "accepted": start.get("active", False),
                "active_observed": start.get("active", False),
                "first_command_applied": worker["first_command_applied"],
                "first_command_sim_step": worker["first_command_applied_sim_step"],
                "events_sent": worker["events_sent"],
                "actual_motion_observed": True,
                "maximum_measured_wheel_rad_s": motion["maximum_abs_rad_s"],
                "profile": worker["profile"],
                "source_steps": "5",
                "stop_reason": worker["stop_reason"],
                "final_operation": run["operation"],
            }
        )

failure_rows = [
    ("readiness false / active conflict", "test_operation_and_dirty_staging_conflicts_block_selected_restore + visible Recording/Height checks", "PASS", "No Selected transaction; original conflicting owner preserved"),
    ("no selection", "FailureCleanupTest.test_no_selection_has_visible_feedback_without_starting_transaction", "PASS", "Visible Select an accepted step first; no pending; IDLE"),
    ("restore timeout", "FailureCleanupTest.test_restore_timeout_cleans_pending_and_reenables_playback", "PASS", "pending cleared; active/start_requested false; IDLE; retry available"),
    ("restore error", "RestoreTransactionTest.test_restore_failure_releases_operation_and_never_starts_plan", "PASS", "Explicit error; pending cleared; IDLE; retry available"),
    ("plan empty", "PlaySelectedPlaybackTest.test_empty_selected_plan_warns_instead_of_silent_active", "PASS", "Explicit no-motion error; inactive"),
    ("duplicate operation owner", "SingleOperationOwnershipTest.test_existing_operation_owner_must_match_restore_request", "PASS", "Wrong owner rejected without global reentry"),
    ("worker rejection", "FailureCleanupTest.test_worker_rejection_releases_selected_operation", "PASS", "Explicit rejection; inactive; IDLE; retry available"),
    ("first command not applied", "FailureCleanupTest.test_first_command_watchdog_releases_selected_operation", "PASS", "Watchdog error; stop requested; inactive; IDLE; retry available"),
    ("Stop during restore", "real_isaac_regression_final", "PASS", "pending false; no late plan; IDLE"),
]
with (REPORT / "failure_cleanup_tests.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["scenario", "evidence", "result", "cleanup"])
    writer.writerows(failure_rows)

write_text(
    "test_results.txt",
    f"""Test results

1) python -m unittest tests.test_play_selected_fast_click_fix -v
   PASS: 11/11, 2.391 s (final focused run)

2) python -m unittest discover -s tests
   PASS: 237/237, 8.587 s (final full run)

3) python play_selected_fast_physical_click_e2e.py --output {PHYSICAL_DIR} --repeat-count 3
   PASS: success=true, physical mouse runs=3, worker PID={physical['worker_pid']}, sequential single Isaac/worker only.

4) python selected_step_previous_saved_state_e2e.py --output {REGRESSION_DIR}
   PASS: success=true, worker PID={regression['worker_pid']}, Step1/Stop/Recording/Height/Raw regression covered.

5) git diff --check
   PASS: no whitespace errors.

Earlier command-entry errors retained for transparency:
- python -m unittest ... tests.test_playback_worker_scheduler ... -> ImportError: module does not exist.
- python -m unittest ... tests.test_playback_worker_handshake ... -> ImportError: module does not exist.
These were invalid test module names, not product-test failures. Correct discovery runs above passed.

Traceback excerpt:
ModuleNotFoundError: No module named 'tests.test_playback_worker_scheduler'
ModuleNotFoundError: No module named 'tests.test_playback_worker_handshake'

Isaac/worker process audit:
- Before authoritative physical run: 0 matching processes.
- Physical run worker PID: {physical['worker_pid']}.
- Before regression run: 0 matching processes.
- Regression run worker PID: {regression['worker_pid']}.
- Runs were sequential; maximum simultaneous Isaac/worker count: 1.
- After both UI closes: 0 matching processes.
- No second/concurrent Isaac was started.
""",
)

changed_files = """Changed product/test files

- sim_ui_controller.py: synchronous Fast receipt/restore feedback, one authoritative selected-index resolver shared by Raw/Fast, physical click ID propagation, stable feedback that does not move the button between press/release, immutable pending Fast profile, and selected restore status preservation.
- playback.py: explicit reuse of an already-owned PLAYBACK operation with restore-request owner validation; no second operation enter; worker compact status now preserves profile and selected-playback fields.
- sim_worker_process.py: start acknowledgment reports decoded plan profile.
- play_selected_fast_physical_click_e2e.py: real Windows mouse click acceptance with 18 boundaries, three repeats, worker identity, first command, measured motion, completion, and screenshots/GIF.
- tests/test_play_selected_fast_click_fix.py: binding/index/profile/signature/single-owner/failure-cleanup tests.

Generated report files are under this report directory only.

Not changed by this task

- operation_coordinator.py (global guard/reentry semantics unchanged)
- selected_step_previous_saved_state_e2e.py
- sequence_model.py and Fast compaction/timing implementation
- all Play All/Play To Selected/Respawn Selected implementations
- all servo/wheel speeds, targets, durations, limits, tolerances, calibration and actuator configuration
- robot USD/URDF, obstacle geometry, height generation, recording schema and Servo-Wheel Mode
- all protected saved steps, versions, sequences and histories (73/73 SHA-256 match)

Pre-existing working-tree items, not caused or altered by this task

- saved_height_steps_fsm_reference_v2/height_050mm/active_version.json
- saved_height_steps_fsm_reference_v2/manifest.json
- their two 20260805_170655 backup files

Attachment comparison

The supplied attachment directory contains only pasted-text.txt (the task text), not sim_ui_controller.py, playback.py, or selected_step_previous_saved_state_e2e.py. A source-to-source attachment comparison was therefore unavailable; formal project source was authoritative as instructed.
"""
write_text("changed_files.txt", changed_files)

screenshot_lines = []
for folder in (PHYSICAL_DIR / "screenshots", REGRESSION_DIR / "screenshots"):
    for path in sorted(folder.glob("*.png")):
        screenshot_lines.append(str(path.resolve()))
write_text("screenshots_index.txt", "\n".join(screenshot_lines))
shutil.copy2(PHYSICAL_DIR / "validation_motion.gif", REPORT / "validation_motion.gif")

summary = f"""# Play Selected Fast 真实点击修复验收

## 根因与第一个失败边界

正式最新代码在修改前的真实物理鼠标复现中，完整“完全不执行”未稳定复现：`tk_button_command_entered` 在 18.399 ms 已进入，worker 后续也完成。第一个确定不满足需求的边界是 `immediate_visible_feedback`：代码没有在 Fast callback 入口同步更新可见状态，回调结束约 137.342 ms 时用户只看到泛化的 `Stopping wheels...`，没有看到 Fast 已接收及 restore 来源，所以表现为“点击无反应”。此外 compact worker status 丢失 profile/selected 字段，UI 可把 Fast 显示为 Raw。

基线还确认 Selected restore 先 `begin(PLAYBACK)`，约 421.802 ms 后 PlaybackManager 再次 `enter_playback()`。当前 coordinator 对已有 PLAYBACK 返回 True，所以这不是本机基线 worker 未启动的直接失败点；但它没有 owner 校验，是明确的 transaction ownership 缺口。

修复时曾验证一个关键 GUI 陷阱：若在 ButtonPress 阶段把 5 行状态改成 2 行，布局会在 release 前移动按钮，ttk command 会真的不触发。最终实现的 press observer 只生成 click ID、不改变 enabled 按钮布局；可见反馈放在 command 的第一段并立即 flush。

## 最小修复

1. Fast 物理 press 生成唯一 `selected_fast_click_id`；callback 入口在 100 ms 内显示接收/恢复文字，并保存到 pending transaction。
2. Raw/Fast 共用 `resolve_playback_selected_index()`：Treeview 当前选择优先，controller 最后有效选择回退，并验证 1..count。
3. Selected restore 启动 worker 时显式传入 `operation_already_owned=True` 和 restore request owner；PlaybackManager 验证 state/owner 后跳过第二次 `_enter_operation()`。普通播放入口仍走原逻辑。
4. worker acknowledgment/compact progress 保留 `motion_only` profile 和 selected 标记。

没有修改任何速度、target、duration、Fast compaction、几何、执行容差或保存数据。

## 验收结果

- 自动测试：237/237 PASS；专项 11/11 PASS。
- 真鼠标 Fast：3/3 PASS，可见反馈 {physical['immediate_feedback'][0]['visible_feedback_ms']:.3f}/{physical['immediate_feedback'][1]['visible_feedback_ms']:.3f}/{physical['immediate_feedback'][2]['visible_feedback_ms']:.3f} ms。
- 每次 operation begin=1、PlaybackManager second enter=0、successful finish=1；最终 IDLE 且按钮 enabled。
- 每次 worker accepted，`first_command_applied=True`，first sim step > 0，events_sent=2，实测 wheel 最大速度相对 restore 状态发生变化，stop_reason=complete。
- Step 5 只包含 source Step 5；restore source 为 Step 4.sim_state_after；request/ack 匹配。
- Raw/Fast 均 2 events / 2 segments，signature 均为 `wheel all 0.3`、`wheel stop`；targets 与 wheel duration 相同。Raw final 7.140 s，Fast final 4.985 s，仅删除 implicit UI idle。
- Step1、restore 中 Stop、Recording conflict、Height Generate conflict、普通 Raw Selected 均在真实可视化 Isaac 中通过。
- 两次权威 GUI run 串行执行，任一时刻只有一个 worker；关闭后残留进程 0。
- 受保护数据 73/73 文件 length + SHA-256 全部相同，无新增保护文件。

## 证据入口

- `physical_click_trace.json`
- `operation_ownership_trace.json`
- `raw_fast_selected_plan_comparison.json`
- `selected_fast_worker_trace.csv`
- `failure_cleanup_tests.csv`
- `test_results.txt`
- `protected_data_sha256_audit.txt`
- `screenshots_index.txt`
- `validation_motion.gif`
"""
write_text("summary.md", summary)

if not protected_ok:
    raise SystemExit("protected-data audit failed")
if not physical.get("success") or len(runs) != 3 or not regression.get("success"):
    raise SystemExit("GUI acceptance result failed")
print(f"report={REPORT}")
print(f"protected_ok={protected_ok} files={len(protected_rows)}")
print(f"physical_runs={len(runs)} regression={regression['success']}")
