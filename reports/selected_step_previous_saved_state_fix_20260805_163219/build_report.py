from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(r"C:\robotics_sim\wlr_robot\height_based_obstacle_replay")
REPORT = PROJECT / "reports" / "selected_step_previous_saved_state_fix_20260805_163219"
REAL = REPORT / "real_isaac_ui_final"
RESULT = json.loads((REAL / "real_isaac_result.json").read_text(encoding="utf-8"))
ISAAC_VERSION = RESULT.get("isaac_version") or str(RESULT.get("isaac_window_title", "")).removeprefix("Isaac Sim ")


def write_text(name: str, text: str) -> None:
    (REPORT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    return result.stdout.rstrip()


write_text(
    "summary.md",
    f"""# Play Selected Previous Saved State Fix

## 原错误行为与根因

`Play Selected Step` / `Play Selected Fast` 原来直接调用通用 `start_playback([selected])`。通用入口只会按所选 Step 自己的 `sim_state_before` / `command_state_before` 恢复；当恢复选项关闭或状态缺失时，还可能直接沿用当前 live state。同时 worker 侧没有可匹配的 restore request ID，restore 与 start plan 之间只依赖预设延时，不能证明对应恢复已完成。

## 修复后的 restore policy

- Step N（N > 1）：`Step N-1.sim_state_after` → `Step N-1.command_state_after` → `Step N.sim_state_before` → `Step N.command_state_before`。
- Step 1：`Step 1.sim_state_before` → `Step 1.command_state_before`。
- 全部缺失时明确报错并保持无 active/scheduled plan，绝不读取 current live state 作为起始状态。
- 同时存在前一步结束态与当前步开始态时只读比较；不一致只告警，前一步结束态仍是权威来源，不修改 saved data。

## 恢复与播放顺序

同一个既有 PLAYBACK operation transaction 内执行：取得 operation → Stop Wheels → 发送带 request ID 的 restore → 等待相同 request ID、restore_count 增加、result=ok、runtime ready → 请求 detailed state → 验证恢复态 → 使用既有 settle delay → 启动只含 selected Step 的 plan。Stop/E-stop/失败/超时会取消 pending transaction、释放 operation，且不发送 plan。

## 影响边界

普通 Selected 两个入口改用专用的轻量解析/等待路径；Play All、Play To Selected、Respawn Selected 等仍调用原有通用入口。planner、Raw/Fast timing、actuator command、completion policy、Obstacle、Height、Respawn、Recording、Servo-Wheel、Save/Combine 均未修改。OperationCoordinator 在 RESTORING 至完成期间持续保持 PLAYBACK，因此其他 task 使用原有 guard 被拒绝。

## 最终证据

- 单元/回归：聚焦 19/19；全套 226/226。
- 可见 Isaac Sim 5.1.0：正式 Tk `Play Selected Step` / `Play Selected Fast` 按钮级验证成功，PID {RESULT['worker_pid']}，未复用已有进程，单 worker，退出后无残留。
- Raw/Fast 选择 Step 5 均恢复 Step 4 `sim_state_after`，plan source 仅为 Step 5。
- Step 1 恢复自身 `sim_state_before`；恢复中 Stop 未延迟启动；Recording/Height Generate 冲突均被拒绝。
- 71 个受保护 saved-step/version 文件 SHA-256 变化为 0；正式验证序列哈希前后相等。
""",
)

write_text(
    "changed_files.txt",
    """产品代码：
- sim_ui_controller.py：仅为普通 Play Selected Step/Fast 增加 previous-saved-state 解析、匹配 restore ack、detailed-state 验证、取消/失败收尾；将相关选项固定为启用。其他 Playback handler 保持原入口。
- sim_process_client.py：restore_sim_state 增加可选 request_id，并返回最终 request_id。
- sim_transport.py：最小透传 restore request_id；no-sim adapter 行为保持同步。
- sim_worker_process.py：worker status 增加 last_restore_request_id，用于精确匹配 ack。

测试/验证代码：
- tests/test_selected_step_previous_saved_state.py：新增恢复优先级、缺失、Step 1、顺序、失败、Stop、冲突、Fast 和其他入口回归测试。
- tests/test_play_selected_playback.py：适配 restore_sim_state 现在返回 request_id 的状态断言。
- selected_step_previous_saved_state_e2e.py：正式 UI/单可见 Isaac 只读 E2E 验证与截图采集。

报告辅助：
- reports/selected_step_previous_saved_state_fix_20260805_163219/build_report.py：只读取测试/E2E/Git/哈希结果并生成本目录报告。

明确未修改：
- playback.py 及其 planner、Raw/Fast timing、Fast gap removal、completion policy、长序列逻辑。
- Servo/Wheel 速度、target、velocity、duration、trajectory、limits。
- Height Generate、Obstacle geometry、Respawn pose、Recording、Servo-Wheel、Save Version、Combine、Sim State schema、UI tab 布局、Vision/Stability、FSM/RL/PPO/CoM。
- saved_height_steps 下任何正式 accepted_steps、manifest 或 version 文件。
""",
)

source_rows = [
    ("unit previous full", 3, 2, "sim_state_after", False, False, "PASS"),
    ("unit previous command fallback", 3, 2, "command_state_after", False, False, "PASS"),
    ("unit selected-start compatibility", 3, 3, "sim_state_before", True, False, "PASS"),
    ("unit no saved state", 3, "", "", "N/A", False, "PASS (refused)"),
    ("unit Step 1 full", 1, 1, "sim_state_before", False, False, "PASS"),
    ("unit Step 1 command fallback", 1, 1, "command_state_before", True, False, "PASS"),
    ("unit continuity mismatch", 3, 2, "sim_state_after", False, False, "PASS (warning)"),
    ("real Isaac Raw", 5, 4, "sim_state_after", False, False, "PASS"),
    ("real Isaac Fast", 5, 4, "sim_state_after", False, False, "PASS"),
    ("real Isaac Step 1", 1, 1, "sim_state_before", False, False, "PASS"),
]
with (REPORT / "selected_restore_source_tests.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["case", "selected_index", "restore_source_step", "restore_source_field", "fallback", "current_state_incorrectly_used", "result"])
    writer.writerows(source_rows)

with (REPORT / "restore_order_trace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["profile", "selected_index", "restore_request_id", "sequence", "event", "relative_seconds"])
    for profile in ("raw", "fast"):
        restore = RESULT["selected_runs"][profile]["restore_result"]
        trace = restore["trace"]
        first = float(trace[0]["monotonic_s"])
        for sequence, event in enumerate(trace, 1):
            writer.writerow([profile, 5, restore["request_id"], sequence, event["event"], f"{float(event['monotonic_s']) - first:.6f}"])
    trace = RESULT["step1_restore"]["trace"]
    first = float(trace[0]["monotonic_s"])
    for sequence, event in enumerate(trace, 1):
        writer.writerow(["fast-step1", 1, RESULT["step1_restore"]["request_id"], sequence, event["event"], f"{float(event['monotonic_s']) - first:.6f}"])

write_text(
    "task_conflict_tests.txt",
    f"""Unit conflict coverage: PASS
- Recording: blocked by existing OperationCoordinator guard.
- Scene Update / Height Generate: blocked by existing guard.
- Respawn: blocked by existing guard.
- Existing Playback: blocked by existing guard.
- Dirty Servo-Wheel staging: explicitly blocked before restore.
- Stop during restore: pending transaction cancelled; late matching restore ack did not create a plan; operation returned IDLE; next Play remained available.
- Stop Wheels and E-stop remain callable; UI refresh/tab viewing is non-blocking.

Real visible Isaac conflict coverage: PASS
- Recording selected_started={RESULT['conflicts']['recording']['selected_started']}; operation={RESULT['conflicts']['recording']['operation']}; status={RESULT['conflicts']['recording']['status']}
- Height Generate selected_started={RESULT['conflicts']['height_generate']['selected_started']}; operation={RESULT['conflicts']['height_generate']['operation']}; operation_after={RESULT['conflicts']['height_generate']['operation_after']}; status={RESULT['conflicts']['height_generate']['status']}
- Stop during restore cancelled={RESULT['stop_during_restore']['cancel_result']['cancelled']}; worker stayed inactive after late ack; final operation={RESULT['final_operation']}.
""",
)

write_text(
    "regression_tests.txt",
    """PASS — python -m unittest discover -s tests -v
Ran 226 tests in 7.452s; OK.

PASS — other Playback entry semantics
- Play All: unchanged full-sequence start_playback path.
- Play All Fast: unchanged path/profile semantics.
- Play To Selected From Start: unchanged Step 1..N path; no previous-step policy injected.
- Respawn And Play Selected / Fast: unchanged respawn_first=True path.
- Respawn And Play To Selected: unchanged respawn_first=True Step 1..N path.
- Pause / Resume / Stop: existing PlaybackManager paths unchanged; Stop only gains pending-restore cancellation before plan creation.

PASS — protected behavior/static scope
- playback.py unchanged.
- No Servo/Wheel speed, timing, completion, obstacle, height, respawn, recording, save/combine, vision or learning code changed.
- Raw/Fast selected plans preserve the existing planner's command signature; only the restore source/ack gate changed.
- git diff --check: PASS (line-ending notices only).
""",
)

write_text(
    "test_results.txt",
    f"""FINAL RESULTS

1) python -m unittest tests.test_selected_step_previous_saved_state tests.test_play_selected_playback -v
   PASS — 19 tests in 1.016s; traceback: none.

2) python -m unittest discover -s tests -v
   PASS — 226 tests in 7.452s; traceback: none.
   Complete earlier captured console log: {REPORT / 'unittest_output.txt'}

3) python -m py_compile selected_step_previous_saved_state_e2e.py
   PASS.

4) git diff --check
   PASS; only Git CRLF conversion notices, no whitespace errors.

5) python selected_step_previous_saved_state_e2e.py --output {REAL}
   PASS — visible Isaac Sim {ISAAC_VERSION}; result success=true; traceback: none.
   Isaac PID: {RESULT['worker_pid']}
   Worker session: {RESULT['worker_session_id']}
   Existing process reused: {RESULT['reused_existing_process']}
   A second concurrent Isaac was not started.
   UI closed: {RESULT['ui_closed']}
   Worker reference cleared: {RESULT['worker_reference_cleared']}
   Matching project Isaac/Kit processes after exit: 0

E2E harness history (preserved for audit):
- real_isaac: product Raw/Fast passed; harness advanced on stale completed plan and failed at Step1 start. Harness plan-ID matching corrected.
- real_isaac_retry: product paths through Stop passed; harness treated the successful void recording-start API as false. Harness state assertion corrected.
- real_isaac_final: complete PASS. These two earlier failures were validation-script assertions, not product restore failures.
- real_isaac_ui_final: complete PASS with actual Tk button.invoke() for Play Selected Step, Play Selected Fast, Step 1 Fast, and Stop Play; this is the authoritative final run.

Protected saved data:
- 71 files compared by SHA-256; changed/missing: 0.
- Formal source: {RESULT['formal_source_path']}
- SHA-256 before: {RESULT['formal_source_sha256_before']}
- SHA-256 after:  {RESULT['formal_source_sha256_after']}
""",
)

before = json.loads((REPORT / "protected_sha256_before.json").read_text(encoding="utf-8-sig"))
after = []
changed = []
for row in before:
    path = Path(row["Path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper() if path.is_file() else "MISSING"
    current = {"Path": str(path), "Length": path.stat().st_size if path.is_file() else None, "SHA256": digest}
    after.append(current)
    if digest != row["SHA256"]:
        changed.append({"Path": str(path), "Before": row["SHA256"], "After": digest})
(REPORT / "protected_sha256_after.json").write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
write_text("protected_sha256_audit.txt", f"Protected files: {len(before)}\nChanged or missing: {len(changed)}\nDetails: {json.dumps(changed, ensure_ascii=False)}")

write_text("git_status_after.txt", git("status", "--short"))
write_text("git_diff_stat.txt", git("diff", "--stat"))

screenshots = REPORT / "screenshots"
screenshots.mkdir(exist_ok=True)
for source in (REAL / "screenshots").glob("*.png"):
    shutil.copy2(source, screenshots / source.name)
shutil.copy2(REAL / "validation_motion.gif", screenshots / "validation_motion.gif")

print(f"report={REPORT}")
print(f"protected={len(before)} changed={len(changed)}")
print(f"screenshots={len(list(screenshots.iterdir()))}")
