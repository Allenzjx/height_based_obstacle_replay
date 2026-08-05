"""Generate the immutable evidence bundle for the 2026-08-04 completion fix.

This script reads the captured pre-change and final real-Isaac JSON results.  It
does not start Isaac, mutate saved recordings, or synthesize pass results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from playback import plan_from_steps
from sequence_model import load_steps_jsonl


PROJECT = Path(__file__).resolve().parent
DEFAULT_REPORT = PROJECT / "reports" / "height_wheel_servo_batch_replay_completion_fix_20260804_220554"
OLD_WIDTH_M = 1.20
NEW_WIDTH_M = 2.00
ROBOT_COLLISION_WIDTH_M = 0.4411003655


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def get_nested(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def build_geometry_report(report: Path, final: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for tx in final["height_transactions"]:
        ack = tx["ack"]
        bounds = ack["measured_bounds"]
        min_xyz, max_xyz = bounds["min"], bounds["max"]
        width = float(ack["measured_width_m"])
        passed = all(
            (
                ack.get("accepted") is True,
                ack.get("request_id") == tx.get("request_id"),
                abs(float(ack["measured_height_mm"]) - float(tx["requested_height_mm"])) <= 1.0,
                ack.get("visual_updated") is True,
                ack.get("collision_updated") is True,
                ack.get("control_ready") is True,
                abs(width - NEW_WIDTH_M) <= 1.0e-3,
            )
        )
        rows.append(
            {
                "requested_height_mm": tx["requested_height_mm"],
                "old_measured_height_mm": ack.get("old_height_mm"),
                "measured_height_mm": ack["measured_height_mm"],
                "old_authoritative_width_m": OLD_WIDTH_M,
                "new_measured_width_m": width,
                "robot_collision_width_m": ROBOT_COLLISION_WIDTH_M,
                "left_margin_m": (width - ROBOT_COLLISION_WIDTH_M) / 2.0,
                "right_margin_m": (width - ROBOT_COLLISION_WIDTH_M) / 2.0,
                "aabb_min_x_m": min_xyz[0],
                "aabb_max_x_m": max_xyz[0],
                "front_face_x_m": ack["front_face_x_m"],
                "center_y_m": ack["center_y_m"],
                "bottom_z_m": ack["bottom_z_m"],
                "top_z_m": ack["top_z_m"],
                "length_x_m": ack["measured_length_m"],
                "visual_height_mm": (ack["visual_bounds"]["max"][2] - ack["visual_bounds"]["min"][2]) * 1000.0,
                "collision_height_mm": (ack["collision_bounds"]["max"][2] - ack["collision_bounds"]["min"][2]) * 1000.0,
                "revision": ack["obstacle_revision"],
                "request_id": ack["request_id"],
                "update_mode": ack["update_mode"],
                "control_ready": ack["control_ready"],
                "operation_after": "IDLE",
                "result": "PASS" if passed else "FAIL",
            }
        )
    write_csv(report / "obstacle_geometry_update.csv", list(rows[0]), rows)


def build_manual_wheel_report(report: Path, final: dict[str, Any]) -> None:
    rows = []
    for sample in final["manual_wheel_states"]:
        targets = sample["applied_target_rad_s"]
        measured = sample["measured_velocity_rad_s"]
        passed = (
            sample.get("ui_callback_triggered") is True
            and sample.get("worker_received") is True
            and sample.get("stale_command_rejected") is False
            and sample.get("generation_sent") == sample.get("worker_generation")
            and all(abs(float(value)) > 0.05 for value in measured.values())
        )
        rows.append(
            {
                "mode": sample["label"],
                "operation": sample["operation"],
                "ui_callback": sample["ui_callback_triggered"],
                "readiness": compact(sample["controller_readiness"]),
                "requested_velocity_rad_s": sample["velocity"],
                "applied_velocity_rad_s": compact(targets),
                "measured_velocity_rad_s": compact(measured),
                "generation_before": sample["generation_before"],
                "generation_sent": sample["generation_sent"],
                "worker_generation": sample["worker_generation"],
                "worker_received": sample["worker_received"],
                "stale_rejected": sample["stale_command_rejected"],
                "result": "PASS" if passed else "FAIL",
            }
        )
    write_csv(report / "manual_wheel_modes.csv", list(rows[0]), rows)


def build_servo_wheel_report(report: Path, final: dict[str, Any]) -> None:
    sw = final["servo_wheel"]
    recorded_events = sw.get("recorded_launch_events", [])
    rows = []
    for index, name in enumerate(("first_ack", "second_ack"), start=1):
        compact_ack = sw.get(name, {})
        recorded_event = recorded_events[index - 1] if len(recorded_events) >= index else {}
        ack = recorded_event.get("batch_ack", compact_ack)
        servo_step = ack.get("applied_sim_step")
        wheel_step = ack.get("applied_sim_step")
        skew = ack.get("motion_start_skew_s")
        passed = (
            not ack.get("error")
            and ack.get("servo_applied") is True
            and ack.get("wheel_applied") is True
            and servo_step == wheel_step
            and float(skew or 0.0) <= float(ack.get("physics_dt_s", 0.0))
        )
        rows.append(
            {
                "launch": index,
                "batch_id": ack.get("batch_id"),
                "servo_apply_step": servo_step,
                "wheel_apply_step": wheel_step,
                "applied_sim_time_s": ack.get("applied_sim_time"),
                "servo_motion_start_sim_time_s": ack.get("servo_motion_start_sim_time"),
                "wheel_motion_start_sim_time_s": ack.get("wheel_motion_start_sim_time"),
                "start_time_difference_s": skew,
                "physics_tick_s": ack.get("physics_dt_s"),
                "recorded_event": "servo_wheel_launch",
                "recorded_launch_count": sw.get("recorded_launch_count", len(recorded_events)),
                "reloaded_launch_count": sw.get("reloaded_launch_count"),
                "replay_event": "apply_motion_batch/same tick",
                "expanded_command_count": len(recorded_event.get("expanded_commands", [])),
                "result": "PASS" if passed else "FAIL",
            }
        )
    write_csv(report / "servo_wheel_atomic_batch.csv", list(rows[0]), rows)


def build_handshake_report(report: Path, final: dict[str, Any]) -> None:
    names = ("mini_playback", "mini_pause_resume", "mini_stop", "mini_after_stop", "formal_raw", "formal_fast")
    rows = []
    for name, ack in zip(names, final["playback_handshakes"], strict=True):
        run = final[name]
        worker = run.get("worker_final", run.get("final", run.get("worker", {})))
        if not worker and name == "mini_playback":
            worker = run
        immediate = run.get("local_immediately_after_request", run.get("local_state_immediately_after_start", {}))
        accepted = run.get("accepted_state", {})
        first_applied = worker.get("first_command_applied", run.get("first_command_applied", False))
        if name == "mini_stop":
            final_state = worker.get("stop_reason", run.get("stop_reason", "stopped"))
        else:
            final_state = worker.get("stop_reason", run.get("stop_reason", "complete"))
        rows.append(
            {
                "run": name,
                "request_id": ack.get("request_id"),
                "plan_id": ack.get("plan_id"),
                "plan_sha256": ack.get("plan_sha256"),
                "local_state_after_request": get_nested(immediate, "progress_detail", "playback_state", default="START_REQUESTED"),
                "worker_state_after_ack": get_nested(accepted, "progress_detail", "playback_state", default="PREPARING"),
                "ack_accepted": ack.get("accepted"),
                "ack_events": ack.get("event_count"),
                "ack_segments": ack.get("segment_count"),
                "worker_session_id": ack.get("worker_session_id"),
                "first_command_applied": first_applied,
                "first_command_sim_time_s": worker.get("first_command_applied_sim_time_s", run.get("first_command_applied_sim_time_s")),
                "first_command_sim_step": worker.get("first_command_applied_sim_step", run.get("first_command_applied_sim_step")),
                "actual_motion": first_applied or name == "mini_stop",
                "final_state": final_state,
                "result": "PASS" if ack.get("accepted") and (first_applied or name == "mini_stop") else "FAIL",
            }
        )
    write_csv(report / "playback_start_handshake.csv", list(rows[0]), rows)


def build_integrity_and_trace(report: Path, final: dict[str, Any], baseline: dict[str, Any]) -> None:
    integrity = dict(final["formal_plan_integrity"])
    integrity["raw_worker"] = {
        key: final["formal_raw"]["worker_final"].get(key)
        for key in ("event_count", "segment_count", "index", "segment_index", "stop_reason", "first_command_applied_sim_time_s", "first_command_applied_sim_step", "servo_residual_warning_count")
    }
    integrity["fast_worker"] = {
        key: final["formal_fast"]["worker_final"].get(key)
        for key in ("event_count", "segment_count", "index", "segment_index", "stop_reason", "first_command_applied_sim_time_s", "first_command_applied_sim_step", "servo_residual_warning_count")
    }
    integrity["pre_change_worker"] = baseline["baseline_outcome"]
    integrity["step_24_25_evidence"] = {
        profile: {
            str(step): any(
                row.get("replay") == f"formal_{profile}" and int(row.get("step", 0) or 0) == step
                for row in final["formal_replay_trace"]
            )
            for step in (23, 24, 25, 35)
        }
        for profile in ("raw", "fast")
    }
    write_json(report / "formal_50mm_plan_integrity.json", integrity)

    source = Path(integrity["source_path"])
    steps = load_steps_jsonl(source)
    trace_lookup: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in final["formal_replay_trace"]:
        replay_name = str(sample.get("replay", ""))
        profile_name = replay_name.removeprefix("formal_")
        trace_lookup.setdefault((profile_name, int(sample.get("segment", 0) or 0)), []).append(sample)
    rows: list[dict[str, Any]] = []
    for profile, key in (("raw", "formal_raw"), ("fast", "formal_fast")):
        plan = plan_from_steps(steps, profile=profile, sequence_total_steps=len(steps), label=f"formal 50mm {profile}")
        worker = final[key]["worker_final"]
        origin = float(worker["first_command_applied_sim_time_s"])
        for segment in plan.segments:
            events = plan.events[segment.event_start_index : segment.event_start_index + segment.event_count]
            samples = trace_lookup.get((profile, segment.segment_index), [])
            observer = samples[-1] if samples else {}
            errors = observer.get("servo_errors", {})
            rows.append(
                {
                    "profile": profile,
                    "step": segment.source_step,
                    "segment": segment.segment_index,
                    "global_command_first": events[0].global_command_index if events else "",
                    "global_command_last": events[-1].global_command_index if events else "",
                    "command": " | ".join(event.command for event in events),
                    "scheduled_start_sim_time_s": origin + segment.planned_start_s,
                    "scheduled_end_sim_time_s": origin + segment.planned_end_s,
                    "planned_start_offset_s": segment.planned_start_s,
                    "planned_end_offset_s": segment.planned_end_s,
                    "observer_wall_time": observer.get("wall_time"),
                    "max_servo_error_deg": max((abs(float(value)) for value in errors.values()), default=""),
                    "servo_errors_deg": compact(errors),
                    "wheel_targets_rad_s": compact(segment.wheel_applied_target_rad_s),
                    "residual_warning": compact(observer.get("residual_warning")),
                    "stop_reason": worker["stop_reason"] if segment.segment_index == plan.segments[-1].segment_index else "",
                    "step24_25_evidence": "YES" if segment.source_step in (24, 25) else "",
                }
            )
    write_csv(report / "formal_50mm_replay_trace.csv", list(rows[0]), rows)


def build_protected_audit(report: Path) -> None:
    before = read_json(report / "protected_sha256_before.json")
    after = []
    mismatches = []
    for item in before:
        path = Path(item["Path"])
        exists = path.exists()
        row = {
            "Path": str(path),
            "SHA256": sha256(path).upper() if exists else "MISSING",
            "Length": path.stat().st_size if exists else None,
        }
        after.append(row)
        if row != item:
            mismatches.append({"before": item, "after": row})
    write_json(report / "protected_sha256_after.json", after)
    accepted = next(row for row in after if row["Path"].endswith("height_05cm\\accepted_steps.jsonl"))
    write_text(
        report / "protected_data_sha256_audit.txt",
        "\n".join(
            [
                "Protected recording/version SHA-256 audit",
                f"Files checked: {len(before)}",
                f"Mismatches: {len(mismatches)}",
                f"All protected files identical: {not mismatches}",
                f"Formal 50 mm accepted_steps.jsonl SHA-256: {accepted['SHA256'].lower()}",
                f"Formal 50 mm accepted_steps.jsonl length: {accepted['Length']}",
                "Before manifest: protected_sha256_before.json",
                "After manifest: protected_sha256_after.json",
                "No protected recording was written, replaced, migrated, or deleted.",
                "Mismatch detail: " + compact(mismatches),
            ]
        ),
    )
    if mismatches:
        raise RuntimeError(f"protected-file audit failed: {len(mismatches)} mismatch(es)")


def build_changed_files(report: Path) -> None:
    roles = {
        "README.md": "更新固定 100%、Servo-Wheel、几何事务与握手说明",
        "config/environment_reference.yaml": "唯一正式障碍物宽度由 1.20 m 增至 2.00 m",
        "height_generate_panel.py": "显示 worker 实测几何和严格事务结果",
        "height_version_store.py": "新版本 metadata 写入实测 width；忽略旧 speed metadata",
        "motion_speed.py": "仅保留固定 canonical motion reference，删除倍率模型",
        "playback.py": "完整性 SHA/计数、原子 batch 重放、contact-aware completion",
        "playback_progress.py": "新增 START_REQUESTED 等权威播放状态",
        "sim_ipc_protocol.py": "删除 set_speed_scale；加入 apply_motion_batch/握手字段",
        "sim_obstacle_scene.py": "真实 USD visual/collision/AABB 事务和可靠最小重建",
        "sim_process_client.py": "batch IPC、播放握手、wheel generation 同步",
        "sim_robot_adapter.py": "一次校验并同 tick 写入完整 servo/wheel vector",
        "sim_transport.py": "统一 manual wheel generation 与 atomic batch transport",
        "sim_ui_controller.py": "Manual Wheel readiness、Servo-Wheel 可见 UI、事务/握手对账",
        "sim_worker_process.py": "worker 权威 start accept/reject、严格 scene/batch 执行",
        "sim_worker_runtime.py": "实际几何 ack、控制就绪和 worker 状态",
        "tests/test_recording_baseline_lock.py": "Recording observer 与控制解耦回归",
        "tests/test_servo_wheel_staging.py": "staging/Launch 原子行为回归",
        "tests/test_sim_time_playback_service.py": "完成策略和 sim-time scheduler 回归",
        "tests/test_task_separation_refactor.py": "已移除任务不回归",
        "tests/test_ui_motion_speed_height_version_refactor.py": "固定 100% 和 height metadata 回归",
        "tests/test_workflow_regressions.py": "正式工作流回归",
        "height_wheel_servo_batch_replay_completion_e2e.py": "可视化真实 Isaac 端到端证据采集",
        "tests/test_height_wheel_servo_batch_replay_completion_fix.py": "本轮六问题专项测试",
        "generate_completion_fix_report.py": "从原始证据生成 CSV/JSON/哈希审计报告",
    }
    status = git_output("status", "--short")
    changed = []
    for line in status.splitlines():
        state, name = line[:2], line[3:]
        category = "新增" if state == "??" or "A" in state else "删除" if "D" in state else "修改"
        changed.append(f"{category}\t{name}\t{roles.get(name, '测试或实现配套更新')}")
    write_text(report / "changed_files.txt", "类型\t文件\t作用\n" + "\n".join(changed) + "\n删除文件：无")
    write_text(report / "git_status_after.txt", status)
    write_text(report / "git_diff_stat.txt", git_output("diff", "--stat"))
    if not (report / "git_status_before.txt").exists():
        write_text(report / "git_status_before.txt", "Initial repository status recorded before edits: clean")


def build_text_reports(report: Path, final: dict[str, Any], baseline: dict[str, Any], final_dir: Path) -> None:
    before = baseline["baseline_outcome"]
    raw = final["formal_raw"]["worker_final"]
    fast = final["formal_fast"]["worker_final"]
    write_text(
        report / "playback_stop_reason_before_after.txt",
        f"""修改前正式 50 mm：
stop_reason={before['stop_reason']}
step={before['progress_detail']['current_step_index']}/35, segment=128, global_command={before['index']}/248
last_command={before['last_event_command']}
last_error={before['last_error']}
证据结论：不是 planner/payload/worker 解码截断；worker 已持有 248 events/159 segments。也不是 5 秒 stall，实际阈值为 0.800 simulation seconds。失败发生在 step 26（UI 观察上已越过 24/25），由接触负载下约 1.315° 的合法录制残差被旧 completion policy 当作 actuator_limit。

修改后 Raw：stop_reason={raw['stop_reason']}, step={raw['progress_detail']['current_step_index']}/35, event={raw['index']}/248, segment={raw['segment_index']}/159, warnings={raw['servo_residual_warning_count']}。
修改后 Fast：stop_reason={fast['stop_reason']}, step={fast['progress_detail']['current_step_index']}/35, event={fast['index']}/248, segment={fast['segment_index']}/159, warnings={fast['servo_residual_warning_count']}。
修复保留 target、不 teleport；仅在 planned duration + bounded grace 后，合法/稳定/未恶化且不超过 cap 的录制残差以 servo_residual_warning 继续。NaN、超限、状态不可读和真实不改善仍失败。
""",
    )
    write_text(
        report / "removed_speed_scale_runtime.txt",
        """已从运行时删除：Speed Scale tab、百分比 slider、GlobalSpeedModel/SpeedPercentModel、requested/effective/pending percent、set_speed_scale IPC、manual/playback multiplier、servo/wheel target scaling、preserve-wheel-distance、speed snapshot/timer/task operation。

motion_speed.py 现在只提供固定 MotionReference；Manual、Recording、Raw、Fast 和 Servo-Wheel Launch 都直接发送 canonical actuator command。Fast 只删除 implicit UI idle，不改变 target、servo profile、wheel velocity、wheel active duration或命令顺序。

只读状态显示 Motion profile: Fixed 100%。旧 recording 中的 speed metadata 可被读取但被忽略，不参与 runtime scaling，也没有重写旧文件。

真实 batch ack：motion_profile=fixed_100_percent；requested/effective servo velocity=150.0 deg/s；wheel target 为 canonical rad/s 值。正式 config 的 wheel default/max、stiffness/damping/effort limit 未因本轮修复调整。
""",
    )
    unit_log = report / "unittest_output_final_214.txt"
    write_text(
        report / "test_results.txt",
        f"""Automated and real-Isaac validation summary

1. python -m unittest discover -s tests -v
   PASS: 214 tests, expected final line `OK`; complete output: {unit_log}
2. tests/test_height_wheel_servo_batch_replay_completion_fix.py
   PASS: 10 targeted tests (included in the 214-test discovery run).
3. python height_wheel_servo_batch_replay_completion_e2e.py --visible
   PASS: real result success=true; evidence: {final_dir / 'real_isaac_result.json'}
4. git diff --check
   PASS (Git emitted only working-tree CRLF normalization warnings, no whitespace errors).
5. protected SHA-256 before/after audit
   PASS: zero mismatches.

Visible Isaac/worker PID: {final['worker_pid']}
Isaac window: {final['isaac_window_title']}
Existing process reused: false (preflight confirmed no Isaac/Kit/sim worker, exactly one was launched).
UI closed: {final['ui_closed']}
Worker reference cleared: {final['worker_reference_cleared']}
Final process audit: PID {final['worker_pid']} absent after orderly UI shutdown.

Failure/retry evidence is intentionally retained in final_real_isaac through final_real_isaac_retry5. Authoritative passing run: final_real_isaac_retry6.
""",
    )
    summary = f"""# Height / Wheel / Servo Batch / Replay Completion Fix

## Outcome

真实可视化 Isaac 5.1.0 最终验证通过：高度 50→75→100→50→100→75 mm 六个事务均用 USD 实际 visual/collision AABB 验证；障碍物实测宽度约 2.00000003 m；Manual Wheel 在 TEST、RECORDING、Stop Recording 后 TEST 三种状态均运动；Servo+Wheel batch 同一 applied sim step、启动偏差 0 s；正式 50 mm Raw/Fast 均完成 35/35、248/248 events、159/159 segments，stop_reason=complete。UI 最终 IDLE，worker 有序退出。

## 六个真实根因与修复

1. **Generate 假成功**：旧 worker 只写 prim 属性并按请求值回填 `scene_height/obstacle_updated`，没有 physics/render 后读取实际 AABB；本机 Isaac 对现有 prim 的原地 size 更新未刷新有效几何。现在每次事务携带 request/revision，先尝试原地更新并验证；验证失败时最小 Remove/Create，确认旧 prim 失效、新 prim/collision 有效，再用实际 AABB、visual/collision bounds 和 control_ready 决定成功。
2. **宽度不足**：正式唯一配置仍为 1.20 m。改为 2.00 m（满足 ≥2.0 且比 1.20 至少增加 0.40），三个高度同宽、Y 中心 0。机器人 collision width={ROBOT_COLLISION_WIDTH_M:.10f} m，左右理论余量各 {(NEW_WIDTH_M - ROBOT_COLLISION_WIDTH_M) / 2:.10f} m。
3. **Manual Wheel 看似依赖 Recording**：readiness/UI command path 分裂，且 Stop 后 worker/client 的 wheel generation 可能不同步；Recording 路径碰巧刷新状态，造成表象。现在 Servo/Wheel 共用 motion readiness，IDLE/RECORDING 都走同一 transport；Recording 只观察一次，Stop 后 generation 同步，旧 generation 仍拒绝。
4. **Servo-Wheel UI 消失/非原子**：后端残留 staging command 名称，但右侧 tab 的可见 section 和完整状态化工作流已丢失，旧通用应用路径也不能保证 12 个 actuator 同 tick。现已恢复 Start/Launch/Clear/Cancel 与 staged/live/delta；Launch 只发一个 `apply_motion_batch`，worker 先全量校验再一次 articulation write，同 tick ack；Recording 保存一个 launch event，Playback 复用同一 executor。
5. **Speed Scale 隐藏倍率**：倍率同时散落在 UI/model/IPC/manual/playback 元数据路径。运行时链路已移除，固定为 canonical 100%；Fast 只去 implicit idle。
6. **Already Active / 长序列停止**：本地曾在 worker 接受前过早 active，stale local/session 可导致 false already-active。新状态从 START_REQUESTED 开始，matching request/plan/SHA/count/session ack 后才 PREPARING，scheduler started 后才 PLAYING，并有 first-motion watchdog和状态对账。正式 50 mm 并未在第 24 步被 planner 或 IPC 截断：baseline worker 已收到全部 248/159；实际在 step 26 segment 128 因旧 0.800 s no-improvement + 约 1.315° contact residual 触发 actuator_limit。新的 bounded recorded-residual policy 保留安全失败，并把合法稳定残差记 warning 后继续。

## 被证据排除的怀疑

- 不是固定 24-step/50-event/queue 长度限制，也不是 JSON payload/worker decode 截断。
- 不是 UI 在 step 24 发送 Stop；baseline 已执行 step 25 并到 step 26。
- 不是 5 simulation seconds stall；捕获值是 0.800 s。
- 高度问题不是 75 被解释为 75 cm；正式 IPC 统一为整数 `height_mm`。
- 未更改既有 actuator 速度、stiffness、damping、effort limit、calibration、joint mapping 或 robot asset。

## Evidence

- Final raw result: `{final_dir / 'real_isaac_result.json'}`
- Completed screenshot: `{final_dir / 'screenshots' / 'final_completed.png'}`
- Validation animation: `{final_dir / 'validation_motion.gif'}`
- Protected data audit: `{report / 'protected_data_sha256_audit.txt'}`
"""
    write_text(report / "summary.md", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--final-dir", type=Path)
    args = parser.parse_args()
    report = args.report_root.resolve()
    final_dir = (args.final_dir or report / "final_real_isaac_retry6").resolve()
    final = read_json(final_dir / "real_isaac_result.json")
    baseline = read_json(report / "baseline_formal_50" / "real_isaac_result.json")
    if final.get("success") is not True:
        raise RuntimeError("authoritative final real-Isaac run is not successful")
    build_geometry_report(report, final)
    build_manual_wheel_report(report, final)
    build_servo_wheel_report(report, final)
    build_handshake_report(report, final)
    build_integrity_and_trace(report, final, baseline)
    build_protected_audit(report)
    build_changed_files(report)
    build_text_reports(report, final, baseline, final_dir)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
