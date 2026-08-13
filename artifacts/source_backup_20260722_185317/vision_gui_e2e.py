"""Automated visible-GUI Vision workflow for regression smoke testing.

The runner intentionally drives the same controller methods as the Tk buttons.
It records snapshots and block reasons, but does not write controller readiness
or validation fields directly.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable


class VisionGuiE2ERunner:
    def __init__(self, ui: Any, controller: Any, args: Any):
        self.ui = ui
        self.root = ui.root
        self.controller = controller
        self.args = args
        self.height_cm = int(getattr(args, "e2e_height_cm", 5))
        self.timeout_s = max(1.0, float(getattr(args, "e2e_timeout_s", 300.0)))
        self.playback_probe_s = max(0.0, float(getattr(args, "e2e_playback_probe_s", 0.0)))
        self.keep_open_on_failure = bool(getattr(args, "e2e_keep_open_on_failure", False))
        self.open_camera_viewport = bool(getattr(args, "e2e_open_camera_viewport", False))
        self.test_camera_fallback = bool(getattr(args, "e2e_test_camera_fallback", False))
        self.save_screenshots = bool(getattr(args, "e2e_save_screenshots", False))
        self.capture_startup_trace = bool(getattr(args, "e2e_capture_startup_trace", False))
        self.camera_counterfactual = bool(getattr(args, "e2e_camera_counterfactual", False))
        self.camera_pose_ab = bool(getattr(args, "e2e_camera_pose_ab", False))
        self.ground_calibration = bool(getattr(args, "e2e_ground_calibration", False))
        self.started = 0.0
        self.stage_started = 0.0
        self.stage_name = ""
        self.stages: list[dict[str, Any]] = []
        self.exit_code = 1
        self.report_path = self._resolve_output_path()
        self.error = ""
        self._playback_started_wall = 0.0
        self._stopping = False

    def start(self) -> None:
        self.started = time.monotonic()
        self._advance("CAPTURE_RAW_STARTUP" if self.capture_startup_trace else "WAIT_RUNTIME")

    def _advance(self, stage: str) -> None:
        self.stage_name = stage
        self.stage_started = time.monotonic()
        self.stages.append(
            {
                "name": stage,
                "started_at": time.time(),
                "completed_at": None,
                "elapsed_s": 0.0,
                "snapshot_before": self._snapshot(),
                "snapshot_after": {},
                "result": "running",
                "error": "",
                "block_reason": "",
            }
        )
        self._schedule(0, self._tick)

    def _tick(self) -> None:
        if self._stopping:
            return
        if time.monotonic() - self.started > self.timeout_s:
            self._finish_current("timeout", error=f"E2E timeout after {self.timeout_s:.1f}s", block_reason=self._block_reason())
            self._save_and_shutdown(exit_code=1)
            return
        try:
            handler = getattr(self, f"_stage_{self.stage_name.lower()}")
            handler()
        except Exception as exc:
            self.error = str(exc)
            self._finish_current("error", error=f"{exc}\n{traceback.format_exc()}", block_reason=self._block_reason())
            self._save_and_shutdown(exit_code=1)

    def _stage_wait_runtime(self) -> None:
        if bool(getattr(self.controller, "runtime_ready", False)):
            self._finish_current("ok")
            self._advance("CAPTURE_FIRST_VISIBLE_POSE" if self.capture_startup_trace else "START_VISION_TASK")
            return
        self._reschedule()

    def _stage_capture_raw_startup(self) -> None:
        self._finish_current("ok")
        self._advance("WAIT_FIRST_VISIBLE_RENDER")

    def _stage_wait_first_visible_render(self) -> None:
        snap = self._snapshot()
        status = _dig(snap, "sim", "worker_status")
        history = status.get("startup_phase_history", []) if isinstance(status, dict) else []
        phases = [str(row.get("phase", "")) for row in history if isinstance(row, dict)]
        if "first_visible_render_completed" in phases or bool(getattr(self.controller, "runtime_ready", False)):
            self._finish_current("ok")
            self._advance("WAIT_RUNTIME")
            return
        self._reschedule()

    def _stage_capture_first_visible_pose(self) -> None:
        self._finish_current("ok")
        self._advance("START_VISION_TASK")

    def _stage_start_vision_task(self) -> None:
        ok = bool(self.controller.start_vision_task("generated"))
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("GENERATE_5CM" if ok else "SAVE_REPORT")

    def _stage_generate_5cm(self) -> None:
        ok = bool(self.controller.generate_vision_test_obstacle(self.height_cm))
        if ok:
            self.controller.set_vision_enabled(True)
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("WAIT_SCENE_CONFIRMATION" if ok else "SAVE_REPORT")

    def _stage_wait_scene_confirmation(self) -> None:
        snap = self._snapshot()
        if bool(snap.get("vision_scene_ready", False)) or bool(_dig(snap, "vision", "scene_ready")):
            self._finish_current("ok")
            self._advance("WAIT_STABLE_DETECTION")
            return
        self._reschedule()

    def _stage_wait_stable_detection(self) -> None:
        snap = self._snapshot()
        vision = snap.get("vision", {}) if isinstance(snap.get("vision"), dict) else {}
        stable = bool(vision.get("stable", False))
        detected = vision.get("detected_height_cm")
        confidence = float(vision.get("confidence", 0.0) or 0.0)
        threshold = float(getattr(self.args, "vision_confidence_threshold", 0.75))
        if stable and detected == self.height_cm and confidence >= threshold:
            self._finish_current("ok")
            self._advance("VALIDATE_HEIGHT")
            return
        self._reschedule()

    def _stage_validate_height(self) -> None:
        ok = bool(self.controller.validate_current_generated_height())
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("WAIT_VALIDATION" if ok else "SAVE_REPORT")

    def _stage_wait_validation(self) -> None:
        validation = _dig(self._snapshot(), "vision", "height_validation")
        if isinstance(validation, dict) and bool(validation.get("checked", False)):
            passed = bool(validation.get("passed", False))
            self._finish_current("ok" if passed else "blocked", block_reason="" if passed else self._block_reason())
            self._advance("WAIT_STEPS" if passed else "SAVE_REPORT")
            return
        self._reschedule()

    def _stage_wait_steps(self) -> None:
        snap = self._snapshot()
        vision = snap.get("vision", {}) if isinstance(snap.get("vision"), dict) else {}
        steps_count = int(snap.get("vision_steps_count", vision.get("steps_count", 0)) or 0)
        steps_ready = bool(snap.get("vision_steps_ready", vision.get("steps_ready", False)))
        if steps_ready and steps_count > 0:
            self._finish_current("ok")
            self._advance("VALIDATE_GROUND")
            return
        self._reschedule()

    def _stage_validate_ground(self) -> None:
        ok = bool(self.controller.validate_robot_ground_contact())
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("CALIBRATE_GROUND" if ok else "SAVE_REPORT")

    def _stage_calibrate_ground(self) -> None:
        ok = bool(self.controller.calibrate_ground_reference())
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("WAIT_GROUND_RESULT" if ok else "SAVE_REPORT")

    def _stage_wait_ground_result(self) -> None:
        snap = self._snapshot()
        ground = _robot_ground_from_snapshot(snap)
        calibration = _dig(snap, "sim", "worker_status", "vision", "ground_calibration")
        if bool(ground.get("checked", False)) or isinstance(calibration, dict):
            self._finish_current("ok")
            if self.open_camera_viewport:
                self._advance("OPEN_SECONDARY_CAMERA_VIEW")
            elif self.test_camera_fallback:
                self._advance("TEST_MAIN_VIEW_FALLBACK")
            else:
                self._advance("PLAYBACK_PROBE")
            return
        self._reschedule()

    def _stage_open_secondary_camera_view(self) -> None:
        ok = bool(self.controller.open_onboard_camera_viewport())
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("WAIT_CAMERA_VIEW_ACK" if ok else ("TEST_MAIN_VIEW_FALLBACK" if self.test_camera_fallback else "PLAYBACK_PROBE"))

    def _stage_wait_camera_view_ack(self) -> None:
        camera_view = _dig(self._snapshot(), "vision", "camera_view")
        target_revision = int(getattr(self.controller, "camera_view_pending_revision", 0) or 0)
        completed_revision = int(camera_view.get("completed_revision", 0) or 0) if isinstance(camera_view, dict) else 0
        if target_revision > 0 and completed_revision < target_revision:
            self._reschedule()
            return
        if isinstance(camera_view, dict) and not bool(camera_view.get("pending", False)):
            error = str(camera_view.get("error", "") or "")
            active = bool(camera_view.get("active", False))
            path_verified = bool(camera_view.get("camera_path_verified", False))
            window_visible = bool(camera_view.get("window_visible", camera_view.get("visible", False)))
            ok = not error and active and path_verified and window_visible
            self._finish_current("ok" if ok else "blocked", error=error, block_reason="" if ok else self._block_reason())
            if self.test_camera_fallback:
                self._advance("TEST_MAIN_VIEW_FALLBACK")
            elif self.camera_counterfactual:
                self._advance("RUN_CAMERA_COUNTERFACTUAL")
            else:
                self._advance("PLAYBACK_PROBE")
            return
        self._reschedule()

    def _stage_test_main_view_fallback(self) -> None:
        ok = bool(self.controller.show_camera_in_main_view_fallback())
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("RETURN_TO_PERSPECTIVE")

    def _stage_return_to_perspective(self) -> None:
        ok = bool(self.controller.return_main_view_to_perspective())
        self._finish_current("ok" if ok else "blocked", block_reason="" if ok else self._block_reason())
        self._advance("RUN_CAMERA_COUNTERFACTUAL" if self.camera_counterfactual else "PLAYBACK_PROBE")

    def _stage_run_camera_counterfactual(self) -> None:
        self._finish_current(
            "skipped",
            error="GUI runner does not mutate Isaac geometry directly; use --worker-smoke-camera-counterfactual for the real counterfactual worker smoke.",
        )
        self._advance("PLAYBACK_PROBE")

    def _stage_playback_probe(self) -> None:
        can_play, reason = self.controller.playback_readiness(respawn_first=bool(getattr(self.controller, "vision_respawn_before_replay", True)))
        if not bool(can_play):
            self._finish_current("blocked", block_reason=str(reason or self._block_reason()))
            self._advance("SAVE_REPORT")
            return
        snap = self._snapshot()
        detection_revision = int(_dig(snap, "vision", "detection_revision") or 0)
        ok = bool(self.controller.replay_validated_vision_steps(self.height_cm, detection_revision, require_auto_replay=False))
        if not ok:
            self._finish_current("blocked", block_reason=self._block_reason())
            self._advance("SAVE_REPORT")
            return
        self._playback_started_wall = time.monotonic()
        self._finish_current("ok")
        self._advance("STOP_PLAYBACK")

    def _stage_stop_playback(self) -> None:
        if time.monotonic() - self._playback_started_wall < self.playback_probe_s:
            self._reschedule(delay_ms=50)
            return
        try:
            self.controller.playback.stop("E2E playback probe complete")
        except TypeError:
            self.controller.playback.stop()
        except Exception:
            pass
        try:
            self.controller.stop_wheels()
        except Exception:
            pass
        self._finish_current("ok")
        self._advance("SAVE_REPORT")

    def _stage_save_report(self) -> None:
        self._finish_current("ok")
        failures = [row for row in self.stages if row.get("result") in {"blocked", "error", "timeout"}]
        self._save_report()
        self._advance("SHUTDOWN")
        self.exit_code = 0 if not failures else 1

    def _stage_shutdown(self) -> None:
        self._finish_current("ok")
        self._save_and_shutdown(exit_code=self.exit_code)

    def _finish_current(self, result: str, *, error: str = "", block_reason: str = "") -> None:
        if not self.stages:
            return
        row = self.stages[-1]
        if row.get("result") != "running":
            return
        row["completed_at"] = time.time()
        row["elapsed_s"] = max(0.0, time.monotonic() - self.stage_started)
        row["snapshot_after"] = self._snapshot()
        row["result"] = str(result)
        row["error"] = str(error or "")
        row["block_reason"] = str(block_reason or "")

    def _save_and_shutdown(self, *, exit_code: int) -> None:
        self._stopping = True
        self.exit_code = int(exit_code)
        self._save_report()
        if self.exit_code != 0 and self.keep_open_on_failure:
            return
        try:
            self.ui._window_close()
        except Exception:
            try:
                self.controller.shutdown()
            finally:
                self.root.destroy()

    def _save_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "mode": "e2e_vision_gui_smoke",
            "ok": not any(row.get("result") in {"blocked", "error", "timeout"} for row in self.stages),
            "exit_code": int(self.exit_code),
            "height_cm": int(self.height_cm),
            "started_at": time.time() - max(0.0, time.monotonic() - self.started) if self.started else time.time(),
            "elapsed_s": max(0.0, time.monotonic() - self.started) if self.started else 0.0,
            "args": vars(self.args) if hasattr(self.args, "__dict__") else {},
            "stage_count": len(self.stages),
            "stages": self.stages,
            "final_snapshot": self._snapshot(),
            "stdout_stderr_paths": self._worker_log_paths(),
            "startup_pose_trace": _dig(self._snapshot(), "sim", "worker_status", "startup_pose_trace") or [],
            "screenshot_path": "",
            "screenshot_note": "Tk/Isaac screenshot capture is not implemented in this runner." if self.save_screenshots else "",
        }
        self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")

    def _snapshot(self) -> dict[str, Any]:
        try:
            snap = self.controller.snapshot()
            if isinstance(snap, dict):
                return _json_safe(snap)
        except Exception as exc:
            return {"snapshot_error": str(exc)}
        return {}

    def _block_reason(self) -> str:
        snap = self._snapshot()
        reasons: list[str] = []
        for key in ("playback_block_reason", "motion_block_reason", "respawn_block_reason"):
            value = snap.get(key)
            if value:
                reasons.append(str(value))
        sim = snap.get("sim", {}) if isinstance(snap.get("sim"), dict) else {}
        for key in ("playback_block_reason", "motion_block_reason", "respawn_block_reason"):
            value = sim.get(key)
            if value:
                reasons.append(str(value))
        ground = _robot_ground_from_snapshot(snap)
        for reason in ground.get("reasons", []) if isinstance(ground.get("reasons", []), list) else []:
            reasons.append(str(reason))
        vision = snap.get("vision", {}) if isinstance(snap.get("vision"), dict) else {}
        if vision.get("failure_reason"):
            reasons.append(str(vision.get("failure_reason")))
        return "; ".join(dict.fromkeys(item for item in reasons if item))

    def _worker_log_paths(self) -> dict[str, str]:
        client = getattr(self.controller, "sim_client", None)
        if client is None:
            return {}
        result: dict[str, str] = {}
        for attr, key in (("stdout_path", "stdout"), ("stderr_path", "stderr")):
            value = getattr(client, attr, "")
            if value:
                result[key] = str(value)
        return result

    def _resolve_output_path(self) -> Path:
        value = str(getattr(self.args, "e2e_output", "") or "").strip()
        if value:
            return Path(value)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return Path(__file__).resolve().parent / "saved_height_steps" / "vision_debug" / f"e2e_vision_gui_{stamp}.json"

    def _reschedule(self, *, delay_ms: int = 250) -> None:
        self._schedule(delay_ms, self._tick)

    def _schedule(self, delay_ms: int, callback: Callable[[], None]) -> None:
        self.root.after(max(0, int(delay_ms)), callback)


def _dig(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _robot_ground_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    direct = snapshot.get("robot_ground", {})
    if isinstance(direct, dict) and direct:
        return direct
    nested = _dig(snapshot, "sim", "worker_status", "robot_ground")
    if isinstance(nested, dict):
        return nested
    return {}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
