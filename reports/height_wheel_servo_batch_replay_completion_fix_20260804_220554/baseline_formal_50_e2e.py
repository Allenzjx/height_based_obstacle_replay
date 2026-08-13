"""Read-only pre-change reproduction of the formal legacy 50 mm replay."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from playback import plan_from_steps, playback_plan_from_payload, playback_plan_to_payload
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


FORMAL_PATH = PROJECT_ROOT / "saved_height_steps" / "height_05cm" / "accepted_steps.jsonl"


class BaselineFormal50E2E(RefactorGuiE2E):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir, timeout_s=700.0)
        self.last_trace_key: tuple[int, int, int] | None = None
        self.result["baseline_only"] = True
        self.result["formal_replay_trace"] = []

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        rows = [json.loads(line) for line in FORMAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.controller.current_height_mm = 50
        self.controller.current_version_id = "legacy_5cm_readonly"
        self.controller.current_version_metadata = {
            "legacy": True,
            "read_only": True,
            "accepted_steps_path": str(FORMAL_PATH.resolve()),
        }
        self.controller.manager.adopt_steps(rows, dirty=False)
        plan = plan_from_steps(
            rows,
            profile="raw",
            max_wheel_speed=self.controller.max_wheel_speed,
            label="formal 50mm pre-change baseline",
            sequence_total_steps=len(rows),
        )
        payload = playback_plan_to_payload(plan)
        decoded = playback_plan_from_payload(payload)
        represented = sorted({int(event.source_step) for event in plan.events if event.source_step})
        self.result["formal_plan_integrity"] = {
            "source_path": str(FORMAL_PATH.resolve()),
            "sha256": hashlib.sha256(FORMAL_PATH.read_bytes()).hexdigest(),
            "input_steps": len(rows),
            "input_events": sum(len(row.get("events", [])) for row in rows),
            "plan_events": len(plan.events),
            "plan_segments": len(plan.segments),
            "payload_events": len(payload.get("events", [])),
            "payload_segments": len(payload.get("segments", [])),
            "worker_decoded_events": len(decoded.events),
            "worker_decoded_segments": len(decoded.segments),
            "represented_steps": represented,
            "missing_steps": [index for index in range(1, len(rows) + 1) if index not in represented],
            "plan_sha256": plan.plan_sha256,
        }
        self.result["isaac_pid"] = int(self.controller.latest_sim_status.get("worker_pid", 0) or 0)
        self._capture_ui("formal_50_before_play", "formal_50_before_play.png")
        ok = self.controller.start_playback(rows, label="formal 50mm pre-change baseline", profile="raw")
        self.result["start_returned"] = bool(ok)
        self.result["local_state_immediately_after_start"] = self.controller.playback.status_dict()
        if not ok:
            raise AssertionError(self.controller.playback.last_error or "formal playback start returned false")
        self._advance("FORMAL_WAIT")

    def _stage_formal_wait(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        progress = dict(worker.get("progress_detail", {}) or {})
        key = (
            int(progress.get("current_step_index", 0) or 0),
            int(worker.get("segment_index", 0) or 0),
            int(progress.get("global_command_index", 0) or 0),
        )
        if key != self.last_trace_key:
            self.last_trace_key = key
            self.stage_started = time.monotonic()
            self.result["formal_replay_trace"].append(
                {
                    "wall_time": time.time(),
                    "step": key[0],
                    "segment": key[1],
                    "global_command": key[2],
                    "active": bool(worker.get("active", False)),
                    "started": bool(worker.get("started", False)),
                    "stop_reason": str(worker.get("stop_reason", "") or ""),
                    "last_error": str(worker.get("last_error", "") or ""),
                    "servo_errors": dict(worker.get("current_servo_errors", {}) or {}),
                }
            )
            if key[0] == 24 and "step_24" not in self.result["screenshots"]:
                self._capture_ui("step_24", "step_24.png")
        if bool(worker.get("active", False)):
            return
        reason = str(worker.get("stop_reason", "") or "")
        if not reason:
            return
        self.result["baseline_outcome"] = worker
        self.result["success"] = True
        self.controller.stop_wheels(reason="baseline_complete")
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(350, lambda: self.ui._window_close(force=True))


if __name__ == "__main__":
    raise SystemExit(BaselineFormal50E2E(Path(__file__).resolve().parent / "baseline_formal_50").run())
