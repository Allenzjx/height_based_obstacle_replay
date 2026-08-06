"""Single-worker visible audit/replay E2E for the active formal 50 mm v2 version."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from height_version_store import DEFAULT_VERSION_ROOT, HeightVersionStore
from operation_coordinator import OperationState
from playback import PlaybackPlan, plan_from_steps
from sequence_model import load_steps_jsonl
from sim_state_validation import validate_full_sim_pose_state
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


class Latest50mmManualV2E2E(RefactorGuiE2E):
    def __init__(
        self,
        output_dir: Path,
        *,
        profiles: list[str],
        require_complete: bool,
        timeout_s: float,
    ) -> None:
        super().__init__(output_dir, timeout_s=timeout_s)
        self.profiles = ["fast" if value == "fast" else "raw" for value in profiles]
        self.require_complete = bool(require_complete)
        self.profile_index = 0
        self.steps: list[dict[str, Any]] = []
        self.plans: dict[str, PlaybackPlan] = {}
        self.active_run: dict[str, Any] = {}
        self.last_segment_index = -1
        self.last_detail_request = 0.0
        self.respawn_requested_at = 0.0
        self.result = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "single_worker_requested": True,
            "formal_version_read_only": True,
            "require_complete": self.require_complete,
            "profiles": list(self.profiles),
            "stages": [],
            "screenshots": {},
            "runs": [],
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }

    @staticmethod
    def _plan_summary(plan: PlaybackPlan) -> dict[str, Any]:
        return {
            "profile": plan.profile,
            "event_count": len(plan.events),
            "segment_count": len(plan.segments),
            "final_time_s": plan.final_time_s,
            "plan_sha256": plan.plan_sha256,
            "represented_step_indices": list(
                dict(plan.timing.get("plan_integrity", {}) or {}).get("represented_step_indices", []) or []
            ),
        }

    @staticmethod
    def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
        actual = dict(state.get("actual_joint_state", {}) or {})
        servos = dict(actual.get("servos", {}) or {})
        return {
            "root_pose": copy.deepcopy(state.get("root_pose")),
            "joint_pos": copy.deepcopy(state.get("joint_pos")),
            "joint_vel": copy.deepcopy(state.get("joint_vel")),
            "joint_names": list(state.get("joint_names", []) or []),
            "measured_servos_deg": {
                str(name): dict(row or {}).get("deg") for name, row in servos.items()
            },
            "robot_ground_diagnostics": copy.deepcopy(state.get("robot_ground_diagnostics", {})),
            "sim_time": state.get("sim_time"),
            "adapter_sim_step": state.get("adapter_sim_steps"),
        }

    def _tick(self) -> None:
        if self.ui._closing:
            return
        try:
            if time.monotonic() - self.started > self.timeout_s:
                raise TimeoutError(f"overall timeout at {self.stage}")
            phase = str(self.controller.latest_sim_status.get("phase", "") or "")
            if phase in {"preflight_failed", "launch_plan_failed", "process_spawn_failed", "runtime_failed"}:
                raise RuntimeError(str(self.controller.latest_sim_status.get("error", phase)))
            self._sample_worker()
            getattr(self, f"_stage_{self.stage.lower()}")()
        except Exception as exc:
            self._fail(exc)
        if not self.ui._closing:
            self.ui.root.after(50, self._tick)

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        store = HeightVersionStore(DEFAULT_VERSION_ROOT)
        active_id = store.active_version_id(50)
        if not active_id:
            raise AssertionError("No active formal 50 mm version")
        version_path = store.version_dir(50, active_id)
        steps_path = store.version_steps_path(50, active_id)
        metadata_path = store.version_metadata_path(50, active_id)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.steps = load_steps_jsonl(steps_path)
        saved_state_audit: list[dict[str, Any]] = []
        saved_state_counts: dict[str, int] = {}
        for step in self.steps:
            for field in ("sim_state_before", "sim_state_after"):
                validation = validate_full_sim_pose_state(step.get(field))
                classification = str(validation.get("classification", "INVALID"))
                saved_state_counts[classification] = saved_state_counts.get(classification, 0) + 1
                saved_state_audit.append(
                    {
                        "step_index": int(step.get("index", 0) or 0),
                        "field": field,
                        "classification": classification,
                        "valid": bool(validation.get("valid", False)),
                        "reason": str(validation.get("reason", "") or ""),
                    }
                )
        actual_sha = hashlib.sha256(steps_path.read_bytes()).hexdigest()
        if actual_sha != str(metadata.get("accepted_steps_sha256", "")):
            raise AssertionError("active 50 mm accepted_steps SHA mismatch")
        self.controller.current_height_mm = 50
        self.controller.current_version_id = active_id
        self.controller.current_version_metadata = {
            **metadata,
            "read_only": True,
            "path": str(version_path.resolve()),
            "accepted_steps_path": str(steps_path.resolve()),
        }
        self.controller.manager.adopt_steps(self.steps, dirty=False)
        self.controller.selected_step_index = 1
        self.ui.open_right_tab("Playback")
        self.ui._refresh(force=True)
        self.plans = {
            profile: plan_from_steps(
                self.steps,
                profile=profile,
                max_wheel_speed=self.controller.max_wheel_speed,
                label=f"formal active 50mm {active_id} {profile}",
                sequence_total_steps=len(self.steps),
            )
            for profile in set(self.profiles)
        }
        self.result.update(
            worker_pid=int(self.controller.latest_sim_status.get("pid", 0) or 0),
            worker_session_id=str(self.controller.latest_sim_status.get("worker_session_id", "") or ""),
            active_version_id=active_id,
            version_path=str(version_path.resolve()),
            accepted_steps_path=str(steps_path.resolve()),
            metadata_path=str(metadata_path.resolve()),
            accepted_steps_sha256=actual_sha,
            source_step_count=len(self.steps),
            source_event_count=sum(len(step.get("events", []) or []) for step in self.steps),
            source_command_count=int(metadata.get("command_count", 0) or 0),
            plan_summaries={name: self._plan_summary(plan) for name, plan in self.plans.items()},
            saved_state_audit=saved_state_audit,
            saved_state_counts=saved_state_counts,
        )
        selected_fast_started = self.controller.start_selected_step_playback(2, profile="fast")
        self.result["selected_fast_formal_gate"] = {
            "selected_step_index": 2,
            "started": bool(selected_fast_started),
            "last_error": self.controller.playback.last_error,
            "detail_text": self.controller.detail_text,
            "operation": self.controller.operation.state.value,
        }
        if selected_fast_started:
            raise AssertionError("formal placeholder checkpoints unexpectedly started Selected Fast")
        if "FULL_VALID" not in self.controller.playback.last_error:
            raise AssertionError("formal Selected Fast rejection did not identify the FULL_VALID requirement")
        self._capture_ui("active_version_loaded", "active_50mm_version_loaded.png")
        self._advance("START_PROFILE")

    def _stage_start_profile(self) -> None:
        if self.profile_index >= len(self.profiles):
            self._advance("FINISH")
            return
        if self.controller.operation.state is not OperationState.IDLE:
            return
        profile = self.profiles[self.profile_index]
        plan = self.plans[profile]
        replay_run_id = uuid.uuid4().hex
        started = self.controller.start_playback(
            self.steps,
            label=f"formal active 50mm {self.result['active_version_id']} {profile} run={replay_run_id}",
            profile=profile,
        )
        if not started:
            raise AssertionError(f"{profile} did not start: {self.controller.playback.last_error}")
        self.active_run = {
            "replay_run_id": replay_run_id,
            "profile": profile,
            "started_at": time.time(),
            "plan": self._plan_summary(plan),
            "expected_worker_plan_id": self.controller.playback.worker_plan_id,
            "expected_worker_request_id": self.controller.playback.worker_request_id,
            "segments": [],
            "final_detail_request_id": "",
        }
        self.last_segment_index = -1
        self.last_detail_request = 0.0
        self._capture_ui(f"{profile}_started", f"{profile}_started.png")
        self._advance("WAIT_PROFILE")

    def _stage_wait_profile(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if (
            str(worker.get("plan_id", "") or "") != self.active_run["expected_worker_plan_id"]
            or str(worker.get("request_id", "") or "") != self.active_run["expected_worker_request_id"]
        ):
            return
        segment_index = int(worker.get("segment_index", 0) or 0)
        now = time.monotonic()
        if segment_index != self.last_segment_index or now - self.last_detail_request > 0.5:
            self.controller.transport.request_state(detailed=True)
            self.last_detail_request = now
        if segment_index != self.last_segment_index:
            self.last_segment_index = segment_index
            plan = self.plans[self.active_run["profile"]]
            segment = plan.segments[segment_index] if 0 <= segment_index < len(plan.segments) else None
            detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
            state = dict(detailed.get("sim_state", {}) or {})
            row = {
                "observed_at": time.time(),
                "segment_index": segment_index,
                "worker_active": bool(worker.get("active", False)),
                "events_sent": int(worker.get("events_sent", 0) or 0),
                "current_servo_errors": copy.deepcopy(worker.get("current_servo_errors", {})),
                "last_info": str(worker.get("last_info", "") or ""),
                "progress": copy.deepcopy(worker.get("progress_detail", {})),
                "state": self._state_summary(state),
            }
            if segment is not None:
                row.update(
                    source_step=int(segment.source_step),
                    source_step_id=str(segment.source_step_id),
                    source_event_indices=list(
                        range(int(segment.event_start_index), int(segment.event_start_index + segment.event_count))
                    ),
                    planned_start_s=float(segment.planned_start_s),
                    servo_targets=copy.deepcopy(segment.servo_targets),
                    servo_duration_s=float(segment.servo_duration_s),
                    effective_tolerance_deg=float(segment.servo_tolerance_deg),
                    recorded_servo_residual_deg=copy.deepcopy(segment.recorded_servo_residual_deg),
                    wheel_targets=copy.deepcopy(segment.wheel_applied_target_rad_s),
                    wheel_duration_s=float(segment.wheel_active_duration_s),
                )
            self.active_run["segments"].append(row)
            if int(row.get("source_step", 0) or 0) == 11:
                self._capture_ui(
                    f"{self.active_run['profile']}_step11",
                    f"{self.active_run['profile']}_step11.png",
                )
        if worker.get("active") or not str(worker.get("stop_reason", "") or ""):
            return
        if not self.active_run.get("final_detail_request_id"):
            self.active_run["final_detail_request_id"] = self.controller.transport.request_state(
                detailed=True,
                purpose=f"formal_{self.active_run['profile']}_final_trace",
            )
            return
        detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
        if (
            str(detailed.get("state_capture_request_id", "") or "")
            != str(self.active_run["final_detail_request_id"])
        ):
            return
        detailed_worker = dict(detailed.get("worker_playback", {}) or {})
        if (
            str(detailed_worker.get("plan_id", "") or "") != self.active_run["expected_worker_plan_id"]
            or str(detailed_worker.get("request_id", "") or "") != self.active_run["expected_worker_request_id"]
        ):
            return
        progress = dict(detailed_worker.get("progress_detail", {}) or {})
        self.active_run.update(
            completed_at=time.time(),
            final_worker=copy.deepcopy(detailed_worker),
            last_completed_segment=max(-1, int(detailed_worker.get("segment_index", 0) or 0) - 1),
            last_started_segment=int(detailed_worker.get("segment_index", 0) or 0),
            last_started_step=int(progress.get("current_step_index", 0) or 0),
            final_operation=self.controller.operation.state.value,
        )
        self.result["runs"].append(copy.deepcopy(self.active_run))
        profile = self.active_run["profile"]
        self._capture_ui(f"{profile}_final", f"{profile}_final.png")
        if self.require_complete and str(detailed_worker.get("stop_reason", "")) != "complete":
            raise AssertionError(f"{profile} stopped early: {detailed_worker.get('last_error')}")
        self.profile_index += 1
        if self.profile_index < len(self.profiles):
            self.respawn_requested_at = 0.0
            self._advance("RESPAWN")
        else:
            self._advance("FINISH")

    def _stage_respawn(self) -> None:
        if self.respawn_requested_at <= 0.0:
            if not self.controller.respawn_robot(source="manual"):
                raise AssertionError("Respawn between profiles failed")
            self.respawn_requested_at = time.monotonic()
            return
        if time.monotonic() - self.respawn_requested_at < 1.0:
            return
        if self.controller.operation.state is OperationState.IDLE:
            self._advance("START_PROFILE")

    def _stage_finish(self) -> None:
        self.controller.stop_wheels(reason="latest_50mm_manual_v2_e2e_complete")
        self.result["accepted_steps_sha256_after"] = hashlib.sha256(
            Path(self.result["accepted_steps_path"]).read_bytes()
        ).hexdigest()
        if self.result["accepted_steps_sha256_after"] != self.result["accepted_steps_sha256"]:
            raise AssertionError("formal active 50 mm version changed")
        self.result["final_operation"] = self.controller.operation.state.value
        self.result["success"] = True
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(300, lambda: self.ui._window_close(force=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--profiles", nargs="+", choices=("raw", "fast"), default=["fast", "raw"])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    return Latest50mmManualV2E2E(
        Path(args.output),
        profiles=list(args.profiles),
        require_complete=args.require_complete,
        timeout_s=args.timeout_s,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
