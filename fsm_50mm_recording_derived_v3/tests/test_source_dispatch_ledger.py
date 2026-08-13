from __future__ import annotations

import copy
import unittest

from command_model import WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.recording_fast_plan import (
    build_source_dispatch_ledger,
    fast_plan_rows,
)


class SourceDispatchLedgerTests(unittest.TestCase):
    @staticmethod
    def _steps() -> list[dict]:
        return [
            {
                "index": 1,
                "name": "step-1",
                "duration": 1.0,
                "events": [
                    {
                        "time": 0.0,
                        "command": "wheel stop",
                        "kind": "command",
                    },
                    {
                        "time": 0.1,
                        "command": "servo front_left_hip 10",
                        "kind": "command",
                    },
                    {
                        "time": 0.2,
                        "command": "wheel all 1",
                        "kind": "command",
                        "wheel_active_duration_s": 0.3,
                    },
                ],
                "command_state_before": {
                    "servos": {},
                    "wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES},
                },
            }
        ]

    @staticmethod
    def _live_trace(plan) -> dict:
        commands = []
        for event in plan.events:
            commands.append(
                {
                    "global_command_index": int(event.global_command_index),
                    "segment_index": int(event.segment_index),
                    "planned_start_sim_time": float(event.time_s),
                    "actual_start_sim_time": float(event.time_s),
                    "scheduler_lateness": 0.0,
                }
            )
        batches = [
            {
                "dispatch_kind": "playback_start_boundary",
                "batch_id": "start-boundary",
                "segment_index": 0,
                "source_step_index": 1,
                "ack_valid": True,
                "ack_error": "",
                "applied_sim_step": 1,
                "first_physics_step": 2,
                "motion_start_skew_s": 0.0,
                "servo_targets_deg": {},
                "wheel_targets_rad_s": {
                    name: 0.0 for name in WHEEL_JOINT_NAMES
                },
            }
        ]
        for segment in plan.segments:
            batches.append(
                {
                    "dispatch_kind": "source_segment_start",
                    "batch_id": f"batch-{segment.segment_index}",
                    "segment_index": int(segment.segment_index),
                    "source_step_index": int(segment.source_step),
                    "ack_valid": True,
                    "ack_error": "",
                    "applied_sim_step": int(segment.segment_index + 10),
                    "first_physics_step": int(segment.segment_index + 11),
                    "motion_start_skew_s": 0.0,
                    "servo_targets_deg": dict(segment.servo_targets),
                    "wheel_targets_rad_s": {
                        name: float(segment.wheel_applied_target_rad_s.get(name, 0.0))
                        for name in WHEEL_JOINT_NAMES
                    },
                    "recording_metadata": {
                        "motion_start_readiness_token": "c" * 64,
                        "motion_start_readiness_bound_sim_step": 2,
                    },
                }
            )
        final_applied_step = len(plan.segments) + 10
        batches.append(
            {
                "dispatch_kind": "final_safety_stop",
                "batch_id": "final-stop",
                "segment_index": len(plan.segments) - 1,
                "source_step_index": 1,
                "ack_valid": True,
                "ack_error": "",
                "applied_sim_step": final_applied_step,
                "first_physics_step": final_applied_step + 1,
                "motion_start_skew_s": 0.0,
                "servo_targets_deg": {},
                "wheel_targets_rad_s": {
                    name: 0.0 for name in WHEEL_JOINT_NAMES
                },
            }
        )
        return {"commands": commands, "motion_batches": batches}

    def test_every_source_command_is_applied_or_explicit_semantic_noop(self):
        steps = self._steps()
        plan, _ = fast_plan_rows(
            source_version="v003-test", steps=steps, max_wheel_speed=3.0
        )
        rows, summary = build_source_dispatch_ledger(
            source_version="v003-test",
            steps=steps,
            plan=plan,
            timing_trace=self._live_trace(plan),
        )
        self.assertTrue(summary["complete"], summary["errors"])
        self.assertEqual(len(rows), summary["source_command_count"])
        self.assertEqual(
            [row["source_command_cursor"] for row in rows],
            list(range(1, len(rows) + 1)),
        )
        self.assertEqual(
            summary["retained_plan_event_count"], summary["plan_event_count"]
        )
        self.assertEqual(
            summary["semantic_noop_count"]
            + summary["retained_plan_event_count"],
            summary["source_command_count"],
        )
        self.assertTrue(
            all(
                row["atomic_batch_ack_valid"]
                for row in rows
                if row["compiler_outcome"].startswith("APPLIED")
            )
        )

    def test_missing_timing_or_invalid_batch_ack_fails_closed(self):
        steps = self._steps()
        plan, _ = fast_plan_rows(
            source_version="v003-test", steps=steps, max_wheel_speed=3.0
        )
        trace = self._live_trace(plan)
        trace["commands"].pop()
        trace["motion_batches"][0]["ack_valid"] = False
        trace["motion_batches"][0]["ack_error"] = "synthetic invalid ACK"
        _rows, summary = build_source_dispatch_ledger(
            source_version="v003-test",
            steps=steps,
            plan=plan,
            timing_trace=trace,
        )
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["errors"])

    def test_duplicate_plan_event_mapping_fails_closed(self):
        steps = self._steps()
        plan, _ = fast_plan_rows(
            source_version="v003-test", steps=steps, max_wheel_speed=3.0
        )
        duplicate = copy.deepcopy(plan.events[0])
        duplicate.global_command_index = max(
            event.global_command_index for event in plan.events
        ) + 1
        plan.events.append(duplicate)
        trace = self._live_trace(plan)
        _rows, summary = build_source_dispatch_ledger(
            source_version="v003-test",
            steps=steps,
            plan=plan,
            timing_trace=trace,
        )
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["errors"])

    def test_invalid_runtime_batch_or_missing_final_stop_fails_closed(self):
        steps = self._steps()
        plan, _ = fast_plan_rows(
            source_version="v003-test", steps=steps, max_wheel_speed=3.0
        )
        trace = self._live_trace(plan)
        trace["motion_batches"][-1]["ack_valid"] = False
        trace["motion_batches"][-1]["ack_error"] = "tampered final stop"
        _rows, summary = build_source_dispatch_ledger(
            source_version="v003-test",
            steps=steps,
            plan=plan,
            timing_trace=trace,
        )
        self.assertFalse(summary["complete"])
        self.assertIn("invalid ACK", " ".join(summary["errors"]))

        trace = self._live_trace(plan)
        trace["motion_batches"].pop()
        _rows, summary = build_source_dispatch_ledger(
            source_version="v003-test",
            steps=steps,
            plan=plan,
            timing_trace=trace,
        )
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["final_safety_stop_count"], 0)


if __name__ == "__main__":
    unittest.main()
