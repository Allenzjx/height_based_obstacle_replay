from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from playback import PlaybackEvent, PlaybackManager, PlaybackPlan, PlaybackSegment, SimTimePlaybackService, plan_fingerprint, playback_plan_to_payload  # noqa: E402
from sim_process_client import SimProcessClient  # noqa: E402


class FakeAdapter:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.stop_count = 0

    def handle_command(self, message: Any) -> None:
        self.commands.append(str(message.text))

    def stop_wheels(self) -> None:
        self.stop_count += 1

    def apply_commands_to_robot(self) -> None:
        pass


class AtomicBatchAdapter(FakeAdapter):
    def __init__(self, *, valid_ack: bool = True) -> None:
        super().__init__()
        self.valid_ack = bool(valid_ack)
        self.wheel_speeds = {
            "front_left_ankle": 0.0,
            "front_right_ankle": 0.0,
            "rear_left_ankle": 0.0,
            "rear_right_ankle": 0.0,
        }
        self.batches: list[dict[str, Any]] = []
        self.sim_steps = 10

    def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.batches.append(dict(payload))
        self.wheel_speeds.update(dict(payload.get("wheel_targets_rad_s", {}) or {}))
        ack = {
            "batch_id": payload["batch_id"],
            "error": "",
            "applied_sim_step": int(self.sim_steps),
            "first_physics_step": int(self.sim_steps) + 1,
            "motion_start_skew_s": 0.0,
            "servo_applied": bool(payload.get("servo_targets_deg")),
            "wheel_applied": True,
            "servo_targets_applied": dict(payload.get("servo_targets_deg", {}) or {}),
            "wheel_targets_applied": dict(payload.get("wheel_targets_rad_s", {}) or {}),
            "recording_metadata": dict(
                payload.get("recording_metadata", {}) or {}
            ),
        }
        if not self.valid_ack:
            ack["applied_sim_step"] = int(self.sim_steps) - 1
        return ack


class FakeWorkerTransport:
    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.stopped: list[str] = []

    def start_playback_plan(
        self,
        plan: PlaybackPlan,
        *,
        start_delay_sim_s: float = 0.0,
        plan_id: str = "",
        request_id: str = "",
        plan_sha256: str = "",
    ) -> None:
        self.started.append({
            "plan": plan,
            "delay": float(start_delay_sim_s),
            "plan_id": str(plan_id),
            "request_id": str(request_id),
            "plan_sha256": str(plan_sha256),
        })

    def stop_playback(self, *, reason: str = "stopped", stop_wheels: bool = True) -> None:
        self.stopped.append(str(reason))


class FakePlaybackController:
    def __init__(self) -> None:
        self.transport = FakeWorkerTransport()
        self.latest_sim_status: dict[str, Any] = {}


def _plan() -> PlaybackPlan:
    plan = PlaybackPlan(
        path=Path("saved_height_steps/height_05cm/accepted_steps.jsonl"),
        events=[
            PlaybackEvent(0.0, "wheel all 1.0", source_step=1),
            PlaybackEvent(0.1, "wheel stop", source_step=1),
        ],
        final_time_s=0.1,
        label="unit sim-time plan",
    )
    return plan


def _worker_plan() -> PlaybackPlan:
    plan = _plan()
    plan.segments = [
        PlaybackSegment(
            segment_index=0,
            source_step=1,
            source_step_id="step-1",
            event_start_index=0,
            event_count=2,
            planned_start_s=0.0,
            planned_end_s=0.1,
            base_duration_s=0.1,
            servo_base_duration_s=0.0,
            servo_duration_s=0.0,
        )
    ]
    plan.timing["plan_integrity"] = {
        "input_step_count": 1,
        "required_step_indices": [1],
        "represented_step_indices": [1],
        "missing_required_step_indices": [],
    }
    plan.plan_sha256 = plan_fingerprint(plan)
    return plan


def _args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=5,
        robot_usd="robot.usd",
        save_usd="scene.usd",
        spawn_z=0.04,
        obstacle_x=1.55,
        obstacle_width=None,
        obstacle_length=None,
        infer_obstacle_size=True,
        robot_width=0.80,
        robot_length=0.55,
        physics_dt=1.0 / 120.0,
        render_interval=2,
        device="cuda:0",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        apply_physx_joint_limits=False,
        ui_refresh_ms=100,
        sim_status_refresh_ms=250,
        full_refresh_ms=1000,
        no_continuous_sim_step=False,
        wheel_direction=1.0,
        servo_stiffness=600.0,
        servo_damping=60.0,
        wheel_damping=20.0,
        save_scene=False,
        sim_launch_mode="disabled",
        restore_step_start_state_before_selected_playback=True,
        restore_full_sim_pose_if_available=True,
        fallback_to_command_state_before=True,
        playback_pre_step_settle_s=0.30,
        respawn_play_settle_s=0.30,
    )


class SimTimePlaybackServiceTest(unittest.TestCase):
    @staticmethod
    def _atomic_segment_plan() -> PlaybackPlan:
        wheel_targets = {
            "front_left_ankle": 1.0,
            "front_right_ankle": 1.0,
            "rear_left_ankle": 1.0,
            "rear_right_ankle": 1.0,
        }
        event = PlaybackEvent(
            0.0,
            "wheel all 1.0",
            source_step=1,
            source_step_id="step-1",
            global_command_index=1,
            segment_index=0,
            channel="wheel",
            wheel_applied_target_rad_s=tuple(wheel_targets.values()),
            wheel_active_duration_s=0.1,
        )
        segment = PlaybackSegment(
            segment_index=0,
            source_step=1,
            source_step_id="step-1",
            event_start_index=0,
            event_count=1,
            planned_start_s=0.0,
            planned_end_s=0.1,
            base_duration_s=0.1,
            servo_base_duration_s=0.0,
            servo_duration_s=0.0,
            wheel_active_duration_s=0.1,
            wheel_base_velocity=wheel_targets,
            wheel_requested_velocity_rad_s=wheel_targets,
            wheel_applied_target_rad_s=wheel_targets,
        )
        return PlaybackPlan(
            path=None,
            events=[event],
            segments=[segment],
            final_time_s=0.1,
            label="atomic batch plan",
        )

    def test_segment_records_and_validates_independent_motion_batch_ack(self) -> None:
        adapter = AtomicBatchAdapter(valid_ack=True)
        service = SimTimePlaybackService()
        plan = self._atomic_segment_plan()
        self.assertTrue(
            service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0)
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=10,
            current_wall_time_s=0.0,
        )
        rows = service.timing_trace["motion_batches"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ack_valid"])
        self.assertEqual(rows[0]["applied_sim_step"], 10)
        self.assertEqual(rows[0]["first_physics_step"], 11)
        self.assertEqual(rows[0]["motion_start_skew_s"], 0.0)
        self.assertEqual(
            set(rows[0]["wheel_targets_rad_s"]), set(adapter.wheel_speeds)
        )

    def test_invalid_motion_batch_ack_aborts_scheduler_fail_closed(self) -> None:
        adapter = AtomicBatchAdapter(valid_ack=False)
        service = SimTimePlaybackService()
        plan = self._atomic_segment_plan()
        service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0)
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=10,
            current_wall_time_s=0.0,
        )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "atomic_dispatch_invalid")
        self.assertIn("applied_sim_step", service.last_error)
        self.assertFalse(service.timing_trace["motion_batches"][0]["ack_valid"])
        self.assertEqual(len(adapter.batches), 1)

    def test_segment_dispatches_compiled_applied_wheel_target_not_raw_request(self) -> None:
        wheel_limit = 2.0943951023931953
        requested = {
            "front_left_ankle": 0.0,
            "front_right_ankle": 2.0944,
            "rear_left_ankle": 0.0,
            "rear_right_ankle": 0.0,
        }
        applied = {**requested, "front_right_ankle": wheel_limit}
        event = PlaybackEvent(
            0.0,
            "wheel fr 2.0944",
            source_step=14,
            source_step_id="step-14",
            global_command_index=73,
            segment_index=56,
            channel="wheel",
            wheel_requested_velocity_rad_s=(2.0944,),
            wheel_applied_target_rad_s=(wheel_limit,),
            wheel_active_duration_s=1.6,
        )
        segment = PlaybackSegment(
            segment_index=56,
            source_step=14,
            source_step_id="step-14",
            event_start_index=0,
            event_count=1,
            planned_start_s=0.0,
            planned_end_s=1.6,
            base_duration_s=1.6,
            servo_base_duration_s=0.0,
            servo_duration_s=0.0,
            wheel_active_duration_s=1.6,
            wheel_base_velocity=requested,
            wheel_requested_velocity_rad_s=requested,
            wheel_applied_target_rad_s=applied,
        )
        plan = PlaybackPlan(
            path=None,
            events=[event],
            segments=[segment],
            final_time_s=1.6,
            label="recorded wheel clamp",
        )
        adapter = AtomicBatchAdapter(valid_ack=True)
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0)
        )

        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=10,
            current_wall_time_s=0.0,
        )

        self.assertEqual(
            adapter.batches[0]["wheel_targets_rad_s"]["front_right_ankle"],
            wheel_limit,
        )
        self.assertEqual(
            segment.wheel_requested_velocity_rad_s["front_right_ankle"],
            2.0944,
        )
        self.assertTrue(service.timing_trace["motion_batches"][0]["ack_valid"])

    def test_start_boundary_and_final_stop_are_separate_acked_ticks(self) -> None:
        adapter = AtomicBatchAdapter(valid_ack=True)
        service = SimTimePlaybackService()
        plan = self._atomic_segment_plan()
        service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0)
        self.assertTrue(
            service.apply_playback_start_boundary(
                adapter, current_sim_time_s=0.0, current_sim_step=10
            )
        )
        adapter.sim_steps = 11
        service.update(
            adapter,
            current_sim_time_s=1.0 / 120.0,
            current_sim_step=11,
            current_wall_time_s=0.01,
        )
        adapter.sim_steps = 12
        service.update(
            adapter,
            current_sim_time_s=0.2,
            current_sim_step=12,
            current_wall_time_s=0.2,
        )
        adapter.sim_steps = 13
        service.update(
            adapter,
            current_sim_time_s=0.2 + 1.0 / 120.0,
            current_sim_step=13,
            current_wall_time_s=0.21,
        )
        rows = service.timing_trace["motion_batches"]
        self.assertEqual(
            [row["dispatch_kind"] for row in rows],
            ["playback_start_boundary", "source_segment_start", "final_safety_stop"],
        )
        # Tick 12 has no source batch, so its dispatch slot is free for the
        # independently ACKed final stop; an unconditional extra idle tick is
        # neither required nor part of the recording timeline.
        self.assertEqual([row["applied_sim_step"] for row in rows], [10, 11, 12])

    def test_zero_duration_stop_then_next_segment_uses_distinct_ticks(self) -> None:
        wheels_zero = {
            "front_left_ankle": 0.0,
            "front_right_ankle": 0.0,
            "rear_left_ankle": 0.0,
            "rear_right_ankle": 0.0,
        }
        wheels_next = {name: 1.0 for name in wheels_zero}
        events = [
            PlaybackEvent(
                0.0,
                "wheel stop",
                source_step=1,
                global_command_index=1,
                segment_index=0,
                channel="wheel",
                wheel_applied_target_rad_s=tuple(wheels_zero.values()),
            ),
            PlaybackEvent(
                0.0,
                "wheel all 1.0",
                source_step=1,
                global_command_index=2,
                segment_index=1,
                channel="wheel",
                wheel_applied_target_rad_s=tuple(wheels_next.values()),
            ),
        ]
        segments = [
            PlaybackSegment(
                segment_index=index,
                source_step=1,
                source_step_id="step-1",
                event_start_index=index,
                event_count=1,
                planned_start_s=0.0,
                planned_end_s=0.0,
                base_duration_s=0.0,
                servo_base_duration_s=0.0,
                servo_duration_s=0.0,
                wheel_active_duration_s=0.0,
                wheel_base_velocity=targets,
                wheel_requested_velocity_rad_s=targets,
                wheel_applied_target_rad_s=targets,
            )
            for index, targets in enumerate((wheels_zero, wheels_next))
        ]
        service = SimTimePlaybackService()
        adapter = AtomicBatchAdapter(valid_ack=True)
        self.assertTrue(
            service.start_plan(
                PlaybackPlan(
                    path=None,
                    events=events,
                    segments=segments,
                    final_time_s=0.0,
                ),
                current_sim_time_s=0.0,
                current_wall_time_s=0.0,
            )
        )
        for step in (10, 11, 12):
            adapter.sim_steps = step
            service.update(
                adapter,
                current_sim_time_s=(step - 10) / 120.0,
                current_sim_step=step,
                current_wall_time_s=float(step),
            )
        rows = service.timing_trace["motion_batches"]
        self.assertEqual(
            [row["dispatch_kind"] for row in rows],
            ["source_segment_start", "source_segment_start", "final_safety_stop"],
        )
        self.assertEqual([row["applied_sim_step"] for row in rows], [10, 11, 12])
        self.assertEqual(len(adapter.batches), 3)
        self.assertEqual(service.stop_reason, "complete")

    def test_central_motion_batch_slot_rejects_duplicate_without_adapter_call(self) -> None:
        adapter = AtomicBatchAdapter(valid_ack=True)
        service = SimTimePlaybackService()
        plan = self._atomic_segment_plan()
        service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0)
        self.assertTrue(
            service.apply_playback_start_boundary(
                adapter, current_sim_time_s=0.0, current_sim_step=10
            )
        )
        self.assertFalse(
            service.apply_playback_start_boundary(
                adapter, current_sim_time_s=0.0, current_sim_step=10
            )
        )
        self.assertEqual(len(adapter.batches), 1)
        self.assertFalse(service.timing_trace["motion_batches"][-1]["ack_valid"])
        self.assertFalse(service.timing_trace["motion_batches"][-1]["adapter_called"])

    def test_dispatches_by_sim_time_not_wall_time(self) -> None:
        adapter = FakeAdapter()
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(_plan(), current_sim_time_s=10.0, current_wall_time_s=100.0))

        service.update(adapter, current_sim_time_s=10.0, current_sim_step=1, current_wall_time_s=500.0)
        self.assertEqual(adapter.commands, ["wheel all 1.0"])

        service.update(adapter, current_sim_time_s=10.05, current_sim_step=2, current_wall_time_s=900.0)
        self.assertEqual(adapter.commands, ["wheel all 1.0"])

        service.update(adapter, current_sim_time_s=10.1, current_sim_step=3, current_wall_time_s=901.0)
        self.assertEqual(adapter.commands, ["wheel all 1.0", "wheel stop"])
        status = service.status_dict(current_sim_time_s=10.1, current_wall_time_s=901.0)
        self.assertFalse(status["active"])
        self.assertEqual(status["stop_reason"], "complete")
        self.assertEqual(status["dispatch_clock"], "simulation_time")

    def test_pause_resume_freezes_sim_time_offset(self) -> None:
        adapter = FakeAdapter()
        service = SimTimePlaybackService()
        service.start_plan(_plan(), current_sim_time_s=0.0, current_wall_time_s=0.0)
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=0, current_wall_time_s=0.0)
        service.pause(current_sim_time_s=0.05)
        service.update(adapter, current_sim_time_s=0.30, current_sim_step=30, current_wall_time_s=30.0)
        self.assertEqual(adapter.commands, ["wheel all 1.0"])
        service.resume(current_sim_time_s=0.30)
        service.update(adapter, current_sim_time_s=0.34, current_sim_step=34, current_wall_time_s=34.0)
        self.assertEqual(adapter.commands, ["wheel all 1.0"])
        service.update(adapter, current_sim_time_s=0.35, current_sim_step=35, current_wall_time_s=35.0)
        self.assertEqual(adapter.commands, ["wheel all 1.0", "wheel stop"])

    def test_process_client_queues_full_plan_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = SimProcessClient(_args(Path(tmp)))
            payload = playback_plan_to_payload(_plan())
            client.start_playback_plan(payload, start_delay_sim_s=0.25, plan_id="request-123")
            message = client.pending_messages[-1]
            self.assertEqual(message["type"], "start_playback_plan")
            self.assertEqual(message["start_delay_sim_s"], 0.25)
            self.assertEqual(message["plan_id"], "request-123")
            self.assertEqual(message["plan"]["events"][0]["command"], "wheel all 1.0")

    def test_manager_waits_for_matching_worker_ack_before_syncing_idle(self) -> None:
        controller = FakePlaybackController()
        manager = PlaybackManager(controller)
        plan = _worker_plan()
        self.assertTrue(manager.start_worker_plan(plan))

        manager.sync_worker_status({"active": False, "paused": False, "plan_id": "", "index": 0, "events_sent": 0})
        self.assertFalse(manager.active)
        self.assertTrue(manager.start_requested)
        self.assertTrue(manager.worker_managed)
        self.assertFalse(manager.worker_acknowledged)
        self.assertEqual(manager.progress.playback_state, "START_REQUESTED")

        ack = {
            "operation": "start_playback_plan",
            "request_id": manager.worker_request_id,
            "accepted": True,
            "plan_id": manager.worker_plan_id,
            "plan_sha256": plan.plan_sha256,
            "event_count": len(plan.events),
            "segment_count": len(plan.segments),
            "worker_session_id": "session-1",
        }
        manager.sync_worker_status(
            {
                "active": True,
                "paused": False,
                "plan_id": manager.worker_plan_id,
                "request_id": manager.worker_request_id,
                "worker_session_id": "session-1",
                "started": True,
                "first_command_applied": True,
                "index": 1,
                "events_sent": 1,
                "stop_reason": "",
                "last_info": "worker playback running",
            },
            operation_ack=ack,
        )
        self.assertTrue(manager.active)
        self.assertTrue(manager.worker_acknowledged)
        self.assertEqual(manager.index, 1)

    def test_repeated_identical_plan_uses_unique_request_ids_and_ignores_stale_status(self) -> None:
        controller = FakePlaybackController()
        manager = PlaybackManager(controller)
        plan = _worker_plan()

        self.assertTrue(manager.start_worker_plan(plan))
        first_id = manager.worker_plan_id
        manager.stop(silent=True)
        self.assertTrue(manager.start_worker_plan(plan))
        second_id = manager.worker_plan_id

        self.assertNotEqual(first_id, second_id)
        self.assertEqual(controller.transport.started[0]["plan_id"], first_id)
        self.assertEqual(controller.transport.started[1]["plan_id"], second_id)
        self.assertNotEqual(controller.transport.started[0]["request_id"], controller.transport.started[1]["request_id"])
        manager.sync_worker_status(
            {
                "active": False,
                "paused": False,
                "plan_id": first_id,
                "stop_reason": "stopped",
                "events_sent": 1,
            }
        )
        self.assertFalse(manager.active)
        self.assertTrue(manager.start_requested)
        self.assertTrue(manager.worker_managed)
        self.assertFalse(manager.worker_acknowledged)

        ack = {
            "operation": "start_playback_plan",
            "request_id": manager.worker_request_id,
            "accepted": True,
            "plan_id": second_id,
            "plan_sha256": plan.plan_sha256,
            "event_count": len(plan.events),
            "segment_count": len(plan.segments),
            "worker_session_id": "session-2",
        }
        manager.sync_worker_status(
            {
                "active": True,
                "paused": False,
                "plan_id": second_id,
                "request_id": manager.worker_request_id,
                "worker_session_id": "session-2",
                "started": True,
                "first_command_applied": True,
                "stop_reason": "",
                "events_sent": 1,
            },
            operation_ack=ack,
        )
        self.assertTrue(manager.active)
        self.assertTrue(manager.worker_acknowledged)


if __name__ == "__main__":
    unittest.main()
