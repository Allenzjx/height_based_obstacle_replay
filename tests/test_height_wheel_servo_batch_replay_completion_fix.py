from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from operation_coordinator import OperationState
from playback import (
    PlaybackEvent,
    PlaybackManager,
    PlaybackPlan,
    PlaybackSegment,
    SimTimePlaybackService,
    plan_fingerprint,
    plan_from_steps,
    playback_plan_from_payload,
    playback_plan_to_payload,
    validate_plan_integrity,
)
from sequence_model import empty_command_state, make_event, make_step
from sim_ui_controller import HeightReplayController
from tests.controller_test_utils import make_args


class FakeHeightClient:
    def __init__(self) -> None:
        self.connected = True
        self.latest_detailed_status: dict[str, Any] = {}
        self.last_status_time = time.monotonic()
        self.payloads: list[dict[str, Any]] = []
        self.current_status: dict[str, Any] = {
            "runtime_ready": True,
            "ready": True,
            "phase": "running",
            "height_mm": 50,
            "obstacle_revision": 0,
        }

    def set_height_mm(self, height_mm: int, **payload: Any) -> None:
        self.payloads.append({"height_mm": height_mm, **payload})

    def set_height_respawn(self, height_mm: int, **payload: Any) -> None:
        self.payloads.append({"height_mm": height_mm, "respawn": True, **payload})

    def poll(self) -> None:
        self.last_status_time = time.monotonic()

    def status(self) -> dict[str, Any]:
        return dict(self.current_status)


class HeightTransactionTest(unittest.TestCase):
    def test_ui_rejects_false_success_and_accepts_measured_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            fake = FakeHeightClient()
            controller.no_sim = False
            controller.sim_launch_mode = "subprocess"
            controller.sim_client = fake  # type: ignore[assignment]
            controller.sim_connection_enabled = True
            controller.latest_sim_status = dict(fake.current_status)
            controller.transport.attach_process_client(fake)
            controller.current_height_mm = 75
            request_id = controller.generate_or_update_height_obstacle()
            self.assertTrue(request_id)
            self.assertEqual(controller.operation.state, OperationState.SCENE_UPDATE)
            self.assertEqual(fake.payloads[-1]["requested_revision"], 1)

            fake.current_status["last_operation_ack"] = {
                "operation": "set_height",
                "request_id": request_id,
                "accepted": True,
                "prim_valid": True,
                "measured_height_mm": 50.0,
                "visual_updated": True,
                "collision_updated": True,
                "obstacle_revision": 1,
                "control_ready": True,
                "error": "",
            }
            controller.update()
            self.assertTrue(controller.operation.idle)
            self.assertIn("ERROR", controller.status)
            self.assertNotEqual(controller.loaded_sim_height_mm, 75)

            request_id = controller.generate_or_update_height_obstacle()
            fake.current_status.update(height_mm=75, obstacle_revision=2)
            fake.current_status["last_operation_ack"] = {
                "operation": "set_height",
                "request_id": request_id,
                "accepted": True,
                "prim_valid": True,
                "requested_height_mm": 75,
                "measured_height_mm": 75.2,
                "measured_width_m": 2.0,
                "measured_bounds": {"min": [0.5, -1.0, 0.0], "max": [2.5, 1.0, 0.0752]},
                "visual_updated": True,
                "collision_updated": True,
                "obstacle_revision": 2,
                "control_ready": True,
                "error": "",
            }
            controller.update()
            self.assertTrue(controller.operation.idle)
            self.assertEqual(controller.loaded_sim_height_mm, 75)
            self.assertIn("geometry verified", controller.status)


class WheelReadinessTest(unittest.TestCase):
    def test_recording_uses_motion_readiness_not_respawn_reference_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.no_sim = False
            controller.latest_sim_status = {
                "phase": "running",
                "runtime_ready": True,
                "grounded_reference_valid": False,
                "grounded_reference_stable": False,
                "robot_ground": {
                    "ground_state": "PASS_WITH_VISUAL_WARNING",
                    "classification": "VISUAL_ONLY_INTERSECTION",
                    "physical_ground_safe": True,
                },
            }
            allowed, reason = controller.can_start_recording()
            self.assertTrue(allowed, reason)

    def test_manual_wheel_uses_same_path_before_during_and_after_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            calls: list[str] = []
            original = controller.transport.send

            def spy(command: str, **kwargs: Any) -> None:
                calls.append(command)
                original(command, **kwargs)

            controller.transport.send = spy  # type: ignore[method-assign]
            controller.handle_command("wheel all 0.2")
            controller.handle_command("step_record start")
            controller.handle_command("wheel all 0.3")
            controller.stop_step_recording()
            controller.update()
            controller.handle_command("wheel all 0.4")
            self.assertIn("wheel all 0.2", calls)
            self.assertIn("wheel all 0.3", calls)
            self.assertIn("wheel all 0.4", calls)

    def test_worker_generation_sync_prevents_post_stop_stale_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.transport.wheel_generation = 2
            controller.transport.update_worker_status({
                "runtime_ready": True,
                "wheel_command": {"generation": 9},
            })
            self.assertEqual(controller.transport.wheel_generation, 9)
            controller.handle_command("wheel all 0.25")
            self.assertEqual(controller.transport.adapter.wheel_generation, 9)
            self.assertFalse(controller.transport.adapter.wheel_command_status["stale_command_rejected"])


class PlanIntegrityAndCompletionTest(unittest.TestCase):
    def test_recorded_residual_margin_is_bounded_by_three_degree_safety_cap(self) -> None:
        before = empty_command_state()
        step = make_step(
            index=1,
            step_type="recorded",
            duration=0.1,
            events=[make_event(0.0, "servo front_left_hip 10")],
            command_state_before=before,
            command_state_after=before,
        )
        step["sim_state_after"] = {
            "actual_joint_state": {"servos": {"front_left_hip": {"deg": 12.0}}},
            "target_joint_state": {"servos": {"front_left_hip": {"target_actual_deg": 10.0}}},
        }
        segment = plan_from_steps([step]).segments[0]
        self.assertAlmostEqual(segment.servo_tolerance_deg, 2.75)

        step["sim_state_after"]["actual_joint_state"]["servos"]["front_left_hip"]["deg"] = 15.0
        capped = plan_from_steps([step]).segments[0]
        self.assertEqual(capped.servo_tolerance_deg, 3.0)

    def test_fifty_steps_five_hundred_commands_round_trip_without_truncation(self) -> None:
        steps = []
        for step_index in range(1, 51):
            before = empty_command_state()
            events = [
                make_event(command_index * 0.01, f"servo front_left_hip {((step_index + command_index) % 30):.1f}")
                for command_index in range(10)
            ]
            steps.append(
                make_step(
                    index=step_index,
                    step_type="recorded",
                    duration=0.1,
                    events=events,
                    command_state_before=before,
                    command_state_after=before,
                )
            )
        plan = plan_from_steps(steps, profile="raw", sequence_total_steps=50)
        self.assertEqual(len(plan.events), 500)
        payload = playback_plan_to_payload(plan)
        decoded = playback_plan_from_payload(payload)
        integrity = validate_plan_integrity(
            decoded,
            expected_plan_sha256=plan.plan_sha256,
            expected_event_count=500,
            expected_segment_count=len(plan.segments),
        )
        self.assertTrue(integrity["ok"], integrity)
        self.assertEqual(integrity["represented_step_indices"], list(range(1, 51)))

    def test_plan_sha_detects_payload_tampering(self) -> None:
        before = empty_command_state()
        step = make_step(
            index=1,
            step_type="recorded",
            duration=0.1,
            events=[make_event(0.0, "servo front_left_hip 10")],
            command_state_before=before,
            command_state_after=before,
        )
        plan = plan_from_steps([step], profile="raw")
        payload = playback_plan_to_payload(plan)
        payload["events"][0]["command"] = "servo front_left_hip 11"
        decoded = playback_plan_from_payload(payload)
        result = validate_plan_integrity(decoded, expected_plan_sha256=plan.plan_sha256)
        self.assertFalse(result["ok"])
        self.assertIn("plan sha mismatch", "; ".join(result["errors"]))

    def test_recorded_contact_residual_completes_without_target_rewrite(self) -> None:
        class LoadedAdapter:
            def __init__(self) -> None:
                self.joint_command_deg = {"front_left_hip": 0.0}
                self.applied: list[dict[str, Any]] = []
                self.motion_reference = SimpleNamespace(
                    servo_reference_velocity_deg_s=150.0,
                    servo_velocity_limit_deg_s=150.0,
                )

            def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
                self.applied.append(dict(payload))
                self.joint_command_deg.update(payload.get("servo_targets_deg", {}))
                return {}

            def get_actual_joint_state(self) -> dict[str, Any]:
                return {"servos": {"front_left_hip": {"deg": 11.3}}}

            def command_to_actual_target_deg(self, _name: str, target: float) -> float:
                return float(target)

            def stop_wheels(self) -> None:
                pass

            def apply_commands_to_robot(self) -> None:
                pass

        event = PlaybackEvent(
            time_s=0.0,
            command="servo front_left_hip 10",
            source_step=26,
            source_step_id="step-26",
            commands_in_step=1,
            global_command_index=1,
            segment_index=0,
            channel="servo",
            servo_targets=(("front_left_hip", 10.0),),
        )
        segment = PlaybackSegment(
            segment_index=0,
            source_step=26,
            source_step_id="step-26",
            event_start_index=0,
            event_count=1,
            planned_start_s=0.0,
            planned_end_s=0.1,
            base_duration_s=0.1,
            servo_base_duration_s=0.1,
            servo_duration_s=0.1,
            servo_targets={"front_left_hip": 10.0},
            servo_tolerance_deg=1.6,
            recorded_servo_residual_deg={"front_left_hip": 1.1},
        )
        plan = PlaybackPlan(path=None, events=[event], segments=[segment], final_time_s=0.1, total_steps=26)
        plan.plan_sha256 = plan_fingerprint(plan)
        adapter = LoadedAdapter()
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0))
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=1, current_wall_time_s=0.0)
        service.update(adapter, current_sim_time_s=0.1, current_sim_step=13, current_wall_time_s=0.1)
        self.assertTrue(service.active)
        self.assertEqual(service.progress.command_phase, "contact_residual_grace")
        service.update(adapter, current_sim_time_s=0.36, current_sim_step=44, current_wall_time_s=0.36)
        status = service.status_dict(current_sim_time_s=0.36, current_wall_time_s=0.36)
        self.assertEqual(status["stop_reason"], "complete")
        self.assertEqual(status["servo_residual_warning_count"], 1)
        self.assertEqual(adapter.applied[0]["servo_targets_deg"]["front_left_hip"], 10.0)

    def test_bounded_contact_residual_jitter_inside_effective_tolerance_completes(self) -> None:
        class JitteringLoadedAdapter:
            def __init__(self) -> None:
                self.joint_command_deg = {"front_left_hip": 0.0}
                self.motion_reference = SimpleNamespace(
                    servo_reference_velocity_deg_s=150.0,
                    servo_velocity_limit_deg_s=150.0,
                )
                self.reads = 0

            def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
                self.joint_command_deg.update(payload.get("servo_targets_deg", {}))
                return {}

            def get_actual_joint_state(self) -> dict[str, Any]:
                self.reads += 1
                # More than the 0.05 degree improvement epsilon from one
                # scheduler tick to the next, but always bounded by the
                # recording-derived 1.6 degree tolerance.
                actual = 11.30 if self.reads % 2 else 11.38
                return {"servos": {"front_left_hip": {"deg": actual}}}

            def command_to_actual_target_deg(self, _name: str, target: float) -> float:
                return float(target)

            def stop_wheels(self) -> None:
                pass

            def apply_commands_to_robot(self) -> None:
                pass

        event = PlaybackEvent(
            time_s=0.0,
            command="servo front_left_hip 10",
            source_step=23,
            source_step_id="step-23",
            commands_in_step=1,
            global_command_index=1,
            segment_index=0,
            channel="servo",
            servo_targets=(("front_left_hip", 10.0),),
        )
        segment = PlaybackSegment(
            segment_index=0,
            source_step=23,
            source_step_id="step-23",
            event_start_index=0,
            event_count=1,
            planned_start_s=0.0,
            planned_end_s=0.1,
            base_duration_s=0.1,
            servo_base_duration_s=0.1,
            servo_duration_s=0.1,
            servo_targets={"front_left_hip": 10.0},
            servo_tolerance_deg=1.6,
            recorded_servo_residual_deg={"front_left_hip": 1.1},
        )
        plan = PlaybackPlan(path=None, events=[event], segments=[segment], final_time_s=0.1, total_steps=23)
        plan.plan_sha256 = plan_fingerprint(plan)
        adapter = JitteringLoadedAdapter()
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0))
        for tick in range(0, 51):
            now = tick * 0.01
            service.update(adapter, current_sim_time_s=now, current_sim_step=tick, current_wall_time_s=now)
            if not service.active:
                break
        status = service.status_dict(current_sim_time_s=now, current_wall_time_s=now)
        self.assertEqual(status["stop_reason"], "complete")
        self.assertEqual(status["servo_residual_warning_count"], 1)
        self.assertEqual(
            status["last_servo_residual_warning"]["stability_basis"],
            "bounded_recent_contact_residual_window",
        )


class PlaybackHandshakeTest(unittest.TestCase):
    class Transport:
        def start_playback_plan(self, plan: PlaybackPlan, **payload: Any) -> None:
            self.payload = {"plan": plan, **payload}

        def stop_playback(self, **_payload: Any) -> None:
            pass

    def test_explicit_rejection_never_becomes_active(self) -> None:
        controller = SimpleNamespace(transport=self.Transport(), latest_sim_status={})
        manager = PlaybackManager(controller)
        before = empty_command_state()
        plan = plan_from_steps([
            make_step(
                index=1,
                step_type="recorded",
                duration=0.1,
                events=[make_event(0.0, "servo front_left_hip 5")],
                command_state_before=before,
                command_state_after=before,
            )
        ])
        self.assertTrue(manager.start_worker_plan(plan))
        self.assertFalse(manager.active)
        self.assertTrue(manager.start_requested)
        manager.sync_worker_status(
            {},
            operation_ack={
                "operation": "start_playback_plan",
                "request_id": manager.worker_request_id,
                "accepted": False,
                "rejection_reason": "worker busy",
            },
        )
        self.assertFalse(manager.active)
        self.assertFalse(manager.worker_managed)
        self.assertEqual(manager.progress.playback_state, "ERROR")
        self.assertIn("worker busy", manager.last_error)


if __name__ == "__main__":
    unittest.main()
