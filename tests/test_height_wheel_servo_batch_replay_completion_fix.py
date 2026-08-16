from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from completion_aware_segment import (
    MEASURED_ENDPOINT_V1,
    RECORDED_TIMELINE_OPEN_LOOP_V1,
)
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
from sim_robot_adapter import SimRobotAdapter, SimRobotAdapterConfig
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
        self.assertEqual(decoded.execution_semantics, MEASURED_ENDPOINT_V1)

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

    def test_fast_execution_semantics_is_payload_and_fingerprint_bound(self) -> None:
        before = empty_command_state()
        step = make_step(
            index=1,
            step_type="recorded",
            duration=0.1,
            events=[make_event(0.0, "servo front_left_hip 10")],
            command_state_before=before,
            command_state_after=before,
        )
        plan = plan_from_steps([step], profile="fast")
        self.assertEqual(plan.profile, "motion_only")
        self.assertEqual(
            plan.execution_semantics, RECORDED_TIMELINE_OPEN_LOOP_V1
        )
        payload = playback_plan_to_payload(plan)
        self.assertEqual(
            payload["execution_semantics"], RECORDED_TIMELINE_OPEN_LOOP_V1
        )
        decoded = playback_plan_from_payload(payload)
        self.assertEqual(decoded.execution_semantics, plan.execution_semantics)
        payload["execution_semantics"] = MEASURED_ENDPOINT_V1
        tampered = playback_plan_from_payload(payload)
        integrity = validate_plan_integrity(
            tampered, expected_plan_sha256=plan.plan_sha256
        )
        self.assertFalse(integrity["ok"])
        self.assertIn("plan sha mismatch", "; ".join(integrity["errors"]))

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
                return {
                    "servos": {
                        "front_left_hip": {
                            "deg": 11.3,
                            "velocity_deg_s": 0.0,
                        }
                    }
                }

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
                return {
                    "servos": {
                        "front_left_hip": {
                            "deg": actual,
                            "velocity_deg_s": 0.0,
                        }
                    }
                }

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
            "finite_grace_bounded_recorded_contact_tolerance",
        )
        self.assertEqual(
            status["last_servo_residual_warning"]["velocity_evidence_role"],
            "diagnostic_only",
        )

    def test_recent_contact_window_is_not_polluted_by_divergence_history(self) -> None:
        class RecoveringAdapter:
            def __init__(self) -> None:
                self.joint_command_deg = {"front_left_hip": 0.0}
                self.actual_deg = 12.9
                self.motion_reference = SimpleNamespace(
                    servo_reference_velocity_deg_s=150.0,
                    servo_velocity_limit_deg_s=150.0,
                )

            def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
                self.joint_command_deg.update(payload.get("servo_targets_deg", {}))
                return {}

            def get_actual_joint_state(self) -> dict[str, Any]:
                return {
                    "servos": {
                        "front_left_hip": {
                            "deg": self.actual_deg,
                            "velocity_deg_s": 0.0,
                        }
                    }
                }

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
        plan = PlaybackPlan(
            path=None,
            events=[event],
            segments=[segment],
            final_time_s=0.1,
            total_steps=23,
        )
        plan.plan_sha256 = plan_fingerprint(plan)
        adapter = RecoveringAdapter()
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        for tick in range(0, 51):
            now = tick * 0.01
            if now >= 0.25:
                adapter.actual_deg = 11.3
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
            if not service.active:
                break
        self.assertEqual(service.stop_reason, "complete")
        self.assertGreater(
            max(value for _at, value in service.segment_contact_error_history),
            service.contact_residual_hard_cap_deg - 0.2,
        )

    @staticmethod
    def _mixed_tracking_plan(*, wheel_duration_s: float) -> PlaybackPlan:
        events = [
            PlaybackEvent(
                time_s=0.0,
                command="servo front_left_hip 10",
                source_step=1,
                source_step_id="step-1",
                commands_in_step=2,
                global_command_index=1,
                segment_index=0,
                channel="servo",
                servo_targets=(("front_left_hip", 10.0),),
            ),
            PlaybackEvent(
                time_s=0.0,
                command="wheel fl 1",
                source_step=1,
                source_step_id="step-1",
                commands_in_step=2,
                global_command_index=2,
                segment_index=0,
                channel="wheel",
                wheel_applied_target_rad_s=(1.0,),
            ),
        ]
        segment = PlaybackSegment(
            segment_index=0,
            source_step=1,
            source_step_id="step-1",
            event_start_index=0,
            event_count=2,
            planned_start_s=0.0,
            planned_end_s=wheel_duration_s,
            base_duration_s=wheel_duration_s,
            servo_base_duration_s=0.1,
            servo_duration_s=0.1,
            servo_targets={"front_left_hip": 10.0},
            legacy_missing_endpoint=True,
            wheel_base_velocity={"front_left_ankle": 1.0},
            wheel_requested_velocity_rad_s={"front_left_ankle": 1.0},
            wheel_applied_target_rad_s={"front_left_ankle": 1.0},
            wheel_active_duration_s=wheel_duration_s,
        )
        plan = PlaybackPlan(
            path=None,
            events=events,
            segments=[segment],
            final_time_s=wheel_duration_s,
            total_steps=1,
        )
        plan.plan_sha256 = plan_fingerprint(plan)
        return plan

    class _TrackingAdapter:
        def __init__(
            self,
            *,
            converged: bool,
            saturated: bool,
            velocity: float,
            current_feedback_valid: bool = True,
            measured_deg: float = 10.2,
        ) -> None:
            self.joint_command_deg = {"front_left_hip": 0.0}
            self.motion_reference = SimpleNamespace(
                servo_reference_velocity_deg_s=150.0,
                servo_velocity_limit_deg_s=150.0,
            )
            self.converged = converged
            self.saturated = saturated
            self.velocity = velocity
            self.current_feedback_valid = current_feedback_valid
            self.measured_deg = measured_deg
            self.batches: list[dict[str, Any]] = []
            self.end_calls = 0
            self.end_attempts = 0
            self.tracking_active = False

        def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.batches.append(dict(payload))
            self.joint_command_deg.update(payload.get("servo_targets_deg", {}))
            return {}

        def begin_servo_tracking(self, _targets: Any) -> None:
            self.tracking_active = True

        def servo_tracking_completion_evidence(self, _targets: Any) -> dict[str, Any]:
            row_converged = (
                self.converged
                and not self.saturated
                and self.current_feedback_valid
            )
            return {
                "supported": True,
                "converged": row_converged,
                "joints": {
                    "front_left_hip": {
                        "stable_ticks": 2 if self.converged else 0,
                        "stable_evidence_valid": self.converged,
                        "feedback_phase": "HOLD" if self.converged else "TRACK",
                        "current_feedback_valid": self.current_feedback_valid,
                        "saturated": self.saturated,
                        "converged": row_converged,
                    }
                },
            }

        def end_servo_tracking(self, targets: Any) -> dict[str, Any]:
            self.end_attempts += 1
            evidence = self.servo_tracking_completion_evidence(targets)
            self.end_calls += 1
            self.tracking_active = False
            return {
                **evidence,
                "ended": True,
                "end_policy": "unconditional_segment_boundary",
                "convergence_diagnostic_only": True,
            }

        def get_actual_joint_state(self) -> dict[str, Any]:
            return {
                "servos": {
                    "front_left_hip": {
                        "deg": self.measured_deg,
                        "velocity_deg_s": self.velocity,
                    }
                }
            }

        def command_to_actual_target_deg(self, _name: str, target: float) -> float:
            return float(target)

        def stop_wheels(self) -> None:
            pass

        def apply_commands_to_robot(self) -> None:
            pass

    def test_mixed_long_wheel_segment_ends_saturated_bias_at_boundary(self) -> None:
        adapter = self._TrackingAdapter(
            converged=False,
            saturated=True,
            velocity=12.0,
        )
        service = SimTimePlaybackService()
        plan = self._mixed_tracking_plan(wheel_duration_s=0.5)
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        for tick in range(0, 81):
            now = tick * 0.01
            adapter.sim_steps = tick
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(adapter.end_calls, 1)
        self.assertEqual(adapter.end_attempts, 1)
        self.assertFalse(adapter.tracking_active)
        segment_trace = service.timing_trace["segments"][0]
        self.assertFalse(segment_trace["tracking_completion_deferred"])
        self.assertFalse(
            segment_trace["tracking_completion_evidence"]["converged"]
        )
        self.assertTrue(
            segment_trace["tracking_completion_evidence"]["joints"][
                "front_left_hip"
            ]["saturated"]
        )
        self.assertEqual(
            segment_trace["completion_decision"],
            "exact_completion",
        )
        dispatch_kinds = [
            row["dispatch_kind"]
            for row in service.timing_trace["motion_batches"]
        ]
        self.assertIn("final_safety_stop", dispatch_kinds)

    def test_mixed_long_wheel_segment_can_freeze_stable_unsaturated_bias(self) -> None:
        adapter = self._TrackingAdapter(
            converged=True,
            saturated=False,
            velocity=0.0,
        )
        service = SimTimePlaybackService()
        plan = self._mixed_tracking_plan(wheel_duration_s=0.5)
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        for tick in range(0, 61):
            now = tick * 0.01
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
            if not service.active:
                break
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(adapter.end_calls, 1)

    def test_wheel_only_segment_ignores_empty_servo_tracking_evidence(self) -> None:
        event = PlaybackEvent(
            time_s=0.0,
            command="wheel fl 1",
            source_step=1,
            source_step_id="step-1",
            commands_in_step=1,
            global_command_index=1,
            segment_index=0,
            channel="wheel",
            wheel_applied_target_rad_s=(1.0,),
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
            wheel_base_velocity={"front_left_ankle": 1.0},
            wheel_requested_velocity_rad_s={"front_left_ankle": 1.0},
            wheel_applied_target_rad_s={"front_left_ankle": 1.0},
            wheel_active_duration_s=0.1,
        )
        plan = PlaybackPlan(
            path=None,
            events=[event],
            segments=[segment],
            final_time_s=0.1,
            total_steps=1,
        )
        plan.plan_sha256 = plan_fingerprint(plan)
        adapter = self._TrackingAdapter(
            converged=False,
            saturated=True,
            velocity=12.0,
        )
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        for tick in range(0, 21):
            now = tick * 0.01
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
            if not service.active:
                break
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(adapter.end_calls, 0)

    def test_adapter_end_tracking_unconditionally_freezes_saturated_high_velocity_bias(self) -> None:
        adapter = object.__new__(SimRobotAdapter)
        adapter.config = SimRobotAdapterConfig()
        adapter.servo_nominal_target_reached = {"front_left_hip": True}
        adapter.servo_tracking_active = {"front_left_hip": True}
        adapter.servo_tracking_stable_ticks = {"front_left_hip": 3}
        adapter.servo_tracking_compensation_deg = {"front_left_hip": 10.0}
        # Four physics substeps can elapse in one rendered worker iteration;
        # age 3 is still the current four-tick sample-and-hold epoch.
        adapter.servo_tracking_feedback_tick = 104
        adapter.servo_tracking_feedback_evidence = {
            "front_left_hip": {
                "sample_tick": 100,
                "phase": "HOLD",
                "sample_valid": True,
                "actual_error_deg": 0.2,
                "measured_velocity_deg_s": 12.0,
                "damping_horizon_s": None,
            }
        }

        frozen_correction = adapter.servo_tracking_compensation_deg[
            "front_left_hip"
        ]
        accepted = adapter.end_servo_tracking(["front_left_hip"])
        self.assertTrue(accepted["ended"])
        self.assertFalse(adapter.servo_tracking_active["front_left_hip"])
        self.assertFalse(accepted["converged"])
        self.assertTrue(accepted["joints"]["front_left_hip"]["saturated"])
        self.assertFalse(
            accepted["joints"]["front_left_hip"]["sampled_velocity_valid"]
        )
        self.assertEqual(
            accepted["end_policy"], "unconditional_segment_boundary"
        )
        self.assertTrue(accepted["convergence_diagnostic_only"])
        self.assertEqual(
            adapter.servo_tracking_compensation_deg["front_left_hip"],
            frozen_correction,
        )

    def test_tracking_activity_does_not_leak_across_different_joint_segments(self) -> None:
        first = "front_left_hip"
        second = "front_right_hip"
        adapter = object.__new__(SimRobotAdapter)
        adapter.config = SimRobotAdapterConfig()
        adapter.servo_nominal_target_reached = {first: True, second: True}
        adapter.servo_tracking_active = {first: True, second: False}
        adapter.servo_tracking_stable_ticks = {first: 0, second: 0}
        adapter.servo_tracking_compensation_deg = {first: 1.25, second: -2.5}
        adapter.servo_tracking_feedback_tick = 0
        adapter.servo_tracking_feedback_evidence = {
            first: adapter._empty_servo_tracking_feedback_evidence(),
            second: adapter._empty_servo_tracking_feedback_evidence(),
        }

        first_result = adapter.end_servo_tracking([first])
        self.assertTrue(first_result["ended"])
        self.assertFalse(adapter.servo_tracking_active[first])
        self.assertEqual(adapter.servo_tracking_compensation_deg[first], 1.25)

        adapter.begin_servo_tracking([second])
        self.assertFalse(adapter.servo_tracking_active[first])
        self.assertTrue(adapter.servo_tracking_active[second])
        second_result = adapter.end_servo_tracking([second])
        self.assertTrue(second_result["ended"])
        self.assertFalse(adapter.servo_tracking_active[first])
        self.assertFalse(adapter.servo_tracking_active[second])
        self.assertEqual(adapter.servo_tracking_compensation_deg[second], -2.5)

    def test_live_step1_in_band_velocity_uses_historical_gain(self) -> None:
        logical_target_deg = -19.7794263347525
        measured_deg = -19.70489768
        measured_velocity_deg_s = 1.3848288583480188
        command_sign = -1.0
        actual_error_deg = logical_target_deg - measured_deg
        correction, phase, desired = (
            SimRobotAdapter._servo_tracking_correction_step(
                actual_error_deg=actual_error_deg,
                measured_velocity_deg_s=measured_velocity_deg_s,
                command_sign=command_sign,
                previous_correction_deg=0.0,
                previous_phase="DAMP",
                maximum_delta_deg=1.25,
                gain=8.0,
                limit_deg=10.0,
                physics_dt_s=1.0 / 120.0,
                feedback_interval_ticks=4,
            )
        )
        drive_target_deg = logical_target_deg + command_sign * correction
        expected_correction = (
            actual_error_deg / command_sign
        ) * 8.0
        expected_drive_target_deg = (
            logical_target_deg + command_sign * expected_correction
        )
        self.assertEqual(phase, "HOLD")
        self.assertAlmostEqual(correction, desired, places=12)
        self.assertAlmostEqual(correction, expected_correction, places=12)
        self.assertAlmostEqual(
            drive_target_deg, expected_drive_target_deg, places=12
        )

    def test_in_band_at_existing_velocity_limit_still_uses_gain_eight(self) -> None:
        logical_target_deg = -19.7794263347525
        measured_deg = -19.70489768
        command_sign = -1.0
        correction, phase, desired = (
            SimRobotAdapter._servo_tracking_correction_step(
                actual_error_deg=logical_target_deg - measured_deg,
                measured_velocity_deg_s=0.5,
                command_sign=command_sign,
                previous_correction_deg=-0.028,
                previous_phase="DAMP",
                maximum_delta_deg=1.25,
                gain=8.0,
                limit_deg=10.0,
                physics_dt_s=1.0 / 120.0,
                feedback_interval_ticks=4,
            )
        )
        drive_target_deg = logical_target_deg + command_sign * correction
        self.assertEqual(phase, "HOLD")
        self.assertAlmostEqual(correction, desired, places=12)
        self.assertAlmostEqual(
            drive_target_deg,
            logical_target_deg + 8.0 * (logical_target_deg - measured_deg),
            places=12,
        )

    def test_in_band_low_velocity_recomputes_historical_gain_bias(self) -> None:
        correction, phase, desired = (
            SimRobotAdapter._servo_tracking_correction_step(
                actual_error_deg=-0.2,
                measured_velocity_deg_s=0.5,
                command_sign=-1.0,
                previous_correction_deg=0.37,
                previous_phase="HOLD",
                maximum_delta_deg=1.25,
                gain=8.0,
                limit_deg=10.0,
                physics_dt_s=1.0 / 120.0,
                feedback_interval_ticks=4,
            )
        )
        self.assertEqual(phase, "HOLD")
        self.assertEqual(desired, 1.6)
        self.assertEqual(correction, 1.6)

    def test_in_band_correction_is_identical_for_all_velocity_evidence(self) -> None:
        results: list[tuple[float, str, float]] = []
        for velocity in (0.0, 1000.0, -1000.0, None):
            with self.subTest(velocity=velocity):
                result = SimRobotAdapter._servo_tracking_correction_step(
                    actual_error_deg=0.5,
                    measured_velocity_deg_s=velocity,
                    command_sign=1.0,
                    previous_correction_deg=0.0,
                    previous_phase="INVALID",
                    maximum_delta_deg=1.25,
                    gain=8.0,
                    limit_deg=10.0,
                    physics_dt_s=1.0 / 120.0,
                    feedback_interval_ticks=4,
                )
                results.append(result)
                correction, phase, desired = result
                self.assertEqual(correction, 1.25)
                self.assertEqual(phase, "TRACK")
                self.assertEqual(desired, 4.0)
        self.assertTrue(all(result == results[0] for result in results))

    def test_adapter_records_velocity_without_using_it_for_control(self) -> None:
        class Vector:
            def __init__(self, values: list[float]) -> None:
                self.values = values

            def detach(self) -> "Vector":
                return self

            def cpu(self) -> "Vector":
                return self

            def tolist(self) -> list[float]:
                return list(self.values)

        class Tensor:
            def __init__(self, values: list[float]) -> None:
                self.values = values

            def __getitem__(self, key: tuple[int, list[int]]) -> Vector:
                _row, ids = key
                return Vector([self.values[index] for index in ids])

        adapter = object.__new__(SimRobotAdapter)
        adapter.config = SimRobotAdapterConfig()
        adapter.motion_reference = SimpleNamespace(
            servo_reference_velocity_deg_s=150.0,
            servo_velocity_limit_deg_s=150.0,
        )
        adapter.servo_motion_enabled = True
        adapter.servo_joint_ids = list(range(len(SERVO_JOINT_NAMES)))
        adapter.servo_name_to_id = {
            name: index for index, name in enumerate(SERVO_JOINT_NAMES)
        }
        logical_target_deg = -19.7794263347525
        measured_deg = -19.70489768
        velocity_deg_s = 1.3848288583480188
        positions = [0.0] * len(SERVO_JOINT_NAMES)
        velocities = [0.0] * len(SERVO_JOINT_NAMES)
        rear_left_index = SERVO_JOINT_NAMES.index("rear_left_hip")
        positions[rear_left_index] = math.radians(measured_deg)
        velocities[rear_left_index] = math.radians(velocity_deg_s)
        adapter.robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=Tensor(positions),
                joint_vel=Tensor(velocities),
            )
        )
        adapter.standing_pose_deg = {
            name: 0.0 for name in SERVO_JOINT_NAMES
        }
        adapter.standing_pose_deg["rear_left_hip"] = (
            logical_target_deg + 19.6
        )
        adapter.joint_command_deg = {
            name: 0.0 for name in SERVO_JOINT_NAMES
        }
        adapter.joint_command_deg["rear_left_hip"] = 19.6
        adapter.servo_applied_command_deg = dict(adapter.joint_command_deg)
        adapter.servo_nominal_target_reached = {
            name: True for name in SERVO_JOINT_NAMES
        }
        adapter.servo_tracking_active = {
            name: name == "rear_left_hip" for name in SERVO_JOINT_NAMES
        }
        adapter.servo_tracking_stable_ticks = {
            name: 0 for name in SERVO_JOINT_NAMES
        }
        adapter.servo_tracking_compensation_deg = {
            name: 0.0 for name in SERVO_JOINT_NAMES
        }
        adapter.servo_tracking_feedback_evidence = {
            name: adapter._empty_servo_tracking_feedback_evidence()
            for name in SERVO_JOINT_NAMES
        }
        adapter.servo_tracking_feedback_tick = 0

        adapter._advance_servo_targets(1.0 / 120.0)
        first_correction = adapter.servo_tracking_compensation_deg[
            "rear_left_hip"
        ]
        first_evidence = adapter.servo_tracking_feedback_evidence[
            "rear_left_hip"
        ]
        self.assertEqual(first_evidence["phase"], "HOLD")
        self.assertEqual(adapter.servo_tracking_stable_ticks["rear_left_hip"], 1)
        expected_correction = (
            (logical_target_deg - measured_deg) / -1.0
        ) * 8.0
        self.assertAlmostEqual(
            first_correction, expected_correction, places=12
        )
        self.assertEqual(first_evidence["velocity_role"], "diagnostic_only")
        self.assertAlmostEqual(
            first_evidence["measured_velocity_deg_s"],
            velocity_deg_s,
            places=12,
        )

        adapter.robot.data.joint_vel.values[rear_left_index] = math.radians(-50.0)
        adapter.servo_tracking_feedback_tick = 4
        adapter._advance_servo_targets(1.0 / 120.0)
        self.assertAlmostEqual(
            adapter.servo_tracking_compensation_deg["rear_left_hip"],
            first_correction,
            places=12,
        )

        adapter.robot.data.joint_pos.values[rear_left_index] = math.radians(-19.73)
        adapter.robot.data.joint_vel.values[rear_left_index] = math.radians(0.3)
        adapter.servo_tracking_feedback_tick = 8
        adapter._advance_servo_targets(1.0 / 120.0)
        self.assertEqual(
            adapter.servo_tracking_feedback_evidence["rear_left_hip"][
                "phase"
            ],
            "HOLD",
        )
        self.assertAlmostEqual(
            adapter.servo_tracking_compensation_deg["rear_left_hip"],
            ((logical_target_deg - -19.73) / -1.0) * 8.0,
            places=12,
        )

        del adapter.robot.data.joint_vel
        adapter.robot.data.joint_pos.values[rear_left_index] = math.radians(-19.75)
        adapter.servo_tracking_feedback_tick = 12
        adapter._advance_servo_targets(1.0 / 120.0)
        missing_velocity_evidence = adapter.servo_tracking_feedback_evidence[
            "rear_left_hip"
        ]
        self.assertTrue(missing_velocity_evidence["sample_valid"])
        self.assertFalse(missing_velocity_evidence["velocity_sample_valid"])
        self.assertIsNone(missing_velocity_evidence["measured_velocity_deg_s"])
        self.assertAlmostEqual(
            adapter.servo_tracking_compensation_deg["rear_left_hip"],
            ((logical_target_deg - -19.75) / -1.0) * 8.0,
            places=12,
        )

    def test_in_band_high_velocity_keeps_gain_slew_and_clamp(self) -> None:
        correction, phase, desired = (
            SimRobotAdapter._servo_tracking_correction_step(
                actual_error_deg=0.1,
                measured_velocity_deg_s=1000.0,
                command_sign=1.0,
                previous_correction_deg=0.0,
                previous_phase="HOLD",
                maximum_delta_deg=1.25,
                gain=8.0,
                limit_deg=10.0,
                physics_dt_s=1.0 / 120.0,
                feedback_interval_ticks=4,
            )
        )
        self.assertEqual(phase, "HOLD")
        self.assertEqual(desired, 0.8)
        self.assertEqual(correction, 0.8)

    def test_tracking_band_exit_restores_existing_gain_clamp_and_slew(self) -> None:
        previous = -0.5682083347524998
        correction, phase, desired = SimRobotAdapter._servo_tracking_correction_step(
            actual_error_deg=0.80,
            measured_velocity_deg_s=1000.0,
            command_sign=-1.0,
            previous_correction_deg=previous,
            previous_phase="HOLD",
            maximum_delta_deg=1.25,
            gain=8.0,
            limit_deg=10.0,
            physics_dt_s=1.0 / 120.0,
            feedback_interval_ticks=4,
        )
        self.assertEqual(phase, "TRACK")
        self.assertAlmostEqual(desired, -6.4, places=12)
        self.assertAlmostEqual(correction, previous - 1.25, places=12)

    def test_service_ends_boundary_with_stale_convergence_evidence(self) -> None:
        adapter = self._TrackingAdapter(
            converged=True,
            saturated=False,
            velocity=0.0,
            current_feedback_valid=False,
        )
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self._mixed_tracking_plan(wheel_duration_s=0.5),
                current_sim_time_s=0.0,
                current_wall_time_s=0.0,
            )
        )
        for tick in range(0, 71):
            now = tick * 0.01
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(adapter.end_calls, 1)
        self.assertEqual(adapter.end_attempts, 1)
        trace = service.timing_trace["segments"][0]
        self.assertFalse(trace["tracking_completion_deferred"])
        self.assertFalse(
            trace["tracking_completion_evidence"]["joints"][
                "front_left_hip"
            ]["current_feedback_valid"]
        )

    def test_service_uses_reference_position_tolerance_not_tracking_band(self) -> None:
        adapter = self._TrackingAdapter(
            converged=True,
            saturated=False,
            velocity=0.0,
            measured_deg=10.8,
        )
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self._mixed_tracking_plan(wheel_duration_s=0.5),
                current_sim_time_s=0.0,
                current_wall_time_s=0.0,
            )
        )
        for tick in range(0, 71):
            now = tick * 0.01
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(adapter.end_calls, 1)

    def test_servo_position_completion_requires_every_finite_target_row(self) -> None:
        class StateAdapter:
            def __init__(self, servos: dict[str, dict[str, float]]) -> None:
                self.servos = servos

            def get_actual_joint_state(self) -> dict[str, Any]:
                return {"servos": self.servos}

            def command_to_actual_target_deg(
                self, _name: str, target: float
            ) -> float:
                return float(target)

        service = SimTimePlaybackService()
        targets = {"front_left_hip": 10.0, "front_right_hip": 20.0}
        missing_done, missing_errors = service._servo_targets_complete(
            StateAdapter({"front_left_hip": {"deg": 10.0}}), targets
        )
        self.assertFalse(missing_done)
        self.assertEqual(set(missing_errors), {"front_left_hip"})

        nonfinite_done, nonfinite_errors = service._servo_targets_complete(
            StateAdapter(
                {
                    "front_left_hip": {"deg": 10.0},
                    "front_right_hip": {"deg": float("nan")},
                }
            ),
            targets,
        )
        self.assertFalse(nonfinite_done)
        self.assertTrue(math.isnan(nonfinite_errors["front_right_hip"]))

    def test_v003_step1_three_waypoints_advance_on_position_with_high_velocity(
        self,
    ) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "saved_height_steps_fsm_reference_v2"
            / "height_050mm"
            / "versions"
            / "v003_20260805_224517_157723_manual"
            / "accepted_steps.jsonl"
        )
        with source.open("r", encoding="utf-8") as stream:
            step1 = json.loads(stream.readline())
        plan = plan_from_steps([step1], profile="fast", sequence_total_steps=1)
        plan.segments = plan.segments[:3]
        event_stop = sum(segment.event_count for segment in plan.segments)
        plan.events = plan.events[:event_stop]
        plan.final_time_s = plan.segments[-1].planned_end_s
        plan.plan_sha256 = plan_fingerprint(plan)

        class V003HighVelocityAdapter:
            def __init__(self) -> None:
                self.joint_command_deg = {"rear_left_hip": 0.0}
                self.motion_reference = SimpleNamespace(
                    servo_reference_velocity_deg_s=150.0,
                    servo_velocity_limit_deg_s=150.0,
                )
                self.sim_steps = 0
                self.batches: list[dict[str, Any]] = []
                self.tracking_active = False
                self.end_attempts = 0

            def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
                copied = dict(payload)
                self.batches.append(copied)
                servos = {
                    str(name): float(value)
                    for name, value in dict(
                        payload.get("servo_targets_deg", {}) or {}
                    ).items()
                }
                wheels = {
                    str(name): float(value)
                    for name, value in dict(
                        payload.get("wheel_targets_rad_s", {}) or {}
                    ).items()
                }
                self.joint_command_deg.update(servos)
                return {
                    "batch_id": payload["batch_id"],
                    "applied_sim_step": self.sim_steps,
                    "first_physics_step": self.sim_steps + 1,
                    "motion_start_skew_s": 0.0,
                    "servo_targets_applied": servos,
                    "wheel_targets_applied": wheels,
                    "servo_applied": bool(servos),
                    "wheel_applied": bool(wheels),
                    "recording_metadata": dict(
                        payload.get("recording_metadata", {}) or {}
                    ),
                }

            def begin_servo_tracking(self, _targets: Any) -> None:
                self.tracking_active = True

            def servo_tracking_completion_evidence(
                self, _targets: Any
            ) -> dict[str, Any]:
                velocity_by_target = {19.6: 6.16, 25.9: -7.14, 28.1: 1.38}
                target = float(self.joint_command_deg["rear_left_hip"])
                velocity = velocity_by_target[target]
                return {
                    "supported": True,
                    "converged": False,
                    "joints": {
                        "rear_left_hip": {
                            "stable_ticks": 0,
                            "stable_evidence_valid": False,
                            "feedback_phase": "TRACK",
                            "current_feedback_valid": True,
                            "sampled_velocity_deg_s": velocity,
                            "saturated": False,
                            "converged": False,
                        }
                    },
                }

            def end_servo_tracking(self, targets: Any) -> dict[str, Any]:
                self.end_attempts += 1
                self.tracking_active = False
                return {
                    **self.servo_tracking_completion_evidence(targets),
                    "ended": True,
                    "end_policy": "unconditional_segment_boundary",
                    "convergence_diagnostic_only": True,
                }

            def get_actual_joint_state(self) -> dict[str, Any]:
                target = float(self.joint_command_deg["rear_left_hip"])
                velocity = {19.6: 6.16, 25.9: -7.14, 28.1: 1.38}[target]
                return {
                    "servos": {
                        "rear_left_hip": {
                            "deg": target + 0.07453,
                            "velocity_deg_s": velocity,
                        }
                    }
                }

            def command_to_actual_target_deg(
                self, _name: str, target: float
            ) -> float:
                return float(target)

            def stop_wheels(self) -> None:
                pass

            def apply_commands_to_robot(self) -> None:
                pass

        adapter = V003HighVelocityAdapter()
        service = SimTimePlaybackService()
        self.assertEqual(
            [segment.servo_targets["rear_left_hip"] for segment in plan.segments],
            [19.6, 25.9, 28.1],
        )
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        for sim_step, now in ((0, 0.0), (14, 0.14), (19, 0.19), (21, 0.21)):
            adapter.sim_steps = sim_step
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=sim_step,
                current_wall_time_s=now,
            )

        source_batches = [
            row
            for row in service.timing_trace["motion_batches"]
            if row["dispatch_kind"] == "source_segment_start"
        ]
        self.assertEqual(
            [row["servo_targets_deg"]["rear_left_hip"] for row in source_batches],
            [19.6, 25.9, 28.1],
        )
        source_ticks = [row["scheduler_sim_step"] for row in source_batches]
        self.assertEqual(source_ticks, [0, 14, 19])
        self.assertEqual(len(source_ticks), len(set(source_ticks)))
        self.assertTrue(
            all(
                row["first_physics_step"] == row["scheduler_sim_step"] + 1
                for row in source_batches
            )
        )
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(service.events_sent, 3)
        self.assertEqual(adapter.end_attempts, 3)
        self.assertFalse(adapter.tracking_active)
        self.assertTrue(
            all(
                row["completion_decision"]
                == "recorded_timeline_open_loop_complete"
                for row in service.timing_trace["segments"]
            )
        )
        self.assertTrue(
            all(
                row["tracking_completion_deferred"] is False
                for row in service.timing_trace["segments"]
            )
        )

    def test_v003_step24_source_freezes_bias_despite_high_servo_velocity(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "saved_height_steps_fsm_reference_v2"
            / "height_050mm"
            / "versions"
            / "v003_20260805_224517_157723_manual"
            / "accepted_steps.jsonl"
        )
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
        ]
        step24 = rows[23]["sim_state_after"]
        target_servos = step24["target_joint_state"]["servos"]
        actual_servos = step24["actual_joint_state"]["servos"]

        self.assertEqual(set(target_servos), set(SERVO_JOINT_NAMES))
        self.assertTrue(
            all(
                abs(float(actual_servos[name]["velocity_deg_s"])) > 0.5
                for name in SERVO_JOINT_NAMES
            )
        )
        self.assertTrue(
            all(
                not math.isclose(
                    float(target_servos[name]["tracking_compensation_deg"]),
                    0.0,
                    abs_tol=1.0e-12,
                )
                for name in SERVO_JOINT_NAMES
            )
        )

    def test_v003_all_112_segments_end_tracking_at_formal_boundaries(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "saved_height_steps_fsm_reference_v2"
            / "height_050mm"
            / "versions"
            / "v003_20260805_224517_157723_manual"
            / "accepted_steps.jsonl"
        )
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
        ]
        plan = plan_from_steps(rows, profile="fast", sequence_total_steps=24)
        self.assertEqual(len(plan.events), 160)
        self.assertEqual(len(plan.segments), 112)
        self.assertEqual(
            sum(bool(segment.servo_targets) for segment in plan.segments), 96
        )

        class FormalBoundaryAdapter:
            def __init__(self) -> None:
                self.joint_command_deg = {
                    name: 0.0 for name in SERVO_JOINT_NAMES
                }
                self.wheel_speeds = {name: 0.0 for name in WHEEL_JOINT_NAMES}
                self.motion_reference = SimpleNamespace(
                    servo_reference_velocity_deg_s=150.0,
                    servo_velocity_limit_deg_s=150.0,
                )
                self.sim_steps = 0
                self.tracking_active = {
                    name: False for name in SERVO_JOINT_NAMES
                }
                self.end_attempts = 0

            def apply_motion_batch(
                self, payload: dict[str, Any]
            ) -> dict[str, Any]:
                servos = {
                    str(name): float(value)
                    for name, value in dict(
                        payload.get("servo_targets_deg", {}) or {}
                    ).items()
                }
                wheels = {
                    str(name): float(value)
                    for name, value in dict(
                        payload.get("wheel_targets_rad_s", {}) or {}
                    ).items()
                }
                self.joint_command_deg.update(servos)
                self.wheel_speeds.update(wheels)
                return {
                    "batch_id": payload["batch_id"],
                    "applied_sim_step": self.sim_steps,
                    "first_physics_step": self.sim_steps + 1,
                    "motion_start_skew_s": 0.0,
                    "servo_targets_applied": servos,
                    "wheel_targets_applied": wheels,
                    "servo_applied": bool(servos),
                    "wheel_applied": bool(wheels),
                    "recording_metadata": dict(
                        payload.get("recording_metadata", {}) or {}
                    ),
                }

            def begin_servo_tracking(self, targets: Any) -> None:
                for name in targets:
                    self.tracking_active[str(name)] = True

            def servo_tracking_completion_evidence(
                self, targets: Any
            ) -> dict[str, Any]:
                return {
                    "supported": True,
                    "converged": False,
                    "completion_role": "diagnostic_only",
                    "joints": {
                        str(name): {
                            "stable_ticks": 0,
                            "stable_evidence_valid": False,
                            "feedback_phase": "TRACK",
                            "current_feedback_valid": True,
                            "sampled_velocity_deg_s": 12.0,
                            "saturated": False,
                            "converged": False,
                        }
                        for name in targets
                    },
                }

            def end_servo_tracking(self, targets: Any) -> dict[str, Any]:
                names = [str(name) for name in targets]
                evidence = self.servo_tracking_completion_evidence(names)
                self.end_attempts += 1
                for name in names:
                    self.tracking_active[name] = False
                return {
                    **evidence,
                    "ended": True,
                    "end_policy": "unconditional_segment_boundary",
                    "convergence_diagnostic_only": True,
                }

            def get_actual_joint_state(self) -> dict[str, Any]:
                return {
                    "servos": {
                        name: {
                            "deg": float(target),
                            "velocity_deg_s": 12.0,
                        }
                        for name, target in self.joint_command_deg.items()
                    }
                }

            def command_to_actual_target_deg(
                self, _name: str, target: float
            ) -> float:
                return float(target)

            def stop_wheels(self) -> None:
                self.wheel_speeds = {
                    name: 0.0 for name in WHEEL_JOINT_NAMES
                }

            def apply_commands_to_robot(self) -> None:
                pass

        adapter = FormalBoundaryAdapter()
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        self.assertTrue(
            service.apply_playback_start_boundary(
                adapter, current_sim_time_s=0.0, current_sim_step=0
            )
        )
        for tick in range(1, 20_000):
            now = tick / 120.0
            adapter.sim_steps = tick
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
            if not service.active:
                break

        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(len(service.timing_trace["segments"]), 112)
        servo_traces = [
            row
            for row in service.timing_trace["segments"]
            if row["servo_target"]
        ]
        self.assertEqual(len(servo_traces), 96)
        self.assertEqual(adapter.end_attempts, 96)
        self.assertTrue(
            all(
                row["tracking_completion_evidence"]["ended"] is True
                and row["tracking_completion_deferred"] is False
                for row in servo_traces
            )
        )
        self.assertFalse(any(adapter.tracking_active.values()))
        motion_batches = service.timing_trace["motion_batches"]
        self.assertEqual(
            sum(
                row["dispatch_kind"] == "source_segment_start"
                for row in motion_batches
            ),
            112,
        )
        self.assertIn(
            "playback_start_boundary",
            {row["dispatch_kind"] for row in motion_batches},
        )
        self.assertIn(
            "final_safety_stop",
            {row["dispatch_kind"] for row in motion_batches},
        )
        self.assertTrue(
            all(
                row["ack_valid"] is True
                for row in motion_batches
            )
        )

    def test_segment_104_boundary_freeze_completes_and_next_segment_starts(
        self,
    ) -> None:
        events = [
            PlaybackEvent(
                time_s=0.0,
                command="servo front_left_hip 10",
                source_step=104,
                source_step_id="step-104",
                global_command_index=1,
                segment_index=104,
                channel="servo",
                servo_targets=(("front_left_hip", 10.0),),
            ),
            PlaybackEvent(
                time_s=0.1,
                command="servo front_left_hip 20",
                source_step=105,
                source_step_id="step-105",
                global_command_index=2,
                segment_index=105,
                channel="servo",
                servo_targets=(("front_left_hip", 20.0),),
            ),
        ]
        segments = [
            PlaybackSegment(
                segment_index=104,
                source_step=104,
                source_step_id="step-104",
                event_start_index=0,
                event_count=1,
                planned_start_s=0.0,
                planned_end_s=0.1,
                base_duration_s=0.1,
                servo_base_duration_s=0.1,
                servo_duration_s=0.1,
                servo_targets={"front_left_hip": 10.0},
                legacy_missing_endpoint=True,
            ),
            PlaybackSegment(
                segment_index=105,
                source_step=105,
                source_step_id="step-105",
                event_start_index=1,
                event_count=1,
                planned_start_s=0.1,
                planned_end_s=0.2,
                base_duration_s=0.1,
                servo_base_duration_s=0.1,
                servo_duration_s=0.1,
                servo_targets={"front_left_hip": 20.0},
                legacy_missing_endpoint=True,
            ),
        ]
        plan = PlaybackPlan(
            path=None,
            events=events,
            segments=segments,
            final_time_s=0.2,
            total_steps=105,
        )
        plan.plan_sha256 = plan_fingerprint(plan)
        adapter = self._TrackingAdapter(
            converged=False,
            saturated=True,
            velocity=12.0,
            measured_deg=10.0,
        )
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=0,
            current_wall_time_s=0.0,
        )
        service.update(
            adapter,
            current_sim_time_s=0.07,
            current_sim_step=7,
            current_wall_time_s=0.07,
        )

        self.assertTrue(service.active)
        self.assertEqual(service.segment_index, 1)
        self.assertEqual(service.events_sent, 2)
        self.assertEqual(adapter.joint_command_deg["front_left_hip"], 20.0)
        self.assertEqual(adapter.end_attempts, 1)
        self.assertEqual(adapter.end_calls, 1)
        self.assertTrue(adapter.tracking_active)
        trace = service.timing_trace["segments"][0]
        self.assertEqual(trace["segment_index"], 104)
        self.assertFalse(trace["tracking_completion_deferred"])
        self.assertTrue(
            trace["tracking_completion_evidence"]["joints"][
                "front_left_hip"
            ]["saturated"]
        )
        self.assertEqual(
            trace["completion_decision"],
            "exact_completion",
        )

    def test_malformed_tracking_end_result_fails_closed(self) -> None:
        adapter = self._TrackingAdapter(
            converged=False,
            saturated=False,
            velocity=6.0,
            measured_deg=10.0,
        )
        adapter.end_servo_tracking = lambda _targets: None  # type: ignore[method-assign]
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self._mixed_tracking_plan(wheel_duration_s=0.1),
                current_sim_time_s=0.0,
                current_wall_time_s=0.0,
            )
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=0,
            current_wall_time_s=0.0,
        )
        service.update(
            adapter,
            current_sim_time_s=0.11,
            current_sim_step=11,
            current_wall_time_s=0.11,
        )
        self.assertFalse(service.active)
        self.assertEqual(
            service.stop_reason, "servo_tracking_completion_invalid"
        )
        self.assertEqual(service.segment_index, 0)
        self.assertEqual(service.timing_trace.get("segments", []), [])

    def test_hard_liveness_classifies_fast_nonconvergent_servo_unstable(self) -> None:
        adapter = self._TrackingAdapter(
            converged=False,
            saturated=False,
            velocity=6.0,
            measured_deg=12.0,
        )
        service = SimTimePlaybackService()
        self.assertEqual(service.servo_tracking_hard_liveness_s, 2.25)
        self.assertTrue(
            service.start_plan(
                self._mixed_tracking_plan(wheel_duration_s=0.5),
                current_sim_time_s=0.0,
                current_wall_time_s=0.0,
            )
        )
        for tick, now in enumerate((0.0, 0.5, 1.6, 2.36)):
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "actuator_unstable")
        self.assertIn("hard_liveness_bound_s=2.250000", service.last_error)
        self.assertIn("worst_error_joint=front_left_hip", service.last_error)
        self.assertIn("fastest_joint=front_left_hip", service.last_error)
        self.assertIn("fastest_joint_velocity_deg_s=6.0", service.last_error)

    def test_hard_liveness_classifies_nearzero_unconverged_servo_limit(self) -> None:
        adapter = self._TrackingAdapter(
            converged=False,
            saturated=True,
            velocity=0.0,
            measured_deg=13.0,
        )
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self._mixed_tracking_plan(wheel_duration_s=0.5),
                current_sim_time_s=0.0,
                current_wall_time_s=0.0,
            )
        )
        # Keep producing >0.05-degree improvements so the existing 0.75-second
        # stall path does not fire; the derived 2.25-second bound must still do
        # so deterministically.
        for tick, (now, measured) in enumerate(
            ((0.0, 13.0), (0.8, 12.3), (1.6, 11.7), (2.36, 11.1))
        ):
            adapter.measured_deg = measured
            service.update(
                adapter,
                current_sim_time_s=now,
                current_sim_step=tick,
                current_wall_time_s=now,
            )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "actuator_limit")
        self.assertIn("hard_liveness_bound_s=2.250000", service.last_error)


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
