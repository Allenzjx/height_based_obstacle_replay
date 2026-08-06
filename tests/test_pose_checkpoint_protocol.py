from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from playback import PlaybackEvent, PlaybackManager, PlaybackPlan, PlaybackSegment, SimTimePlaybackService, plan_fingerprint
from sim_state_validation import (
    COMMAND_ONLY,
    FULL_VALID,
    PLACEHOLDER_NO_SIM,
    validate_full_sim_pose_state,
    verify_restored_full_sim_pose,
)
from sim_ui_controller import MODE_RECORDING_PREPARING, MODE_RECORDING_STEP
from sim_robot_adapter import SimRobotAdapter
from tests.test_selected_step_previous_saved_state import asynchronous_controller, command_state, sim_state
from tests.test_sim_time_playback_service import FakePlaybackController, _worker_plan


class PoseClassifierTest(unittest.TestCase):
    def test_full_command_only_and_no_sim_are_distinct(self) -> None:
        full = sim_state(12.0, 1.25)
        self.assertEqual(validate_full_sim_pose_state(full)["classification"], FULL_VALID)
        self.assertEqual(
            validate_full_sim_pose_state({"command_state": command_state(1.0)})["classification"],
            COMMAND_ONLY,
        )
        placeholder = {
            "capture_source": "NullSimRobotAdapter",
            "command_state": command_state(1.0),
            "root_pose": None,
            "root_velocity": None,
            "joint_pos": None,
            "joint_vel": None,
        }
        self.assertEqual(validate_full_sim_pose_state(placeholder)["classification"], PLACEHOLDER_NO_SIM)

    def test_joint_name_reorder_is_explicit_and_pose_tolerances_are_enforced(self) -> None:
        expected = sim_state(12.0, 1.25)
        names = list(expected["joint_names"])
        measured = copy.deepcopy(expected)
        measured["joint_names"] = list(reversed(names))
        measured["joint_pos"] = [list(reversed(expected["joint_pos"][0]))]
        measured["joint_vel"] = [list(reversed(expected["joint_vel"][0]))]
        validation = validate_full_sim_pose_state(measured, names)
        self.assertTrue(validation["valid"])
        self.assertNotEqual(validation["joint_reorder_indices"], list(range(len(names))))
        self.assertTrue(verify_restored_full_sim_pose(expected, measured, names)["verified"])
        measured["root_pose"][0][0] += 0.006
        result = verify_restored_full_sim_pose(expected, measured, names)
        self.assertFalse(result["verified"])
        self.assertGreater(result["root_position_error_m"], result["root_position_tolerance_m"])

    def test_missing_pose_fields_and_nan_are_invalid(self) -> None:
        for field, value in (
            ("root_pose", None),
            ("joint_pos", None),
            ("joint_names", []),
            ("joint_vel", [[float("nan")] * 12]),
        ):
            state = sim_state(1.0, 0.0)
            state[field] = value
            result = validate_full_sim_pose_state(state)
            self.assertFalse(result["valid"], field)
            self.assertEqual(result["classification"], "INVALID", field)


class RecordingCheckpointTransactionTest(unittest.TestCase):
    def test_recording_does_not_start_until_matching_full_worker_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            controller.start_step_recording()
            self.assertEqual(controller.mode, MODE_RECORDING_PREPARING)
            self.assertIsNone(controller.record_command_state_before)
            pending = dict(controller.pending_full_state_capture or {})
            self.assertTrue(pending.get("request_id"))

            fake.latest_detailed_status = {
                **fake.latest_status,
                "detail_status": True,
                "state_capture_request_id": "stale-request",
                "state_capture_purpose": "recording_start",
                "state_capture_worker_session_id": "worker-test",
                "robot_joint_names": list(SERVO_JOINT_NAMES) + list(WHEEL_JOINT_NAMES),
                "sim_state": sim_state(10.0, 2.0),
            }
            controller.update()
            self.assertEqual(controller.mode, MODE_RECORDING_PREPARING)

            fake.latest_detailed_status.update(state_capture_request_id=pending["request_id"])
            controller.update()
            self.assertEqual(controller.mode, MODE_RECORDING_STEP)
            self.assertEqual(
                controller.recording_capture_metadata["start"]["validation"]["classification"],
                FULL_VALID,
            )
            self.assertEqual(
                controller.recording_capture_metadata["start"]["worker_session_id"],
                "worker-test",
            )

    def test_start_capture_timeout_creates_no_placeholder_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, _fake = asynchronous_controller(Path(tmp))
            controller.start_step_recording()
            self.assertIsNotNone(controller.pending_full_state_capture)
            controller.pending_full_state_capture["deadline"] = 0.0
            controller.update()
            self.assertIsNone(controller.pending_step)
            self.assertIsNone(controller.record_sim_state_before)
            self.assertTrue(controller.operation.idle)
            self.assertIn("Recording blocked", controller.detail_text)
            controller.start_step_recording()
            self.assertEqual(controller.mode, MODE_RECORDING_PREPARING)
            self.assertIsNotNone(controller.pending_full_state_capture)


class AdapterRestoreOrderTest(unittest.TestCase):
    def test_reset_precedes_all_saved_state_writes_and_targets_follow_joint_state(self) -> None:
        source = inspect.getsource(SimRobotAdapter.restore_sim_state)
        reset_at = source.index("self.robot.reset()")
        root_at = source.index("write_root_pose_to_sim")
        velocity_at = source.index("write_root_velocity_to_sim")
        joint_at = source.index("write_joint_state_to_sim")
        targets_at = source.index("self.apply_commands_to_robot()")
        write_data_at = source.index("self.robot.write_data_to_sim()")
        self.assertLess(reset_at, root_at)
        self.assertLess(root_at, velocity_at)
        self.assertLess(velocity_at, joint_at)
        self.assertLess(joint_at, targets_at)
        self.assertLess(targets_at, write_data_at)


class MixedChannelCompletionTest(unittest.TestCase):
    class Adapter:
        def __init__(self, *, measured_deg: float, velocity_deg_s: float) -> None:
            self.measured_deg = measured_deg
            self.velocity_deg_s = velocity_deg_s
            self.commands: list[str] = []
            self.stop_count = 0

        def handle_command(self, message) -> None:
            self.commands.append(str(message.text))

        def stop_wheels(self) -> None:
            self.stop_count += 1

        def apply_commands_to_robot(self) -> None:
            return None

        def command_to_actual_target_deg(self, _name: str, target: float) -> float:
            return float(target)

        def get_actual_joint_state(self) -> dict:
            return {
                "servos": {
                    "front_left_hip": {
                        "deg": self.measured_deg,
                        "velocity_deg_s": self.velocity_deg_s,
                    }
                }
            }

    @staticmethod
    def plan(*, wheel_duration_s: float = 1.0) -> PlaybackPlan:
        plan = PlaybackPlan(
            path=None,
            events=[
                PlaybackEvent(
                    0.0,
                    "servo front_left_hip 10; wheel all 0.5",
                    source_step=7,
                    segment_index=0,
                )
            ],
            segments=[
                PlaybackSegment(
                    segment_index=0,
                    source_step=7,
                    source_step_id="step-7",
                    event_start_index=0,
                    event_count=1,
                    planned_start_s=0.0,
                    planned_end_s=wheel_duration_s,
                    base_duration_s=wheel_duration_s,
                    servo_base_duration_s=0.1,
                    servo_duration_s=0.1,
                    servo_targets={"front_left_hip": 10.0},
                    wheel_active_duration_s=wheel_duration_s,
                    servo_tolerance_deg=1.0,
                    legacy_missing_endpoint=True,
                )
            ],
            final_time_s=wheel_duration_s,
            total_steps=7,
        )
        plan.plan_sha256 = plan_fingerprint(plan)
        return plan

    def test_servo_inside_tolerance_cannot_false_stall_while_wheel_timer_is_active(self) -> None:
        adapter = self.Adapter(measured_deg=10.2, velocity_deg_s=0.0)
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0))
        for step, sim_time in enumerate((0.0, 0.1, 0.5, 0.95, 1.0, 1.01)):
            service.update(adapter, current_sim_time_s=sim_time, current_sim_step=step, current_wall_time_s=sim_time)
        self.assertEqual(service.stop_reason, "complete")
        self.assertEqual(service.last_error, "")

    def test_stall_requires_near_zero_joint_velocity(self) -> None:
        adapter = self.Adapter(measured_deg=15.0, velocity_deg_s=4.0)
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(wheel_duration_s=2.0), current_sim_time_s=0.0, current_wall_time_s=0.0))
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=0, current_wall_time_s=0.0)
        service.update(adapter, current_sim_time_s=1.0, current_sim_step=1, current_wall_time_s=1.0)
        self.assertTrue(service.active)
        self.assertEqual(service.last_error, "")
        adapter.velocity_deg_s = 0.0
        service.update(adapter, current_sim_time_s=1.9, current_sim_step=2, current_wall_time_s=1.9)
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "actuator_limit")


class PlaybackFailureVisibilityTest(unittest.TestCase):
    def test_worker_failure_exposes_structured_step_segment_and_joint_detail(self) -> None:
        controller = FakePlaybackController()
        manager = PlaybackManager(controller)
        plan = _worker_plan()
        self.assertTrue(manager.start_worker_plan(plan))
        ack = {
            "operation": "start_playback_plan",
            "request_id": manager.worker_request_id,
            "accepted": True,
            "plan_id": manager.worker_plan_id,
            "plan_sha256": plan.plan_sha256,
            "event_count": len(plan.events),
            "segment_count": len(plan.segments),
            "worker_session_id": "session-visible-failure",
        }
        manager.sync_worker_status(
            {
                "active": False,
                "started": True,
                "plan_id": manager.worker_plan_id,
                "request_id": manager.worker_request_id,
                "worker_session_id": "session-visible-failure",
                "first_command_applied": True,
                "stop_reason": "actuator_limit",
                "last_error": (
                    "actuator_limit: step=14 segment=56 joint=rear_left_hip "
                    "requested_command_deg=1.6 measured_actual_deg=-1.63 "
                    "error_deg=0.14 tolerance_deg=1.0"
                ),
                "segment_index": 56,
                "progress_detail": {"current_step_index": 14},
            },
            operation_ack=ack,
        )
        text = manager.progress.status_text
        for expected in (
            "Playback stopped at Step 14 / Segment 56",
            "Reason: actuator_limit",
            "Joint: rear_left_hip",
            "Requested: 1.6",
            "Actual: -1.63",
            "Error: 0.14",
            "Tolerance: 1.0",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
