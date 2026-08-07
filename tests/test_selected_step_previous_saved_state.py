from __future__ import annotations

import argparse
import copy
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from operation_coordinator import OperationState
from playback import plan_from_steps
from sequence_model import empty_command_state, make_event, make_step
from sim_ui_controller import HeightReplayController, MODE_TEST


def make_args(store_root: Path, *, no_sim: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=no_sim,
        store_root=store_root,
        height_mm=50,
        robot_usd="C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd",
        save_usd="C:/robotics_sim/wlr_robot/usd/wlr_robot_height_replay_env.usd",
        physics_dt=1.0 / 120.0,
        render_interval=2,
        device="cuda:0",
        max_wheel_speed_rad_s=2.0943951023931953,
        default_wheel_speed_rad_s=0.5235987755982988,
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
        sim_launch_mode="disabled" if no_sim else "subprocess",
        restore_step_start_state_before_selected_playback=True,
        restore_full_sim_pose_if_available=True,
        fallback_to_command_state_before=True,
        playback_pre_step_settle_s=0.30,
        respawn_play_settle_s=0.30,
    )


def command_state(hip: float) -> dict[str, Any]:
    state = empty_command_state()
    state["servos"]["front_left_hip"] = float(hip)
    return state


def sim_state(hip: float, root_x: float) -> dict[str, Any]:
    names = list(SERVO_JOINT_NAMES) + list(WHEEL_JOINT_NAMES)
    command = command_state(hip)
    servo_positions = {
        name: (float(hip) if name == "front_left_hip" else 0.0)
        for name in SERVO_JOINT_NAMES
    }
    return {
        "capture_source": "FakeFullIsaacAdapter",
        "pose_restore_eligible": True,
        "command_state": command,
        "target_joint_state": {
            "servos": {
                name: {"target_actual_deg": value}
                for name, value in servo_positions.items()
            },
            "wheels": {name: {"target_rad_s": 0.0} for name in WHEEL_JOINT_NAMES},
        },
        "actual_joint_state": {
            "servos": {
                name: {"deg": value, "velocity_deg_s": 0.0}
                for name, value in servo_positions.items()
            },
            "wheels": {name: {"rad_s": 0.0} for name in WHEEL_JOINT_NAMES},
        },
        "root_pose": [[root_x, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]],
        "root_velocity": [[0.0] * 6],
        "joint_pos": [[servo_positions[name] / 57.29577951308232 for name in SERVO_JOINT_NAMES] + [0.0] * len(WHEEL_JOINT_NAMES)],
        "joint_vel": [[0.0] * len(names)],
        "joint_names": names,
    }


def step(index: int, before: float, target: float) -> dict[str, Any]:
    return make_step(
        index=index,
        step_type="test",
        duration=0.2,
        events=[make_event(0.0, f"servo front_left_hip {target}")],
        command_state_before=command_state(before),
        command_state_after=command_state(target),
        name=f"step_{index:03d}",
    )


def controller_with_three_steps(root: Path) -> HeightReplayController:
    controller = HeightReplayController(make_args(root))
    controller.manager.steps = [step(1, 0.0, 10.0), step(2, 10.0, 20.0), step(3, 20.0, 30.0)]
    for index, row in enumerate(controller.manager.steps, start=1):
        row["sim_state_before"] = sim_state(float((index - 1) * 10), float(index - 1))
        row["sim_state_after"] = sim_state(float(index * 10), float(index))
    controller.manager.revision += 1
    # --no-sim tests still need a pose-capable in-memory adapter to exercise
    # the strict restore verifier; NullSim placeholders are intentionally not
    # FULL_VALID checkpoints.
    adapter = controller.transport.adapter
    original_restore = adapter.restore_sim_state
    restored_state: dict[str, Any] = {"value": sim_state(0.0, 0.0)}

    def restore_and_remember(state: dict[str, Any]) -> None:
        restored_state["value"] = copy.deepcopy(state)
        original_restore(state)

    adapter.restore_sim_state = restore_and_remember  # type: ignore[method-assign]
    adapter.capture_sim_state = lambda: copy.deepcopy(restored_state["value"])  # type: ignore[method-assign]
    return controller


class FakeSimClient:
    def __init__(self) -> None:
        self.connected = True
        self.wheel_generation = 0
        self.last_status_time = time.monotonic()
        self.latest_status: dict[str, Any] = {
            "type": "status",
            "phase": "running",
            "ready": True,
            "runtime_ready": True,
            "restore_count": 0,
            "last_restore_result": "",
            "last_restore_error": "",
            "last_restore_request_id": "",
            "worker_playback": {"active": False},
            "wheel_command": {"generation": 0},
            "worker_session_id": "worker-test",
            "robot_joint_names": list(SERVO_JOINT_NAMES) + list(WHEEL_JOINT_NAMES),
        }
        self.latest_detailed_status: dict[str, Any] = {}
        self.restore_calls: list[dict[str, Any]] = []
        self.start_calls: list[dict[str, Any]] = []
        self.state_requests: list[bool] = []
        self.last_state_request_id = ""
        self.last_state_request_purpose = ""
        self.stop_calls = 0

    def poll(self) -> list[dict[str, Any]]:
        self.last_status_time = time.monotonic()
        return []

    def status(self) -> dict[str, Any]:
        status = dict(self.latest_status)
        if (
            str(status.get("last_restore_result", "") or "") == "ok"
            and str(status.get("last_restore_request_id", "") or "")
        ):
            status["last_restore_verification"] = {
                "request_id": str(status["last_restore_request_id"]),
                "verified": True,
            }
        return status

    def restore_sim_state(self, state: dict[str, Any], *, request_id: str = "") -> str:
        self.restore_calls.append({"sim_state": copy.deepcopy(state), "request_id": request_id})
        return request_id

    def request_state(
        self,
        *,
        detailed: bool = False,
        request_id: str = "",
        purpose: str = "",
    ) -> str:
        self.state_requests.append(bool(detailed))
        self.last_state_request_id = request_id
        self.last_state_request_purpose = purpose
        if self.latest_detailed_status:
            self.latest_detailed_status.update(
                state_capture_request_id=request_id,
                state_capture_purpose=purpose,
                state_capture_worker_session_id="worker-test",
            )
        return request_id

    def stop_wheels(self, **payload: Any) -> dict[str, Any]:
        self.stop_calls += 1
        self.wheel_generation = max(self.wheel_generation, int(payload.get("generation", 0) or 0))
        return {"generation": self.wheel_generation, "command_id": payload.get("command_id", "")}

    def start_playback_plan(self, payload: dict[str, Any], **metadata: Any) -> None:
        self.start_calls.append({"payload": copy.deepcopy(payload), **metadata})

    def stop_playback(self, **_payload: Any) -> None:
        self.stop_calls += 1

    def pause_playback(self) -> None:
        pass

    def resume_playback(self) -> None:
        pass


def asynchronous_controller(root: Path) -> tuple[HeightReplayController, FakeSimClient]:
    controller = controller_with_three_steps(root)
    fake = FakeSimClient()
    controller.no_sim = False
    controller.sim_launch_mode = "subprocess"
    controller.sim_client = fake  # type: ignore[assignment]
    controller.transport.process_client = fake
    controller.latest_sim_status = fake.status()
    controller.sim_ready = True
    return controller, fake


class RestoreSourceTest(unittest.TestCase):
    def test_previous_full_state_is_authoritative_and_only_selected_step_is_planned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            state_b = sim_state(20.0, 2.0)
            controller.manager.steps[1]["sim_state_after"] = state_b
            controller.manager.steps[2]["sim_state_before"] = copy.deepcopy(state_b)
            controller.transport.adapter.restore_sim_state(sim_state(-15.0, 9.0))
            sent: list[dict[str, Any]] = []
            original = controller.transport.restore_sim_state

            def capture(payload: dict[str, Any], *, request_id: str = "") -> str:
                sent.append(copy.deepcopy(payload))
                return original(payload, request_id=request_id)

            controller.transport.restore_sim_state = capture  # type: ignore[method-assign]
            self.assertTrue(controller.start_selected_step_playback(3, profile="raw"))
            self.assertEqual(sent, [state_b])
            self.assertNotEqual(sent[0]["command_state"], command_state(-15.0))
            result = controller.last_selected_restore_result
            self.assertEqual(result["restore_source_step_index"], 2)
            self.assertEqual(result["restore_source_field"], "sim_state_after")
            self.assertFalse(result["fallback_used"])
            self.assertTrue(result["plan_selected_playback"])
            self.assertEqual(result["plan_source_steps"], [3])
            self.assertEqual({event.source_step for event in controller.playback.plan.events}, {3})

    def test_previous_command_state_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.manager.steps[1].pop("sim_state_after", None)
            controller.manager.steps[1]["command_state_after"] = command_state(22.0)
            resolved = controller.resolve_selected_step_restore_state(3)
            self.assertEqual(resolved["restore_source_step_index"], 2)
            self.assertEqual(resolved["restore_source_field"], "command_state_after")
            self.assertEqual(resolved["restore_command_state"], command_state(22.0))
            self.assertFalse(resolved["fallback_used"])

    def test_selected_start_compatibility_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.manager.steps[1].pop("sim_state_after", None)
            controller.manager.steps[1].pop("command_state_after", None)
            controller.manager.steps[2]["sim_state_before"] = sim_state(20.0, 3.0)
            resolved = controller.resolve_selected_step_restore_state(3)
            self.assertEqual(resolved["restore_source_step_index"], 3)
            self.assertEqual(resolved["restore_source_field"], "sim_state_before")
            self.assertTrue(resolved["fallback_used"])

    def test_missing_all_saved_states_never_starts_from_live_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            for key in ("sim_state_after", "command_state_after"):
                controller.manager.steps[1].pop(key, None)
            for key in ("sim_state_before", "command_state_before"):
                controller.manager.steps[2].pop(key, None)
            controller.transport.adapter.restore_sim_state(sim_state(44.0, 8.0))
            self.assertFalse(controller.start_selected_step_playback(3, profile="raw"))
            self.assertFalse(controller.playback.active)
            self.assertFalse(controller.playback.start_requested)
            self.assertEqual(controller.playback.scheduled_start_at, 0.0)
            self.assertIs(controller.operation.state, OperationState.IDLE)
            self.assertIn("FULL_VALID saved Isaac pose is required", controller.detail_text)

    def test_first_step_uses_own_saved_start_and_does_not_respawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            first_state = sim_state(0.0, 1.0)
            controller.manager.steps[0]["sim_state_before"] = first_state
            respawn_calls: list[str] = []
            controller.respawn_robot = lambda **_payload: respawn_calls.append("respawn") or True  # type: ignore[method-assign]
            self.assertTrue(controller.start_selected_step_playback(1, profile="raw"))
            self.assertEqual(respawn_calls, [])
            result = controller.last_selected_restore_result
            self.assertEqual(result["restore_source_step_index"], 1)
            self.assertEqual(result["restore_source_field"], "sim_state_before")
            self.assertEqual(result["plan_source_steps"], [1])

    def test_continuity_mismatch_warns_but_previous_end_remains_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.manager.steps[1]["sim_state_after"] = sim_state(20.0, 2.0)
            controller.manager.steps[2]["sim_state_before"] = sim_state(99.0, 7.0)
            resolved = controller.resolve_selected_step_restore_state(3)
            self.assertEqual(resolved["continuity"], "WARNING")
            self.assertIn("authoritative previous-step end state", resolved["continuity_warning"])
            self.assertEqual(resolved["restore_sim_state"], controller.manager.steps[1]["sim_state_after"])


class RestoreTransactionTest(unittest.TestCase):
    def test_restore_ack_and_detailed_state_precede_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            controller.manager.steps[1]["sim_state_after"] = sim_state(20.0, 2.0)
            self.assertTrue(controller.start_selected_step_playback(3, profile="raw"))
            self.assertEqual(len(fake.start_calls), 0)
            request_id = fake.restore_calls[0]["request_id"]
            fake.latest_status.update(
                restore_count=1,
                last_restore_result="ok",
                last_restore_error="",
                last_restore_request_id=request_id,
            )
            controller.update()
            self.assertEqual(fake.state_requests, [True])
            self.assertEqual(len(fake.start_calls), 0)
            fake.latest_detailed_status = {
                **fake.latest_status,
                "detail_status": True,
                "state_capture_request_id": fake.last_state_request_id,
                "state_capture_purpose": fake.last_state_request_purpose,
                "state_capture_worker_session_id": "worker-test",
                "sim_state": copy.deepcopy(fake.restore_calls[0]["sim_state"]),
            }
            controller.update()
            self.assertEqual(len(fake.start_calls), 1)
            result = controller.last_selected_restore_result
            events = [row["event"] for row in result["trace"]]
            self.assertLess(events.index("operation_acquired"), events.index("stop_wheels_sent"))
            self.assertLess(events.index("stop_wheels_sent"), events.index("restore_sent"))
            self.assertLess(events.index("restore_sent"), events.index("restore_acknowledged"))
            self.assertLess(events.index("restore_acknowledged"), events.index("state_verified"))
            self.assertLess(events.index("state_verified"), events.index("playback_start_requested"))

    def test_restore_failure_releases_operation_and_never_starts_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            self.assertTrue(controller.start_selected_step_playback(3, profile="raw"))
            request_id = fake.restore_calls[0]["request_id"]
            fake.latest_status.update(
                restore_count=1,
                last_restore_result="error",
                last_restore_error="synthetic restore failure",
                last_restore_request_id=request_id,
            )
            controller.update()
            self.assertEqual(fake.start_calls, [])
            self.assertFalse(controller.playback.active)
            self.assertFalse(controller.playback.start_requested)
            self.assertEqual(controller.playback.scheduled_start_at, 0.0)
            self.assertIs(controller.operation.state, OperationState.IDLE)
            self.assertEqual(controller.mode, MODE_TEST)
            self.assertIn("synthetic restore failure", controller.playback.last_error)
            fake.latest_status.update(last_restore_result="ok", last_restore_error="")
            self.assertTrue(controller.start_selected_step_playback(3, profile="raw"))

    def test_stop_during_restore_cancels_pending_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            self.assertTrue(controller.start_selected_step_playback(3, profile="raw"))
            request_id = fake.restore_calls[0]["request_id"]
            controller.handle_command("stop_play")
            self.assertIsNone(controller.pending_selected_playback)
            self.assertIs(controller.operation.state, OperationState.IDLE)
            fake.latest_status.update(
                restore_count=1,
                last_restore_result="ok",
                last_restore_error="",
                last_restore_request_id=request_id,
            )
            fake.latest_detailed_status = {
                **fake.latest_status,
                "detail_status": True,
                "sim_state": copy.deepcopy(fake.restore_calls[0]["sim_state"]),
            }
            controller.update()
            self.assertEqual(fake.start_calls, [])
            self.assertTrue(controller.last_selected_restore_result["cancelled"])
            self.assertTrue(controller.can_playback()[0])


class ConflictAndRegressionTest(unittest.TestCase):
    def test_operation_and_dirty_staging_conflicts_block_selected_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for state in (OperationState.RECORDING, OperationState.SCENE_UPDATE, OperationState.RESPAWNING):
                controller = controller_with_three_steps(root / state.value)
                controller.operation.begin(state)
                self.assertFalse(controller.start_selected_step_playback(3, profile="raw"), state.value)
                self.assertIsNone(controller.pending_selected_playback)
            controller = controller_with_three_steps(root / "playback")
            controller.operation.begin(OperationState.PLAYBACK)
            controller.playback.active = True
            self.assertFalse(controller.start_selected_step_playback(3, profile="raw"))
            controller = controller_with_three_steps(root / "staging")
            controller.servo_wheel_staging_active = True
            controller.servo_wheel_staged_dirty = True
            self.assertFalse(controller.start_selected_step_playback(3, profile="raw"))
            self.assertTrue(controller.operation.idle)

    def test_selected_fast_uses_same_restore_source_and_keeps_actuator_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            controller.manager.steps[1]["sim_state_after"] = sim_state(20.0, 2.0)
            expected_fast = plan_from_steps(
                [controller.manager.steps[2]],
                profile="fast",
                max_wheel_speed=controller.max_wheel_speed,
                sequence_total_steps=3,
            )
            self.assertTrue(controller.start_selected_step_playback(3, profile="fast"))
            request_id = fake.restore_calls[0]["request_id"]
            fake.latest_status.update(
                restore_count=1,
                last_restore_result="ok",
                last_restore_error="",
                last_restore_request_id=request_id,
                last_restore_verification={"verified": True, "request_id": request_id},
            )
            controller.update()
            fake.latest_detailed_status = {
                **fake.latest_status,
                "detail_status": True,
                "state_capture_request_id": fake.last_state_request_id,
                "state_capture_purpose": fake.last_state_request_purpose,
                "state_capture_worker_session_id": "worker-test",
                "sim_state": copy.deepcopy(fake.restore_calls[0]["sim_state"]),
            }
            controller.update()
            result = controller.last_selected_restore_result
            self.assertEqual(result["restore_source_step_index"], 2)
            self.assertEqual(result["restore_source_field"], "sim_state_after")
            self.assertEqual(
                [event.command for event in controller.playback.plan.events],
                [event.command for event in expected_fast.events],
            )
            self.assertEqual(controller.playback.plan.profile, expected_fast.profile)

    def test_other_playback_handlers_keep_their_existing_entry_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            calls: list[dict[str, Any]] = []

            def capture(steps: list[dict[str, Any]], **kwargs: Any) -> bool:
                calls.append({"indices": [row["index"] for row in steps], **kwargs})
                return True

            controller.start_playback = capture  # type: ignore[method-assign]
            controller.selected_step_index = 2
            controller.play_all(fast=False)
            controller._handle_play_to_step(["2"])
            controller._handle_respawn_play(["selected", "2"])
            controller._handle_respawn_play(["to", "2"])
            self.assertEqual(calls[0]["indices"], [1, 2, 3])
            self.assertEqual(calls[1]["indices"], [1, 2])
            self.assertNotIn("respawn_first", calls[1])
            self.assertEqual(calls[2]["indices"], [2])
            self.assertTrue(calls[2]["respawn_first"])
            self.assertEqual(calls[3]["indices"], [1, 2])
            self.assertTrue(calls[3]["respawn_first"])


if __name__ == "__main__":
    unittest.main()
