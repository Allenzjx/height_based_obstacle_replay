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

from playback import PlaybackEvent, PlaybackManager, PlaybackPlan, SimTimePlaybackService, playback_plan_to_payload  # noqa: E402
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


class FakeWorkerTransport:
    def __init__(self) -> None:
        self.started: list[tuple[PlaybackPlan, float, str]] = []
        self.stopped: list[str] = []

    def start_playback_plan(
        self,
        plan: PlaybackPlan,
        *,
        start_delay_sim_s: float = 0.0,
        plan_id: str = "",
    ) -> None:
        self.started.append((plan, float(start_delay_sim_s), str(plan_id)))

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
        plan = _plan()
        self.assertTrue(manager.start_worker_plan(plan))

        manager.sync_worker_status({"active": False, "paused": False, "plan_id": "", "index": 0, "events_sent": 0})
        self.assertTrue(manager.active)
        self.assertTrue(manager.worker_managed)
        self.assertFalse(manager.worker_acknowledged)
        self.assertIn("acknowledgement", manager.last_info)

        manager.sync_worker_status(
            {
                "active": True,
                "paused": False,
                "plan_id": manager.worker_plan_id,
                "index": 1,
                "events_sent": 1,
                "stop_reason": "",
                "last_info": "worker playback running",
            }
        )
        self.assertTrue(manager.active)
        self.assertTrue(manager.worker_acknowledged)
        self.assertEqual(manager.index, 1)

    def test_repeated_identical_plan_uses_unique_request_ids_and_ignores_stale_status(self) -> None:
        controller = FakePlaybackController()
        manager = PlaybackManager(controller)
        plan = _plan()

        self.assertTrue(manager.start_worker_plan(plan))
        first_id = manager.worker_plan_id
        manager.stop(silent=True)
        self.assertTrue(manager.start_worker_plan(plan))
        second_id = manager.worker_plan_id

        self.assertNotEqual(first_id, second_id)
        self.assertEqual(controller.transport.started[0][2], first_id)
        self.assertEqual(controller.transport.started[1][2], second_id)
        manager.sync_worker_status(
            {
                "active": False,
                "paused": False,
                "plan_id": first_id,
                "stop_reason": "stopped",
                "events_sent": 1,
            }
        )
        self.assertTrue(manager.active)
        self.assertTrue(manager.worker_managed)
        self.assertFalse(manager.worker_acknowledged)

        manager.sync_worker_status(
            {
                "active": True,
                "paused": False,
                "plan_id": second_id,
                "stop_reason": "",
                "events_sent": 1,
            }
        )
        self.assertTrue(manager.active)
        self.assertTrue(manager.worker_acknowledged)


if __name__ == "__main__":
    unittest.main()
