from __future__ import annotations

import copy
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from height_version_store import HeightVersionStore, file_sha256
from operation_coordinator import OperationState
from playback import (
    PlaybackEvent,
    PlaybackPlan,
    PlaybackSegment,
    SimTimePlaybackService,
    plan_fingerprint,
)
from sequence_model import empty_command_state
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi
from tests.test_selected_step_previous_saved_state import (
    controller_with_three_steps,
    make_args,
    sim_state,
    step,
)


class SelectedRawFastContractTest(unittest.TestCase):
    def test_worker_selected_resume_uses_saved_progress_without_undefined_event(self) -> None:
        service = SimTimePlaybackService()
        service.active = True
        service.paused = True
        service.pause_started_sim_time_s = 1.0
        service.plan = PlaybackPlan(
            path=None,
            events=[PlaybackEvent(0.0, "servo front_left_hip 8", source_step=2)],
            final_time_s=0.1,
            selected_playback=True,
            total_steps=3,
        )
        service.progress.current_step_index = 2
        service.progress.total_steps = 3
        service.resume(current_sim_time_s=1.25)
        self.assertFalse(service.paused)
        self.assertEqual(service.progress.status_text, "Restore verified. Playing Selected Step 2 / 3")

    def test_raw_and_fast_restore_full_previous_pose_and_plan_only_selected_step(self) -> None:
        for profile in ("raw", "fast"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                controller = controller_with_three_steps(Path(tmp))
                self.assertTrue(controller.start_selected_step_playback(3, profile=profile))
                result = controller.last_selected_restore_result
                self.assertEqual(result["restore_source_step_index"], 2)
                self.assertEqual(result["restore_source_field"], "sim_state_after")
                self.assertTrue(result["restore_verification"]["verified"])
                self.assertEqual(result["plan_source_steps"], [3])
                self.assertTrue(controller.playback.plan.selected_playback)
                self.assertEqual({event.source_step for event in controller.playback.plan.events}, {3})
                self.assertTrue(controller.playback.plan.timing["selected_actuator_motion_changes"])

    def test_raw_rejects_incomplete_checkpoint_instead_of_using_command_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.manager.steps[1]["sim_state_after"] = {
                "root_pose": None,
                "joint_pos": None,
                "command_state": copy.deepcopy(
                    controller.manager.steps[1]["command_state_after"]
                ),
            }
            controller.manager.steps[2].pop("sim_state_before", None)
            self.assertFalse(controller.start_selected_step_playback(3, profile="raw"))
            self.assertIn("FULL_VALID", controller.playback.last_error)
            self.assertTrue(controller.operation.idle)
            self.assertIsNone(controller.pending_selected_playback)

    def test_selected_noop_is_explicit_and_never_flashes_as_completed_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.manager.steps[2] = step(3, 20.0, 20.0)
            controller.manager.steps[2]["sim_state_before"] = sim_state(20.0, 2.0)
            controller.manager.steps[2]["sim_state_after"] = sim_state(20.0, 3.0)
            self.assertFalse(controller.start_selected_step_playback(3, profile="fast"))
            self.assertIn("contains no actuator motion", controller.playback.last_error)
            self.assertTrue(controller.operation.idle)


class RunManagementContractTest(unittest.TestCase):
    def test_combobox_selection_only_previews_until_open_button(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v2"
            controller = HeightReplayController(make_args(root))
            first = controller.store.save_new_version(50, [step(1, 0.0, 10.0)], version_name="first")
            second = controller.store.save_new_version(50, [step(1, 0.0, 20.0)], version_name="second")
            controller.load_steps_for_current_height(discard_dirty=True, version_id=first.name)
            ui = RealRobotStyleHeightReplayUi(controller)
            ui.root.withdraw()
            try:
                panel = ui.height_generate_panel
                panel._sync_run_choices()
                second_label = next(
                    label for label, run_id in panel._version_labels.items() if run_id == second.name
                )
                revision_before = controller.manager.revision
                steps_before = copy.deepcopy(controller.manager.steps)
                active_before = controller.store.active_version_id(50)
                panel.run_var.set(second_label)
                panel._select_pending_run()
                self.assertEqual(controller.pending_selected_run_id, second.name)
                self.assertEqual(controller.current_version_id, first.name)
                self.assertEqual(controller.manager.revision, revision_before)
                self.assertEqual(controller.manager.steps, steps_before)
                self.assertEqual(controller.store.active_version_id(50), active_before)
                panel._open_selected_run()
                self.assertEqual(controller.current_version_id, second.name)
                self.assertEqual(controller.manager.steps[0]["events"][0]["command"], "servo front_left_hip 20.0")
                for label in (
                    "New Empty Run",
                    "Open Selected Run",
                    "💾 Update Current Run",
                    "💾 Save As New Run",
                    "Refresh Runs",
                ):
                    self.assertIn(label, [
                        panel.new_run_button.cget("text"),
                        panel.open_run_button.cget("text"),
                        panel.update_run_button.cget("text"),
                        panel.save_as_run_button.cget("text"),
                        panel.refresh_runs_button.cget("text"),
                    ])
            finally:
                ui._window_close(force=True)

    def test_update_keeps_id_and_save_as_new_preserves_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp) / "v2"))
            controller.manager.adopt_steps([step(1, 0.0, 10.0)], dirty=True)
            first_path = controller.save_new_version(version_name="first")
            self.assertIsNotNone(first_path)
            first_id = controller.current_version_id
            first_meta = controller.store.inspect_version(50, first_id)
            first_created = first_meta["created_at"]
            version_count = len(controller.store.list_versions(50, include_legacy=False))

            controller.manager.adopt_steps([step(1, 0.0, 20.0)], dirty=True)
            updated = controller.save_current_version(confirmed=True)
            self.assertEqual(updated, first_path)
            self.assertEqual(controller.current_version_id, first_id)
            updated_meta = controller.store.inspect_version(50, first_id)
            self.assertEqual(updated_meta["created_at"], first_created)
            self.assertNotEqual(updated_meta["updated_at"], first_meta["updated_at"])
            self.assertEqual(len(controller.store.list_versions(50, include_legacy=False)), version_count)
            self.assertTrue(list(Path(first_path).glob("accepted_steps.jsonl.backup_*")))

            parent_hash = updated_meta["accepted_steps_sha256"]
            controller.manager.adopt_steps([step(1, 0.0, 30.0)], dirty=True)
            second_path = controller.save_new_version(version_name="child")
            self.assertIsNotNone(second_path)
            second_id = controller.current_version_id
            self.assertNotEqual(second_id, first_id)
            self.assertEqual(controller.current_version_metadata["parent_version_id"], first_id)
            self.assertEqual(controller.store.inspect_version(50, first_id)["accepted_steps_sha256"], parent_hash)
            self.assertEqual(len(controller.store.list_versions(50, include_legacy=False)), version_count + 1)

    def test_new_empty_run_is_unsaved_and_does_not_create_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp) / "v2"))
            controller.manager.adopt_steps([step(1, 0.0, 10.0)], dirty=True)
            controller.save_new_version()
            count_before = len(controller.store.list_versions(50, include_legacy=False))
            controller.new_empty_sequence_for_current_height(discard_dirty=True)
            self.assertEqual(controller.current_version_id, "")
            self.assertEqual(controller.manager.count, 0)
            self.assertTrue(controller.manager.dirty)
            self.assertEqual(len(controller.store.list_versions(50, include_legacy=False)), count_before)
            self.assertIn("Unsaved New Run", controller.detail_text)

    def test_failed_atomic_update_restores_original_run_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightVersionStore(Path(tmp) / "v2", legacy_root=Path(tmp) / "legacy")
            version = store.save_new_version(50, [step(1, 0.0, 10.0)])
            target = version / "accepted_steps.jsonl"
            metadata = version / "metadata.json"
            manifest = store.manifest_path
            before = {
                target: target.read_bytes(),
                metadata: metadata.read_bytes(),
                manifest: manifest.read_bytes(),
            }
            with mock.patch.object(
                store,
                "_replace_manifest_version",
                side_effect=RuntimeError("synthetic manifest failure"),
            ):
                with self.assertRaises(RuntimeError):
                    store.save_current_version(
                        50,
                        version.name,
                        [step(1, 0.0, 25.0)],
                        confirmed=True,
                    )
            for path, payload in before.items():
                self.assertEqual(path.read_bytes(), payload, path)
            self.assertEqual(file_sha256(target), store.inspect_version(50, version.name)["accepted_steps_sha256"])


class PlaybackReconciliationContractTest(unittest.TestCase):
    def test_resume_accepts_fresh_matching_worker_pause_before_local_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            playback = controller.playback
            playback.active = True
            playback.paused = False
            playback.worker_managed = True
            playback.worker_plan_id = "plan-live"
            playback.worker_request_id = "request-live"
            controller.latest_sim_status = {
                "worker_playback": {
                    "active": True,
                    "paused": True,
                    "plan_id": "plan-live",
                    "request_id": "request-live",
                }
            }
            sent: list[str] = []
            controller.transport.resume_playback = lambda: sent.append("resume")  # type: ignore[method-assign]
            playback.resume()
            self.assertEqual(sent, ["resume"])
            self.assertFalse(playback.paused)

    @staticmethod
    def controller() -> HeightReplayController:
        root = Path(tempfile.mkdtemp()) / "v2"
        controller = controller_with_three_steps(root)
        fake = SimpleNamespace(connected=True, last_status_time=time.monotonic())
        controller.no_sim = False
        controller.sim_client = fake  # type: ignore[assignment]
        controller.sim_connection_enabled = True
        controller.sim_ready = True
        controller.latest_sim_status = {
            "phase": "running",
            "runtime_ready": True,
            "worker_playback": {"active": False, "stop_reason": "complete"},
        }
        return controller

    def test_stale_local_playback_and_operation_are_cleared(self) -> None:
        controller = self.controller()
        controller.playback.active = True
        controller.playback.worker_managed = True
        controller.playback.worker_plan_id = "stale-plan"
        controller.playback.worker_request_id = "stale-request"
        controller.operation.begin(OperationState.PLAYBACK, detail="stale")
        result = controller.reconcile_playback_state_before_start()
        self.assertTrue(result["reconciled"], result)
        self.assertFalse(controller.playback.active)
        self.assertFalse(controller.playback.start_requested)
        self.assertTrue(controller.operation.idle)
        self.assertIsNone(controller.playback.plan)

    def test_real_worker_activity_and_live_matching_request_are_not_cleared(self) -> None:
        controller = self.controller()
        controller.latest_sim_status["worker_playback"] = {
            "active": True,
            "plan_id": "real-plan",
            "request_id": "real-request",
            "worker_session_id": "real-session",
        }
        controller.playback.active = True
        active = controller.reconcile_playback_state_before_start()
        self.assertFalse(active["reconciled"])
        self.assertTrue(controller.playback.active)

        controller.latest_sim_status["worker_playback"] = {"active": False}
        controller.playback.active = False
        controller.playback.start_requested = True
        controller.playback.worker_request_id = "pending-request"
        controller.playback.worker_requested_at = time.monotonic()
        pending = controller.reconcile_playback_state_before_start()
        self.assertFalse(pending["reconciled"])
        self.assertTrue(controller.playback.start_requested)

    def test_selection_and_dirty_working_run_do_not_block_play_all(self) -> None:
        controller = self.controller()
        controller.manager.dirty = True
        controller.selected_step_index = 3
        controller.pending_selected_run_id = "pending-not-opened"
        availability = controller.playback_availability()
        self.assertTrue(availability.can_start, availability.reason)
        controller.servo_wheel_staging_active = True
        controller.servo_wheel_staged_dirty = True
        blocked = controller.playback_availability()
        self.assertFalse(blocked.can_start)
        self.assertIn("Launch, Clear, or Cancel", blocked.reason)

    def test_expired_start_selected_restore_worker_error_and_plan_mismatch_reconcile(self) -> None:
        cases = ("expired_start", "expired_selected", "worker_error", "plan_mismatch")
        for case in cases:
            with self.subTest(case=case):
                controller = self.controller()
                controller.operation.begin(OperationState.PLAYBACK, detail=case)
                if case == "expired_start":
                    controller.playback.start_requested = True
                    controller.playback.worker_request_id = "expired"
                    controller.playback.worker_requested_at = time.monotonic() - 30.0
                elif case == "expired_selected":
                    controller.pending_selected_playback = {"deadline": 0.0}
                    controller.playback.start_requested = True
                    controller.playback.worker_request_id = "expired-selected"
                    controller.playback.worker_requested_at = time.monotonic() - 30.0
                elif case == "worker_error":
                    controller.playback.active = True
                    controller.playback.worker_managed = True
                    controller.latest_sim_status["worker_playback"] = {
                        "active": False,
                        "stop_reason": "worker_exception",
                        "last_error": "synthetic worker error",
                    }
                else:
                    controller.playback.active = True
                    controller.playback.worker_managed = True
                    controller.playback.worker_plan_id = "local-plan"
                    controller.latest_sim_status["worker_playback"] = {
                        "active": False,
                        "plan_id": "other-plan",
                        "stop_reason": "complete",
                    }
                result = controller.reconcile_playback_state_before_start()
                self.assertTrue(result["reconciled"], result)
                self.assertTrue(controller.operation.idle)
                self.assertIsNone(controller.pending_selected_playback)

    def test_task_conflict_matrix_blocks_only_real_conflicts(self) -> None:
        controller = self.controller()
        controller.manager.dirty = True
        controller.pending_selected_run_id = "preview-only"
        self.assertTrue(controller.playback_availability(selected_index=2).can_start)
        self.assertTrue(controller.playback_availability(selected_index=2).can_play_selected)

        for state in (
            OperationState.RECORDING,
            OperationState.SCENE_UPDATE,
            OperationState.RESPAWNING,
            OperationState.RUN_MANAGEMENT,
        ):
            with self.subTest(state=state):
                controller.operation.finish()
                controller.operation.begin(state, detail=f"{state.value} test")
                self.assertFalse(controller.playback_availability(selected_index=2).can_start)
        controller.operation.finish()
        controller.pending_step = {"events": []}
        self.assertFalse(controller.playback_availability(selected_index=2).can_start)
        controller.pending_step = None
        controller.pending_replacement = {"events": []}
        self.assertFalse(controller.playback_availability(selected_index=2).can_start)
        controller.pending_replacement = None
        controller.playback.active = True
        self.assertFalse(controller.playback_availability(selected_index=2).can_start)


class ContactResidualCompletionContractTest(unittest.TestCase):
    class Adapter:
        def __init__(self, actual_deg: float, velocity_deg_s: float = 0.0) -> None:
            self.actual_deg = actual_deg
            self.velocity_deg_s = velocity_deg_s
            self.joint_command_deg = {"front_left_hip": 0.0}

        def apply_motion_batch(self, _payload):
            self.joint_command_deg.update(_payload.get("servo_targets_deg", {}))
            return {}

        def get_actual_joint_state(self):
            return {
                "servos": {
                    "front_left_hip": {
                        "deg": self.actual_deg,
                        "velocity_deg_s": self.velocity_deg_s,
                    }
                }
            }

        def command_to_actual_target_deg(self, _name, target):
            return float(target)

        def stop_wheels(self):
            return None

        def apply_commands_to_robot(self):
            return None

    @staticmethod
    def plan() -> PlaybackPlan:
        event = PlaybackEvent(
            time_s=0.0,
            command="servo front_left_hip 0",
            source_step=10,
            source_step_id="step-10",
            commands_in_step=1,
            global_command_index=1,
            segment_index=37,
            channel="servo",
            servo_targets=(("front_left_hip", 0.0),),
        )
        segment = PlaybackSegment(
            segment_index=37,
            source_step=10,
            source_step_id="step-10",
            event_start_index=0,
            event_count=1,
            planned_start_s=0.0,
            planned_end_s=0.1,
            base_duration_s=0.1,
            servo_base_duration_s=0.1,
            servo_duration_s=0.1,
            servo_targets={"front_left_hip": 0.0},
            servo_tolerance_deg=3.0,
            recorded_servo_residual_deg={"front_left_hip": 2.9},
        )
        plan = PlaybackPlan(path=None, events=[event], segments=[segment], final_time_s=0.1, total_steps=12)
        plan.plan_sha256 = plan_fingerprint(plan)
        return plan

    def test_2986098_inside_3000_completes_with_contact_warning(self) -> None:
        adapter = self.Adapter(2.986098)
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0))
        for tick, now in enumerate((0.0, 0.1, 0.20, 0.36)):
            service.update(adapter, current_sim_time_s=now, current_sim_step=tick, current_wall_time_s=now)
        status = service.status_dict(current_sim_time_s=0.36, current_wall_time_s=0.36)
        self.assertEqual(status["stop_reason"], "complete")
        self.assertEqual(status["last_servo_residual_warning"]["warning"], "contact_residual_accepted")
        self.assertAlmostEqual(status["last_servo_residual_warning"]["max_error_deg"], 2.986098)

    def test_3200_outside_3000_still_fails(self) -> None:
        adapter = self.Adapter(3.2, velocity_deg_s=0.0)
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0))
        for tick, now in enumerate((0.0, 0.1, 0.5, 0.9)):
            service.update(adapter, current_sim_time_s=now, current_sim_step=tick, current_wall_time_s=now)
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "actuator_limit")
        self.assertIn("recent_min_deg", service.last_error)

    def test_nan_fails_immediately(self) -> None:
        adapter = self.Adapter(float("nan"))
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0))
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=0, current_wall_time_s=0.0)
        service.update(adapter, current_sim_time_s=0.1, current_sim_step=1, current_wall_time_s=0.1)
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "invalid_joint_state")

    def test_worsening_outside_hard_tolerance_is_actuator_unstable(self) -> None:
        adapter = self.Adapter(3.2, velocity_deg_s=2.0)
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0))
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=0, current_wall_time_s=0.0)
        service.update(adapter, current_sim_time_s=0.1, current_sim_step=1, current_wall_time_s=0.1)
        adapter.actual_deg = 3.6
        service.update(adapter, current_sim_time_s=0.8, current_sim_step=2, current_wall_time_s=0.8)
        adapter.actual_deg = 5.2
        service.update(adapter, current_sim_time_s=1.6, current_sim_step=3, current_wall_time_s=1.6)
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "actuator_unstable")
        self.assertIn("recent_slope_deg_s", service.last_error)

    def test_positive_error_with_negative_velocity_is_recovering_not_unstable(self) -> None:
        adapter = self.Adapter(3.2, velocity_deg_s=-2.0)
        service = SimTimePlaybackService()
        self.assertTrue(service.start_plan(self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0))
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=0, current_wall_time_s=0.0)
        service.update(adapter, current_sim_time_s=0.1, current_sim_step=1, current_wall_time_s=0.1)
        adapter.actual_deg = 3.6
        service.update(adapter, current_sim_time_s=0.8, current_sim_step=2, current_wall_time_s=0.8)
        adapter.actual_deg = 5.2
        service.update(adapter, current_sim_time_s=1.6, current_sim_step=3, current_wall_time_s=1.6)
        self.assertTrue(service.active)
        self.assertNotEqual(service.stop_reason, "actuator_unstable")


if __name__ == "__main__":
    unittest.main()
