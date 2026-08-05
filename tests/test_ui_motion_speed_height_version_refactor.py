from __future__ import annotations

import json
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from command_model import KNEE_JOINT_NAMES, SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES, validate_motion_command
from height_manifest import (
    SUPPORTED_HEIGHTS_MM,
    HeightValidationError,
    height_folder_name_mm,
    legacy_cm_to_mm,
    normalize_height_mm,
)
from height_replay_ui import build_parser, normalize_motion_args
from height_version_store import HeightVersionStore, file_sha256
from motion_speed import SpeedScale, load_motion_reference
from playback import SimTimePlaybackService, plan_from_steps
from sequence_model import empty_command_state, make_event, make_step, save_steps_jsonl
from sim_ipc_protocol import MESSAGE_TYPES, encode_message, make_message
from sim_obstacle_scene import (
    ENVIRONMENT_REFERENCE,
    OBSTACLE_FRONT_FACE_X_M,
    OBSTACLE_LENGTH_M,
    OBSTACLE_WIDTH_M,
    ROBOT_COLLISION_WIDTH_M,
)
from sim_robot_adapter import NullSimRobotAdapter
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi
from sim_worker_process import WorkerIpc, build_parser as build_worker_parser
from sim_worker_runtime import build_common_worker_status
from tests.controller_test_utils import make_args, motion_step


class HeightAndVersionContractTest(unittest.TestCase):
    def test_only_integer_mm_heights_are_accepted(self) -> None:
        self.assertEqual(SUPPORTED_HEIGHTS_MM, (50, 75, 100))
        for value in SUPPORTED_HEIGHTS_MM:
            self.assertEqual(normalize_height_mm(value), value)
            self.assertEqual(height_folder_name_mm(value), f"height_{value:03d}mm")
        for value in (0, 5, 10, 74, 76, 150, 7.5):
            with self.assertRaises(HeightValidationError):
                normalize_height_mm(value)

    def test_legacy_cm_maps_only_five_and_ten(self) -> None:
        self.assertEqual(legacy_cm_to_mm(5), 50)
        self.assertEqual(legacy_cm_to_mm(10), 100)
        for value in (0, 7.5, 15, 75):
            with self.assertRaises(HeightValidationError):
                legacy_cm_to_mm(value)

    def test_two_new_versions_are_immutable_and_load_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightVersionStore(Path(tmp) / "v2", legacy_root=Path(tmp) / "legacy")
            first = [motion_step(index=1, height_cm=5, command="servo front_left_hip 10")]
            second = [motion_step(index=1, height_cm=5, command="servo front_left_hip 20")]
            v1 = store.save_new_version(50, first, version_name="first")
            hash1 = file_sha256(v1 / "accepted_steps.jsonl")
            v2 = store.save_new_version(50, second, version_name="second", parent_version_id=v1.name)
            hash2 = file_sha256(v2 / "accepted_steps.jsonl")
            self.assertNotEqual(v1, v2)
            self.assertNotEqual(hash1, hash2)
            self.assertEqual(file_sha256(v1 / "accepted_steps.jsonl"), hash1)
            self.assertEqual(store.load_version(50, v1.name)[0][0]["events"][0]["command"], "servo front_left_hip 10")
            self.assertEqual(store.load_version(50, v2.name)[0][0]["events"][0]["command"], "servo front_left_hip 20")
            self.assertEqual(store.status_rows()[0]["version_count"], 2)

    def test_save_current_requires_confirmation_and_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightVersionStore(Path(tmp) / "v2", legacy_root=Path(tmp) / "legacy")
            version = store.save_new_version(75, [motion_step(command="servo front_left_hip 1")])
            with self.assertRaises(PermissionError):
                store.save_current_version(75, version.name, [], confirmed=False)
            store.save_current_version(75, version.name, [motion_step(command="servo front_left_hip 2")], confirmed=True)
            self.assertTrue(list(version.glob("accepted_steps.jsonl.backup_*")))
            loaded, metadata = store.load_version(75, version.name)
            playback_plan = plan_from_steps(loaded, profile="raw", max_wheel_speed=2.0943951023931953)
            self.assertEqual(metadata["height_mm"], 75)
            self.assertTrue(playback_plan.events)

    def test_legacy_versions_are_read_only_and_75mm_can_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy" / "height_05cm"
            legacy.mkdir(parents=True)
            save_steps_jsonl([motion_step(height_cm=5)], legacy / "accepted_steps.jsonl")
            store = HeightVersionStore(root / "v2", legacy_root=root / "legacy")
            rows = store.list_versions(50)
            self.assertEqual(rows[-1]["version_id"], "legacy_5cm_readonly")
            self.assertTrue(rows[-1]["read_only"])
            self.assertEqual(store.list_versions(75), [])
            with self.assertRaises(PermissionError):
                store.save_current_version(50, "legacy_5cm_readonly", [], confirmed=True)


class SpeedSemanticsContractTest(unittest.TestCase):
    def test_servo_target_never_scales_and_rate_does(self) -> None:
        reference = load_motion_reference()
        model = SpeedScale(reference)
        target = -60.0
        values = {}
        for percent in (50, 100, 200, 300):
            model.set_percent(percent)
            requested, effective = model.servo_velocity()
            values[percent] = (target, requested, effective, abs(target) / effective)
        self.assertEqual({row[0] for row in values.values()}, {target})
        self.assertAlmostEqual(values[200][1], values[100][1] * 2.0)
        self.assertAlmostEqual(values[300][1], values[100][1] * 3.0)
        self.assertAlmostEqual(values[200][3], values[100][3] / 2.0)

    def test_wheel_velocity_and_joint_path_scale_but_duration_does_not(self) -> None:
        model = SpeedScale(load_motion_reference())
        canonical = 0.3
        duration = 2.0
        paths = {}
        for percent in (50, 100, 200, 300):
            model.set_percent(percent)
            _requested, effective = model.wheel_velocity(canonical)
            paths[percent] = effective * duration
        self.assertAlmostEqual(paths[200], paths[100] * 2.0)
        self.assertAlmostEqual(paths[300], paths[100] * 3.0)
        self.assertAlmostEqual(paths[50], paths[100] * 0.5)

    def test_adapter_applies_scale_once(self) -> None:
        adapter = NullSimRobotAdapter()
        adapter.set_speed_percent(200.0)
        ack = adapter.apply_motion_batch(
            {
                "batch_id": "scale-once",
                "servo_targets_deg": {"front_left_knee": -60.0},
                "wheel_base_velocity_rad_s": {name: 0.3 for name in WHEEL_JOINT_NAMES},
                "speed_percent_snapshot": 200.0,
            }
        )
        self.assertEqual(adapter.capture_command_state()["servos"]["front_left_knee"], -60.0)
        self.assertAlmostEqual(ack["effective_wheel_velocity_rad_s"]["front_left_ankle"], 0.6)
        self.assertNotAlmostEqual(ack["effective_wheel_velocity_rad_s"]["front_left_ankle"], 1.2)

    def test_legacy_plan_is_canonical_and_raw_fast_signatures_match(self) -> None:
        before = empty_command_state()
        events = [make_event(0.0, "wheel all 0.3"), make_event(2.0, "wheel stop")]
        step = make_step(
            index=1,
            step_type="recorded",
            duration=2.0,
            events=events,
            command_state_before=before,
            command_state_after=before,
            extra={"recorded_speed_percent": 200.0},
        )
        raw = plan_from_steps([step], profile="raw")
        fast = plan_from_steps([step], profile="fast")
        raw_signature = [(row.command, row.wheel_applied_target_rad_s, row.wheel_active_duration_s) for row in raw.events]
        fast_signature = [(row.command, row.wheel_applied_target_rad_s, row.wheel_active_duration_s) for row in fast.events]
        self.assertEqual(raw_signature, fast_signature)
        self.assertEqual(next(row for row in raw.events if row.command.startswith("wheel all")).wheel_applied_target_rad_s, (0.3,))

    def test_playback_segment_uses_atomic_executor(self) -> None:
        before = empty_command_state()
        step = make_step(
            index=1,
            step_type="recorded",
            duration=1.0,
            events=[
                make_event(0.0, "servo front_left_hip 15"),
                make_event(0.0, "wheel all 0.3"),
                make_event(1.0, "wheel stop"),
            ],
            command_state_before=before,
            command_state_after=before,
        )
        adapter = NullSimRobotAdapter()
        calls: list[dict] = []
        original = adapter.apply_motion_batch

        def spy(payload: dict) -> dict:
            calls.append(dict(payload))
            return original(payload)

        adapter.apply_motion_batch = spy  # type: ignore[method-assign]
        service = SimTimePlaybackService()
        plan = plan_from_steps([step], profile="raw")
        self.assertTrue(service.start_plan(plan, current_sim_time_s=0.0, current_wall_time_s=0.0))
        service.update(adapter, current_sim_time_s=0.0, current_sim_step=0, current_wall_time_s=0.0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["servo_targets_deg"])
        self.assertTrue(calls[0]["wheel_base_velocity_rad_s"])


class LightweightRuntimeContractTest(unittest.TestCase):
    def test_protocol_contains_every_refactor_request_and_ack(self) -> None:
        required = {
            "set_height_respawn",
            "recalibrate_ground_reference",
            "set_speed_scale",
            "apply_motion_batch",
            "operation_ack",
            "stop_ack",
        }
        self.assertTrue(required.issubset(MESSAGE_TYPES))
        self.assertNotIn("live_telemetry", MESSAGE_TYPES)

    def test_worker_parser_uses_authoritative_motion_reference_fields(self) -> None:
        args = build_worker_parser().parse_args([])
        reference = load_motion_reference()
        self.assertAlmostEqual(args.default_wheel_speed_rad_s, reference.wheel_reference_velocity_rad_s)
        self.assertAlmostEqual(args.max_wheel_speed_rad_s, reference.wheel_velocity_limit_rad_s)

    def test_light_status_is_small_and_excludes_detailed_payloads(self) -> None:
        args = SimpleNamespace(physics_dt=1.0 / 120.0)
        status = build_common_worker_status(args=args, adapter=NullSimRobotAdapter(), detailed=False)
        payload = encode_message(make_message("status", **status))
        self.assertLess(len(payload), 16 * 1024)
        for forbidden in ("sim_state", "joint_diagnostics", "joint_catalog", "scene_baseline", "latest_worker_log_tail", "final_window_frames"):
            self.assertNotIn(forbidden, status)

    def test_worker_ipc_replaces_stale_status_and_preserves_critical(self) -> None:
        class BlockedSocket:
            def send(self, _payload: bytes) -> int:
                raise BlockingIOError(10035, "would block")

        ipc = WorkerIpc("", 0)
        ipc.sock = BlockedSocket()  # type: ignore[assignment]
        ipc.send(make_message("status", revision=1))
        ipc.send(make_message("status", revision=2))
        ipc.send(make_message("error", error="critical"))
        queued = ([ipc.current_outbound] if ipc.current_outbound else []) + list(ipc.outbound)
        self.assertEqual(sum(row["kind"] == "status" for row in queued), 1)
        self.assertEqual(sum(row["kind"] == "error" for row in queued), 1)
        self.assertGreaterEqual(ipc.status()["status_replaced"], 1)
        self.assertEqual(ipc.status()["socket_send_blocking_ms"], 0.0)

    def test_generate_callback_is_lock_free_and_fast_for_all_heights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            samples = []
            for _index in range(100):
                for height in SUPPORTED_HEIGHTS_MM:
                    controller.current_height_mm = height
                    started = time.perf_counter()
                    controller.generate_or_update_height_obstacle()
                    samples.append((time.perf_counter() - started) * 1000.0)
                    self.assertTrue(controller.operation.idle)
            self.assertLess(statistics.quantiles(samples, n=20)[18], 30.0)
            self.assertLess(max(samples), 100.0)

    def test_speed_callback_is_lock_free_and_parser_runs_5_to_10hz_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            samples = []
            for value in range(0, 301, 3):
                started = time.perf_counter()
                controller.set_speed_percent(value)
                samples.append((time.perf_counter() - started) * 1000.0)
                self.assertTrue(controller.operation.idle)
            self.assertLess(statistics.quantiles(samples, n=20)[18], 30.0)
        args = build_parser().parse_args(["--ui", "--no-sim"])
        normalize_motion_args(args)
        self.assertGreaterEqual(args.sim_status_refresh_ms, 100)
        self.assertLessEqual(args.sim_status_refresh_ms, 200)

    def test_ui_tabs_save_icon_and_debounced_speed_slider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            ui = RealRobotStyleHeightReplayUi(controller)
            ui.root.withdraw()
            try:
                tabs = [str(ui.right_notebook.tab(item, "text")) for item in ui.right_notebook.tabs()]
                self.assertIn("Speed Scale", tabs)
                self.assertNotIn("Vision Auto Replay", tabs)
                self.assertNotIn("Stability Replay", tabs)
                self.assertIn("Save New Version", ui.save_modified_steps_button.cget("text"))
                first_id = None
                for value in range(100):
                    ui._schedule_speed_scale(str(value * 3))
                    first_id = ui.speed_scale_after_id
                self.assertIsNotNone(first_id)
                self.assertEqual(len([key for key in ui.root.tk.call("after", "info") if key == first_id]), 1)
            finally:
                ui._window_close()


class GeometryAndRegressionContractTest(unittest.TestCase):
    def test_fixed_wider_obstacle_geometry_for_all_heights(self) -> None:
        self.assertEqual(OBSTACLE_WIDTH_M, 1.20)
        self.assertGreaterEqual(OBSTACLE_WIDTH_M, float(ENVIRONMENT_REFERENCE["old_obstacle_width_m"]) + 0.20)
        self.assertEqual(OBSTACLE_LENGTH_M, 2.057375557085507)
        self.assertEqual(OBSTACLE_FRONT_FACE_X_M, 0.5213121737735307)
        self.assertAlmostEqual((OBSTACLE_WIDTH_M - ROBOT_COLLISION_WIDTH_M) / 2.0, 0.3794498172320399)
        rows = [
            (height, OBSTACLE_WIDTH_M, OBSTACLE_FRONT_FACE_X_M, 0.0, height / 1000.0)
            for height in SUPPORTED_HEIGHTS_MM
        ]
        self.assertEqual({row[1] for row in rows}, {1.20})
        self.assertEqual({row[2] for row in rows}, {OBSTACLE_FRONT_FACE_X_M})
        self.assertEqual([row[4] for row in rows], [0.05, 0.075, 0.1])

    def test_all_four_knees_still_accept_negative_sixty(self) -> None:
        self.assertEqual(len(KNEE_JOINT_NAMES), 4)
        for name in KNEE_JOINT_NAMES:
            command = validate_motion_command(
                f"servo {name} -60",
                default_wheel_speed_rad_s=0.5235987755982988,
                max_wheel_speed_rad_s=2.0943951023931953,
            )
            self.assertEqual(command.command, f"servo {name} -60")


if __name__ == "__main__":
    unittest.main()
