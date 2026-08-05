from __future__ import annotations

import copy
import hashlib
import inspect
import math
import statistics
import tempfile
import time
import unittest
from pathlib import Path

from command_model import (
    JOINT_COMMAND_SIGN,
    KNEE_JOINT_NAMES,
    SERVO_JOINT_NAMES,
    WHEEL_FORWARD_SIGN,
    WHEEL_JOINT_NAMES,
    command_limits_for_servo,
    validate_motion_command,
)
from height_manifest import SUPPORTED_HEIGHTS_CM
from height_replay_ui import build_parser, normalize_motion_args
from playback import plan_from_steps
from recording_baseline import (
    BASELINE_CONFIG_PATH,
    DEFAULT_RECORDING_WHEEL_VELOCITY_RAD_S,
    baseline_identity,
    load_baseline,
    validate_recording_baseline,
)
from sequence_model import empty_command_state, make_event, make_step
from sim_obstacle_scene import SimSceneConfig, resolve_obstacle_dimensions
from sim_robot_adapter import NullSimRobotAdapter
from sim_transport import SimTransport
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi
from tests.controller_test_utils import make_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _motion_step(*, legacy: bool = False) -> dict:
    state = empty_command_state()
    events = [
        make_event(2.0, "servo front_left_hip 30", command_state_before=state),
        make_event(3.0, "wheel all 0.3", command_state_before=state),
        make_event(4.0, "wheel stop", command_state_before=state),
        make_event(4.1, "wait 0.25", command_state_before=state),
    ]
    step = make_step(
        index=1,
        step_type="recorded",
        duration=5.0,
        events=events,
        command_state_before=state,
        command_state_after=state,
        name="direct_actuator_reference",
    )
    if legacy:
        step["speed" + "_percent"] = 275
        step["preserve" + "_wheel_distance"] = True
        for event in step["events"]:
            event["recorded" + "_speed" + "_percent"] = 50
            event["canonical" + "_velocity_100"] = 99.0
    return step


def _actuator_signature(plan) -> list[tuple]:
    return [
        (
            event.command,
            event.channel,
            event.planned_duration_s,
            event.servo_targets,
            event.servo_base_velocity_deg_s,
            event.wheel_requested_velocity_rad_s,
            event.wheel_applied_target_rad_s,
            event.wheel_active_duration_s,
            event.wheel_displacement,
            event.dispatch_command,
        )
        for event in plan.events
    ]


def _live_status_for_baseline(data: dict) -> dict:
    return {
        "height_cm": 5,
        "joint_catalog": [
            {"joint_name": name, "available": True}
            for name in data["robot"]["joint_order"]
        ],
        "joint_diagnostics": [
            {
                "joint_name": row["articulation_joint"],
                "command_limit_deg": [row["lower_deg"], row["upper_deg"]],
                "target_inside_current_limit": True,
            }
            for row in data["servo_profile"]["joints"]
        ],
        "scene_baseline": {
            "available": True,
            "ground_z_m": data["ground"]["z_m"],
            "obstacle_bottom_z_m": data["ground"]["z_m"],
            "obstacle_front_face_x_m": data["obstacle"]["authoritative_front_face_x_m"],
            "obstacle_length_m": data["obstacle"]["authoritative_length_m"],
            "obstacle_width_m": data["obstacle"]["authoritative_width_m"],
            "wheel_radius_m": data["wheel_actuator"]["wheel_radius_m"],
            "robot_collision_bounds_min_m": [0.0, 0.0, 0.0],
            "robot_root_pose": list(data["respawn"]["grounded_settled_reference_root_pose_wxyz"]),
        },
        "grounded_reference_valid": True,
        "wheel_command": {
            "zero_target_applied": True,
            "physically_stopped": True,
        },
    }


class RemovalAndBaselineLockTest(unittest.TestCase):
    def test_a_one_speed_scale_and_legacy_metadata_is_opaque(self) -> None:
        runtime_files = [
            path
            for path in PROJECT_ROOT.glob("*.py")
            if path.name not in {"recording_baseline_gui_e2e.py"}
        ]
        class_hits = []
        for path in runtime_files:
            source = path.read_text(encoding="utf-8", errors="replace")
            if "class SpeedScale" in source:
                class_hits.append(path.name)
            self.assertNotIn("preserve_wheel_distance", source.lower())
        self.assertEqual(class_hits, ["motion_speed.py"])

        ui_source = inspect.getsource(RealRobotStyleHeightReplayUi._build_right_notebook)
        for tab in (
            "Sim Connection",
            "Run Manager",
            "Record / Servo+Wheel",
            "Speed Scale",
            "Playback",
            "Height Generate",
            "Combine",
            "Sim State",
        ):
            self.assertIn(tab, ui_source)

        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        for option in ("--speed-percent", "--speed-scale", "--playback-speed-multiplier"):
            self.assertNotIn(option, option_strings)

        clean_raw = plan_from_steps([_motion_step()], profile="raw")
        legacy_raw = plan_from_steps([_motion_step(legacy=True)], profile="raw")
        legacy_fast = plan_from_steps([_motion_step(legacy=True)], profile="fast")
        self.assertEqual(_actuator_signature(clean_raw), _actuator_signature(legacy_raw))
        self.assertEqual(_actuator_signature(legacy_raw), _actuator_signature(legacy_fast))
        self.assertLess(legacy_fast.final_time_s, legacy_raw.final_time_s)

    def test_b_stop_priority_rejects_stale_nonzero_fifty_times(self) -> None:
        adapter = NullSimRobotAdapter()
        transport = SimTransport(adapter)
        callback_latencies: list[float] = []
        enqueue_latencies: list[float] = []
        apply_latencies: list[float] = []
        stop_latencies: list[float] = []
        for index in range(50):
            transport.send("wheel all 0.3", source="test")
            old_generation = transport.wheel_generation
            self.assertTrue(any(abs(value) > 0.0 for value in adapter.command_state["wheels"].values()))
            started = time.perf_counter()
            result = transport.stop_wheels(reason=f"priority_test_{index}")
            callback_latencies.append(time.perf_counter() - started)
            status = dict(adapter.wheel_command_status)
            enqueue_latencies.append(max(0.0, status["enqueued_wall_time"] - status["requested_wall_time"]))
            apply_latencies.append(max(0.0, status["target_applied_wall_time"] - status["enqueued_wall_time"]))
            stop_latencies.append(max(0.0, status["measured_stop_wall_time"] - status["target_applied_wall_time"]))
            accepted = adapter.apply_wheel_velocity(
                {name: 0.3 for name in WHEEL_JOINT_NAMES},
                generation=old_generation,
                command_id=f"stale-{index}",
            )
            self.assertFalse(accepted)
            self.assertEqual(result["wheel_generation"], old_generation + 1)
            self.assertTrue(adapter.wheel_command_status["stale_command_rejected"])
            self.assertTrue(all(value == 0.0 for value in adapter.command_state["wheels"].values()))
        self.assertLess(statistics.quantiles(callback_latencies, n=20)[18], 0.05)
        self.assertTrue(all(value >= 0.0 for value in enqueue_latencies + apply_latencies + stop_latencies))

    def test_c_recording_stop_waits_for_zero_ack_and_writes_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("step_record start")
            controller.handle_command("wheel all 0.3")
            controller.handle_command("wheel stop")
            generation_before_finalize = controller.transport.wheel_generation
            controller.handle_command("step_record stop")
            self.assertIsNone(controller.record_stop_pending)
            self.assertIsNotNone(controller.pending_step)
            step = controller.pending_step or {}
            self.assertEqual(step["events"][-1]["command"], "wheel stop")
            self.assertTrue(step["events"][-1]["final_stop_command"])
            self.assertTrue(step["wheel_stop_status"]["zero_target_applied"])
            self.assertTrue(all(value == 0.0 for value in controller.transport.capture_command_state()["wheels"].values()))
            accepted = controller.transport.adapter.apply_wheel_velocity(
                {name: 0.3 for name in WHEEL_JOINT_NAMES},
                generation=generation_before_finalize,
                command_id="recording-stale",
            )
            self.assertFalse(accepted)
            self.assertTrue(all(value == 0.0 for value in controller.transport.capture_command_state()["wheels"].values()))
            motion = step["motion_semantics"]
            self.assertEqual(len(motion["servo_records"]), len(SERVO_JOINT_NAMES))
            self.assertEqual(len(motion["wheel_records"]), len(WHEEL_JOINT_NAMES))

    def test_d_manual_and_playback_use_identical_actuator_values(self) -> None:
        validated_servo = validate_motion_command(
            "servo front_left_hip 30",
            default_wheel_speed_rad_s=DEFAULT_RECORDING_WHEEL_VELOCITY_RAD_S,
            max_wheel_speed_rad_s=2.1,
        )
        validated_wheel = validate_motion_command(
            "wheel all 0.3",
            default_wheel_speed_rad_s=DEFAULT_RECORDING_WHEEL_VELOCITY_RAD_S,
            max_wheel_speed_rad_s=2.1,
        )
        step = _motion_step()
        plan = plan_from_steps([step], profile="raw", max_wheel_speed=2.1)
        commands = [event.command for event in plan.events]
        self.assertIn(validated_servo.command, commands)
        self.assertIn(validated_wheel.command, commands)
        wheel_event = next(event for event in plan.events if event.command == validated_wheel.command)
        self.assertEqual(wheel_event.wheel_requested_velocity_rad_s, (0.3,))
        self.assertEqual(wheel_event.wheel_applied_target_rad_s, (0.3,))
        self.assertAlmostEqual(wheel_event.wheel_active_duration_s, 1.0)

    def test_e_fast_and_raw_have_equal_actuator_signatures(self) -> None:
        raw = plan_from_steps([_motion_step()], profile="raw")
        fast = plan_from_steps([_motion_step()], profile="fast")
        self.assertEqual(_actuator_signature(raw), _actuator_signature(fast))
        self.assertGreater(raw.final_time_s, fast.final_time_s)
        self.assertGreater(raw.timing["final_implicit_idle_s"], fast.timing["final_implicit_idle_s"])

    def test_f_servo_mapping_sign_units_and_knee_limit(self) -> None:
        baseline = load_baseline()
        rows = baseline["servo_profile"]["joints"]
        self.assertEqual([row["ui_name"] for row in rows], SERVO_JOINT_NAMES)
        self.assertEqual([row["articulation_joint"] for row in rows], SERVO_JOINT_NAMES)
        self.assertEqual(baseline["servo_profile"]["ui_unit"], "deg")
        self.assertEqual(baseline["servo_profile"]["internal_unit"], "rad")
        for row in rows:
            name = row["articulation_joint"]
            self.assertEqual(row["direction"], JOINT_COMMAND_SIGN[name])
            self.assertEqual((row["lower_deg"], row["upper_deg"]), command_limits_for_servo(name))
        for name in KNEE_JOINT_NAMES:
            self.assertEqual(command_limits_for_servo(name)[0], -60.0)
            self.assertEqual(validate_motion_command(
                f"servo {name} -60",
                default_wheel_speed_rad_s=0.3,
                max_wheel_speed_rad_s=2.1,
            ).command, f"servo {name} -60")

    def test_g_wheel_mapping_direction_and_units(self) -> None:
        baseline = load_baseline()
        rows = baseline["wheel_actuator"]["joints"]
        self.assertEqual(baseline["wheel_actuator"]["internal_unit"], "rad/s")
        self.assertEqual([row["articulation_joint"] for row in rows], WHEEL_JOINT_NAMES)
        self.assertEqual(
            {row["articulation_joint"]: row["forward_direction"] for row in rows},
            WHEEL_FORWARD_SIGN,
        )
        self.assertEqual({math.copysign(1.0, WHEEL_FORWARD_SIGN[name]) for name in WHEEL_JOINT_NAMES[:2]}, {-1.0, 1.0})
        self.assertEqual(WHEEL_FORWARD_SIGN["front_left_ankle"], WHEEL_FORWARD_SIGN["rear_left_ankle"])
        self.assertEqual(WHEEL_FORWARD_SIGN["front_right_ankle"], WHEEL_FORWARD_SIGN["rear_right_ankle"])
        for value in (-0.3, 0.3):
            command = validate_motion_command(
                f"wheel all {value}",
                default_wheel_speed_rad_s=0.3,
                max_wheel_speed_rad_s=baseline["wheel_actuator"]["velocity_limit_rad_s"],
            )
            self.assertEqual(command.applied_wheel_values_rad_s, (value,))

    def test_h_no_sim_respawn_is_deterministic_ten_times(self) -> None:
        adapter = NullSimRobotAdapter()
        snapshots = []
        for _ in range(10):
            adapter.apply_wheel_velocity({name: 0.3 for name in WHEEL_JOINT_NAMES})
            adapter.respawn_robot()
            snapshots.append(adapter.capture_command_state())
        self.assertTrue(all(row == snapshots[0] for row in snapshots))
        self.assertTrue(all(value == 0.0 for value in snapshots[0]["wheels"].values()))

    def test_i_all_height_geometry_keeps_front_footprint_and_bottom(self) -> None:
        config = SimSceneConfig(
            obstacle_height_m=0.05,
            obstacle_x=1.55,
            obstacle_length=1.65,
            obstacle_width=1.60,
            infer_obstacle_size=False,
            ground_z_m=0.0,
        )
        fake_sim_utils = object()
        rows = []
        for height_cm in SUPPORTED_HEIGHTS_CM:
            config.obstacle_height_m = height_cm / 100.0
            resolve_obstacle_dimensions(config, fake_sim_utils)
            rows.append(
                (
                    config.obstacle_front_x,
                    config.obstacle_x,
                    config.obstacle_width,
                    config.obstacle_length,
                    config.ground_z_m,
                    config.ground_z_m + config.obstacle_height_m,
                )
            )
        self.assertEqual({row[0] for row in rows}, {rows[0][0]})
        self.assertEqual({row[1] for row in rows}, {rows[0][1]})
        self.assertEqual({row[2:5] for row in rows}, {rows[0][2:5]})
        self.assertEqual([row[5] for row in rows], [height / 100.0 for height in SUPPORTED_HEIGHTS_CM])

    def test_j_baseline_hash_is_stable_and_live_mismatch_is_explicit(self) -> None:
        baseline = load_baseline()
        identity_a = baseline_identity(baseline)
        identity_b = baseline_identity(load_baseline(BASELINE_CONFIG_PATH))
        self.assertEqual(identity_a, identity_b)
        changed = copy.deepcopy(baseline)
        changed["wheel_actuator"]["damping"] += 1.0
        self.assertNotEqual(identity_a["baseline_sha256"], baseline_identity(changed)["baseline_sha256"])

        args = build_parser().parse_args([])
        normalize_motion_args(args)
        live = _live_status_for_baseline(baseline)
        passed = validate_recording_baseline(
            baseline,
            args=args,
            worker_status=live,
            output_root=baseline["recording_output_root"],
            no_sim=False,
        )
        self.assertTrue(passed["passed"], passed["mismatches"])
        args.wheel_damping += 1.0
        failed = validate_recording_baseline(
            baseline,
            args=args,
            worker_status=live,
            output_root=baseline["recording_output_root"],
            no_sim=False,
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(any("wheel.damping" in row for row in failed["mismatches"]))

        with tempfile.TemporaryDirectory() as tmp:
            old_recording = Path(tmp) / "old_recording.jsonl"
            old_recording.write_text('{"legacy":true}\n', encoding="utf-8")
            digest_before = hashlib.sha256(old_recording.read_bytes()).hexdigest()
            baseline_identity(baseline)
            digest_after = hashlib.sha256(old_recording.read_bytes()).hexdigest()
            self.assertEqual(digest_before, digest_after)

    def test_k_stop_callback_p95_and_no_background_resend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            calls: list[str] = []
            original = controller.transport.adapter.handle_command

            def spy(message) -> None:
                calls.append(str(getattr(message, "text", message)))
                original(message)

            controller.transport.adapter.handle_command = spy
            controller.handle_command("wheel all 0.3")
            sent_before_updates = len(calls)
            for _ in range(30):
                controller.update()
            self.assertEqual(len(calls), sent_before_updates)
            latencies = []
            for _ in range(100):
                started = time.perf_counter()
                controller.stop_wheels(reason="ui_responsiveness_test")
                latencies.append(time.perf_counter() - started)
            p95 = statistics.quantiles(latencies, n=20)[18]
            self.assertLess(p95, 0.05)
            callback_source = inspect.getsource(RealRobotStyleHeightReplayUi._stop_wheels_ui)
            self.assertNotIn("speed" + "_scale", callback_source)
            self.assertIn("after_cancel", callback_source)


if __name__ == "__main__":
    unittest.main()
