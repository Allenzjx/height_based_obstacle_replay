from __future__ import annotations

import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from playback import PlaybackManager  # noqa: E402
from sequence_model import apply_events_to_state, coalesce_record_events, empty_command_state, format_step_json, make_event, make_step  # noqa: E402
from sim_ui_controller import HeightReplayController, MODE_PENDING_RECORDED_STEP, MODE_PENDING_REPLACEMENT, MODE_RECORDING_STEP, MODE_TEST  # noqa: E402


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=10,
        sim_launch_mode="disabled",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        global_motion_speed_scale=1.0,
        wheel_speed_scale=1.0,
        servo_command_scale=1.0,
        playback_speed_scale=1.0,
        preserve_wheel_distance=True,
        apply_speed_scale_to_manual=True,
        apply_speed_scale_to_playback=True,
        ui_refresh_ms=100,
        sim_status_refresh_ms=250,
        full_refresh_ms=1000,
        no_continuous_sim_step=False,
        record_event_min_interval_ms=50.0,
        record_event_max_hz=20.0,
        record_coalesce_slider_events=True,
        record_max_events_per_step=2000,
        max_text_widget_chars=2000,
        disable_auto_sim_state_json=False,
        sim_state_json_on_demand=True,
    )


def event_series(count: int, *, command_prefix: str = "servo front_left_hip") -> list[dict[str, object]]:
    events = []
    for index in range(count):
        value = float(index % 90)
        events.append(make_event(index * 0.001, f"{command_prefix} {value:.3f}", kind="slider"))
    return events


def step_with_events(index: int, events: list[dict[str, object]], *, name: str = "step") -> dict[str, object]:
    before = empty_command_state()
    after = apply_events_to_state(before, events)
    return make_step(
        index=index,
        step_type="recorded",
        duration=max([float(event.get("time", 0.0)) for event in events] + [0.0]),
        events=events,
        command_state_before=before,
        command_state_after=after,
        name=f"{name}_{index:03d}",
        note="height=10cm",
        extra={"height_cm": 10},
    )


class AntiFreezeWorkflowTest(unittest.TestCase):
    def test_accept_recording_performance_uses_compact_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.pending_step = step_with_events(1, event_series(1000))
            controller.mode = MODE_PENDING_RECORDED_STEP
            started = time.perf_counter()
            controller.accept_pending_step()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.assertLess(elapsed_ms, 200.0)
            self.assertEqual(controller.manager.count, 1)
            self.assertTrue(controller.manager.dirty)
            self.assertEqual(controller.mode, MODE_TEST)
            self.assertIsNone(controller.pending_step)
            self.assertIn("events=1000", controller.detail_text)
            self.assertNotIn('"events":', controller.detail_text)

    def test_step_details_truncation_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            step = step_with_events(1, event_series(500))
            summary = controller.compact_step_details(step, title="summary")
            self.assertIn("events=500", summary)
            self.assertNotIn('"events":', summary)
            json_text = controller.step_json_text(step, max_chars=1000)
            self.assertTrue(json_text.startswith("[TRUNCATED]"))
            self.assertLess(len(json_text), len(format_step_json(step)))
            controller.manager.add_step(step)
            exported = controller.export_step_json(1)
            self.assertIsNotNone(exported)
            self.assertGreater(len(Path(exported).read_text(encoding="utf-8")), len(json_text))

    def test_record_event_coalescing_preserves_final_state(self) -> None:
        servo_events = event_series(120, command_prefix="servo front_left_hip")
        wheel_events = event_series(80, command_prefix="wheel fl")
        events = servo_events + wheel_events
        coalesced, stats = coalesce_record_events(events, min_interval_s=0.05, max_events=2000)
        self.assertTrue(stats["coalesced"])
        self.assertLess(len(coalesced), len(events))
        original_state = apply_events_to_state(empty_command_state(), events)
        coalesced_state = apply_events_to_state(empty_command_state(), coalesced)
        self.assertEqual(original_state["servos"]["front_left_hip"], coalesced_state["servos"]["front_left_hip"])
        self.assertEqual(original_state["wheels"], coalesced_state["wheels"])

    def test_replace_no_freeze_uses_compact_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(step_with_events(1, event_series(20)))
            controller.pending_replacement = step_with_events(1, event_series(1000, command_prefix="servo front_right_hip"), name="replacement")
            controller.replace_target_index = 1
            controller.mode = MODE_PENDING_REPLACEMENT
            controller.accept_replacement()
            self.assertEqual(controller.manager.count, 1)
            self.assertEqual(controller.mode, MODE_TEST)
            self.assertIsNone(controller.pending_replacement)
            self.assertNotIn('"events":', controller.detail_text)

    def test_playback_batch_limit(self) -> None:
        class FakeController:
            max_wheel_speed = 3.0

            def __init__(self) -> None:
                self.commands: list[str] = []
                self.stopped = False

            def handle_command(self, message: object) -> None:
                self.commands.append(getattr(message, "text"))

            def stop_wheels(self) -> None:
                self.stopped = True

        fake = FakeController()
        playback = PlaybackManager(fake)
        playback.max_events_per_update = 50
        step = step_with_events(1, [make_event(0.0, f"servo front_left_hip {index}", kind="slider") for index in range(1000)])
        self.assertTrue(playback.start_steps([step]))
        playback.update()
        self.assertEqual(len(fake.commands), 50)
        self.assertTrue(playback.active)
        playback.stop()
        self.assertFalse(playback.active)
        self.assertTrue(fake.stopped)

    def test_combine_lazy_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(step_with_events(1, event_series(10, command_prefix="servo front_left_hip")))
            controller.manager.add_step(step_with_events(2, event_series(10, command_prefix="servo front_right_hip")))
            controller.combine_mode_enabled = True
            controller.set_combine_selection([1, 2])
            self.assertIn("Click Preview Combined Step", controller.combine_preview_text)
            self.assertNotIn('"events":', controller.combine_preview_text)
            controller.preview_combine_steps()
            self.assertIn("Combined preview", controller.combine_preview_text)
            self.assertNotIn('"events":', controller.combine_preview_text)
            controller.commit_combine_steps()
            self.assertEqual(controller.manager.count, 1)
            self.assertNotIn('"events":', controller.detail_text)

    def test_state_machine_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            self.assertFalse(controller.can_accept_recorded_step()[0])
            controller.mode = MODE_RECORDING_STEP
            self.assertFalse(controller.can_playback()[0])
            self.assertFalse(controller.can_prepare_replacement()[0])
            controller.mode = MODE_PENDING_RECORDED_STEP
            controller.pending_step = step_with_events(1, event_series(1))
            self.assertTrue(controller.can_accept_recorded_step()[0])
            self.assertFalse(controller.can_playback()[0])
            controller.mode = MODE_TEST
            controller.pending_step = None
            controller.playback.active = True
            self.assertFalse(controller.can_start_recording()[0])

    def test_no_sim_smoke_accept_dummy_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.start_sim_if_needed()
            controller.handle_command("step_record start")
            for index in range(20):
                controller.handle_command(f"servo front_left_hip {index}")
            controller.handle_command("step_record stop")
            self.assertIsNotNone(controller.pending_step)
            controller.handle_command("step_record accept")
            self.assertEqual(controller.manager.count, 1)
            self.assertEqual(controller.mode, MODE_TEST)


if __name__ == "__main__":
    unittest.main()
