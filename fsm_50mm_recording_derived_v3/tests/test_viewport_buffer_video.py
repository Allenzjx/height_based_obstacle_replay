from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsm_50mm_recording_derived_v3.viewport_buffer_video import (  # noqa: E402
    CAPTURE_BACKEND,
    ActiveViewportBufferVideoRecorder,
)
from sim_robot_adapter import SimRobotAdapter  # noqa: E402


class _FakeEncoder:
    def __init__(self, *, start_ok: bool = True) -> None:
        self.start_ok = start_ok
        self.path: Path | None = None
        self.start_args = None
        self.frames: list[np.ndarray] = []
        self.finalize_calls = 0

    def start_encoding(self, path, fps, nframes, overwrite):
        self.path = Path(path)
        self.start_args = (path, fps, nframes, overwrite)
        return self.start_ok

    def encode_next_frame_from_buffer(self, frame, width=0, height=0):
        self.last_dimensions = (width, height)
        self.frames.append(np.asarray(frame).copy())
        return True

    def finalize_encoding(self):
        self.finalize_calls += 1
        if self.path is not None and self.start_ok:
            self.path.write_bytes(b"synthetic-mp4")


class _FakeCapturePipeline:
    def __init__(self, *, width: int = 2, height: int = 1) -> None:
        self.width = width
        self.height = height
        self.pending = None
        self.schedule_calls = 0
        self.wait_calls = 0
        self.callback_enabled = True
        self.size_delta = 0

    def schedule(self, callback, resource):
        self.schedule_calls += 1
        self.pending = callback
        self.resource = resource

    def wait(self):
        self.wait_calls += 1
        callback, self.pending = self.pending, None
        if callback is not None and self.callback_enabled:
            payload = bytes(range(self.width * self.height * 4))
            callback(
                payload,
                len(payload) + self.size_delta,
                self.width,
                self.height,
                "RGBA8_UNORM",
            )


class _FakeSimulation:
    def __init__(self, *, substeps: int) -> None:
        self.substeps = substeps
        self.step_calls: list[object] = []
        self.render_calls = 0

    def get_physics_dt(self):
        return 1.0 / 120.0

    def step(self, render=None):
        self.step_calls.append(render)

    def render(self):
        self.render_calls += 1


class _FakeRobot:
    def __init__(self) -> None:
        self.write_calls = 0
        self.update_calls = 0

    def write_data_to_sim(self):
        self.write_calls += 1

    def update(self, _dt):
        self.update_calls += 1


def _adapter(*, substeps: int) -> SimRobotAdapter:
    adapter = object.__new__(SimRobotAdapter)
    adapter.sim = _FakeSimulation(substeps=substeps)
    adapter.robot = _FakeRobot()
    adapter.sim_time = 0.0
    adapter.sim_steps = 0
    adapter.telemetry_collector = None
    adapter.artifact_render_observer = None
    adapter.artifact_render_observer_errors = []
    adapter._render_step_timing = lambda: (substeps / 120.0, substeps)
    adapter._advance_servo_targets = lambda _dt: None
    adapter.apply_commands_to_robot = lambda: None
    adapter._update_wheel_stop_measurement = lambda: None
    return adapter


def _recorder(root: Path, pipeline: _FakeCapturePipeline, encoder: _FakeEncoder):
    resource = object()
    viewport = SimpleNamespace(
        render_product_path="/Render/ActiveViewport",
        _hydra_texture=SimpleNamespace(
            get_drawable_ldr_resource=lambda _handle=0: resource
        ),
    )
    return ActiveViewportBufferVideoRecorder(
        root,
        enabled=True,
        fps=15.0,
        viewport_provider=lambda: viewport,
        capture_scheduler=pipeline.schedule,
        renderer_wait=pipeline.wait,
        encoder_provider=lambda: encoder,
        format_validator=lambda value: (
            None
            if value == "RGBA8_UNORM"
            else (_ for _ in ()).throw(ValueError("wrong format"))
        ),
        video_validator=lambda _path: {
            "valid": True,
            "decoded_frame_count": len(encoder.frames),
            "decoded_width": pipeline.width,
            "decoded_height": pipeline.height,
            "decoded_channels": 4,
        },
    )


class ActiveViewportBufferVideoRecorderTests(unittest.TestCase):
    def test_direct_buffer_path_encodes_one_frame_per_existing_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            recorder = _recorder(root, pipeline, encoder)
            self.assertTrue(recorder.start())
            self.assertEqual(encoder.start_args[1:], (15, 0, True))
            for index in range(3):
                recorder.before_render(sim_step=8 * (index + 1), sim_time_s=index / 15)
                recorder.after_render()
            manifest = recorder.finalize()

            self.assertTrue(manifest["valid"], manifest)
            self.assertEqual(pipeline.schedule_calls, 3)
            self.assertEqual(pipeline.wait_calls, 3)
            self.assertIsNotNone(pipeline.resource)
            self.assertEqual(len(encoder.frames), 3)
            self.assertEqual(encoder.last_dimensions, (2, 1))
            self.assertTrue(all(frame.shape == (1, 2, 4) for frame in encoder.frames))
            self.assertEqual(encoder.finalize_calls, 1)
            self.assertEqual(manifest["frame_count"], 3)
            self.assertEqual(manifest["full_decode"]["decoded_frame_count"], 3)
            self.assertEqual(manifest["capture_backend"], CAPTURE_BACKEND)
            self.assertEqual(manifest["extra_app_update_count"], 0)
            self.assertEqual(manifest["maximum_pending_captures"], 1)
            self.assertTrue(manifest["render_product_unchanged"])
            self.assertEqual(manifest["viewport_identity_check_count"], 7)
            self.assertTrue(Path(manifest["first_frame_path"]).is_file())
            self.assertTrue(Path(manifest["last_frame_path"]).is_file())
            rows = [
                json.loads(line)
                for line in recorder.ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["sim_step"] for row in rows], [8, 16, 24])
            self.assertEqual([row["render_sequence"] for row in rows], [0, 1, 2])

    def test_active_viewport_identity_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            viewport = SimpleNamespace(
                render_product_path="/Render/ActiveViewport",
                _hydra_texture=SimpleNamespace(
                    get_drawable_ldr_resource=lambda _handle=0: object()
                ),
            )
            active = [viewport]
            recorder = ActiveViewportBufferVideoRecorder(
                Path(temporary),
                enabled=True,
                fps=15.0,
                viewport_provider=lambda: active[0],
                capture_scheduler=pipeline.schedule,
                renderer_wait=pipeline.wait,
                encoder_provider=lambda: encoder,
                format_validator=lambda _value: None,
                video_validator=lambda _path: {
                    "valid": True,
                    "decoded_frame_count": len(encoder.frames),
                    "decoded_width": pipeline.width,
                    "decoded_height": pipeline.height,
                    "decoded_channels": 4,
                },
            )
            self.assertTrue(recorder.start())
            recorder.before_render(sim_step=8, sim_time_s=8.0 / 120.0)
            active[0] = SimpleNamespace(
                render_product_path="/Render/OtherViewport",
                _hydra_texture=viewport._hydra_texture,
            )
            recorder.after_render()
            manifest = recorder.finalize()
            self.assertFalse(manifest["valid"])
            self.assertFalse(manifest["render_product_unchanged"])
            self.assertIn("identity changed", manifest["error"])

    def test_render_product_path_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            viewport = SimpleNamespace(
                render_product_path="/Render/ActiveViewport",
                _hydra_texture=SimpleNamespace(
                    get_drawable_ldr_resource=lambda _handle=0: object()
                ),
            )
            recorder = ActiveViewportBufferVideoRecorder(
                Path(temporary),
                enabled=True,
                fps=15.0,
                viewport_provider=lambda: viewport,
                capture_scheduler=pipeline.schedule,
                renderer_wait=pipeline.wait,
                encoder_provider=lambda: encoder,
                format_validator=lambda _value: None,
                video_validator=lambda _path: {},
            )
            self.assertTrue(recorder.start())
            viewport.render_product_path = "/Render/Rebound"
            recorder.before_render(sim_step=8, sim_time_s=8.0 / 120.0)
            manifest = recorder.finalize()
            self.assertFalse(manifest["valid"])
            self.assertFalse(manifest["render_product_unchanged"])
            self.assertIn("render_product_path changed", manifest["error"])

    def test_one_pending_capture_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            recorder = _recorder(Path(temporary), pipeline, _FakeEncoder())
            self.assertTrue(recorder.start())
            recorder.before_render(sim_step=1, sim_time_s=0.0)
            recorder.before_render(sim_step=2, sim_time_s=0.1)
            self.assertIn("still pending", recorder.error)
            manifest = recorder.finalize()
            self.assertFalse(manifest["valid"])

    def test_missing_callback_fails_instead_of_dropping_a_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            pipeline.callback_enabled = False
            recorder = _recorder(Path(temporary), pipeline, _FakeEncoder())
            self.assertTrue(recorder.start())
            recorder.before_render(sim_step=1, sim_time_s=0.0)
            recorder.after_render()
            self.assertIn("callback_count=0", recorder.error)
            self.assertFalse(recorder.finalize()["valid"])

    def test_missing_live_ldr_resource_fails_before_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            viewport = SimpleNamespace(
                render_product_path="/Render/ActiveViewport",
                _hydra_texture=SimpleNamespace(
                    get_drawable_ldr_resource=lambda _handle=0: None
                ),
            )
            recorder = ActiveViewportBufferVideoRecorder(
                Path(temporary),
                enabled=True,
                fps=15.0,
                viewport_provider=lambda: viewport,
                capture_scheduler=pipeline.schedule,
                renderer_wait=pipeline.wait,
                encoder_provider=lambda: encoder,
                format_validator=lambda _value: None,
                video_validator=lambda _path: {},
            )
            self.assertTrue(recorder.start())
            recorder.before_render(sim_step=1, sim_time_s=0.0)
            self.assertIn("LdrColor RpResource is unavailable", recorder.error)
            self.assertEqual(pipeline.schedule_calls, 0)
            self.assertFalse(recorder.finalize()["valid"])

    def test_invalid_rgba_buffer_size_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            pipeline.size_delta = -1
            recorder = _recorder(Path(temporary), pipeline, _FakeEncoder())
            self.assertTrue(recorder.start())
            recorder.before_render(sim_step=1, sim_time_s=0.0)
            recorder.after_render()
            self.assertIn("buffer_size", recorder.error)
            self.assertFalse(recorder.finalize()["valid"])

    def test_encoder_start_failure_never_claims_valid_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            recorder = _recorder(
                Path(temporary), pipeline, _FakeEncoder(start_ok=False)
            )
            self.assertFalse(recorder.start())
            recorder.before_render(sim_step=1, sim_time_s=0.0)
            recorder.after_render()
            manifest = recorder.finalize()
            self.assertFalse(manifest["valid"])
            self.assertIn("start_encoding", manifest["error"])

    def test_decoded_frame_count_must_equal_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            recorder = _recorder(Path(temporary), pipeline, encoder)
            recorder.video_validator = lambda _path: {
                "valid": True,
                "decoded_frame_count": 1,
                "decoded_width": 2,
                "decoded_height": 1,
                "decoded_channels": 4,
            }
            self.assertTrue(recorder.start())
            for index in range(2):
                recorder.before_render(sim_step=index, sim_time_s=float(index))
                recorder.after_render()
            manifest = recorder.finalize()
            self.assertFalse(manifest["valid"])
            self.assertIn("frame count", manifest["error"])

    def test_adapter_observes_existing_render_without_extra_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            recorder = _recorder(Path(temporary), pipeline, encoder)
            adapter = _adapter(substeps=8)
            self.assertTrue(recorder.start())
            adapter.attach_artifact_render_observer(recorder)
            adapter.step()
            adapter.step()
            adapter.detach_artifact_render_observer(recorder)
            manifest = recorder.finalize()

            self.assertTrue(manifest["valid"], manifest)
            self.assertEqual(adapter.sim.step_calls, [False] * 16)
            self.assertEqual(adapter.sim.render_calls, 2)
            self.assertEqual(adapter.sim_steps, 16)
            self.assertEqual([row.shape for row in encoder.frames], [(1, 2, 4)] * 2)
            ledger = [
                json.loads(line)
                for line in recorder.ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["sim_step"] for row in ledger], [8, 16])
            self.assertEqual(manifest["extra_app_update_count"], 0)

    def test_single_substep_render_is_bracketed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = _FakeCapturePipeline()
            encoder = _FakeEncoder()
            recorder = _recorder(Path(temporary), pipeline, encoder)
            adapter = _adapter(substeps=1)
            self.assertTrue(recorder.start())
            adapter.attach_artifact_render_observer(recorder)
            adapter.step()
            adapter.step()
            adapter.detach_artifact_render_observer(recorder)
            manifest = recorder.finalize()
            self.assertTrue(manifest["valid"], manifest)
            self.assertEqual(adapter.sim.step_calls, [None, None])
            self.assertEqual(adapter.sim.render_calls, 0)
            ledger = [
                json.loads(line)
                for line in recorder.ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["sim_step"] for row in ledger], [1, 2])


if __name__ == "__main__":
    unittest.main()
