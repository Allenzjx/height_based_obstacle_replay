from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fsm_50mm_recording_derived_v3.run_fsm50 import (
    _ACTUAL_VIEWPORT_VIDEO_SOURCE,
    _RecordingReplayViewportCapture,
    _apply_recording_artifact_policy,
    _finalize_recording_viewport_video_contract,
    _recording_video_capture_requested,
    _recording_visual_manifest,
    _run_recording_version,
    _write_checksums,
    build_parser,
)
from fsm_50mm_recording_derived_v3.viewport_buffer_video import CAPTURE_BACKEND


_MINIMAL_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


def _write_test_mp4(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MINIMAL_MP4)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_raw_video(run_dir: Path) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = (run_dir / "actual_viewport_video.mp4").resolve()
    video_sha256 = _write_test_mp4(video_path)
    ledger_path = (run_dir / "viewport_frame_ledger.jsonl").resolve()
    render_product_path = "/Render/Product/ActiveViewport"
    ledger_rows = [
        {
            "render_sequence": index,
            "encoded_frame_index": index,
            "sim_step": 8 * (index + 1),
            "sim_time_s": 8 * (index + 1) / 120.0,
            "capture_backend": CAPTURE_BACKEND,
            "render_product_path": render_product_path,
        }
        for index in range(2)
    ]
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    first_frame_path = (run_dir / "viewport_first_frame.png").resolve()
    last_frame_path = (run_dir / "viewport_last_frame.png").resolve()
    first_frame_path.write_bytes(b"first-png")
    last_frame_path.write_bytes(b"last-png")
    return {
        "valid": True,
        "not_camera_video": False,
        "source": _ACTUAL_VIEWPORT_VIDEO_SOURCE,
        "capture_backend": CAPTURE_BACKEND,
        "render_product_path": render_product_path,
        "render_product_unchanged": True,
        "active_render_product_identity_proven": True,
        "capture_graph_created": False,
        "render_observer_only": True,
        "extra_app_update_count": 0,
        "extra_render_count": 0,
        "maximum_pending_captures": 1,
        "frame_count": 2,
        "frame_ledger_complete": True,
        "ledger_path": str(ledger_path),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "full_decode": {
            "valid": True,
            "decoded_frame_count": 2,
            "decoded_width": 1280,
            "decoded_height": 720,
            "decoded_channels": 4,
        },
        "full_decode_all_frames": True,
        "first_frame_path": str(first_frame_path),
        "first_frame_sha256": hashlib.sha256(
            first_frame_path.read_bytes()
        ).hexdigest(),
        "last_frame_path": str(last_frame_path),
        "last_frame_sha256": hashlib.sha256(
            last_frame_path.read_bytes()
        ).hexdigest(),
        "fps": 15.0,
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "error": "",
    }


class _DisabledRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.starts = 0
        self.finalizes = 0

    def start(self) -> None:
        self.starts += 1

    def finalize(self) -> dict[str, object]:
        self.finalizes += 1
        return {
            "valid": False,
            "source": _ACTUAL_VIEWPORT_VIDEO_SOURCE,
            "render_product_path": "",
            "frame_count": 0,
            "video_path": str(self.run_dir / "actual_viewport_video.mp4"),
            "video_sha256": "",
            "error": "viewport capture disabled",
        }


class _FakeActiveViewportRecorder:
    def __init__(self, run_dir: Path, viewport: SimpleNamespace) -> None:
        self.run_dir = run_dir
        self.viewport = viewport
        self.original_path = str(viewport.render_product_path)
        self.render_product_path = self.original_path
        self.error = ""
        self.finalizes = 0

    def start(self) -> bool:
        return True

    def finalize(self) -> dict[str, object]:
        self.finalizes += 1
        return _valid_raw_video(self.run_dir)


class _FakeAdapter:
    def __init__(self) -> None:
        self.observer = None
        self.attach_count = 0
        self.detach_count = 0

    def attach_artifact_render_observer(self, observer) -> None:
        self.attach_count += 1
        self.observer = observer

    def detach_artifact_render_observer(self, observer) -> None:
        self.detach_count += 1
        if self.observer is observer:
            self.observer = None


class RecordingViewportVideoContractTests(unittest.TestCase):
    def test_replay_defaults_to_gui_viewport_capture(self) -> None:
        args = build_parser().parse_args(
            ["replay-recordings", "--trial-id", "1"]
        )
        self.assertFalse(args.headless)
        self.assertFalse(args.no_video)
        self.assertEqual(15.0, args.video_fps)
        self.assertTrue(_recording_video_capture_requested(args))

        for switch in ("--headless", "--no-video"):
            with self.subTest(switch=switch):
                diagnostic = build_parser().parse_args(
                    ["replay-recordings", "--trial-id", "1", switch]
                )
                self.assertFalse(_recording_video_capture_requested(diagnostic))

    def test_valid_actual_viewport_mp4_is_manifested_with_hash_and_contact_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            video = _finalize_recording_viewport_video_contract(
                run_dir,
                _valid_raw_video(run_dir),
                contact_mode="instrumented",
                capture_requested=True,
            )

            self.assertTrue(video["valid"])
            self.assertTrue(video["actual_viewport_video"])
            self.assertFalse(video["not_camera_video"])
            self.assertEqual("instrumented", video["contact_mode"])
            self.assertEqual(
                _ACTUAL_VIEWPORT_VIDEO_SOURCE,
                video["source"],
            )
            manifest_path = Path(str(video["manifest_path"]))
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                video["manifest_sha256"],
            )
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(video["video_path"], persisted["video_path"])
            self.assertEqual(video["video_sha256"], persisted["video_sha256"])

            _write_checksums(run_dir)
            checksum_text = (run_dir / "checksums.sha256").read_text(
                encoding="utf-8"
            )
            self.assertIn("actual_viewport_video.mp4", checksum_text)
            self.assertIn("viewport_frame_ledger.jsonl", checksum_text)
            self.assertIn("viewport_video_manifest.json", checksum_text)

    def test_fake_or_incoherent_video_evidence_is_fail_closed(self) -> None:
        cases = (
            "telemetry_claim",
            "wrong_source",
            "missing_render_product",
            "one_frame",
            "outside_run",
            "bad_sha256",
            "invalid_mp4",
            "wrong_backend",
            "render_product_changed",
            "capture_graph",
            "extra_app_update",
            "ledger_tamper",
            "decode_incomplete",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in cases:
                with self.subTest(case=case):
                    run_dir = root / case / "run"
                    raw = _valid_raw_video(run_dir)
                    if case == "telemetry_claim":
                        raw["not_camera_video"] = True
                    elif case == "wrong_source":
                        raw["source"] = "telemetry_visualization"
                    elif case == "missing_render_product":
                        raw["render_product_path"] = ""
                    elif case == "one_frame":
                        raw["frame_count"] = 1
                    elif case == "outside_run":
                        outside = root / f"outside_{case}.mp4"
                        raw["video_path"] = str(outside.resolve())
                        raw["video_sha256"] = _write_test_mp4(outside)
                    elif case == "bad_sha256":
                        raw["video_sha256"] = "0" * 64
                    elif case == "invalid_mp4":
                        video_path = Path(str(raw["video_path"]))
                        video_path.write_bytes(b"not an mp4")
                        raw["video_sha256"] = hashlib.sha256(
                            video_path.read_bytes()
                        ).hexdigest()
                    elif case == "wrong_backend":
                        raw["capture_backend"] = "png_movie_capture"
                    elif case == "render_product_changed":
                        raw["render_product_unchanged"] = False
                    elif case == "capture_graph":
                        raw["capture_graph_created"] = True
                    elif case == "extra_app_update":
                        raw["extra_app_update_count"] = 1
                    elif case == "ledger_tamper":
                        Path(str(raw["ledger_path"])).write_text(
                            "{}\n", encoding="utf-8"
                        )
                    elif case == "decode_incomplete":
                        raw["full_decode"]["decoded_frame_count"] = 1

                    video = _finalize_recording_viewport_video_contract(
                        run_dir,
                        raw,
                        contact_mode="formal",
                        capture_requested=True,
                    )
                    self.assertFalse(video["valid"])
                    self.assertFalse(video["actual_viewport_video"])
                    self.assertTrue(video["error"])

    def test_no_video_is_diagnostic_and_forces_artifact_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            args = build_parser().parse_args(
                ["replay-recordings", "--trial-id", "1", "--no-video"]
            )
            recorder = _DisabledRecorder(run_dir)
            capture = _RecordingReplayViewportCapture(
                run_dir,
                args,
                contact_mode="instrumented",
                recorder=recorder,
            )
            capture.start()
            video = capture.finalize()
            second = capture.finalize()

            self.assertEqual(1, recorder.starts)
            self.assertEqual(1, recorder.finalizes)
            self.assertEqual(video, second)
            self.assertTrue(video["diagnostic_only"])
            self.assertFalse(video["actual_viewport_video"])

            result = {
                "classification": "FULL_SUCCESS",
                "first_failure_phase": "NONE",
                "strict_full_success": True,
                "strict_success": True,
                "source_integrity": {"ok": True},
            }
            valid = _apply_recording_artifact_policy(
                result,
                video=video,
                visualization={"ok": True},
            )
            self.assertFalse(valid)
            self.assertEqual("ARTIFACT_INVALID", result["classification"])
            self.assertEqual(
                "VIEWPORT_VIDEO_MISSING_OR_INVALID",
                result["first_failure_phase"],
            )
            self.assertFalse(result["strict_full_success"])
            self.assertEqual(
                {"finalized": False, "failed": True, "strict_success": False},
                result["lifecycle"],
            )

    def test_each_version_preserves_active_identity_and_creates_no_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(
                ["replay-recordings", "--trial-id", "1"]
            )
            viewport = SimpleNamespace(
                render_product_path="/Render/Product/ActiveViewport"
            )
            videos: list[dict[str, object]] = []
            recorders: list[_FakeActiveViewportRecorder] = []

            adapters: list[_FakeAdapter] = []
            for version in ("v001", "v002"):
                run_dir = root / version
                recorder = _FakeActiveViewportRecorder(run_dir, viewport)
                adapter = _FakeAdapter()
                adapters.append(adapter)
                recorders.append(recorder)
                capture = _RecordingReplayViewportCapture(
                    run_dir,
                    args,
                    contact_mode="formal",
                    adapter=adapter,
                    recorder=recorder,
                )
                capture.start()
                self.assertIs(adapter.observer, recorder)
                self.assertEqual(
                    "/Render/Product/ActiveViewport",
                    viewport.render_product_path,
                )
                videos.append(capture.finalize())
                self.assertIsNone(adapter.observer)
                self.assertEqual(
                    "/Render/Product/ActiveViewport",
                    viewport.render_product_path,
                )

            self.assertTrue(all(video["actual_viewport_video"] for video in videos))
            self.assertTrue(
                all(video["capture_graph_created"] is False for video in videos)
            )
            self.assertNotEqual(videos[0]["video_path"], videos[1]["video_path"])
            self.assertEqual([1, 1], [recorder.finalizes for recorder in recorders])
            self.assertEqual([1, 1], [adapter.attach_count for adapter in adapters])
            self.assertEqual([1, 1], [adapter.detach_count for adapter in adapters])

    def test_visual_manifest_never_labels_telemetry_as_camera_video(self) -> None:
        manifest = _recording_visual_manifest(
            video={
                "actual_viewport_video": True,
                "video_path": "C:/run/actual_viewport_video.mp4",
                "video_sha256": "a" * 64,
                "manifest_path": "C:/run/viewport_video_manifest.json",
                "manifest_sha256": "b" * 64,
            },
            visualization={"ok": True, "kind": "telemetry"},
            contact_mode="formal",
            artifact_valid=True,
        )
        self.assertTrue(manifest["actual_viewport_video"])
        self.assertFalse(manifest["not_camera_video"])
        self.assertEqual("formal", manifest["contact_mode"])
        self.assertIn("telemetry_visualization", manifest)

    def test_video_finalize_precedes_result_checksums_and_marker(self) -> None:
        source = inspect.getsource(_run_recording_version)
        success_video = source.index("video = video_capture.finalize()")
        success_checksums = source.index("_write_checksums(run_dir)", success_video)
        success_marker = source.index(
            "_mark_artifact_root(artifact_root, valid=artifact_valid)",
            success_checksums,
        )
        self.assertLess(success_video, success_checksums)
        self.assertLess(success_checksums, success_marker)

        failure_video = source.index("video_capture.finalize()", success_video + 1)
        failure_checksums = source.index(
            "_write_checksums(Path(run_dir))", failure_video
        )
        failure_marker = source.index(
            "_mark_artifact_root(artifact_root, valid=False)",
            failure_checksums,
        )
        self.assertLess(failure_video, failure_checksums)
        self.assertLess(failure_checksums, failure_marker)


if __name__ == "__main__":
    unittest.main()
