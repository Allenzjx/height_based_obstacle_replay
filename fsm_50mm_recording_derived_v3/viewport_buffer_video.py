"""Direct active-viewport MP4 capture without per-frame PNG intermediates.

The recorder is a render observer, not a simulation driver.  Its caller must
invoke :meth:`before_render` immediately before the already-required render and
:meth:`after_render` immediately after it.  The observer never calls
``app.update`` or renders an extra frame.  At most one GPU-to-CPU capture is in
flight, and encoding happens synchronously on the render-calling thread.

Isaac/Kit imports are lazy so lifecycle and evidence contracts remain covered
by pure-Python tests.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from telemetry.exporters import write_json, write_jsonl


CAPTURE_BACKEND = "active_viewport_ldr_byte_buffer_to_omni_videoencoding"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capsule_rgba_bytes(buffer: Any, size: int) -> bytes:
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        payload = bytes(buffer)
        if len(payload) != int(size):
            raise ValueError(
                f"capture buffer has {len(payload)} bytes, expected {int(size)}"
            )
        return payload
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    address = get_pointer(buffer, None)
    if not address:
        raise RuntimeError("PyCapsule_GetPointer returned a null address")
    return ctypes.string_at(address, int(size))


def _strict_rgba8_format(byte_format: Any) -> None:
    try:
        import omni.ui as ui  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in Kit
        raise RuntimeError(f"omni.ui TextureFormat unavailable: {exc}") from exc
    if byte_format != ui.TextureFormat.RGBA8_UNORM:
        raise ValueError(
            "active viewport LdrColor format is not RGBA8_UNORM: "
            f"{byte_format!r}"
        )


def _decode_all_frames(path: Path) -> dict[str, Any]:
    """Decode every encoded frame; metadata alone is not sufficient proof."""

    import imageio.v2 as imageio  # type: ignore

    reader = imageio.get_reader(path)
    count = 0
    shape: tuple[int, int, int] | None = None
    try:
        for frame in reader:
            array = np.asarray(frame)
            if array.ndim != 3 or int(array.shape[2]) not in (3, 4):
                raise ValueError(f"decoded frame {count} has invalid shape {array.shape}")
            current = tuple(int(value) for value in array.shape)
            if shape is None:
                shape = current
            elif current != shape:
                raise ValueError(
                    f"decoded frame {count} shape {current} differs from {shape}"
                )
            count += 1
    finally:
        reader.close()
    return {
        "valid": bool(count > 0 and shape is not None),
        "decoded_frame_count": int(count),
        "decoded_height": None if shape is None else int(shape[0]),
        "decoded_width": None if shape is None else int(shape[1]),
        "decoded_channels": None if shape is None else int(shape[2]),
    }


class ActiveViewportBufferVideoRecorder:
    """One-pending-frame active viewport recorder with a strict frame ledger."""

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool,
        fps: float,
        viewport_provider: Callable[[], Any] | None = None,
        capture_scheduler: Callable[..., Any] | None = None,
        renderer_wait: Callable[[], Any] | None = None,
        encoder_provider: Callable[[], Any] | None = None,
        buffer_copier: Callable[[Any, int], bytes] = _capsule_rgba_bytes,
        format_validator: Callable[[Any], None] = _strict_rgba8_format,
        video_validator: Callable[[Path], dict[str, Any]] = _decode_all_frames,
    ) -> None:
        self.root = Path(root)
        self.video_path = self.root / "actual_viewport_video.mp4"
        self.ledger_path = self.root / "viewport_frame_ledger.jsonl"
        self.manifest_path = self.root / "viewport_buffer_video_manifest.json"
        self.first_frame_path = self.root / "viewport_first_frame.png"
        self.last_frame_path = self.root / "viewport_last_frame.png"
        self.enabled = bool(enabled)
        parsed_fps = float(fps)
        if not math.isfinite(parsed_fps) or parsed_fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        self.fps = parsed_fps
        self.viewport_provider = viewport_provider
        self.capture_scheduler = capture_scheduler
        self.renderer_wait = renderer_wait
        self.encoder_provider = encoder_provider
        self.buffer_copier = buffer_copier
        self.format_validator = format_validator
        self.video_validator = video_validator
        self.viewport: Any | None = None
        self.viewport_identity = 0
        self.encoder: Any | None = None
        self.render_product_path = ""
        self.render_product_unchanged = False
        self.viewport_identity_check_count = 0
        self.started = False
        self._encoding_started = False
        self.finalized = False
        self.error = ""
        self._pending: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._first_frame: tuple[bytes, int, int] | None = None
        self._last_frame: tuple[bytes, int, int] | None = None

    def _load_production_dependencies(self) -> None:
        if self.viewport_provider is None:
            from omni.kit.viewport.utility import get_active_viewport  # type: ignore

            self.viewport_provider = get_active_viewport
        if self.capture_scheduler is None or self.renderer_wait is None:
            import omni.kit.renderer_capture  # type: ignore

            interface = (
                omni.kit.renderer_capture.acquire_renderer_capture_interface()
            )
            if self.capture_scheduler is None:
                self.capture_scheduler = (
                    interface.capture_next_frame_rp_resource_callback
                )
            if self.renderer_wait is None:
                self.renderer_wait = interface.wait_async_capture
        if self.encoder_provider is None:
            from video_encoding import get_video_encoding_interface  # type: ignore

            self.encoder_provider = get_video_encoding_interface

    def start(self) -> bool:
        if self.started:
            return not self.error
        if not self.enabled:
            self.error = "viewport buffer video disabled"
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_production_dependencies()
            assert self.viewport_provider is not None
            assert self.encoder_provider is not None
            self.viewport = self.viewport_provider()
            if self.viewport is None:
                raise RuntimeError("active GUI viewport is unavailable")
            self.viewport_identity = id(self.viewport)
            self.render_product_path = str(
                getattr(self.viewport, "render_product_path", "") or ""
            )
            if not self.render_product_path:
                raise RuntimeError("active viewport render_product_path is unavailable")
            self.encoder = self.encoder_provider()
            if self.encoder is None:
                raise RuntimeError("omni.videoencoding interface is unavailable")
            started = self.encoder.start_encoding(
                str(self.video_path),
                int(round(self.fps)),
                0,
                True,
            )
            if started is not True:
                raise RuntimeError("omni.videoencoding.start_encoding returned false")
            self._encoding_started = True
            self.render_product_unchanged = True
            self.started = True
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def _verify_active_viewport_identity(self, phase: str) -> None:
        """Prove that every captured frame still belongs to the start viewport."""

        if self.viewport_provider is None or self.viewport is None:
            raise RuntimeError("active viewport identity evidence is unavailable")
        active = self.viewport_provider()
        self.viewport_identity_check_count += 1
        if active is not self.viewport or id(active) != self.viewport_identity:
            self.render_product_unchanged = False
            raise RuntimeError(f"active viewport identity changed during {phase}")
        observed_path = str(getattr(active, "render_product_path", "") or "")
        if observed_path != self.render_product_path:
            self.render_product_unchanged = False
            raise RuntimeError(
                "active viewport render_product_path changed during "
                f"{phase}: {observed_path!r} != {self.render_product_path!r}"
            )

    def before_render(self, *, sim_step: int, sim_time_s: float) -> None:
        """Schedule exactly one LdrColor capture for the imminent render."""

        if not self.started or self.finalized or self.error:
            return
        if self._pending is not None:
            self.error = "previous viewport capture is still pending"
            return
        try:
            self._verify_active_viewport_identity("before_render")
            step = int(sim_step)
            time_s = float(sim_time_s)
            if step < 0 or not math.isfinite(time_s):
                raise ValueError("render evidence step/time is invalid")
            self._pending = {
                "render_sequence": len(self._rows),
                "sim_step": step,
                "sim_time_s": time_s,
                "callback_count": 0,
                "capture_bytes": None,
                "width": None,
                "height": None,
                "byte_format": "",
                "capture_resource_identity": 0,
            }

            def _callback(
                buffer: Any,
                buffer_size: int,
                width: int,
                height: int,
                byte_format: Any,
            ) -> None:
                pending = self._pending
                if pending is None:
                    self.error = "viewport callback arrived without a pending render"
                    return
                pending["callback_count"] = int(pending["callback_count"]) + 1
                if int(pending["callback_count"]) != 1:
                    self.error = "viewport render produced more than one callback"
                    return
                try:
                    parsed_width = int(width)
                    parsed_height = int(height)
                    parsed_size = int(buffer_size)
                    if parsed_width < 1 or parsed_height < 1:
                        raise ValueError("captured viewport dimensions are invalid")
                    expected_size = parsed_width * parsed_height * 4
                    if parsed_size != expected_size:
                        raise ValueError(
                            f"RGBA buffer_size={parsed_size}, expected {expected_size}"
                        )
                    self.format_validator(byte_format)
                    captured = self.buffer_copier(buffer, parsed_size)
                    if len(captured) != expected_size:
                        raise ValueError(
                            f"copied RGBA bytes={len(captured)}, expected {expected_size}"
                        )
                    pending["capture_bytes"] = captured
                    pending["width"] = parsed_width
                    pending["height"] = parsed_height
                    pending["byte_format"] = str(byte_format)
                except Exception as exc:
                    self.error = f"viewport callback failed: {type(exc).__name__}: {exc}"

            # ByteCapture is a two-stage request: its viewport delegate runs
            # during the drawable event and only then asks renderer-capture
            # for the *next* frame.  Waiting immediately after the same render
            # therefore observes zero callbacks.  Request the active
            # viewport's current LdrColor RpResource directly before the
            # caller's one existing render, then drain that exact request in
            # ``after_render``.  No extra app.update or render is introduced.
            hydra_texture = getattr(self.viewport, "_hydra_texture", None)
            if hydra_texture is None:
                raise RuntimeError("active viewport HydraTexture is unavailable")
            resource_getter = getattr(
                hydra_texture, "get_drawable_ldr_resource", None
            ) or getattr(hydra_texture, "_get_drawable_ldr_resource", None)
            if not callable(resource_getter):
                raise RuntimeError(
                    "active viewport LdrColor RpResource getter is unavailable"
                )
            resource = resource_getter(0)
            if resource is None:
                raise RuntimeError("active viewport LdrColor RpResource is unavailable")
            self._pending["capture_resource_identity"] = int(id(resource))
            assert self.capture_scheduler is not None
            self.capture_scheduler(_callback, resource)
        except Exception as exc:
            self.error = f"viewport schedule failed: {type(exc).__name__}: {exc}"

    def after_render(self) -> None:
        """Drain the one pending GPU copy and encode it on this same thread."""

        if not self.started or self.finalized or self.error:
            return
        pending = self._pending
        if pending is None:
            self.error = "render completed without a scheduled viewport capture"
            return
        try:
            assert self.renderer_wait is not None
            self.renderer_wait()
            self._verify_active_viewport_identity("after_render")
            if self.error:
                # The asynchronous callback already recorded the precise
                # evidence failure.  Do not obscure it with follow-on shape
                # errors while handling the rejected frame.
                self._pending = None
                return
            if int(pending["callback_count"]) != 1:
                raise RuntimeError(
                    f"viewport callback_count={pending['callback_count']}, expected 1"
                )
            payload = pending["capture_bytes"]
            width = int(pending["width"])
            height = int(pending["height"])
            if not isinstance(payload, bytes):
                raise RuntimeError("viewport callback did not provide copied bytes")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                height, width, 4
            )
            assert self.encoder is not None
            encoded = self.encoder.encode_next_frame_from_buffer(
                frame, width, height
            )
            if encoded is not True:
                raise RuntimeError(
                    "omni.videoencoding.encode_next_frame_from_buffer returned false"
                )
            stored = (payload, width, height)
            if self._first_frame is None:
                self._first_frame = stored
            self._last_frame = stored
            self._rows.append(
                {
                    "render_sequence": int(pending["render_sequence"]),
                    "sim_step": int(pending["sim_step"]),
                    "sim_time_s": float(pending["sim_time_s"]),
                    "encoded_frame_index": len(self._rows),
                    "width": width,
                    "height": height,
                    "rgba_buffer_size": len(payload),
                    "byte_format": str(pending["byte_format"]),
                    "capture_backend": CAPTURE_BACKEND,
                    "render_product_path": self.render_product_path,
                    "viewport_identity": self.viewport_identity,
                }
            )
            self._pending = None
        except Exception as exc:
            self.error = f"viewport encode failed: {type(exc).__name__}: {exc}"

    @staticmethod
    def _write_checkpoint(path: Path, captured: tuple[bytes, int, int]) -> None:
        import imageio.v2 as imageio  # type: ignore

        payload, width, height = captured
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 4)
        imageio.imwrite(path, frame)

    def finalize(self) -> dict[str, Any]:
        if self.finalized:
            try:
                return dict(
                    __import__("json").loads(
                        self.manifest_path.read_text(encoding="utf-8")
                    )
                )
            except Exception:
                return {"valid": False, "error": self.error or "manifest unavailable"}
        self.finalized = True
        if self.started and not self.error:
            try:
                self._verify_active_viewport_identity("finalize")
            except Exception as exc:
                self.error = f"viewport identity failed: {type(exc).__name__}: {exc}"
        if self._pending is not None and not self.error:
            self.error = "video finalized with a pending viewport capture"
        if self.encoder is not None and self._encoding_started:
            try:
                self.encoder.finalize_encoding()
            except Exception as exc:
                self.error = self.error or (
                    f"video finalize failed: {type(exc).__name__}: {exc}"
                )
        checkpoint_error = ""
        try:
            if self._first_frame is not None:
                self._write_checkpoint(self.first_frame_path, self._first_frame)
            if self._last_frame is not None:
                self._write_checkpoint(self.last_frame_path, self._last_frame)
        except Exception as exc:
            checkpoint_error = f"checkpoint PNG failed: {type(exc).__name__}: {exc}"
            self.error = self.error or checkpoint_error
        write_jsonl(self.ledger_path, self._rows)
        decode: dict[str, Any] = {
            "valid": False,
            "decoded_frame_count": 0,
            "decoded_width": None,
            "decoded_height": None,
            "decoded_channels": None,
        }
        if self.video_path.is_file() and self.video_path.stat().st_size > 0:
            try:
                decode = dict(self.video_validator(self.video_path))
            except Exception as exc:
                self.error = self.error or (
                    f"full video decode failed: {type(exc).__name__}: {exc}"
                )
        frame_count = len(self._rows)
        decode_matches = bool(
            decode.get("valid") is True
            and int(decode.get("decoded_frame_count", -1)) == frame_count
            and frame_count >= 2
        )
        if not decode_matches and not self.error:
            self.error = "decoded MP4 frame count does not match the capture ledger"
        frame_ledger_complete = bool(
            frame_count >= 2
            and [int(row["render_sequence"]) for row in self._rows]
            == list(range(frame_count))
            and [int(row["encoded_frame_index"]) for row in self._rows]
            == list(range(frame_count))
        )
        active_render_product_identity_proven = bool(
            self.render_product_unchanged
            and self.viewport_identity_check_count >= (2 * frame_count + 1)
        )
        valid = bool(
            self.started
            and not self.error
            and frame_count >= 2
            and decode_matches
            and frame_ledger_complete
            and active_render_product_identity_proven
            and self.video_path.is_file()
            and self.video_path.stat().st_size > 0
            and self.first_frame_path.is_file()
            and self.last_frame_path.is_file()
        )
        manifest = {
            "schema_version": "fsm50.active_viewport_buffer_video.v1",
            "valid": valid,
            "artifact_valid": valid,
            "actual_viewport_video": valid,
            "not_camera_video": False,
            "capture_backend": CAPTURE_BACKEND,
            "source": "actual_active_isaac_gui_viewport_render_product",
            "render_product_path": self.render_product_path,
            "viewport_identity": self.viewport_identity,
            "viewport_identity_check_count": self.viewport_identity_check_count,
            "render_product_unchanged": bool(self.render_product_unchanged),
            "active_render_product_identity_proven": (
                active_render_product_identity_proven
            ),
            "capture_graph_created": False,
            "extra_app_update_count": 0,
            "extra_render_count": 0,
            "render_observer_only": True,
            "maximum_pending_captures": 1,
            "fps": self.fps,
            "frame_count": frame_count,
            "frame_ledger_complete": frame_ledger_complete,
            "ledger_path": str(self.ledger_path),
            "ledger_sha256": _sha256(self.ledger_path),
            "video_path": str(self.video_path),
            "video_sha256": _sha256(self.video_path) if self.video_path.is_file() else "",
            "video_size": self.video_path.stat().st_size if self.video_path.is_file() else 0,
            "first_frame_path": str(self.first_frame_path),
            "first_frame_sha256": (
                _sha256(self.first_frame_path) if self.first_frame_path.is_file() else ""
            ),
            "last_frame_path": str(self.last_frame_path),
            "last_frame_sha256": (
                _sha256(self.last_frame_path) if self.last_frame_path.is_file() else ""
            ),
            "full_decode": decode,
            "full_decode_all_frames": bool(decode_matches),
            "error": self.error,
            "checkpoint_error": checkpoint_error,
        }
        write_json(self.manifest_path, manifest)
        return manifest
