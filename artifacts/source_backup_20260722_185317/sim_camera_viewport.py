"""Fail-soft helpers for showing the onboard camera in Isaac viewports."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass
from typing import Any


CAMERA_VIEWPORT_NAME = "Onboard Camera"
PERSPECTIVE_MODE = "perspective"


@dataclass
class ViewportViewState:
    viewport_id: str = ""
    viewport_name: str = ""
    mode: str = PERSPECTIVE_MODE
    camera_prim_path: str = ""
    was_free_perspective: bool = True
    eye: tuple[float, float, float] | None = None
    target: tuple[float, float, float] | None = None
    active: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraViewportStatus:
    requested: bool = False
    completed: bool = False
    active: bool = False
    supported: bool = False
    mode: str = ""
    main_view_unchanged: bool = False
    camera_prim_path: str = ""
    camera_viewport_name: str = ""
    viewport_name: str = ""
    previous_view_mode: str = ""
    previous_camera_prim_path: str = ""
    requested_action: str = ""
    request_id: str = ""
    request_revision: int = 0
    completed_revision: int = 0
    action_revision: int = 0
    error: str = ""
    pending: bool = False
    retry_count: int = 0
    created_new_window: bool = False
    reused_existing_window: bool = False
    active_fallback_allowed: bool = False
    active_fallback_used: bool = False
    render_only: bool = True
    physics_guard_checked: bool = False
    physics_guard_passed: bool = False
    root_pose_delta_m: float = 0.0
    root_rotation_delta_deg: float = 0.0
    max_joint_delta_rad: float = 0.0
    sim_time_delta: float = 0.0
    sim_steps_delta: int = 0
    ground_classification_before: str = ""
    ground_classification_after: str = ""
    rtf_before: float | None = None
    rtf_after: float | None = None
    fabric_warning_detected: bool = False
    physics_guard: dict[str, Any] | None = None
    main_viewport_id: str = ""
    main_viewport_name: str = ""
    main_camera_path_before: str = ""
    main_camera_path_after: str = ""
    secondary_viewport_name: str = ""
    secondary_camera_path: str = ""
    perspective_restore_verified: bool = False
    perspective_restore_method: str = ""
    postcondition_error: str = ""
    trigger_stage: str = ""
    api_availability: dict[str, bool] | None = None
    api_diagnostics: dict[str, Any] | None = None
    utility_module_path: str = ""
    window_class: str = ""
    viewport_class: str = ""
    window_visible: bool | None = None
    window_docked: bool | None = None
    bound_camera_path: str = ""
    camera_path_verified: bool = False
    frame_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("viewport_name"):
            data["viewport_name"] = data.get("camera_viewport_name", "")
        return data


def default_camera_viewport_status(error: str = "") -> dict[str, Any]:
    return CameraViewportStatus(error=str(error or "")).to_dict()


class CameraViewportManager:
    """Manage an optional second viewport without touching the detector.

    Isaac/Kit imports are intentionally dynamic. Unit tests and --no-sim can
    import this module without having Isaac installed.
    """

    def __init__(self, *, viewport_name: str = CAMERA_VIEWPORT_NAME):
        self.viewport_name = str(viewport_name or CAMERA_VIEWPORT_NAME)
        self.previous_main_view = ViewportViewState()
        self.main_viewport_api: Any | None = None
        self.main_viewport_window: Any | None = None
        self.main_viewport_id = ""
        self.main_viewport_name = ""
        self.main_viewport_camera_path = ""
        self.main_viewport_was_perspective = True
        self.camera_viewport: Any | None = None
        self.camera_window: Any | None = None
        self.camera_window_created_by_project = False
        self.last_status = CameraViewportStatus(error="not requested")
        self.pending_camera_path = ""
        self.pending_request_id = ""
        self.pending_revision = 0
        self.pending_started_at = 0.0
        self.pending_timeout_s = 10.0
        self.pending_max_retries = 30
        self.pending_retry_count = 0
        self.pending_active_fallback_allowed = False

    def open_onboard_camera_viewport(
        self,
        camera_prim_path: str,
        *,
        request_id: str = "",
        action_revision: int = 0,
        active_fallback_allowed: bool = False,
        pending_timeout_s: float = 10.0,
        max_pending_retries: int = 30,
    ) -> CameraViewportStatus:
        camera_path = str(camera_prim_path or "").strip()
        if not camera_path:
            return self._status(
                requested_action="open_camera_viewport",
                request_id=request_id,
                revision=action_revision,
                supported=False,
                active_fallback_allowed=bool(active_fallback_allowed),
                error="camera prim path missing",
            )
        utility, error = _load_viewport_utility()
        if utility is None:
            return self._status(
                requested_action="open_camera_viewport",
                request_id=request_id,
                revision=action_revision,
                camera_prim_path=camera_path,
                supported=False,
                active_fallback_allowed=bool(active_fallback_allowed),
                error=error,
            )
        self._capture_main_viewport_identity(utility)
        active = _safe_call(getattr(utility, "get_active_viewport", None))
        failed_bound_camera_path = ""
        failed_viewport_class = ""
        viewport, window, create_error, created_new, reused_existing = self._get_or_create_camera_viewport(utility, camera_path)
        if viewport is not None:
            ok, set_error = _set_viewport_camera(viewport, camera_path)
            bound_path = _read_viewport_camera(viewport) or _read_window_camera_path(window)
            verified = _camera_path_matches(bound_path, camera_path)
            if ok and verified:
                self._clear_pending()
                self.camera_viewport = viewport
                self.camera_window = window or self.camera_window
                self.camera_window_created_by_project = bool(created_new) or self.camera_window_created_by_project
                return self._status(
                    requested_action="open_camera_viewport",
                    request_id=request_id,
                    revision=action_revision,
                    active=True,
                    supported=True,
                    completed=True,
                    mode="secondary_viewport",
                    main_view_unchanged=True,
                    camera_prim_path=camera_path,
                    camera_viewport_name=_viewport_name(viewport) or self.viewport_name,
                    created_new_window=bool(created_new),
                    reused_existing_window=bool(reused_existing),
                    active_fallback_allowed=bool(active_fallback_allowed),
                    previous_view_mode=self.previous_main_view.mode,
                    previous_camera_prim_path=self.previous_main_view.camera_prim_path,
                    secondary_viewport_name=_viewport_name(viewport) or self.viewport_name,
                    secondary_camera_path=bound_path or camera_path,
                    bound_camera_path=bound_path or "",
                    camera_path_verified=True,
                    window_class=_class_name(window),
                    viewport_class=_class_name(viewport),
                    window_visible=_window_visible(window),
                    window_docked=_window_docked(window),
                    frame_ready=_viewport_frame_ready(viewport, window),
                )
            failed_bound_camera_path = bound_path or ""
            failed_viewport_class = _class_name(viewport)
            create_error = set_error or f"camera path postcondition failed: expected {camera_path!r}, got {bound_path!r}"
        if created_new or window is not None:
            self._set_pending(
                camera_path,
                request_id=request_id,
                revision=action_revision,
                window=window,
                active_fallback_allowed=bool(active_fallback_allowed),
                pending_timeout_s=pending_timeout_s,
                max_pending_retries=max_pending_retries,
                project_created_window=bool(created_new),
            )
            return self._status(
                requested_action="open_camera_viewport",
                request_id=request_id,
                revision=action_revision,
                camera_prim_path=camera_path,
                supported=True,
                completed=False,
                pending=True,
                active=False,
                mode="secondary_viewport_pending",
                main_view_unchanged=True,
                camera_viewport_name=self.viewport_name,
                created_new_window=bool(created_new),
                reused_existing_window=bool(reused_existing),
                active_fallback_allowed=bool(active_fallback_allowed),
                retry_count=self.pending_retry_count,
                previous_view_mode=self.previous_main_view.mode,
                previous_camera_prim_path=self.previous_main_view.camera_prim_path,
                secondary_viewport_name=self.viewport_name,
                secondary_camera_path=camera_path,
                error=create_error or "created viewport window; waiting for viewport API",
            )
        if not bool(active_fallback_allowed):
            return self._status(
                requested_action="open_camera_viewport",
                request_id=request_id,
                revision=action_revision,
                camera_prim_path=camera_path,
                supported=False,
                active=False,
                completed=True,
                mode="secondary_viewport_failed",
                main_view_unchanged=True,
                camera_viewport_name=self.viewport_name,
                created_new_window=bool(created_new),
                reused_existing_window=bool(reused_existing),
                active_fallback_allowed=False,
                secondary_viewport_name=self.viewport_name,
                secondary_camera_path=camera_path,
                bound_camera_path=failed_bound_camera_path,
                camera_path_verified=False,
                viewport_class=failed_viewport_class,
                window_class=_class_name(window),
                error=create_error or "secondary viewport unavailable; active viewport fallback disabled",
            )
        if active is None:
            return self._status(
                requested_action="open_camera_viewport",
                request_id=request_id,
                revision=action_revision,
                camera_prim_path=camera_path,
                supported=False,
                active_fallback_allowed=bool(active_fallback_allowed),
                error=create_error or "active Isaac viewport unavailable",
            )
        ok, set_error = _set_viewport_camera(active, camera_path)
        bound_path = _read_viewport_camera(active)
        verified = _camera_path_matches(bound_path, camera_path)
        if not ok or not verified:
            return self._status(
                requested_action="open_camera_viewport",
                request_id=request_id,
                revision=action_revision,
                camera_prim_path=camera_path,
                supported=False,
                active_fallback_allowed=bool(active_fallback_allowed),
                bound_camera_path=bound_path or "",
                camera_path_verified=False,
                viewport_class=_class_name(active),
                error=create_error or set_error or f"active viewport camera path postcondition failed: expected {camera_path!r}, got {bound_path!r}",
            )
        self._clear_pending()
        return self._status(
            requested_action="open_camera_viewport",
            request_id=request_id,
            revision=action_revision,
            active=True,
            supported=True,
            completed=True,
            mode="active_viewport_fallback",
            main_view_unchanged=False,
            camera_prim_path=camera_path,
            camera_viewport_name=_viewport_name(active) or "active viewport",
            active_fallback_allowed=True,
            active_fallback_used=True,
            previous_view_mode=self.previous_main_view.mode,
            previous_camera_prim_path=self.previous_main_view.camera_prim_path,
            secondary_viewport_name=_viewport_name(active) or "active viewport",
            secondary_camera_path=bound_path or camera_path,
            bound_camera_path=bound_path or "",
            camera_path_verified=True,
            viewport_class=_class_name(active),
            frame_ready=_viewport_frame_ready(active, None),
        )

    def service_pending_camera_viewport(self) -> CameraViewportStatus:
        if not self.pending_camera_path:
            return self.last_status
        self.pending_retry_count += 1
        camera_path = self.pending_camera_path
        utility, error = _load_viewport_utility()
        viewport = _viewport_from_window_or_api(self.camera_window)
        if viewport is None and utility is not None:
            getter = getattr(utility, "get_viewport_from_window_name", None)
            if callable(getter):
                viewport = _viewport_from_window_or_api(_safe_call(getter, self.viewport_name))
        if viewport is not None:
            ok, set_error = _set_viewport_camera(viewport, camera_path)
            bound_path = _read_viewport_camera(viewport) or _read_window_camera_path(self.camera_window)
            verified = _camera_path_matches(bound_path, camera_path)
            if ok and verified:
                self.camera_viewport = viewport
                status = self._status(
                    requested_action="open_camera_viewport",
                    request_id=self.pending_request_id,
                    revision=self.pending_revision,
                    active=True,
                    supported=True,
                    completed=True,
                    pending=False,
                    retry_count=self.pending_retry_count,
                    mode="secondary_viewport",
                    main_view_unchanged=True,
                    camera_prim_path=camera_path,
                    camera_viewport_name=_viewport_name(viewport) or self.viewport_name,
                    reused_existing_window=True,
                    active_fallback_allowed=bool(self.pending_active_fallback_allowed),
                    previous_view_mode=self.previous_main_view.mode,
                    previous_camera_prim_path=self.previous_main_view.camera_prim_path,
                    secondary_viewport_name=_viewport_name(viewport) or self.viewport_name,
                    secondary_camera_path=bound_path or camera_path,
                    bound_camera_path=bound_path or "",
                    camera_path_verified=True,
                    window_class=_class_name(self.camera_window),
                    viewport_class=_class_name(viewport),
                    window_visible=_window_visible(self.camera_window),
                    window_docked=_window_docked(self.camera_window),
                    frame_ready=_viewport_frame_ready(viewport, self.camera_window),
                )
                self._clear_pending()
                return status
            error = set_error or f"camera path postcondition failed: expected {camera_path!r}, got {bound_path!r}"
            if utility is not None:
                getter = getattr(utility, "get_viewport_from_window_name", None)
                if callable(getter):
                    named_viewport = _viewport_from_window_or_api(_safe_call(getter, self.viewport_name))
                    if named_viewport is not None and named_viewport is not viewport:
                        ok, set_error = _set_viewport_camera(named_viewport, camera_path)
                        bound_path = _read_viewport_camera(named_viewport) or _read_window_camera_path(self.camera_window)
                        verified = _camera_path_matches(bound_path, camera_path)
                        if ok and verified:
                            self.camera_viewport = named_viewport
                            status = self._status(
                                requested_action="open_camera_viewport",
                                request_id=self.pending_request_id,
                                revision=self.pending_revision,
                                active=True,
                                supported=True,
                                completed=True,
                                pending=False,
                                retry_count=self.pending_retry_count,
                                mode="secondary_viewport",
                                main_view_unchanged=True,
                                camera_prim_path=camera_path,
                                camera_viewport_name=_viewport_name(named_viewport) or self.viewport_name,
                                reused_existing_window=True,
                                active_fallback_allowed=bool(self.pending_active_fallback_allowed),
                                previous_view_mode=self.previous_main_view.mode,
                                previous_camera_prim_path=self.previous_main_view.camera_prim_path,
                                secondary_viewport_name=_viewport_name(named_viewport) or self.viewport_name,
                                secondary_camera_path=bound_path or camera_path,
                                bound_camera_path=bound_path or "",
                                camera_path_verified=True,
                                window_class=_class_name(self.camera_window),
                                viewport_class=_class_name(named_viewport),
                                window_visible=_window_visible(self.camera_window),
                                window_docked=_window_docked(self.camera_window),
                                frame_ready=_viewport_frame_ready(named_viewport, self.camera_window),
                            )
                            self._clear_pending()
                            return status
                        error = set_error or f"camera path postcondition failed: expected {camera_path!r}, got {bound_path!r}"
        timed_out = False
        if self.pending_started_at > 0.0:
            import time

            timed_out = (time.monotonic() - self.pending_started_at) >= float(self.pending_timeout_s)
        if self.pending_retry_count >= int(self.pending_max_retries):
            timed_out = True
        if timed_out:
            status = self._status(
                requested_action="open_camera_viewport",
                request_id=self.pending_request_id,
                revision=self.pending_revision,
                camera_prim_path=camera_path,
                supported=False,
                completed=True,
                pending=False,
                retry_count=self.pending_retry_count,
                active=False,
                mode="secondary_viewport_timeout",
                main_view_unchanged=True,
                camera_viewport_name=self.viewport_name,
                active_fallback_allowed=bool(self.pending_active_fallback_allowed),
                secondary_viewport_name=self.viewport_name,
                secondary_camera_path=camera_path,
                error=error or "timed out waiting for Onboard Camera viewport API",
            )
            self._clear_pending()
            return status
        return self._status(
            requested_action="open_camera_viewport",
            request_id=self.pending_request_id,
            revision=self.pending_revision,
            camera_prim_path=camera_path,
            supported=True,
            completed=False,
            pending=True,
            retry_count=self.pending_retry_count,
            active=False,
            mode="secondary_viewport_pending",
            main_view_unchanged=True,
            camera_viewport_name=self.viewport_name,
            active_fallback_allowed=bool(self.pending_active_fallback_allowed),
            secondary_viewport_name=self.viewport_name,
            secondary_camera_path=camera_path,
            error=error or "waiting for Onboard Camera viewport API",
        )

    def return_main_view_to_perspective(
        self,
        *,
        scene_handle: Any | None = None,
        request_id: str = "",
        action_revision: int = 0,
    ) -> CameraViewportStatus:
        previous = self.previous_main_view
        active = self.main_viewport_api
        restore_error = ""
        ok = False
        mode = "return_main_view_to_perspective"
        camera_path = ""
        main_camera_path_before = _read_viewport_camera(active) if active is not None else ""
        main_camera_path_after = main_camera_path_before
        method = ""
        postcondition_error = ""
        if active is not None:
            if self.main_viewport_camera_path:
                ok, restore_error = _set_viewport_camera(active, self.main_viewport_camera_path)
                camera_path = self.main_viewport_camera_path
                method = "restore_saved_main_camera_path"
                mode = "restore_saved_main_view"
            else:
                ok, restore_error = _clear_viewport_camera(active)
                method = "clear_saved_main_viewport_camera"
                mode = "return_main_view_to_perspective"
            main_camera_path_after = _read_viewport_camera(active)
        else:
            restore_error = "saved main viewport identity unavailable"
        if not ok:
            sim_ok, sim_error = _restore_sim_default_view(scene_handle)
            ok = sim_ok
            restore_error = "" if sim_ok else (restore_error or sim_error)
            mode = "sim_set_camera_view" if sim_ok else mode
            method = "scene_sim_set_camera_view" if sim_ok else method
            main_camera_path_after = _read_viewport_camera(active) if active is not None else main_camera_path_after
        onboard_path = str(getattr(scene_handle, "camera_prim_path", "") or self.pending_camera_path or "")
        verified = bool(ok) and _restore_postcondition_ok(
            main_camera_path_after,
            expected_path=camera_path,
            onboard_camera_path=onboard_path,
            expected_perspective=bool(self.main_viewport_was_perspective),
        )
        if bool(ok) and not verified:
            postcondition_error = (
                "Perspective restore postcondition failed for saved main viewport; "
                f"after={main_camera_path_after!r} onboard={onboard_path!r}"
            )
            ok = False
        return self._status(
            requested_action="return_main_view_to_perspective",
            request_id=request_id,
            revision=action_revision,
            active=ok,
            supported=ok,
            completed=True,
            mode=mode,
            main_view_unchanged=False,
            camera_prim_path=camera_path,
            camera_viewport_name=_viewport_name(active) if active is not None else "",
            previous_view_mode=previous.mode,
            previous_camera_prim_path=previous.camera_prim_path,
            main_camera_path_before=main_camera_path_before,
            main_camera_path_after=main_camera_path_after,
            secondary_viewport_name=_viewport_name(self.camera_viewport) or self.viewport_name,
            secondary_camera_path=_read_viewport_camera(self.camera_viewport),
            perspective_restore_verified=bool(verified),
            perspective_restore_method=method,
            postcondition_error=postcondition_error,
            error="" if ok else postcondition_error or restore_error or "could not restore Perspective view",
        )

    def restore_previous_view(
        self,
        *,
        scene_handle: Any | None = None,
        request_id: str = "",
        action_revision: int = 0,
    ) -> CameraViewportStatus:
        previous = self.previous_main_view
        if previous.was_free_perspective:
            status = self.return_main_view_to_perspective(
                scene_handle=scene_handle,
                request_id=request_id,
                action_revision=action_revision,
            )
            status.requested_action = "restore_camera_view"
            self.last_status = status
            return status
        main = self.main_viewport_api
        before = _read_viewport_camera(main) if main is not None else ""
        ok = False
        error = ""
        if main is not None and previous.camera_prim_path:
            ok, error = _set_viewport_camera(main, previous.camera_prim_path)
        else:
            error = "saved main viewport identity or previous camera path unavailable"
        after = _read_viewport_camera(main) if main is not None else before
        verified = bool(ok and after == previous.camera_prim_path)
        status = self._status(
            requested_action="restore_camera_view",
            request_id=request_id,
            revision=action_revision,
            active=verified,
            supported=verified,
            completed=True,
            mode="restore_saved_previous_camera",
            main_view_unchanged=False,
            camera_prim_path=previous.camera_prim_path,
            camera_viewport_name=_viewport_name(main) if main is not None else "",
            previous_view_mode=previous.mode,
            previous_camera_prim_path=previous.camera_prim_path,
            main_camera_path_before=before,
            main_camera_path_after=after,
            perspective_restore_verified=verified,
            perspective_restore_method="restore_saved_previous_camera",
            postcondition_error="" if verified else "saved previous camera restore postcondition failed",
            error="" if verified else error or "could not restore saved previous camera",
        )
        self.last_status = status
        return status

    def close_camera_viewport(self, *, request_id: str = "", action_revision: int = 0) -> CameraViewportStatus:
        closed = False
        error = ""
        window = self.camera_window if self.camera_window_created_by_project else None
        for obj in (window,):
            if obj is None:
                continue
            for name in ("destroy", "close", "hide"):
                method = getattr(obj, name, None)
                if callable(method):
                    try:
                        method()
                        closed = True
                        break
                    except Exception as exc:
                        error = str(exc)
            if closed:
                break
        self.camera_window = None
        self.camera_viewport = None
        self.camera_window_created_by_project = False
        self._clear_pending()
        return self._status(
            requested_action="close_camera_viewport",
            request_id=request_id,
            revision=action_revision,
            active=False,
            supported=closed,
            completed=True,
            mode="close_secondary_viewport",
            main_view_unchanged=True,
            camera_viewport_name=self.viewport_name,
            previous_view_mode=self.previous_main_view.mode,
            previous_camera_prim_path=self.previous_main_view.camera_prim_path,
            secondary_viewport_name=self.viewport_name,
            secondary_camera_path="",
            error="" if closed else (error or "no project-created camera viewport to close"),
        )

    def _capture_main_viewport_identity(self, utility: Any) -> ViewportViewState:
        if self.main_viewport_api is not None:
            return self.previous_main_view
        window = _safe_call(getattr(utility, "get_active_viewport_window", None))
        viewport = _viewport_from_window_or_api(window)
        if viewport is None:
            viewport = _safe_call(getattr(utility, "get_active_viewport", None))
        state = _capture_view_state(viewport, fallback_name="active viewport")
        self.previous_main_view = state
        self.main_viewport_api = viewport
        self.main_viewport_window = window
        self.main_viewport_id = state.viewport_id
        self.main_viewport_name = state.viewport_name
        self.main_viewport_camera_path = state.camera_prim_path
        self.main_viewport_was_perspective = bool(state.was_free_perspective)
        return state

    def _get_or_create_camera_viewport(self, utility: Any, camera_path: str) -> tuple[Any | None, Any | None, str, bool, bool]:
        window = None
        viewport = None
        created_new = False
        reused_existing = False
        getter = getattr(utility, "get_viewport_from_window_name", None)
        if callable(getter):
            viewport = _safe_call(getter, self.viewport_name)
            viewport = _viewport_from_window_or_api(viewport)
        if viewport is not None:
            reused_existing = True
            return viewport, None, "", created_new, reused_existing
        creator = getattr(utility, "create_viewport_window", None)
        if not callable(creator):
            return None, None, "create_viewport_window unavailable", created_new, reused_existing
        try:
            camera_arg = _sdf_path(camera_path)
            try:
                window = creator(self.viewport_name, width=800, height=450, camera_path=camera_arg)
            except TypeError:
                try:
                    window = creator(window_name=self.viewport_name, width=800, height=450, camera_path=camera_arg)
                except TypeError:
                    try:
                        window = creator(self.viewport_name, camera_path=camera_arg)
                    except TypeError:
                        try:
                            window = creator(window_name=self.viewport_name, camera_path=camera_arg)
                        except TypeError:
                            try:
                                window = creator(self.viewport_name)
                            except TypeError:
                                window = creator(window_name=self.viewport_name)
            created_new = True
            _show_window(window)
            viewport = _viewport_from_window_or_api(window)
            if viewport is None and callable(getter):
                viewport = _viewport_from_window_or_api(_safe_call(getter, self.viewport_name))
            if viewport is None:
                return None, window, "created viewport window but could not resolve viewport api", created_new, reused_existing
            return viewport, window, "", created_new, reused_existing
        except Exception as exc:
            return None, None, str(exc), created_new, reused_existing

    def _set_pending(
        self,
        camera_path: str,
        *,
        request_id: str,
        revision: int,
        window: Any | None,
        active_fallback_allowed: bool,
        pending_timeout_s: float,
        max_pending_retries: int,
        project_created_window: bool,
    ) -> None:
        import time

        self.pending_camera_path = str(camera_path)
        self.pending_request_id = str(request_id or "")
        self.pending_revision = int(revision or 0)
        self.pending_started_at = time.monotonic()
        self.pending_timeout_s = max(0.05, float(pending_timeout_s or 10.0))
        self.pending_max_retries = max(1, int(max_pending_retries or 30))
        self.pending_retry_count = 0
        self.pending_active_fallback_allowed = bool(active_fallback_allowed)
        self.camera_window = window or self.camera_window
        self.camera_window_created_by_project = bool(project_created_window) or self.camera_window_created_by_project

    def _clear_pending(self) -> None:
        self.pending_camera_path = ""
        self.pending_request_id = ""
        self.pending_revision = 0
        self.pending_started_at = 0.0
        self.pending_retry_count = 0

    def _status(self, **kwargs: Any) -> CameraViewportStatus:
        revision = int(kwargs.pop("revision", 0) or 0)
        completed = bool(kwargs.pop("completed", True))
        status = CameraViewportStatus(
            requested=True,
            completed=completed,
            request_revision=revision,
            completed_revision=revision if completed else 0,
            action_revision=revision,
            **kwargs,
        )
        if status.api_availability is None:
            utility, _error = _load_viewport_utility()
            status.api_availability = _viewport_api_availability(utility)
            status.api_diagnostics = _viewport_api_diagnostics(utility)
            if utility is not None and not status.utility_module_path:
                status.utility_module_path = str(getattr(utility, "__file__", "") or getattr(utility, "__name__", ""))
        if status.api_diagnostics is None:
            utility, _error = _load_viewport_utility()
            status.api_diagnostics = _viewport_api_diagnostics(utility)
        if status.window_visible is None and self.camera_window is not None:
            status.window_visible = _window_visible(self.camera_window)
        if status.window_docked is None and self.camera_window is not None:
            status.window_docked = _window_docked(self.camera_window)
        if status.request_revision and not status.completed_revision:
            status.completed_revision = status.request_revision
        if not status.viewport_name:
            status.viewport_name = status.camera_viewport_name
        if not status.main_viewport_id:
            status.main_viewport_id = self.main_viewport_id
        if not status.main_viewport_name:
            status.main_viewport_name = self.main_viewport_name
        if not status.main_camera_path_before:
            status.main_camera_path_before = self.main_viewport_camera_path
        if not status.secondary_viewport_name:
            status.secondary_viewport_name = status.camera_viewport_name if status.camera_viewport_name == self.viewport_name else self.viewport_name
        if not status.secondary_camera_path:
            status.secondary_camera_path = status.camera_prim_path if status.camera_viewport_name == self.viewport_name else _read_viewport_camera(self.camera_viewport)
        self.last_status = status
        return status


_MANAGER = CameraViewportManager()


def show_camera_in_isaac_viewport(camera_prim_path: str, **payload: Any) -> CameraViewportStatus:
    return _MANAGER.open_onboard_camera_viewport(
        camera_prim_path,
        request_id=str(payload.get("request_id", "") or ""),
        action_revision=int(payload.get("action_revision", payload.get("request_revision", 0)) or 0),
        active_fallback_allowed=bool(payload.get("active_fallback_allowed", payload.get("camera_view_active_fallback", False))),
        pending_timeout_s=float(payload.get("pending_timeout_s", payload.get("camera_view_pending_timeout_s", 10.0)) or 10.0),
        max_pending_retries=int(payload.get("max_pending_retries", payload.get("camera_view_pending_max_retries", 30)) or 30),
    )


def open_onboard_camera_viewport(camera_prim_path: str, **payload: Any) -> CameraViewportStatus:
    return show_camera_in_isaac_viewport(camera_prim_path, **payload)


def return_main_view_to_perspective(scene_handle: Any | None = None, **payload: Any) -> CameraViewportStatus:
    return _MANAGER.return_main_view_to_perspective(
        scene_handle=scene_handle,
        request_id=str(payload.get("request_id", "") or ""),
        action_revision=int(payload.get("action_revision", payload.get("request_revision", 0)) or 0),
    )


def close_onboard_camera_viewport(**payload: Any) -> CameraViewportStatus:
    return _MANAGER.close_camera_viewport(
        request_id=str(payload.get("request_id", "") or ""),
        action_revision=int(payload.get("action_revision", payload.get("request_revision", 0)) or 0),
    )


def restore_previous_isaac_viewport(scene_handle: Any | None = None, **payload: Any) -> CameraViewportStatus:
    return _MANAGER.restore_previous_view(
        scene_handle=scene_handle,
        request_id=str(payload.get("request_id", "") or ""),
        action_revision=int(payload.get("action_revision", payload.get("request_revision", 0)) or 0),
    )


def service_pending_camera_viewport() -> CameraViewportStatus:
    return _MANAGER.service_pending_camera_viewport()


def _load_viewport_utility() -> tuple[Any | None, str]:
    try:
        return importlib.import_module("omni.kit.viewport.utility"), ""
    except Exception as exc:
        return None, f"Isaac viewport utility unavailable: {exc}"


def _viewport_api_availability(utility: Any | None) -> dict[str, bool]:
    names = [
        "get_active_viewport",
        "get_active_viewport_window",
        "get_active_viewport_and_window",
        "get_viewport_from_window_name",
        "get_viewport_window_camera_path",
        "create_viewport_window",
    ]
    result = {name: bool(callable(getattr(utility, name, None))) for name in names}
    result["viewport_window.viewport_api"] = False
    result["set_camera_prim_path"] = False
    result["camera_path"] = False
    if utility is not None:
        active = _safe_call(getattr(utility, "get_active_viewport", None))
        window = _safe_call(getattr(utility, "get_active_viewport_window", None))
        viewport = _viewport_from_window_or_api(window) or active
        result["viewport_window.viewport_api"] = bool(getattr(window, "viewport_api", None) is not None)
        result["set_camera_prim_path"] = callable(getattr(viewport, "set_camera_prim_path", None))
        result["camera_path"] = hasattr(viewport, "camera_path") or hasattr(viewport, "camera_prim_path")
    return result


def _viewport_api_diagnostics(utility: Any | None) -> dict[str, Any]:
    if utility is None:
        return {"available": False}
    names = [
        "get_active_viewport",
        "get_active_viewport_window",
        "get_active_viewport_and_window",
        "get_viewport_from_window_name",
        "get_viewport_window_camera_path",
        "create_viewport_window",
    ]
    functions: dict[str, dict[str, Any]] = {}
    for name in names:
        func = getattr(utility, name, None)
        functions[name] = {
            "available": callable(func),
            "signature": _safe_signature(func),
            "class": _class_name(func),
        }
    return {
        "available": True,
        "module_name": str(getattr(utility, "__name__", "") or ""),
        "module_file": str(getattr(utility, "__file__", "") or ""),
        "functions": functions,
    }


def _safe_signature(func: Any) -> str:
    if not callable(func):
        return ""
    try:
        return str(inspect.signature(func))
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _sdf_path(camera_path: str) -> Any:
    try:
        from pxr import Sdf  # type: ignore

        return Sdf.Path(str(camera_path))
    except Exception:
        return str(camera_path)


def _camera_path_matches(actual: Any, expected: str) -> bool:
    actual_s = str(actual or "")
    expected_s = str(expected or "")
    return actual_s == expected_s or actual_s.rstrip("/") == expected_s.rstrip("/")


def _read_window_camera_path(window: Any) -> str:
    if window is None:
        return ""
    for name in ("camera_path", "camera_prim_path"):
        try:
            value = getattr(window, name)
            if value:
                return str(value)
        except Exception:
            pass
    for name in ("get_camera_prim_path", "get_camera_path"):
        method = getattr(window, name, None)
        if callable(method):
            try:
                value = method()
                if value:
                    return str(value)
            except Exception:
                pass
    return _read_viewport_camera(_viewport_from_window_or_api(window))


def _viewport_frame_ready(viewport: Any, window: Any | None) -> bool:
    for obj in (viewport, window):
        if obj is None:
            continue
        for name in ("frame_ready", "is_frame_ready", "visible"):
            value = getattr(obj, name, None)
            try:
                if callable(value):
                    return bool(value())
                if value is not None:
                    return bool(value)
            except Exception:
                pass
    return False


def _class_name(obj: Any) -> str:
    if obj is None:
        return ""
    return f"{type(obj).__module__}.{type(obj).__name__}"


def _window_visible(window: Any | None) -> bool | None:
    if window is None:
        return None
    for name in ("visible", "shown", "is_visible"):
        value = getattr(window, name, None)
        try:
            if callable(value):
                return bool(value())
            if value is not None:
                return bool(value)
        except Exception:
            pass
    return None


def _window_docked(window: Any | None) -> bool | None:
    if window is None:
        return None
    for name in ("docked", "is_docked"):
        value = getattr(window, name, None)
        try:
            if callable(value):
                return bool(value())
            if value is not None:
                return bool(value)
        except Exception:
            pass
    return None


def _show_window(window: Any | None) -> None:
    if window is None:
        return
    for attr in ("visible", "shown"):
        try:
            if hasattr(window, attr):
                setattr(window, attr, True)
        except Exception:
            pass
    for name in ("show", "focus", "bring_to_front"):
        method = getattr(window, name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _safe_call(func: Any, *args: Any, **kwargs: Any) -> Any | None:
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def _viewport_from_window_or_api(value: Any) -> Any | None:
    if value is None:
        return None
    for attr in ("viewport_api", "viewport", "viewport_api_instance"):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            return candidate
    getter = getattr(value, "get_viewport_api", None)
    if callable(getter):
        try:
            candidate = getter()
            if candidate is not None:
                return candidate
        except Exception:
            pass
    return value


def _viewport_name(viewport: Any) -> str:
    return str(getattr(viewport, "name", "") or getattr(viewport, "viewport_name", "") or "")


def _capture_view_state(viewport: Any, *, fallback_name: str = "") -> ViewportViewState:
    if viewport is None:
        return ViewportViewState(viewport_name=fallback_name, active=False, error="viewport unavailable")
    camera_path = _read_viewport_camera(viewport)
    mode = PERSPECTIVE_MODE if _is_perspective_camera_path(camera_path) else "usd_camera"
    return ViewportViewState(
        viewport_id=str(getattr(viewport, "id", "") or getattr(viewport, "viewport_id", "") or ""),
        viewport_name=_viewport_name(viewport) or fallback_name,
        mode=mode,
        camera_prim_path=camera_path,
        was_free_perspective=mode == PERSPECTIVE_MODE,
        eye=_tuple3(getattr(viewport, "eye", None)),
        target=_tuple3(getattr(viewport, "target", None)),
        active=True,
    )


def _read_viewport_camera(viewport: Any) -> str:
    for name in ("camera_path", "camera_prim_path"):
        value = getattr(viewport, name, "")
        if value:
            return str(value)
    for name in ("get_camera_prim_path", "get_camera_path"):
        getter = getattr(viewport, name, None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                pass
    return ""


def _is_perspective_camera_path(camera_path: str) -> bool:
    path = str(camera_path or "").strip()
    if not path:
        return True
    lowered = path.lower()
    return "persp" in lowered or "perspective" in lowered or lowered.startswith("/omniversekit")


def _restore_postcondition_ok(
    camera_path_after: str,
    *,
    expected_path: str,
    onboard_camera_path: str,
    expected_perspective: bool,
) -> bool:
    after = str(camera_path_after or "")
    expected = str(expected_path or "")
    onboard = str(onboard_camera_path or "")
    if onboard and after == onboard:
        return False
    if expected:
        return after == expected
    if expected_perspective:
        return _is_perspective_camera_path(after)
    return True


def _set_viewport_camera(viewport: Any, camera_path: str) -> tuple[bool, str]:
    setter = getattr(viewport, "set_camera_prim_path", None)
    if callable(setter):
        try:
            setter(str(camera_path))
            return True, ""
        except Exception as exc:
            return False, str(exc)
    for attr in ("camera_path", "camera_prim_path"):
        if hasattr(viewport, attr):
            try:
                setattr(viewport, attr, str(camera_path))
                return True, ""
            except Exception as exc:
                return False, str(exc)
    return False, "viewport does not expose set_camera_prim_path or camera_path"


def _clear_viewport_camera(viewport: Any) -> tuple[bool, str]:
    errors: list[str] = []
    setter = getattr(viewport, "set_camera_prim_path", None)
    if callable(setter):
        for value in ("", None):
            try:
                setter(value)
                return True, ""
            except Exception as exc:
                errors.append(str(exc))
    for attr in ("camera_path", "camera_prim_path"):
        if hasattr(viewport, attr):
            try:
                setattr(viewport, attr, "")
                return True, ""
            except Exception as exc:
                errors.append(str(exc))
    return False, "; ".join(errors) or "viewport camera binding cannot be cleared"


def _restore_sim_default_view(scene_handle: Any | None) -> tuple[bool, str]:
    if scene_handle is None:
        return False, "scene handle unavailable"
    sim = getattr(scene_handle, "sim", None)
    setter = getattr(sim, "set_camera_view", None)
    if not callable(setter):
        return False, "scene sim.set_camera_view unavailable"
    eye = getattr(scene_handle, "default_camera_eye", (1.45, -1.25, 0.80))
    target = getattr(scene_handle, "default_camera_target", (0.45, 0.0, 0.12))
    try:
        setter(eye=list(eye), target=list(target))
        return True, ""
    except TypeError:
        try:
            setter(list(eye), list(target))
            return True, ""
        except Exception as exc:
            return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        values = list(value)
        if len(values) < 3:
            return None
        return (float(values[0]), float(values[1]), float(values[2]))
    except Exception:
        return None
