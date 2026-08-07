"""Visible one-worker Run-management acceptance using only a temporary store."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

from operation_coordinator import OperationState
from sequence_model import empty_command_state, make_event, make_step
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


def _step(target_deg: float, note: str) -> dict[str, Any]:
    before = empty_command_state()
    after = copy.deepcopy(before)
    after["servos"]["front_left_hip"] = float(target_deg)
    row = make_step(
        index=1,
        step_type="run_management_e2e",
        duration=0.25,
        events=[make_event(0.0, f"servo front_left_hip {target_deg:.3f}")],
        command_state_before=before,
        command_state_after=after,
        name="temporary_run_management_step",
    )
    row["note"] = note
    return row


class RunManagementGuiE2E(RefactorGuiE2E):
    def __init__(self, output_dir: Path, *, timeout_s: float) -> None:
        super().__init__(output_dir, timeout_s=timeout_s)
        self.result = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "single_worker_requested": True,
            "temporary_version_root": str(self.version_root),
            "formal_recordings_read_only": True,
            "stages": [],
            "screenshots": {},
            "run_management_rows": [],
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }
        self.run_a = ""
        self.run_b = ""
        self.run_c = ""
        self.update_started = False
        self.save_as_started = False
        self.update_before: dict[str, Any] = {}
        self.save_as_before: dict[str, Any] = {}
        self.version_count_before = 0

    def _tick(self) -> None:
        if self.ui._closing:
            return
        if bool(getattr(self, "_e2e_tick_active", False)):
            return
        self._e2e_tick_active = True
        try:
            if time.monotonic() - self.started > self.timeout_s:
                raise TimeoutError(f"overall timeout at {self.stage}")
            if self.stage != "WAIT_READY" and time.monotonic() - self.stage_started > 120.0:
                raise TimeoutError(f"stage timeout: {self.stage}")
            phase = str(self.controller.latest_sim_status.get("phase", "") or "")
            if phase in {"preflight_failed", "launch_plan_failed", "process_spawn_failed", "runtime_failed"}:
                raise RuntimeError(str(self.controller.latest_sim_status.get("error", phase)))
            self._sample_worker()
            getattr(self, f"_stage_{self.stage.lower()}")()
        except Exception as exc:
            self._fail(exc)
        finally:
            self._e2e_tick_active = False
        if not self.ui._closing:
            self.ui.root.after(50, self._tick)

    def _record(self, action: str, **values: Any) -> None:
        self.result["run_management_rows"].append({"action": action, **copy.deepcopy(values)})

    def _select_combo_run(self, run_id: str) -> None:
        panel = self.ui.height_generate_panel
        panel._sync_run_choices()
        label = next(
            (label for label, candidate in panel._version_labels.items() if candidate == run_id),
            "",
        )
        if not label:
            raise AssertionError(f"Run {run_id} is missing from the visible combobox")
        panel.run_var.set(label)
        # Exercise the exact callback bound to <<ComboboxSelected>>. Calling it
        # synchronously keeps the staged E2E deterministic while retaining the
        # visible combobox selection and its production selection-only path.
        panel._select_pending_run()
        self.ui.root.update_idletasks()

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        self.controller.current_height_mm = 50
        self.controller.manager.adopt_steps([_step(4.0, "seed-a")], dirty=True)
        self.controller.save_new_version(version_name="run-management-a")
        self.run_a = self.controller.current_version_id
        self.controller.manager.adopt_steps([_step(7.0, "seed-b")], dirty=True)
        self.controller.save_new_version(version_name="run-management-b")
        self.run_b = self.controller.current_version_id
        self.controller.load_steps_for_current_height(discard_dirty=True, version_id=self.run_a)
        self.version_count_before = len(self.controller.store.list_versions(50, include_legacy=False))
        self.ui.open_right_tab("Height Generate")
        self.ui._refresh(force=True)
        self.result.update(
            worker_pid=int(self.controller.latest_sim_status.get("pid", 0) or 0),
            worker_session_id=str(self.controller.latest_sim_status.get("worker_session_id", "") or ""),
            seeded_runs=[self.run_a, self.run_b],
        )
        self._capture_ui("run_management_ui", "run_management_ui.png")
        self._advance("SELECT_PENDING")

    def _stage_select_pending(self) -> None:
        current_before = self.controller.current_version_id
        active_before = str(self.controller.store.active_version_id(50) or "")
        steps_before = copy.deepcopy(self.controller.manager.steps)
        revision_before = self.controller.manager.revision
        self._select_combo_run(self.run_b)
        passed = bool(
            self.controller.pending_selected_run_id == self.run_b
            and self.controller.current_version_id == current_before
            and str(self.controller.store.active_version_id(50) or "") == active_before
            and self.controller.manager.steps == steps_before
            and self.controller.manager.revision == revision_before
            and self.controller.operation.state is OperationState.IDLE
        )
        self._record(
            "combobox_selection_only",
            source_run=current_before,
            destination_run=self.run_b,
            current_after=self.controller.current_version_id,
            active_before=active_before,
            active_after=self.controller.store.active_version_id(50),
            sequence_unchanged=self.controller.manager.steps == steps_before,
            revision_unchanged=self.controller.manager.revision == revision_before,
            result="PASS" if passed else "FAIL",
        )
        if not passed:
            raise AssertionError(f"combobox selection opened or mutated the Run: {self.result['run_management_rows'][-1]}")
        self.ui._refresh(force=True)
        self._capture_ui("pending_selection_not_opened", "pending_selection_not_opened.png")
        self._advance("OPEN_SELECTED")

    def _stage_open_selected(self) -> None:
        self.ui.height_generate_panel.open_run_button.invoke()
        self.ui.root.update_idletasks()
        passed = bool(
            self.controller.current_version_id == self.run_b
            and str(self.controller.store.active_version_id(50) or "") == self.run_b
            and not self.controller.manager.dirty
            and self.controller.operation.state is OperationState.IDLE
        )
        self._record(
            "open_selected_run",
            source_run=self.run_a,
            destination_run=self.run_b,
            dirty=self.controller.manager.dirty,
            read_only=self.controller.visible_steps_are_read_only(),
            result="PASS" if passed else "FAIL",
        )
        if not passed:
            raise AssertionError(f"Open Selected Run failed: {self.result['run_management_rows'][-1]}")
        self.ui._refresh(force=True)
        self._capture_ui("open_selected_run", "open_selected_run.png")
        self._advance("NEW_EMPTY")

    def _stage_new_empty(self) -> None:
        version_count = len(self.controller.store.list_versions(50, include_legacy=False))
        self.ui.height_generate_panel.new_run_button.invoke()
        self.ui.root.update_idletasks()
        count_after = len(self.controller.store.list_versions(50, include_legacy=False))
        passed = bool(
            not self.controller.current_version_id
            and self.controller.manager.count == 0
            and self.controller.manager.dirty
            and count_after == version_count
            and self.controller.operation.state is OperationState.IDLE
        )
        self._record(
            "new_empty_run",
            source_run=self.run_b,
            destination_run="Unsaved New Run",
            version_count_before=version_count,
            version_count_after=count_after,
            disk_directory_created=count_after != version_count,
            dirty=self.controller.manager.dirty,
            result="PASS" if passed else "FAIL",
        )
        if not passed:
            raise AssertionError(f"New Empty Run contract failed: {self.result['run_management_rows'][-1]}")
        self.ui._refresh(force=True)
        self._capture_ui("new_empty_run", "new_empty_run.png")
        self._advance("REOPEN_FOR_UPDATE")

    def _stage_reopen_for_update(self) -> None:
        self._select_combo_run(self.run_b)
        # The visible Open button path was already verified above. Reopen the
        # seed synchronously with explicit discard so this setup step cannot be
        # delayed by a native dirty-choice dialog from the unsaved empty Run.
        self.controller.open_selected_run(discard_dirty=True)
        if self.controller.current_version_id != self.run_b:
            raise AssertionError(
                f"could not reopen seed Run before Update: expected={self.run_b} "
                f"actual={self.controller.current_version_id} pending={self.controller.pending_selected_run_id}"
            )
        self.controller.manager.adopt_steps([_step(9.0, "updated-in-place")], dirty=True)
        self.ui._refresh(force=True)
        self.update_before = self.controller.store.inspect_version(50, self.run_b)
        self.version_count_before = len(self.controller.store.list_versions(50, include_legacy=False))
        self.ui.messagebox.askyesno = lambda *_args, **_kwargs: True
        self.ui.height_generate_panel.update_run_button.invoke()
        self.update_started = True
        self._advance("WAIT_UPDATE")

    def _stage_wait_update(self) -> None:
        if not self.update_started or self.ui.save_in_progress:
            return
        after = self.controller.store.inspect_version(50, self.run_b)
        count_after = len(self.controller.store.list_versions(50, include_legacy=False))
        report = copy.deepcopy(self.controller.last_save_report)
        backup_path = Path(str(report.get("backup_path", "") or ""))
        passed = bool(
            self.controller.current_version_id == self.run_b
            and after.get("created_at") == self.update_before.get("created_at")
            and after.get("updated_at") != self.update_before.get("updated_at")
            and after.get("accepted_steps_sha256") != self.update_before.get("accepted_steps_sha256")
            and count_after == self.version_count_before
            and backup_path.is_file()
            and not self.controller.manager.dirty
            and self.controller.operation.state is OperationState.IDLE
        )
        self._record(
            "update_current_run",
            source_run=self.run_b,
            destination_run=self.controller.current_version_id,
            old_sha=self.update_before.get("accepted_steps_sha256", ""),
            new_sha=after.get("accepted_steps_sha256", ""),
            version_count_before=self.version_count_before,
            version_count_after=count_after,
            created_at_preserved=after.get("created_at") == self.update_before.get("created_at"),
            backup_path=str(backup_path),
            dirty=self.controller.manager.dirty,
            result="PASS" if passed else "FAIL",
        )
        if not passed:
            raise AssertionError(f"Update Current Run contract failed: {self.result['run_management_rows'][-1]}")
        self.ui._refresh(force=True)
        self._capture_ui("update_current_run", "update_current_run.png")
        self.save_as_before = after
        self.controller.manager.adopt_steps([_step(12.0, "save-as-child")], dirty=True)
        self.ui._refresh(force=True)
        self.ui.height_generate_panel.save_as_run_button.invoke()
        self.save_as_started = True
        self._advance("WAIT_SAVE_AS")

    def _stage_wait_save_as(self) -> None:
        if not self.save_as_started or self.ui.save_in_progress:
            return
        self.run_c = self.controller.current_version_id
        old_after = self.controller.store.inspect_version(50, self.run_b)
        child = self.controller.store.inspect_version(50, self.run_c)
        count_after = len(self.controller.store.list_versions(50, include_legacy=False))
        passed = bool(
            self.run_c
            and self.run_c != self.run_b
            and str(child.get("parent_version_id", "") or "") == self.run_b
            and old_after.get("accepted_steps_sha256") == self.save_as_before.get("accepted_steps_sha256")
            and count_after == self.version_count_before + 1
            and str(self.controller.store.active_version_id(50) or "") == self.run_c
            and not self.controller.manager.dirty
            and self.controller.operation.state is OperationState.IDLE
        )
        self._record(
            "save_as_new_run",
            source_run=self.run_b,
            destination_run=self.run_c,
            old_sha=self.save_as_before.get("accepted_steps_sha256", ""),
            old_sha_after=old_after.get("accepted_steps_sha256", ""),
            new_sha=child.get("accepted_steps_sha256", ""),
            parent_run_id=child.get("parent_version_id", ""),
            version_count_before=self.version_count_before,
            version_count_after=count_after,
            result="PASS" if passed else "FAIL",
        )
        if not passed:
            raise AssertionError(f"Save As New Run contract failed: {self.result['run_management_rows'][-1]}")
        self.ui._refresh(force=True)
        self._capture_ui("save_as_new_run", "save_as_new_run.png")
        self._advance("FINISH")

    def _stage_finish(self) -> None:
        self.controller.stop_wheels(reason="run_management_gui_e2e_complete")
        self.result.update(
            success=True,
            current_run_id=self.controller.current_version_id,
            active_run_id=self.controller.store.active_version_id(50),
            final_operation=self.controller.operation.state.value,
            temporary_version_count=len(self.controller.store.list_versions(50, include_legacy=False)),
        )
        self._save_gif()
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(300, lambda: self.ui._window_close(force=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    return RunManagementGuiE2E(Path(args.output), timeout_s=args.timeout_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
