"""Lightweight millimetre height and explicit Run management controls."""

from __future__ import annotations

from typing import Any

from height_manifest import SUPPORTED_HEIGHTS_MM


class HeightGeneratePanel:
    def __init__(self, ui: Any, parent: Any):
        self.ui = ui
        self.controller = ui.controller
        self.ttk = ui.ttk
        self.status_var = ui.tk.StringVar(value="Select 50, 75, or 100 mm. Generate never starts Playback.")
        self.run_var = ui.tk.StringVar(value="")
        self.run_detail_var = ui.tk.StringVar(value="Current Run: none")
        self.pending_run_preview_var = ui.tk.StringVar(value="Pending Selected Run: none")
        # Compatibility aliases for older smoke tests and callers.
        self.version_var = self.run_var
        self.version_detail_var = self.run_detail_var
        self._version_labels: dict[str, str] = {}
        self._build(parent)

    def _build(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="Height Generate")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.ttk.Label(frame, text="Target obstacle height").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        self.height_combo = self.ttk.Combobox(
            frame,
            textvariable=self.ui.height_var,
            values=[f"{height} mm" for height in SUPPORTED_HEIGHTS_MM],
            width=12,
            state="readonly",
        )
        self.height_combo.grid(row=0, column=1, sticky="w", padx=3, pady=3)
        self.height_combo.bind("<<ComboboxSelected>>", self._select_height)
        buttons = [
            ("Generate / Update Obstacle", self.generate),
            ("Generate + Respawn Robot", self.generate_and_respawn),
            ("Recalibrate Ground Reference", self.recalibrate),
        ]
        for index, (label, callback) in enumerate(buttons):
            self.ttk.Button(frame, text=label, command=callback).grid(
                row=1 + index // 2,
                column=index % 2,
                sticky="ew",
                padx=3,
                pady=3,
            )
        self.ttk.Label(frame, textvariable=self.status_var, wraplength=560).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=(5, 3)
        )
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        versions = self.ttk.LabelFrame(parent, text="Run Management")
        versions.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.ttk.Label(versions, text="Select Run").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        self.version_combo = self.ttk.Combobox(versions, textvariable=self.run_var, state="readonly", width=42)
        self.version_combo.grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        self.version_combo.bind("<<ComboboxSelected>>", self._select_pending_run)
        self.new_run_button = self.ttk.Button(versions, text="New Empty Run", command=self._new_empty_run)
        self.open_run_button = self.ttk.Button(versions, text="Open Selected Run", command=self._open_selected_run)
        self.update_run_button = self.ttk.Button(versions, text="💾 Update Current Run", command=self.ui._update_current_run_async)
        self.save_as_run_button = self.ttk.Button(versions, text="💾 Save As New Run", command=self.ui._save_as_new_run_async)
        self.refresh_runs_button = self.ttk.Button(versions, text="Refresh Runs", command=self._refresh_runs)
        self.new_run_button.grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        self.open_run_button.grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        self.update_run_button.grid(row=2, column=0, sticky="ew", padx=3, pady=3)
        self.save_as_run_button.grid(row=2, column=1, sticky="ew", padx=3, pady=3)
        self.refresh_runs_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.save_button = self.save_as_run_button
        self.ttk.Label(versions, textvariable=self.pending_run_preview_var, wraplength=560, justify="left").grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=3, pady=3
        )
        self.ttk.Label(versions, textvariable=self.run_detail_var, wraplength=560, justify="left").grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=3, pady=3
        )
        versions.columnconfigure(1, weight=1)
        self.manifest_tree = self.ui._build_manifest_tree(parent, row=2)

    def _target_height(self) -> int:
        return int(str(self.ui.height_var.get()).split()[0])

    def _select_height(self, _event: Any | None = None) -> bool:
        try:
            height = self._target_height()
            if self.controller.manager.dirty and height != self.controller.current_height_mm:
                if not self.ui._resolve_unsaved_changes("selecting another height"):
                    self.ui.height_var.set(f"{self.controller.current_height_mm} mm")
                    return False
                discard = True
            else:
                discard = False
            self.controller.set_current_height(height, discard_dirty=discard, load_steps=False, generate_obstacle=False)
            self._sync_version_choices()
            self.status_var.set(f"Selected {height} mm. No obstacle update or Playback was started.")
            result = True
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not select height: {exc}")
            self.ui.height_var.set(f"{self.controller.current_height_mm} mm")
            result = False
        self.ui._refresh(force=False)
        return result

    def generate(self) -> None:
        if not self._select_height():
            return
        request_id = self.controller.generate_or_update_height_obstacle()
        self.status_var.set(f"Updating {self.controller.current_height_mm} mm… request {request_id[:8]}; motion controls are locked until geometry verification.")

    def generate_and_respawn(self) -> None:
        if not self._select_height():
            return
        request_id = self.controller.generate_height_and_respawn()
        self.status_var.set(f"Updating {self.controller.current_height_mm} mm and respawning… request {request_id[:8]}.")

    def recalibrate(self) -> None:
        request_id = self.controller.recalibrate_ground_reference()
        self.status_var.set(f"Ground recalibration requested: {request_id[:8]}.")

    def _refresh_runs(self) -> None:
        self.controller.refresh_manifest()
        self._sync_run_choices()
        self.ui._refresh(force=False)

    def _refresh_versions(self) -> None:
        self._refresh_runs()

    def _sync_run_choices(self) -> None:
        versions = self.controller.store.list_versions(self.controller.current_height_mm, include_legacy=True)
        self._version_labels = {}
        labels: list[str] = []
        for row in versions:
            version_id = str(row.get("version_id", "") or "")
            label = f"{version_id} — {row.get('version_name', '')}"
            labels.append(label)
            self._version_labels[label] = version_id
        self.version_combo.configure(values=labels)
        pending = self.controller.pending_selected_run_id or self.controller.current_version_id
        selected = next((label for label, version_id in self._version_labels.items() if version_id == pending), "")
        self.run_var.set(selected)

    def _sync_version_choices(self) -> None:
        self._sync_run_choices()

    def _select_pending_run(self, _event: Any | None = None) -> None:
        version_id = self._version_labels.get(self.run_var.get(), "")
        if not version_id:
            return
        try:
            metadata = self.controller.select_pending_run(version_id)
            self.pending_run_preview_var.set(
                f"Pending Selected Run: {version_id}\n"
                f"Path: {metadata.get('path', '-')}\n"
                f"Steps: {metadata.get('step_count', 0)} | Commands: {metadata.get('command_count', 0)} | "
                f"Read-only: {bool(metadata.get('read_only', False))}\n"
                f"SHA-256: {metadata.get('accepted_steps_sha256', '-') or '-'}\n"
                "Selection only; click Open Selected Run to load."
            )
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not preview Run {version_id}: {exc}")
            self.controller.pending_selected_run_id = ""
            self.controller.pending_selected_run_metadata = {}
        self.ui._refresh(force=False)

    def _load_selected_version(self, _event: Any | None = None) -> None:
        self._open_selected_run()

    def _open_selected_run(self) -> None:
        if not self.controller.pending_selected_run_id:
            self.controller._warn("[WARN] Select a Run first; combobox selection does not auto-open.")
            return
        if self.controller.manager.dirty and not self.ui._resolve_run_dirty_changes("opening another Run"):
            return
        self.controller.open_selected_run(discard_dirty=True)
        self._sync_run_choices()
        self.ui._refresh(force=False)

    def _new_empty_run(self) -> None:
        if self.controller.manager.dirty and not self.ui._resolve_run_dirty_changes("creating a New Empty Run"):
            return
        self.controller.new_empty_sequence_for_current_height(discard_dirty=True)
        self._sync_run_choices()
        self.ui._refresh(force=False)

    def refresh(self, snapshot: dict[str, Any], *, sync_versions: bool = False) -> None:
        height = snapshot["height"]
        sequence = snapshot["sequence"]
        if sync_versions:
            self._sync_run_choices()
        current = str(height.get("current_version_id", "") or "none")
        metadata = dict(height.get("current_version_metadata", {}) or {})
        pending = str(height.get("pending_selected_run_id", "") or "none")
        pending_metadata = dict(height.get("pending_selected_run_metadata", {}) or {})
        self.status_var.set(
            f"Target={height['current_mm']} mm | Scene={height.get('scene_mm') if height.get('scene_mm') is not None else '-'} mm | "
            f"Measured={height.get('measured_height_mm') if height.get('measured_height_mm') is not None else '-'} mm | "
            f"Width={float(height.get('measured_width_m') or 0.0):.3f} m | revision={height.get('obstacle_revision', 0)} | "
            f"control_ready={snapshot['sim']['motion_ready']} | request={self.controller.pending_height_request_id[:8] or '-'}"
        )
        self.run_detail_var.set(
            f"Current height: {height['current_mm']} mm\n"
            f"Current Run ID: {current}\n"
            f"Current Run Name: {metadata.get('version_name', 'Unsaved New Run') if current != 'none' else 'Unsaved New Run'}\n"
            f"Pending Selected Run: {pending}\n"
            f"Path: {sequence['accepted_path']}\n"
            f"Read-only: {sequence['read_only']} | Dirty: {sequence['dirty']} | "
            f"Steps: {sequence['count']} | Commands: {sequence['event_count']}\n"
            f"Created: {metadata.get('created_at', '-') or '-'} | Updated: {metadata.get('updated_at', '-') or '-'}\n"
            f"SHA-256: {metadata.get('accepted_steps_sha256', '-') or '-'}\n"
            f"Parent Run ID: {metadata.get('parent_version_id', '-') or '-'}"
        )
        if pending != "none":
            self.pending_run_preview_var.set(
                f"Pending Selected Run: {pending}\n"
                f"Path: {pending_metadata.get('path', '-') or '-'}\n"
                f"Steps: {pending_metadata.get('step_count', 0)} | Commands: {pending_metadata.get('command_count', 0)} | "
                f"Read-only: {bool(pending_metadata.get('read_only', False))}\n"
                f"SHA-256: {pending_metadata.get('accepted_steps_sha256', '-') or '-'}\n"
                "Selection only; click Open Selected Run to load."
            )
        else:
            self.pending_run_preview_var.set("Pending Selected Run: none")
        idle = bool(snapshot.get("operation", {}).get("idle", False))
        self.new_run_button.configure(state="normal" if idle else "disabled")
        self.open_run_button.configure(state="normal" if idle and pending != "none" else "disabled")
        can_update = bool(
            idle
            and current != "none"
            and not sequence["read_only"]
            and sequence["dirty"]
        )
        self.update_run_button.configure(state="normal" if can_update else "disabled")
        self.save_as_run_button.configure(
            state="normal" if idle and (sequence["count"] or sequence["dirty"]) else "disabled"
        )
