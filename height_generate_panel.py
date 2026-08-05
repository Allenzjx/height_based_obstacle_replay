"""Lightweight millimetre height and immutable version controls."""

from __future__ import annotations

from typing import Any

from height_manifest import SUPPORTED_HEIGHTS_MM


class HeightGeneratePanel:
    def __init__(self, ui: Any, parent: Any):
        self.ui = ui
        self.controller = ui.controller
        self.ttk = ui.ttk
        self.status_var = ui.tk.StringVar(value="Select 50, 75, or 100 mm. Generate never starts Playback.")
        self.version_var = ui.tk.StringVar(value="")
        self.version_detail_var = ui.tk.StringVar(value="Current version: none")
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

        versions = self.ttk.LabelFrame(parent, text="Recording Versions")
        versions.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.ttk.Label(versions, text="Version").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        self.version_combo = self.ttk.Combobox(versions, textvariable=self.version_var, state="readonly", width=42)
        self.version_combo.grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        self.version_combo.bind("<<ComboboxSelected>>", self._load_selected_version)
        self.ttk.Button(versions, text="Load Version", command=self._load_selected_version).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        self.ttk.Button(versions, text="Refresh Versions", command=self._refresh_versions).grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        self.save_button = self.ttk.Button(versions, text="💾 Save New Version", command=self.ui._save_new_version_async)
        self.save_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.ttk.Label(versions, textvariable=self.version_detail_var, wraplength=560, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=3
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
        self.status_var.set(f"Updating {self.controller.current_height_mm} mm… request {request_id[:8]}; controls remain available.")

    def generate_and_respawn(self) -> None:
        if not self._select_height():
            return
        request_id = self.controller.generate_height_and_respawn()
        self.status_var.set(f"Updating {self.controller.current_height_mm} mm and respawning… request {request_id[:8]}.")

    def recalibrate(self) -> None:
        request_id = self.controller.recalibrate_ground_reference()
        self.status_var.set(f"Ground recalibration requested: {request_id[:8]}.")

    def _refresh_versions(self) -> None:
        self.controller.refresh_manifest()
        self._sync_version_choices()
        self.ui._refresh(force=False)

    def _sync_version_choices(self) -> None:
        versions = self.controller.store.list_versions(self.controller.current_height_mm, include_legacy=True)
        self._version_labels = {}
        labels: list[str] = []
        for row in versions:
            version_id = str(row.get("version_id", "") or "")
            label = f"{version_id} — {row.get('version_name', '')}"
            labels.append(label)
            self._version_labels[label] = version_id
        self.version_combo.configure(values=labels)
        current = self.controller.current_version_id
        selected = next((label for label, version_id in self._version_labels.items() if version_id == current), "")
        self.version_var.set(selected)

    def _load_selected_version(self, _event: Any | None = None) -> None:
        version_id = self._version_labels.get(self.version_var.get(), "")
        if not version_id:
            return
        if self.controller.manager.dirty and not self.ui._resolve_unsaved_changes("loading another version"):
            self._sync_version_choices()
            return
        self.controller.load_steps_for_current_height(discard_dirty=True, version_id=version_id)
        self.ui._refresh(force=False)

    def refresh(self, snapshot: dict[str, Any], *, sync_versions: bool = False) -> None:
        height = snapshot["height"]
        sequence = snapshot["sequence"]
        if sync_versions:
            self._sync_version_choices()
        current = str(height.get("current_version_id", "") or "none")
        metadata = dict(height.get("current_version_metadata", {}) or {})
        self.status_var.set(
            f"Target={height['current_mm']} mm | Scene={height.get('scene_mm') if height.get('scene_mm') is not None else '-'} mm | "
            f"control_ready={snapshot['sim']['motion_ready']} | request={self.controller.pending_height_request_id[:8] or '-'}"
        )
        self.version_detail_var.set(
            f"Current height: {height['current_mm']} mm\n"
            f"Current version: {current}\n"
            f"Path: {sequence['accepted_path']}\n"
            f"Dirty: {sequence['dirty']} | Steps: {sequence['count']} | Commands: {sequence['event_count']}\n"
            f"SHA-256: {metadata.get('accepted_steps_sha256', '-') or '-'}"
        )
        self.save_button.configure(state="normal" if sequence["dirty"] or sequence["count"] else "disabled")
