"""Height task tab for the real-robot-style height replay UI."""

from __future__ import annotations

from typing import Any

from height_manifest import SUPPORTED_HEIGHTS_CM


class HeightTaskPanel:
    def __init__(self, ui: Any, parent: Any):
        self.ui = ui
        self.controller = ui.controller
        self.tk = ui.tk
        self.ttk = ui.ttk
        self.parent = parent
        self.current_task_var = self.tk.StringVar(value="Current Task Height: none")
        self._build(parent)

    def _build(self, parent: Any) -> None:
        mode_frame = self.ttk.LabelFrame(parent, text="Height Task Mode")
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.ttk.Radiobutton(mode_frame, text="Height Recording Mode", variable=self.ui.height_mode_var, value="recording").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        self.ttk.Radiobutton(mode_frame, text="Height Replay Mode", variable=self.ui.height_mode_var, value="replay").grid(row=0, column=1, sticky="w", padx=3, pady=2)
        self.ttk.Label(mode_frame, text="Current height").grid(row=1, column=0, sticky="e", padx=3, pady=2)
        self.height_combo = self.ttk.Combobox(
            mode_frame,
            textvariable=self.ui.height_var,
            values=[str(height) for height in SUPPORTED_HEIGHTS_CM],
            width=8,
            state="readonly",
        )
        self.height_combo.grid(row=1, column=1, sticky="w", padx=3, pady=2)
        self.height_combo.bind("<<ComboboxSelected>>", lambda _event: self._switch_to_combo_height())
        self.ttk.Label(mode_frame, textvariable=self.current_task_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        mode_frame.columnconfigure(1, weight=1)

        heights_frame = self.ttk.LabelFrame(parent, text="Supported Heights")
        heights_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for index, height in enumerate(SUPPORTED_HEIGHTS_CM):
            self.ttk.Checkbutton(
                heights_frame,
                text=f"{height}cm",
                variable=self.ui.height_selected_vars[height],
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=4, pady=2)
        self.ttk.Button(heights_frame, text="Select All Heights", command=self.select_all).grid(row=3, column=0, sticky="ew", padx=3, pady=4)
        self.ttk.Button(heights_frame, text="Clear Selection", command=self.clear_selection).grid(row=3, column=1, sticky="ew", padx=3, pady=4)

        task_frame = self.ttk.LabelFrame(parent, text="Height Recording / Replay Task")
        task_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        buttons = [
            ("Start Height Recording Task", lambda: self.start_task("recording")),
            ("Start Height Replay Task", lambda: self.start_task("replay")),
            ("Generate / Load Current Height Obstacle", self.generate_current),
            ("Load Steps For Current Height", self.load_current),
            ("Save Steps For Current Height", self.save_current),
            ("Save And Next Height", self.save_and_next),
            ("Previous Height", self.previous_height),
            ("Next Height", self.next_height),
            ("Finish Task", self.finish_task),
            ("Auto Replay Current Height", self.auto_replay_current),
            ("Respawn And Auto Replay Current Height", self.respawn_auto_replay_current),
            ("Auto Replay Selected Heights", self.auto_replay_selected),
        ]
        for index, (label, callback) in enumerate(buttons):
            self.ttk.Button(task_frame, text=label, command=callback).grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        task_frame.columnconfigure(0, weight=1)
        task_frame.columnconfigure(1, weight=1)

        self.manifest_tree = self.ui._build_manifest_tree(parent, row=3)

    def selected_heights(self) -> list[int]:
        selected = [height for height in SUPPORTED_HEIGHTS_CM if self.ui.height_selected_vars[height].get()]
        if not selected:
            selected = [self.controller.current_height_cm]
        return selected

    def select_all(self) -> None:
        for var in self.ui.height_selected_vars.values():
            var.set(True)

    def clear_selection(self) -> None:
        for var in self.ui.height_selected_vars.values():
            var.set(False)

    def start_task(self, mode: str) -> None:
        try:
            if self.controller.manager.dirty:
                discard = self.ui.messagebox.askyesno(
                    "Discard Unsaved Changes",
                    "Current sequence has unsaved changes. Discard and start the height task?",
                )
                if not discard:
                    return
            self.ui.height_mode_var.set(mode)
            self.controller.start_height_task(mode, self.selected_heights())
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not start height task: {exc}")
        self.ui._refresh()

    def generate_current(self) -> None:
        self.ui._generate_current_obstacle()

    def load_current(self) -> None:
        self.ui._load_current_height_steps()

    def save_current(self) -> None:
        self.ui._save_current_height_steps()

    def save_and_next(self) -> None:
        ok, reason = self.controller.can_save()
        if not ok:
            self.controller._warn("[WARN] " + reason)
            self.ui._refresh()
            return
        self.controller.next_height(save_current=True)
        self.ui._refresh()

    def previous_height(self) -> None:
        self.controller.previous_height()
        self.ui._refresh()

    def next_height(self) -> None:
        self.controller.next_height(save_current=False)
        self.ui._refresh()

    def finish_task(self) -> None:
        self.controller.finish_height_task()
        self.ui._refresh()

    def auto_replay_current(self) -> None:
        self.controller.auto_replay_current_height()
        self.ui._refresh()

    def respawn_auto_replay_current(self) -> None:
        self.controller.respawn_and_auto_replay_current_height()
        self.ui._refresh()

    def auto_replay_selected(self) -> None:
        self.controller.auto_replay_selected_heights(self.selected_heights())
        self.ui._refresh()

    def refresh(self, snapshot: dict[str, Any]) -> None:
        task = snapshot["task"]
        current = task.get("current_height")
        if current is None:
            self.current_task_var.set("Current Task Height: none")
        else:
            self.current_task_var.set(
                f"Current Task Height: {current}cm ({task.get('index', 0)}/{task.get('total', 0)})"
            )

    def _switch_to_combo_height(self) -> None:
        try:
            height = int(self.ui.height_var.get())
            discard = False
            if self.controller.manager.dirty:
                discard = self.ui.messagebox.askyesno(
                    "Discard Unsaved Changes",
                    "Current sequence has unsaved changes. Discard and switch height?",
                )
                if not discard:
                    self.ui.height_var.set(str(self.controller.current_height_cm))
                    return
            self.controller.set_current_height(height, discard_dirty=discard, load_steps=True, generate_obstacle=True)
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not switch height: {exc}")
        self.ui._refresh()
