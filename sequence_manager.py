"""Small accepted-step manager for one height bucket."""

from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sequence_model import (
    accepted_rows,
    atomic_save_steps_jsonl,
    build_combined_step,
    load_steps_jsonl,
    normalize_step,
    apply_events_to_state,
    clone_command_state,
    rebuild_sequence_continuity,
    reindex_steps,
    step_summary,
)


class SequenceManager:
    def __init__(self, accepted_path: str | Path | None = None):
        self.accepted_path = Path(accepted_path) if accepted_path else None
        self.steps: list[dict[str, Any]] = []
        self.dirty = False
        self.revision = 0
        self.last_continuity_report: dict[str, Any] = {"ok": True, "errors": []}
        self.last_operation_report: dict[str, Any] = {}

    @property
    def count(self) -> int:
        return len(self.steps)

    def add_step(self, step: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        accepted = normalize_step(step, index=len(self.steps) + 1)
        accepted["index"] = len(self.steps) + 1
        if self.steps:
            before = clone_command_state(self.steps[-1].get("command_state_after"))
            accepted["command_state_before"] = before
            accepted["command_state_after"] = apply_events_to_state(before, accepted.get("events", []))
        self.steps.append(accepted)
        self.last_continuity_report = {"ok": True, "start_index": len(self.steps), "steps_rebuilt": 1, "errors": []}
        self._mark_dirty()
        self._record_operation("add_step", started, events=len(accepted.get("events", [])), index=int(accepted["index"]))
        return self.steps[-1]

    def replace_step(self, index: int, step: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        zero_index = self._zero_index(index)
        replacement = normalize_step(step, index=index)
        replacement["index"] = index
        replacement["type"] = str(replacement.get("type") or "replacement")
        replacement["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.steps[zero_index] = replacement
        self._rebuild_continuity_from(index)
        self._mark_dirty()
        self._record_operation("replace_step", started, events=len(replacement.get("events", [])), index=int(index))
        return self.steps[zero_index]

    def replace_step_range_with_combined(
        self,
        indices: list[int],
        *,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        selected = sorted(int(index) for index in indices)
        if len(selected) < 2:
            raise ValueError("Combine requires at least two steps.")
        expected = list(range(selected[0], selected[-1] + 1))
        if selected != expected:
            raise ValueError("Combine selection must be contiguous.")
        original = copy.deepcopy(self.steps)
        first = selected[0]
        try:
            selected_steps = [self.get_step(index) for index in selected]
            combined = build_combined_step(selected_steps, allow_conflicts=allow_conflicts)
            kept = [copy.deepcopy(step) for step in self.steps if int(step.get("index", 0)) not in selected]
            kept.insert(first - 1, normalize_step(combined, index=first))
            candidate, report = rebuild_sequence_continuity(kept, start_index=first)
            if not report.get("ok", False):
                raise ValueError("Could not rebuild sequence continuity: " + "; ".join(report.get("errors", [])))
            self.steps = candidate
            self.last_continuity_report = report
            self._mark_dirty()
            self._record_operation("combine_steps", started, selected=selected, index=first, events=len(self.steps[first - 1].get("events", [])))
            return self.steps[first - 1]
        except Exception:
            self.steps = original
            raise

    def delete_step(self, index: int) -> dict[str, Any]:
        started = time.perf_counter()
        removed = self.steps.pop(self._zero_index(index))
        if self.steps:
            self._rebuild_continuity_from(max(1, min(index, len(self.steps))))
        else:
            self.last_continuity_report = {"ok": True, "errors": []}
        self._mark_dirty()
        self._record_operation("delete_step", started, index=int(index))
        return removed

    def undo(self) -> dict[str, Any] | None:
        started = time.perf_counter()
        if not self.steps:
            return None
        removed = self.steps.pop()
        if self.steps:
            self._rebuild_continuity_from(len(self.steps))
        else:
            self.last_continuity_report = {"ok": True, "errors": []}
        self._mark_dirty()
        self._record_operation("undo_step", started, index=int(removed.get("index", 0)))
        return removed

    def clear(self) -> None:
        started = time.perf_counter()
        self.steps = []
        self.last_continuity_report = {"ok": True, "errors": []}
        self._mark_dirty()
        self._record_operation("clear_steps", started)

    def adopt_steps(self, steps: list[dict[str, Any]], *, dirty: bool = False) -> int:
        started = time.perf_counter()
        self.steps = reindex_steps(steps)
        if self.steps:
            self._rebuild_continuity_from(1)
        else:
            self.last_continuity_report = {"ok": True, "errors": []}
        self.dirty = bool(dirty)
        self.revision += 1
        self._record_operation("adopt_steps", started, count=len(self.steps))
        return len(self.steps)

    def load(self, path: str | Path | None = None) -> int:
        source = Path(path) if path else self.accepted_path
        if source is None:
            raise ValueError("No accepted_path is configured.")
        steps = load_steps_jsonl(source)
        self.accepted_path = source
        self.adopt_steps(steps, dirty=False)
        return len(self.steps)

    def save(self, path: str | Path | None = None) -> Path:
        started = time.perf_counter()
        target = Path(path) if path else self.accepted_path
        if target is None:
            raise ValueError("No accepted_path is configured.")
        atomic_save_steps_jsonl(self.steps, target)
        self.accepted_path = target
        self.dirty = False
        self._record_operation("save_steps", started, count=len(self.steps))
        return target

    def rows(self) -> list[dict[str, Any]]:
        return accepted_rows(self.steps)

    def summaries(self) -> list[str]:
        return [step_summary(step) for step in reindex_steps(self.steps)]

    def get_step(self, index: int) -> dict[str, Any]:
        return self.steps[self._zero_index(index)]

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.revision += 1

    def _rebuild_continuity_from(self, start_index: int) -> None:
        rebuilt, report = rebuild_sequence_continuity(self.steps, start_index=start_index)
        if not report.get("ok", False):
            raise ValueError("Could not rebuild sequence continuity: " + "; ".join(report.get("errors", [])))
        self.steps = rebuilt
        self.last_continuity_report = report

    def _zero_index(self, index: int) -> int:
        zero_index = int(index) - 1
        if zero_index < 0 or zero_index >= len(self.steps):
            raise IndexError(f"Accepted step index out of range: {index}")
        return zero_index

    def _record_operation(self, name: str, started: float, **extra: Any) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_operation_report = {"operation": name, "elapsed_ms": elapsed_ms, **extra}
