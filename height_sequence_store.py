"""Height bucket storage for accepted-step JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from height_manifest import HeightManifest, height_folder_name, normalize_height_cm
from sequence_model import atomic_save_steps_jsonl, load_steps_jsonl


class HeightSequenceStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else Path(__file__).resolve().parent / "saved_height_steps"
        self.manifest = HeightManifest(self.root / "manifest.json")
        self._manifest_snapshot: dict[str, Any] | None = None
        self._last_status_rows: list[dict[str, Any]] = []
        self.last_status_warning = ""
        self.last_save_report: dict[str, Any] = {}
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest.ensure()

    def height_dir(self, height_cm: int | float | str) -> Path:
        return self.root / height_folder_name(height_cm)

    def steps_path(self, height_cm: int | float | str) -> Path:
        return self.manifest.steps_path(height_cm)

    def has_saved_steps(self, height_cm: int | float | str) -> bool:
        path = self.steps_path(height_cm)
        return path.exists() and path.stat().st_size > 0 and len(load_steps_jsonl(path)) > 0

    def load_steps(self, height_cm: int | float | str) -> list[dict[str, Any]]:
        height = normalize_height_cm(height_cm)
        path = self.steps_path(height)
        if not path.exists() or path.stat().st_size == 0:
            return []
        return load_steps_jsonl(path)

    def save_steps(self, height_cm: int | float | str, steps: list[dict[str, Any]]) -> Path:
        height = normalize_height_cm(height_cm)
        self.manifest.ensure()
        path = self.steps_path(height)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_save_report = atomic_save_steps_jsonl(steps, path)
        self.manifest.update_height(height, step_count=len(steps))
        self._manifest_snapshot = None
        return path

    def clear_saved_steps(self, height_cm: int | float | str) -> Path:
        height = normalize_height_cm(height_cm)
        self.manifest.ensure()
        path = self.steps_path(height)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_save_report = atomic_save_steps_jsonl([], path)
        self.manifest.update_height(height, step_count=0, saved_at="")
        self._manifest_snapshot = None
        return path

    def new_empty_sequence(self, height_cm: int | float | str, *, write_file: bool = False) -> Path:
        height = normalize_height_cm(height_cm)
        self.manifest.ensure()
        path = self.steps_path(height)
        path.parent.mkdir(parents=True, exist_ok=True)
        if write_file:
            self.last_save_report = atomic_save_steps_jsonl([], path)
            self.manifest.update_height(height, step_count=0, saved_at="")
            self._manifest_snapshot = None
        return path

    def status_rows(self) -> list[dict[str, Any]]:
        try:
            manifest = self._manifest_snapshot
            if manifest is None:
                manifest = self.manifest.load()
                self.manifest._merge_defaults(manifest)
            rows = self._rows_from_manifest(manifest)
            self._last_status_rows = rows
            self.last_status_warning = ""
            return rows
        except Exception as exc:
            self.last_status_warning = f"Could not read manifest status rows: {exc}"
            if self._last_status_rows:
                return list(self._last_status_rows)
            manifest = self.manifest._default_manifest()
            return self._rows_from_manifest(manifest)

    def _rows_from_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for height_cm in manifest["supported_heights_cm"]:
            entry = manifest["heights"][str(height_cm)]
            rows.append(
                {
                    "height_cm": int(height_cm),
                    "recorded": bool(entry.get("recorded", False)),
                    "step_count": int(entry.get("step_count", 0)),
                    "steps_path": str(entry.get("steps_path", "")),
                    "last_saved_at": str(entry.get("last_saved_at", "")),
                }
            )
        return rows

    def refresh_manifest(self) -> dict[str, Any]:
        manifest = self.manifest.load()
        self.manifest._merge_defaults(manifest)
        for height_cm in manifest["supported_heights_cm"]:
            path = self.steps_path(height_cm)
            step_count = self.manifest._count_jsonl_objects(path)
            entry = manifest["heights"][str(height_cm)]
            entry["recorded"] = step_count > 0
            entry["step_count"] = step_count
            entry["steps_path"] = str(path)
            if step_count == 0:
                entry["last_saved_at"] = ""
        self._manifest_snapshot = manifest
        self._last_status_rows = self._rows_from_manifest(manifest)
        self.last_status_warning = ""
        return manifest
