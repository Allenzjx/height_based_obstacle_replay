"""Manifest and height validation for height-indexed replay steps."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_HEIGHTS_MM = (50, 75, 100)
# Retained only for the isolated legacy HeightSequenceStore.  Production UI,
# worker, manifests, and v2 persistence use SUPPORTED_HEIGHTS_MM exclusively.
SUPPORTED_HEIGHTS_CM = tuple(range(0, 45, 5))
HEIGHT_INTERVAL_CM = 5
HEIGHT_ERROR_MESSAGE = "当前只支持 50、75、100 mm"


class HeightValidationError(ValueError):
    pass


def normalize_height_cm(value: int | float | str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HeightValidationError(HEIGHT_ERROR_MESSAGE) from exc
    if not numeric.is_integer():
        raise HeightValidationError(HEIGHT_ERROR_MESSAGE)
    height_cm = int(numeric)
    if height_cm not in SUPPORTED_HEIGHTS_CM:
        raise HeightValidationError(HEIGHT_ERROR_MESSAGE)
    return height_cm


def normalize_height_mm(value: int | float | str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HeightValidationError(HEIGHT_ERROR_MESSAGE) from exc
    if not numeric.is_integer() or int(numeric) not in SUPPORTED_HEIGHTS_MM:
        raise HeightValidationError(HEIGHT_ERROR_MESSAGE)
    return int(numeric)


def legacy_cm_to_mm(value: int | float | str) -> int:
    legacy_cm = normalize_height_cm(value)
    if legacy_cm not in {5, 10}:
        raise HeightValidationError("Legacy --height-cm accepts only 5 or 10")
    return legacy_cm * 10


def obstacle_height_m_mm(height_mm: int | float | str) -> float:
    return normalize_height_mm(height_mm) / 1000.0


def height_folder_name_mm(height_mm: int | float | str) -> str:
    return f"height_{normalize_height_mm(height_mm):03d}mm"


def obstacle_height_m(height_cm: int | float | str) -> float:
    return normalize_height_cm(height_cm) / 100.0


def height_folder_name(height_cm: int | float | str) -> str:
    return f"height_{normalize_height_cm(height_cm):02d}cm"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class HeightManifest:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    def ensure(self) -> dict[str, Any]:
        self._ensure_directory(self.root)
        for height_cm in SUPPORTED_HEIGHTS_CM:
            self._ensure_directory(self.root / height_folder_name(height_cm))
        manifest = self.load()
        changed = self._merge_defaults(manifest)
        if changed or not self.manifest_path.exists():
            self.save(manifest)
        return manifest

    def load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._default_manifest()
        try:
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON: {self.manifest_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"Manifest root must be an object: {self.manifest_path}")
        return loaded

    def save(self, manifest: dict[str, Any]) -> None:
        self._ensure_directory(self.manifest_path.parent)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def refresh_from_files(self) -> dict[str, Any]:
        self._ensure_directory(self.root)
        for height_cm in SUPPORTED_HEIGHTS_CM:
            self._ensure_directory(self.root / height_folder_name(height_cm))
        manifest = self.load()
        self._merge_defaults(manifest)
        for height_cm in SUPPORTED_HEIGHTS_CM:
            steps_path = self.steps_path(height_cm)
            step_count = self._count_jsonl_objects(steps_path)
            entry = manifest["heights"][str(height_cm)]
            entry["recorded"] = step_count > 0
            entry["step_count"] = step_count
            entry["steps_path"] = str(steps_path)
            if step_count == 0:
                entry["last_saved_at"] = ""
        manifest["updated_at"] = now_iso()
        self.save(manifest)
        return manifest

    def update_height(self, height_cm: int | float | str, *, step_count: int, saved_at: str | None = None) -> dict[str, Any]:
        height = normalize_height_cm(height_cm)
        manifest = self.load()
        self._merge_defaults(manifest)
        entry = manifest["heights"][str(height)]
        entry["recorded"] = int(step_count) > 0
        entry["step_count"] = int(step_count)
        entry["steps_path"] = str(self.steps_path(height))
        entry["last_saved_at"] = saved_at if saved_at is not None else (now_iso() if step_count > 0 else "")
        manifest["updated_at"] = now_iso()
        self.save(manifest)
        return manifest

    def entry(self, height_cm: int | float | str) -> dict[str, Any]:
        height = normalize_height_cm(height_cm)
        manifest = self.ensure()
        return dict(manifest["heights"][str(height)])

    def steps_path(self, height_cm: int | float | str) -> Path:
        return self.root / height_folder_name(height_cm) / "accepted_steps.jsonl"

    def _default_manifest(self) -> dict[str, Any]:
        created = now_iso()
        return {
            "schema_version": 1,
            "created_at": created,
            "updated_at": created,
            "supported_heights_cm": list(SUPPORTED_HEIGHTS_CM),
            "height_interval_cm": HEIGHT_INTERVAL_CM,
            "heights": {
                str(height_cm): {
                    "height_cm": height_cm,
                    "height_m": height_cm / 100.0,
                    "folder": height_folder_name(height_cm),
                    "recorded": False,
                    "steps_path": str(self.steps_path(height_cm)),
                    "last_saved_at": "",
                    "step_count": 0,
                }
                for height_cm in SUPPORTED_HEIGHTS_CM
            },
        }

    def _merge_defaults(self, manifest: dict[str, Any]) -> bool:
        changed = False
        defaults = self._default_manifest()
        for key, value in defaults.items():
            if key not in manifest:
                manifest[key] = value
                changed = True
        if manifest.get("supported_heights_cm") != list(SUPPORTED_HEIGHTS_CM):
            manifest["supported_heights_cm"] = list(SUPPORTED_HEIGHTS_CM)
            changed = True
        heights = manifest.setdefault("heights", {})
        for height_cm in SUPPORTED_HEIGHTS_CM:
            key = str(height_cm)
            default_entry = defaults["heights"][key]
            entry = heights.setdefault(key, {})
            for entry_key, entry_value in default_entry.items():
                if entry_key not in entry:
                    entry[entry_key] = entry_value
                    changed = True
            expected_path = str(self.steps_path(height_cm))
            if entry.get("steps_path") != expected_path:
                entry["steps_path"] = expected_path
                changed = True
        return changed

    @staticmethod
    def _count_jsonl_objects(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                count += 1
        return count

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.exists() and path.is_dir():
            return
        if path.exists() and not path.is_dir():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = path.with_name(f"{path.name}.invalid_file_{stamp}")
            suffix = 1
            while target.exists():
                target = path.with_name(f"{path.name}.invalid_file_{stamp}_{suffix}")
                suffix += 1
            path.rename(target)
        path.mkdir(parents=True, exist_ok=True)
