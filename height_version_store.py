"""Immutable multi-version persistence for the three supported obstacle heights."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from height_manifest import SUPPORTED_HEIGHTS_MM, height_folder_name_mm, normalize_height_mm
from sequence_model import atomic_save_steps_jsonl, load_steps_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VERSION_ROOT = PROJECT_ROOT / "saved_height_steps_fsm_reference_v2"
DEFAULT_LEGACY_ROOT = PROJECT_ROOT / "saved_height_steps"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HeightVersionStore:
    """One persistence service; polling reads only its in-memory manifest cache."""

    SCHEMA_VERSION = "height-steps-versions.v2"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        legacy_root: str | Path | None = None,
        robot_asset_path: str | Path | None = None,
    ):
        self.root = Path(root) if root else DEFAULT_VERSION_ROOT
        if legacy_root is not None:
            self.legacy_root = Path(legacy_root)
        elif self.root.resolve() == DEFAULT_VERSION_ROOT.resolve():
            self.legacy_root = DEFAULT_LEGACY_ROOT
        else:
            self.legacy_root = self.root.parent / "saved_height_steps"
        self.manifest_path = self.root / "manifest.json"
        self.robot_asset_path = Path(robot_asset_path) if robot_asset_path else None
        self.robot_asset_sha256 = file_sha256(self.robot_asset_path) if self.robot_asset_path and self.robot_asset_path.is_file() else ""
        self.last_save_report: dict[str, Any] = {}
        self.last_status_warning = ""
        self._legacy_steps_cache: dict[int, Path] = {}
        self._manifest = self._default_manifest()
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for height_mm in SUPPORTED_HEIGHTS_MM:
            (self.height_dir(height_mm) / "versions").mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self._manifest = self._load_json(self.manifest_path)
            self._merge_defaults(self._manifest)
        else:
            self._manifest = self._default_manifest()
            self._write_json_atomic(self.manifest_path, self._manifest, backup_existing=False)
        self._refresh_legacy_cache()

    def height_dir(self, height_mm: int | float | str) -> Path:
        return self.root / height_folder_name_mm(self._compat_height_mm(height_mm))

    @staticmethod
    def _compat_height_mm(value: int | float | str) -> int:
        numeric = float(value)
        if numeric in {5.0, 10.0}:
            return int(numeric * 10.0)
        return normalize_height_mm(value)

    def active_path(self, height_mm: int | float | str) -> Path:
        return self.height_dir(height_mm) / "active_version.json"

    def version_dir(self, height_mm: int | float | str, version_id: str) -> Path:
        return self.height_dir(height_mm) / "versions" / str(version_id)

    def version_steps_path(self, height_mm: int | float | str, version_id: str) -> Path:
        return self.version_dir(height_mm, version_id) / "accepted_steps.jsonl"

    def version_metadata_path(self, height_mm: int | float | str, version_id: str) -> Path:
        return self.version_dir(height_mm, version_id) / "metadata.json"

    def active_version_id(self, height_mm: int | float | str) -> str:
        height = self._compat_height_mm(height_mm)
        return str(self._manifest["heights"][str(height)].get("active_version_id", "") or "")

    def status_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for height_mm in SUPPORTED_HEIGHTS_MM:
            entry = dict(self._manifest["heights"][str(height_mm)])
            rows.append(
                {
                    "height_mm": height_mm,
                    "version_count": int(entry.get("version_count", 0)),
                    "active_version_id": str(entry.get("active_version_id", "") or ""),
                    "last_saved_at": str(entry.get("last_saved_at", "") or ""),
                    "legacy_available": self.legacy_steps_path(height_mm) is not None,
                }
            )
        return rows

    def list_versions(self, height_mm: int | float | str, *, include_legacy: bool = True) -> list[dict[str, Any]]:
        height = self._compat_height_mm(height_mm)
        versions = [dict(row) for row in self._manifest["heights"][str(height)].get("versions", [])]
        if include_legacy:
            legacy = self.legacy_steps_path(height)
            if legacy is not None:
                versions.append(
                    {
                        "version_id": f"legacy_{height // 10}cm_readonly",
                        "version_name": f"Legacy {height // 10} cm (read-only)",
                        "path": str(legacy.parent.resolve()),
                        "accepted_steps_path": str(legacy.resolve()),
                        "read_only": True,
                        "legacy": True,
                    }
                )
        return versions

    def legacy_steps_path(self, height_mm: int | float | str) -> Path | None:
        height = self._compat_height_mm(height_mm)
        return self._legacy_steps_cache.get(height)

    def inspect_version(self, height_mm: int | float | str, version_id: str) -> dict[str, Any]:
        """Read and hash one Run without changing the active Run."""

        height = self._compat_height_mm(height_mm)
        version = str(version_id or "")
        if version.startswith("legacy_"):
            path = self.legacy_steps_path(height)
            if path is None:
                raise FileNotFoundError(f"No legacy steps exist for {height} mm")
            steps = load_steps_jsonl(path)
            return {
                "version_id": version,
                "version_name": f"Legacy {height // 10} cm (read-only)",
                "height_mm": height,
                "path": str(path.parent.resolve()),
                "accepted_steps_path": str(path.resolve()),
                "step_count": len(steps),
                "command_count": sum(len(step.get("events", []) or []) for step in steps),
                "accepted_steps_sha256": file_sha256(path),
                "read_only": True,
                "legacy": True,
                "speed_semantics_version": "legacy-100-percent-canonical",
            }
        metadata_path = self.version_metadata_path(height, version)
        steps_path = self.version_steps_path(height, version)
        metadata = self._load_json(metadata_path)
        actual_hash = file_sha256(steps_path)
        if actual_hash != str(metadata.get("accepted_steps_sha256", "")):
            raise IOError(f"Version hash mismatch: {steps_path}")
        return {
            **metadata,
            "path": str(steps_path.parent.resolve()),
            "accepted_steps_path": str(steps_path.resolve()),
            "accepted_steps_sha256": actual_hash,
            "read_only": False,
            "legacy": False,
        }

    def load_version(
        self,
        height_mm: int | float | str,
        version_id: str,
        *,
        activate: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        height = self._compat_height_mm(height_mm)
        version = str(version_id or "")
        metadata = self.inspect_version(height, version)
        steps = load_steps_jsonl(Path(str(metadata["accepted_steps_path"])))
        if activate and not bool(metadata.get("legacy", False)):
            self._set_active(height, version)
        return steps, metadata

    def save_new_version(
        self,
        height_mm: int | float | str,
        steps: list[dict[str, Any]],
        *,
        version_name: str = "manual",
        note: str = "",
        parent_version_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        height = self._compat_height_mm(height_mm)
        created = now_iso()
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        index = self._next_index(height)
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", str(version_name or "manual")).strip("-")[:48] or "manual"
        version_id = f"v{index:03d}_{stamp}_{safe_name}"
        target_dir = self.version_dir(height, version_id)
        target_dir.mkdir(parents=True, exist_ok=False)
        steps_path = target_dir / "accepted_steps.jsonl"
        try:
            save_report = atomic_save_steps_jsonl(steps, steps_path)
            accepted_hash = file_sha256(steps_path)
            extra = dict(metadata or {})
            record = {
                "schema_version": self.SCHEMA_VERSION,
                "version_id": version_id,
                "version_name": str(version_name or "manual"),
                "created_at": created,
                "updated_at": created,
                "height_mm": height,
                "step_count": len(steps),
                "command_count": sum(len(step.get("events", []) or []) for step in steps),
                "accepted_steps_sha256": accepted_hash,
                "robot_asset_path": str(self.robot_asset_path.resolve()) if self.robot_asset_path else str(extra.pop("robot_asset_path", "")),
                "robot_asset_sha256": self.robot_asset_sha256 or str(extra.pop("robot_asset_sha256", "")),
                "actuator_baseline_id": str(extra.pop("actuator_baseline_id", "")),
                "environment_baseline_id": str(extra.pop("environment_baseline_id", "")),
                "obstacle_width_m": extra.pop("obstacle_width_m", None),
                "obstacle_length_m": extra.pop("obstacle_length_m", None),
                "obstacle_front_face_x_m": extra.pop("obstacle_front_face_x_m", None),
                "robot_respawn_pose": extra.pop("robot_respawn_pose", None),
                "note": str(note or ""),
                "parent_version_id": str(parent_version_id or ""),
                **extra,
            }
            self._write_json_atomic(target_dir / "metadata.json", record, backup_existing=False)
            self._append_manifest_version(height, record, target_dir)
            self._set_active(height, version_id)
            self.last_save_report = {
                **save_report,
                "version_id": version_id,
                "version_name": record["version_name"],
                "path": str(target_dir.resolve()),
                "accepted_steps_path": str(steps_path.resolve()),
                "accepted_steps_sha256": accepted_hash,
                "saved_at": created,
                "step_count": record["step_count"],
                "command_count": record["command_count"],
                "parent_run_id": record["parent_version_id"],
                "version_count_after": int(
                    self._manifest["heights"][str(height)].get("version_count", 0) or 0
                ),
            }
            return target_dir
        except Exception:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            if self.manifest_path.exists():
                self._manifest = self._load_json(self.manifest_path)
                self._merge_defaults(self._manifest)
            raise

    def save_current_version(
        self,
        height_mm: int | float | str,
        version_id: str,
        steps: list[dict[str, Any]],
        *,
        confirmed: bool,
    ) -> Path:
        if not confirmed:
            raise PermissionError("Save Current Version requires explicit confirmation")
        height = self._compat_height_mm(height_mm)
        version = str(version_id)
        if version.startswith("legacy_"):
            raise PermissionError("Legacy versions are read-only")
        target = self.version_steps_path(height, version)
        metadata_path = self.version_metadata_path(height, version)
        active_path = self.active_path(height)
        paths = (target, metadata_path, self.manifest_path, active_path)
        originals = {path: path.read_bytes() if path.exists() else None for path in paths}
        before = self.inspect_version(height, version)
        entry = self._manifest["heights"][str(height)]
        version_count_before = int(entry.get("version_count", 0) or 0)
        created_at = str(before.get("created_at", "") or "")
        old_hash = str(before.get("accepted_steps_sha256", "") or "")
        try:
            report = atomic_save_steps_jsonl(steps, target)
            metadata = self._load_json(metadata_path)
            metadata.update(
                updated_at=now_iso(),
                step_count=len(steps),
                command_count=sum(len(step.get("events", []) or []) for step in steps),
                accepted_steps_sha256=file_sha256(target),
            )
            metadata["created_at"] = created_at
            self._write_json_atomic(metadata_path, metadata, backup_existing=True)
            self._replace_manifest_version(height, metadata, target.parent)
            verified_steps, verified = self.load_version(height, version, activate=False)
            current_entry = self._manifest["heights"][str(height)]
            if len(verified_steps) != len(steps):
                raise IOError("Updated Run step count verification failed")
            if str(verified.get("created_at", "") or "") != created_at:
                raise IOError("Updated Run created_at changed")
            if int(current_entry.get("version_count", 0) or 0) != version_count_before:
                raise IOError("Update Current Run changed version_count")
            if str(current_entry.get("active_version_id", "") or "") != version:
                raise IOError("Update Current Run changed the active Run ID")
            self.last_save_report = {
                **report,
                **verified,
                "path": str(target.parent.resolve()),
                "old_accepted_steps_sha256": old_hash,
                "new_accepted_steps_sha256": str(verified.get("accepted_steps_sha256", "") or ""),
                "version_count_before": version_count_before,
                "version_count_after": int(current_entry.get("version_count", 0) or 0),
                "run_id_unchanged": True,
            }
            return target.parent
        except Exception:
            for path, payload in originals.items():
                self._restore_bytes_atomic(path, payload)
            self._manifest = self._load_json(self.manifest_path)
            self._merge_defaults(self._manifest)
            raise

    def refresh_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            self._manifest = self._load_json(self.manifest_path)
        self._merge_defaults(self._manifest)
        self._refresh_legacy_cache()
        return json.loads(json.dumps(self._manifest))

    def _refresh_legacy_cache(self) -> None:
        cache: dict[int, Path] = {}
        for height_mm, legacy_cm in ((50, 5), (100, 10)):
            path = self.legacy_root / f"height_{legacy_cm:02d}cm" / "accepted_steps.jsonl"
            if path.is_file() and path.stat().st_size > 0:
                cache[height_mm] = path
        self._legacy_steps_cache = cache

    # Narrow compatibility for tests/tools that used the former single-file
    # store.  Every write still creates a new immutable v2 version.
    def save_steps(self, height: int | float | str, steps: list[dict[str, Any]]) -> Path:
        version_dir = self.save_new_version(height, steps, version_name="compat-import")
        return version_dir / "accepted_steps.jsonl"

    def clear_saved_steps(self, height: int | float | str) -> Path:
        return self.save_steps(height, [])

    def load_steps(self, height: int | float | str) -> list[dict[str, Any]]:
        height_mm = self._compat_height_mm(height)
        version_id = self.active_version_id(height_mm)
        if not version_id:
            rows = self.list_versions(height_mm, include_legacy=True)
            version_id = str(rows[0].get("version_id", "")) if rows else ""
        return self.load_version(height_mm, version_id)[0] if version_id else []

    def steps_path(self, height: int | float | str) -> Path:
        height_mm = self._compat_height_mm(height)
        version_id = self.active_version_id(height_mm)
        if version_id and not version_id.startswith("legacy_"):
            return self.version_steps_path(height_mm, version_id)
        legacy = self.legacy_steps_path(height_mm)
        return legacy if legacy is not None else self.height_dir(height_mm) / "unsaved_accepted_steps.jsonl"

    def _next_index(self, height_mm: int) -> int:
        values = []
        for row in self._manifest["heights"][str(height_mm)].get("versions", []):
            match = re.match(r"v(\d+)_", str(row.get("version_id", "")))
            if match:
                values.append(int(match.group(1)))
        return max(values + [0]) + 1

    def _append_manifest_version(self, height_mm: int, metadata: dict[str, Any], path: Path) -> None:
        row = self._manifest_row(metadata, path)
        entry = self._manifest["heights"][str(height_mm)]
        entry.setdefault("versions", []).append(row)
        entry["version_count"] = len(entry["versions"])
        entry["last_saved_at"] = metadata["updated_at"]
        self._manifest["updated_at"] = now_iso()
        self._write_json_atomic(self.manifest_path, self._manifest, backup_existing=True)

    def _replace_manifest_version(self, height_mm: int, metadata: dict[str, Any], path: Path) -> None:
        entry = self._manifest["heights"][str(height_mm)]
        replacement = self._manifest_row(metadata, path)
        entry["versions"] = [replacement if row.get("version_id") == metadata["version_id"] else row for row in entry.get("versions", [])]
        entry["version_count"] = len(entry["versions"])
        entry["last_saved_at"] = metadata["updated_at"]
        self._manifest["updated_at"] = now_iso()
        self._write_json_atomic(self.manifest_path, self._manifest, backup_existing=True)

    def _set_active(self, height_mm: int, version_id: str) -> None:
        current_id = str(
            self._manifest["heights"][str(height_mm)].get("active_version_id", "") or ""
        )
        active_path = self.active_path(height_mm)
        if current_id == str(version_id) and active_path.exists():
            try:
                if str(self._load_json(active_path).get("version_id", "") or "") == str(version_id):
                    return
            except Exception:
                pass
        active = {"height_mm": height_mm, "version_id": str(version_id), "updated_at": now_iso()}
        self._write_json_atomic(active_path, active, backup_existing=True)
        self._manifest["heights"][str(height_mm)]["active_version_id"] = str(version_id)
        self._manifest["updated_at"] = now_iso()
        self._write_json_atomic(self.manifest_path, self._manifest, backup_existing=True)

    @staticmethod
    def _manifest_row(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "version_id": metadata["version_id"],
            "version_name": metadata["version_name"],
            "path": str(path.resolve()),
            "accepted_steps_path": str((path / "accepted_steps.jsonl").resolve()),
            "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"],
            "step_count": metadata["step_count"],
            "command_count": metadata["command_count"],
            "accepted_steps_sha256": metadata["accepted_steps_sha256"],
            "read_only": False,
            "legacy": False,
        }

    def _default_manifest(self) -> dict[str, Any]:
        created = now_iso()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": created,
            "updated_at": created,
            "supported_heights_mm": list(SUPPORTED_HEIGHTS_MM),
            "heights": {
                str(height): {
                    "height_mm": height,
                    "folder": height_folder_name_mm(height),
                    "version_count": 0,
                    "active_version_id": "",
                    "last_saved_at": "",
                    "versions": [],
                }
                for height in SUPPORTED_HEIGHTS_MM
            },
        }

    def _merge_defaults(self, manifest: dict[str, Any]) -> None:
        defaults = self._default_manifest()
        manifest["schema_version"] = self.SCHEMA_VERSION
        manifest["supported_heights_mm"] = list(SUPPORTED_HEIGHTS_MM)
        manifest.setdefault("created_at", defaults["created_at"])
        manifest.setdefault("updated_at", defaults["updated_at"])
        heights = manifest.setdefault("heights", {})
        for height in SUPPORTED_HEIGHTS_MM:
            entry = heights.setdefault(str(height), {})
            for key, value in defaults["heights"][str(height)].items():
                entry.setdefault(key, value)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"JSON root must be an object: {path}")
        return loaded

    @classmethod
    def _write_json_atomic(cls, path: Path, payload: dict[str, Any], *, backup_existing: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        backup = path.with_name(f"{path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}") if backup_existing and path.exists() else None
        try:
            with tmp.open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if cls._load_json(tmp) != payload:
                raise IOError(f"Temporary JSON verification failed: {tmp}")
            if backup is not None:
                shutil.copy2(path, backup)
                with backup.open("r+b") as stream:
                    os.fsync(stream.fileno())
            os.replace(tmp, path)
            if cls._load_json(path) != payload:
                raise IOError(f"Final JSON verification failed: {path}")
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _restore_bytes_atomic(path: Path, payload: bytes | None) -> None:
        if payload is None:
            if path.exists():
                path.unlink()
            return
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback.tmp")
        try:
            with tmp.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
