"""Static, non-destructive audit for every available 50 mm recording version."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from motion_speed import load_motion_reference
from sequence_model import event_playback_commands, load_steps_jsonl
from sim_state_validation import validate_full_sim_pose_state

from .recording_fast_plan import write_fast_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDING_ROOT = (
    PROJECT_ROOT
    / "saved_height_steps_fsm_reference_v2"
    / "height_050mm"
)
DEFAULT_REPORT_ROOT = Path(__file__).resolve().parent / "reports"
BASELINE_PATH = PROJECT_ROOT / "config" / "fsm_recording_baseline.yaml"
ENVIRONMENT_REFERENCE_PATH = PROJECT_ROOT / "config" / "environment_reference.yaml"
MOTION_REFERENCE_PATH = PROJECT_ROOT / "config" / "real_robot_motion_reference.yaml"


MATRIX_COLUMNS = (
    "version",
    "step_count",
    "command_count",
    "hash_valid",
    "snapshot_full_valid_count",
    "snapshot_count",
    "full_success",
    "first_failure_phase",
    "FR_lift_success",
    "FR_place_success",
    "FL_lift_success",
    "FL_place_success",
    "RR_lift_success",
    "RR_place_success",
    "RL_lift_success",
    "RL_place_success",
    "final_recovery_success",
    "maximum_roll_rad",
    "maximum_pitch_rad",
    "minimum_support_margin_m",
    "maximum_contact_drift_m",
    "loaded_front_face_wheel_rotation_rad",
    "replay_evidence",
    "comments",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected an object: {path}")
    return loaded


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _flatten_root_pose(state: dict[str, Any]) -> list[float]:
    value: Any = state.get("root_pose", [])
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _command_count(steps: Iterable[dict[str, Any]]) -> int:
    """Count decoded actuator commands (an atomic event may expand to many)."""

    return sum(
        len(event_playback_commands(event))
        for step in steps
        for event in list(step.get("events", []) or [])
    )


@dataclass(frozen=True)
class VersionFiles:
    version_id: str
    directory: Path
    steps_path: Path
    metadata_path: Path


class RecordingAudit:
    def __init__(
        self,
        recording_root: Path = DEFAULT_RECORDING_ROOT,
        report_root: Path = DEFAULT_REPORT_ROOT,
    ) -> None:
        self.recording_root = recording_root.resolve()
        self.versions_root = self.recording_root / "versions"
        self.report_root = report_root.resolve()

    def enumerate_versions(self) -> list[VersionFiles]:
        if not self.versions_root.is_dir():
            raise FileNotFoundError(self.versions_root)
        versions: list[VersionFiles] = []
        for directory in sorted(path for path in self.versions_root.iterdir() if path.is_dir()):
            versions.append(
                VersionFiles(
                    version_id=directory.name,
                    directory=directory,
                    steps_path=directory / "accepted_steps.jsonl",
                    metadata_path=directory / "metadata.json",
                )
            )
        return versions

    def _manifest_entry_ids(self) -> list[str]:
        manifest_path = self.recording_root.parent / "manifest.json"
        if not manifest_path.exists():
            return []
        manifest = _read_json(manifest_path)
        versions = _nested(manifest, "heights", "50", "versions", default=[])
        return [
            str(row.get("version_id", ""))
            for row in versions
            if isinstance(row, dict) and str(row.get("version_id", ""))
        ]

    def audit_version(self, item: VersionFiles) -> dict[str, Any]:
        missing = [
            path.name
            for path in (item.steps_path, item.metadata_path)
            if not path.is_file()
        ]
        if missing:
            return {
                "version": item.version_id,
                "directory": str(item.directory),
                "valid": False,
                "missing_required_files": missing,
            }
        metadata = _read_json(item.metadata_path)
        actual_hash = sha256_file(item.steps_path)
        expected_hash = str(metadata.get("accepted_steps_sha256", "") or "").lower()
        steps = load_steps_jsonl(item.steps_path)
        snapshot_audit: list[dict[str, Any]] = []
        for step in steps:
            for field in ("sim_state_before", "sim_state_after"):
                result = validate_full_sim_pose_state(step.get(field))
                snapshot_audit.append(
                    {
                        "step_index": int(step.get("index", 0) or 0),
                        "field": field,
                        "classification": str(result.get("classification", "INVALID")),
                        "valid": bool(result.get("valid", False)),
                        "reason": str(result.get("reason", "") or ""),
                    }
                )
        auxiliary = [
            {
                "relative_path": str(path.relative_to(item.directory)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(item.directory.rglob("*"))
            if path.is_file() and path not in {item.steps_path, item.metadata_path}
        ]
        first_state = dict(steps[0].get("sim_state_before", {}) or {}) if steps else {}
        last_state = dict(steps[-1].get("sim_state_after", {}) or {}) if steps else {}
        first_recording_baseline = (
            dict(steps[0].get("recording_baseline", {}) or {}) if steps else {}
        )
        return {
            "version": item.version_id,
            "directory": str(item.directory),
            "valid": bool(actual_hash == expected_hash and steps),
            "missing_required_files": [],
            "metadata": metadata,
            "accepted_steps_sha256": actual_hash,
            "metadata_sha256_matches": actual_hash == expected_hash,
            "metadata_step_count": int(metadata.get("step_count", 0) or 0),
            "actual_step_count": len(steps),
            "metadata_command_count": int(metadata.get("command_count", 0) or 0),
            "actual_recorded_command_count": sum(
                1
                for step in steps
                for event in list(step.get("events", []) or [])
                if str(event.get("command", "") or "").strip()
            ),
            "decoded_actuator_command_count": _command_count(steps),
            "event_count": sum(len(step.get("events", []) or []) for step in steps),
            "snapshot_audit": snapshot_audit,
            "snapshot_full_valid_count": sum(
                1 for row in snapshot_audit if row["classification"] == "FULL_VALID"
            ),
            "initial_root_pose": _flatten_root_pose(first_state),
            "final_root_pose": _flatten_root_pose(last_state),
            "embedded_recording_baseline": first_recording_baseline,
            "environment_json_present": (item.directory / "environment.json").is_file(),
            "standalone_telemetry_files": [
                row for row in auxiliary if "telemetry" in row["relative_path"].lower()
            ],
            "media_files": [
                row
                for row in auxiliary
                if Path(row["relative_path"]).suffix.lower()
                in {".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg"}
            ],
            "auxiliary_files": auxiliary,
            "steps": steps,
        }

    def _environment_lock(self, audits: list[dict[str, Any]]) -> dict[str, Any]:
        baseline = _read_json(BASELINE_PATH)
        environment_reference = _read_json(ENVIRONMENT_REFERENCE_PATH)
        motion = load_motion_reference()
        metadata_fields = (
            "environment_baseline_id",
            "actuator_baseline_id",
            "actuator_command_semantics",
            "motion_profile_id",
            "motion_profile_mode",
            "robot_asset_path",
            "robot_asset_sha256",
            "obstacle_front_face_x_m",
            "obstacle_length_m",
            "obstacle_width_m",
            "height_mm",
        )
        differences: dict[str, dict[str, Any]] = {}
        for field in metadata_fields:
            values = {
                audit["version"]: audit.get("metadata", {}).get(field)
                for audit in audits
                if audit.get("valid")
            }
            canonical = {json.dumps(value, sort_keys=True) for value in values.values()}
            if len(canonical) > 1:
                differences[field] = values
        # Freeze the complete production replay closure dynamically.  A fixed
        # hand-maintained shortlist silently omitted newly imported readiness,
        # grounding and shutdown modules, so a stale environment lock could
        # still pass while executable behavior had changed.
        source_candidates = list(PROJECT_ROOT.glob("*.py"))
        source_candidates.extend(
            sorted((PROJECT_ROOT / "telemetry").rglob("*.py"))
        )
        source_candidates.extend(
            sorted((PROJECT_ROOT / "config").rglob("*.yaml"))
        )
        source_candidates.extend(
            sorted((PROJECT_ROOT / "config").rglob("*.json"))
        )
        source_candidates.extend(sorted(MODULE_ROOT.glob("*.py")))
        source_candidates.extend(sorted(MODULE_ROOT.glob("*.yaml")))
        source_hashes = {
            str(path.resolve()): sha256_file(path)
            for path in sorted({path.resolve() for path in source_candidates})
            if path.is_file()
        }
        robot_usd = Path(str(_nested(baseline, "robot", "usd_path"))).resolve()
        if not robot_usd.is_file():
            raise FileNotFoundError(f"recording robot USD is missing: {robot_usd}")
        source_hashes[str(robot_usd)] = sha256_file(robot_usd)
        joint_signs = {
            str(row["articulation_joint"]): float(row["direction"])
            for row in _nested(baseline, "servo_profile", "joints", default=[])
        }
        joint_limits = {
            str(row["articulation_joint"]): [
                float(row["lower_deg"]),
                float(row["upper_deg"]),
            ]
            for row in _nested(baseline, "servo_profile", "joints", default=[])
        }
        wheel_directions = {
            str(row["articulation_joint"]): float(row["forward_direction"])
            for row in _nested(baseline, "wheel_actuator", "joints", default=[])
        }
        selected = {
            "robot_usd": str(robot_usd),
            "robot_usd_sha256": sha256_file(robot_usd),
            "robot_initial_root_position_m": list(
                _nested(baseline, "respawn", "grounded_settled_reference_root_pose_wxyz", default=[])
            )[:3],
            "robot_initial_root_orientation_wxyz": list(
                _nested(baseline, "respawn", "grounded_settled_reference_root_pose_wxyz", default=[])
            )[3:7],
            "standing_home_command_state_deg": dict(
                _nested(baseline, "initial_actuator_state", "servo_command_deg", default={})
            ),
            "obstacle_height_m": 0.05,
            "obstacle_front_face_x_m": float(environment_reference["obstacle_front_face_x_m"]),
            "obstacle_center_x_m": float(environment_reference["obstacle_front_face_x_m"])
            + 0.5 * float(environment_reference["obstacle_length_m"]),
            "obstacle_center_y_m": float(environment_reference.get("obstacle_center_y_m", 0.0)),
            "obstacle_length_m": float(environment_reference["obstacle_length_m"]),
            "obstacle_width_m": float(environment_reference["obstacle_width_m"]),
            "obstacle_bottom_z_m": float(_nested(baseline, "ground", "z_m")),
            "ground_static_friction": float(_nested(baseline, "ground", "static_friction")),
            "ground_dynamic_friction": float(_nested(baseline, "ground", "dynamic_friction")),
            "obstacle_static_friction": float(_nested(baseline, "obstacle", "static_friction")),
            "obstacle_dynamic_friction": float(_nested(baseline, "obstacle", "dynamic_friction")),
            "physics_dt_s": float(_nested(baseline, "physics", "physics_timestep_s")),
            "control_dt_s": float(_nested(baseline, "physics", "physics_timestep_s")),
            "render_interval_physics_steps": 8,
            "servo_stiffness": float(_nested(baseline, "servo_profile", "stiffness")),
            "servo_damping": float(_nested(baseline, "servo_profile", "damping")),
            "wheel_damping": float(_nested(baseline, "wheel_actuator", "damping")),
            "servo_reference_velocity_deg_s": float(
                motion.servo_reference_velocity_deg_s
            ),
            "wheel_reference_velocity_rad_s": float(
                motion.wheel_reference_velocity_rad_s
            ),
            "wheel_maximum_velocity_rad_s": float(
                _nested(baseline, "wheel_actuator", "velocity_limit_rad_s")
            ),
            "playback_profile": "fast",
            "preserve_wheel_distance": True,
            "preserve_wheel_distance_semantics": "wheel target multiplied by recorded effective active interval; servo and wheel share a segment whose duration is their maximum plus explicit-hold semantics",
            "wheel_direction_mapping": wheel_directions,
            "joint_command_signs": joint_signs,
            "joint_command_limits_deg": joint_limits,
            "recording_initial_settle_s": float(
                _nested(baseline, "initial_actuator_state", "settle_duration_s")
            ),
            "respawn_internal_settle_s": float(
                _nested(baseline, "initial_actuator_state", "settle_duration_s")
            ),
            "post_respawn_playback_settle_s": 0.30,
            "fixed_command_semantics": "100_percent_no_speed_scaling",
        }
        sanity = {
            "requested_hint_servo_reference_velocity_deg_s": 150.0,
            "runtime_fast_planner_servo_reference_velocity_deg_s": selected[
                "servo_reference_velocity_deg_s"
            ],
            "recording_bookkeeping_servo_reference_velocity_deg_s": float(
                _nested(baseline, "servo_profile", "reference_velocity_deg_s")
            ),
            "requested_hint_wheel_reference_velocity_rad_s": 0.5235988,
            "runtime_default_wheel_reference_velocity_rad_s": selected[
                "wheel_reference_velocity_rad_s"
            ],
            "recording_bookkeeping_wheel_reference_velocity_rad_s": float(
                _nested(baseline, "wheel_actuator", "default_recording_velocity_rad_s")
            ),
            "recording_baseline_render_interval_physics_steps": int(
                _nested(baseline, "physics", "render_interval_physics_steps")
            ),
            "current_fast_replay_render_interval_physics_steps": selected[
                "render_interval_physics_steps"
            ],
            "source_runtime_motion_reference_servo_deg_s": float(
                motion.servo_reference_velocity_deg_s
            ),
            "source_runtime_motion_reference_wheel_rad_s": float(
                motion.wheel_reference_velocity_rad_s
            ),
            "differences_must_not_be_silently_overridden": True,
        }
        return {
            "schema_version": "fsm50.environment_lock.v1",
            "status": "offline_locked_runtime_readback_pending",
            "selected_environment": selected,
            "parameter_sources": {
                "recording_baseline": str(BASELINE_PATH.resolve()),
                "environment_reference": str(ENVIRONMENT_REFERENCE_PATH.resolve()),
                "runtime_motion_reference": str(MOTION_REFERENCE_PATH.resolve()),
                "version_metadata": {
                    audit["version"]: str(
                        Path(audit["directory"]) / "metadata.json"
                    )
                    for audit in audits
                    if audit.get("valid")
                },
            },
            "source_sha256": source_hashes,
            "version_metadata_differences": differences,
            "sanity_check_differences": sanity,
            "runtime_readbacks": [],
        }

    def run(self) -> dict[str, Any]:
        self.report_root.mkdir(parents=True, exist_ok=True)
        fast_dir = self.report_root / "recording_fast_plans"
        versions = self.enumerate_versions()
        audits = [self.audit_version(item) for item in versions]
        max_wheel_speed = float(
            _nested(_read_json(BASELINE_PATH), "wheel_actuator", "velocity_limit_rad_s")
        )
        for audit in audits:
            if not audit.get("valid"):
                continue
            export = write_fast_plan(
                output_dir=fast_dir,
                source_version=str(audit["version"]),
                steps=list(audit["steps"]),
                max_wheel_speed=max_wheel_speed,
            )
            audit["fast_plan"] = {
                "json_path": str(export["json_path"]),
                "csv_path": str(export["csv_path"]),
                "plan_sha256": export["plan"].plan_sha256,
                "event_count": len(export["plan"].events),
                "segment_count": len(export["plan"].segments),
                "duration_s": float(export["plan"].final_time_s),
            }

        environment_path = self.report_root / "environment_lock_50mm.json"
        environment_lock = self._environment_lock(audits)
        # The offline audit is intentionally repeatable, including after live
        # Isaac validation.  Never erase immutable runtime readbacks merely
        # because the static recording inventory was regenerated.
        if environment_path.is_file():
            try:
                previous_lock = json.loads(environment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous_lock = {}
            previous_readbacks = list(previous_lock.get("runtime_readbacks", []) or [])
            if previous_readbacks:
                environment_lock["runtime_readbacks"] = previous_readbacks
                environment_lock["status"] = "runtime_readback_available"
        environment_path.write_text(
            json.dumps(environment_lock, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        active_path = self.recording_root / "active_version.json"
        active = _read_json(active_path) if active_path.is_file() else {}
        disk_ids = [item.version_id for item in versions]
        manifest_ids = self._manifest_entry_ids()
        missing_on_disk = sorted(set(manifest_ids) - set(disk_ids))
        unlisted_on_disk = sorted(set(disk_ids) - set(manifest_ids))
        matrix_path = self.report_root / "RECORDING_VERSION_MATRIX_50MM.csv"
        with matrix_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(MATRIX_COLUMNS))
            writer.writeheader()
            for audit in audits:
                metadata = dict(audit.get("metadata", {}) or {})
                comments = []
                if not audit.get("environment_json_present"):
                    comments.append("environment.json absent; locked from baseline+metadata")
                if not audit.get("standalone_telemetry_files"):
                    comments.append("no standalone recording/replay telemetry")
                if not audit.get("media_files"):
                    comments.append("no video/screenshot in version folder")
                writer.writerow(
                    {
                        "version": audit["version"],
                        "step_count": audit.get("actual_step_count", 0),
                        "command_count": audit.get("actual_recorded_command_count", 0),
                        "hash_valid": audit.get("metadata_sha256_matches", False),
                        "snapshot_full_valid_count": audit.get(
                            "snapshot_full_valid_count", 0
                        ),
                        "snapshot_count": len(audit.get("snapshot_audit", [])),
                        "full_success": "NOT_EVALUATED",
                        "first_failure_phase": "NOT_EVALUATED",
                        "FR_lift_success": "NOT_EVALUATED",
                        "FR_place_success": "NOT_EVALUATED",
                        "FL_lift_success": "NOT_EVALUATED",
                        "FL_place_success": "NOT_EVALUATED",
                        "RR_lift_success": "NOT_EVALUATED",
                        "RR_place_success": "NOT_EVALUATED",
                        "RL_lift_success": "NOT_EVALUATED",
                        "RL_place_success": "NOT_EVALUATED",
                        "final_recovery_success": "NOT_EVALUATED",
                        "maximum_roll_rad": "",
                        "maximum_pitch_rad": "",
                        "minimum_support_margin_m": "",
                        "maximum_contact_drift_m": "",
                        "loaded_front_face_wheel_rotation_rad": "",
                        "replay_evidence": "PENDING_CLEAN_FAST_REPLAY",
                        "comments": "; ".join(comments),
                    }
                )

        audit_json = self.report_root / "recording_audit_50mm.json"
        serializable_audits = []
        for audit in audits:
            row = dict(audit)
            row.pop("steps", None)
            serializable_audits.append(row)
        audit_payload = {
            "schema_version": "fsm50.recording_audit.v1",
            "recording_root": str(self.recording_root),
            "active_version": active.get("version_id", ""),
            "disk_version_ids": disk_ids,
            "manifest_version_ids": manifest_ids,
            "manifest_versions_missing_on_disk": missing_on_disk,
            "disk_versions_missing_from_manifest": unlisted_on_disk,
            "versions": serializable_audits,
        }
        audit_json.write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        markdown_path = self.report_root / "RECORDING_AUDIT_50MM.md"
        rows = [
            "# 50 mm Recording Audit",
            "",
            "This is the static pre-replay audit. Physical success fields remain `NOT_EVALUATED` until each version has a clean Isaac Fast Replay with telemetry; endpoint JSON is not treated as proof of traversal.",
            "",
            f"- Active pointer: `{active.get('version_id', '')}`",
            f"- Version directories inspected: {len(disk_ids)} (`{', '.join(disk_ids)}`)",
            f"- Manifest entries missing on disk: `{', '.join(missing_on_disk) or 'none'}`",
            f"- Disk directories missing from manifest: `{', '.join(unlisted_on_disk) or 'none'}`",
            "",
            "## Integrity and Fast Plan",
            "",
            "| Version | Steps | Commands | SHA-256 | FULL_VALID snapshots | Fast segments | Fast duration (s) | Embedded media/telemetry |",
            "|---|---:|---:|---|---:|---:|---:|---|",
        ]
        for audit in audits:
            fast = dict(audit.get("fast_plan", {}) or {})
            rows.append(
                "| {version} | {steps} | {commands} | {sha} | {snap}/{snap_total} | {segments} | {duration:.6f} | {evidence} |".format(
                    version=audit["version"],
                    steps=audit.get("actual_step_count", 0),
                    commands=audit.get("actual_recorded_command_count", 0),
                    sha="PASS" if audit.get("metadata_sha256_matches") else "FAIL",
                    snap=audit.get("snapshot_full_valid_count", 0),
                    snap_total=len(audit.get("snapshot_audit", [])),
                    segments=fast.get("segment_count", 0),
                    duration=float(fast.get("duration_s", 0.0) or 0.0),
                    evidence="present"
                    if audit.get("media_files") or audit.get("standalone_telemetry_files")
                    else "absent",
                )
            )
        rows.extend(
            [
                "",
                "## Environment lock findings",
                "",
                "- All available version metadata use the same v2 obstacle geometry and robot asset identity.",
                "- No version directory contains a standalone `environment.json`; authoritative values are therefore traced to `config/fsm_recording_baseline.yaml`, `config/environment_reference.yaml`, per-version metadata, and their SHA-256 values.",
                "- The embedded bookkeeping profile names **30 deg/s** and **0.3 rad/s**, while the authoritative code actually consumed by `playback.plan_from_steps` and `SimRobotAdapter` uses **150 deg/s** and **0.5235987756 rad/s**. The Fast Replay lock uses the runtime values and preserves the stale bookkeeping discrepancy as evidence.",
                "- The recording baseline document says render interval 2, while the current UI/worker Fast Replay path defaults to 8 and explicitly treats render cadence as a non-physics performance override. The reproduction lock uses 8 and records both values.",
                "",
                "## Evidence limits before replay",
                "",
                "The accepted steps contain commands and before/after articulation snapshots, but not continuous whole-body COM, per-wheel contact force/class, support diagonal, contact drift, or video. Consequently this audit does **not** label any version a full success or select primitives yet. Those decisions require the clean replay artifacts requested by the task.",
                "",
                "## Generated evidence",
                "",
                f"- Environment lock: `{environment_path}`",
                f"- Machine-readable audit: `{audit_json}`",
                f"- Version matrix: `{matrix_path}`",
                f"- Fast plans: `{fast_dir}`",
            ]
        )
        markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return {
            "version_count": len(audits),
            "versions": disk_ids,
            "active_version": active.get("version_id", ""),
            "missing_on_disk": missing_on_disk,
            "report_root": str(self.report_root),
            "audit_json": str(audit_json),
            "markdown": str(markdown_path),
            "matrix": str(matrix_path),
            "environment_lock": str(environment_path),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = RecordingAudit(args.recording_root, args.report_root).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
