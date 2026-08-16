"""Fail-closed residual authority for the first FSM50 PPO envelope.

This module is deliberately Isaac-free and has no controller/worker side
effects.  It validates one canonical, SHA-bound configuration and answers a
single question: which additive command residual, if any, is authorized for a
reviewed source/state/action tuple?

The nominal FSM command remains the authority.  Unknown sources, unlisted
states, non-source decisions, zero-wheel nominal targets, and every excluded
source receive an immediate all-zero residual.  Checkpoint-derived action
truth is rejected rather than silently accepted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .fsm50_direct_command_residual import ResidualPhaseContract


SCHEMA_VERSION = "fsm50.residual_envelope.v1"
ENVELOPE_ID = "fsm50-residual-r1-reviewed-success-sources-v1"
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "fsm50_residual_r1_envelope.json"

CANONICAL_ACTION_ORDER = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
SERVO_ACTION_COUNT = 8
ACTION_DIMENSION = len(CANONICAL_ACTION_ORDER)
ZERO_RESIDUAL = (0.0,) * ACTION_DIMENSION
ALLOWED_DECISION_PROVENANCE = ("SOURCE_ACTION",)

S5 = "S5_PRE_RR_COM_SHIFT"
S7 = "S7_PRE_RL_SUPPORT_SETUP"
S8 = "S8_RL_COM_SHIFT_AND_TRAVERSE"
S10 = "S10_POSTURE_RECOVERY"
R1_STATES = (S5, S7, S8, S10)

V003 = "v003_20260805_224517_157723_manual"
V008 = "v008_20260806_211408_578700_manual"
V009 = "v009_20260806_215232_433234_manual"
V010_FAILED = "v010_20260806_220745_363972_manual"
ALLOWED_SOURCE_VERSIONS = (V003, V008, V009)

GLOBAL_LIMITS = MappingProxyType(
    {
        "active_servo_max_abs_deg": 2.0,
        "support_servo_max_abs_deg": 1.0,
        "wheel_max_abs_rad_s": 0.10,
        "servo_slew_deg_s": 10.0,
        "wheel_slew_rad_s2": 0.5,
        "wheel_nominal_nonzero_epsilon_rad_s": 1.0e-12,
    }
)

EVIDENCE_STATISTICS = MappingProxyType(
    {
        "servo_nonzero_delta_count": 324,
        "servo_delta_deg_min": 0.5,
        "servo_delta_deg_p10": 1.1,
        "servo_delta_deg_median": 4.3,
        "servo_delta_deg_p90": 20.1,
        "servo_delta_deg_max": 47.6,
        "wheel_nonzero_delta_count": 141,
        "wheel_delta_rad_s_min": 0.3,
        "wheel_delta_rad_s_p10": 0.3,
        "wheel_delta_rad_s_median": 0.3,
        "wheel_delta_rad_s_p90": 1.07,
        "wheel_delta_rad_s_max": 2.0943951023931953,
    }
)


def _mask(*values: float) -> tuple[float, ...]:
    if len(values) != ACTION_DIMENSION:
        raise AssertionError("internal residual mask has the wrong width")
    return tuple(float(value) for value in values)


EXPECTED_STATE_MASKS: Mapping[str, Mapping[str, tuple[float, ...]]] = MappingProxyType(
    {
        V003: MappingProxyType(
            {
                S5: ZERO_RESIDUAL,
                S7: ZERO_RESIDUAL,
                S8: _mask(1, 1, 1, 0, 2, 2, 0, 0, 0.1, 0, 0, 0),
                S10: _mask(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
            }
        ),
        V008: MappingProxyType(
            {
                S5: _mask(1, 1, 0, 1, 0, 1, 0, 1, 0, 0.1, 0, 0),
                S7: ZERO_RESIDUAL,
                S8: _mask(1, 1, 1, 1, 2, 2, 1, 1, 0.1, 0.1, 0.1, 0.1),
                S10: _mask(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
            }
        ),
        V009: MappingProxyType(
            {
                S5: _mask(1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
                S7: _mask(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.1, 0),
                S8: _mask(1, 1, 1, 1, 2, 2, 1, 1, 0.1, 0.1, 0.1, 0.1),
                S10: _mask(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
            }
        ),
    }
)


EXPECTED_SOURCE_EVIDENCE: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        V003: MappingProxyType(
            {
                "bundle_sha256": "5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78",
                "run_dir": (
                    "fsm_50mm_recording_derived_v3/runs/"
                    "v003_macro_fsm_completion_aware_coalesced_r4/"
                    "v003_20260805_224517_157723_manual/baseline/"
                    "20260815T105857_601132Z_baseline_00_5742cda6e438"
                ),
                "accepted_steps": MappingProxyType(
                    {
                        "path": (
                            "saved_height_steps_fsm_reference_v2/height_050mm/versions/"
                            "v003_20260805_224517_157723_manual/accepted_steps.jsonl"
                        ),
                        "sha256": "06e13153b7ba75a4283e117d875f1da4895748835a9032c6faadef2bda25b394",
                    }
                ),
                "manual_verdict": MappingProxyType(
                    {
                        "path": "manual_macro_video_verdict.json",
                        "sha256": "63141b536216b8aac1c123df13db3eb39392d00ac3a6a808b481f023235f1782",
                    }
                ),
                "task_inputs": MappingProxyType(
                    {
                        "path": "macro_task_inputs.json",
                        "sha256": "10cb7ad90b9b1d5ceeab43b24813596dd8ea02ec4d206c6d950e4983cfdb8848",
                    }
                ),
                "video": MappingProxyType(
                    {
                        "path": "actual_viewport_video.mp4",
                        "sha256": "9a446fbfab7aae0ae84d167e5ac15e1bf38a3495582c7aa04d884fb27fed00f7",
                    }
                ),
                "worker_result": MappingProxyType(
                    {
                        "path": "worker_macro_fsm_result.json",
                        "sha256": "dc7dd6d4e08d82ad82e5eab847194c25342a6cab72d6df7e4ae0cae75d5daa30",
                    }
                ),
            }
        ),
        V008: MappingProxyType(
            {
                "bundle_sha256": "0a47cee05cbb41a3a09b23836386bddcf9ae1ab2aca5bbe451ee20a0cbaa8c41",
                "run_dir": (
                    "fsm_50mm_recording_derived_v3/runs/"
                    "cross_version_macro_fsm_completion_aware_coalesced_r4/"
                    "v008_20260806_211408_578700_manual/trials/"
                    "20260815T113449_254770Z_cross_version_00_0a47cee05cbb"
                ),
                "accepted_steps": MappingProxyType(
                    {
                        "path": (
                            "saved_height_steps_fsm_reference_v2/height_050mm/versions/"
                            "v008_20260806_211408_578700_manual/accepted_steps.jsonl"
                        ),
                        "sha256": "737b23a2152275e6ca716cd49006f3f34f1ae01466ac16960bc64d528dc4ea8c",
                    }
                ),
                "manual_verdict": MappingProxyType(
                    {
                        "path": "manual_macro_video_verdict.json",
                        "sha256": "05721928e11a83708bb542538a5c06884fd47023d9353a614d73900ab05095b1",
                    }
                ),
                "task_inputs": MappingProxyType(
                    {
                        "path": "macro_task_inputs.json",
                        "sha256": "5b1c1d7efc26fbeaff17a4f42a7cbf0d12df7e0ef2ef1197383962b4a86cfa12",
                    }
                ),
                "video": MappingProxyType(
                    {
                        "path": "actual_viewport_video.mp4",
                        "sha256": "4920b7f2f79f2caf61423b9e0c105794654d4d153313e02ad60682ac71ccf3a1",
                    }
                ),
                "worker_result": MappingProxyType(
                    {
                        "path": "worker_macro_fsm_result.json",
                        "sha256": "332ba356dbf2d52d271bed991509d9f9378a6aec9898df035bda7a96725abbc9",
                    }
                ),
            }
        ),
        V009: MappingProxyType(
            {
                "bundle_sha256": "0e5686354eed7d89fb57d3aba722602eee7115b44749d531b6634dd0420ee919",
                "run_dir": (
                    "fsm_50mm_recording_derived_v3/runs/"
                    "cross_version_macro_fsm_completion_aware_coalesced_r4/"
                    "v009_20260806_215232_433234_manual/trials/"
                    "20260815T114309_706871Z_cross_version_00_0e5686354eed"
                ),
                "accepted_steps": MappingProxyType(
                    {
                        "path": (
                            "saved_height_steps_fsm_reference_v2/height_050mm/versions/"
                            "v009_20260806_215232_433234_manual/accepted_steps.jsonl"
                        ),
                        "sha256": "60db11a138088fd4a4c2886a681e0a195402a896f66932220c6c393d3d6c17c1",
                    }
                ),
                "manual_verdict": MappingProxyType(
                    {
                        "path": "manual_macro_video_verdict.json",
                        "sha256": "1fe1e673b44b812a10698d043554f8c761695699f289fb3da1741fb317daaa93",
                    }
                ),
                "task_inputs": MappingProxyType(
                    {
                        "path": "macro_task_inputs.json",
                        "sha256": "6df6529da34104afb5f64f9196ddd6c52720882f54461359ff5ec24cf6acb5cf",
                    }
                ),
                "video": MappingProxyType(
                    {
                        "path": "actual_viewport_video.mp4",
                        "sha256": "2168ba48840b460297e4953421b83a3ce304619e49492ab1aed50ea48ee722c3",
                    }
                ),
                "worker_result": MappingProxyType(
                    {
                        "path": "worker_macro_fsm_result.json",
                        "sha256": "64c3ad74ae5248a0371344a9f166db00f0c5eb7ca0e3461fe985573de697ba0f",
                    }
                ),
            }
        ),
    }
)

EXPECTED_HARD_EXCLUSIONS = MappingProxyType(
    {
        "source_versions": (V010_FAILED,),
        "unknown_sources_default_zero": True,
        "legacy_checkpoints_permitted": False,
        "old_action_truth_permitted": False,
    }
)


class ResidualEnvelopeError(ValueError):
    """Raised when residual authority or its provenance is not valid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ResidualEnvelopeError(f"{label} must be a 64-character SHA256")
    return digest


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResidualEnvelopeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ResidualEnvelopeError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except ResidualEnvelopeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResidualEnvelopeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResidualEnvelopeError(f"{label} must contain a JSON object: {path}")
    return payload


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical JSON while excluding its self-declared hash field."""

    if not isinstance(payload, Mapping):
        raise ResidualEnvelopeError("residual envelope payload must be a mapping")
    body = copy.deepcopy(dict(payload))
    body.pop("canonical_payload_sha256", None)
    try:
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidualEnvelopeError(f"payload is not canonical-JSON compatible: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualEnvelopeError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ResidualEnvelopeError(
            f"{label} keys mismatch: missing={sorted(expected - observed)!r} "
            f"unexpected={sorted(observed - expected)!r}"
        )


def _finite_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResidualEnvelopeError(f"{label} must be a {ACTION_DIMENSION}-element array")
    if len(value) != ACTION_DIMENSION:
        raise ResidualEnvelopeError(f"{label} must contain {ACTION_DIMENSION} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ResidualEnvelopeError(f"{label} must contain only numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ResidualEnvelopeError(f"{label} must contain only finite numbers")
    return result


def _resolve_project_relative(project_root: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ResidualEnvelopeError(f"{label} path is required")
    declared = Path(text)
    if declared.is_absolute() or ".." in declared.parts:
        raise ResidualEnvelopeError(f"{label} must be a project-relative path")
    resolved = (project_root / declared).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ResidualEnvelopeError(f"{label} escapes the project root") from exc
    return resolved


def _resolve_run_relative(run_dir: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ResidualEnvelopeError(f"{label} path is required")
    declared = Path(text)
    if declared.is_absolute() or ".." in declared.parts:
        raise ResidualEnvelopeError(f"{label} must be a run-relative path")
    resolved = (run_dir / declared).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ResidualEnvelopeError(f"{label} escapes the reviewed run") from exc
    return resolved


def _resolve_declared_artifact(value: Any, *, run_dir: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ResidualEnvelopeError(f"{label} is required")
    declared = Path(text)
    resolved = (declared if declared.is_absolute() else run_dir / declared).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ResidualEnvelopeError(f"{label} escapes the reviewed run") from exc
    return resolved


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    digest = _validated_sha256(expected, label=f"{label} SHA256")
    if not path.is_file():
        raise ResidualEnvelopeError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ResidualEnvelopeError(
            f"{label} SHA256 mismatch: expected={digest} actual={actual}"
        )
    return actual


def _plain_expected_evidence(source_version: str) -> dict[str, Any]:
    expected = EXPECTED_SOURCE_EVIDENCE[source_version]
    return {
        "run_dir": expected["run_dir"],
        "accepted_steps": dict(expected["accepted_steps"]),
        "manual_verdict": dict(expected["manual_verdict"]),
        "task_inputs": dict(expected["task_inputs"]),
        "video": dict(expected["video"]),
        "worker_result": dict(expected["worker_result"]),
    }


def _verify_reviewed_source_evidence(
    *,
    source_version: str,
    bundle_sha256: str,
    evidence: Mapping[str, Any],
    project_root: Path,
) -> None:
    run_dir = _resolve_project_relative(
        project_root, evidence["run_dir"], label=f"{source_version}.run_dir"
    )
    if not run_dir.is_dir():
        raise ResidualEnvelopeError(f"reviewed run is missing: {run_dir}")

    accepted_row = _mapping(
        evidence["accepted_steps"], label=f"{source_version}.accepted_steps"
    )
    accepted_path = _resolve_project_relative(
        project_root,
        accepted_row["path"],
        label=f"{source_version}.accepted_steps",
    )
    _verify_hash(
        accepted_path,
        accepted_row["sha256"],
        label=f"{source_version} accepted_steps",
    )

    resolved: dict[str, Path] = {}
    for artifact_key in ("manual_verdict", "task_inputs", "video", "worker_result"):
        row = _mapping(evidence[artifact_key], label=f"{source_version}.{artifact_key}")
        path = _resolve_run_relative(
            run_dir, row["path"], label=f"{source_version}.{artifact_key}"
        )
        _verify_hash(path, row["sha256"], label=f"{source_version} {artifact_key}")
        resolved[artifact_key] = path

    verdict = _read_json_mapping(
        resolved["manual_verdict"], label=f"{source_version} manual verdict"
    )
    if verdict.get("schema_version") != "fsm50.manual_macro_video_verdict.v1":
        raise ResidualEnvelopeError(f"{source_version} manual verdict schema mismatch")
    if verdict.get("review_complete") is not True:
        raise ResidualEnvelopeError(f"{source_version} manual review is incomplete")
    if str(verdict.get("source_version", "")) != source_version:
        raise ResidualEnvelopeError(f"{source_version} manual verdict source mismatch")
    if str(verdict.get("bundle_sha256", "")).lower() != bundle_sha256:
        raise ResidualEnvelopeError(f"{source_version} manual verdict bundle mismatch")
    if _resolve_declared_artifact(
        verdict.get("run_dir"), run_dir=run_dir, label="manual verdict run_dir"
    ) != run_dir:
        raise ResidualEnvelopeError(f"{source_version} manual verdict run_dir mismatch")

    verdict_artifact_fields = {
        "task_inputs": ("task_inputs_path", "task_inputs_sha256"),
        "video": ("video_path", "video_sha256"),
        "worker_result": ("worker_result_path", "worker_result_sha256"),
    }
    for artifact_key, (path_key, sha_key) in verdict_artifact_fields.items():
        declared = _resolve_declared_artifact(
            verdict.get(path_key), run_dir=run_dir, label=f"manual verdict {path_key}"
        )
        if declared != resolved[artifact_key]:
            raise ResidualEnvelopeError(
                f"{source_version} manual verdict {artifact_key} path mismatch"
            )
        expected_sha = str(evidence[artifact_key]["sha256"]).lower()
        if str(verdict.get(sha_key, "")).lower() != expected_sha:
            raise ResidualEnvelopeError(
                f"{source_version} manual verdict {artifact_key} SHA mismatch"
            )

    visual = _mapping(verdict.get("verdict"), label=f"{source_version}.verdict")
    required_true = (
        "task_completed",
        "required_leg_lift_completed",
        "body_crossed_front_face",
        "final_recoverable",
        "posture_incomplete",
    )
    required_false = (
        "robot_fell",
        "body_stuck",
        "dangerous_body_collision",
        "severe_penetration",
        "joint_limit_violation",
        "wheel_drive_up_without_required_lift",
    )
    if any(visual.get(key) is not True for key in required_true):
        raise ResidualEnvelopeError(f"{source_version} reviewed success claims are incomplete")
    if any(visual.get(key) is not False for key in required_false):
        raise ResidualEnvelopeError(f"{source_version} reviewed hard-failure claims are unsafe")

    worker_result = _read_json_mapping(
        resolved["worker_result"], label=f"{source_version} worker result"
    )
    required_worker_values = {
        "schema_version": "fsm50.worker_macro_fsm_session.v1",
        "source_version": source_version,
        "bundle_sha256": bundle_sha256,
        "macro_fsm_complete": True,
        "controller_terminal_outcome": "TASK_SUCCESS_POSTURE_INCOMPLETE",
        "error": "",
        "source_action_coverage_complete": True,
        "segment_completion_coverage_complete": True,
        "safe_stop_verified": True,
        "video_writer_quiesced": True,
    }
    for key, expected_value in required_worker_values.items():
        if worker_result.get(key) != expected_value:
            raise ResidualEnvelopeError(
                f"{source_version} worker result {key} mismatch: "
                f"expected={expected_value!r} observed={worker_result.get(key)!r}"
            )
    if _resolve_declared_artifact(
        worker_result.get("run_dir"), run_dir=run_dir, label="worker result run_dir"
    ) != run_dir:
        raise ResidualEnvelopeError(f"{source_version} worker result run_dir mismatch")


@dataclass(frozen=True)
class ResidualDecision:
    """One offline authorization preview in canonical action order.

    This record is diagnostic only.  It deliberately contains no final command
    map; ``compose_direct_command_residual`` is the sole command composer.
    """

    source_version: str
    state_id: str
    decision_provenance: str
    authority_granted: bool
    authority_reason: str
    maximum_abs_residual: tuple[float, ...]
    requested_residual: tuple[float, ...]
    applied_residual: tuple[float, ...]
    clipped: bool
    slew_limited: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.residual_authorization.v1",
            "source_version": self.source_version,
            "state_id": self.state_id,
            "decision_provenance": self.decision_provenance,
            "authority_granted": self.authority_granted,
            "authority_reason": self.authority_reason,
            "canonical_action_order": list(CANONICAL_ACTION_ORDER),
            "maximum_abs_residual": list(self.maximum_abs_residual),
            "requested_residual": list(self.requested_residual),
            "applied_residual": list(self.applied_residual),
            "clipped": self.clipped,
            "slew_limited": self.slew_limited,
        }


@dataclass(frozen=True)
class ResidualEnvelope:
    """Validated R1 authority; construction is only through validation helpers."""

    canonical_sha256: str
    evidence_verified: bool
    project_root: Path
    state_masks: Mapping[str, Mapping[str, tuple[float, ...]]]

    @property
    def source_versions(self) -> tuple[str, ...]:
        return ALLOWED_SOURCE_VERSIONS

    def is_source_allowed(self, source_version: str) -> bool:
        return str(source_version) in ALLOWED_SOURCE_VERSIONS

    def _maximum_abs_residual(
        self,
        *,
        source_version: str,
        state_id: str,
        decision_provenance: str,
        nominal_action: tuple[float, ...],
    ) -> tuple[tuple[float, ...], str]:
        source = str(source_version)
        state = str(getattr(state_id, "value", state_id))
        provenance = str(decision_provenance)
        if not self.evidence_verified:
            return ZERO_RESIDUAL, "EVIDENCE_NOT_VERIFIED"
        if source == V010_FAILED or source.startswith("v010_"):
            return ZERO_RESIDUAL, "SOURCE_EXCLUDED"
        if source not in ALLOWED_SOURCE_VERSIONS:
            return ZERO_RESIDUAL, "SOURCE_NOT_ALLOWLISTED"
        if provenance not in ALLOWED_DECISION_PROVENANCE:
            return ZERO_RESIDUAL, "DECISION_PROVENANCE_DEFAULT_ZERO"
        raw = self.state_masks[source].get(state, ZERO_RESIDUAL)
        if not any(raw):
            return ZERO_RESIDUAL, "STATE_DEFAULT_ZERO"
        bounds = list(raw)
        epsilon = float(GLOBAL_LIMITS["wheel_nominal_nonzero_epsilon_rad_s"])
        for index in range(SERVO_ACTION_COUNT, ACTION_DIMENSION):
            if abs(nominal_action[index]) <= epsilon:
                bounds[index] = 0.0
        return tuple(bounds), "AUTHORIZED"

    def maximum_abs_residual(
        self,
        *,
        source_version: str,
        state_id: str,
        nominal_action: Sequence[float],
        decision_provenance: str = "SOURCE_ACTION",
    ) -> tuple[float, ...]:
        nominal = _finite_vector(nominal_action, label="nominal_action")
        bounds, _ = self._maximum_abs_residual(
            source_version=source_version,
            state_id=state_id,
            decision_provenance=decision_provenance,
            nominal_action=nominal,
        )
        return bounds

    def phase_contract(
        self,
        *,
        source_version: str,
        profile_strategy: str,
        macro_state: str,
        subphase: str,
        nominal_servo_targets_deg: Mapping[str, Any],
        nominal_wheel_targets_rad_s: Mapping[str, Any],
        decision_provenance: str = "SOURCE_ACTION",
    ) -> "ResidualPhaseContract":
        """Return the exact contract consumed by the sole command composer."""

        from .fsm50_direct_command_residual import (
            RESIDUAL_ACTION_NAMES,
            ResidualPhaseContract,
        )

        if tuple(RESIDUAL_ACTION_NAMES) != CANONICAL_ACTION_ORDER:
            raise ResidualEnvelopeError(
                "direct-command composer action order differs from the envelope"
            )
        nominal = canonical_action_from_targets(
            nominal_servo_targets_deg, nominal_wheel_targets_rad_s
        )
        state = str(getattr(macro_state, "value", macro_state))
        bounds, _ = self._maximum_abs_residual(
            source_version=str(source_version),
            state_id=state,
            decision_provenance=str(decision_provenance),
            nominal_action=nominal,
        )
        enabled = tuple(value > 0.0 for value in bounds)
        rates = tuple(
            (
                float(GLOBAL_LIMITS["servo_slew_deg_s"])
                if index < SERVO_ACTION_COUNT
                else float(GLOBAL_LIMITS["wheel_slew_rad_s2"])
            )
            if enabled[index]
            else 0.0
            for index in range(ACTION_DIMENSION)
        )
        return ResidualPhaseContract(
            source_version=str(source_version),
            profile_strategy=str(profile_strategy),
            macro_state=state,
            subphase=str(subphase),
            enabled_mask=enabled,
            residual_min_command_units=tuple(-value for value in bounds),
            residual_max_command_units=bounds,
            maximum_rate_command_units_per_s=rates,
        )

    def authorize(
        self,
        *,
        source_version: str,
        state_id: str,
        nominal_action: Sequence[float],
        requested_residual: Sequence[float],
        previous_residual: Sequence[float] | None = None,
        dt_s: float,
        decision_provenance: str = "SOURCE_ACTION",
        checkpoint_reference: str | Path | None = None,
    ) -> ResidualDecision:
        """Preview clip/slew authority offline; never use this as dispatch output.

        Runtime command composition belongs exclusively to
        :func:`fsm50_direct_command_residual.compose_direct_command_residual`
        using :meth:`phase_contract`.
        """

        if checkpoint_reference is not None:
            raise ResidualEnvelopeError(
                "checkpoint-derived action truth is forbidden by the R1 envelope"
            )
        nominal = _finite_vector(nominal_action, label="nominal_action")
        requested = _finite_vector(requested_residual, label="requested_residual")
        previous = (
            ZERO_RESIDUAL
            if previous_residual is None
            else _finite_vector(previous_residual, label="previous_residual")
        )
        try:
            dt = float(dt_s)
        except (TypeError, ValueError) as exc:
            raise ResidualEnvelopeError("dt_s must be a finite positive number") from exc
        if not math.isfinite(dt) or dt <= 0.0:
            raise ResidualEnvelopeError("dt_s must be a finite positive number")

        state = str(getattr(state_id, "value", state_id))
        provenance = str(decision_provenance)
        bounds, reason = self._maximum_abs_residual(
            source_version=str(source_version),
            state_id=state,
            decision_provenance=provenance,
            nominal_action=nominal,
        )
        if not any(bounds):
            applied = ZERO_RESIDUAL
            clipped = requested != ZERO_RESIDUAL
            slew_limited = False
        else:
            bounded = tuple(
                max(-limit, min(limit, value))
                for value, limit in zip(requested, bounds)
            )
            applied_values: list[float] = []
            for index, (target, prior, limit) in enumerate(zip(bounded, previous, bounds)):
                slew = (
                    float(GLOBAL_LIMITS["servo_slew_deg_s"])
                    if index < SERVO_ACTION_COUNT
                    else float(GLOBAL_LIMITS["wheel_slew_rad_s2"])
                )
                maximum_step = slew * dt
                candidate = prior + max(-maximum_step, min(maximum_step, target - prior))
                applied_values.append(max(-limit, min(limit, candidate)))
            applied = tuple(applied_values)
            clipped = bounded != requested
            slew_limited = applied != bounded

        return ResidualDecision(
            source_version=str(source_version),
            state_id=state,
            decision_provenance=provenance,
            authority_granted=any(bounds),
            authority_reason=reason,
            maximum_abs_residual=bounds,
            requested_residual=requested,
            applied_residual=applied,
            clipped=clipped,
            slew_limited=slew_limited,
        )


def canonical_action_from_targets(
    servo_targets_deg: Mapping[str, Any],
    wheel_targets_rad_s: Mapping[str, Any],
) -> tuple[float, ...]:
    """Create the canonical 12-D mixed-unit action from complete target maps."""

    expected_servos = set(CANONICAL_ACTION_ORDER[:SERVO_ACTION_COUNT])
    expected_wheels = set(CANONICAL_ACTION_ORDER[SERVO_ACTION_COUNT:])
    if set(servo_targets_deg) != expected_servos:
        raise ResidualEnvelopeError("servo target keys differ from canonical action order")
    if set(wheel_targets_rad_s) != expected_wheels:
        raise ResidualEnvelopeError("wheel target keys differ from canonical action order")
    return _finite_vector(
        [
            *(servo_targets_deg[name] for name in CANONICAL_ACTION_ORDER[:SERVO_ACTION_COUNT]),
            *(wheel_targets_rad_s[name] for name in CANONICAL_ACTION_ORDER[SERVO_ACTION_COUNT:]),
        ],
        label="canonical target maps",
    )


def split_canonical_action(
    action: Sequence[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Split a canonical mixed-unit action into servo and wheel maps."""

    values = _finite_vector(action, label="canonical action")
    servos = {
        name: values[index]
        for index, name in enumerate(CANONICAL_ACTION_ORDER[:SERVO_ACTION_COUNT])
    }
    wheels = {
        name: values[index]
        for index, name in enumerate(
            CANONICAL_ACTION_ORDER[SERVO_ACTION_COUNT:], start=SERVO_ACTION_COUNT
        )
    }
    return servos, wheels


def phase_contract(
    envelope: ResidualEnvelope,
    *,
    source_version: str,
    profile_strategy: str,
    macro_state: str,
    subphase: str,
    nominal_servo_targets_deg: Mapping[str, Any],
    nominal_wheel_targets_rad_s: Mapping[str, Any],
    decision_provenance: str = "SOURCE_ACTION",
) -> "ResidualPhaseContract":
    """Pure adapter from a validated envelope to the runtime contract type."""

    if not isinstance(envelope, ResidualEnvelope):
        raise ResidualEnvelopeError("envelope must be a validated ResidualEnvelope")
    return envelope.phase_contract(
        source_version=source_version,
        profile_strategy=profile_strategy,
        macro_state=macro_state,
        subphase=subphase,
        nominal_servo_targets_deg=nominal_servo_targets_deg,
        nominal_wheel_targets_rad_s=nominal_wheel_targets_rad_s,
        decision_provenance=decision_provenance,
    )


def validate_envelope_mapping(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    verify_evidence: bool = True,
) -> ResidualEnvelope:
    """Validate a complete envelope mapping and optionally hash every artifact."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ResidualEnvelopeError(f"project root is missing: {root}")
    document = _mapping(payload, label="residual envelope")
    _exact_keys(
        document,
        {
            "schema_version",
            "envelope_id",
            "canonical_payload_sha256",
            "canonical_action_order",
            "units_by_action",
            "decision_provenance_allowlist",
            "default_max_abs_residual",
            "global_limits",
            "hard_exclusions",
            "evidence_statistics",
            "source_allowlist",
        },
        label="residual envelope",
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ResidualEnvelopeError("residual envelope schema_version mismatch")
    if document.get("envelope_id") != ENVELOPE_ID:
        raise ResidualEnvelopeError("residual envelope id mismatch")
    declared_sha = _validated_sha256(
        document.get("canonical_payload_sha256"), label="canonical_payload_sha256"
    )
    actual_sha = canonical_payload_sha256(document)
    if declared_sha != actual_sha:
        raise ResidualEnvelopeError(
            f"canonical payload SHA256 mismatch: expected={declared_sha} actual={actual_sha}"
        )
    if tuple(document.get("canonical_action_order", ())) != CANONICAL_ACTION_ORDER:
        raise ResidualEnvelopeError("canonical action order mismatch")
    expected_units = ("deg",) * SERVO_ACTION_COUNT + ("rad_s",) * (
        ACTION_DIMENSION - SERVO_ACTION_COUNT
    )
    if tuple(document.get("units_by_action", ())) != expected_units:
        raise ResidualEnvelopeError("units_by_action mismatch")
    if tuple(document.get("decision_provenance_allowlist", ())) != ALLOWED_DECISION_PROVENANCE:
        raise ResidualEnvelopeError("decision provenance allowlist mismatch")
    if _finite_vector(
        document.get("default_max_abs_residual"), label="default_max_abs_residual"
    ) != ZERO_RESIDUAL:
        raise ResidualEnvelopeError("default residual authority must be exactly zero")

    limits = _mapping(document.get("global_limits"), label="global_limits")
    if dict(limits) != dict(GLOBAL_LIMITS):
        raise ResidualEnvelopeError("global residual limits differ from reviewed R1 limits")
    statistics = _mapping(document.get("evidence_statistics"), label="evidence_statistics")
    if dict(statistics) != dict(EVIDENCE_STATISTICS):
        raise ResidualEnvelopeError("evidence statistics differ from reviewed-source derivation")

    exclusions = _mapping(document.get("hard_exclusions"), label="hard_exclusions")
    normalized_exclusions = dict(exclusions)
    normalized_exclusions["source_versions"] = tuple(
        normalized_exclusions.get("source_versions", ())
    )
    if normalized_exclusions != dict(EXPECTED_HARD_EXCLUSIONS):
        raise ResidualEnvelopeError("hard exclusions differ from the fail-closed R1 policy")

    sources = _mapping(document.get("source_allowlist"), label="source_allowlist")
    if set(sources) != set(ALLOWED_SOURCE_VERSIONS):
        raise ResidualEnvelopeError("source allowlist must be exactly v003/v008/v009")

    validated_masks: dict[str, Mapping[str, tuple[float, ...]]] = {}
    for source_version in ALLOWED_SOURCE_VERSIONS:
        source = _mapping(sources[source_version], label=f"source_allowlist.{source_version}")
        _exact_keys(
            source,
            {"bundle_sha256", "state_max_abs_residual", "evidence"},
            label=f"source_allowlist.{source_version}",
        )
        expected_evidence = EXPECTED_SOURCE_EVIDENCE[source_version]
        bundle_sha256 = _validated_sha256(
            source.get("bundle_sha256"), label=f"{source_version}.bundle_sha256"
        )
        if bundle_sha256 != expected_evidence["bundle_sha256"]:
            raise ResidualEnvelopeError(f"{source_version} bundle SHA256 mismatch")

        state_rows = _mapping(
            source.get("state_max_abs_residual"),
            label=f"{source_version}.state_max_abs_residual",
        )
        if set(state_rows) != set(R1_STATES):
            raise ResidualEnvelopeError(
                f"{source_version} must encode exactly the four R1 phase masks"
            )
        source_masks: dict[str, tuple[float, ...]] = {}
        for state_id in R1_STATES:
            observed = _finite_vector(
                state_rows[state_id], label=f"{source_version}.{state_id} mask"
            )
            if any(value < 0.0 for value in observed):
                raise ResidualEnvelopeError("residual mask bounds cannot be negative")
            expected = EXPECTED_STATE_MASKS[source_version][state_id]
            if observed != expected:
                raise ResidualEnvelopeError(
                    f"{source_version}.{state_id} differs from reviewed R1 authority"
                )
            source_masks[state_id] = observed

        evidence = _mapping(source.get("evidence"), label=f"{source_version}.evidence")
        _exact_keys(
            evidence,
            {
                "run_dir",
                "accepted_steps",
                "manual_verdict",
                "task_inputs",
                "video",
                "worker_result",
            },
            label=f"{source_version}.evidence",
        )
        if dict(evidence) != _plain_expected_evidence(source_version):
            raise ResidualEnvelopeError(f"{source_version} evidence binding mismatch")
        for row_name in (
            "accepted_steps",
            "manual_verdict",
            "task_inputs",
            "video",
            "worker_result",
        ):
            row = _mapping(evidence[row_name], label=f"{source_version}.{row_name}")
            _exact_keys(row, {"path", "sha256"}, label=f"{source_version}.{row_name}")
            _validated_sha256(row.get("sha256"), label=f"{source_version}.{row_name}.sha256")
        if verify_evidence:
            _verify_reviewed_source_evidence(
                source_version=source_version,
                bundle_sha256=bundle_sha256,
                evidence=evidence,
                project_root=root,
            )
        validated_masks[source_version] = MappingProxyType(source_masks)

    return ResidualEnvelope(
        canonical_sha256=actual_sha,
        evidence_verified=bool(verify_evidence),
        project_root=root,
        state_masks=MappingProxyType(validated_masks),
    )


def load_residual_envelope(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path = PROJECT_ROOT,
    verify_evidence: bool = True,
) -> ResidualEnvelope:
    """Load the strict JSON envelope and validate its full evidence chain."""

    path = Path(config_path).resolve()
    payload = _read_json_mapping(path, label="residual envelope config")
    return validate_envelope_mapping(
        payload,
        project_root=project_root,
        verify_evidence=verify_evidence,
    )


__all__ = [
    "ACTION_DIMENSION",
    "ALLOWED_SOURCE_VERSIONS",
    "CANONICAL_ACTION_ORDER",
    "DEFAULT_CONFIG_PATH",
    "ENVELOPE_ID",
    "GLOBAL_LIMITS",
    "ResidualDecision",
    "ResidualEnvelope",
    "ResidualEnvelopeError",
    "SCHEMA_VERSION",
    "V003",
    "V008",
    "V009",
    "V010_FAILED",
    "ZERO_RESIDUAL",
    "canonical_action_from_targets",
    "canonical_payload_sha256",
    "load_residual_envelope",
    "phase_contract",
    "sha256_file",
    "split_canonical_action",
    "validate_envelope_mapping",
]
