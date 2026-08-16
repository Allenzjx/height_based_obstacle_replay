"""Strict Gate-E R0 ZERO-residual request and worker-session wiring.

This module does not widen ``fsm50.worker_macro_fsm_request.v1``.  Gate-E uses
an independent exact schema which SHA-binds an already-valid base request, the
reviewed residual envelope, the canonical environment lock, the Macro graph,
profile library, source bundle, and every code file on the residual dispatch
path.

R0 permits only :class:`ZeroResidualPolicy`.  There is no checkpoint or
nonzero-action field in the schema.  The base WorkerMacroFSMSession must expose
both ``residual_policy`` and ``residual_contract_provider`` keyword hooks before
this module will construct it; until then construction fails closed while
request validation remains usable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .fsm50_direct_command_residual import ResidualContractError, ZeroResidualPolicy
from .fsm50_residual_envelope import (
    ALLOWED_SOURCE_VERSIONS,
    DEFAULT_CONFIG_PATH,
    ResidualEnvelope,
    ResidualEnvelopeError,
    load_residual_envelope,
)
from .worker_macro_fsm_session import (
    WorkerMacroFSMRequest,
    WorkerMacroFSMSession,
    load_worker_macro_fsm_request,
)


REQUEST_SCHEMA = "fsm50.worker_residual_macro_fsm_request.v1"
START_SCHEMA = "fsm50.start_residual_macro_fsm.v1"
GATE_E_IDENTITY_SCHEMA = "fsm50.gate_e_zero_residual_identity.v1"
GATE_E_TASK_INPUTS_SCHEMA = "fsm50.residual_macro_task_inputs.v1"
GATE_E_WORKER_RESULT_SCHEMA = "fsm50.worker_residual_macro_fsm_result.v1"
GATE_E_COMMAND_EVIDENCE_SCHEMA = "fsm50.zero_residual_command_evidence.v1"
EXECUTION_MODE = "gate_e_r0_zero_residual"
OPERATION = "residual_macro_fsm"
POLICY_KIND = "ZERO"
NO_ACTIVE_PROFILE_STRATEGY = "NO_ACTIVE_PROFILE"
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
ENVIRONMENT_LOCK_PATH = MODULE_ROOT / "reports" / "environment_lock_50mm.json"
RESIDUAL_CORE_PATH = MODULE_ROOT / "fsm50_direct_command_residual.py"
SIM_WORKER_PROCESS_PATH = PROJECT_ROOT / "sim_worker_process.py"
GATE_E_TASK_INPUTS_NAME = "gate_e_residual_macro_task_inputs.json"
GATE_E_WORKER_RESULT_NAME = "worker_residual_macro_fsm_result.json"
EXPECTED_PROFILE_ID = "fsm50-gate-c-successful-recording-profiles-v1"
EXPECTED_GRAPH_ID = "fsm50-recording-derived-macro-v1"
EXPECTED_GRAPH_SHA256 = (
    "ffa5acfbf64b65c22eee54709a2afae5a56fa0b9345d8db84eb86acac58447c5"
)
EXPECTED_PROFILE_LIBRARY_SHA256 = (
    "762665911785eb755b7000fbb2cb450a52f5579f3945d8f214129f41eaa50066"
)
EXPECTED_ENVELOPE_CANONICAL_SHA256 = (
    "fa5002690737d94fab7304f40044293da46de34cd197af19f3f1140047ae7fbe"
)
EXPECTED_ENVELOPE_CONFIG_FILE_SHA256 = (
    "ba1f3f417eede01beca226b402227c216e7211cba0bc6c89f7ebb69626ed3037"
)
EXPECTED_FROZEN_CODE_SHA256 = MappingProxyType(
    {
        "worker_macro_fsm_session.py": (
            "d0e8f5392747365410645a5b3ce3f3f2b5740422418911cdfe3f86faebf0f45d"
        ),
        "fsm50_macro_controller.py": (
            "82439ba04ee1e3110190f96cffbe23b189d8bec7633f764039e54007c4c1d56d"
        ),
        "fsm50_macro_state_model.py": (
            "2cd9d4a67fa4b114bd1265e07f1471639d779e1c1fa1ad7120479236d5e579dc"
        ),
        "fsm50_motion_profiles.py": (
            "46dfbfcfb8b56e882ae750049f12c9c81489c6037bc831120756c9248a8f308f"
        ),
        "fsm50_direct_command_residual.py": (
            "169fde5d2ba656d0f1ea1aa3a504d2347dcc8b3151d9160fa9ec6bee7e8b54eb"
        ),
        "fsm50_residual_envelope.py": (
            "cf13ea71a1aa75d1770f2324faf87a67572f1630a7a4956bdf8641daa6bda2ae"
        ),
    }
)
EXPECTED_BUNDLE_SHA256 = MappingProxyType(
    {
        "v003_20260805_224517_157723_manual": (
            "5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78"
        ),
        "v008_20260806_211408_578700_manual": (
            "0a47cee05cbb41a3a09b23836386bddcf9ae1ab2aca5bbe451ee20a0cbaa8c41"
        ),
        "v009_20260806_215232_433234_manual": (
            "0e5686354eed7d89fb57d3aba722602eee7115b44749d531b6634dd0420ee919"
        ),
    }
)

CODE_BINDING_PATHS = MappingProxyType(
    {
        "worker_residual_macro_fsm_session.py": Path(__file__).resolve(),
        "worker_macro_fsm_session.py": MODULE_ROOT / "worker_macro_fsm_session.py",
        "fsm50_macro_controller.py": MODULE_ROOT / "fsm50_macro_controller.py",
        "fsm50_macro_state_model.py": MODULE_ROOT / "fsm50_macro_state_model.py",
        "fsm50_motion_profiles.py": MODULE_ROOT / "fsm50_motion_profiles.py",
        "fsm50_direct_command_residual.py": (
            MODULE_ROOT / "fsm50_direct_command_residual.py"
        ),
        "fsm50_residual_envelope.py": MODULE_ROOT / "fsm50_residual_envelope.py",
        "sim_worker_process.py": SIM_WORKER_PROCESS_PATH,
        "sim_process_client.py": PROJECT_ROOT / "sim_process_client.py",
        "sim_ipc_protocol.py": PROJECT_ROOT / "sim_ipc_protocol.py",
    }
)

_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "enabled",
        "execution_mode",
        "operation",
        "request_id",
        "request_identity_sha256",
        "base_request_path",
        "base_request_sha256",
        "base_request_id",
        "source_version",
        "profile_id",
        "graph_id",
        "graph_sha256",
        "profile_library_sha256",
        "bundle_sha256",
        "run_dir",
        "policy_kind",
        "policy_id",
        "policy_sha256",
        "residual_core_sha256",
        "envelope_config_path",
        "envelope_config_file_sha256",
        "envelope_canonical_sha256",
        "environment_lock_path",
        "environment_lock_sha256",
        "code_sha256",
    }
)

_START_KEYS = frozenset(
    {
        "schema_version",
        "type",
        "operation",
        "request_id",
        "request_identity_sha256",
        "worker_session_id",
        "source_version",
        "profile_id",
        "graph_id",
        "graph_sha256",
        "profile_library_sha256",
        "bundle_sha256",
        "policy_kind",
        "policy_sha256",
        "residual_core_sha256",
        "envelope_canonical_sha256",
        "enqueued_wall_time",
    }
)

_GATE_E_TASK_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "execution_mode",
        "operation",
        "gate_e_zero_residual",
        "base_task_inputs_path",
        "base_task_inputs_sha256",
        "base_task_inputs",
        "residual_command_evidence",
    }
)

_GATE_E_WORKER_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "execution_mode",
        "operation",
        "gate_e_zero_residual",
        "base_worker_result_path",
        "base_worker_result_sha256",
        "base_worker_result",
        "task_inputs_path",
        "task_inputs_sha256",
        "task_inputs",
        "residual_command_evidence",
        "macro_fsm_complete",
        "error",
    }
)


class ResidualWorkerRequestError(ValueError):
    """Raised when the independent Gate-E request is not exact and current."""


class BaseResidualAPIUnavailable(RuntimeError):
    """Raised until the base worker exposes both frozen residual hooks."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or value != value.strip() or value != value.lower():
        raise ResidualWorkerRequestError(f"{label} must be a lowercase SHA256")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ResidualWorkerRequestError(f"{label} must be a lowercase SHA256")
    return value


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if type(value) is not str or not value or value != value.strip():
        raise ResidualWorkerRequestError(f"{key} must be exact non-empty text")
    return value


def _strict_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ResidualWorkerRequestError(
            f"{label} contains non-finite JSON constant {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResidualWorkerRequestError(f"{label} duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except ResidualWorkerRequestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResidualWorkerRequestError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResidualWorkerRequestError(f"{label} must contain one JSON object")
    return value


def _strict_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualWorkerRequestError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = set(value)
    if observed != set(expected):
        raise ResidualWorkerRequestError(
            f"{label} keys mismatch: missing={sorted(set(expected) - observed)!r} "
            f"unexpected={sorted(observed - set(expected))!r}"
        )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidualWorkerRequestError(
            f"request is not strict canonical JSON data: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ResidualWorkerRequestError(
            f"cannot read {label}: {path}: {exc}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ResidualWorkerRequestError(
                f"{label} contains a blank row at line {index}"
            )
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ResidualWorkerRequestError(
                        f"{label} contains non-finite JSON constant {value}"
                    )
                ),
                object_pairs_hook=lambda pairs: _pairs_without_duplicates(
                    pairs, label=f"{label} line {index}"
                ),
            )
        except ResidualWorkerRequestError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResidualWorkerRequestError(
                f"{label} line {index} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ResidualWorkerRequestError(
                f"{label} line {index} must be an object"
            )
        rows.append(row)
    return rows


def _pairs_without_duplicates(
    pairs: list[tuple[str, Any]], *, label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResidualWorkerRequestError(f"{label} duplicate JSON key {key!r}")
        result[key] = value
    return result


def request_identity_sha256(value: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(value))
    body.pop("request_identity_sha256", None)
    return _canonical_json_sha256(body)


def current_code_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in CODE_BINDING_PATHS.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ResidualWorkerRequestError(f"Gate-E code binding is missing: {resolved}")
        result[name] = sha256_file(resolved)
    return result


def _validate_frozen_code_sha256(code_sha256: Mapping[str, str]) -> None:
    for name, expected in EXPECTED_FROZEN_CODE_SHA256.items():
        if code_sha256.get(name) != expected:
            raise ResidualWorkerRequestError(
                f"Gate-E frozen code SHA256 mismatch: {name}"
            )


def validate_current_environment_lock() -> dict[str, Any]:
    """Run the existing canonical lock content/hash validator.

    The expected lock digest is deliberately not stored in this source file:
    this file is itself part of the lock closure.  A Gate-E request binds the
    digest returned by this validation instead.
    """

    from .fsm50_macro_runner import _validate_canonical_environment_lock

    try:
        verification = dict(
            _validate_canonical_environment_lock(
                lock_path=ENVIRONMENT_LOCK_PATH,
            )
        )
    except Exception as exc:
        raise ResidualWorkerRequestError(
            f"canonical environment lock is not current: {exc}"
        ) from exc
    expected_keys = {
        "environment_lock_path",
        "environment_lock_sha256",
        "locked_source_file_count",
        "required_source_file_count",
        "source_closure_complete",
        "source_verification_sha256",
    }
    if set(verification) != expected_keys:
        raise ResidualWorkerRequestError(
            "canonical environment lock verification shape mismatch"
        )
    resolved = Path(
        _required_text(verification, "environment_lock_path")
    ).resolve()
    if resolved != ENVIRONMENT_LOCK_PATH.resolve():
        raise ResidualWorkerRequestError(
            "canonical environment lock validator returned a noncanonical path"
        )
    declared_sha = _validated_sha256(
        verification.get("environment_lock_sha256"),
        label="environment_lock_sha256",
    )
    if not resolved.is_file() or sha256_file(resolved) != declared_sha:
        raise ResidualWorkerRequestError(
            "canonical environment lock changed after content validation"
        )
    _validated_sha256(
        verification.get("source_verification_sha256"),
        label="source_verification_sha256",
    )
    locked = verification.get("locked_source_file_count")
    required = verification.get("required_source_file_count")
    if (
        type(locked) is not int
        or type(required) is not int
        or locked <= 0
        or locked != required
        or verification.get("source_closure_complete") is not True
    ):
        raise ResidualWorkerRequestError(
            "canonical environment lock source closure is not exact"
        )
    return verification


def _validate_environment_binding(
    declared_sha256: str,
    *,
    environment_validator: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    validator = environment_validator or validate_current_environment_lock
    try:
        verification = dict(validator())
    except ResidualWorkerRequestError:
        raise
    except Exception as exc:
        raise ResidualWorkerRequestError(
            f"canonical environment lock validation failed: {exc}"
        ) from exc
    current_sha = _validated_sha256(
        verification.get("environment_lock_sha256"),
        label="environment_lock_sha256",
    )
    if current_sha != declared_sha256:
        raise ResidualWorkerRequestError(
            "Gate-E environment lock SHA256 differs from current validated bytes"
        )
    if verification.get("source_closure_complete") is not True:
        raise ResidualWorkerRequestError(
            "Gate-E environment lock source closure is not current"
        )
    return verification


def base_residual_api_available() -> bool:
    parameters = inspect.signature(WorkerMacroFSMSession.__init__).parameters
    return {
        "residual_policy",
        "residual_contract_provider",
    }.issubset(parameters)


def validate_request_mode_exclusivity(
    *,
    residual_request_path: str | Path | None,
    task_request_path: str | Path | None,
    macro_request_path: str | Path | None,
) -> None:
    """Reject any Gate-E/base-task/base-macro dual admission."""

    residual = bool(str(residual_request_path or "").strip())
    task = bool(str(task_request_path or "").strip())
    macro = bool(str(macro_request_path or "").strip())
    if residual and (task or macro):
        conflicts = [
            name
            for name, active in (("task", task), ("macro", macro))
            if active
        ]
        raise ResidualWorkerRequestError(
            "Gate-E residual request is mutually exclusive with " + ", ".join(conflicts)
        )


@dataclass(frozen=True)
class WorkerResidualMacroFSMRequest:
    request_path: Path
    request_file_sha256: str
    request_id: str
    request_identity_sha256: str
    base_request_path: Path
    base_request_sha256: str
    base_request: WorkerMacroFSMRequest
    source_version: str
    profile_id: str
    graph_id: str
    graph_sha256: str
    profile_library_sha256: str
    bundle_sha256: str
    run_dir: Path
    policy_id: str
    policy_sha256: str
    residual_core_sha256: str
    envelope_config_path: Path
    envelope_config_file_sha256: str
    envelope_canonical_sha256: str
    environment_lock_path: Path
    environment_lock_sha256: str
    environment_lock_verification: Mapping[str, Any]
    code_sha256: Mapping[str, str]
    envelope: ResidualEnvelope

    @property
    def policy_kind(self) -> str:
        return POLICY_KIND

    def gate_e_identity(self, *, payload_role: str) -> dict[str, Any]:
        if payload_role not in {
            "status",
            "start_ack",
            "terminal",
            "worker_result",
            "task_inputs",
            "shutdown_ack",
            "close_requested",
            "close_returned",
        }:
            raise ResidualWorkerRequestError(
                "unsupported Gate-E identity payload role"
            )
        return {
            "schema_version": GATE_E_IDENTITY_SCHEMA,
            "payload_role": payload_role,
            "execution_mode": EXECUTION_MODE,
            "operation": OPERATION,
            "request_path": str(self.request_path),
            "request_file_sha256": self.request_file_sha256,
            "request_id": self.request_id,
            "request_identity_sha256": self.request_identity_sha256,
            "base_request_id": self.base_request.request_id,
            "base_request_sha256": self.base_request_sha256,
            "source_version": self.source_version,
            "profile_id": self.profile_id,
            "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "profile_library_sha256": self.profile_library_sha256,
            "bundle_sha256": self.bundle_sha256,
            "policy_kind": POLICY_KIND,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "residual_core_sha256": self.residual_core_sha256,
            "envelope_config_path": str(self.envelope_config_path),
            "envelope_config_file_sha256": self.envelope_config_file_sha256,
            "envelope_canonical_sha256": self.envelope_canonical_sha256,
            "environment_lock_path": str(self.environment_lock_path),
            "environment_lock_sha256": self.environment_lock_sha256,
            "environment_lock_verification": dict(
                self.environment_lock_verification
            ),
            "code_sha256": dict(self.code_sha256),
            "runtime_policy_authority": "EXACT_ZERO_ONLY",
            "ppo_training_performed": False,
            "checkpoint_loaded": False,
        }

    def validate_environment_lock_current(
        self,
        *,
        environment_validator: Callable[[], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return _validate_environment_binding(
            self.environment_lock_sha256,
            environment_validator=environment_validator,
        )

    def preflight_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA,
            "enabled": True,
            "execution_mode": EXECUTION_MODE,
            "operation": OPERATION,
            "request_id": self.request_id,
            "request_identity_sha256": self.request_identity_sha256,
            "source_version": self.source_version,
            "profile_id": self.profile_id,
            "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "profile_library_sha256": self.profile_library_sha256,
            "bundle_sha256": self.bundle_sha256,
            "policy_kind": POLICY_KIND,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "residual_core_sha256": self.residual_core_sha256,
            "envelope_canonical_sha256": self.envelope_canonical_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "base_residual_api_available": base_residual_api_available(),
            "preflight_ok": True,
            "ppo_training_claimed": False,
        }


def build_worker_residual_macro_fsm_request(
    *,
    base_request_path: str | Path,
    request_id: str,
    environment_validator: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build, but do not write, one exact ZERO-only request mapping."""

    if type(request_id) is not str or not request_id or request_id != request_id.strip():
        raise ResidualWorkerRequestError("request_id must be exact non-empty text")
    base_path = Path(base_request_path).resolve()
    base_request = load_worker_macro_fsm_request(base_path)
    if base_request is None:
        raise ResidualWorkerRequestError("base macro request path is required")
    if base_request.source_version not in ALLOWED_SOURCE_VERSIONS:
        raise ResidualWorkerRequestError(
            "Gate-E source must be one of reviewed v003/v008/v009"
        )
    expected_bundle = EXPECTED_BUNDLE_SHA256[base_request.source_version]
    if (
        base_request.profile_id != EXPECTED_PROFILE_ID
        or base_request.graph_id != EXPECTED_GRAPH_ID
        or base_request.graph_sha256 != EXPECTED_GRAPH_SHA256
        or base_request.profile_library_sha256 != EXPECTED_PROFILE_LIBRARY_SHA256
        or base_request.bundle_sha256 != expected_bundle
    ):
        raise ResidualWorkerRequestError("base macro request identity is not Gate-E canonical")

    envelope = load_residual_envelope(verify_evidence=True)
    if envelope.canonical_sha256 != EXPECTED_ENVELOPE_CANONICAL_SHA256:
        raise ResidualWorkerRequestError("reviewed residual envelope identity mismatch")
    config_file_sha = sha256_file(DEFAULT_CONFIG_PATH)
    if config_file_sha != EXPECTED_ENVELOPE_CONFIG_FILE_SHA256:
        raise ResidualWorkerRequestError(
            "reviewed residual envelope config file SHA256 mismatch"
        )
    validator = environment_validator or validate_current_environment_lock
    try:
        environment_verification = dict(validator())
    except ResidualWorkerRequestError:
        raise
    except Exception as exc:
        raise ResidualWorkerRequestError(
            f"canonical environment lock validation failed: {exc}"
        ) from exc
    environment_sha = _validated_sha256(
        environment_verification.get("environment_lock_sha256"),
        label="environment_lock_sha256",
    )
    _validate_environment_binding(
        environment_sha,
        environment_validator=lambda: environment_verification,
    )

    code_sha256 = current_code_sha256()
    _validate_frozen_code_sha256(code_sha256)

    policy = ZeroResidualPolicy()
    payload: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": EXECUTION_MODE,
        "operation": OPERATION,
        "request_id": request_id,
        "request_identity_sha256": "0" * 64,
        "base_request_path": str(base_path),
        "base_request_sha256": sha256_file(base_path),
        "base_request_id": base_request.request_id,
        "source_version": base_request.source_version,
        "profile_id": base_request.profile_id,
        "graph_id": base_request.graph_id,
        "graph_sha256": base_request.graph_sha256,
        "profile_library_sha256": base_request.profile_library_sha256,
        "bundle_sha256": base_request.bundle_sha256,
        "run_dir": str(base_request.run_dir),
        "policy_kind": POLICY_KIND,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_sha256,
        "residual_core_sha256": sha256_file(RESIDUAL_CORE_PATH),
        "envelope_config_path": str(DEFAULT_CONFIG_PATH.resolve()),
        "envelope_config_file_sha256": config_file_sha,
        "envelope_canonical_sha256": envelope.canonical_sha256,
        "environment_lock_path": str(ENVIRONMENT_LOCK_PATH.resolve()),
        "environment_lock_sha256": environment_sha,
        "code_sha256": code_sha256,
    }
    payload["request_identity_sha256"] = request_identity_sha256(payload)
    return payload


def load_worker_residual_macro_fsm_request(
    request_path: str | Path | None,
    *,
    environment_validator: Callable[[], Mapping[str, Any]] | None = None,
) -> WorkerResidualMacroFSMRequest | None:
    text = str(request_path or "").strip()
    if not text:
        return None
    path = Path(text).resolve()
    raw = _strict_object(path, label="Gate-E residual request")
    _exact_keys(raw, _REQUEST_KEYS, label="Gate-E residual request")
    if raw.get("schema_version") != REQUEST_SCHEMA:
        raise ResidualWorkerRequestError("unsupported Gate-E residual request schema")
    if raw.get("enabled") is not True:
        raise ResidualWorkerRequestError("Gate-E residual request requires enabled=true")
    if raw.get("execution_mode") != EXECUTION_MODE:
        raise ResidualWorkerRequestError("Gate-E residual execution_mode mismatch")
    if raw.get("operation") != OPERATION:
        raise ResidualWorkerRequestError("Gate-E residual operation mismatch")

    request_id = _required_text(raw, "request_id")
    declared_identity = _validated_sha256(
        raw.get("request_identity_sha256"), label="request_identity_sha256"
    )
    actual_identity = request_identity_sha256(raw)
    if declared_identity != actual_identity:
        raise ResidualWorkerRequestError(
            "Gate-E residual request canonical identity SHA256 mismatch"
        )

    source_version = _required_text(raw, "source_version")
    if source_version not in ALLOWED_SOURCE_VERSIONS:
        raise ResidualWorkerRequestError(
            "Gate-E source must be one of reviewed v003/v008/v009"
        )
    if raw.get("profile_id") != EXPECTED_PROFILE_ID:
        raise ResidualWorkerRequestError("Gate-E profile_id mismatch")
    if raw.get("graph_id") != EXPECTED_GRAPH_ID:
        raise ResidualWorkerRequestError("Gate-E graph_id mismatch")
    graph_sha = _validated_sha256(raw.get("graph_sha256"), label="graph_sha256")
    library_sha = _validated_sha256(
        raw.get("profile_library_sha256"), label="profile_library_sha256"
    )
    bundle_sha = _validated_sha256(raw.get("bundle_sha256"), label="bundle_sha256")
    if graph_sha != EXPECTED_GRAPH_SHA256:
        raise ResidualWorkerRequestError("Gate-E graph SHA256 mismatch")
    if library_sha != EXPECTED_PROFILE_LIBRARY_SHA256:
        raise ResidualWorkerRequestError("Gate-E profile library SHA256 mismatch")
    if bundle_sha != EXPECTED_BUNDLE_SHA256[source_version]:
        raise ResidualWorkerRequestError("Gate-E source bundle SHA256 mismatch")

    if raw.get("policy_kind") != POLICY_KIND:
        raise ResidualWorkerRequestError("Gate-E R0 only permits policy_kind=ZERO")
    policy = ZeroResidualPolicy()
    if raw.get("policy_id") != policy.policy_id:
        raise ResidualWorkerRequestError("Gate-E ZERO policy_id mismatch")
    policy_sha = _validated_sha256(raw.get("policy_sha256"), label="policy_sha256")
    if policy_sha != policy.policy_sha256:
        raise ResidualWorkerRequestError("Gate-E ZERO policy SHA256 mismatch")
    residual_core_sha = _validated_sha256(
        raw.get("residual_core_sha256"), label="residual_core_sha256"
    )
    if (
        residual_core_sha != EXPECTED_FROZEN_CODE_SHA256[RESIDUAL_CORE_PATH.name]
        or residual_core_sha != sha256_file(RESIDUAL_CORE_PATH)
    ):
        raise ResidualWorkerRequestError("Gate-E residual core SHA256 mismatch")

    base_path = Path(_required_text(raw, "base_request_path")).resolve()
    base_sha = _validated_sha256(
        raw.get("base_request_sha256"), label="base_request_sha256"
    )
    if not base_path.is_file() or sha256_file(base_path) != base_sha:
        raise ResidualWorkerRequestError("Gate-E base request SHA256 mismatch")
    base_request = load_worker_macro_fsm_request(base_path)
    if base_request is None:
        raise ResidualWorkerRequestError("Gate-E base request is missing")
    expected_base = {
        "base_request_id": base_request.request_id,
        "source_version": base_request.source_version,
        "profile_id": base_request.profile_id,
        "graph_id": base_request.graph_id,
        "graph_sha256": base_request.graph_sha256,
        "profile_library_sha256": base_request.profile_library_sha256,
        "bundle_sha256": base_request.bundle_sha256,
        "run_dir": str(base_request.run_dir),
    }
    for key, expected in expected_base.items():
        if raw.get(key) != expected:
            raise ResidualWorkerRequestError(
                f"Gate-E {key} does not match the exact base macro request"
            )

    envelope_path = Path(_required_text(raw, "envelope_config_path")).resolve()
    if envelope_path != DEFAULT_CONFIG_PATH.resolve():
        raise ResidualWorkerRequestError("Gate-E envelope config path is not canonical")
    envelope_file_sha = _validated_sha256(
        raw.get("envelope_config_file_sha256"),
        label="envelope_config_file_sha256",
    )
    if (
        envelope_file_sha != EXPECTED_ENVELOPE_CONFIG_FILE_SHA256
        or not envelope_path.is_file()
        or sha256_file(envelope_path) != envelope_file_sha
    ):
        raise ResidualWorkerRequestError("Gate-E envelope config file SHA256 mismatch")
    envelope = load_residual_envelope(envelope_path, verify_evidence=True)
    envelope_canonical_sha = _validated_sha256(
        raw.get("envelope_canonical_sha256"), label="envelope_canonical_sha256"
    )
    if (
        envelope_canonical_sha != EXPECTED_ENVELOPE_CANONICAL_SHA256
        or envelope.canonical_sha256 != envelope_canonical_sha
        or not envelope.evidence_verified
        or not envelope.is_source_allowed(source_version)
    ):
        raise ResidualWorkerRequestError("Gate-E residual envelope is not reviewed/current")

    environment_path = Path(_required_text(raw, "environment_lock_path")).resolve()
    if environment_path != ENVIRONMENT_LOCK_PATH.resolve():
        raise ResidualWorkerRequestError("Gate-E environment lock path is not canonical")
    environment_sha = _validated_sha256(
        raw.get("environment_lock_sha256"), label="environment_lock_sha256"
    )
    environment_verification = _validate_environment_binding(
        environment_sha,
        environment_validator=environment_validator,
    )

    declared_code = _strict_mapping(raw.get("code_sha256"), label="code_sha256")
    expected_code = current_code_sha256()
    _validate_frozen_code_sha256(expected_code)
    if set(declared_code) != set(expected_code):
        raise ResidualWorkerRequestError("Gate-E code SHA256 keys mismatch")
    for name, actual_sha in expected_code.items():
        if _validated_sha256(declared_code.get(name), label=f"code_sha256.{name}") != actual_sha:
            raise ResidualWorkerRequestError(f"Gate-E code SHA256 mismatch: {name}")
    if declared_code[RESIDUAL_CORE_PATH.name] != residual_core_sha:
        raise ResidualWorkerRequestError(
            "Gate-E residual core SHA256 differs from its code binding"
        )

    return WorkerResidualMacroFSMRequest(
        request_path=path,
        request_file_sha256=sha256_file(path),
        request_id=request_id,
        request_identity_sha256=declared_identity,
        base_request_path=base_path,
        base_request_sha256=base_sha,
        base_request=base_request,
        source_version=source_version,
        profile_id=EXPECTED_PROFILE_ID,
        graph_id=EXPECTED_GRAPH_ID,
        graph_sha256=graph_sha,
        profile_library_sha256=library_sha,
        bundle_sha256=bundle_sha,
        run_dir=base_request.run_dir,
        policy_id=policy.policy_id,
        policy_sha256=policy_sha,
        residual_core_sha256=residual_core_sha,
        envelope_config_path=envelope_path,
        envelope_config_file_sha256=envelope_file_sha,
        envelope_canonical_sha256=envelope_canonical_sha,
        environment_lock_path=environment_path,
        environment_lock_sha256=environment_sha,
        environment_lock_verification=MappingProxyType(
            copy.deepcopy(environment_verification)
        ),
        code_sha256=MappingProxyType(dict(declared_code)),
        envelope=envelope,
    )


@dataclass(frozen=True)
class VerifiedPhaseContractProvider:
    """Source-bound adapter from nominal targets to reviewed phase contracts."""

    request: WorkerResidualMacroFSMRequest

    def __call__(self, context: Mapping[str, Any]) -> Any:
        """Adapt the base worker's frozen context mapping, without composing."""

        if not isinstance(context, Mapping):
            raise ResidualWorkerRequestError(
                "residual contract context must be a mapping"
            )
        if context.get("schema_version") != "fsm50.worker_residual_observation.v1":
            raise ResidualWorkerRequestError(
                "residual contract context schema mismatch"
            )
        source_version = str(context.get("source_version", "") or "")
        if source_version != self.request.source_version:
            raise ResidualWorkerRequestError(
                "residual contract source differs from the Gate-E request"
            )
        if context.get("outer_render_boundary_permit") is not True:
            raise ResidualWorkerRequestError(
                "residual contract requires an outer render boundary permit"
            )
        provenance = _strict_mapping(
            context.get("command_provenance"), label="command_provenance"
        )
        provenance_kind = provenance.get("kind", "")
        if type(provenance_kind) is not str:
            raise ResidualWorkerRequestError(
                "command_provenance.kind must be text"
            )
        profile_id = context.get("profile_id")
        profile_source_version = context.get("profile_source_version")
        profile_strategy = context.get("profile_strategy")
        for label, value in (
            ("profile_id", profile_id),
            ("profile_source_version", profile_source_version),
            ("profile_strategy", profile_strategy),
        ):
            if type(value) is not str:
                raise ResidualWorkerRequestError(f"{label} must be text")
        if profile_source_version not in {"", self.request.source_version}:
            raise ResidualWorkerRequestError(
                "residual contract profile source differs from the Gate-E request"
            )
        if provenance_kind == "SOURCE_ACTION" and (
            not profile_id
            or profile_source_version != self.request.source_version
            or not profile_strategy
        ):
            raise ResidualWorkerRequestError(
                "SOURCE_ACTION residual contract lacks its bound source profile"
            )
        contract_profile_strategy = (
            profile_strategy or NO_ACTIVE_PROFILE_STRATEGY
        )
        if not self.request.envelope.evidence_verified:
            raise ResidualWorkerRequestError("residual envelope evidence is unverified")
        try:
            return self.request.envelope.phase_contract(
                source_version=source_version,
                profile_strategy=contract_profile_strategy,
                macro_state=_required_text(context, "macro_state"),
                subphase=_required_text(context, "subphase"),
                nominal_servo_targets_deg=_strict_mapping(
                    context.get("nominal_servo_targets_deg"),
                    label="nominal_servo_targets_deg",
                ),
                nominal_wheel_targets_rad_s=_strict_mapping(
                    context.get("nominal_wheel_targets_rad_s"),
                    label="nominal_wheel_targets_rad_s",
                ),
                decision_provenance=provenance_kind,
            )
        except (ResidualEnvelopeError, ResidualContractError) as exc:
            raise ResidualWorkerRequestError(
                f"cannot build verified residual phase contract: {exc}"
            ) from exc


def _is_exact_zero_action(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == 12
        and all(
            type(item) in (int, float) and float(item) == 0.0
            for item in value
        )
    )


def _build_residual_command_evidence(
    request: WorkerResidualMacroFSMRequest,
    base_result: Mapping[str, Any],
) -> dict[str, Any]:
    direct = _strict_mapping(
        base_result.get("direct_command_residual"),
        label="base worker direct_command_residual",
    )
    if direct.get("enabled") is not True:
        raise ResidualWorkerRequestError(
            "base worker result did not enable direct-command residual composition"
        )
    if (
        direct.get("policy_id") != request.policy_id
        or direct.get("policy_sha256") != request.policy_sha256
    ):
        raise ResidualWorkerRequestError(
            "base worker residual policy identity differs from Gate-E ZERO"
        )
    transform_count = direct.get("transform_count")
    if type(transform_count) is not int or transform_count < 0:
        raise ResidualWorkerRequestError(
            "base worker residual transform_count is invalid"
        )
    last_transform = _strict_mapping(
        direct.get("last_transform"), label="base worker last residual transform"
    )
    last_available = bool(last_transform)
    last_zero = None
    if last_available:
        last_zero = bool(
            _is_exact_zero_action(last_transform.get("raw_normalized_action"))
            and _is_exact_zero_action(last_transform.get("clipped_normalized_action"))
            and _is_exact_zero_action(last_transform.get("applied_residual"))
            and last_transform.get("zero_identity") is True
        )
        if last_zero is not True:
            raise ResidualWorkerRequestError(
                "base worker last residual transform is not exact ZERO identity"
            )
    elif transform_count != 0:
        raise ResidualWorkerRequestError(
            "base worker residual transforms lack last-transform evidence"
        )

    dispatch_path = Path(
        _required_text(base_result, "dispatch_ledger_path")
    ).resolve()
    expected_dispatch_path = request.run_dir / "macro_dispatch_ledger.jsonl"
    if dispatch_path != expected_dispatch_path.resolve() or not dispatch_path.is_file():
        raise ResidualWorkerRequestError(
            "base worker residual dispatch ledger path is not canonical"
        )
    dispatch_rows = _strict_jsonl(
        dispatch_path, label="base worker residual dispatch ledger"
    )
    residual_dispatch_rows = []
    for index, row in enumerate(dispatch_rows):
        if "residual_transform" not in row:
            raise ResidualWorkerRequestError(
                f"Gate-E dispatch row {index} lacks residual transform evidence"
            )
        transform = _strict_mapping(
            row.get("residual_transform"),
            label=f"Gate-E dispatch row {index} residual_transform",
        )
        if (
            row.get("residual_policy_id") != request.policy_id
            or row.get("residual_policy_sha256") != request.policy_sha256
            or not _is_exact_zero_action(transform.get("raw_normalized_action"))
            or not _is_exact_zero_action(transform.get("clipped_normalized_action"))
            or not _is_exact_zero_action(transform.get("applied_residual"))
            or transform.get("zero_identity") is not True
        ):
            raise ResidualWorkerRequestError(
                f"Gate-E dispatch row {index} is not exact ZERO identity"
            )
        residual_dispatch_rows.append(row)

    return {
        "schema_version": GATE_E_COMMAND_EVIDENCE_SCHEMA,
        "policy_kind": POLICY_KIND,
        "policy_id": request.policy_id,
        "policy_sha256": request.policy_sha256,
        "residual_core_sha256": request.residual_core_sha256,
        "envelope_canonical_sha256": request.envelope_canonical_sha256,
        "transform_count": transform_count,
        "physical_command_epoch": direct.get("physical_command_epoch"),
        "last_verified_physical_command_epoch": direct.get(
            "last_verified_physical_command_epoch"
        ),
        "last_transform_available": last_available,
        "last_transform_zero_identity_verified": last_zero,
        "last_transform_sha256": str(
            direct.get("last_transform_sha256", "") or ""
        ),
        "last_transform": copy.deepcopy(dict(last_transform)),
        "dispatch_ledger_path": str(dispatch_path),
        "dispatch_ledger_sha256": sha256_file(dispatch_path),
        "dispatch_row_count": len(dispatch_rows),
        "residual_dispatch_zero_identity_count": len(residual_dispatch_rows),
        "all_durable_residual_dispatches_zero_identity": True,
        "checkpoint_loaded": False,
        "ppo_training_performed": False,
    }


class GateEResidualMacroFSMSession:
    """Thin Gate-E lifecycle proxy around the frozen nominal base session."""

    def __init__(
        self,
        request: WorkerResidualMacroFSMRequest,
        base_session: WorkerMacroFSMSession,
        *,
        environment_validator: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.residual_request = request
        self.base_session = base_session
        self.environment_validator = environment_validator
        self.terminal_payload: dict[str, Any] | None = None

    @property
    def request(self) -> WorkerMacroFSMRequest:
        """Retain the frozen base-session request API for nominal ownership."""

        return self.base_session.request

    @property
    def outer_request_id(self) -> str:
        return self.residual_request.request_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_session, name)

    def status_dict(self) -> dict[str, Any]:
        status = copy.deepcopy(dict(self.base_session.status_dict()))
        status["execution_mode"] = EXECUTION_MODE
        status["operation"] = OPERATION
        status["residual_macro_fsm_request_id"] = self.residual_request.request_id
        status["base_macro_fsm_request_id"] = self.request.request_id
        status["gate_e_zero_residual"] = self.residual_request.gate_e_identity(
            payload_role="status"
        )
        return status

    def start(self) -> dict[str, Any]:
        current_environment = self.residual_request.validate_environment_lock_current(
            environment_validator=self.environment_validator
        )
        if current_environment != dict(
            self.residual_request.environment_lock_verification
        ):
            raise ResidualWorkerRequestError(
                "canonical environment lock/source verification changed before session start"
            )
        base_start = dict(self.base_session.start())
        if base_start.get("request_id") != self.request.request_id:
            raise ResidualWorkerRequestError(
                "base Macro start payload lost its exact request binding"
            )
        payload = copy.deepcopy(base_start)
        payload.update(
            {
                "execution_mode": EXECUTION_MODE,
                "operation": OPERATION,
                "request_id": self.residual_request.request_id,
                "request_identity_sha256": (
                    self.residual_request.request_identity_sha256
                ),
                "residual_macro_fsm_request_id": (
                    self.residual_request.request_id
                ),
                "base_macro_fsm_request_id": self.request.request_id,
                "base_macro_fsm_request_sha256": (
                    self.residual_request.base_request_sha256
                ),
                "base_start_payload_sha256": _canonical_json_sha256(base_start),
                "gate_e_zero_residual": self.residual_request.gate_e_identity(
                    payload_role="start_ack"
                ),
            }
        )
        return payload

    def before_adapter_step(self) -> None:
        self.base_session.before_adapter_step()

    def after_adapter_step(self) -> dict[str, Any] | None:
        if self.terminal_payload is not None:
            return None
        terminal = self.base_session.after_adapter_step()
        if terminal is None:
            return None
        return self._wrap_terminal(terminal)

    def fail(
        self,
        error: str,
        *,
        infrastructure_failure: bool = False,
        simulation_app_stopped: bool = False,
    ) -> dict[str, Any] | None:
        if self.terminal_payload is not None:
            return copy.deepcopy(self.terminal_payload)
        terminal = self.base_session.fail(
            error,
            infrastructure_failure=infrastructure_failure,
            simulation_app_stopped=simulation_app_stopped,
        )
        if terminal is None:
            return None
        return self._wrap_terminal(terminal)

    def _wrap_terminal(self, base_terminal: Mapping[str, Any]) -> dict[str, Any]:
        if self.terminal_payload is not None:
            return copy.deepcopy(self.terminal_payload)
        if not isinstance(base_terminal, Mapping):
            raise ResidualWorkerRequestError(
                "base Macro terminal must be a mapping"
            )
        base_terminal_copy = copy.deepcopy(dict(base_terminal))
        if (
            base_terminal_copy.get("type")
            not in {"macro_fsm_complete", "macro_fsm_failed"}
            or base_terminal_copy.get("request_id") != self.request.request_id
            or base_terminal_copy.get("source_version")
            != self.residual_request.source_version
            or base_terminal_copy.get("profile_id")
            != self.residual_request.profile_id
            or base_terminal_copy.get("graph_sha256")
            != self.residual_request.graph_sha256
            or base_terminal_copy.get("profile_library_sha256")
            != self.residual_request.profile_library_sha256
            or base_terminal_copy.get("bundle_sha256")
            != self.residual_request.bundle_sha256
        ):
            raise ResidualWorkerRequestError(
                "base Macro terminal identity differs from Gate-E nominal binding"
            )

        run_dir = self.residual_request.run_dir.resolve()
        base_inputs_path = Path(
            _required_text(base_terminal_copy, "task_inputs_path")
        ).resolve()
        base_result_path = Path(
            _required_text(base_terminal_copy, "worker_result_path")
        ).resolve()
        if (
            base_inputs_path != (run_dir / "macro_task_inputs.json").resolve()
            or base_result_path
            != (run_dir / "worker_macro_fsm_result.json").resolve()
        ):
            raise ResidualWorkerRequestError(
                "base Macro terminal artifact paths are not canonical"
            )
        base_inputs = _strict_object(
            base_inputs_path, label="base Macro task inputs"
        )
        base_result = _strict_object(
            base_result_path, label="base Macro worker result"
        )
        if base_inputs.get("schema_version") != "fsm50.macro_task_inputs.v1":
            raise ResidualWorkerRequestError(
                "base Macro task inputs schema mismatch"
            )
        if (
            base_result.get("schema_version")
            != "fsm50.worker_macro_fsm_session.v1"
            or base_result.get("request_id") != self.request.request_id
            or base_result.get("source_version")
            != self.residual_request.source_version
            or base_result.get("profile_id") != self.residual_request.profile_id
            or base_result.get("graph_sha256")
            != self.residual_request.graph_sha256
            or base_result.get("profile_library_sha256")
            != self.residual_request.profile_library_sha256
            or base_result.get("bundle_sha256")
            != self.residual_request.bundle_sha256
        ):
            raise ResidualWorkerRequestError(
                "base Macro worker result identity differs from Gate-E request"
            )
        if _canonical_json_sha256(
            _strict_mapping(
                base_terminal_copy.get("task_inputs"),
                label="base terminal task_inputs",
            )
        ) != _canonical_json_sha256(base_inputs):
            raise ResidualWorkerRequestError(
                "base terminal task_inputs differ from the durable artifact"
            )

        command_evidence = _build_residual_command_evidence(
            self.residual_request, base_result
        )
        base_inputs_sha = sha256_file(base_inputs_path)
        base_result_sha = sha256_file(base_result_path)
        outer_inputs = {
            "schema_version": GATE_E_TASK_INPUTS_SCHEMA,
            "execution_mode": EXECUTION_MODE,
            "operation": OPERATION,
            "gate_e_zero_residual": self.residual_request.gate_e_identity(
                payload_role="task_inputs"
            ),
            "base_task_inputs_path": str(base_inputs_path),
            "base_task_inputs_sha256": base_inputs_sha,
            "base_task_inputs": base_inputs,
            "residual_command_evidence": command_evidence,
        }
        _exact_keys(
            outer_inputs,
            _GATE_E_TASK_INPUT_KEYS,
            label="Gate-E task inputs",
        )
        outer_inputs_path = run_dir / GATE_E_TASK_INPUTS_NAME
        _atomic_write_json(outer_inputs_path, outer_inputs)
        outer_inputs_sha = sha256_file(outer_inputs_path)

        complete = base_terminal_copy.get("type") == "macro_fsm_complete"
        outer_result = {
            "schema_version": GATE_E_WORKER_RESULT_SCHEMA,
            "execution_mode": EXECUTION_MODE,
            "operation": OPERATION,
            "gate_e_zero_residual": self.residual_request.gate_e_identity(
                payload_role="worker_result"
            ),
            "base_worker_result_path": str(base_result_path),
            "base_worker_result_sha256": base_result_sha,
            "base_worker_result": base_result,
            "task_inputs_path": str(outer_inputs_path),
            "task_inputs_sha256": outer_inputs_sha,
            "task_inputs": outer_inputs,
            "residual_command_evidence": command_evidence,
            "macro_fsm_complete": complete,
            "error": str(base_terminal_copy.get("error", "") or ""),
        }
        _exact_keys(
            outer_result,
            _GATE_E_WORKER_RESULT_KEYS,
            label="Gate-E worker result",
        )
        outer_result_path = run_dir / GATE_E_WORKER_RESULT_NAME
        _atomic_write_json(outer_result_path, outer_result)
        outer_result_sha = sha256_file(outer_result_path)

        terminal = copy.deepcopy(base_terminal_copy)
        terminal.update(
            {
                "operation": OPERATION,
                "request_id": self.residual_request.request_id,
                "request_identity_sha256": (
                    self.residual_request.request_identity_sha256
                ),
                "execution_mode": EXECUTION_MODE,
                "residual_macro_fsm_request_id": (
                    self.residual_request.request_id
                ),
                "base_macro_fsm_request_id": self.request.request_id,
                "base_macro_fsm_request_path": str(
                    self.residual_request.base_request_path
                ),
                "base_macro_fsm_request_sha256": (
                    self.residual_request.base_request_sha256
                ),
                "base_macro_fsm_terminal": base_terminal_copy,
                "base_macro_fsm_terminal_sha256": _canonical_json_sha256(
                    base_terminal_copy
                ),
                "base_task_inputs_path": str(base_inputs_path),
                "base_task_inputs_sha256": base_inputs_sha,
                "base_worker_result_path": str(base_result_path),
                "base_worker_result_sha256": base_result_sha,
                "task_inputs_path": str(outer_inputs_path),
                "task_inputs_sha256": outer_inputs_sha,
                "worker_result_path": str(outer_result_path),
                "worker_result_sha256": outer_result_sha,
                "task_inputs": outer_inputs,
                "residual_command_evidence": command_evidence,
                "gate_e_zero_residual": (
                    self.residual_request.gate_e_identity(
                        payload_role="terminal"
                    )
                ),
            }
        )
        self.terminal_payload = terminal
        return copy.deepcopy(terminal)


@dataclass(frozen=True)
class GateEResidualMacroFSMSessionFactory:
    """Construct the base session only after its residual API is frozen."""

    request: WorkerResidualMacroFSMRequest

    @property
    def policy(self) -> ZeroResidualPolicy:
        return ZeroResidualPolicy()

    @property
    def contract_provider(self) -> VerifiedPhaseContractProvider:
        return VerifiedPhaseContractProvider(self.request)

    def build_base_session(
        self,
        *,
        worker_session_id: str,
        **base_session_kwargs: Any,
    ) -> WorkerMacroFSMSession:
        reserved = {"residual_policy", "residual_contract_provider"}
        if reserved.intersection(base_session_kwargs):
            raise ResidualWorkerRequestError(
                "caller cannot override Gate-E residual policy or contract provider"
            )
        if not base_residual_api_available():
            raise BaseResidualAPIUnavailable(
                "WorkerMacroFSMSession residual_policy/residual_contract_provider "
                "API is not frozen"
            )
        return WorkerMacroFSMSession(
            self.request.base_request,
            worker_session_id=str(worker_session_id),
            residual_policy=self.policy,
            residual_contract_provider=self.contract_provider,
            **base_session_kwargs,
        )

    def build_session(
        self,
        *,
        worker_session_id: str,
        environment_validator: Callable[[], Mapping[str, Any]] | None = None,
        **base_session_kwargs: Any,
    ) -> GateEResidualMacroFSMSession:
        return GateEResidualMacroFSMSession(
            self.request,
            self.build_base_session(
                worker_session_id=worker_session_id,
                **base_session_kwargs,
            ),
            environment_validator=environment_validator,
        )


def gate_e_identity_for_payload(
    payload: Mapping[str, Any],
    request: WorkerResidualMacroFSMRequest,
    *,
    payload_role: str,
) -> dict[str, Any]:
    """Return an identity-bearing copy without mutating nominal payload data."""

    if not isinstance(payload, Mapping):
        raise ResidualWorkerRequestError("Gate-E identity payload must be a mapping")
    if "gate_e_zero_residual" in payload:
        raise ResidualWorkerRequestError("Gate-E identity is already present")
    result = copy.deepcopy(dict(payload))
    result["gate_e_zero_residual"] = request.gate_e_identity(
        payload_role=payload_role
    )
    return result


def validate_worker_residual_start_binding(
    request: WorkerResidualMacroFSMRequest,
    message: Mapping[str, Any],
    *,
    expected_worker_session_id: str,
) -> list[str]:
    if not isinstance(message, Mapping):
        return ["Gate-E residual start message is not an object"]
    observed = set(message)
    if observed != set(_START_KEYS):
        return [
            "Gate-E residual start keys mismatch: "
            f"missing={sorted(set(_START_KEYS) - observed)!r} "
            f"unexpected={sorted(observed - set(_START_KEYS))!r}"
        ]
    expected = {
        "schema_version": START_SCHEMA,
        "type": "start_macro_fsm",
        "operation": OPERATION,
        "request_id": request.request_id,
        "request_identity_sha256": request.request_identity_sha256,
        "worker_session_id": str(expected_worker_session_id),
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "policy_kind": POLICY_KIND,
        "policy_sha256": request.policy_sha256,
        "residual_core_sha256": request.residual_core_sha256,
        "envelope_canonical_sha256": request.envelope_canonical_sha256,
    }
    errors = [
        f"Gate-E residual start {key} does not match the request"
        for key, value in expected.items()
        if message.get(key) != value
    ]
    enqueued = message.get("enqueued_wall_time")
    if type(enqueued) not in (int, float) or not math.isfinite(float(enqueued)):
        errors.append("Gate-E residual start enqueued_wall_time must be finite")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one strict Gate-E R0 ZERO residual worker request."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and print identity without constructing a base session.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    request = load_worker_residual_macro_fsm_request(args.request)
    if request is None:
        raise ResidualWorkerRequestError("Gate-E request path is required")
    preflight = request.preflight_payload()
    print(json.dumps(preflight, sort_keys=True, separators=(",", ":")))
    if args.preflight_only:
        return 0
    if not base_residual_api_available():
        raise BaseResidualAPIUnavailable(
            "live Gate-E route is blocked until the base residual API is frozen"
        )
    raise BaseResidualAPIUnavailable(
        "standalone validation never launches a worker; use sim_worker_process "
        "--fsm50-residual-macro-request-path"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BaseResidualAPIUnavailable",
    "ENVIRONMENT_LOCK_PATH",
    "EXECUTION_MODE",
    "GATE_E_COMMAND_EVIDENCE_SCHEMA",
    "GATE_E_IDENTITY_SCHEMA",
    "GATE_E_TASK_INPUTS_NAME",
    "GATE_E_TASK_INPUTS_SCHEMA",
    "GATE_E_WORKER_RESULT_NAME",
    "GATE_E_WORKER_RESULT_SCHEMA",
    "GateEResidualMacroFSMSession",
    "GateEResidualMacroFSMSessionFactory",
    "OPERATION",
    "POLICY_KIND",
    "REQUEST_SCHEMA",
    "ResidualWorkerRequestError",
    "START_SCHEMA",
    "VerifiedPhaseContractProvider",
    "WorkerResidualMacroFSMRequest",
    "base_residual_api_available",
    "build_worker_residual_macro_fsm_request",
    "current_code_sha256",
    "gate_e_identity_for_payload",
    "load_worker_residual_macro_fsm_request",
    "main",
    "request_identity_sha256",
    "sha256_file",
    "validate_request_mode_exclusivity",
    "validate_current_environment_lock",
    "validate_worker_residual_start_binding",
]
