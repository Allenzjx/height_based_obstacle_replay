"""Immutable, current-only phase-entry snapshots for residual training.

The bank is deliberately an artifact reader, not a controller serializer.  It
derives physical reset state and a public source-action cursor from the sealed,
reviewed r4 Macro runs for v003, v008, and v009.  v010, legacy PPO outputs, and
arbitrary caller-selected run directories are outside this module's authority.

The canonical telemetry sample is the first 15 Hz row whose physics step is at
or after the 120 Hz state-entry transition.  Telemetry is sampled before the
controller decision in a physics callback, so a same-step row is the physical
observation that triggers entry and precedes any same-step source dispatch.  If
no same-step row exists, the first later sample is used.  In v003/v008, S7 is a
one-tick, profile-free logical state coalesced into S8 before the next telemetry
sample; its entry still binds the honest same-step pre-decision physical row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

try:  # Package import in tests and callers.
    from .fsm50_residual_scene import (
        ENV_REGEX_NS,
        OBSTACLE_BOTTOM_Z_M,
        OBSTACLE_CENTER_Y_M,
        OBSTACLE_FRONT_X_M,
        OBSTACLE_HEIGHT_M,
        OBSTACLE_LENGTH_M,
        OBSTACLE_PRIM_PATH,
        OBSTACLE_WIDTH_M,
        SERVO_JOINT_NAMES,
        WHEEL_JOINT_NAMES,
    )
except ImportError:  # Direct module execution with the replay root on PYTHONPATH.
    from fsm50_residual_scene import (  # type: ignore[no-redef]
        ENV_REGEX_NS,
        OBSTACLE_BOTTOM_Z_M,
        OBSTACLE_CENTER_Y_M,
        OBSTACLE_FRONT_X_M,
        OBSTACLE_HEIGHT_M,
        OBSTACLE_LENGTH_M,
        OBSTACLE_PRIM_PATH,
        OBSTACLE_WIDTH_M,
        SERVO_JOINT_NAMES,
        WHEEL_JOINT_NAMES,
    )


SCHEMA_VERSION = "fsm50.phase_entry_bank.v1"
BANK_ID = "fsm50-current-r4-reviewed-phase-entry-bank-v1"
SELECTION_POLICY = "FIRST_TELEMETRY_ROW_AT_OR_AFTER_STATE_ENTRY"
CURSOR_POLICY = "FIRST_SOURCE_ACTION_AT_OR_AFTER_STATE_ENTRY"

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "fsm50_phase_entry_bank.json"

TARGET_STATES = (
    "S5_PRE_RR_COM_SHIFT",
    "S7_PRE_RL_SUPPORT_SETUP",
    "S8_RL_COM_SHIFT_AND_TRAVERSE",
    "S10_POSTURE_RECOVERY",
)

CURRENT_SOURCE_VERSIONS = (
    "v003_20260805_224517_157723_manual",
    "v008_20260806_211408_578700_manual",
    "v009_20260806_215232_433234_manual",
)

GRAPH_ID = "fsm50-recording-derived-macro-v1"
GRAPH_SHA256 = "ffa5acfbf64b65c22eee54709a2afae5a56fa0b9345d8db84eb86acac58447c5"
PROFILE_LIBRARY_ID = "fsm50-gate-c-successful-recording-profiles-v1"
PROFILE_LIBRARY_SHA256 = "762665911785eb755b7000fbb2cb450a52f5579f3945d8f214129f41eaa50066"

ARTIFACT_FILENAMES = MappingProxyType(
    {
        "manifest": "macro_fsm_runner_manifest.json",
        "verdict": "manual_macro_video_verdict.json",
        "request": "worker_macro_fsm_request.json",
        "result": "worker_macro_fsm_result.json",
        "telemetry": "minimal_macro_telemetry.jsonl",
        "source": "macro_source_action_consumption.jsonl",
        "completion": "macro_segment_completion_ledger.jsonl",
        "transition": "macro_transition_evidence.jsonl",
    }
)


class PhaseEntryBankError(ValueError):
    """The current-only bank or one of its sealed inputs is invalid."""


@dataclass(frozen=True)
class _CurrentRunSpec:
    source_version: str
    run_relative_path: str
    trial_kind: str
    expected_count: int
    bundle_sha256: str
    artifact_sha256: Mapping[str, str]


_RUN_SPECS = (
    _CurrentRunSpec(
        source_version=CURRENT_SOURCE_VERSIONS[0],
        run_relative_path=(
            "runs/v003_macro_fsm_completion_aware_coalesced_r4/"
            "v003_20260805_224517_157723_manual/baseline/"
            "20260815T105857_601132Z_baseline_00_5742cda6e438"
        ),
        trial_kind="baseline",
        expected_count=112,
        bundle_sha256="5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78",
        artifact_sha256=MappingProxyType(
            {
                "manifest": "e4f74586cda839724ae070473e1dfd34704e1750d44e9129751ec0e6d12fb499",
                "verdict": "63141b536216b8aac1c123df13db3eb39392d00ac3a6a808b481f023235f1782",
                "request": "56279c4739378a055ea29d6457202c383623ab24d51bf4b663bd09cd45f932ce",
                "result": "dc7dd6d4e08d82ad82e5eab847194c25342a6cab72d6df7e4ae0cae75d5daa30",
                "telemetry": "72e2bedc991c455e318865aa92f2b62e14a1ff042b93fa54bc622915d2c805ea",
                "source": "41c3907d79dd429cb5e3831ac4bbc9725288be8ee3be8a950746c6c28ad0eb6a",
                "completion": "046704d92da8728e1c6a4791f7855d11922b99ecea4d30f9a67d53596a618a1a",
                "transition": "c3424f4b67b43ce34e9c923231758300b81f35b651cba9bd86d777b4dd05bca9",
            }
        ),
    ),
    _CurrentRunSpec(
        source_version=CURRENT_SOURCE_VERSIONS[1],
        run_relative_path=(
            "runs/cross_version_macro_fsm_completion_aware_coalesced_r4/"
            "v008_20260806_211408_578700_manual/trials/"
            "20260815T113449_254770Z_cross_version_00_0a47cee05cbb"
        ),
        trial_kind="cross_version",
        expected_count=119,
        bundle_sha256="0a47cee05cbb41a3a09b23836386bddcf9ae1ab2aca5bbe451ee20a0cbaa8c41",
        artifact_sha256=MappingProxyType(
            {
                "manifest": "7108c69d9f03a843a19405114a61c696616d068e5e05e8f7dc4c4d1d68ef619e",
                "verdict": "05721928e11a83708bb542538a5c06884fd47023d9353a614d73900ab05095b1",
                "request": "50db1315559ca58b9949e9dbaea4b31e09fba9c8e5647bfa19dd0afd472ce8a8",
                "result": "332ba356dbf2d52d271bed991509d9f9378a6aec9898df035bda7a96725abbc9",
                "telemetry": "fe6092d5abf2189b39ca16b40886c966e25376e8a783b25d0f54c9f6d8a0859e",
                "source": "e70a39b4948bfec60705885cd49a83753b01c3b0272ffd195f0323557afc9620",
                "completion": "c34bc3b14c37d6d59396f26a9e454d484b53314db17c2f80518b95c03157483c",
                "transition": "cb497ac87ceb241c0bb72dfb66afa4e3f6c4bf56816c383a14015966c8959674",
            }
        ),
    ),
    _CurrentRunSpec(
        source_version=CURRENT_SOURCE_VERSIONS[2],
        run_relative_path=(
            "runs/cross_version_macro_fsm_completion_aware_coalesced_r4/"
            "v009_20260806_215232_433234_manual/trials/"
            "20260815T114309_706871Z_cross_version_00_0e5686354eed"
        ),
        trial_kind="cross_version",
        expected_count=132,
        bundle_sha256="0e5686354eed7d89fb57d3aba722602eee7115b44749d531b6634dd0420ee919",
        artifact_sha256=MappingProxyType(
            {
                "manifest": "e46d6d06fb02ce967c911b93f14a8fdfed4b7bef13a41a4b0f66bfd3df000691",
                "verdict": "1fe1e673b44b812a10698d043554f8c761695699f289fb3da1741fb317daaa93",
                "request": "54c935fb20aa2e80a3b5fc9f971844682ebdc9eb33263a139d0456e5e6915002",
                "result": "64c3ad74ae5248a0371344a9f166db00f0c5eb7ca0e3461fe985573de697ba0f",
                "telemetry": "d54addcd284719ced25a64d534d090bca728b8cbcad40bb17af2d24e1f26c962",
                "source": "9017275f103d53fddd6653aae33da1f3a34f985ac7d455ee390184ec1a6aff47",
                "completion": "a29f4961dc73524a7b163695bedeaf0f0049dc9a0b937cd6a4f06512ab5e4357",
                "transition": "8470d9a351572575807712c46daaf30d4d75e3b2f4134ad778efd30fd6d224f0",
            }
        ),
    ),
)
_RUN_SPEC_BY_SOURCE = MappingProxyType({spec.source_version: spec for spec in _RUN_SPECS})


@dataclass(frozen=True)
class PhaseEntryBank:
    """Deeply immutable validated bank.

    Nested mappings are :class:`types.MappingProxyType` and JSON arrays are
    tuples.  ``to_mapping`` returns a detached mutable JSON-compatible copy.
    """

    schema_version: str
    bank_sha256: str
    payload: Mapping[str, Any]

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return self.payload["entries"]

    @property
    def source_bindings(self) -> tuple[Mapping[str, Any], ...]:
        return self.payload["source_bindings"]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bank_sha256": self.bank_sha256,
            "payload": _thaw(self.payload),
        }


def canonical_sha256(value: Any) -> str:
    """SHA-256 of strict canonical JSON (sorted keys, no NaN/Infinity)."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseEntryBankError(f"value is not strict canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def build_current_phase_entry_bank_mapping() -> dict[str, Any]:
    """Rebuild the deterministic bank from the three allowlisted r4 runs."""

    obstacle = _build_obstacle_identity()
    obstacle_sha = obstacle["identity_sha256"]
    bindings: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for spec in _RUN_SPECS:
        artifacts = _load_and_validate_current_run(spec)
        bindings.append(_build_source_binding(spec, artifacts))
        for state in TARGET_STATES:
            entries.append(_build_phase_entry(spec, artifacts, state, obstacle_sha))

    payload = {
        "bank_id": BANK_ID,
        "selection_policy": SELECTION_POLICY,
        "cursor_policy": CURSOR_POLICY,
        "source_versions": list(CURRENT_SOURCE_VERSIONS),
        "phase_states": list(TARGET_STATES),
        "joint_order": {
            "servo": list(SERVO_JOINT_NAMES),
            "wheel": list(WHEEL_JOINT_NAMES),
            "all": list(SERVO_JOINT_NAMES) + list(WHEEL_JOINT_NAMES),
        },
        "obstacle_identity": obstacle,
        "source_bindings": bindings,
        "entries": entries,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "bank_sha256": canonical_sha256(payload),
        "payload": payload,
    }


def validate_phase_entry_bank_mapping(
    value: Mapping[str, Any], *, verify_artifacts: bool = True
) -> PhaseEntryBank:
    """Validate strict schema, hashes, current-only paths, and optional files."""

    root = _require_mapping("bank", value)
    _require_exact_keys("bank", root, {"schema_version", "bank_sha256", "payload"})
    _require_equal("schema_version", root["schema_version"], SCHEMA_VERSION)
    bank_sha = _require_sha256("bank_sha256", root["bank_sha256"])
    payload = _require_mapping("payload", root["payload"])
    _validate_payload_shape(payload)
    if canonical_sha256(payload) != bank_sha:
        raise PhaseEntryBankError("bank_sha256 does not match canonical payload")

    if verify_artifacts:
        expected = build_current_phase_entry_bank_mapping()
        if canonical_sha256(root) != canonical_sha256(expected):
            raise PhaseEntryBankError(
                "bank is not the exact deterministic projection of the sealed current r4 artifacts"
            )

    return PhaseEntryBank(
        schema_version=SCHEMA_VERSION,
        bank_sha256=bank_sha,
        payload=_freeze(payload),
    )


def load_phase_entry_bank(
    path: str | Path = DEFAULT_CONFIG_PATH, *, verify_artifacts: bool = True
) -> PhaseEntryBank:
    """Load the allowlisted config path and validate it fail-closed."""

    candidate = Path(path)
    resolved = candidate.resolve(strict=True)
    expected = DEFAULT_CONFIG_PATH.resolve(strict=True)
    if resolved != expected:
        raise PhaseEntryBankError(
            f"only the current bank config is accepted: expected {expected}, got {resolved}"
        )
    _require_contained(resolved, MODULE_ROOT.resolve())
    value = _read_json(resolved)
    return validate_phase_entry_bank_mapping(value, verify_artifacts=verify_artifacts)


def render_current_phase_entry_bank_json() -> str:
    """Return the deterministic config text without writing any file."""

    return json.dumps(
        build_current_phase_entry_bank_mapping(),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _load_and_validate_current_run(spec: _CurrentRunSpec) -> dict[str, Any]:
    run_dir = _resolve_run_dir(spec.run_relative_path, spec)
    paths = {name: run_dir / filename for name, filename in ARTIFACT_FILENAMES.items()}
    for name, path in paths.items():
        actual = _file_sha256(path)
        expected = spec.artifact_sha256[name]
        if actual != expected:
            raise PhaseEntryBankError(
                f"{spec.source_version} {name} SHA drift: expected {expected}, got {actual}"
            )

    manifest = _read_json(paths["manifest"])
    verdict = _read_json(paths["verdict"])
    request = _read_json(paths["request"])
    result = _read_json(paths["result"])
    telemetry = _read_jsonl(paths["telemetry"])
    source = _read_jsonl(paths["source"])
    completion = _read_jsonl(paths["completion"])
    transition = _read_jsonl(paths["transition"])

    request_id = _require_text("request.request_id", request.get("request_id"))
    for label, actual in (
        ("manifest.request_id", manifest.get("request_id")),
        ("result.request_id", result.get("request_id")),
        ("verdict.request_id", verdict.get("request_id")),
    ):
        _require_equal(label, actual, request_id)
    for label, document in (
        ("request", request),
        ("result", result),
        ("manifest", manifest),
        ("verdict", verdict),
    ):
        _require_equal(f"{label}.source_version", document.get("source_version"), spec.source_version)

    for label, actual in (
        ("request.run_dir", request.get("run_dir")),
        ("result.run_dir", result.get("run_dir")),
        ("manifest.run_dir", manifest.get("run_dir")),
        ("verdict.run_dir", verdict.get("run_dir")),
    ):
        _require_absolute_path(label, actual, run_dir, run_dir)

    _require_absolute_path("manifest.request_path", manifest.get("request_path"), paths["request"], run_dir)
    _require_absolute_path(
        "manifest.manual_verdict_path", manifest.get("manual_verdict_path"), paths["verdict"], run_dir
    )
    _require_absolute_path(
        "verdict.worker_result_path", verdict.get("worker_result_path"), paths["result"], run_dir
    )
    _require_absolute_path(
        "result.telemetry_jsonl_path", result.get("telemetry_jsonl_path"), paths["telemetry"], run_dir
    )
    _require_absolute_path(
        "result.source_action_consumption_path",
        result.get("source_action_consumption_path"),
        paths["source"],
        run_dir,
    )
    _require_absolute_path(
        "result.segment_completion_ledger_path",
        result.get("segment_completion_ledger_path"),
        paths["completion"],
        run_dir,
    )

    _require_equal("manifest.request_sha256", manifest.get("request_sha256"), spec.artifact_sha256["request"])
    _require_equal(
        "manifest.manual_verdict_sha256", manifest.get("manual_verdict_sha256"), spec.artifact_sha256["verdict"]
    )
    _require_equal(
        "verdict.worker_result_sha256", verdict.get("worker_result_sha256"), spec.artifact_sha256["result"]
    )
    _require_equal(
        "result.source_action_consumption_sha256",
        result.get("source_action_consumption_sha256"),
        spec.artifact_sha256["source"],
    )
    _require_equal(
        "result.segment_completion_ledger_sha256",
        result.get("segment_completion_ledger_sha256"),
        spec.artifact_sha256["completion"],
    )

    for label, document in (("request", request), ("result", result)):
        _require_equal(f"{label}.graph_id", document.get("graph_id"), GRAPH_ID)
        _require_equal(f"{label}.graph_sha256", document.get("graph_sha256"), GRAPH_SHA256)
        _require_equal(
            f"{label}.profile_library_sha256",
            document.get("profile_library_sha256"),
            PROFILE_LIBRARY_SHA256,
        )
        _require_equal(f"{label}.bundle_sha256", document.get("bundle_sha256"), spec.bundle_sha256)

    bundle = _require_mapping("manifest.bundle", manifest.get("bundle"))
    _require_equal("manifest.bundle_sha256", manifest.get("bundle_sha256"), spec.bundle_sha256)
    _require_equal("manifest.bundle.graph_id", bundle.get("graph_id"), GRAPH_ID)
    _require_equal("manifest.bundle.graph_sha256", bundle.get("graph_sha256"), GRAPH_SHA256)
    _require_equal(
        "manifest.bundle.profile_library_id", bundle.get("profile_library_id"), PROFILE_LIBRARY_ID
    )
    _require_equal(
        "manifest.bundle.profile_library_sha256",
        bundle.get("profile_library_sha256"),
        PROFILE_LIBRARY_SHA256,
    )

    _require_equal("request.trial_kind", request.get("trial_kind"), spec.trial_kind)
    _require_equal("request.trial_index", request.get("trial_index"), 0)
    _require_equal("manifest.trial_kind", manifest.get("trial_kind"), spec.trial_kind)
    _require_equal("manifest.trial_index", manifest.get("trial_index"), 0)
    _require_equal(
        "manifest.manual_review_status",
        manifest.get("manual_review_status"),
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
    )
    _require_equal("verdict.review_complete", verdict.get("review_complete"), True)
    verdict_body = _require_mapping("verdict.verdict", verdict.get("verdict"))
    _require_equal("verdict.verdict.task_completed", verdict_body.get("task_completed"), True)
    _require_equal("verdict.verdict.robot_fell", verdict_body.get("robot_fell"), False)
    _require_equal("verdict.verdict.final_recoverable", verdict_body.get("final_recoverable"), True)
    _require_equal("result.macro_fsm_complete", result.get("macro_fsm_complete"), True)
    _require_equal(
        "result.controller_terminal_outcome",
        result.get("controller_terminal_outcome"),
        "TASK_SUCCESS_POSTURE_INCOMPLETE",
    )
    _require_equal("result.safe_stop_verified", result.get("safe_stop_verified"), True)
    _require_equal("result.segment_completion_coverage_complete", result.get("segment_completion_coverage_complete"), True)
    _require_equal("manifest.shutdown_verified", manifest.get("shutdown_verified"), True)

    for label, actual in (
        ("result.expected_source_action_count", result.get("expected_source_action_count")),
        ("result.source_action_consumption_count", result.get("source_action_consumption_count")),
        ("result.expected_segment_completion_count", result.get("expected_segment_completion_count")),
        ("result.segment_completion_count", result.get("segment_completion_count")),
        ("manifest.expected_segment_completion_count", manifest.get("expected_segment_completion_count")),
        ("manifest.segment_completion_count", manifest.get("segment_completion_count")),
    ):
        _require_equal(label, actual, spec.expected_count)
    if len(source) != spec.expected_count or len(completion) != spec.expected_count:
        raise PhaseEntryBankError(
            f"{spec.source_version} ledger count mismatch: source={len(source)}, completion={len(completion)}"
        )

    _validate_monotonic_telemetry(telemetry, spec.source_version)
    for index, row in enumerate(source):
        _require_equal(f"source[{index}].source_version", row.get("command_provenance", {}).get("source_version"), spec.source_version)
        _require_equal(f"source[{index}].bundle_sha256", row.get("bundle_sha256"), spec.bundle_sha256)
        _require_equal(
            f"source[{index}].profile_library_sha256",
            row.get("profile_library_sha256"),
            PROFILE_LIBRARY_SHA256,
        )
        _require_equal(f"source[{index}].source_action_index", row.get("source_action_index"), index)
    for index, row in enumerate(completion):
        _require_equal(f"completion[{index}].source_version", row.get("source_version"), spec.source_version)
        _require_equal(f"completion[{index}].segment_completion_index", row.get("segment_completion_index"), index)
        _require_equal(f"completion[{index}].terminal_kind", row.get("terminal_kind"), "COMPLETE")

    return {
        "run_dir": run_dir,
        "paths": paths,
        "manifest": manifest,
        "verdict": verdict,
        "request": request,
        "result": result,
        "telemetry": telemetry,
        "source": source,
        "completion": completion,
        "transition": transition,
    }


def _build_source_binding(spec: _CurrentRunSpec, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    manifest = artifacts["manifest"]
    request = artifacts["request"]
    binding = {
        "source_version": spec.source_version,
        "run_relative_path": spec.run_relative_path,
        "request_id": request["request_id"],
        "trial_kind": spec.trial_kind,
        "trial_index": 0,
        "reviewed_utc": manifest["reviewed_utc"],
        "manual_review_status": manifest["manual_review_status"],
        "expected_source_action_count": spec.expected_count,
        "expected_segment_completion_count": spec.expected_count,
        "graph_id": GRAPH_ID,
        "graph_sha256": GRAPH_SHA256,
        "profile_library_id": PROFILE_LIBRARY_ID,
        "profile_library_sha256": PROFILE_LIBRARY_SHA256,
        "bundle_sha256": spec.bundle_sha256,
        "artifacts": {
            name: {
                "path": f"{spec.run_relative_path}/{ARTIFACT_FILENAMES[name]}",
                "sha256": spec.artifact_sha256[name],
            }
            for name in ARTIFACT_FILENAMES
        },
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _build_phase_entry(
    spec: _CurrentRunSpec,
    artifacts: Mapping[str, Any],
    phase_state: str,
    obstacle_sha256: str,
) -> dict[str, Any]:
    transitions = artifacts["transition"]
    matching_entries = [
        row
        for row in transitions
        if row.get("to_state") == phase_state and row.get("from_state") != phase_state
    ]
    if len(matching_entries) != 1:
        raise PhaseEntryBankError(
            f"{spec.source_version} must contain exactly one entry transition for {phase_state}; "
            f"got {len(matching_entries)}"
        )
    transition = matching_entries[0]
    entry_step = _require_nonnegative_int("entry sim_step", transition.get("sim_step"))
    entry_time = _require_finite_number("entry sim_time_s", transition.get("sim_time_s"))

    telemetry_rows = artifacts["telemetry"]
    selected: tuple[int, Mapping[str, Any]] | None = None
    for index, row in enumerate(telemetry_rows):
        if _require_nonnegative_int(f"telemetry[{index}].sim_step", row.get("sim_step")) >= entry_step:
            selected = (index, row)
            break
    if selected is None:
        raise PhaseEntryBankError(f"{spec.source_version} has no telemetry sample after entry {phase_state}")
    telemetry_index, telemetry = selected
    snapshot_step = _require_nonnegative_int("snapshot sim_step", telemetry.get("sim_step"))
    snapshot_time = _require_finite_number("snapshot sim_time_s", telemetry.get("sim_time_s"))
    latency = snapshot_step - entry_step
    if latency < 0 or latency > 8:
        raise PhaseEntryBankError(
            f"{spec.source_version} {phase_state} telemetry latency must be 0..8 ticks, got {latency}"
        )

    observed_state = _require_text("telemetry.macro_state", telemetry.get("macro_state"), allow_empty=False)
    observation_relation = (
        "SAME_STEP_PRE_DECISION_OBSERVATION"
        if snapshot_step == entry_step
        else "FIRST_POST_ENTRY_TELEMETRY_OBSERVATION"
    )
    if snapshot_step == entry_step:
        expected_observed = _require_text(
            "transition.from_state", transition.get("from_state"), allow_empty=False
        )
        if observed_state != expected_observed:
            raise PhaseEntryBankError(
                f"same-step entry telemetry must retain pre-decision state {expected_observed}, "
                f"got {observed_state}"
            )
    elif observed_state != phase_state:
        raise PhaseEntryBankError(
            f"first later telemetry after {phase_state} entry unexpectedly observed {observed_state}"
        )

    coalesced_state = ""
    coalesced_index: int | None = None
    if phase_state == "S7_PRE_RL_SUPPORT_SETUP":
        next_telemetry_steps = [
            _require_nonnegative_int("next telemetry sim_step", row.get("sim_step"))
            for row in telemetry_rows
            if row.get("sim_step", -1) > entry_step
        ]
        next_telemetry_step = next_telemetry_steps[0] if next_telemetry_steps else entry_step
        bridges = [
            row
            for row in transitions
            if row.get("from_state") == phase_state
            and row.get("to_state") == "S8_RL_COM_SHIFT_AND_TRAVERSE"
            and entry_step < row.get("sim_step", -1) <= next_telemetry_step
        ]
        if len(bridges) > 1:
            raise PhaseEntryBankError(
                f"{spec.source_version} {phase_state} has duplicate coalescing transitions"
            )
        if bridges:
            coalesced_state = "S8_RL_COM_SHIFT_AND_TRAVERSE"
            coalesced_index = _require_nonnegative_int(
                "coalesced transition_index", bridges[0].get("transition_index")
            )

    source_rows = artifacts["source"]
    cursor_candidates = [row for row in source_rows if row.get("sim_step", -1) >= entry_step]
    if not cursor_candidates:
        raise PhaseEntryBankError(f"{spec.source_version} {phase_state} has no source cursor")
    cursor_row = cursor_candidates[0]
    cursor_index = _require_nonnegative_int("source_action_index", cursor_row.get("source_action_index"))
    provenance = _require_mapping("command_provenance", cursor_row.get("command_provenance"))
    segment_index = _require_nonnegative_int("source_segment_index", provenance.get("source_segment_index"))
    completion_matches = [
        row
        for row in artifacts["completion"]
        if row.get("source_action_consumption_index") == cursor_index
        and row.get("source_segment_index") == segment_index
    ]
    if len(completion_matches) != 1:
        raise PhaseEntryBankError(
            f"{spec.source_version} {phase_state} cursor must bind one completion row; got {len(completion_matches)}"
        )
    completion = completion_matches[0]

    root_position = _require_vector("root_position_w", telemetry.get("root_position_w"), 3)
    root_orientation = _require_vector("root_orientation_wxyz", telemetry.get("root_orientation_wxyz"), 4)
    root_pose = _require_vector("root_pose_w", telemetry.get("root_pose_w"), 7)
    if root_pose != root_position + root_orientation:
        raise PhaseEntryBankError("root_pose_w is not exact position + wxyz orientation")
    quat_norm = math.sqrt(sum(value * value for value in root_orientation))
    if abs(quat_norm - 1.0) > 1.0e-4:
        raise PhaseEntryBankError(f"root orientation is not unit wxyz: norm={quat_norm}")

    joint_q = _ordered_numeric_map("joint_q_rad", telemetry.get("joint_q_rad"), SERVO_JOINT_NAMES + WHEEL_JOINT_NAMES)
    joint_qd = _ordered_numeric_map(
        "joint_qd_rad_s", telemetry.get("joint_qd_rad_s"), SERVO_JOINT_NAMES + WHEEL_JOINT_NAMES
    )
    nominal_servo = _ordered_numeric_map("servo_targets_deg", telemetry.get("servo_targets_deg"), SERVO_JOINT_NAMES)
    nominal_wheel = _ordered_numeric_map("wheel_targets_rad_s", telemetry.get("wheel_targets_rad_s"), WHEEL_JOINT_NAMES)

    _validate_telemetry_obstacle(telemetry)
    transition_profile_fraction = _require_finite_number(
        "transition.profile_fraction", transition.get("profile_fraction")
    )
    snapshot_profile_fraction = _require_finite_number(
        "telemetry.profile_fraction", telemetry.get("profile_fraction")
    )
    cursor_servo = _ordered_numeric_map(
        "cursor.servo_targets_deg", cursor_row.get("servo_targets_deg"), SERVO_JOINT_NAMES
    )
    cursor_wheel = _ordered_numeric_map(
        "cursor.wheel_targets_rad_s", cursor_row.get("wheel_targets_rad_s"), WHEEL_JOINT_NAMES
    )
    event_indices = _require_int_list("source_event_indices", provenance.get("source_event_indices"))
    commands = _require_text_list("commands", provenance.get("commands"))

    entry = {
        "entry_id": f"{spec.source_version}:{phase_state}",
        "source_version": spec.source_version,
        "phase_state": phase_state,
        "entry_transition_index": _require_nonnegative_int(
            "entry transition_index", transition.get("transition_index")
        ),
        "entry_transition_row_sha256": canonical_sha256(transition),
        "entry_sim_step": entry_step,
        "entry_sim_time_s": entry_time,
        "telemetry_row_index": telemetry_index,
        "telemetry_row_sha256": canonical_sha256(telemetry),
        "snapshot_sim_step": snapshot_step,
        "snapshot_sim_time_s": snapshot_time,
        "telemetry_latency_physics_steps": latency,
        "telemetry_relation_to_entry": observation_relation,
        "observed_macro_state": observed_state,
        "coalesced_successor_state": coalesced_state,
        "coalesced_transition_index": coalesced_index,
        "root_position_m": root_position,
        "root_orientation_wxyz": root_orientation,
        "root_linear_velocity_m_s": _require_vector(
            "root_linear_velocity_w", telemetry.get("root_linear_velocity_w"), 3
        ),
        "root_angular_velocity_rad_s": _require_vector(
            "root_angular_velocity_w", telemetry.get("root_angular_velocity_w"), 3
        ),
        "joint_q_rad": joint_q,
        "joint_qd_rad_s": joint_qd,
        "nominal_servo_targets_deg": nominal_servo,
        "nominal_wheel_targets_rad_s": nominal_wheel,
        "entry_profile": {
            "profile_id": _require_text("transition.profile_id", transition.get("profile_id")),
            "profile_source_version": _require_text(
                "transition.profile_source_version", transition.get("profile_source_version")
            ),
            "profile_strategy": _require_text(
                "transition.profile_strategy", transition.get("profile_strategy")
            ),
            "profile_fraction": transition_profile_fraction,
            "subphase": _require_text("transition.subphase", transition.get("subphase")),
            "command_epoch": _require_nonnegative_int(
                "transition.command_epoch", transition.get("command_epoch")
            ),
            "profile_free": transition.get("profile_id") == "",
        },
        "snapshot_context": {
            "macro_state": observed_state,
            "profile_id": _require_text("telemetry.profile_id", telemetry.get("profile_id")),
            "profile_source_version": _require_text(
                "telemetry.profile_source_version", telemetry.get("profile_source_version")
            ),
            "profile_strategy": _require_text(
                "telemetry.profile_strategy", telemetry.get("profile_strategy")
            ),
            "profile_fraction": snapshot_profile_fraction,
            "subphase": _require_text("telemetry.subphase", telemetry.get("subphase")),
            "command_epoch": _require_nonnegative_int(
                "telemetry.command_epoch", telemetry.get("command_epoch")
            ),
            "active_completion_source_segment_index": _optional_nonnegative_int(
                "telemetry.active_completion_source_segment_index",
                telemetry.get("active_completion_source_segment_index"),
            ),
        },
        "source_cursor": {
            "source_action_index": cursor_index,
            "source_action_identity": _require_sha256(
                "source_action_identity", provenance.get("source_action_identity")
            ),
            "source_action_row_sha256": canonical_sha256(cursor_row),
            "source_segment_index": segment_index,
            "source_step_index": _require_nonnegative_int(
                "source_step_index", provenance.get("source_step_index")
            ),
            "source_step_id": _require_text(
                "completion.source_step_id", completion.get("source_step_id"), allow_empty=False
            ),
            "source_time_s": _require_finite_number("source_time_s", provenance.get("source_time_s")),
            "source_event_indices": event_indices,
            "commands": commands,
            "source_plan_sha256": _require_sha256(
                "source_plan_sha256", cursor_row.get("source_plan_sha256")
            ),
            "source_plan_payload_sha256": _require_sha256(
                "source_plan_payload_sha256", completion.get("source_plan_payload_sha256")
            ),
            "owner_state": _require_text("cursor.macro_state", cursor_row.get("macro_state"), allow_empty=False),
            "profile_id": _require_text("cursor.profile_id", cursor_row.get("profile_id"), allow_empty=False),
            "profile_source_version": _require_text(
                "cursor.profile_source_version", cursor_row.get("profile_source_version"), allow_empty=False
            ),
            "profile_strategy": _require_text(
                "cursor.profile_strategy", cursor_row.get("profile_strategy"), allow_empty=False
            ),
            "subphase": _require_text("cursor.subphase", cursor_row.get("subphase")),
            "dispatch_sim_step": _require_nonnegative_int(
                "cursor.sim_step", cursor_row.get("sim_step")
            ),
            "dispatch_sim_time_s": _require_finite_number(
                "cursor.sim_time_s", cursor_row.get("sim_time_s")
            ),
            "segment_completion_index": _require_nonnegative_int(
                "segment_completion_index", completion.get("segment_completion_index")
            ),
            "completion_row_sha256": canonical_sha256(completion),
            "servo_targets_deg": cursor_servo,
            "wheel_targets_rad_s": cursor_wheel,
        },
        "obstacle_identity_sha256": obstacle_sha256,
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry


def _build_obstacle_identity() -> dict[str, Any]:
    obstacle = {
        "schema_version": "fsm50.formal_obstacle_identity.v1",
        "prim_path": OBSTACLE_PRIM_PATH,
        "env_namespace_token": ENV_REGEX_NS,
        "front_face_x_m": OBSTACLE_FRONT_X_M,
        "rear_face_x_m": OBSTACLE_FRONT_X_M + OBSTACLE_LENGTH_M,
        "center_x_m": OBSTACLE_FRONT_X_M + 0.5 * OBSTACLE_LENGTH_M,
        "center_y_m": OBSTACLE_CENTER_Y_M,
        "bottom_z_m": OBSTACLE_BOTTOM_Z_M,
        "top_z_m": OBSTACLE_BOTTOM_Z_M + OBSTACLE_HEIGHT_M,
        "length_m": OBSTACLE_LENGTH_M,
        "width_m": OBSTACLE_WIDTH_M,
        "height_m": OBSTACLE_HEIGHT_M,
    }
    obstacle["identity_sha256"] = canonical_sha256(obstacle)
    return obstacle


def _validate_payload_shape(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(
        "payload",
        payload,
        {
            "bank_id",
            "selection_policy",
            "cursor_policy",
            "source_versions",
            "phase_states",
            "joint_order",
            "obstacle_identity",
            "source_bindings",
            "entries",
        },
    )
    _require_equal("payload.bank_id", payload["bank_id"], BANK_ID)
    _require_equal("payload.selection_policy", payload["selection_policy"], SELECTION_POLICY)
    _require_equal("payload.cursor_policy", payload["cursor_policy"], CURSOR_POLICY)
    _require_equal("payload.source_versions", payload["source_versions"], list(CURRENT_SOURCE_VERSIONS))
    _require_equal("payload.phase_states", payload["phase_states"], list(TARGET_STATES))

    joint_order = _require_mapping("payload.joint_order", payload["joint_order"])
    _require_exact_keys("payload.joint_order", joint_order, {"servo", "wheel", "all"})
    _require_equal("joint_order.servo", joint_order["servo"], list(SERVO_JOINT_NAMES))
    _require_equal("joint_order.wheel", joint_order["wheel"], list(WHEEL_JOINT_NAMES))
    _require_equal(
        "joint_order.all", joint_order["all"], list(SERVO_JOINT_NAMES) + list(WHEEL_JOINT_NAMES)
    )
    _validate_obstacle_shape(payload["obstacle_identity"])

    bindings = _require_list("payload.source_bindings", payload["source_bindings"])
    if len(bindings) != len(_RUN_SPECS):
        raise PhaseEntryBankError(f"source_bindings must contain exactly {len(_RUN_SPECS)} rows")
    seen_sources: set[str] = set()
    for index, raw_binding in enumerate(bindings):
        binding = _require_mapping(f"source_bindings[{index}]", raw_binding)
        _validate_binding_shape(binding, _RUN_SPECS[index])
        source = binding["source_version"]
        if source in seen_sources:
            raise PhaseEntryBankError(f"duplicate source binding {source}")
        seen_sources.add(source)

    entries = _require_list("payload.entries", payload["entries"])
    expected_entry_count = len(CURRENT_SOURCE_VERSIONS) * len(TARGET_STATES)
    if len(entries) != expected_entry_count:
        raise PhaseEntryBankError(f"entries must contain exactly {expected_entry_count} rows")
    seen_entries: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(f"entries[{index}]", raw_entry)
        source = _require_text(f"entries[{index}].source_version", entry.get("source_version"))
        state = _require_text(f"entries[{index}].phase_state", entry.get("phase_state"))
        key = (source, state)
        if key in seen_entries:
            raise PhaseEntryBankError(f"duplicate phase entry {key}")
        seen_entries.add(key)
    seen_entries.clear()
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(f"entries[{index}]", raw_entry)
        expected_source = CURRENT_SOURCE_VERSIONS[index // len(TARGET_STATES)]
        expected_state = TARGET_STATES[index % len(TARGET_STATES)]
        _validate_entry_shape(entry, expected_source, expected_state, payload["obstacle_identity"])
        key = (entry["source_version"], entry["phase_state"])
        if key in seen_entries:
            raise PhaseEntryBankError(f"duplicate phase entry {key}")
        seen_entries.add(key)


_BINDING_KEYS = {
    "source_version",
    "run_relative_path",
    "request_id",
    "trial_kind",
    "trial_index",
    "reviewed_utc",
    "manual_review_status",
    "expected_source_action_count",
    "expected_segment_completion_count",
    "graph_id",
    "graph_sha256",
    "profile_library_id",
    "profile_library_sha256",
    "bundle_sha256",
    "artifacts",
    "binding_sha256",
}


def _validate_binding_shape(binding: Mapping[str, Any], spec: _CurrentRunSpec) -> None:
    _require_exact_keys(f"binding[{spec.source_version}]", binding, _BINDING_KEYS)
    _require_equal("binding.source_version", binding["source_version"], spec.source_version)
    _resolve_run_dir(_require_text("binding.run_relative_path", binding["run_relative_path"]), spec)
    _require_text("binding.request_id", binding["request_id"], allow_empty=False)
    _require_equal("binding.trial_kind", binding["trial_kind"], spec.trial_kind)
    _require_equal("binding.trial_index", binding["trial_index"], 0)
    _require_text("binding.reviewed_utc", binding["reviewed_utc"], allow_empty=False)
    _require_equal(
        "binding.manual_review_status",
        binding["manual_review_status"],
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
    )
    _require_equal("binding.expected_source_action_count", binding["expected_source_action_count"], spec.expected_count)
    _require_equal(
        "binding.expected_segment_completion_count",
        binding["expected_segment_completion_count"],
        spec.expected_count,
    )
    _require_equal("binding.graph_id", binding["graph_id"], GRAPH_ID)
    _require_equal("binding.graph_sha256", binding["graph_sha256"], GRAPH_SHA256)
    _require_equal("binding.profile_library_id", binding["profile_library_id"], PROFILE_LIBRARY_ID)
    _require_equal(
        "binding.profile_library_sha256", binding["profile_library_sha256"], PROFILE_LIBRARY_SHA256
    )
    _require_equal("binding.bundle_sha256", binding["bundle_sha256"], spec.bundle_sha256)

    artifacts = _require_mapping("binding.artifacts", binding["artifacts"])
    _require_exact_keys("binding.artifacts", artifacts, set(ARTIFACT_FILENAMES))
    for name, filename in ARTIFACT_FILENAMES.items():
        ref = _require_mapping(f"binding.artifacts.{name}", artifacts[name])
        _require_exact_keys(f"binding.artifacts.{name}", ref, {"path", "sha256"})
        expected_path = f"{spec.run_relative_path}/{filename}"
        _require_equal(f"binding.artifacts.{name}.path", ref["path"], expected_path)
        _resolve_module_relative(ref["path"], MODULE_ROOT / Path(expected_path))
        _require_equal(
            f"binding.artifacts.{name}.sha256",
            _require_sha256(f"binding.artifacts.{name}.sha256", ref["sha256"]),
            spec.artifact_sha256[name],
        )

    sealed = dict(binding)
    supplied = _require_sha256("binding.binding_sha256", sealed.pop("binding_sha256"))
    if canonical_sha256(sealed) != supplied:
        raise PhaseEntryBankError(f"binding SHA mismatch for {spec.source_version}")


_ENTRY_KEYS = {
    "entry_id",
    "source_version",
    "phase_state",
    "entry_transition_index",
    "entry_transition_row_sha256",
    "entry_sim_step",
    "entry_sim_time_s",
    "telemetry_row_index",
    "telemetry_row_sha256",
    "snapshot_sim_step",
    "snapshot_sim_time_s",
    "telemetry_latency_physics_steps",
    "telemetry_relation_to_entry",
    "observed_macro_state",
    "coalesced_successor_state",
    "coalesced_transition_index",
    "root_position_m",
    "root_orientation_wxyz",
    "root_linear_velocity_m_s",
    "root_angular_velocity_rad_s",
    "joint_q_rad",
    "joint_qd_rad_s",
    "nominal_servo_targets_deg",
    "nominal_wheel_targets_rad_s",
    "entry_profile",
    "snapshot_context",
    "source_cursor",
    "obstacle_identity_sha256",
    "entry_sha256",
}


def _validate_entry_shape(
    entry: Mapping[str, Any],
    expected_source: str,
    expected_state: str,
    obstacle: Any,
) -> None:
    label = f"entry[{expected_source}:{expected_state}]"
    _require_exact_keys(label, entry, _ENTRY_KEYS)
    _require_equal(f"{label}.source_version", entry["source_version"], expected_source)
    _require_equal(f"{label}.phase_state", entry["phase_state"], expected_state)
    _require_equal(f"{label}.entry_id", entry["entry_id"], f"{expected_source}:{expected_state}")
    _require_nonnegative_int(f"{label}.entry_transition_index", entry["entry_transition_index"])
    _require_sha256(f"{label}.entry_transition_row_sha256", entry["entry_transition_row_sha256"])
    entry_step = _require_nonnegative_int(f"{label}.entry_sim_step", entry["entry_sim_step"])
    entry_time = _require_finite_number(f"{label}.entry_sim_time_s", entry["entry_sim_time_s"])
    _require_nonnegative_int(f"{label}.telemetry_row_index", entry["telemetry_row_index"])
    _require_sha256(f"{label}.telemetry_row_sha256", entry["telemetry_row_sha256"])
    snapshot_step = _require_nonnegative_int(f"{label}.snapshot_sim_step", entry["snapshot_sim_step"])
    snapshot_time = _require_finite_number(f"{label}.snapshot_sim_time_s", entry["snapshot_sim_time_s"])
    latency = _require_nonnegative_int(
        f"{label}.telemetry_latency_physics_steps", entry["telemetry_latency_physics_steps"]
    )
    if snapshot_step - entry_step != latency or latency < 0 or latency > 8:
        raise PhaseEntryBankError(f"{label} has invalid entry telemetry latency")
    if abs(entry_time - entry_step / 120.0) > 1.0e-9 or abs(snapshot_time - snapshot_step / 120.0) > 1.0e-9:
        raise PhaseEntryBankError(f"{label} step/time identity is not 120 Hz")

    relation = _require_text(
        f"{label}.telemetry_relation_to_entry",
        entry["telemetry_relation_to_entry"],
        allow_empty=False,
    )
    expected_relation = (
        "SAME_STEP_PRE_DECISION_OBSERVATION"
        if latency == 0
        else "FIRST_POST_ENTRY_TELEMETRY_OBSERVATION"
    )
    _require_equal(f"{label}.telemetry_relation_to_entry", relation, expected_relation)
    observed = _require_text(f"{label}.observed_macro_state", entry["observed_macro_state"], allow_empty=False)
    coalesced = _require_text(f"{label}.coalesced_successor_state", entry["coalesced_successor_state"])
    coalesced_index = entry["coalesced_transition_index"]
    if coalesced:
        if expected_state != TARGET_STATES[1] or coalesced != TARGET_STATES[2]:
            raise PhaseEntryBankError(f"{label} has an unauthorized coalesced successor")
        _require_nonnegative_int(f"{label}.coalesced_transition_index", coalesced_index)
    elif coalesced_index is not None:
        raise PhaseEntryBankError(f"{label} has a coalesced index without a successor")
    if latency > 0 and observed != expected_state:
        raise PhaseEntryBankError(f"{label} later telemetry does not observe the entered state")

    _require_vector(f"{label}.root_position_m", entry["root_position_m"], 3)
    quat = _require_vector(f"{label}.root_orientation_wxyz", entry["root_orientation_wxyz"], 4)
    if abs(math.sqrt(sum(value * value for value in quat)) - 1.0) > 1.0e-4:
        raise PhaseEntryBankError(f"{label} quaternion is not unit wxyz")
    _require_vector(f"{label}.root_linear_velocity_m_s", entry["root_linear_velocity_m_s"], 3)
    _require_vector(f"{label}.root_angular_velocity_rad_s", entry["root_angular_velocity_rad_s"], 3)
    _require_vector(f"{label}.joint_q_rad", entry["joint_q_rad"], 12)
    _require_vector(f"{label}.joint_qd_rad_s", entry["joint_qd_rad_s"], 12)
    _require_vector(f"{label}.nominal_servo_targets_deg", entry["nominal_servo_targets_deg"], 8)
    _require_vector(f"{label}.nominal_wheel_targets_rad_s", entry["nominal_wheel_targets_rad_s"], 4)

    profile = _require_mapping(f"{label}.entry_profile", entry["entry_profile"])
    _validate_profile_shape(f"{label}.entry_profile", profile)
    context = _require_mapping(f"{label}.snapshot_context", entry["snapshot_context"])
    _require_exact_keys(
        f"{label}.snapshot_context",
        context,
        {
            "macro_state",
            "profile_id",
            "profile_source_version",
            "profile_strategy",
            "profile_fraction",
            "subphase",
            "command_epoch",
            "active_completion_source_segment_index",
        },
    )
    _require_equal(f"{label}.snapshot_context.macro_state", context["macro_state"], observed)
    for name in ("profile_id", "profile_source_version", "profile_strategy", "subphase"):
        _require_text(f"{label}.snapshot_context.{name}", context[name])
    _require_finite_number(f"{label}.snapshot_context.profile_fraction", context["profile_fraction"])
    _require_nonnegative_int(f"{label}.snapshot_context.command_epoch", context["command_epoch"])
    _optional_nonnegative_int(
        f"{label}.snapshot_context.active_completion_source_segment_index",
        context["active_completion_source_segment_index"],
    )
    _validate_cursor_shape(
        f"{label}.source_cursor",
        entry["source_cursor"],
        expected_source,
        entry_step,
        snapshot_step,
    )

    obstacle_map = _require_mapping("payload.obstacle_identity", obstacle)
    _require_equal(
        f"{label}.obstacle_identity_sha256",
        entry["obstacle_identity_sha256"],
        obstacle_map["identity_sha256"],
    )
    sealed = dict(entry)
    supplied = _require_sha256(f"{label}.entry_sha256", sealed.pop("entry_sha256"))
    if canonical_sha256(sealed) != supplied:
        raise PhaseEntryBankError(f"{label} entry_sha256 mismatch")


def _validate_profile_shape(label: str, profile: Mapping[str, Any]) -> None:
    _require_exact_keys(
        label,
        profile,
        {
            "profile_id",
            "profile_source_version",
            "profile_strategy",
            "profile_fraction",
            "subphase",
            "command_epoch",
            "profile_free",
        },
    )
    for name in ("profile_id", "profile_source_version", "profile_strategy", "subphase"):
        _require_text(f"{label}.{name}", profile[name])
    _require_finite_number(f"{label}.profile_fraction", profile["profile_fraction"])
    _require_nonnegative_int(f"{label}.command_epoch", profile["command_epoch"])
    if not isinstance(profile["profile_free"], bool):
        raise PhaseEntryBankError(f"{label}.profile_free must be boolean")
    expected_profile_free = profile["profile_id"] == ""
    if profile["profile_free"] is not expected_profile_free:
        raise PhaseEntryBankError(f"{label}.profile_free contradicts profile_id")
    if expected_profile_free and (
        profile["profile_source_version"] != "" or profile["profile_strategy"] != ""
    ):
        raise PhaseEntryBankError(f"{label} profile-free entry carries profile identity")


_CURSOR_KEYS = {
    "source_action_index",
    "source_action_identity",
    "source_action_row_sha256",
    "source_segment_index",
    "source_step_index",
    "source_step_id",
    "source_time_s",
    "source_event_indices",
    "commands",
    "source_plan_sha256",
    "source_plan_payload_sha256",
    "owner_state",
    "profile_id",
    "profile_source_version",
    "profile_strategy",
    "subphase",
    "dispatch_sim_step",
    "dispatch_sim_time_s",
    "segment_completion_index",
    "completion_row_sha256",
    "servo_targets_deg",
    "wheel_targets_rad_s",
}


def _validate_cursor_shape(
    label: str,
    raw_cursor: Any,
    expected_source: str,
    entry_step: int,
    snapshot_step: int,
) -> None:
    cursor = _require_mapping(label, raw_cursor)
    _require_exact_keys(label, cursor, _CURSOR_KEYS)
    action_index = _require_nonnegative_int(f"{label}.source_action_index", cursor["source_action_index"])
    _require_sha256(f"{label}.source_action_identity", cursor["source_action_identity"])
    _require_sha256(f"{label}.source_action_row_sha256", cursor["source_action_row_sha256"])
    segment_index = _require_nonnegative_int(f"{label}.source_segment_index", cursor["source_segment_index"])
    if action_index != segment_index:
        raise PhaseEntryBankError(f"{label} action/segment indices must be identical in the current fast plan")
    _require_nonnegative_int(f"{label}.source_step_index", cursor["source_step_index"])
    _require_text(f"{label}.source_step_id", cursor["source_step_id"], allow_empty=False)
    _require_finite_number(f"{label}.source_time_s", cursor["source_time_s"])
    _require_int_list(f"{label}.source_event_indices", cursor["source_event_indices"])
    _require_text_list(f"{label}.commands", cursor["commands"])
    _require_sha256(f"{label}.source_plan_sha256", cursor["source_plan_sha256"])
    _require_sha256(f"{label}.source_plan_payload_sha256", cursor["source_plan_payload_sha256"])
    for name in ("owner_state", "profile_id", "profile_source_version", "profile_strategy"):
        _require_text(f"{label}.{name}", cursor[name], allow_empty=False)
    _require_equal(f"{label}.profile_source_version", cursor["profile_source_version"], expected_source)
    _require_text(f"{label}.subphase", cursor["subphase"])
    dispatch_step = _require_nonnegative_int(f"{label}.dispatch_sim_step", cursor["dispatch_sim_step"])
    if dispatch_step < entry_step:
        raise PhaseEntryBankError(f"{label} dispatch precedes state entry")
    if dispatch_step < snapshot_step:
        raise PhaseEntryBankError(f"{label} cursor dispatch precedes its physical snapshot")
    dispatch_time = _require_finite_number(f"{label}.dispatch_sim_time_s", cursor["dispatch_sim_time_s"])
    if abs(dispatch_time - dispatch_step / 120.0) > 1.0e-9:
        raise PhaseEntryBankError(f"{label} dispatch step/time identity is not 120 Hz")
    _require_nonnegative_int(f"{label}.segment_completion_index", cursor["segment_completion_index"])
    _require_sha256(f"{label}.completion_row_sha256", cursor["completion_row_sha256"])
    _require_vector(f"{label}.servo_targets_deg", cursor["servo_targets_deg"], 8)
    _require_vector(f"{label}.wheel_targets_rad_s", cursor["wheel_targets_rad_s"], 4)


def _validate_obstacle_shape(raw_obstacle: Any) -> None:
    obstacle = _require_mapping("payload.obstacle_identity", raw_obstacle)
    keys = {
        "schema_version",
        "prim_path",
        "env_namespace_token",
        "front_face_x_m",
        "rear_face_x_m",
        "center_x_m",
        "center_y_m",
        "bottom_z_m",
        "top_z_m",
        "length_m",
        "width_m",
        "height_m",
        "identity_sha256",
    }
    _require_exact_keys("payload.obstacle_identity", obstacle, keys)
    expected = _build_obstacle_identity()
    if obstacle != expected:
        raise PhaseEntryBankError("obstacle identity drifted from the formal namespaced 50 mm scene")
    sealed = dict(obstacle)
    supplied = _require_sha256("obstacle.identity_sha256", sealed.pop("identity_sha256"))
    if canonical_sha256(sealed) != supplied:
        raise PhaseEntryBankError("obstacle identity SHA mismatch")


def _validate_telemetry_obstacle(row: Mapping[str, Any]) -> None:
    _require_equal("telemetry.obstacle_front_face_x_m", row.get("obstacle_front_face_x_m"), OBSTACLE_FRONT_X_M)
    _require_equal(
        "telemetry.obstacle_rear_face_x_m",
        row.get("obstacle_rear_face_x_m"),
        OBSTACLE_FRONT_X_M + OBSTACLE_LENGTH_M,
    )
    _require_equal(
        "telemetry.obstacle_top_z_m",
        row.get("obstacle_top_z_m"),
        OBSTACLE_BOTTOM_Z_M + OBSTACLE_HEIGHT_M,
    )


def _validate_monotonic_telemetry(rows: Sequence[Mapping[str, Any]], source_version: str) -> None:
    if not rows:
        raise PhaseEntryBankError(f"{source_version} telemetry is empty")
    previous = -1
    for index, row in enumerate(rows):
        _require_equal(f"telemetry[{index}].source_version", row.get("source_version"), source_version)
        step = _require_nonnegative_int(f"telemetry[{index}].sim_step", row.get("sim_step"))
        if step < previous:
            raise PhaseEntryBankError(f"{source_version} telemetry steps move backwards")
        if step == previous:
            previous_row = rows[index - 1]
            if not (
                index == len(rows) - 1
                and row.get("controller_terminal") is True
                and previous_row.get("controller_terminal") is True
                and row.get("macro_state") == "SUCCESS"
                and previous_row.get("macro_state") == "SUCCESS"
            ):
                raise PhaseEntryBankError(
                    f"{source_version} only permits the sealed duplicate terminal sample"
                )
        previous = step


def _resolve_run_dir(path_text: str, spec: _CurrentRunSpec) -> Path:
    if path_text != spec.run_relative_path:
        raise PhaseEntryBankError(
            f"run path is not the allowlisted current r4 {spec.source_version} path: {path_text!r}"
        )
    return _resolve_module_relative(path_text, MODULE_ROOT / Path(spec.run_relative_path))


def _resolve_module_relative(path_text: Any, expected: Path) -> Path:
    text = _require_text("relative artifact path", path_text, allow_empty=False)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.anchor:
        raise PhaseEntryBankError(f"artifact path must be a contained relative path: {text!r}")
    try:
        resolved = (MODULE_ROOT / path).resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise PhaseEntryBankError(f"artifact path does not resolve: {text!r}: {exc}") from exc
    _require_contained(resolved, MODULE_ROOT.resolve())
    if resolved != expected_resolved:
        raise PhaseEntryBankError(
            f"artifact path resolved outside its exact allowlisted identity: {text!r}"
        )
    return resolved


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PhaseEntryBankError(f"path is outside the phase-bank root: {path}") from exc


def _require_absolute_path(label: str, actual: Any, expected: Path, run_dir: Path) -> None:
    text = _require_text(label, actual, allow_empty=False)
    try:
        resolved = Path(text).resolve(strict=True)
    except OSError as exc:
        raise PhaseEntryBankError(f"{label} does not resolve: {exc}") from exc
    _require_contained(resolved, run_dir.resolve())
    if resolved != expected.resolve(strict=True):
        raise PhaseEntryBankError(f"{label} path identity mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PhaseEntryBankError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, PhaseEntryBankError) as exc:
        if isinstance(exc, PhaseEntryBankError):
            raise
        raise PhaseEntryBankError(f"cannot parse strict JSON {path}: {exc}") from exc
    return _require_mapping(str(path), value)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.rstrip("\r\n")
                if not line:
                    raise PhaseEntryBankError(f"blank JSONL row at {path}:{line_number}")
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_object_keys,
                    parse_constant=_reject_nonfinite_json_constant,
                )
                rows.append(_require_mapping(f"{path}:{line_number}", value))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEntryBankError(f"cannot parse strict JSONL {path}: {exc}") from exc
    return rows


def _reject_duplicate_object_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseEntryBankError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise PhaseEntryBankError(f"non-finite JSON constant {value} is forbidden")


def _require_mapping(label: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseEntryBankError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise PhaseEntryBankError(f"{label} keys must be strings")
    return value


def _require_list(label: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise PhaseEntryBankError(f"{label} must be an array")
    return value


def _require_exact_keys(label: str, value: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise PhaseEntryBankError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise PhaseEntryBankError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_text(label: str, value: Any, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise PhaseEntryBankError(f"{label} must be {'a nonempty' if not allow_empty else 'a'} string")
    return value


def _require_sha256(label: str, value: Any) -> str:
    text = _require_text(label, value, allow_empty=False)
    if len(text) != 64 or text.lower() != text or any(ch not in "0123456789abcdef" for ch in text):
        raise PhaseEntryBankError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _require_nonnegative_int(label: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhaseEntryBankError(f"{label} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(label: str, value: Any) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(label, value)


def _require_finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEntryBankError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PhaseEntryBankError(f"{label} must be finite")
    return number


def _require_vector(label: str, value: Any, length: int) -> list[float]:
    vector = _require_list(label, value)
    if len(vector) != length:
        raise PhaseEntryBankError(f"{label} must contain exactly {length} values")
    return [_require_finite_number(f"{label}[{index}]", item) for index, item in enumerate(vector)]


def _ordered_numeric_map(label: str, value: Any, names: Sequence[str]) -> list[float]:
    mapping = _require_mapping(label, value)
    if set(mapping) != set(names):
        raise PhaseEntryBankError(
            f"{label} must contain exact keys: missing={sorted(set(names) - set(mapping))}, "
            f"extra={sorted(set(mapping) - set(names))}"
        )
    return [_require_finite_number(f"{label}.{name}", mapping[name]) for name in names]


def _require_int_list(label: str, value: Any) -> list[int]:
    items = _require_list(label, value)
    return [_require_nonnegative_int(f"{label}[{index}]", item) for index, item in enumerate(items)]


def _require_text_list(label: str, value: Any) -> list[str]:
    items = _require_list(label, value)
    return [_require_text(f"{label}[{index}]", item, allow_empty=False) for index, item in enumerate(items)]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the sealed current r4 phase-entry bank")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="print the deterministic bank JSON; this module never writes artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.print_json:
        raise PhaseEntryBankError("--print-json is required; file creation is caller-owned")
    print(render_current_phase_entry_bank_json(), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the pure CLI test.
    raise SystemExit(main())


__all__ = [
    "BANK_ID",
    "CURRENT_SOURCE_VERSIONS",
    "CURSOR_POLICY",
    "DEFAULT_CONFIG_PATH",
    "PhaseEntryBank",
    "PhaseEntryBankError",
    "SCHEMA_VERSION",
    "SELECTION_POLICY",
    "TARGET_STATES",
    "build_current_phase_entry_bank_mapping",
    "canonical_sha256",
    "load_phase_entry_bank",
    "render_current_phase_entry_bank_json",
    "validate_phase_entry_bank_mapping",
]
