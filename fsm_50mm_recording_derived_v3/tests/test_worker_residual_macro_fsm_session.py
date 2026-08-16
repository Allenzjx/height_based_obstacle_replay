from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.fsm50_direct_command_residual import (
    RESIDUAL_ACTION_DIM,
    ResidualPhaseContract,
    ZeroResidualPolicy,
)
from fsm_50mm_recording_derived_v3.worker_residual_macro_fsm_session import (
    ENVIRONMENT_LOCK_PATH,
    EXECUTION_MODE,
    EXPECTED_BUNDLE_SHA256,
    EXPECTED_ENVELOPE_CANONICAL_SHA256,
    EXPECTED_GRAPH_ID,
    EXPECTED_GRAPH_SHA256,
    EXPECTED_PROFILE_ID,
    EXPECTED_PROFILE_LIBRARY_SHA256,
    GATE_E_COMMAND_EVIDENCE_SCHEMA,
    GATE_E_IDENTITY_SCHEMA,
    GATE_E_TASK_INPUTS_NAME,
    GATE_E_TASK_INPUTS_SCHEMA,
    GATE_E_WORKER_RESULT_NAME,
    GATE_E_WORKER_RESULT_SCHEMA,
    OPERATION,
    POLICY_KIND,
    REQUEST_SCHEMA,
    START_SCHEMA,
    BaseResidualAPIUnavailable,
    GateEResidualMacroFSMSession,
    GateEResidualMacroFSMSessionFactory,
    ResidualWorkerRequestError,
    VerifiedPhaseContractProvider,
    base_residual_api_available,
    build_worker_residual_macro_fsm_request,
    current_code_sha256,
    gate_e_identity_for_payload,
    load_worker_residual_macro_fsm_request,
    main,
    request_identity_sha256,
    sha256_file,
    validate_request_mode_exclusivity,
    validate_worker_residual_start_binding,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
BASE_REQUEST_PATHS = {
    "v003_20260805_224517_157723_manual": (
        MODULE_ROOT
        / "runs"
        / "v003_macro_fsm_completion_aware_coalesced_r4"
        / "v003_20260805_224517_157723_manual"
        / "baseline"
        / "20260815T105857_601132Z_baseline_00_5742cda6e438"
        / "worker_macro_fsm_request.json"
    ),
    "v008_20260806_211408_578700_manual": (
        MODULE_ROOT
        / "runs"
        / "cross_version_macro_fsm_completion_aware_coalesced_r4"
        / "v008_20260806_211408_578700_manual"
        / "trials"
        / "20260815T113449_254770Z_cross_version_00_0a47cee05cbb"
        / "worker_macro_fsm_request.json"
    ),
    "v009_20260806_215232_433234_manual": (
        MODULE_ROOT
        / "runs"
        / "cross_version_macro_fsm_completion_aware_coalesced_r4"
        / "v009_20260806_215232_433234_manual"
        / "trials"
        / "20260815T114309_706871Z_cross_version_00_0e5686354eed"
        / "worker_macro_fsm_request.json"
    ),
}


def _write_payload(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _reseal(payload: dict) -> dict:
    payload["request_identity_sha256"] = request_identity_sha256(payload)
    return payload


def _current_environment_verification() -> dict:
    return {
        "environment_lock_path": str(ENVIRONMENT_LOCK_PATH.resolve()),
        "environment_lock_sha256": sha256_file(ENVIRONMENT_LOCK_PATH),
        "locked_source_file_count": 1,
        "required_source_file_count": 1,
        "source_closure_complete": True,
        "source_verification_sha256": "f" * 64,
    }


def _load_request(path: Path | str | None):
    return load_worker_residual_macro_fsm_request(
        path,
        environment_validator=_current_environment_verification,
    )


@pytest.fixture(scope="module")
def valid_payload() -> dict:
    return build_worker_residual_macro_fsm_request(
        base_request_path=BASE_REQUEST_PATHS[
            "v003_20260805_224517_157723_manual"
        ],
        request_id="gate-e-r0-zero-unit",
        environment_validator=_current_environment_verification,
    )


@pytest.fixture()
def loaded_request(tmp_path: Path, valid_payload: dict):
    path = _write_payload(tmp_path / "gate_e_request.json", valid_payload)
    request = _load_request(path)
    assert request is not None
    return request


@pytest.mark.parametrize(
    "source_version,base_request_path", tuple(BASE_REQUEST_PATHS.items())
)
def test_builder_and_loader_admit_only_each_reviewed_source(
    tmp_path: Path,
    source_version: str,
    base_request_path: Path,
):
    payload = build_worker_residual_macro_fsm_request(
        base_request_path=base_request_path,
        request_id=f"gate-e-r0-{source_version[:4]}",
        environment_validator=_current_environment_verification,
    )
    request = _load_request(
        _write_payload(tmp_path / "request.json", payload)
    )

    assert request is not None
    assert request.source_version == source_version
    assert request.bundle_sha256 == EXPECTED_BUNDLE_SHA256[source_version]
    assert request.policy_kind == POLICY_KIND
    assert request.envelope.evidence_verified is True
    assert request.envelope.is_source_allowed(source_version)


def test_request_binds_all_gate_e_identities(valid_payload: dict):
    policy = ZeroResidualPolicy()
    code_sha256 = current_code_sha256()

    assert valid_payload["schema_version"] == REQUEST_SCHEMA
    assert valid_payload["enabled"] is True
    assert valid_payload["execution_mode"] == EXECUTION_MODE
    assert valid_payload["operation"] == OPERATION
    assert valid_payload["profile_id"] == EXPECTED_PROFILE_ID
    assert valid_payload["graph_id"] == EXPECTED_GRAPH_ID
    assert valid_payload["graph_sha256"] == EXPECTED_GRAPH_SHA256
    assert (
        valid_payload["profile_library_sha256"]
        == EXPECTED_PROFILE_LIBRARY_SHA256
    )
    assert (
        valid_payload["envelope_canonical_sha256"]
        == EXPECTED_ENVELOPE_CANONICAL_SHA256
    )
    assert valid_payload["environment_lock_sha256"] == sha256_file(
        ENVIRONMENT_LOCK_PATH
    )
    assert valid_payload["policy_kind"] == "ZERO"
    assert valid_payload["policy_id"] == policy.policy_id
    assert valid_payload["policy_sha256"] == policy.policy_sha256
    assert valid_payload["code_sha256"] == code_sha256
    assert (
        valid_payload["residual_core_sha256"]
        == code_sha256["fsm50_direct_command_residual.py"]
    )
    assert valid_payload["request_identity_sha256"] == request_identity_sha256(
        valid_payload
    )
    assert "checkpoint" not in " ".join(valid_payload).lower()
    assert "action" not in valid_payload


def test_request_identity_is_canonical_and_order_independent(valid_payload: dict):
    reverse_order = dict(reversed(tuple(valid_payload.items())))
    assert request_identity_sha256(reverse_order) == request_identity_sha256(
        valid_payload
    )


@pytest.mark.parametrize(
    "extra,value",
    (
        ("checkpoint_path", "old.pt"),
        ("normalized_action", [0.0] * RESIDUAL_ACTION_DIM),
        ("policy_config", {}),
        ("ppo_training_result", {"success": True}),
    ),
)
def test_exact_schema_rejects_checkpoint_action_and_unknown_extras(
    tmp_path: Path,
    valid_payload: dict,
    extra: str,
    value,
):
    payload = copy.deepcopy(valid_payload)
    payload[extra] = value
    _reseal(payload)

    with pytest.raises(ResidualWorkerRequestError, match="keys mismatch"):
        _load_request(
            _write_payload(tmp_path / f"extra_{extra}.json", payload)
        )


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("policy_kind", "PPO", "only permits policy_kind=ZERO"),
        ("policy_id", "fsm50.some_checkpoint.v1", "policy_id mismatch"),
        ("policy_sha256", "0" * 64, "policy SHA256 mismatch"),
        ("residual_core_sha256", "0" * 64, "core SHA256 mismatch"),
        ("graph_sha256", "0" * 64, "graph SHA256 mismatch"),
        ("profile_library_sha256", "0" * 64, "library SHA256 mismatch"),
        ("bundle_sha256", "0" * 64, "bundle SHA256 mismatch"),
        (
            "envelope_canonical_sha256",
            "0" * 64,
            "envelope is not reviewed/current",
        ),
        (
            "environment_lock_sha256",
            "0" * 64,
            "differs from current validated bytes",
        ),
        ("base_request_sha256", "0" * 64, "base request SHA256 mismatch"),
    ),
)
def test_loader_rejects_resealed_identity_tampering(
    tmp_path: Path,
    valid_payload: dict,
    field: str,
    value,
    match: str,
):
    payload = copy.deepcopy(valid_payload)
    payload[field] = value
    _reseal(payload)

    with pytest.raises(ResidualWorkerRequestError, match=match):
        _load_request(
            _write_payload(tmp_path / f"tampered_{field}.json", payload)
        )


def test_loader_rejects_unreviewed_v010_even_when_resealed(
    tmp_path: Path, valid_payload: dict
):
    payload = copy.deepcopy(valid_payload)
    payload["source_version"] = "v010_20260806_220745_363972_manual"
    payload["bundle_sha256"] = (
        "246f073b36adda9e23ab325133c7aafea6ccd9cfaa983b33e83af011e071e396"
    )
    _reseal(payload)

    with pytest.raises(ResidualWorkerRequestError, match="reviewed v003/v008/v009"):
        _load_request(
            _write_payload(tmp_path / "v010.json", payload)
        )


def test_loader_rejects_current_code_binding_tamper(
    tmp_path: Path, valid_payload: dict
):
    payload = copy.deepcopy(valid_payload)
    payload["code_sha256"]["worker_macro_fsm_session.py"] = "0" * 64
    _reseal(payload)

    with pytest.raises(ResidualWorkerRequestError, match="code SHA256 mismatch"):
        _load_request(
            _write_payload(tmp_path / "code_tamper.json", payload)
        )


def test_loader_rejects_code_binding_key_drift(
    tmp_path: Path, valid_payload: dict
):
    payload = copy.deepcopy(valid_payload)
    payload["code_sha256"].pop("fsm50_macro_controller.py")
    _reseal(payload)

    with pytest.raises(ResidualWorkerRequestError, match="code SHA256 keys mismatch"):
        _load_request(
            _write_payload(tmp_path / "code_keys.json", payload)
        )


def test_loader_rejects_request_identity_tamper(
    tmp_path: Path, valid_payload: dict
):
    payload = copy.deepcopy(valid_payload)
    payload["request_id"] = "tampered-without-reseal"

    with pytest.raises(ResidualWorkerRequestError, match="canonical identity"):
        _load_request(
            _write_payload(tmp_path / "identity_tamper.json", payload)
        )


def test_loader_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path, valid_payload: dict
):
    exact = json.dumps(valid_payload, sort_keys=True)
    duplicate = exact.replace(
        '"enabled": true,', '"enabled": true, "enabled": true,', 1
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ResidualWorkerRequestError, match="duplicate JSON key"):
        _load_request(duplicate_path)

    nonfinite = exact.replace('"enabled": true', '"enabled": NaN', 1)
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ResidualWorkerRequestError, match="non-finite JSON"):
        _load_request(nonfinite_path)


@pytest.mark.parametrize(
    "residual,task,macro",
    (
        ("gate_e.json", "task.json", None),
        ("gate_e.json", None, "macro.json"),
        ("gate_e.json", "task.json", "macro.json"),
    ),
)
def test_gate_e_request_is_mutually_exclusive_with_old_routes(
    residual, task, macro
):
    with pytest.raises(ResidualWorkerRequestError, match="mutually exclusive"):
        validate_request_mode_exclusivity(
            residual_request_path=residual,
            task_request_path=task,
            macro_request_path=macro,
        )


def test_mode_exclusivity_accepts_gate_e_alone():
    validate_request_mode_exclusivity(
        residual_request_path="gate_e.json",
        task_request_path=None,
        macro_request_path=None,
    )


def _provider_context(request, *, provenance_kind="SOURCE_ACTION"):
    return {
        "schema_version": "fsm50.worker_residual_observation.v1",
        "source_version": request.source_version,
        "profile_id": "fsm50-unit-source-profile",
        "profile_source_version": request.source_version,
        "profile_strategy": "recording_exact",
        "macro_state": "S8_RL_COM_SHIFT_AND_TRAVERSE",
        "subphase": "segment_start",
        "outer_render_boundary_permit": True,
        "nominal_servo_targets_deg": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "nominal_wheel_targets_rad_s": {
            name: 1.0 for name in WHEEL_JOINT_NAMES
        },
        "command_provenance": {"kind": provenance_kind},
    }


def test_verified_provider_returns_phase_contract_and_preserves_subphase(
    loaded_request,
):
    provider = VerifiedPhaseContractProvider(loaded_request)
    context = _provider_context(loaded_request)

    contract = provider(context)

    assert isinstance(contract, ResidualPhaseContract)
    assert contract.source_version == loaded_request.source_version
    assert contract.profile_strategy == context["profile_strategy"]
    assert contract.macro_state == context["macro_state"]
    assert contract.subphase == context["subphase"]
    assert len(contract.enabled_mask) == RESIDUAL_ACTION_DIM
    assert any(contract.enabled_mask)


def test_verified_provider_defaults_non_source_provenance_to_zero(loaded_request):
    context = _provider_context(
        loaded_request, provenance_kind="BOUNDARY_ZERO_WHEELS"
    )
    context.update(profile_id="", profile_source_version="", profile_strategy="")
    contract = VerifiedPhaseContractProvider(loaded_request)(context)
    assert contract.enabled_mask == (False,) * RESIDUAL_ACTION_DIM
    assert contract.residual_max_command_units == (0.0,) * RESIDUAL_ACTION_DIM
    assert contract.profile_strategy == "NO_ACTIVE_PROFILE"
    assert contract.subphase == context["subphase"]


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda value: value.update(schema_version="wrong"), "schema mismatch"),
        (lambda value: value.update(source_version="v009"), "source differs"),
        (lambda value: value.update(profile_id=3), "profile_id must be text"),
        (
            lambda value: value.update(profile_source_version="wrong"),
            "profile source differs",
        ),
        (
            lambda value: value.update(outer_render_boundary_permit=False),
            "outer render boundary permit",
        ),
        (
            lambda value: value.update(profile_id=""),
            "lacks its bound source profile",
        ),
    ),
)
def test_verified_provider_rejects_unbound_context(
    loaded_request, mutation, match
):
    context = _provider_context(loaded_request)
    mutation(context)
    with pytest.raises(ResidualWorkerRequestError, match=match):
        VerifiedPhaseContractProvider(loaded_request)(context)


def test_factory_supplies_exact_zero_policy_and_verified_provider(loaded_request):
    assert base_residual_api_available() is True
    factory = GateEResidualMacroFSMSessionFactory(loaded_request)

    session = factory.build_base_session(worker_session_id="gate-e-worker-unit")

    assert isinstance(session.residual_policy, ZeroResidualPolicy)
    assert session.residual_policy.act({}) == (0.0,) * RESIDUAL_ACTION_DIM
    assert isinstance(
        session.residual_contract_provider, VerifiedPhaseContractProvider
    )
    assert session.residual_enabled is True
    assert session.request is loaded_request.base_request
    assert session.worker_session_id == "gate-e-worker-unit"


def test_factory_rejects_policy_or_provider_override(loaded_request):
    factory = GateEResidualMacroFSMSessionFactory(loaded_request)
    with pytest.raises(ResidualWorkerRequestError, match="cannot override"):
        factory.build_base_session(
            worker_session_id="unit", residual_policy=ZeroResidualPolicy()
        )


def test_factory_fails_closed_when_base_api_is_absent(monkeypatch, loaded_request):
    import fsm_50mm_recording_derived_v3.worker_residual_macro_fsm_session as module

    monkeypatch.setattr(module, "base_residual_api_available", lambda: False)
    with pytest.raises(BaseResidualAPIUnavailable, match="API is not frozen"):
        GateEResidualMacroFSMSessionFactory(loaded_request).build_base_session(
            worker_session_id="unit"
        )


def _temp_run_request(tmp_path: Path):
    base_payload = json.loads(
        BASE_REQUEST_PATHS[
            "v003_20260805_224517_157723_manual"
        ].read_text(encoding="utf-8")
    )
    base_payload["run_dir"] = str((tmp_path / "run").resolve())
    base_path = _write_payload(tmp_path / "base_request.json", base_payload)
    outer_payload = build_worker_residual_macro_fsm_request(
        base_request_path=base_path,
        request_id="gate-e-temp-run",
        environment_validator=_current_environment_verification,
    )
    outer_path = _write_payload(tmp_path / "gate_e_request.json", outer_payload)
    request = _load_request(outer_path)
    assert request is not None
    return request


def test_environment_lock_is_dynamic_and_content_validator_is_fail_closed(
    tmp_path: Path, valid_payload: dict
):
    production_source = (
        MODULE_ROOT / "worker_residual_macro_fsm_session.py"
    ).read_text(encoding="utf-8")
    assert "ENVIRONMENT_LOCK_SHA256" not in production_source
    assert valid_payload["environment_lock_sha256"] == sha256_file(
        ENVIRONMENT_LOCK_PATH
    )
    path = _write_payload(tmp_path / "request.json", valid_payload)

    def failed_validator():
        raise ValueError("source closure stale")

    with pytest.raises(ResidualWorkerRequestError, match="source closure stale"):
        load_worker_residual_macro_fsm_request(
            path,
            environment_validator=failed_validator,
        )


def test_wrapper_revalidates_environment_immediately_before_base_start(
    tmp_path: Path,
):
    request = _temp_run_request(tmp_path)

    class _StartBase:
        def __init__(self):
            self.request = request.base_request
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            return {
                "accepted": True,
                "request_id": self.request.request_id,
                "source_version": self.request.source_version,
                "profile_id": self.request.profile_id,
            }

    current_base = _StartBase()
    current = GateEResidualMacroFSMSession(
        request,
        current_base,
        environment_validator=_current_environment_verification,
    )
    start = current.start()
    assert current_base.start_calls == 1
    assert start["request_id"] == request.request_id
    assert start["residual_macro_fsm_request_id"] == request.request_id
    assert start["base_macro_fsm_request_id"] == request.base_request.request_id
    assert start["gate_e_zero_residual"]["payload_role"] == "start_ack"
    assert start["gate_e_zero_residual"]["request_file_sha256"] == (
        request.request_file_sha256
    )

    changed_verification = _current_environment_verification()
    changed_verification["source_verification_sha256"] = "e" * 64
    stale_base = _StartBase()
    stale = GateEResidualMacroFSMSession(
        request,
        stale_base,
        environment_validator=lambda: changed_verification,
    )
    with pytest.raises(ResidualWorkerRequestError, match="changed before session"):
        stale.start()
    assert stale_base.start_calls == 0


def _base_terminal_artifacts(request, *, nonzero: bool = False):
    run_dir = request.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    zero = [0.0] * RESIDUAL_ACTION_DIM
    raw = list(zero)
    if nonzero:
        raw[0] = 0.25
    transform = {
        "raw_normalized_action": raw,
        "clipped_normalized_action": list(raw),
        "applied_residual": list(raw),
        "zero_identity": not nonzero,
    }
    dispatch_path = run_dir / "macro_dispatch_ledger.jsonl"
    dispatch_path.write_text(
        json.dumps(
            {
                "residual_policy_id": request.policy_id,
                "residual_policy_sha256": request.policy_sha256,
                "residual_transform": transform,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = {
        "schema_version": "fsm50.macro_task_inputs.v1",
        "completed_result": {
            "source_version": request.source_version,
            "direct_command_residual": {
                "enabled": True,
                "policy_id": request.policy_id,
                "policy_sha256": request.policy_sha256,
            },
        },
        "physical_evidence": {},
        "final_telemetry_row": {},
    }
    inputs_path = run_dir / "macro_task_inputs.json"
    _write_payload(inputs_path, inputs)
    direct = {
        "enabled": True,
        "policy_id": request.policy_id,
        "policy_sha256": request.policy_sha256,
        "transform_count": 1,
        "physical_command_epoch": 1,
        "last_verified_physical_command_epoch": 1,
        "last_transform_sha256": "a" * 64,
        "last_transform": transform,
    }
    result = {
        "schema_version": "fsm50.worker_macro_fsm_session.v1",
        "request_id": request.base_request.request_id,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "dispatch_ledger_path": str(dispatch_path),
        "direct_command_residual": direct,
    }
    result_path = run_dir / "worker_macro_fsm_result.json"
    _write_payload(result_path, result)
    terminal = {
        "type": "macro_fsm_complete",
        "operation": "macro_fsm",
        "request_id": request.base_request.request_id,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "task_inputs_path": str(inputs_path),
        "worker_result_path": str(result_path),
        "task_inputs": inputs,
        "error": "",
    }
    return terminal, inputs_path, result_path


def test_wrapper_writes_atomic_outer_terminal_result_and_task_inputs(tmp_path: Path):
    request = _temp_run_request(tmp_path)
    terminal, base_inputs_path, base_result_path = _base_terminal_artifacts(request)
    base_inputs_sha = sha256_file(base_inputs_path)
    base_result_sha = sha256_file(base_result_path)
    base = SimpleNamespace(request=request.base_request)
    wrapper = GateEResidualMacroFSMSession(request, base)

    outer_terminal = wrapper._wrap_terminal(terminal)

    assert sha256_file(base_inputs_path) == base_inputs_sha
    assert sha256_file(base_result_path) == base_result_sha
    assert outer_terminal["type"] == "macro_fsm_complete"
    assert outer_terminal["operation"] == OPERATION
    assert outer_terminal["request_id"] == request.request_id
    assert outer_terminal["residual_macro_fsm_request_id"] == request.request_id
    assert outer_terminal["base_macro_fsm_request_id"] == (
        request.base_request.request_id
    )
    assert outer_terminal["gate_e_zero_residual"]["payload_role"] == "terminal"
    assert outer_terminal["residual_command_evidence"]["schema_version"] == (
        GATE_E_COMMAND_EVIDENCE_SCHEMA
    )
    assert outer_terminal["residual_command_evidence"][
        "all_durable_residual_dispatches_zero_identity"
    ] is True
    assert outer_terminal["residual_command_evidence"][
        "ppo_training_performed"
    ] is False
    assert outer_terminal["residual_command_evidence"]["checkpoint_loaded"] is False

    outer_inputs_path = request.run_dir / GATE_E_TASK_INPUTS_NAME
    outer_result_path = request.run_dir / GATE_E_WORKER_RESULT_NAME
    outer_inputs = json.loads(outer_inputs_path.read_text(encoding="utf-8"))
    outer_result = json.loads(outer_result_path.read_text(encoding="utf-8"))
    assert outer_inputs["schema_version"] == GATE_E_TASK_INPUTS_SCHEMA
    assert outer_result["schema_version"] == GATE_E_WORKER_RESULT_SCHEMA
    assert outer_inputs["base_task_inputs_sha256"] == base_inputs_sha
    assert outer_result["base_worker_result_sha256"] == base_result_sha
    assert outer_result["task_inputs_sha256"] == sha256_file(outer_inputs_path)
    assert outer_terminal["worker_result_sha256"] == sha256_file(outer_result_path)
    assert outer_result["gate_e_zero_residual"]["policy_kind"] == "ZERO"


def test_wrapper_rejects_nonzero_base_residual_evidence(tmp_path: Path):
    request = _temp_run_request(tmp_path)
    terminal, _inputs_path, _result_path = _base_terminal_artifacts(
        request, nonzero=True
    )
    wrapper = GateEResidualMacroFSMSession(
        request, SimpleNamespace(request=request.base_request)
    )
    with pytest.raises(ResidualWorkerRequestError, match="not exact ZERO"):
        wrapper._wrap_terminal(terminal)
    assert not (request.run_dir / GATE_E_TASK_INPUTS_NAME).exists()
    assert not (request.run_dir / GATE_E_WORKER_RESULT_NAME).exists()


@pytest.mark.parametrize("role", ("terminal", "worker_result", "task_inputs"))
def test_gate_e_identity_helper_adds_honest_copy(role, loaded_request):
    original = {"schema_version": f"test.{role}.v1", "success": True}
    result = gate_e_identity_for_payload(
        original, loaded_request, payload_role=role
    )
    identity = result["gate_e_zero_residual"]

    assert result is not original
    assert "gate_e_zero_residual" not in original
    assert identity["schema_version"] == GATE_E_IDENTITY_SCHEMA
    assert identity["payload_role"] == role
    assert identity["policy_kind"] == "ZERO"
    assert identity["runtime_policy_authority"] == "EXACT_ZERO_ONLY"
    assert identity["ppo_training_performed"] is False
    assert identity["checkpoint_loaded"] is False
    assert identity["request_identity_sha256"] == (
        loaded_request.request_identity_sha256
    )
    assert identity["residual_core_sha256"] == (
        loaded_request.residual_core_sha256
    )


def test_gate_e_identity_helper_rejects_duplicate_or_unknown_role(loaded_request):
    with pytest.raises(ResidualWorkerRequestError, match="already present"):
        gate_e_identity_for_payload(
            {"gate_e_zero_residual": {}},
            loaded_request,
            payload_role="terminal",
        )
    with pytest.raises(ResidualWorkerRequestError, match="unsupported"):
        gate_e_identity_for_payload({}, loaded_request, payload_role="training")


def _start_message(request, worker_session_id="worker-unit"):
    return {
        "schema_version": START_SCHEMA,
        "type": "start_macro_fsm",
        "operation": OPERATION,
        "request_id": request.request_id,
        "request_identity_sha256": request.request_identity_sha256,
        "worker_session_id": worker_session_id,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "policy_kind": request.policy_kind,
        "policy_sha256": request.policy_sha256,
        "residual_core_sha256": request.residual_core_sha256,
        "envelope_canonical_sha256": request.envelope_canonical_sha256,
        "enqueued_wall_time": 1.0,
    }


def test_start_binding_is_exact_and_source_bound(loaded_request):
    message = _start_message(loaded_request)
    assert validate_worker_residual_start_binding(
        loaded_request,
        message,
        expected_worker_session_id="worker-unit",
    ) == []

    tampered = dict(message, source_version="v009")
    assert validate_worker_residual_start_binding(
        loaded_request,
        tampered,
        expected_worker_session_id="worker-unit",
    ) == ["Gate-E residual start source_version does not match the request"]

    extra = dict(message, checkpoint_path="forbidden.pt")
    errors = validate_worker_residual_start_binding(
        loaded_request,
        extra,
        expected_worker_session_id="worker-unit",
    )
    assert len(errors) == 1
    assert "unexpected=['checkpoint_path']" in errors[0]


def test_preflight_cli_validates_without_dispatch(
    tmp_path: Path, valid_payload: dict, capsys, monkeypatch
):
    import fsm_50mm_recording_derived_v3.worker_residual_macro_fsm_session as module

    monkeypatch.setattr(
        module,
        "validate_current_environment_lock",
        _current_environment_verification,
    )
    path = _write_payload(tmp_path / "request.json", valid_payload)

    assert main(["--request", str(path), "--preflight-only"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["preflight_ok"] is True
    assert output["policy_kind"] == "ZERO"
    assert output["ppo_training_claimed"] is False
    assert output["base_residual_api_available"] is True


def test_cli_without_dedicated_process_route_fails_closed(
    tmp_path: Path, valid_payload: dict, monkeypatch
):
    import fsm_50mm_recording_derived_v3.worker_residual_macro_fsm_session as module

    monkeypatch.setattr(
        module,
        "validate_current_environment_lock",
        _current_environment_verification,
    )
    path = _write_payload(tmp_path / "request.json", valid_payload)
    with pytest.raises(BaseResidualAPIUnavailable, match="standalone validation"):
        main(["--request", str(path)])


def test_empty_request_path_remains_disabled():
    assert _load_request(None) is None
    assert _load_request("") is None
