from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from fsm_50mm_recording_derived_v3.fsm50_direct_command_residual import (
    ResidualPhaseContract,
    ResidualTransformInput,
    compose_direct_command_residual,
)
from fsm_50mm_recording_derived_v3.fsm50_residual_envelope import (
    CANONICAL_ACTION_ORDER,
    DEFAULT_CONFIG_PATH,
    S5,
    S7,
    S8,
    S10,
    V003,
    V008,
    V009,
    V010_FAILED,
    ZERO_RESIDUAL,
    ResidualEnvelopeError,
    canonical_action_from_targets,
    canonical_payload_sha256,
    load_residual_envelope,
    phase_contract,
    split_canonical_action,
    validate_envelope_mapping,
)


EXPECTED_CANONICAL_SHA256 = (
    "fa5002690737d94fab7304f40044293da46de34cd197af19f3f1140047ae7fbe"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def envelope():
    return load_residual_envelope(verify_evidence=True)


def _reseal(payload: dict[str, object]) -> dict[str, object]:
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def _all_wheels_nominal(value: float = 0.3) -> tuple[float, ...]:
    return (0.0,) * 8 + (value,) * 4


def _target_maps(
    wheel_values: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3),
) -> tuple[dict[str, float], dict[str, float]]:
    servos = {name: float(index) for index, name in enumerate(CANONICAL_ACTION_ORDER[:8])}
    wheels = {
        name: float(value)
        for name, value in zip(CANONICAL_ACTION_ORDER[8:], wheel_values)
    }
    return servos, wheels


def test_default_config_canonical_sha_and_complete_evidence_chain(envelope) -> None:
    assert envelope.canonical_sha256 == EXPECTED_CANONICAL_SHA256
    assert envelope.evidence_verified is True
    assert envelope.source_versions == (V003, V008, V009)


def test_unverified_mapping_cannot_grant_runtime_authority() -> None:
    unverified = load_residual_envelope(verify_evidence=False)
    result = unverified.authorize(
        source_version=V009,
        state_id=S8,
        nominal_action=_all_wheels_nominal(),
        requested_residual=(1.0,) * 12,
        dt_s=1.0,
    )
    assert result.authority_granted is False
    assert result.authority_reason == "EVIDENCE_NOT_VERIFIED"
    assert result.applied_residual == ZERO_RESIDUAL


def test_canonical_hash_is_independent_of_object_key_order(
    payload: dict[str, object],
) -> None:
    reordered = dict(reversed(list(payload.items())))
    assert canonical_payload_sha256(payload) == EXPECTED_CANONICAL_SHA256
    assert canonical_payload_sha256(reordered) == EXPECTED_CANONICAL_SHA256


def test_canonical_target_map_round_trip() -> None:
    servos = {
        name: float(index + 1)
        for index, name in enumerate(CANONICAL_ACTION_ORDER[:8])
    }
    wheels = {
        name: -0.1 * float(index + 1)
        for index, name in enumerate(CANONICAL_ACTION_ORDER[8:])
    }
    action = canonical_action_from_targets(servos, wheels)
    assert action == pytest.approx(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -0.1, -0.2, -0.3, -0.4)
    )
    split_servos, split_wheels = split_canonical_action(action)
    assert split_servos == servos
    assert split_wheels == wheels


@pytest.mark.parametrize(
    ("source_version", "state_id", "expected"),
    [
        (V003, S5, ZERO_RESIDUAL),
        (V003, S7, ZERO_RESIDUAL),
        (V003, S8, (1, 1, 1, 0, 2, 2, 0, 0, 0.1, 0, 0, 0)),
        (V008, S5, (1, 1, 0, 1, 0, 1, 0, 1, 0, 0.1, 0, 0)),
        (V008, S7, ZERO_RESIDUAL),
        (V008, S8, (1, 1, 1, 1, 2, 2, 1, 1, 0.1, 0.1, 0.1, 0.1)),
        (V009, S5, (1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0)),
        (V009, S7, (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.1, 0)),
        (V009, S8, (1, 1, 1, 1, 2, 2, 1, 1, 0.1, 0.1, 0.1, 0.1)),
        (V003, S10, (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
        (V008, S10, (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
        (V009, S10, (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)),
    ],
)
def test_exact_source_conditioned_phase_masks(
    envelope,
    source_version: str,
    state_id: str,
    expected: tuple[float, ...],
) -> None:
    observed = envelope.maximum_abs_residual(
        source_version=source_version,
        state_id=state_id,
        nominal_action=_all_wheels_nominal(),
    )
    assert observed == tuple(float(value) for value in expected)


def test_wheel_residual_is_gated_by_each_nominal_wheel_target(envelope) -> None:
    nominal = (0.0,) * 8 + (0.0, 0.3, 1.0e-13, -0.3)
    bounds = envelope.maximum_abs_residual(
        source_version=V008,
        state_id=S8,
        nominal_action=nominal,
    )
    assert bounds[:8] == (1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0)
    assert bounds[8:] == (0.0, 0.1, 0.0, 0.1)


def test_phase_contract_adapts_validated_bounds_and_passes_context(envelope) -> None:
    servos, wheels = _target_maps((0.0, 0.3, 1.0e-13, -0.3))
    contract = phase_contract(
        envelope,
        source_version=V008,
        profile_strategy="reviewed-v008-strategy",
        macro_state=S8,
        subphase="segment_start:75",
        nominal_servo_targets_deg=servos,
        nominal_wheel_targets_rad_s=wheels,
    )
    assert isinstance(contract, ResidualPhaseContract)
    assert contract.source_version == V008
    assert contract.profile_strategy == "reviewed-v008-strategy"
    assert contract.macro_state == S8
    assert contract.subphase == "segment_start:75"
    assert contract.residual_max_command_units == (
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        1.0,
        1.0,
        0.0,
        0.1,
        0.0,
        0.1,
    )
    assert contract.residual_min_command_units == tuple(
        -value for value in contract.residual_max_command_units
    )
    assert contract.maximum_rate_command_units_per_s == (
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        0.0,
        0.5,
        0.0,
        0.5,
    )


def test_s10_never_authorizes_wheel_residual(envelope) -> None:
    for source_version in (V003, V008, V009):
        bounds = envelope.maximum_abs_residual(
            source_version=source_version,
            state_id=S10,
            nominal_action=_all_wheels_nominal(2.0),
        )
        assert bounds[8:] == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("source_version", "state_id", "provenance", "reason"),
    [
        (V010_FAILED, S8, "SOURCE_ACTION", "SOURCE_EXCLUDED"),
        ("v777_unknown", S8, "SOURCE_ACTION", "SOURCE_NOT_ALLOWLISTED"),
        (V008, "S9_FINAL_ADVANCE", "SOURCE_ACTION", "STATE_DEFAULT_ZERO"),
        (V008, S8, "BOUNDARY", "DECISION_PROVENANCE_DEFAULT_ZERO"),
        (V008, S8, "HOLD", "DECISION_PROVENANCE_DEFAULT_ZERO"),
        (V008, S8, "RETRY", "DECISION_PROVENANCE_DEFAULT_ZERO"),
        (V008, S8, "SAFE_STOP", "DECISION_PROVENANCE_DEFAULT_ZERO"),
        (V008, S8, "SUCCESS", "DECISION_PROVENANCE_DEFAULT_ZERO"),
    ],
)
def test_default_zero_is_immediate_and_cannot_leak_previous_residual(
    envelope,
    source_version: str,
    state_id: str,
    provenance: str,
    reason: str,
) -> None:
    nominal = tuple(float(index) for index in range(12))
    result = envelope.authorize(
        source_version=source_version,
        state_id=state_id,
        decision_provenance=provenance,
        nominal_action=nominal,
        requested_residual=(4.0,) * 12,
        previous_residual=(0.5,) * 12,
        dt_s=1.0 / 120.0,
    )
    assert result.authority_granted is False
    assert result.authority_reason == reason
    assert result.maximum_abs_residual == ZERO_RESIDUAL
    assert result.applied_residual == ZERO_RESIDUAL
    assert result.clipped is True
    assert result.slew_limited is False


def test_clip_then_slew_limit_respects_mixed_units(envelope) -> None:
    nominal = _all_wheels_nominal()
    first = envelope.authorize(
        source_version=V008,
        state_id=S8,
        nominal_action=nominal,
        requested_residual=(9.0,) * 12,
        dt_s=0.1,
    )
    assert first.maximum_abs_residual == (
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        1.0,
        1.0,
        0.1,
        0.1,
        0.1,
        0.1,
    )
    assert first.applied_residual == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.05,
        0.05,
        0.05,
        0.05,
    )
    assert first.clipped is True
    assert first.slew_limited is True

    second = envelope.authorize(
        source_version=V008,
        state_id=S8,
        nominal_action=nominal,
        requested_residual=(9.0,) * 12,
        previous_residual=first.applied_residual,
        dt_s=1.0,
    )
    assert second.applied_residual == first.maximum_abs_residual
    assert second.clipped is True
    assert second.slew_limited is False


def test_authorize_is_offline_preview_not_a_command_composer(envelope) -> None:
    nominal = tuple((-1.0) ** index * (index + 0.125) for index in range(12))
    result = envelope.authorize(
        source_version=V009,
        state_id=S8,
        nominal_action=nominal,
        requested_residual=ZERO_RESIDUAL,
        dt_s=1.0 / 120.0,
    )
    assert result.authority_granted is True
    assert result.applied_residual == ZERO_RESIDUAL
    assert result.clipped is False
    assert result.slew_limited is False
    assert not hasattr(result, "command_targets")
    assert "command_targets" not in result.to_mapping()


def test_phase_contract_uses_sole_composer_for_exact_zero_identity(envelope) -> None:
    servos, wheels = _target_maps()
    contract = envelope.phase_contract(
        source_version=V009,
        profile_strategy="reviewed-v009-strategy",
        macro_state=S8,
        subphase="segment_start:93",
        nominal_servo_targets_deg=servos,
        nominal_wheel_targets_rad_s=wheels,
    )
    transform_input = ResidualTransformInput(
        source_version=V009,
        profile_strategy="reviewed-v009-strategy",
        macro_state=S8,
        subphase="segment_start:93",
        nominal_servo_targets_deg=servos,
        nominal_wheel_targets_rad_s=wheels,
        normalized_action=ZERO_RESIDUAL,
        previous_applied_residual=(0.5,) * 12,
        decision_dt_s=1.0 / 120.0,
        maximum_wheel_speed_rad_s=3.0,
    )
    result = compose_direct_command_residual(transform_input, contract)
    assert result.zero_identity is True
    assert result.applied_residual == ZERO_RESIDUAL
    assert result.applied_servo_targets_deg == servos
    assert result.applied_wheel_targets_rad_s == wheels


def test_checkpoint_reference_is_hard_rejected(envelope) -> None:
    with pytest.raises(ResidualEnvelopeError, match="checkpoint-derived action truth"):
        envelope.authorize(
            source_version=V009,
            state_id=S8,
            nominal_action=_all_wheels_nominal(),
            requested_residual=ZERO_RESIDUAL,
            dt_s=1.0 / 120.0,
            checkpoint_reference="old_failed_policy.pt",
        )


def test_unsealed_change_is_rejected(
    payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(payload)
    changed["global_limits"]["wheel_max_abs_rad_s"] = 0.2  # type: ignore[index]
    with pytest.raises(ResidualEnvelopeError, match="canonical payload SHA256 mismatch"):
        validate_envelope_mapping(changed, verify_evidence=False)


def test_resealed_authority_expansion_is_still_rejected(
    payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(payload)
    source = changed["source_allowlist"][V003]  # type: ignore[index]
    source["state_max_abs_residual"][S5][0] = 1.0  # type: ignore[index]
    _reseal(changed)
    with pytest.raises(ResidualEnvelopeError, match="reviewed R1 authority"):
        validate_envelope_mapping(changed, verify_evidence=False)


def test_resealed_v010_or_checkpoint_admission_is_rejected(
    payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(payload)
    changed["hard_exclusions"]["legacy_checkpoints_permitted"] = True  # type: ignore[index]
    _reseal(changed)
    with pytest.raises(ResidualEnvelopeError, match="hard exclusions"):
        validate_envelope_mapping(changed, verify_evidence=False)

    changed = copy.deepcopy(payload)
    changed["source_allowlist"][V010_FAILED] = copy.deepcopy(  # type: ignore[index]
        changed["source_allowlist"][V009]  # type: ignore[index]
    )
    _reseal(changed)
    with pytest.raises(ResidualEnvelopeError, match="source allowlist"):
        validate_envelope_mapping(changed, verify_evidence=False)


def test_resealed_evidence_path_or_hash_change_is_rejected(
    payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(payload)
    evidence = changed["source_allowlist"][V008]["evidence"]  # type: ignore[index]
    evidence["accepted_steps"]["sha256"] = "a" * 64  # type: ignore[index]
    _reseal(changed)
    with pytest.raises(ResidualEnvelopeError, match="evidence binding mismatch"):
        validate_envelope_mapping(changed, verify_evidence=False)


def test_extra_checkpoint_field_is_not_part_of_schema(
    payload: dict[str, object],
) -> None:
    changed = copy.deepcopy(payload)
    changed["checkpoint_path"] = "old.pt"
    _reseal(changed)
    with pytest.raises(ResidualEnvelopeError, match="unexpected"):
        validate_envelope_mapping(changed, verify_evidence=False)


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"a","schema_version":"b"}\n', encoding="utf-8"
    )
    with pytest.raises(ResidualEnvelopeError, match="duplicate JSON key"):
        load_residual_envelope(path, verify_evidence=False)


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, math.inf, math.nan, "bad"])
def test_authorization_rejects_invalid_dt(envelope, bad_dt: object) -> None:
    with pytest.raises(ResidualEnvelopeError, match="dt_s"):
        envelope.authorize(
            source_version=V009,
            state_id=S8,
            nominal_action=_all_wheels_nominal(),
            requested_residual=ZERO_RESIDUAL,
            dt_s=bad_dt,  # type: ignore[arg-type]
        )


def test_target_maps_must_be_complete() -> None:
    with pytest.raises(ResidualEnvelopeError, match="servo target keys"):
        canonical_action_from_targets({}, {})
