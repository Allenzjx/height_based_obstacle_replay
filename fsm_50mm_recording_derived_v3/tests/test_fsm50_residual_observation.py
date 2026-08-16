from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.fsm50_residual_observation import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_FIELD_SLICES,
    ACTOR_OBSERVATION_SCHEMA_SHA256,
    AUTHORIZED_SOURCE_VERSIONS,
    ResidualObservationError,
    build_residual_actor_observation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V003_RUN = (
    PROJECT_ROOT
    / "fsm_50mm_recording_derived_v3"
    / "runs"
    / "v003_macro_fsm_completion_aware_coalesced_r4"
    / "v003_20260805_224517_157723_manual"
    / "baseline"
    / "20260815T105857_601132Z_baseline_00_5742cda6e438"
)


def _sample() -> dict:
    path = V003_RUN / "minimal_macro_telemetry.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("macro_state") == "S8_RL_COM_SHIFT_AND_TRAVERSE" and row.get("profile_fraction") is not None:
            return row
    raise AssertionError("missing usable S8 telemetry row")


def test_actor_observation_schema_is_stable_and_complete() -> None:
    assert ACTOR_OBSERVATION_DIM == 115
    assert len(ACTOR_OBSERVATION_SCHEMA_SHA256) == 64
    assert ACTOR_OBSERVATION_FIELD_SLICES["macro_state_one_hot"] == (0, 11)
    assert ACTOR_OBSERVATION_FIELD_SLICES["body_crossed_front_face"] == (114, 115)


def test_real_reviewed_telemetry_builds_finite_deployable_vector() -> None:
    row = _sample()
    observation = build_residual_actor_observation(row, [0.0] * 12)
    assert len(observation.values) == ACTOR_OBSERVATION_DIM
    assert all(math.isfinite(value) for value in observation.values)
    state_slice = ACTOR_OBSERVATION_FIELD_SLICES["macro_state_one_hot"]
    assert sum(observation.values[slice(*state_slice)]) == 1.0
    source_slice = ACTOR_OBSERVATION_FIELD_SLICES["source_version_one_hot"]
    assert sum(observation.values[slice(*source_slice)]) == 1.0
    previous_slice = ACTOR_OBSERVATION_FIELD_SLICES["previous_residual"]
    assert observation.values[slice(*previous_slice)] == (0.0,) * 12


def test_com_proxy_is_geometry_only_and_support_conditioned() -> None:
    row = _sample()
    observation = build_residual_actor_observation(row, [0.0] * 12)
    start, end = ACTOR_OBSERVATION_FIELD_SLICES["com_proxy_relative_support_xy_m"]
    actual = observation.values[start:end]
    centers = row["wheel_center_w_m"]
    support = row["support_legs"]
    expected = (
        row["base_position_m"]["x"] - sum(centers[leg][0] for leg in support) / len(support),
        row["base_position_m"]["y"] - sum(centers[leg][1] for leg in support) / len(support),
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.__setitem__("source_version", "v010_20260806_220745_363972_manual"), "source_version"),
        (lambda row: row["servo_targets_deg"].pop(SERVO_JOINT_NAMES[0]), "keys mismatch"),
        (lambda row: row["joint_qd_rad_s"].__setitem__(WHEEL_JOINT_NAMES[0], float("nan")), "finite"),
        (lambda row: row.__setitem__("support_legs", []), "must not be empty"),
        (lambda row: row.__setitem__("body_crossed_front_face", 1), "must be bool"),
    ],
)
def test_actor_schema_fails_closed_on_unavailable_or_malformed_inputs(mutation, match: str) -> None:
    row = copy.deepcopy(_sample())
    mutation(row)
    with pytest.raises(ResidualObservationError, match=match):
        build_residual_actor_observation(row, [0.0] * 12)


def test_actor_schema_rejects_nonfinite_or_wrong_residual_history() -> None:
    row = _sample()
    with pytest.raises(ResidualObservationError, match="length"):
        build_residual_actor_observation(row, [0.0] * 11)
    bad = [0.0] * 12
    bad[-1] = float("inf")
    with pytest.raises(ResidualObservationError, match="finite"):
        build_residual_actor_observation(row, bad)


def test_only_current_reviewed_sources_are_authorized() -> None:
    assert AUTHORIZED_SOURCE_VERSIONS == (
        "v003_20260805_224517_157723_manual",
        "v008_20260806_211408_578700_manual",
        "v009_20260806_215232_433234_manual",
    )
