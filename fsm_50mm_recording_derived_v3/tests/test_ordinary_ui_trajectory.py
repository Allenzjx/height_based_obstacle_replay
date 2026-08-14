from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3 import ordinary_ui_trajectory as trajectory


SOURCE_VERSION = "v003_20260805_224517_157723_manual"
ACCEPTED_SHA = "1" * 64
PLAN_SHA = "2" * 64
READINESS_SHA = "3" * 64
PLAN_ID = "fsm50-v003-ordinary-ui-diagnostic"
REQUEST_ID = "request-ordinary-ui-001"
WORKER_ID = "worker-session-001"
ADAPTER_ID = "adapter-runtime-001"
JOINT_NAMES = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
LEGS = ("FL", "FR", "RL", "RR")
LEG_TO_WHEEL = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}
SIGNS = {
    "front_left_ankle": -1.0,
    "front_right_ankle": 1.0,
    "rear_left_ankle": -1.0,
    "rear_right_ankle": 1.0,
}


def _identity_values() -> dict[str, object]:
    return {
        "source_version": SOURCE_VERSION,
        "accepted_steps_sha256": ACCEPTED_SHA,
        "plan_sha256": PLAN_SHA,
        "plan_id": PLAN_ID,
        "request_id": REQUEST_ID,
        "worker_session_id": WORKER_ID,
        "adapter_runtime_instance_id": ADAPTER_ID,
        "readiness_token_sha256": READINESS_SHA,
        "root_state_write_count": 0,
    }


def _envelope() -> dict[str, object]:
    identity = _identity_values()

    def fields(*names: str) -> dict[str, object]:
        return {name: identity[name] for name in names}

    return {
        "schema_version": trajectory.IDENTITY_SCHEMA,
        **identity,
        "evidence": {
            "durable_result": {
                "artifact_sha256": "a" * 64,
                "identity_fields": fields(*identity),
            },
            "immutable_request": {
                "artifact_sha256": "b" * 64,
                "identity_fields": fields(
                    "source_version",
                    "accepted_steps_sha256",
                    "plan_sha256",
                    "plan_id",
                    "request_id",
                    "root_state_write_count",
                ),
            },
            "readiness": {
                "artifact_sha256": "c" * 64,
                "identity_fields": fields(
                    "source_version",
                    "plan_sha256",
                    "plan_id",
                    "request_id",
                    "worker_session_id",
                    "adapter_runtime_instance_id",
                    "root_state_write_count",
                ),
            },
            "dispatch_ledger": {
                "artifact_sha256": "d" * 64,
                "identity_fields": fields(
                    "source_version",
                    "plan_id",
                    "readiness_token_sha256",
                ),
            },
        },
    }


def _row(index: int) -> dict[str, object]:
    measured_position = {
        name: 0.01 * (joint_index + 1) + 0.001 * index
        for joint_index, name in enumerate(JOINT_NAMES)
    }
    measured_velocity = {
        name: -0.02 * (joint_index + 1) + 0.002 * index
        for joint_index, name in enumerate(JOINT_NAMES)
    }
    position_target = {
        name: 0.005 * (joint_index + 1) + 0.0005 * index
        for joint_index, name in enumerate(JOINT_NAMES)
    }
    velocity_target = {name: 0.0 for name in JOINT_NAMES}
    wheel_command = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    if index == 1:
        wheel_command["front_left_ankle"] = -0.5
        velocity_target["front_left_ankle"] = -0.5
    logical = {
        leg: wheel_command[joint_name]
        for leg, joint_name in LEG_TO_WHEEL.items()
    }
    physx = {
        leg: velocity_target[joint_name]
        for leg, joint_name in LEG_TO_WHEEL.items()
    }
    canonical_angle = {
        leg: SIGNS[joint_name] * measured_position[joint_name]
        for leg, joint_name in LEG_TO_WHEEL.items()
    }
    canonical_velocity = {
        leg: SIGNS[joint_name] * measured_velocity[joint_name]
        for leg, joint_name in LEG_TO_WHEEL.items()
    }
    return {
        "sample_index": index,
        "sim_step": 100 + index,
        "time_s": 2.0 + index / 120.0,
        "physics_dt_s": 1.0 / 120.0,
        "source_version": SOURCE_VERSION,
        "robot_joint_names": list(JOINT_NAMES),
        "root_pose_w": [0.1 * index, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
        "root_linear_velocity_w": [0.01 * index, 0.0, 0.0],
        "root_angular_velocity_w": [0.0, 0.0, 0.01 * index],
        "com_position_w": [0.1 * index, 0.0, 0.15],
        "com_velocity_w": [0.01 * index, 0.0, 0.0],
        "measured_joint_position_rad": measured_position,
        "measured_joint_velocity_rad_s": measured_velocity,
        "physx_joint_position_target_rad": position_target,
        "joint_position_target_buffer_rad": dict(position_target),
        "physx_joint_velocity_target_rad_s": velocity_target,
        "joint_velocity_target_buffer_rad_s": dict(velocity_target),
        "servo_command_target_rad": {
            name: position_target[name] for name in SERVO_JOINT_NAMES
        },
        "physx_drive_target_evidence_valid": True,
        "wheel_command_velocity_rad_s": wheel_command,
        "wheel_canonical_forward_angle_rad": canonical_angle,
        "wheel_canonical_forward_velocity_rad_s": canonical_velocity,
        "wheel_logical_target_rad_s_by_leg": logical,
        "wheel_physx_target_rad_s_by_leg": physx,
        "wheel_forward_sign": dict(SIGNS),
        "wheel_direction": 1.0,
        "fsm_state": "RECORDING_FAST_REPLAY",
        "scheduler_phase": "SOURCE" if index else "READY",
        "macro_state_cursor": "RECORDING_FAST_REPLAY",
        "command_cursor": index,
        "segment_cursor": index,
        "source_command": "" if index == 0 else "wheel fl 0.5",
        "source_event_index": None if index == 0 else index - 1,
        "source_event_indices": [] if index == 0 else [index - 1],
        "source_fast_segment": None if index == 0 else index - 1,
        "source_step": None if index == 0 else 1,
        "planned_dispatch_time_s": None if index == 0 else 2.0,
        "actual_dispatch_time_s": None if index == 0 else 2.0 + index / 120.0,
        "atomic_batch_id": "" if index == 0 else f"batch-{index}",
        "dispatch_kind": "" if index == 0 else "source_segment_start",
        "atomic_concurrent": index == 1,
        "planned_servo_target_deg": (
            {} if index == 0 else {"front_left_hip": 1.0}
        ),
        "planned_wheel_target_rad_s": (
            {} if index == 0 else dict(wheel_command)
        ),
        # Deliberately present in input: contact data must not reach output.
        "wheel_contact_classes": {leg: "GROUND" for leg in LEGS},
        "wheel_contact_force_up_n": {leg: 10.0 for leg in LEGS},
    }


def _rows() -> list[dict[str, object]]:
    return [_row(index) for index in range(3)]


def test_identity_envelope_is_strictly_cross_bound() -> None:
    observed = trajectory.validate_identity_envelope(_envelope())
    assert observed["root_state_write_count"] == 0
    assert observed["evidence"]["durable_result"]["identity_fields"] == _identity_values()

    mismatch = _envelope()
    mismatch["evidence"]["readiness"]["identity_fields"]["worker_session_id"] = "other"
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match="evidence mismatch"):
        trajectory.validate_identity_envelope(mismatch)

    nonzero = _envelope()
    nonzero["root_state_write_count"] = 1
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match="exactly 0"):
        trajectory.validate_identity_envelope(nonzero)

    missing = _envelope()
    del missing["evidence"]["dispatch_ledger"]["identity_fields"][
        "readiness_token_sha256"
    ]
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match="missing required fields"):
        trajectory.validate_identity_envelope(missing)


def test_build_projects_only_sensor_independent_fields_and_denies_eligibility() -> None:
    bundle = trajectory.build_ordinary_ui_trajectory(
        _rows(), identity_envelope=_envelope()
    )
    assert bundle["schema_version"] == trajectory.BUNDLE_SCHEMA
    assert len(bundle["rows"]) == 3
    assert bundle["rows"][0]["root"]["pose_w"] == [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    assert set(bundle["rows"][0]["joints"]["measured_position_rad"]) == set(
        JOINT_NAMES
    )
    serialized_rows = json.dumps(bundle["rows"], sort_keys=True)
    assert "contact" not in serialized_rows
    manifest = bundle["manifest"]
    assert manifest["contact_evidence"] == {
        "available": False,
        "verdict": trajectory.NOT_EVALUABLE,
        "reason": "ordinary-UI trajectory intentionally contains no contact evidence",
    }
    assert manifest["physical_evidence"]["full_verdict"] == trajectory.NOT_EVALUABLE
    assert manifest["eligibility"] == {
        "gate1": False,
        "environment_equivalence": False,
        "physical_success_claim": False,
    }
    assert trajectory.ordinary_ui_diagnostic_complete(bundle) is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[1].__setitem__("sim_step", 103), "continuous 120 Hz grid"),
        (lambda rows: rows[1].__setitem__("physics_dt_s", 1.0 / 60.0), "exactly 1/120"),
        (lambda rows: rows[2].__setitem__("time_s", 2.1), "off the exact 120 Hz grid"),
        (lambda rows: rows[1].__setitem__("sample_index", 9), "sample_index is not continuous"),
    ],
)
def test_grid_is_exact_and_continuous(mutate, message: str) -> None:
    rows = _rows()
    mutate(rows)
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match=message):
        trajectory.build_ordinary_ui_trajectory(rows, identity_envelope=_envelope())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row["measured_joint_position_rad"].pop(JOINT_NAMES[0]),
            "identity differs",
        ),
        (
            lambda row: row["measured_joint_velocity_rad_s"].__setitem__(
                JOINT_NAMES[0], math.nan
            ),
            "must be finite",
        ),
        (
            lambda row: row.__setitem__("physx_drive_target_evidence_valid", False),
            "must be true",
        ),
        (
            lambda row: row["wheel_physx_target_rad_s_by_leg"].__setitem__("FL", 9.0),
            "differs from joint readback",
        ),
        (
            lambda row: row.__setitem__("root_state_write_count", 1),
            "exactly 0",
        ),
        (
            lambda row: row.__setitem__("plan_sha256", "9" * 64),
            "differs from sealed identity",
        ),
    ],
)
def test_rows_fail_closed_on_incomplete_or_inconsistent_readback(mutate, message: str) -> None:
    rows = _rows()
    mutate(rows[1])
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match=message):
        trajectory.build_ordinary_ui_trajectory(rows, identity_envelope=_envelope())


def test_write_validate_and_refuse_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "ordinary_ui_diagnostic"
    receipt = trajectory.write_ordinary_ui_trajectory(
        output, _rows(), identity_envelope=_envelope()
    )
    assert receipt["schema_version"] == trajectory.RECEIPT_SCHEMA
    assert receipt["diagnostic_complete"] is True
    assert receipt["row_count"] == 3
    assert receipt["full_physical_verdict"] == trajectory.NOT_EVALUABLE
    assert receipt["gate1_eligible"] is False
    assert receipt["environment_equivalence_eligible"] is False
    assert trajectory.validate_ordinary_ui_trajectory(output) == receipt
    assert trajectory.ordinary_ui_diagnostic_complete(output) is True
    assert trajectory.ordinary_ui_diagnostic_complete(receipt) is True
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match="refusing to overwrite"):
        trajectory.write_ordinary_ui_trajectory(
            output, _rows(), identity_envelope=_envelope()
        )


@pytest.mark.parametrize("filename", [trajectory.TRAJECTORY_FILENAME, trajectory.MANIFEST_FILENAME, trajectory.SEAL_FILENAME])
def test_any_file_tamper_invalidates_diagnostic_complete(tmp_path: Path, filename: str) -> None:
    output = tmp_path / filename.replace(".", "_")
    receipt = trajectory.write_ordinary_ui_trajectory(
        output, _rows(), identity_envelope=_envelope()
    )
    target = output / filename
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(trajectory.OrdinaryUITrajectoryError):
        trajectory.validate_ordinary_ui_trajectory(output)
    assert trajectory.ordinary_ui_diagnostic_complete(output) is False
    assert trajectory.ordinary_ui_diagnostic_complete(receipt) is False


def test_helper_rejects_forged_claims_and_strict_json_duplicates(tmp_path: Path) -> None:
    assert trajectory.ordinary_ui_diagnostic_complete(
        {
            "schema_version": trajectory.RECEIPT_SCHEMA,
            "output_dir": str(tmp_path),
            "diagnostic_complete": True,
        }
    ) is False
    assert trajectory.ordinary_ui_diagnostic_complete(
        {"schema_version": trajectory.MANIFEST_SCHEMA, "diagnostic_complete": True}
    ) is False

    output = tmp_path / "duplicate"
    trajectory.write_ordinary_ui_trajectory(output, _rows(), identity_envelope=_envelope())
    manifest_path = output / trajectory.MANIFEST_FILENAME
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        '{"schema_version":"forged",' + original[1:], encoding="utf-8"
    )
    with pytest.raises(trajectory.OrdinaryUITrajectoryError, match="duplicate JSON key"):
        trajectory.validate_ordinary_ui_trajectory(output)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bundle: bundle["manifest"]["eligibility"].__setitem__("gate1", 0),
        lambda bundle: bundle["manifest"]["trajectory"].__setitem__(
            "row_count", 3.0
        ),
        lambda bundle: bundle["rows"][0].__setitem__("time_s", 2),
    ],
)
def test_in_memory_bundle_comparison_is_type_strict(mutate) -> None:
    bundle = trajectory.build_ordinary_ui_trajectory(
        _rows(), identity_envelope=_envelope()
    )
    mutate(bundle)
    assert trajectory.ordinary_ui_diagnostic_complete(bundle) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.__setitem__("diagnostic_complete", 1),
        lambda receipt: receipt.__setitem__("row_count", 3.0),
        lambda receipt: receipt.__setitem__("gate1_eligible", 0),
    ],
)
def test_receipt_comparison_is_type_strict(tmp_path: Path, mutate) -> None:
    output = tmp_path / "receipt_types"
    receipt = trajectory.write_ordinary_ui_trajectory(
        output, _rows(), identity_envelope=_envelope()
    )
    mutate(receipt)
    assert trajectory.ordinary_ui_diagnostic_complete(receipt) is False


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        (
            trajectory.MANIFEST_FILENAME,
            lambda value: value.__setitem__("diagnostic_complete", 1),
        ),
        (
            trajectory.MANIFEST_FILENAME,
            lambda value: value["trajectory"].__setitem__("row_count", 3.0),
        ),
        (
            trajectory.SEAL_FILENAME,
            lambda value: value.__setitem__("gate1_eligible", 0),
        ),
    ],
)
def test_disk_manifest_and_seal_comparison_is_type_strict(
    tmp_path: Path, filename: str, mutate
) -> None:
    output = tmp_path / (filename.replace(".", "_") + "_types")
    trajectory.write_ordinary_ui_trajectory(
        output, _rows(), identity_envelope=_envelope()
    )
    path = output / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(trajectory.OrdinaryUITrajectoryError):
        trajectory.validate_ordinary_ui_trajectory(output)
    assert trajectory.ordinary_ui_diagnostic_complete(output) is False
