from __future__ import annotations

from command_model import JOINT_COMMAND_SIGN

from fsm_50mm_recording_derived_v3.kinematics_sign_validation import (
    BodyTransformResponse,
    KinematicsValidationStatus,
    LEG_CONTRACTS,
    command_to_actual_joint_delta_deg,
    current_robot_geometry_identity,
    validate_front_rear_placement_response,
)


def _response(
    leg: str,
    *,
    command_delta_deg: float,
    dz_m: float,
) -> BodyTransformResponse:
    contract = LEG_CONTRACTS[leg]
    identity = current_robot_geometry_identity()
    assert identity.available
    return BodyTransformResponse(
        leg=leg,
        joint_name=contract.knee_joint,
        wheel_body_name=contract.wheel_body,
        command_delta_deg=command_delta_deg,
        measured_joint_delta_deg=command_to_actual_joint_delta_deg(
            contract.knee_joint,
            command_delta_deg,
        ),
        wheel_center_before_root_m=(0.2, 0.3, -0.1),
        wheel_center_after_root_m=(0.2, 0.3, -0.1 + dz_m),
        other_servo_max_abs_delta_deg=0.0,
        robot_usd_sha256=identity.sha256,
        source_artifact=f"unit-test isolated {leg} ROBOT_ROOT body transform",
    )


def test_same_command_delta_uses_authoritative_opposite_front_rear_joint_signs() -> None:
    command_delta = 7.5
    front_joint = LEG_CONTRACTS["FL"].knee_joint
    rear_joint = LEG_CONTRACTS["RL"].knee_joint
    front_actual = command_to_actual_joint_delta_deg(front_joint, command_delta)
    rear_actual = command_to_actual_joint_delta_deg(rear_joint, command_delta)

    assert front_actual == JOINT_COMMAND_SIGN[front_joint] * command_delta
    assert rear_actual == JOINT_COMMAND_SIGN[rear_joint] * command_delta
    assert front_actual == -rear_actual


def test_fk_response_is_not_available_without_real_body_transform_evidence() -> None:
    result = validate_front_rear_placement_response(
        front_leg="FL",
        rear_leg="RL",
        command_delta_deg=5.0,
    )

    assert result.status == KinematicsValidationStatus.NOT_AVAILABLE
    assert not result.physically_verified
    assert result.placement_z_sign == {}
    assert "no current-project Isaac-independent FK" in result.reason


def test_root_frame_body_transform_contract_verifies_distinct_placement_signs() -> None:
    command_delta = 5.0
    result = validate_front_rear_placement_response(
        front_leg="FR",
        rear_leg="RR",
        command_delta_deg=command_delta,
        body_transform_responses={
            "FR": _response("FR", command_delta_deg=command_delta, dz_m=0.012),
            "RR": _response("RR", command_delta_deg=command_delta, dz_m=-0.010),
        },
    )

    assert result.status == KinematicsValidationStatus.VERIFIED
    assert result.physically_verified
    assert result.placement_z_sign == {"FR": 1, "RR": -1}
    assert result.expected_actual_joint_delta_deg["FR"] == 5.0
    assert result.expected_actual_joint_delta_deg["RR"] == -5.0


def test_same_front_rear_body_response_sign_is_rejected() -> None:
    command_delta = 5.0
    result = validate_front_rear_placement_response(
        front_leg="FL",
        rear_leg="RL",
        command_delta_deg=command_delta,
        body_transform_responses={
            "FL": _response("FL", command_delta_deg=command_delta, dz_m=0.010),
            "RL": _response("RL", command_delta_deg=command_delta, dz_m=0.008),
        },
    )

    assert result.status == KinematicsValidationStatus.INVALID
    assert not result.physically_verified
    assert "not distinct" in result.reason
