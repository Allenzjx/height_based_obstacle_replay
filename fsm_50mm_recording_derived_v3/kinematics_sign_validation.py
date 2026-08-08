"""Fail-closed front/rear kinematics and placement-sign validation.

The current height-replay project owns the command-to-articulation mapping and
the robot USD, but it does not expose an Isaac-independent FK implementation.
This module therefore never invents link lengths or imports the legacy FSM/PPO
kinematics.  It provides two deliberately separate checks:

* the command delta is mapped with the authoritative ``JOINT_COMMAND_SIGN``;
* an FK/placement response is verified only when isolated wheel-body transforms
  measured in the robot-root frame are supplied for the exact robot USD.

Without those body-transform observations, the result is ``NOT_AVAILABLE`` --
not a guessed placement direction and not a physical-validation claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from command_model import JOINT_COMMAND_SIGN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_USD_PATH = PROJECT_ROOT.parent / "usd" / "wlr_robot_drive_test.usd"


class KinematicsValidationStatus(str, Enum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    VERIFIED = "VERIFIED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class LegKinematicsContract:
    leg: str
    region: str
    side: str
    knee_joint: str
    wheel_body: str


LEG_CONTRACTS: Mapping[str, LegKinematicsContract] = {
    "FL": LegKinematicsContract(
        leg="FL",
        region="FRONT",
        side="LEFT",
        knee_joint="front_left_knee",
        wheel_body="front_left_wheel",
    ),
    "FR": LegKinematicsContract(
        leg="FR",
        region="FRONT",
        side="RIGHT",
        knee_joint="front_right_knee",
        wheel_body="front_right_wheel",
    ),
    "RL": LegKinematicsContract(
        leg="RL",
        region="REAR",
        side="LEFT",
        knee_joint="rear_left_knee",
        wheel_body="rear_left_wheel",
    ),
    "RR": LegKinematicsContract(
        leg="RR",
        region="REAR",
        side="RIGHT",
        knee_joint="rear_right_knee",
        wheel_body="rear_right_wheel",
    ),
}


@dataclass(frozen=True)
class RobotGeometryIdentity:
    path: Path
    available: bool
    sha256: str
    reason: str = ""


@dataclass(frozen=True)
class BodyTransformResponse:
    """Isolated finite-difference response from a trusted articulation sample.

    Wheel centers must be expressed in the robot-root frame.  World-frame
    points would mix chassis motion into the leg FK response and are rejected.
    ``other_servo_max_abs_delta_deg`` proves that the knee perturbation was
    isolated from other servo motion within the caller's stated tolerance.
    """

    leg: str
    joint_name: str
    wheel_body_name: str
    command_delta_deg: float
    measured_joint_delta_deg: float
    wheel_center_before_root_m: tuple[float, float, float]
    wheel_center_after_root_m: tuple[float, float, float]
    other_servo_max_abs_delta_deg: float
    robot_usd_sha256: str
    source_artifact: str
    coordinate_frame: str = "ROBOT_ROOT"


@dataclass(frozen=True)
class FrontRearKinematicsResult:
    status: KinematicsValidationStatus
    reason: str
    front_leg: str
    rear_leg: str
    command_delta_deg: float
    expected_actual_joint_delta_deg: Mapping[str, float]
    measured_actual_joint_delta_deg: Mapping[str, float]
    wheel_center_delta_root_m: Mapping[str, tuple[float, float, float]]
    placement_z_sign: Mapping[str, int]
    robot_usd_path: str
    robot_usd_sha256: str
    evidence_sources: tuple[str, ...]

    @property
    def physically_verified(self) -> bool:
        return self.status == KinematicsValidationStatus.VERIFIED


def current_robot_geometry_identity(
    robot_usd_path: str | Path = DEFAULT_ROBOT_USD_PATH,
) -> RobotGeometryIdentity:
    path = Path(robot_usd_path).resolve()
    if not path.is_file():
        return RobotGeometryIdentity(
            path=path,
            available=False,
            sha256="",
            reason="robot USD is unavailable",
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return RobotGeometryIdentity(
            path=path,
            available=False,
            sha256="",
            reason=f"robot USD could not be read: {exc}",
        )
    return RobotGeometryIdentity(path=path, available=True, sha256=digest.hexdigest())


def command_to_actual_joint_delta_deg(joint_name: str, command_delta_deg: float) -> float:
    """Apply the authoritative height-replay command-space sign mapping."""

    if joint_name not in JOINT_COMMAND_SIGN:
        raise KeyError(f"unknown command-space joint: {joint_name}")
    value = float(command_delta_deg)
    if not math.isfinite(value):
        raise ValueError("command_delta_deg must be finite")
    return float(JOINT_COMMAND_SIGN[joint_name]) * value


def _vec3(value: Sequence[float]) -> tuple[float, float, float] | None:
    try:
        row = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(row) != 3 or not all(math.isfinite(item) for item in row):
        return None
    return row


def _sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _result(
    *,
    status: KinematicsValidationStatus,
    reason: str,
    front_leg: str,
    rear_leg: str,
    command_delta_deg: float,
    expected: Mapping[str, float],
    measured: Mapping[str, float] | None,
    wheel_delta: Mapping[str, tuple[float, float, float]] | None,
    placement_sign: Mapping[str, int] | None,
    identity: RobotGeometryIdentity,
    sources: tuple[str, ...] = (),
) -> FrontRearKinematicsResult:
    return FrontRearKinematicsResult(
        status=status,
        reason=reason,
        front_leg=front_leg,
        rear_leg=rear_leg,
        command_delta_deg=float(command_delta_deg),
        expected_actual_joint_delta_deg=dict(expected),
        measured_actual_joint_delta_deg=dict(measured or {}),
        wheel_center_delta_root_m=dict(wheel_delta or {}),
        placement_z_sign=dict(placement_sign or {}),
        robot_usd_path=str(identity.path),
        robot_usd_sha256=identity.sha256,
        evidence_sources=tuple(sources),
    )


def validate_front_rear_placement_response(
    *,
    front_leg: str,
    rear_leg: str,
    command_delta_deg: float,
    body_transform_responses: Mapping[str, BodyTransformResponse] | None = None,
    robot_usd_path: str | Path = DEFAULT_ROBOT_USD_PATH,
    joint_delta_tolerance_deg: float = 0.05,
    isolation_tolerance_deg: float = 0.05,
    minimum_wheel_center_response_m: float = 1.0e-5,
) -> FrontRearKinematicsResult:
    """Validate front/rear placement signs from isolated root-frame transforms.

    The user-observed front/rear planar response is encoded only as the
    fail-closed invariant that the two non-zero vertical response signs differ.
    Which leg moves toward ``+z`` is learned from the supplied body transforms,
    never inferred from a command sign alone.
    """

    front_leg = str(front_leg).upper()
    rear_leg = str(rear_leg).upper()
    if front_leg not in LEG_CONTRACTS or LEG_CONTRACTS[front_leg].region != "FRONT":
        raise ValueError(f"front_leg must be FL or FR, got {front_leg!r}")
    if rear_leg not in LEG_CONTRACTS or LEG_CONTRACTS[rear_leg].region != "REAR":
        raise ValueError(f"rear_leg must be RL or RR, got {rear_leg!r}")
    if LEG_CONTRACTS[front_leg].side != LEG_CONTRACTS[rear_leg].side:
        raise ValueError("front/rear response pair must use the same robot side")
    delta = float(command_delta_deg)
    if not math.isfinite(delta) or abs(delta) <= 1.0e-12:
        raise ValueError("command_delta_deg must be finite and non-zero")
    for label, value in (
        ("joint_delta_tolerance_deg", joint_delta_tolerance_deg),
        ("isolation_tolerance_deg", isolation_tolerance_deg),
        ("minimum_wheel_center_response_m", minimum_wheel_center_response_m),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")

    identity = current_robot_geometry_identity(robot_usd_path)
    expected = {
        leg: command_to_actual_joint_delta_deg(
            LEG_CONTRACTS[leg].knee_joint,
            delta,
        )
        for leg in (front_leg, rear_leg)
    }
    if not identity.available:
        return _result(
            status=KinematicsValidationStatus.NOT_AVAILABLE,
            reason=identity.reason,
            front_leg=front_leg,
            rear_leg=rear_leg,
            command_delta_deg=delta,
            expected=expected,
            measured=None,
            wheel_delta=None,
            placement_sign=None,
            identity=identity,
        )
    if not body_transform_responses:
        return _result(
            status=KinematicsValidationStatus.NOT_AVAILABLE,
            reason=(
                "no current-project Isaac-independent FK or isolated ROBOT_ROOT "
                "wheel-body transform response was supplied"
            ),
            front_leg=front_leg,
            rear_leg=rear_leg,
            command_delta_deg=delta,
            expected=expected,
            measured=None,
            wheel_delta=None,
            placement_sign=None,
            identity=identity,
        )

    measured: dict[str, float] = {}
    wheel_delta: dict[str, tuple[float, float, float]] = {}
    placement_sign: dict[str, int] = {}
    sources: list[str] = []
    errors: list[str] = []
    for leg in (front_leg, rear_leg):
        contract = LEG_CONTRACTS[leg]
        response = body_transform_responses.get(leg)
        if response is None:
            errors.append(f"{leg} body-transform response is missing")
            continue
        if str(response.leg).upper() != leg:
            errors.append(f"{leg} response leg identity mismatch")
        if response.joint_name != contract.knee_joint:
            errors.append(f"{leg} response joint must be {contract.knee_joint}")
        if response.wheel_body_name != contract.wheel_body:
            errors.append(f"{leg} response wheel body must be {contract.wheel_body}")
        if str(response.coordinate_frame).upper() != "ROBOT_ROOT":
            errors.append(f"{leg} wheel centers must use ROBOT_ROOT coordinates")
        if response.robot_usd_sha256 != identity.sha256:
            errors.append(f"{leg} robot USD SHA256 mismatch")
        if not str(response.source_artifact).strip():
            errors.append(f"{leg} source artifact is missing")
        if not math.isfinite(float(response.command_delta_deg)) or abs(
            float(response.command_delta_deg) - delta
        ) > 1.0e-9:
            errors.append(f"{leg} command delta does not match the paired probe")
        measured_delta = float(response.measured_joint_delta_deg)
        measured[leg] = measured_delta
        if not math.isfinite(measured_delta) or abs(measured_delta - expected[leg]) > float(
            joint_delta_tolerance_deg
        ):
            errors.append(f"{leg} measured joint delta violates command-space mapping")
        other_delta = float(response.other_servo_max_abs_delta_deg)
        if not math.isfinite(other_delta) or other_delta > float(isolation_tolerance_deg):
            errors.append(f"{leg} probe is not an isolated knee response")
        before = _vec3(response.wheel_center_before_root_m)
        after = _vec3(response.wheel_center_after_root_m)
        if before is None or after is None:
            errors.append(f"{leg} wheel-center transform is non-finite")
            continue
        response_delta = tuple(after[index] - before[index] for index in range(3))
        wheel_delta[leg] = response_delta
        placement_sign[leg] = _sign(
            response_delta[2],
            float(minimum_wheel_center_response_m),
        )
        if placement_sign[leg] == 0:
            errors.append(f"{leg} vertical wheel-center response is unresolved")
        sources.append(str(response.source_artifact))

    if not errors and placement_sign[front_leg] == placement_sign[rear_leg]:
        errors.append("front and rear placement response signs are not distinct")
    return _result(
        status=(
            KinematicsValidationStatus.INVALID
            if errors
            else KinematicsValidationStatus.VERIFIED
        ),
        reason="; ".join(errors),
        front_leg=front_leg,
        rear_leg=rear_leg,
        command_delta_deg=delta,
        expected=expected,
        measured=measured,
        wheel_delta=wheel_delta,
        placement_sign=placement_sign,
        identity=identity,
        sources=tuple(sources),
    )


__all__ = [
    "BodyTransformResponse",
    "DEFAULT_ROBOT_USD_PATH",
    "FrontRearKinematicsResult",
    "KinematicsValidationStatus",
    "LEG_CONTRACTS",
    "LegKinematicsContract",
    "RobotGeometryIdentity",
    "command_to_actual_joint_delta_deg",
    "current_robot_geometry_identity",
    "validate_front_rear_placement_response",
]
