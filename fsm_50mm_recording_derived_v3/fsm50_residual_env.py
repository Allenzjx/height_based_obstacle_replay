"""Phase-local Gate-E R1 environment contract and lazy Isaac Lab adapter.

The module is deliberately importable before :class:`AppLauncher` exists.
Everything through :class:`PhaseLocalResidualKernel` is Isaac-free; Isaac Lab,
Torch, and Gym are imported only by the explicit construction/registration
seams at the bottom of the file.

The environment is not a second FSM.  One episode is rooted in one immutable
phase-entry-bank snapshot and may consume only that entry's exact public
recording profile cursor.  Logical source actions are evaluated at 15 Hz while
safety, target slew, and physical readback remain visible at 120 Hz.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from command_model import (
    DEFAULT_MAX_WHEEL_SPEED_RAD_S,
    JOINT_COMMAND_SIGN,
    SERVO_JOINT_NAMES,
    WHEEL_FORWARD_SIGN,
    WHEEL_JOINT_NAMES,
)
from completion_aware_segment import (
    CompletionAwareSegmentExecutor,
    SegmentCompletionSpec,
    SegmentDecisionKind,
    SegmentFeedback,
)

from .fsm50_direct_command_residual import (
    RESIDUAL_ACTION_DIM,
    ZERO_RESIDUAL_ACTION,
    ResidualTransformInput,
    compose_direct_command_residual,
)
from .fsm50_macro_state_model import MacroGuardKind, build_default_macro_graph
from .fsm50_motion_profiles import (
    MotionProfileLibrary,
    PhaseMotionProfile,
    build_profile_library,
)
from .fsm50_phase_entry_bank import (
    PhaseEntryBank,
    canonical_sha256 as phase_bank_payload_sha256,
    load_phase_entry_bank,
    validate_phase_entry_bank_mapping,
)
from .fsm50_residual_envelope import ResidualEnvelope, load_residual_envelope
from .fsm50_residual_observation import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_SCHEMA_SHA256,
    ResidualActorObservation,
    build_residual_actor_observation,
)
from .fsm50_residual_outer_cycle import (
    EXPECTED_PHYSICS_HZ,
    EXPECTED_POLICY_HZ,
    EXPECTED_RENDER_SUBSTEPS,
    OuterCycleContext,
)
from .fsm50_residual_reward import (
    ResidualRewardInput,
    ResidualRewardResult,
    compute_residual_reward,
)
from .fsm50_residual_scene import (
    DIRECT_RL_DECIMATION,
    OBSTACLE_BOTTOM_Z_M,
    OBSTACLE_FRONT_X_M,
    OBSTACLE_HEIGHT_M,
    OBSTACLE_LENGTH_M,
    OBSTACLE_WIDTH_M,
    PHYSICS_DT_S,
    RENDER_INTERVAL_PHYSICS_STEPS,
    build_isaaclab_scene_bundle,
    load_formal_scene_spec,
    spawn_global_scene_assets,
)
from .support_classifier import ObstacleGeometry, WheelObservation, classify_wheel_contact


SCHEMA_VERSION = "fsm50.residual_phase_local_env.v1"
EPISODE_METRICS_SCHEMA_VERSION = "fsm50.residual_phase_local_metrics.v1"
GYM_ENV_ID = "Isaac-Fsm50-Residual-PhaseLocal-Direct-v0"
NO_ACTIVE_PROFILE_STRATEGY = "NO_ACTIVE_PROFILE"
PHYSX_TARGET_READBACK_SOURCE = (
    "robot.root_physx_view.get_dof_position_targets/get_dof_velocity_targets"
)
MODULE_ROOT = Path(__file__).resolve().parent
REPLAY_ROOT = MODULE_ROOT.parent

SERVO_REFERENCE_VELOCITY_DEG_S = 150.0
SERVO_MAX_DELTA_DEG_PER_PHYSICS_STEP = SERVO_REFERENCE_VELOCITY_DEG_S * PHYSICS_DT_S
POLICY_DT_S = EXPECTED_RENDER_SUBSTEPS * PHYSICS_DT_S
CRITIC_STATE_DIM = ACTOR_OBSERVATION_DIM
DEFAULT_EPISODE_LENGTH_S = 60.0
DEFAULT_TIMEOUT_PHYSICS_STEPS = int(DEFAULT_EPISODE_LENGTH_S * EXPECTED_PHYSICS_HZ)
WHEEL_RADIUS_M = 0.04998999834060672
WHEEL_BODY_NAMES = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_TO_WHEEL_BODY = dict(zip(LEG_NAMES, WHEEL_BODY_NAMES))
ACTIVE_LEG_BY_STATE = MappingProxyType(
    {
        "S5_PRE_RR_COM_SHIFT": "RR",
        "S7_PRE_RL_SUPPORT_SETUP": "RL",
        "S8_RL_COM_SHIFT_AND_TRAVERSE": "RL",
        "S10_POSTURE_RECOVERY": "",
    }
)
EPISODE_METRICS_FIELDS = (
    "schema_version",
    "env_source_sha256",
    "scene_manifest_sha256",
    "bank_sha256",
    "entry_sha256",
    "entry_id",
    "source_version",
    "phase_state",
    "profile_id",
    "profile_sha256",
    "source_plan_sha256",
    "residual_envelope_sha256",
    "actor_observation_schema_sha256",
    "task_success",
    "hard_failure",
    "terminal_reason",
    "body_crossed_front_face",
    "required_leg_lift_seen",
    "final_recoverable",
    "posture_complete",
    "completion_time_s",
    "return",
    "physics_length",
    "outer_policy_length",
    "peak_abs_roll_rad",
    "peak_abs_pitch_rad",
    "peak_root_angular_speed_rad_s",
    "peak_abs_servo_error_deg",
    "peak_residual_l2",
    "peak_residual_slew_l2",
    "source_consumption_count",
    "residual_transform_count",
    "physical_batch_count",
    "physical_command_epoch",
    "last_verified_physical_command_epoch",
    "n_plus_one_verified_count",
    "dynamic_wheel_stop_count",
    "exact_eight_cadence_verified",
    "source_cursor_boundary_verified",
    "one_batch_per_tick_verified",
    "n_plus_one_verified",
    "terminal_latched",
    "terminal_ready",
)


class ResidualEnvContractError(RuntimeError):
    """A bank, cadence, action, feedback, or actuation contract failed."""


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidualEnvContractError(f"value is not strict canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


EPISODE_METRICS_SCHEMA_SHA256 = _canonical_sha256(
    {
        "schema_version": EPISODE_METRICS_SCHEMA_VERSION,
        "fields": list(EPISODE_METRICS_FIELDS),
    }
)
_ENV_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def env_source_sha256() -> str:
    """Return the current environment source bytes identity."""

    return _ENV_SOURCE_SHA256


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise ResidualEnvContractError(f"{label} must be exact finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ResidualEnvContractError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _exact_int(value: Any, label: str, *, nonnegative: bool = True) -> int:
    if type(value) is not int or (nonnegative and value < 0):
        raise ResidualEnvContractError(f"{label} must be an exact nonnegative int")
    return value


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ResidualEnvContractError(f"{label} must be a {length}-element numeric sequence")
    if len(value) != length:
        raise ResidualEnvContractError(f"{label} must contain exactly {length} values")
    return tuple(_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _exact_finite_map(value: Any, names: Sequence[str], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ResidualEnvContractError(f"{label} must contain exactly {list(names)!r}")
    return {name: _finite_number(value[name], f"{label}.{name}") for name in names}


def _map_from_vector(value: Any, names: Sequence[str], label: str) -> dict[str, float]:
    vector = _finite_vector(value, len(names), label)
    return dict(zip(names, vector))


def _frozen_map(value: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({str(name): float(number) for name, number in value.items()})


def _finite_action(action: Sequence[float]) -> tuple[float, ...]:
    return _finite_vector(action, RESIDUAL_ACTION_DIM, "normalized residual action")


def build_actor_critic_observation_tensors(
    values: Sequence[Sequence[float]],
    *,
    torch_module: Any,
    device: Any,
) -> Mapping[str, Any]:
    """Build equal deployable actor and non-privileged critic tensors lazily."""

    policy = torch_module.tensor(values, dtype=torch_module.float32, device=device)
    expected_rows = len(values)
    if tuple(policy.shape) != (expected_rows, ACTOR_OBSERVATION_DIM) or not bool(
        torch_module.isfinite(policy).all().item()
    ):
        raise ResidualEnvContractError("actor/critic observation tensor is not finite [num_envs,115]")
    critic = policy.clone()
    if not bool(torch_module.equal(policy, critic)):
        raise ResidualEnvContractError("critic state differs from the frozen actor observation")
    return MappingProxyType({"policy": policy, "critic": critic})


@dataclass(frozen=True)
class PhysxTargetReadbackEvidence:
    """Fail-closed evidence from the low-level PhysX articulation view."""

    verified: bool
    failure_reason: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.verified) is not bool:
            raise ResidualEnvContractError("PhysX target evidence verified must be exact bool")
        if not isinstance(self.failure_reason, str):
            raise ResidualEnvContractError("PhysX target evidence failure_reason must be text")
        if self.verified == bool(self.failure_reason):
            raise ResidualEnvContractError(
                "PhysX target evidence verified/failure_reason identity is inconsistent"
            )
        digest = self.evidence_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ResidualEnvContractError("PhysX target evidence SHA-256 is invalid")


def validate_physx_target_readback(
    *,
    robot: Any,
    num_envs: int,
    env_index: int,
    servo_joint_ids: Sequence[int],
    wheel_joint_ids: Sequence[int],
    expected_servo_position_targets: Any,
    expected_wheel_velocity_targets: Any,
) -> PhysxTargetReadbackEvidence:
    """Validate exact N+1 targets from ``root_physx_view`` only.

    Isaac Lab's ``robot.data.joint_*_target`` tensors are command buffers, not
    low-level drive readback. They are deliberately not inspected here.
    Every malformed/missing/mismatched runtime input produces unverified
    evidence so the kernel can issue its fail-closed safe stop.
    """

    evidence_payload: dict[str, Any] = {
        "schema_version": "fsm50.physx_target_readback.v1",
        "source": PHYSX_TARGET_READBACK_SOURCE,
        "verified": False,
        "failure_reason": "",
    }

    def finish(*, verified: bool, failure_reason: str, details: Mapping[str, Any]) -> PhysxTargetReadbackEvidence:
        payload = dict(evidence_payload)
        payload.update(
            {
                "verified": verified,
                "failure_reason": failure_reason,
                "details": dict(details),
            }
        )
        return PhysxTargetReadbackEvidence(
            verified=verified,
            failure_reason=failure_reason,
            evidence_sha256=_canonical_sha256(payload),
        )

    def matrix_rows(value: Any, expected_shape: tuple[int, int], label: str) -> list[list[float]]:
        raw_shape = getattr(value, "shape", None)
        try:
            shape = tuple(int(item) for item in raw_shape)
        except Exception as exc:
            raise ResidualEnvContractError(f"{label} shape is unavailable: {type(exc).__name__}: {exc}") from exc
        if shape != expected_shape:
            raise ResidualEnvContractError(f"{label} shape {shape} != {expected_shape}")
        try:
            detached = value.detach() if callable(getattr(value, "detach", None)) else value
            host = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
            raw_rows = host.tolist() if callable(getattr(host, "tolist", None)) else list(host)
        except Exception as exc:
            raise ResidualEnvContractError(f"{label} rows are unreadable: {type(exc).__name__}: {exc}") from exc
        if not isinstance(raw_rows, list) or len(raw_rows) != expected_shape[0]:
            raise ResidualEnvContractError(f"{label} row count is not exact")
        rows: list[list[float]] = []
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, (list, tuple)) or len(raw_row) != expected_shape[1]:
                raise ResidualEnvContractError(f"{label}[{row_index}] width is not exact")
            row: list[float] = []
            for column_index, item in enumerate(raw_row):
                if type(item) is bool:
                    raise ResidualEnvContractError(
                        f"{label}[{row_index},{column_index}] must not be bool"
                    )
                try:
                    number = float(item.item() if callable(getattr(item, "item", None)) else item)
                except Exception as exc:
                    raise ResidualEnvContractError(
                        f"{label}[{row_index},{column_index}] is not numeric: {type(exc).__name__}: {exc}"
                    ) from exc
                if not math.isfinite(number):
                    raise ResidualEnvContractError(
                        f"{label}[{row_index},{column_index}] is non-finite"
                    )
                row.append(number)
            rows.append(row)
        return rows

    def vector_values(value: Any, expected_length: int, label: str) -> list[float]:
        raw_shape = getattr(value, "shape", None)
        try:
            shape = tuple(int(item) for item in raw_shape)
        except Exception as exc:
            raise ResidualEnvContractError(f"{label} shape is unavailable: {type(exc).__name__}: {exc}") from exc
        if shape != (expected_length,):
            raise ResidualEnvContractError(f"{label} shape {shape} != {(expected_length,)}")
        try:
            detached = value.detach() if callable(getattr(value, "detach", None)) else value
            host = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
            raw_values = host.tolist() if callable(getattr(host, "tolist", None)) else list(host)
        except Exception as exc:
            raise ResidualEnvContractError(f"{label} is unreadable: {type(exc).__name__}: {exc}") from exc
        if not isinstance(raw_values, list) or len(raw_values) != expected_length:
            raise ResidualEnvContractError(f"{label} count is not exact")
        values: list[float] = []
        for index, item in enumerate(raw_values):
            if type(item) is bool:
                raise ResidualEnvContractError(f"{label}[{index}] must not be bool")
            try:
                number = float(item.item() if callable(getattr(item, "item", None)) else item)
            except Exception as exc:
                raise ResidualEnvContractError(
                    f"{label}[{index}] is not numeric: {type(exc).__name__}: {exc}"
                ) from exc
            if not math.isfinite(number):
                raise ResidualEnvContractError(f"{label}[{index}] is non-finite")
            values.append(number)
        return values

    try:
        count = _exact_int(num_envs, "PhysX readback num_envs")
        if count <= 0:
            raise ResidualEnvContractError("PhysX readback num_envs must be positive")
        row_index = _exact_int(env_index, "PhysX readback env_index")
        if row_index >= count:
            raise ResidualEnvContractError("PhysX readback env_index is out of range")
        names_value = getattr(robot, "joint_names", None)
        if (
            isinstance(names_value, (str, bytes, bytearray))
            or not isinstance(names_value, Sequence)
            or not names_value
        ):
            raise ResidualEnvContractError("robot.joint_names is unavailable or empty")
        joint_names = tuple(str(name) for name in names_value)
        if len(set(joint_names)) != len(joint_names):
            raise ResidualEnvContractError("robot.joint_names contains duplicates")

        def exact_ids(value: Sequence[int], expected_names: Sequence[str], label: str) -> tuple[int, ...]:
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
                raise ResidualEnvContractError(f"{label} must be an exact ID sequence")
            if len(value) != len(expected_names):
                raise ResidualEnvContractError(f"{label} count is not exact")
            ids = tuple(_exact_int(item, f"{label}[{index}]") for index, item in enumerate(value))
            if len(set(ids)) != len(ids) or any(item >= len(joint_names) for item in ids):
                raise ResidualEnvContractError(f"{label} contains duplicate or out-of-range IDs")
            resolved = tuple(joint_names[item] for item in ids)
            if resolved != tuple(expected_names):
                raise ResidualEnvContractError(f"{label} does not resolve the exact canonical names")
            return ids

        servo_ids = exact_ids(servo_joint_ids, SERVO_JOINT_NAMES, "servo_joint_ids")
        wheel_ids = exact_ids(wheel_joint_ids, WHEEL_JOINT_NAMES, "wheel_joint_ids")
        if set(servo_ids) & set(wheel_ids):
            raise ResidualEnvContractError("servo/wheel joint IDs overlap")

        view = getattr(robot, "root_physx_view", None)
        position_getter = None if view is None else getattr(view, "get_dof_position_targets", None)
        velocity_getter = None if view is None else getattr(view, "get_dof_velocity_targets", None)
        if not callable(position_getter) or not callable(velocity_getter):
            raise ResidualEnvContractError("root_physx_view target getters are unavailable")
        try:
            raw_position_targets = position_getter()
            raw_velocity_targets = velocity_getter()
        except Exception as exc:
            raise ResidualEnvContractError(
                f"root_physx_view target getter failed: {type(exc).__name__}: {exc}"
            ) from exc

        full_shape = (count, len(joint_names))
        position_targets = matrix_rows(
            raw_position_targets,
            full_shape,
            "root_physx_view.get_dof_position_targets",
        )
        velocity_targets = matrix_rows(
            raw_velocity_targets,
            full_shape,
            "root_physx_view.get_dof_velocity_targets",
        )
        expected_servo = vector_values(
            expected_servo_position_targets,
            len(SERVO_JOINT_NAMES),
            "expected servo position targets",
        )
        expected_wheel = vector_values(
            expected_wheel_velocity_targets,
            len(WHEEL_JOINT_NAMES),
            "expected wheel velocity targets",
        )
        actual_servo = [position_targets[row_index][joint_id] for joint_id in servo_ids]
        actual_wheel = [velocity_targets[row_index][joint_id] for joint_id in wheel_ids]
        details = {
            "num_envs": count,
            "env_index": row_index,
            "joint_names": list(joint_names),
            "servo_joint_ids": list(servo_ids),
            "wheel_joint_ids": list(wheel_ids),
            "actual_servo_position_targets": actual_servo,
            "expected_servo_position_targets": expected_servo,
            "actual_wheel_velocity_targets": actual_wheel,
            "expected_wheel_velocity_targets": expected_wheel,
        }
        if actual_servo != expected_servo:
            return finish(
                verified=False,
                failure_reason="PhysX servo position target mismatch",
                details=details,
            )
        if actual_wheel != expected_wheel:
            return finish(
                verified=False,
                failure_reason="PhysX wheel velocity target mismatch",
                details=details,
            )
        return finish(verified=True, failure_reason="", details=details)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return finish(verified=False, failure_reason=reason, details={"exception": reason})


def slew_servo_targets_150_deg_s(
    current_command_deg: Mapping[str, Any],
    endpoint_command_deg: Mapping[str, Any],
) -> dict[str, float]:
    """Advance exact command-space drive targets by one 120 Hz physics tick.

    This is actuator plumbing, not a source-action generator.  Calling it does
    not advance a profile cursor and must not create a source ledger row.
    """

    current = _exact_finite_map(current_command_deg, SERVO_JOINT_NAMES, "current servo drive target")
    endpoint = _exact_finite_map(endpoint_command_deg, SERVO_JOINT_NAMES, "servo endpoint")
    result: dict[str, float] = {}
    limit = SERVO_MAX_DELTA_DEG_PER_PHYSICS_STEP
    for name in SERVO_JOINT_NAMES:
        delta = endpoint[name] - current[name]
        result[name] = current[name] + max(-limit, min(limit, delta))
    return result


@dataclass(frozen=True)
class PhaseResetState:
    entry_id: str
    entry_sha256: str
    bank_sha256: str
    source_version: str
    phase_state: str
    root_position_m: tuple[float, float, float]
    root_orientation_wxyz: tuple[float, float, float, float]
    root_linear_velocity_m_s: tuple[float, float, float]
    root_angular_velocity_rad_s: tuple[float, float, float]
    joint_q_rad: tuple[float, ...]
    joint_qd_rad_s: tuple[float, ...]
    nominal_servo_targets_deg: Mapping[str, float]
    nominal_wheel_targets_rad_s: Mapping[str, float]
    snapshot_sim_step: int
    snapshot_sim_time_s: float

    def __post_init__(self) -> None:
        if not self.entry_id or not self.source_version or not self.phase_state:
            raise ResidualEnvContractError("phase reset identity is incomplete")
        for label, digest in (("entry_sha256", self.entry_sha256), ("bank_sha256", self.bank_sha256)):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ResidualEnvContractError(f"{label} must be a lowercase SHA-256")
        object.__setattr__(self, "root_position_m", _finite_vector(self.root_position_m, 3, "root_position_m"))
        orientation = _finite_vector(self.root_orientation_wxyz, 4, "root_orientation_wxyz")
        norm = math.sqrt(sum(item * item for item in orientation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ResidualEnvContractError("root_orientation_wxyz must be normalized")
        object.__setattr__(self, "root_orientation_wxyz", orientation)
        object.__setattr__(self, "root_linear_velocity_m_s", _finite_vector(self.root_linear_velocity_m_s, 3, "root_linear_velocity_m_s"))
        object.__setattr__(self, "root_angular_velocity_rad_s", _finite_vector(self.root_angular_velocity_rad_s, 3, "root_angular_velocity_rad_s"))
        object.__setattr__(self, "joint_q_rad", _finite_vector(self.joint_q_rad, 12, "joint_q_rad"))
        object.__setattr__(self, "joint_qd_rad_s", _finite_vector(self.joint_qd_rad_s, 12, "joint_qd_rad_s"))
        object.__setattr__(self, "nominal_servo_targets_deg", _frozen_map(_exact_finite_map(self.nominal_servo_targets_deg, SERVO_JOINT_NAMES, "nominal_servo_targets_deg")))
        object.__setattr__(self, "nominal_wheel_targets_rad_s", _frozen_map(_exact_finite_map(self.nominal_wheel_targets_rad_s, WHEEL_JOINT_NAMES, "nominal_wheel_targets_rad_s")))
        _exact_int(self.snapshot_sim_step, "snapshot_sim_step")
        object.__setattr__(self, "snapshot_sim_time_s", _finite_number(self.snapshot_sim_time_s, "snapshot_sim_time_s", nonnegative=True))


@dataclass(frozen=True)
class PhaseEpisodePlan:
    reset: PhaseResetState
    profile: PhaseMotionProfile | None
    profile_free: bool
    profile_id: str
    profile_strategy: str
    entry_subphase: str
    source_cursor_segment_index: int
    source_cursor_delay_physics_steps: int
    source_plan_sha256: str
    segment_frames: tuple[Any, ...]

    def __post_init__(self) -> None:
        if type(self.profile_free) is not bool:
            raise ResidualEnvContractError("profile_free must be exact bool")
        if not isinstance(self.entry_subphase, str) or not self.entry_subphase:
            raise ResidualEnvContractError("entry_subphase must be non-empty text")
        _exact_int(self.source_cursor_segment_index, "source_cursor_segment_index")
        delay = _exact_int(self.source_cursor_delay_physics_steps, "source_cursor_delay_physics_steps")
        if delay % EXPECTED_RENDER_SUBSTEPS != 0:
            raise ResidualEnvContractError("bank cursor delay must lie on an exact-eight boundary")
        if not isinstance(self.source_plan_sha256, str) or len(self.source_plan_sha256) != 64:
            raise ResidualEnvContractError("source_plan_sha256 is invalid")
        if self.profile_free:
            if self.profile is not None or self.segment_frames or self.profile_id or self.profile_strategy:
                raise ResidualEnvContractError("profile-free entry must not borrow successor actions")
        else:
            if self.profile is None or not self.profile_id or not self.profile_strategy:
                raise ResidualEnvContractError("profile-backed entry lacks its exact profile")
            if self.profile.profile_id != self.profile_id:
                raise ResidualEnvContractError("profile identity differs from bank entry")
            indices = tuple(int(frame.source_segment_index) for frame in self.segment_frames)
            expected = tuple(range(self.source_cursor_segment_index, self.profile.source_segment_range[1] + 1))
            if indices != expected:
                raise ResidualEnvContractError("phase-local source cursor is not exact and contiguous")


def _validated_bank(
    bank: PhaseEntryBank | Mapping[str, Any] | None,
    *,
    verify_artifacts: bool,
) -> PhaseEntryBank:
    if bank is None:
        result = load_phase_entry_bank(verify_artifacts=verify_artifacts)
    elif isinstance(bank, PhaseEntryBank):
        result = bank
    elif isinstance(bank, Mapping):
        result = validate_phase_entry_bank_mapping(bank, verify_artifacts=verify_artifacts)
    else:
        raise ResidualEnvContractError("bank must be PhaseEntryBank, mapping, or None")
    detached = result.to_mapping()
    if phase_bank_payload_sha256(detached["payload"]) != result.bank_sha256:
        raise ResidualEnvContractError("phase-entry bank payload was tampered after validation")
    return result


def build_phase_episode_plan(
    entry_id: str,
    *,
    bank: PhaseEntryBank | Mapping[str, Any] | None = None,
    profile_library: MotionProfileLibrary | None = None,
    verify_artifacts: bool = True,
) -> PhaseEpisodePlan:
    """Resolve one immutable bank entry to its sole allowed public cursor."""

    if not isinstance(entry_id, str) or not entry_id:
        raise ResidualEnvContractError("entry_id must be non-empty text")
    resolved_bank = _validated_bank(bank, verify_artifacts=verify_artifacts)
    detached_payload = resolved_bank.to_mapping()["payload"]
    matches = [entry for entry in detached_payload["entries"] if entry.get("entry_id") == entry_id]
    if len(matches) != 1:
        raise ResidualEnvContractError(f"entry_id must resolve exactly once: {entry_id!r}")
    entry = matches[0]
    if _canonical_sha256({key: copy.deepcopy(value) for key, value in entry.items() if key != "entry_sha256"}) != entry["entry_sha256"]:
        raise ResidualEnvContractError("phase entry payload SHA is invalid")

    reset = PhaseResetState(
        entry_id=entry_id,
        entry_sha256=str(entry["entry_sha256"]),
        bank_sha256=resolved_bank.bank_sha256,
        source_version=str(entry["source_version"]),
        phase_state=str(entry["phase_state"]),
        root_position_m=tuple(entry["root_position_m"]),
        root_orientation_wxyz=tuple(entry["root_orientation_wxyz"]),
        root_linear_velocity_m_s=tuple(entry["root_linear_velocity_m_s"]),
        root_angular_velocity_rad_s=tuple(entry["root_angular_velocity_rad_s"]),
        joint_q_rad=tuple(entry["joint_q_rad"]),
        joint_qd_rad_s=tuple(entry["joint_qd_rad_s"]),
        nominal_servo_targets_deg=_map_from_vector(entry["nominal_servo_targets_deg"], SERVO_JOINT_NAMES, "entry nominal servos"),
        nominal_wheel_targets_rad_s=_map_from_vector(entry["nominal_wheel_targets_rad_s"], WHEEL_JOINT_NAMES, "entry nominal wheels"),
        snapshot_sim_step=int(entry["snapshot_sim_step"]),
        snapshot_sim_time_s=float(entry["snapshot_sim_time_s"]),
    )
    profile_row = entry["entry_profile"]
    cursor = entry["source_cursor"]
    cursor_segment = int(cursor["source_segment_index"])
    cursor_delay = int(cursor["dispatch_sim_step"]) - int(entry["snapshot_sim_step"])
    if cursor_delay < 0:
        raise ResidualEnvContractError("bank source cursor precedes its physical snapshot")
    source_plan_sha = str(cursor["source_plan_sha256"])
    profile_free = bool(profile_row["profile_free"])
    if profile_free:
        # The cursor may identify the successor for provenance, but the R1
        # episode may not execute that successor under this phase's authority.
        return PhaseEpisodePlan(
            reset=reset,
            profile=None,
            profile_free=True,
            profile_id="",
            profile_strategy="",
            entry_subphase=str(profile_row["subphase"]),
            source_cursor_segment_index=cursor_segment,
            source_cursor_delay_physics_steps=cursor_delay,
            source_plan_sha256=source_plan_sha,
            segment_frames=(),
        )

    library = profile_library or build_profile_library(REPLAY_ROOT)
    profile = library.get(
        reset.source_version,
        reset.phase_state,
        strategy=str(profile_row["profile_strategy"]),
    )
    exact = (
        profile.profile_id == profile_row["profile_id"] == cursor["profile_id"]
        and profile.source_version == reset.source_version == cursor["profile_source_version"]
        and profile.strategy == profile_row["profile_strategy"] == cursor["profile_strategy"]
        and profile.source_plan_sha256 == source_plan_sha
        and cursor["owner_state"] == reset.phase_state
    )
    if not exact:
        raise ResidualEnvContractError("bank/profile/cursor identity mismatch")
    frames = tuple(
        frame
        for frame in profile.keyframes
        if frame.dispatch_kind == "segment_start" and frame.source_segment_index >= cursor_segment
    )
    if not frames or frames[0].source_segment_index != cursor_segment:
        raise ResidualEnvContractError("bank cursor has no exact phase-local segment start")
    first = frames[0]
    if (
        tuple(first.commands) != tuple(cursor["commands"])
        or tuple(first.source_event_indices) != tuple(cursor["source_event_indices"])
        or tuple(first.servo_targets_deg[name] for name in SERVO_JOINT_NAMES) != tuple(cursor["servo_targets_deg"])
        or tuple(first.wheel_targets_rad_s[name] for name in WHEEL_JOINT_NAMES) != tuple(cursor["wheel_targets_rad_s"])
    ):
        raise ResidualEnvContractError("bank cursor payload differs from first profile action")
    return PhaseEpisodePlan(
        reset=reset,
        profile=profile,
        profile_free=False,
        profile_id=profile.profile_id,
        profile_strategy=profile.strategy,
        entry_subphase=str(profile_row["subphase"]),
        source_cursor_segment_index=cursor_segment,
        source_cursor_delay_physics_steps=cursor_delay,
        source_plan_sha256=source_plan_sha,
        segment_frames=frames,
    )


@dataclass(frozen=True)
class BatchReadback:
    batch_id: str
    applied_sim_step: int
    first_physics_step: int
    physical_command_epoch: int
    physx_evidence_source: str
    physx_evidence_sha256: str
    physx_failure_reason: str
    verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id:
            raise ResidualEnvContractError("batch readback needs a non-empty batch_id")
        applied = _exact_int(self.applied_sim_step, "batch readback applied_sim_step")
        first = _exact_int(self.first_physics_step, "batch readback first_physics_step")
        if first != applied + 1:
            raise ResidualEnvContractError("batch readback is not exact N+1")
        epoch = _exact_int(self.physical_command_epoch, "batch readback physical_command_epoch")
        if epoch <= 0:
            raise ResidualEnvContractError("batch readback physical_command_epoch must be positive")
        if self.physx_evidence_source != PHYSX_TARGET_READBACK_SOURCE:
            raise ResidualEnvContractError("batch readback does not identify low-level PhysX target evidence")
        digest = self.physx_evidence_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ResidualEnvContractError("batch readback PhysX evidence SHA-256 is invalid")
        if not isinstance(self.physx_failure_reason, str):
            raise ResidualEnvContractError("batch readback PhysX failure reason must be text")
        if type(self.verified) is not bool:
            raise ResidualEnvContractError("batch readback verified must be exact bool")
        if self.verified == bool(self.physx_failure_reason):
            raise ResidualEnvContractError(
                "batch readback PhysX verified/failure-reason identity is inconsistent"
            )


@dataclass(frozen=True)
class PhasePhysicsFeedback:
    sim_step: int
    sim_time_s: float
    telemetry: Mapping[str, Any]
    effective_servo_actual_error_deg: Mapping[str, float]
    servo_velocity_deg_s: Mapping[str, float]
    batch_readback: BatchReadback | None = None
    hard_failure: bool = False
    hard_failure_reason: str = ""

    def __post_init__(self) -> None:
        step = _exact_int(self.sim_step, "feedback sim_step")
        sim_time = _finite_number(self.sim_time_s, "feedback sim_time_s", nonnegative=True)
        if not math.isclose(sim_time, step * PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-8):
            raise ResidualEnvContractError("feedback time/step is not exact 120 Hz")
        if not isinstance(self.telemetry, Mapping):
            raise ResidualEnvContractError("feedback telemetry must be a mapping")
        object.__setattr__(self, "telemetry", copy.deepcopy(dict(self.telemetry)))
        object.__setattr__(self, "effective_servo_actual_error_deg", _frozen_map(_exact_finite_map(self.effective_servo_actual_error_deg, SERVO_JOINT_NAMES, "effective servo errors")))
        object.__setattr__(self, "servo_velocity_deg_s", _frozen_map(_exact_finite_map(self.servo_velocity_deg_s, SERVO_JOINT_NAMES, "servo velocity")))
        if self.batch_readback is not None and not isinstance(self.batch_readback, BatchReadback):
            raise ResidualEnvContractError("batch_readback must be BatchReadback or None")
        if type(self.hard_failure) is not bool:
            raise ResidualEnvContractError("hard_failure must be exact bool")
        if not isinstance(self.hard_failure_reason, str):
            raise ResidualEnvContractError("hard_failure_reason must be text")
        if self.hard_failure and not self.hard_failure_reason:
            raise ResidualEnvContractError("hard failure needs an exact reason")
        if not self.hard_failure and self.hard_failure_reason:
            raise ResidualEnvContractError("non-failure feedback cannot carry a failure reason")


@dataclass(frozen=True)
class PhaseCommandBatch:
    batch_id: str
    kind: str
    decision_sim_step: int
    expected_first_physics_step: int
    physical_command_epoch: int
    source_segment_index: int | None
    source_action: bool
    nominal_servo_targets_deg: Mapping[str, float]
    nominal_wheel_targets_rad_s: Mapping[str, float]
    applied_servo_targets_deg: Mapping[str, float]
    applied_wheel_targets_rad_s: Mapping[str, float]
    residual_transform_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id:
            raise ResidualEnvContractError("command batch_id must be non-empty")
        if self.kind not in {
            "SOURCE_ACTION",
            "RESIDUAL_ONLY",
            "WHEEL_STOP_OVERRIDE",
            "SAFE_STOP",
            "SUCCESS_STOP",
        }:
            raise ResidualEnvContractError("unsupported command batch kind")
        step = _exact_int(self.decision_sim_step, "batch decision_sim_step")
        if self.expected_first_physics_step != step + 1:
            raise ResidualEnvContractError("command batch must own exact N+1")
        physical_epoch = _exact_int(self.physical_command_epoch, "physical_command_epoch")
        if physical_epoch <= 0:
            raise ResidualEnvContractError("physical_command_epoch must be positive")
        if self.source_segment_index is not None:
            _exact_int(self.source_segment_index, "source_segment_index")
        if type(self.source_action) is not bool:
            raise ResidualEnvContractError("source_action must be exact bool")
        if (self.kind == "SOURCE_ACTION") is not self.source_action:
            raise ResidualEnvContractError("command batch kind/source_action identity is inconsistent")
        if self.source_action and self.source_segment_index is None:
            raise ResidualEnvContractError("source command batch lacks its source segment")
        for field_name, names in (
            ("nominal_servo_targets_deg", SERVO_JOINT_NAMES),
            ("nominal_wheel_targets_rad_s", WHEEL_JOINT_NAMES),
            ("applied_servo_targets_deg", SERVO_JOINT_NAMES),
            ("applied_wheel_targets_rad_s", WHEEL_JOINT_NAMES),
        ):
            object.__setattr__(self, field_name, _frozen_map(_exact_finite_map(getattr(self, field_name), names, field_name)))
        digest = self.residual_transform_sha256
        if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ResidualEnvContractError("residual_transform_sha256 is invalid")


@dataclass(frozen=True)
class PhaseKernelStepResult:
    observation: ResidualActorObservation
    reward: ResidualRewardResult
    terminated: bool
    truncated: bool
    terminal_latched: bool
    phase_success: bool
    terminal_reason: str
    command_batch: PhaseCommandBatch | None
    source_consumed_segment_index: int | None
    source_consumption_sha256: str
    completion_decision: Mapping[str, Any]
    tracking_begin_targets_deg: Mapping[str, float]
    tracking_end_targets_deg: Mapping[str, float]
    profile_fraction: float
    info: Mapping[str, Any]


@dataclass
class _PendingBatch:
    batch: PhaseCommandBatch
    wheel_stop_ack: bool = False


class PhaseLocalResidualKernel:
    """Isaac-free executor for exactly one immutable phase-local episode."""

    def __init__(
        self,
        plan: PhaseEpisodePlan,
        envelope: ResidualEnvelope,
        *,
        timeout_physics_steps: int = DEFAULT_TIMEOUT_PHYSICS_STEPS,
        scene_manifest_sha256: str | None = None,
    ) -> None:
        if not isinstance(plan, PhaseEpisodePlan):
            raise ResidualEnvContractError("plan must be PhaseEpisodePlan")
        if not isinstance(envelope, ResidualEnvelope) or not envelope.evidence_verified:
            raise ResidualEnvContractError("residual envelope must carry verified evidence")
        if not envelope.is_source_allowed(plan.reset.source_version):
            raise ResidualEnvContractError("phase source is outside the frozen R1 envelope")
        self.plan = plan
        self.envelope = envelope
        self.scene_manifest_sha256 = (
            load_formal_scene_spec(verify_files=True).manifest_sha256
            if scene_manifest_sha256 is None
            else str(scene_manifest_sha256)
        )
        if (
            len(self.scene_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.scene_manifest_sha256)
        ):
            raise ResidualEnvContractError("scene_manifest_sha256 must be a lowercase SHA-256")
        self.timeout_physics_steps = _exact_int(timeout_physics_steps, "timeout_physics_steps")
        if self.timeout_physics_steps < EXPECTED_RENDER_SUBSTEPS:
            raise ResidualEnvContractError("timeout must span at least one outer cycle")
        self._graph_state = build_default_macro_graph().get(plan.reset.phase_state)
        self._completion = CompletionAwareSegmentExecutor()
        self.reset()

    @classmethod
    def from_current_artifacts(
        cls,
        entry_id: str,
        *,
        timeout_physics_steps: int = DEFAULT_TIMEOUT_PHYSICS_STEPS,
    ) -> "PhaseLocalResidualKernel":
        return cls(
            build_phase_episode_plan(entry_id),
            load_residual_envelope(verify_evidence=True),
            timeout_physics_steps=timeout_physics_steps,
        )

    @property
    def reset_state(self) -> PhaseResetState:
        return self.plan.reset

    @property
    def current_nominal_servo_targets_deg(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._nominal_servos))

    @property
    def current_nominal_wheel_targets_rad_s(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._nominal_wheels))

    @property
    def current_applied_servo_targets_deg(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._applied_servos))

    @property
    def current_applied_wheel_targets_rad_s(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._applied_wheels))

    @property
    def previous_applied_residual(self) -> tuple[float, ...]:
        return self._previous_residual

    @property
    def profile_fraction(self) -> float:
        if self.plan.profile_free:
            return 1.0
        return len(self._completed_segments) / len(self.plan.segment_frames)

    @property
    def terminal_latched(self) -> bool:
        return self._terminal_latched

    @property
    def physical_command_epoch(self) -> int:
        return self._physical_command_epoch

    @property
    def active_completion_effective_targets_deg(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._active_effective_sparse_targets))

    @property
    def active_completion_latched_servo_residual_deg(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._active_completion_latched_servo_residual_deg))

    def reset(self) -> PhaseResetState:
        self._next_frame = 0
        self._active_frame: Any | None = None
        self._active_binding: Any | None = None
        self._active_effective_sparse_targets: dict[str, float] = {}
        self._active_completion_latched_servo_residual_deg: dict[str, float] = {}
        self._completed_segments: list[int] = []
        self._nominal_servos = dict(self.plan.reset.nominal_servo_targets_deg)
        self._nominal_wheels = dict(self.plan.reset.nominal_wheel_targets_rad_s)
        self._applied_servos = dict(self._nominal_servos)
        self._applied_wheels = dict(self._nominal_wheels)
        self._previous_residual = ZERO_RESIDUAL_ACTION
        self._held_action = ZERO_RESIDUAL_ACTION
        self._previous_policy_action = ZERO_RESIDUAL_ACTION
        self._pending_batch: _PendingBatch | None = None
        self._last_sim_step: int | None = None
        self._last_batch_decision_step: int | None = None
        self._physics_callback_count = 0
        self._outer_boundary_count = 0
        self._source_consumption_count = 0
        self._residual_transform_count = 0
        self._physical_batch_count = 0
        self._physical_command_epoch = 0
        self._last_verified_physical_command_epoch = 0
        self._n_plus_one_verified_count = 0
        self._last_residual_transform_sha256 = ""
        self._dynamic_wheel_stop_count = 0
        self._peak_abs_roll_rad = 0.0
        self._peak_abs_pitch_rad = 0.0
        self._peak_root_angular_speed_rad_s = 0.0
        self._peak_abs_servo_error_deg = 0.0
        self._peak_residual_l2 = 0.0
        self._peak_residual_slew_l2 = 0.0
        self._reward_return = 0.0
        self._completion_time_s: float | None = None
        self._terminal_latched = False
        self._terminal_ready = False
        self._terminal_truncated = False
        self._phase_success = False
        self._terminal_reason = ""
        self._safe_stop_kind = ""
        self._deferred_terminal_stop_kind = ""
        self._completion.reset()
        self._entry_base_xy: tuple[float, float] | None = None
        self._entry_target_unit_xy: tuple[float, float] | None = None
        self._airborne_before_cross = {leg: False for leg in LEG_NAMES}
        self._crossed_seen = {leg: False for leg in LEG_NAMES}
        self._top_seen = {leg: False for leg in LEG_NAMES}
        self._last_contact_classes = {leg: "AIR" for leg in LEG_NAMES}
        return self.plan.reset

    def _validate_cadence(self, context: OuterCycleContext, feedback: PhasePhysicsFeedback) -> None:
        if not isinstance(context, OuterCycleContext):
            raise ResidualEnvContractError("context must be OuterCycleContext")
        expected_step = 0 if self._last_sim_step is None else self._last_sim_step + 1
        if feedback.sim_step != expected_step:
            raise ResidualEnvContractError(
                f"physics callbacks must be consecutive: expected {expected_step}, got {feedback.sim_step}"
            )
        expected_outer, expected_substep = divmod(feedback.sim_step, EXPECTED_RENDER_SUBSTEPS)
        if (context.outer_cycle_index, context.physics_substep_index) != (expected_outer, expected_substep):
            raise ResidualEnvContractError("outer-cycle context differs from exact 120/15 Hz cadence")
        self._last_sim_step = feedback.sim_step

    def _validate_or_ack_pending(self, feedback: PhasePhysicsFeedback) -> bool:
        pending = self._pending_batch
        readback = feedback.batch_readback
        if pending is None:
            if readback is not None:
                raise ResidualEnvContractError("stale or unsolicited batch readback")
            return False
        batch = pending.batch
        if feedback.sim_step != batch.expected_first_physics_step:
            raise ResidualEnvContractError("pending batch was not checked on exact N+1")
        if readback is None:
            raise ResidualEnvContractError("pending batch has no N+1 readback")
        if (
            readback.batch_id != batch.batch_id
            or readback.applied_sim_step != batch.decision_sim_step
            or readback.first_physics_step != batch.expected_first_physics_step
            or readback.physical_command_epoch != batch.physical_command_epoch
        ):
            raise ResidualEnvContractError("pending batch N+1 readback identity failed")
        if readback.verified is not True:
            raise ResidualEnvContractError(
                "pending batch low-level PhysX N+1 target verification failed: "
                + readback.physx_failure_reason
            )
        if pending.wheel_stop_ack:
            self._completion.acknowledge_wheel_stop(
                applied_sim_step=batch.decision_sim_step,
                first_physics_step=batch.expected_first_physics_step,
                batch_id=batch.batch_id,
            )
        self._pending_batch = None
        self._last_verified_physical_command_epoch = batch.physical_command_epoch
        self._n_plus_one_verified_count += 1
        if (
            self._terminal_latched
            and not self._deferred_terminal_stop_kind
            and pending.batch.kind == self._safe_stop_kind
        ):
            self._terminal_ready = True
        return True

    def _updated_telemetry(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        telemetry = copy.deepcopy(dict(raw))
        telemetry.update(
            {
                "source_version": self.plan.reset.source_version,
                "macro_state": self.plan.reset.phase_state,
                "profile_fraction": self.profile_fraction,
                "active_leg": ACTIVE_LEG_BY_STATE[self.plan.reset.phase_state],
                "servo_targets_deg": dict(self._nominal_servos),
                "wheel_targets_rad_s": dict(self._nominal_wheels),
            }
        )
        return telemetry

    def _update_guard_latches(self, telemetry: Mapping[str, Any]) -> bool:
        classes = telemetry.get("wheel_contact_classes")
        face = telemetry.get("wheel_front_face_clearance_m")
        centers = telemetry.get("wheel_center_w_m")
        if not isinstance(classes, Mapping) or set(classes) != set(LEG_NAMES):
            raise ResidualEnvContractError("guard wheel_contact_classes are not exact")
        if not isinstance(face, Mapping) or set(face) != set(LEG_NAMES):
            raise ResidualEnvContractError("guard wheel_front_face_clearance_m is not exact")
        if not isinstance(centers, Mapping) or set(centers) != set(LEG_NAMES):
            raise ResidualEnvContractError("guard wheel_center_w_m is not exact")
        base = telemetry.get("base_position_m")
        if not isinstance(base, Mapping) or set(base) != {"x", "y", "z"}:
            raise ResidualEnvContractError("guard base_position_m is not exact")
        base_xy = (
            _finite_number(base["x"], "base_position_m.x"),
            _finite_number(base["y"], "base_position_m.y"),
        )
        if self._entry_base_xy is None:
            self._entry_base_xy = base_xy
            fl = _finite_vector(centers["FL"], 3, "wheel_center_w_m.FL")
            dx, dy = fl[0] - base_xy[0], fl[1] - base_xy[1]
            norm = math.hypot(dx, dy)
            self._entry_target_unit_xy = None if norm <= 1.0e-12 else (dx / norm, dy / norm)
        for leg in LEG_NAMES:
            contact = str(classes[leg])
            if contact not in {"AIR", "GROUND", "FRONT_FACE", "TOP"}:
                raise ResidualEnvContractError(f"unsupported guard contact class for {leg}: {contact!r}")
            clearance = _finite_number(face[leg], f"wheel_front_face_clearance_m.{leg}")
            if contact == "AIR" and not self._crossed_seen[leg]:
                self._airborne_before_cross[leg] = True
            if clearance > 0.0:
                self._crossed_seen[leg] = True
            if contact == "TOP":
                self._top_seen[leg] = True
            self._last_contact_classes[leg] = contact

        roll = abs(_finite_number(telemetry.get("base_roll_rad"), "base_roll_rad"))
        pitch = abs(_finite_number(telemetry.get("base_pitch_rad"), "base_pitch_rad"))
        angular = _finite_vector(
            telemetry.get("root_angular_velocity_w"),
            3,
            "root_angular_velocity_w",
        )
        canonical_errors = _exact_finite_map(
            telemetry.get("canonical_servo_actual_error_deg"),
            SERVO_JOINT_NAMES,
            "canonical_servo_actual_error_deg",
        )
        self._peak_abs_roll_rad = max(self._peak_abs_roll_rad, roll)
        self._peak_abs_pitch_rad = max(self._peak_abs_pitch_rad, pitch)
        self._peak_root_angular_speed_rad_s = max(
            self._peak_root_angular_speed_rad_s,
            math.sqrt(sum(value * value for value in angular)),
        )
        self._peak_abs_servo_error_deg = max(
            self._peak_abs_servo_error_deg,
            max(abs(value) for value in canonical_errors.values()),
        )
        guard = self._graph_state.completion_guard
        if roll > guard.maximum_abs_roll_rad or pitch > guard.maximum_abs_pitch_rad:
            return False
        kind = guard.kind
        if kind == MacroGuardKind.COM_SHIFT_OR_UNLOAD:
            projected = 0.0
            if self._entry_base_xy is not None and self._entry_target_unit_xy is not None:
                projected = (
                    (base_xy[0] - self._entry_base_xy[0]) * self._entry_target_unit_xy[0]
                    + (base_xy[1] - self._entry_base_xy[1]) * self._entry_target_unit_xy[1]
                )
            return bool(self._airborne_before_cross["RR"] or projected >= guard.minimum_com_displacement_m)
        if kind == MacroGuardKind.SUPPORT_SETUP:
            return all(self._last_contact_classes[leg] != "AIR" for leg in guard.required_support_legs)
        if kind == MacroGuardKind.LEG_TRAVERSED:
            leg = str(guard.active_leg or "")
            return bool(
                leg
                and self._crossed_seen[leg]
                and self._top_seen[leg]
                and (self._airborne_before_cross[leg] or not guard.require_airborne_before_crossing)
                and all(self._top_seen[item] for item in guard.required_top_legs)
            )
        if kind == MacroGuardKind.POSTURE_RECOVERED:
            return bool(
                telemetry.get("final_recoverable") is True
                and (telemetry.get("body_crossed_front_face") is True if guard.require_body_crossed else True)
            )
        raise ResidualEnvContractError(f"unsupported phase-local guard kind: {kind}")

    def _profile_complete(self) -> bool:
        return bool(
            self.plan.profile_free
            or (
                self._active_binding is None
                and self._next_frame == len(self.plan.segment_frames)
                and len(self._completed_segments) == len(self.plan.segment_frames)
            )
        )

    def _make_batch(
        self,
        *,
        kind: str,
        sim_step: int,
        source_segment_index: int | None,
        source_action: bool,
        residual_sha: str = "",
    ) -> PhaseCommandBatch:
        if self._last_batch_decision_step == sim_step:
            raise ResidualEnvContractError("more than one 8+4 batch was requested in one physics tick")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entry_id": self.plan.reset.entry_id,
            "kind": kind,
            "decision_sim_step": sim_step,
            "source_segment_index": source_segment_index,
            "source_action": source_action,
            "nominal_servo_targets_deg": self._nominal_servos,
            "nominal_wheel_targets_rad_s": self._nominal_wheels,
            "applied_servo_targets_deg": self._applied_servos,
            "applied_wheel_targets_rad_s": self._applied_wheels,
            "residual_transform_sha256": residual_sha,
            "physical_command_epoch": self._physical_command_epoch + 1,
        }
        next_physical_epoch = self._physical_command_epoch + 1
        batch = PhaseCommandBatch(
            batch_id=f"fsm50-r1:{_canonical_sha256(payload)[:24]}:{sim_step:08d}",
            kind=kind,
            decision_sim_step=sim_step,
            expected_first_physics_step=sim_step + 1,
            physical_command_epoch=next_physical_epoch,
            source_segment_index=source_segment_index,
            source_action=source_action,
            nominal_servo_targets_deg=self._nominal_servos,
            nominal_wheel_targets_rad_s=self._nominal_wheels,
            applied_servo_targets_deg=self._applied_servos,
            applied_wheel_targets_rad_s=self._applied_wheels,
            residual_transform_sha256=residual_sha,
        )
        self._last_batch_decision_step = sim_step
        self._physical_batch_count += 1
        self._physical_command_epoch = next_physical_epoch
        return batch

    def _latch_terminal(
        self,
        *,
        success: bool,
        truncated: bool,
        reason: str,
        sim_step: int,
    ) -> PhaseCommandBatch | None:
        if self._terminal_latched:
            return None
        self._terminal_latched = True
        self._phase_success = bool(success)
        self._terminal_truncated = bool(truncated)
        self._terminal_reason = str(reason)
        self._completion_time_s = sim_step * PHYSICS_DT_S
        kind = "SUCCESS_STOP" if success else "SAFE_STOP"
        wheels_changed = any(value != 0.0 for value in self._applied_wheels.values())
        if wheels_changed or not success:
            self._safe_stop_kind = kind
            if self._last_batch_decision_step == sim_step:
                # A source or wheel-stop batch already owns this callback.  A
                # fail-closed terminal stop may not become a second batch; it
                # is emitted after that batch's exact N+1 verification.
                self._deferred_terminal_stop_kind = kind
                return None
            self._nominal_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
            self._applied_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
            batch = self._make_batch(
                kind=kind,
                sim_step=sim_step,
                source_segment_index=None,
                source_action=False,
            )
            self._pending_batch = _PendingBatch(batch=batch)
            return batch
        self._nominal_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self._applied_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self._terminal_ready = True
        return None

    def _current_subphase(self) -> str:
        if self._active_frame is not None:
            return str(self._active_frame.subphase.value)
        if self._next_frame < len(self.plan.segment_frames):
            return str(self.plan.segment_frames[self._next_frame].subphase.value)
        if self.plan.segment_frames:
            return str(self.plan.segment_frames[-1].subphase.value)
        return self.plan.entry_subphase

    def _compose_current_nominal(
        self,
        action: tuple[float, ...],
        *,
        subphase: str,
        latched_servo_residual_deg: Mapping[str, float] | None = None,
        force_zero_wheels: bool = False,
    ) -> tuple[Any, bool]:
        """Run the sole composer for one permitted 15 Hz policy decision."""

        if self._pending_batch is not None:
            raise ResidualEnvContractError("residual transform cannot replace a pending N+1 readback")
        strategy = self.plan.profile_strategy or NO_ACTIVE_PROFILE_STRATEGY
        latch = (
            dict(self._active_completion_latched_servo_residual_deg)
            if latched_servo_residual_deg is None
            else dict(latched_servo_residual_deg)
        )
        if self._active_binding is None:
            if latch:
                raise ResidualEnvContractError("completion residual latch exists without an active segment")
        elif set(latch) != set(self._active_effective_sparse_targets):
            raise ResidualEnvContractError("active completion residual latch keys drifted")
        contract = self.envelope.phase_contract(
            source_version=self.plan.reset.source_version,
            profile_strategy=strategy,
            macro_state=self.plan.reset.phase_state,
            subphase=subphase,
            nominal_servo_targets_deg=self._nominal_servos,
            nominal_wheel_targets_rad_s=self._nominal_wheels,
            # The current nominal map remains bound to its reviewed source
            # action even when this physical epoch is residual-only.
            decision_provenance="SOURCE_ACTION",
        )
        previous_servos = dict(self._applied_servos)
        previous_wheels = dict(self._applied_wheels)
        transform = compose_direct_command_residual(
            ResidualTransformInput(
                source_version=self.plan.reset.source_version,
                profile_strategy=strategy,
                macro_state=self.plan.reset.phase_state,
                subphase=subphase,
                nominal_servo_targets_deg=self._nominal_servos,
                nominal_wheel_targets_rad_s=self._nominal_wheels,
                normalized_action=action,
                previous_applied_residual=self._previous_residual,
                decision_dt_s=POLICY_DT_S,
                maximum_wheel_speed_rad_s=DEFAULT_MAX_WHEEL_SPEED_RAD_S,
                latched_servo_residual_deg=latch,
                force_zero_wheels=force_zero_wheels,
            ),
            contract,
        )
        applied_servos = dict(transform.applied_servo_targets_deg)
        applied_wheels = dict(transform.applied_wheel_targets_rad_s)
        if self._active_binding is not None:
            for name, target in self._active_effective_sparse_targets.items():
                if applied_servos[name] != target:
                    raise ResidualEnvContractError(
                        f"active completion effective target drifted during policy update: {name}"
                    )
        self._applied_servos = applied_servos
        self._applied_wheels = applied_wheels
        self._previous_residual = tuple(transform.applied_residual)
        self._residual_transform_count += 1
        self._last_residual_transform_sha256 = transform.sha256
        previous_l2 = math.sqrt(sum(value * value for value in transform.previous_applied_residual))
        current_l2 = math.sqrt(sum(value * value for value in transform.applied_residual))
        slew_l2 = math.sqrt(
            sum(
                (current - previous) ** 2
                for current, previous in zip(
                    transform.applied_residual,
                    transform.previous_applied_residual,
                )
            )
        )
        self._peak_residual_l2 = max(self._peak_residual_l2, current_l2, previous_l2)
        self._peak_residual_slew_l2 = max(self._peak_residual_slew_l2, slew_l2)
        changed = bool(applied_servos != previous_servos or applied_wheels != previous_wheels)
        return transform, changed

    def _apply_policy_update(
        self,
        action: tuple[float, ...],
        feedback: PhasePhysicsFeedback,
        *,
        kind: str = "RESIDUAL_ONLY",
        require_dispatch: bool = False,
        wheel_stop_ack: bool = False,
    ) -> tuple[PhaseCommandBatch | None, str]:
        transform, physical_changed = self._compose_current_nominal(
            action,
            subphase=self._current_subphase(),
            force_zero_wheels=wheel_stop_ack,
        )
        batch: PhaseCommandBatch | None = None
        if physical_changed or require_dispatch:
            batch = self._make_batch(
                kind=kind,
                sim_step=feedback.sim_step,
                source_segment_index=(
                    None if self._active_binding is None else self._active_binding.segment_index
                ),
                source_action=False,
                residual_sha=transform.sha256,
            )
            self._pending_batch = _PendingBatch(batch=batch, wheel_stop_ack=wheel_stop_ack)
        return batch, transform.sha256

    def _consume_next_source(
        self,
        action: tuple[float, ...],
        feedback: PhasePhysicsFeedback,
    ) -> tuple[PhaseCommandBatch | None, int, str, Mapping[str, float], str]:
        if self.plan.profile_free or self.plan.profile is None:
            raise ResidualEnvContractError("profile-free phase attempted to consume a successor action")
        if self._active_binding is not None or self._next_frame >= len(self.plan.segment_frames):
            raise ResidualEnvContractError("source cursor is not ready")
        frame = self.plan.segment_frames[self._next_frame]
        expected_segment = self.plan.source_cursor_segment_index + self._next_frame
        if frame.source_segment_index != expected_segment:
            raise ResidualEnvContractError("source cursor order drifted")
        binding = self.plan.profile.segment_binding(expected_segment)
        if binding.segment_index != frame.source_segment_index:
            raise ResidualEnvContractError("source frame/completion binding mismatch")

        previous_applied_servos = dict(self._applied_servos)
        nominal_servos = dict(frame.servo_targets_deg)
        nominal_wheels = dict(frame.wheel_targets_rad_s)
        self._nominal_servos = nominal_servos
        self._nominal_wheels = nominal_wheels
        transform, physical_changed = self._compose_current_nominal(
            action,
            subphase=frame.subphase.value,
            # A new segment establishes a new sparse completion latch.
            latched_servo_residual_deg={},
        )

        spec = binding.completion_spec
        effective_sparse = {name: self._applied_servos[name] for name in spec.servo_targets_deg}
        maximum_delta = max(
            [abs(effective_sparse[name] - previous_applied_servos[name]) for name in effective_sparse]
            or [0.0]
        )
        dynamic_duration = maximum_delta / SERVO_REFERENCE_VELOCITY_DEG_S
        self._completion.start(
            spec,
            start_elapsed_s=feedback.sim_time_s,
            start_sim_time_s=feedback.sim_time_s,
            start_sim_step=feedback.sim_step,
            servo_duration_s_override=dynamic_duration,
        )
        self._active_frame = frame
        self._active_binding = binding
        self._active_effective_sparse_targets = effective_sparse
        self._active_completion_latched_servo_residual_deg = {
            name: effective_sparse[name] - float(spec.servo_targets_deg[name])
            for name in effective_sparse
        }
        self._next_frame += 1

        source_payload = {
            "entry_id": self.plan.reset.entry_id,
            "source_version": self.plan.reset.source_version,
            "profile_id": self.plan.profile_id,
            "source_plan_sha256": self.plan.source_plan_sha256,
            "source_segment_index": frame.source_segment_index,
            "source_step_index": frame.source_step_index,
            "source_event_indices": list(frame.source_event_indices),
            "commands": list(frame.commands),
            "decision_sim_step": feedback.sim_step,
            "nominal_servo_targets_deg": nominal_servos,
            "nominal_wheel_targets_rad_s": nominal_wheels,
            "applied_servo_targets_deg": self._applied_servos,
            "applied_wheel_targets_rad_s": self._applied_wheels,
            "core_transform_sha256": transform.sha256,
            "zero_identity": transform.zero_identity,
            "physical_dispatch": physical_changed,
            "physical_command_epoch": self._physical_command_epoch + int(physical_changed),
        }
        consumption_sha = _canonical_sha256(source_payload)
        self._source_consumption_count += 1
        batch: PhaseCommandBatch | None = None
        if physical_changed:
            batch = self._make_batch(
                kind="SOURCE_ACTION",
                sim_step=feedback.sim_step,
                source_segment_index=frame.source_segment_index,
                source_action=True,
                residual_sha=transform.sha256,
            )
            self._pending_batch = _PendingBatch(batch=batch)
        tracking_begin = effective_sparse
        return batch, frame.source_segment_index, consumption_sha, tracking_begin, transform.sha256

    def _observe_completion(
        self,
        feedback: PhasePhysicsFeedback,
    ) -> tuple[
        Mapping[str, Any],
        PhaseCommandBatch | None,
        Mapping[str, float],
        bool,
    ]:
        if self._active_binding is None:
            return {}, None, {}, False
        sparse_names = tuple(self._active_effective_sparse_targets)
        decision = self._completion.observe(
            SegmentFeedback(
                elapsed_s=feedback.sim_time_s,
                sim_time_s=feedback.sim_time_s,
                sim_step=feedback.sim_step,
                servo_errors_deg={name: feedback.effective_servo_actual_error_deg[name] for name in sparse_names},
                servo_velocity_deg_s={name: feedback.servo_velocity_deg_s[name] for name in sparse_names},
                tracking_evidence={"source": "phase_local_measured_actual_space"},
            )
        )
        mapping = decision.to_mapping()
        if decision.kind == SegmentDecisionKind.FAIL:
            batch = self._latch_terminal(
                success=False,
                truncated=False,
                reason=f"completion:{decision.failure_reason}:{decision.failure_code}",
                sim_step=feedback.sim_step,
            )
            return mapping, batch, {}, False
        if decision.kind == SegmentDecisionKind.WHEEL_STOP_DUE:
            if not any(value != 0.0 for value in self._applied_wheels.values()):
                raise ResidualEnvContractError("completion requested wheel stop for zero wheels")
            self._nominal_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
            # The policy transform below is coalesced into this exact stop
            # epoch, with wheel residuals forcibly zeroed.
            return mapping, None, {}, True
        if decision.kind == SegmentDecisionKind.COMPLETE:
            segment = int(self._active_binding.segment_index)
            if segment in self._completed_segments:
                raise ResidualEnvContractError("source segment completed twice")
            self._completed_segments.append(segment)
            tracking_end = dict(self._active_effective_sparse_targets)
            self._active_frame = None
            self._active_binding = None
            self._active_effective_sparse_targets = {}
            self._active_completion_latched_servo_residual_deg = {}
            return mapping, None, tracking_end, False
        return mapping, None, {}, False

    def _reward(
        self,
        telemetry: Mapping[str, Any],
        action: tuple[float, ...],
        *,
        phase_success: bool,
        hard_failure: bool,
    ) -> ResidualRewardResult:
        classes = dict(telemetry["wheel_contact_classes"])
        active = ACTIVE_LEG_BY_STATE[self.plan.reset.phase_state]
        servo_errors = _exact_finite_map(
            telemetry["canonical_servo_actual_error_deg"],
            SERVO_JOINT_NAMES,
            "canonical_servo_actual_error_deg",
        )
        reward_input = ResidualRewardInput(
            macro_state=self.plan.reset.phase_state,
            base_roll_rad=telemetry["base_roll_rad"],
            base_pitch_rad=telemetry["base_pitch_rad"],
            root_linear_velocity_w=tuple(telemetry["root_linear_velocity_w"]),
            root_angular_velocity_w=tuple(telemetry["root_angular_velocity_w"]),
            normalized_action=action,
            previous_normalized_action=self._previous_policy_action,
            geometry_support_candidate_count=int(telemetry["geometry_support_candidate_count"]),
            max_servo_endpoint_error_deg=max(abs(value) for value in servo_errors.values()),
            active_leg_airborne_event=bool(active and self._airborne_before_cross[active]),
            active_leg_front_face_cross_event=bool(active and self._crossed_seen[active]),
            active_leg_top_event=bool(active and self._top_seen[active]),
            body_cross_event=telemetry["body_crossed_front_face"] is True,
            next_phase_completed=phase_success,
            all_wheels_top=all(classes[leg] == "TOP" for leg in LEG_NAMES),
            posture_stable=str(telemetry.get("stability_state", "")) == "stable",
            fsm_success=bool(phase_success and self.plan.reset.phase_state == "S10_POSTURE_RECOVERY"),
            hard_failure=hard_failure,
        )
        return compute_residual_reward(reward_input)

    def metrics_snapshot(self, telemetry: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the stable, exact-key per-environment evaluation record."""

        if not isinstance(telemetry, Mapping):
            raise ResidualEnvContractError("metrics telemetry must be a mapping")
        classes = telemetry.get("wheel_contact_classes")
        if not isinstance(classes, Mapping) or set(classes) != set(LEG_NAMES):
            raise ResidualEnvContractError("metrics wheel contact classes are not exact")
        active = ACTIVE_LEG_BY_STATE[self.plan.reset.phase_state]
        profile_sha = "" if self.plan.profile is None else self.plan.profile.sha256
        metrics = {
            "schema_version": EPISODE_METRICS_SCHEMA_VERSION,
            "env_source_sha256": env_source_sha256(),
            "scene_manifest_sha256": self.scene_manifest_sha256,
            "bank_sha256": self.plan.reset.bank_sha256,
            "entry_sha256": self.plan.reset.entry_sha256,
            "entry_id": self.plan.reset.entry_id,
            "source_version": self.plan.reset.source_version,
            "phase_state": self.plan.reset.phase_state,
            "profile_id": self.plan.profile_id,
            "profile_sha256": profile_sha,
            "source_plan_sha256": self.plan.source_plan_sha256,
            "residual_envelope_sha256": self.envelope.canonical_sha256,
            "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
            "task_success": self._phase_success,
            "hard_failure": bool(self._terminal_latched and not self._phase_success),
            "terminal_reason": self._terminal_reason,
            "body_crossed_front_face": telemetry.get("body_crossed_front_face") is True,
            "required_leg_lift_seen": bool(active and self._airborne_before_cross[active]),
            "final_recoverable": telemetry.get("final_recoverable") is True,
            "posture_complete": bool(
                str(telemetry.get("stability_state", "")) == "stable"
                and all(str(classes[leg]) == "TOP" for leg in LEG_NAMES)
            ),
            "completion_time_s": self._completion_time_s,
            "return": self._reward_return,
            "physics_length": self._physics_callback_count,
            "outer_policy_length": self._outer_boundary_count,
            "peak_abs_roll_rad": self._peak_abs_roll_rad,
            "peak_abs_pitch_rad": self._peak_abs_pitch_rad,
            "peak_root_angular_speed_rad_s": self._peak_root_angular_speed_rad_s,
            "peak_abs_servo_error_deg": self._peak_abs_servo_error_deg,
            "peak_residual_l2": self._peak_residual_l2,
            "peak_residual_slew_l2": self._peak_residual_slew_l2,
            "source_consumption_count": self._source_consumption_count,
            "residual_transform_count": self._residual_transform_count,
            "physical_batch_count": self._physical_batch_count,
            "physical_command_epoch": self._physical_command_epoch,
            "last_verified_physical_command_epoch": self._last_verified_physical_command_epoch,
            "n_plus_one_verified_count": self._n_plus_one_verified_count,
            "dynamic_wheel_stop_count": self._dynamic_wheel_stop_count,
            "exact_eight_cadence_verified": True,
            "source_cursor_boundary_verified": True,
            "one_batch_per_tick_verified": True,
            "n_plus_one_verified": bool(
                self._pending_batch is None
                and self._n_plus_one_verified_count == self._physical_batch_count
            ),
            "terminal_latched": self._terminal_latched,
            "terminal_ready": self._terminal_ready,
        }
        if tuple(metrics) != EPISODE_METRICS_FIELDS:
            raise ResidualEnvContractError("episode metrics keys drifted")
        return MappingProxyType(metrics)

    def step(
        self,
        action: Sequence[float],
        context: OuterCycleContext,
        feedback: PhasePhysicsFeedback,
    ) -> PhaseKernelStepResult:
        """Consume one completed 120 Hz callback and optionally emit one batch."""

        if not isinstance(feedback, PhasePhysicsFeedback):
            raise ResidualEnvContractError("feedback must be PhasePhysicsFeedback")
        normalized_action = _finite_action(action)
        self._validate_cadence(context, feedback)
        self._physics_callback_count += 1
        if context.source_cursor_permit:
            self._outer_boundary_count += 1
        command_batch: PhaseCommandBatch | None = None
        consumed_segment: int | None = None
        consumption_sha = ""
        completion_mapping: Mapping[str, Any] = {}
        tracking_begin: Mapping[str, float] = {}
        tracking_end: Mapping[str, float] = {}
        residual_transform_sha = ""
        n_plus_one_verified_this_step = False
        try:
            n_plus_one_verified_this_step = self._validate_or_ack_pending(feedback)
            if self._terminal_latched and self._deferred_terminal_stop_kind:
                if self._pending_batch is not None:
                    raise ResidualEnvContractError("deferred terminal stop still has an unverified predecessor")
                kind = self._deferred_terminal_stop_kind
                self._deferred_terminal_stop_kind = ""
                self._nominal_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
                self._applied_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
                command_batch = self._make_batch(
                    kind=kind,
                    sim_step=feedback.sim_step,
                    source_segment_index=None,
                    source_action=False,
                )
                self._pending_batch = _PendingBatch(batch=command_batch)
            if context.policy_update_permit:
                self._previous_policy_action = self._held_action
                self._held_action = normalized_action
            elif normalized_action != self._held_action:
                raise ResidualEnvContractError("policy action changed inside an exact-eight outer cycle")

            telemetry = self._updated_telemetry(feedback.telemetry)
            guard_complete = self._update_guard_latches(telemetry)
            if feedback.hard_failure and not self._terminal_latched:
                command_batch = self._latch_terminal(
                    success=False,
                    truncated=False,
                    reason=feedback.hard_failure_reason,
                    sim_step=feedback.sim_step,
                )
            elif not self._terminal_latched and feedback.sim_step >= self.timeout_physics_steps:
                command_batch = self._latch_terminal(
                    success=False,
                    truncated=True,
                    reason="phase-local timeout",
                    sim_step=feedback.sim_step,
                )
            elif not self._terminal_latched and context.source_cursor_permit:
                completion_mapping, command_batch, tracking_end, wheel_stop_due = (
                    self._observe_completion(feedback)
                )
                if not self._terminal_latched and command_batch is None:
                    if wheel_stop_due:
                        command_batch, residual_transform_sha = self._apply_policy_update(
                            self._held_action,
                            feedback,
                            kind="WHEEL_STOP_OVERRIDE",
                            require_dispatch=True,
                            wheel_stop_ack=True,
                        )
                        self._dynamic_wheel_stop_count += 1
                    elif self._profile_complete() and guard_complete:
                        # Profile-free entries can already satisfy their
                        # physical guard at reset.  They must still traverse
                        # the sole residual composer once before success is
                        # latched, including the deterministic ZERO arms.
                        # If a nonzero residual changes physical targets, its
                        # batch owns this callback and the success stop is
                        # deferred until that predecessor is N+1 verified.
                        policy_batch: PhaseCommandBatch | None = None
                        if self._residual_transform_count == 0:
                            policy_batch, residual_transform_sha = self._apply_policy_update(
                                self._held_action,
                                feedback,
                            )
                        terminal_batch = self._latch_terminal(
                            success=True,
                            truncated=False,
                            reason="phase profile and physical guard complete",
                            sim_step=feedback.sim_step,
                        )
                        if policy_batch is not None:
                            if terminal_batch is not None:
                                raise ResidualEnvContractError(
                                    "profile-complete residual and terminal stop shared one callback"
                                )
                            command_batch = policy_batch
                        else:
                            command_batch = terminal_batch
                    elif (
                        not self.plan.profile_free
                        and self._active_binding is None
                        and self._next_frame < len(self.plan.segment_frames)
                        and feedback.sim_step >= self.plan.source_cursor_delay_physics_steps
                    ):
                        (
                            command_batch,
                            consumed_segment,
                            consumption_sha,
                            tracking_begin,
                            residual_transform_sha,
                        ) = self._consume_next_source(
                            self._held_action, feedback
                        )
                    else:
                        command_batch, residual_transform_sha = self._apply_policy_update(
                            self._held_action,
                            feedback,
                        )
            observation = build_residual_actor_observation(telemetry, self._previous_residual)
            reward = self._reward(
                telemetry,
                self._held_action,
                phase_success=self._phase_success,
                hard_failure=not self._phase_success and self._terminal_latched,
            )
            self._reward_return += reward.total
        except Exception as exc:
            if isinstance(exc, ResidualEnvContractError) and self._terminal_latched:
                raise
            # Contract/completion errors fail closed.  If this callback still
            # owns an empty batch slot, request one full safe-stop map.
            if not self._terminal_latched:
                try:
                    terminal_batch = self._latch_terminal(
                        success=False,
                        truncated=False,
                        reason=f"{type(exc).__name__}: {exc}",
                        sim_step=feedback.sim_step,
                    )
                    if terminal_batch is not None:
                        command_batch = terminal_batch
                except Exception:
                    self._terminal_latched = True
                    self._terminal_ready = True
                    self._phase_success = False
                    self._terminal_reason = f"{type(exc).__name__}: {exc}"
            telemetry = self._updated_telemetry(feedback.telemetry)
            observation = build_residual_actor_observation(telemetry, self._previous_residual)
            reward = self._reward(telemetry, self._held_action, phase_success=False, hard_failure=True)
            self._reward_return += reward.total

        metrics = self.metrics_snapshot(telemetry)
        info = MappingProxyType(
            {
                "schema_version": SCHEMA_VERSION,
                "entry_id": self.plan.reset.entry_id,
                "bank_sha256": self.plan.reset.bank_sha256,
                "source_version": self.plan.reset.source_version,
                "phase_state": self.plan.reset.phase_state,
                "profile_id": self.plan.profile_id,
                "source_cursor_next_offset": self._next_frame,
                "source_segment_count": len(self.plan.segment_frames),
                "completed_source_segments": tuple(self._completed_segments),
                "pending_batch_id": "" if self._pending_batch is None else self._pending_batch.batch.batch_id,
                "terminal_ready": self._terminal_ready,
                "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
                "physics_hz": EXPECTED_PHYSICS_HZ,
                "policy_hz": EXPECTED_POLICY_HZ,
                "raw_direct_rl_decimation": DIRECT_RL_DECIMATION,
                "n_plus_one_verified_this_step": n_plus_one_verified_this_step,
                "command_batch_count_this_step": int(command_batch is not None),
                "residual_transform_sha256_this_step": residual_transform_sha,
                "last_residual_transform_sha256": self._last_residual_transform_sha256,
                "active_completion_effective_targets_deg": MappingProxyType(
                    dict(self._active_effective_sparse_targets)
                ),
                "active_completion_latched_servo_residual_deg": MappingProxyType(
                    dict(self._active_completion_latched_servo_residual_deg)
                ),
                "physical_command_epoch": self._physical_command_epoch,
                "last_verified_physical_command_epoch": self._last_verified_physical_command_epoch,
                "episode_metrics_schema_sha256": EPISODE_METRICS_SCHEMA_SHA256,
                "episode_metrics": metrics,
            }
        )
        return PhaseKernelStepResult(
            observation=observation,
            reward=reward,
            terminated=bool(self._terminal_ready and not self._terminal_truncated),
            truncated=bool(self._terminal_ready and self._terminal_truncated),
            terminal_latched=self._terminal_latched,
            phase_success=self._phase_success,
            terminal_reason=self._terminal_reason,
            command_batch=command_batch,
            source_consumed_segment_index=consumed_segment,
            source_consumption_sha256=consumption_sha,
            completion_decision=MappingProxyType(dict(completion_mapping)),
            tracking_begin_targets_deg=MappingProxyType(dict(tracking_begin)),
            tracking_end_targets_deg=MappingProxyType(dict(tracking_end)),
            profile_fraction=self.profile_fraction,
            info=info,
        )


def _quat_wxyz_to_rpy(quaternion: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = _finite_vector(quaternion, 4, "root quaternion")
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _resolve_exact(resolver: Any, names: Sequence[str], label: str) -> tuple[list[int], list[str]]:
    ids, resolved_names = resolver(list(names), preserve_order=True)
    normalized_ids = [int(value) for value in ids]
    normalized_names = [str(value) for value in resolved_names]
    if normalized_names != list(names) or len(normalized_ids) != len(names) or len(set(normalized_ids)) != len(names):
        raise ResidualEnvContractError(f"{label} resolution is not exact")
    return normalized_ids, normalized_names


def _expand_asset_cfg(cfg: Any, env_regex_ns: str, label: str) -> Any:
    prim_path = getattr(cfg, "prim_path", None)
    if not isinstance(prim_path, str) or prim_path.count("{ENV_REGEX_NS}") != 1:
        raise ResidualEnvContractError(f"{label} prim path lacks exact namespace placeholder")
    replace_method = getattr(cfg, "replace", None)
    if not callable(replace_method):
        raise ResidualEnvContractError(f"{label} cfg has no replace()")
    return replace_method(prim_path=prim_path.format(ENV_REGEX_NS=env_regex_ns))


def build_env_cfg(
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
    entry_ids: Sequence[str] | None = None,
    episode_length_s: float = DEFAULT_EPISODE_LENGTH_S,
) -> Any:
    """Build the real DirectRLEnvCfg after the caller launched AppLauncher."""

    from isaaclab.envs import DirectRLEnvCfg  # type: ignore
    from isaaclab.utils import configclass  # type: ignore

    count = _exact_int(num_envs, "num_envs")
    if count <= 0:
        raise ResidualEnvContractError("num_envs must be positive")
    duration = _finite_number(episode_length_s, "episode_length_s")
    if duration <= 0.0:
        raise ResidualEnvContractError("episode_length_s must be positive")
    bank = load_phase_entry_bank(verify_artifacts=True)
    available = tuple(str(entry["entry_id"]) for entry in bank.entries)
    selected = tuple(entry_ids or (available[0],))
    if not selected or any(item not in available for item in selected):
        raise ResidualEnvContractError("entry_ids must select only immutable current-bank entries")
    expanded = tuple(selected[index % len(selected)] for index in range(count))
    bundle = build_isaaclab_scene_bundle(num_envs=count, device=device)

    @configclass
    class Fsm50ResidualEnvCfg(DirectRLEnvCfg):
        decimation = DIRECT_RL_DECIMATION
        episode_length_s = duration
        is_finite_horizon = True
        action_space = RESIDUAL_ACTION_DIM
        observation_space = ACTOR_OBSERVATION_DIM
        # R1 intentionally has no privileged critic-only truth.  The SKRL
        # state seam is the same frozen deployable 115-D actor vector.
        state_space = CRITIC_STATE_DIM

    cfg = Fsm50ResidualEnvCfg()
    cfg.sim = bundle.simulation_cfg
    cfg.scene = bundle.interactive_scene_cfg
    cfg.fsm50_scene_bundle = bundle
    cfg.fsm50_entry_ids = expanded
    cfg.fsm50_bank_sha256 = bank.bank_sha256
    cfg.fsm50_outer_policy_substeps = EXPECTED_RENDER_SUBSTEPS
    if cfg.decimation != 1 or cfg.sim.dt != PHYSICS_DT_S or cfg.sim.render_interval != RENDER_INTERVAL_PHYSICS_STEPS:
        raise ResidualEnvContractError("DirectRLEnv cfg drifted from raw 120 Hz formal scene")
    return cfg


def make_direct_rl_env(cfg: Any | None = None, render_mode: str | None = None, **kwargs: Any) -> Any:
    """Create the lazy DirectRLEnv implementation after AppLauncher."""

    import torch  # type: ignore
    from isaaclab.assets import Articulation, RigidObject  # type: ignore
    from isaaclab.envs import DirectRLEnv  # type: ignore

    resolved_cfg = build_env_cfg() if cfg is None else cfg

    class Fsm50ResidualDirectRLEnv(DirectRLEnv):
        def __init__(self, env_cfg: Any, render_mode: str | None = None, **env_kwargs: Any):
            if int(env_cfg.decimation) != 1:
                raise ResidualEnvContractError("R1 DirectRLEnv raw decimation must be one")
            self._fsm50_bundle = env_cfg.fsm50_scene_bundle
            self._fsm50_entry_ids = tuple(env_cfg.fsm50_entry_ids)
            if (
                len(self._fsm50_entry_ids) != int(env_cfg.scene.num_envs)
                or any(not isinstance(entry_id, str) or not entry_id for entry_id in self._fsm50_entry_ids)
            ):
                raise ResidualEnvContractError("cfg must bind one non-empty immutable entry_id per environment")
            if int(env_cfg.fsm50_outer_policy_substeps) != EXPECTED_RENDER_SUBSTEPS:
                raise ResidualEnvContractError("cfg outer policy cadence is not exact-eight")
            self._fsm50_outer_substep = 0
            self._fsm50_outer_index = 0
            self._fsm50_held_actions = None
            self._fsm50_kernels: list[PhaseLocalResidualKernel] = []
            self._fsm50_episode_steps: list[int] = []
            self._fsm50_staged_batches: list[PhaseCommandBatch | None] = []
            self._fsm50_pending_echo: list[tuple[PhaseCommandBatch, Any, Any] | None] = []
            self._fsm50_last_results: list[PhaseKernelStepResult | None] = []
            self._fsm50_drive_commands: list[dict[str, float]] = []
            super().__init__(env_cfg, render_mode, **env_kwargs)
            # DirectRLEnv's raw cfg intentionally reports a 120 Hz step.  This
            # adapter returns only after eight such steps, so video consumers
            # must see the public 15 Hz cadence.
            self.metadata = dict(self.metadata)
            self.metadata["render_fps"] = EXPECTED_POLICY_HZ
            self._servo_ids, _ = _resolve_exact(self.robot.find_joints, SERVO_JOINT_NAMES, "servo joints")
            self._wheel_ids, _ = _resolve_exact(self.robot.find_joints, WHEEL_JOINT_NAMES, "wheel joints")
            self._wheel_body_ids, _ = _resolve_exact(self.robot.find_bodies, WHEEL_BODY_NAMES, "wheel bodies")
            bank = load_phase_entry_bank(verify_artifacts=True)
            library = build_profile_library(REPLAY_ROOT)
            envelope = load_residual_envelope(verify_evidence=True)
            self._fsm50_kernels = [
                PhaseLocalResidualKernel(
                    build_phase_episode_plan(
                        entry_id,
                        bank=bank,
                        profile_library=library,
                        verify_artifacts=True,
                    ),
                    envelope,
                    timeout_physics_steps=int(math.ceil(env_cfg.episode_length_s / PHYSICS_DT_S)),
                )
                for entry_id in self._fsm50_entry_ids
            ]
            self._fsm50_episode_steps = [0] * self.num_envs
            self._fsm50_staged_batches = [None] * self.num_envs
            self._fsm50_pending_echo = [None] * self.num_envs
            self._fsm50_last_results = [None] * self.num_envs
            self._fsm50_drive_commands = [dict(kernel.current_applied_servo_targets_deg) for kernel in self._fsm50_kernels]

        def _setup_scene(self) -> None:
            robot_cfg = _expand_asset_cfg(self._fsm50_bundle.robot_cfg, self.scene.env_regex_ns, "robot")
            obstacle_cfg = _expand_asset_cfg(self._fsm50_bundle.obstacle_cfg, self.scene.env_regex_ns, "obstacle")
            self.robot = Articulation(robot_cfg)
            self.obstacle = RigidObject(obstacle_cfg)
            self.scene.articulations["robot"] = self.robot
            self.scene.rigid_objects["obstacle"] = self.obstacle
            spawn_global_scene_assets(self._fsm50_bundle)
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=list(self._fsm50_bundle.global_collision_prim_paths))

        def _pre_physics_step(self, actions: Any) -> None:
            if actions.shape != (self.num_envs, RESIDUAL_ACTION_DIM) or not bool(torch.isfinite(actions).all().item()):
                raise ResidualEnvContractError("policy action tensor must be finite [num_envs,12]")
            if self._fsm50_outer_substep == 0:
                self._fsm50_held_actions = actions.clone()
            elif self._fsm50_held_actions is None or not bool(torch.equal(actions, self._fsm50_held_actions)):
                raise ResidualEnvContractError("policy action tensor changed inside exact-eight outer cycle")

        def _apply_action(self) -> None:
            if self._fsm50_held_actions is None:
                raise ResidualEnvContractError("outer policy action is unavailable")
            applied_batches: list[PhaseCommandBatch | None] = [None] * self.num_envs
            for index, kernel in enumerate(self._fsm50_kernels):
                staged = self._fsm50_staged_batches[index]
                if staged is not None:
                    expected = self._fsm50_episode_steps[index]
                    if staged.expected_first_physics_step != expected:
                        raise ResidualEnvContractError("staged batch missed its exact N+1 physics slot")
                    self._fsm50_staged_batches[index] = None
                    applied_batches[index] = staged
                self._fsm50_drive_commands[index] = slew_servo_targets_150_deg_s(
                    self._fsm50_drive_commands[index],
                    kernel.current_applied_servo_targets_deg,
                )
            standing = self.robot.data.default_joint_pos[:, self._servo_ids]
            command_sign = torch.tensor(
                [JOINT_COMMAND_SIGN[name] for name in SERVO_JOINT_NAMES],
                dtype=standing.dtype,
                device=standing.device,
            )
            servo_commands = torch.tensor(
                [[self._fsm50_drive_commands[i][name] for name in SERVO_JOINT_NAMES] for i in range(self.num_envs)],
                dtype=standing.dtype,
                device=standing.device,
            )
            wheel_commands = torch.tensor(
                [[self._fsm50_kernels[i].current_applied_wheel_targets_rad_s[name] for name in WHEEL_JOINT_NAMES] for i in range(self.num_envs)],
                dtype=standing.dtype,
                device=standing.device,
            )
            wheel_sign = torch.tensor(
                [WHEEL_FORWARD_SIGN[name] for name in WHEEL_JOINT_NAMES],
                dtype=standing.dtype,
                device=standing.device,
            )
            self.robot.set_joint_position_target(standing + command_sign * torch.deg2rad(servo_commands), joint_ids=self._servo_ids)
            self.robot.set_joint_velocity_target(wheel_commands * wheel_sign, joint_ids=self._wheel_ids)
            servo_echo_targets = standing + command_sign * torch.deg2rad(servo_commands)
            wheel_echo_targets = wheel_commands * wheel_sign
            for index in range(self.num_envs):
                # A staged logical batch is first written in this N+1 slot.
                # Persist its exact PhysX target tensors for post-step echo
                # validation rather than self-claiming readback success.
                prior = self._fsm50_pending_echo[index]
                if prior is not None:
                    raise ResidualEnvContractError("unconsumed target echo crossed a physics callback")
                applied = applied_batches[index]
                if applied is not None:
                    self._fsm50_pending_echo[index] = (
                        applied,
                        servo_echo_targets[index].clone(),
                        wheel_echo_targets[index].clone(),
                    )

        def _telemetry_for(self, index: int) -> tuple[dict[str, Any], dict[str, float], dict[str, float], bool, str]:
            kernel = self._fsm50_kernels[index]
            root_pos = [float(value) for value in self.robot.data.root_pos_w[index].tolist()]
            root_quat = [float(value) for value in self.robot.data.root_quat_w[index].tolist()]
            root_lin = [float(value) for value in self.robot.data.root_lin_vel_w[index].tolist()]
            root_ang = [float(value) for value in self.robot.data.root_ang_vel_w[index].tolist()]
            ordered_ids = self._servo_ids + self._wheel_ids
            q = [float(value) for value in self.robot.data.joint_pos[index, ordered_ids].tolist()]
            qd = [float(value) for value in self.robot.data.joint_vel[index, ordered_ids].tolist()]
            all_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
            if len(q) != len(all_names) or len(qd) != len(all_names):
                raise ResidualEnvContractError("Isaac articulation joint shape is not exact 12D")
            joint_q = dict(zip(all_names, q))
            joint_qd = dict(zip(all_names, qd))
            standing_deg = {
                name: math.degrees(float(self.robot.data.default_joint_pos[index, joint_id].item()))
                for name, joint_id in zip(SERVO_JOINT_NAMES, self._servo_ids)
            }
            nominal_errors = {
                name: math.degrees(joint_q[name])
                - (standing_deg[name] + JOINT_COMMAND_SIGN[name] * kernel.current_nominal_servo_targets_deg[name])
                for name in SERVO_JOINT_NAMES
            }
            effective_errors = {
                name: math.degrees(joint_q[name])
                - (standing_deg[name] + JOINT_COMMAND_SIGN[name] * kernel.current_applied_servo_targets_deg[name])
                for name in SERVO_JOINT_NAMES
            }
            servo_velocity = {name: math.degrees(joint_qd[name]) for name in SERVO_JOINT_NAMES}
            centers = {
                leg: tuple(float(value) for value in self.robot.data.body_pos_w[index, body_id, :].tolist())
                for leg, body_id in zip(LEG_NAMES, self._wheel_body_ids)
            }
            obstacle = ObstacleGeometry(
                front_face_x_m=OBSTACLE_FRONT_X_M + float(self.scene.env_origins[index, 0].item()),
                top_z_m=OBSTACLE_BOTTOM_Z_M + OBSTACLE_HEIGHT_M,
                bottom_z_m=OBSTACLE_BOTTOM_Z_M,
                rear_face_x_m=OBSTACLE_FRONT_X_M + OBSTACLE_LENGTH_M + float(self.scene.env_origins[index, 0].item()),
                center_y_m=float(self.scene.env_origins[index, 1].item()),
                width_m=OBSTACLE_WIDTH_M,
            )
            contacts = {
                leg: classify_wheel_contact(WheelObservation(leg=leg, center_w=centers[leg]), obstacle, wheel_radius_m=WHEEL_RADIUS_M)
                for leg in LEG_NAMES
            }
            classes = {leg: contacts[leg].contact_class.value for leg in LEG_NAMES}
            support = tuple(leg for leg in LEG_NAMES if classes[leg] != "AIR")
            roll, pitch, yaw = _quat_wxyz_to_rpy(root_quat)
            finite = all(math.isfinite(value) for value in (*root_pos, *root_quat, *root_lin, *root_ang, *q, *qd))
            hard_failure = not finite or abs(roll) >= math.radians(85.0) or abs(pitch) >= math.radians(85.0)
            reason = "non-finite Isaac articulation state" if not finite else "fallen attitude" if hard_failure else ""
            recoverable = bool(finite and abs(roll) < math.radians(70.0) and abs(pitch) < math.radians(70.0) and len(support) >= 2)
            velocity_stable = bool(max((abs(value) for value in (*root_lin, *root_ang, *qd)), default=0.0) <= 0.10)
            telemetry = {
                "macro_state": kernel.plan.reset.phase_state,
                "source_version": kernel.plan.reset.source_version,
                "profile_fraction": kernel.profile_fraction,
                "active_leg": ACTIVE_LEG_BY_STATE[kernel.plan.reset.phase_state],
                "support_legs": support,
                "base_position_m": {"x": root_pos[0], "y": root_pos[1], "z": root_pos[2]},
                "base_roll_rad": roll,
                "base_pitch_rad": pitch,
                "base_yaw_rad": yaw,
                "root_linear_velocity_w": root_lin,
                "root_angular_velocity_w": root_ang,
                "joint_q_rad": joint_q,
                "joint_qd_rad_s": joint_qd,
                "servo_targets_deg": dict(kernel.current_nominal_servo_targets_deg),
                "canonical_servo_actual_error_deg": nominal_errors,
                "wheel_targets_rad_s": dict(kernel.current_nominal_wheel_targets_rad_s),
                "wheel_center_w_m": centers,
                "wheel_contact_classes": classes,
                "wheel_front_face_clearance_m": {leg: contacts[leg].front_face_clearance_m for leg in LEG_NAMES},
                "wheel_top_clearance_m": {leg: contacts[leg].clearance_over_top_m for leg in LEG_NAMES},
                "obstacle_front_face_x_m": obstacle.front_face_x_m,
                "obstacle_top_z_m": obstacle.top_z_m,
                "geometry_support_candidate_count": len(support),
                "body_crossed_front_face": root_pos[0] > obstacle.front_face_x_m,
                "final_recoverable": recoverable,
                "stability_state": "stable" if recoverable and velocity_stable else "recoverable" if recoverable else "fallen",
            }
            return telemetry, effective_errors, servo_velocity, hard_failure, reason

        def _get_dones(self) -> tuple[Any, Any]:
            actions = self._fsm50_held_actions.detach().cpu().tolist()
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for index, kernel in enumerate(self._fsm50_kernels):
                telemetry, errors, velocities, hard_failure, reason = self._telemetry_for(index)
                step = self._fsm50_episode_steps[index]
                echo = self._fsm50_pending_echo[index]
                readback = None
                if echo is not None:
                    batch, expected_servo, expected_wheel = echo
                    # This runs after DirectRLEnv has called
                    # scene.write_data_to_sim(), stepped physics, and updated
                    # the scene.  Only the low-level PhysX view can attest the
                    # N+1 drive targets; robot.data target buffers are not ACK
                    # evidence.
                    physx_evidence = validate_physx_target_readback(
                        robot=self.robot,
                        num_envs=self.num_envs,
                        env_index=index,
                        servo_joint_ids=self._servo_ids,
                        wheel_joint_ids=self._wheel_ids,
                        expected_servo_position_targets=expected_servo,
                        expected_wheel_velocity_targets=expected_wheel,
                    )
                    readback = BatchReadback(
                        batch_id=batch.batch_id,
                        applied_sim_step=batch.decision_sim_step,
                        first_physics_step=batch.expected_first_physics_step,
                        physical_command_epoch=batch.physical_command_epoch,
                        physx_evidence_source=PHYSX_TARGET_READBACK_SOURCE,
                        physx_evidence_sha256=physx_evidence.evidence_sha256,
                        physx_failure_reason=physx_evidence.failure_reason,
                        verified=physx_evidence.verified,
                    )
                    self._fsm50_pending_echo[index] = None
                result = kernel.step(
                    actions[index],
                    OuterCycleContext(
                        outer_cycle_index=step // EXPECTED_RENDER_SUBSTEPS,
                        physics_substep_index=step % EXPECTED_RENDER_SUBSTEPS,
                        source_cursor_permit=step % EXPECTED_RENDER_SUBSTEPS == 0,
                        policy_update_permit=step % EXPECTED_RENDER_SUBSTEPS == 0,
                    ),
                    PhasePhysicsFeedback(
                        sim_step=step,
                        sim_time_s=step * PHYSICS_DT_S,
                        telemetry=telemetry,
                        effective_servo_actual_error_deg=errors,
                        servo_velocity_deg_s=velocities,
                        batch_readback=readback,
                        hard_failure=hard_failure,
                        hard_failure_reason=reason,
                    ),
                )
                self._fsm50_staged_batches[index] = result.command_batch
                self._fsm50_last_results[index] = result
                self._fsm50_episode_steps[index] += 1
                # Terminal events are latched immediately but exposed to
                # DirectRLEnv only at an exact outer boundary after safe-stop
                # N+1 verification, preventing mid-cycle policy/reset drift.
                if self._fsm50_outer_substep == EXPECTED_RENDER_SUBSTEPS - 1:
                    terminated[index] = result.terminated
                    truncated[index] = result.truncated
            self.extras["fsm50"] = {
                "schema_version": EPISODE_METRICS_SCHEMA_VERSION,
                "schema_sha256": EPISODE_METRICS_SCHEMA_SHA256,
                "per_env": [
                    None
                    if result is None
                    else dict(result.info["episode_metrics"])
                    for result in self._fsm50_last_results
                ],
            }
            return terminated, truncated

        def _get_rewards(self) -> Any:
            return torch.tensor(
                [0.0 if result is None else result.reward.total for result in self._fsm50_last_results],
                dtype=torch.float32,
                device=self.device,
            )

        def _get_observations(self) -> dict[str, Any]:
            observations: list[tuple[float, ...]] = []
            for index, kernel in enumerate(self._fsm50_kernels):
                result = self._fsm50_last_results[index]
                if result is not None:
                    observations.append(result.observation.values)
                else:
                    telemetry, _errors, _velocities, _hard, _reason = self._telemetry_for(index)
                    observations.append(
                        build_residual_actor_observation(
                            kernel._updated_telemetry(telemetry),
                            kernel.previous_applied_residual,
                        ).values
                    )
            return dict(
                build_actor_critic_observation_tensors(
                    observations,
                    torch_module=torch,
                    device=self.device,
                )
            )

        def _reset_idx(self, env_ids: Any) -> None:
            super()._reset_idx(env_ids)
            indices = [int(value) for value in env_ids.detach().cpu().tolist()]
            all_joint_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
            for index in indices:
                reset = self._fsm50_kernels[index].reset()
                root_pose = torch.tensor(
                    [(*reset.root_position_m, *reset.root_orientation_wxyz)],
                    dtype=self.robot.data.default_root_state.dtype,
                    device=self.device,
                )
                root_pose[:, :3] += self.scene.env_origins[index : index + 1]
                root_velocity = torch.tensor(
                    [(*reset.root_linear_velocity_m_s, *reset.root_angular_velocity_rad_s)],
                    dtype=self.robot.data.default_root_state.dtype,
                    device=self.device,
                )
                q_by_name = dict(zip(all_joint_names, reset.joint_q_rad))
                qd_by_name = dict(zip(all_joint_names, reset.joint_qd_rad_s))
                ordered_ids = self._servo_ids + self._wheel_ids
                q_tensor = torch.tensor(
                    [[q_by_name[name] for name in all_joint_names]],
                    dtype=self.robot.data.joint_pos.dtype,
                    device=self.device,
                )
                qd_tensor = torch.tensor(
                    [[qd_by_name[name] for name in all_joint_names]],
                    dtype=self.robot.data.joint_vel.dtype,
                    device=self.device,
                )
                env_tensor = torch.tensor([index], dtype=torch.long, device=self.device)
                self.robot.write_root_pose_to_sim(root_pose, env_ids=env_tensor)
                self.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_tensor)
                self.robot.write_joint_state_to_sim(q_tensor, qd_tensor, joint_ids=ordered_ids, env_ids=env_tensor)
                self._fsm50_episode_steps[index] = 0
                self._fsm50_staged_batches[index] = None
                self._fsm50_pending_echo[index] = None
                self._fsm50_last_results[index] = None
                self._fsm50_drive_commands[index] = dict(reset.nominal_servo_targets_deg)

        def step(self, action: Any) -> Any:
            reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            final = None
            for substep in range(EXPECTED_RENDER_SUBSTEPS):
                self._fsm50_outer_substep = substep
                observation, reward, terminated, truncated, extras = super().step(action)
                reward_sum += reward
                if substep < EXPECTED_RENDER_SUBSTEPS - 1 and bool((terminated | truncated).any().item()):
                    raise ResidualEnvContractError("DirectRLEnv exposed a terminal before exact-eight boundary")
                final = (observation, reward_sum, terminated, truncated, extras)
            self._fsm50_outer_index += 1
            return final

    return Fsm50ResidualDirectRLEnv(resolved_cfg, render_mode=render_mode, **kwargs)


def register_gym_env(*, gym_module: Any | None = None) -> str:
    """Idempotently register the lazy R1 task without importing Isaac Lab."""

    gym = gym_module
    if gym is None:
        import gymnasium as gym  # type: ignore[no-redef]
    entry_point = f"{__name__}:make_direct_rl_env"
    cfg_entry_point = f"{__name__}:build_env_cfg"
    try:
        existing = gym.spec(GYM_ENV_ID)
    except Exception:
        existing = None
    if existing is not None:
        if existing.entry_point != entry_point or existing.kwargs.get("env_cfg_entry_point") != cfg_entry_point:
            raise ResidualEnvContractError("gym id is already registered to a different environment")
        return GYM_ENV_ID
    gym.register(
        id=GYM_ENV_ID,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": cfg_entry_point},
    )
    return GYM_ENV_ID


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "BatchReadback",
    "CRITIC_STATE_DIM",
    "DEFAULT_EPISODE_LENGTH_S",
    "DEFAULT_TIMEOUT_PHYSICS_STEPS",
    "EPISODE_METRICS_FIELDS",
    "EPISODE_METRICS_SCHEMA_SHA256",
    "EPISODE_METRICS_SCHEMA_VERSION",
    "GYM_ENV_ID",
    "PHYSX_TARGET_READBACK_SOURCE",
    "PhaseCommandBatch",
    "PhaseEpisodePlan",
    "PhaseKernelStepResult",
    "PhaseLocalResidualKernel",
    "PhasePhysicsFeedback",
    "PhaseResetState",
    "PhysxTargetReadbackEvidence",
    "ResidualEnvContractError",
    "SCHEMA_VERSION",
    "SERVO_MAX_DELTA_DEG_PER_PHYSICS_STEP",
    "SERVO_REFERENCE_VELOCITY_DEG_S",
    "build_env_cfg",
    "build_actor_critic_observation_tensors",
    "build_phase_episode_plan",
    "env_source_sha256",
    "make_direct_rl_env",
    "register_gym_env",
    "slew_servo_targets_150_deg_s",
    "validate_physx_target_readback",
]
