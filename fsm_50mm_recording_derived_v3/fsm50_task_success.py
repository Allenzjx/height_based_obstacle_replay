"""Gate-A task-level classification for 50 mm recording replays.

This module intentionally does not reuse the legacy strict-physical verdict as a
task verdict.  Strict rest, contact-point drift, and final posture quality remain
diagnostics; task failure is reserved for incomplete execution, incomplete
traversal, an unrecoverable final state, or an explicit hard-safety failure.

The classifier is pure Python and only consumes already-completed artifacts.  It
does not import or start Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping, Sequence


REPLAY_TASK_SUCCESS = "REPLAY_TASK_SUCCESS"
REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE = (
    "REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE"
)
REPLAY_TASK_FAIL = "REPLAY_TASK_FAIL"
REPLAY_TASK_NOT_EVALUATED = "NOT_EVALUATED"

EVALUATED = "EVALUATED"
NOT_EVALUATED = "NOT_EVALUATED"

POSTURE_COMPLETE = "POSTURE_COMPLETE"
POSTURE_INCOMPLETE = "POSTURE_INCOMPLETE"
POSTURE_NOT_APPLICABLE = "NOT_APPLICABLE"

LEGS = ("FL", "FR", "RL", "RR")
SERVO_JOINTS = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_JOINTS = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
ALL_JOINTS = (*SERVO_JOINTS, *WHEEL_JOINTS)

# Only non-finite values in live robot state or commanded/applied targets are a
# task-level hard safety failure.  Every other NaN/Inf is evidence quality, not
# proof that the robot failed the traversal.
CORE_FINITE_TELEMETRY_FIELDS = frozenset(
    {
        "actual_joint_target_rad",
        "base_angular_velocity_rad_s",
        "base_linear_velocity_m_s",
        "base_pitch_rad",
        "base_position_m",
        "base_quaternion_wxyz",
        "base_qw",
        "base_qx",
        "base_qy",
        "base_qz",
        "base_roll_rad",
        "base_vx_m_s",
        "base_vy_m_s",
        "base_vz_m_s",
        "base_wx_rad_s",
        "base_wy_rad_s",
        "base_wz_rad_s",
        "base_x_m",
        "base_y_m",
        "base_yaw_rad",
        "base_z_m",
        "command_space_servo_target_deg",
        "joint_position_target_buffer_rad",
        "joint_q_rad",
        "joint_qd_rad_s",
        "joint_velocity_target_buffer_rad_s",
        "measured_joint_position_rad",
        "measured_joint_velocity_rad_s",
        "root_angular_velocity_w",
        "root_linear_velocity_w",
        "root_orientation_wxyz",
        "root_pose_w",
        "root_position_w",
        "servo_command_target_rad",
        "servo_targets_deg",
        "wheel_command_velocity_rad_s",
        "wheel_physx_target_rad_s_by_leg",
        "wheel_targets_rad_s",
    }
)

NORMAL_SESSION_VECTOR_SHAPES = {
    "base_quaternion_wxyz": 4,
    "base_linear_velocity_m_s": 3,
    "base_angular_velocity_rad_s": 3,
}
NORMAL_SESSION_MAPPING_SHAPES = {
    "base_position_m": frozenset(("x", "y", "z")),
    "joint_q_rad": frozenset(ALL_JOINTS),
    "joint_qd_rad_s": frozenset(ALL_JOINTS),
    "servo_targets_deg": frozenset(SERVO_JOINTS),
    "wheel_targets_rad_s": frozenset(WHEEL_JOINTS),
}
NORMAL_SESSION_CORE_FIELDS = frozenset(
    (*NORMAL_SESSION_VECTOR_SHAPES, *NORMAL_SESSION_MAPPING_SHAPES)
)

TABLE_COLUMNS = (
    "version",
    "step_count",
    "fast_segment_count",
    "evaluation_status",
    "task_result",
    "posture_result",
    "body_crossed_front_face",
    "required_leg_lift_completed",
    "final_recoverable",
    "final_wheel_classes",
    "peak_roll",
    "peak_pitch",
    "video_path",
    "first_actual_failure_phase",
    "hard_failure_reasons",
    "not_evaluated_reasons",
    "secondary_diagnostics",
    "notes",
    "classification_reasons",
)


@dataclass(frozen=True)
class ManualVideoVerdict:
    """Optional human review facts.

    ``None`` means the reviewer did not make a claim about that fact.  A video
    verdict may fill evidence gaps, but a positive video observation never
    suppresses a hard failure reported by machine evidence.
    """

    task_completed: bool | None = None
    body_crossed_front_face: bool | None = None
    required_leg_lift_completed: bool | None = None
    final_recoverable: bool | None = None
    posture_incomplete: bool | None = None
    robot_fell: bool | None = None
    body_stuck: bool | None = None
    wheel_drive_up_without_required_lift: bool | None = None
    dangerous_body_collision: bool | None = None
    joint_limit_violation: bool | None = None
    severe_penetration: bool | None = None
    irrecoverable: bool | None = None
    first_actual_failure_phase: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "task_completed",
            "body_crossed_front_face",
            "required_leg_lift_completed",
            "final_recoverable",
            "posture_incomplete",
            "robot_fell",
            "body_stuck",
            "wheel_drive_up_without_required_lift",
            "dangerous_body_collision",
            "joint_limit_violation",
            "severe_penetration",
            "irrecoverable",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{field_name} must be bool or None")
        if type(self.first_actual_failure_phase) is not str:
            raise TypeError("first_actual_failure_phase must be str")
        if type(self.notes) is not tuple or any(
            type(note) is not str for note in self.notes
        ):
            raise TypeError("notes must be a tuple of strings")


@dataclass(frozen=True)
class TaskSuccessAssessment:
    version: str
    step_count: int
    fast_segment_count: int
    evaluation_status: str
    task_result: str
    posture_result: str
    body_crossed_front_face: bool
    required_leg_lift_completed: bool
    final_recoverable: bool
    final_wheel_classes: Mapping[str, str]
    peak_roll: float | None
    peak_pitch: float | None
    video_path: str
    first_actual_failure_phase: str
    hard_failure_reasons: tuple[str, ...]
    not_evaluated_reasons: tuple[str, ...]
    secondary_diagnostics: tuple[str, ...]
    notes: tuple[str, ...]
    classification_reasons: tuple[str, ...]

    def to_table_row(self) -> dict[str, object]:
        """Return a deterministic CSV/Markdown-ready row."""

        final_classes = ";".join(
            f"{leg}={self.final_wheel_classes[leg]}"
            for leg in LEGS
            if leg in self.final_wheel_classes
        )
        combined_notes = _unique((*self.notes, *self.secondary_diagnostics))
        row: dict[str, object] = {
            "version": self.version,
            "step_count": self.step_count,
            "fast_segment_count": self.fast_segment_count,
            "evaluation_status": self.evaluation_status,
            "task_result": self.task_result,
            "posture_result": self.posture_result,
            "body_crossed_front_face": self.body_crossed_front_face,
            "required_leg_lift_completed": self.required_leg_lift_completed,
            "final_recoverable": self.final_recoverable,
            "final_wheel_classes": final_classes,
            "peak_roll": self.peak_roll,
            "peak_pitch": self.peak_pitch,
            "video_path": self.video_path,
            "first_actual_failure_phase": self.first_actual_failure_phase,
            "hard_failure_reasons": "; ".join(self.hard_failure_reasons),
            "not_evaluated_reasons": "; ".join(
                self.not_evaluated_reasons
            ),
            "secondary_diagnostics": "; ".join(self.secondary_diagnostics),
            "notes": "; ".join(combined_notes),
            "classification_reasons": "; ".join(self.classification_reasons),
        }
        # Protect the table contract from accidental field-order drift.
        return {column: row[column] for column in TABLE_COLUMNS}


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _bool_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if type(value) is not bool:
        errors.append(f"EVIDENCE_TYPE_ERROR:{path}.{key}")
        return None
    return value


def _optional_bool_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> bool | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _bool_field(mapping, key, path, errors)


def _int_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> int | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if type(value) is not int or value < 0:
        errors.append(f"EVIDENCE_TYPE_ERROR:{path}.{key}")
        return None
    return value


def _optional_int_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> int | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _int_field(mapping, key, path, errors)


def _number_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if type(value) not in (int, float):
        errors.append(f"EVIDENCE_TYPE_ERROR:{path}.{key}")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"EVIDENCE_NONFINITE:{path}.{key}")
        return None
    return number


def _string_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if type(value) is not str:
        errors.append(f"EVIDENCE_TYPE_ERROR:{path}.{key}")
        return None
    return value


def _optional_string_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> str | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _string_field(mapping, key, path, errors)


def _mapping_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, Mapping):
        errors.append(f"EVIDENCE_TYPE_ERROR:{path}.{key}")
        return None
    return value


def _optional_mapping_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _mapping_field(mapping, key, path, errors)


def _nonfinite_paths(value: object, path: str) -> tuple[str, ...]:
    found: list[str] = []

    def visit(item: object, item_path: str) -> None:
        if type(item) is float and not math.isfinite(item):
            found.append(item_path)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{item_path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")

    visit(value, path)
    return tuple(found)


def _is_core_nonfinite_path(path: str) -> bool:
    prefix = "final_telemetry_row."
    if not path.startswith(prefix):
        return False
    top_level = path[len(prefix) :].split(".", 1)[0].split("[", 1)[0]
    return top_level in CORE_FINITE_TELEMETRY_FIELDS


def _validate_normal_session_core_telemetry(
    telemetry: Mapping[str, Any],
    hard: list[str],
    errors: list[str],
) -> None:
    """Validate the exact JSON shapes emitted by the normal replay worker.

    Legacy completed artifacts do not carry these grouped fields, so this
    contract activates only when at least one normal-session marker is present.
    Once active, an absent/malformed group cannot silently yield success, while
    an explicit null/non-finite core state remains a task-level safety failure.
    """

    normal_session = "robot_state_finite" in telemetry or any(
        field in telemetry for field in NORMAL_SESSION_CORE_FIELDS
    )
    if not normal_session:
        return

    state_path = "final_telemetry_row.robot_state_finite"
    if "robot_state_finite" not in telemetry:
        errors.append(f"EVIDENCE_MISSING:{state_path}")
    else:
        state_finite = telemetry["robot_state_finite"]
        if state_finite is None or state_finite is False:
            hard.append(f"NONFINITE_CORE_STATE:{state_path}")
        elif state_finite is not True:
            errors.append(f"EVIDENCE_TYPE_ERROR:{state_path}")

    def validate_number(value: object, path: str) -> None:
        if value is None:
            hard.append(f"NONFINITE_CORE_STATE:{path}")
        elif type(value) not in (int, float):
            errors.append(f"EVIDENCE_TYPE_ERROR:{path}")
        elif not math.isfinite(float(value)):
            hard.append(f"NONFINITE_CORE_STATE:{path}")

    for field, length in NORMAL_SESSION_VECTOR_SHAPES.items():
        path = f"final_telemetry_row.{field}"
        if field not in telemetry:
            errors.append(f"EVIDENCE_MISSING:{path}")
            continue
        value = telemetry[field]
        if value is None:
            hard.append(f"NONFINITE_CORE_STATE:{path}")
            continue
        if not isinstance(value, (list, tuple)) or len(value) != length:
            errors.append(f"EVIDENCE_SHAPE_ERROR:{path}")
            continue
        for index, item in enumerate(value):
            validate_number(item, f"{path}[{index}]")

    for field, expected_keys in NORMAL_SESSION_MAPPING_SHAPES.items():
        path = f"final_telemetry_row.{field}"
        if field not in telemetry:
            errors.append(f"EVIDENCE_MISSING:{path}")
            continue
        value = telemetry[field]
        if value is None:
            hard.append(f"NONFINITE_CORE_STATE:{path}")
            continue
        if not isinstance(value, Mapping) or frozenset(value) != expected_keys:
            errors.append(f"EVIDENCE_SHAPE_ERROR:{path}")
            continue
        for key in sorted(expected_keys):
            validate_number(value[key], f"{path}.{key}")


def _hard_failure_is_blocked(reasons: Sequence[str]) -> bool:
    """Return whether evidence integrity/infrastructure prevents evaluation.

    Missing unrelated evidence (including video or an unavailable sensor) does
    not erase a separately observed hard failure.  A type/conflict boundary or
    an infrastructure-invalid run still cannot be labelled as a robot outcome.
    """

    blocking_exact = {
        "ARTIFACT_INVALID",
        "INFRASTRUCTURE_PROCESS_FAILURE",
        "MOTION_START_NOT_READY",
        "SECOND_SIMULATOR_PROCESS",
    }
    return any(
        reason in blocking_exact
        or reason.startswith("EVIDENCE_CONFLICT:")
        or reason.startswith("EVIDENCE_TYPE_ERROR:")
        for reason in reasons
    )


def _diagnostic_bool_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    secondary: list[str],
) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        secondary.append(f"SECONDARY_DIAGNOSTIC_UNAVAILABLE:{path}.{key}")
        return None
    if type(value) is not bool:
        secondary.append(f"SECONDARY_DIAGNOSTIC_UNAVAILABLE:{path}.{key}")
        return None
    return value


def _diagnostic_number_field(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    secondary: list[str],
) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        secondary.append(f"SECONDARY_DIAGNOSTIC_UNAVAILABLE:{path}.{key}")
        return None
    return float(value)


def _normalise_video_verdict(
    value: ManualVideoVerdict | Mapping[str, object] | None,
    errors: list[str],
) -> ManualVideoVerdict:
    if value is None:
        return ManualVideoVerdict()
    if isinstance(value, ManualVideoVerdict):
        return value
    if not isinstance(value, Mapping):
        errors.append("EVIDENCE_TYPE_ERROR:video_verdict")
        return ManualVideoVerdict()

    allowed = {field.name for field in fields(ManualVideoVerdict)}
    for key in value:
        if type(key) is not str or key not in allowed:
            errors.append(f"EVIDENCE_TYPE_ERROR:video_verdict.{key}")

    kwargs: dict[str, object] = {}
    bool_names = allowed - {"notes", "first_actual_failure_phase"}
    for name in bool_names:
        if name not in value:
            continue
        candidate = value[name]
        if candidate is not None and type(candidate) is not bool:
            errors.append(f"EVIDENCE_TYPE_ERROR:video_verdict.{name}")
        else:
            kwargs[name] = candidate

    phase = value.get("first_actual_failure_phase", "")
    if type(phase) is not str:
        errors.append(
            "EVIDENCE_TYPE_ERROR:video_verdict.first_actual_failure_phase"
        )
    else:
        kwargs["first_actual_failure_phase"] = phase

    notes = value.get("notes", ())
    if not isinstance(notes, (list, tuple)) or any(
        type(note) is not str for note in notes
    ):
        errors.append("EVIDENCE_TYPE_ERROR:video_verdict.notes")
    else:
        kwargs["notes"] = tuple(notes)
    return ManualVideoVerdict(**kwargs)


def _criterion_passed(
    physical: Mapping[str, Any],
    name: str,
    errors: list[str],
) -> bool | None:
    criteria = _optional_mapping_field(
        physical, "criteria", "physical_evidence", errors
    )
    if criteria is None or name not in criteria:
        return None
    record = criteria[name]
    if not isinstance(record, Mapping):
        errors.append(f"EVIDENCE_TYPE_ERROR:physical_evidence.criteria.{name}")
        return None
    return _optional_bool_field(
        record,
        "passed",
        f"physical_evidence.criteria.{name}",
        errors,
    )


def _diagnostic_criterion_passed(
    physical: Mapping[str, Any],
    name: str,
    secondary: list[str],
) -> bool | None:
    criteria = physical.get("criteria")
    if criteria is None:
        return None
    if not isinstance(criteria, Mapping):
        secondary.append(
            "SECONDARY_DIAGNOSTIC_UNAVAILABLE:physical_evidence.criteria"
        )
        return None
    if name not in criteria:
        return None
    record = criteria[name]
    path = f"physical_evidence.criteria.{name}"
    if not isinstance(record, Mapping):
        secondary.append(f"SECONDARY_DIAGNOSTIC_UNAVAILABLE:{path}")
        return None
    return _diagnostic_bool_field(record, "passed", path, secondary)


def _valid_video_artifact(
    result: Mapping[str, Any],
    errors: list[str],
) -> bool:
    video = result.get("video")
    if video is None:
        return False
    if not isinstance(video, Mapping):
        errors.append("EVIDENCE_TYPE_ERROR:completed_result.video")
        return False

    required = (
        "valid",
        "actual_viewport_video",
        "artifact_valid",
        "full_decode_all_frames",
    )
    values: list[bool] = []
    for key in required:
        value = video.get(key)
        if value is None:
            return False
        if type(value) is not bool:
            errors.append(f"EVIDENCE_TYPE_ERROR:completed_result.video.{key}")
            return False
        values.append(value)

    full_decode = video.get("full_decode")
    if not isinstance(full_decode, Mapping):
        return False
    decode_valid = full_decode.get("valid")
    if type(decode_valid) is not bool:
        if decode_valid is not None:
            errors.append(
                "EVIDENCE_TYPE_ERROR:completed_result.video.full_decode.valid"
            )
        return False
    return all(values) and decode_valid


def _final_wheel_classes(
    physical: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    errors: list[str],
    secondary: list[str],
) -> dict[str, str]:
    def parse(candidate: object, path: str) -> dict[str, str] | None:
        if not isinstance(candidate, Mapping):
            errors.append(f"EVIDENCE_TYPE_ERROR:{path}")
            return None
        parsed: dict[str, str] = {}
        for leg in LEGS:
            value = candidate.get(leg)
            if type(value) is not str or not value:
                errors.append(f"EVIDENCE_TYPE_ERROR:{path}.{leg}")
                return None
            parsed[leg] = value.upper()
        return parsed

    physical_classes = None
    telemetry_classes = None
    if physical.get("final_wheel_contact_classes") is not None:
        physical_classes = parse(
            physical["final_wheel_contact_classes"],
            "physical_evidence.final_wheel_contact_classes",
        )
    if telemetry.get("wheel_contact_classes") is not None:
        telemetry_classes = parse(
            telemetry["wheel_contact_classes"],
            "final_telemetry_row.wheel_contact_classes",
        )
    if physical_classes is None and telemetry_classes is None:
        secondary.append("FINAL_WHEEL_CLASSES_UNAVAILABLE")
        return {}
    if (
        physical_classes is not None
        and telemetry_classes is not None
        and physical_classes != telemetry_classes
    ):
        errors.append("EVIDENCE_CONFLICT:final_wheel_classes")
    return dict(physical_classes or telemetry_classes or {})


def _derived_crossing(
    result: Mapping[str, Any],
    physical: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    video: ManualVideoVerdict,
    errors: list[str],
) -> bool | None:
    claims: list[bool] = []
    for mapping, path in (
        (result, "completed_result"),
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        value = _optional_bool_field(
            mapping, "body_crossed_front_face", path, errors
        )
        if value is not None:
            claims.append(value)
    if video.body_crossed_front_face is not None:
        claims.append(video.body_crossed_front_face)

    clearances = telemetry.get("wheel_front_face_clearance_m")
    if clearances is not None:
        if not isinstance(clearances, Mapping):
            errors.append(
                "EVIDENCE_TYPE_ERROR:final_telemetry_row."
                "wheel_front_face_clearance_m"
            )
        else:
            values: list[float] = []
            valid = True
            for leg in LEGS:
                raw = clearances.get(leg)
                if type(raw) not in (int, float) or not math.isfinite(float(raw)):
                    errors.append(
                        "EVIDENCE_TYPE_ERROR:final_telemetry_row."
                        f"wheel_front_face_clearance_m.{leg}"
                    )
                    valid = False
                else:
                    values.append(float(raw))
            if valid:
                claims.append(all(value > 0.0 for value in values))

    if False in claims:
        return False
    if True in claims:
        return True
    return None


def _derived_required_lift(
    physical: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    video: ManualVideoVerdict,
    errors: list[str],
) -> bool | None:
    claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        value = _optional_bool_field(
            mapping, "required_leg_lift_completed", path, errors
        )
        if value is not None:
            claims.append(value)
    if video.required_leg_lift_completed is not None:
        claims.append(video.required_leg_lift_completed)

    traversal = physical.get("traversal")
    if traversal is not None:
        if not isinstance(traversal, Mapping):
            errors.append("EVIDENCE_TYPE_ERROR:physical_evidence.traversal")
        else:
            legs = traversal.get("legs")
            if not isinstance(legs, Mapping):
                errors.append(
                    "EVIDENCE_TYPE_ERROR:physical_evidence.traversal.legs"
                )
            else:
                all_sequences_complete = True
                explicit_sequence_failure = False
                for leg in LEGS:
                    record = legs.get(leg)
                    leg_path = f"physical_evidence.traversal.legs.{leg}"
                    if not isinstance(record, Mapping):
                        errors.append(f"EVIDENCE_TYPE_ERROR:{leg_path}")
                        all_sequences_complete = False
                        continue
                    airborne = _optional_bool_field(
                        record, "airborne_seen_before_top", leg_path, errors
                    )
                    crossing = _number_field(
                        record, "front_face_crossing_s", leg_path, errors
                    )
                    if airborne is False:
                        explicit_sequence_failure = True
                    if airborne is not True or crossing is None:
                        all_sequences_complete = False
                if explicit_sequence_failure:
                    claims.append(False)
                elif all_sequences_complete:
                    claims.append(True)

    if False in claims:
        return False
    if True in claims:
        return True
    return None


def _derived_recoverable(
    result: Mapping[str, Any],
    physical: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    video: ManualVideoVerdict,
    final_classes: Mapping[str, str],
    errors: list[str],
) -> bool | None:
    claims: list[bool] = []
    for mapping, path in (
        (result, "completed_result"),
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        value = _optional_bool_field(
            mapping, "final_recoverable", path, errors
        )
        if value is not None:
            claims.append(value)
    if video.final_recoverable is not None:
        claims.append(video.final_recoverable)
    if video.irrecoverable is not None:
        claims.append(not video.irrecoverable)

    state = _optional_string_field(
        telemetry, "stability_state", "final_telemetry_row", errors
    )
    if state is not None:
        normalized = state.strip().lower()
        if normalized in {"fallen", "irrecoverable", "stuck"}:
            claims.append(False)
        elif normalized in {"safe", "stable", "recoverable"}:
            active = _optional_int_field(
                telemetry,
                "active_contact_count",
                "final_telemetry_row",
                errors,
            )
            inferred_support_count = sum(
                contact_class != "AIR"
                for contact_class in final_classes.values()
            )
            support_count = active if active is not None else inferred_support_count
            if support_count >= 3:
                claims.append(True)

    if False in claims:
        return False
    if True in claims:
        return True
    return None


def _first_available_count(
    result: Mapping[str, Any],
    direct_key: str,
    nested_keys: tuple[str, ...],
    errors: list[str],
) -> int | None:
    direct = _int_field(result, direct_key, "completed_result", errors)
    if direct is not None:
        return direct
    current: object = result
    path = "completed_result"
    for key in nested_keys[:-1]:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
        path = f"{path}.{key}"
        if not isinstance(current, Mapping):
            errors.append(f"EVIDENCE_TYPE_ERROR:{path}")
            return None
    if not isinstance(current, Mapping):
        return None
    return _int_field(current, nested_keys[-1], path, errors)


def _peak(
    result: Mapping[str, Any],
    physical: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    axis: str,
    errors: list[str],
    secondary: list[str],
) -> float | None:
    values: list[float] = []
    key = f"maximum_abs_{axis}_rad"
    for mapping, path in (
        (result, "completed_result"),
        (physical, "physical_evidence"),
    ):
        value = _diagnostic_number_field(mapping, key, path, secondary)
        if value is not None:
            values.append(abs(value))
    final_value = _number_field(
        telemetry, f"base_{axis}_rad", "final_telemetry_row", errors
    )
    if final_value is not None:
        values.append(abs(final_value))
    return max(values) if values else None


def classify_replay_task(
    *,
    completed_result: Mapping[str, object],
    physical_evidence: Mapping[str, object],
    final_telemetry_row: Mapping[str, object],
    video_verdict: ManualVideoVerdict | Mapping[str, object] | None = None,
) -> TaskSuccessAssessment:
    """Classify one completed Fast Replay at Gate-A task level.

    Wrongly typed or missing core facts cannot produce success.  Secondary
    diagnostics can lower the posture result, but cannot independently produce
    ``REPLAY_TASK_FAIL``.
    """

    result = _require_mapping(completed_result, "completed_result")
    physical = _require_mapping(physical_evidence, "physical_evidence")
    telemetry = _require_mapping(final_telemetry_row, "final_telemetry_row")

    validation_errors: list[str] = []
    hard: list[str] = []
    not_evaluated: list[str] = []
    secondary: list[str] = []
    notes: list[str] = []
    video = _normalise_video_verdict(video_verdict, validation_errors)
    notes.extend(video.notes)

    for name, payload in (
        ("completed_result", result),
        ("physical_evidence", physical),
        ("final_telemetry_row", telemetry),
    ):
        for path in _nonfinite_paths(payload, name):
            if _is_core_nonfinite_path(path):
                hard.append(f"NONFINITE_CORE_STATE:{path}")
            else:
                secondary.append(
                    f"SECONDARY_DIAGNOSTIC_UNAVAILABLE:{path}"
                )
    _validate_normal_session_core_telemetry(
        telemetry, hard, validation_errors
    )

    version_claims: list[str] = []
    for mapping, path in (
        (result, "completed_result"),
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        source_version = _string_field(
            mapping, "source_version", path, validation_errors
        )
        if source_version:
            version_claims.append(source_version)
    if not version_claims:
        not_evaluated.append("EVIDENCE_MISSING:source_version")
        version = ""
    else:
        version = version_claims[0]
        if any(claim != version for claim in version_claims[1:]):
            not_evaluated.append("EVIDENCE_CONFLICT:source_version")

    step_count = _first_available_count(
        result,
        "step_count",
        ("scheduler_status", "progress_detail", "total_steps"),
        validation_errors,
    )
    if step_count is None:
        not_evaluated.append("EVIDENCE_MISSING:step_count")
        step_count = 0
    fast_segment_count = _first_available_count(
        result,
        "plan_segment_count",
        ("scheduler_status", "segment_count"),
        validation_errors,
    )
    if fast_segment_count is None:
        not_evaluated.append("EVIDENCE_MISSING:fast_segment_count")
        fast_segment_count = 0

    plan_event_count = _int_field(
        result, "plan_event_count", "completed_result", validation_errors
    )
    if plan_event_count is None:
        not_evaluated.append("EVIDENCE_MISSING:plan_event_count")
    scheduler_status = _mapping_field(
        result, "scheduler_status", "completed_result", validation_errors
    )
    if scheduler_status is None:
        not_evaluated.append("EVIDENCE_MISSING:scheduler_status")
    else:
        for key in ("count", "event_count", "events_sent", "index"):
            observed = _int_field(
                scheduler_status,
                key,
                "completed_result.scheduler_status",
                validation_errors,
            )
            if observed is None:
                not_evaluated.append(
                    f"EVIDENCE_MISSING:scheduler_status.{key}"
                )
            elif plan_event_count is not None and observed != plan_event_count:
                hard.append("SCHEDULER_EVENT_COUNT_MISMATCH")
        for key in ("segment_count", "segment_index"):
            observed = _int_field(
                scheduler_status,
                key,
                "completed_result.scheduler_status",
                validation_errors,
            )
            if observed is None:
                not_evaluated.append(
                    f"EVIDENCE_MISSING:scheduler_status.{key}"
                )
            elif observed != fast_segment_count:
                hard.append("SCHEDULER_SEGMENT_COUNT_MISMATCH")

    dispatch_complete = _bool_field(
        result, "dispatch_complete", "completed_result", validation_errors
    )
    if dispatch_complete is False:
        hard.append("COMMAND_DISPATCH_INCOMPLETE")
    elif dispatch_complete is None and not any(
        error.endswith("completed_result.dispatch_complete")
        for error in validation_errors
    ):
        not_evaluated.append("EVIDENCE_MISSING:dispatch_complete")

    scheduler_complete = _bool_field(
        result, "scheduler_complete", "completed_result", validation_errors
    )
    if scheduler_complete is False:
        hard.append("SCHEDULER_INCOMPLETE")
    elif scheduler_complete is None and not any(
        error.endswith("completed_result.scheduler_complete")
        for error in validation_errors
    ):
        not_evaluated.append("EVIDENCE_MISSING:scheduler_complete")
    stop_reason = _string_field(
        result, "scheduler_stop_reason", "completed_result", validation_errors
    )
    if stop_reason is None:
        not_evaluated.append("EVIDENCE_MISSING:scheduler_stop_reason")
    elif stop_reason != "complete":
        hard.append("SCHEDULER_INCOMPLETE")

    dispatch_ledger = _optional_mapping_field(
        result, "dispatch_ledger", "completed_result", validation_errors
    )
    if dispatch_ledger is not None:
        ledger_complete = _bool_field(
            dispatch_ledger,
            "complete",
            "completed_result.dispatch_ledger",
            validation_errors,
        )
        if ledger_complete is False:
            hard.append("COMMAND_DISPATCH_INCOMPLETE")
        for key, expected in (
            ("plan_event_count", plan_event_count),
            ("plan_segment_count", fast_segment_count),
        ):
            if key not in dispatch_ledger:
                continue
            observed = _int_field(
                dispatch_ledger,
                key,
                "completed_result.dispatch_ledger",
                validation_errors,
            )
            if observed is not None and expected is not None and observed != expected:
                hard.append("DISPATCH_LEDGER_PLAN_COUNT_MISMATCH")

    for key in ("timed_out", "simulation_app_stopped"):
        value = _bool_field(result, key, "completed_result", validation_errors)
        if value is True:
            not_evaluated.append("INFRASTRUCTURE_PROCESS_FAILURE")
    lifecycle = _optional_mapping_field(
        result, "lifecycle", "completed_result", validation_errors
    )
    if lifecycle is not None:
        failed = _bool_field(
            lifecycle,
            "failed",
            "completed_result.lifecycle",
            validation_errors,
        )
        if failed is True:
            not_evaluated.append("INFRASTRUCTURE_PROCESS_FAILURE")
    artifact_valid = _bool_field(
        result, "artifact_valid", "completed_result", validation_errors
    )
    if artifact_valid is False:
        not_evaluated.append("ARTIFACT_INVALID")

    manual_video_available = (
        video_verdict is not None and video.task_completed is not None
    )
    if not manual_video_available and not _valid_video_artifact(
        result, validation_errors
    ):
        not_evaluated.append("VIDEO_EVIDENCE_UNAVAILABLE")

    for mapping, path in (
        (result, "completed_result"),
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        if _optional_bool_field(
            mapping,
            "nonfinite_core_state_detected",
            path,
            validation_errors,
        ) is True:
            hard.append(
                f"NONFINITE_CORE_STATE:{path}.nonfinite_core_state_detected"
            )
        for key in (
            "second_isaac_instance",
            "second_isaac_process",
            "second_simulator_process",
            "second_simulator_process_detected",
        ):
            if _optional_bool_field(
                mapping, key, path, validation_errors
            ) is True:
                not_evaluated.append("SECOND_SIMULATOR_PROCESS")
        for key in ("active_leg_trapped", "leg_trapped_by_obstacle"):
            if _optional_bool_field(
                mapping, key, path, validation_errors
            ) is True:
                hard.append("ACTIVE_LEG_TRAPPED")
        for key in (
            "unsafe_joint_target",
            "joint_target_outside_safe_limit",
        ):
            if _optional_bool_field(
                mapping, key, path, validation_errors
            ) is True:
                hard.append("UNSAFE_JOINT_TARGET")

    motion_ready = _optional_bool_field(
        result, "motion_start_ready", "completed_result", validation_errors
    )
    targets_applied = _optional_bool_field(
        result, "actuator_targets_applied", "completed_result", validation_errors
    )
    if motion_ready is False:
        not_evaluated.append("MOTION_START_NOT_READY")
    if targets_applied is False:
        hard.append("ACTUATOR_TARGET_NOT_APPLIED")

    final_classes = _final_wheel_classes(
        physical, telemetry, validation_errors, secondary
    )
    body_crossed = _derived_crossing(
        result, physical, telemetry, video, validation_errors
    )
    lift_completed = _derived_required_lift(
        physical, telemetry, video, validation_errors
    )
    final_recoverable = _derived_recoverable(
        result,
        physical,
        telemetry,
        video,
        final_classes,
        validation_errors,
    )

    if body_crossed is False:
        hard.append("BODY_NOT_CROSSED_FRONT_FACE")
    elif body_crossed is None:
        not_evaluated.append("EVIDENCE_MISSING:body_crossed_front_face")
    if lift_completed is False:
        hard.append("REQUIRED_LEG_LIFT_MISSING")
    elif lift_completed is None:
        not_evaluated.append("EVIDENCE_MISSING:required_leg_lift_completed")
    if final_recoverable is False:
        hard.append("FINAL_STATE_IRRECOVERABLE")
    elif final_recoverable is None:
        not_evaluated.append("EVIDENCE_MISSING:final_recoverable")

    fall_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        for key in ("robot_fell", "fall_detected"):
            value = _optional_bool_field(
                mapping, key, path, validation_errors
            )
            if value is not None:
                fall_claims.append(value)
    if video.robot_fell is not None:
        fall_claims.append(video.robot_fell)
    attitude_safe = _criterion_passed(
        physical, "attitude_safe", validation_errors
    )
    final_stability = telemetry.get("stability_state")
    fall_safe_observed = (
        False in fall_claims
        or attitude_safe is True
        or (
            type(final_stability) is str
            and final_stability.strip().lower() in {"safe", "stable", "recoverable"}
        )
    )
    if True in fall_claims:
        hard.append("ROBOT_FALL")
    elif not fall_safe_observed:
        not_evaluated.append("EVIDENCE_MISSING:robot_fall")

    stuck_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        for key in ("body_stuck", "permanently_stuck"):
            value = _optional_bool_field(
                mapping, key, path, validation_errors
            )
            if value is not None:
                stuck_claims.append(value)
    if video.body_stuck is not None:
        stuck_claims.append(video.body_stuck)
    if True in stuck_claims:
        hard.append("BODY_STUCK")
    elif body_crossed is None and False not in stuck_claims:
        not_evaluated.append("EVIDENCE_MISSING:body_stuck")

    illegal_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        for key in (
            "wheel_drive_up_without_required_lift",
            "illegal_drive_up",
        ):
            value = _optional_bool_field(
                mapping, key, path, validation_errors
            )
            if value is not None:
                illegal_claims.append(value)
    traversal = _optional_mapping_field(
        physical, "traversal", "physical_evidence", validation_errors
    )
    if traversal is not None:
        illegal = _optional_bool_field(
            traversal,
            "any_illegal_drive_up",
            "physical_evidence.traversal",
            validation_errors,
        )
        if illegal is not None:
            illegal_claims.append(illegal)
    if video.wheel_drive_up_without_required_lift is not None:
        illegal_claims.append(video.wheel_drive_up_without_required_lift)
    no_illegal_criterion = _criterion_passed(
        physical, "no_illegal_drive_up", validation_errors
    )
    if no_illegal_criterion is not None:
        illegal_claims.append(not no_illegal_criterion)
    if True in illegal_claims:
        hard.append("WHEEL_DRIVE_UP_WITHOUT_REQUIRED_LIFT")
    elif not illegal_claims:
        not_evaluated.append("EVIDENCE_MISSING:illegal_drive_up")

    collision_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        value = _optional_bool_field(
            mapping, "dangerous_collision", path, validation_errors
        )
        if value is not None:
            collision_claims.append(value)
    collision_count = _optional_int_field(
        physical,
        "dangerous_collision_count",
        "physical_evidence",
        validation_errors,
    )
    if collision_count is not None:
        collision_claims.append(collision_count > 0)
    if video.dangerous_body_collision is not None:
        collision_claims.append(video.dangerous_body_collision)
    collision_safe = _criterion_passed(
        physical, "collision_safe", validation_errors
    )
    if collision_safe is not None:
        collision_claims.append(not collision_safe)
    if True in collision_claims:
        hard.append("DANGEROUS_BODY_COLLISION")
    elif not collision_claims:
        not_evaluated.append("EVIDENCE_MISSING:dangerous_collision")

    joint_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        value = _optional_bool_field(
            mapping, "joint_limit_violation", path, validation_errors
        )
        if value is not None:
            joint_claims.append(value)
    joint_count = _optional_int_field(
        physical,
        "joint_limit_violation_count",
        "physical_evidence",
        validation_errors,
    )
    if joint_count is not None:
        joint_claims.append(joint_count > 0)
    if video.joint_limit_violation is not None:
        joint_claims.append(video.joint_limit_violation)
    joint_safe = _criterion_passed(
        physical, "joint_limits_safe", validation_errors
    )
    if joint_safe is not None:
        joint_claims.append(not joint_safe)
    if True in joint_claims:
        hard.append("JOINT_LIMIT_VIOLATION")
    elif not joint_claims:
        not_evaluated.append("EVIDENCE_MISSING:joint_limit_violation")

    penetration_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        value = _optional_bool_field(
            mapping, "severe_penetration", path, validation_errors
        )
        if value is not None:
            penetration_claims.append(value)
    if video.severe_penetration is not None:
        penetration_claims.append(video.severe_penetration)
    penetration_safe = _criterion_passed(
        physical, "penetration_safe", validation_errors
    )
    if penetration_safe is not None:
        penetration_claims.append(not penetration_safe)
    max_penetration = _number_field(
        physical,
        "maximum_collision_penetration_m",
        "physical_evidence",
        validation_errors,
    )
    allowed_penetration = _number_field(
        physical,
        "maximum_allowed_penetration_m",
        "physical_evidence",
        validation_errors,
    )
    if max_penetration is not None and allowed_penetration is not None:
        penetration_claims.append(max_penetration > allowed_penetration)
    if True in penetration_claims:
        hard.append("SEVERE_PENETRATION")
    elif not penetration_claims:
        not_evaluated.append("EVIDENCE_MISSING:severe_penetration")

    if video.irrecoverable is True:
        hard.append("FINAL_STATE_IRRECOVERABLE")
    if video.task_completed is False:
        hard.append("MANUAL_VIDEO_CONFIRMED_FAILURE")

    contact_drift_safe = _diagnostic_criterion_passed(
        physical, "contact_drift_safe", secondary
    )
    explicit_contact_drift_safe = _diagnostic_bool_field(
        physical,
        "contact_drift_safe",
        "physical_evidence",
        secondary,
    )
    if contact_drift_safe is False or explicit_contact_drift_safe is False:
        secondary.append("CONTACT_DRIFT_DIAGNOSTIC_FAILED")

    legacy_lift_safe = _diagnostic_criterion_passed(
        physical, "all_legs_linkage_lift_valid", secondary
    )
    if legacy_lift_safe is False and lift_completed is True:
        secondary.append("LEGACY_STRICT_LIFT_CRITERION_FAILED")

    all_top = _diagnostic_bool_field(
        physical, "final_all_top", "physical_evidence", secondary
    )
    all_loaded = _diagnostic_bool_field(
        physical, "final_all_loaded", "physical_evidence", secondary
    )
    final_stable = _diagnostic_bool_field(
        physical,
        "final_velocity_stable",
        "physical_evidence",
        secondary,
    )
    if all_top is False:
        secondary.append("FINAL_NOT_ALL_TOP")
    if all_loaded is False:
        secondary.append("FINAL_NOT_ALL_LOADED")
    if final_stable is False:
        secondary.append("STRICT_REST_INCOMPLETE")

    if len(final_classes) == len(LEGS):
        air_count = sum(value == "AIR" for value in final_classes.values())
        if air_count == 1:
            secondary.append("FINAL_SINGLE_WHEEL_AIR")
        elif any(value != "TOP" for value in final_classes.values()):
            secondary.append("FINAL_POSTURE_CONTACTS_INCOMPLETE")

    posture_complete_claims: list[bool] = []
    for mapping, path in (
        (physical, "physical_evidence"),
        (telemetry, "final_telemetry_row"),
    ):
        for key in ("final_posture_complete", "final_home_pose_complete"):
            value = _diagnostic_bool_field(
                mapping, key, path, secondary
            )
            if value is not None:
                posture_complete_claims.append(value)
                if value is False and key == "final_home_pose_complete":
                    secondary.append("FINAL_HOME_POSE_INCOMPLETE")
    if video.posture_incomplete is not None:
        posture_complete_claims.append(not video.posture_incomplete)
        if video.posture_incomplete:
            secondary.append("MANUAL_VIDEO_POSTURE_INCOMPLETE")

    if final_classes and all(value == "TOP" for value in final_classes.values()):
        posture_complete_claims.append(True)
    elif final_classes:
        posture_complete_claims.append(False)
    if all_top is not None:
        posture_complete_claims.append(all_top)
    if all_loaded is not None:
        posture_complete_claims.append(all_loaded)
    if final_stable is not None:
        posture_complete_claims.append(final_stable)
    if not posture_complete_claims:
        secondary.append("POSTURE_EVIDENCE_INCOMPLETE")
        posture_complete = False
    else:
        posture_complete = False not in posture_complete_claims

    not_evaluated.extend(validation_errors)
    hard = list(_unique(hard))
    not_evaluated = list(_unique(not_evaluated))
    secondary = list(_unique(secondary))

    nonfinite_hard_failure = any(
        reason.startswith("NONFINITE_CORE_STATE:") for reason in hard
    )
    hard_failure_blocked = _hard_failure_is_blocked(not_evaluated)
    if hard and not hard_failure_blocked:
        evaluation_status = EVALUATED
        task_result = REPLAY_TASK_FAIL
        posture_result = POSTURE_NOT_APPLICABLE
        classification_reasons = (
            (
                "a non-finite task state was observed as a hard safety failure"
                if nonfinite_hard_failure
                else "an evaluated task-level hard failure was observed"
            ),
        )
    elif not_evaluated:
        evaluation_status = NOT_EVALUATED
        task_result = REPLAY_TASK_NOT_EVALUATED
        posture_result = POSTURE_NOT_APPLICABLE
        classification_reasons = (
            "task was not evaluated because infrastructure or required "
            "evidence was invalid or incomplete",
        )
    elif posture_complete:
        evaluation_status = EVALUATED
        task_result = REPLAY_TASK_SUCCESS
        posture_result = POSTURE_COMPLETE
        classification_reasons = (
            "task core complete: dispatch, crossing, required lifting, "
            "recoverability, and hard-safety checks passed",
            "final posture complete; secondary diagnostics do not redefine traversal success",
        )
    else:
        evaluation_status = EVALUATED
        task_result = REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
        posture_result = POSTURE_INCOMPLETE
        classification_reasons = (
            "task core complete: dispatch, crossing, required lifting, "
            "recoverability, and hard-safety checks passed",
            "final posture or settling remains incomplete but recoverable",
        )

    first_failure_phase = ""
    if evaluation_status == EVALUATED and task_result == REPLAY_TASK_FAIL:
        if video.first_actual_failure_phase:
            first_failure_phase = video.first_actual_failure_phase
        else:
            for mapping, path in (
                (physical, "physical_evidence"),
                (telemetry, "final_telemetry_row"),
                (result, "completed_result"),
            ):
                phase = _string_field(
                    mapping,
                    "first_actual_failure_phase",
                    path,
                    validation_errors,
                )
                if phase:
                    first_failure_phase = phase
                    break

    video_path = _string_field(
        result, "video_path", "completed_result", validation_errors
    )
    if video_path is None:
        video_path = ""
        secondary = list(_unique((*secondary, "VIDEO_PATH_UNAVAILABLE")))

    peak_roll = _peak(
        result,
        physical,
        telemetry,
        "roll",
        validation_errors,
        secondary,
    )
    peak_pitch = _peak(
        result,
        physical,
        telemetry,
        "pitch",
        validation_errors,
        secondary,
    )

    # Peak/video extraction should not normally add errors after classification,
    # but a malformed present field still cannot silently yield success.
    late_errors = [
        error for error in validation_errors if error not in not_evaluated
    ]
    if late_errors:
        not_evaluated = list(_unique((*not_evaluated, *late_errors)))
        if not hard or _hard_failure_is_blocked(not_evaluated):
            evaluation_status = NOT_EVALUATED
            task_result = REPLAY_TASK_NOT_EVALUATED
            posture_result = POSTURE_NOT_APPLICABLE
            classification_reasons = (
                "task was not evaluated because infrastructure or required "
                "evidence was invalid or incomplete",
            )

    return TaskSuccessAssessment(
        version=version,
        step_count=step_count,
        fast_segment_count=fast_segment_count,
        evaluation_status=evaluation_status,
        task_result=task_result,
        posture_result=posture_result,
        body_crossed_front_face=body_crossed is True,
        required_leg_lift_completed=lift_completed is True,
        final_recoverable=final_recoverable is True,
        final_wheel_classes=dict(final_classes),
        peak_roll=peak_roll,
        peak_pitch=peak_pitch,
        video_path=video_path,
        first_actual_failure_phase=first_failure_phase,
        hard_failure_reasons=tuple(hard),
        not_evaluated_reasons=tuple(not_evaluated),
        secondary_diagnostics=tuple(secondary),
        notes=tuple(notes),
        classification_reasons=tuple(classification_reasons),
    )


__all__ = [
    "EVALUATED",
    "ManualVideoVerdict",
    "NOT_EVALUATED",
    "POSTURE_COMPLETE",
    "POSTURE_INCOMPLETE",
    "POSTURE_NOT_APPLICABLE",
    "REPLAY_TASK_FAIL",
    "REPLAY_TASK_NOT_EVALUATED",
    "REPLAY_TASK_SUCCESS",
    "REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE",
    "TABLE_COLUMNS",
    "TaskSuccessAssessment",
    "classify_replay_task",
]
