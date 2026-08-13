"""Isaac-free command vocabulary shared by height replay modules.

The names and command words mirror ``real_robot_ui_controller`` so saved JSONL
steps stay compatible, while the command-space limits follow the height replay
requirement.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


SERVO_JOINT_NAMES = [
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
]

WHEEL_JOINT_NAMES = [
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
]

WHEEL_SHORT_NAMES = {
    "fl": "front_left_ankle",
    "fr": "front_right_ankle",
    "rl": "rear_left_ankle",
    "rr": "rear_right_ankle",
}

WHEEL_NAME_TO_SHORT = {value: key for key, value in WHEEL_SHORT_NAMES.items()}

SERVO_SHORT_NAMES = {
    "front_left_hip": "fl_hip",
    "front_left_knee": "fl_knee",
    "front_right_hip": "fr_hip",
    "front_right_knee": "fr_knee",
    "rear_left_hip": "rl_hip",
    "rear_left_knee": "rl_knee",
    "rear_right_hip": "rr_hip",
    "rear_right_knee": "rr_knee",
}

SERVO_NAME_ALIASES = {short: full for full, short in SERVO_SHORT_NAMES.items()}

HIP_LIMIT_DEG = (-135.0, 135.0)
KNEE_LIMIT_DEG = (-60.0, 210.0)

KNEE_JOINT_NAMES = {
    "front_left_knee",
    "front_right_knee",
    "rear_left_knee",
    "rear_right_knee",
}

JOINT_COMMAND_SIGN = {
    "front_left_hip": 1.0,
    "front_left_knee": 1.0,
    "front_right_hip": 1.0,
    "front_right_knee": 1.0,
    "rear_left_hip": -1.0,
    "rear_left_knee": -1.0,
    "rear_right_hip": -1.0,
    "rear_right_knee": -1.0,
}

WHEEL_FORWARD_SIGN = {
    "front_left_ankle": -1.0,
    "rear_left_ankle": -1.0,
    "front_right_ankle": 1.0,
    "rear_right_ankle": 1.0,
}

JOINT_ALIASES = {
    "left_front_joint_1": "front_left_hip",
    "left_front_joint_2": "front_left_knee",
    "right_front_joint_1": "front_right_hip",
    "right_front_joint_2": "front_right_knee",
    "left_rear_joint_1": "rear_left_hip",
    "left_rear_joint_2": "rear_left_knee",
    "right_rear_joint_1": "rear_right_hip",
    "right_rear_joint_2": "rear_right_knee",
    **SERVO_NAME_ALIASES,
}

SERVO_GROUPS = {
    "fl": ["front_left_hip", "front_left_knee"],
    "fr": ["front_right_hip", "front_right_knee"],
    "rl": ["rear_left_hip", "rear_left_knee"],
    "rr": ["rear_right_hip", "rear_right_knee"],
    "front_left": ["front_left_hip", "front_left_knee"],
    "front_right": ["front_right_hip", "front_right_knee"],
    "rear_left": ["rear_left_hip", "rear_left_knee"],
    "rear_right": ["rear_right_hip", "rear_right_knee"],
    "front": ["front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee"],
    "rear": ["rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee"],
    "left": ["front_left_hip", "front_left_knee", "rear_left_hip", "rear_left_knee"],
    "right": ["front_right_hip", "front_right_knee", "rear_right_hip", "rear_right_knee"],
    "hips": ["front_left_hip", "front_right_hip", "rear_left_hip", "rear_right_hip"],
    "hip": ["front_left_hip", "front_right_hip", "rear_left_hip", "rear_right_hip"],
    "knees": ["front_left_knee", "front_right_knee", "rear_left_knee", "rear_right_knee"],
    "knee": ["front_left_knee", "front_right_knee", "rear_left_knee", "rear_right_knee"],
    "all": list(SERVO_JOINT_NAMES),
}

WHEEL_MAX_SPEED_RPM = 20.0
DEFAULT_MAX_WHEEL_SPEED_RAD_S = WHEEL_MAX_SPEED_RPM * 2.0 * math.pi / 60.0


@dataclass
class CommandMessage:
    text: str
    source: str = "ui"
    kind: str = "command"
    target: str = ""
    log_history: bool = True
    quiet: bool = False
    playback_label: str = ""
    playback_event_index: int | None = None
    playback_event_count: int = 0
    playback_final_time_s: float = 0.0
    source_step: int | None = None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def is_float_token(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def command_limits_for_servo(joint_name: str) -> tuple[float, float]:
    return KNEE_LIMIT_DEG if joint_name in KNEE_JOINT_NAMES else HIP_LIMIT_DEG


def clamp_servo_command(joint_name: str, angle_deg: float) -> float:
    lower, upper = command_limits_for_servo(joint_name)
    return clamp(float(angle_deg), lower, upper)


def resolve_servo_name(name: str) -> str:
    key = name.strip().lower()
    candidate = JOINT_ALIASES.get(key, key)
    if candidate not in SERVO_JOINT_NAMES:
        raise ValueError(f"Unknown servo joint '{name}'.")
    return candidate


def resolve_wheel_name(name: str) -> str:
    key = name.strip().lower()
    candidate = WHEEL_SHORT_NAMES.get(key, key)
    if candidate not in WHEEL_JOINT_NAMES:
        raise ValueError(f"Unknown wheel '{name}'. Use fl/fr/rl/rr or an ankle joint name.")
    return candidate


def resolve_servo_targets_for_command(name: str) -> list[str]:
    key = name.strip().lower()
    if key in SERVO_GROUPS:
        return list(SERVO_GROUPS[key])
    return [resolve_servo_name(key)]


def resolve_servo_targets_for_group_part(group: str, part: str) -> list[str]:
    part_key = part.strip().lower()
    if part_key not in {"hip", "knee"}:
        raise ValueError("Servo part must be 'hip' or 'knee'.")
    targets = resolve_servo_targets_for_command(group)
    filtered = [name for name in targets if name.endswith(f"_{part_key}")]
    if not filtered:
        raise ValueError(f"Group '{group}' has no {part_key} servo targets.")
    return filtered


def split_semicolon_commands(command: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == ";":
            text = "".join(current).strip()
            if text:
                parts.append(text)
            current = []
            continue
        current.append(char)
    text = "".join(current).strip()
    if text:
        parts.append(text)
    return parts


def parse_scalar(value: str) -> Any:
    raw = str(value).strip()
    if raw == "":
        return ""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"[-+]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw):
        return float(raw)
    return raw
