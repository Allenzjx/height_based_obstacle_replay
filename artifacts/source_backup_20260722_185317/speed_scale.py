"""Manual and playback speed scaling helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from command_model import is_float_token


@dataclass
class MotionScaleConfig:
    global_motion_speed_scale: float = 1.0
    wheel_speed_scale: float = 1.0
    servo_command_scale: float = 1.0
    playback_speed_scale: float = 1.0
    apply_to_manual_control: bool = True
    apply_to_playback: bool = True
    preserve_wheel_distance: bool = True

    @property
    def manual_wheel_scale(self) -> float:
        if not self.apply_to_manual_control:
            return 1.0
        return max(0.0, float(self.global_motion_speed_scale) * float(self.wheel_speed_scale))

    @property
    def manual_servo_scale(self) -> float:
        if not self.apply_to_manual_control:
            return 1.0
        return max(0.0, float(self.global_motion_speed_scale) * float(self.servo_command_scale))

    @property
    def playback_scale(self) -> float:
        if not self.apply_to_playback:
            return 1.0
        return max(0.1, min(5.0, float(self.global_motion_speed_scale) * float(self.playback_speed_scale)))


@dataclass
class ScaledCommand:
    raw_command: str
    scaled_command: str
    raw_speed_values: tuple[float, ...] = ()
    scaled_speed_values: tuple[float, ...] = ()
    was_wheel_command: bool = False
    warnings: tuple[str, ...] = ()


def scale_manual_motion_command(
    command: str,
    *,
    default_wheel_speed: float,
    max_wheel_speed: float,
    wheel_speed_scale: float,
    servo_command_scale: float = 1.0,
) -> ScaledCommand:
    raw = str(command).strip()
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return ScaledCommand(raw, raw)
    if not tokens:
        return ScaledCommand(raw, raw)

    verb = tokens[0].lower()
    scale = max(0.0, float(wheel_speed_scale))
    servo_scale = max(0.0, float(servo_command_scale))
    max_speed = abs(float(max_wheel_speed))

    def clamp(value: float) -> float:
        return max(-max_speed, min(max_speed, float(value)))

    def scaled_value(value: float) -> float:
        return clamp(float(value) * scale)

    if verb == "w":
        raw_value = float(default_wheel_speed)
        scaled = scaled_value(raw_value)
        return ScaledCommand(raw, f"wheel all {scaled:.6g}", (raw_value,), (scaled,), True)
    if verb == "s":
        raw_value = -float(default_wheel_speed)
        scaled = scaled_value(raw_value)
        return ScaledCommand(raw, f"wheel all {scaled:.6g}", (raw_value,), (scaled,), True)
    if verb == "a":
        left = -float(default_wheel_speed)
        right = float(default_wheel_speed)
        scaled_left = scaled_value(left)
        scaled_right = scaled_value(right)
        return ScaledCommand(raw, f"wheel {scaled_left:.6g} {scaled_right:.6g}", (left, right), (scaled_left, scaled_right), True)
    if verb == "d":
        left = float(default_wheel_speed)
        right = -float(default_wheel_speed)
        scaled_left = scaled_value(left)
        scaled_right = scaled_value(right)
        return ScaledCommand(raw, f"wheel {scaled_left:.6g} {scaled_right:.6g}", (left, right), (scaled_left, scaled_right), True)
    if verb in {"x", "stop"}:
        return ScaledCommand(raw, "wheel stop", (0.0,), (0.0,), True)
    if verb in {"wheels", "speed"}:
        if len(tokens) != 3:
            return ScaledCommand(raw, raw)
        left = float(tokens[1])
        right = float(tokens[2])
        scaled_left = scaled_value(left)
        scaled_right = scaled_value(right)
        return ScaledCommand(raw, f"{tokens[0]} {scaled_left:.6g} {scaled_right:.6g}", (left, right), (scaled_left, scaled_right), True)
    if verb == "wheel":
        args = tokens[1:]
        if not args:
            return ScaledCommand(raw, raw)
        if args[0].lower() == "stop":
            return ScaledCommand(raw, "wheel stop", (0.0,), (0.0,), True)
        if len(args) == 2 and is_float_token(args[0]):
            left = float(args[0])
            right = float(args[1])
            scaled_left = scaled_value(left)
            scaled_right = scaled_value(right)
            return ScaledCommand(raw, f"wheel {scaled_left:.6g} {scaled_right:.6g}", (left, right), (scaled_left, scaled_right), True)
        if len(args) == 2:
            value = float(args[1])
            scaled = scaled_value(value)
            return ScaledCommand(raw, f"wheel {args[0]} {scaled:.6g}", (value,), (scaled,), True)
    if verb in {"servo", "angle"}:
        scaled = list(tokens)
        try:
            if len(tokens) == 3:
                raw_value = float(tokens[2])
                scaled_value = _scaled_servo_value(raw_value, servo_scale)
                scaled[2] = f"{scaled_value:.6g}"
                warnings = _servo_scale_warnings(servo_scale)
                return ScaledCommand(raw, " ".join(shlex.quote(token) for token in scaled), (raw_value,), (scaled_value,), False, warnings)
            if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
                raw_value = float(tokens[3])
                scaled_value = _scaled_servo_value(raw_value, servo_scale)
                scaled[3] = f"{scaled_value:.6g}"
                warnings = _servo_scale_warnings(servo_scale)
                return ScaledCommand(raw, " ".join(shlex.quote(token) for token in scaled), (raw_value,), (scaled_value,), False, warnings)
        except (TypeError, ValueError):
            return ScaledCommand(raw, raw)
    return ScaledCommand(raw, raw)


def _scaled_servo_value(value: float, servo_scale: float) -> float:
    scaled = float(value) * float(servo_scale)
    return 0.0 if abs(scaled) < 1.0e-12 else scaled


def _servo_scale_warnings(servo_scale: float) -> tuple[str, ...]:
    if float(servo_scale) == 0.0:
        return ("servo_command_scale is 0; servo commands will not move",)
    return ()
