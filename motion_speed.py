"""Single worker-owned motion speed model for manual control and playback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "config" / "real_robot_motion_reference.yaml"


@dataclass(frozen=True)
class MotionReference:
    profile_id: str = "real-robot-motion-reference-v1"
    servo_reference_velocity_deg_s: float = 150.0
    servo_acceleration_limit_deg_s2: float | None = None
    servo_velocity_limit_deg_s: float | None = None
    wheel_reference_velocity_rad_s: float = math.pi / 6.0
    wheel_velocity_limit_rad_s: float = math.pi / 1.5
    wheel_radius_m: float | None = None
    wheel_transmission_ratio: float | None = None
    verified: bool = False
    verification_method: str = "mixed: source-config verified speeds; radius/transmission unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "servo_reference_velocity_deg_s": self.servo_reference_velocity_deg_s,
            "servo_acceleration_limit_deg_s2": self.servo_acceleration_limit_deg_s2,
            "servo_velocity_limit_deg_s": self.servo_velocity_limit_deg_s,
            "wheel_reference_velocity_rad_s": self.wheel_reference_velocity_rad_s,
            "wheel_velocity_limit_rad_s": self.wheel_velocity_limit_rad_s,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_transmission_ratio": self.wheel_transmission_ratio,
            "verified": self.verified,
            "verification_method": self.verification_method,
        }


class SpeedScale:
    """The only percentage model. Scaling is consumed only by the adapter."""

    MIN_PERCENT = 0.0
    MAX_PERCENT = 300.0

    def __init__(self, reference: MotionReference | None = None, *, speed_percent: float = 100.0):
        self.reference = reference or load_motion_reference()
        self._speed_percent = 100.0
        self.set_percent(speed_percent)

    @property
    def speed_percent(self) -> float:
        return self._speed_percent

    @property
    def scale(self) -> float:
        return self._speed_percent / 100.0

    def set_percent(self, value: float) -> float:
        percent = float(value)
        if not math.isfinite(percent):
            raise ValueError("speed_percent must be finite")
        self._speed_percent = max(self.MIN_PERCENT, min(self.MAX_PERCENT, percent))
        return self._speed_percent

    def servo_velocity(self) -> tuple[float, float]:
        requested = self.reference.servo_reference_velocity_deg_s * self.scale
        limit = self.reference.servo_velocity_limit_deg_s
        effective = requested if limit is None else min(requested, float(limit))
        return requested, effective

    def wheel_velocity(self, canonical_rad_s: float) -> tuple[float, float]:
        requested = float(canonical_rad_s) * self.scale
        limit = abs(self.reference.wheel_velocity_limit_rad_s)
        effective = max(-limit, min(limit, requested))
        return requested, effective

    def status(self) -> dict[str, Any]:
        servo_requested, servo_effective = self.servo_velocity()
        wheel_requested, wheel_effective = self.wheel_velocity(self.reference.wheel_reference_velocity_rad_s)
        return {
            "speed_percent": self.speed_percent,
            "speed_scale": self.scale,
            "reference_profile_id": self.reference.profile_id,
            "reference_verified": self.reference.verified,
            "requested_servo_velocity_deg_s": servo_requested,
            "effective_servo_velocity_deg_s": servo_effective,
            "requested_reference_wheel_velocity_rad_s": wheel_requested,
            "effective_reference_wheel_velocity_rad_s": wheel_effective,
            "semantics_version": "worker-motion-executor-v1",
        }


def load_motion_reference(path: str | Path = DEFAULT_REFERENCE_PATH) -> MotionReference:
    """Load the small authoritative YAML without adding a runtime dependency."""

    source = Path(path)
    values: dict[str, Any] = {}
    if source.exists():
        for raw in source.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line or line.startswith("-"):
                continue
            key, raw_value = (part.strip() for part in line.split(":", 1))
            if not key or not raw_value or raw_value.startswith("["):
                continue
            values[key] = _yaml_scalar(raw_value)
    defaults = MotionReference()
    return MotionReference(
        profile_id=str(values.get("profile_id", defaults.profile_id)),
        servo_reference_velocity_deg_s=float(values.get("servo_reference_velocity_deg_s", defaults.servo_reference_velocity_deg_s)),
        servo_acceleration_limit_deg_s2=_optional_float(values.get("servo_acceleration_limit_deg_s2")),
        servo_velocity_limit_deg_s=_optional_float(values.get("servo_velocity_limit_deg_s")),
        wheel_reference_velocity_rad_s=float(values.get("wheel_reference_velocity_rad_s", defaults.wheel_reference_velocity_rad_s)),
        wheel_velocity_limit_rad_s=float(values.get("wheel_velocity_limit_rad_s", defaults.wheel_velocity_limit_rad_s)),
        wheel_radius_m=_optional_float(values.get("wheel_radius_m")),
        wheel_transmission_ratio=_optional_float(values.get("wheel_transmission_ratio")),
        verified=bool(values.get("verified", defaults.verified)),
        verification_method=str(values.get("verification_method", defaults.verification_method)),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "null", "none", "unverified"}:
        return None
    return float(value)


def _yaml_scalar(value: str) -> Any:
    text = value.strip().strip('"').strip("'")
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return float(text) if any(ch in text.lower() for ch in (".", "e")) else int(text)
    except ValueError:
        return text
