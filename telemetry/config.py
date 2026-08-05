"""Configuration helpers for runtime telemetry.

This module is intentionally importable without Isaac Sim.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "height_replay_telemetry.v1"
MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = MODULE_ROOT / "config" / "telemetry.yaml"


@dataclass
class TelemetryConfig:
    enabled: bool = True
    sample_hz: float = 50.0
    flush_interval_s: float = 5.0
    output_root: str = "runs"
    env_id: int = 0
    save_npz: bool = True
    save_csv: bool = True
    save_contacts: bool = True
    save_events: bool = True
    max_full_samples: int = 200000
    enable_contact_sensor: bool = False
    live_queue_max_frames: int = 5


@dataclass
class VisualizationConfig:
    live_enabled: bool = True
    update_hz: float = 20.0
    show_com: bool = True
    show_com_trail: bool = True
    com_trail_duration_s: float = 5.0
    show_com_projection: bool = True
    show_support_polygon: bool = True
    show_equilibrium_region: bool = True
    show_contact_points: bool = True
    show_contact_forces: bool = True
    show_base_trajectory: bool = True
    show_hud: bool = True
    force_arrow_scale: float = 0.002
    marker_limit: int = 256
    visualized_env_id: int = 0


@dataclass
class StabilityConfig:
    contact_force_threshold_n: float = 2.0
    safe_margin_m: float = 0.05
    warning_margin_m: float = 0.02
    equilibrium_enabled: bool = True
    equilibrium_hz: float = 5.0
    friction_pyramid_sides: int = 8
    direction_samples: int = 48
    solver_tolerance: float = 1.0e-7
    friction_coefficient_default: float = 1.05


@dataclass
class FilterConfig:
    acceleration_filter_enabled: bool = True
    acceleration_cutoff_hz: float = 10.0
    contact_force_filter_enabled: bool = True
    contact_force_cutoff_hz: float = 20.0


@dataclass
class WarningConfig:
    torque_utilization: float = 0.9
    friction_utilization: float = 0.8
    wheel_slip_ratio: float = 0.2
    roll_rad: float = 0.35
    pitch_rad: float = 0.35
    replay_position_rmse_rad: float = 0.1
    impact_force_n: float = 200.0


@dataclass
class RuntimeTelemetryConfig:
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    warnings: WarningConfig = field(default_factory=WarningConfig)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_telemetry_args(parser: argparse.ArgumentParser) -> None:
    """Add telemetry CLI switches without changing existing parser behavior."""

    parser.set_defaults(
        telemetry_enabled=None,
        live_viz_enabled=None,
        equilibrium_region_enabled=None,
        telemetry_rate=None,
        visualized_env_id=None,
        output_dir=None,
        telemetry_config="",
    )
    parser.add_argument("--telemetry", dest="telemetry_enabled", action="store_true", help="Enable telemetry capture.")
    parser.add_argument("--no-telemetry", dest="telemetry_enabled", action="store_false", help="Disable telemetry capture.")
    parser.add_argument("--live-viz", dest="live_viz_enabled", action="store_true", help="Enable viewport telemetry overlay.")
    parser.add_argument("--no-live-viz", dest="live_viz_enabled", action="store_false", help="Disable viewport telemetry overlay.")
    parser.add_argument("--equilibrium-region", dest="equilibrium_region_enabled", action="store_true", help="Enable friction-aware equilibrium region solving.")
    parser.add_argument("--no-equilibrium-region", dest="equilibrium_region_enabled", action="store_false", help="Disable friction-aware equilibrium region solving.")
    parser.add_argument("--telemetry-rate", dest="telemetry_rate", type=float, default=None, help="Telemetry sample rate in Hz.")
    parser.add_argument("--visualized-env-id", dest="visualized_env_id", type=int, default=None, help="Environment id for live overlay.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None, help="Telemetry output root or run directory.")
    parser.add_argument("--telemetry-config", dest="telemetry_config", type=str, default="", help="Telemetry YAML config path.")


def load_telemetry_config(args: Any | None = None, config_path: str | Path | None = None) -> RuntimeTelemetryConfig:
    cfg = RuntimeTelemetryConfig()
    selected = _select_config_path(args, config_path)
    if selected is not None and selected.exists():
        _deep_update_dataclass(cfg, _read_yaml(selected))
    _apply_arg_overrides(cfg, args)
    _normalize(cfg)
    return cfg


def config_snapshot_dict(config: RuntimeTelemetryConfig) -> dict[str, Any]:
    return copy.deepcopy(config.to_dict())


def telemetry_enabled_from_args(args: Any) -> bool:
    return bool(load_telemetry_config(args).telemetry.enabled)


def _select_config_path(args: Any | None, config_path: str | Path | None) -> Path | None:
    explicit = str(config_path or "").strip()
    if not explicit and args is not None:
        explicit = str(getattr(args, "telemetry_config", "") or "").strip()
    if explicit:
        return Path(explicit)
    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return _read_simple_yaml(path)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Tiny nested YAML fallback for the flat config file used here."""

    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            current = root.setdefault(key, {})
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(text: str) -> Any:
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"').strip("'")


def _deep_update_dataclass(obj: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _deep_update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _apply_arg_overrides(cfg: RuntimeTelemetryConfig, args: Any | None) -> None:
    if args is None:
        return
    if getattr(args, "telemetry_enabled", None) is not None:
        cfg.telemetry.enabled = bool(getattr(args, "telemetry_enabled"))
    if getattr(args, "live_viz_enabled", None) is not None:
        cfg.visualization.live_enabled = bool(getattr(args, "live_viz_enabled"))
    if getattr(args, "equilibrium_region_enabled", None) is not None:
        cfg.stability.equilibrium_enabled = bool(getattr(args, "equilibrium_region_enabled"))
    if getattr(args, "telemetry_rate", None) is not None:
        cfg.telemetry.sample_hz = float(getattr(args, "telemetry_rate"))
    if getattr(args, "visualized_env_id", None) is not None:
        cfg.visualization.visualized_env_id = int(getattr(args, "visualized_env_id"))
        cfg.telemetry.env_id = int(getattr(args, "visualized_env_id"))
    if getattr(args, "output_dir", None):
        cfg.telemetry.output_root = str(getattr(args, "output_dir"))


def _normalize(cfg: RuntimeTelemetryConfig) -> None:
    cfg.telemetry.sample_hz = max(0.1, float(cfg.telemetry.sample_hz))
    cfg.telemetry.flush_interval_s = max(0.1, float(cfg.telemetry.flush_interval_s))
    cfg.telemetry.env_id = max(0, int(cfg.telemetry.env_id))
    cfg.telemetry.max_full_samples = max(0, int(cfg.telemetry.max_full_samples))
    cfg.visualization.update_hz = max(0.1, float(cfg.visualization.update_hz))
    cfg.visualization.com_trail_duration_s = max(0.0, float(cfg.visualization.com_trail_duration_s))
    cfg.visualization.marker_limit = max(1, int(cfg.visualization.marker_limit))
    cfg.visualization.visualized_env_id = max(0, int(cfg.visualization.visualized_env_id))
    cfg.stability.contact_force_threshold_n = max(0.0, float(cfg.stability.contact_force_threshold_n))
    cfg.stability.safe_margin_m = max(0.0, float(cfg.stability.safe_margin_m))
    cfg.stability.warning_margin_m = max(0.0, float(cfg.stability.warning_margin_m))
    cfg.stability.equilibrium_hz = max(0.01, float(cfg.stability.equilibrium_hz))
    cfg.stability.friction_pyramid_sides = max(4, int(cfg.stability.friction_pyramid_sides))
    cfg.stability.direction_samples = max(8, int(cfg.stability.direction_samples))
    cfg.stability.solver_tolerance = max(1.0e-12, float(cfg.stability.solver_tolerance))
    cfg.stability.friction_coefficient_default = max(0.0, float(cfg.stability.friction_coefficient_default))
    cfg.filters.acceleration_cutoff_hz = max(0.01, float(cfg.filters.acceleration_cutoff_hz))
    cfg.filters.contact_force_cutoff_hz = max(0.01, float(cfg.filters.contact_force_cutoff_hz))
