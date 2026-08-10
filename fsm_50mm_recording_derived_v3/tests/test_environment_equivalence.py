from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import sim_obstacle_scene as scene_module
from fsm_50mm_recording_derived_v3.environment_equivalence import (
    ALLOWED_INSTRUMENTATION_CONFIG_FIELDS,
    build_static_environment_fingerprint,
    compare_instrumentation_configs,
    compare_trajectory_equivalence,
    normalize_physical_scene_config,
    sha256_file,
    write_environment_equivalence_report,
)
from playback import plan_from_steps
from sequence_model import empty_command_state, make_event, make_step
from sim_obstacle_scene import SimSceneConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeSimulationContext:
    def __init__(self, config):
        self.config = config
        self.reset_count = 0
        self.camera = None

    def set_camera_view(self, *, eye, target):
        self.camera = (list(eye), list(target))

    def reset(self):
        self.reset_count += 1

    def get_physics_dt(self):
        return float(self.config.dt)


class _FakeRobot:
    def __init__(self, config):
        self.config = config
        self.update_calls = []

    def update(self, dt):
        self.update_calls.append(float(dt))


def _install_fake_scene_dependencies(monkeypatch):
    fake_sim_utils = SimpleNamespace(
        SimulationCfg=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(scene_module, "_validate_robot_usd", lambda path: None)
    monkeypatch.setattr(
        scene_module,
        "_isaac_imports",
        lambda: {
            "sim_utils": fake_sim_utils,
            "SimulationContext": _FakeSimulationContext,
            "Articulation": _FakeRobot,
        },
    )
    monkeypatch.setattr(scene_module, "build_robot_cfg", lambda config, imports: {"robot": True})
    monkeypatch.setattr(
        scene_module,
        "create_ground_plane",
        lambda sim_utils, *, ground_z_m: {
            "configured_ground_z_m": float(ground_z_m),
            "actual_ground_z_m": float(ground_z_m),
            "ground_resolution_ok": True,
        },
    )
    monkeypatch.setattr(scene_module, "add_lighting", lambda sim_utils: None)
    monkeypatch.setattr(scene_module, "resolve_obstacle_dimensions", lambda config, sim_utils: None)
    monkeypatch.setattr(scene_module, "create_obstacle", lambda config, imports: [1.55, 0.0, 0.025])
    monkeypatch.setattr(scene_module, "detect_articulation_roots", lambda path, imports: [path])
    monkeypatch.setattr(scene_module, "save_scene", lambda path, sim_utils: None)


def test_scene_contact_factory_none_uses_default_and_custom_is_honored(monkeypatch):
    """Exercise the production create_scene branch with no Isaac imports."""

    _install_fake_scene_dependencies(monkeypatch)
    calls = {"default": 0, "custom": 0}
    default_sensor = object()
    custom_sensor = object()

    def default_factory():
        calls["default"] += 1
        return default_sensor, ""

    def custom_factory():
        calls["custom"] += 1
        return custom_sensor, "custom-readback-warning"

    monkeypatch.setattr(scene_module, "create_robot_contact_sensor", default_factory)

    default_config = SimSceneConfig(
        obstacle_height_m=0.05,
        telemetry_contact_sensors_enabled=True,
        contact_sensor_factory=None,
        save_scene=False,
    )
    default_physical_before = normalize_physical_scene_config(default_config)
    default_handle = scene_module.create_scene(default_config)
    assert default_handle.contact_sensor is default_sensor
    assert default_handle.contact_sensor_error == ""
    assert calls == {"default": 1, "custom": 0}
    assert normalize_physical_scene_config(default_config) == default_physical_before

    custom_config = replace(default_config, contact_sensor_factory=custom_factory)
    custom_physical_before = normalize_physical_scene_config(custom_config)
    custom_handle = scene_module.create_scene(custom_config)
    assert custom_handle.contact_sensor is custom_sensor
    assert custom_handle.contact_sensor_error == "custom-readback-warning"
    assert calls == {"default": 1, "custom": 1}
    assert custom_config.contact_sensor_factory is custom_factory
    assert normalize_physical_scene_config(custom_config) == custom_physical_before
    assert normalize_physical_scene_config(default_config) == normalize_physical_scene_config(custom_config)


def test_instrumentation_normalization_allows_only_declared_fields_and_readback():
    def custom_factory():
        return object(), ""

    baseline = SimSceneConfig(obstacle_height_m=0.05, save_scene=False)
    instrumented = replace(
        baseline,
        telemetry_contact_sensors_enabled=True,
        contact_sensor_factory=custom_factory,
    )
    comparison = compare_instrumentation_configs(
        baseline,
        instrumented,
        baseline_sensor_readback={"forces_n": [0.0, 0.0]},
        instrumented_sensor_readback={"forces_n": [1.0, 2.0], "contact": "GROUND"},
    )
    assert comparison["ok"] is True
    assert set(comparison["allowed_config_fields"]) == ALLOWED_INSTRUMENTATION_CONFIG_FIELDS
    assert comparison["physical_differences"] == []
    assert comparison["allowed_instrumentation_differences"]
    assert comparison["allowed_sensor_readback_differences"]

    physically_changed = replace(instrumented, servo_stiffness=instrumented.servo_stiffness + 1.0)
    rejected = compare_instrumentation_configs(baseline, physically_changed)
    assert rejected["ok"] is False
    assert "servo_stiffness" in rejected["physical_differences"]

    # A sensor-looking name inside physical config is not a wildcard escape.
    rejected_mapping = compare_instrumentation_configs(
        {"physics_dt": 1.0 / 120.0, "sensor_gain": 1.0},
        {"physics_dt": 1.0 / 120.0, "sensor_gain": 2.0},
    )
    assert rejected_mapping["ok"] is False
    assert rejected_mapping["physical_differences"] == ["sensor_gain"]


def test_static_fingerprint_hashes_formal_sources_and_selects_150_and_8(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"environment-equivalence")
    assert sha256_file(sample) == hashlib.sha256(b"environment-equivalence").hexdigest()

    fingerprint = build_static_environment_fingerprint(
        project_root=PROJECT_ROOT,
        source_commit="unit-test-commit",
    )
    scene_path = PROJECT_ROOT / "sim_obstacle_scene.py"
    assert fingerprint["source_commit"] == "unit-test-commit"
    assert fingerprint["source_files"]["sim_obstacle_scene.py"]["sha256"] == hashlib.sha256(
        scene_path.read_bytes()
    ).hexdigest()
    assert fingerprint["robot_usd"]["sha256"] == sha256_file(fingerprint["robot_usd"]["path"])
    assert fingerprint["prims"] == {
        "robot": "/World/WLRRobot",
        "ground": "/World/defaultGroundPlane",
        "obstacle": "/World/Obstacle",
    }
    assert fingerprint["initial_state"]["servo_command_deg"]
    assert set(fingerprint["initial_state"]["wheel_target_rad_s"]) == {
        "front_left_ankle",
        "front_right_ankle",
        "rear_left_ankle",
        "rear_right_ankle",
    }
    assert fingerprint["obstacle"]["height_m"] == pytest.approx(0.05)
    assert fingerprint["obstacle"]["front_face_x_m"] == pytest.approx(0.5213121737735307)
    assert fingerprint["obstacle"]["width_m"] == pytest.approx(2.0)
    assert fingerprint["physics"]["physics_dt_s"] == pytest.approx(1.0 / 120.0)
    assert fingerprint["physics"]["render_interval_physics_steps"] == 8
    assert fingerprint["physics"]["obstacle_contact_offset_m"] == pytest.approx(0.005)
    assert fingerprint["physics"]["obstacle_rest_offset_m"] == 0.0
    assert fingerprint["motion_reference"]["profile_id"] == "real-robot-ui-controller-20260626-v1"
    assert fingerprint["actuators"]["servo"]["reference_velocity_deg_s"] == 150.0
    assert fingerprint["actuators"]["wheel"]["radius_m"] == pytest.approx(0.04998999834060672)
    assert fingerprint["runtime_versions"]["isaac_sim"] == "unknown_pending_runtime_readback"
    assert fingerprint["runtime_readback_required"]

    legacy = fingerprint["legacy_metadata_differences"]
    assert legacy["servo_reference_velocity_deg_s"] == {
        "legacy_baseline_metadata": 30.0,
        "selected_runtime": 150.0,
        "classification": "metadata_only_not_runtime_selection",
    }
    assert legacy["render_interval_physics_steps"] == {
        "legacy_baseline_metadata": 2,
        "selected_runtime": 8,
        "classification": "metadata_only_not_runtime_selection",
    }
    assert fingerprint["fast_replay"]["profile_normalized"] == "motion_only"
    assert fingerprint["fast_replay"]["segment_duration_rule"] == (
        "max(servo_duration_s, wheel_duration_s, explicit_hold_s)"
    )
    json.dumps(fingerprint, sort_keys=True)


def test_production_fast_planner_uses_150_degree_reference_and_max_duration_rule():
    before = empty_command_state()
    step = make_step(
        index=1,
        step_type="recorded",
        duration=1.0,
        events=[
            make_event(0.0, "servo front_left_hip 30", command_state_before=before),
            make_event(0.0, "wheel all 0.3", command_state_before=before),
            make_event(0.4, "wheel stop", command_state_before=before),
        ],
        command_state_before=before,
        command_state_after=before,
    )
    plan = plan_from_steps([step], profile="fast")
    servo_segments = [segment for segment in plan.segments if segment.servo_targets]
    assert servo_segments
    segment = servo_segments[0]
    segment_events = plan.events[
        segment.event_start_index : segment.event_start_index + segment.event_count
    ]
    servo_events = [event for event in segment_events if event.servo_targets]
    assert servo_events
    assert servo_events[0].servo_base_velocity_deg_s == 150.0
    assert segment.servo_duration_s == pytest.approx(30.0 / 150.0)
    assert segment.planned_end_s - segment.planned_start_s == pytest.approx(
        max(segment.servo_duration_s, segment.wheel_active_duration_s, segment.explicit_hold_s)
    )


def _trajectory(offset: float = 0.0, *, contact_class: str = "GROUND", force_offset: float = 0.0):
    return {
        "root_trajectory": [[0.0 + offset, 0.0, 0.10], [1.0 + offset, 0.0, 0.10]],
        "joint_trajectory": {
            "front_left_hip": [0.0 + offset, 0.2 + offset],
            "front_left_knee": [0.0, -0.1 + offset],
        },
        "wheel_rotation": {"front_left_ankle": [0.0, 1.0 + offset]},
        "wheel_travel": {"front_left_ankle": [0.0, 0.05 + offset]},
        "final_pose": [1.0 + offset, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
        "obstacle_geometry": {
            "bounds_min": [0.5213121737735307 + offset, -1.0, 0.0],
            "bounds_max": [2.5786877308590375 + offset, 1.0, 0.05],
        },
        "contact_class": [contact_class, contact_class],
        "contact_force": [[10.0 + force_offset, 0.0], [12.0 + force_offset, 0.0]],
    }


def test_aa_ab_comparator_uses_self_error_and_fails_closed_on_large_b_drift():
    a1 = _trajectory(0.0, force_offset=0.0)
    a2 = _trajectory(0.001, force_offset=0.1)
    within_b = _trajectory(0.002, force_offset=0.2)
    within = compare_trajectory_equivalence(a1, a2, within_b)
    assert within["ok"] is True
    assert within["failed_metrics"] == []
    assert within["metrics"]["root_trajectory"]["baseline_self_error"] == pytest.approx(0.001)
    assert within["metrics"]["root_trajectory"]["tolerance"] == pytest.approx(0.003)
    assert within["metrics"]["root_trajectory"]["instrumented_error"] == pytest.approx(0.002)

    bad_b = copy.deepcopy(within_b)
    bad_b["root_trajectory"][1][0] += 0.05
    bad_b["contact_class"][1] = "TOP"
    bad_b["contact_force"][1][0] += 1.0
    rejected = compare_trajectory_equivalence(a1, a2, bad_b)
    assert rejected["ok"] is False
    assert {"root_trajectory", "contact_class", "contact_force"} <= set(
        rejected["failed_metrics"]
    )
    assert rejected["metrics"]["contact_class"]["tolerance"] == 0.0
    assert rejected["metrics"]["contact_class"]["instrumented_error"] > 0.0

    missing = copy.deepcopy(within_b)
    del missing["wheel_travel"]
    missing_result = compare_trajectory_equivalence(a1, a2, missing)
    assert missing_result["ok"] is False
    assert missing_result["metrics"]["wheel_travel"]["reason"] == "missing metric in B"


def test_json_report_requires_runtime_readback_and_both_comparisons(tmp_path):
    destination = tmp_path / "equivalence.json"
    fingerprint = {"schema_version": "test", "status": "STATIC_LOCK_RUNTIME_READBACK_PENDING"}
    instrumentation = {"ok": True}
    trajectory = {"ok": True}
    written = write_environment_equivalence_report(
        destination,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
        runtime_readback={
            "ok": True,
            "isaac_sim": "test-version",
            "readback_complete": True,
        },
        extra={"artifact_conversion": {"ok": True}},
    )
    assert written == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["environment_equivalent"] is True
    assert not destination.with_name(f".{destination.name}.tmp").exists()

    pending_path = tmp_path / "pending.json"
    write_environment_equivalence_report(
        pending_path,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
    )
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    assert pending["status"] == "PENDING_RUNTIME_A_B"
    assert pending["environment_equivalent"] is False

    empty_readback_path = tmp_path / "empty-readback.json"
    write_environment_equivalence_report(
        empty_readback_path,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
        runtime_readback={},
    )
    empty_readback = json.loads(empty_readback_path.read_text(encoding="utf-8"))
    assert empty_readback["status"] == "PENDING_RUNTIME_A_B"
    assert empty_readback["environment_equivalent"] is False

    complete_without_ok_path = tmp_path / "complete-without-ok.json"
    write_environment_equivalence_report(
        complete_without_ok_path,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
        runtime_readback={"readback_complete": True},
    )
    complete_without_ok = json.loads(
        complete_without_ok_path.read_text(encoding="utf-8")
    )
    assert complete_without_ok["status"] == "FAIL"
    assert complete_without_ok["environment_equivalent"] is False

    incomplete_but_ok_path = tmp_path / "incomplete-but-ok.json"
    write_environment_equivalence_report(
        incomplete_but_ok_path,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
        runtime_readback={"ok": True, "readback_complete": False},
    )
    incomplete_but_ok = json.loads(incomplete_but_ok_path.read_text(encoding="utf-8"))
    assert incomplete_but_ok["status"] == "PENDING_RUNTIME_A_B"
    assert incomplete_but_ok["environment_equivalent"] is False

    missing_conversion_path = tmp_path / "missing-conversion.json"
    write_environment_equivalence_report(
        missing_conversion_path,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
        runtime_readback={"ok": True, "readback_complete": True},
    )
    missing_conversion = json.loads(
        missing_conversion_path.read_text(encoding="utf-8")
    )
    assert missing_conversion["status"] == "PENDING_RUNTIME_A_B"
    assert missing_conversion["environment_equivalent"] is False

    failed_conversion_path = tmp_path / "failed-conversion.json"
    write_environment_equivalence_report(
        failed_conversion_path,
        fingerprint=fingerprint,
        instrumentation_comparison=instrumentation,
        trajectory_comparison=trajectory,
        runtime_readback={"ok": True, "readback_complete": True},
        extra={"artifact_conversion": {"ok": False}},
    )
    failed_conversion = json.loads(
        failed_conversion_path.read_text(encoding="utf-8")
    )
    assert failed_conversion["status"] == "FAIL"
    assert failed_conversion["environment_equivalent"] is False

    failed_path = tmp_path / "failed.json"
    write_environment_equivalence_report(
        failed_path,
        fingerprint=fingerprint,
        instrumentation_comparison={"ok": False},
        trajectory_comparison=trajectory,
        runtime_readback={"readback_complete": True},
    )
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed["status"] == "FAIL"
    assert failed["environment_equivalent"] is False
