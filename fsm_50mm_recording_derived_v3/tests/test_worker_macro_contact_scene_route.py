from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from fsm_50mm_recording_derived_v3.worker_macro_fsm_session import (
    WorkerMacroFSMSession,
    configure_scene_for_macro_fsm,
)
from sim_worker_process import run_worker


def _request(*, enabled: bool = True):
    return SimpleNamespace(
        request_id="macro-contact-route",
        source_version="v003_20260805_224517_157723_manual",
        profile_id="test-profiles",
        bundle_sha256="a" * 64,
        filtered_contact_bank_enabled=enabled,
    )


def _default_scene_config():
    return SimpleNamespace(
        telemetry_contact_sensors_enabled=False,
        contact_sensor_factory=None,
        spawn_z=0.04,
        obstacle_height_m=0.05,
        obstacle_x=1.55,
        ground_static_friction=1.25,
        ground_dynamic_friction=1.05,
        obstacle_static_friction=1.20,
        obstacle_dynamic_friction=1.00,
        servo_stiffness=600.0,
        servo_damping=60.0,
        wheel_damping=20.0,
        physics_dt=1.0 / 120.0,
        render_interval=8,
    )


def _combined_sensor(*, threshold_n: float = 1.0):
    wheel_sensors = {leg: object() for leg in ("FL", "FR", "RL", "RR")}
    nonwheel_path = "/World/WLRRobot/base_link"
    obstacle_path = "/World/Obstacle"
    nonwheel_child = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path=nonwheel_path,
            filter_prim_paths_expr=[obstacle_path],
        )
    )
    wheel = SimpleNamespace(
        is_filtered_wheel_contact_bank=True,
        force_threshold_n=threshold_n,
        sensors=wheel_sensors,
    )
    nonwheel = SimpleNamespace(
        is_nonwheel_obstacle_contact_bank=True,
        force_threshold_n=threshold_n,
        sensors={nonwheel_path: nonwheel_child},
        specs=(SimpleNamespace(body_name="base_link", prim_path=nonwheel_path),),
        obstacle_prim_path=obstacle_path,
    )
    return SimpleNamespace(
        is_filtered_wheel_contact_bank=True,
        is_nonwheel_obstacle_contact_bank=True,
        wheel_bank=wheel,
        nonwheel_bank=nonwheel,
    )


def test_none_route_preserves_even_an_already_instrumented_ordinary_scene():
    sentinel_factory = object()
    config = SimpleNamespace(
        telemetry_contact_sensors_enabled=True,
        contact_sensor_factory=sentinel_factory,
        ordinary_recording_identity="unchanged",
    )
    before = dict(vars(config))
    assert configure_scene_for_macro_fsm(config, None) is config
    assert vars(config) == before


def test_exact_route_installs_combined_one_newton_bank_only():
    config = _default_scene_config()
    immutable_physics = {
        key: value
        for key, value in vars(config).items()
        if key
        not in {"telemetry_contact_sensors_enabled", "contact_sensor_factory"}
    }

    assert configure_scene_for_macro_fsm(config, _request()) is config
    assert config.telemetry_contact_sensors_enabled is True
    assert callable(config.contact_sensor_factory)
    assert {
        key: vars(config)[key]
        for key in immutable_physics
    } == immutable_physics

    combined_factory = config.contact_sensor_factory
    assert combined_factory.keywords["force_threshold_n"] == 1.0
    wheel_factory = combined_factory.keywords["wheel_factory"]
    assert wheel_factory.keywords["force_threshold_n"] == 1.0


@pytest.mark.parametrize(
    "enabled,factory",
    ((True, None), (False, object())),
)
def test_exact_route_rejects_nondefault_incoming_scene(enabled, factory):
    config = _default_scene_config()
    config.telemetry_contact_sensors_enabled = enabled
    config.contact_sensor_factory = factory
    with pytest.raises(ValueError, match="incoming production default"):
        configure_scene_for_macro_fsm(config, _request())


def test_request_must_explicitly_require_the_contact_bank():
    with pytest.raises(ValueError, match="does not require"):
        configure_scene_for_macro_fsm(_default_scene_config(), _request(enabled=False))


def test_session_claim_turns_true_only_after_live_combined_bank_binding():
    request = _request()
    session = WorkerMacroFSMSession(request, worker_session_id="worker-session")
    assert session.status_dict()["filtered_contact_bank_enabled"] is False

    config = _default_scene_config()
    configure_scene_for_macro_fsm(config, request)
    scene = SimpleNamespace(
        config=config,
        contact_sensor=_combined_sensor(),
        contact_sensor_error="",
    )
    session.bind_filtered_contact_bank_scene(scene)
    assert session.status_dict()["filtered_contact_bank_enabled"] is True
    assert session.filtered_contact_scene_handle is scene


@pytest.mark.parametrize(
    "scene",
    (
        SimpleNamespace(
            config=_default_scene_config(),
            contact_sensor=_combined_sensor(),
            contact_sensor_error="",
        ),
        SimpleNamespace(
            config=None,
            contact_sensor=_combined_sensor(),
            contact_sensor_error="",
        ),
    ),
)
def test_session_rejects_a_scene_not_selected_by_the_macro_helper(scene):
    session = WorkerMacroFSMSession(_request(), worker_session_id="worker-session")
    with pytest.raises(RuntimeError, match="not selected"):
        session.bind_filtered_contact_bank_scene(scene)
    assert session.status_dict()["filtered_contact_bank_enabled"] is False


@pytest.mark.parametrize(
    "sensor,error,match",
    (
        (None, "factory failure", "unavailable"),
        (_combined_sensor(threshold_n=2.0), "", "not exactly 1.0 N"),
        (
            SimpleNamespace(
                is_filtered_wheel_contact_bank=True,
                is_nonwheel_obstacle_contact_bank=False,
            ),
            "",
            "not the combined",
        ),
    ),
)
def test_session_rejects_missing_wrong_or_wrong_threshold_live_bank(
    sensor, error, match
):
    request = _request()
    config = _default_scene_config()
    configure_scene_for_macro_fsm(config, request)
    scene = SimpleNamespace(
        config=config,
        contact_sensor=sensor,
        contact_sensor_error=error,
    )
    session = WorkerMacroFSMSession(request, worker_session_id="worker-session")
    with pytest.raises(RuntimeError, match=match):
        session.bind_filtered_contact_bank_scene(scene)
    assert session.status_dict()["filtered_contact_bank_enabled"] is False


def test_worker_configures_macro_or_residual_base_before_create_then_binds():
    source = inspect.getsource(run_worker)
    selection = source.index("macro_scene_request = (")
    configure = source.index(
        "configure_scene_for_macro_fsm(scene_config, macro_scene_request)"
    )
    create = source.index("scene_handle = create_scene(")
    bind = source.index("macro_session.bind_filtered_contact_bank_scene(scene_handle)")
    first_status = source.index("publish_status(ready=False, starting=True)", create)
    assert selection < configure < create < bind < first_status
    assert "residual_request.base_request" in source[selection:configure]
