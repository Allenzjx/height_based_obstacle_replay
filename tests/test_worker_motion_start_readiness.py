from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker_process  # noqa: E402
from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES  # noqa: E402
from sim_worker_runtime import capture_worker_motion_start_readiness  # noqa: E402


def _zero_wheels() -> dict[str, float]:
    return {name: 0.0 for name in WHEEL_JOINT_NAMES}


def _request_identity() -> dict[str, object]:
    state = {
        "servos": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "wheels": _zero_wheels(),
    }
    state_sha = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "request_id": "request-1",
        "plan_id": "plan-1",
        "plan_sha256": "a" * 64,
        "validated_plan_sha256": "a" * 64,
        "event_count": 3,
        "validated_event_count": 3,
        "segment_count": 2,
        "validated_segment_count": 2,
        "integrity_ok": True,
        "source_initial_command_state": state,
        "source_initial_command_state_sha256": state_sha,
    }


class _Adapter:
    def __init__(
        self,
        *,
        strict_rest_stable: bool = True,
        ground: dict[str, object] | None = None,
        base: dict[str, object] | None = None,
    ) -> None:
        self.sim_steps = 42
        self.runtime_instance_id = "adapter-instance-1"
        self.root_state_write_count = 0
        self.grounded_reference_valid = bool(strict_rest_stable)
        self.grounded_reference_stable = bool(strict_rest_stable)
        self.last_ground_settle_result = {"stable": bool(strict_rest_stable)}
        self.validation_apply_correction: list[bool] = []
        self._ground = ground or {
            "checked": True,
            "classification": "OK",
            "ground_state": "PASS",
            "physical_ground_safe": True,
            "sim_steps": self.sim_steps,
            "correction_applied": False,
            "wheel_command_values": _zero_wheels(),
        }
        self._base = base or {
            "adapter_runtime_instance_id": self.runtime_instance_id,
            "root_state_write_count": self.root_state_write_count,
            "root_state_write_events": [],
            "sim_step": self.sim_steps,
            "command_state": {
                "servos": {name: 0.0 for name in SERVO_JOINT_NAMES},
                "wheels": _zero_wheels(),
            },
            "joint_state_evidence_valid": True,
            "wheel_target_evidence_valid": True,
            "joint_velocity_by_name": {
                **_zero_wheels(),
                "front_left_hip": 0.0,
            },
            "wheel_target_velocity_by_name": _zero_wheels(),
            "wheel_target_readback_velocity_by_name": _zero_wheels(),
        }
        self.wheel_command_status = {
            "generation": 0,
            "state": "physically_stopped",
            "zero_target_applied": True,
            "physically_stopped": True,
            "stop_tolerance_rad_s": 0.20,
            "applied_target_rad_s": _zero_wheels(),
            "measured_velocity_rad_s": _zero_wheels(),
        }

    def validate_robot_ground_contact(self, *, apply_correction: bool) -> dict[str, object]:
        self.validation_apply_correction.append(apply_correction)
        return copy.deepcopy(self._ground)

    def capture_motion_start_base_evidence(self) -> dict[str, object]:
        return copy.deepcopy(self._base)


def _capture(adapter: object, identity: dict[str, object] | None = None) -> dict[str, object]:
    return capture_worker_motion_start_readiness(
        adapter,
        runtime_ready=True,
        current_sim_step=42,
        worker_session_id="worker-session-1",
        request_identity=identity or _request_identity(),
    )


class WorkerMotionStartReadinessTest(unittest.TestCase):
    def test_ordinary_safe_worker_start_passes_with_read_only_live_evidence(self) -> None:
        adapter = _Adapter(strict_rest_stable=True)

        result = _capture(adapter)

        self.assertTrue(result["motion_start_ready"])
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(adapter.validation_apply_correction, [False])
        self.assertFalse(result["state_writes_performed"])
        self.assertEqual(result["identity"]["worker_session_id"], "worker-session-1")
        self.assertEqual(result["adapter_runtime_instance_id"], "adapter-instance-1")
        self.assertEqual(result["root_state_write_count_before"], 0)
        self.assertEqual(result["root_state_write_count_after"], 0)
        self.assertEqual(result["current_sim_step"], 42)

    def test_strict_rest_failure_does_not_block_current_safe_ground(self) -> None:
        adapter = _Adapter(strict_rest_stable=False)

        result = _capture(adapter)

        self.assertTrue(result["motion_start_ready"])
        self.assertFalse(result["strict_rest_context"]["grounded_reference_valid"])
        self.assertFalse(result["strict_rest_context"]["grounded_reference_stable"])
        self.assertFalse(result["strict_rest_context"]["required_for_motion_start"])
        self.assertTrue(result["ground_motion_ready"])

    def test_unsafe_live_ground_is_rejected(self) -> None:
        adapter = _Adapter(
            ground={
                "checked": True,
                "classification": "COLLISION_PENETRATION",
                "ground_state": "FAIL",
                "physical_ground_safe": False,
                "sim_steps": 42,
                "correction_applied": False,
                "wheel_command_values": _zero_wheels(),
            }
        )

        result = _capture(adapter)

        self.assertFalse(result["motion_start_ready"])
        self.assertIn("ground diagnostics failed", result["rejection_reason"])

    def test_missing_live_ground_evidence_is_rejected(self) -> None:
        adapter = _Adapter(ground={"sim_steps": 42})

        result = _capture(adapter)

        self.assertFalse(result["motion_start_ready"])
        self.assertIn("checked=true", result["rejection_reason"])
        self.assertIn("unverified", result["rejection_reason"])

    def test_stale_ground_or_base_step_is_rejected(self) -> None:
        adapter = _Adapter()
        adapter._ground["sim_steps"] = 41
        adapter._base["sim_step"] = 40

        result = _capture(adapter)

        self.assertFalse(result["motion_start_ready"])
        self.assertIn("live ground diagnostics are stale", result["rejection_reason"])
        self.assertIn("motion-start base evidence is stale", result["rejection_reason"])

    def test_nonzero_wheel_target_is_rejected(self) -> None:
        adapter = _Adapter()
        wheel = WHEEL_JOINT_NAMES[0]
        adapter._base["wheel_target_velocity_by_name"][wheel] = 0.1

        result = _capture(adapter)

        self.assertFalse(result["motion_start_ready"])
        self.assertIn(f"logical_targets is nonzero for {wheel}", result["rejection_reason"])

    def test_missing_adapter_instance_or_root_write_evidence_is_rejected(self) -> None:
        adapter = _Adapter()
        adapter.runtime_instance_id = ""
        adapter.root_state_write_count = None

        result = _capture(adapter)

        self.assertFalse(result["motion_start_ready"])
        self.assertIn("adapter runtime instance identity is missing", result["rejection_reason"])
        self.assertIn("root-state write count evidence is missing", result["rejection_reason"])

    def test_incomplete_identity_is_rejected(self) -> None:
        identity = _request_identity()
        identity["request_id"] = ""

        result = _capture(_Adapter(), identity)

        self.assertFalse(result["motion_start_ready"])
        self.assertIn("missing request_id", result["rejection_reason"])

    def test_worker_checks_gate_before_scheduler_and_preserves_ack_evidence(self) -> None:
        source = inspect.getsource(sim_worker_process.run_worker)

        capture_at = source.index("capture_worker_motion_start_readiness(")
        scheduler_at = source.index("playback_service.start_plan(", capture_at)
        boundary_at = source.index(
            "playback_service.apply_playback_start_boundary(", scheduler_at
        )
        ack_evidence_at = source.index("motion_start_readiness=copy.deepcopy(", boundary_at)
        self.assertLess(capture_at, scheduler_at)
        self.assertLess(scheduler_at, boundary_at)
        self.assertLess(boundary_at, ack_evidence_at)


if __name__ == "__main__":
    unittest.main()
