from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from completion_aware_segment import (
    CompletionAwareSegmentExecutor,
    SegmentCompletionSpec,
    SegmentDecisionKind,
    SegmentFeedback,
)
from playback import (
    PlaybackEvent,
    PlaybackPlan,
    PlaybackSegment,
    SimTimePlaybackService,
    plan_fingerprint,
    plan_from_jsonl,
)


def _spec(**updates: Any) -> SegmentCompletionSpec:
    values: dict[str, Any] = {
        "segment_index": 7,
        "source_step": 3,
        "source_step_id": "step-3",
        "servo_targets_deg": {"front_left_hip": 10.0},
        "servo_duration_s": 0.1,
        "servo_tolerance_deg": 1.0,
        "recorded_servo_residual_deg": {"front_left_hip": 0.0},
        "legacy_missing_endpoint": False,
        "wheel_active_duration_s": 0.0,
        "explicit_hold_s": 0.0,
    }
    values.update(updates)
    return SegmentCompletionSpec(**values)


def _feedback(
    elapsed_s: float,
    sim_step: int,
    *,
    errors: dict[str, Any] | None = None,
    velocities: dict[str, Any] | None = None,
) -> SegmentFeedback:
    return SegmentFeedback(
        elapsed_s=elapsed_s,
        sim_time_s=elapsed_s,
        sim_step=sim_step,
        servo_errors_deg=(
            {"front_left_hip": 0.0} if errors is None else errors
        ),
        servo_velocity_deg_s=(
            {"front_left_hip": 0.0}
            if velocities is None
            else velocities
        ),
        tracking_evidence={"supported": True, "converged": False},
    )


class SegmentCompletionSpecContractTest(unittest.TestCase):
    def test_spec_round_trip_preserves_execution_policy(self) -> None:
        original = _spec(
            servo_tolerance_deg=2.25,
            recorded_servo_residual_deg={"front_left_hip": -1.5},
            wheel_active_duration_s=0.2,
            explicit_hold_s=0.3,
        )
        decoded = SegmentCompletionSpec.from_mapping(original.to_mapping())
        self.assertEqual(decoded.to_mapping(), original.to_mapping())

    def test_residual_legacy_and_tolerance_contracts_are_strict(self) -> None:
        for field in ("segment_index", "source_step"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "non-negative int"):
                    _spec(**{field: -1})
        with self.assertRaisesRegex(ValueError, "non-legacy residuals"):
            _spec(recorded_servo_residual_deg={})
        with self.assertRaisesRegex(ValueError, "legacy-missing"):
            _spec(
                legacy_missing_endpoint=True,
                recorded_servo_residual_deg={"front_left_hip": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            _spec(
                servo_targets_deg={
                    "front_left_hip": 10.0,
                    "front_right_hip": 20.0,
                },
                recorded_servo_residual_deg={"front_left_hip": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "without servo targets"):
            _spec(
                servo_targets_deg={},
                recorded_servo_residual_deg={"front_left_hip": 0.0},
            )
        for tolerance in (0.999, 3.001):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "within 1..3"):
                    _spec(servo_tolerance_deg=tolerance)
        legacy = _spec(
            legacy_missing_endpoint=True,
            recorded_servo_residual_deg={},
        )
        self.assertEqual(legacy.servo_tolerance_deg, 1.0)

    def test_v003_all_52_legacy_servo_segments_build_strict_specs(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "saved_height_steps_fsm_reference_v2"
            / "height_050mm"
            / "versions"
            / "v003_20260805_224517_157723_manual"
            / "accepted_steps.jsonl"
        )
        plan = plan_from_jsonl(source, profile="fast")
        legacy_count = 0
        for segment in plan.segments:
            SegmentCompletionSpec(
                segment_index=segment.segment_index,
                source_step=segment.source_step,
                source_step_id=segment.source_step_id,
                servo_targets_deg=segment.servo_targets,
                servo_duration_s=segment.servo_duration_s,
                servo_tolerance_deg=segment.servo_tolerance_deg,
                recorded_servo_residual_deg=segment.recorded_servo_residual_deg,
                legacy_missing_endpoint=segment.legacy_missing_endpoint,
                wheel_active_duration_s=segment.wheel_active_duration_s,
                explicit_hold_s=segment.explicit_hold_s,
            )
            legacy_count += bool(
                segment.servo_targets and segment.legacy_missing_endpoint
            )
        self.assertEqual(len(plan.segments), 112)
        self.assertEqual(legacy_count, 52)


class CompletionAwareSegmentExecutorTest(unittest.TestCase):
    def test_n_plus_one_monotonic_and_terminal_exact_once(self) -> None:
        executor = CompletionAwareSegmentExecutor()
        executor.start(
            _spec(),
            start_elapsed_s=0.0,
            start_sim_time_s=0.0,
            start_sim_step=10,
        )
        with self.assertRaisesRegex(ValueError, "after start_sim_step"):
            executor.observe(_feedback(0.1, 10))
        decision = executor.observe(_feedback(0.1, 18))
        self.assertEqual(decision.kind, SegmentDecisionKind.COMPLETE)
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            executor.observe(_feedback(0.2, 26))

    def test_missing_nonfinite_and_empty_target_measurements_fail_closed(self) -> None:
        vectors = (
            ({}, {"front_left_hip": 0.0}),
            ({"front_left_hip": float("nan")}, {"front_left_hip": 0.0}),
            ({"front_left_hip": 0.0}, {}),
            ({"front_left_hip": 0.0}, {"front_left_hip": float("inf")}),
        )
        for errors, velocities in vectors:
            with self.subTest(errors=errors, velocities=velocities):
                executor = CompletionAwareSegmentExecutor()
                executor.start(
                    _spec(servo_duration_s=0.0),
                    start_elapsed_s=0.0,
                    start_sim_time_s=0.0,
                    start_sim_step=0,
                )
                decision = executor.observe(
                    _feedback(
                        0.1,
                        8,
                        errors=errors,
                        velocities=velocities,
                    )
                )
                self.assertEqual(decision.kind, SegmentDecisionKind.FAIL)
                self.assertEqual(decision.failure_reason, "invalid_joint_state")

        executor = CompletionAwareSegmentExecutor()
        executor.start(
            _spec(
                servo_targets_deg={},
                recorded_servo_residual_deg={},
                servo_duration_s=0.0,
            ),
            start_elapsed_s=0.0,
            start_sim_time_s=0.0,
            start_sim_step=0,
        )
        decision = executor.observe(
            _feedback(
                0.1,
                8,
                errors={"unexpected": 0.0},
                velocities={},
            )
        )
        self.assertEqual(decision.kind, SegmentDecisionKind.FAIL)

    def test_dynamic_duration_override_delays_measured_completion(self) -> None:
        executor = CompletionAwareSegmentExecutor()
        executor.start(
            _spec(servo_duration_s=0.1),
            start_elapsed_s=0.0,
            start_sim_time_s=0.0,
            start_sim_step=0,
            servo_duration_s_override=0.2,
        )
        waiting = executor.observe(
            _feedback(0.1, 8, errors={}, velocities={})
        )
        self.assertEqual(waiting.kind, SegmentDecisionKind.WAIT)
        self.assertFalse(waiting.servo_planned_done)
        complete = executor.observe(_feedback(0.2, 16))
        self.assertEqual(complete.kind, SegmentDecisionKind.COMPLETE)

    def test_contact_grace_starts_at_nominal_servo_end(self) -> None:
        executor = CompletionAwareSegmentExecutor()
        executor.start(
            _spec(
                servo_tolerance_deg=2.25,
                recorded_servo_residual_deg={"front_left_hip": -1.5},
            ),
            start_elapsed_s=0.0,
            start_sim_time_s=0.0,
            start_sim_step=0,
        )
        nominal = executor.observe(
            _feedback(0.1, 8, errors={"front_left_hip": -2.0})
        )
        self.assertEqual(nominal.kind, SegmentDecisionKind.WAIT)
        self.assertTrue(nominal.contact_candidate)
        self.assertFalse(nominal.contact_grace_done)
        almost = executor.observe(
            _feedback(0.349, 40, errors={"front_left_hip": -2.0})
        )
        self.assertEqual(almost.kind, SegmentDecisionKind.WAIT)
        complete = executor.observe(
            _feedback(0.36, 48, errors={"front_left_hip": -2.0})
        )
        self.assertEqual(complete.kind, SegmentDecisionKind.COMPLETE)
        self.assertAlmostEqual(complete.contact_extension_s, 0.26)

    def test_wheel_stop_is_dynamic_external_and_exact_once(self) -> None:
        executor = CompletionAwareSegmentExecutor()
        executor.start(
            _spec(
                servo_duration_s=0.1,
                wheel_active_duration_s=0.1,
            ),
            start_elapsed_s=0.0,
            start_sim_time_s=0.0,
            start_sim_step=0,
        )
        due = executor.observe(
            _feedback(
                0.1,
                8,
                errors={"front_left_hip": -2.0},
                velocities={"front_left_hip": 10.0},
            )
        )
        self.assertEqual(due.kind, SegmentDecisionKind.WHEEL_STOP_DUE)
        self.assertTrue(due.wheel_stop_due)
        with self.assertRaisesRegex(ValueError, r"exact N\+1"):
            executor.acknowledge_wheel_stop(
                applied_sim_step=9,
                first_physics_step=10,
                batch_id="stale",
            )
        executor.acknowledge_wheel_stop(
            applied_sim_step=8,
            first_physics_step=9,
            batch_id="wheel-stop-1",
        )
        with self.assertRaisesRegex(RuntimeError, "already acknowledged"):
            executor.acknowledge_wheel_stop(
                applied_sim_step=8,
                first_physics_step=9,
                batch_id="wheel-stop-2",
            )
        waiting = executor.observe(
            _feedback(
                0.2,
                16,
                errors={"front_left_hip": -1.5},
                velocities={"front_left_hip": 10.0},
            )
        )
        self.assertEqual(waiting.kind, SegmentDecisionKind.WAIT)
        self.assertFalse(waiting.wheel_stop_due)

    def test_explicit_hold_runs_concurrently_from_segment_start(self) -> None:
        executor = CompletionAwareSegmentExecutor()
        executor.start(
            _spec(
                servo_targets_deg={},
                recorded_servo_residual_deg={},
                servo_duration_s=0.0,
                wheel_active_duration_s=0.1,
                explicit_hold_s=0.3,
            ),
            start_elapsed_s=0.0,
            start_sim_time_s=0.0,
            start_sim_step=0,
        )
        due = executor.observe(
            _feedback(0.1, 8, errors={}, velocities={})
        )
        self.assertEqual(due.kind, SegmentDecisionKind.WHEEL_STOP_DUE)
        self.assertTrue(due.wheel_done)
        self.assertFalse(due.hold_done)
        executor.acknowledge_wheel_stop(
            applied_sim_step=8,
            first_physics_step=9,
            batch_id="wheel-stop-hold",
        )
        complete = executor.observe(
            _feedback(0.3, 24, errors={}, velocities={})
        )
        self.assertEqual(complete.kind, SegmentDecisionKind.COMPLETE)


class PlaybackCompletionSeamTest(unittest.TestCase):
    class Adapter:
        def __init__(
            self,
            *,
            measured_deg: float | None,
            velocity_deg_s: float = 0.0,
            actual_offset_deg: float = 0.0,
            reference_deg_s: Any = 150.0,
            velocity_limit_deg_s: Any = 150.0,
            canonical_command: Any = 0.0,
        ) -> None:
            self.joint_command_deg = {"front_left_hip": canonical_command}
            self.motion_reference = SimpleNamespace(
                servo_reference_velocity_deg_s=reference_deg_s,
                servo_velocity_limit_deg_s=velocity_limit_deg_s,
            )
            self.measured_deg = measured_deg
            self.velocity_deg_s = velocity_deg_s
            self.actual_offset_deg = actual_offset_deg

        def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.joint_command_deg.update(payload.get("servo_targets_deg", {}))
            return {}

        def get_actual_joint_state(self) -> dict[str, Any]:
            servos = {}
            if self.measured_deg is not None:
                servos["front_left_hip"] = {
                    "deg": self.measured_deg,
                    "velocity_deg_s": self.velocity_deg_s,
                }
            return {"servos": servos}

        def command_to_actual_target_deg(self, _name: str, target: float) -> float:
            return float(target) + self.actual_offset_deg

        def stop_wheels(self) -> None:
            return None

        def apply_commands_to_robot(self) -> None:
            return None

    @staticmethod
    def plan() -> PlaybackPlan:
        event = PlaybackEvent(
            time_s=0.0,
            command="servo front_left_hip 10",
            source_step=1,
            source_step_id="step-1",
            global_command_index=1,
            segment_index=0,
            channel="servo",
            servo_targets=(("front_left_hip", 10.0),),
        )
        segment = PlaybackSegment(
            segment_index=0,
            source_step=1,
            source_step_id="step-1",
            event_start_index=0,
            event_count=1,
            planned_start_s=0.0,
            planned_end_s=0.1,
            base_duration_s=0.1,
            servo_base_duration_s=0.1,
            servo_duration_s=0.1,
            servo_targets={"front_left_hip": 10.0},
            servo_tolerance_deg=1.0,
            recorded_servo_residual_deg={"front_left_hip": 0.0},
        )
        plan = PlaybackPlan(
            path=None,
            events=[event],
            segments=[segment],
            final_time_s=0.1,
            total_steps=1,
        )
        plan.plan_sha256 = plan_fingerprint(plan)
        return plan

    def test_wrapper_observes_only_later_outer_loop_steps_and_uses_conversion(self) -> None:
        adapter = self.Adapter(measured_deg=110.0, actual_offset_deg=100.0)
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=0,
            current_wall_time_s=0.0,
        )
        executor = service.segment_completion_executor
        self.assertIsNone(executor.last_feedback)
        self.assertEqual(executor.observation_count, 0)
        service.update(
            adapter,
            current_sim_time_s=0.1,
            current_sim_step=8,
            current_wall_time_s=0.1,
        )
        self.assertEqual(executor.observation_count, 1)
        self.assertEqual(executor.last_feedback.sim_step, 8)
        self.assertEqual(service.stop_reason, "complete")

    def test_wrapper_missing_measured_joint_fails_at_first_completion_check(self) -> None:
        adapter = self.Adapter(measured_deg=None)
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=0,
            current_wall_time_s=0.0,
        )
        service.update(
            adapter,
            current_sim_time_s=0.1,
            current_sim_step=8,
            current_wall_time_s=0.1,
        )
        self.assertFalse(service.active)
        self.assertEqual(service.stop_reason, "invalid_joint_state")
        self.assertIn("keys are not exact", service.last_error)

    def test_dynamic_duration_uses_finite_positive_effective_speed(self) -> None:
        adapter = self.Adapter(
            measured_deg=10.0,
            reference_deg_s=150.0,
            velocity_limit_deg_s=75.0,
        )
        service = SimTimePlaybackService()
        plan = self.plan()
        self.assertTrue(
            service.start_plan(
                plan, current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=0,
            current_wall_time_s=0.0,
        )
        expected = 10.0 / 75.0
        self.assertAlmostEqual(service.plan.segments[0].servo_duration_s, expected)
        self.assertAlmostEqual(
            service.segment_completion_executor.spec.servo_duration_s,
            expected,
        )

    def test_invalid_dynamic_duration_inputs_fail_through_service(self) -> None:
        vectors = (
            {"canonical_command": float("nan")},
            {"canonical_command": "not-a-number"},
            {"reference_deg_s": float("nan")},
            {"reference_deg_s": 0.0},
            {"reference_deg_s": -1.0},
            {"velocity_limit_deg_s": float("inf")},
            {"velocity_limit_deg_s": 0.0},
            {"velocity_limit_deg_s": -1.0},
        )
        for values in vectors:
            with self.subTest(values=values):
                adapter = self.Adapter(measured_deg=10.0, **values)
                service = SimTimePlaybackService()
                self.assertTrue(
                    service.start_plan(
                        self.plan(),
                        current_sim_time_s=0.0,
                        current_wall_time_s=0.0,
                    )
                )
                service.update(
                    adapter,
                    current_sim_time_s=0.0,
                    current_sim_step=0,
                    current_wall_time_s=0.0,
                )
                self.assertFalse(service.active)
                self.assertEqual(
                    service.stop_reason, "invalid_servo_duration_reference"
                )
                self.assertIn(
                    "invalid_servo_duration_reference", service.last_error
                )

        adapter = self.Adapter(measured_deg=10.0)
        adapter.joint_command_deg = {}
        service = SimTimePlaybackService()
        self.assertTrue(
            service.start_plan(
                self.plan(), current_sim_time_s=0.0, current_wall_time_s=0.0
            )
        )
        service.update(
            adapter,
            current_sim_time_s=0.0,
            current_sim_step=0,
            current_wall_time_s=0.0,
        )
        self.assertFalse(service.active)
        self.assertEqual(
            service.stop_reason, "invalid_servo_duration_reference"
        )
        self.assertIn("missing target keys", service.last_error)


if __name__ == "__main__":
    unittest.main()
