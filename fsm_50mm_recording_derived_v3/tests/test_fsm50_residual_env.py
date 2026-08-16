from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.fsm50_motion_profiles import build_profile_library
from fsm_50mm_recording_derived_v3.fsm50_phase_entry_bank import load_phase_entry_bank
from fsm_50mm_recording_derived_v3.fsm50_residual_envelope import load_residual_envelope
from fsm_50mm_recording_derived_v3.fsm50_residual_env import (
    ACTOR_OBSERVATION_DIM,
    BatchReadback,
    CRITIC_STATE_DIM,
    EPISODE_METRICS_FIELDS,
    EPISODE_METRICS_SCHEMA_SHA256,
    EPISODE_METRICS_SCHEMA_VERSION,
    GYM_ENV_ID,
    PHYSX_TARGET_READBACK_SOURCE,
    PhaseLocalResidualKernel,
    PhasePhysicsFeedback,
    ResidualEnvContractError,
    SERVO_MAX_DELTA_DEG_PER_PHYSICS_STEP,
    build_actor_critic_observation_tensors,
    build_phase_episode_plan,
    env_source_sha256,
    register_gym_env,
    slew_servo_targets_150_deg_s,
    validate_physx_target_readback,
)
from fsm_50mm_recording_derived_v3.fsm50_residual_outer_cycle import (
    EXPECTED_RENDER_SUBSTEPS,
    OuterCycleContext,
)
from fsm_50mm_recording_derived_v3.fsm50_residual_observation import (
    ACTOR_OBSERVATION_SCHEMA_SHA256,
)
from fsm_50mm_recording_derived_v3.fsm50_residual_scene import (
    OBSTACLE_FRONT_X_M,
    OBSTACLE_HEIGHT_M,
    PHYSICS_DT_S,
)


REPLAY_ROOT = Path(__file__).resolve().parents[2]
V003 = "v003_20260805_224517_157723_manual"
V008 = "v008_20260806_211408_578700_manual"
V009 = "v009_20260806_215232_433234_manual"
S5 = "S5_PRE_RR_COM_SHIFT"
S7 = "S7_PRE_RL_SUPPORT_SETUP"
S8 = "S8_RL_COM_SHIFT_AND_TRAVERSE"
S10 = "S10_POSTURE_RECOVERY"
ZERO = (0.0,) * 12


def _context(step: int) -> OuterCycleContext:
    outer, substep = divmod(step, EXPECTED_RENDER_SUBSTEPS)
    return OuterCycleContext(
        outer_cycle_index=outer,
        physics_substep_index=substep,
        source_cursor_permit=substep == 0,
        policy_update_permit=substep == 0,
    )


def _telemetry(kernel: PhaseLocalResidualKernel, **overrides):
    reset = kernel.reset_state
    all_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    base = {
        "x": reset.root_position_m[0],
        "y": reset.root_position_m[1],
        "z": reset.root_position_m[2],
    }
    centers = {
        "FL": (base["x"] + 0.18, base["y"] + 0.14, 0.05),
        "FR": (base["x"] + 0.18, base["y"] - 0.14, 0.05),
        "RL": (base["x"] - 0.18, base["y"] + 0.14, 0.05),
        "RR": (base["x"] - 0.18, base["y"] - 0.14, 0.05),
    }
    payload = {
        "macro_state": reset.phase_state,
        "source_version": reset.source_version,
        "profile_fraction": kernel.profile_fraction,
        "active_leg": {S5: "RR", S7: "RL", S8: "RL", S10: ""}[reset.phase_state],
        "support_legs": ("FL", "FR", "RL", "RR"),
        "base_position_m": base,
        "base_roll_rad": 0.0,
        "base_pitch_rad": 0.0,
        "base_yaw_rad": 0.0,
        "root_linear_velocity_w": (0.0, 0.0, 0.0),
        "root_angular_velocity_w": (0.0, 0.0, 0.0),
        "joint_q_rad": dict(zip(all_names, reset.joint_q_rad)),
        "joint_qd_rad_s": dict(zip(all_names, (0.0,) * 12)),
        "servo_targets_deg": dict(kernel.current_nominal_servo_targets_deg),
        "canonical_servo_actual_error_deg": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "wheel_targets_rad_s": dict(kernel.current_nominal_wheel_targets_rad_s),
        "wheel_center_w_m": centers,
        "wheel_contact_classes": {leg: "GROUND" for leg in ("FL", "FR", "RL", "RR")},
        "wheel_front_face_clearance_m": {leg: -0.10 for leg in ("FL", "FR", "RL", "RR")},
        "wheel_top_clearance_m": {leg: -0.05 for leg in ("FL", "FR", "RL", "RR")},
        "obstacle_front_face_x_m": OBSTACLE_FRONT_X_M,
        "obstacle_top_z_m": OBSTACLE_HEIGHT_M,
        "geometry_support_candidate_count": 4,
        "body_crossed_front_face": False,
        "final_recoverable": True,
        "stability_state": "stable",
    }
    payload.update(overrides)
    return payload


def _feedback(
    kernel: PhaseLocalResidualKernel,
    step: int,
    *,
    readback: BatchReadback | None = None,
    telemetry_overrides=None,
    hard_failure: bool = False,
    hard_failure_reason: str = "",
    effective_errors=None,
    servo_velocities=None,
) -> PhasePhysicsFeedback:
    return PhasePhysicsFeedback(
        sim_step=step,
        sim_time_s=step * PHYSICS_DT_S,
        telemetry=_telemetry(kernel, **(telemetry_overrides or {})),
        effective_servo_actual_error_deg=dict(
            effective_errors
            if effective_errors is not None
            else {name: 0.0 for name in SERVO_JOINT_NAMES}
        ),
        servo_velocity_deg_s=dict(
            servo_velocities
            if servo_velocities is not None
            else {name: 0.0 for name in SERVO_JOINT_NAMES}
        ),
        batch_readback=readback,
        hard_failure=hard_failure,
        hard_failure_reason=hard_failure_reason,
    )


def _readback(batch):
    return BatchReadback(
        batch_id=batch.batch_id,
        applied_sim_step=batch.decision_sim_step,
        first_physics_step=batch.expected_first_physics_step,
        physical_command_epoch=batch.physical_command_epoch,
        physx_evidence_source=PHYSX_TARGET_READBACK_SOURCE,
        physx_evidence_sha256="0" * 64,
        physx_failure_reason="",
        verified=True,
    )


class _FakeArray:
    def __init__(self, values, *, shape=None):
        self._values = copy.deepcopy(values)
        if shape is not None:
            self.shape = tuple(shape)
        elif values and isinstance(values[0], (list, tuple)):
            self.shape = (len(values), len(values[0]))
        else:
            self.shape = (len(values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self._values)


def _physx_readback_fixture(*, num_envs=2, env_index=1):
    joint_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    servo_ids = tuple(range(len(SERVO_JOINT_NAMES)))
    wheel_ids = tuple(range(len(SERVO_JOINT_NAMES), len(joint_names)))
    expected_servo = [0.125 * (index + 1) for index in range(len(SERVO_JOINT_NAMES))]
    expected_wheel = [0.25 * (index + 1) for index in range(len(WHEEL_JOINT_NAMES))]
    position_rows = [[0.0] * len(joint_names) for _ in range(num_envs)]
    velocity_rows = [[0.0] * len(joint_names) for _ in range(num_envs)]
    for column, value in zip(servo_ids, expected_servo):
        position_rows[env_index][column] = value
    for column, value in zip(wheel_ids, expected_wheel):
        velocity_rows[env_index][column] = value
    view = SimpleNamespace(
        get_dof_position_targets=lambda: _FakeArray(position_rows),
        get_dof_velocity_targets=lambda: _FakeArray(velocity_rows),
    )
    # Matching Isaac Lab command buffers are diagnostic bait: verification
    # must remain exclusively rooted in ``root_physx_view``.
    data = SimpleNamespace(
        joint_pos_target=_FakeArray(position_rows),
        joint_vel_target=_FakeArray(velocity_rows),
    )
    robot = SimpleNamespace(joint_names=joint_names, root_physx_view=view, data=data)
    kwargs = {
        "robot": robot,
        "num_envs": num_envs,
        "env_index": env_index,
        "servo_joint_ids": servo_ids,
        "wheel_joint_ids": wheel_ids,
        "expected_servo_position_targets": _FakeArray(expected_servo),
        "expected_wheel_velocity_targets": _FakeArray(expected_wheel),
    }
    return kwargs, position_rows, velocity_rows


class ResidualEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = load_phase_entry_bank(verify_artifacts=True)
        cls.library = build_profile_library(REPLAY_ROOT)
        cls.envelope = load_residual_envelope(verify_evidence=True)

    def plan(self, source: str, state: str):
        return build_phase_episode_plan(
            f"{source}:{state}",
            bank=self.bank,
            profile_library=self.library,
            verify_artifacts=True,
        )

    def kernel(self, source=V008, state=S5):
        return PhaseLocalResidualKernel(self.plan(source, state), self.envelope)

    def test_low_level_physx_target_readback_exact_match_is_verified(self):
        kwargs, _position, _velocity = _physx_readback_fixture()
        evidence = validate_physx_target_readback(**kwargs)
        self.assertTrue(evidence.verified)
        self.assertEqual(evidence.failure_reason, "")
        self.assertEqual(len(evidence.evidence_sha256), 64)

    def test_physx_target_readback_fail_closed_vectors(self):
        def position_mismatch(kwargs, positions, _velocities):
            positions[1][0] += 1.0
            kwargs["robot"].root_physx_view.get_dof_position_targets = lambda: _FakeArray(positions)

        def velocity_mismatch(kwargs, _positions, velocities):
            velocities[1][len(SERVO_JOINT_NAMES)] += 1.0
            kwargs["robot"].root_physx_view.get_dof_velocity_targets = lambda: _FakeArray(velocities)

        def missing_getter(kwargs, _positions, _velocities):
            kwargs["robot"].root_physx_view = SimpleNamespace(
                get_dof_position_targets=kwargs["robot"].root_physx_view.get_dof_position_targets
            )

        def nonfinite(kwargs, positions, _velocities):
            positions[0][-1] = math.nan
            kwargs["robot"].root_physx_view.get_dof_position_targets = lambda: _FakeArray(positions)

        def wrong_shape(kwargs, positions, _velocities):
            kwargs["robot"].root_physx_view.get_dof_position_targets = lambda: _FakeArray(
                positions,
                shape=(1, len(positions[0])),
            )

        def wrong_ids(kwargs, _positions, _velocities):
            kwargs["servo_joint_ids"] = tuple(reversed(kwargs["servo_joint_ids"]))

        for label, mutate in (
            ("position mismatch despite matching internal buffer", position_mismatch),
            ("velocity mismatch", velocity_mismatch),
            ("missing getter", missing_getter),
            ("nonfinite full PhysX matrix", nonfinite),
            ("wrong env-row shape", wrong_shape),
            ("wrong canonical joint IDs", wrong_ids),
        ):
            with self.subTest(label=label):
                kwargs, positions, velocities = _physx_readback_fixture()
                buffered_position_before = kwargs["robot"].data.joint_pos_target.tolist()
                mutate(kwargs, positions, velocities)
                evidence = validate_physx_target_readback(**kwargs)
                self.assertFalse(evidence.verified)
                self.assertTrue(evidence.failure_reason)
                self.assertEqual(len(evidence.evidence_sha256), 64)
                if label.startswith("position mismatch"):
                    self.assertEqual(
                        kwargs["robot"].data.joint_pos_target.tolist(),
                        buffered_position_before,
                    )

    def test_unverified_physx_target_evidence_drives_kernel_safe_stop(self):
        kwargs, positions, _velocities = _physx_readback_fixture()
        positions[1][0] += 1.0
        kwargs["robot"].root_physx_view.get_dof_position_targets = lambda: _FakeArray(positions)
        evidence = validate_physx_target_readback(**kwargs)
        self.assertFalse(evidence.verified)

        kernel = self.kernel()
        source = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        batch = source.command_batch
        readback = BatchReadback(
            batch_id=batch.batch_id,
            applied_sim_step=batch.decision_sim_step,
            first_physics_step=batch.expected_first_physics_step,
            physical_command_epoch=batch.physical_command_epoch,
            physx_evidence_source=PHYSX_TARGET_READBACK_SOURCE,
            physx_evidence_sha256=evidence.evidence_sha256,
            physx_failure_reason=evidence.failure_reason,
            verified=False,
        )
        failed = kernel.step(ZERO, _context(1), _feedback(kernel, 1, readback=readback))
        self.assertTrue(failed.terminal_latched)
        self.assertFalse(failed.phase_success)
        self.assertIn("low-level PhysX", failed.terminal_reason)
        self.assertIsNotNone(failed.command_batch)
        self.assertEqual(failed.command_batch.kind, "SAFE_STOP")

    def test_all_three_by_four_bank_entries_reset_without_controller_state(self):
        observed = set()
        profile_free = set()
        for source in (V003, V008, V009):
            for state in (S5, S7, S8, S10):
                plan = self.plan(source, state)
                observed.add((plan.reset.source_version, plan.reset.phase_state))
                self.assertEqual(len(plan.reset.joint_q_rad), 12)
                self.assertEqual(len(plan.reset.joint_qd_rad_s), 12)
                self.assertEqual(tuple(plan.reset.nominal_servo_targets_deg), tuple(SERVO_JOINT_NAMES))
                self.assertEqual(tuple(plan.reset.nominal_wheel_targets_rad_s), tuple(WHEEL_JOINT_NAMES))
                if plan.profile_free:
                    profile_free.add((source, state))
                    self.assertIsNone(plan.profile)
                    self.assertEqual(plan.segment_frames, ())
        self.assertEqual(len(observed), 12)
        self.assertEqual(profile_free, {(V003, S5), (V003, S7), (V008, S7)})

    def test_bank_tamper_fails_before_plan_creation(self):
        tampered = self.bank.to_mapping()
        tampered["payload"]["entries"][0]["root_position_m"][0] += 0.001
        with self.assertRaisesRegex(Exception, "bank_sha256|entry_sha256|canonical|exact deterministic"):
            build_phase_episode_plan(
                tampered["payload"]["entries"][0]["entry_id"],
                bank=tampered,
                profile_library=self.library,
                verify_artifacts=False,
            )

    def test_servo_slew_is_exact_150_deg_per_second_and_strict(self):
        current = {name: 0.0 for name in SERVO_JOINT_NAMES}
        endpoint = {name: (10.0 if index % 2 == 0 else -10.0) for index, name in enumerate(SERVO_JOINT_NAMES)}
        advanced = slew_servo_targets_150_deg_s(current, endpoint)
        self.assertEqual(
            tuple(advanced[name] for name in SERVO_JOINT_NAMES),
            tuple(SERVO_MAX_DELTA_DEG_PER_PHYSICS_STEP if index % 2 == 0 else -SERVO_MAX_DELTA_DEG_PER_PHYSICS_STEP for index in range(8)),
        )
        with self.assertRaises(ResidualEnvContractError):
            slew_servo_targets_150_deg_s({**current, "extra": 0.0}, endpoint)

    def test_zero_action_is_bit_exact_nominal_and_creates_no_extra_dispatch(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        self.assertEqual(first.source_consumed_segment_index, 41)
        self.assertIsNotNone(first.command_batch)
        self.assertEqual(dict(first.command_batch.applied_servo_targets_deg), dict(first.command_batch.nominal_servo_targets_deg))
        self.assertEqual(dict(first.command_batch.applied_wheel_targets_rad_s), dict(first.command_batch.nominal_wheel_targets_rad_s))
        pending = first.command_batch
        for step in range(1, 8):
            result = kernel.step(
                ZERO,
                _context(step),
                _feedback(kernel, step, readback=_readback(pending) if step == 1 else None),
            )
            self.assertIsNone(result.command_batch)
            self.assertIsNone(result.source_consumed_segment_index)
        self.assertEqual(len(first.observation.values), ACTOR_OBSERVATION_DIM)

    def test_episode_metrics_schema_is_exact_stable_and_identity_bound(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        metrics = first.info["episode_metrics"]
        self.assertEqual(tuple(metrics), EPISODE_METRICS_FIELDS)
        self.assertEqual(metrics["schema_version"], EPISODE_METRICS_SCHEMA_VERSION)
        self.assertEqual(first.info["episode_metrics_schema_sha256"], EPISODE_METRICS_SCHEMA_SHA256)
        self.assertEqual(metrics["env_source_sha256"], env_source_sha256())
        self.assertEqual(len(metrics["env_source_sha256"]), 64)
        self.assertEqual(metrics["entry_sha256"], kernel.reset_state.entry_sha256)
        self.assertEqual(metrics["bank_sha256"], kernel.reset_state.bank_sha256)
        self.assertEqual(metrics["source_consumption_count"], 1)
        self.assertEqual(metrics["physical_batch_count"], 1)
        self.assertEqual(metrics["residual_transform_count"], 1)
        self.assertEqual(metrics["physical_command_epoch"], 1)
        self.assertEqual(metrics["last_verified_physical_command_epoch"], 0)
        self.assertEqual(metrics["n_plus_one_verified_count"], 0)
        self.assertFalse(metrics["n_plus_one_verified"])
        # The public metrics record is strict JSON and contains no tensors or
        # implementation-private state.
        json.dumps(dict(metrics), allow_nan=False, sort_keys=True)

    def test_nonzero_policy_update_between_sources_emits_residual_only_n_plus_one_epoch(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        pending = first.command_batch
        for step in range(1, 8):
            result = kernel.step(
                ZERO,
                _context(step),
                _feedback(kernel, step, readback=_readback(pending) if step == 1 else None),
            )
            self.assertIsNone(result.command_batch)
        action = (1.0,) + (0.0,) * 11
        waiting_errors = {name: 5.0 for name in SERVO_JOINT_NAMES}
        updated = kernel.step(
            action,
            _context(8),
            _feedback(kernel, 8, effective_errors=waiting_errors),
        )
        self.assertEqual(updated.completion_decision["kind"], "WAIT")
        self.assertIsNone(updated.source_consumed_segment_index)
        self.assertEqual(updated.command_batch.kind, "RESIDUAL_ONLY")
        self.assertFalse(updated.command_batch.source_action)
        self.assertEqual(updated.command_batch.physical_command_epoch, 2)
        self.assertNotEqual(
            updated.command_batch.applied_servo_targets_deg[SERVO_JOINT_NAMES[0]],
            updated.command_batch.nominal_servo_targets_deg[SERVO_JOINT_NAMES[0]],
        )
        verified = kernel.step(
            action,
            _context(9),
            _feedback(kernel, 9, readback=_readback(updated.command_batch)),
        )
        self.assertTrue(verified.info["n_plus_one_verified_this_step"])
        self.assertEqual(verified.info["last_verified_physical_command_epoch"], 2)
        self.assertIsNone(verified.command_batch)

    def test_zero_outer_policy_update_is_exact_identity_without_extra_batch(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        for step in range(1, 8):
            kernel.step(
                ZERO,
                _context(step),
                _feedback(kernel, step, readback=_readback(first.command_batch) if step == 1 else None),
            )
        before_applied = dict(kernel.current_applied_servo_targets_deg)
        waiting_errors = {name: 5.0 for name in SERVO_JOINT_NAMES}
        unchanged = kernel.step(
            ZERO,
            _context(8),
            _feedback(kernel, 8, effective_errors=waiting_errors),
        )
        self.assertEqual(unchanged.completion_decision["kind"], "WAIT")
        self.assertIsNone(unchanged.command_batch)
        self.assertEqual(dict(kernel.current_applied_servo_targets_deg), before_applied)
        self.assertTrue(unchanged.info["residual_transform_sha256_this_step"])
        self.assertEqual(unchanged.info["physical_command_epoch"], 1)

    def test_source_and_residual_are_coalesced_into_one_physical_epoch(self):
        kernel = self.kernel()
        action = (1.0,) + (0.0,) * 11
        result = kernel.step(action, _context(0), _feedback(kernel, 0))
        self.assertEqual(result.source_consumed_segment_index, 41)
        self.assertEqual(result.command_batch.kind, "SOURCE_ACTION")
        self.assertTrue(result.command_batch.source_action)
        self.assertEqual(result.command_batch.physical_command_epoch, 1)
        self.assertEqual(result.info["command_batch_count_this_step"], 1)
        self.assertEqual(result.info["episode_metrics"]["source_consumption_count"], 1)
        self.assertEqual(result.info["episode_metrics"]["residual_transform_count"], 1)
        self.assertEqual(result.info["episode_metrics"]["physical_batch_count"], 1)

    def test_active_completion_sparse_effective_target_follows_deployment_latch(self):
        kernel = self.kernel()
        action = (0.0, 1.0) + (0.0,) * 10
        nominal_spec = kernel.plan.profile.segment_binding(41).completion_spec.to_mapping()
        first = kernel.step(action, _context(0), _feedback(kernel, 0))
        sparse = dict(kernel.active_completion_effective_targets_deg)
        latch = dict(kernel.active_completion_latched_servo_residual_deg)
        self.assertEqual(set(sparse), set(latch))
        self.assertTrue(any(value != 0.0 for value in latch.values()))
        for step in range(1, 8):
            kernel.step(
                action,
                _context(step),
                _feedback(kernel, step, readback=_readback(first.command_batch) if step == 1 else None),
            )
        waiting_errors = {name: 5.0 for name in SERVO_JOINT_NAMES}
        updated = kernel.step(
            ZERO,
            _context(8),
            _feedback(kernel, 8, effective_errors=waiting_errors),
        )
        self.assertEqual(updated.completion_decision["kind"], "WAIT")
        self.assertEqual(dict(kernel.active_completion_effective_targets_deg), sparse)
        self.assertEqual(dict(kernel.active_completion_latched_servo_residual_deg), latch)
        live_spec = kernel._completion.spec.to_mapping()
        self.assertEqual(
            {key: value for key, value in live_spec.items() if key != "servo_duration_s"},
            {key: value for key, value in nominal_spec.items() if key != "servo_duration_s"},
        )
        self.assertNotEqual(live_spec["servo_duration_s"], nominal_spec["servo_duration_s"])
        self.assertEqual(
            kernel.plan.profile.segment_binding(41).completion_spec.to_mapping(),
            nominal_spec,
        )
        self.assertIsNone(updated.command_batch)
        self.assertTrue(updated.info["residual_transform_sha256_this_step"])

    def test_pending_n_plus_one_and_occupied_slot_reject_residual_second_batch(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        original_epoch = kernel.physical_command_epoch
        with self.assertRaisesRegex(ResidualEnvContractError, "pending N\\+1"):
            kernel._apply_policy_update(
                (1.0,) + (0.0,) * 11,
                _feedback(kernel, 0),
            )
        with self.assertRaisesRegex(ResidualEnvContractError, "more than one"):
            kernel._make_batch(
                kind="RESIDUAL_ONLY",
                sim_step=0,
                source_segment_index=None,
                source_action=False,
            )
        self.assertEqual(kernel.physical_command_epoch, original_epoch)
        self.assertEqual(first.command_batch.physical_command_epoch, original_epoch)

        mismatched = self.kernel()
        source = mismatched.step(ZERO, _context(0), _feedback(mismatched, 0))
        wrong_epoch = replace(
            _readback(source.command_batch),
            physical_command_epoch=source.command_batch.physical_command_epoch + 1,
        )
        failed = mismatched.step(
            ZERO,
            _context(1),
            _feedback(mismatched, 1, readback=wrong_epoch),
        )
        self.assertTrue(failed.terminal_latched)
        self.assertEqual(failed.command_batch.kind, "SAFE_STOP")
        self.assertEqual(failed.info["last_verified_physical_command_epoch"], 0)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("torch", "gymnasium", "skrl")),
        "torch/gymnasium/skrl are available in env_isaaclab",
    )
    def test_wrapper_style_actor_and_critic_views_are_equal_finite_and_value_compatible(self):
        import torch

        from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper

        from fsm_50mm_recording_derived_v3.fsm50_residual_models import (
            Fsm50ResidualValue,
            make_actor_space,
            make_observation_space,
        )

        views = build_actor_critic_observation_tensors(
            [[0.0] * ACTOR_OBSERVATION_DIM for _ in range(3)],
            torch_module=torch,
            device="cpu",
        )
        self.assertEqual(tuple(views["policy"].shape), (3, ACTOR_OBSERVATION_DIM))
        self.assertEqual(tuple(views["critic"].shape), (3, CRITIC_STATE_DIM))
        self.assertEqual(CRITIC_STATE_DIM, ACTOR_OBSERVATION_DIM)
        self.assertEqual(
            ACTOR_OBSERVATION_SCHEMA_SHA256,
            "6552a6cad5358f36ef4d60b0291658951c4a3901203e3943fd8770e5605bdb83",
        )
        self.assertTrue(torch.equal(views["policy"], views["critic"]))
        self.assertTrue(torch.isfinite(views["critic"]).all().item())

        observation_space = make_observation_space()
        action_space = make_actor_space()

        class _FakeIsaacLabEnv:
            device = "cpu"
            num_envs = 3
            single_observation_space = {
                "policy": observation_space,
                "critic": observation_space,
            }
            single_action_space = action_space

            def reset(self, *, seed=None):
                return dict(views), {"seed": seed}

            def step(self, actions):
                return (
                    dict(views),
                    torch.zeros(3),
                    torch.zeros(3, dtype=torch.bool),
                    torch.zeros(3, dtype=torch.bool),
                    {},
                )

            def close(self):
                return None

        wrapper = IsaacLabWrapper(_FakeIsaacLabEnv())
        wrapper_policy, _ = wrapper.reset()
        wrapper_state = wrapper.state()
        self.assertEqual(tuple(wrapper_policy.shape), (3, ACTOR_OBSERVATION_DIM))
        self.assertEqual(tuple(wrapper_state.shape), (3, CRITIC_STATE_DIM))
        self.assertTrue(torch.equal(wrapper_policy, wrapper_state))
        self.assertTrue(torch.isfinite(wrapper_state).all().item())
        self.assertEqual(wrapper.state_space.shape, (CRITIC_STATE_DIM,))

        value = Fsm50ResidualValue(observation_space, action_space, "cpu")
        output, extra = value.compute({"states": wrapper_state}, "value")
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(extra, {})

    def test_nonzero_action_obeys_frozen_phase_mask(self):
        kernel = self.kernel()
        frame = kernel.plan.segment_frames[0]
        contract = self.envelope.phase_contract(
            source_version=V008,
            profile_strategy=kernel.plan.profile_strategy,
            macro_state=S5,
            subphase=frame.subphase.value,
            nominal_servo_targets_deg=frame.servo_targets_deg,
            nominal_wheel_targets_rad_s=frame.wheel_targets_rad_s,
        )
        result = kernel.step((1.0,) * 12, _context(0), _feedback(kernel, 0))
        self.assertIsNotNone(result.command_batch)
        nominal = tuple(result.command_batch.nominal_servo_targets_deg[name] for name in SERVO_JOINT_NAMES) + tuple(
            result.command_batch.nominal_wheel_targets_rad_s[name] for name in WHEEL_JOINT_NAMES
        )
        applied = tuple(result.command_batch.applied_servo_targets_deg[name] for name in SERVO_JOINT_NAMES) + tuple(
            result.command_batch.applied_wheel_targets_rad_s[name] for name in WHEEL_JOINT_NAMES
        )
        for index, enabled in enumerate(contract.enabled_mask):
            if not enabled:
                self.assertEqual(applied[index], nominal[index])
        self.assertTrue(any(applied[index] != nominal[index] for index, enabled in enumerate(contract.enabled_mask) if enabled))

    def test_exact_eight_cadence_and_held_action_are_fail_closed(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        changed = kernel.step(
            (0.1,) + (0.0,) * 11,
            _context(1),
            _feedback(kernel, 1, readback=_readback(first.command_batch)),
        )
        self.assertTrue(changed.terminal_latched)
        self.assertEqual(changed.command_batch.kind, "SAFE_STOP")

        kernel = self.kernel()
        with self.assertRaisesRegex(ResidualEnvContractError, "outer-cycle context"):
            kernel.step(
                ZERO,
                OuterCycleContext(0, 1, False, False),
                _feedback(kernel, 0),
            )

    def test_source_batch_requires_exact_n_plus_one_readback(self):
        kernel = self.kernel()
        first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        self.assertIsNotNone(first.command_batch)
        failed = kernel.step(ZERO, _context(1), _feedback(kernel, 1))
        self.assertTrue(failed.terminal_latched)
        self.assertFalse(failed.terminated)
        self.assertEqual(failed.command_batch.kind, "SAFE_STOP")
        closed = kernel.step(
            ZERO,
            _context(2),
            _feedback(kernel, 2, readback=_readback(failed.command_batch)),
        )
        self.assertTrue(closed.terminated)
        self.assertFalse(closed.phase_success)

    def test_post_batch_failure_defers_safe_stop_without_breaking_one_batch_or_n_plus_one(self):
        kernel = self.kernel()
        original_reward = kernel._reward
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic post-batch failure")
            return original_reward(*args, **kwargs)

        with patch.object(kernel, "_reward", side_effect=fail_once):
            failed = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
        self.assertTrue(failed.terminal_latched)
        self.assertFalse(failed.terminated)
        self.assertEqual(failed.command_batch.kind, "SOURCE_ACTION")

        stop = kernel.step(
            ZERO,
            _context(1),
            _feedback(kernel, 1, readback=_readback(failed.command_batch)),
        )
        self.assertEqual(stop.command_batch.kind, "SAFE_STOP")
        self.assertFalse(stop.command_batch.source_action)
        self.assertFalse(stop.terminated)
        self.assertTrue(stop.info["n_plus_one_verified_this_step"])

        closed = kernel.step(
            ZERO,
            _context(2),
            _feedback(kernel, 2, readback=_readback(stop.command_batch)),
        )
        self.assertTrue(closed.terminated)
        self.assertFalse(closed.phase_success)
        self.assertIsNone(closed.command_batch)

    def test_profile_free_guard_runs_zero_composer_before_success_across_three_resets(self):
        for source in (V003, V008):
            kernel = self.kernel(source, S7)
            for _episode in range(3):
                first = kernel.step(ZERO, _context(0), _feedback(kernel, 0))
                self.assertTrue(first.phase_success)
                self.assertFalse(first.terminated)
                self.assertEqual(first.command_batch.kind, "SUCCESS_STOP")
                self.assertIsNone(first.source_consumed_segment_index)
                self.assertEqual(first.info["command_batch_count_this_step"], 1)
                self.assertTrue(first.info["residual_transform_sha256_this_step"])
                first_metrics = first.info["episode_metrics"]
                self.assertEqual(first_metrics["residual_transform_count"], 1)
                self.assertEqual(first_metrics["physical_batch_count"], 1)
                self.assertEqual(first_metrics["physical_command_epoch"], 1)
                self.assertEqual(first_metrics["last_verified_physical_command_epoch"], 0)
                self.assertEqual(first_metrics["n_plus_one_verified_count"], 0)
                self.assertFalse(first_metrics["n_plus_one_verified"])

                second = kernel.step(
                    ZERO,
                    _context(1),
                    _feedback(kernel, 1, readback=_readback(first.command_batch)),
                )
                self.assertTrue(second.phase_success)
                self.assertTrue(second.terminated)
                self.assertIsNone(second.command_batch)
                self.assertEqual(second.info["source_cursor_next_offset"], 0)
                metrics = second.info["episode_metrics"]
                self.assertEqual(metrics["residual_transform_count"], 1)
                self.assertEqual(metrics["physical_batch_count"], 1)
                self.assertEqual(metrics["physical_command_epoch"], 1)
                self.assertEqual(metrics["last_verified_physical_command_epoch"], 1)
                self.assertEqual(metrics["n_plus_one_verified_count"], 1)
                self.assertTrue(metrics["n_plus_one_verified"])
                kernel.reset()

    def test_profile_free_nonzero_residual_precedes_deferred_success_stop(self):
        action = (1.0,) + (0.0,) * 11
        for source in (V003, V008):
            # The reviewed S7 masks for these profile-free entries are all
            # zero.  Give this pure protocol test one synthetic authorized
            # servo axis so the physical-change/deferred-stop branch remains
            # reachable without weakening the production envelope.
            state_masks = {
                version: dict(masks)
                for version, masks in self.envelope.state_masks.items()
            }
            state_masks[source][S7] = (1.0,) + (0.0,) * 11
            envelope = replace(self.envelope, state_masks=state_masks)
            kernel = PhaseLocalResidualKernel(self.plan(source, S7), envelope)
            residual = kernel.step(action, _context(0), _feedback(kernel, 0))
            self.assertTrue(residual.phase_success)
            self.assertFalse(residual.terminated)
            self.assertEqual(residual.command_batch.kind, "RESIDUAL_ONLY")
            self.assertEqual(residual.info["command_batch_count_this_step"], 1)
            self.assertEqual(residual.info["episode_metrics"]["residual_transform_count"], 1)
            self.assertEqual(residual.info["episode_metrics"]["physical_batch_count"], 1)
            self.assertEqual(
                dict(kernel.current_applied_wheel_targets_rad_s),
                dict(residual.command_batch.applied_wheel_targets_rad_s),
            )

            stop = kernel.step(
                action,
                _context(1),
                _feedback(kernel, 1, readback=_readback(residual.command_batch)),
            )
            self.assertFalse(stop.terminated)
            self.assertEqual(stop.command_batch.kind, "SUCCESS_STOP")
            self.assertEqual(stop.info["command_batch_count_this_step"], 1)
            self.assertTrue(stop.info["n_plus_one_verified_this_step"])
            self.assertTrue(
                all(value == 0.0 for value in stop.command_batch.applied_wheel_targets_rad_s.values())
            )
            self.assertEqual(stop.info["episode_metrics"]["physical_batch_count"], 2)
            self.assertEqual(stop.info["episode_metrics"]["n_plus_one_verified_count"], 1)

            closed = kernel.step(
                action,
                _context(2),
                _feedback(kernel, 2, readback=_readback(stop.command_batch)),
            )
            self.assertTrue(closed.terminated)
            self.assertTrue(closed.phase_success)
            self.assertIsNone(closed.command_batch)
            metrics = closed.info["episode_metrics"]
            self.assertEqual(metrics["residual_transform_count"], 1)
            self.assertEqual(metrics["physical_batch_count"], 2)
            self.assertEqual(metrics["physical_command_epoch"], 2)
            self.assertEqual(metrics["last_verified_physical_command_epoch"], 2)
            self.assertEqual(metrics["n_plus_one_verified_count"], 2)
            self.assertTrue(metrics["n_plus_one_verified"])

    def test_completion_advances_only_on_outer_boundary_in_exact_source_order(self):
        kernel = self.kernel()
        pending = None
        consumed = []
        completion_steps = []
        for step in range(80):
            result = kernel.step(
                ZERO,
                _context(step),
                _feedback(kernel, step, readback=_readback(pending) if pending is not None else None),
            )
            pending = result.command_batch
            if result.source_consumed_segment_index is not None:
                consumed.append(result.source_consumed_segment_index)
                self.assertEqual(step % EXPECTED_RENDER_SUBSTEPS, 0)
            if result.completion_decision.get("kind") == "COMPLETE":
                completion_steps.append(step)
                self.assertEqual(step % EXPECTED_RENDER_SUBSTEPS, 0)
            if len(consumed) >= 3:
                break
        self.assertEqual(consumed[:3], [41, 42, 43])
        self.assertTrue(completion_steps)
        self.assertEqual(len(set(consumed)), len(consumed))

    def test_dynamic_wheel_stop_is_non_source_one_batch_and_acknowledged_n_plus_one(self):
        plan = self.plan(V003, S10)
        first_binding = plan.profile.segment_bindings[0]
        segment_payload = dict(first_binding.segment_payload)
        # Synthetic completion vector: the production helper must stop the
        # active wheel while the 0.254 s servo slew remains incomplete.
        segment_payload["wheel_active_duration_s"] = 0.05
        synthetic_binding = replace(first_binding, segment_payload=segment_payload)
        synthetic_profile = replace(
            plan.profile,
            segment_bindings=(synthetic_binding, *plan.profile.segment_bindings[1:]),
        )
        kernel = PhaseLocalResidualKernel(replace(plan, profile=synthetic_profile), self.envelope)
        pending = None
        due = None
        for step in range(40):
            result = kernel.step(
                ZERO,
                _context(step),
                _feedback(kernel, step, readback=_readback(pending) if pending is not None else None),
            )
            pending = result.command_batch
            if result.command_batch is not None and result.command_batch.kind == "WHEEL_STOP_OVERRIDE":
                due = result.command_batch
                self.assertFalse(due.source_action)
                self.assertIsNone(result.source_consumed_segment_index)
                self.assertEqual(step % EXPECTED_RENDER_SUBSTEPS, 0)
                break
        self.assertIsNotNone(due)
        acknowledged = kernel.step(
            ZERO,
            _context(due.expected_first_physics_step),
            _feedback(kernel, due.expected_first_physics_step, readback=_readback(due)),
        )
        self.assertIsNone(acknowledged.command_batch)
        self.assertFalse(acknowledged.terminal_latched)

    def test_feedback_missing_nonfinite_and_stale_readback_fail_closed(self):
        kernel = self.kernel()
        with self.assertRaises(ResidualEnvContractError):
            PhasePhysicsFeedback(
                sim_step=0,
                sim_time_s=0.0,
                telemetry=_telemetry(kernel),
                effective_servo_actual_error_deg={name: math.nan for name in SERVO_JOINT_NAMES},
                servo_velocity_deg_s={name: 0.0 for name in SERVO_JOINT_NAMES},
            )
        with self.assertRaises(ResidualEnvContractError):
            PhasePhysicsFeedback(
                sim_step=0,
                sim_time_s=0.001,
                telemetry=_telemetry(kernel),
                effective_servo_actual_error_deg={name: 0.0 for name in SERVO_JOINT_NAMES},
                servo_velocity_deg_s={name: 0.0 for name in SERVO_JOINT_NAMES},
            )

    def test_module_import_is_isaac_torch_gym_and_skrl_lazy(self):
        command = (
            "import sys; "
            "import fsm_50mm_recording_derived_v3.fsm50_residual_env as m; "
            "assert not any(n == 'isaaclab' or n.startswith('isaaclab.') for n in sys.modules); "
            "assert 'torch' not in sys.modules; assert 'gymnasium' not in sys.modules; "
            "assert not any(n == 'skrl' or n.startswith('skrl.') for n in sys.modules); "
            "print(m.GYM_ENV_ID)"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPLAY_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), GYM_ENV_ID)

    def test_gym_registration_seam_is_exact_and_idempotent(self):
        class Spec:
            def __init__(self, entry_point, kwargs):
                self.entry_point = entry_point
                self.kwargs = kwargs

        class FakeGym:
            def __init__(self):
                self.specs = {}

            def spec(self, name):
                if name not in self.specs:
                    raise KeyError(name)
                return self.specs[name]

            def register(self, *, id, entry_point, disable_env_checker, kwargs):
                self.assertions = (disable_env_checker, kwargs)
                self.specs[id] = Spec(entry_point, kwargs)

        gym = FakeGym()
        self.assertEqual(register_gym_env(gym_module=gym), GYM_ENV_ID)
        self.assertEqual(register_gym_env(gym_module=gym), GYM_ENV_ID)
        spec = gym.spec(GYM_ENV_ID)
        self.assertTrue(spec.entry_point.endswith(":make_direct_rl_env"))
        self.assertTrue(spec.kwargs["env_cfg_entry_point"].endswith(":build_env_cfg"))


if __name__ == "__main__":
    unittest.main()
