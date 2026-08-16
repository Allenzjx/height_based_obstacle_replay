from __future__ import annotations

import json
import math
import unittest

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.fsm50_direct_command_residual import (
    RESIDUAL_ACTION_DIM,
    RESIDUAL_ACTION_NAMES,
    ZERO_RESIDUAL_ACTION,
    ResidualContractError,
    ResidualPhaseContract,
    ResidualTransformInput,
    ZeroResidualPolicy,
    canonical_mapping_sha256,
    compose_direct_command_residual,
)


SOURCE = "v003_20260805_224517_157723_manual"
STRATEGY = "PRIMARY_PROFILE"
STATE = "S8_RL_COM_SHIFT_AND_TRAVERSE"
SUBPHASE = "RL_FACE_CROSS"


def _nominal_servos(value: float = 0.0) -> dict[str, float]:
    return {name: float(value) for name in SERVO_JOINT_NAMES}


def _nominal_wheels() -> dict[str, float]:
    return {
        "front_left_ankle": 0.5,
        "front_right_ankle": -0.5,
        "rear_left_ankle": 0.0,
        "rear_right_ankle": 0.25,
    }


def _contract(
    *,
    enabled: tuple[int, ...] = tuple(range(RESIDUAL_ACTION_DIM)),
    lower: dict[int, float] | None = None,
    upper: dict[int, float] | None = None,
    rates: dict[int, float] | None = None,
    action_names: tuple[str, ...] = RESIDUAL_ACTION_NAMES,
) -> ResidualPhaseContract:
    lower = dict(lower or {})
    upper = dict(upper or {})
    rates = dict(rates or {})
    enabled_set = set(enabled)
    return ResidualPhaseContract(
        source_version=SOURCE,
        profile_strategy=STRATEGY,
        macro_state=STATE,
        subphase=SUBPHASE,
        action_names=action_names,
        enabled_mask=tuple(index in enabled_set for index in range(RESIDUAL_ACTION_DIM)),
        residual_min_command_units=tuple(
            lower.get(index, -2.0) if index in enabled_set else 0.0
            for index in range(RESIDUAL_ACTION_DIM)
        ),
        residual_max_command_units=tuple(
            upper.get(index, 2.0) if index in enabled_set else 0.0
            for index in range(RESIDUAL_ACTION_DIM)
        ),
        maximum_rate_command_units_per_s=tuple(
            rates.get(index, 100.0) if index in enabled_set else 0.0
            for index in range(RESIDUAL_ACTION_DIM)
        ),
    )


def _input(
    *,
    action: tuple[float, ...] = ZERO_RESIDUAL_ACTION,
    previous: tuple[float, ...] = ZERO_RESIDUAL_ACTION,
    servos: dict[str, float] | None = None,
    wheels: dict[str, float] | None = None,
    dt: float = 1.0,
    servo_safety: dict[str, tuple[float, float]] | None = None,
    wheel_limit: float = 3.0,
    latched: dict[str, float] | None = None,
    force_zero_residual: bool = False,
    force_zero_wheels: bool = False,
    state: str = STATE,
) -> ResidualTransformInput:
    return ResidualTransformInput(
        source_version=SOURCE,
        profile_strategy=STRATEGY,
        macro_state=state,
        subphase=SUBPHASE,
        nominal_servo_targets_deg=servos or _nominal_servos(),
        nominal_wheel_targets_rad_s=wheels or _nominal_wheels(),
        normalized_action=action,
        previous_applied_residual=previous,
        decision_dt_s=dt,
        maximum_wheel_speed_rad_s=wheel_limit,
        servo_safe_command_limits_deg=servo_safety,
        latched_servo_residual_deg=latched,
        force_zero_residual=force_zero_residual,
        force_zero_wheels=force_zero_wheels,
    )


def _action(**values: float) -> tuple[float, ...]:
    unknown = set(values) - set(RESIDUAL_ACTION_NAMES)
    if unknown:
        raise AssertionError(f"unknown test action names: {sorted(unknown)}")
    return tuple(float(values.get(name, 0.0)) for name in RESIDUAL_ACTION_NAMES)


class DirectCommandResidualTests(unittest.TestCase):
    def test_action_vocabulary_is_the_exact_canonical_8_plus_4_order(self) -> None:
        self.assertEqual(RESIDUAL_ACTION_DIM, 12)
        self.assertEqual(
            RESIDUAL_ACTION_NAMES,
            tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES),
        )
        with self.assertRaisesRegex(ResidualContractError, "canonical 8-servo"):
            _contract(action_names=tuple(reversed(RESIDUAL_ACTION_NAMES)))

    def test_contract_mapping_round_trip_and_digest_are_deterministic(self) -> None:
        contract = _contract(enabled=(0, 8), lower={0: -1.25, 8: -0.1})
        mapping = contract.to_mapping()
        rebuilt = ResidualPhaseContract.from_mapping(
            json.loads(json.dumps(mapping))
        )
        self.assertEqual(rebuilt, contract)
        self.assertEqual(rebuilt.sha256, contract.sha256)
        self.assertEqual(contract.sha256, canonical_mapping_sha256(mapping))
        self.assertRegex(contract.sha256, r"^[0-9a-f]{64}$")

        extra = dict(mapping)
        extra["unexpected"] = True
        with self.assertRaisesRegex(ResidualContractError, "keys are not exact"):
            ResidualPhaseContract.from_mapping(extra)

        malformed = dict(mapping)
        malformed["enabled_mask"] = None
        with self.assertRaisesRegex(ResidualContractError, "must be a sequence"):
            ResidualPhaseContract.from_mapping(malformed)

    def test_contract_is_fail_closed_for_disabled_or_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ResidualContractError, "disabled residual channel"):
            ResidualPhaseContract(
                source_version=SOURCE,
                profile_strategy=STRATEGY,
                macro_state=STATE,
                subphase=SUBPHASE,
                enabled_mask=(False,) * RESIDUAL_ACTION_DIM,
                residual_min_command_units=(-1.0,) + (0.0,) * 11,
                residual_max_command_units=(0.0,) * RESIDUAL_ACTION_DIM,
                maximum_rate_command_units_per_s=(0.0,) * RESIDUAL_ACTION_DIM,
            )
        with self.assertRaisesRegex(ResidualContractError, "min <= 0 <= max"):
            _contract(enabled=(0,), lower={0: 0.1})

    def test_transform_input_rejects_wrong_shape_nonfinite_and_unsafe_nominal(self) -> None:
        with self.assertRaisesRegex(ResidualContractError, "exactly 12"):
            _input(action=(0.0,) * 11)
        bad_action = list(ZERO_RESIDUAL_ACTION)
        bad_action[3] = math.nan
        with self.assertRaisesRegex(ResidualContractError, "finite numeric"):
            _input(action=tuple(bad_action))
        servos = _nominal_servos()
        servos["front_left_hip"] = 136.0
        with self.assertRaisesRegex(ResidualContractError, "outside hard safety"):
            _input(servos=servos)
        wheels = _nominal_wheels()
        wheels["front_left_ankle"] = 3.1
        with self.assertRaisesRegex(ResidualContractError, "outside hard safety"):
            _input(wheels=wheels, wheel_limit=3.0)

    def test_exact_zero_short_circuit_is_nominal_identity_without_clip_flags(self) -> None:
        servos = {
            name: float(index) + 0.125
            for index, name in enumerate(SERVO_JOINT_NAMES)
        }
        wheels = _nominal_wheels()
        previous = tuple(float(index + 1) for index in range(RESIDUAL_ACTION_DIM))
        result = compose_direct_command_residual(
            _input(
                action=ZERO_RESIDUAL_ACTION,
                previous=previous,
                servos=servos,
                wheels=wheels,
            ),
            _contract(enabled=(0, 8)),
        )
        self.assertIs(result.zero_identity, True)
        self.assertEqual(result.applied_residual, ZERO_RESIDUAL_ACTION)
        self.assertEqual(result.applied_servo_targets_deg, servos)
        self.assertEqual(result.applied_wheel_targets_rad_s, wheels)
        self.assertTrue(all(not reasons for reasons in result.clip_reasons_by_action))
        self.assertEqual(result.sha256, canonical_mapping_sha256(result.to_mapping()))
        self.assertEqual(result.sha256, result.sha256)

    def test_exact_zero_with_nonzero_completion_latch_uses_general_composer(self) -> None:
        first, second = SERVO_JOINT_NAMES[:2]
        contract = _contract(
            enabled=(0, 1),
            lower={0: -2.0, 1: -2.0},
            upper={0: 2.0, 1: 2.0},
            rates={0: 2.0, 1: 2.0},
        )
        previous = _action(**{first: 1.0, second: 1.0})
        result = compose_direct_command_residual(
            _input(
                action=ZERO_RESIDUAL_ACTION,
                previous=previous,
                dt=0.25,
                latched={first: 0.75},
            ),
            contract,
        )
        self.assertIs(result.zero_identity, False)
        self.assertEqual(result.raw_normalized_action, ZERO_RESIDUAL_ACTION)
        self.assertEqual(result.applied_residual[0], 0.75)
        self.assertEqual(result.applied_residual[1], 0.5)
        self.assertIn("ACTIVE_COMPLETION_LATCH", result.clip_reasons_by_action[0])
        self.assertIn("RESIDUAL_RATE_LIMIT", result.clip_reasons_by_action[1])
        self.assertEqual(result.applied_servo_targets_deg[first], 0.75)
        self.assertEqual(result.applied_servo_targets_deg[second], 0.5)

    def test_phase_mask_hard_zero_overrides_prior_residual_without_slew_tail(self) -> None:
        action = _action(front_left_hip=1.0, front_left_knee=1.0)
        previous = _action(front_left_hip=1.0, front_left_knee=1.5)
        contract = _contract(
            enabled=(0,),
            lower={0: -2.0},
            upper={0: 2.0},
            rates={0: 100.0},
        )
        result = compose_direct_command_residual(
            _input(action=action, previous=previous), contract
        )
        self.assertEqual(result.applied_residual[0], 2.0)
        self.assertEqual(result.applied_residual[1], 0.0)
        self.assertIn("PHASE_MASK_HARD_ZERO", result.clip_reasons_by_action[1])

    def test_asymmetric_bounds_action_clip_and_residual_slew_are_applied_in_order(self) -> None:
        contract = _contract(
            enabled=(0,),
            lower={0: -4.0},
            upper={0: 10.0},
            rates={0: 2.0},
        )
        positive = compose_direct_command_residual(
            _input(
                action=_action(front_left_hip=2.0),
                previous=ZERO_RESIDUAL_ACTION,
                dt=0.25,
            ),
            contract,
        )
        self.assertEqual(positive.clipped_normalized_action[0], 1.0)
        self.assertEqual(positive.requested_residual[0], 10.0)
        self.assertEqual(positive.rate_limited_residual[0], 0.5)
        self.assertEqual(positive.applied_residual[0], 0.5)
        self.assertEqual(
            positive.clip_reasons_by_action[0],
            ("NORMALIZED_ACTION_CLIP", "RESIDUAL_RATE_LIMIT"),
        )

        negative = compose_direct_command_residual(
            _input(
                action=_action(front_left_hip=-0.5),
                previous=ZERO_RESIDUAL_ACTION,
                dt=1.0,
            ),
            contract,
        )
        self.assertEqual(negative.requested_residual[0], -2.0)
        self.assertEqual(negative.applied_residual[0], -2.0)

    def test_servo_residual_is_clipped_to_command_and_runtime_safe_intersection(self) -> None:
        safety = {
            name: (-100.0, 100.0) for name in SERVO_JOINT_NAMES
        }
        servos = _nominal_servos()
        servos["front_left_hip"] = 99.5
        result = compose_direct_command_residual(
            _input(
                action=_action(front_left_hip=1.0),
                servos=servos,
                servo_safety=safety,
            ),
            _contract(enabled=(0,), upper={0: 5.0}),
        )
        self.assertEqual(result.applied_servo_targets_deg["front_left_hip"], 100.0)
        self.assertEqual(result.applied_residual[0], 0.5)
        self.assertIn("SERVO_HARD_SAFETY_CLIP", result.clip_reasons_by_action[0])

    def test_nominal_zero_wheel_cannot_be_created_by_a_residual(self) -> None:
        index = RESIDUAL_ACTION_NAMES.index("rear_left_ankle")
        result = compose_direct_command_residual(
            _input(action=_action(rear_left_ankle=1.0)),
            _contract(enabled=(index,), upper={index: 2.0}),
        )
        self.assertEqual(result.applied_wheel_targets_rad_s["rear_left_ankle"], 0.0)
        self.assertEqual(result.applied_residual[index], 0.0)
        self.assertIn(
            "NOMINAL_ZERO_WHEEL_HARD_ZERO",
            result.clip_reasons_by_action[index],
        )

    def test_wheel_residual_preserves_nominal_sign_and_hard_speed(self) -> None:
        positive_index = RESIDUAL_ACTION_NAMES.index("front_left_ankle")
        negative_index = RESIDUAL_ACTION_NAMES.index("front_right_ankle")
        sign_result = compose_direct_command_residual(
            _input(
                action=_action(
                    front_left_ankle=-1.0,
                    front_right_ankle=1.0,
                )
            ),
            _contract(
                enabled=(positive_index, negative_index),
                lower={positive_index: -2.0, negative_index: -2.0},
                upper={positive_index: 2.0, negative_index: 2.0},
            ),
        )
        self.assertEqual(
            sign_result.applied_wheel_targets_rad_s["front_left_ankle"], 0.0
        )
        self.assertEqual(
            sign_result.applied_wheel_targets_rad_s["front_right_ankle"], 0.0
        )
        self.assertIn(
            "WHEEL_SIGN_PRESERVATION",
            sign_result.clip_reasons_by_action[positive_index],
        )
        self.assertIn(
            "WHEEL_SIGN_PRESERVATION",
            sign_result.clip_reasons_by_action[negative_index],
        )

        wheels = _nominal_wheels()
        wheels["front_left_ankle"] = 0.9
        speed_result = compose_direct_command_residual(
            _input(
                action=_action(front_left_ankle=1.0),
                wheels=wheels,
                wheel_limit=1.0,
            ),
            _contract(enabled=(positive_index,), upper={positive_index: 2.0}),
        )
        self.assertEqual(
            speed_result.applied_wheel_targets_rad_s["front_left_ankle"], 1.0
        )
        self.assertAlmostEqual(speed_result.applied_residual[positive_index], 0.1)
        self.assertIn(
            "WHEEL_HARD_SPEED_CLIP",
            speed_result.clip_reasons_by_action[positive_index],
        )

    def test_active_completion_servo_latch_is_exact_but_still_hard_safe(self) -> None:
        result = compose_direct_command_residual(
            _input(
                action=_action(front_left_hip=-1.0),
                latched={"front_left_hip": 1.25},
            ),
            _contract(enabled=(0,), lower={0: -2.0}, upper={0: 2.0}),
        )
        self.assertEqual(result.applied_residual[0], 1.25)
        self.assertIn("ACTIVE_COMPLETION_LATCH", result.clip_reasons_by_action[0])

        safety = {
            name: (-100.0, 100.0) for name in SERVO_JOINT_NAMES
        }
        servos = _nominal_servos()
        servos["front_left_hip"] = 99.5
        clipped = compose_direct_command_residual(
            _input(
                action=_action(front_left_hip=-1.0),
                servos=servos,
                servo_safety=safety,
                latched={"front_left_hip": 1.25},
            ),
            _contract(enabled=(0,)),
        )
        self.assertEqual(clipped.applied_residual[0], 0.5)
        self.assertIn("SERVO_HARD_SAFETY_CLIP", clipped.clip_reasons_by_action[0])

    def test_forced_zero_contract_requires_nominal_zero_wheels(self) -> None:
        with self.assertRaisesRegex(
            ResidualContractError, "requires an exact zero nominal wheel map"
        ):
            _input(force_zero_wheels=True)

        wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        result = compose_direct_command_residual(
            _input(
                action=tuple(1.0 for _ in range(RESIDUAL_ACTION_DIM)),
                wheels=wheels,
                force_zero_residual=True,
                force_zero_wheels=True,
            ),
            _contract(),
        )
        self.assertEqual(result.applied_servo_targets_deg, _nominal_servos())
        self.assertEqual(result.applied_wheel_targets_rad_s, wheels)
        self.assertEqual(result.applied_residual, ZERO_RESIDUAL_ACTION)
        self.assertIs(result.zero_identity, False)

    def test_context_mismatch_is_rejected_before_composition(self) -> None:
        with self.assertRaisesRegex(ResidualContractError, "context mismatch"):
            compose_direct_command_residual(
                _input(
                    action=_action(front_left_hip=1.0),
                    state="S10_POSTURE_RECOVERY",
                ),
                _contract(enabled=(0,)),
            )

    def test_zero_policy_is_deterministic_and_requires_a_mapping_observation(self) -> None:
        policy = ZeroResidualPolicy()
        self.assertEqual(policy.act({"finite": True}), ZERO_RESIDUAL_ACTION)
        self.assertEqual(policy.policy_sha256, policy.policy_sha256)
        self.assertRegex(policy.policy_sha256, r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(ResidualContractError, "must be a mapping"):
            policy.act(())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
