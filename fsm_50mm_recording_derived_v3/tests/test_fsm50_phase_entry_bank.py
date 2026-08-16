from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from fsm_50mm_recording_derived_v3 import fsm50_phase_entry_bank as bank


REPLAY_ROOT = Path(__file__).resolve().parents[2]


EXPECTED_ENTRY_STEPS = {
    ("v003_20260805_224517_157723_manual", "S5_PRE_RR_COM_SHIFT"): (7181, 7188, 41),
    ("v003_20260805_224517_157723_manual", "S7_PRE_RL_SUPPORT_SETUP"): (8460, 8460, 57),
    ("v003_20260805_224517_157723_manual", "S8_RL_COM_SHIFT_AND_TRAVERSE"): (8461, 8468, 57),
    ("v003_20260805_224517_157723_manual", "S10_POSTURE_RECOVERY"): (10764, 10764, 104),
    ("v008_20260806_211408_578700_manual", "S5_PRE_RR_COM_SHIFT"): (7180, 7180, 41),
    ("v008_20260806_211408_578700_manual", "S7_PRE_RL_SUPPORT_SETUP"): (8428, 8428, 75),
    ("v008_20260806_211408_578700_manual", "S8_RL_COM_SHIFT_AND_TRAVERSE"): (8429, 8436, 75),
    ("v008_20260806_211408_578700_manual", "S10_POSTURE_RECOVERY"): (11405, 11412, 118),
    ("v009_20260806_215232_433234_manual", "S5_PRE_RR_COM_SHIFT"): (6900, 6900, 54),
    ("v009_20260806_215232_433234_manual", "S7_PRE_RL_SUPPORT_SETUP"): (7572, 7572, 91),
    ("v009_20260806_215232_433234_manual", "S8_RL_COM_SHIFT_AND_TRAVERSE"): (7732, 7732, 93),
    ("v009_20260806_215232_433234_manual", "S10_POSTURE_RECOVERY"): (10869, 10876, 131),
}


def _reseal_entry(mapping: dict[str, object], index: int) -> None:
    payload = mapping["payload"]
    assert isinstance(payload, dict)
    entries = payload["entries"]
    assert isinstance(entries, list)
    entry = entries[index]
    assert isinstance(entry, dict)
    entry.pop("entry_sha256", None)
    entry["entry_sha256"] = bank.canonical_sha256(entry)
    mapping["bank_sha256"] = bank.canonical_sha256(payload)


def _reseal_binding(mapping: dict[str, object], index: int) -> None:
    payload = mapping["payload"]
    assert isinstance(payload, dict)
    bindings = payload["source_bindings"]
    assert isinstance(bindings, list)
    binding = bindings[index]
    assert isinstance(binding, dict)
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = bank.canonical_sha256(binding)
    mapping["bank_sha256"] = bank.canonical_sha256(payload)


class CurrentPhaseEntryBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping = bank.build_current_phase_entry_bank_mapping()

    def fresh(self) -> dict[str, object]:
        return copy.deepcopy(self.mapping)

    def test_exact_three_by_four_entry_inventory_and_real_steps(self) -> None:
        payload = self.mapping["payload"]
        self.assertEqual(payload["source_versions"], list(bank.CURRENT_SOURCE_VERSIONS))
        self.assertEqual(payload["phase_states"], list(bank.TARGET_STATES))
        entries = payload["entries"]
        self.assertEqual(len(entries), 12)
        self.assertEqual(
            [(entry["source_version"], entry["phase_state"]) for entry in entries],
            [
                (source, state)
                for source in bank.CURRENT_SOURCE_VERSIONS
                for state in bank.TARGET_STATES
            ],
        )
        for entry in entries:
            key = (entry["source_version"], entry["phase_state"])
            expected_entry, expected_snapshot, expected_action = EXPECTED_ENTRY_STEPS[key]
            self.assertEqual(entry["entry_sim_step"], expected_entry)
            self.assertEqual(entry["snapshot_sim_step"], expected_snapshot)
            self.assertEqual(entry["source_cursor"]["source_action_index"], expected_action)
            self.assertGreaterEqual(entry["snapshot_sim_step"], entry["entry_sim_step"])
            self.assertLessEqual(entry["telemetry_latency_physics_steps"], 8)
            expected_relation = (
                "SAME_STEP_PRE_DECISION_OBSERVATION"
                if entry["snapshot_sim_step"] == entry["entry_sim_step"]
                else "FIRST_POST_ENTRY_TELEMETRY_OBSERVATION"
            )
            self.assertEqual(entry["telemetry_relation_to_entry"], expected_relation)

    def test_s7_coalescing_is_honest_and_profile_free_where_no_row_exists(self) -> None:
        entries = {
            (entry["source_version"], entry["phase_state"]): entry
            for entry in self.mapping["payload"]["entries"]
        }
        for source in bank.CURRENT_SOURCE_VERSIONS[:2]:
            entry = entries[(source, "S7_PRE_RL_SUPPORT_SETUP")]
            self.assertEqual(entry["entry_profile"]["profile_id"], "")
            self.assertEqual(entry["entry_profile"]["profile_strategy"], "")
            self.assertIs(entry["entry_profile"]["profile_free"], True)
            self.assertEqual(entry["observed_macro_state"], "S6_RR_TRAVERSE")
            self.assertEqual(entry["coalesced_successor_state"], "S8_RL_COM_SHIFT_AND_TRAVERSE")
            self.assertEqual(entry["source_cursor"]["owner_state"], "S8_RL_COM_SHIFT_AND_TRAVERSE")
        v009 = entries[(bank.CURRENT_SOURCE_VERSIONS[2], "S7_PRE_RL_SUPPORT_SETUP")]
        self.assertEqual(v009["observed_macro_state"], "S6_RR_TRAVERSE")
        self.assertEqual(v009["coalesced_successor_state"], "")
        self.assertIsNone(v009["coalesced_transition_index"])
        self.assertIn("S7_PRE_RL_SUPPORT_SETUP", v009["entry_profile"]["profile_id"])
        self.assertIs(v009["entry_profile"]["profile_free"], False)

    def test_snapshot_has_complete_finite_reset_and_nominal_state(self) -> None:
        payload = self.mapping["payload"]
        self.assertEqual(len(payload["joint_order"]["all"]), 12)
        self.assertEqual(len(payload["joint_order"]["servo"]), 8)
        self.assertEqual(len(payload["joint_order"]["wheel"]), 4)
        obstacle_sha = payload["obstacle_identity"]["identity_sha256"]
        for entry in payload["entries"]:
            vectors = (
                (entry["root_position_m"], 3),
                (entry["root_orientation_wxyz"], 4),
                (entry["root_linear_velocity_m_s"], 3),
                (entry["root_angular_velocity_rad_s"], 3),
                (entry["joint_q_rad"], 12),
                (entry["joint_qd_rad_s"], 12),
                (entry["nominal_servo_targets_deg"], 8),
                (entry["nominal_wheel_targets_rad_s"], 4),
                (entry["source_cursor"]["servo_targets_deg"], 8),
                (entry["source_cursor"]["wheel_targets_rad_s"], 4),
            )
            for values, size in vectors:
                self.assertEqual(len(values), size)
                self.assertTrue(all(math.isfinite(value) for value in values))
            norm = math.sqrt(sum(value * value for value in entry["root_orientation_wxyz"]))
            self.assertAlmostEqual(norm, 1.0, places=4)
            self.assertEqual(entry["obstacle_identity_sha256"], obstacle_sha)
            self.assertEqual(
                entry["source_cursor"]["source_action_index"],
                entry["source_cursor"]["source_segment_index"],
            )

    def test_binding_covers_exact_required_sha_chain_and_transition_evidence(self) -> None:
        expected_artifacts = {
            "manifest",
            "verdict",
            "request",
            "result",
            "telemetry",
            "source",
            "completion",
            "transition",
        }
        for binding in self.mapping["payload"]["source_bindings"]:
            self.assertEqual(set(binding["artifacts"]), expected_artifacts)
            self.assertEqual(binding["graph_sha256"], bank.GRAPH_SHA256)
            self.assertEqual(binding["profile_library_sha256"], bank.PROFILE_LIBRARY_SHA256)
            self.assertIn("coalesced_r4", binding["run_relative_path"])
            self.assertNotIn("v010", binding["run_relative_path"])
            self.assertNotIn("ppo", binding["run_relative_path"].lower())
            for artifact in binding["artifacts"].values():
                self.assertEqual(len(artifact["sha256"]), 64)

    def test_generated_config_is_exact_and_full_artifact_validation_passes(self) -> None:
        disk = json.loads(bank.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(disk, self.mapping)
        self.assertEqual(
            bank.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
            bank.render_current_phase_entry_bank_json(),
        )
        loaded = bank.load_phase_entry_bank()
        self.assertEqual(loaded.bank_sha256, self.mapping["bank_sha256"])
        self.assertEqual(len(loaded.entries), 12)

    def test_build_is_deterministic_and_canonical(self) -> None:
        rebuilt = bank.build_current_phase_entry_bank_mapping()
        self.assertEqual(rebuilt, self.mapping)
        self.assertEqual(
            rebuilt["bank_sha256"],
            bank.canonical_sha256(rebuilt["payload"]),
        )

    def test_validated_bank_is_deeply_immutable(self) -> None:
        loaded = bank.validate_phase_entry_bank_mapping(self.mapping, verify_artifacts=False)
        with self.assertRaises(TypeError):
            loaded.payload["bank_id"] = "changed"
        with self.assertRaises(TypeError):
            loaded.entries[0]["phase_state"] = "changed"
        with self.assertRaises(TypeError):
            loaded.entries[0]["joint_q_rad"][0] = 0.0

    def test_hash_tamper_is_rejected(self) -> None:
        tampered = self.fresh()
        tampered["payload"]["entries"][0]["joint_q_rad"][0] += 0.01
        with self.assertRaisesRegex(bank.PhaseEntryBankError, "entry_sha256|bank_sha256"):
            bank.validate_phase_entry_bank_mapping(tampered, verify_artifacts=False)

        tampered = self.fresh()
        tampered["payload"]["source_bindings"][0]["artifacts"]["telemetry"]["sha256"] = "0" * 64
        _reseal_binding(tampered, 0)
        with self.assertRaises(bank.PhaseEntryBankError):
            bank.validate_phase_entry_bank_mapping(tampered, verify_artifacts=False)

    def test_v010_and_old_paths_are_rejected_even_when_resealed(self) -> None:
        for replacement in (
            "runs/cross_version_macro_fsm_completion_aware_coalesced_r4/"
            "v010_20260806_220745_363972_manual/trials/"
            "20260815T115036_921727Z_cross_version_00_246f073b36ad",
            "runs/fsm_residual_ppo/legacy/checkpoint",
        ):
            tampered = self.fresh()
            binding = tampered["payload"]["source_bindings"][0]
            binding["run_relative_path"] = replacement
            _reseal_binding(tampered, 0)
            with self.assertRaisesRegex(bank.PhaseEntryBankError, "allowlisted current r4"):
                bank.validate_phase_entry_bank_mapping(tampered, verify_artifacts=False)

    def test_artifact_path_traversal_is_rejected(self) -> None:
        tampered = self.fresh()
        binding = tampered["payload"]["source_bindings"][0]
        binding["artifacts"]["telemetry"]["path"] = "../minimal_macro_telemetry.jsonl"
        _reseal_binding(tampered, 0)
        with self.assertRaises(bank.PhaseEntryBankError):
            bank.validate_phase_entry_bank_mapping(tampered, verify_artifacts=False)

    def test_state_shape_and_nonfinite_values_are_rejected_after_reseal(self) -> None:
        bad_state = self.fresh()
        entry = bad_state["payload"]["entries"][0]
        entry["phase_state"] = "S6_RR_TRAVERSE"
        entry["entry_id"] = f"{entry['source_version']}:S6_RR_TRAVERSE"
        _reseal_entry(bad_state, 0)
        with self.assertRaises(bank.PhaseEntryBankError):
            bank.validate_phase_entry_bank_mapping(bad_state, verify_artifacts=False)

        bad_shape = self.fresh()
        bad_shape["payload"]["entries"][0]["joint_q_rad"].pop()
        _reseal_entry(bad_shape, 0)
        with self.assertRaisesRegex(bank.PhaseEntryBankError, "exactly 12"):
            bank.validate_phase_entry_bank_mapping(bad_shape, verify_artifacts=False)

        nonfinite = self.fresh()
        nonfinite["payload"]["entries"][0]["joint_qd_rad_s"][0] = float("nan")
        with self.assertRaisesRegex(bank.PhaseEntryBankError, "finite"):
            bank.validate_phase_entry_bank_mapping(nonfinite, verify_artifacts=False)

    def test_duplicate_source_state_pair_is_rejected(self) -> None:
        tampered = self.fresh()
        duplicate = copy.deepcopy(tampered["payload"]["entries"][0])
        tampered["payload"]["entries"][-1] = duplicate
        tampered["bank_sha256"] = bank.canonical_sha256(tampered["payload"])
        with self.assertRaisesRegex(bank.PhaseEntryBankError, "duplicate phase entry"):
            bank.validate_phase_entry_bank_mapping(tampered, verify_artifacts=False)

    def test_module_does_not_import_or_deserialize_private_controller_state(self) -> None:
        source = Path(bank.__file__).read_text(encoding="utf-8")
        self.assertNotIn("fsm50_macro_controller", source)
        self.assertNotIn("pickle", source)
        self.assertNotIn("controller.__dict__", source)

    def test_cli_is_pure_stdout_and_matches_config(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPLAY_ROOT)
        completed = subprocess.run(
            [sys.executable, "-m", "fsm_50mm_recording_derived_v3.fsm50_phase_entry_bank", "--print-json"],
            cwd=REPLAY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(completed.stdout, bank.render_current_phase_entry_bank_json())


if __name__ == "__main__":
    unittest.main()
