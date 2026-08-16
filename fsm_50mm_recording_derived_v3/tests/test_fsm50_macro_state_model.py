from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from fsm_50mm_recording_derived_v3.fsm50_macro_state_model import (
    LEGACY_57_STATE_CONFIG,
    LEGACY_57_STATE_CONTROL_AUTHORITY,
    LEGACY_57_STATE_COUNT,
    MacroGuardKind,
    MacroStateId,
    PHYSICAL_PHASE_TO_MACRO_STATE,
    build_default_macro_graph,
)


class MacroStateModelTests(unittest.TestCase):
    def test_graph_is_small_physical_and_deterministic(self) -> None:
        graph = build_default_macro_graph()
        self.assertEqual(graph.active_state_count, 11)
        self.assertLessEqual(graph.active_state_count, 12)
        self.assertEqual(graph.sha256, build_default_macro_graph().sha256)
        self.assertIn("LEGACY_SEMANTIC_GRAPH", graph.legacy_graph_role)
        for state_id in graph.active_state_ids:
            self.assertNotIn("STEP", state_id.value)
            self.assertNotIn("SEGMENT", state_id.value)

    def test_57_state_config_is_explicitly_non_authoritative_legacy_input(self) -> None:
        graph = build_default_macro_graph()
        mapping = graph.to_mapping()
        self.assertEqual(mapping["legacy_graph_path"], LEGACY_57_STATE_CONFIG)
        self.assertEqual(mapping["legacy_graph_state_count"], LEGACY_57_STATE_COUNT)
        self.assertEqual(LEGACY_57_STATE_COUNT, 57)
        self.assertIs(LEGACY_57_STATE_CONTROL_AUTHORITY, False)
        self.assertIs(mapping["legacy_graph_control_authority"], False)
        config_path = Path(__file__).resolve().parents[1] / LEGACY_57_STATE_CONFIG
        legacy = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(legacy["metadata"]["graph_role"], "LEGACY_SEMANTIC_GRAPH")
        self.assertEqual(legacy["metadata"]["legacy_state_count"], 57)
        self.assertIs(legacy["metadata"]["control_authority"], False)

    def test_graph_uses_physical_events_not_time_as_completion(self) -> None:
        graph = build_default_macro_graph()
        kinds = {state.completion_guard.kind for state in graph.states}
        self.assertIn(MacroGuardKind.COM_SHIFT_OR_UNLOAD, kinds)
        self.assertIn(MacroGuardKind.LEG_TRAVERSED, kinds)
        self.assertIn(MacroGuardKind.SUPPORT_SETUP, kinds)
        self.assertIn(MacroGuardKind.POSTURE_RECOVERED, kinds)
        for state in graph.states:
            if state.profile_required:
                self.assertTrue(state.completion_guard.profile_must_complete)

    def test_rl_completion_does_not_reintroduce_all_top_gate(self) -> None:
        state = build_default_macro_graph().get(
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE
        )
        self.assertEqual(state.completion_guard.required_top_legs, ("RL",))
        self.assertTrue(state.completion_guard.require_airborne_before_crossing)

    def test_support_setup_is_source_optional_not_a_manufactured_profile(self) -> None:
        state = build_default_macro_graph().get(
            MacroStateId.S7_PRE_RL_SUPPORT_SETUP
        )
        self.assertFalse(state.profile_required)
        self.assertEqual(state.completion_guard.required_support_legs, ("FL", "RR"))

    def test_feedback_acknowledgements_are_optional_and_holds_are_evidence_bounded(self) -> None:
        graph = build_default_macro_graph()
        for state_id in (
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            MacroStateId.S5_PRE_RR_COM_SHIFT,
            MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
            MacroStateId.S9_FINAL_ADVANCE,
        ):
            self.assertFalse(graph.get(state_id).profile_required)
        self.assertEqual(
            {
                state_id: graph.get(state_id).retry_policy.maximum_hold_s
                for state_id in (
                    MacroStateId.S2_FR_TRAVERSE,
                    MacroStateId.S3_FL_TRAVERSE,
                    MacroStateId.S6_RR_TRAVERSE,
                    MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
                )
            },
            {
                MacroStateId.S2_FR_TRAVERSE: 2.25,
                MacroStateId.S3_FL_TRAVERSE: 6.25,
                MacroStateId.S6_RR_TRAVERSE: 3.75,
                MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE: 7.25,
            },
        )
        self.assertTrue(
            all(
                graph.get(state_id).retry_policy.guard_observation_window_basis
                for state_id in (
                    MacroStateId.S2_FR_TRAVERSE,
                    MacroStateId.S3_FL_TRAVERSE,
                    MacroStateId.S6_RR_TRAVERSE,
                    MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
                )
            )
        )

    def test_twenty_canonical_phases_map_to_one_graph(self) -> None:
        expected = {
            "INITIAL_APPROACH",
            "PRE_FR_COM_SHIFT",
            "FR_UNLOAD_AND_LIFT",
            "FR_FACE_CROSS",
            "FR_TOP_PLACE",
            "FL_UNLOAD_AND_LIFT",
            "FL_FACE_CROSS",
            "FL_TOP_PLACE",
            "FRONT_PAIR_ADVANCE",
            "PRE_RR_COM_SHIFT",
            "RR_UNLOAD_AND_LIFT",
            "RR_FACE_CROSS",
            "RR_TOP_PLACE",
            "PRE_RL_SUPPORT_SETUP",
            "PRE_RL_COM_SHIFT",
            "RL_UNLOAD_AND_LIFT",
            "RL_FACE_CROSS",
            "RL_TOP_PLACE",
            "FINAL_ADVANCE",
            "FINAL_POSTURE_RECOVERY",
        }
        self.assertTrue(expected.issubset(PHYSICAL_PHASE_TO_MACRO_STATE))
        self.assertEqual(
            PHYSICAL_PHASE_TO_MACRO_STATE["RR_UNLOAD_AND_LIFT"],
            MacroStateId.S6_RR_TRAVERSE,
        )
        self.assertEqual(
            PHYSICAL_PHASE_TO_MACRO_STATE["RL_UNLOAD_AND_LIFT"],
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
        )


if __name__ == "__main__":
    unittest.main()
