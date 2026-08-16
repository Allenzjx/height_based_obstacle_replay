from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from fsm_50mm_recording_derived_v3.fsm50_macro_state_model import (
    MacroStateId,
    MacroSubphase,
)
from fsm_50mm_recording_derived_v3.fsm50_motion_profiles import (
    CANONICAL_PHASE_SOURCE_STATE,
    CANONICAL_SEGMENT_OWNERSHIP_RANGES,
    DEFAULT_PRIMARY_VERSION,
    MotionKeyframe,
    PhaseSpan,
    PhaseWindow,
    RecordingSegmentOwnership,
    RecordingProfileSource,
    build_profile_library,
    discover_successful_gate_a_sources,
    load_phase_spans,
    load_recording_phase_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "fsm_50mm_recording_derived_v3"


class MotionProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = discover_successful_gate_a_sources(PROJECT_ROOT)
        cls.library = build_profile_library(PROJECT_ROOT)

    def test_only_four_gate_a_successes_are_profile_sources(self) -> None:
        self.assertEqual(
            [source.source_version.split("_", 1)[0] for source in self.sources],
            ["v003", "v008", "v009", "v010"],
        )
        self.assertTrue(all(source.plan_sha256 for source in self.sources))
        self.assertTrue(all(source.video_sha256 for source in self.sources))

    def test_real_v003_merged_s1_preserves_global_segment_order(self) -> None:
        profile = self.library.get(
            DEFAULT_PRIMARY_VERSION,
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            strategy="PRIMARY_PROFILE",
        )
        segments = [frame.source_segment_index for frame in profile.keyframes]
        self.assertEqual(segments, sorted(segments))
        self.assertEqual((min(segments), max(segments)), (0, 6))
        self.assertEqual(profile.source_segment_range, (0, 6))
        self.assertIn("servo rear_left_hip 19.6", profile.source_commands)
        self.assertIn("wheel all 0.300", profile.source_commands)
        self.assertEqual(
            tuple(frame.sequence_index for frame in profile.keyframes),
            tuple(range(len(profile.keyframes))),
        )

    def test_real_profiles_keep_full_targets_and_atomic_concurrency(self) -> None:
        concurrent = []
        for profile in self.library.profiles:
            for frame in profile.keyframes:
                self.assertEqual(set(frame.servo_targets_deg), set(SERVO_JOINT_NAMES))
                self.assertEqual(set(frame.wheel_targets_rad_s), set(WHEEL_JOINT_NAMES))
                if frame.atomic_concurrent:
                    concurrent.append(frame)
        self.assertTrue(concurrent)
        frame = concurrent[0]
        self.assertTrue(any(command.startswith("servo ") for command in frame.commands))
        self.assertTrue(any(abs(value) > 0.0 for value in frame.wheel_targets_rad_s.values()))

    def test_full_completion_segments_are_exactly_bound_for_all_four_sources(self) -> None:
        expected_counts = {"v003": 112, "v008": 119, "v009": 132, "v010": 142}
        legacy_counts: dict[str, int] = {}
        for source in self.sources:
            prefix = source.source_version.split("_", 1)[0]
            bindings = [
                binding
                for profile in self.library.profiles
                if profile.source_version == source.source_version
                for binding in profile.segment_bindings
            ]
            with self.subTest(source=source.source_version):
                self.assertEqual(len(bindings), expected_counts[prefix])
                self.assertEqual(
                    [binding.segment_index for binding in bindings],
                    list(range(expected_counts[prefix])),
                )
                self.assertTrue(source.accepted_steps_path.is_file())
                self.assertTrue(source.worker_request_path.is_file())
                self.assertTrue(source.full_plan_payload_sha256)
                self.assertTrue(
                    all(
                        binding.source_plan_payload_sha256
                        == source.full_plan_payload_sha256
                        and binding.accepted_steps_sha256
                        == source.accepted_steps_sha256
                        and binding.completion_spec.segment_index
                        == binding.segment_index
                        for binding in bindings
                    )
                )
                legacy_counts[prefix] = sum(
                    binding.completion_spec.legacy_missing_endpoint
                    for binding in bindings
                )
        self.assertEqual(legacy_counts, {"v003": 52, "v008": 0, "v009": 0, "v010": 0})

    def test_full_plan_payload_and_library_sha_are_deterministic_across_rebuilds(self) -> None:
        rebuilt = build_profile_library(PROJECT_ROOT)
        self.assertEqual(rebuilt.sha256, self.library.sha256)
        self.assertEqual(
            [source.full_plan_payload_sha256 for source in rebuilt.successful_sources],
            [source.full_plan_payload_sha256 for source in self.library.successful_sources],
        )
        self.assertEqual(
            [profile.sha256 for profile in rebuilt.profiles],
            [profile.sha256 for profile in self.library.profiles],
        )

    def test_source_hash_tamper_is_rejected_at_construction(self) -> None:
        source = self.sources[0]
        with self.assertRaisesRegex(ValueError, "accepted_steps SHA mismatch"):
            replace(source, accepted_steps_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "worker request SHA mismatch"):
            replace(source, worker_request_sha256="0" * 64)

    def test_all_profiles_form_one_exactly_owned_ordered_action_stream_per_source(self) -> None:
        self.assertEqual(len(self.library.profiles), 33)
        self.assertEqual(
            {
                source.source_version: len(
                    [
                        profile
                        for profile in self.library.profiles
                        if profile.source_version == source.source_version
                    ]
                )
                for source in self.sources
            },
            {
                next(source.source_version for source in self.sources if source.source_version.startswith("v003_")): 7,
                next(source.source_version for source in self.sources if source.source_version.startswith("v008_")): 8,
                next(source.source_version for source in self.sources if source.source_version.startswith("v009_")): 9,
                next(source.source_version for source in self.sources if source.source_version.startswith("v010_")): 9,
            },
        )
        for source in self.sources:
            owned = sorted(
                (
                    item
                    for item in self.library.segment_ownership
                    if item.source_version == source.source_version
                ),
                key=lambda item: item.first_segment,
            )
            emitted = []
            identities: list[tuple[object, ...]] = []
            for item in owned:
                profile = self.library.profiles_for_state(
                    source.source_version, item.state_id
                )[0]
                self.assertEqual(
                    profile.source_segment_range,
                    (item.first_segment, item.last_segment),
                )
                emitted.extend(profile.keyframes)
                identities.extend(
                    (
                        profile.source_version,
                        frame.source_segment_index,
                        frame.dispatch_kind,
                        frame.commands,
                        frame.source_event_indices,
                    )
                    for frame in profile.keyframes
                )
            with self.subTest(source=source.source_version):
                self.assertEqual(len(identities), len(set(identities)))
                segment_starts = [
                    frame.source_segment_index
                    for frame in emitted
                    if frame.dispatch_kind == "segment_start"
                ]
                self.assertEqual(segment_starts, sorted(segment_starts))
                self.assertEqual(len(segment_starts), len(set(segment_starts)))
                with source.plan_path.open("r", encoding="utf-8") as stream:
                    segment_count = len(json.load(stream)["segments"])
                self.assertEqual(segment_starts, list(range(segment_count)))
                completion_stops = [
                    (
                        frame.source_segment_index,
                        frame.dispatch_kind,
                        frame.source_event_indices,
                    )
                    for frame in emitted
                    if frame.dispatch_kind == "wheel_channel_completion_stop"
                ]
                self.assertEqual(len(completion_stops), len(set(completion_stops)))

    def test_canonical_ownership_table_is_sha_bound_and_preserves_source_labels(self) -> None:
        self.assertEqual(
            {
                source.source_version: tuple(
                    (item.state_id, item.first_segment, item.last_segment)
                    for item in self.library.segment_ownership
                    if item.source_version == source.source_version
                )
                for source in self.sources
            },
            dict(CANONICAL_SEGMENT_OWNERSHIP_RANGES),
        )
        source_sha = {
            source.source_version: source.plan_sha256 for source in self.sources
        }
        self.assertTrue(
            all(
                item.source_plan_sha256 == source_sha[item.source_version]
                and item.evidence_basis
                for item in self.library.segment_ownership
            )
        )
        v009 = next(
            source.source_version
            for source in self.sources
            if source.source_version.startswith("v009_")
        )
        s7 = self.library.get(
            v009,
            MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
            strategy="ALTERNATE_PROFILE_1",
        )
        self.assertEqual(s7.source_segment_range, (91, 92))
        self.assertIn("PRE_RL_COM_SHIFT", s7.physical_phases)
        self.assertNotIn("PRE_RL_SUPPORT_SETUP", s7.physical_phases)
        self.assertEqual(
            CANONICAL_PHASE_SOURCE_STATE[
                (v009, MacroStateId.S7_PRE_RL_SUPPORT_SETUP)
            ],
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
        )

    def test_segment_ownership_rejects_unbound_source(self) -> None:
        first = self.library.segment_ownership[0]
        unbound = RecordingSegmentOwnership(
            source_version="unbound_success",
            state_id=first.state_id,
            phase_source_state_id=first.phase_source_state_id,
            first_segment=first.first_segment,
            last_segment=first.last_segment,
            source_plan_sha256=first.source_plan_sha256,
            evidence_basis="negative-test-only unbound source",
        )
        with self.assertRaisesRegex(
            ValueError, "ownership sources must exactly match successful sources"
        ):
            replace(
                self.library,
                segment_ownership=self.library.segment_ownership + (unbound,),
            )

    def test_keyframe_rejects_incomplete_target_maps(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact canonical actuator set"):
            MotionKeyframe(
                time_s=0.0,
                source_time_s=0.0,
                sequence_index=0,
                source_segment_index=0,
                source_step_index=1,
                physical_phase="PRE_FR_COM_SHIFT",
                subphase=MacroSubphase.PRELOAD,
                servo_targets_deg={SERVO_JOINT_NAMES[0]: 0.0},
                wheel_targets_rad_s={name: 0.0 for name in WHEEL_JOINT_NAMES},
            )

    def test_mixed_channel_profile_synthesizes_wheel_completion_stop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "run"
            run_dir.mkdir()
            plan_path = root / "plan.json"
            plan = {
                "source_version": "synthetic_success",
                "plan_sha256": "a" * 64,
                "segments": [
                    {
                        "decoded_segment_index": 0,
                        "source_step_index": 1,
                        "source_event_indices": [0],
                        "command_start_s": 0.0,
                        "command_end_s": 2.0,
                        "final_segment_duration_s": 2.0,
                        "wheel_duration_s": 1.0,
                        "commands": ["servo front_right_hip 10", "wheel all 0.3"],
                        "servo_target_deg": {"front_right_hip": 10.0},
                        "wheel_target_rad_s": {name: 0.3 for name in WHEEL_JOINT_NAMES},
                        "concurrent": True,
                    }
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            source = RecordingProfileSource(
                source_version="synthetic_success",
                task_result="REPLAY_TASK_SUCCESS",
                plan_path=plan_path,
                gate_a_run_dir=run_dir,
                plan_sha256="a" * 64,
                plan_file_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            )
            span = PhaseSpan(
                source_version="synthetic_success",
                state_id=MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
                strategy="PRIMARY_PROFILE",
                windows=(PhaseWindow("PRE_FR_COM_SHIFT", 0, 0),),
            )
            profile = load_recording_phase_profile(source, span)
            self.assertEqual(len(profile.keyframes), 2)
            self.assertTrue(profile.keyframes[0].atomic_concurrent)
            self.assertEqual(
                profile.keyframes[1].dispatch_kind,
                "wheel_channel_completion_stop",
            )
            self.assertTrue(
                all(value == 0.0 for value in profile.keyframes[1].wheel_targets_rad_s.values())
            )

    def test_alignment_plan_sha_is_bound(self) -> None:
        alignment = MODULE_ROOT / "reports" / "50MM_COMMON_PHASE_ALIGNMENT.csv"
        with tempfile.TemporaryDirectory() as raw:
            bad = Path(raw) / "alignment.csv"
            with alignment.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["plan_sha256"] = "0" * 64
            with bad.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            expected = {source.source_version: source.plan_sha256 for source in self.sources}
            with self.assertRaisesRegex(ValueError, "plan_sha256 mismatch"):
                load_phase_spans(
                    bad,
                    successful_versions=expected,
                    expected_plan_sha256=expected,
                )


if __name__ == "__main__":
    unittest.main()
