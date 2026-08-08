from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from command_model import DEFAULT_MAX_WHEEL_SPEED_RAD_S
from sequence_model import load_steps_jsonl

from fsm_50mm_recording_derived_v3.recording_audit import (
    DEFAULT_RECORDING_ROOT,
    RecordingAudit,
)
from fsm_50mm_recording_derived_v3.recording_fast_plan import fast_plan_rows
from fsm_50mm_recording_derived_v3.support_classifier import (
    ContactClass,
    ObstacleGeometry,
    PrimaryDiagonal,
    TraversalEvidenceTracker,
    WheelContact,
    WheelObservation,
    classify_diagonal_support,
    classify_wheel_contact,
    diagonal_support_corridor,
    polygon_support_margin,
)


LEGS = ("FL", "FR", "RL", "RR")


def _contact(
    leg: str,
    kind: ContactClass,
    *,
    force: float,
    x: float = 0.0,
    bottom_z: float = 0.0,
) -> WheelContact:
    return WheelContact(
        leg=leg,
        contact_class=kind,
        active=kind not in {ContactClass.AIR, ContactClass.UNKNOWN},
        center_w=(x, 0.0, bottom_z + 0.05),
        contact_point_w=(x, 0.0, bottom_z),
        obstacle_relative=(x - 0.5, 0.0, bottom_z - 0.05),
        upward_force_n=force,
        total_force_n=abs(force),
        clearance_over_top_m=bottom_z - 0.05,
        front_face_clearance_m=x - 0.5,
        source="test.filtered_contact_sensor",
        confidence="FORCE_AND_GEOMETRY",
    )


class RecordingIntegrityTests(unittest.TestCase):
    def test_recording_version_enumeration_is_physical_directory_based(self) -> None:
        versions = RecordingAudit().enumerate_versions()
        self.assertEqual(9, len(versions))
        self.assertEqual(
            ["v003", "v005", "v006", "v007", "v008", "v009", "v010", "v011", "v012"],
            [item.version_id.split("_", 1)[0] for item in versions],
        )
        self.assertTrue(all(item.steps_path.is_file() for item in versions))

    def test_accepted_steps_parser_reads_every_version(self) -> None:
        expected_steps = {
            "v003": 24,
            "v005": 22,
            "v006": 22,
            "v007": 21,
            "v008": 23,
            "v009": 23,
            "v010": 26,
            "v011": 28,
            "v012": 19,
        }
        for item in RecordingAudit().enumerate_versions():
            with self.subTest(version=item.version_id):
                steps = load_steps_jsonl(item.steps_path)
                self.assertEqual(expected_steps[item.version_id.split("_", 1)[0]], len(steps))
                self.assertTrue(all(isinstance(step.get("events"), list) for step in steps))

    def test_fast_plan_decoding_and_source_mapping(self) -> None:
        item = RecordingAudit().enumerate_versions()[-1]
        steps = load_steps_jsonl(item.steps_path)
        plan, rows = fast_plan_rows(
            source_version=item.version_id,
            steps=steps,
            max_wheel_speed=DEFAULT_MAX_WHEEL_SPEED_RAD_S,
        )
        self.assertEqual(len(plan.segments), len(rows))
        self.assertGreater(len(rows), 0)
        provenance = [entry for row in rows for entry in row["command_provenance"]]
        self.assertFalse([entry for entry in provenance if entry["mapping_error"]])
        self.assertTrue(all(row["command_end_s"] >= row["command_start_s"] for row in rows))

    def test_fast_plan_preserves_wheel_displacement(self) -> None:
        for item in RecordingAudit().enumerate_versions():
            steps = load_steps_jsonl(item.steps_path)
            _plan, rows = fast_plan_rows(
                source_version=item.version_id,
                steps=steps,
                max_wheel_speed=DEFAULT_MAX_WHEEL_SPEED_RAD_S,
            )
            with self.subTest(version=item.version_id):
                for row in rows:
                    for name, target in row["wheel_target_rad_s"].items():
                        expected = float(target) * float(row["wheel_duration_s"])
                        self.assertAlmostEqual(
                            expected,
                            float(row["expected_wheel_displacement_rad"][name]),
                            places=12,
                        )
                    self.assertAlmostEqual(
                        max(
                            float(row["servo_duration_s"]),
                            float(row["wheel_duration_s"]),
                            float(row["explicit_hold_s"]),
                        ),
                        float(row["final_segment_duration_s"]),
                        places=8,
                    )

    def test_offline_audit_preserves_existing_runtime_readbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_root = Path(directory)
            lock_path = report_root / "environment_lock_50mm.json"
            lock_path.write_text(
                json.dumps({"runtime_readbacks": [{"sentinel": "live-run"}]}),
                encoding="utf-8",
            )
            RecordingAudit(DEFAULT_RECORDING_ROOT, report_root).run()
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual([{"sentinel": "live-run"}], lock["runtime_readbacks"])
            self.assertEqual("runtime_readback_available", lock["status"])


class ContactAndSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.obstacle = ObstacleGeometry(
            front_face_x_m=0.5,
            top_z_m=0.05,
            bottom_z_m=0.0,
            rear_face_x_m=2.5,
            width_m=2.0,
        )

    def test_support_contact_classifier_distinguishes_surfaces_and_air(self) -> None:
        cases = {
            ContactClass.GROUND: WheelObservation("FL", (0.2, 0.0, 0.05), 5.0, 5.0),
            ContactClass.FRONT_FACE: WheelObservation("FL", (0.45, 0.0, 0.075), 1.0, 5.0),
            ContactClass.TOP: WheelObservation("FL", (0.56, 0.0, 0.10), 5.0, 5.0),
            ContactClass.AIR: WheelObservation("FL", (0.40, 0.0, 0.14), 0.0, 0.0),
        }
        for expected, observation in cases.items():
            with self.subTest(expected=expected):
                actual = classify_wheel_contact(
                    observation,
                    self.obstacle,
                    wheel_radius_m=0.05,
                    force_threshold_n=2.0,
                )
                self.assertEqual(expected, actual.contact_class)

    def test_geometry_only_contact_never_supplies_force_support(self) -> None:
        contact = classify_wheel_contact(
            WheelObservation("FL", (0.56, 0.0, 0.10)),
            self.obstacle,
            wheel_radius_m=0.05,
        )
        self.assertEqual(ContactClass.TOP, contact.contact_class)
        self.assertEqual("GEOMETRY_ONLY", contact.confidence)
        diagonal = classify_diagonal_support({"FL": contact})
        self.assertEqual((), diagonal.support_legs)
        self.assertEqual(PrimaryDiagonal.UNKNOWN, diagonal.primary)

    def test_diagonal_support_uses_force_persistence_and_drift(self) -> None:
        contacts = {
            "FL": _contact("FL", ContactClass.TOP, force=8.0),
            "FR": _contact("FR", ContactClass.TOP, force=0.8),
            "RL": _contact("RL", ContactClass.AIR, force=0.0),
            "RR": _contact("RR", ContactClass.GROUND, force=7.0),
        }
        result = classify_diagonal_support(
            contacts,
            persistence_s={"FL": 0.2, "FR": 0.2, "RR": 0.2},
            contact_drift_m={"FL": 0.001, "FR": 0.001, "RR": 0.001},
        )
        self.assertEqual(PrimaryDiagonal.FL_RR, result.primary)
        self.assertEqual(("FL", "RR"), result.support_legs)
        self.assertEqual(("FR",), result.light_support_legs)
        self.assertEqual(("FL", "RR"), result.stable_contact_legs)

    def test_two_leg_corridor_is_translation_invariant(self) -> None:
        base = diagonal_support_corridor(
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.52),
            corridor_half_width_m=0.03,
        )
        shifted = diagonal_support_corridor(
            (7.0, -3.0),
            (8.0, -2.0),
            (7.5, -2.48),
            corridor_half_width_m=0.03,
        )
        self.assertTrue(base.valid)
        self.assertTrue(shifted.valid)
        self.assertAlmostEqual(base.segment_fraction, shifted.segment_fraction)
        self.assertAlmostEqual(base.perpendicular_distance_m, shifted.perpendicular_distance_m)

    def test_two_leg_corridor_rejects_degenerate_and_lateral_escape(self) -> None:
        self.assertFalse(
            diagonal_support_corridor(
                (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), corridor_half_width_m=0.03
            ).valid
        )
        escaped = diagonal_support_corridor(
            (0.0, 0.0), (1.0, 1.0), (0.5, 0.8), corridor_half_width_m=0.03
        )
        self.assertFalse(escaped.valid)
        self.assertFalse(escaped.within_corridor_width)

    def test_three_leg_polygon_has_signed_margin(self) -> None:
        inside = polygon_support_margin([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], (0.2, 0.2))
        outside = polygon_support_margin([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], (0.9, 0.9))
        degenerate = polygon_support_margin([(0.0, 0.0), (1.0, 1.0)], (0.5, 0.5))
        self.assertTrue(inside.valid)
        self.assertGreater(inside.signed_margin_m, 0.0)
        self.assertFalse(outside.valid)
        self.assertLess(outside.signed_margin_m, 0.0)
        self.assertTrue(degenerate.degenerate)


class TraversalEvidenceTests(unittest.TestCase):
    def _tracker(self) -> TraversalEvidenceTracker:
        return TraversalEvidenceTracker(
            unload_force_n=1.0,
            load_confirm_force_n=2.0,
            top_load_dwell_s=0.10,
            loaded_front_face_rotation_limit_rad=0.15,
            wheel_forward_sign={leg: 1.0 for leg in LEGS},
        )

    def test_place_requires_continuous_top_load_dwell(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0) for leg in LEGS}
        tracker.update(0.0, contacts, angles)
        contacts["FL"] = _contact("FL", ContactClass.AIR, force=0.0, x=0.45, bottom_z=0.07)
        tracker.update(0.1, contacts, angles)
        contacts["FL"] = _contact("FL", ContactClass.AIR, force=0.0, x=0.55, bottom_z=0.07)
        tracker.update(0.2, contacts, angles)
        contacts["FL"] = _contact("FL", ContactClass.TOP, force=3.0, x=0.60, bottom_z=0.05)
        tracker.update(0.25, contacts, angles)
        self.assertIsNone(tracker.legs["FL"].top_load_confirm_s)
        contacts["FL"] = _contact("FL", ContactClass.TOP, force=0.5, x=0.60, bottom_z=0.05)
        tracker.update(0.30, contacts, angles)
        contacts["FL"] = _contact("FL", ContactClass.TOP, force=3.0, x=0.60, bottom_z=0.05)
        tracker.update(0.35, contacts, angles)
        tracker.update(0.46, contacts, angles)
        self.assertAlmostEqual(0.46, tracker.legs["FL"].top_load_confirm_s or -1.0)

    def test_sensor_warmup_air_cannot_become_lift_evidence(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {
            leg: _contact(leg, ContactClass.AIR, force=0.0, x=0.0, bottom_z=0.0)
            for leg in LEGS
        }
        tracker.update(0.0, contacts, angles)
        tracker.update(0.10, contacts, angles)
        contacts["FR"] = _contact(
            "FR", ContactClass.TOP, force=4.0, x=0.60, bottom_z=0.05
        )
        tracker.update(0.20, contacts, angles)
        evidence = tracker.result()["legs"]["FR"]
        self.assertIsNone(evidence["unload_start_s"])
        self.assertIsNone(evidence["airborne_start_s"])
        self.assertFalse(evidence["linkage_lift_valid"])
        self.assertIn(
            "GROUND/FRONT_FACE to TOP transition without AIR",
            evidence["illegal_reasons"],
        )

    def test_loaded_front_face_drive_up_is_rejected(self) -> None:
        tracker = self._tracker()
        contacts = {leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0) for leg in LEGS}
        angles = {leg: 0.0 for leg in LEGS}
        tracker.update(0.0, contacts, angles)
        contacts["FR"] = _contact("FR", ContactClass.FRONT_FACE, force=5.0, x=0.45, bottom_z=0.01)
        for index, angle in enumerate((0.06, 0.12, 0.18), start=1):
            angles["FR"] = angle
            tracker.update(index * 0.05, contacts, angles)
        contacts["FR"] = _contact("FR", ContactClass.TOP, force=5.0, x=0.60, bottom_z=0.05)
        tracker.update(0.25, contacts, angles)
        result = tracker.result()["legs"]["FR"]
        self.assertFalse(result["linkage_lift_valid"])
        self.assertIn("loaded front-face wheel rotation exceeded limit", result["illegal_reasons"])
        self.assertIn("GROUND/FRONT_FACE to TOP transition without AIR", result["illegal_reasons"])

    def test_disconnected_attempts_cannot_be_stitched_into_success(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0) for leg in LEGS}
        tracker.update(0.0, contacts, angles)
        contacts["RR"] = _contact("RR", ContactClass.AIR, force=0.0, x=0.40, bottom_z=0.07)
        tracker.update(0.10, contacts, angles)
        tracker.update(0.15, contacts, angles)
        contacts["RR"] = _contact("RR", ContactClass.GROUND, force=5.0, x=0.40)
        tracker.update(0.20, contacts, angles)
        contacts["RR"] = _contact("RR", ContactClass.AIR, force=0.0, x=0.55, bottom_z=0.07)
        tracker.update(0.30, contacts, angles)
        contacts["RR"] = _contact("RR", ContactClass.TOP, force=3.0, x=0.60, bottom_z=0.05)
        tracker.update(0.36, contacts, angles)
        tracker.update(0.47, contacts, angles)
        evidence = tracker.result()["legs"]["RR"]
        self.assertFalse(evidence["linkage_lift_valid"])
        self.assertIn(
            "airborne attempt ended before front-face crossing",
            evidence["illegal_reasons"],
        )

    def test_complete_lift_chain_can_be_proven_per_leg(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0) for leg in LEGS}
        tracker.update(0.0, contacts, angles)
        for leg_index, leg in enumerate(LEGS):
            t = 1.0 + leg_index
            contacts[leg] = _contact(leg, ContactClass.AIR, force=0.0, x=0.45, bottom_z=0.07)
            tracker.update(t, contacts, angles)
            contacts[leg] = _contact(leg, ContactClass.AIR, force=0.0, x=0.55, bottom_z=0.07)
            tracker.update(t + 0.05, contacts, angles)
            contacts[leg] = _contact(leg, ContactClass.TOP, force=3.0, x=0.60, bottom_z=0.05)
            tracker.update(t + 0.10, contacts, angles)
            tracker.update(t + 0.21, contacts, angles)
        result = tracker.result()
        self.assertTrue(result["all_legs_valid"])
        self.assertFalse(result["any_illegal_drive_up"])


if __name__ == "__main__":
    unittest.main()
