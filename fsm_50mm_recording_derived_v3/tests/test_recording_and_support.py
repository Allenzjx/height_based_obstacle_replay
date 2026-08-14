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
    ContactPersistenceMode,
    ContactPersistenceTracker,
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

    @staticmethod
    def _filtered_surface_evidence(
        *,
        leg: str = "RL",
        ground_active: bool = False,
        obstacle_active: bool = False,
        ground_force_n: float | None = None,
        obstacle_force_n: float | None = None,
    ) -> dict[str, dict[str, object]]:
        forces = {
            "ground": (
                12.934568405151484 if ground_active else 0.0
                if ground_force_n is None
                else ground_force_n
            ),
            "obstacle": (
                12.934568405151484 if obstacle_active else 0.0
                if obstacle_force_n is None
                else obstacle_force_n
            ),
        }
        active = {"ground": ground_active, "obstacle": obstacle_active}
        return {
            surface: {
                "leg": leg,
                "surface": surface,
                "identity_valid": True,
                "force_valid": True,
                "contact_point_valid": enabled,
                "active": enabled,
                "normal_force_n": forces[surface],
            }
            for surface, enabled in active.items()
        }

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

    def test_exact_ground_surface_resolves_real_7984_7985_geometry_boundary(self) -> None:
        obstacle = ObstacleGeometry(
            front_face_x_m=0.5213121734675507,
            top_z_m=0.05,
            bottom_z_m=0.0,
            rear_face_x_m=2.5,
            width_m=2.0,
        )
        radius = 0.04998999834060672
        evidence = self._filtered_surface_evidence(ground_active=True)
        # Real trial sample_index 7984 is just outside the face candidate;
        # 7985 crosses into the ground/face overlap while the exact filtered
        # ground pair remains the only active pair.
        samples = (
            (0.4592247009277344, 0.07858600467443466, 0.05212927609682083),
            (0.4593568742275238, 0.07857198268175125, 0.052130326628685),
        )
        classified_samples: list[WheelContact] = []
        for sample_index, center in zip((7984, 7985), samples):
            with self.subTest(sample_index=sample_index):
                contact = classify_wheel_contact(
                    WheelObservation("RL", center, 12.934568405151367, 12.934568405151484),
                    obstacle,
                    wheel_radius_m=radius,
                    force_threshold_n=2.0,
                    filtered_surface_evidence=evidence,
                )
                self.assertEqual(ContactClass.GROUND, contact.contact_class)
                self.assertTrue(contact.active)
                self.assertEqual(
                    "EXACT_FILTERED_SURFACE_AND_GEOMETRY", contact.confidence
                )
                classified_samples.append(contact)

        traversal = TraversalEvidenceTracker(
            unload_force_n=1.0,
            load_confirm_force_n=2.0,
            top_load_dwell_s=0.05,
            loaded_front_face_rotation_limit_rad=0.1,
        )
        other_contacts = {
            leg: _contact(leg, ContactClass.GROUND, force=8.0)
            for leg in ("FL", "FR", "RR")
        }
        traversal.update(
            68.03333333333354,
            {**other_contacts, "RL": classified_samples[0]},
            {leg: 0.0 for leg in LEGS},
        )
        traversal.update(
            68.04166666666688,
            {**other_contacts, "RL": classified_samples[1]},
            {**{leg: 0.0 for leg in LEGS}, "RL": 1.0},
        )
        self.assertEqual(0.0, traversal.legs["RL"].loaded_front_face_rotation_rad)
        self.assertEqual([], traversal.legs["RL"].illegal_reasons)

        fallback = classify_wheel_contact(
            WheelObservation("RL", samples[1], 12.934568405151367, 12.934568405151484),
            obstacle,
            wheel_radius_m=radius,
            force_threshold_n=2.0,
        )
        self.assertEqual(ContactClass.FRONT_FACE, fallback.contact_class)
        self.assertEqual("FORCE_AND_GEOMETRY", fallback.confidence)
        self.assertNotIn("EXACT", fallback.confidence)

    def test_exact_obstacle_and_both_surface_cases_keep_geometric_precedence(self) -> None:
        face_center = (0.45, 0.0, 0.075)
        obstacle_only = classify_wheel_contact(
            WheelObservation("RL", face_center, 5.0, 5.0),
            self.obstacle,
            wheel_radius_m=0.05,
            filtered_surface_evidence=self._filtered_surface_evidence(
                obstacle_active=True
            ),
        )
        self.assertEqual(ContactClass.FRONT_FACE, obstacle_only.contact_class)

        both_face = classify_wheel_contact(
            WheelObservation("RL", face_center, 10.0, 10.0),
            self.obstacle,
            wheel_radius_m=0.05,
            filtered_surface_evidence=self._filtered_surface_evidence(
                ground_active=True,
                obstacle_active=True,
            ),
        )
        self.assertEqual(ContactClass.FRONT_FACE, both_face.contact_class)

        upper_corner = classify_wheel_contact(
            WheelObservation("RL", (0.45, 0.0, 0.10), 10.0, 10.0),
            self.obstacle,
            wheel_radius_m=0.05,
            filtered_surface_evidence=self._filtered_surface_evidence(
                ground_active=True,
                obstacle_active=True,
            ),
        )
        self.assertEqual(ContactClass.TOP, upper_corner.contact_class)

    def test_exact_neither_surface_fails_closed_for_loaded_aggregate(self) -> None:
        loaded = classify_wheel_contact(
            WheelObservation("RL", (0.45, 0.0, 0.075), 8.0, 8.0),
            self.obstacle,
            wheel_radius_m=0.05,
            filtered_surface_evidence=self._filtered_surface_evidence(),
        )
        self.assertEqual(ContactClass.UNKNOWN, loaded.contact_class)
        self.assertFalse(loaded.active)
        self.assertEqual("FILTERED_SURFACE_GEOMETRY_MISMATCH", loaded.confidence)

        unloaded = classify_wheel_contact(
            WheelObservation("RL", (0.45, 0.0, 0.075), 0.0, 0.0),
            self.obstacle,
            wheel_radius_m=0.05,
            filtered_surface_evidence=self._filtered_surface_evidence(),
        )
        self.assertEqual(ContactClass.AIR, unloaded.contact_class)
        self.assertEqual("EXACT_FILTERED_SURFACE_NO_CONTACT", unloaded.confidence)

    def test_invalid_exact_surface_evidence_never_falls_back_to_geometry(self) -> None:
        base = self._filtered_surface_evidence(ground_active=True)
        invalid_cases = {
            "identity": {**base, "ground": {**base["ground"], "identity_valid": False}},
            "wrong_leg": {**base, "ground": {**base["ground"], "leg": "FL"}},
            "force": {**base, "ground": {**base["ground"], "force_valid": False}},
            "point": {
                **base,
                "ground": {**base["ground"], "contact_point_valid": False},
            },
            "normal": {
                **base,
                "ground": {**base["ground"], "normal_force_n": float("nan")},
            },
            "activity": {
                **base,
                "ground": {**base["ground"], "normal_force_n": 0.0},
            },
        }
        for label, evidence in invalid_cases.items():
            with self.subTest(label=label):
                contact = classify_wheel_contact(
                    WheelObservation("RL", (0.45, 0.0, 0.075), 8.0, 8.0),
                    self.obstacle,
                    wheel_radius_m=0.05,
                    filtered_surface_evidence=evidence,
                )
                self.assertEqual(ContactClass.UNKNOWN, contact.contact_class)
                self.assertFalse(contact.active)
                self.assertEqual(
                    "FILTERED_SURFACE_EVIDENCE_INVALID", contact.confidence
                )

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


class ContactPersistenceTrackerTests(unittest.TestCase):
    @staticmethod
    def _contacts() -> dict[str, WheelContact]:
        return {
            leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0)
            for leg in LEGS
        }

    @staticmethod
    def _targets(value: float = 0.0) -> dict[str, float]:
        return {leg: value for leg in LEGS}

    @staticmethod
    def _point_flags(value: bool = True) -> dict[str, bool]:
        return {leg: value for leg in LEGS}

    def _update(
        self,
        tracker: ContactPersistenceTracker,
        time_s: float,
        contacts: dict[str, WheelContact],
        *,
        logical: dict[str, float] | None = None,
        physx: dict[str, float] | None = None,
        point_flags: dict[str, bool] | None = None,
        drive_valid: bool | dict[str, bool] | None = True,
    ) -> tuple[dict[str, float], dict[str, float]]:
        return tracker.update(
            time_s,
            contacts,
            logical_wheel_target_rad_s=self._targets() if logical is None else logical,
            physx_wheel_target_rad_s=self._targets() if physx is None else physx,
            measured_contact_point_valid=(
                self._point_flags() if point_flags is None else point_flags
            ),
            drive_evidence_valid=drive_valid,
        )

    def test_zero_targets_accumulate_measured_contact_point_displacement(self) -> None:
        tracker = ContactPersistenceTracker(load_confirm_force_n=2.0)
        contacts = self._contacts()
        self._update(tracker, 0.0, contacts)
        contacts["FL"] = _contact("FL", ContactClass.GROUND, force=5.0, x=0.012)
        persistence, drift = self._update(tracker, 0.10, contacts)

        sample = tracker.last_samples["FL"]
        self.assertEqual(
            ContactPersistenceMode.ZERO_TARGET_LOADED_CONTACT_EPOCH,
            sample.mode,
        )
        self.assertTrue(sample.valid)
        self.assertTrue(sample.evidence_valid)
        self.assertFalse(sample.physical_anchoring_proven)
        self.assertFalse(sample.material_point_identity_available)
        self.assertEqual(
            "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT",
            sample.contact_point_displacement_semantics,
        )
        self.assertAlmostEqual(0.10, persistence["FL"])
        self.assertAlmostEqual(0.012, drift["FL"])
        self.assertAlmostEqual(
            0.012,
            sample.zero_target_contact_epoch_max_displacement_m or -1.0,
        )
        self.assertAlmostEqual(
            0.012,
            tracker.maximum_zero_target_contact_point_displacement_m["FL"],
        )
        self.assertNotIn("ANCHORED", sample.as_dict().values())

    def test_rolling_diagnostic_does_not_pollute_zero_target_maximum(self) -> None:
        tracker = ContactPersistenceTracker()
        contacts = self._contacts()
        self._update(tracker, 0.0, contacts)
        logical = self._targets()
        physx = self._targets()
        # A zero logical command cannot start a zero-target measurement epoch
        # while the independent PhysX target is still non-zero.
        logical["FL"] = 0.0
        physx["FL"] = 1.0
        contacts["FL"] = _contact("FL", ContactClass.GROUND, force=5.0, x=0.10)
        self._update(tracker, 0.10, contacts, logical=logical, physx=physx)
        contacts["FL"] = _contact("FL", ContactClass.GROUND, force=5.0, x=0.22)
        persistence, drift = self._update(
            tracker, 0.20, contacts, logical=logical, physx=physx
        )

        sample = tracker.last_samples["FL"]
        self.assertEqual(ContactPersistenceMode.ACTIVE_ROLLING, sample.mode)
        self.assertFalse(sample.valid)
        self.assertTrue(sample.evidence_valid)
        self.assertTrue(math.isnan(drift["FL"]))
        self.assertAlmostEqual(0.20, persistence["FL"])
        self.assertAlmostEqual(0.12, sample.rolling_displacement_m or -1.0)
        self.assertAlmostEqual(
            0.12, sample.maximum_rolling_displacement_m_so_far or -1.0
        )
        self.assertEqual(
            0.0,
            tracker.maximum_zero_target_contact_point_displacement_m["FL"],
        )

    def test_rolling_to_zero_starts_new_measurement_epoch_at_transition(self) -> None:
        tracker = ContactPersistenceTracker()
        contacts = self._contacts()
        rolling_logical = self._targets()
        rolling_physx = self._targets()
        rolling_logical["FR"] = 1.0
        rolling_physx["FR"] = 1.0
        self._update(
            tracker,
            0.0,
            contacts,
            logical=rolling_logical,
            physx=rolling_physx,
        )
        contacts["FR"] = _contact("FR", ContactClass.GROUND, force=5.0, x=0.20)
        self._update(
            tracker,
            0.10,
            contacts,
            logical=rolling_logical,
            physx=rolling_physx,
        )

        contacts["FR"] = _contact("FR", ContactClass.GROUND, force=5.0, x=0.25)
        _, transition_drift = self._update(tracker, 0.20, contacts)
        transition = tracker.last_samples["FR"]
        self.assertEqual(
            ContactPersistenceMode.ZERO_TARGET_LOADED_CONTACT_EPOCH,
            transition.mode,
        )
        self.assertEqual(0.0, transition_drift["FR"])
        self.assertEqual(
            0.0,
            transition.zero_target_contact_point_displacement_m,
        )
        self.assertAlmostEqual(
            0.20, transition.maximum_rolling_displacement_m_so_far or -1.0
        )

        contacts["FR"] = _contact("FR", ContactClass.GROUND, force=5.0, x=0.26)
        _, next_drift = self._update(tracker, 0.30, contacts)
        self.assertAlmostEqual(0.01, next_drift["FR"])

    def test_zero_target_motion_is_drift_even_without_a_qd_gate(self) -> None:
        tracker = ContactPersistenceTracker()
        contacts = self._contacts()
        self._update(tracker, 0.0, contacts)
        # Contact-point motion with both commanded/read-back targets at exactly
        # zero remains displacement evidence, but never proves anchoring.
        contacts["RR"] = _contact("RR", ContactClass.GROUND, force=5.0, x=0.03)
        _, drift = self._update(tracker, 0.10, contacts)
        self.assertEqual(
            ContactPersistenceMode.ZERO_TARGET_LOADED_CONTACT_EPOCH,
            tracker.last_samples["RR"].mode,
        )
        self.assertFalse(tracker.last_samples["RR"].physical_anchoring_proven)
        self.assertAlmostEqual(0.03, drift["RR"])

    def test_missing_nonfinite_or_geometric_only_evidence_never_returns_zero(self) -> None:
        contacts = self._contacts()

        missing = ContactPersistenceTracker()
        _, missing_drift = missing.update(0.0, contacts)
        self.assertEqual(
            ContactPersistenceMode.INVALID_EVIDENCE,
            missing.last_samples["FL"].mode,
        )
        self.assertTrue(math.isnan(missing_drift["FL"]))
        self.assertIn("measured contact-point", missing.last_samples["FL"].reason)

        nonfinite = ContactPersistenceTracker()
        physx = self._targets()
        physx["FL"] = float("nan")
        _, nonfinite_drift = self._update(nonfinite, 0.0, contacts, physx=physx)
        self.assertTrue(math.isnan(nonfinite_drift["FL"]))
        self.assertIn("PhysX", nonfinite.last_samples["FL"].reason)

        geometric_only = ContactPersistenceTracker()
        flags = self._point_flags()
        flags["FL"] = False
        _, geometric_drift = self._update(
            geometric_only, 0.0, contacts, point_flags=flags
        )
        self.assertTrue(math.isnan(geometric_drift["FL"]))
        self.assertFalse(geometric_only.last_samples["FL"].valid)

        invalid_drive = ContactPersistenceTracker()
        _, invalid_drive_drift = self._update(
            invalid_drive, 0.0, contacts, drive_valid=False
        )
        self.assertTrue(math.isnan(invalid_drive_drift["FL"]))
        self.assertIn("drive-target", invalid_drive.last_samples["FL"].reason)

        invalid_point = ContactPersistenceTracker()
        contacts["FL"] = _contact(
            "FL", ContactClass.GROUND, force=5.0, x=float("nan")
        )
        _, invalid_point_drift = self._update(invalid_point, 0.0, contacts)
        self.assertTrue(math.isnan(invalid_point_drift["FL"]))
        self.assertIn("non-finite", invalid_point.last_samples["FL"].reason)

    def test_loss_unload_and_class_change_reset_epochs(self) -> None:
        tracker = ContactPersistenceTracker()
        contacts = self._contacts()
        self._update(tracker, 0.0, contacts)
        contacts["RL"] = _contact("RL", ContactClass.GROUND, force=5.0, x=0.02)
        self._update(tracker, 0.10, contacts)
        self.assertAlmostEqual(
            0.02,
            tracker.maximum_zero_target_contact_point_displacement_m["RL"],
        )

        contacts["RL"] = _contact("RL", ContactClass.GROUND, force=0.5, x=0.02)
        persistence, drift = self._update(tracker, 0.20, contacts)
        self.assertEqual(ContactPersistenceMode.UNLOADED, tracker.last_samples["RL"].mode)
        self.assertEqual(0.0, persistence["RL"])
        self.assertTrue(math.isnan(drift["RL"]))
        self.assertNotIn("RL", tracker.zero_target_contact_anchor_w)

        contacts["RL"] = _contact("RL", ContactClass.TOP, force=5.0, x=0.60, bottom_z=0.05)
        _, top_drift = self._update(tracker, 0.30, contacts)
        self.assertEqual(0.0, top_drift["RL"])
        self.assertEqual(0.30, tracker.zero_target_contact_epoch_start_s["RL"])

        contacts["RL"] = _contact("RL", ContactClass.AIR, force=0.0, x=0.60, bottom_z=0.08)
        _, air_drift = self._update(tracker, 0.40, contacts)
        self.assertEqual(ContactPersistenceMode.NO_CONTACT, tracker.last_samples["RL"].mode)
        self.assertTrue(math.isnan(air_drift["RL"]))
        self.assertNotIn("RL", tracker.zero_target_contact_anchor_w)

    def test_loaded_front_face_is_not_zero_target_displacement_eligible(self) -> None:
        tracker = ContactPersistenceTracker()
        contacts = self._contacts()
        contacts["FL"] = _contact(
            "FL", ContactClass.FRONT_FACE, force=5.0, x=0.45, bottom_z=0.01
        )
        _, drift = self._update(tracker, 0.0, contacts)
        sample = tracker.last_samples["FL"]
        self.assertEqual(ContactPersistenceMode.INVALID_EVIDENCE, sample.mode)
        self.assertTrue(math.isnan(drift["FL"]))
        self.assertIn("not eligible for zero-target", sample.reason)

    def test_contact_persistence_continues_during_valid_rolling(self) -> None:
        tracker = ContactPersistenceTracker()
        contacts = self._contacts()
        logical = self._targets(0.5)
        physx = self._targets(0.5)
        self._update(tracker, 0.0, contacts, logical=logical, physx=physx)
        contacts["FL"] = _contact("FL", ContactClass.GROUND, force=5.0, x=0.10)
        persistence, _ = self._update(
            tracker, 0.25, contacts, logical=logical, physx=physx
        )
        self.assertAlmostEqual(0.25, persistence["FL"])
        self.assertEqual(
            ContactPersistenceMode.ACTIVE_ROLLING,
            tracker.last_samples["FL"].mode,
        )


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
        self.assertEqual(
            "INCOMPLETE_PRE_FACE",
            evidence["episodes"][0]["status"],
        )
        self.assertEqual([], evidence["episodes"][0]["illegal_reasons"])
        self.assertNotIn(
            "airborne attempt ended before front-face crossing",
            evidence["illegal_reasons"],
        )
        # The second AIR sample did not sustain a fresh unload dwell, so none
        # of the first episode's timestamps may be reused to validate it.
        self.assertIsNone(evidence["episodes"][1]["unload_start_s"])
        self.assertIsNone(evidence["episodes"][1]["airborne_start_s"])
        self.assertIn(
            "GROUND/FRONT_FACE to TOP transition without AIR",
            evidence["illegal_reasons"],
        )

    def test_preface_retry_is_nonillegal_and_can_reprove_the_full_chain(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {
            leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0)
            for leg in LEGS
        }
        tracker.update(0.0, contacts, angles)

        contacts["RL"] = _contact(
            "RL", ContactClass.AIR, force=0.0, x=0.40, bottom_z=0.07
        )
        tracker.update(0.10, contacts, angles)
        tracker.update(0.15, contacts, angles)
        contacts["RL"] = _contact("RL", ContactClass.GROUND, force=5.0, x=0.40)
        tracker.update(0.20, contacts, angles)

        after_retry = tracker.result()["legs"]["RL"]
        self.assertEqual("INCOMPLETE_PRE_FACE", after_retry["episodes"][0]["status"])
        self.assertEqual([], after_retry["illegal_reasons"])
        self.assertIsNone(after_retry["unload_start_s"])
        self.assertIsNone(after_retry["airborne_start_s"])

        # Re-establish loaded support, sustain the same configured unload dwell,
        # then complete AIR->CLEAR_FACE->TOP->LOAD as one new episode.
        contacts["RL"] = _contact(
            "RL", ContactClass.AIR, force=0.0, x=0.45, bottom_z=0.07
        )
        tracker.update(0.30, contacts, angles)
        contacts["RL"] = _contact(
            "RL", ContactClass.AIR, force=0.0, x=0.55, bottom_z=0.07
        )
        tracker.update(0.35, contacts, angles)
        contacts["RL"] = _contact(
            "RL", ContactClass.TOP, force=3.0, x=0.60, bottom_z=0.05
        )
        tracker.update(0.40, contacts, angles)
        tracker.update(0.51, contacts, angles)

        evidence = tracker.result()["legs"]["RL"]
        self.assertTrue(evidence["linkage_lift_valid"])
        self.assertEqual([], evidence["illegal_reasons"])
        self.assertEqual(
            ["INCOMPLETE_PRE_FACE", "VALID"],
            [row["status"] for row in evidence["episodes"]],
        )
        self.assertEqual("VALID", evidence["canonical_episode"]["status"])
        # Legacy summary keys mirror the canonical episode.
        self.assertEqual(
            evidence["canonical_episode"]["airborne_start_s"],
            evidence["airborne_start_s"],
        )
        self.assertEqual(
            evidence["canonical_episode"]["top_load_confirm_s"],
            evidence["top_load_confirm_s"],
        )

    def test_postface_non_top_landing_is_genuinely_illegal(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {
            leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0)
            for leg in LEGS
        }
        tracker.update(0.0, contacts, angles)
        contacts["FL"] = _contact(
            "FL", ContactClass.AIR, force=0.0, x=0.45, bottom_z=0.07
        )
        tracker.update(0.10, contacts, angles)
        contacts["FL"] = _contact(
            "FL", ContactClass.AIR, force=0.0, x=0.55, bottom_z=0.07
        )
        tracker.update(0.15, contacts, angles)
        contacts["FL"] = _contact("FL", ContactClass.GROUND, force=5.0, x=0.55)
        tracker.update(0.20, contacts, angles)

        evidence = tracker.result()["legs"]["FL"]
        self.assertFalse(evidence["linkage_lift_valid"])
        self.assertEqual(
            "ILLEGAL_POST_FACE_NON_TOP",
            evidence["episodes"][0]["status"],
        )
        self.assertIn(
            "airborne attempt ended after front-face crossing on non-TOP contact",
            evidence["illegal_reasons"],
        )

    def test_unknown_contact_ends_continuity_without_fabricating_illegal(self) -> None:
        tracker = self._tracker()
        angles = {leg: 0.0 for leg in LEGS}
        contacts = {
            leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0)
            for leg in LEGS
        }
        tracker.update(0.0, contacts, angles)
        contacts["FR"] = _contact(
            "FR", ContactClass.AIR, force=0.0, x=0.45, bottom_z=0.07
        )
        tracker.update(0.10, contacts, angles)
        tracker.update(0.15, contacts, angles)
        contacts["FR"] = _contact(
            "FR", ContactClass.UNKNOWN, force=0.0, x=0.55, bottom_z=0.07
        )
        tracker.update(0.20, contacts, angles)

        result = tracker.result()
        evidence = result["legs"]["FR"]
        self.assertFalse(result["any_illegal_drive_up"])
        self.assertEqual([], evidence["illegal_reasons"])
        self.assertEqual("NOT_EVALUABLE", evidence["episodes"][0]["status"])
        self.assertIsNone(evidence["active_episode_status"])

    def test_loaded_front_face_rotation_accumulates_across_episode_reset(self) -> None:
        tracker = self._tracker()
        contacts = {
            leg: _contact(leg, ContactClass.GROUND, force=5.0, x=0.0)
            for leg in LEGS
        }
        angles = {leg: 0.0 for leg in LEGS}
        tracker.update(0.0, contacts, angles)
        contacts["FR"] = _contact(
            "FR", ContactClass.FRONT_FACE, force=5.0, x=0.45, bottom_z=0.01
        )
        angles["FR"] = 0.08
        tracker.update(0.05, contacts, angles)
        contacts["FR"] = _contact(
            "FR", ContactClass.UNKNOWN, force=0.0, x=0.45, bottom_z=0.01
        )
        tracker.update(0.10, contacts, angles)
        contacts["FR"] = _contact("FR", ContactClass.GROUND, force=5.0, x=0.40)
        tracker.update(0.15, contacts, angles)
        contacts["FR"] = _contact(
            "FR", ContactClass.FRONT_FACE, force=5.0, x=0.45, bottom_z=0.01
        )
        angles["FR"] = 0.16
        tracker.update(0.20, contacts, angles)

        evidence = tracker.result()["legs"]["FR"]
        self.assertAlmostEqual(0.16, evidence["loaded_front_face_rotation_rad"])
        self.assertIn(
            "loaded front-face wheel rotation exceeded limit",
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
