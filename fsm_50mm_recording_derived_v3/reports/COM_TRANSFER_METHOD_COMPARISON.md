# 50 mm COM Transfer Method Comparison

## Evidence boundary

This is an offline endpoint comparison. No Isaac process was started. The nine recording directories contain commands and before/after articulation snapshots, but no standalone continuous COM, support force/class, contact-point drift, or video telemetry.

- Physical version directories enumerated: 9
- Source steps aligned: 208
- Steps with complete start/end root, attitude, joint, and velocity boundaries: 192
- Adjacent step boundaries compared: 199
- Endpoint compatibility counts: {"MISSING_ENDPOINT": 16, "POSE_COMPATIBLE_ENDPOINTS": 110, "POSE_GAP_EXCEEDS_RESTORE_TOLERANCE": 73}
- Steps with nonzero expected wheel displacement in the authoritative Fast plan: 96
- Steps containing at least one concurrent servo/wheel Fast segment: 48
- Detailed alignment CSV: C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\RECORDING_PHASE_ALIGNMENT_50MM.csv

Every row is marked ENDPOINT_CANDIDATE and PENDING_REPLAY. These labels must not be promoted using endpoint data alone.

## Physical recording directories

- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v003_20260805_224517_157723_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v005_20260805_225441_439112_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v006_20260805_233948_654778_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v007_20260806_190636_100857_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v008_20260806_211408_578700_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v009_20260806_215232_433234_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v010_20260806_220745_363972_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v011_20260806_223621_672618_manual
- C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v012_20260806_231025_027004_manual

## Per-version endpoint inventory

| Version | Steps | Full boundaries | Nonzero wheel-displacement steps | Concurrent steps | Endpoint adjacency results |
|---|---:|---:|---:|---:|---|
| v003_20260805_224517_157723_manual | 24 | 8 | 11 | 6 | FIRST_STEP=1, MISSING_ENDPOINT=16, POSE_COMPATIBLE_ENDPOINTS=5, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=2 |
| v005_20260805_225441_439112_manual | 22 | 22 | 11 | 6 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=13, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=8 |
| v006_20260805_233948_654778_manual | 22 | 22 | 10 | 6 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=9, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=12 |
| v007_20260806_190636_100857_manual | 21 | 21 | 8 | 3 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=14, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=6 |
| v008_20260806_211408_578700_manual | 23 | 23 | 11 | 5 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=15, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=7 |
| v009_20260806_215232_433234_manual | 23 | 23 | 12 | 7 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=11, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=11 |
| v010_20260806_220745_363972_manual | 26 | 26 | 11 | 4 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=16, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=9 |
| v011_20260806_223621_672618_manual | 28 | 28 | 13 | 7 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=17, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=10 |
| v012_20260806_231025_027004_manual | 19 | 19 | 9 | 4 | FIRST_STEP=1, POSE_COMPATIBLE_ENDPOINTS=10, POSE_GAP_EXCEEDS_RESTORE_TOLERANCE=8 |

## Method comparison

| Method | What endpoints can show | What endpoints cannot establish | Relevant audited failure constraints | Required clean-replay evidence | Current status |
|---|---|---|---|---|---|
| Impulse | Short successive Fast segments, command deltas, endpoint velocity before/after, and source timing can identify an impulse-shaped command candidate. | Endpoint velocity does not prove a contact-force impulse, momentum transfer, preload, coast, or causal COM shift during the step. | Old-failure audit findings 2, 5, 7, 14, and 16: static compression loses dynamics; entry steps and transition discontinuities matter; time alone is not a physical guard. | High-rate COM position/velocity, per-wheel force and contact class, contact-point drift, joint velocity, command dispatch time, and pre/post impulse dwell. | ENDPOINT_CANDIDATE / PENDING_REPLAY |
| Anchored support-angle | Servo-only posture changes with zero commanded wheel speed identify support-angle endpoint candidates and expose joint/root boundary changes. | Zero wheel command does not prove an anchored contact, and an endpoint root shift does not prove the support points stayed fixed or loaded. | Findings 3, 9, 10, 11, 12, 13, 15, and 18: isolate the active leg; verify contact anchoring/load; use per-leg IK/signs; relatch COM targets; use a diagonal corridor rather than a fabricated polygon. | Relative COM to measured support contacts, support-point drift, retained upward load with dwell, per-leg applied IK, and diagonal segment coordinates s and d_perp. | ENDPOINT_CANDIDATE / PENDING_REPLAY |
| Wheel assist | Nonzero wheel targets, Fast active duration, expected angular displacement, and servo/wheel concurrency are preserved and reported per step. | Wheel angular displacement does not prove body-relative COM improvement; the support boundary and body may translate together, and slip is unknown. | Findings 5, 6, 7, 8, 9, and 19: no phase-entry speed step, rear-local ramping only, preserve sustained travel, and never treat wheel travel as relative COM transfer. | Wheel/world displacement, body displacement, slip, contact-point drift, support-boundary motion, active-wheel zero-command compliance, and COM relative to the moving support set. | ENDPOINT_CANDIDATE / PENDING_REPLAY |
| Hybrid | A step with concurrent or sequential servo and wheel commands can be identified without discarding command order, Fast duration, or expected wheel displacement. | Endpoints cannot attribute the observed root/joint change to the angle action, wheel action, their order, an impulse, or uncontrolled contact dynamics. | All constraints above, especially findings 2-8 and 14: preserve timing/concurrency, isolate the active leg, scope ramps locally, and maintain boundary continuity. | Primitive-isolation A/B replays first, then a preregistered hybrid replay with the same full telemetry, unchanged safety gates, and component-wise ablation. | ENDPOINT_CANDIDATE / PENDING_REPLAY |

## Interpretation

The endpoint inventory is sufficient to define replay candidates and to reject incompatible step joins. It is not sufficient to select a winning COM-transfer mechanism.

A conservative experimental order is:

1. verify an anchored support-angle primitive while measuring actual contact anchoring and load;
2. verify wheel assist separately using relative COM and support-boundary motion, not wheel travel;
3. replay impulse-shaped candidates with high-rate force and velocity telemetry;
4. evaluate a hybrid only after each component has an isolated, repeatable mechanical effect.

This order is a safety and identifiability recommendation, not a claim that any recording has already demonstrated one of these mechanisms.

## Compatibility semantics

POSE_COMPATIBLE_ENDPOINTS uses the repository's existing restore tolerances: root position <= 0.005 m, root orientation <= 0.5 deg, servo position <= 1.0 deg, and wheel position <= 0.05 rad.

Velocity gaps are reported but do not decide endpoint compatibility. Whether momentum should be retained or reset is precisely one of the mechanisms that requires replay. MISSING_ENDPOINT is retained for legacy v003 snapshots and is never silently filled.

## Evidence sources

- Endpoint alignment: C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\RECORDING_PHASE_ALIGNMENT_50MM.csv
- Old-failure evidence audit: C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\OLD_FSM_FAILURES_TO_AVOID.md
- Recording audit: C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\RECORDING_AUDIT_50MM.md
- Environment lock: C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\environment_lock_50mm.json
