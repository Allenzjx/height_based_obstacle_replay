# 50 mm replay and Selected Fast pose restore fix

## Outcome

- Active formal version: `v002_20260805_185955_854964_manual` at `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v002_20260805_185955_854964_manual`.
- Formal SHA-256: `f43061e9d398eb0cb955e48a059abc13135761166c33e088e389b779cb8f38cd`; 16 steps, 80 source events/commands, 96 planned events, 64 segments.
- Before: Fast and Raw both false-stopped at Step 14 / Segment 56 (zero-based), after 81/96 events, with `actuator_limit` even though their errors (0.144823° and 0.289960°) were already inside the 1° normal tolerance.
- After: Raw and Fast each reached Step 16, sent 96/96 events, completed 64/64 segments, `stop_reason=complete`, `operation=IDLE`.
- New real recordings produced four FULL_VALID boundaries. Three real mouse Selected Fast runs restored Step 1 `sim_state_after`, verified the physical pose, then played only Step 2; all completed.
- All 94 protected files, including active v002, are byte-identical before/after.

## Root causes

1. Replay false stop: in a mixed servo+wheel segment, the scheduler ran servo-stall logic whenever the whole segment was not complete. It did so even after `servo_done=True` while only the wheel duration remained. The formal failures prove this: error was below tolerance but a fixed no-improvement window still raised `actuator_limit`.
2. Placeholder recording: lightweight subprocess status omitted `sim_state`; synchronous capture fell back to local NullSim and persisted null root/joint fields.
3. Selected Fast: the resolver treated any non-empty dictionary as a pose and even wrapped command_state as sim_state. The worker acknowledged restore before a verified physics boundary, and adapter reset ordering could overwrite writes.

The missing-state bug explains old unusable pose checkpoints and the Selected behavior. The formal mid-run stop had a separate immediate cause (the mixed-channel false-stall branch); missing physical endpoint data forced the legacy conservative policy but was not the direct reason an in-tolerance joint failed.

## Runtime repair

- One shared validator classifies all states and requires finite root pose/velocity, complete named articulation joint pos/vel, complete servo/wheel command and actual state, and a non-NullSim source.
- Recording enters `RECORDING_PREPARING`, correlates request/purpose/session/height/version/revision, and begins/finalizes only on FULL_VALID detailed worker snapshots. Subprocess capture cannot fall back to NullSim. A stop-capture failure creates no pending step and supports explicit retry.
- The completion change is minimal: never invoke servo-stall logic after the servo channel is done; a real stall additionally requires error outside tolerance, a full no-improvement window, readable state, and near-zero target-joint velocity. The 3° hard contact cap remains.
- Selected Fast accepts only previous `sim_state_after` FULL_VALID, or a FULL_VALID selected start with continuity; Step 1 requires its own FULL_VALID start. Command-only and placeholder sources are rejected.
- Adapter reset occurs before root/joint writes. Worker applies a safe wheel boundary, advances one bounded physics step, captures measured state, and verifies 5 mm / 0.5° / 1° / 0.05 rad tolerances before publishing restore OK or starting playback.

## Before failure evidence

- Fast: `actuator_limit: servo target did not improve for 0.800s sim; step=14 segment=56 joint=rear_left_hip requested_command_deg=1.6 expected_actual_deg=-1.7794263347525243 measured_actual_deg=-1.634603646931802 error_deg=0.14482268782072238 max_error_deg=0.144823 tolerance_deg=1.000000`
- Raw: `actuator_limit: servo target did not improve for 0.800s sim; step=14 segment=56 joint=rear_right_knee requested_command_deg=-40.9 expected_actual_deg=41.11524399350242 measured_actual_deg=41.40520439016967 error_deg=0.28996039666725437 max_error_deg=0.289960 tolerance_deg=1.000000`

## Compatibility

The current v002 has `{'PLACEHOLDER_NO_SIM': 32}` across 32 boundaries and cannot support exact original manual pose restore. It was not rewritten and no replay-derived version was created. Re-record with the repaired flow for exact manual poses.
