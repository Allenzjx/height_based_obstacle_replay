# 50 mm Recording Timing Audit (Static)

Status: **STATIC AUDIT COMPLETE; PHYSICAL REPLAY PENDING**

Audit date: 2026-08-10

Scope: the nine accepted 50 mm recording versions under
`saved_height_steps_fsm_reference_v2/height_050mm/versions`.

This report audits saved timing data and the current Fast-plan/executor path.
It does not claim that any recording has completed a clean physical Isaac
replay. No A1/A2/B trajectory-equivalence triplet or nine-recording physical
success set exists as evidence for this report.

## 1. Version timing inventory

The `Raw duration` column is the sum of the saved step durations. `Fast @ 150`
is the currently materialized authoritative Fast-plan output. `Fast @ 30` is a
pure-Python counterfactual rebuild of the same accepted steps with only
`servo_reference_velocity_deg_s=30`; it is not an Isaac result.

| Version | Steps | Source events | Raw duration (s) | Fast @ 150 (s) | Fast @ 30 (s) | 30 minus 150 (s) |
|---|---:|---:|---:|---:|---:|---:|
| `v003_20260805_224517_157723_manual` | 24 | 136 | 317.933333 | 78.522667 | 93.483333 | 14.960667 |
| `v005_20260805_225441_439112_manual` | 22 | 139 | 331.266667 | 79.637333 | 94.296667 | 14.659333 |
| `v006_20260805_233948_654778_manual` | 22 | 135 | 337.200000 | 73.254000 | 85.800000 | 12.546000 |
| `v007_20260806_190636_100857_manual` | 21 | 113 | 303.866667 | 74.650667 | 88.576667 | 13.926000 |
| `v008_20260806_211408_578700_manual` | 23 | 142 | 280.400000 | 78.515333 | 95.053333 | 16.538000 |
| `v009_20260806_215232_433234_manual` | 23 | 154 | 337.200000 | 75.951333 | 91.256667 | 15.305333 |
| `v010_20260806_220745_363972_manual` | 26 | 168 | 326.066667 | 82.467333 | 98.863333 | 16.396000 |
| `v011_20260806_223621_672618_manual` | 28 | 185 | 398.133333 | 84.588667 | 100.120000 | 15.531333 |
| `v012_20260806_231025_027004_manual` | 19 | 118 | 249.000000 | 69.699333 | 81.030000 | 11.330667 |

Totals: 208 accepted steps and 1,290 source events.

## 2. What timing the recordings actually preserve

All 208 accepted rows satisfy the following static invariants:

- `duration == recording_timing.actual_duration_s == max(events[*].time)`.
- Every event has `time == actual_recording_time_s`.
- Event times are monotonic within a step and remain in `[0, duration]`.
- Every event says `recording_timing_source == "simulation_time"`.
- Within each step, `command_start_sim_time - time` is constant; therefore the
  relative event clock and its absolute simulation-time origin agree.
- Every event records `canonical_servo_velocity_deg_s == 150.0`.
- Every row contains before/after articulation snapshots, including root and
  joint state and `sim_time`.

The timestamps originate in `sim_ui_controller.py:860-889`. Event metadata is
written at `sim_ui_controller.py:919-926`, event relative/actual times at
`sim_ui_controller.py:2311-2329`, and the final step timing block at
`sim_ui_controller.py:2622-2638,2753-2760`.

Important limitations of the saved fields:

- All 1,290 event-level `wheel_active_duration_s` keys are present but are
  `null`. The row-level `recording_timing.wheel_active_duration_s` equals the
  whole step duration for all 208 rows; it is not a per-event wheel interval.
  The 832 `motion_semantics.wheel_records[*].active_duration_s` values are
  finite, and the Fast planner also derives wheel-active intervals from the
  event state/timestamps.
- `sim_state_after.sim_time - sim_state_before.sim_time` exceeds `duration` by
  0.2 to 1.2 seconds. Some curated versions also contain backward absolute
  simulation-time jumps between rows. The endpoint snapshots are valid
  per-step evidence, but their absolute times must not be concatenated into one
  continuous source trajectory.
- Version directories contain accepted JSONL and metadata (plus a few backup
  copies), not continuous root/joint/contact telemetry or video. See
  `reports/recording_audit_50mm.md:3,12,33` and the per-version
  `standalone_telemetry_files: []` entries in
  `reports/recording_audit_50mm.json`.

## 3. Legacy 30 deg/s bookkeeping versus selected 150 deg/s runtime

The two values have different evidentiary meanings:

- `config/fsm_recording_baseline.yaml:16-23` defines the legacy bookkeeping
  profile `fixed-linear-command-space-30deg-s-v1` with 30 deg/s.
- `config/real_robot_motion_reference.yaml:7` selects 150 deg/s, with no servo
  velocity limit. `motion_speed.py:18,60`, `sim_robot_adapter.py:306-308`, and
  `sim_robot_adapter.py:1337-1345` consume that runtime reference.
- The recorder writes the selected motion-reference value into each event at
  `sim_ui_controller.py:919-923`; all 1,290 saved values are 150.
- The recorder separately copies the legacy baseline profile ID into
  `motion_semantics.servo_records` at `sim_ui_controller.py:1037-1047` and sets
  each `completion_time_s` to the whole step's `reference_duration_s`.

Across the nine versions, all 1,664 servo records have the legacy 30-deg/s
profile ID, and all 1,664 completion times equal their step endpoint. They are
bookkeeping/provenance, not measured per-joint completion times and not proof
that a joint moved at 30 deg/s. Version-level `metadata.json` instead identifies
`real-robot-ui-controller-20260626-v1` and `fixed_100_percent`.

The current planner computes

```text
servo_duration = maximum_changed_joint_delta / servo_reference_velocity
segment_duration = max(servo_duration, wheel_duration, explicit_hold)
```

at `playback.py:159-191`. Fast compaction preserves wheel-active intervals,
same-step repeated-servo intervals, and explicit waits, while removing other UI
idle gaps and step-boundary delay (`sequence_model.py:267-274,384-395`). This is
why a 30-deg/s counterfactual does not make the whole plan five times longer,
but still shifts subsequent phases by 11.33 to 16.54 seconds. Such a shift can
change servo/wheel concurrency, contact transitions, load transfer, COM/root
motion, and the full time history even if final commands are identical.

Current environment fingerprinting correctly classifies the 30-deg/s value as
`metadata_only_not_runtime_selection` at
`fsm_50mm_recording_derived_v3/environment_equivalence.py:468-477`.

## 4. Read, execution, and comparison path

The formal recording replay path is:

1. `run_fsm50.py:2397-2403` loads the accepted steps and runtime motion
   reference, then requests the authoritative Fast plan.
2. `recording_fast_plan.py:103-176` delegates to
   `playback.plan_from_steps(profile="fast")` and exports each segment's source
   mapping, start/end, servo duration, wheel duration, and expected wheel
   displacement.
3. `playback.py:1302-1312` recomputes each servo duration at actual segment
   start from adapter command state and effective runtime speed.
4. `playback.py:1003-1288` advances on measured completion and may extend a
   segment for bounded contact residual/stall handling.
5. `playback.py:1502-1532,1757-1787` records planned/actual segment and command
   times, scheduler lateness, and completion extension. `run_fsm50.py:2237-2241,
   2285` places the non-compact trace in `result.scheduler_status`.
6. `fsm50_telemetry.py:683-724,953-995` records source alignment, planned
   targets, measured joints, wheel motion, and contact evidence per sample.

The existing A/A-B converter already compares complete sampled arrays, not
only endpoints: `environment_ab_artifacts.py:1010-1135` extracts root, joint,
wheel, contact-class, and contact-force histories, while
`environment_ab_artifacts.py:1138-1183,1431-1444` requires identical contiguous
physics grids. `environment_equivalence.py:594-671` derives B tolerances from
A1/A2 self-error.

However, the converter does not currently consume `scheduler_status` or its
timing trace. Consequently it does not independently compare target/applied
target history, measured joint/wheel/root velocity, actual segment boundaries,
scheduler lateness, completion extension, or contact-transition time. Also,
the numeric comparison reduces each whole metric to global maximum/RMS errors,
which does not localize a phase error.

There is an additional consistency risk: `run_fsm50.py:2398-2403` loads the
motion reference but passes only the wheel limit to `fast_plan_rows`;
`recording_fast_plan.py:103-114` has no explicit servo-speed argument and thus
uses the `playback.py:111` import-time default. The current values both resolve
to 150, but a formal gate must assert equality instead of relying on that
coincidence. Runtime segment duration can also be recomputed without rewriting
the static planned start/end fields, making the execution trace essential.

## 5. Current proof boundary

The present artifacts can prove:

- integrity of the accepted source hashes and saved command-event timing;
- the exact raw-to-Fast transformation and its source-event mapping;
- the selected 150-deg/s command reference in the saved events and current
  configuration;
- per-step before/after articulation endpoints;
- what a nominal 150-deg/s Fast plan schedules statically.

They cannot yet prove:

- the original recording's continuous measured servo velocity or joint path;
- continuous root/COM, wheel, contact-class, or force history inside each saved
  step;
- equivalence of a completed physical replay to the original physical motion;
- successful A1/A2/B environment equivalence for any of the nine versions;
- physical success of all nine recordings.

In particular, neither the old 30 profile ID nor the saved 150 command reference
is measured full-trajectory evidence. The former is stale bookkeeping; the
latter proves the selected command semantics only.

## 6. Fail-closed gate for future full-time-trajectory claims

A future report may claim full-time-trajectory equivalence only if every gate
below passes for each recording:

1. **Source timing integrity:** exact accepted-steps SHA; all duration/time
   invariants above; complete monotonic source-event mapping; no missing or
   non-finite required field.
2. **Explicit speed lock:** event canonical speed, planner speed, plan timing,
   runtime environment readback, adapter requested speed, and adapter effective
   speed must all equal 150. Legacy 30 remains metadata-only. Any mismatch
   fails before execution.
3. **Exact plan reproduction:** rebuild with an explicit 150-deg/s parameter;
   require the full event/segment table and plan SHA to match, not merely final
   time or final pose.
4. **Complete execution trace:** every plan segment and command has exactly one
   finite, monotonic planned/actual start and end; record effective speed,
   recomputed delta/duration, dispatch jitter, and classified completion
   extension. Missing, duplicate, or reordered rows fail.
5. **Common physics grid:** every tick from pre-command through post-settle has
   contiguous `sample_index`, `sim_step`, and simulation time. Persist requested
   command, rate-limited applied target, actual articulation target, measured
   joint/root/wheel position and velocity, wheel travel, contact class, and
   finite common contact force.
6. **A/A-B full-trajectory comparison:** use two independent formal runs
   (A1/A2) and one instrumented run (B) with the same recording SHA, plan SHA,
   source closure, runtime versions, device, physics grid, and initial state.
   Compare every sample and both A-to-B pairs.
7. **Phase-sensitive criteria:** in addition to max/RMS error, require bounded
   time-integrated and sliding-window error; compare every command/segment/contact
   boundary; reject cross-correlation lag greater than one physics tick. Do not
   use time warping to hide phase shifts.
8. **Closed artifacts:** only finalized normal-exit batches with verified
   checksums, complete runtime readback, and a real viewport MP4/manifest may
   enter the comparison. Video is corroborating evidence, not a substitute for
   numeric telemetry.

Because the historical recordings contain no continuous physical telemetry,
new duplicate 150-runtime formal replays are required to establish a physical
time-series baseline. Until those artifacts exist and pass every gate, the
correct status for all nine versions remains **PENDING_PHYSICAL_REPLAY**.
