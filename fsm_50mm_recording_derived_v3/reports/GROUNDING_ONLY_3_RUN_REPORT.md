# Grounding-only clean-reset qualification report

Status: **FAIL (0/3 PASS); A1/A2/B NOT STARTED**

Run date: 2026-08-12

Frozen source: `af1bb751134eaffed6114ed6e3625f6dddbdcc20`

Environment lock SHA-256:
`a029c9d02f545314af4885981e8cde24ddd9e333947f433469ab78bded35228f`

## Qualification contract

- Three independent supervised Isaac Sim 5.1 processes; no in-process reset
  retry was counted as a clean reset.
- Physics `dt = 1/120 s`, maximum 180 ticks, rolling diagnostic window 60
  frames, and 10 consecutive strict-stable ticks required for early exit.
- Existing limits were unchanged: root vertical speed `0.01 m/s`, servo joint
  speed `0.02 rad/s`, wheel joint speed `0.20 rad/s`, ground clearance
  `0.002 m`, and maximum penetration `0.003 m`.
- Historical grounded root-pose seeding was disabled. Formal initialization
  also performed no root-Z correction; penetration or clearance failures are
  rejected fail-closed.
- The diagnostic artifact is distinct from the qualification artifact. A
  diagnostic `.finalized` marker does not qualify a failed run; each failed
  qualification has a `.failed` marker and `qualified=false`.

## Results

| Trial | Result | Ticks | Stable count | Trace | Video | Preclose | Shutdown |
|---:|---|---:|---:|---:|---|---|---|
| 1 | FAIL | 180 | 0/10 | 180 valid rows | 26 frames, valid MP4 | 22 files / 44 checksum rows | `FAST_EXIT_VERIFIED` |
| 2 | FAIL | 180 | 0/10 | 180 valid rows | 26 frames, valid MP4 | 22 files / 44 checksum rows | `FAST_EXIT_VERIFIED` |
| 3 | FAIL | 180 | 0/10 | 180 valid rows | 26 frames, valid MP4 | 22 files / 44 checksum rows | `FAST_EXIT_VERIFIED` |

All three `grounding_telemetry.jsonl` files have the same SHA-256:
`7187477b44d29b5964e4565751df13803d6889c1fdaad4defaea1906958e73d0`.
The failure is therefore exactly reproducible under this frozen configuration.

## Terminal physical evidence

The following values are identical in all three trials:

| Metric | Observed | Existing limit | Gate |
|---|---:|---:|---|
| Maximum final servo speed | 0.069964 rad/s | 0.020000 rad/s | FAIL |
| Acceptance-window maximum servo speed | 0.069970 rad/s | 0.020000 rad/s | FAIL |
| Rolling-60 maximum servo speed | 0.095783 rad/s | 0.020000 rad/s | FAIL |
| Final root vertical speed | 0.006041 m/s | 0.010000 m/s | PASS |
| Final maximum wheel speed | 0.029033 rad/s | 0.200000 rad/s | PASS |
| Maximum collision penetration | 0.001049 m | 0.003000 m | PASS |
| Non-wheel contact rows | 0 | 0 required | PASS |

Final upward wheel forces were FL `7.4024 N`, FR `6.8224 N`, RL `6.9854 N`,
and RR `7.5391 N`. Thus the robot was load-bearing rather than motionless in
free space. The terminal ground diagnostic was nevertheless classified
`VISUAL_ONLY_INTERSECTION`; visual AABB intersection was not reclassified or
repaired by a root-pose write.

## Per-servo analysis

The table is identical for all three trials. A servo is a terminal offender
when `abs(final qd) > 0.02 rad/s`.

| Servo | Final qd (rad/s) | First post-step target-q (rad) | Final target-q (rad) | Peak abs target-q (rad) | Command-to-PhysX target error | Terminal offender |
|---|---:|---:|---:|---:|---:|---|
| front_left_hip | -0.031850 | -0.003840 | -0.001785 | 0.008031 | 0 | yes |
| front_left_knee | +0.045743 | -0.001805 | +0.006316 | 0.012720 | 0 | yes |
| front_right_hip | -0.019739 | -0.004653 | +0.000474 | 0.008421 | 0 | no |
| front_right_knee | +0.031893 | -0.000760 | +0.005570 | 0.013364 | 0 | yes |
| rear_left_hip | +0.069964 | -0.000267 | +0.003712 | 0.004488 | 0 | yes |
| rear_left_knee | -0.062118 | -0.002942 | -0.006955 | 0.022479 | 0 | yes |
| rear_right_hip | +0.048005 | -0.000393 | +0.003038 | 0.003880 | 0 | yes |
| rear_right_knee | -0.040079 | -0.002668 | -0.006650 | 0.022791 | 0 | yes |

The shared adapter takes `command_zero_joint_pos` from the live articulation
joint position, constructs the zero servo targets from it, and writes those
targets before the first settling physics tick. The actual PhysX drive
readback matched the requested command on every tick for every servo; wheel
velocity targets likewise remained zero and matched PhysX readback. The small
`target-q` difference after physics begins is load-dependent articulation
deflection, not a command-buffer/write mismatch. Consequently, the evidence
does not justify another target-to-current-q synchronization phase: doing so
would remove part of the supporting actuator error while the robot is loaded.

## Per-tick evidence coverage

All 540 recorded ticks contain finite/evidence-valid root position and
orientation, linear and angular velocity, roll and pitch, 12-joint `q`, `qd`,
actual PhysX target and `target-q`, command/target readback error, four wheel
`net_forces_w` vectors, non-wheel contact evidence, per-wheel penetration,
rolling-window metrics, and consecutive stable count. Each run rendered 23
settling ticks and encoded 26 unique viewport frames.

## Artifacts

### Trial 1

- Batch: `../runs/grounding_only/20260812T040419_064662Z_grounding_only_trial_01_221758a6ca`
- Telemetry: `diagnostic_artifact/grounding_telemetry.jsonl` and `.csv`
- Joint analysis: `diagnostic_artifact/joint_velocity_analysis.json`
- Video: `diagnostic_artifact/fsm50_viewport.mp4`
- Qualification: `qualification_artifact/result.json`
- Shutdown: `shutdown_outcome.json`

### Trial 2

- Batch: `../runs/grounding_only/20260812T040839_603858Z_grounding_only_trial_02_5c46205340`
- Telemetry: `diagnostic_artifact/grounding_telemetry.jsonl` and `.csv`
- Joint analysis: `diagnostic_artifact/joint_velocity_analysis.json`
- Video: `diagnostic_artifact/fsm50_viewport.mp4`
- Qualification: `qualification_artifact/result.json`
- Shutdown: `shutdown_outcome.json`

### Trial 3

- Batch: `../runs/grounding_only/20260812T041221_837907Z_grounding_only_trial_03_957540f80a`
- Telemetry: `diagnostic_artifact/grounding_telemetry.jsonl` and `.csv`
- Joint analysis: `diagnostic_artifact/joint_velocity_analysis.json`
- Video: `diagnostic_artifact/fsm50_viewport.mp4`
- Qualification: `qualification_artifact/result.json`
- Shutdown: `shutdown_outcome.json`

## Replay timing boundary

The separate static timing audit found 208 accepted steps and 1,290 source
events across the nine recordings. Every accepted step preserves a duration;
every event preserves a simulation-time-relative timestamp; and all 1,290
events record a canonical command speed of 150 deg/s. The 1,664 nested
`servo_records` carrying a 30-deg/s profile are legacy bookkeeping, not measured
velocity. Rebuilding at hypothetical 30 deg/s changes the nine Fast-plan
durations by 11.331 to 16.538 seconds, so endpoint-only comparison is invalid.

The historical versions do not contain continuous physical telemetry, so full
original physical trajectory equivalence cannot yet be established. The
required future comparison is a complete common-grid time series including
commands, actual targets, q/qd, root/wheel motion, segment/contact boundaries,
and force histories. Because grounding is 0/3, no A1/A2/B or recording replay
was run and this physical timing comparison remains pending.

## Gate decision

The grounding gate is closed. No A1/A2/B run, nine-recording replay, or full
FSM tuning is admissible from these artifacts. These results do not establish
17 primitives, diagonal support order, RL-lift forward-FR COM transfer,
impulse/anchored-COM selection, or a physically completed 57-state FSM.
