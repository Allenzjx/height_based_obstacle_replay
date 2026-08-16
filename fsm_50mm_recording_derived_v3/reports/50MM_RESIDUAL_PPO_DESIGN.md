# 50 mm FSM Residual PPO Design and R1 Authority

## Status and scope

This document defines residual **authority**, not a PPO result. No PPO policy
has been trained, evaluated, exported, or admitted. The authority module is
pure Python and is not integrated with the macro controller, worker, scene, or
Isaac runtime.

The nominal recording-derived FSM remains the action authority. The envelope
only returns a `ResidualPhaseContract`; the sole final command composer is
`fsm50_direct_command_residual.compose_direct_command_residual`. The
envelope's `authorize()` method is an offline clip/slew preview and never
returns a physical command map. A zero residual through the sole composer
reproduces the nominal 12-D target exactly. Boundary, hold, retry, safe-stop,
success, unknown-source, excluded-source, unverified-evidence, and
unlisted-state decisions all receive an immediate all-zero contract.

Canonical config:

- Schema: `fsm50.residual_envelope.v1`
- Envelope: `fsm50-residual-r1-reviewed-success-sources-v1`
- Canonical payload SHA256:
  `fa5002690737d94fab7304f40044293da46de34cd197af19f3f1140047ae7fbe`
- Config: `configs/fsm50_residual_r1_envelope.json`
- Authority: `fsm50_residual_envelope.py`

Only these reviewed-success sources are admitted:

1. `v003_20260805_224517_157723_manual`
2. `v008_20260806_211408_578700_manual`
3. `v009_20260806_215232_433234_manual`

`v010_20260806_220745_363972_manual`, every other v010 identity, unknown
sources, old policies, and legacy checkpoints are excluded. The code rejects a
checkpoint reference rather than accepting it as action truth.

## Evidence-derived scale

The v003/v008/v009 successful Gate-A profiles contain 324 nonzero adjacent
servo target changes and 141 nonzero adjacent wheel target changes:

| Quantity | Min | P10 | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Servo change (deg) | 0.5 | 1.1 | 4.3 | 20.1 | 47.6 |
| Wheel change (rad/s) | 0.30 | 0.30 | 0.30 | 1.07 | 2.094395 |

The support-servo residual cap of `1 deg` is below the median nominal servo
change. The active RL hip/knee cap of `2 deg` is still below the median and is
small relative to the successful RL traversal ranges. The wheel cap of
`0.10 rad/s` is one third of the smallest successful nonzero wheel change.

The authority additionally limits residual slew to `10 deg/s` for servo
channels and `0.5 rad/s^2` for wheel channels. These are implementation safety
limits, not learned values.

## Canonical 12-D command space

The direct command residual order is:

1. `front_left_hip` (deg)
2. `front_left_knee` (deg)
3. `front_right_hip` (deg)
4. `front_right_knee` (deg)
5. `rear_left_hip` (deg)
6. `rear_left_knee` (deg)
7. `rear_right_hip` (deg)
8. `rear_right_knee` (deg)
9. `front_left_ankle` wheel target (rad/s)
10. `front_right_ankle` wheel target (rad/s)
11. `rear_left_ankle` wheel target (rad/s)
12. `rear_right_ankle` wheel target (rad/s)

For an authorized source action, `phase_contract(...)` validates the context,
passes `subphase` through exactly, applies the source/state and nominal-wheel
gates, and returns bounds/rates in `ResidualPhaseContract`. The sole composer
then performs:

1. Clip each requested residual to the exact source/state mask.
2. Gate each wheel residual to zero whenever that wheel's nominal target is
   zero (epsilon `1e-12 rad/s`).
3. Slew-limit relative to the previous residual.
4. Add the applied residual to the nominal target and produce the final maps.

Default-zero contracts hard-zero all channels; an old residual therefore
cannot leak through a boundary or safe decision. No second physical
composition seam is defined by the envelope module.

## R1 source/state masks

Notation: active RL hip/knee channels are capped at `2 deg`; every other listed
servo channel is capped at `1 deg`; every listed wheel channel is capped at
`0.10 rad/s` and is still subject to the nonzero-nominal gate.

| Source/state | Authorized servo channels | Authorized wheel channels |
|---|---|---|
| v003 / S5 PRE_RR | none | none |
| v003 / S7 PRE_RL | none | none |
| v003 / S8 RL traverse | FL hip/knee, FR hip, RL hip/knee | FL |
| v003 / S10 posture | all eight at `1 deg` | none |
| v008 / S5 PRE_RR | FL hip/knee, FR knee, RL knee, RR knee | FR |
| v008 / S7 PRE_RL | none | none |
| v008 / S8 RL traverse | all eight; RL hip/knee at `2 deg` | all four |
| v008 / S10 posture | all eight at `1 deg` | none |
| v009 / S5 PRE_RR | FL hip/knee, RL knee, RR hip | none |
| v009 / S7 PRE_RL | RL knee | RL |
| v009 / S8 RL traverse | all eight; RL hip/knee at `2 deg` | all four |
| v009 / S10 posture | all eight at `1 deg` | none |

S5 in v003 and S7 in v003/v008 are explicitly zero because those old states
are not action-backed profiles. S10 deliberately gives all eight servo channels
a small recovery authority because all three reviewed successful runs ended
with `posture_incomplete=true`; it never gives S10 wheel authority.

## Actor observation and reward proposal

The initial actor should use only fields already present in deployment
telemetry plus its own previous residual:

- macro-state one-hot, reviewed source one-hot, profile fraction;
- active-leg one-hot and controller support-leg mask;
- base roll/pitch, root linear velocity, root angular velocity;
- eight servo positions, eight servo velocities, four wheel velocities;
- the twelve nominal command targets;
- each wheel's front-face and top clearance;
- geometry support-candidate count and body-crossed flag;
- previous 12-D residual.

This is an 85-D proposal. Wheel angle is omitted because continuous wheel
rotation is not a useful posture coordinate. Source identity is retained
because successful cross-version strategies are categorical and must not be
averaged.

Do not expose these unavailable or privileged quantities to the actor:

- exact/mass-weighted COM (`com_measurement_available=false`);
- wheel contact loads (`wheel_contact_load_available=false`);
- a true support-polygon/load margin;
- dangerous body collision, severe penetration, body stuck, or trapped-leg
  fields while their runtime source remains unavailable.

Wheel contact class and support count are geometry proxies, not measured load.
The critic may use simulator-only COM/contact state during training, but that
state must never become an actor dependency.

Proposed normalized phase-local reward terms are:

- S5/S7: roll/pitch, angular velocity, geometry support count, and a one-time
  bonus for entering the next traversal state;
- S8: ordered AIR, front-face-cross, and TOP events with bonuses `+1/+2/+4`,
  plus body crossing `+5`;
- S10: servo endpoint error, servo velocity, root settling velocity, four-wheel
  TOP/support geometry, and settling time;
- FSM task success `+10`;
- hard failure `-100` for fall, nonfinite state, joint-limit violation, unsafe
  target, or drive-up without required lift.

Continuous regularization proposal:

```text
-0.10 * ((roll / 20 deg)^2 + (pitch / 15 deg)^2)
-0.05 * min((root_angular_speed / 25 deg/s)^2, 4)
-0.02 * ((root_v_y / 0.10 m/s)^2 + (root_v_z / 0.10 m/s)^2)
-0.01 * ||normalized_residual||^2
-0.02 * ||delta_normalized_residual||^2
```

Forward velocity/drift is not penalized merely for being nonzero. Collision or
penetration must not be presented as an online reward/termination until a
reliable deployment-time sensor exists.

The proposed S10 completion window is ten consecutive 120 Hz frames with:

- maximum servo endpoint error at most `3 deg`;
- root linear speed at most `0.03 m/s`;
- root angular speed at most `5 deg/s`;
- four geometry support candidates and all four wheels classified TOP.

## Phase-local start evidence

The reviewed r4 directories contain full telemetry and transition evidence but
do not contain `fsm50.trusted_sim_state_before.v1` or
`fsm50.verified_prefix_replay_manifest.v1` packages. Consequently, telemetry
rows are not restore assets. The currently defensible start is clean S0 plus a
verified nominal prefix replay.

Candidate future trusted-snapshot capture coordinates are:

| Source | S5 | S7 | S8 | S10 |
|---|---:|---:|---:|---:|
| v003 | step 7181 / 59.8417 s (feedback-only) | 8460 / 70.5000 s (feedback-only) | 8461 / 70.5083 s | 10764 / 89.7000 s |
| v008 | 7180 / 59.8333 s | 8428 / 70.2333 s (feedback-only) | 8429 / 70.2417 s | 11405 / 95.0417 s |
| v009 | 6900 / 57.5000 s | 7572 / 63.1000 s | 7732 / 64.4333 s | 10869 / 90.5750 s |

Capture only v008/v009 for S5, only v009 for S7, all three sources for S8,
and all three source-specific states for S10. Do not average states across
sources. v003 repeats can validate restore determinism but are not independent
state diversity.

## SHA-bound evidence

| Source | Bundle | accepted_steps | reviewed verdict | task inputs | video | worker result |
|---|---|---|---|---|---|---|
| v003 | `5742cda6...90b78` | `06e13153...5b394` | `63141b53...f1782` | `10cb7ad9...b8848` | `9a446fbf...00f7` | `dc7dd6d4...aa30` |
| v008 | `0a47cee0...a8c41` | `737b23a2...ea8c` | `05721928...95b1` | `5b1c1d7e...cfa12` | `4920b7f2...f3a1` | `332ba356...bbc9` |
| v009 | `0e568635...e919` | `60db11a1...17c1` | `1fe1e673...aa93` | `6df6529d...b5cf` | `2168ba48...22c3` | `64c3ad74...ba0f` |

The config contains every full SHA and project-relative evidence path. Loading
with evidence verification enabled hashes all fifteen files, validates each
manual verdict's internal run/source/bundle/path relationships, and requires a
complete worker result with terminal outcome
`TASK_SUCCESS_POSTURE_INCOMPLETE`, complete source/segment coverage, verified
safe stop, and a quiesced video writer.

## R0 / R1 / R2 plan

### R0 — contract and zero-residual shadow

1. Keep all residuals identically zero.
2. Validate canonical ordering, evidence closure, exact nominal passthrough,
   default-zero transitions, and telemetry-only actor inputs.
3. Run the future policy in shadow only; its output cannot reach actuators.
4. Require byte/SHA-stable config and nominal trajectory equivalence before any
   physical authority is granted.

Current work completes only the pure authority/config/test portion of R0.

### R1 — independent phase-local training

Train and evaluate source-conditioned phase tasks separately in this order:

1. S5 PRE_RR: v008 and v009.
2. S7 PRE_RL: v009.
3. S8 RL COM shift/traverse: v003, v008, and v009.
4. S10 posture recovery: all three, with a dedicated bounded hold window.

Each task must first pass zero-residual equivalence, then nominal-plus-residual
hard-safety tests, then multi-seed evaluation. No R1 phase result grants full
FSM authority.

### R2 — combined FSM residual

Only after every R1 phase independently passes should a combined actor be
evaluated across complete v003/v008/v009 FSM runs. R2 must preserve source
conditioning, default-zero decisions, one source action per controller
decision, one physical batch, N+1 source accounting, and prior-zero boundary
semantics. Any future checkpoint must be newly produced under this evidence
chain and explicitly SHA-bound; legacy checkpoints remain inadmissible.

## Current blockers before PPO or physical claims

1. No SHA-bound PPO training run, admitted checkpoint, or policy evaluation
   report exists. This authority-only change does not implement or validate an
   actor, critic, reward, or training environment.
2. This envelope is not connected to controller/worker dispatch and has not
   run in Isaac or on hardware.
3. No macro-aware trusted phase snapshot packages exist; phase-local starts
   currently require clean S0 prefix replay.
4. v008/v009 S10 consists of a zero-duration wheel-stop action. A separately
   reviewed residual hold/settling window is required before S10 PPO authority
   is useful.
5. Exact COM, wheel load, true support margin, and reliable non-wheel
   collision/penetration sensing are unavailable in deployment telemetry.
6. Environment-lock and production bundle admission have not been updated for
   these new Gate-E files; this task intentionally did not edit or reseal them.
7. No full-regression, Isaac, audit, reseal, or physical promotion has been
   performed by this authority-only change.

These blockers are admission conditions. Passing the pure envelope tests is
not evidence of PPO performance or physical robustness.
