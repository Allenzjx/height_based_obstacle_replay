# 50 mm recording-derived FSM checkpoint summary

Last updated: 2026-08-09 (America/New_York)

## Outcome

The existing `fsm_50mm_recording_derived_v3` project was extended in place and
checkpointed; it was not regenerated. The requested controller/runtime and
fail-closed evidence contracts now exist and pass the repository's pure-Python
tests. Physical completion is **not** claimed.

The first formal A1 run exposed a reproducible clean-grounding timing blocker
before any replay command. Environment equivalence therefore remains
`PENDING_RUNTIME_A_B`; all nine recording replays, all 57 states, the full FSM,
and the five-run gate remain pending.

## Starting checkpoint versus current code

The starting commit `92abb432a10d15e82bd9490495b348b5cf9e8953`
contained recording audit/alignment/Fast planning, contact telemetry, support
classification, COM guard helpers, a state model/YAML, and a replay runner. It
did not contain a real controller, unified observation, atomic FSM executor,
complete guard registry, trusted state restore layer, actual state/full/5-run
runtime, robust resume/finalization, or qualifying viewport-video evidence.

Code checkpoint `4b9e1549164c3c79da78d411d03a1ef8094d3c2d`
adds those missing contracts through:

- `fsm50_controller.py`
- `fsm50_observation.py`
- `fsm50_executor.py`
- `fsm50_guard_registry.py`
- `fsm50_state_restore.py`
- `fsm50_isaac_runtime.py`
- `environment_equivalence.py`
- `environment_ab_artifacts.py`
- command generators in `com_transfer_primitives.py`
- expanded `run_fsm50.py`, configuration, tests, and tracked documentation

Checkpoint `6616e9c52178923c74e7930d2cb6667acf627718` then hardens
the supervised shutdown outcome: only `CLOSE_RETURNED` with return code zero is
`NORMAL_EXIT`; a preclose-only exit can no longer masquerade as normal closure.

The controller resolves all configured guards at startup, drives state
entry/update/exit, emits one full atomic Servo+Wheel batch per physics tick,
requires live evidence for transitions, tracks ordered leg traversal, retries
within bounds, and enters latched SAFE_STOP on abort/timeout/execution failure.

## Environment audit

`git diff ab7ed11..92abb432 -- sim_obstacle_scene.py` has exactly two semantic
changes: add optional `SimSceneConfig.contact_sensor_factory`, and select that
factory instead of the original `create_robot_contact_sensor` only when one is
supplied. With `factory=None`, the original constructor remains selected. The
diff changes no physics, geometry, robot USD, spawn pose, material, solver,
actuator, wheel, sign, timestep, or render parameter.

Static values are locked to the current formal project: `dt=1/120 s`, render
interval `8`, servo command interpolation `150 deg/s`, servo stiffness/damping
`600/60`, wheel damping `20`, wheel reference/limit
`0.5235987756/2.0943951024 rad/s`, solver iterations `8/2`, root spawn Z
`0.04 m`, ground Z `0`, and the measured 50 mm obstacle bounds. The historical
`30 deg/s` and render interval `2` remain recording metadata, not runtime
authority.

The sensor's effect on a complete physical trajectory is still unknown because
no admissible A1/A2/B set exists. It is therefore neither claimed equivalent
nor claimed perturbing.

## Real A1 result

The formal GUI A1 command used v012, `contact_mode=formal`, real-video enabled,
clean Git/source/USD closure, and no historical root seed. It created:

```text
runs/environment_equivalence/A1/20260810T004238_372852Z_recording_replays_c91edb0e48
```

Ground contact was physically safe, with maximum penetration about `1.055 mm`,
but the current shared timing computes `min(180, ceil(0.75/dt)) = 90` ticks
while retaining a 60-tick final window. That window still held the landing
transient, producing `stable_frames=0/10`, final-window maximum root vertical
speed `0.66814047098 m/s`, servo/joint speed `1.75069868565 rad/s`, and wheel
speed `0.82960325480 rad/s`. The final root sample was much closer to stable,
but no authoritative grounded reference was saved. The full retained trajectory
matches the old failed batch exactly.

Replay never began. The batch has complete failure checksums and source
integrity, but no per-version result, runtime readback, telemetry, visual
manifest, viewport MP4, or normal shutdown closure. Native close timed out
after `PRECLOSE_COMPLETE`; the supervisor terminated only its owned child PID
tree and recorded `SIMULATION_CLOSE_TIMEOUT`. The A/B consumer correctly
rejects the directory.

The native hang is after Replicator shutdown and simulation `onStop`, inside
`UsdContext.close_stage()` before Isaac 5.1 emits `Stage closed`. Twenty-nine
local Kit logs enter `SimulationApp.close`; none reaches that marker. Because
the failed A1 never constructed a replay viewport recorder, video cleanup is
not the cause. No unverified fast-close workaround or environment upgrade was
applied.

No grounding timing/window change was made because the task explicitly forbids
changing reset/grounding behavior. A shared change that allows the same
zero-command clean-reset articulation to continue toward the already configured
180-step bound is the evidence-backed next hypothesis, but it requires explicit
authorization and a new real A1 run.

## Validation scoreboard

| Requirement | Real result |
|---|---:|
| Static fingerprint | generated; runtime readback pending |
| Formal A1/A2 + instrumented B | `0/3`; first A1 attempt rejected |
| Contact-sensor trajectory equivalence | unknown / not measured |
| Recording replays | `0/9` reliable completed |
| Primitive selections | `0/17` physically verified |
| State-level runs | `0/57`; all `PENDING_REPLAY` |
| Full A0→F5 runs | `0` |
| Clean deterministic runs | `0/5` |
| Qualifying real viewport videos | `0` |

Pure-Python results are `197 passed + 44 subtests` for the focused FSM/telemetry
set and `495 passed + 54 subtests` for the complete repository suite. They do
not change the table above. After the close-outcome hardening, the affected set
is `231 passed + 47 subtests` and the runner suite is `27 passed + 9 subtests`.
Two complete combined reruns each had `495 passed + 57 subtests` plus one
transient Tk GUI initialization failure; the failing tests passed alone, and no
FSM/shutdown assertion failed. A fully green post-change combined rerun is not
claimed.

## Primitive, diagonal, and COM truth

No final primitive provenance is claimed because no recording has a reliable
live replay. No final diagonal sequence is claimed. The runtime contains both
command-side `IMPULSE_REACTION_TRANSFER` and anchored support-angle generators,
but no transfer state has been assigned one of them from physical all-recording
evidence. Live correction feedback and per-leg IK acceptance remain unwired to
those generators, so they remain partial.

## Remaining blockers

1. Decide whether to authorize a shared formal grounding-time semantic change
   from the present 90-tick hard cap toward the configured 180-step deadline,
   without changing thresholds, pose, physics, or using a seed.
2. Establish an acceptable native-close lifecycle that remains truthful if
   `SimulationApp.close` does not return after durable preclose evidence.
3. Obtain admissible A1, A2, and B runs and a real environment-equivalence
   `PASS`.
4. Replay all nine recordings, select compatible primitives, wire live
   corrections/IK, validate states, then attempt full FSM and 5/5.

PPO and residual policy work remain prohibited until the final gate is truly
5/5.
