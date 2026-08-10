# 50 mm Recording-Derived FSM — Current Status

Last updated: 2026-08-09 (America/New_York)

## Recovery checkpoint

- Repository: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay`
- Working directory: `fsm_50mm_recording_derived_v3`
- Starting commit: `92abb432a10d15e82bd9490495b348b5cf9e8953`
- Runtime/code checkpoint: `4b9e1549164c3c79da78d411d03a1ef8094d3c2d`
- Supervisor close-outcome hardening:
  `6616e9c52178923c74e7930d2cb6667acf627718`
- PPO work remains out of scope until the FSM reaches 5/5 physical validation.

## Code milestone reached

The working tree now contains the requested pure-Python/runtime architecture:

- `fsm50_controller.py`: event-gated A0→F5 controller lifecycle, retries,
  timeouts, and SAFE_STOP transitions;
- `fsm50_observation.py`: unified strict observation model;
- `fsm50_executor.py`: atomic Servo+Wheel command execution contract;
- `fsm50_guard_registry.py`: guard resolution and startup validation;
- `fsm50_state_restore.py`: trusted restore/prefix provenance contracts;
- `fsm50_isaac_runtime.py`: live runtime integration and real active-viewport
  video capture support;
- `run_fsm50.py`: `run-fsm`, `test-state`, `validate-5`,
  `validate-environment`, replay resume, formal/instrumented contact modes, and
  durable artifact handling;
- `environment_equivalence.py` and `environment_ab_artifacts.py`: static
  fingerprinting plus fail-closed A/A--A/B artifact conversion/comparison;
- command-side impulse and anchored support-angle generators in
  `com_transfer_primitives.py`.

The focused FSM/telemetry Isaac-free suite at this checkpoint is
**197 passed + 44 subtests**. The complete repository suite is
**495 passed + 54 subtests** under `env_isaaclab`. These are code-contract
results only; they are not Isaac physics evidence.

After the A1 audit, the supervisor exit contract was tightened so only a
durable `CLOSE_RETURNED` handshake with child return code zero can become
`NORMAL_EXIT`. Its runner contract suite is **27 passed + 9 subtests**, and the
expanded affected FSM/telemetry regression is **231 passed + 47 subtests**.
Two post-change attempts to run the entire combined repository suite each
ended at **495 passed + 57 subtests + 1 Tk initialization failure**; the failing
GUI test differed between attempts and each passed immediately in isolation.
No FSM/shutdown assertion failed. This transient Tk ordering/resource issue is
recorded rather than reported as an all-green full-suite rerun.

## What remains provisional

- All **57 states remain `PENDING_REPLAY`**.
- The COM command generators are implemented, but parameters have not been
  selected and verified against all nine recordings.
- Those generators are not yet connected to live runtime correction feedback
  or per-leg IK correction/acceptance. They therefore cannot be described as
  runtime-complete transfer primitives.
- Offline recording audit and Fast-plan artifacts exist for all nine physical
  50 mm versions (`v003`, `v005`–`v012`), but none has a reliable completed
  live replay result.

## Existing simulation evidence

- The old replay batch
  `runs/recording_replays/20260808T005031_049278Z_recording_replays_bab6fb2016`
  remains a stale `.partial` artifact. It failed during initial grounding
  before any recording command started.
- A real formal A1 attempt was run on 2026-08-09 at
  `runs/environment_equivalence/A1/20260810T004238_372852Z_recording_replays_c91edb0e48`.
  It followed the unseeded production grounding chain, but failed after 90
  physics ticks with `stable_frames=0/10`. The final-window maxima were root
  vertical speed `0.6681404709815979 m/s`, servo/joint speed
  `1.7506986856460571 rad/s`, and wheel speed `0.8296032547950745 rad/s`.
  No replay command ran (`batch_results=[]`).
- That A1 attempt also ended with `SIMULATION_CLOSE_TIMEOUT` after the child
  wrote `PRECLOSE_COMPLETE`; the parent waited 60.062 seconds and terminated
  only its owned child PID tree. The batch has a checksum-complete failure
  evidence set, `.failed`, and no `.partial`, but it is not an admissible A1
  artifact because it has no version result, runtime readback, telemetry, or
  viewport MP4 and did not reach `NORMAL_EXIT`/`SHUTDOWN_COMPLETE`.
- The formal replay path no longer pre-seeds the historical locked grounded
  pose. Its production chain is clean reset → live settle/validation → save
  the live grounded reference. `_seed_adapter_from_locked_ground_pose` is not
  invoked by that path.
- Real active-viewport video production and validation code now exists, but no
  qualifying completed viewport video artifact exists yet.

## Physical validation scoreboard

| Gate | Current real-Isaac result |
|---|---:|
| Environment A1/A2/B | `0/3`; A1 attempted but rejected; `PENDING_RUNTIME_A_B` |
| Nine recording replays | `0/9` reliable completed |
| State-level validation | `0/57`; every state `PENDING_REPLAY` |
| Full A0→F5 run | `0` successful |
| Five clean full-FSM runs | `0/5` |

The environment converter is implemented and pure-Python tested, including
source-closure, sample-grid, contact-mode, and real-video admission checks.
`reports/ENVIRONMENT_EQUIVALENCE_REPORT.json` now exists as a static
`PENDING_RUNTIME_A_B` report with `environment_equivalent=false`; it is not a
runtime A/B result. There are no admissible A1/A2/B artifacts, so no report
with a real `PASS` exists.

## Next gates

1. Resolve the formal clean-grounding blocker without changing the locked
   robot root/standing pose, physical parameters, or reset/grounding behavior,
   and without using a historical-pose pre-seed. The present production
   contract caps the standard settle at 90 ticks while retaining a 60-frame
   terminal window, so that window still contains the landing transient even
   though the terminal root sample is nearly stable. Changing that shared
   timing contract requires explicit user authority.
2. Resolve or explicitly redefine the supervised native-close acceptance
   contract; the present A/B loader correctly rejects the observed
   `SIMULATION_CLOSE_TIMEOUT`.
3. Capture formal A1/A2 and instrumented B with matching source closure,
   recording, Fast plan, device, simulation grid, and real viewport video.
4. Obtain an environment-equivalence `PASS` from real artifacts.
5. Replay all nine recording versions and select/verify command parameters.
6. Wire and validate live corrections/IK, then execute all 57 state gates,
   a complete A0→F5 run, and finally 5/5 clean runs.

## Truthful completion state

The implementation milestone is substantial and pure-Python tested. The
project is still **physically unvalidated**: no environment A/B run, recording
replay, state, full FSM, or five-run gate has succeeded in real Isaac.
