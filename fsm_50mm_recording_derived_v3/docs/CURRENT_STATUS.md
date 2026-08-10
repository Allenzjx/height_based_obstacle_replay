# 50 mm Recording-Derived FSM — Current Status

Last updated: 2026-08-09 (America/New_York)

## Recovery checkpoint

- Repository: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay`
- Working directory: `fsm_50mm_recording_derived_v3`
- Starting commit: `92abb432a10d15e82bd9490495b348b5cf9e8953`
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

- The only replay batch remains
  `runs/recording_replays/20260808T005031_049278Z_recording_replays_bab6fb2016`.
- It is a stale `.partial` artifact that failed closed during initial ground
  initialization before any recording command started; `results_so_far` is
  empty.
- The formal replay path no longer pre-seeds the historical locked grounded
  pose. Its production chain is clean reset → live settle/validation → save
  the live grounded reference. `_seed_adapter_from_locked_ground_pose` is not
  invoked by that path.
- Real active-viewport video production and validation code now exists, but no
  qualifying completed viewport video artifact exists yet.

## Physical validation scoreboard

| Gate | Current real-Isaac result |
|---|---:|
| Environment A1/A2/B | `0/3`; `PENDING_RUNTIME_A_B` |
| Nine recording replays | `0/9` reliable completed |
| State-level validation | `0/57`; every state `PENDING_REPLAY` |
| Full A0→F5 run | `0` successful |
| Five clean full-FSM runs | `0/5` |

The environment converter is implemented and pure-Python tested, including
source-closure, sample-grid, contact-mode, and real-video admission checks.
There are no real A1/A2/B artifacts, so no
`ENVIRONMENT_EQUIVALENCE_REPORT.json` with a real `PASS` exists.

## Next gates

1. Diagnose and rerun clean grounding without changing locked physical
   parameters or using a historical-pose pre-seed.
2. Capture formal A1/A2 and instrumented B with matching source closure,
   recording, Fast plan, device, simulation grid, and real viewport video.
3. Obtain an environment-equivalence `PASS` from real artifacts.
4. Replay all nine recording versions and select/verify command parameters.
5. Wire and validate live corrections/IK, then execute all 57 state gates,
   a complete A0→F5 run, and finally 5/5 clean runs.

## Truthful completion state

The implementation milestone is substantial and pure-Python tested. The
project is still **physically unvalidated**: no environment A/B run, recording
replay, state, full FSM, or five-run gate has succeeded in real Isaac.
