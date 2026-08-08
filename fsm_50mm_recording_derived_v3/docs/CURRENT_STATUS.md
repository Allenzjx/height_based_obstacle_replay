# 50 mm Recording-Derived FSM — Current Status

Last updated: 2026-08-08 (America/New_York)

## Recovery checkpoint

- Repository: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay`
- Working directory: `fsm_50mm_recording_derived_v3`
- Starting commit: `92abb432a10d15e82bd9490495b348b5cf9e8953`
- Starting branch/worktree state: `main`, clean, aligned with `origin/main`
- Recent commits: `92abb43`, `ab7ed11`, `74e5a3b`
- Active Isaac/Kit/Python simulation or training process at audit start: none found
- PPO work: not started and out of scope until the FSM reaches 5/5 physical validation

## Verified existing work

- Offline recording audit and Fast-plan artifacts exist for all nine physical 50 mm version directories (`v003`, `v005`–`v012`).
- Existing recording parsing, alignment, Fast-plan decoding, support/contact classification, traversal evidence, telemetry, and guard/detector helpers are present.
- Baseline package tests run with `python -m unittest discover -s fsm_50mm_recording_derived_v3\\tests -t . -v`: **44/44 passed** on 2026-08-08.
- `sim_obstacle_scene.py` differs from `ab7ed11` by only the optional `SimSceneConfig.contact_sensor_factory` field and selection of that factory in the telemetry-sensor creation branch. Runtime A/B equivalence has not yet been established.

## Existing simulation evidence

- One supervised replay batch exists at `runs/recording_replays/20260808T005031_049278Z_recording_replays_bab6fb2016`.
- It failed closed during the initial grounded settle check before any recording command began (`results_so_far` is empty and `recording_commands_started` is false).
- Therefore **zero recording versions currently have a reliable clean live Fast Replay result**. Offline plans/audits are not replay success.
- There is no valid environment A/B result, state-level FSM result, full-FSM result, 5-run result, or camera video yet.

## Current blocking gates

1. Complete implementation-gap audit and guard/profile/provenance validation.
2. Implement and unit-test the real controller, observation layer, atomic executor, command-side transfer primitives, SAFE_STOP, and runtime CLI.
3. Implement environment fingerprinting plus scene-factory regression tests.
4. Diagnose the clean-reset settle failure without changing physical parameters.
5. Obtain A/A baseline repeatability and A/B instrumentation equivalence before full FSM execution.
6. Replay every physical recording version with resumable durable results, then select and physically verify primitives.
7. Run provenance-backed state-level tests, followed by five clean full-FSM runs.

## Truthful completion state

The project is in active implementation/audit. It is **not** physically complete: no recording version replay, state-level run, or full FSM run has yet passed in this checkpoint.

