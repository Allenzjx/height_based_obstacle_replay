# Current Implementation Gap Audit

Audit basis: clean starting commit
`92abb432a10d15e82bd9490495b348b5cf9e8953`; current findings describe the
2026-08-09 working tree. Code-contract completion and physical validation are
tracked separately.

## Executive finding

The starting checkpoint was not a runnable closed-loop FSM. The working tree
now includes the controller, observation layer, atomic executor, guard
registry, restore/provenance layer, runtime integration, CLI commands, replay
resume/finalization, actual active-viewport capture contract, and environment
A/A--A/B converter. The final pre-Isaac regression at this checkpoint was
**197 passed + 44 subtests** for the focused FSM/telemetry set and
**495 passed + 54 subtests** for the complete repository suite.

That implementation work does not close the physics gap. All 57 states remain
`PENDING_REPLAY`; there is no successful environment A1/A2/B set, reliable
recording replay, state run, full run, or 5/5 run.

## Audit resolution table

| # | Audit item | Starting checkpoint | Current truthful finding |
|---:|---|---|---|
| 1 | CLI commands | `audit`, `replay-recordings`, `report` | Adds `run-fsm`, `test-state`, `validate-5`, and `validate-environment` |
| 2 | Real A0→F5 controller | Absent | `fsm50_controller.py` implemented and pure-tested; no successful real run |
| 3 | Entry/update/exit lifecycle | Absent | Implemented with retry/timeout/SAFE_STOP behavior |
| 4 | Unified observation | Absent | `fsm50_observation.py` implemented with strict evidence handling |
| 5 | Atomic Servo+Wheel executor | Absent | `fsm50_executor.py` implemented with fake-adapter contract tests |
| 6 | Complete guard registry | Absent | `fsm50_guard_registry.py` implements resolution and startup validation |
| 7 | Restore/prefix provenance | Absent | `fsm50_state_restore.py` and `test-state` validation implemented |
| 8 | Replay resume/durable artifacts | Partial | Resume, lifecycle, source-freeze, checksum, and failure artifacts implemented; 0/9 reliable runs |
| 9 | Real viewport video | Telemetry plots only | Per-run active GUI viewport MP4, manifest, SHA, and container checks implemented; no real qualifying artifact |
| 10 | Environment fingerprint | Partial offline lock | Static fingerprint and strict runtime converter/report writer implemented |
| 11 | Environment A/A--A/B | Not run | A1/A2/B all absent; report has no real `PASS` |
| 12 | Formal grounding parity | Historical path could pre-seed | Formal path no longer invokes `_seed_adapter_from_locked_ground_pose`; clean live settle/reference is authoritative |
| 13 | Impulse generator | Guard/detector only | Command generator implemented, but not selected across all recordings or wired to live corrections/IK |
| 14 | Anchored support-angle generator | Guard/detector only | Command generator implemented with the same selection/runtime-wiring gap |
| 15 | Per-leg IK/live correction | Helpers only | Still not integrated into the COM transfer generators; runtime completion cannot be claimed |
| 16 | Per-state provenance | Mainly v012/default metadata | All 57 states remain `PENDING_REPLAY`; no real replay-backed selection |
| 17 | State validation | None | 0/57 |
| 18 | Full deterministic validation | None | 0 full runs; `validate-5` is 0/5 |

## Remaining critical implementation/evidence gap

The COM impulse and anchored support-angle generators are now real command
generators, not placeholders. However, their parameters have not been selected
from and validated against all nine recordings, and their command stream is not
yet adjusted by live correction feedback or per-leg IK acceptance. Until those
connections are implemented and replay-proven, they remain **PARTIAL**, not
runtime-complete.

Likewise, static reachability and guard tests do not upgrade the 57 YAML states:
every state remains `PENDING_REPLAY` until its source version/step/event/interval
and physical transition are demonstrated in durable real-Isaac evidence.

## Existing artifacts are not physical completion

- `reports/recording_fast_plans/*` are offline derived plans, not live replays.
- `reports/RECORDING_VERSION_MATRIX_50MM.csv` and
  `RECORDING_PHASE_ALIGNMENT_50MM.csv` are analysis inputs, not proof that
  selected primitives connect physically.
- The only replay directory,
  `runs/recording_replays/20260808T005031_049278Z_recording_replays_bab6fb2016`,
  is a stale `.partial` batch. It failed during initial ground initialization
  before recording commands; `results_so_far` is empty.
- There are no complete A1/A2/B artifacts, no nine-version replay set, no
  state/full result, and no five-run result.

## Environment source diff

`git diff ab7ed11..92abb432 -- sim_obstacle_scene.py` contains exactly two
semantic changes:

1. Add `contact_sensor_factory: Any | None = None` to `SimSceneConfig`.
2. When telemetry contact sensors are enabled, use the supplied factory or
   fall back to the existing `create_robot_contact_sensor`.

That range changes no physical scene, robot, actuator, gravity, timestep,
solver, material, obstacle, or spawn parameter. Runtime equivalence is still
unproven because the required real A1/A2/B trajectories do not exist.

## Current evidence standard

The environment converter fails closed unless A1/A2 are formal aggregate
contact runs and B is the instrumented run, all three share the recording SHA,
Fast plan/source version, source-freeze closure and git HEAD, device, runtime,
simulation/sample grid, physical configuration, and real active-viewport video.
It then evaluates root/joint/wheel/final-pose/geometry/contact metrics against an
A/A-derived tolerance. The converter is implemented and tested; without the
three real artifacts its project-level status remains
`PENDING_RUNTIME_A_B`, never `PASS`.

## Baseline validation

- Starting checkpoint: 44/44 unit tests.
- Current working-tree checkpoint used here: focused FSM/telemetry
  **197 passed + 44 subtests**; complete repository
  **495 passed + 54 subtests** in `env_isaaclab`.
- No Isaac simulation was run to produce this documentation update.
