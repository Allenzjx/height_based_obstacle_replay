# Current Implementation Gap Audit

Audit basis: clean commit `92abb432a10d15e82bd9490495b348b5cf9e8953`. This document is a live audit checkpoint and will be expanded with exact guard/profile/provenance tables before implementation is declared complete.

## Executive finding

The starting checkpoint is a substantial evidence, replay-runner, telemetry, state-schema, and safety-helper foundation. It is not yet a runnable closed-loop FSM. In particular, `fsm50_controller.py` does not exist and `run_fsm50.py` has no `run-fsm`, `test-state`, `validate-environment`, or `validate-5` command.

## Required audit items

| # | Question | Starting checkpoint finding |
|---:|---|---|
| 1 | Actual CLI commands | `audit`, `replay-recordings`, `report` |
| 2 | Real `run-fsm` command | No |
| 3 | Real `fsm50_controller.py` | No |
| 4 | Entry/update/exit runtime | No controller lifecycle runtime found |
| 5 | Every YAML guard mapped to predicate | No complete runtime registry found |
| 6 | Unknown/unimplemented guards | Registry validation absent; exact table pending below |
| 7 | Actual A0→F5 transitions | State ordering is data only; not executed by a controller |
| 8 | Command profiles executed | Replay executor exists; state command profiles are loaded/expanded but no FSM controller executes them |
| 9 | Servo/wheel trajectory executor | Official replay path has execution machinery; no dedicated FSM executor contract |
| 10 | Same-tick atomic Servo+Wheel | Official adapter/replay support exists; no FSM runtime wiring/fake-adapter proof yet |
| 11 | Impulse command generator | No; guard/detector helpers only |
| 12 | Anchored support-angle command generator | No; guard/detector helpers only |
| 13 | Active-leg isolation runtime wiring | No controller wiring; helper exists |
| 14 | Per-leg IK acceptance used live | No controller wiring; helper exists |
| 15 | SAFE_STOP runtime | State/config entry exists; no executable controller behavior |
| 16 | State restore/prefix replay | No state-level CLI/runtime |
| 17 | Five-run deterministic validation | No |
| 18 | Real camera video | No; manifest-based telemetry plots do not qualify |
| 19 | `PENDING_REPLAY` config | Global provenance and state/default evidence remain pending; detailed table pending |
| 20 | Provisional thresholds | Safety/event thresholds are labelled provisional pending replay; detailed table pending |
| 21 | States only from v012 | Starting config is primarily sourced from v012; detailed state table pending |
| 22 | Every profile used | Static usage audit pending |
| 23 | Valid source version/step/event/interval per state | Not established; config uses defaults/candidate metadata and lacks live replay provenance |

## Existing artifacts are not physical completion

- `reports/recording_fast_plans/*` are offline derived plans, not live replays.
- `reports/RECORDING_VERSION_MATRIX_50MM.csv` and `RECORDING_PHASE_ALIGNMENT_50MM.csv` are useful analysis inputs, not evidence that selected primitives connect physically.
- The only `runs/recording_replays/*` batch failed during grounded settle with no recording commands started.
- No `result.json`, `batch_results.json`, `runtime_environment_readback.json`, or `failure_diagnostics.json` with the required standard names was found under the FSM directory at audit start.

## Environment source diff

`git diff ab7ed11..92abb432 -- sim_obstacle_scene.py` contains exactly two semantic changes (four diff lines):

1. Add `contact_sensor_factory: Any | None = None` to `SimSceneConfig`.
2. When telemetry contact sensors are enabled, call `config.contact_sensor_factory` if supplied, otherwise call the existing `create_robot_contact_sensor`.

No physical scene/robot/actuator parameter was changed in this file by that commit range. Runtime trajectory equivalence remains unproven until A/A and A/B tests run.

## Baseline validation

- `python -m unittest discover -s fsm_50mm_recording_derived_v3\\tests -t . -v`: 44 tests passed.
- Base Python lacks pytest; `env_isaaclab` reports pytest 9.0.3. No package or environment was upgraded.

## Detailed static tables

Guard registry completeness, command-profile usage, reachability, and per-state provenance tables are being generated next. Until those tables and runtime tests are complete, all corresponding requirements remain open.
