# V003 Source Audit

> Static evidence only. Isaac was not launched; this document does not claim a physical Fast Replay PASS.

## Status

- Static source/compile status: `STATIC_PRODUCTION_PLAN_ONLY`
- Physical Fast Replay: `NOT_RUN`
- Live dispatch trace: `NOT_COLLECTED`
- Selected source: `v003_20260805_224517_157723_manual` (explicit v003; no active-pointer fallback)
- accepted_steps SHA-256 matches metadata: `True`
- robot asset SHA-256 matches metadata: `True`
- source files unchanged during audit: `True`

## Source counts

| Measure | Value |
|---|---:|
| Steps | 24 |
| Raw source events / metadata commands | 136 / 136 |
| Expanded actuator commands | 202 |
| Expanded servo / wheel commands | 138 / 64 |
| Atomic servo-wheel batches | 6 |
| Production plan events / segments | 160 / 112 |
| Production semantic no-ops elided | 42 |

The three command counts are intentionally distinct: top-level metadata counts 136 recorded events; atomic expansion produces 202 actuator commands; the production compiler retains 160 effective plan events after semantic no-op removal.
All `source_event_index` values in the JSON/CSV are zero-based within their source Step; Step indices retain the recording's one-based values.

## Timestamp, duration, and schema evidence

- All 136 events contain relative `time`, `actual_recording_time_s`, and `command_start_sim_time` fields.
- All recording timing sources are `simulation_time`.
- `wheel_active_duration_s` exists on every event but is null; wheel intervals come from step motion semantics and the production simulation-time grouping.
- Missing required Step fields: `0`
- Missing required event fields: `0`
- Non-finite numeric values: `0`
- Snapshot classifications: `{"FULL_VALID": 16, "PLACEHOLDER_NO_SIM": 32}`

## Atomic source-to-plan preservation

| Step:event | Batch | Source commands | Retained | Elided zero/no-op | One segment | Concurrent |
|---|---|---:|---:|---:|---|---|
| 5:0 | `f1ef4b66d2ca48cfa70949e14ef0ed1e` | 12 | 9 | 3 | True | True |
| 11:0 | `ddd7007fea0a4d3c965d4581f9322a77` | 12 | 9 | 3 | True | True |
| 14:0 | `0315e9c4803c476189b934ddf661f8e1` | 12 | 9 | 3 | True | True |
| 16:0 | `b801c4acb31243168874d42efacabb2c` | 12 | 9 | 3 | True | True |
| 17:0 | `d35ac68e10f8443d8120b0ca79d24cf9` | 12 | 9 | 3 | True | True |
| 23:0 | `5bc185e974364a1580762302215ffaf5` | 12 | 9 | 3 | True | True |

The production compiler drops explicit zero-wheel semantic no-ops from each atomic launch, but keeps all eight servo commands and the active wheel command in one concurrent segment. It does not split the effective servo-wheel batch across ticks at the plan level.

## Step provenance

| Step | Duration (s) | Events | Expanded commands | Atomic | Plan segments |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.199999999994 | 4 | 4 | 0 | 3 |
| 2 | 5.399999999995 | 4 | 4 | 0 | 3 |
| 3 | 17.399999999984 | 3 | 3 | 0 | 2 |
| 4 | 10.133333333324 | 9 | 9 | 0 | 8 |
| 5 | 21.199999999981 | 3 | 14 | 1 | 2 |
| 6 | 8.399999999992 | 3 | 3 | 0 | 2 |
| 7 | 11.39999999999 | 5 | 5 | 0 | 4 |
| 8 | 14.199999999987 | 8 | 8 | 0 | 7 |
| 9 | 10.59999999999 | 8 | 8 | 0 | 7 |
| 10 | 36.799999999967 | 3 | 3 | 0 | 2 |
| 11 | 19.999999999982 | 3 | 14 | 1 | 2 |
| 12 | 12.399999999989 | 13 | 13 | 0 | 12 |
| 13 | 9.933333333324 | 3 | 3 | 0 | 2 |
| 14 | 17.733333333317 | 3 | 14 | 1 | 2 |
| 15 | 6.200000000079 | 5 | 5 | 0 | 4 |
| 16 | 21.066666666935 | 3 | 14 | 1 | 2 |
| 17 | 22.00000000028 | 3 | 14 | 1 | 2 |
| 18 | 6.733333333419 | 11 | 11 | 0 | 10 |
| 19 | 3.533333333378 | 4 | 4 | 0 | 3 |
| 20 | 11.133333333475 | 12 | 12 | 0 | 11 |
| 21 | 6.666666666752 | 13 | 13 | 0 | 12 |
| 22 | 14.066666666846 | 3 | 3 | 0 | 2 |
| 23 | 19.266666666912 | 3 | 14 | 1 | 2 |
| 24 | 5.466666666736 | 7 | 7 | 0 | 6 |

Initial/final joint command targets for every Step, every source event identity, and every source-to-production command mapping are stored in `V003_FAST_REPLAY_PLAN.json`.

## Locked hashes

| Source | SHA-256 | Path |
|---|---|---|
| accepted_steps | `06e13153b7ba75a4283e117d875f1da4895748835a9032c6faadef2bda25b394` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v003_20260805_224517_157723_manual\accepted_steps.jsonl` |
| metadata | `606f6306bf1781f1462b5a31e8d38b78bbd7363ec062f637851d1eb3ccad0ee3` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps_fsm_reference_v2\height_050mm\versions\v003_20260805_224517_157723_manual\metadata.json` |
| production_playback | `fa808305b1d7167d8a005cb0f8a312d8ff2aac862b7da72e381a7ff52c3ed613` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\playback.py` |
| production_fast_plan_adapter | `3661860d0b2a901a2b4fb7e6c0e9ae7f35a3aba54348df69fe9747cd04c816e5` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\recording_fast_plan.py` |
| sequence_model | `f2d6ffbb6901dd985149154cc3308341a1b8fd0014224dd9f1520c0b769630a0` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\sequence_model.py` |
| command_model | `70f1e4183fc711f1dcded5d44a45c661960a383634b252c944ca21705c86e905` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\command_model.py` |
| motion_speed | `5e41f5501b841ff7764ee6c2a9975afb256d6c121d22519ffc38842737b49421` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\motion_speed.py` |
| runtime_motion_reference | `143bbb71fe50fcde22360901552f36b397b9d102bde21172213bf3a316333747` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\config\real_robot_motion_reference.yaml` |
| recording_bookkeeping_baseline | `399e08d6395e7bb82fefaa03542d9829e937113967900c63802ef4ec09cc1ae1` | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\config\fsm_recording_baseline.yaml` |
| robot_asset | `e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892` | `C:\robotics_sim\wlr_robot\usd\wlr_robot_drive_test.usd` |

- Production plan SHA-256: `a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c`
- Static provenance fingerprint: `6ed847eb582fff43b6c1e4147a7c3ce8a99cb4d244f53ae7497c52c7c987fa72`

## Evidence boundary

This source audit validates bytes, schema presence, timing fields, command provenance, atomic compile structure, and planned wheel integrals. It contains no viewport video, PhysX readback, live dispatch tick, contact, COM, or clearance evidence. Physical traversal, successful replay, support diagonal, and COM-transfer mechanism all remain unverified.
