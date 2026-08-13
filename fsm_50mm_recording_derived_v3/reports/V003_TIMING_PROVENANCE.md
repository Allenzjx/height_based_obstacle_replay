# V003 Timing Provenance

> Offline production-compiler evidence only. Live dispatch and physical Fast Replay are pending.

## Authority chain

1. Original `accepted_steps.jsonl` and `metadata.json` from the explicit v003 directory.
2. `fsm_50mm_recording_derived_v3.recording_fast_plan.fast_plan_rows`.
3. Production `playback.plan_from_steps(profile="fast")`.

No approximate replay compiler and no Step-endpoint fallback were used.

## What the recording actually stores

- Step durations: 24/24; sum `317.933333334629 s`.
- `recording_timing.actual_duration_s` and `motion_semantics.actual_recording_duration_s`: 24/24 and 24/24; both have zero maximum difference from the Step duration.
- Event timestamps: 136/136 relative timestamps and 136/136 simulation timestamps.
- All events identify `simulation_time` as their recording clock.
- Per-event `wheel_active_duration_s` is present but null; the recorded Step motion semantics and production grouping supply wheel intervals.

## 30 deg/s bookkeeping versus 150 deg/s production runtime

- Top-level `metadata.json` numeric servo speed fields: `{}`.
- Nested recording bookkeeping profiles: `fixed-linear-command-space-30deg-s-v1`; baseline numeric reference `30 deg/s`.
- Source-event canonical speeds: `[150.0]` deg/s.
- Current runtime motion reference and production plan: `150` / `150` deg/s.

Therefore 30 deg/s is stale nested bookkeeping, not a numeric field in top-level `metadata.json`. Static code/data provenance selects 150 deg/s for the current compiler. Which speed the historically successful UI run actually dispatched still requires the requested live dispatch trace and is not claimed here.

## Full production trajectory comparison

| Compile | Events | Segments | Final planned time (s) | Plan SHA-256 |
|---|---:|---:|---:|---|
| Runtime 150 deg/s | 160 | 112 | 78.5226666668 | `a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c` |
| Counterfactual 30 deg/s | 160 | 112 | 93.483333333451 | `65afdf68caa1e712d05818e412dff93db364c7c2d7246a13796b309a45a47f5e` |

- Final-time increase at 30 deg/s: `14.960666666651 s`.
- Segments whose full start/end/duration trajectory changes: `112/112`.
- Same command/target path: `True`; endpoint-only comparison used: `False`.
- The CSV includes both 150 and 30 deg/s start/end/duration for every segment, so the conclusion is not based only on final joint targets.

## Wheel integral

| Wheel | Source requested/canonical (rad) | Source applied/derived (rad) | Production plan (rad) | Plan - applied (rad) |
|---|---:|---:|---:|---:|
| front_left_ankle | 17.629333333346 | 17.629333333346 | 17.629333333346 | 0 |
| front_right_ankle | 20.545040000027 | 20.545032163856 | 20.545032163856 | 0 |
| rear_left_ankle | 20.120000000028 | 20.120000000028 | 20.120000000028 | 0 |
| rear_right_ankle | 20.120000000028 | 20.120000000028 | 20.120000000028 | 0 |

The tiny front-right requested/applied difference is the production clamp of a recorded `2.0944 rad/s` command to the runtime limit `2.0943951023931953 rad/s`. The production plan exactly matches the source applied/derived integral; it does not substitute endpoints for wheel travel.

## Pending evidence

- Live dispatch trace: `NOT_COLLECTED`
- Physical Fast Replay: `NOT_RUN`
- No claim is made for obstacle traversal, contact sequence, support diagonal, COM transfer, or FSM completion.
