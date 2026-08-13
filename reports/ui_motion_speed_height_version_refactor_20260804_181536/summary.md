# UI Motion / Speed / Height / Version Refactor

Final status: PASS. A real visible Tk UI and exactly one Isaac subprocess (PID 110528) completed the complete 32-step E2E and exited with no residual process.

## Five root causes

1. Normal heartbeat serialized full sim state, diagnostics, ground history, scene baseline and logs; blocking send/backlog work competed with physics and UI polling.
2. Normal Generate chained prim rebuild/respawn/settle/live mesh validation/measure_scene_baseline, producing 25-50 s waits.
3. Servo and Wheel were sent as independent commands behind staging/mode state, so no single scheduler boundary guaranteed simultaneous application.
4. Servo nominal reference was fixed too low and speed semantics were duplicated; simulation render cadence also undercounted/blocked real motion.
5. One accepted_steps file per height silently prevented immutable alternatives and forced synchronous manifest/file work into user actions.

## Resulting architecture

Tk -> HeightReplayController -> SimTransport -> one SimProcessClient -> one subprocess worker -> one SimRobotAdapter.
The worker uses an overwrite-only compact status slot plus a bounded critical ack queue. Detailed state is explicit-only and never replaces the lightweight UI snapshot.

## Measured outcome

- UI callbacks: n=207, mean=5.302 ms, P95=15.263 ms, max=49.321 ms.
- Ordinary Tk probe: max=147.567 ms (<200 ms).
- Heartbeat: average=10367.1 B, max=10512 B, 7.516 Hz, socket blocking=0.000 ms.
- RTF: P50=1.282 (before 0.388).
- Generate 50/75/100 mm ack: 113.353/160.649/163.196 ms; none respawned or ran baseline scan.
- Generate + cached respawn: 1762.894 ms.
- Atomic batch: tick 1105 for both channels; scheduler ack skew=0.000000 s. Heartbeat-observed skew 0.066667 s is bounded by 5-10 Hz observation resolution, not apply timing.
- Servo 100/200% direct measured moving-average speed: 76.273/86.702 deg/s; peak 141.113/211.856 deg/s; target remains 60 deg; final errors -0.501/0.811 deg.
- Both Servo runs reached the real 2.7 N*m effort limit; 200% computed demand peaked at 48.509 N*m, explaining non-2x physical average despite 2x requested/effective trajectory rate.
- Wheel 100/200% joint path: 0.328366/0.707169 rad, ratio=2.153603; body path=0.022938/0.033730 m. Slip ratio remains UNVERIFIED because real radius/transmission are unavailable.
- v001/v002 SHA differ and both remain loadable below `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\reports\ui_motion_speed_height_version_refactor_20260804_181536\temporary_version_store_v2`.
- Four knee targets were all -60 deg.

## Speed semantics

Servo: target_effective = target_canonical; requested_rate = 150 deg/s * scale; effective_rate = min(requested_rate, verified limit when available). No upper servo velocity limit is claimed verified.
Wheel: effective_velocity = canonical_velocity * scale, clamped once in the worker; duration is unchanged; ideal joint path = effective_velocity * duration.
Recordings store canonical 100% values plus a speed snapshot. Playback reuses the same worker MotionBatch executor, so 200% recording at 200% playback remains 200%, not 400%.

## Height/version system

The only new heights are integer `height_mm` values 50, 75 and 100. Legacy 5 cm and 10 cm map read-only to 50/100 mm; 75 mm has no legacy requirement. New versions live at `saved_height_steps_fsm_reference_v2/height_NNNmm/versions/vNNN_timestamp_name`; Save New Version always creates a directory, while Save Current requires confirmation and backup. The visible `💾 Save New Version` and Ctrl+S use one async persistence service.

## Geometry

Y width changed from 0.8822007310718405 m to 1.2 m. Robot collision width is 0.44110036553592025 m, leaving 0.3794498172320399 m each side. X length/front face remain 2.057375557085507 / 0.5213121737735307 m.

## Most expensive measured post-change callbacks

[
  {
    "name": "tab_switch",
    "elapsed_ms": 49.32129988446832
  },
  {
    "name": "tab_switch",
    "elapsed_ms": 46.20230011641979
  },
  {
    "name": "tab_switch",
    "elapsed_ms": 43.72009984217584
  }
]

No FSM, RL/PPO, CoM controller, Vision Auto Replay, Stability Replay or Preserve Wheel Distance path was added.
