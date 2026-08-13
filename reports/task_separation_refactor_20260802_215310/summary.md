# Task Separation Refactor Summary

Completed: 2026-08-02 (America/New_York)

## Original problem

The 5 cm bucket loaded the correct `accepted_steps.jsonl` and showed an idle
Playback state, but all Play actions remained unavailable. Vision Auto Replay,
Stability Replay, Height Task, and Playback each contributed task-local state,
polling, and readiness gates to the same controls.

## Root cause

The old availability path was not owned by Playback. Removed-task readiness,
validation, camera, task mode, and stale error/busy state could override a valid
sequence on the next UI poll. Height switching also had a lifecycle in which a
sequence manager could be replaced while other handlers retained the previous
authority. Multiple task-specific refresh paths could therefore disable a
button after another path enabled it.

The fix is architectural rather than a forced widget state:

- one permanent `SequenceManager` is shared by Height Generate, Record,
  Combine, and Playback;
- one `PlaybackManager`, one `SimProcessClient`, and one operation coordinator
  are constructed by the controller;
- one pure availability evaluator owns every Playback button rule;
- the UI has one state polling loop and cancels it on close;
- Vision/Stability task objects, state machines, schedulers, IPC, worker
  handlers, callbacks, tabs, and automatic report flow were removed.

Real Isaac validation found and fixed one additional stale-state issue: two
identical plans formerly reused the same content-hash `plan_id`, so a previous
stop acknowledgement could terminate the next request. Every launch now has a
unique request id while retaining `plan_sha256` for content integrity.

## Current task ownership

| Task | Sole responsibility |
| --- | --- |
| Sim Connection | Attach/detach command transport and show heartbeat/readiness/IPC state. |
| Run Manager | Start, stop, or restart the simulation worker and show process state. |
| Record / Servo+Wheel | Manual commands, recording, Accept/Replace, and shared-sequence edits. |
| Speed Scale | The only authoritative timing/speed model, including preserve-wheel-distance semantics. |
| Playback | The only queue/scheduler and owner of Play, Respawn+Play, Pause, Resume, Stop, analysis, debug, and export. |
| Height Generate | Select/generate/reset obstacle height and load height-matched steps into the shared sequence; never auto-plays. |
| Combine | Combine/edit steps and publish the result to the shared sequence; never plays. |
| Sim State | Read/export simulation state and call the shared standalone respawn service. |

## Final Playback availability

A new normal playback requires: simulation connected, worker runtime ready,
valid non-empty shared sequence, operation state `IDLE`, and no active or
scheduled playback. Selected actions additionally require a valid selected
step. Respawn+Play requires the valid sequence, connected/ready simulation, and
no conflicting operation. Pause requires active and not paused; Resume requires
active and paused; Stop requires active or scheduled. Timing analysis and
motion export require only a valid sequence.

No rule includes selected tab, Vision/Stability state, camera readiness,
dashboard/report state, ground diagnostics, or a historical task error.

## Data safety

Manifest refresh is now a read-only in-memory snapshot. The final pre/post
SHA-256 comparison covered 54 protected recording/manifest/command files:
`changed=0`, `missing=0`. Final Isaac validation also used
`save_scene=false` and `telemetry_enabled=false`, so it did not write USD or
telemetry artifacts.

Scope note: the first exploratory Isaac run, before `--no-save-scene` was added
to the validator, created or recreated
`C:\robotics_sim\wlr_robot\usd\wlr_robot_height_replay_env.usd` at 21:21:21.
Its current SHA-256 is
`8992B4374014FFE5C0F55AF77DB1D605980C1EB1CAD0E8136632CC38455D3241`.
There is no start-of-task hash for this external path, so it was left untouched
rather than risk deleting or guessing a restore for an USD file.

## Outcome

The real 5 cm sequence loaded 35 steps and produced a 269-event plan. Play All
became available, sent `wheel all 0.3`, changed actual robot joint state,
paused without advancing, resumed and advanced, then stopped with the queue
cleared and Play available again. Selected, Respawn+Selected, Play-to-selected,
and Fast variants also reached the real worker. Record recovery, in-memory
Combine, no auto-play, and tab-switch invariance passed.
