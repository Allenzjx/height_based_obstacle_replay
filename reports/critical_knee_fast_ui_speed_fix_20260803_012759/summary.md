# Critical knee / Fast / GUI / Save / Speed fix report

Generated: 2026-08-03 (America/New_York)  
Project: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay`  
Formal 5 cm source was used read-only; destructive workflow tests used report-local temporary copies.

## Result

All five requested fault classes were changed in runtime source and verified. Final automated regression: **201 tests PASS**. Three final visible real-Isaac workflows passed sequentially with one worker at a time. The final protected-data audit compared 65 files by length and SHA-256: **0 mismatches**. Vision Auto Replay and Stability Replay were not restored; the UI structure regression still confirms only retained tabs and one poll loop.

## 1. Four knee joints

Root cause: the UI, command model and sequence model already shared `KNEE_LIMIT_DEG=(-60, 210)` and did not contain a hidden -35 clamp. The defect was at articulation initialization: safe runtime records were calculated but `apply_joint_limits_to_sim` defaulted false, so the imported articulation's stale runtime limit remained authoritative. This was not a degree/radian conversion, sign/zero-offset, actuator-effort, or collision root cause.

Fix: runtime limit installation is on by default. The live articulation joint names are discovered dynamically. Each named servo RevoluteJoint receives a session-stage USD lower/upper override from the command-space source of truth; source USD/URDF files are untouched. This per-joint path avoids Isaac Lab's tensor helper, which resends continuous-wheel unlimited sentinels and caused PhysX `setLimitParams` errors. A 1e-5 rad numerical envelope keeps float32 endpoints inside the runtime limit without disabling any limit or collision.

Safe lifted A-test -60 errors: front_left_knee=-0.080deg, front_right_knee=+0.164deg, rear_left_knee=+0.260deg, rear_right_knee=+0.309deg.  
Normal obstacle B-test -60 errors: front_left_knee=+0.026deg, front_right_knee=+0.011deg, rear_left_knee=+0.010deg, rear_right_knee=+0.005deg.  
The final visible pose put all four command sliders and worker targets at -60; its measured errors were -0.003°, +0.016°, +0.302°, -0.008°. All are below 1°.

## 2. True semantic Fast

Root cause: the old `motion_only_events()` removed only record markers and moved the first event of each step to zero. It still preserved internal UI timestamps/no-op commands and effectively treated step boundaries as timing barriers; it also retained an unconditional trailing pad.

Fix: Fast now scans the complete in-memory sequence while carrying actuator state. It keeps a servo command only when its target changes beyond epsilon, keeps non-zero wheel intervals even when velocity repeats, preserves direction changes and wheel stop, groups simultaneous Servo+Wheel commands, and preserves only explicit `wait`/`hold`/`sleep`. Repeated zero-motion commands and implicit UI timestamp gaps are removed. Step boundaries remain progress metadata only; fixed per-step/final pad is 0. Worker simulation time is the scheduler clock.

Formal 5 cm read-only timing: Raw 278.872 s; Fast 123.249 s; removed 155.623 s. Fixed step gap: 0. Final real full Fast at requested 300% had planned simulation time 46.184533 s, 192 events, median/max inter-step scheduler gap 0/0 s, maximum lateness 0.008225 s (about one 1/120 s physics tick), and ended active=False, scheduled=False, operation=IDLE. Fast never changes or secretly forces the global percentage.

## 3. GUI responsiveness

Root cause: every 100 ms `_refresh_button_states` recomputed full playback availability and normalized all 35 steps separately for roughly 40 guarded buttons. `snapshot()` repeatedly normalized rows/summary and rehashed the 374,808-byte formal JSONL. `_post()` also forced `update_idletasks()`.

Fix: sequence validity, visible rows, summary and hash are revision/stat cached; button availability is computed once per refresh; unchanged widget states/text/tree selection are not rewritten; the formal JSONL is not read in polling; full table redraw occurs only on sequence/step change; slider IPC is debounced; `_post()` no longer nests idle processing; close cancels poll/slider/speed/smoke callbacks. No Vision/Stability timer was added.

Before `_poll`: mean 240.996 ms, P95 277.054 ms, max 289.889 ms, 2.43 Hz. After: mean 2.021 ms, P95 4.062 ms, max 4.527 ms, 9.59 Hz. After `snapshot` P95 was 0.699 ms; button refresh P95 0.338 ms. During the profile: full-table redraw count 0, sequence JSON reads 0, IPC polls 48. This meets the requested <20 ms P95 target.

## 4. Save Modified Steps

The Accepted Steps pane now shows `💾 Save Modified Steps`, the absolute target path and `Unsaved changes`. The button is disabled when clean and enabled for Replace/Delete/Combine/Reorder/command edits; Ctrl+S uses the same path. Close, height switch, reload and sequence-overwrite operations ask Save/Discard/Cancel.

Persistence writes a same-directory temp file, flushes/fsyncs it, reload-validates fields/counts, creates a timestamped backup, atomically replaces, then reload-validates the destination. Dirty clears only after all stages succeed. Injected temp corruption and disk errors preserved the original and dirty state. The temporary GUI save landed at `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\reports\critical_knee_fast_ui_speed_fix_20260803_012759\temporary_save_copy\saved_height_steps\height_05cm\accepted_steps.jsonl` and reloaded successfully.

## 5. One speed authority and recording equivalence

Before: ordinary playback scaled once in the planner and the worker did not rescale, but Manual/Recording and Playback held copied percentages. More importantly, a manual wheel recording persisted the already-scaled command; replay then scaled that value again—two semantic applications across record -> replay. Recording timestamps were wall-clock.

Now one `SpeedPercentModel` instance is shared by UI/controller/manual/staging/recording/playback. Planner events assert `speed_scale_application_count=1`, and the worker rejects any other count. Servo target angles are never scaled. Wheel velocity is scaled once and duration inversely scaled; if the velocity limit is reached, duration is recomputed so base angular displacement is retained. 0% blocks new motion and pauses active worker playback while preserving progress; Stop remains available.

Recording stores 100% reference commands and a simulation-time-normalized reference duration. It also stores servo start/target plus real articulation wheel joint displacement; if joint state is unavailable, the explicit fallback is marked `command_velocity_x_simulation_duration`. In the real 200% temporary recording, actual 1.541667 s became a 3.083333 s 100% reference, `wheel all 0.6` was stored while `wheel all 1.2` was executed. Same-speed playback duration was 1.541667 s (difference 0), servo endpoint difference was 0.032643°, and maximum four-wheel displacement difference was 0.040225 rad.

Across 50/100/200/300% real playback, measured articulation wheel travel was 3.557249/3.573561/3.579091/3.580776 rad; maximum deviation from the physical median was 0.533421%. The planner command integral was 1.8 rad at every percentage. The approximately 2x articulation-coordinate/command-integral relationship is consistent at all speeds and in recording/replay; no wheel radius/transmission calibration exists to convert it honestly to linear ground distance, so mapping/calibration was not guessed or altered. A dedicated same-start, same-target front-left-knee action used command -45° with a 2 s 100% reference: planned durations were 4/2/1/0.667 s and real endpoint errors were +0.02223°/+0.02228°/+0.02226°/+0.02228° at 50/100/200/300%. A separate formal 5 cm Step 2 stress sample was extremely short after compaction and produced a 200% transient error of +2.261°; it is retained in the general E2E JSON, but is not used as the controlled endpoint-invariant result.

## Verification inventory

- Automated: `python -m unittest discover -s tests -p "test_*.py" -v` -> 201 PASS in 15.373 s.
- Real GUI/Isaac general workflow PID 169796: Raw start, Pause/Resume/Stop, selected Raw/Fast, 50-300%, 0% pause, tab switching, full formal Fast completion -> PASS; process closed cleanly.
- Real GUI/Isaac runtime-USD knee workflow PID 4820: safe A, normal B and visible four-at--60 pose -> PASS; no `setLimitParams` errors; process closed cleanly.
- Real GUI/Isaac recording/playback workflow PID 19980: temporary record, atomic save, start-state restore and same-speed playback -> PASS; process closed cleanly.
- Real GUI/Isaac servo endpoint workflow PID 20016: identical -45° target at 50/100/200/300%, maximum endpoint error 0.022285° -> PASS; process closed cleanly.
- Save/dirty/Ctrl+S/rollback test on temporary formal copy -> PASS.
- Protected formal recording/assets audit: 65/65 exact length + SHA-256 matches.

## Known evidence boundary

No source or project tool provided a video recorder, so no video was produced and no dependency was installed. A true linear ground-travel number is not claimed because the protected project has no validated wheel radius/transmission calibration; direct articulation wheel angle is fully measured and reported. The formal short-step transient noted above remains diagnostic evidence, while the controlled four-speed endpoint test passes. There are no unverified code paths within the five requested fixes; real hardware behavior is outside this Isaac Sim task.
