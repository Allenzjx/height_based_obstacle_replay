# Height Based Obstacle Replay

This folder is an isolated height-indexed replay library for the WLR robot:

`obstacle height -> saved accepted_steps.jsonl -> Isaac Sim playback`

It is not RL and does not train a policy. The UI now follows the real robot UI controller layout, but motion commands are routed to Isaac Sim through `SimRobotAdapter` instead of serial hardware.

## Supported Heights

Only these heights are supported:

`0cm, 5cm, 10cm, 15cm, 20cm, 25cm, 30cm, 35cm, 40cm`

The UI uses centimeters. Isaac Sim receives meters:

```python
obstacle_height_m = height_cm / 100.0
```

Unsupported values print:

```text
当前只支持 5cm 间隔高度
```

## Saved Files

Steps are always saved under the current height bucket:

```text
C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps\height_00cm\accepted_steps.jsonl
C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps\height_05cm\accepted_steps.jsonl
C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps\height_10cm\accepted_steps.jsonl
...
C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps\height_40cm\accepted_steps.jsonl
```

`saved_height_steps\manifest.json` records height, recorded yes/no, steps path, last save time, and step count.

## Telemetry

Runtime telemetry is documented in `README_TELEMETRY.md`. It records whole-body COM, link COM contributions, joint tracking and torque, contact/support geometry, stability margins, replay events, CSV/NPZ exports, and an offline `dashboard.html`.

Common switches:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --auto-play --height-cm 5 --telemetry --no-live-viz --report
python .\height_based_obstacle_replay\height_replay_ui.py --ui --height-cm 5 --no-telemetry
```

## Open The UI

From `C:\robotics_sim\wlr_robot`:

```powershell
conda activate env_isaaclab
python .\height_based_obstacle_replay\height_replay_ui.py --ui
```

Default UI behavior:

1. Opens the real-robot-style UI immediately.
2. Starts an Isaac worker subprocess in the background by default.
3. The worker subprocess creates Isaac Sim / Isaac Lab SimulationApp in its own main thread.
4. The worker generates the default height obstacle, currently `0cm`.
5. The UI shows sim ready, worker phase, worker pid, physics dt, sim step Hz, UI refresh Hz, real-time factor, wheel speed limits, current height, task status, loaded steps path, playback state, recording state, and dirty state in the top status area.

For UI/data debugging without Isaac Sim:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-sim
```

## UI Layout

The main screen is modeled after `real_robot_ui_controller`:

- Left column: `Servos`, `Wheels`, `Quick Commands`
- Center column: `Accepted Steps`, `Accepted Step Actions`, `Step Details`
- Right tabs: `Sim Connection`, `Run Manager`, `Record / Servo+Wheel`, `Speed Scale`, `Playback`, `Height Task`, `Combine`, `Vision Auto Replay`, `Sim State`

Serial-only controls are replaced with sim status/actions. Servo sliders and wheel sliders still emit the same command vocabulary used by the real robot UI.

The UI does not call `sim.step()` from Tk polling. In the default `--sim-launch-mode subprocess`, Isaac Sim startup, scene updates, command application, and continuous stepping run in `sim_worker_process.py`; Tk only communicates through localhost newline-JSON IPC via `sim_process_client.py`. Fast labels/sliders refresh at `--ui-refresh-ms`, sim summaries at `--sim-status-refresh-ms`, and full JSON/tree/manifest refreshes at `--full-refresh-ms` or on revision changes.

`--sim-launch-mode thread` is kept only for debugging. Isaac Sim / Kit may hang when `SimulationApp` is created inside a Tkinter background thread, so it is not the default.

## Anti-Freeze UI Behavior

`Accept Recorded Step` and `Accept Replacement` now keep the Tk path light:

- Accepted steps are appended with a `SequenceManager.add_step` fast path.
- Step Details defaults to a compact summary: name, type, duration, event count, height, and command-state summary.
- Full step JSON is not inserted automatically after Accept, Replace, Show Before/After, Combine, Load, or Save.
- Use `Show Step Summary` for the compact view, `Show Step JSON` / `Show JSON Truncated` for an on-demand bounded JSON view, and `Export Step JSON` for the full file.
- If JSON/text exceeds `--max-text-widget-chars` (default `200000`), the UI truncates it and prints a message instead of inserting multi-MB text.

Recording slider events are coalesced at `Stop Record Step` so dragging controls does not create a huge pending step:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --record-event-min-interval-ms 50 --record-event-max-hz 20 --record-max-events-per-step 2000
```

`--record-coalesce-slider-events` is on by default; use `--no-record-coalesce-slider-events` only when debugging raw slider traces. The coalescer preserves final servo and wheel command states and records coalescing stats in the step metadata.

The `Sim State` tab is lazy. Full snapshot JSON is not refreshed after ordinary button clicks. By default it updates only when you click `Refresh Sim State`; pass `--no-sim-state-json-on-demand` if you want the visible Sim State tab to auto-refresh on the full refresh interval. `--disable-auto-sim-state-json` disables that auto path entirely.

Useful anti-freeze/debug parameters:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --ui-refresh-ms 100 --full-refresh-ms 1000 --max-text-widget-chars 200000 --disable-auto-sim-state-json
```

The controller enforces state-machine guards for the heavy workflows:

- Start recording only in `TEST`, `SERVO_WHEEL`, or prepared replacement mode.
- `Servo+Wheel Mode` is now a real `SERVO_WHEEL` mode, not an alias for `TEST`.
- Accept recorded steps only from `PENDING_RECORDED_STEP`.
- Accept replacements only from `PENDING_REPLACEMENT`.
- Play is allowed from clean `TEST` mode or clean `SERVO_WHEEL` mode with no dirty staged command; combine is allowed from clean `TEST` mode.
- Save requires no pending step.
- Height switching is blocked while recording, playing, or holding a pending step; dirty switches require explicit discard.
- E-stop remains available and stops playback/wheels immediately.

## Combine Steps

Combine mode keeps its own selected index set so table refreshes do not collapse multi-selection back to one row.

Recommended workflow:

1. Select one or more rows in `Accepted Steps`.
2. Open `Combine` and click `Combine Mode`.
3. Use `Add Selected To Combine`, `Remove Selected From Combine`, or `Toggle Selected For Combine`.
4. If the selection is not contiguous, click `Select Contiguous Range`.
5. Click `Preview Combined Step` for a compact preview.
6. Click `Combine Selected Steps`.

The combine selection display shows the selected indices, count, and whether they are contiguous. Combine commit replaces the original contiguous range with one combined step, removes the old steps from memory, reindexes the remaining table, clears the combine selection, marks the sequence dirty, and shows a compact summary.

## Playback Start State

`Play Selected Step` and `Play Selected Fast` restore the selected step's beginning state before playback by default:

- New recordings save optional `sim_state_before` and `sim_state_after` fields when the worker reports sim state.
- If `sim_state_before` exists and `Restore full sim pose if available` is enabled, the UI asks the worker to restore it.
- Old steps without `sim_state_before` fall back to `command_state_before`.
- If neither start state is available, the UI warns and still plays from the current robot state.
- `Play To Selected From Start` still plays steps `1..N` from the sequence beginning and does not restore only the selected step.

Plain `Play Selected Step` does not respawn the robot. It stops wheels, restores the selected step start state when available, shows a scheduled playback state during the default `0.30s` settle delay, then sends the selected step's motion events. `Stop Play` cancels both active playback and scheduled playback.

Use `--playback-pre-step-settle-s` to tune the settle delay after restoring the start state. If the selected step produces no playback events, the UI warns `Selected step XXX has no motion events to play.` and does not silently start/stop playback.

`Respawn And Play All`, `Respawn And Play Selected Step`, and `Respawn And Play To Selected From Start` first stop wheels, request a worker respawn, wait the configured `--respawn-play-settle-s`, then start playback. In `--no-sim` mode respawn is a harmless no-op and playback planning still runs.

The left `Quick Commands` panel also has `↻ Respawn` immediately to the right of `E-stop`. It sends the existing `respawn` command path, stops/cancels playback, stops wheels, disarms Vision Auto Replay while preserving the enabled setting, requests the worker respawn, and refreshes state. It does not change obstacle height, `current_height_cm`, saved files, or the Height Task `SequenceManager`. Unlike `Home`, Respawn resets the robot to its initial simulation pose. Respawn does not clear E-stop; use the existing TEST mode / clear E-stop flow before arming Vision again.

Respawn now uses a grounded respawn reference instead of the raw root pose captured immediately after `sim.reset()`. At worker startup, before the worker reports ready, the adapter sets wheel targets to zero, holds the command-zero standing servo pose, steps a bounded physics settle, validates wheel collision clearance against the ground, and then saves `grounded_respawn_root_pose` plus `grounded_respawn_joint_pos`. `set_height`, manual Respawn, Respawn And Play, and auto-play all use that grounded reference. The default `--spawn-z 0.04` is unchanged; use `--spawn-z 0.08` only as an A/B diagnostic, not as the fix.

Ground settle and correction parameters:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui `
  --worker-launch-mode auto `
  --onboard-camera `
  --viewport-physics-guard `
  --robot-auto-ground-correction `
  --robot-ground-settle-s 0.75 `
  --robot-ground-settle-max-steps 180 `
  --robot-ground-stable-frames 10 `
  --robot-ground-vertical-speed-threshold-m-s 0.01 `
  --robot-ground-joint-speed-threshold-rad-s 0.02 `
  --robot-ground-clearance-m 0.002 `
  --robot-ground-penetration-tolerance-m 0.003 `
  --robot-max-ground-correction-m 0.10
```

Automatic Z correction is bounded and conservative. It is considered only when wheel collision AABB diagnostics classify the state as `COLLISION_PENETRATION`; visual-only ground intersection never moves the root. The correction is positive-only, clamped to `--robot-max-ground-correction-m`, applied at most once per validation/respawn flow, followed by zero root velocity, bounded settle, and re-validation.

Playback diagnostics are visible in the Playback tab status line: active/scheduled, starts-in seconds, event index/count, events sent, last command, stop reason, and last info/error. Click `Debug Selected Playback` or run `playback_debug_selected` to print the controller-selected index, manager count, selected step summary, generated plan count, final time, and first playback commands. Button clicks for selected playback also log the Treeview selection IDs, parsed indices, generated command, and current `can_playback` reason.

## Runtime, Ground, And Vision Readiness

The worker now separates runtime readiness from motion safety. `runtime_ready` means the worker, scene, adapter, and simulation loop are running. The legacy `ready` field is kept for compatibility and follows `runtime_ready`; ground validation failures do not make the whole worker not ready.

Motion commands use the stricter `motion_ready` gate. `motion_ready` requires reliable collision bounds and a physical ground diagnostic state of `PASS` or `PASS_WITH_VISUAL_WARNING`; `UNVERIFIED` and `FAIL` block motion. `respawn_ready` is stricter: it requires `motion_ready` plus a valid stable grounded respawn reference. Plain playback uses `motion_ready`, while Respawn, Respawn Before Playback, and Vision auto replay with `Respawn Before Auto Replay` enabled use `respawn_ready`. Ground problems do not block camera health, RGB-D detection, generated obstacle updates, camera geometry checks, camera viewport actions, or generated-height validation.

Ground diagnostics report the configured ground Z and the resolved world-space ground Z from `/World/defaultGroundPlane`, including the collision prim path and Z delta. Wheel diagnostics distinguish `COLLIDER_CONFIRMED_MISSING` from `COLLIDER_RESOLUTION_FAILED`; the latter is treated as `UNVERIFIED` rather than a confirmed missing wheel collision.

Generated Vision test obstacles use `respawn_policy=if_motion_ready`. The worker can confirm `obstacle_updated`, `scene_ready`, `scene_height_cm`, and `obstacle_revision` even when ground safety prevents respawn. In that state camera detection and `Validate Current Generated Height` remain usable, while playback stays blocked until `motion_ready=True`.

The Vision tab keeps a fixed high-level readiness area visible, uses a stable `Field | Value` status table for changing diagnostics, and keeps raw diagnostics in `Raw Vision Diagnostics`. Raw details can auto-refresh, refresh on demand, or pause; updates preserve the user's scroll position.

## Save And Replace Semantics

`Save Steps For Current Height` overwrites the current height bucket's `accepted_steps.jsonl`; it never appends. The store writes `accepted_steps.jsonl.tmp`, atomically replaces the target file, reloads the result, and verifies the saved count matches the in-memory manager count before updating the manifest.

Saving an empty current sequence is explicit: the UI asks before overwriting a height bucket as empty. This prevents stale old files from being mistaken for current steps.

`Replace Selected Step` replaces the step object at that index, rebuilds continuity for following steps, clears pending replacement state and combine preview state, marks the sequence dirty, and after save the JSONL contains only the replacement, not the old step.

## Servo+Wheel Staging

`Servo+Wheel Mode` is a staging mode. After entering it, servo and wheel sliders do not move the Isaac robot immediately. Slider changes update only the staged Servo+Wheel state and the preview panel.

Use this workflow:

1. Click `Servo+Wheel Mode`.
2. Drag servo and wheel sliders. The preview shows live state, staged state, delta, and launch commands.
3. Click `Launch Servo+Wheel` to send the staged servo commands first and wheel commands second.
4. Keep staging more commands, or click `Cancel Servo+Wheel Mode` to return to `TEST`.

`Clear Servo+Wheel Staged` resets the staged state to the current live command state. It is not Home and does not move the robot.

During recording, staged slider movement is not recorded because the robot has not moved. `Launch Servo+Wheel` records one event with `command="servo_wheel launch"` and `expanded_commands` containing the actual servo/wheel commands. Playback uses `expanded_commands`, so existing JSONL remains compatible and staged launches replay as real motion.

`Stop Wheels` and E-stop still execute immediately for safety. `Stop Wheels` also stages wheel speed zero; E-stop stops wheels and playback while preserving staged Servo+Wheel state.

If a saved height folder path such as `saved_height_steps\height_20cm` exists as a file instead of a directory, the manifest layout repair renames it to `height_20cm.invalid_file_YYYYMMDD_HHMMSS` and recreates the directory. The UI status snapshot path is guarded so a manifest layout problem does not crash the Tk polling callback.

## Record 10cm In The UI

1. Open the UI.
2. Open the `Height Task` tab.
3. Select `Height Recording Mode`.
4. Select `10cm`.
5. Click `Start Height Recording Task`.
6. The system generates/loads the 10cm obstacle in Isaac Sim.
7. Use the left `Servos` and `Wheels` panels to move the robot.
8. In `Record / Servo+Wheel`, click `Start Record Step`.
9. Drive the step with the UI controls.
10. Click `Stop Record Step`.
11. Click `Accept Recorded Step`.
12. Click `Save Steps For Current Height`.

The file is written automatically to:

```text
C:\robotics_sim\wlr_robot\height_based_obstacle_replay\saved_height_steps\height_10cm\accepted_steps.jsonl
```

## Record All Heights

1. Open the `Height Task` tab.
2. Select `Height Recording Mode`.
3. Click `Select All Heights`.
4. Click `Start Height Recording Task`.
5. Record and accept steps for the current height.
6. Click `Save And Next Height`.
7. Repeat until `40cm`.
8. Click `Finish Task`.

The manifest table shows recorded yes/no, step count, saved time, and the current task marker.

## Replay 10cm In The UI

1. Open the `Height Task` tab.
2. Select `Height Replay Mode`.
3. Select `10cm`.
4. Click `Start Height Replay Task` or `Generate / Load Current Height Obstacle`.
5. Click `Load Steps For Current Height`.
6. Click `Auto Replay Current Height` or use `Playback -> Play All`.

If no 10cm steps exist, the UI status/log shows:

```text
No saved steps found for 10cm. Please record steps first.
```

## Vision Auto Replay

Vision Auto Replay adds an onboard virtual RGB-D camera to the robot body inside the Isaac worker. The Tk process never reads Isaac camera objects and never receives full images or point clouds over IPC. Worker status only includes small fields such as `camera_ready`, `raw_height_cm`, `detected_height_cm`, `confidence`, `stable_count`, `detection_revision`, `failure_reason`, and an optional debug image path.

The camera is created before `sim.reset()` with Isaac Lab `CameraCfg` and outputs `rgb` plus `distance_to_image_plane`. By default it is mounted under a resolved robot rigid body prim, preferring names containing `base`, `body`, `chassis`, or `trunk`; the WLR USD contains candidates such as `base_link` / `body`. Use `--camera-parent-prim` to override the parent path. If no valid parent can be found, vision is disabled and the original recording/playback UI remains usable.

The height decision does not read `obstacle_height_m`, the UI selected height, worker `height_cm`, obstacle scale/size/bounding box, the manifest, or the `set_height` command. Runtime detection uses camera `distance_to_image_plane`, camera intrinsics, camera world pose, configured world ground `z`, and ROI selection only. Worker status includes `vision.height_provenance` so the UI can show `Height Source: Isaac RGB-D depth geometry`, `Uses Depth/Intrinsics/Camera Pose: YES`, and `Uses Expected/Generated/Scene Height In Detector: NO`.

The pure Python estimator back-projects depth to a world point cloud, searches the forward ROI, uses world ground `z=0` as the reference plane, extracts non-ground obstacle points, estimates the top plane with a robust median, computes `raw_height_cm`, then quantizes only to `0, 5, ..., 40cm`. Generated Test Obstacle mode uses `roi_source=generated_scene_x_prior` and passes only the generated obstacle's `x` position as a non-height prior. External / Unknown mode uses `roi_source=camera_forward_auto` and passes `obstacle_x_m=None`, so the ROI is selected from the legal camera-forward range. The default tolerance is `2.0cm`, below half the 5cm bucket spacing; outside tolerance is invalid and does not choose the nearest bucket.

0cm is a normal bucket. A single frame with no obstacle points does not auto replay. The temporal filter uses a 7-frame window and requires 5 recent consecutive valid frames of the same height with median confidence at least `0.75`. A stable result increments `detection_revision`; the same stable height is latched so it does not create a new revision every status tick. Resetting the filter clears the latch so the same height can be recognized again.

The UI now separates two workflows:

- `Height Task` owns height selection, obstacle generation, recording, Accept/Replace/Delete/Combine, saving `accepted_steps.jsonl`, and manual height-bucket playback.
- `Vision Task` owns generated test obstacles, RGB-D detection, validation, independent Vision steps display, and Vision playback.

Only one task is active at a time. Starting Vision Task is blocked while Height Task is active, recording is active, a pending step/replacement exists, the Height sequence is dirty, playback is active, or an operation is busy. Starting Height Task is blocked while Vision Task is active. Vision Task never overwrites `self.manager`, never changes `current_height_cm`, and never discards unsaved Height Task data. Finish Vision Task restores the central Accepted Steps view to the Height Task manager.

Vision Task has two sources:

- `Generated Test Obstacle`: choose `0, 5, ..., 40cm` in the Vision tab, click `Start Vision Task`, then `Generate Vision Test Obstacle`. This uses the existing worker `set_height` path with extra compatible fields (`source=vision_task`, `request_id`, `generation_revision`) and records scene/detection baselines.
- `External / Unknown Obstacle`: does not call `set_height`, does not use an expected height, and keeps the stable-detection auto replay workflow for unknown obstacles.

Generated Test Obstacle mode has a strict validation gate. Detect Once, stable detection, Validate Isaac Camera, Save RGB-D Diagnostic, Enable Auto Replay, and Arm Auto Replay do not load steps. Before validation PASS, the central Accepted Steps tree is empty and shows `Accepted Steps Source: Vision - waiting for validation`. Only `Validate Current Generated Height` compares the generated expected height to the latest stable detector output. If validation FAILs, uses a stale detection revision, predates the generation baseline, or the detected height differs from the generated height, Vision steps remain empty and playback does not start.

After validation PASS, Vision loads exactly:

```text
saved_height_steps\height_XXcm\accepted_steps.jsonl
```

There is no fallback to neighboring heights and no fallback to Height Task current height. The loaded data is stored in `vision_steps`, not in `self.manager`. The central tree switches to `Accepted Steps Source: Vision - detected XXcm`; those rows are read-only. Replace, Delete, Clear, Save, Combine, Record replacement, and Accept replacement are disabled. Show summary/JSON, Play selected, Play all, and Stop playback remain available.

Automatic replay is off by default. Enable it with the `Vision Auto Replay` tab or `--vision-auto-replay`, then explicitly arm it. In Generated Test Obstacle mode, armed auto replay starts only after validation PASS and successful Vision step load. In External / Unknown mode, stable detection can load and play the detected height's `vision_steps` without a generated expected height. After one successful vision-triggered playback, auto replay disarms. E-stop also disarms auto replay and it is not automatically re-armed by Respawn.

`replay_detected_height()` now delegates to `replay_validated_vision_steps()`. Vision playback uses `vision_steps` or loads them with `load_validated_vision_steps(height_cm, detection_revision)`. It does not call `set_current_height()`, does not load into the Height Task manager, does not generate/modify obstacles, and does not change `current_height_cm`.

Auto replay is blocked without moving the robot when camera is not ready, detection is not stable, confidence is below threshold, the revision was already consumed, no saved steps exist, recording/pending/replacement/playback is active, an operation is busy, E-stop is active, Servo+Wheel staging is dirty, or Isaac Sim is not ready. If no saved steps exist for the detected height, the UI reports `No saved steps found for XXcm. Please record steps first.`

Debug camera frames are saved only on request with `Save Debug Camera Frame` / `vision save_debug_frame`. The worker writes a small PPM mosaic plus JSON sidecar under `saved_height_steps\vision_debug\`. The sidecar includes `camera_geometry`, `camera_coverage`, `height_provenance`, `frame_revision`, `detection_revision`, `source_mode`, and `roi_source`. The UI shows the saved path; it does not stream RGB/depth frames over IPC.

Camera viewport controls are fail-soft worker actions with request/ACK status. `Open Onboard Camera Viewport` uses `omni.kit.viewport.utility.get_viewport_from_window_name()` and `create_viewport_window()` to create or reuse a second viewport named `Onboard Camera`, then assigns `/World/WLRRobot/base_link/onboard_rgbd_camera` to that viewport. The main viewport should remain Perspective. If the second viewport window is created before its viewport API is ready, worker main-thread loop retries for a bounded time and reports `camera_view.pending`, `retry_count`, and any timeout error. Default behavior does not fall back to the active viewport; pass `--camera-view-active-fallback` only when you explicitly want the worker to modify the active viewport and mark `camera_view.active_fallback_used=true`. `Return Main View To Perspective` no longer requires a saved USD camera path; if clearing the viewport camera binding fails, it restores the scene's saved default eye/target through `sim.set_camera_view()`. `Close Onboard Camera Viewport` closes only the project-created camera viewport. Viewport failures do not disable the RGB-D detector.

`--viewport-physics-guard` wraps open, return, close, and restore viewport actions. The worker captures root pose, root velocity, joint state, wheel command state, sim time/steps, timeline state, and robot ground diagnostics immediately before and after the viewport API call, before any ordinary `adapter.step()` can run. The action fails its ACK if immediate root/joint/timeline state changes exceed the guard tolerances. In idle state, open/return/restore guard failures restore the exact pre-action sim state and zero wheel commands; close remains fail-soft and never restores robot pose.

The Vision tab and Sim State tab now include read-only ground diagnostics:

- `Ground Contact`: `PASS`, `FAIL`, `VISUAL ONLY`, or `VISUAL/FABRIC SUSPECTED`.
- `Root Z`, minimum collision clearance, maximum collision penetration, and minimum visual clearance.
- Missing wheel collision list.
- Grounded respawn reference validity and last bounded ground correction.
- Camera viewport physics guard result, root/joint/sim-time deltas, active fallback use, and Fabric/render warning flag.

The Vision tab also has `Validate Robot Ground Contact` and `Respawn And Validate Ground`. Both are worker actions; Tk never reads Isaac objects directly.

If automatic viewport APIs are unsupported, use the manual path:

```text
1. Click the Camera button in the upper-left corner of the Isaac Viewport.
2. Select Perspective to return to the free view.
3. Or open Window > Viewports > Viewport 2.
4. In Viewport 2 select Cameras > onboard_rgbd_camera.
5. Keep the main Viewport on Perspective.

Alternative Camera Inspector path:
Tools > Sensors > Camera Inspector > Refresh
Select /World/WLRRobot/base_link/onboard_rgbd_camera > Create Viewport
```

Recommended default camera tuning:

```text
offset x=0.35 m, y=0.00 m, z=0.18 m
pitch=14 deg down
resolution=424x240
update_period=0.10 s
ROI horizontal fraction=0.72
near/far clip=0.05/6.0 m
```

`Validate Camera Geometry` reports the actual camera position, world/ROS quaternion, optical axis, center ray, target direction, ground intersection, valid depth fraction, ground/obstacle/top-plane fractions, and a framing state. Expected states include `GOOD_OBLIQUE_VIEW`, `TOO_TOP_DOWN`, `TOO_HORIZONTAL`, `OBSTACLE_NOT_VISIBLE`, `TOP_PLANE_NOT_VISIBLE`, `GROUND_NOT_VISIBLE`, and `CAMERA_POINTS_BACKWARD`. Warnings do not block detection by default. `--camera-coverage-strict` makes bad framing invalidate that frame.

Camera aim defaults to the old pitch mode. For oblique forward-down testing, use `--camera-aim-mode look-at --camera-target-frame world --camera-target-x 1.55 --camera-target-y 0.0 --camera-target-z 0.02`. The target z is fixed near ground and does not use generated or expected obstacle height. Look-at now computes the camera offset in the Isaac Lab `world` camera convention, where local `+X` is optical forward and local `+Z` is up; world targets are converted into the camera parent frame before computing the offset rotation. The UI buttons `Apply Recommended Oblique Camera Pose` and `Restart Worker With Camera Pose` stage these parameters and restart the worker rather than trying to rebuild the Isaac camera sensor in-place.

Generated Vision workflow:

1. Record and save steps for each height first.
2. Start the UI.
3. Open `Vision Auto Replay`.
4. Select `Generated Test Obstacle`.
5. Click `Start Vision Task`.
6. Choose the Vision tab `Test Obstacle Height`.
7. Click `Generate Vision Test Obstacle`.
8. Confirm the Height Task combobox/current height did not change and Accepted Steps is empty.
9. Wait for stable detection.
10. Click `Validate Current Generated Height`.
11. PASS loads `saved_height_steps\height_XXcm\accepted_steps.jsonl` into read-only Vision steps.
12. Click `Play Validated Vision Steps`, or enable and arm Auto Replay before the next validated generation.
13. Click `Finish Vision Task` to restore the Height Task steps view.

## Windows Isaac Worker Launch

The default subprocess launcher is now:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode auto
```

`auto` runs a pure Python preflight before starting Isaac. It executes `isaac_launch_preflight.py` as a file in each candidate interpreter, not `isaaclab.bat -p -c "..."`, so it avoids the Windows nested-quote path that can produce:

```text
"" was unexpected at this time.
```

Candidate order:

1. `--worker-python-exe`, when provided.
2. Current `sys.executable`.
3. `%CONDA_PREFIX%\python.exe`.
4. `IsaacLab\_isaac_sim\python.bat` or bundled Python, when present.
5. `isaaclab.bat -p` as the final fallback.

Every candidate must import `isaacsim`, import `isaaclab`, import `isaaclab.app.AppLauncher`, and pass the Python compatibility check.

| Isaac Sim | Required Python |
| --- | --- |
| 5.x | 3.11 |
| 4.x | 3.10 |

Activating Conda can make `isaaclab.bat -p` choose `%CONDA_PREFIX%\python.exe`; if that interpreter does not have Isaac Sim installed, preflight reports `missing_isaacsim` instead of opening a black viewport and waiting.

Run preflight without starting Tk or Isaac:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --launch-preflight-only --worker-launch-mode auto
python .\height_based_obstacle_replay\height_replay_ui.py --launch-preflight-only --worker-launch-mode explicit-python --worker-python-exe C:\path\to\python.exe
```

The worker launch writes a temporary config JSON under:

```text
%TEMP%\height_replay_worker_configs\worker_<timestamp>.json
```

Direct Python mode starts `sim_worker_process.py --worker --worker-config-file <config.json>`. Batch fallback writes a small `.cmd` wrapper and calls only:

```text
isaaclab.bat -p sim_worker_process.py --worker --worker-config-file <config.json>
```

This keeps long paths, spaces, parentheses, `&`, and non-ASCII characters out of IsaacLab's fragile long `%*` reconstruction path. The `Sim Connection` tab shows the display command, config path, wrapper path, cwd, pid, return code, stdout/stderr paths, selected interpreter, Isaac versions, preflight result, startup phase, phase elapsed time, last log activity, and the last meaningful stdout/stderr lines.

EULA handling is explicit. The worker never silently accepts NVIDIA Omniverse terms. `OMNI_KIT_ACCEPT_EULA=YES` is passed only when the environment already has it or when you start with:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --accept-isaac-eula
```

If logs contain an EULA prompt, the UI stops waiting and reports `eula_required`. Run Isaac Sim once in the foreground, or use `--accept-isaac-eula` only if you explicitly agree.

GUI mode is the default. Unless `--headless` is passed, the child environment is normalized to:

```text
HEADLESS=0
LIVESTREAM=0
ENABLE_CAMERAS=1  # only when onboard camera is enabled
```

`--onboard-camera` enables camera rendering but does not force headless. If the UI requested GUI and the worker resolves to headless, the Sim Connection tab warns: `GUI was requested but AppLauncher resolved to headless mode.`

First Isaac Sim launches can spend several minutes downloading extensions or compiling shaders. While stdout/stderr files are still growing, startup is treated as progressing and the tab shows `last_log_activity_at` / `startup_progress_message`. Use `--sim-startup-timeout-s 600` or higher for first-run machines.

Black-screen triage uses startup phase, not pixels:

- No IPC: launcher, batch, Python, or preflight problem.
- `starting_app`: AppLauncher, EULA, extension download, experience, or environment problem.
- `app_created` but not `simulation_context_created`: SimulationContext/device problem.
- `robot_created` but not `sim_reset_completed`: scene, camera, or reset problem.
- `adapter_ready` but viewport black: viewport camera, renderer, headless/livestream, GPU, or render warmup problem.

An empty Isaac `create_empty` viewport before scene phases is different from a project black screen after `adapter_ready`; check `startup_phase_history` and the worker logs before changing scene code.

Isolation commands:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode auto --no-onboard-camera
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode auto --onboard-camera
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode isaaclab-bat --no-onboard-camera
```

Use `Restart Without Onboard Camera` if `camera_error` appears. Camera creation failures are reported but should not prevent ground, robot, obstacle, manual control, recording, or replay from working.

## Motion Units And Scaling

Servo command `0` means the loaded standing pose. Command-space limits are:

- hip: `-135 deg` to `+135 deg`
- knee: `-60 deg` to `+210 deg`

Out-of-range servo commands are clamped in command space. Joint direction signs are unchanged from the earlier height replay adapter. Isaac targets are computed as:

```text
actual_target = standing_pose + JOINT_COMMAND_SIGN[joint] * command
```

That means front and rear knees can move in different physical degree directions for the same command-space value. A rear knee command of `-30 deg` is expected to move in the sign=-1 physical direction, not necessarily toward a smaller measured actual degree.

Knee negative command-space motion is supported. By default the worker safely writes servo physical joint limits broad enough to cover the command-space targets, including knee negative commands. The safe writer clamps any requested PhysX limit to `[-2pi, 2pi]`, skips invalid min/max pairs, and records warnings in worker status instead of using the older broad legacy write path.

The `Sim State` tab includes diagnostics for every servo joint:

- `command_deg`
- `target_actual_deg`
- `measured_joint_pos_deg`
- `current_physx_or_usd_limit_deg`
- `target_inside_current_limit`
- safe limit write record

Wheel commands use radians per second, matching `real_robot_ui_controller`. The default max wheel speed is `2.0943951023931953 rad/s` (20 rpm). Manual `w/a/s/d` commands use the configured default wheel speed, which defaults to 25% of the max. Use `--max-wheel-speed-rad-s` and `--default-wheel-speed-rad-s` to override; old `--max-wheel-speed` and `--default-wheel-speed` aliases still work and are treated as rad/s.

The `Speed Scale` tab controls:

- global motion scale
- wheel speed scale
- servo command scale
- playback speed scale
- apply to manual control
- apply to playback
- preserve wheel distance

Manual wheel commands are multiplied by `global * wheel` when manual scaling is enabled. Manual servo commands are multiplied by `global * servo command` and then clamped to the command-space servo limits. Playback timing uses the Playback tab speed multiplied by `global * playback` when playback scaling is enabled. With `preserve wheel distance` enabled, fast-profile playback also multiplies wheel commands by the effective playback speed and clamps them to the configured max wheel speed.

## CLI Helpers

CLI is kept for quick tests, not as the primary recording workflow:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --sim-launch-mode subprocess --worker-launch-mode auto
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-sim
python .\height_based_obstacle_replay\height_replay_ui.py --height-cm 10 --auto-play
```

Worker-only smoke test:

```powershell
python .\height_based_obstacle_replay\sim_worker_process.py --worker --height-cm 0 --worker-smoke-test-s 20
```

Negative knee worker smoke test:

```powershell
python .\height_based_obstacle_replay\sim_worker_process.py --worker --height-cm 0 --worker-smoke-negative-knee-test --worker-smoke-test-s 10
```

Camera detection worker smoke test, when Isaac can launch:

```powershell
python .\height_based_obstacle_replay\sim_worker_process.py --worker --height-cm 10 --worker-smoke-camera-detection --worker-smoke-test-s 20 --enable_cameras
```

Camera provenance worker smoke test, when Isaac can launch:

```powershell
python .\height_based_obstacle_replay\sim_worker_process.py --worker --height-cm 5 --worker-smoke-camera-provenance --worker-smoke-camera-output .\height_based_obstacle_replay\saved_height_steps\vision_debug\camera_provenance_smoke.json --enable_cameras
```

Camera viewport ground-contact smoke test, when a real Isaac GUI can launch:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py `
  --ui `
  --worker-launch-mode auto `
  --onboard-camera `
  --robot-auto-ground-correction `
  --robot-ground-settle-s 0.75 `
  --worker-smoke-camera-view-ground-contact
```

Viewport physics guard is enabled by default in GUI/worker paths. Use `--no-viewport-physics-guard` only for diagnostics when you intentionally want to observe raw Kit viewport behavior.

The worker smoke path records fixed trigger-isolation stages: startup baseline, after 5cm `set_height + respawn`, immediately before/after opening the onboard camera viewport, 1/10/60 physics steps after open, before/after returning the main view to Perspective, 60 steps after return, and after closing the project-created camera viewport. It does not restore sim state around the viewport action. It writes:

```text
height_based_obstacle_replay\saved_height_steps\vision_debug\camera_view_ground_contact_<timestamp>.json
```

If the current machine cannot run a real Isaac GUI, run the pure Python tests and no-sim UI smoke instead and treat the real physical/visual wheel-ground result as not yet verified.

Old worker launcher aliases are still accepted and map to the new launch modes:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-python-mode current-python
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-python-mode isaaclab-bat
```

If Kit reports `Kit appears to be hanging`, check the `Sim Connection` tab. It includes worker phase, phase elapsed time, pid, return code, traceback, stdout/stderr paths and tails, preflight result, selected interpreter, effective `HEADLESS` / `LIVESTREAM` / `ENABLE_CAMERAS`, EULA source, and startup diagnosis. Startup timeouts are controlled by `--sim-startup-timeout-s` and status silence warnings by `--sim-worker-status-timeout-s`.

Useful debug and motion parameters:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --max-wheel-speed-rad-s 4.0 --default-wheel-speed-rad-s 1.0
python .\height_based_obstacle_replay\height_replay_ui.py --ui --wheel-speed-scale 0.5 --playback-speed-scale 1.5
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-preserve-wheel-distance
python .\height_based_obstacle_replay\height_replay_ui.py --ui --ui-refresh-ms 100 --sim-status-refresh-ms 250 --full-refresh-ms 1000
python .\height_based_obstacle_replay\height_replay_ui.py --ui --max-text-widget-chars 200000 --disable-auto-sim-state-json
python .\height_based_obstacle_replay\height_replay_ui.py --ui --record-event-min-interval-ms 50 --record-event-max-hz 20 --record-max-events-per-step 2000
python .\height_based_obstacle_replay\height_replay_ui.py --ui --playback-pre-step-settle-s 0.30 --respawn-play-settle-s 0.30
python .\height_based_obstacle_replay\height_replay_ui.py --ui --apply-physx-joint-limits
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-apply-safe-servo-joint-limits
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-continuous-sim-step
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode auto
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode explicit-python --worker-python-exe C:\path\to\python.exe
```

Vision parameters:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --onboard-camera --camera-offset-x 0.35 --camera-offset-z 0.18 --camera-pitch-deg 14
python .\height_based_obstacle_replay\height_replay_ui.py --ui --camera-aim-mode look-at --camera-target-x 1.55 --camera-target-y 0.0 --camera-target-z 0.02 --camera-look-at-roll-deg 0
python .\height_based_obstacle_replay\height_replay_ui.py --ui --camera-width 424 --camera-height 240 --camera-update-period-s 0.10
python .\height_based_obstacle_replay\height_replay_ui.py --ui --camera-coverage-strict
python .\height_based_obstacle_replay\height_replay_ui.py --ui --vision-confidence-threshold 0.75 --vision-stable-frames 5 --vision-window-size 7 --vision-height-tolerance-cm 2.0
python .\height_based_obstacle_replay\height_replay_ui.py --ui --vision-auto-replay --vision-respawn-before-replay
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-onboard-camera --no-vision-auto-replay
```

`--apply-safe-servo-joint-limits` is on by default. `--apply-physx-joint-limits` is kept as a legacy/debug flag and is not needed for normal use.

## Reuse Notes

Copied/adapted from `real_robot_ui_controller`:

- UI layout: Servos, Wheels, Quick Commands, Accepted Steps, Step Details, right-side notebook tabs
- Servo/wheel slider behavior and command vocabulary
- Step workflow names: record, accept, replace, delete, undo, clear, show before/after, combine
- Playback controls: fast/raw profile, speed, trailing pad, max idle gap, play all/selected/to selected, pause/resume/stop, analyze timing
- Accepted-step JSONL data shape

Referenced from `manual_obstacle_climb_trainer`:

- Isaac Sim `CuboidCfg` obstacle generation
- default obstacle sizing style
- WLR robot USD articulation setup
- standing-pose-zero servo command conversion
- wheel forward sign conversion into Isaac joint velocity targets

Command-space limits:

- hip: `-135 deg` to `+135 deg`
- knee: `-60 deg` to `+210 deg`

## Validation

Pure Python checks:

```powershell
python -m py_compile @(Get-ChildItem .\height_based_obstacle_replay -Filter *.py | ForEach-Object { $_.FullName })
python -m unittest discover .\height_based_obstacle_replay\tests
```

No-sim UI smoke:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --no-sim --smoke-test-ms 1000
```

Subprocess UI smoke without waiting for manual close:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --sim-launch-mode subprocess --worker-launch-mode auto --smoke-test-ms 1500
```

Launch preflight:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --launch-preflight-only --worker-launch-mode auto
```

No-camera isolation:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode auto --no-onboard-camera
```

Camera isolation:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode auto --onboard-camera
```

Batch fallback isolation:

```powershell
python .\height_based_obstacle_replay\height_replay_ui.py --ui --worker-launch-mode isaaclab-bat --no-onboard-camera
```

Worker startup success is confirmed when the Sim Connection phase history reaches:

```text
ipc_connected
starting_app
app_created
simulation_context_created
ground_created
lighting_created
robot_created
obstacle_created
sim_reset_completed
first_render_completed
adapter_ready
running
```
