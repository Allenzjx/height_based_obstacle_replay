# Height-Based Obstacle Replay

Tk/Isaac Lab application for recording, loading, combining, and replaying WLR
robot commands against a height-controlled obstacle.

## Start

Use the existing Isaac Lab environment and the formal entry point:

```powershell
conda activate env_isaaclab
python height_replay_ui.py --ui
```

For a UI-only smoke check that does not start Isaac:

```powershell
python height_replay_ui.py --ui --no-sim --smoke-test-ms 2000
```

## UI ownership

The right notebook has exactly these task areas:

1. **Sim Connection** — connect/disconnect, heartbeat, readiness, IPC errors.
2. **Run Manager** — start, stop, or restart the simulation worker and inspect its process state.
3. **Record / Servo+Wheel** — manual commands, atomic Servo-Wheel staging/Launch, and sequence recording/editing.
4. **Playback** — the only owner of replay queue creation and Play/Pause/Resume/Stop.
5. **Height Generate** — update the obstacle through a verified USD geometry transaction and manage height versions.
6. **Combine** — combine steps and publish the result as the shared current sequence.
7. **Sim State** — inspect simulation/robot state and use the shared standalone respawn service.

`Height Generate`, `Record`, and `Combine` all update the same permanent
`SequenceManager` instance read by `Playback`. Loading or combining never starts
playback automatically.

### Servo-Wheel Mode

`Start Servo-Wheel Mode` copies the live command state into a staging buffer.
Servo and wheel sliders then change only that buffer; the robot does not move.
`Launch Servo-Wheel` sends one `apply_motion_batch` IPC message containing all
eight servo and four wheel targets, and the worker writes both channels in one
articulation cycle for the same next physics tick. Recording stores one
`servo_wheel_launch` event and Playback uses the same atomic executor. `Clear
Staged` reloads the live state without moving; `Cancel` clears staging and
immediately stops wheels.

Manual Wheel uses the same readiness policy as Manual Servo and is available in
ordinary `IDLE` as well as `RECORDING`; Recording observes commands but never
gates or resends them. Stop Wheels remains the generation-advancing safety
boundary, and worker wheel generations are synchronized back to the controller.

### Fixed motion profile

The runtime has one non-adjustable motion profile: **Fixed 100%**. There is no
Speed Scale task, slider, timer, IPC request, manual multiplier, or playback
multiplier. Servo targets/rates and wheel velocities are canonical actuator
commands. Legacy speed metadata is accepted as opaque input and ignored.

## Height-indexed steps

Supported obstacle heights are integer millimetres only: `50`, `75`, and `100`.
Legacy `5 cm` and `10 cm` paths remain read-only compatibility sources, for
example:

- `5 cm` → `saved_height_steps_fsm_reference_v1\height_05cm`
- `10 cm` → `saved_height_steps_fsm_reference_v1\height_10cm`

Missing or invalid steps clear the shared current sequence and show a direct
error. A later valid load recovers normally; it does not leave a task lock.

Generate/Update creates a request ID and requested obstacle revision. The UI
reports success only after the worker measures `/World/Obstacle` and confirms
the prim, visual and collision height (within 1 mm), the new revision, and
control readiness. The authoritative Y width is 2.00 m at every height; X front
face, X length, Y centre, and ground-aligned bottom remain fixed. Ordinary
Generate never respawns the robot; `Generate + Respawn Robot` is separate.

## Playback availability

A new playback can start only when:

- the simulation transport is connected;
- the worker reports runtime ready;
- the shared current sequence is valid and non-empty;
- the operation coordinator is `IDLE`;
- playback is neither start-requested, active, nor scheduled.

Selected-step actions additionally require a valid selection. Pause, Resume,
and Stop each use their own playback-state rule. Timing analysis and motion
export require only a valid sequence. Ground diagnostics are still used by the
shared respawn safety service, but they do not gate ordinary Playback buttons.

Worker playback starts with a request/accept handshake containing unique
request and plan IDs, the plan SHA-256, decoded event/segment counts, and the
worker session ID. The UI stays in `START_REQUESTED` until the matching explicit
acceptance arrives, then follows worker status as the authority. A first-command
watchdog and completion/error reconciliation prevent a stale local “active”
state. Long plans carry count/SHA/step-boundary integrity diagnostics through
planner, IPC, decode, scheduler, and UI progress.

## Data safety

New recordings use `saved_height_steps_fsm_reference_v1`. Historical commands
and sequences under `saved_height_steps`, `saved_sequences`, and the
accepted/history files are read-only user data. Tests use temporary stores.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Telemetry capture and state diagnostics are documented in
[`README_TELEMETRY.md`](README_TELEMETRY.md).
