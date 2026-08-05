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
3. **Record / Servo+Wheel** — manual commands and sequence recording/editing.
4. **Playback** — the only owner of replay queue creation and Play/Pause/Resume/Stop.
5. **Height Generate** — generate/reset the obstacle and load height-matched steps into the shared current sequence.
6. **Combine** — combine steps and publish the result as the shared current sequence.
7. **Sim State** — inspect simulation/robot state and use the shared standalone respawn service.

`Height Generate`, `Record`, and `Combine` all update the same permanent
`SequenceManager` instance read by `Playback`. Loading or combining never starts
playback automatically.

## Height-indexed steps

Supported heights use the project manifest and existing folders, for example:

- `5 cm` → `saved_height_steps_fsm_reference_v1\height_05cm`
- `10 cm` → `saved_height_steps_fsm_reference_v1\height_10cm`

Missing or invalid steps clear the shared current sequence and show a direct
error. A later valid load recovers normally; it does not leave a task lock.

## Playback availability

A new playback can start only when:

- the simulation transport is connected;
- the worker reports runtime ready;
- the shared current sequence is valid and non-empty;
- the operation coordinator is `IDLE`;
- playback is neither active nor scheduled.

Selected-step actions additionally require a valid selection. Pause, Resume,
and Stop each use their own playback-state rule. Timing analysis and motion
export require only a valid sequence. Ground diagnostics are still used by the
shared respawn safety service, but they do not gate ordinary Playback buttons.

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
