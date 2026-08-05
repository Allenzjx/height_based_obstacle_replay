# Troubleshooting

## Playback buttons are disabled

Read the reason shown at the top of the Playback tab. A new playback requires
an attached, runtime-ready worker, a non-empty shared sequence, and an idle
operation coordinator. Recording, respawning, a scene update, or an already
active/scheduled playback temporarily blocks new playback requests.

Height Generate and Combine update the same shared sequence used by Playback.
They never start playback automatically.

## A height loads no steps

The expected path is
`saved_height_steps\height_XXcm\accepted_steps.jsonl`. A missing or invalid
file clears the shared current sequence and reports a recoverable error. Select
a supported height with a valid bucket and load it again; no restart is needed.

## Ground state FAIL but respawn is ready

This can be recoverable. Ground diagnostics distinguish visible-mesh overlap
from physical collision penetration. Use the standalone respawn action in Sim
State, then recheck ground contact. Ordinary Playback availability is not gated
by ground state; the ground reference applies to respawn safety only.

## Pause, Resume, or Stop is unavailable

- Pause requires an active, unpaused playback.
- Resume requires an active, paused playback.
- Stop requires an active or scheduled playback.

Stop clears the local plan, worker queue identity, schedule, and operation
state. Play becomes available again as soon as the worker reports idle.

## Worker hard failures

Scan worker stdout/stderr for:

```text
setLimitParams
only supports limit angles
CUDA error
illegal memory access
Traceback
ERROR_FATAL
```

Keep the exact worker logs and GUI result JSON together when diagnosing a
failure. For telemetry capture or report data-quality questions, see
`README_TELEMETRY.md`.
