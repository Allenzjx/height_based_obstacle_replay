# Telemetry and simulation diagnostics

Telemetry is a shared low-level simulation facility. It can record robot pose,
joint/wheel state, contact data, playback timing, and derived physics metrics.
It is not a task controller and never starts, schedules, pauses, or stops
playback.

The **Sim State** tab presents the current worker snapshot. Playback debug and
timing analysis may read the same underlying status. Worker-side capture uses
bounded live buffers while retaining the configured CSV/JSONL/NPZ exports.

Telemetry configuration remains available through the existing `--telemetry`,
`--no-telemetry`, rate, output directory, and report options. Its metric names
may include physical stability terminology; these are measurements only and do
not create a UI task, task state machine, scheduler, dashboard, or replay path.

User telemetry outputs are preserved. Regression tests must write to temporary
directories or a new report directory and must not modify prior runs.
