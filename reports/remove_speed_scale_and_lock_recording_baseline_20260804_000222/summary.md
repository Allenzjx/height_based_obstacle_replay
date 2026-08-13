# Remove Speed Scale and lock recording baseline

## Outcome

The retired global percentage task is absent from runtime source, configuration, documentation, UI, tests, IPC, worker scheduling, progress, and recording metadata. Manual, Recording observation, Raw playback, and Fast playback now carry direct actuator values. No FSM, phase controller, trajectory generator, CoM controller, RL policy, Vision Auto Replay, or Stability Replay was introduced.

The final visible Isaac run used PID 92564 with requested/effective headless both false. It passed 20/20 real Stop cycles, a real simulation-time recording with zero-target acknowledgment before finalize, Raw and Fast worker playback, four knee commands at -60 degrees, all 8 formal obstacle heights, and 10 settled respawns. The UI closed through its normal safety path and the worker reference was cleared.

## Wheel stop root cause and corrected pipeline

The old design distributed percentage semantics and wheel state across eight integration layers: UI/tab callbacks, controller percentage models, command fields, recording snapshots, playback planning, progress/status, IPC/worker messages, and percentage-only tests/report runners. The mathematical multiplication itself was intended to occur once in the planner, but delayed slider callbacks and ordinary FIFO wheel commands could still leave an older nonzero command behind a zero command or reactivate it after Stop.

The current path is:

`requested wheel velocity rad/s -> validate_motion_command (single clamp) -> SimTransport/SimProcessClient (one IPC message) -> worker high-priority/FIFO queues -> SimRobotAdapter.apply_wheel_velocity -> Isaac actuator -> applied target + measured velocity status`.

`stop_wheels` advances a generation, removes pending ordinary wheel messages, prepends one atomic all-wheel zero command, and the worker drains safety messages before ordinary FIFO work. Any older-generation nonzero message is rejected by the adapter. Stop status separately timestamps receipt, zero-target application, and measured physical stop. Stop is reused by UI Stop, recording stop, playback stop/complete/error, respawn, obstacle update, disconnect, worker shutdown, UI close, operation errors, and command timeout.

Final real latency (20 samples): command target-zero min/P50/P95/max = 1.077/11.835/18.895/19.444 ms; physical stop after target-zero = 26.022/27.250/31.055/41.059 ms; UI callback P95/max = 20.263/20.964 ms. All 20 injected stale-generation probes were rejected and every applied target remained zero. Physical stop uses a locked 0.10 rad/s measured contact-jitter floor; this changes classification only, not target, damping, friction, inertia, physics, or telemetry.

## Recording and playback semantics

Stop Recording first rejects new manual actuator commands, records the end simulation timestamp, submits the authoritative high-priority stop, waits for the matching command id and `zero_target_applied`, appends the final stop boundary, and only then finalizes. The real recording reports `simulation_time` as its timing source and `zero_target_applied=true`.

New recordings contain servo name/start/target/command simulation time/fixed profile/requested target/measured final/error/completion, wheel name/requested and applied rad/s/start/stop simulation time/active duration/signed joint displacement/measured average/final stop/data source, plus baseline identity, scene and state snapshots. The temporary real sample requested and applied 0.3 rad/s for 1.2666666667 s. Raw and Fast plan signatures are identical; both executed 3/3 events and completed with `wheel stop`. Raw duration was 2.2916666667 s and Fast 1.6000000000 s; only implicit UI idle differs. Command differences for servo target, wheel velocity, and wheel active duration are all zero. Physical signed displacement is retained as diagnostic telemetry and is not used to rescale either plan.

## Actuator baseline

The authoritative file is `config/fsm_recording_baseline.yaml`. Servo command units are degrees, articulation units are radians, the fixed reference profile is 30 deg/s, effort 2.7 Nm, stiffness 600, damping 60, armature 0.005. Hip limits are [-135, 135] degrees and knee limits are [-60, 210] degrees. Front servo directions are +1; rear servo directions are -1.

All four real knee UI/worker commands were -60 degrees. Measured articulation values were front-left -59.980, front-right -59.864, rear-left +59.999, rear-right +60.222 degrees; the positive rear physical values are the expected result of the locked -1 rear mapping and are within about 0.22 degrees of their mapped targets.

Wheel limit is 2.0943951024 rad/s, default recording command 0.3 rad/s, damping 20, armature 0.002, and live collision-local radius 0.0499899983 m. Directions are front-left -1, front-right +1, rear-left -1, rear-right +1. In the real Raw run the signed joint displacements were -0.7600, +0.7140, -0.7169, +0.7577 rad, confirming the four mappings for the same positive command.

## Environment, respawn, and obstacle geometry

Physics is 120 Hz with render every 2 physics steps, gravity -9.81 m/s2, solver iterations 8/2, ground static/dynamic friction 1.25/1.05, and obstacle friction 1.2/1.0. These values were not tuned for this result.

The original collision fallback rotated the corners of an already axis-aligned local AABB. For an oriented wheel this invented non-mesh corners and falsely reported about 20 mm penetration, causing an incorrect upward placement. The corrected fallback transforms every real collider mesh point by the live body pose before forming the world AABB. The authoritative live grounded root pose is `[-0.003959653899073601,-0.0009929761290550232,0.09901974350214005,0.9999808073043823,-0.00001026025438477518,-0.006009926088154316,0.0015246823895722628]`. Ten settled respawns had maximum pairwise position spread 0.00012054 m and differed from the reference by 0.00022525 m. Minimum collision clearance was -0.00094204 m, inside the explicit 0.003 m penetration tolerance; the former false upward jump is absent.

The obstacle is centered initially at x=1.55 m, y=0; length 2.0573755571 m; width 0.8822007311 m; fixed front face x=0.5213121738 m; bottom z=0; and top z alone changes to 0.05 through 0.40 m. Across all 8 heights, front-face maximum error was 1.0608e-7 m and bottom error was 0. Canonical future-FSM reference distance is defined only as `root_to_obstacle_front_m = obstacle_front_face_x_m - robot_root_x_m`; the authoritative initial value was 0.5252718277 m, and the 8-height observed maximum variation was 0.0001149510 m. Front-wheel and collision-front distances remain recorded as verification fields.

These locked direct actuator units, mappings, limits, physical settings, grounded pose, collision-measured geometry, and hash are suitable inputs for future FSM work because recording and both playback profiles already consume the same lower-level API. This task deliberately contains no FSM behavior.

## Protection and verification

The project is not a Git repository, so a pre-change SHA-256 inventory was used. The old `saved_height_steps` tree is exactly 55 files before and after: changed=0, missing=0, added=0. All save/replace/delete/recording tests used temporary directories. The formal new root `saved_height_steps_fsm_reference_v1` contains only its empty height layout and manifest; no test step was written there.

Final results: 187/187 unit tests PASS; compileall PASS; runtime/source banned percentage terms = 0; forbidden FSM/Vision/Stability terms = 0; visible Isaac result PASS; UI closed and worker exited. Baseline ID is `fsm-recording-reference-v1:8e1f6009c1984a18`, SHA-256 `8e1f6009c1984a185846e27944b9878329b94bac77649e7645b3f134dd02ba1d`.
