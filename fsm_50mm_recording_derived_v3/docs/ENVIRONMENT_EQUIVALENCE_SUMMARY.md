# 50 mm FSM environment equivalence summary

## Verdict

**Current status: `PENDING_RUNTIME_A_B`.**

The source audit and Isaac-free contract tests establish a static lock and an
A/A--A/B acceptance procedure. They do **not** establish runtime environment
equivalence. As of 2026-08-09, neither the two required real-Isaac baseline
runs nor the instrumented B run exists at the registered artifact locations:

- `ENV_AA_BASELINE_1`: `PENDING`
- `ENV_AA_BASELINE_2`: `PENDING`
- `ENV_AB_INSTRUMENTED`: `PENDING`
- `reports/ENVIRONMENT_EQUIVALENCE_REPORT.json`: absent
- `reports/environment_equivalence/`: absent

The existing `reports/environment_lock_50mm.json` is an offline static lock. Its
`runtime_readbacks` array is empty and its status is
`offline_locked_runtime_readback_pending`; it is not an empirical A/B artifact.

Consequently, no environment-equivalence `PASS` may be cited. The failed
`REPLAY_BATCH_20260808T005031` stopped during initial grounding before any
recording command and is neither an A/A sample nor A/B evidence. It remains a
stale `.partial` ground-initialization failure.

The current working tree now implements the artifact converter in
[`environment_ab_artifacts.py`](../environment_ab_artifacts.py), including
source-freeze/git closure, recording/Fast-plan/runtime/sample-grid matching,
role-specific contact evidence, eight trajectory metrics, and a real
active-viewport MP4 admission gate. The implementation and pure-Python tests do
not substitute for the missing A1/A2/B artifacts, so the report remains absent
rather than being populated with a synthetic `PASS`.

Admission also requires the owning supervised batch to have a normal shutdown
closure: `.finalized` with no partial/failed marker,
`shutdown_outcome.status=NORMAL_EXIT`, live
`batch_finalization.phase=SHUTDOWN_COMPLETE`, coherent immutable preclose
snapshots, and valid live/preclose checksums. The four per-wheel force channels
compared across A and B come from the shared complete
`ContactSensor.net_forces_w` layout; a sensor-proven airborne wheel is finite
zero, while a missing/duplicate/non-finite layout is rejected rather than
converted to zero.

## Exact `ab7ed11..92abb432` scene diff

The complete diff of `sim_obstacle_scene.py` over this range contains exactly
two changes:

```diff
@@ class SimSceneConfig:
     telemetry_contact_sensors_enabled: bool = False
+    contact_sensor_factory: Any | None = None

@@ create_scene(...):
     if bool(config.telemetry_contact_sensors_enabled):
-        contact_sensor, contact_error = create_robot_contact_sensor()
+        contact_sensor_factory = config.contact_sensor_factory or create_robot_contact_sensor
+        contact_sensor, contact_error = contact_sensor_factory()
```

There is no change in that commit range to gravity, physics timestep, render
cadence, solver iterations, collision/contact offsets, materials, robot USD,
spawn pose, obstacle geometry, actuator gains/limits, wheel radius, or command
signs. With `contact_sensor_factory=None`, behavior still resolves to
`create_robot_contact_sensor`. A supplied factory changes only how the optional
telemetry sensor object is created.

This source diff is necessary evidence, not sufficient evidence. Contact
instrumentation can still perturb engine execution, allocation, or scheduling;
therefore the real A/A--A/B trajectory comparison below remains mandatory.

Source anchors:

- [`SimSceneConfig`](../../sim_obstacle_scene.py#L45) and its two telemetry fields
- [`create_scene`](../../sim_obstacle_scene.py#L156) factory selection
- `activate_contact_sensors` remains controlled by
  `telemetry_contact_sensors_enabled` in [`build_robot_cfg`](../../sim_obstacle_scene.py#L766)

## Formal clean-reset grounding chain

The only environment-equivalent production initialization chain is:

1. Construct the scene with the formal `SimSceneConfig`.
2. `create_scene` performs `sim.reset()` and `robot.update(0.0)` while the first
   visible render remains deferred.
3. Construct `SimRobotAdapter` from the freshly reset live articulation.
4. Call `initialize_adapter_ground_reference(adapter)` exactly once. It delegates
   to `initialize_grounded_respawn_reference`, settles the live robot, validates
   ground/collision stability, and only then saves the live grounded root/joint
   reference.
5. Reject the run if `ground_reference_result_is_valid` is false.
6. Call `finalize_scene_after_grounding`; only after this may measurement and
   replay proceed.

This is the formal worker ordering in
[`sim_worker_process.py`](../../sim_worker_process.py#L755), and the recording
replay path mirrors it in [`run_fsm50.py`](../run_fsm50.py#L2386):

```text
create_scene
  -> SimRobotAdapter
  -> initialize_adapter_ground_reference
  -> validate grounded reference
  -> finalize_scene_after_grounding
  -> live environment readback
  -> replay
```

The standard grounding settings are locked as follows:

| Setting | Formal value |
|---|---:|
| Raw root spawn Z | `0.04 m` |
| Ground Z | `0.0 m` |
| Physics timestep | `1/120 s` |
| Settle duration | `0.75 s` |
| Settle max steps | `180` |
| Effective standard duration cap | `ceil(0.75 / (1/120)) = 90` physics steps |
| Required stable frames | `10` |
| Maximum root vertical speed | `0.01 m/s` |
| Maximum servo speed | `0.02 rad/s` |
| Maximum wheel speed | `0.20 rad/s` |
| Ground clearance | `0.002 m` |
| Penetration tolerance | `0.003 m` |
| Automatic ground correction | `false` |

### Pre-seed is forbidden in the formal path

The historical locked grounded pose is read-only comparison evidence. It must
not be written into root pose, root velocity, joint state, or adapter state
before clean grounding. In particular, the formal replay path must not call
`_seed_adapter_from_locked_ground_pose`.

The helper remains in source only for explicit, non-default diagnostics; the
formal replay path no longer invokes it. Any run that invokes it must record
`environment_equivalent=false`, use a distinct experiment ID, and cannot be
accepted as A1, A2, B, or a production replay. Merely reproducing a historical
pose does not reproduce the worker's reset/settle trajectory.

The current formal path records `locked_ground_seed.applied=false` and describes
the lock as comparison evidence in [`run_fsm50.py`](../run_fsm50.py#L2391).

## `30` versus `150 deg/s`

| Value | Actual meaning | Runtime authority |
|---:|---|---|
| `30 deg/s` | Legacy `fsm_recording_baseline.yaml` metadata, including the old profile ID `fixed-linear-command-space-30deg-s-v1` | **No** |
| `150 deg/s` | Selected command-space servo interpolation reference | **Yes** |

The selected value comes from
[`real_robot_motion_reference.yaml`](../../config/real_robot_motion_reference.yaml#L7).
[`playback.py`](../../playback.py#L32) loads it for Fast planning and
[`sim_robot_adapter.py`](../../sim_robot_adapter.py#L1151) uses it to advance
logical servo commands. With `dt=1/120 s` and no configured servo velocity cap,
the maximum logical-command increment is `150/120 = 1.25 deg` per physics tick.
This is a command interpolation rate, not a PhysX actuator hard velocity limit;
the implicit drive can lag under load.

The legacy `30` at
[`fsm_recording_baseline.yaml`](../../config/fsm_recording_baseline.yaml#L18)
is retained only as an auditable metadata difference. It must not override the
motion reference, the adapter, or the planner.

For a Fast semantic motion group:

```text
servo_delta_deg = max(abs(target_joint - current_command_joint))
effective_servo_rate = min(reference_rate, velocity_limit) if a limit exists,
                       otherwise reference_rate
servo_duration_s = servo_delta_deg / effective_servo_rate
wheel_duration_s = semantic base interval while a nonzero wheel target is active
segment_duration_s = max(servo_duration_s, wheel_duration_s, explicit_hold_s)
```

The planner calculation is in [`playback.py`](../../playback.py#L159), and the
servo duration is recomputed at segment start from the adapter's current logical
command in [`playback.py`](../../playback.py#L1302). For the selected profile,
the effective rate is `150 deg/s` because the motion-reference velocity limit is
`null`. Fast normalizes to `motion_only`, so implicit timestamp gaps are removed;
explicit holds and active wheel intervals are preserved by the `max` rule.

## Render interval `2` versus `8`

| Value | Actual meaning | Runtime authority |
|---:|---|---|
| `2` | Legacy baseline metadata (`2 * 1/120 = 1/60 s` per render) | **No** |
| `8` | Selected formal render cadence (`8 * 1/120 = 1/15 s` per render) | **Yes** |

The current default is `render_interval=8` in
[`SimSceneConfig`](../../sim_obstacle_scene.py#L60). It is also explicit in the
50 mm replay configuration in [`run_fsm50.py`](../run_fsm50.py#L2334), and both
formal UI/worker parsers default to `8`. The value means one render after eight
physics integration steps; it does **not** change the physics timestep from
`1/120 s` and is not a PhysX substep multiplier.

The `2` at
[`fsm_recording_baseline.yaml`](../../config/fsm_recording_baseline.yaml#L69)
is legacy metadata only. `sim_process_client.py` still contains a compatibility
fallback of `2` when an argument object entirely lacks `render_interval`; that
fallback is not reached by the formal parser/direct FSM paths. An A/A or A/B run
with a missing cadence attribute is invalid: its runtime readback must report
`8`, or equivalence fails closed.

## Static fingerprint contract

[`build_static_environment_fingerprint`](../environment_equivalence.py#L267)
emits schema `fsm50.environment_static_fingerprint.v1`. It hashes the formal
inputs without importing or starting Isaac. The static record includes:

- `source_commit` and `source_commit_scope`; per-file SHA-256 and byte size for
  scene, adapter, command model, motion reference loader, playback/sequence
  planner, environment/motion/baseline configs, FSM config, Fast export, and
  runner sources;
- `robot_usd.path`, `robot_usd.sha256`, and byte size;
- robot, ground, and obstacle prim paths;
- raw root spawn pose, all-zero initial servo commands, all-zero wheel targets,
  clean-grounding policy, legacy pose metadata, and a pending live grounded pose;
- authoritative 50 mm obstacle center/bounds, ground geometry, gravity,
  `dt`, render interval, solver settings, contact/rest offsets, and materials;
- servo/wheel actuator modes, limits, gains, armatures, motion reference,
  command signs, command limits, wheel directions, and the legacy measured wheel
  radius with an explicit live-readback requirement;
- Fast profile normalization and exact duration rules;
- selected runtime `150/8` and legacy metadata `30/2` as separate fields;
- instrumentation allow-list, runtime version placeholders, and mandatory live
  readback categories.

Important selected static values are:

| Fingerprinted item | Value |
|---|---|
| Robot USD | `C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd`; SHA-256 `e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892` |
| Robot prim/articulation root | `/World/WLRRobot` |
| Ground prim | `/World/defaultGroundPlane` |
| Obstacle prim | `/World/Obstacle` |
| Raw root pose `xyz+wxyz` | `[0, 0, 0.04, 1, 0, 0, 0]` |
| Obstacle bounds at 50 mm | min `[0.5213121737735307, -1.0, 0.0]`; max `[2.5786877308590377, 1.0, 0.05]` |
| Gravity / physics timestep | `[0, 0, -9.81] m/s^2`; `1/120 s` |
| Solver iterations | position `8`; velocity `2` |
| Obstacle contact/rest offsets | `0.005 m` / `0.0 m` |
| Ground friction | static `1.25`; dynamic `1.05`; restitution `0` |
| Obstacle friction | static `1.20`; dynamic `1.00`; restitution `0` |
| Servo drive | effort `2.7 Nm`; stiffness `600`; damping `60`; armature `0.005` |
| Wheel drive | velocity limit `2.0943951023931953 rad/s`; damping `20`; armature `0.002` |
| Wheel radius | legacy live-mesh metadata `0.04998999834060672 m`; runtime revalidation required |
| Servo signs | front `+1`; rear `-1` |
| Wheel forward signs | left `-1`; right `+1` |

The fingerprint deliberately emits
`status=STATIC_LOCK_RUNTIME_READBACK_PENDING` and
`environment_equivalent=false`. Default Isaac Sim, Isaac Lab, PhysX, and Torch
versions are `unknown_pending_runtime_readback`. A static hash match never
upgrades the result to `PASS`.

## Instrumentation normalization

The A and B scene configurations may differ only in:

- `telemetry_contact_sensors_enabled`
- `contact_sensor_factory`
- the separately supplied sensor readback payload

`normalize_physical_scene_config` removes exactly the first two configuration
fields. It does not ignore arbitrary fields containing words such as `sensor`,
`contact`, or `telemetry`. Any other exact configuration difference is reported
as a physical difference and fails the comparison. Sensor readback values may
differ because B exists to observe them; they cannot be copied into or used to
compensate physics settings.

A1 and A2 must use `contact_mode=formal`: the production aggregate contact
sensor with `contact_sensor_factory=None`. They are not required to expose B's
filtered wheel/non-wheel fields. B must use `contact_mode=instrumented` and
must provide complete filtered wheel and non-wheel instrumentation evidence.
All three must use the same source-freeze files map and git HEAD, recording SHA,
Fast plan/source version, USD, initialization chain, physical configuration,
command stream, device/runtime versions, timestep, seed policy, absolute
simulation-time/sample grid, and real active-viewport video. Only the two
allow-listed instrumentation fields and sensor readback may differ. No gravity,
material, solver, drive, contact-offset, cadence, or timing compensation is
permitted.

## A/A--A/B trajectory decision rule

[`compare_trajectory_equivalence`](../environment_equivalence.py#L594) requires
all three aligned runs and all eight metrics:

1. root trajectory;
2. joint trajectory;
3. wheel rotation;
4. wheel travel;
5. final pose;
6. obstacle geometry;
7. contact class;
8. contact force.

For every numeric metric, the pairwise primary error is maximum absolute error;
RMS and sample count are also reported. Contact class uses mismatch rate. Keys,
shapes, and sample alignment must be identical.

For metric `m`:

```text
self_error[m] = error(A1[m], A2[m])
tolerance[m] = max(absolute_floor[m], 3 * self_error[m])
B_error[m] = max(error(A1[m], B[m]), error(A2[m], B[m]))
metric_ok[m] = B_error[m] <= tolerance[m]
```

The absolute floors are `1e-6` for root, joint, wheel rotation, final pose, and
contact force; `1e-7` for wheel travel and obstacle geometry; and `0` for contact
class. Units follow each serialized metric. These floors cover readback
quantization only; they are not permission to tune physical parameters.

The compared contact class and force are physical quantities available in both
formal aggregate and instrumented modes; B-only filtered/non-wheel fields are
admission evidence and are never copied into A or fabricated. The comparison
fails closed when any metric is missing, non-finite, unsupported, misaligned,
structurally different, or exceeds its A/A-derived tolerance. B is checked
against both baseline repeats; closeness to only one repeat is insufficient.

The JSON writer may emit `PASS` only when all of the following are present and
true:

- instrumentation comparison `ok=true`;
- trajectory comparison `ok=true` for every metric;
- runtime readback exists with `readback_complete=true` and no runtime failure;
- the artifacts are real Isaac runs following the unseeded formal chain.
- each run contains a valid active-GUI viewport MP4 and coherent v1/v2 video
  manifests; the old `not_camera_video=true` telemetry visualization is
  explicitly insufficient.

Missing comparisons or incomplete readback produce `PENDING_RUNTIME_A_B`; an
explicit failed check produces `FAIL`. Because the required real Isaac artifacts
do not currently exist, the only accurate project-level conclusion is:

> **Environment equivalence: `PENDING`, not `PASS`.**
