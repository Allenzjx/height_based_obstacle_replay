# 50 mm Recording Audit

This is the static pre-replay audit. Physical success fields remain `NOT_EVALUATED` until each version has a clean Isaac Fast Replay with telemetry; endpoint JSON is not treated as proof of traversal.

- Active pointer: `v012_20260806_231025_027004_manual`
- Version directories inspected: 9 (`v003_20260805_224517_157723_manual, v005_20260805_225441_439112_manual, v006_20260805_233948_654778_manual, v007_20260806_190636_100857_manual, v008_20260806_211408_578700_manual, v009_20260806_215232_433234_manual, v010_20260806_220745_363972_manual, v011_20260806_223621_672618_manual, v012_20260806_231025_027004_manual`)
- Manifest entries missing on disk: `v001_20260805_162057_740045_manual, v002_20260805_185955_854964_manual, v004_20260805_224517_863127_manual`
- Disk directories missing from manifest: `none`

## Integrity and Fast Plan

| Version | Steps | Commands | SHA-256 | FULL_VALID snapshots | Fast segments | Fast duration (s) | Embedded media/telemetry |
|---|---:|---:|---|---:|---:|---:|---|
| v003_20260805_224517_157723_manual | 24 | 136 | PASS | 16/48 | 112 | 78.522667 | absent |
| v005_20260805_225441_439112_manual | 22 | 139 | PASS | 44/44 | 116 | 79.637333 | absent |
| v006_20260805_233948_654778_manual | 22 | 135 | PASS | 44/44 | 113 | 73.254000 | absent |
| v007_20260806_190636_100857_manual | 21 | 113 | PASS | 42/42 | 92 | 74.650667 | absent |
| v008_20260806_211408_578700_manual | 23 | 142 | PASS | 46/46 | 119 | 78.515333 | absent |
| v009_20260806_215232_433234_manual | 23 | 154 | PASS | 46/46 | 132 | 75.951333 | absent |
| v010_20260806_220745_363972_manual | 26 | 168 | PASS | 52/52 | 142 | 82.467333 | absent |
| v011_20260806_223621_672618_manual | 28 | 185 | PASS | 56/56 | 158 | 84.588667 | absent |
| v012_20260806_231025_027004_manual | 19 | 118 | PASS | 38/38 | 99 | 69.699333 | absent |

## Environment lock findings

- All available version metadata use the same v2 obstacle geometry and robot asset identity.
- No version directory contains a standalone `environment.json`; authoritative values are therefore traced to `config/fsm_recording_baseline.yaml`, `config/environment_reference.yaml`, per-version metadata, and their SHA-256 values.
- The embedded bookkeeping profile names **30 deg/s** and **0.3 rad/s**, while the authoritative code actually consumed by `playback.plan_from_steps` and `SimRobotAdapter` uses **150 deg/s** and **0.5235987756 rad/s**. The Fast Replay lock uses the runtime values and preserves the stale bookkeeping discrepancy as evidence.
- The recording baseline document says render interval 2, while the current UI/worker Fast Replay path defaults to 8 and explicitly treats render cadence as a non-physics performance override. The reproduction lock uses 8 and records both values.

## Evidence limits before replay

The accepted steps contain commands and before/after articulation snapshots, but not continuous whole-body COM, per-wheel contact force/class, support diagonal, contact drift, or video. Consequently this audit does **not** label any version a full success or select primitives yet. Those decisions require the clean replay artifacts requested by the task.

## Generated evidence

- Environment lock: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\environment_lock_50mm.json`
- Machine-readable audit: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\recording_audit_50mm.json`
- Version matrix: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\RECORDING_VERSION_MATRIX_50MM.csv`
- Fast plans: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\recording_fast_plans`
