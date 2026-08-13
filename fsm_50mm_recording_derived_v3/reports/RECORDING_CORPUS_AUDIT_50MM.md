# 50 mm Recording Corpus Static Audit

Status: **STATIC_INCOMPLETE**

This is a static, Isaac-free audit of the directories physically present in `height_050mm/versions`. It does not establish physical replay success, contact validity, primitive quality, or FSM success.

## Membership and selection policy

- Physical version directories discovered dynamically: **9**.
- Highest audit priority: `PRIMARY_BASELINE` for `v003_20260805_224517_157723_manual`.
- Active pointer observed: `v012_20260806_231025_027004_manual`; it is **not** a selection input.
- No recording is automatically selected. Version recency, the active pointer, or fewer steps cannot promote v012 or any other version.
- v003 is the first complete action sequence and remains the primary corpus priority; its snapshot limitations are reported rather than hidden.

Manifest discrepancies are inventory evidence, not membership overrides:

- Manifest-only IDs: `v001_20260805_162057_740045_manual, v002_20260805_185955_854964_manual, v004_20260805_224517_863127_manual`
- Disk-only IDs: `none`

## Per-version index

| Priority | Version | Static | SHA | Steps | Source / decoded commands | Servo / wheel | Atomic valid/total | FULL snapshots | Wheel segments source/fast | Duplicates | Empty waits | Overlap | Mid-stop | Schema drift |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY_BASELINE | `v003_20260805_224517_157723_manual` | FAIL | PASS | 24 | 136/202 | 138/64 | 6/6 | 16/48 | 11/11 | 0 | 0 | 3 | 11 | 16 |
| CROSS_VERSION_REFERENCE | `v005_20260805_225441_439112_manual` | PASS | PASS | 22 | 139/205 | 142/63 | 6/6 | 44/44 | 11/11 | 0 | 0 | 9 | 12 | 0 |
| CROSS_VERSION_REFERENCE | `v006_20260805_233948_654778_manual` | PASS | PASS | 22 | 135/201 | 141/60 | 6/6 | 44/44 | 10/10 | 0 | 0 | 9 | 10 | 0 |
| CROSS_VERSION_REFERENCE | `v007_20260806_190636_100857_manual` | PASS | PASS | 21 | 113/146 | 100/46 | 3/3 | 42/42 | 8/8 | 0 | 0 | 6 | 8 | 0 |
| CROSS_VERSION_REFERENCE | `v008_20260806_211408_578700_manual` | PASS | PASS | 23 | 142/197 | 137/60 | 5/5 | 46/46 | 11/11 | 0 | 0 | 8 | 11 | 0 |
| CROSS_VERSION_REFERENCE | `v009_20260806_215232_433234_manual` | PASS | PASS | 23 | 154/231 | 164/67 | 7/7 | 46/46 | 12/12 | 0 | 0 | 10 | 11 | 0 |
| CROSS_VERSION_REFERENCE | `v010_20260806_220745_363972_manual` | PASS | PASS | 26 | 168/212 | 152/60 | 4/4 | 52/52 | 11/11 | 0 | 0 | 11 | 11 | 0 |
| CROSS_VERSION_REFERENCE | `v011_20260806_223621_672618_manual` | PASS | PASS | 28 | 185/262 | 188/74 | 7/7 | 56/56 | 13/13 | 0 | 0 | 9 | 12 | 0 |
| CROSS_VERSION_REFERENCE | `v012_20260806_231025_027004_manual` | PASS | PASS | 19 | 118/162 | 113/49 | 4/4 | 38/38 | 9/9 | 0 | 0 | 6 | 9 | 0 |

Counts use these precise meanings:

- `Source commands` are accepted JSONL events and are checked against `metadata.command_count`; `decoded commands` expand atomic source events.
- `Atomic` means one source event contains both servo and wheel commands and its batch acknowledgement proves a common next physics tick.
- `Overlap` means separate events at the same source timestamp command at least one common actuator. JSONL order is retained, but this is not treated as atomic.
- `Mid-stop` means a wheel-stop command occurs before the end of the step. It is preserved as timing evidence, not silently discarded.
- Every source and authoritative Fast wheel segment is stored in the JSON index with speed, duration, source event indices, and `theta = omega * dt`.

## Aggregate evidence

- Steps: **208**
- Source events / decoded commands: **1290 / 1818**
- Servo / wheel decoded commands: **1275 / 543**
- Same-tick atomic batches: **48 / 48 valid**
- Before/after snapshots: **384 / 416 FULL_VALID**
- Source / Fast wheel segments: **96 / 96**
- Missing required fields / non-finite numerics: **0 / 0**
- Empty steps / empty waits: **0 / 0**
- Same-timestamp overlaps / snapshot-clock overlaps: **71 / 4**

## Fail-closed findings

- `v003_20260805_224517_157723_manual`: `SCHEMA_DRIFT, SNAPSHOT_INCOMPLETE`

The v003 priority does not turn incomplete evidence into a pass. A priority is an audit/repair order, not a physical primitive selection.

## Schema and evidence boundary

The JSON index contains the corpus modal key schema and every deviation. Required metadata, step, timing, motion and event fields are checked independently, and all numeric leaves are scanned for NaN/Infinity. Optional historical `null` values are counted by normalized path rather than converted to zero.

The recordings contain accepted commands and endpoint snapshots, not continuous contact/COM telemetry. Therefore every version remains physical replay `NOT_RUN` until a separately finalized physical replay supplies that evidence.
