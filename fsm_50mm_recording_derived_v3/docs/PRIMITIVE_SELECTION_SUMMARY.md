# 50 mm primitive selection summary

Last updated: 2026-08-09 (America/New_York)

## Verdict

No FSM primitive has been promoted to `PHYSICALLY_VERIFIED`.

All nine physical recording versions have been enumerated and converted into
offline audit/Fast-plan inputs, but none has a reliable completed live replay.
The first formal A1 attempt failed during clean grounding before any recording
command. Selecting a source primitive now would therefore confuse offline
candidate analysis with physical replay evidence.

## Available source recordings

| Short ID | Physical version | Live replay status |
|---|---|---|
| v003 | `v003_20260805_224517_157723_manual` | `PENDING` |
| v005 | `v005_20260805_225441_439112_manual` | `PENDING` |
| v006 | `v006_20260805_233948_654778_manual` | `PENDING` |
| v007 | `v007_20260806_190636_100857_manual` | `PENDING` |
| v008 | `v008_20260806_211408_578700_manual` | `PENDING` |
| v009 | `v009_20260806_215232_433234_manual` | `PENDING` |
| v010 | `v010_20260806_220745_363972_manual` | `PENDING` |
| v011 | `v011_20260806_223621_672618_manual` | `PENDING` |
| v012 | `v012_20260806_231025_027004_manual` | `PENDING` |

Offline analysis inputs exist at:

- `reports/RECORDING_VERSION_MATRIX_50MM.csv`
- `reports/RECORDING_PHASE_ALIGNMENT_50MM.csv`
- `reports/RECORDING_TO_FSM_PROVENANCE.csv`
- `reports/recording_fast_plans/`

These files may rank candidates, but they cannot provide the required live
root/joint/contact/support/COM compatibility proof.

## Required primitive decisions

Every required choice remains pending:

| Phase | Selected source | Verification |
|---|---|---|
| COM to RL | none | `PENDING_REPLAY` |
| FR reaction preload/pulse | none | `PENDING_REPLAY` |
| FR unload/lift | none | `PENDING_REPLAY` |
| FR advance | none | `PENDING_REPLAY` |
| FR place | none | `PENDING_REPLAY` |
| FL COM transfer | none | `PENDING_REPLAY` |
| FL unload/lift/place | none | `PENDING_REPLAY` |
| Front-pair advance | none | `PENDING_REPLAY` |
| COM to FL | none | `PENDING_REPLAY` |
| RR unload/lift/place | none | `PENDING_REPLAY` |
| FL+RR diagonal preparation | none | `PENDING_REPLAY` |
| RL lever preload | none | `PENDING_REPLAY` |
| RL downward reaction pulse | none | `PENDING_REPLAY` |
| COM to FR | none | `PENDING_REPLAY` |
| RL unload/lift/place | none | `PENDING_REPLAY` |
| Final advance | none | `PENDING_REPLAY` |
| Concurrent home recovery | none | `PENDING_REPLAY` |

Consequently:

- no final primary-diagonal sequence is claimed;
- no COM transfer is assigned to impulse versus anchored support-angle;
- no cross-version splice is accepted;
- every one of the 57 YAML states remains `PENDING_REPLAY`;
- v012 endpoint candidates remain candidates only, not selected truth.

## Promotion gate

A primitive may be promoted only after the environment A1/A2/B report is a
real `PASS`, the source recording has a reliable live Fast replay with actual
viewport video, and the connection boundary is compatible in root pose,
measured joints, wheel angle/velocity, contact class, support set, primary
diagonal, COM state, body attitude/rate, and obstacle-relative wheel position.
The final provenance row must include version, step/event/Fast-segment indices,
telemetry interval, run directory, source SHA-256, selection reason, and the
compatibility result.

