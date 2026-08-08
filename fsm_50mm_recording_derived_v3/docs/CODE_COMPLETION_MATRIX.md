# Code Completion Matrix

Last updated: 2026-08-08

Status meanings: **EXISTING** = present and baseline-tested; **PARTIAL** = useful implementation exists but required runtime wiring/evidence is missing; **MISSING** = no implementation at the starting checkpoint; **PENDING PHYSICS** = code alone cannot establish completion.

| Requirement | Starting checkpoint status | Current status | Evidence / next gate |
|---|---:|---:|---|
| Recording audit and physical version enumeration | EXISTING | EXISTING | Nine physical version directories enumerated |
| Official Fast-plan reuse | EXISTING | EXISTING | `recording_fast_plan.py`; offline plans for v003/v005–v012 |
| Resumable per-version live replay artifacts | PARTIAL | PARTIAL | Batch supervision exists; first batch failed before commands; no per-version result |
| Filtered wheel-contact telemetry | EXISTING | EXISTING | `filtered_wheel_contact.py` plus tests |
| Non-wheel obstacle-contact telemetry | EXISTING | EXISTING | `nonwheel_obstacle_contact.py` plus tests |
| Support classifier / diagonal corridor | EXISTING | EXISTING | `support_classifier.py` plus tests |
| Traversal evidence / illegal drive-up rejection | EXISTING | EXISTING | Per-leg evidence tracker plus tests |
| COM guard/detector helpers | EXISTING | EXISTING | `com_transfer_primitives.py` |
| Command-side impulse primitive | MISSING | MISSING | Needs PRELOAD→PUSH→RELEASE→COAST→SETTLE→VERIFY generator |
| Command-side anchored support-angle primitive | MISSING | MISSING | Needs anchor/ramp/hold/settle/verify generator |
| Unified live FSM observation | MISSING | MISSING | `fsm50_observation.py` or equivalent required |
| Atomic Servo+Wheel FSM executor | MISSING | MISSING | Formal adapter protocol and fake-adapter test required |
| Active-leg correction isolation runtime wiring | PARTIAL | PARTIAL | Helper/tests exist; controller wiring absent |
| Per-leg IK acceptance runtime wiring | PARTIAL | PARTIAL | Decision helper exists; controller wiring absent |
| Complete guard registry with startup validation | MISSING | MISSING | Every YAML guard must resolve; no placeholder/always-true |
| State entry/update/exit lifecycle | MISSING | MISSING | No controller at starting checkpoint |
| Retry/timeout/SAFE_STOP runtime | MISSING | MISSING | State model carries metadata only |
| Actual A0→F5 controller | MISSING | MISSING | `fsm50_controller.py` absent at starting checkpoint |
| `run-fsm` CLI | MISSING | MISSING | Starting CLI supports only audit/replay/report |
| `test-state` CLI with restore/prefix provenance | MISSING | MISSING | Requires runtime and trusted restore input |
| `validate-5` CLI | MISSING | MISSING | Requires runtime and environment gate |
| Environment fingerprint | PARTIAL | PARTIAL | Offline `environment_lock_50mm.json` exists; requested complete fingerprint/report absent |
| Scene factory regression tests | MISSING | MISSING | Must prove default/custom factory behavior and config non-mutation |
| Runtime A/A and A/B environment equivalence | PENDING PHYSICS | PENDING PHYSICS | Must compare clean settle and identical replay trajectories |
| All-recording primitive selection and provenance | PARTIAL | PARTIAL | Matrix/alignment exist; config is mainly v012 and PENDING_REPLAY |
| Actual camera video | MISSING | MISSING | Telemetry visualization is explicitly not camera video |
| State-level Isaac validation | PENDING PHYSICS | PENDING PHYSICS | 0/9 run |
| Five clean full-FSM validations | PENDING PHYSICS | PENDING PHYSICS | 0/5 run |

