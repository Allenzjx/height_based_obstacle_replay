# Code Completion Matrix

Last updated: 2026-08-09

Status meanings: **EXISTING** = present at the starting checkpoint;
**IMPLEMENTED** = now present with pure-Python contract tests;
**PARTIAL** = useful code exists but required selection/wiring is incomplete;
**PENDING PHYSICS** = code alone cannot establish completion.

| Requirement | Starting checkpoint | Current code status | Evidence / remaining gate |
|---|---:|---:|---|
| Recording audit and physical version enumeration | EXISTING | EXISTING | Nine physical version directories enumerated |
| Official Fast-plan reuse | EXISTING | EXISTING | `recording_fast_plan.py`; offline plans for v003/v005–v012 |
| Resumable per-version live replay artifacts | PARTIAL | IMPLEMENTED | Durable finalization/resume/source/video checks implemented; real reliable result remains 0/9 |
| Filtered wheel-contact telemetry | EXISTING | EXISTING | `filtered_wheel_contact.py` plus tests |
| Non-wheel obstacle-contact telemetry | EXISTING | EXISTING | `nonwheel_obstacle_contact.py` plus tests |
| Support classifier / diagonal corridor | EXISTING | EXISTING | `support_classifier.py` plus tests |
| Traversal evidence / illegal drive-up rejection | EXISTING | EXISTING | Per-leg evidence tracker plus tests |
| COM guard/detector helpers | EXISTING | EXISTING | `com_transfer_primitives.py` |
| Command-side impulse primitive | MISSING | PARTIAL | PRELOAD→PUSH→RELEASE→COAST→SETTLE→VERIFY generator implemented; not selected across all recordings or wired to live correction/IK |
| Command-side anchored support-angle primitive | MISSING | PARTIAL | Anchor/ramp/hold/settle/verify generator implemented; same runtime-selection/wiring gap |
| Unified live FSM observation | MISSING | IMPLEMENTED | `fsm50_observation.py` strict model and tests |
| Atomic Servo+Wheel FSM executor | MISSING | IMPLEMENTED | `fsm50_executor.py` adapter contract and fake-adapter tests |
| Active-leg correction isolation runtime wiring | PARTIAL | PARTIAL | Helpers and command generators exist; live correction feedback is not wired into the transfer generators |
| Per-leg IK acceptance runtime wiring | PARTIAL | PARTIAL | Acceptance helpers exist; transfer generators do not yet consume live per-leg IK corrections |
| Complete guard registry with startup validation | MISSING | IMPLEMENTED | `fsm50_guard_registry.py`; YAML resolution/startup validation tests |
| State entry/update/exit lifecycle | MISSING | IMPLEMENTED | `fsm50_controller.py` lifecycle and tests |
| Retry/timeout/SAFE_STOP runtime | MISSING | IMPLEMENTED | Controller/executor behavior and tests |
| Actual A0→F5 controller | MISSING | IMPLEMENTED | Controller code exists; full real-Isaac completion remains 0 |
| `run-fsm` CLI | MISSING | IMPLEMENTED | Command and runtime dispatch exist; no successful full run |
| `test-state` CLI with restore/prefix provenance | MISSING | IMPLEMENTED | `fsm50_state_restore.py` and CLI validation; 0/57 state results |
| `validate-5` CLI | MISSING | IMPLEMENTED | Command exists; physical gate remains 0/5 |
| Environment fingerprint | PARTIAL | IMPLEMENTED | Hash-backed static fingerprint with runtime placeholders |
| Scene factory regression tests | MISSING | IMPLEMENTED | Default/custom factory and non-mutation contracts tested without Isaac |
| Environment artifact converter/report writer | MISSING | IMPLEMENTED | Strict A1/A2 formal + B instrumented converter, source closure, sample grid, eight metrics, real-video gate |
| Runtime A/A and A/B environment equivalence | PENDING PHYSICS | PENDING PHYSICS | A1/A2/B do not exist; no real report `PASS` |
| All-recording primitive selection and provenance | PARTIAL | PARTIAL | Offline matrix/alignment exist; all 57 states remain `PENDING_REPLAY` |
| Actual active-viewport video | MISSING | IMPLEMENTED | Capture/manifest/SHA/container validation code and tests exist; no qualifying real artifact yet |
| Replay grounding parity | PARTIAL | IMPLEMENTED | Formal path uses unseeded clean reset/settle chain; stale batch predates the fix and failed ground init |
| Isaac-free regression suite | 44/44 | IMPLEMENTED | Focused FSM/telemetry: 197 passed + 44 subtests; complete repository: 495 passed + 54 subtests |
| State-level Isaac validation | PENDING PHYSICS | PENDING PHYSICS | 0/57; every state remains `PENDING_REPLAY` |
| Complete A0→F5 Isaac validation | PENDING PHYSICS | PENDING PHYSICS | 0 successful full runs |
| Five clean full-FSM validations | PENDING PHYSICS | PENDING PHYSICS | 0/5 |

“IMPLEMENTED” in this table never means physically validated. The runtime
completion gate still requires real environment A/A--A/B evidence, all nine
recording replays, all state checks, a full run, and 5/5 clean repetitions.
