# 50 mm FSM run commands

Last updated: 2026-08-09 (America/New_York)

All commands use repository root:

```text
C:\robotics_sim\wlr_robot\height_based_obstacle_replay
```

and the existing environment interpreter:

```text
C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe
```

## Commands actually run at this checkpoint

Pure-Python complete repository regression:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -m pytest -q -p no:cacheprovider tests fsm_50mm_recording_derived_v3/tests
```

Result: `495 passed, 54 subtests passed`. A focused FSM/telemetry selection
also produced `197 passed, 44 subtests passed`. These are code-contract results,
not physics evidence.

Post-A1 supervisor/affected regression commands:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -m pytest -q -p no:cacheprovider fsm_50mm_recording_derived_v3/tests/test_runner_contract.py
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -m pytest -q -p no:cacheprovider fsm_50mm_recording_derived_v3/tests tests/test_fsm50_filtered_wheel_contact.py tests/test_fsm50_nonwheel_obstacle_contact.py tests/test_fsm50_state_model.py tests/test_com_transfer_primitives.py
```

Results: `27 passed + 9 subtests` and `231 passed + 47 subtests`. Two
post-change attempts at the complete combined command each produced
`495 passed + 57 subtests + 1 Tk initialization failure`; the failing GUI test
changed between attempts, and each passed when rerun alone. This is not recorded
as a fully green post-change combined run.

Syntax verification:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m compileall -q fsm_50mm_recording_derived_v3
```

Recording audit and lock refresh:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m fsm_50mm_recording_derived_v3.run_fsm50 audit
```

Static environment report:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m fsm_50mm_recording_derived_v3.run_fsm50 validate-environment
```

Expected current exit: nonzero, because the truthful report is
`PENDING_RUNTIME_A_B`, not `PASS`.

Formal A1 attempt actually run:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -u -m fsm_50mm_recording_derived_v3.run_fsm50 replay-recordings --versions v012 --contact-mode formal --output-root 'C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\environment_equivalence\A1' --fail-fast
```

Result: failed before replay at initial grounding; no A1 artifact was admitted.

## Existing commands that have not passed the physical gates

The following are real CLI surfaces, but they were deliberately not launched
after A1 failed. They are examples for the next authorized run, not claims of
execution:

```powershell
# Formal repeat, only after the A1 grounding contract is resolved.
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -u -m fsm_50mm_recording_derived_v3.run_fsm50 replay-recordings --versions v012 --contact-mode formal --output-root 'C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\environment_equivalence\A2' --fail-fast

# Instrumented B, only after admissible A1/A2 runs exist.
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -u -m fsm_50mm_recording_derived_v3.run_fsm50 replay-recordings --versions v012 --contact-mode instrumented --output-root 'C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\environment_equivalence\B' --fail-fast

# All recordings, only after a real environment-equivalence PASS.
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -u -m fsm_50mm_recording_derived_v3.run_fsm50 replay-recordings --versions all --contact-mode instrumented --resume

# Runtime entry points; exact restore/report arguments must reference verified artifacts.
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m fsm_50mm_recording_derived_v3.run_fsm50 test-state --help
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m fsm_50mm_recording_derived_v3.run_fsm50 run-fsm --help
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m fsm_50mm_recording_derived_v3.run_fsm50 validate-5 --help
```

Do not run A2, B, all recordings, state tests, full FSM, or `validate-5` while
the preceding gate is unresolved. Never start a second Isaac instance, and do
not use `--headless` or `--no-video` for evidence-bearing replay runs.
