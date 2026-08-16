# Current background process stop report

Pre-stop snapshot: `2026-08-16T05:14:14.9883707Z`

Requested final evaluation state: `INTERRUPTED_BY_USER_FOR_FSM_REVIEW`  
Physical promotion: `false`

## Process ownership before interrupt

The only live project runtime tree was:

```text
Codex desktop/app server (do not stop)
└─ PID 20844 pwsh.exe — Codex owning terminal
   └─ PID 4176 python.exe — evaluation coordinator and singleton owner
      └─ PID 48284 python.exe — play_fsm50_residual_ppo + embedded Isaac/Kit
```

There was no separate `kit.exe`, `isaac-sim`, training child, Macro FSM worker, or second simulator. PID `60448` was the read-only snapshot process itself and exited after producing the snapshot. Desktop ancestors PID `18544`, `19516`, and app-server PID `36404` are not stop targets.

Exact evaluation child command:

```text
C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe -m fsm_50mm_recording_derived_v3.play_fsm50_residual_ppo --stage R1 --checkpoint-manifest C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\fsm_residual_ppo\r1_phase_local_seed20260815_lockf270_smoke01\fsm50_residual_ppo_checkpoint_manifest.json --output C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\fsm_residual_ppo\r1_phase_local_seed20260815_lockf270_smoke01\fsm50_residual_ppo_evaluation.json --seed 20260815 --episodes-per-entry 3 --device cuda:0 --headless
```

Coordinator command identity:

```text
PID 20844 pwsh.exe -Command <inline coordinator>
└─ PID 4176 "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe" -
```

The inline coordinator acquired `ReplaySingletonLock`, ran the exact child command above with `subprocess.run`, required a canonical three-arm `36/36/36` artifact, then intended to release the lock in `finally`.

| PID | Parent | Role | CPU seconds | Working set bytes |
|---:|---:|---|---:|---:|
| 20844 | 36404 | Codex owning terminal | 2.078125 | 81,702,912 |
| 4176 | 20844 | evaluation coordinator / lock owner | 0.3125 | 16,441,344 |
| 48284 | 4176 | PPO evaluation child + embedded Isaac/Kit | 21,351.078125 | 598,507,520 |

## Singleton before interrupt

- Path: `C:\Users\kskzz\AppData\Local\Temp\fsm50_replay_0ab14a3020e34b71.pid.lock`
- Exists: `true`
- SHA-256: `b66a4c46199059a178b86964942ba6fb6a880c3ab3b8fd433b9d0441f4724eb4`
- Owner PID: `4176` (alive at snapshot)
- Owner token: `3b0ed81fca284bedad2f21620fe58358`
- Recovery: send exactly one normal interrupt to the owning terminal; allow coordinator cleanup first. Use owner-checked stale recovery only if the process tree has exited without releasing it. Never blind-delete it.

## Training artifacts before interrupt

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `fsm50_residual_ppo_checkpoint.pt` | 3,128,101 | `582df8517f26bd6c1820e54625621813882ca132964b20519ac26b989665d8a0` |
| `fsm50_residual_ppo_checkpoint_manifest.json` | 26,673 | `1058d431000877fbde7d49cb25d4480c610aad466aff191fa9f37c21a6b5ac3a` |
| `fsm50_residual_ppo_training_manifest.json` | 3,416 | `40040ba479d1d3fd41ce301be84f7ca0985db3263bc2c45720b4ab1ddd9bd601` |

The formal evaluation output did not exist at the pre-stop snapshot:

```text
...\r1_phase_local_seed20260815_lockf270_smoke01\fsm50_residual_ppo_evaluation.json
exists = false
size = 0
partial/tmp candidates = []
```

Therefore this run cannot be labeled `PASS`, `FAIL`, or `PPO_PROMOTED`. Its requested terminal classification is `INTERRUPTED_BY_USER_FOR_FSM_REVIEW`; `physical_promotion=false`.

## Git before report creation

- HEAD: `2acd2852ee73e1d378de26df1abb0b86845ae4a1`
- Last commit: `2acd285 Update latest obstacle replay FSM code and validation artifacts`
- Branch: `main`
- `git status --short`: empty
- `git diff --stat`: empty
- `git diff --check`: empty

## Stop action and verified result

Exactly one normal `Ctrl+C` was sent to Codex owning terminal session `43503`. The session returned exit code `1`, as expected for an interrupt. No second interrupt and no process-name-wide stop command was issued.

Post-stop verification:

- PID `20844` owning terminal: not alive.
- PID `4176` evaluation coordinator / singleton owner: not alive.
- PID `48284` evaluation child with embedded Isaac/Kit: not alive.
- Standalone Isaac/Kit survivors: none.
- Simulator conflicts: `[]`.
- Formal evaluation output: absent.
- Partial evaluation/tmp candidates: none.

The interrupt ended the process tree before the coordinator could release its singleton. The remaining file still named dead owner PID `4176` and token `3b0ed81fca284bedad2f21620fe58358`. After rechecking that the child and simulator were gone and all three training artifact hashes were unchanged, the project’s production `ReplaySingletonLock.acquire()` performed its built-in dead-owner/token re-read recovery. Recovery temporarily acquired the same lock as PID `43456`, token `2750fda8c0004cb49baa29a1fb87a5e2`, then owner-checked `release()` removed it. Singleton exists after recovery: `false`.

Post-stop artifact hashes are exactly unchanged:

| Artifact | SHA-256 after stop |
|---|---|
| `fsm50_residual_ppo_checkpoint.pt` | `582df8517f26bd6c1820e54625621813882ca132964b20519ac26b989665d8a0` |
| `fsm50_residual_ppo_checkpoint_manifest.json` | `1058d431000877fbde7d49cb25d4480c610aad466aff191fa9f37c21a6b5ac3a` |
| `fsm50_residual_ppo_training_manifest.json` | `40040ba479d1d3fd41ce301be84f7ca0985db3263bc2c45720b4ab1ddd9bd601` |

Final status:

```text
INTERRUPTED_BY_USER_FOR_FSM_REVIEW
physical_promotion = false
```
