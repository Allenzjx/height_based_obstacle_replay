# Resume After Usage Limit Snapshot

Captured at `2026-08-16T17:05:48.5731388Z` before any resumed code edit or simulator launch.

## Resume authority

- Request: `C:\Users\kskzz\.codex\attachments\5cd0f4c7-ff66-4b8a-bc85-91fab2e67ec3\pasted-text.txt`
- Size: `26,754` bytes; `1,401` lines
- SHA-256: `8d0e04e97379adede126d11973501a1f4e9f69c1874d4b1386405291ca325b9d`

## Repository snapshot

- Project: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay`
- Branch: `main`
- HEAD: `551f868ca827a39a451d7777b33a647cd5aa353f`
- `git status --short`: empty at capture; the worktree was clean.
- Most recent commit: `551f868ca827a39a451d7777b33a647cd5aa353f` — `Update latest Codex obstacle replay FSM implementation`.

## Runtime and singleton snapshot

The process scan found **zero** project Python, Isaac Sim, Kit, IsaacLab, FSM, replay, or PPO runtime processes. Two `codex.exe` application processes were present (PIDs `36404` and `58184`); neither was classified as a project runtime.

The singleton path `C:\Users\kskzz\AppData\Local\Temp\fsm50_replay_0ab14a3020e34b71.pid.lock` was absent. No stale-lock recovery or deletion was performed.

## Preserved prior PPO artifact

The prior R1 smoke checkpoint remains byte-identical and is not physically promoted:

- Checkpoint: `3,128,101` bytes, SHA-256 `582df8517f26bd6c1820e54625621813882ca132964b20519ac26b989665d8a0`
- Checkpoint manifest: `26,673` bytes, SHA-256 `1058d431000877fbde7d49cb25d4480c610aad466aff191fa9f37c21a6b5ac3a`

## Resume decision

Status: `SAFE_TO_CONTINUE_OFFLINE_LEDGER_CLOSURE_ONLY`.

No Isaac, Macro FSM, Fast Replay, updated v003, evaluation, or PPO process may start from this snapshot. The next admissible work is the existing-ledger external anchor closure, its tests, one source-lock reseal, and two stable read-only verifications. Only then may exactly one updated v003 FSM validation run be considered.

`PPO_RETRAINING_PAUSED`

`PHYSICAL_PROMOTION_FALSE`
