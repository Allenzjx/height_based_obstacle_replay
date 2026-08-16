# 50 mm Support-Diagonal Findings

## 结论

Gate-B 成功 Recording 一致显示：PRE_FR 和 PRE_RL transfer windows 中，FL/RR 反复形成所需的 **candidate support geometry**。Gate-C v003 的 S7 guard也在 baseline + 3 repeats中四次观察到 `FL=true, RR=true`，随后 RL 完成 unload、cross和TOP。因此“FL/RR-biased workspace 是 v003 RL traversal 前有效且可重复的几何条件”有直接证据。

当前证据不能升级为“FL/RR 是唯一主要承重腿”“temporary two-leg support 已测得”“接触点 anchored”或“对角载荷平衡已验证”。normal Gate-A/Gate-C telemetry 没有 contact load/force，且 S6 tail 附近常有多于两条腿同时属于 geometry support candidate。

## Cross-recording geometry candidates

来源：[`50MM_COMMON_PHASE_ALIGNMENT.csv`](50MM_COMMON_PHASE_ALIGNMENT.csv) 和 [`50MM_COMMON_PHASES.md`](50MM_COMMON_PHASES.md)。每个集合都是 wheel class/geometry候选，不是 force-ranked support set。

| Phase | v003 | v008 | v009 | v010 | Safe finding |
|---|---|---|---|---|---|
| PRE_FR_COM_SHIFT | FL, RR | FL, RR | FL, RR | FL, RR | FR unload前，FL/RR diagonal candidate在四个成功版本中一致 |
| PRE_RR_COM_SHIFT | FR, RL | FR, FL, RL | FL, RL | FR, FL, RL | 后腿准备策略存在 categorical差异；不应平均或强制一个 diagonal |
| PRE_RL_COM_SHIFT | FL, RR | FL, RR | FL, RR | FL, RR | RL lift前，FL/RR candidate geometry在四个成功版本中一致 |

`PRE_RL_SUPPORT_SETUP` 在 Gate-B 没有可单独切出的通用 command window，所以没有人为发明一段动作。runtime graph仍保留 S7 作为 feedback state：它确认当前 geometry是否适合进入 RL profile。

## Gate-C v003 causal sequence

| Evidence point | Baseline + repeats | What it proves | What it does not prove |
|---|---|---|---|
| S6 RR traversal complete | RR airborne-before-crossing、crossed、TOP；FR/FL/RR TOP history ready | rear-right traversal完成且进入 support setup boundary | 各腿normal load大小 |
| S7 → S8 at sim step 7693 / epoch 58 | `candidate_support_evidence={FL:true, RR:true}` in all 4 runs | FL/RR geometry candidates同时存在 | FL/RR anchored、exclusive或承担主要载荷 |
| S8 RL profile | v003 segments 57–101；segment 57 same-target no-op after boundary，后续 source order完整 | 没有借用/遗漏 Recording action；support guard后才开始RL owner stream | 哪个候选腿提供了多少 reaction force |
| S8 → S9 at step 8115 / epoch 102 | RL airborne-before-crossing、crossed、TOP | 候选 workspace与完整 profile共同产生成功 RL traversal | 仅靠 diagonal geometry即可产生同样结果 |
| Final | 4/4 task complete；FL=AIR，其他三轮TOP；recoverable | traversal成功、recovery仍不完整 | final all-loaded 或 load balance |

S7 在 v003 中是纯 feedback-only state，零 source action。profile library 的静态 cross-version ownership显示 v009 是一个有证据的例外：它在 S7 拥有 segments 91–92 的 causal prefix，因为 FL 在 S6 tail为 AIR，segment 92 后才重新进入 TOP；这说明 graph允许 source-specific profile ownership，而不是把 v003 的空 S7 机械套给每个版本。该设计事实不等于 v009 已完成 live Gate-D validation。

## Runtime measurement boundary

[worker_macro_fsm_session.py](../worker_macro_fsm_session.py) 在本次 normal mode 中明确输出：

```text
wheel_contact_load_n = {FL: None, FR: None, RL: None, RR: None}
wheel_contact_load_available = false
filtered_contact_bank_enabled = false
wheel_classification = GEOMETRY_ONLY
com_measurement_available = false
```

[fsm50_macro_controller.py](../fsm50_macro_controller.py) 的 `leg_support_candidate()` 只有在 load可用时才使用 `load > 0`；否则 `GROUND` 或 `TOP` 即为 candidate。transition evidence也将 claim写成 `candidate geometry/load only; not an anchored-support claim`。

在 accepted v003 baseline 的 S6→S7 边界附近，telemetry 的 `geometry_support_candidate_count` 为 4，说明 FL/RR 虽然满足 S7 required subset，但不是当时唯一可见候选。因此不能从 S7 boolean guard推导“只有两腿承重”。

## Approved terminology

在 load/force/contact-manifold 证据缺失时使用：

- `FL/RR candidate support geometry`；
- `FL/RR-biased support workspace`；
- `SUPPORT_CONTACT_WITH_BODY_MOTION`；
- `PASSIVE_SUPPORT_RESPONSE`；
- `LOADED_ZERO_COMMAND_CONTACT` 仅在未来确有 load evidence 时使用。

当前不得使用：

- `ANCHORED_FL_RR`；
- `exclusive two-leg support`；
- `balanced diagonal load`；
- `measured support margin`；
- `no-slip contact`。

同样，zero wheel target不表示 wheel center在世界坐标中静止，passive displacement不表示 command failure。只有 passive response导致 fall、必要支撑丢失、swing leg无法cross、危险碰撞或不可恢复时，才升级为 task failure。

## Control and future optimization implications

第一版 nominal FSM 的正确做法已经实现：S7只要求 FL/RR candidate geometry，保持 servo target、zero wheels并 bounded recheck；它不等待虚构的 load threshold，也不 replay profile。四次无 retry成功说明该条件对 v003 deterministic baseline足够。

未来 PPO residual若进入 Gate E，可以把 phase、candidate support mask、base attitude和active-wheel clearance作为 actor可用输入；真实 load/support margin只有在部署时也可用时才进入 actor，否则只可作为 privileged critic/reward diagnostics。不得仅因 passive drift惩罚策略，也不得让 residual改变 FR→FL→RR→RL 顺序或绕过 S7/S8 的必要 RL lift。

COM证据边界见 [50MM_COM_TRANSFER_FINDINGS.md](50MM_COM_TRANSFER_FINDINGS.md)，完整 Gate-C运行见 [V003_MACRO_FSM_VALIDATION.md](V003_MACRO_FSM_VALIDATION.md)。
