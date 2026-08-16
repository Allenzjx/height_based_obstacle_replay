# v003 50 mm Recording-Derived Macro FSM Map

## 封存控制身份

| Artifact | Identity |
|---|---|
| Graph | `fsm50-recording-derived-macro-v1` |
| Graph SHA-256 | `ffa5acfbf64b65c22eee54709a2afae5a56fa0b9345d8db84eb86acac58447c5` |
| Profile library | `fsm50-gate-c-successful-recording-profiles-v1` |
| Profile-library SHA-256 | `3fb1501c40a8669681f5a073036a496ae221b7d91c3ceb629e93e37ac7c2ceea` |
| Gate-C bundle SHA-256 | `0579218825bcfdd4dabf7ab6225268ac04279a83ed96cad2ed390955414deebc` |
| v003 source-plan SHA-256 | `a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c` |
| Gate-B alignment SHA-256 | `f1b0e55b76ffddcf45c3727a490e3355a3531840443e0b086f743cd22efce358` |
| Active-state count | 11（S0..S10）；另有 `SUCCESS`、`SAFE_STOP` |
| Legacy graph | 57 states，`LEGACY_SEMANTIC_GRAPH`，无 Gate-C control authority |

正常成功路径固定为：

```text
S0 → S1 → S2(FR) → S3(FL) → S4 → S5 → S6(RR)
   → S7 → S8(RL) → S9 → S10 → SUCCESS
```

任何 hard failure、bounded timeout 或 exhausted hold/retry 都进入 `SAFE_STOP`；本次 accepted v003 baseline 和三次 repeat 都没有进入该失败状态。

## State map

候选支撑腿是几何/可用 load 的候选集合，不表示 anchored 或主要载荷路径。`profile complete` 对有 profile 的 state 是 transition 必要条件，但从来不是充分条件。

| State | Physical purpose / active leg | Candidate support | v003 source owner | Live completion guard | Bounded hold / retry | Next |
|---|---|---|---|---|---|---|
| S0_INITIALIZE | finite live state、writable target、建立 episode/base-or-COM baseline | — | feedback-only | finite articulation + actuator targets applied；profile completion 不要求 | minimum timeout 1.0 s | S1 |
| S1_APPROACH_AND_PRE_FR_SHIFT | approach；向 RL 候选区域转移 body/COM，使 FR unload / active FR | FL, RL, RR | segments 0–6；`PRIMARY_PROFILE` | profile complete +（FR phase-local unload **或** toward-RL projected displacement ≥ 0.003 m）+ attitude ≤35° | 0.8 s；最多 1 次 hold-target guard recheck；不 rewind | S2 |
| S2_FR_TRAVERSE | FR unload、lift、face clear、TOP place / active FR | FL, RL, RR | segments 7–23；`PRIMARY_PROFILE` | profile complete + FR airborne-before-crossing latch + crossed + phase-local TOP | 2.25 s hold；0 replay retry | S3 |
| S3_FL_TRAVERSE | FL unload、lift、face clear、TOP place / active FL | FR, RL, RR | segments 24–40；`PRIMARY_PROFILE` | profile complete + FL airborne-before-crossing + crossed + phase-local TOP + episode FR TOP | 6.25 s hold；0 replay retry | S4 |
| S4_FRONT_PAIR_ADVANCE | 确认 front-pair advancement | FL, FR, RL, RR | feedback-only in v003 | FR/FL TOP +（body progress ≥0.03 m、fresh inherited progress 或 rear-face approach） | 0.8 s live observation | S5 |
| S5_PRE_RR_COM_SHIFT | 向 FL 候选区域移动 body/COM，使 RR unload / active RR | FR, FL, RL | feedback-only in v003 | timeline trivially complete +（RR unload **或** toward-FL projected displacement ≥0.003 m，允许 fresh S4 carry） | 0.8 s；最多 1 次 guard recheck；不 rewind | S6 |
| S6_RR_TRAVERSE | RR unload、lift、face clear、TOP place / active RR | FR, FL, RL | segments 41–56；`PRIMARY_PROFILE` | profile complete + RR airborne-before-crossing + crossed + phase-local TOP + episode FR/FL TOP | 3.75 s hold；0 replay retry | S7 |
| S7_PRE_RL_SUPPORT_SETUP | 建立 RL lift 前的 FL/RR-biased candidate workspace / active RL | FR, FL, RR | feedback-only in v003 | FL 和 RR 均为 `leg_support_candidate`；本次无 load，故实际依据为 GROUND/TOP geometry class | 0.8 s；最多 1 次 guard recheck；不 rewind | S8 |
| S8_RL_COM_SHIFT_AND_TRAVERSE | toward-FR body/COM shift + RL unload/lift/cross/TOP / active RL | FR, FL, RR | segments 57–101；`PRIMARY_PROFILE` | profile complete + RL airborne-before-crossing + crossed + phase-local TOP；final simultaneous all-TOP 不要求 | 7.25 s hold；0 replay retry | S9 |
| S9_FINAL_ADVANCE | body 越过 front face并产生 recovery workspace | FL, FR, RL, RR | segments 102–103；`PRIMARY_PROFILE` | body crossed +（state/inherited body progress ≥0.03 m 或 ≥3 wheels crossed）+ attitude ≤35° | 1.0 s live observation | S10 |
| S10_POSTURE_RECOVERY | 执行 v003 recovery profile并以 recoverable task state 结束 | FL, FR, RL, RR | segments 104–111；`RECOVERY_PROFILE_2` | profile complete + body crossed + final recoverable + attitude ≤30°；`posture_complete` 只记录、不阻止 task success | 1.5 s；最多 1 次 guard recheck；不 replay recovery | SUCCESS |

动态 timeout 由 [fsm50_macro_controller.py](../fsm50_macro_controller.py) 的 state-local profile duration、`timeout_scale`、hold budget 和 retry budget计算；它限制动作持续时间，但不会把 timeout 当成成功事件。

## v003 source ownership 与 physical dispatch

| Owner | Source segments | Count | Runtime note |
|---|---:|---:|---|
| S1 | 0–6 | 7 | approach / pre-FR profile |
| S2 | 7–23 | 17 | segment 7 在 S1→S2 wheel-zero boundary 后为相同 target no-op |
| S3 | 24–40 | 17 | FL post-cross TOP 使用 phase-local latch；不会借用后续 RR command |
| S4 | — | 0 | feedback-only |
| S5 | — | 0 | feedback-only |
| S6 | 41–56 | 16 | segment 41 在 S3→S4 boundary 后为相同 target no-op |
| S7 | — | 0 | feedback-only；不伪造 PRE_RL profile |
| S8 | 57–101 | 45 | segment 57 在 S6→S7 boundary 后为相同 target no-op |
| S9 | 102–103 | 2 | final advance |
| S10 | 104–111 | 8 | recovery profile |
| **Total** | **0–111 exactly once** | **112** | 109 changed source targets + 3 boundary wheel-zero dispatches = 112 physical epochs |

每个 changed action 的 physical ledger 都包含完整 8-servo + 4-wheel target、atomic adapter ack、command epoch 和 N+1 PhysX readback。segments 7/41/57 仍保留完整 source identity，但 `target_changed=false`、不重复写入 target。

## Accepted v003 transition evidence

下面的 sim step 和 epoch 在 reviewed baseline 与三次 fresh repeat 中完全相同。`None` 的 body-stuck/active-leg-trapped producer仍保持 unknown；危险事件由完整 SHA-bound video review补足，而不是制造 `False`。

| # | Transition | Sim step | Command epoch | Guard evidence that passed |
|---:|---|---:|---:|---|
| 0 | reset → S0 | 189 | 0 | controller reset |
| 1 | S0 → S1 | 190 | 0 | finite articulation；writable target |
| 2 | S1 → S2 | 1808 | 8 | FR unload observed；profile complete；attitude safe |
| 3 | S2 → S3 | 2468 | 24 | FR airborne-before-crossing、crossed、TOP、required TOP ready |
| 4 | S3 → S4 | 6600 | 42 | FL airborne-before-crossing、crossed、TOP；FR/FL TOP evidence ready |
| 5 | S4 → S5 | 6601 | 42 | front pair TOP + fresh inherited body progress |
| 6 | S5 → S6 | 6602 | 42 | RR unload observed；profile timeline complete |
| 7 | S6 → S7 | 7692 | 58 | RR airborne-before-crossing、crossed、TOP；FR/FL/RR TOP evidence ready |
| 8 | S7 → S8 | 7693 | 58 | FL=true、RR=true candidate support geometry |
| 9 | S8 → S9 | 8115 | 102 | RL airborne-before-crossing、crossed、TOP |
| 10 | S9 → S10 | 9460 | 104 | body crossed；4-wheel crossing history；recovery workspace |
| 11 | S10 → SUCCESS | 9620 | 112 | body crossed；final recoverable；`posture_complete=false` |

实际四次运行的 transition retry count 全为 0；physical dispatch ledger 中没有 target-changing `HOLD` 或 replay dispatch。成功后 worker 仍执行原子 wheel-zero closure safe-stop，并在后续 physics step验证保留的 servo target与零 wheel target；这不是 controller 进入失败 `SAFE_STOP` state。

## 术语和证据边界

- `body_or_com_position()` 只有在 `com_position_m` 可用时才称 `MEASURED_COM`；本次所有 Gate-C runs 均为 `BASE_POSITION_PROXY`。
- `leg_support_candidate()` 优先使用正 load；load 为 `None` 时退化为 GROUND/TOP geometry class。本次 load 全部不可用，因此 S7 只能证明 FL/RR 候选几何存在。
- phase-local TOP 是 crossing 后的几何 class latch；它不要求四轮在同一 final tick 全 TOP。
- 最终 automated class 为 `FL=AIR; FR=TOP; RL=TOP; RR=TOP`，速度未 strict-stable，但 body 已越障、机器人可恢复，故正确语义是 `TASK_SUCCESS_POSTURE_INCOMPLETE`，不是 traversal failure。

运行级验证和完整 hashes 见 [V003_MACRO_FSM_VALIDATION.md](V003_MACRO_FSM_VALIDATION.md)。
