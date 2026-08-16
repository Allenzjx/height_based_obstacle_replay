# 50 mm Recording、Fast Replay、Macro FSM 与 PPO Residual 的边界

## 结论

Gate C 已经证明：v003-derived Macro FSM 不是把 24 个 Recording Step 改名后按时钟重播。它仍把 v003 的 112 个已封存 source actions 作为动作真值，但由 11 个 active macro states 赋予这些动作唯一的 phase owner，并且只有实时物理事件 guard 通过后才允许进入下一状态。一次 reviewed baseline 和随后恰好三次 fresh-worker repeat 都完成了 50 mm 越障，终态均为 `TASK_SUCCESS_POSTURE_INCOMPLETE`。

这份报告只陈述已完成的 Gate A/B/C。profile library 已包含四个 Gate-A 成功版本，但其他 profile 的 live FSM 运行属于 Gate D；PPO residual 的训练、zero-residual identity 和效果比较属于 Gate E，本文不把它们写成已完成结果。

## 控制层比较

| 控制层 | 输入 | 输出 | 是否固定时间轨迹 | 是否使用反馈 | 是否学习 | 当前证据状态 |
|---|---|---|---|---|---|---|
| Recording | 人工 UI 操作 | 保存的 servo/wheel commands、顺序和 duration | 是 | 否 | 否 | v003 原始示范：24 Steps、112 Fast segments、160 commands |
| Fast Replay | Recording segments | production compiler/scheduler 按时间执行的 commands | 是 | 主要否 | 否 | Gate A 已逐版本完成；v003 为 `REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE` |
| Recording-Derived FSM | live phase、wheel geometry class/clearance、base attitude/progress、可用时的 COM/load | phase-local full 8-servo + 4-wheel reference | 部分：时间只推进 state-local primitive | 是 | 否 | Gate C：v003 baseline + 3 repeats，4/4 task success，posture incomplete |
| PPO Residual | observation + FSM phase + nominal target | phase-masked bounded correction | 否 | 是 | 是 | 尚无 Gate-E 结果；不得解释为已训练或已优于 nominal FSM |

责任边界保持为：

```text
Recording 决定原始示范动作
FSM 决定当前应该执行哪个物理 phase、何时允许离开该 phase
PPO 未来只决定该 phase 内增加多少小幅 bounded correction
```

## 实际代码路径

1. [fsm50_motion_profiles.py](../fsm50_motion_profiles.py) 从 Gate-A production Fast plans 构建 phase-local profiles，保留 source version、segment、Step、event、command、原始相对时间、servo/wheel concurrency 和 source-plan SHA。稀疏 Recording command 被展开为完整 actuator target map，但数值、顺序、速度符号不被重新解释。
2. 同一文件的 `CANONICAL_SEGMENT_OWNERSHIP_RANGES` 使每个 source segment 只属于一个 macro state。v003 的 0..111 全覆盖且无重叠；S4、S5、S7 在 v003 中是 feedback-only acknowledgement states，未伪造空动作 profile。
3. [fsm50_macro_state_model.py](../fsm50_macro_state_model.py) 定义 11 个 active states、`SUCCESS` 和 `SAFE_STOP`。旧 57-state YAML 被标为 `LEGACY_SEMANTIC_GRAPH`，`legacy_graph_control_authority=false`。
4. [fsm50_macro_controller.py](../fsm50_macro_controller.py) 让时间只推进当前 profile cursor；transition 同时要求 profile timeline（若该 state 有 profile）和 live guard。guard 包括 unload、airborne-before-crossing、front-face crossing、phase/episode-latched TOP、body progress、candidate support geometry、recoverable attitude 和 final recoverability。
5. timeline 完成但 guard 未满足时，controller 保留 servo target、把 wheels 置零并进行 bounded hold/guard recheck。允许的 retry 不 rewind source cursor、不 replay action、不热切换其他 source/profile。
6. [worker_macro_fsm_session.py](../worker_macro_fsm_session.py) 在 120 Hz post-physics hook 上将每个实际 target change 原子地写入完整 8+4 target map，并在 N+1 physics step 做独立 PhysX target readback。source-action consumption 与 physical dispatch 使用不同 ledger，因此相同 target 的合法 source action仍被消费，但不会伪造新的 command epoch。

## v003：保留动作真值，但改变控制语义

| 维度 | v003 Fast Replay | v003 Macro FSM 的实际行为 |
|---|---|---|
| source action | 112 segments 按 production 时间轨迹执行 | 同一 112 actions 依 source identity 精确消费一次；无遗漏、重复或 cross-source switch |
| 组织单位 | 24 Recording Steps / 112 segments | 11 active physical states；FR → FL → RR → RL crossing order |
| 时间 | scheduler 时间决定下一 command | 时间只决定当前 state 中哪个 keyframe 到期；不能单独宣告 unload/cross/TOP/success |
| transition | command cursor 前进 | profile complete + live physical guard；source action与 transition boundary 不能占用同一 physics slot |
| 无动作 state | 无此抽象 | v003 S4、S5、S7 是 feedback-only states，使用已观察/边界携带证据，不借用后续 Recording command |
| guard 未满足 | open-loop 继续时间轨迹 | bounded hold、wheel-zero、guard recheck；耗尽则 fail-closed `SAFE_STOP` |
| 分支/恢复 | 固定轨迹 | state-local bounded retry policy 已实现；本次四个成功运行实际 retry count 均为 0 |
| 最终姿态 | scheduler 结束 | S10 独立判断 body crossed + final recoverable；`posture_complete=false` 仍可形成 task success posture-incomplete |

本次 v003 运行中三个 source actions（segments 7、41、57）在 state boundary 已经把 wheels 置零后与当前完整 target 相同。因此三行仍以精确 provenance 消费，但 `target_changed=false`、无 physical dispatch；对应 physical stream 是 109 个 `SOURCE_ACTION` dispatch 加 3 个 `BOUNDARY_ZERO_WHEELS`，command epochs 仍严格为 1..112。这是“source truth”和“物理写入”分离的实例，不是 action omission。

## Feedback 不是 force/load 伪装

Gate-C normal-development telemetry 明确记录：

- `com_measurement_available=false`，position guard 使用 `BASE_POSITION_PROXY`；
- `wheel_contact_load_available=false`，每腿 load 为 `None`；
- `filtered_contact_bank_enabled=false`；
- support guard 在没有 load 时使用 `GROUND`/`TOP` geometry class，只能称为 candidate support；
- runtime non-wheel collision/penetration producer 不可用时保留 `None`，最终由 SHA-bound full-video review 排除危险事件。

因此 FSM 使用了真实反馈，但当前反馈并不证明 mass-weighted COM、normal force、load balance、anchored contact 或 impulse mechanism。详细边界见 [50MM_COM_TRANSFER_FINDINGS.md](50MM_COM_TRANSFER_FINDINGS.md) 和 [50MM_SUPPORT_DIAGONAL_FINDINGS.md](50MM_SUPPORT_DIAGONAL_FINDINGS.md)。

## Gate-C 实证与后续边界

封存身份：graph `ffa5acfbf64b65c22eee54709a2afae5a56fa0b9345d8db84eb86acac58447c5`，profile library `3fb1501c40a8669681f5a073036a496ae221b7d91c3ceb629e93e37ac7c2ceea`，bundle `0579218825bcfdd4dabf7ab6225268ac04279a83ed96cad2ed390955414deebc`。

四个 reviewed runs 均为 112/112 source actions、160/160 commands、24/24 Steps、12 transitions、0 root-state writes、verified closure safe-stop，且视频逐帧 full decode 后确认 task complete、required lifts complete、body crossed、final recoverable，无 fall、wheel-only drive-up、dangerous body collision、severe penetration 或 joint-limit-like pose。完整运行表和 SHA 见 [V003_MACRO_FSM_VALIDATION.md](V003_MACRO_FSM_VALIDATION.md)。

尚未由本报告证明：

- 同一 graph 使用 v008/v009/v010 profile 的 live task success；
- cross-version parameter generalization；
- `FSM + zero residual == FSM nominal`；
- PPO residual 的稳定性或终态改善；
- measured COM/support margin/contact-load 改善。

