# Selected Pose、Run Management 与完整回放修复报告

## 结论

三个问题不是同一个 bug：Selected Playback 是 restore 合约/即时反馈/陈旧所有权的组合问题；Run 管理是 UI 语义和持久化事务缺口；完整回放是 completion policy 将容差内接触残差误判为 actuator_unstable。

## 最小修复

1. Selected Raw/Fast 统一要求 `FULL_VALID`，Step N 严格恢复 Step N-1 `sim_state_after`，经过 worker ACK、physics boundary 与实际 pose 校验后仅构建 Step N plan；command-only、placeholder、home/current pose 均不能继续。
2. Run 区改为 New Empty / Open Selected / Update Current / Save As New / Refresh；combobox 仅预览。Update 使用临时文件、fsync、重读校验、timestamp backup、atomic replace 和失败回滚。
3. 每次 Playback 前用新鲜 worker 状态清理 proven-stale 本地 ownership；selection/dirty Run 不再成为冲突。容差内 bounded contact residual 记录 `contact_residual_accepted` 并继续，3.0° hard cap、NaN、超差无响应及持续恶化仍失败。

## 正式数据与真实 Isaac

- 正式 Run：`v006_20260805_233948_654778_manual`，12 steps，78 events/commands，62 segments。
- checkpoint：12/12 steps 的 before/after 共 24 个状态全部 `FULL_VALID`。
- Selected Raw：root 0.000079746 m，orientation 0.005882°，servo max 0.022047°，实际最大关节变化 5.209682°。
- Selected Fast：root 0.000079746 m，orientation 0.005882°，servo max 0.022047°，实际最大关节变化 4.758651°。
- Formal Raw：12/12、78/78、62/62，Step 10/Segment 37=`contact_residual_accepted`，进入 Step 11/12，`stop_reason=complete`，operation=IDLE。
- Formal Fast：12/12、78/78、62/62，`stop_reason=complete`，operation=IDLE。
- 265 项自动测试通过；1021 个受保护文件 SHA-256 全部不变；正式 accepted_steps SHA 始终为 `2f7800916e26d20b4c098883f812220152c00e8480bf93a966ec544b073dba0f`。

## 唯一未完成的运行时动作

正式 Raw 完成后请求 Respawn 时，当前 Isaac ground reference 报告 grounded/stable/physics reference 无效，因此 Respawn 被安全门阻止。未绕过安全条件，也未把这一项伪报为成功；Raw/Fast 完整回放分别在独立、顺序、单 worker 的真实 Isaac 运行中均已完成。
