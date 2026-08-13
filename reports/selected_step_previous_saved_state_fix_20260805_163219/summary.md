# Play Selected Previous Saved State Fix

## 原错误行为与根因

`Play Selected Step` / `Play Selected Fast` 原来直接调用通用 `start_playback([selected])`。通用入口只会按所选 Step 自己的 `sim_state_before` / `command_state_before` 恢复；当恢复选项关闭或状态缺失时，还可能直接沿用当前 live state。同时 worker 侧没有可匹配的 restore request ID，restore 与 start plan 之间只依赖预设延时，不能证明对应恢复已完成。

## 修复后的 restore policy

- Step N（N > 1）：`Step N-1.sim_state_after` → `Step N-1.command_state_after` → `Step N.sim_state_before` → `Step N.command_state_before`。
- Step 1：`Step 1.sim_state_before` → `Step 1.command_state_before`。
- 全部缺失时明确报错并保持无 active/scheduled plan，绝不读取 current live state 作为起始状态。
- 同时存在前一步结束态与当前步开始态时只读比较；不一致只告警，前一步结束态仍是权威来源，不修改 saved data。

## 恢复与播放顺序

同一个既有 PLAYBACK operation transaction 内执行：取得 operation → Stop Wheels → 发送带 request ID 的 restore → 等待相同 request ID、restore_count 增加、result=ok、runtime ready → 请求 detailed state → 验证恢复态 → 使用既有 settle delay → 启动只含 selected Step 的 plan。Stop/E-stop/失败/超时会取消 pending transaction、释放 operation，且不发送 plan。

## 影响边界

普通 Selected 两个入口改用专用的轻量解析/等待路径；Play All、Play To Selected、Respawn Selected 等仍调用原有通用入口。planner、Raw/Fast timing、actuator command、completion policy、Obstacle、Height、Respawn、Recording、Servo-Wheel、Save/Combine 均未修改。OperationCoordinator 在 RESTORING 至完成期间持续保持 PLAYBACK，因此其他 task 使用原有 guard 被拒绝。

## 最终证据

- 单元/回归：聚焦 19/19；全套 226/226。
- 可见 Isaac Sim 5.1.0：正式 Tk `Play Selected Step` / `Play Selected Fast` 按钮级验证成功，PID 143148，未复用已有进程，单 worker，退出后无残留。
- Raw/Fast 选择 Step 5 均恢复 Step 4 `sim_state_after`，plan source 仅为 Step 5。
- Step 1 恢复自身 `sim_state_before`；恢复中 Stop 未延迟启动；Recording/Height Generate 冲突均被拒绝。
- 71 个受保护 saved-step/version 文件 SHA-256 变化为 0；正式验证序列哈希前后相等。
