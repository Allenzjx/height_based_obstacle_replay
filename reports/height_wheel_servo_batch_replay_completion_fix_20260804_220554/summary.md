# Height / Wheel / Servo Batch / Replay Completion Fix

## Outcome

真实可视化 Isaac 5.1.0 最终验证通过：高度 50→75→100→50→100→75 mm 六个事务均用 USD 实际 visual/collision AABB 验证；障碍物实测宽度约 2.00000003 m；Manual Wheel 在 TEST、RECORDING、Stop Recording 后 TEST 三种状态均运动；Servo+Wheel batch 同一 applied sim step、启动偏差 0 s；正式 50 mm Raw/Fast 均完成 35/35、248/248 events、159/159 segments，stop_reason=complete。UI 最终 IDLE，worker 有序退出。

## 六个真实根因与修复

1. **Generate 假成功**：旧 worker 只写 prim 属性并按请求值回填 `scene_height/obstacle_updated`，没有 physics/render 后读取实际 AABB；本机 Isaac 对现有 prim 的原地 size 更新未刷新有效几何。现在每次事务携带 request/revision，先尝试原地更新并验证；验证失败时最小 Remove/Create，确认旧 prim 失效、新 prim/collision 有效，再用实际 AABB、visual/collision bounds 和 control_ready 决定成功。
2. **宽度不足**：正式唯一配置仍为 1.20 m。改为 2.00 m（满足 ≥2.0 且比 1.20 至少增加 0.40），三个高度同宽、Y 中心 0。机器人 collision width=0.4411003655 m，左右理论余量各 0.7794498172 m。
3. **Manual Wheel 看似依赖 Recording**：readiness/UI command path 分裂，且 Stop 后 worker/client 的 wheel generation 可能不同步；Recording 路径碰巧刷新状态，造成表象。现在 Servo/Wheel 共用 motion readiness，IDLE/RECORDING 都走同一 transport；Recording 只观察一次，Stop 后 generation 同步，旧 generation 仍拒绝。
4. **Servo-Wheel UI 消失/非原子**：后端残留 staging command 名称，但右侧 tab 的可见 section 和完整状态化工作流已丢失，旧通用应用路径也不能保证 12 个 actuator 同 tick。现已恢复 Start/Launch/Clear/Cancel 与 staged/live/delta；Launch 只发一个 `apply_motion_batch`，worker 先全量校验再一次 articulation write，同 tick ack；Recording 保存一个 launch event，Playback 复用同一 executor。
5. **Speed Scale 隐藏倍率**：倍率同时散落在 UI/model/IPC/manual/playback 元数据路径。运行时链路已移除，固定为 canonical 100%；Fast 只去 implicit idle。
6. **Already Active / 长序列停止**：本地曾在 worker 接受前过早 active，stale local/session 可导致 false already-active。新状态从 START_REQUESTED 开始，matching request/plan/SHA/count/session ack 后才 PREPARING，scheduler started 后才 PLAYING，并有 first-motion watchdog和状态对账。正式 50 mm 并未在第 24 步被 planner 或 IPC 截断：baseline worker 已收到全部 248/159；实际在 step 26 segment 128 因旧 0.800 s no-improvement + 约 1.315° contact residual 触发 actuator_limit。新的 bounded recorded-residual policy 保留安全失败，并把合法稳定残差记 warning 后继续。

## 被证据排除的怀疑

- 不是固定 24-step/50-event/queue 长度限制，也不是 JSON payload/worker decode 截断。
- 不是 UI 在 step 24 发送 Stop；baseline 已执行 step 25 并到 step 26。
- 不是 5 simulation seconds stall；捕获值是 0.800 s。
- 高度问题不是 75 被解释为 75 cm；正式 IPC 统一为整数 `height_mm`。
- 未更改既有 actuator 速度、stiffness、damping、effort limit、calibration、joint mapping 或 robot asset。

## Evidence

- Final raw result: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\reports\height_wheel_servo_batch_replay_completion_fix_20260804_220554\final_real_isaac_retry6\real_isaac_result.json`
- Completed screenshot: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\reports\height_wheel_servo_batch_replay_completion_fix_20260804_220554\final_real_isaac_retry6\screenshots\final_completed.png`
- Validation animation: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\reports\height_wheel_servo_batch_replay_completion_fix_20260804_220554\final_real_isaac_retry6\validation_motion.gif`
- Protected data audit: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\reports\height_wheel_servo_batch_replay_completion_fix_20260804_220554\protected_data_sha256_audit.txt`
