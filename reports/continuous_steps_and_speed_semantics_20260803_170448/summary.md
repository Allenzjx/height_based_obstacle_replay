# Continuous steps and Speed Semantics v2

## 结论

正式 5 cm 数据以只读方式完成 Raw 与 Fast 全序列验证。新调度器一次提交完整 sequence，worker 按 simulation time 连续执行 segment；step boundary 仅为 progress metadata。Raw 保留 implicit UI idle，Fast 只删除它；两者在相同百分比下使用完全相同的 actuator motion 参数。Raw 与 Fast 是两次顺序执行、互不重叠的可见 GUI 验收；每次均由一个 worker 从头执行到尾。

旧实现以离散 event timestamp 为完成条件，没有按 servo 实际到达与 wheel 独立 active duration 建立 channel completion，step/event 边界还会重发 wheel 与依赖事件时隙。新实现预构建 159 个 completion-aware segments；相同 wheel 值跨 step 只更新 progress，不 stop/restart；servo waypoint 在误差 <=1° 后同一 scheduler cycle 进入下一段，没有固定 settle pad。

## 正式真实 Isaac 结果（5 cm）

- Raw PID 94444：248/248 events、159/159 segments、35 steps，stop_reason=complete，planned/actual end 291.274333/295.175000s；median/max inter-step gap 3300.000000/15533.333333ms（保留 implicit recorded UI idle）；最大实际 endpoint error 0.997217°。
- Fast PID 164384：248/248 events、159/159 segments、35 steps，stop_reason=complete，planned/actual end 129.739333/136.225000s；median/max inter-step gap 0.000000/0.000000ms；最大实际 endpoint error 0.976407°。
- 安全未触限区间：base 0.1 rad/s；100% requested/effective 0.1 rad/s、duration 2.000000s、path 1.539674rad；200% requested/effective 0.2 rad/s、duration 2.000000s、path 3.141249rad，ratio 2.040204。
- 用户指定 0.3→0.6 rad/s 结果：100%/200% active duration 均为 2.000000s；measured joint path 4.646057→3.392575rad（ratio 0.730205）。该点已进入真实 actuator/contact 限制区，实际路径降低，但调度器没有延长 duration 补距离；未把物理触限误报为线性未触限结果。
- Pause/Resume、移动窗口、切换 tab 不影响 worker。独立录制验收最终 Stop 后 active=False、scheduled=False、operation=IDLE、Play 可重新启用。

Raw runner 最后一次状态文件的顶层 `success=false` 是验收壳层在已写完完成截图并计划关闭窗口后又执行了一次 tick，Tk 根窗口已销毁而触发 `winfo` 异常；其 worker payload 已完整满足上述计数和终态。该 after-terminal tick 竞态已修复，报告生成器只接受严格的 `complete + 248 events + 159 segments + inactive + unscheduled` payload，不以顶层布尔值掩盖或替代 worker 证据。

## 修改前 / 修改后时序

修改前的 event-dispatch instrumentation 在旧报告中也给出 Fast median/max 0/0ms，但它仅量 timestamp dispatch，无法观测 servo 是否实际到位，因此是无效的完成指标。修改后 timing 以各通道实际完成为边界：Raw median/max 3299.999999/15533.333333ms，Fast 0/0ms；Fast 删除的只是 implicit UI idle，servo 必要 extension 仍保留并单列到 CSV。Raw planned 291.274333s，Fast planned 129.739333s，差值 161.535s 是可审计的 Raw-only idle/时序差异。

## 最终速度数学定义

`scale = speed_percent / 100`。Servo target 不变，requested angular velocity = 30°/s × scale，未触限时 duration = base duration / scale；实际未到 <=1° 时只记录并延长必要的 `servo_completion_extension`。Wheel requested velocity = canonical velocity × scale，effective velocity 按 limit clamp，active duration 永远等于 recorded base duration，displacement/path = effective velocity × base duration；不延长 duration 恢复距离。

Recording 与 Manual/Raw/Fast/Selected/Play-To/Respawn-And-Play 共享同一个 SpeedPercentModel。新录制存 canonical 100% wheel command、实际 wheel active duration、servo canonical duration 与 command-boundary speed snapshot。Planner 是唯一执行百分比换算的层，worker 只执行计划并验证 application_count=1，因此 200% recording→200% playback 不会变成 400%。真实录制/同速回放在 100%/200% 的 servo endpoint 差分别为 0.756583/0.727179°，wheel velocity/duration 差均为 0；200% playback requested wheel 是 0.6 而不是 1.2 rad/s。legacy 正式记录在内存中解释为 100% base command，原文件不改写。

## 验证边界

项目没有受验证的 wheel radius/transmission→地面线位移标定，因此 CSV 报告 articulation wheel joint rotation/path，不伪造米制地面距离。受保护数据 SHA-256 由最终独立审计文件给出；Vision/Stability 未恢复。未引入视频工具/依赖，因此保存逐阶段 GUI 截图而没有新视频。
