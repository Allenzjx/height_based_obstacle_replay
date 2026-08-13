# Playback speed / progress 修复报告

时间戳：2026-08-02 23:34:08（America/New_York）  
正式项目：`C:\robotics_sim\wlr_robot\height_based_obstacle_replay`  
正式数据：`saved_height_steps\height_05cm\accepted_steps.jsonl`（35 steps / 269 commands）

## 结论

本次已直接修改正式项目并完成自动测试、可见 Tk GUI、真实 Isaac Sim/IsaacLab worker 与真实 articulation 运动验证。最终结果：

- UI 使用唯一的 0%～300% `speed_percent`，默认 100%。
- Playback 进度由 worker 已实际派发的 command 更新，能显示原 sequence 的 step/command/global index，并高亮、滚动当前 step。
- Raw 保留并按百分比缩放录制时间关系；Motion-only 删除隐式人工空白，不删除显式 wait/hold/stop。
- Motion-only 不再在每个 step 后重复 0.050 s pad；35 个 step 的实测 inter-step gap 中位数和最大值均为 0。
- Servo target 不乘速度，50/100/200/300% 的最终 command target 均为 18.7 deg。
- Wheel 使用 `v' = v*s` 与 `T' = T/s`；触发上限后以实际速度反算 duration，计划积分不变。
- 50/100/200/300% 真实 articulation 轮角位移相对中位数最大偏差 0.8286%。
- 同一个 3 rad 基准片段在 100% 与请求 300%/有效 209.44% 限速后的真实轮角位移差 1.7246%。
- 0% 会暂停并停止活动轮目标，队列和 mapping 保留；恢复到 100% 后从正确位置继续。
- 完整 Motion-only 真实回放结束为 `COMPLETED`, step `35/35`, command `269/269`, `active=False`, `scheduled=False`, `operation=IDLE`。

## 真实根因

### Fast Play 的 step 间等待

旧 planner 为每个 step 计算 `last_event_time + trailing_pad`，再把 cursor 推进到下一 step。默认 `trailing_pad=0.050` 因而被重复应用 35 次；旧 Fast 5 cm/100% 的 34 次 step transition 全部约为 0.050 s。它并不是 Isaac physics frame 必需间隔，而是派生 plan 的固定时序。

新 Motion-only plan 把每个 step 的第一个有效 motion 接到上一 step 的有效末端，隐式 timestamp 空白归零；显式 wait/hold 仍作为事件保留并缩放。final pad 如使用只出现一次，不再每 step 重复。

### Play Selected Step 的“卡顿”

沿 UI → controller → plan → IPC → worker 检查后，未发现每条 command 重新读 JSON、每条 command restore 或逐 command 同步 IPC round trip；selected step 已是一次构建、一次 restore、一次完整 plan 提交。实际停顿来源是：

1. Raw selected plan 仍含该 step 的录制前导 timestamp/内部 gap；
2. 旧 Fast selected 仍附加每 step 0.050 s pad；
3. restore、0.30 s simulation start delay 和 worker ready 没有与 PLAYING 状态清楚分开，用户看到的是准备时间与动作时间混在一起；
4. GUI 过去只有 flattened index，无法判断是在 restore、等待首命令还是执行 command。

现在一次性构建/提交完整 selected plan，restore 只执行一次，worker 以 simulation time 连续派发；进度区分 PREPARING/RESTORING/PLAYING/PAUSED/COMPLETED，并保留原 sequence index。真实 step 2：plan build 0.701 ms，Motion-only 首命令到结束 0.257 s，最大 scheduler lateness 7 ms；包括 restore/worker startup 的按钮到完成墙钟时间为 2.150 s。

### Wheel 速度变化后距离错误

旧路径没有统一数学语义：Playback 的有效值来自多个 UI multiplier 的乘积；Fast planner 另行提高 wheel velocity/缩短 timestamp，而 Raw 和 manual 路径处理不同。continuous manual jog 提高 velocity 后，松开时间仍由用户决定，因此距离会自然变长；旧 limit 路径还会 clamp velocity 却保留理想的缩短 duration，造成另一方向的距离损失。核心问题是 velocity 和 segment duration 没有作为同一个距离不变量共同求解。

现在每个派生事件保存 base velocity、base duration、base distance、requested/effective percent 和 `speed_scale_application_count=1`。未限速时同时缩放 velocity 与 duration；限速时用 `base_angle / effective_velocity` 反算 duration。正式 recording 完全不写回。

### 旧代码中 speed scale 的应用位置/次数

旧 Playback 的用户值先由 `speed × global_motion_speed_scale × playback_speed_scale` 合成为一个值，planner 又用这个合成值分别做“timestamp 除法”和 Fast wheel velocity 乘法；这两处本应是同一个距离公式的互补两边，但旧架构允许三个上游倍率叠乘且各路径不一致。worker 对已构建的 plan 没有再乘一次。Manual wheel 使用 `global × wheel`，manual servo 使用 `global × servo`，后者错误地改变了 target position。

新实现只有一个 `SpeedPercentModel`。百分比只在 planner 生成派生 plan 时应用一次；worker 校验 metadata 后原样执行。Manual continuous wheel jog 从同一 model 取瞬时 velocity 比例；manual servo target 永远不乘比例。旧 1.0/3.0 分别迁移为 100%/300%，旧 `preserve-wheel-distance=False` 被忽略。

## 新速度定义

- `scale = speed_percent / 100`
- 0%：Hold/Pause，不开始新运动，无除零或无限 duration。
- 50%：0.5 倍 recording/真实机器人参考速度。
- 100%：正式 recording 的原始 timestamp、servo target 和 wheel command velocity，即当前真实机器人/现有配置的参考，不用 datasheet 最大速度替代。
- 200%/300%：请求 2/3 倍；实际 actuator/wheel 限制可降低 effective percent。

Servo endpoint 始终使用原 command target；速度只改变派生 trajectory timing。Manual servo 是现有 articulation target-control：slider target/range 不变，0% 阻止新目标，连续拖动采用“最新 target 覆盖旧 target”的现有 target 语义，不向 command queue 堆积缩放后的角度。

Manual wheel UI 是 continuous jog（按下/发送速度，停止时刻由用户决定），所以百分比改变瞬时 velocity，本身没有预定义总距离；一旦录成有起止 timestamp 的 step，回放会按 wheel angle invariant 重算 velocity/duration。

## PlaybackProgress 与调度

`PlaybackProgress` 包含 IDLE/PREPARING/RESTORING/PLAYING/PAUSED/STOPPING/COMPLETED/ERROR，以及 step、step id、step 内 command、global command、elapsed/remaining、profile、requested/effective 和 last_error。

planner 给每条事件附加只读 source step metadata。worker 在实际派发 command 时更新进度与 timing trace；UI 只以 10 Hz 读取状态。step 变化时只更新旧/新 row tag 并自动滚动，不在每个 physics frame 重建 table。Pause 停止活动轮目标并保持 index；Resume 重发该轮目标但不重复计数；Stop 清空 active/scheduled queue 并返回 IDLE。

Instrumentation 包含 button click、plan build、restore submit、worker ready、first command、各 command/step start/end、intentional duration、implicit gap、scheduler lateness 和 completion。轮命令还保存派发前关节位置快照，用于真实距离计算。

## 自动测试 A～I

- A Servo endpoint：PASS；30 deg target 不变，duration 为 2T/T/T÷2/T÷3。
- B Wheel distance：PASS；四档 `velocity × duration` 均为 3 rad（浮点精度）。
- C Wheel limit：PASS；300% 请求在 2 rad/s 测试上限下 effective=200%，duration=1.5 s，积分仍 3 rad。
- D Combined servo+wheel：PASS；target=30 deg、wheel=2 rad/s、共同 duration=1.5 s、积分=3 rad。
- E Fast/Raw gap：PASS；Fast 隐式 gap=0 且只一个 final pad；Raw gap 在 200% 为 100% 的一半；显式 wait 保留。
- F Selected step：PASS；restore 一次、一次完整 plan、原 index 15/35。
- G Progress mapping：PASS；Pause/Resume/Stop/Completed mapping 正确。
- H 0%：PASS；阻止启动、活动计划安全暂停、恢复后继续，无除零。
- I Single application：PASS；payload 前后 application count=1，worker 不重复缩放，manual servo target 不变。

全量结果：193 tests PASS；98 个正式 Python 文件编译 PASS；无仿真 GUI smoke PASS。

## 真实 5 cm GUI / Isaac 结果

- 主完整验证 Isaac PID：102360；没有复用既有进程，启动前确认不存在其他 Isaac/worker，且全程只启动一个。
- 限速补充对照 Isaac PID：84060；在主验证完全退出后单独启动，不并发。
- 5 cm obstacle 生成并加载 35 steps / 269 commands。
- Raw：启动、worker 实际进度、Pause、Resume、Stop 均 PASS。
- Selected Raw/Fast：原 step index 2/35、3 commands，restore 一次，均 PASS。
- 50/100/200/300% wheel：真实轮位移相对中位数最大偏差 0.8286%。
- Servo：四档最终 command target 均为 18.7 deg；articulation 保持同一 target，测试记录中的 actual 值是固定 0.75 s 墙钟采样时的瞬态跟踪位置，不是被缩放后的 target。
- 300% wheel limit：请求 300%，有效 209.4395%，`2.0944 × 1.432394 = 3 rad`；真实 100%/limited-300% 位移差 1.7246%。
- 0%：1.5 s hold 内 events 保持 1，轮平均漂移 0.00749 rad，恢复后完整执行 stop event。
- 切 tab：events 从 6 增到 11，计划保持 active。
- 完整 Motion-only：49.2625 s simulation plan，223.3622 s wall，269/269，35/35，inter-step median/max=0，max lateness=8.225 ms，最终 IDLE。

## 数据与环境保护

- 项目不是 git repository；`git status` 返回 exit 128，因此没有执行任何 reset/checkout。
- 开始前后对 65 个正式 recording、accepted/history 文件及 USD/URDF 做 SHA-256 比对：0 个差异。
- 未修改 Conda、Python、Isaac Sim、IsaacLab、physics 参数、joint mapping、calibration、obstacle geometry。
- 所有真实运行均使用 `--no-telemetry --no-save-scene`。
- production Python 中未出现 Vision Auto Replay / Stability Replay。
- 没有新增第二套 PlaybackController、SimProcessClient 或 scheduler。

## 证据与未验证项

最终 JSON：`gui_isaac_result.json` 与 `wheel_limit_real_result.json`。所有要求的 A～I、正式 5 cm GUI/Isaac、实际 wheel movement、limit、0%、tab switch 和 completion 均已验证；没有剩余未验证的验收项。

环境没有现成录屏工具，本次没有引入大型依赖，故未生成视频；等价关键阶段均有 PNG 截图和 worker timing JSON。
