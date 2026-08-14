# v003 正式 Fast Replay 首次完整调度审计

生成日期：2026-08-13  
结论：**诊断运行完整，但不是合格物理基准。Gate 1 仍为 FAIL（0/3 strict physical PASS）。**

## 运行身份

- Source version：`v003_20260805_224517_157723_manual`
- Source HEAD：`b3a6446cf743ad4d8d91dc6ea2fdb0bf2b5f752a`
- Contact role：formal
- Batch：`runs/v003_fast_replay_baseline/20260813T085004_108684Z_recording_replays_c0d1dfa247`
- Run：`v003_20260805_224517_157723_manual/20260813T085053_064259Z_clean_fast_replay_e7f8b3f37f/20260813_045116_obstacle-05cm_recording_fast_v003_20260805_224517_157723_manual`
- Result classification：`PARTIAL_SUCCESS`
- `artifact_valid=true`
- Scheduler/dispatch：PASS
- Full physical result：FAIL / evidence incomplete
- Process close：`FAST_EXIT_VERIFIED`

本次运行没有写回历史 root pose；`MOTION_START_READY` 的 root write count 为 0。

## 已证明的正式调度闭包

| 项目 | 结果 |
|---|---:|
| 原始 source command | 202 / 202 accounted |
| 正式保留 command | 160 / 160 dispatched |
| Semantic no-op | 42 / 42 accounted |
| 编译 segment | 112 / 112 completed |
| Motion batch | 115 |
| Source batch | 112 |
| Start boundary | 1 |
| Wheel-channel completion stop | 1 |
| Final safety stop | 1 |
| ACK invalid | 0 |
| 重复 applied physics step | 0 |
| Atomic servo-wheel batch | 6，全部 skew=0 |
| Readiness token | 单一 token，全部 source batch 精确绑定 |

旧的 segment 56 FR wheel clamp 错误已在本次运行中被实测越过：

- source：Step 14，`wheel fr 2.0944`
- applied scheduler step：8356
- first physics step：8357
- canonical payload / ACK：`2.0943951023931953 rad/s`
- PhysX float32 readback：`2.094395160675049 rad/s`
- ACK：valid

因此，本次失败不是 command 缺失、atomic skew、旧的 FR raw-vs-clamped ACK mismatch 或提前 scheduler abort。

## 启动证据

严格 `REST_QUALIFICATION` 仍未通过：clean reset 后 servo passive velocity 未达到原有 `0.02 rad/s` 连续静止条件。该失败未被伪装成 PASS。

独立 `MOTION_START_READY` 通过：

- 10 个连续采样点；
- 实际采样 physics step 为 196、204、…、268，stride=8，与 production render cadence 一致；
- 当前 ground/contact、joint/target/PhysX readback、wheel-zero boundary、runtime instance、plan identity 均通过；
- start boundary 在 step 260 apply，声明 first physics step 261；
- pre-first source dispatch evidence 在 step 268 重新捕获并绑定 token；
- root state write count=0。

这证明 `REST_QUALIFICATION` 与 `MOTION_START_READY` 是不同门槛，且本次正式调度没有用 historical root seed 绕过 grounding。

## 首个决定性控制偏差

共享 production playback 的 servo tracking lifecycle 在 segment 完成时冻结了尚未收敛的动态补偿。

### 时间线

1. 全程第一次 logical target 与 compensated drive target 分叉发生在 source cursor 1：`servo rear_left_hip 19.6`。batch first physics step=269，step 285 首次加入 `-1.25°` tracking correction。该时点只表示 feedback 已介入，不单独构成失败。
2. 最终关键 source 是 cursor 182、segment 104：`servo front_left_hip 0.5`，同一个 atomic segment 还包含其他 servo 与 `wheel FL 1.93`。batch applied step=10844，first physics step=10845。
3. FL logical actual target 在 step 10865 到达约 `0.579351564°`。
4. step 10869 tracking correction 再次介入；step 10957 correction 到达现有 `-10°` clamp，PhysX drive target 为 `-9.42064855°`。
5. scheduler step 10988 将该 segment 判为 `contact_residual_accepted`。此刻 FL position residual 约 `1.525°`，但 qd 仍为 `-13.843°/s`，recent error slope 约 `-25.277°/s`，不是收敛状态。
6. 旧 `end_servo_tracking()` 随即关闭 feedback，并把这个饱和、动态 correction 冻结为 load bias。
7. step 11062，FL force 从约 11.35 N 降到 4.00 N、再到 1.115 N，contact class 从 `TOP` 变为 `AIR`。
8. 最终 source command 仍为 `0.5°`，但 FL PhysX target 保持 `-9.42064855°`，q 为 `-9.43912848°`，最终只有 FR/RL/RR 在 TOP，FL 为 AIR。

原 v003 Step 24 保存的最终 FL command 为 `0.5°`、q 为 `0.485541126°`；当前 replay 最终 q 相差约 `-9.92467°`。Recording 中的 `target_actual_deg` 是 logical target，不是独立 PhysX readback，因此本报告没有把它误当作历史 drive target。

## 物理证据解释

旧 physical evaluator 还存在三类语义问题，本次旧 artifact 不能据此升级为物理结论：

- formal role 没有 obstacle-attributed non-wheel contact point/force bank，不能从 aggregate net force 证明“无非轮 obstacle collision”；
- 旧 `contact_drift` 把主动 wheel rolling 的世界接触点移动累计为 anchored drift，首次超过 0.03 m 在 step 664、`wheel all 0.3`，最终约 0.578059 m；
- 旧 traversal tracker 把远离 obstacle front face 的 pre-lift AIR→GROUND 当作永久 illegal，最早在 step 395 对 RL 产生假阳性。

这些问题必须分别由 role-aware verdict、anchored-only drift 和 raw traversal episode 修正；不能通过放宽现有物理阈值解决。

## Telemetry、视频与 shutdown

- FSM telemetry：10960 行，physics step 181..11140 连续；SHA-256 `084acb2a499150efe7f592e6cee472d124e1fec255ee7f479d8ef0793caaf640`
- Active GUI viewport：1370 帧，frame number 32..1401 连续
- MP4：12,635,437 bytes；SHA-256 `f82b5a8237b33451925e0c9f17de5517e44225851a25cc79adc327acb31f7487`
- Preclose：24 个关键文件、1419 条 checksum 均通过
- Shutdown：Isaac 5.1 `FAST_EXIT_VERIFIED`，child intended/actual return code=0/0
- Batch shutdown closure SHA-256：`9e34c69d5e81555c3e14b459cb7bfe2a27c04a8797e920dcb50af72099684461`

该 artifact 应永久保留为失败诊断和调度回归证据，但不得作为 Gate 1 的一次 PASS。

## Gate 状态

| Gate 1 子项 | 状态 |
|---|---|
| v003 source/plan provenance | PASS（静态） |
| 正式 command dispatch completeness | PASS（本次 live） |
| Atomic batch / one batch per tick | PASS（本次 live） |
| Motion start readiness | PASS（本次 live） |
| Full physical traversal | FAIL |
| Strict clean Fast Replay | 0 / 3 PASS |
| A2 / B 或其他 Recording | NOT STARTED |
| Exact-Reference FSM | PENDING；Gate 1 未通过，不得开始 |

下一次真实运行前必须先冻结并验证共享 tracking lifecycle、raw traversal episode、anchored-only drift、strict telemetry finalization、真实 PhysX separation 与低开销 active viewport capture。任何源码变化后都要重新生成 environment lock；A1/A2/B 之间不得修改冻结源码。

## 2026-08-13 instrumented 失败诊断（不计 Gate 1）

为验证上述新证据链，随后启动了一次独立 clean-reset instrumented v003 诊断：

- Batch：`runs/v003_fast_replay_baseline/20260813T123231_187080Z_recording_replays_39b5c23ad0`
- Run：`v003_20260805_224517_157723_manual/20260813T123326_258444Z_clean_fast_replay_3c2d2294f1/20260813_083336_obstacle-05cm_recording_fast_v003_20260805_224517_157723_manual`
- `MOTION_START_READY`：PASS，10-frame rich/shared/pre-first evidence 全部闭合，root write count=0
- Result：`ARTIFACT_INVALID`
- Scheduler：`runner_timeout`，未完成 source dispatch
- Physical：role/full 均 FAIL，不能计作 strict replay
- Shutdown closure：`PRECLOSE_CLOSURE_INVALID`

这次失败确定了三项互相独立的实现问题，而不是 v003 source command 本身变化：

1. Isaac 5.1 的 `RigidContactView.filter_paths` 实际为二维 `[sensor][filter]`。旧 decoder 当作一维 path list，导致 17/17 signed-separation pair 都以 layout error 进入 UNKNOWN。
2. source step 1 的 RL hip 在既有 0.75° convergence band 内形成 tracking relay：PhysX target 每 4 tick在约 `-19.211218°` 与 `-20.461218°` 间翻转，q 约为 `-19.69°..-19.85°`、速度约 `+6.16°/s..-7.14°/s`。旧 scheduler 没有可达 liveness failure，最终只由外层 runner timeout 截止。
3. direct viewport recorder 在 drawable event 内才安排“下一帧”GPU readback，因此同一既有 render 后 callback 仍为 0；结果为 0-frame ledger、0-byte MP4、缺 first/last PNG，preclose 正确拒绝认证。

对应修复均落在共享生产路径：signed-separation decoder 严格验证二维 live identity；servo tracking 在既有 band 内 sample-and-hold，并以既有 1.5 s + 0.75 s 窗口导出 2.25 s fail-closed liveness；viewport 在唯一既有 render 前直接请求 active LdrColor RpResource、render 后等待 readback，不增加 physics tick、render 或 `app.update`。失败视频现在仍保持 artifact FAIL，但已与 preclose/shutdown 诊断闭包解耦，不再因不存在的成功视频 PNG 阻止保存失败证据。

本诊断不改变 Gate 1 计数：**仍为 0/3 strict instrumented Fast Replay PASS**。

## 2026-08-13 instrumented 第二次失败诊断（不计 Gate 1）

在二维 PhysX layout、tracking HOLD 与 direct viewport 初修后，又运行了一次独立 clean-reset trial：

- Batch：`runs/v003_fast_replay_baseline/20260813T152437_428969Z_recording_replays_b0075734f0`
- Run：`v003_20260805_224517_157723_manual/20260813T152529_897747Z_clean_fast_replay_5edfa6cec9/20260813_112539_obstacle-05cm_recording_fast_v003_20260805_224517_157723_manual`
- `MOTION_START_READY`：PASS，root write count=0
- Scheduler：step 1 / segment 0 `actuator_unstable`
- Result：`ARTIFACT_INVALID`，Gate 1 不计数

这次运行实测证明两个前置修复已有效：

- 17/17 signed-separation pair 在首个 checkpoint 均为 AVAILABLE，unknown=0；旧二维 layout error 未复现。
- active viewport direct buffer 共 56 帧，sim step 188..628、stride=8、1280×720 RGBA8；ledger、首末 PNG、MP4 全部有效。22/22 run required paths 存在，telemetry canonical 完成。
- Preclose 验证 30 个关键文件、55 条 checksum；Isaac 5.1 shutdown 为 `FAST_EXIT_VERIFIED`，child return code 0。

首段控制失败也更精确：RL hip nominal error 最终仅约 `0.07453°`，drive target 已固定为 `-19.779427°`，旧的 ±1.25° feedback target 翻转不再存在；但 PhysX joint velocity 在 render 相位稳定约 `+1.38°/s`，未达到既有 `0.5°/s` completion 门槛，因而在既有 1.5 s + 0.75 s 导出的 2.25 s liveness bound 后 fail closed。position 本身只在约 `-19.7056°/-19.7049°` 两点间轻微交替，说明剩余问题是带内速度余振，而非 endpoint 偏离。

同时，Kit log 明确报告 GPU contact filter 不支持父 Xform `/World/defaultGroundPlane`。对应 artifact 虽有 common wheel net force，3584 条 filtered wheel/surface rows 却全为零且 contact point 无效；因此 instrumented anchored drift 与 ground contact evidence 不可评估。修复将 ground filter 改到真实 collision prim `/World/defaultGroundPlane/GroundPlane/CollisionPlane`，并增加 common loaded contact 与 exact filtered pair 的 fail-closed 一致性校验，不再允许全零 filtered tensor vacuous PASS。

本次仍不是一次 strict replay：**Gate 1 保持 0/3**。下一次 trial 必须同时实测新的带内速度制动与真实 ground collision filter，任何一个失败都不得启动 trial 2。

## 2026-08-13 instrumented 第三次失败诊断（不计 Gate 1）

在真实 ground collision filter 和带内 DAMP/capture/HOLD 接线后，又运行了一次独立 clean-reset trial：

- Batch：`runs/v003_fast_replay_baseline/20260813T155201_231272Z_recording_replays_ff09748f95`
- Run：`v003_20260805_224517_157723_manual/20260813T155250_626849Z_clean_fast_replay_064f904440/20260813_115300_obstacle-05cm_recording_fast_v003_20260805_224517_157723_manual`
- `MOTION_START_READY`：PASS，root write count=0
- Result：`ARTIFACT_INVALID`；资格门仍为 FAIL
- Causal scheduler result：`SCHEDULER_FAILURE / actuator_unstable`
- Dispatch：2/160 source events、2/112 segments，因此 wheel integral 正确为 `NOT_EVALUABLE`

本次真实运行证明 contact/penetration/drift 修复已生效：

- 504/504 telemetry tick 的 `filtered_contact_consistency_valid=true`；
- ground filtered identity 为实际碰撞 prim `/World/defaultGroundPlane/GroundPlane/CollisionPlane`；
- common wheel net force 与 filtered ground force 对四条腿逐 tick 完全一致；
- 4032 条 wheel signed-separation pair 全部 valid，unknown=0；
- 最大真实 PhysX penetration 为 `0.000028971 m < 0.003 m`；
- anchored drift evidence valid，最大 `0.0137274 m < 0.03 m`；主动滚动位移未再污染 anchored drift；
- dangerous non-wheel obstacle collision count=0，相关证据有效。

唯一先行阻断发生在 physics tick 612（首个 telemetry failure marker 为 tick 613）：

- source Step 1、segment 1、`servo rear_left_hip 25.9`；
- logical actual target `-26.0794263347525 deg`；
- PhysX drive target `-26.106157800326 deg`；
- measured q `-26.0659262785673 deg`；
- position error仅 `0.0135000562 deg`；
- qd `0.782195825 deg/s`，仍高于既有最终静止条件 `0.5 deg/s`；
- extension `2.291333 s`，超过由既有 `1.5 + 0.75 = 2.25 s` 窗口导出的 liveness bound。

该失败揭示的并非需要放宽稳定阈值，而是 reference executor 把最终静止条件错误施加到了每个连续中间 setpoint。旧的正式完整调度 artifact 对同一 Step 1 的前三个 waypoint 在转换时仍分别具有约 `-52.25`、`-38.77`、`-21.08 deg/s` 的关节速度，随后继续下发下一 setpoint；原始 v003 Fast Plan 的三个 segment duration 也仅为 `0.130667 s`、`0.042000 s`、`0.014667 s`。因此中间 waypoint 的权威完成语义是：计划 duration 已到、所有目标关节均有 finite position evidence、位置在既有 tolerance 内；`0.5 deg/s` 仍保留给最终 physical dwell，不得变成每个 source segment 的隐式静止门。

本次 direct viewport video 为 63/63 帧完整 decode；22/22 required evidence 存在；run/batch checksum、telemetry canonical finalization、preclose 均通过；shutdown 为 `FAST_EXIT_VERIFIED`。这证明 `ARTIFACT_INVALID` 是资格结果，不是 diagnostic telemetry、视频或关停文件损坏。

此外发现一个独立的纯数值诊断误判：最终 post-settle tick 654..684 的真实 span 为 `0.25000000000001243 s`，线/角速度均低于既有门槛，但 cutoff 的约 `1.24e-14 s` 浮点误差排除了首 tick。修复只使用 IEEE-754/physics-dt 导出的表示误差 guard，并继续要求实测 span `>=0.25 s`；没有改变 dwell 或速度阈值。

本次仍不计 Gate 1：**strict instrumented Fast Replay 保持 0/3 PASS**。下一次运行前只修共享 `SimTimePlaybackService` 的 reference-position completion/追踪冻结职责边界；不得在 runner 中特判，也不得修改 USD、root pose、物理参数或任何既有稳定阈值。

## 2026-08-13 instrumented 完整调度物理失败诊断（不计 Gate 1）

共享 reference-position completion 修复并重新冻结 94/94 source closure 后，完成了一次完整的 clean-reset v003 instrumented replay：

- Batch：`runs/v003_fast_replay_baseline/20260813T163211_787260Z_recording_replays_a47fc82456`
- Run：`v003_20260805_224517_157723_manual/20260813T163300_794148Z_clean_fast_replay_9b96a13753/20260813_123310_obstacle-05cm_recording_fast_v003_20260805_224517_157723_manual`
- `MOTION_START_READY`：PASS，root state write count=0
- Result：`PARTIAL_SUCCESS`，`artifact_valid=true`，但 role/full physical verdict 均为 FAIL
- Gate 1：仍为 `0/3` strict PASS

正式调度和 artifact closure 已完整通过：

- 24/24 source Steps、160/160 retained commands、112/112 segments；
- 115 motion batches = 1 start boundary + 112 source segment starts + 1 wheel-channel completion stop + 1 final safety stop；
- 115 个 scheduler physics tick 全部唯一并严格递增，ACK valid、atomic skew=0；
- 96 个 servo segment 均明确记录 `reference_position_complete_tracking_deferred`，没有把 position reference completion 冒称 tracking converged；
- 四轮 authoritative target integral 全部 PASS，误差约 `6.79e-7..1.01e-6 rad`，均低于各自一个 physics tick 的明确上界；
- canonical telemetry 11,000 行、9 个 checkpoint，journal 正常移除；
- active viewport MP4 1,375/1,375 帧完整 decode，1280×720、15 fps，SHA-256 `2fcfc32bb9ea33c77027a9ba73352b343b12d895c5c2508a0d8e27f632578cc3`；
- run checksum 40/40、batch-live 60/60、preclose 55/55 全部重算一致；
- preclose 30 个 evidence files 验证通过，shutdown=`FAST_EXIT_VERIFIED`、child return code=0。

因此，本次正式证明了共享 playback 恢复了 v003 的完整 command execution；但它没有证明物理成功。严格 physical FAIL 包含：

- `all_legs_linkage_lift_valid=false`；
- `no_illegal_drive_up=false`；
- `contact_drift_safe=false`，最大 anchored drift `0.0488206383 m > 0.03 m`；
- `final_all_top=false`、`final_all_loaded=false`，终态为 FL=AIR、FR/RL/RR=TOP；
- `final_velocity_stable=false`；
- `contact_evidence_valid=MISSING`。

首个 loader 拒绝点是 telemetry sample 7985（physics step 8166、t=68.05 s）：RL 同时落入 ground 与 front-face 几何容差区，旧 classifier 固定优先 `FRONT_FACE`；但 exact filtered evidence 显示载荷全部来自 ground pair。392/392 受影响帧的 ground pair 均 active，obstacle pair 均未达到既有 2 N 门槛，common load 为 `2.2548..19.5681 N`。这不是 contact sensor identity/layout 错误，而是 surface classification 未使用已经采集的 exact filtered surface load。正确修复边界是在共享 classifier 中用 filtered ground/obstacle evidence消解几何候选重叠；不得放宽 consistency gate、不得改阈值。

关键 artifact SHA-256：

- result：`be67d0e4df2b7bf11a117f98b6dedad32c685876354bb9d14ce47c73f241fc9f`
- physical evidence：`fc5352ca9a0671c59b4b36b0d5b5686fd8d8279a535d5c61c3a2764f46cc81f7`
- dispatch：`d2d61ce6370c064ac74b2e894766704cc6fdc8c8e4aaac048839e6d40f81ef01`
- wheel integral：`c7615139a07d49fe4d65c1f7c9879dba6e0136eaf99e041ce40d5108b705f53d`
- telemetry JSONL：`18c2a9859f283287cc70b20791219a3c1cfac1c3b54ff141f12fe5e6ba3f9716`

该 run 保留为完整 dispatch/视频/closure 回归证据，但官方 B loader 正确 fail-closed，不能计作 baseline PASS。必须先修 surface attribution，并独立解释 anchored drift 与最终 FL=AIR/未稳定；不得直接启动 trial 2 或进入 Gate 2。

## 2026-08-13 正式 worker Trial 1/2 确定性物理失败（当前结论）

上述“不得启动 trial 2”是当时针对旧 direct-runner artifact 的冻结结论，现已由正式 worker 证据取代。完成 controller→IPC→`sim_worker_process.py` 的唯一执行者接线、source-lock 重封与 fast-close receipt 后，按完全相同冻结源码串行运行了两个 clean v003 instrumented trial：

- Trial 1 batch：`runs/v003_fast_replay_baseline/20260813T203256_266975Z_recording_replays_8a05c85769`
- Trial 2 batch：`runs/v003_fast_replay_baseline/20260813T222848_123995Z_recording_replays_9bdfc9235a`
- 正式 Fast plan：SHA-256 `a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c`
- retained commands / segments：160 / 112
- 每次实际 motion batch：117 = start boundary 1 + source 112 + wheel completion stop 3 + final safety stop 1
- 每个 source/derived batch 均有唯一 scheduler tick、有效 ACK，且 first physics tick = applied tick + 1
- root state write count：0

两次运行的 dispatch 投影完全相等；归一 SHA-256 均为 `6d87ebbb33ace0e89939f37461044eaa31499748433d9566ef8932fc5dbf572e`。两份 11,008 行 telemetry 除运行 ID、采样开销和 readiness token 外逐字段相等；归一 SHA-256 均为 `69b5ef2fea5aae277999964634592d5eef351d2ae7acf6944fdb35dbac6afb60`。这排除了 trial 间 IPC、batch 合并、同 tick 重复下发或随机调度差异。

Trial 2 的 worker-owned artifact 完整闭合：

- result：`PARTIAL_SUCCESS`，`artifact_valid=true`，scheduler/dispatch complete，但 physical/strict=false；
- canonical telemetry：11,008 rows、9 checkpoints、journal removed；
- active viewport video：1,376/1,376 frames full decode，1280×720 @ 15 fps，SHA-256 `74877107220551f24aecb0a5090827df1e5bd02719bca637a6e325befd20cab3`；
- four-wheel authoritative target integral：全部 PASS；
- shutdown：`FAST_EXIT_VERIFIED`、`FAST_WORKER_PROCESS_RETURNED`，child/worker rc=0、normal/no-force；
- batch shutdown closure SHA-256：`7c7e45815097e0a93fa23f868fc7527eb31316bafd1246c8bd033437af8c692a`。

严格物理 FAIL 恰为四项：`all_legs_linkage_lift_valid`、`contact_drift_safe`、`final_all_top`、`final_all_loaded`。最终 FL=AIR，FR/RL/RR=TOP；final velocity dwell 实测 PASS。最早确定阈值失败在两次 trial 中逐值相同：

- physics step 8694，t=`72.45000000000373 s`；
- source Step 16、segment 62，命令为 `wheel FL -1.96 rad/s`，RL wheel target=0；
- RL ground pair load/contact point/identity 均有效；
- RL measured contact-point displacement 从上一 tick 的 `0.0269535539 m` 升到 `0.0317374870 m`，超过既有 `0.03 m`；
- 两 tick 后最大为 `0.0344576476 m`；
- 同一 epoch 中 wheel center 约移动 35 mm，而 RL wheel rotation 只对应约 1 mm 轮缘弧长，支持真实被动拖滑，不是 surface 错分或单独 contact centroid 跳变。

### 证据语义更正（不改变 FAIL 或阈值）

旧 producer 把“logical 与独立 PhysX wheel target 均为零、接触承载且 measured contact point 有效”的区间命名为 `ANCHORED`。这不符合证据合同：零 target 只是 command evidence，不能证明 wheel 被物理锚定，也不能证明 ContactSensor 每 tick 返回同一材料点。

新 producer 因此只生成 `ZERO_TARGET_LOADED_CONTACT_EPOCH`，canonical quantity 为 `zero_target_contact_point_displacement_m`，并明确记录：

- `physical_anchoring_proven=false`；
- `material_point_identity_available=false`；
- semantics=`ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT`。

外部 criterion 名 `contact_drift_safe` 与既有 30 mm limit 保留；Trial 2 的 34.4576 mm 仍是 AVAILABLE/FAIL。旧 `ANCHORED` row 不得被新 producer 接受为合格 evidence。该改动只修正证明语义，未修改 USD、root pose、physics/solver、stiffness/damping、控制增益、采样率或任何阈值。

### 下一步：只做冻结的观察层隔离

当前没有足够证据支持 production 控制或物理 patch。正式 UI 与 formal worker 对 v003 编译出完全相同 plan，历史 Fast lifecycle 也一直在 segment boundary 结束 tracking；manual recording 的跨 segment tracking 不能冒充 Fast 语义。

因此下一阶段按同一 source-lock 串行采集，不计 Gate 1：

1. U：production-default 无 ContactSensor；
2. A1、A2：formal aggregate ContactSensor 两次自重复；
3. B：instrumented exact filtered wheel/non-wheel bank。

四者共用同一正式 worker、plan、dt=1/120、render cadence、adapter/service、viewport video、dispatch ledger、root write count=0 与 fast shutdown closure。U 恒为 `TRAJECTORY_DIAGNOSTIC_ONLY / NO_PHYSICAL_CLAIM`；A/B comparator 分开报告六项 sensor-independent trajectory metric 与两项 observation-sensitive contact metric，绝不把诊断结果计入 Gate 1。只有 sealed U/A1/A2/B 证明了某个正式语义差异，才允许做最小共享 restoration；否则保持 production semantics 不变。

Gate 1 当前仍为：**0/3 strict instrumented physical PASS**。Gate 2 及其他 Recording 继续禁止启动。

## 2026-08-14 观察层隔离终验：U 与 sealed A1/A2/B（不计 Gate 1）

### 资格与 provenance 边界

本节只回答 ContactSensor 构型是否改变正式 Fast 轨迹，不作物理成功声明。U 的 durable scope 是 `PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC`，classification=`TRAJECTORY_DIAGNOSTIC_ONLY`，physical verdict=`NOT_EVALUABLE`；A1/A2/B 的 scope 是 `TRAJECTORY_COMPARISON`。sealed comparator 明确输出 `diagnostic_scope=TRAJECTORY_DIAGNOSTIC_ONLY`、`physical_claim=NO_PHYSICAL_CLAIM`、`qualification_eligible=false`。

U 与当前 A1/A2/B 不能声称共用同一 source lock：U 以及第一份旧 A1 使用 metadata 修复前的 worker。旧 A1 batch `20260814T024907_425765Z_recording_replays_b76fc247ce` 因 `runtime_environment.json` 缺顶层 `viewport_video_manifest_path` / `viewport_video_manifest_sha256` 被 strict loader 正确拒绝，永久标记为 **metadata NO-GO**，不得进入 comparator。修复只在 telemetry 已 finish/unbind 且 video 已 finalize 后补写上述两个终态 cross-link；未改 physics、控制、trajectory、root state 或阈值。

修复后 A1/A2/B 共享同一冻结闭包：

- source closure：96/96，environment lock SHA-256 `486719a0053cf29caba11054f5318ec6975c3867cc86312950bae07d5be525e1`；
- worker SHA-256：`a1f38e778889964c0817fc5c0d107f2f675f56add0df3e4717b66e60209e73dc`；
- accepted source SHA-256：`06e13153b7ba75a4283e117d875f1da4895748835a9032c6faadef2bda25b394`；
- plan SHA-256：`a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c`；160 retained events、112 source segments、117 actual batches；
- sample grid：11,008 rows，sim step 181..11188，dt=1/120 s；
- root state write count：0；所有三次 shutdown 均 `FAST_EXIT_VERIFIED`、child/worker rc=0、normal/no-force。

### Durable artifact 表

| role | batch | contact mode | strict admission | metrics SHA-256 | video SHA-256 | shutdown closure SHA-256 |
|---|---|---|---|---|---|---|
| A1 | `20260814T034952_001217Z_recording_replays_b5ce1bb1d5` | formal aggregate | PASS | `d9d6c26c13957ced0e41d7c75fe1c808e84f2a762eb403de1850984aad0ad5c5` | `db2cbf80f03abdc697f9c9bf87d27868048bdf01e090a98694c7b69adc0dca98` | `efd985e3b8df21e5fb0a2172eba6a138b8c7d5265bfe65a2089732bad66706b6` |
| A2 | `20260814T044650_481215Z_recording_replays_dab648eb0e` | formal aggregate | PASS | `d9d6c26c13957ced0e41d7c75fe1c808e84f2a762eb403de1850984aad0ad5c5` | `6e77544103f2d91b7c7ac8373c32ac6c852952cc6233b2eb7607959182023088` | `c1a02a7e72fb7d44e261396ba5ed202fa5c10c5a422e2b7d9a34c7394afd3c0a` |
| B | `20260814T054214_073036Z_recording_replays_eac25c0ff4` | instrumented filtered bank | PASS | `f9d3d7c4021a85058f5d8ac9d20127c360ffda07dce86f76ac63bbf7daa5e5c9` | `684432be28e19347b2d128d227cc067f2e37fed4b3329916f8ebc5f4d6a015ee` | `d90fdf5265a65433fb5a1850748a4ed24d819cd4f23cc178dac1530560bc3b6c` |

三份 artifact 的 loader admission 与 triplet provenance 均 PASS；每份 dispatch 都完整覆盖 160/160 retained events、112/112 primary segments、117 batches，且 one-batch-per-physics-tick=true。三份视频均由 sealed manifest/ledger 证明并逐帧完整解码 1,376/1,376，1280×720 @ 15 fps。

### Strict comparator 结果

正式报告写入 `reports/ENVIRONMENT_EQUIVALENCE_REPORT.json`：9,429,864 bytes，SHA-256 `cfe6eb72d7eee7e92a5aa30e31c152c7d2f247353ec57b819d9255cfd8e678d5`。报告必须保持 `status=FAIL`、`environment_equivalent=false`，因为 8 项指标中的 `contact_class` 不完全相等；不能把六项轨迹 PASS 改写成整份 comparator PASS。

| metric | A1/A2 self error | A/B error | tolerance | verdict |
|---|---:|---:|---:|---|
| root trajectory | 0 | 0 | 1e-6 | PASS |
| joint trajectory | 0 | 0 | 1e-6 | PASS |
| wheel rotation | 0 | 0 | 1e-6 | PASS |
| wheel travel | 0 | 0 | 1e-7 | PASS |
| final pose | 0 | 0 | 1e-6 | PASS |
| obstacle geometry | 0 | 0 | 1e-7 | PASS |
| contact force | 0 | `3.552713678800501e-15` | 1e-6 | PASS |
| contact class | 0 | `17/44032 = 0.0003860828488372093` | 0 | **FAIL** |

六项 sensor-independent trajectory 的 A1/A2、A1/B、A2/B error 全为 0。归一唯一的 run-specific `atomic_batch_id` 后，三份 canonical projected trajectory 逐字节相同：54,763,099 bytes，SHA-256 `d04c341b265540c6d431d5901623b8905c0badc2704e3ff67e24491f89581d7b`。最终 root pose 也逐值相同：

`[0.8431757688522339, -0.03989097476005554, 0.15093284845352173, 0.9975134134292603, 0.0661960244178772, -0.01932697743177414, 0.014547132886946201]`。

17 个 class mismatch 全部是 RL：A1/A2 的 formal aggregate fallback 为 `FRONT_FACE`，B 的 exact filtered evidence 为 `GROUND`。对应 sim steps 为 `8691..8696, 8713, 8715, 8721, 8724, 8727..8731, 8733, 8806`；前 16 个在 source Step 16，最后一个在 Step 17。以 sim 8694 为例，B 的 RL ground normal force=`3.0228245258331574 N`、obstacle force=`0 N`、ground contact point valid；A/B 的 common wheel net force 完全相同。差异来自 formal aggregate 的几何 overlap fallback 与 instrumented exact surface identity，不是动力学分叉，也不是物理成功证明。strict categorical comparator 因 tolerance=0 正确 fail-closed。

### Step 16 根因边界与视频说明

三份同锁运行以及独立 U 均在 sim 8662/8694/8696 逐 bit 复现同一 sensor-independent trace（联合 SHA-256 `5dd894481f1a1b34991c8fcbd0a4431d8924010aaa51a203e09cd4f5f46f6178`）。Step 16 segment 62 中 RL logical/PhysX target 均为 0；8662→8694 wheel-center 水平位移 32.339956874 mm，而轮缘行程仅 0.980448723 mm，留下 31.359508151 mm 非滚动平移；到 8696 留下 33.946138117 mm。零 target 只是命令事实，不是物理锚固证明；canonical evidence 仍为 `ZERO_TARGET_LOADED_CONTACT_EPOCH` / `ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT`，且 `physical_anchoring_proven=false`、`material_point_identity_available=false`。

B 的 instrumented physical observation 仍为 `PARTIAL_SUCCESS`、physical/strict=false：`contact_drift_safe`、`all_legs_linkage_lift_valid`、`final_all_top`、`final_all_loaded` FAIL；最大 zero-target contact-point displacement=`0.03445764763205546 m`，最终 FL=AIR，FR/RL/RR=TOP。该 observation 不参与 diagnostic admission，但与此前正式 instrumented Trial 1/2 的确定性失败一致。

MP4 的 sealed manifest、frame ledger 与逐帧 decode 均有效；但 container header 的 duration/frame-count 元数据不可信（B 的 OpenCV frame count 为 `4294967295`，FFmpeg duration 为溢出值）。因此时长只采用 sealed ledger/full decode 的 1,376 frames / 91.733333 s，不能采用 container header。

### 最终 Gate 决策

- sealed A1/A2/B artifact 与 provenance：**GO**；
- 六项 sensor-independent dynamics/geometry trajectory：**GO，exact**；
- 全 8 项 environment comparator：**NO-GO**，唯一失败为 observation-sensitive `contact_class`；
- physical qualification：**NO-GO / NOT ELIGIBLE**。

观察层隔离已排除 production default、formal aggregate 与 instrumented filtered ContactSensor 构型对该轨迹的动力学扰动。当前可证根因边界是：不可变 v003 source 在正式 Fast 语义和锁定物理环境下确定性地产生 Step 16 被动 RL 非滚动/滑移，并在终态留下 FL=AIR；没有证据支持修改 production runner、physics、root pose、阈值或 tracking lifecycle 来强迫 PASS。

因此 Gate 1 仍为 **0/3 strict instrumented physical PASS**。停止无修复复跑；Gate 2、Exact-Reference FSM 与其他 Recording 继续禁止。若 v003 source 必须不可变，则当前为诚实的物理资格阻塞；只有获得授权创建并用正式 UI 验证新的 Recording 命令序列、重新冻结闭包后，才能从新的 Gate 1 Trial 1 重新开始。
