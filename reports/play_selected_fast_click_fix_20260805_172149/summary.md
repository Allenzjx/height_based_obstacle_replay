# Play Selected Fast 真实点击修复验收

## 根因与第一个失败边界

正式最新代码在修改前的真实物理鼠标复现中，完整“完全不执行”未稳定复现：`tk_button_command_entered` 在 18.399 ms 已进入，worker 后续也完成。第一个确定不满足需求的边界是 `immediate_visible_feedback`：代码没有在 Fast callback 入口同步更新可见状态，回调结束约 137.342 ms 时用户只看到泛化的 `Stopping wheels...`，没有看到 Fast 已接收及 restore 来源，所以表现为“点击无反应”。此外 compact worker status 丢失 profile/selected 字段，UI 可把 Fast 显示为 Raw。

基线还确认 Selected restore 先 `begin(PLAYBACK)`，约 421.802 ms 后 PlaybackManager 再次 `enter_playback()`。当前 coordinator 对已有 PLAYBACK 返回 True，所以这不是本机基线 worker 未启动的直接失败点；但它没有 owner 校验，是明确的 transaction ownership 缺口。

修复时曾验证一个关键 GUI 陷阱：若在 ButtonPress 阶段把 5 行状态改成 2 行，布局会在 release 前移动按钮，ttk command 会真的不触发。最终实现的 press observer 只生成 click ID、不改变 enabled 按钮布局；可见反馈放在 command 的第一段并立即 flush。

## 最小修复

1. Fast 物理 press 生成唯一 `selected_fast_click_id`；callback 入口在 100 ms 内显示接收/恢复文字，并保存到 pending transaction。
2. Raw/Fast 共用 `resolve_playback_selected_index()`：Treeview 当前选择优先，controller 最后有效选择回退，并验证 1..count。
3. Selected restore 启动 worker 时显式传入 `operation_already_owned=True` 和 restore request owner；PlaybackManager 验证 state/owner 后跳过第二次 `_enter_operation()`。普通播放入口仍走原逻辑。
4. worker acknowledgment/compact progress 保留 `motion_only` profile 和 selected 标记。

没有修改任何速度、target、duration、Fast compaction、几何、执行容差或保存数据。

## 验收结果

- 自动测试：237/237 PASS；专项 11/11 PASS。
- 真鼠标 Fast：3/3 PASS，可见反馈 81.583/76.626/78.784 ms。
- 每次 operation begin=1、PlaybackManager second enter=0、successful finish=1；最终 IDLE 且按钮 enabled。
- 每次 worker accepted，`first_command_applied=True`，first sim step > 0，events_sent=2，实测 wheel 最大速度相对 restore 状态发生变化，stop_reason=complete。
- Step 5 只包含 source Step 5；restore source 为 Step 4.sim_state_after；request/ack 匹配。
- Raw/Fast 均 2 events / 2 segments，signature 均为 `wheel all 0.3`、`wheel stop`；targets 与 wheel duration 相同。Raw final 7.140 s，Fast final 4.985 s，仅删除 implicit UI idle。
- Step1、restore 中 Stop、Recording conflict、Height Generate conflict、普通 Raw Selected 均在真实可视化 Isaac 中通过。
- 两次权威 GUI run 串行执行，任一时刻只有一个 worker；关闭后残留进程 0。
- 受保护数据 73/73 文件 length + SHA-256 全部相同，无新增保护文件。

## 证据入口

- `physical_click_trace.json`
- `operation_ownership_trace.json`
- `raw_fast_selected_plan_comparison.json`
- `selected_fast_worker_trace.csv`
- `failure_cleanup_tests.csv`
- `test_results.txt`
- `protected_data_sha256_audit.txt`
- `screenshots_index.txt`
- `validation_motion.gif`
