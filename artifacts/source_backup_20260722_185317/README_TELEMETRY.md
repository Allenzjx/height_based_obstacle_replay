# Height Replay Telemetry

本模块为 `height_replay_ui.py` 和 Isaac worker 增加全身状态、稳定性、接触、关节和回放事件采集。

## 快速运行

```powershell
cd C:\robotics_sim\wlr_robot\height_based_obstacle_replay
python height_replay_ui.py --auto-play --height-cm 5 --telemetry --no-live-viz --report
```

GUI 默认走子进程 worker，telemetry 会随 worker 一起启动：

```powershell
python height_replay_ui.py --ui --height-cm 5 --telemetry --live-viz --report
```

关闭 telemetry：

```powershell
python height_replay_ui.py --ui --height-cm 5 --no-telemetry
```

## 配置

默认配置位于 `config/telemetry.yaml`。常用开关：

- `--telemetry` / `--no-telemetry`
- `--live-viz` / `--no-live-viz`
- `--report` / `--no-report`
- `--equilibrium-region` / `--no-equilibrium-region`
- `--telemetry-rate 50`
- `--output-dir runs`
- `--telemetry-config path\to\telemetry.yaml`

## 输出目录

每次运行会创建类似：

```text
runs/YYYYMMDD_HHMMSS_obstacle-05cm_5cm-accepted-steps
```

主要文件：

- `telemetry_samples.csv`: 主时间序列，包含 base pose、速度、COM、稳定裕度、回放上下文。
- `body_com_timeseries.csv`: 每个 link 的质量、世界系 COM 和对全身 COM 的贡献。
- `joint_timeseries.csv`: 关节位置、速度、加速度、目标、跟踪误差、力矩、功率和能量。
- `contacts.csv`: 接触体、接触点、法向、法向力、滑移指标和数据来源。
- `events.jsonl`: 稳定裕度过低、roll/pitch、关节限位、力矩、冲击、滑移、回放命令等事件。
- `telemetry_timeseries.npz`: 常用数值序列的压缩 NumPy 包。
- `model_audit.json` / `model_audit.txt`: 模型质量、关节限制、接触传感器和 USD schema 审计。
- `dashboard.html`: 离线报告入口。

## 报告和对比

重新生成报告：

```powershell
python tools\generate_report.py runs\YYYYMMDD_HHMMSS_obstacle-05cm_5cm-accepted-steps
```

对比多个 run：

```powershell
python tools\compare_runs.py runs\run_a runs\run_b --output-dir runs\comparison
```

## 数据来源和限制

- 全身 COM 优先使用 `ArticulationData.body_com_state_w` 和 `default_mass`。
- 如果 body COM 状态不可用，会退回 `body_link_state_w + body_com_pose_b`。
- 接触力优先使用 Isaac Lab `ContactSensor.net_forces_w`。
- 若 PhysX 接触点不可用，支撑点使用轮体几何投影近似，`contacts.csv` 的 `source` 和 `geometry_source` 会明确标注。
- 摩擦可行域使用 SciPy `linprog`。如果 SciPy 不可用，报告中会显示 `scipy_unavailable`，不会伪造 equilibrium region。
