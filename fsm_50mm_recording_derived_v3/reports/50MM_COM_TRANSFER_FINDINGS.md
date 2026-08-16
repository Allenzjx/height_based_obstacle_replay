# 50 mm COM Transfer Findings

## 结论

四个 Gate-A 成功 Recording 和 v003 Gate-C nominal FSM 支持“每次关键后腿 traversal 前都存在有方向性的 body-motion / unload preparation”这一控制假设；其中 PRE_FR 与 PRE_RL 的 base-translation proxy 在四个成功版本中方向一致，v003 profile也能在一次 baseline + 三次 repeat中稳定完成 FR→FL→RR→RL 越障。

但现有 normal-development telemetry **没有 mass-weighted COM measurement，也没有 contact force/load**。因此本文只能写 `COM target candidate`、`body/base translation proxy` 和 `unload outcome`，不能把这些数据解释成 measured COM trajectory、support margin、force impulse、anchored support 或因果载荷转移证明。

## Evidence levels

| Level | 本项目当前可用证据 | 可以得出的结论 | 不可以得出的结论 |
|---|---|---|---|
| Recording command | exact servo/wheel command、顺序、时间、concurrency | 动作结构和 target direction hypothesis | 实际 force、load、COM response |
| Gate-A phase alignment | measured base translation、wheel-center geometry、cross/TOP events | candidate body-motion direction、phase是否有效 | mass-weighted COM direction、contact mechanism |
| Gate-C guard | root/base position proxy、active-leg unload、crossing/TOP、attitude | 当前 state 的 event guard 是否满足 | force integral、load redistribution |
| Manual video | 可见 lift/cross/body traversal/fall/collision/posture | task和hard visual safety verdict | 精确 COM、normal force、support margin |

代码中的 [fsm50_macro_controller.py](../fsm50_macro_controller.py) 只有在 `com_position_m` 存在时才返回 `MEASURED_COM`；本次所有 accepted runs 均记录 `com_measurement_available=false`、`com_proxy_source=ROOT_POSITION_AND_WHEEL_GEOMETRY`。下表中的 dx/dy 因而都是 base translation。

## Gate-B transfer windows

数据来自 [`50MM_COMMON_PHASE_ALIGNMENT.csv`](50MM_COMMON_PHASE_ALIGNMENT.csv)。functional windows 可以重叠，尤其 PRE_RR 可能包含 front-pair forward advance，所以跨版本的大 dx 不能直接等同于 lateral COM transfer magnitude。

| Phase | Version | Step / segment | Candidate target | Candidate support geometry | Base dx / dy (m) | Servo+wheel concurrent | Evidence status |
|---|---|---|---|---|---:|---|---|
| PRE_FR_COM_SHIFT | v003 | 1:2 / 0:5 | toward RL | FL, RR | -0.011528 / +0.015370 | no | base proxy only |
| PRE_FR_COM_SHIFT | v008 | 1:1 / 0:3 | toward RL | FL, RR | -0.009083 / +0.012306 | no | base proxy only |
| PRE_FR_COM_SHIFT | v009 | 1:1 / 0:5 | toward RL | FL, RR | -0.010242 / +0.013507 | no | base proxy only |
| PRE_FR_COM_SHIFT | v010 | 1:1 / 0:3 | toward RL | FL, RR | -0.013117 / +0.017355 | no | base proxy only |
| PRE_RR_COM_SHIFT | v003 | 11:11 / 40:40 | toward FL/front candidate | FR, RL | -0.007412 / -0.004767 | yes | base proxy only; overlapping connective action |
| PRE_RR_COM_SHIFT | v008 | 8:13 / 38:60 | toward FL/front candidate | FR, FL, RL | +0.532233 / +0.016586 | yes | base proxy only; substantial forward advance in window |
| PRE_RR_COM_SHIFT | v009 | 7:11 / 48:73 | toward FL/front candidate | FL, RL | +0.493018 / -0.017814 | no | base proxy only; substantial forward advance in window |
| PRE_RR_COM_SHIFT | v010 | 9:14 / 40:60 | toward FL/front candidate | FR, FL, RL | +0.427604 / +0.006630 | yes | base proxy only; substantial forward advance in window |
| PRE_RL_COM_SHIFT | v003 | 14:17 / 56:64 | toward FR/diagonal candidate | FL, RR | +0.012825 / -0.016209 | yes | base proxy only |
| PRE_RL_COM_SHIFT | v008 | 15:20 / 74:101 | toward FR/diagonal candidate | FL, RR | +0.009871 / -0.022215 | yes | base proxy only |
| PRE_RL_COM_SHIFT | v009 | 13:20 / 90:115 | toward FR/diagonal candidate | FL, RR | +0.015206 / -0.032628 | yes | base proxy only |
| PRE_RL_COM_SHIFT | v010 | 18:19 / 91:94 | toward FR/diagonal candidate | FL, RR | +0.023350 / -0.018733 | yes | base proxy only |

可重复的静态发现：

- PRE_FR：四版本均为 -X/+Y base response，candidate target均为 RL；动作主要由 rear-left hip 与 front-right hip 的 phase-local servo sequence构成。
- PRE_RR：结构差异最大。v003 的 window 很短；v008/v009/v010 与明显 forward body advance重叠，且 concurrency并不一致。它们应保留为 categorical profiles，不能平均成单一“COM impulse”。
- PRE_RL：四版本均为 +X/-Y base response，FL/RR candidate geometry相同，四版本均包含 servo+wheel concurrency。这是当前 corpus 中最一致的后腿前 transfer pattern，但仍不是 measured COM 或 force proof。

## Gate-C live guard findings for v003

四次 accepted runs 的 transition evidence相同：

| State boundary | Intended target | Position source | Actual guard path | Proxy detail | Interpretation |
|---|---|---|---|---|---|
| S1 → S2 | RL | `BASE_POSITION_PROXY` | `active_leg_unloaded=true`；`displacement_ready=false` | forward proxy +0.172664 m；target-projected value为 -0.164345 m | FR unload足以通过；不能宣称 measured COM 已达到 RL |
| S5 → S6 | FL | `BASE_POSITION_PROXY` | `active_leg_unloaded=true`；`displacement_ready=false` | fresh inherited projected displacement约 +0.000344 m，小于 0.003 m threshold | RR unload足以通过；S5 没有伪造 COM success |
| S8 → S9 | FR candidate encoded in state | `BASE_POSITION_PROXY` | RL airborne-before-crossing + crossed + TOP | target-projected proxy约 +0.000358 m；LEG_TRAVERSED guard本身没有 COM threshold | profile产生了成功 RL traversal；不能声称 FR-target COM magnitude已验证 |

这三行说明第一版 FSM 的反馈边界是保守的：COM-shift state可以由真正 unload事件完成，而不是强行用不可靠的 base delta充当 COM ground truth；RL combined state则以最终物理 traversal evidence结束。

## Mechanism interpretation

当前数据与三种候选机制都相容，但不能唯一鉴别：

1. **Reaction/inertial transfer**：快速 servo变化与 wheel assist可能先产生 body velocity；没有 force integral，故只保留为候选解释。
2. **Support-angle transfer**：分段 joint posture可移动 body相对候选支撑区域；geometry和成功 traversal支持“有效”，但支撑点不必 anchored。
3. **Wheel assist**：PRE_RR/PRE_RL 和多处 TOP placement 的确保留 concurrent servo+wheel commands；这证明 command concurrency存在，不证明某一 wheel承担了多少冲量。

有效且已验证的控制结论是：完整 Recording-derived profile + live unload/cross/TOP guards 能重复完成 v003 traversal。尚未验证的学术结论是某一个 joint、wheel或 diagonal load path对 COM shift 的独立因果贡献。

## Posture-incomplete 与后续测量

四次 v003 Gate-C final class 都是 `FL=AIR; FR/RL/RR=TOP`，final velocity strict-stable=false，但 body crossed且 final recoverable。因此未完成项是 recovery/settling，不是 COM transfer 导致的 traversal failure。

后续若为了 Gate E reward、privileged critic 或 mechanism analysis增加测量，优先项是 mass-weighted COM、per-wheel normal/contact wrench、support polygon/margin、phase-local force integral和明确的 availability flag。它们应作为优化与解释证据，不能反过来否定已经通过完整视频和事件 guard 的 Gate-C task success。

相关支撑边界见 [50MM_SUPPORT_DIAGONAL_FINDINGS.md](50MM_SUPPORT_DIAGONAL_FINDINGS.md)，运行结果见 [V003_MACRO_FSM_VALIDATION.md](V003_MACRO_FSM_VALIDATION.md)。

