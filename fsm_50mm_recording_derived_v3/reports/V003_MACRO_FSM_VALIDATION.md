# v003 Macro FSM Gate-C Validation

## Gate-C verdict

**GO：Milestone 1 和 Milestone 2 均完成。** 封存的 v003-derived nominal Macro FSM 先完成一次 reviewed baseline，随后通过同一 runner 串行启动恰好三个 fresh workers；三次 repeat 连同 baseline 均完成 50 mm 越障。四次 terminal outcome 都是 `TASK_SUCCESS_POSTURE_INCOMPLETE`：这是 4/4 traversal task success，同时也是 0/4 posture-complete，不能把后者改写为 task failure。

Gate D cross-version runtime 和 Gate E PPO residual 不属于这份结论。

## Frozen inputs

| Input | SHA-256 / value |
|---|---|
| Environment lock | [`environment_lock_50mm.json`](environment_lock_50mm.json), `736872ee80bed46d8a8b4a821bbcbf5e8500ec840c740c3fd651d9577807e18b` |
| Graph | `ffa5acfbf64b65c22eee54709a2afae5a56fa0b9345d8db84eb86acac58447c5` |
| Profile library | `3fb1501c40a8669681f5a073036a496ae221b7d91c3ceb629e93e37ac7c2ceea` |
| Gate-C bundle | `0579218825bcfdd4dabf7ab6225268ac04279a83ed96cad2ed390955414deebc` |
| v003 Fast plan | `a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c` |
| Gate-B alignment | [`50MM_COMMON_PHASE_ALIGNMENT.csv`](50MM_COMMON_PHASE_ALIGNMENT.csv), `f1b0e55b76ffddcf45c3727a490e3355a3531840443e0b086f743cd22efce358` |
| Gate-A success table | [`50MM_REPLAY_TASK_SUCCESS_TABLE.csv`](50MM_REPLAY_TASK_SUCCESS_TABLE.csv), `5549ee54b8e1aa17954c8d7dc0c9c88feee445ad32dbf6f58f6ccf900f45419c` |
| Sealed macro controller source | `a9ee7e6ea1d166a81a2c1351a757b2f110f0a1ff8a0864542152dae812c42cf0` |
| Sealed worker session source | `72d18685d708dbfc5deda5ff750939d754dda0c0489ba873e20a54c58168960d` |
| Sealed state model source | `2cd9d4a67fa4b114bd1265e07f1471639d779e1c1fa1ad7120479236d5e579dc` |
| Sealed motion profiles source | `0648189601e805f83658eb71513ed60da83f001cc97f6fd208bbb08f0e9913a7` |

这些是 accepted Gate-C bundle 的身份；本报告本身在运行后生成，不反向改变 bundle。

## Reviewed run matrix

所有视频均为 1280×720、15 fps、1,187/1,187 frames full-decode valid。roll/pitch 是 secondary stability diagnostics，不参与 task success gate。

| Run | Kind/index | Request | Fresh worker identity | Result | Peak \|roll\| / \|pitch\| rad | Final geometry / strict velocity | SHA-bound review |
|---|---|---|---|---|---|---|---|
| [baseline](../runs/v003_macro_fsm/v003_20260805_224517_157723_manual/baseline/20260815T035949_204251Z_baseline_00_0579218825bc/macro_fsm_runner_manifest.json) | baseline/0 | `be27cb548dc34c8b93e1e544d90a6efe` | PID 14420; session `1e3d7b0cb9734f32bcf350925f3931ad`; adapter `ae3749418da54b65bdf35e3324c5f831` | `TASK_SUCCESS_POSTURE_INCOMPLETE` | 0.171573 / 0.141686 | FL=AIR; FR/RL/RR=TOP; velocity stable=false | complete; task=true; posture-incomplete=true |
| [repeat 0](../runs/v003_macro_fsm/v003_20260805_224517_157723_manual/repeats/20260815T041315_948438Z_repeat_00_0579218825bc/macro_fsm_runner_manifest.json) | repeat/0 | `e2cf3d2756de4f8ba5a3cb5a15d57e00` | PID 12188; session `119fb5df470144bb8a06ee271aa8924d`; adapter `a36b104f1e3146bcb5d10c3d6a3e5814` | `TASK_SUCCESS_POSTURE_INCOMPLETE` | 0.171573 / 0.141686 | FL=AIR; FR/RL/RR=TOP; velocity stable=false | complete; task=true; posture-incomplete=true |
| [repeat 1](../runs/v003_macro_fsm/v003_20260805_224517_157723_manual/repeats/20260815T041558_459908Z_repeat_01_0579218825bc/macro_fsm_runner_manifest.json) | repeat/1 | `d4118e13440b4b7b9436ca4e25c4a62c` | PID 39412; session `d9b7507bad90487888a7cf535289bc58`; adapter `3512a78c0e58406f8e0c4facd7ed31bf` | `TASK_SUCCESS_POSTURE_INCOMPLETE` | 0.171573 / 0.141686 | FL=AIR; FR/RL/RR=TOP; velocity stable=false | complete; task=true; posture-incomplete=true |
| [repeat 2](../runs/v003_macro_fsm/v003_20260805_224517_157723_manual/repeats/20260815T041841_364514Z_repeat_02_0579218825bc/macro_fsm_runner_manifest.json) | repeat/2 | `ca0c0a64501441fca5b84a1c0098557d` | PID 16076; session `65e4a67837ef4edc87726c8ba45f5ebd`; adapter `3f9a26f2bf294c92a180e0c4589b98e7` | `TASK_SUCCESS_POSTURE_INCOMPLETE` | 0.171573 / 0.141686 | FL=AIR; FR/RL/RR=TOP; velocity stable=false | complete; task=true; posture-incomplete=true |

repeat directory count 恰好为 3，trial indices 恰好为 0/1/2；三次 request、PID、worker session、adapter runtime instance、video SHA 和 ledger SHA 全部互异。worker 串行关闭并重新启动，满足 fresh-worker repeat，不是同一 Isaac session 内重复计数。

## Run artifact hashes

manual verdict 的 `video_sha256`、`worker_result_sha256`、`task_inputs_sha256` 已逐文件重新计算并全部匹配。

| Run | Video SHA-256 | Worker result SHA-256 | Task inputs SHA-256 |
|---|---|---|---|
| baseline | `de41fb360fe15be0848ec812574f5be9a35bb4f5f8fb1c8c0aae7d16d1c49815` | `f13ca7ea2fd6533ecfde8a78d61221c2ca16092cc5a7fa3ea714af9cd2ed1505` | `9462eeed4a4c1bda2adefffb3e25440e8921620446905aa96b1d966d8927ed88` |
| repeat 0 | `328c669ff17cabc41b1e0fb8683cdd6889f820da5ad0aa920287349afad44dbe` | `7f9768f48a9498ea3376bfc17ff30a8d6e5e657a796ac90af35717b1efad39dd` | `0125e63588e5f4d6ef62c7982884e729ed2cc49e57dd49652ab301bcc5a1f7fc` |
| repeat 1 | `3a3d323de0b27f745318da95c6c0b4208c379de46a362120ea563d4b86dc9030` | `54b858e0ee97a0303a430682fcfed00b5168b1ea927010d2dd307e207fbcc3b2` | `88289e4ea9ae5176eddc73b19a0abd28af1a99de37f0b00803f0152b85da78df` |
| repeat 2 | `fd870f9afac41334dee0ece2c764920a3340974550f57361b1a26f2002180ab2` | `c175b4eaaa392aa549ed725d4163a1cbdebb4a0d913e4c51bf6b97959d87e976` | `05bf7de6a1bd3c3fb24ac4948fe6d92c21946277aec0baea812b05b282dc2aad` |

| Run | Source-consumption ledger SHA-256 | Physical-dispatch ledger SHA-256 | Transition ledger SHA-256 |
|---|---|---|---|
| baseline | `e575876a21b858f18e6074b7ffd8477d94a35152a66ab2d60a77a1fb4ca2cd6f` | `f215c1ae82005022c18493a5c53edf3aaf26b3ba1d16fe1dff5fe45d7a806b36` | `6668b39301572dde1a4580617df9511919309f338ee825d9f17e0c1665de2f9a` |
| repeat 0 | `ac94cb2b3b00d5485406883d33b0b6534b75a1fffccff2d270b65c3847f410bc` | `a40568bf9505a068726a04e5af5625c1945ef175800733e4c43b68610091a071` | `17c14c838b03958bc8a42ad42444dc61cfb321ee6291a28d0394246696a32d00` |
| repeat 1 | `90516a596cb1e782a7939f2da4d3e5979ccbd6be0f6c075a2f330f60d09fb895` | `6a2a9d9e7a3d447012adbff7b02a09463928d530019dc297e4496970927f4d78` | `58d4d3c8e6db3476d63ca95eb632030aaa707c7aad7bce2d09e6de380b226528` |
| repeat 2 | `7d0825ae945dfaaffdac866f1c2ed454acfc37d9ced0c1bb0bf31a7abc87e671` | `a4886f893100ddf19527b2865d24fdbfac4a3f75e04ca08ce63b1f03554c4788` | `c2f58f0feadeaa69f1f583cdac24f963ad0c0d350087d5392379d4b247082cd3` |

不同 run 的 ledger SHA 不同是正常的：request/batch/runtime identities 和 sim timestamps 属于 ledger payload；逐行重建后的 source coordinates、targets、ownership、epochs 和 transition semantics 相同。

## Reconstructed execution invariants

四次运行分别逐行从 sealed profile library 重建，均满足：

- source actions 112/112，source segments 0..111 exactly once and in order；160/160 commands；24/24 source Steps；
- profile source始终为 v003；无 cross-source switch、identity replay、cursor rewind 或 skipped action；
- segments 7、41、57 是精确的 same-target consumption no-op；
- physical dispatch 112：109 `SOURCE_ACTION` + 3 `BOUNDARY_ZERO_WHEELS`；无 target-changing `HOLD`/`REPLAY` dispatch；
- command epochs 1..112 严格连续；每个 dispatch 是 full 8-servo + 4-wheel atomic batch，ack target完全匹配，且 N+1 readback verified；
- controller ticks 9,432；minimal telemetry samples 1,189；transitions 12；每行 retry count 0；
- transition path、steps 和 epochs 四次完全一致：steps `[189,190,1808,2468,6600,6601,6602,7692,7693,8115,9460,9620]`，epochs `[0,0,8,24,42,42,42,58,58,102,104,112]`；
- `root_state_write_count=0`；physics dt = 1/120 s；target audit available、unsafe=false、violations=[]；
- closure safe-stop 为 atomic servo-hold + zero wheels，后续 physics readback `VERIFIED`；无 timeout、worker error 或 force-close。

state ownership、guard细节和 transition reason见 [V003_MACRO_FSM_MAP.md](V003_MACRO_FSM_MAP.md)。

## Task and safety verdict

四份 SHA-bound manual reviews 都确认：FR、FL、RR、RL 各自在 crossing 前可见地 lift，随后跨过 obstacle front face并到达 top；body crossed并保持在台阶上；无 robot fall、持续不可恢复 body stall、wheel-only drive-up、dangerous body collision、severe penetration 或 joint-limit-like pose。final stance upright/recoverable，但非 neutral，FL 最终为 AIR，且 strict final velocity不稳定。因此：

```text
traversal task: SUCCESS (4/4)
final posture: INCOMPLETE (4/4)
combined terminal: TASK_SUCCESS_POSTURE_INCOMPLETE (4/4)
```

这正是“主越障成功 + final recovery 仍需优化”的语义。final all-TOP、home pose、strict rest、contact drift 和 load balance 仍是 secondary diagnostics。

## Measurement limitations

- Gate-C normal telemetry 没有 measured COM；controller使用 root/base position proxy。
- wheel contact load 全为 unavailable/`None`；support classification 是 geometry-only candidate，不证明载荷或 anchored contact。
- runtime dangerous non-wheel collision/severe penetration producer保持 unavailable；hard detection若为 true 会立即 fail，未检测到时的最终排除依赖完整 SHA-bound video review。
- body-stuck 与 active-leg-trapped 保持 tri-state producer semantics；本次 runtime 为 unknown、manual review 为 false。
- 四次高度确定性的 nominal v003 结果不等于 cross-version generalization，也不等于 sim-to-real validation。

因此 Gate C 可以关闭，但姿态恢复问题仍开放；不能在这份报告中宣称 Gate D 或 PPO residual 已完成。

