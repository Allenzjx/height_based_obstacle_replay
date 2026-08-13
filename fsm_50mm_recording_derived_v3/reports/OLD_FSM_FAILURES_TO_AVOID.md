# Old FSM Failures to Avoid — 50 mm

## 1. Purpose and evidence policy

This is a read-only evidence audit of the twenty historical failure lessons that must constrain the 50 mm recording-derived controller. It is not a claim that the old controller has been repaired, and it is not mechanical validation of the new controller.

No Isaac Sim / Isaac Lab run, PPO run, or source/configuration edit was performed for this audit. Evidence was taken only from retained source, configuration, telemetry, result files, and handoff reports in:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2

Each finding uses one of the following evidence levels:

- **Confirmed** — directly demonstrated by retained source, telemetry, result, or an explicit historical change record.
- **Source-admitted** — the current source explicitly describes and mitigates the old failure mechanism, but the exact old source or a one-variable A/B artifact is not retained.
- **Inference** — strongly implied by the control structure or rigid-body geometry, but not isolated by a retained experiment.
- **No independent evidence** — a reasonable engineering rule, but the retained project does not independently prove the claimed historical event.

The latest handoff explicitly warns that the attempt065 historical source body was not archived, only its runtime/config hashes and telemetry/result were retained:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\chatgpt_handoff_current_version_20260802_200508\CHATGPT_HANDOFF_CURRENT_VERSION_FAILURE_REPORT.md:77-93
- The same limitation is restated at lines 345-346.

Consequently, current-source comments are not treated as a substitute for a strict historical one-variable A/B comparison.

## 2. Executive result

Of the twenty lessons:

- Thirteen are directly confirmed, including two structural data-loss findings whose exact mechanical contribution was not independently isolated.
- Three are source-admitted mechanisms or partially implemented design gaps.
- Two are physics/control inferences.
- Two have no independent retained evidence.

The strongest direct failure channels are:

1. invalid RR-to-RL command reuse;
2. RL knee motion erasing lift clearance;
3. a support-wheel command step coincident with FR load loss;
4. zero wheel speed failing to anchor a contact;
5. support-leg correction lifting a geometrically on-top wheel out of load;
6. all-or-nothing IK rollback caused by an unrelated leg at a floating-point joint boundary;
7. fixed-time phase advancement without unload/airborne/clearance/load events; and
8. PPO execution before the FSM reference passed its own promotion gates.

## 3. Twenty findings

### 1. RR motion cannot be copied or simply mirrored to RL

**Evidence level: Confirmed**

Source evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\tools\reduce_recording_50mm.py:42-47 defines project_rr_active_command_to_rl.
- Lines 50-69 construct an RL phase from the corresponding RR phase and take the RR active joint pair from indices 6:8.

Direct telemetry evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\evaluation\fsm_v2_smoke_0000_attempt018\telemetry.csv:1044-1060
  - fsm_phase=24
  - reference_04/reference_05 changes from approximately [1.0419, 0.0] to [22.8671, 0.0] deg.
  - active_wheel_clearance_m changes from about -0.02603 to +0.01790 m.
- The same file at lines 1061-1420:
  - fsm_phase=25
  - the RL reference reaches [29.8, -60.0] deg;
  - active_wheel_clearance_m ends at -0.02362 m.
- Its result is REAR_SWING_CROSSING_FAIL:
  - C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\evaluation\fsm_v2_smoke_0000_attempt018\result.json:3-10
- attempt020 repeats the same pattern:
  - C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\evaluation\fsm_v2_smoke_0000_attempt020\telemetry.csv:1117-1493
  - final RL reference [29.8, -60.0] deg and final clearance -0.02626 m;
  - result.json:3-10 reports REAR_SWING_CROSSING_FAIL.

**Conclusion:** The retained telemetry directly shows a rear-right-shaped RL swing endpoint removing RL clearance and failing the crossing. The safe rule is per-leg, per-recording kinematics; a numeric mirror is not evidence of physical equivalence.

**Regression requirement:** For every active leg, evaluate the full FK path, not just endpoint joint limits. LIFT-to-SWING clearance must not decrease through zero, and an RR command pair must not be accepted as an RL endpoint merely because it is joint-limit-valid.

### 2. Static keyframe compression does not preserve dynamic action semantics

**Evidence level: Confirmed for structural loss; Inference for the exact physical contribution**

Source/configuration evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\configs\fsm50_com_lift_v2_keyframes.json:4-8 identifies the source as the old accepted_steps.jsonl, with 159 actuator-target events and 278.872 s duration.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\tools\reduce_recording_50mm.py:84-147 stores one sampled pose, servo endpoint, wheel-speed endpoint, nominal duration, guards, and source time/step for each phase.
- That phase schema has no state for momentum, contact preload, force impulse, coast interval, integrated wheel displacement, or servo/wheel concurrency.
- Lines 177-212 infer a small set of phase times from contact transitions and nearest telemetry rows.
- Lines 219-244 turn those sampled times into semantic phase endpoints.
- Lines 543-560 report 35 retained semantic phase keyframes and 48.95 s nominal duration from the 159-event, 278.872 s source.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:1690-1697 reconstructs each ordinary phase with smoothstep interpolation between phase-entry and one endpoint.

**Conclusion:** Loss of the listed dynamic fields is directly confirmed. The retained evidence does not isolate how much of any one mechanical failure was caused by each lost field, so that causal allocation remains an inference.

**Regression requirement:** A recording reducer must preserve and compare integrated wheel displacement, preload/hold/release ordering, dwell/coast duration, and servo-wheel overlap windows. Endpoint equality alone is insufficient.

### 3. Whole-body COM correction can overwrite the active swing target

**Evidence level: Source-admitted**

Source evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:1833-1837 computes reference FK centers and saves active_reference_centers.
- Lines 1898-1923 apply longitudinal and lateral COM corrections to baseline_centers for all legs.
- Lines 1980-2004 define active_isolated_phase and restore the active leg to active_reference_centers during UNLOAD, LIFT, SWING_CLEAR, and the RL transfer sequence.
- Lines 1995-2001 explicitly state that moving the active leg in those phases corrupts its keyframe and can put the mirrored RL linkage back onto the obstacle.
- Lines 2014-2018 explicitly state that no shared support correction is to override active RL.

**Conclusion:** The current source explicitly admits the old mechanism and implements an isolation layer. Because the corresponding historical source body and a one-variable A/B run are absent, this is not classified as a directly measured old causal failure.

**Regression requirement:** With nonzero COM error and support offsets, the active FK center must remain identical to the phase keyframe throughout UNLOAD/LIFT/SWING; only support-leg centers may change. PLACE/CONFIRM must be tested separately because they intentionally rejoin load regulation.

### 4. Changing the RL knee too early after lift erases clearance and causes re-contact

**Evidence level: Confirmed**

Direct telemetry:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\evaluation\fsm_v2_smoke_0000_attempt062\telemetry.csv
  - Line 1128, time_s=56.30, fsm_phase=24: reference_04/reference_05=[11.9806, 0.0] deg, active_wheel_clearance_m=-0.000590.
  - Line 1129, time_s=56.35, fsm_phase=25: knee=0.0143 deg, clearance=+0.002002 m.
  - Line 1141, time_s=56.95: knee=13.0207 deg, clearance peaks at +0.014814 m.
  - Line 1146, time_s=57.20: knee=20.2846 deg, clearance falls to +0.004580 m.
  - Line 1147, time_s=57.25: knee=21.4770 deg, clearance crosses to -0.004253 m.
  - Line 1151, time_s=57.45: knee=24.6551 deg, clearance=-0.024753 m and rl_contact_upward_force_n=2.8734 N.

Current mitigation:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\configs\fsm50_com_lift_v2_keyframes.json:1888-1902 makes LIFT_RL active joints [16.0, 0.0] deg with all wheel targets zero.
- Lines 1956-1970 make SWING_CLEAR_RL use the same active pair [16.0, 0.0] deg with all wheel targets zero.

**Conclusion:** The old transition first gained clearance, then lost it as the knee advanced, crossed below zero clearance, and reloaded RL. This is direct evidence.

**Regression requirement:** Assert exact active-joint continuity at LIFT_RL-to-SWING_CLEAR_RL and evaluate the FK clearance at every interpolation sample, not only phase endpoints.

### 5. A support-wheel speed step at phase entry can coincide with loss of FR support

**Evidence level: Confirmed temporal association; the command step is not isolated as the sole cause**

Direct telemetry:

- attempt062 telemetry line 1128, time_s=56.30:
  - fsm_phase=24;
  - final_wheel_target_rad_s_00..03=[0,0,0,0];
  - fr_contact_upward_force_n=0.687751 N.
- Line 1129, time_s=56.35:
  - fsm_phase=25;
  - final_wheel_target_rad_s_00..03=[0.6,0.6,0,0.6] rad/s;
  - fr_contact_upward_force_n=0 N.

Current implementation:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:1762-1773 assigns support crawl during SWING.
- Lines 1774-1787 replace rear SWING commands with a smooth phase-local ramp.
- Lines 1788-1793 force the active wheel to exactly zero.

**Conclusion:** The retained telemetry proves the speed step and FR force loss occurred on the same transition. It does not prove that no other simultaneous posture/contact change contributed.

**Regression requirement:** Every wheel target at a phase's first control sample must equal its value at the preceding phase's last sample. The active wheel must remain exactly zero and support-wheel acceleration must be phase-local and bounded.

### 6. Applying a new ramp globally to every SWING can break an already valid early prefix

**Evidence level: No independent evidence**

Retained-source status:

- residual_rl_env_v2.py:1762-1787 currently distinguishes the generic SWING command from a rear-SWING smooth ramp.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\tests\test_rl_transfer_phase_graph.py:79-82 only asserts that RL wheel speed is zero throughout the RL sequence.
- No retained test, telemetry comparison, or historical source diff demonstrates that a global ramp specifically broke a previously valid front-wheel prefix.

**Conclusion:** This is a sound change-isolation rule, but it must not be reported as a measured historical fact.

**Regression requirement:** Vary rear-ramp parameters and assert byte-for-byte or tolerance-exact equality of all reference outputs from INIT through the frozen front-wheel prefix.

### 7. A reducer can discard valid sustained wheel advance

**Evidence level: Confirmed for structural loss; Inference for a particular failure's causal share**

Source evidence:

- reduce_recording_50mm.py:84-147 stores an instantaneous wheel_target_rad_s per phase but no integrated wheel travel or active interval.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\src\resume_validation\fsm_trajectory.py:50-64 has an explicit preserve_wheel_distance=True path and forwards it to preserve_wheel_active_gaps.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\src\resume_validation\residual_rl_env.py:1067-1073 zeros wheel commands while a phase gate is waiting, which can truncate a sustained command unless a separately configured recovery command applies.

**Conclusion:** The V2 reduction format structurally lacks the quantity needed to prove sustained travel preservation. A specific mechanical failure caused by that omission is not isolated.

**Regression requirement:** Compare per-wheel sum(omega * delta_time) and the exact nonzero command intervals before and after reduction. Preserve simultaneous servo and wheel actions as an invariant.

### 8. Whole-body wheel advance does not necessarily improve COM relative to the support boundary

**Evidence level: Inference**

Source/geometry evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\src\resume_validation\residual_rl_env.py:1059-1089 applies common rear-transfer wheel commands.
- Lines 1091-1121 can apply common post-transfer forward speed.
- Lines 1697-1719 calculate longitudinal margin from COM x relative to the minimum and maximum support-wheel x positions.
- If the body COM and all support points translate rigidly by the same dx, both differences remain unchanged. This is a mathematical invariant, not a retained experimental measurement.
- The current V2 directional metric explicitly uses COM relative to the FL/RR midpoint:
  - C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:674-683.

**Conclusion:** The control structure makes the concern physically well founded, but no retained trial isolates equal support/body translation as the unique reason a transfer failed.

**Regression requirement:** A rigid translation applied to COM and all support points must leave relative margin and diagonal-transfer progress unchanged; moving COM alone must change them.

### 9. Zero wheel command is not an anchored contact constraint

**Evidence level: Confirmed**

Direct telemetry:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\evaluation\development_h050_0000_recording_transfer_attempt066\telemetry.csv:2225-2226
- At time_s=37.05 and 37.0666667:
  - final_wheel_target_rad_s_00..03 remains [0,0,0,0];
  - fl_wheel_x_m moves from 0.775592744 to 0.776006758 m, approximately 0.414 mm;
  - fl_contact_upward_force_n drops from 5.756548 to 4.897809 N;
  - planned_support_margin_2d_m remains valid and moves from 0.0113441 to 0.0115673 m.
- The audit report records the same transition:
  - C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\chatgpt_handoff_current_version_20260802_200508\DEVELOPMENT_H050_0000_ATTEMPT066_MECHANICAL_AUDIT.md:62-64.

**Conclusion:** A zero velocity target did not prevent contact-point motion or load loss.

**Regression requirement:** Anchoring must require bounded measured contact-point drift, retained load, allowed-surface contact, and dwell. It must never be inferred from commanded wheel speed alone.

### 10. Excessive support-leg correction can remove FL/FR top support

**Evidence level: Confirmed**

Source admission tied to retained attempts:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:1924-1927 states that the maximum longitudinal correction raised the FR center about 9 mm and lost upward load in attempts037-038.

Direct result evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\evaluation\fsm_v2_smoke_0000_attempt037\episodes.jsonl:1
  - failure_reason=COM_TRANSFER_MARGIN_FAIL;
  - terminal_full_wheel_on_top=[true,true,false,false];
  - terminal FR upward force is 0 N;
  - terminal_support_score=0.2578507.
- The corresponding attempt038 episodes.jsonl:1 reports the same geometric/force mismatch and support score 0.2590187.

**Conclusion:** The retained source and episodes jointly confirm that a support correction can preserve a top-geometry flag while removing actual upward load.

**Regression requirement:** Bound every support-leg correction by joint and FK envelopes and separately assert that required support-load hysteresis is not violated.

### 11. All-or-nothing IK rollback can be triggered by an unrelated leg's tiny boundary roundoff

**Evidence level: Confirmed**

Old implementation:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\src\resume_validation\residual_rl_env.py:1659-1663
  - computes ik_valid for all four legs;
  - reduces it with ik_valid.all(dim=1);
  - falls all legs back to baseline_raw when any one leg is invalid.

Exact retained boundary artifact:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\telemetry\debug_v2_placement_path.json:2-3 identifies active=0 and one baseline invalid leg.
- Lines 41-45 contain probe_preloaded_ik_valid=[true,true,true,false].
- Lines 47-55 give the unchanged RR knee solution as 0.032064199447631836 rad.
- Lines 57-65 give the RR knee lower limit as 0.03206435590982437 rad.
- The underflow is approximately 1.56e-7 rad.

Current mitigation:

- residual_rl_env_v2.py:2109-2117 falls invalid baseline legs back individually.
- Lines 2127-2136 similarly apply residual IK fallback per leg.

**Conclusion:** The old all-leg rollback is directly present in source, and the unrelated RR floating-point boundary case is directly retained.

**Regression requirement:** Recreate the exact JSON case. A valid active-leg move must be accepted while only RR falls back; a genuinely out-of-range active leg must still be rejected.

### 12. Front and rear placement require opposite task-space signs

**Evidence level: Confirmed**

Source evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:2096-2099 states that negative planar-z extends FL/FR toward the surface while positive planar-z extends RL/RR.
- Lines 2100-2107 implement placement_direction=(-1,-1,+1,+1).

**Conclusion:** The kinematic sign distinction is explicit. A global placement sign is invalid for this mirrored mechanism.

**Regression requirement:** Use FK finite differences on all four legs and assert that a positive logical placement preload moves each world-space wheel center toward its intended surface.

### 13. A latched COM target can become stale when the support polygon changes

**Evidence level: Source-admitted**

Source evidence:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:748-769 captures a local target at SHIFT entry.
- Lines 777-791 define capture_stale when the held target loses valid geometry or falls below the current support-margin requirement.
- Lines 792-807 describe and implement capture_recovery_closer when hull reshaping leaves an obsolete, overly distant target.
- Lines 808-820 replace the target and track whether the replacement is safe.

**Conclusion:** The present source explicitly admits the stale-target mechanism and relatches around it. No retained telemetry field records relatch reason/count, and no one-variable historical trial proves its exact causal contribution.

**Regression requirement:** A pure target-latch test must change the support geometry after capture, trigger exactly one justified relatch, reject invalid contact transients, and expose a reason/counter.

### 14. Removing offsets abruptly across a state transition causes a command discontinuity

**Evidence level: Confirmed**

Historical change record:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\CHANGELOG.md:191-194 states that a phase-9 offset ramp reached its endpoint and then reset to zero on entry to phase 10; the fix held the scale at one throughout phase 10 and added a continuity regression.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\tests\test_load_balance.py:153-185 contains continuity tests across RECOVER/DRIVE_CLEAR and configurable ramp starts.

V2-specific admission:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:2005-2008 states that removing the recording-safe active RL endpoint at UNLOAD entry reloads RL and retracts planned margin below the unchanged 8 mm guard.

**Conclusion:** The old transition reset is directly documented and tested. The V2 source identifies the same class of failure at a different transition.

**Regression requirement:** Check servo targets, wheel targets, FK centers, and active offsets on both sides of every phase boundary, especially SHIFT-to-UNLOAD and LIFT-to-SWING.

### 15. Geometric on-top status does not prove load-bearing contact

**Evidence level: Confirmed**

Direct evidence:

- attempt037 episodes.jsonl:1 and attempt038 episodes.jsonl:1 both report terminal_full_wheel_on_top=[true,true,false,false] while the FR upward force is 0 N.

Current guard:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\residual_rl_env_v2.py:972-977 defines top_supported as wheel_on_top combined with upward force at or above the configured threshold and accumulates dwell only while both remain true.
- Lines 1311-1327 distinguish PLACE geometry/signature from CONFIRM load dwell.

**Conclusion:** The geometry/load mismatch is directly present in retained episode artifacts.

**Regression requirement:** Test on-top with 0 N, a one-sample force spike, and sustained force. Only sustained load may satisfy CONFIRM.

### 16. Fixed time is not a substitute for unload, airborne, clearance, and load dwell events

**Evidence level: Confirmed**

Old source:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\src\resume_validation\fsm_controller.py:10-23 defines only coarse global phases rather than per-wheel SHIFT/UNLOAD/LIFT/SWING/PLACE/CONFIRM phases.
- Lines 53-65 assign fixed normalized-time windows.
- Lines 91-120 compute the desired phase from elapsed time and advance only once the desired time phase is later than the current phase.
- Lines 122-138 use front/rear top-contact counts and all_wheels_clear; they have no target force ratio, airborne state, clearance threshold, or top-load dwell.

**Conclusion:** The old FSM's phase structure and transition inputs directly lack the required physical events.

**Regression requirement:** Advancing elapsed time alone must never satisfy UNLOAD, LIFT, SWING, or CONFIRM. Each guard must require consecutive physical samples and reset dwell when the condition breaks.

### 17. Relaxing a safety gate instead of fixing the physical action is not supported by retained evidence

**Evidence level: No independent evidence**

Current retained thresholds:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\configs\fsm50_com_lift_v2_keyframes.json:27-29
  - support_margin_target_m=0.008;
  - actual_support_force_threshold_n=2.0.
- Lines 87-100 retain 0.12 s airborne dwell, 0.30 s support-confirm dwell, 0.010 m lift-transition clearance, and 48.95 s nominal duration.
- The latest attempt failed its retained-load guard rather than being relabeled successful:
  - CHATGPT_HANDOFF_CURRENT_VERSION_FAILURE_REPORT.md:5
  - DEVELOPMENT_H050_0000_ATTEMPT066_MECHANICAL_AUDIT.md:11-13.

**Conclusion:** The principle is correct, but the retained evidence does not establish a specific historical case in which a safety threshold was lowered merely to pass.

**Regression requirement:** Freeze threshold values and provenance separately from motion tuning. Any lower threshold must be an explicit reviewed configuration change, never an incidental side effect of an action patch.

### 18. Ordinary support-polygon logic is not valid for two-leg diagonal support; a dedicated corridor is still missing

**Evidence level: Source-admitted / partially implemented design gap**

Old simplification:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\src\resume_validation\residual_rl_env.py:1697-1719 accepts count>=2 and reduces support geometry to support_min/support_max along x. This ignores lateral distance from a diagonal support segment.

Current polygon behavior:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\src\resume_validation\support_polygon_v2.py:37-78 requires at least one non-collinear support triple and marks geometry valid only when count>=3.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\tests\test_support_polygon_v2.py:31-36 explicitly asserts that two supports are invalid for the polygon function.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\tests\test_rl_transfer_phase_graph.py:45-54 requires PREPOSE/SHIFT/ACQUIRE to have mandatory FL/RR support but no fabricated planned triangle, then switches to FL/FR/RR after acquisition.
- residual_rl_env_v2.py:1055-1070 adds retained opposite-diagonal load, but it does not implement a geometric s/d_perp corridor.

**Conclusion:** The project now avoids fabricating a two-point polygon, but it has not implemented the dedicated diagonal support-segment corridor required to constrain along-segment progress and perpendicular drift. The specific claim that an old ordinary 2-D polygon caused a failure is not independently demonstrated; the old x-only simplification and current missing corridor are directly visible.

**Regression requirement:** Add a pure diagonal-corridor geometry helper with segment coordinate s, perpendicular distance d_perp, nondegenerate endpoints, contact drift, and retained-load dwell. It must be translation-invariant.

### 19. Wheel translation must not be mistaken for COM motion relative to the support boundary

**Evidence level: Inference**

Evidence basis:

- The old common wheel commands and relative support-margin calculation are the same source locations cited in finding 8:
  - residual_rl_env.py:1059-1121 and 1697-1719.
- The current controller's explicit relative metric is:
  - residual_rl_env_v2.py:674-683, relative_com=com_xy-fl_rr_midpoint.

**Conclusion:** Rigid-translation invariance proves that wheel displacement alone is not relative COM transfer. However, no retained source statement or isolated telemetry experiment proves that the old implementation literally counted wheel encoder travel as COM-boundary progress. This must remain an inference.

**Regression requirement:** Do not use accumulated wheel travel as COM-transfer progress. A metamorphic test must show that equal translation of COM and support points leaves every relative transfer metric unchanged.

### 20. PPO must not start before the FSM reference passes its mechanical gate

**Evidence level: Confirmed**

Historical outcome:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo\reports\chatgpt_handoff_20260731_171825\HANDOFF_SUMMARY.md:7
  - FSM performance was 12/20 at 50 mm and 7/20 at 75/100 mm.
- Line 8:
  - Method B PPO final checkpoints reached 13/20 at 50 mm and 7/20 at 75/100 mm for every seed;
  - 0/9 development gates promoted.
- Lines 9-10 attribute the outcome in part to a weak FSM reference and state that Method C had no completed result.
- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\50mm_com_lift_review_20260801_123347\BASELINE_AUDIT.md:3 independently records the untouched OLD_FSM_V34 result as 12/20 strict 50 mm scenarios.

Current V2 gate status:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\chatgpt_handoff_current_version_20260802_200508\CHATGPT_HANDOFF_CURRENT_VERSION_FAILURE_REPORT.md:5 states that attempt066 failed before the new RL sequence ran and PPO did not start.
- Lines 313-320 record that the new PPO variants were not trained and no V2 FSM-vs-PPO comparison exists.
- Lines 328-332 state that there is no complete strict single-scenario success and the PPO gate must remain closed.

**Conclusion:** Training was historically executed against an FSM that had not passed its own development target, and it did not produce a promoted controller. Current V2 correctly remains blocked.

**Regression requirement:** The training entry point must consume a frozen, hashed FSM gate artifact. No checkpoint, resume option, or CLI override may bypass the required strict mechanical success count and scenario gate.

## 4. Existing-test gaps that matter most

The current retained tests do not fully cover the highest-risk mechanisms:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\tests\test_recording_reducer_v2.py:20-26 verifies an RR-to-RL projection numerically but does not verify the resulting continuous FK clearance path.
- Lines 44-63 verify selected RL phase endpoints but not LIFT-to-SWING clearance monotonicity or boundary continuity.
- tests\test_rl_transfer_phase_graph.py:79-82 checks only that RL wheel speed is zero, not support-wheel entry continuity or ramp scope.
- tests\test_rl_transfer_phase_graph.py:115-120 checks configured offsets are zero, not runtime active-leg isolation under nonzero COM correction.
- tests\test_support_polygon_v2.py:31-36 correctly rejects two-point polygons but provides no diagonal corridor replacement.
- No retained V2 unit test was found for stale-target relatching, placement-direction FK response, the exact per-leg IK boundary artifact, or reducer wheel-distance/concurrency invariants.

Recommended pure, non-Isaac regression modules:

1. test_active_leg_isolation_v2.py
2. test_rl_lift_swing_clearance_continuity.py
3. test_wheel_ramp_scope_and_entry_continuity.py
4. test_per_leg_ik_boundary_fallback.py
5. test_placement_direction_fk.py
6. test_com_target_relatched_on_support_change.py
7. test_diagonal_support_corridor.py
8. test_reducer_dynamic_invariants.py
9. test_fsm_physical_event_guards.py
10. test_ppo_gate_requires_frozen_fsm_success.py

## 5. Audit boundary

This report establishes what the retained project does and does not prove. It does not establish that the proposed V3 controller is mechanically successful. In particular:

- attempt066 ended at UNLOAD_RR with DIAGONAL_SUPPORT_LOAD_NOT_RETAINED;
- the new explicit RL sequence did not execute;
- attempt065's historical source body is unavailable;
- the latest run reported baseline candidate IK invalidity on at least one leg for all 2,225 control steps, but the rejected targets were not logged;
- no new V2 PPO policy was trained.

Those limitations are documented in:

- C:\robotics_sim\wlr_robot\resume_validation_fsm_residual_ppo_50mm_com_lift_v2\reports\chatgpt_handoff_current_version_20260802_200508\CHATGPT_HANDOFF_CURRENT_VERSION_FAILURE_REPORT.md:5, 77-93, 181-201, 205-215, 233-237, 313-332, 345-357.

Therefore, this audit may be used to define non-regression requirements and controller design constraints, but not as evidence of V3 mechanical success or PPO readiness.
