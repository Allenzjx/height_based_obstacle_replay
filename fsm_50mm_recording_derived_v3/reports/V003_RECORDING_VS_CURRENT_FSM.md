# v003 Recording vs Current FSM Visual Comparison

Offline comparison only: no simulator was launched and source run artifacts were not modified.

- Recording MP4: `0031f043148db9a68170f31e19154203e99ce4e9337976d65a483b2653a8771b`; 15 fps; 1364 frames.
- Current coalesced-r4 baseline MP4: `9a446fbfab7aae0ae84d167e5ac15e1bf38a3495582c7aa04d884fb27fed00f7`; 15 fps; 1361 frames; bundle `5742cda6e43859833b872a250220f18c6696e3d569a12962581e342966990b78`.
- Generated outputs: `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\visual_review\v003_recording_vs_fsm_realtime.mp4`, `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\visual_review\v003_recording_vs_fsm_phase_aligned.mp4`, and `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\reports\visual_review\v003_recording_vs_fsm_keyframes.png`.

## Evidence boundary

The videos are manifest/ledger/SHA bound and phase labels are derived from recording `segment_cursor` ownership and current FSM `macro_state`. They are visual evidence only. They do **not** prove a true COM measurement, filtered or force-based diagonal-support evidence, or the proposed new feedback-landing S10 logic. Those require live runtime instrumentation and validation.

## Phase alignment

- `S1_APPROACH_AND_PRE_FR_SHIFT`: recording frames [0, 213]; FSM frames [0, 213]; aligned 213.
- `S2_FR_TRAVERSE`: recording frames [213, 332]; FSM frames [213, 330]; aligned 119.
- `S3_FL_TRAVERSE`: recording frames [332, 882]; FSM frames [330, 874]; aligned 550.
- `S6_RR_TRAVERSE`: recording frames [882, 1038]; FSM frames [877, 1034]; aligned 157.
- `S8_RL_COM_SHIFT_AND_TRAVERSE`: recording frames [1038, 1152]; FSM frames [1034, 1153]; aligned 119.
- `S9_FINAL_ADVANCE`: recording frames [1152, 1320]; FSM frames [1153, 1322]; aligned 169.
- `S10_POSTURE_RECOVERY`: recording frames [1320, 1356]; FSM frames [1322, 1353]; aligned 36.
