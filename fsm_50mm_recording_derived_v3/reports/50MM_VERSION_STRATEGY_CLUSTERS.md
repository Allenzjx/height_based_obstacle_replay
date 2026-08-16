# 50 mm Recording Strategy Clusters

Clusters use exact categorical structure: physical leg-crossing order, ordered servo-joint participation, wheel-assist participation, concurrency, and whether a phase has a distinct evidence window. Numeric targets and durations are not part of the structural fingerprint and are never averaged between clusters.

## Successful Gate-A inputs

| Version | Task result | Strategy profile | Recovery profile | Plan SHA-256 | Telemetry SHA-256 | Video SHA-256 |
|---|---|---|---|---|---|---|
| v003_20260805_224517_157723_manual | REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE | PRIMARY_PROFILE | RECOVERY_PROFILE_2 | `a53acff942cbb19782c2f804add7feaca3662202107ecffadd110a3fb4acd76c` | `e8be69e66d448fac1b9044125d638f4808e5ecec8af49a151f7f30ccb734cd56` | `0031f043148db9a68170f31e19154203e99ce4e9337976d65a483b2653a8771b` |
| v008_20260806_211408_578700_manual | REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE | ALTERNATE_PROFILE_2 | RECOVERY_PROFILE_1 | `a80928c079c0fdca8080999993d2b8705d17a4686f962914eb5effe4e5993b7c` | `c6a584d6a34c4835c8d5f1c588c98ea206eda630a97cb3582792557d804fb127` | `7ee3c0eb019750843b183060c011572decd45995335bd507e29557fd74f06b5e` |
| v009_20260806_215232_433234_manual | REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE | ALTERNATE_PROFILE_1 | RECOVERY_PROFILE_1 | `f6c6934f631a01f906280ede55b8eec7fef8a37cf45242d970ef5e5285ff4c12` | `f17d8449e77b37550fe2f6047a630d1a5d6b730bb61e6b210f76157ba049c8dc` | `0dc5f08adf75f5e1f08f46032146e1ad90e2943e297fbaea4fd63466173b17d1` |
| v010_20260806_220745_363972_manual | REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE | ALTERNATE_PROFILE_3 | RECOVERY_PROFILE_1 | `6c37a2751ca170954dc617831406d7c43833ef8ccf85deb9991af40c3ee6718e` | `75abea4f01a3142268d0d41e95ac8a7d97aea8bb66f868d958859e8b608a45e3` | `e560219cbde5cf04fb5bc2d8c651762914f2d12cac92531a5a37a3c345fddbc3` |

## Explicit Gate-A exclusions

| Version | Result | First actual failure phase | Run | Plan SHA-256 |
|---|---|---|---|---|
| v005_20260805_225441_439112_manual | REPLAY_TASK_FAIL | FL lift/placement at 26.5 s (source step 8, segment 36; front_left_knee actuator_unstable) | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\50mm_fast_replay\v005_20260805_225441_439112_manual\20260815T001037_476055Z_production_fast_f3273abd30` | `f2e7a7b63d6f9245b5e8ce9c7cd3f493907b948a7f25d842471d93620fcb4e88` |
| v006_20260805_233948_654778_manual | REPLAY_TASK_FAIL | post-FL placement / pre-rear traversal at 55.7 s (source step 10, segment 37; rear_right_knee actuator_unstable) | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\50mm_fast_replay\v006_20260805_233948_654778_manual\20260815T001132_572281Z_production_fast_ef354d0232` | `9cd6cd72e9f73fc5dfddf739f31a6a397069bfad7e58f48b385cc688292ecd1e` |
| v007_20260806_190636_100857_manual | REPLAY_TASK_FAIL | RR placement / before RL traversal at 63.83 s (source step 16, segment 75; front_left_hip actuator_unstable) | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\50mm_fast_replay\v007_20260806_190636_100857_manual\20260815T001310_080566Z_production_fast_2f82878ac6` | `1ac372f238ad10055b116c49aab0bb2a220ab68bd86708169dac950220a14898` |
| v011_20260806_223621_672618_manual | REPLAY_TASK_FAIL | post-RR placement / before RL traversal at 72.03 s (source step 22, segment 136; rear_right_knee actuator_unstable) | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\50mm_fast_replay\v011_20260806_223621_672618_manual\20260815T002058_134091Z_production_fast_3616e5ed39` | `5bdf49b4e46fc48b5c438fa65aeac860794c5a89e221dbe86b7b93d90cdbac27` |
| v012_20260806_231025_027004_manual | REPLAY_TASK_FAIL | front-pair transition / pre-rear traversal at 59.57 s (source step 10, segment 55; front_right_hip actuator_unstable) | `C:\robotics_sim\wlr_robot\height_based_obstacle_replay\fsm_50mm_recording_derived_v3\runs\50mm_fast_replay\v012_20260806_231025_027004_manual\20260815T002239_014202Z_production_fast_44a08a6be4` | `edaf002669c93352519afef5339ab5967e874f6b0104ae7d4005998c2123d584` |

## Structural strategy clusters

### ALTERNATE_PROFILE_1

- Versions: v009_20260806_215232_433234_manual
- Fingerprint: `7712cba1cfb8677efad1e495f55946c0dac072c35a3f04698e188affe7743497`
- Crossing order: FR -> FL -> RR -> RL
- Phase structures:

  - INITIAL_APPROACH: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_FR_COM_SHIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - FR_UNLOAD_AND_LIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_FACE_CROSS: servo=['rear_left_hip', 'front_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_TOP_PLACE: servo=['front_right_knee', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'rear_left_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FL_UNLOAD_AND_LIFT: servo=['front_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_FACE_CROSS: servo=['front_left_hip', 'front_left_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_TOP_PLACE: servo=['front_left_knee', 'front_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FRONT_PAIR_ADVANCE: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_RR_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'rear_left_knee', 'rear_right_hip']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - RR_UNLOAD_AND_LIFT: servo=['rear_right_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_FACE_CROSS: servo=['rear_right_hip', 'rear_right_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_TOP_PLACE: servo=['rear_right_hip', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_knee']; wheel=['front_left_ankle', 'rear_left_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - PRE_RL_SUPPORT_SETUP: servo=[]; wheel=[]; concurrent=False; status=
  - PRE_RL_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'rear_left_ankle', 'rear_right_ankle', 'front_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RL_UNLOAD_AND_LIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RL_FACE_CROSS: servo=['rear_left_hip', 'rear_left_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RL_TOP_PLACE: servo=['rear_left_knee', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FINAL_ADVANCE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - FINAL_POSTURE_RECOVERY: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY

### ALTERNATE_PROFILE_2

- Versions: v008_20260806_211408_578700_manual
- Fingerprint: `e5337ca437b8bcedad95ab8ce2fe3e19f185756c2347ed6d9ada82f27a3d8624`
- Crossing order: FR -> FL -> RR -> RL
- Phase structures:

  - INITIAL_APPROACH: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_FR_COM_SHIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - FR_UNLOAD_AND_LIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_FACE_CROSS: servo=['rear_left_hip', 'front_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_TOP_PLACE: servo=['front_right_knee', 'front_right_hip', 'front_left_hip', 'front_left_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FL_UNLOAD_AND_LIFT: servo=['front_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_FACE_CROSS: servo=['front_left_hip', 'front_left_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_TOP_PLACE: servo=['front_left_knee', 'front_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FRONT_PAIR_ADVANCE: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_RR_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RR_UNLOAD_AND_LIFT: servo=['rear_right_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_FACE_CROSS: servo=['rear_right_knee', 'rear_right_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_TOP_PLACE: servo=['rear_right_hip', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - PRE_RL_SUPPORT_SETUP: servo=[]; wheel=[]; concurrent=False; status=
  - PRE_RL_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RL_UNLOAD_AND_LIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RL_FACE_CROSS: servo=['rear_left_hip', 'rear_left_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RL_TOP_PLACE: servo=['rear_left_knee', 'rear_left_hip', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FINAL_ADVANCE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - FINAL_POSTURE_RECOVERY: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY

### ALTERNATE_PROFILE_3

- Versions: v010_20260806_220745_363972_manual
- Fingerprint: `e6ca4a52b0b15359cb97754686cbd95dba6f98026e9efedecfd45d54993449c8`
- Crossing order: FR -> FL -> RR -> RL
- Phase structures:

  - INITIAL_APPROACH: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_FR_COM_SHIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - FR_UNLOAD_AND_LIFT: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_FACE_CROSS: servo=['rear_left_hip', 'front_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_TOP_PLACE: servo=['front_right_knee', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'rear_left_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FL_UNLOAD_AND_LIFT: servo=['front_left_knee', 'front_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_FACE_CROSS: servo=['front_left_hip']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_TOP_PLACE: servo=['front_left_knee', 'front_left_hip']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FRONT_PAIR_ADVANCE: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_RR_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RR_UNLOAD_AND_LIFT: servo=['rear_left_hip', 'rear_right_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_FACE_CROSS: servo=['rear_right_hip', 'rear_right_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_TOP_PLACE: servo=['rear_right_hip', 'front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - PRE_RL_SUPPORT_SETUP: servo=[]; wheel=[]; concurrent=False; status=
  - PRE_RL_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RL_UNLOAD_AND_LIFT: servo=['rear_right_knee']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RL_FACE_CROSS: servo=['rear_right_knee', 'front_right_hip', 'rear_left_knee', 'rear_left_hip']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RL_TOP_PLACE: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FINAL_ADVANCE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - FINAL_POSTURE_RECOVERY: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY

### PRIMARY_PROFILE

- Versions: v003_20260805_224517_157723_manual
- Fingerprint: `fa834c525361a1537f65c78c014a0c8736af66181d0a1a771ebe809097e14045`
- Crossing order: FR -> FL -> RR -> RL
- Phase structures:

  - INITIAL_APPROACH: servo=[]; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - PRE_FR_COM_SHIFT: servo=['rear_left_hip', 'front_right_hip']; wheel=[]; concurrent=False; status=COMMAND_AND_BASE_PROXY
  - FR_UNLOAD_AND_LIFT: servo=['rear_left_hip', 'front_right_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_FACE_CROSS: servo=['front_right_hip', 'front_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FR_TOP_PLACE: servo=['front_right_hip', 'front_left_hip', 'front_left_knee', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FL_UNLOAD_AND_LIFT: servo=['front_left_knee', 'front_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_FACE_CROSS: servo=['front_left_hip', 'front_left_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FL_TOP_PLACE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - FRONT_PAIR_ADVANCE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - PRE_RR_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RR_UNLOAD_AND_LIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - RR_FACE_CROSS: servo=['rear_right_knee', 'rear_right_hip']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - RR_TOP_PLACE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - PRE_RL_SUPPORT_SETUP: servo=[]; wheel=[]; concurrent=False; status=
  - PRE_RL_COM_SHIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_right_ankle', 'front_left_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - RL_UNLOAD_AND_LIFT: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - RL_FACE_CROSS: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle']; concurrent=True; status=GEOMETRY_EVENT_ANCHORED
  - RL_TOP_PLACE: servo=['rear_left_hip']; wheel=[]; concurrent=False; status=GEOMETRY_EVENT_ANCHORED
  - FINAL_ADVANCE: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY
  - FINAL_POSTURE_RECOVERY: servo=['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee', 'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee']; wheel=['front_left_ankle']; concurrent=True; status=COMMAND_AND_BASE_PROXY

## Profile-selection rule

The cluster containing v003 is `PRIMARY_PROFILE`. Other successful structural clusters are `ALTERNATE_PROFILE_n`. Final-posture command structures are kept separately as `RECOVERY_PROFILE_n`; all current successful Gate-A outcomes are posture-incomplete, so these are demonstrated recovery attempts, not proof of posture-complete recovery.

A first Macro FSM should select one complete profile per phase. It must not average targets across structural clusters. Alternate selection needs a bounded feedback guard tied to unload, clearance, crossing, or top-placement evidence.
