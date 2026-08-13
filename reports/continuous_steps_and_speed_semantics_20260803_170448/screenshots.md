# Visible GUI evidence

All files were visually inspected and are non-empty. The control window shows the connected real Isaac PID, 5 cm accepted sequence, requested/effective speed, command progress, profile, and completion state. `real_gui_runtime_audit.json` independently records `effective_headless=false` and `first_visible_render_completed=true` from each worker.

- Raw paused while UI window was moved and the Speed Scale tab was changed: `real_raw_final/screenshots/formal_raw_paused_ui_independent.png`
- Raw complete, 35/35 steps and 248/248 commands: `real_raw_final/screenshots/formal_raw_completed.png`
- Fast active: `real_fast_final_retry/screenshots/formal_fast_continuous_active.png`
- Fast complete, 35/35 steps and 248/248 commands: `real_fast_final_retry/screenshots/formal_fast_completed.png`
- Requested wheel 0.3 at 100%: `real_fast_final_retry/screenshots/wheel_100_percent.png`
- Requested wheel 0.6 at 200%: `real_fast_final_retry/screenshots/wheel_200_percent.png`
- Safe-range wheel 0.1 at 100%: `real_validation_only/screenshots/wheel_100_percent.png`
- Safe-range wheel 0.2 at 200%: `real_validation_only/screenshots/wheel_200_percent.png`
- Same-speed 200% record/play completion: `real_record_final/screenshots/record_play_same_speed_200.png`
