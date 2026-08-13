# UI Performance Report

Date: 2026-07-22

## Measurements

Baseline artifact:

- Path: `artifacts\ui_performance_before.json`
- Mode: `no_sim_existing_code_tab_switch_baseline`
- Count: 200
- p50: 0.0141 ms
- p95: 0.1255 ms
- max: 1.1291 ms

After-change artifact:

- Path: `artifacts\ui_performance_after.json`
- Mode: `no_sim_tk_tab_switch`
- Count: 200
- Tabs: 10
- p50: 30.2272 ms
- p95: 64.1572 ms
- max: 98.7639 ms
- mean: 34.7897 ms

## Interpretation

The baseline file measured a very small Notebook `select()` operation. The after-change benchmark uses `tools\ui_performance_benchmark.py` and includes real Tk `select + update_idletasks + update`, so it is closer to user-perceived no-sim tab switching but is not directly comparable to the old micro-measurement.

The measured after-change p95 is below 100 ms in no-sim UI conditions. The Live Stability Dashboard now updates cached matplotlib line artists with `set_data()` and clears only event-line overlays, rather than clearing and rebuilding all axes every frame.
