"""Measure Tk UI tab-switch latency without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Height Replay Tk tab-switch latency.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=str, default=str(ROOT / "artifacts" / "ui_performance_after.json"))
    args = parser.parse_args(argv)
    result = measure_tab_switches(max(1, int(args.iterations)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str))
    return 0


def measure_tab_switches(iterations: int) -> dict[str, Any]:
    ui_args = build_parser().parse_args(["--ui", "--no-sim", "--sim-launch-mode", "disabled"])
    normalize_motion_args(ui_args)
    controller = HeightReplayController(ui_args)
    ui = RealRobotStyleHeightReplayUi(controller, smoke_test_ms=0)
    try:
        ui.root.update_idletasks()
        ui.root.update()
        notebook = getattr(ui, "right_notebook", None)
        tab_ids = list(notebook.tabs()) if notebook is not None else []
        samples_ms: list[float] = []
        for index in range(iterations):
            tab_id = tab_ids[index % len(tab_ids)] if tab_ids else None
            started = time.perf_counter()
            if tab_id is not None:
                notebook.select(tab_id)
            ui.root.update_idletasks()
            ui.root.update()
            samples_ms.append((time.perf_counter() - started) * 1000.0)
        return {
            "mode": "no_sim_tk_tab_switch",
            "iterations": int(iterations),
            "tab_count": len(tab_ids),
            "count": len(samples_ms),
            "p50_ms": _percentile(samples_ms, 50.0),
            "p95_ms": _percentile(samples_ms, 95.0),
            "max_ms": max(samples_ms) if samples_ms else 0.0,
            "mean_ms": statistics.fmean(samples_ms) if samples_ms else 0.0,
        }
    finally:
        try:
            ui.root.destroy()
        except Exception:
            pass
        try:
            controller.shutdown()
        except Exception:
            pass


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


if __name__ == "__main__":
    raise SystemExit(main())
