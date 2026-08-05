"""Visible Tk + one real Isaac worker validation for all four -60 degree knees."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import ImageGrab

from height_replay_ui import build_parser, normalize_motion_args
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi


class KneeLimitRealE2E:
    def __init__(self, output_dir: Path, timeout_s: float):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        args = build_parser().parse_args(
            [
                "--ui",
                "--height-cm",
                "5",
                "--worker-launch-mode",
                "explicit-python",
                "--worker-python-exe",
                r"C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe",
                "--sim-startup-timeout-s",
                str(timeout_s),
                "--worker-smoke-negative-knee-test",
                "--no-telemetry",
                "--no-save-scene",
            ]
        )
        normalize_motion_args(args)
        self.controller = HeightReplayController(args)
        self.ui = RealRobotStyleHeightReplayUi(self.controller)
        self.ui.root.geometry("1700x980+0+0")
        self.ui.root.deiconify()
        self.ui.root.lift()
        self.timeout_s = timeout_s
        self.started = time.monotonic()
        self.result: dict[str, Any] = {"success": False, "started_at": time.time()}

    def run(self) -> int:
        self.ui.root.after(50, self._start)
        self.ui.root.after(200, self._tick)
        self.ui.run()
        return 0 if self.result.get("success") else 1

    def _start(self) -> None:
        try:
            self.controller.start_sim_if_needed()
        except Exception as exc:
            self._fail(exc)

    def _tick(self) -> None:
        if self.ui._closing:
            return
        try:
            if time.monotonic() - self.started > self.timeout_s:
                raise TimeoutError("knee validation timed out")
            status = dict(self.controller.latest_sim_status)
            phase = str(status.get("phase", "") or "")
            if phase in {"preflight_failed", "launch_plan_failed", "process_spawn_failed", "runtime_failed"}:
                raise RuntimeError(str(status.get("error", status.get("traceback", phase))))
            smoke = status.get("negative_knee_smoke_result")
            if not isinstance(smoke, dict):
                self.ui.root.after(200, self._tick)
                return
            self.result.update(
                {
                    "success": bool(smoke.get("ok")),
                    "finished_at": time.time(),
                    "isaac_pid": int(status.get("pid", 0) or 0),
                    "physics_dt": float(status.get("physics_dt", 0.0) or 0.0),
                    "phase": phase,
                    "apply_physx_joint_limits": bool(getattr(self.controller.args, "apply_physx_joint_limits", False)),
                    "articulation_joint_names": list(status.get("robot_joint_names", []) or []),
                    "negative_knee_smoke_result": smoke,
                    "joint_diagnostics": status.get("joint_diagnostics", []),
                    "target_joint_state": status.get("target_joint_state"),
                    "actual_joint_state": status.get("actual_joint_state"),
                    "controller_status_log_tail": self.controller.status_log[-100:],
                }
            )
            self.controller.detail_text = json.dumps(smoke, indent=2, ensure_ascii=False)
            self.ui.open_right_tab("Sim State")
            self.ui._refresh(force=True, sim_state=True)
            self.ui.root.update()
            ImageGrab.grab(window=self.ui.root.winfo_id()).save(self.output_dir / "four_knees_minus_60.png")
            self._write()
            self.ui.root.after(250, lambda: self.ui._window_close(force=True))
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        if self.ui._closing:
            return
        self.result.update(
            {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "latest_sim_status": self.controller.latest_sim_status,
                "controller_status_log_tail": self.controller.status_log[-100:],
            }
        )
        self._write()
        try:
            ImageGrab.grab(window=self.ui.root.winfo_id()).save(self.output_dir / "knee_failure.png")
        except Exception:
            pass
        self.ui.root.after(250, lambda: self.ui._window_close(force=True))

    def _write(self) -> None:
        (self.output_dir / "knee_real_result.json").write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    return KneeLimitRealE2E(args.output, args.timeout_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
