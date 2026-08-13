from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from isaac_launch_preflight import IsaacInterpreterReport  # noqa: E402
from sim_process_client import (  # noqa: E402
    build_child_environment,
    build_worker_command,
    build_worker_launch_plan,
    resolve_requested_launch_mode,
)
from sim_ui_controller import HeightReplayController  # noqa: E402


def make_args(extra: list[str] | None = None):
    args = build_parser().parse_args(["--ui", *(extra or [])])
    normalize_motion_args(args)
    return args


def good_report() -> IsaacInterpreterReport:
    return IsaacInterpreterReport(
        executable=sys.executable,
        python_version="3.11.9",
        isaacsim_importable=True,
        isaaclab_importable=True,
        app_launcher_importable=True,
        isaacsim_version="5.0.0",
        isaaclab_version="2.2.0",
        compatible_python=True,
        environment={},
    )


class WorkerLaunchPlanTest(unittest.TestCase):
    def test_empty_camera_parent_is_omitted_from_argv(self) -> None:
        command = build_worker_command(make_args(["--camera-parent-prim", ""]), host="127.0.0.1", port=45678)
        self.assertNotIn("--camera-parent-prim", command)
        self.assertNotIn("", command)

    def test_nonempty_camera_parent_is_passed(self) -> None:
        command = build_worker_command(
            make_args(["--camera-parent-prim", "/World/WLRRobot/base_link"]),
            host="127.0.0.1",
            port=45678,
        )
        self.assertIn("--camera-parent-prim", command)
        self.assertIn("/World/WLRRobot/base_link", command)

    def test_onboard_camera_sets_enable_cameras(self) -> None:
        command = build_worker_command(make_args(["--onboard-camera"]), host="127.0.0.1", port=45678)
        self.assertIn("--enable_cameras", command)

    def test_gui_mode_does_not_pass_headless(self) -> None:
        command = build_worker_command(make_args([]), host="127.0.0.1", port=45678)
        self.assertNotIn("--headless", command)
        env, status = build_child_environment(make_args([]))
        self.assertEqual(env["HEADLESS"], "0")
        self.assertEqual(env["LIVESTREAM"], "0")
        self.assertFalse(status["effective_headless"])

    def test_explicit_headless_propagates(self) -> None:
        args = make_args(["--headless"])
        command = build_worker_command(args, host="127.0.0.1", port=45678)
        self.assertIn("--headless", command)
        env, status = build_child_environment(args)
        self.assertEqual(env["HEADLESS"], "1")
        self.assertTrue(status["effective_headless"])

    def test_current_python_preflight_success_selects_direct_python(self) -> None:
        args = make_args(["--worker-launch-mode", "current-python"])
        with patch("sim_process_client.run_interpreter_preflight", return_value=good_report()):
            plan = build_worker_launch_plan(args, host="127.0.0.1", port=45678)
        self.assertTrue(plan.preflight_ok)
        self.assertEqual(plan.resolved_launch_mode, "current-python")
        self.assertIn("--worker-config-file", plan.argv)
        self.assertTrue(Path(plan.config_path).exists())

    def test_old_worker_python_mode_maps_to_new_launch_mode(self) -> None:
        args = make_args(["--worker-python-mode", "isaaclab-bat"])
        self.assertEqual(args.worker_launch_mode, "isaaclab-bat")
        self.assertEqual(resolve_requested_launch_mode(args), "isaaclab-bat")

    def test_no_sim_does_not_run_preflight_or_start_worker(self) -> None:
        args = make_args(["--no-sim"])
        controller = HeightReplayController(args)
        with patch("sim_ui_controller.run_launch_preflight_for_args") as preflight:
            controller.start_sim_if_needed()
        preflight.assert_not_called()
        self.assertTrue(controller.no_sim)


if __name__ == "__main__":
    unittest.main()
