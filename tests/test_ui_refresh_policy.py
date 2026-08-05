from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi  # noqa: E402


class UiRefreshPolicyTest(unittest.TestCase):
    def test_controller_update_does_not_step_simulation(self) -> None:
        source = inspect.getsource(HeightReplayController.update)
        self.assertNotIn("adapter.step", source)
        self.assertNotIn(".sim.step", source)

    def test_tk_poll_uses_configured_refresh_interval(self) -> None:
        source = inspect.getsource(RealRobotStyleHeightReplayUi._poll)
        self.assertIn("force=False", source)
        self.assertIn("self.ui_refresh_ms", source)
        self.assertNotIn("after(50", source)

    def test_full_json_snapshot_is_gated_by_explicit_sim_state_refresh(self) -> None:
        source = inspect.getsource(RealRobotStyleHeightReplayUi._refresh)
        self.assertIn("if sim_state or auto_sim_state", source)
        self.assertIn("sim_state_json_on_demand", source)
        self.assertIn("json.dumps(snapshot", source)

    def test_refresh_cli_defaults_are_not_50ms_full_refresh(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-sim"])
        normalize_motion_args(args)
        self.assertGreaterEqual(args.ui_refresh_ms, 100)
        self.assertGreaterEqual(args.sim_status_refresh_ms, 100)
        self.assertLessEqual(args.sim_status_refresh_ms, 200)
        self.assertGreaterEqual(args.full_refresh_ms, 1000)


if __name__ == "__main__":
    unittest.main()
