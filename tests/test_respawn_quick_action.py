from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for path in (MODULE_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from controller_test_utils import FakeTransport, make_args, motion_step  # noqa: E402
from operation_coordinator import OperationState  # noqa: E402
from sim_ui_controller import HeightReplayController, MODE_E_STOP, RealRobotStyleHeightReplayUi  # noqa: E402


class RespawnQuickActionTest(unittest.TestCase):
    def test_quick_commands_places_respawn_to_the_right_of_estop(self) -> None:
        source = inspect.getsource(RealRobotStyleHeightReplayUi._build_quick_commands)
        self.assertLess(source.index("E-stop"), source.index("↻ Respawn"))
        self.assertIn('"respawn"', source)
        self.assertIn("Respawn robot to its initial simulation pose. This does not clear E-stop.", source)

    def test_manual_respawn_uses_existing_transport_and_preserves_height_manager_and_estop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.sim_ready = True
            controller.transport = FakeTransport()  # type: ignore[assignment]
            controller.manager.adopt_steps([motion_step(height_cm=10)], dirty=True)
            controller.current_height_cm = 10
            controller.loaded_sim_height_cm = 5
            controller.playback.active = True
            controller.playback.scheduled_start_at = 123.0
            controller.mode = MODE_E_STOP

            ok = controller.respawn_robot(source="manual")

            self.assertTrue(ok)
            actions = [name for name, _payload in controller.transport.calls]
            self.assertIn("respawn", actions)
            self.assertIn("request_state", actions)
            self.assertGreaterEqual(actions.count("stop_wheels"), 1)
            self.assertFalse(controller.playback.active)
            self.assertEqual(controller.playback.scheduled_start_at, 0.0)
            self.assertEqual(controller.mode, MODE_E_STOP)
            self.assertEqual(controller.current_height_cm, 10)
            self.assertEqual(controller.loaded_sim_height_cm, 5)
            self.assertEqual(controller.manager.count, 1)
            self.assertTrue(controller.manager.dirty)

    def test_respawn_command_posts_existing_respawn_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=True))
            controller.transport = FakeTransport()  # type: ignore[assignment]

            controller.handle_command("respawn")

            self.assertIn(("respawn", {}), controller.transport.calls)

    def test_non_playback_respawn_source_releases_operation_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.sim_ready = True
            controller.transport = FakeTransport()  # type: ignore[assignment]

            self.assertTrue(controller.respawn_robot(source="determinism_2"))

            self.assertIs(controller.operation.state, OperationState.IDLE)
            self.assertIn(("respawn", {}), controller.transport.calls)


if __name__ == "__main__":
    unittest.main()
