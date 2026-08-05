from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from command_model import SERVO_JOINT_NAMES
from sim_robot_adapter import (
    PHYSX_SAFE_LIMIT_MAX_RAD,
    PHYSX_SAFE_LIMIT_MIN_RAD,
    SimRobotAdapter,
    physx_write_rad_limits_from_record,
    safe_rad_limits_for_actual_deg,
)


class PhysxJointLimitUnitsTest(unittest.TestCase):
    def test_default_safe_limit_records_do_not_write_to_physx(self) -> None:
        adapter = SimRobotAdapter.__new__(SimRobotAdapter)
        adapter.servo_name_to_id = {name: index for index, name in enumerate(SERVO_JOINT_NAMES)}
        adapter.standing_pose_deg = {name: 0.0 for name in SERVO_JOINT_NAMES}
        adapter.safe_joint_limit_records = {}
        adapter.device = "cpu"
        adapter.robot = type("FakeRobot", (), {"num_instances": 1, "write_calls": 0})()

        def fail_write(*_args: object, **_kwargs: object) -> None:
            adapter.robot.write_calls += 1
            raise AssertionError("PhysX write should not be called")

        adapter.robot.write_joint_position_limit_to_sim = fail_write
        adapter.update_safe_command_space_joint_limit_records(write_to_sim=False)
        self.assertEqual(adapter.robot.write_calls, 0)
        self.assertTrue(adapter.safe_joint_limit_records)
        self.assertTrue(all(record["write_skipped_reason"] for record in adapter.safe_joint_limit_records.values()))

    def test_explicit_physx_write_limits_stay_inside_two_pi(self) -> None:
        record = safe_rad_limits_for_actual_deg(-1000.0, 1000.0)
        low, high, adjusted = physx_write_rad_limits_from_record(record)
        self.assertTrue(adjusted)
        self.assertGreater(low, PHYSX_SAFE_LIMIT_MIN_RAD)
        self.assertLess(high, PHYSX_SAFE_LIMIT_MAX_RAD)
        self.assertLess(high, 2.0 * math.pi)


if __name__ == "__main__":
    unittest.main()
