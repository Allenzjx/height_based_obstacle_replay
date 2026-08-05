from __future__ import annotations

import unittest

from telemetry.live_buffer import LiveTelemetryBuffer


class LiveTelemetryBackpressureTest(unittest.TestCase):
    def test_live_frames_drop_oldest_without_blocking_events(self) -> None:
        buffer = LiveTelemetryBuffer(max_frames=2)

        buffer.push_frame({"live_frame_id": 1})
        buffer.push_frame({"live_frame_id": 2})
        buffer.push_frame({"live_frame_id": 3})
        buffer.push_event({"time_s": 0.1, "event_type": "fresh_frame", "severity": "info"})

        self.assertEqual(buffer.dropped_frames, 1)
        self.assertEqual([frame["live_frame_id"] for frame in buffer.frames], [2, 3])
        self.assertEqual(buffer.latest_frame()["live_frame_id"], 3)
        self.assertEqual(buffer.recent_events()[0]["event_type"], "fresh_frame")
        self.assertEqual(buffer.status()["queued_live_frames"], 2)


if __name__ == "__main__":
    unittest.main()
