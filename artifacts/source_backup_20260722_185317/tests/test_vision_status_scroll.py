from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_ui_controller import RealRobotStyleHeightReplayUi, set_text_preserving_view  # noqa: E402


class FakeText:
    def __init__(self) -> None:
        self.text = ""
        self.view = (0.35, 0.55)
        self.state = "disabled"
        self.scrolls = 0
        self.insert_mark = "3.0"

    def yview(self) -> tuple[float, float]:
        return self.view

    def yview_moveto(self, fraction: float) -> None:
        start = max(0.0, min(1.0, float(fraction)))
        span = self.view[1] - self.view[0]
        self.view = (start, min(1.0, start + span))

    def yview_scroll(self, units: int, _what: str) -> None:
        self.scrolls += int(units)

    def cget(self, name: str) -> str:
        return self.state if name == "state" else ""

    def configure(self, **kwargs: object) -> None:
        if "state" in kwargs:
            self.state = str(kwargs["state"])

    def delete(self, _start: str, _end: str) -> None:
        self.text = ""

    def insert(self, _index: str, text: str) -> None:  # type: ignore[no-redef]
        self.text = str(text)

    def index(self, _mark: str) -> str:
        return self.insert_mark

    def mark_set(self, _mark: str, index: str) -> None:
        self.insert_mark = str(index)


class FakeTree:
    def __init__(self) -> None:
        self.items: dict[str, tuple[str, str]] = {}
        self.deleted: list[str] = []
        self.view = (0.4, 0.7)
        self.selected: tuple[str, ...] = ()

    def yview(self) -> tuple[float, float]:
        return self.view

    def yview_moveto(self, fraction: float) -> None:
        self.view = (float(fraction), float(fraction) + 0.3)

    def yview_scroll(self, _units: int, _what: str) -> None:
        pass

    def get_children(self) -> list[str]:
        return list(self.items)

    def delete(self, iid: str) -> None:
        self.deleted.append(iid)
        self.items.pop(iid, None)

    def item(self, iid: str, option: str | None = None, **kwargs: object) -> tuple[str, str] | None:
        if "values" in kwargs:
            self.items[iid] = tuple(kwargs["values"])  # type: ignore[arg-type]
        if option == "values":
            return self.items[iid]
        return self.items.get(iid)

    def insert(self, _parent: str, _where: str, *, iid: str, values: tuple[str, str]) -> None:
        self.items[iid] = tuple(values)

    def selection(self) -> tuple[str, ...]:
        return self.selected

    def selection_set(self, items: list[str]) -> None:
        self.selected = tuple(items)


class VisionStatusScrollTest(unittest.TestCase):
    def test_text_update_preserves_yview_in_middle(self) -> None:
        widget = FakeText()

        set_text_preserving_view(widget, "Frame Age: 0.100s")

        self.assertEqual(widget.view[0], 0.35)
        self.assertEqual(widget.text, "Frame Age: 0.100s")
        self.assertEqual(widget.state, "disabled")

    def test_recent_user_scroll_is_not_overridden_by_follow_bottom(self) -> None:
        widget = FakeText()
        widget.view = (0.7, 1.0)
        widget._last_user_scroll_at = time.monotonic()

        set_text_preserving_view(widget, "Confidence: 85%", follow_bottom=True)

        self.assertEqual(widget.view[0], 0.7)

    def test_bottom_follow_stays_at_bottom(self) -> None:
        widget = FakeText()
        widget.view = (0.8, 1.0)

        set_text_preserving_view(widget, "new", follow_bottom=True)

        self.assertEqual(widget.view[0], 1.0)

    def test_treeview_updates_rows_without_deleting_all_or_changing_yview(self) -> None:
        ui = object.__new__(RealRobotStyleHeightReplayUi)
        tree = FakeTree()
        ui.vision_status_tree = tree

        snapshot = {"sim": {"runtime_ready": True}, "vision": {"camera_ready": True, "confidence": 0.85, "robot_ground": {}}}
        RealRobotStyleHeightReplayUi._refresh_vision_status_tree(ui, snapshot)
        before_count = len(tree.items)
        snapshot["vision"]["confidence"] = 0.86
        RealRobotStyleHeightReplayUi._refresh_vision_status_tree(ui, snapshot)

        self.assertEqual(len(tree.items), before_count)
        self.assertEqual(tree.deleted, [])
        self.assertEqual(tree.view[0], 0.4)
        self.assertIn("confidence", tree.items)
        self.assertIn("86.00%", tree.items["confidence"][1])

    def test_inner_wheel_event_returns_break(self) -> None:
        ui = object.__new__(RealRobotStyleHeightReplayUi)
        widget = FakeText()
        event = SimpleNamespace(widget=widget, delta=-120)

        result = RealRobotStyleHeightReplayUi._on_text_mousewheel(ui, event)

        self.assertEqual(result, "break")
        self.assertEqual(widget.scrolls, 1)


if __name__ == "__main__":
    unittest.main()
