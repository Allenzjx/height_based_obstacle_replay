"""Timed playback for height-indexed accepted steps."""

from __future__ import annotations

import copy
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from command_model import CommandMessage
from sequence_model import event_playback_commands, load_steps_jsonl, motion_only_events, normalize_events, normalize_step


@dataclass
class PlaybackEvent:
    time_s: float
    command: str
    source_step: int | None = None


@dataclass
class PlaybackPlan:
    path: Path | None
    events: list[PlaybackEvent]
    final_time_s: float
    label: str = ""


def plan_from_steps(
    steps: list[dict[str, Any]],
    *,
    profile: str = "fast",
    speed: float = 1.0,
    trailing_pad: float = 0.05,
    max_idle_gap: float | None = None,
    preserve_wheel_distance: bool = True,
    max_wheel_speed: float | None = None,
    label: str = "accepted steps",
) -> PlaybackPlan:
    normalized_profile = "raw" if str(profile).lower() == "raw" else "fast"
    playback_speed = max(0.1, min(5.0, float(speed)))
    cursor = 0.0
    events: list[PlaybackEvent] = []
    for step in steps:
        normalized = normalize_step(step)
        step_index = int(normalized.get("index", 0))
        step_events = (
            normalize_events(normalized.get("events", []))
            if normalized_profile == "raw"
            else motion_only_events(normalized.get("events", []))
        )
        last_event_time = 0.0
        previous_event_time = 0.0
        compressed_time = 0.0
        for event in step_events:
            raw_time = max(0.0, float(event.get("time", 0.0)))
            if max_idle_gap is None:
                event_time = raw_time
            else:
                gap = max(0.0, raw_time - previous_event_time)
                compressed_time += min(gap, max_idle_gap)
                event_time = compressed_time
                previous_event_time = raw_time
            last_event_time = max(last_event_time, event_time)
            scaled_time = cursor + event_time / playback_speed
            commands = event_playback_commands(event) or [str(event.get("command", ""))]
            for command in commands:
                command = command.strip()
                if command:
                    if normalized_profile == "fast" and preserve_wheel_distance and abs(playback_speed - 1.0) > 1.0e-9:
                        command = scale_wheel_command(
                            command,
                            playback_speed,
                            max_wheel_speed=max_wheel_speed,
                        )
                    events.append(PlaybackEvent(scaled_time, command, source_step=step_index))
        if normalized_profile == "raw":
            step_duration = max(float(normalized.get("duration", 0.0)), last_event_time)
        else:
            step_duration = max(last_event_time, 0.0) + max(0.0, float(trailing_pad))
        cursor += step_duration / playback_speed
    events.sort(key=lambda event: event.time_s)
    return PlaybackPlan(path=None, events=events, final_time_s=cursor, label=label)


def plan_from_jsonl(path: str | Path, **kwargs: Any) -> PlaybackPlan:
    source = Path(path)
    plan = plan_from_steps(load_steps_jsonl(source), label=str(source), **kwargs)
    plan.path = source
    return plan


class PlaybackManager:
    def __init__(self, controller: Any):
        self.controller = controller
        self.active = False
        self.paused = False
        self.plan: PlaybackPlan | None = None
        self.index = 0
        self.start_time = 0.0
        self.scheduled_start_at = 0.0
        self.pause_started = 0.0
        self.started_at = 0.0
        self.events_sent = 0
        self.last_event_command = ""
        self.last_event_time = 0.0
        self.completed_at = 0.0
        self.stop_reason = ""
        self.last_error = ""
        self.last_info = ""
        self.profile = "fast"
        self.speed = 1.0
        self.trailing_pad = 0.05
        self.max_idle_gap: float | None = None
        self.preserve_wheel_distance = True
        self.max_events_per_update = 50

    def start_steps(self, steps: list[dict[str, Any]], *, label: str = "accepted steps", start_delay_s: float = 0.0) -> bool:
        try:
            plan = plan_from_steps(
                steps,
                profile=self.profile,
                speed=self.speed,
                trailing_pad=self.trailing_pad,
                max_idle_gap=self.max_idle_gap,
                preserve_wheel_distance=self.preserve_wheel_distance,
                max_wheel_speed=getattr(self.controller, "max_wheel_speed", None),
                label=label,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return False
        return self.start_plan(plan, start_delay_s=start_delay_s)

    def start_plan(self, plan: PlaybackPlan, *, start_delay_s: float = 0.0) -> bool:
        label = plan.label or "plan"
        if not plan.events:
            self.plan = copy.deepcopy(plan)
            self.index = 0
            self.active = False
            self.paused = False
            self.scheduled_start_at = 0.0
            self.start_time = 0.0
            self.started_at = 0.0
            self.events_sent = 0
            self.last_event_command = ""
            self.last_event_time = 0.0
            self.completed_at = 0.0
            self.stop_reason = "empty_plan"
            self.last_error = f"Selected step/plan {label} has no motion events to play."
            self.last_info = self.last_error
            return False
        self.plan = copy.deepcopy(plan)
        self.index = 0
        self.active = True
        self.paused = False
        now = time.monotonic()
        delay_s = max(0.0, float(start_delay_s))
        self.scheduled_start_at = now + delay_s if delay_s > 0.0 else 0.0
        self.start_time = self.scheduled_start_at or now
        self.started_at = 0.0 if self.scheduled_start_at else time.time()
        self.pause_started = 0.0
        self.events_sent = 0
        self.last_event_command = ""
        self.last_event_time = 0.0
        self.completed_at = 0.0
        self.stop_reason = ""
        self.last_error = ""
        if self.scheduled_start_at:
            self.last_info = f"playback scheduled; starts in {delay_s:.2f}s: {label} ({len(plan.events)} events)"
        else:
            self.last_info = f"playing {label} ({len(plan.events)} events)"
        return True

    def update(self) -> None:
        if not self.active or self.paused or self.plan is None:
            return
        now = time.monotonic()
        if self.scheduled_start_at > 0.0:
            remaining = self.scheduled_start_at - now
            if remaining > 0.0:
                self.last_info = f"playback scheduled; starts in {remaining:.2f}s"
                return
            self.started_at = time.time()
            self.scheduled_start_at = 0.0
            self.last_info = f"playing {self.plan.label or 'plan'} ({len(self.plan.events)} events)"
        elapsed = now - self.start_time
        sent_this_update = 0
        while self.index < len(self.plan.events):
            if sent_this_update >= int(self.max_events_per_update):
                self.last_info = f"playback batching at event {self.index}/{len(self.plan.events)}"
                break
            event = self.plan.events[self.index]
            if event.time_s > elapsed + 1.0e-4:
                break
            self.controller.handle_command(
                CommandMessage(
                    text=event.command,
                    source="playback",
                    log_history=False,
                    quiet=True,
                    playback_label=self.plan.label,
                    playback_event_index=self.index,
                    playback_event_count=len(self.plan.events),
                    playback_final_time_s=float(self.plan.final_time_s),
                    source_step=event.source_step,
                )
            )
            self.index += 1
            sent_this_update += 1
            self.events_sent += 1
            self.last_event_command = event.command
            self.last_event_time = time.time()
        if self.index >= len(self.plan.events) and elapsed >= self.plan.final_time_s:
            self.completed_at = time.time()
            self.stop(silent=True, reason="complete")
            self.last_info = "playback complete"

    def stop(self, *, silent: bool = False, stop_wheels: bool = True, reason: str = "stopped") -> None:
        was_active = self.active
        was_scheduled = self.scheduled_start_at > 0.0
        self.active = False
        self.paused = False
        self.index = 0
        self.scheduled_start_at = 0.0
        if stop_wheels and hasattr(self.controller, "stop_wheels"):
            self.controller.stop_wheels()
        if was_active or was_scheduled or not silent:
            self.stop_reason = reason
            self.last_info = "playback stopped"

    def pause(self) -> None:
        if not self.active or self.paused:
            return
        self.paused = True
        self.pause_started = time.monotonic()
        self.last_info = "playback paused"

    def resume(self) -> None:
        if not self.active or not self.paused:
            return
        paused_for = time.monotonic() - self.pause_started
        self.start_time += paused_for
        if self.scheduled_start_at:
            self.scheduled_start_at += paused_for
        self.pause_started = 0.0
        self.paused = False
        self.last_info = "playback resumed"

    def set_profile(self, profile: str) -> None:
        normalized = profile.lower()
        if normalized in {"motion-only", "motion_only", "fast"}:
            self.profile = "fast"
        elif normalized == "raw":
            self.profile = "raw"
        else:
            raise ValueError("Playback profile must be fast or raw.")

    def set_speed(self, speed: float) -> None:
        value = float(speed)
        if value < 0.1 or value > 5.0:
            raise ValueError("Playback speed must be in the range 0.1..5.0.")
        self.speed = value

    def set_trailing_pad(self, value: float) -> None:
        self.trailing_pad = max(0.0, float(value))

    def set_max_idle_gap(self, value: float | None) -> None:
        self.max_idle_gap = None if value is None else max(0.0, float(value))

    def analyze_steps(self, steps: list[dict[str, Any]]) -> str:
        plan = plan_from_steps(
            steps,
            profile=self.profile,
            speed=self.speed,
            trailing_pad=self.trailing_pad,
            max_idle_gap=self.max_idle_gap,
            preserve_wheel_distance=self.preserve_wheel_distance,
            max_wheel_speed=getattr(self.controller, "max_wheel_speed", None),
        )
        return (
            f"Playback timing: profile={self.profile} speed={self.speed:.2f} "
            f"events={len(plan.events)} duration={plan.final_time_s:.3f}s"
        )

    def status_dict(self) -> dict[str, Any]:
        now = time.monotonic()
        starts_in_s = max(0.0, self.scheduled_start_at - now) if self.scheduled_start_at else 0.0
        return {
            "active": self.active,
            "paused": self.paused,
            "scheduled": bool(self.active and self.scheduled_start_at > 0.0),
            "scheduled_start_at": self.scheduled_start_at,
            "starts_in_s": starts_in_s,
            "index": self.index,
            "count": len(self.plan.events) if self.plan else 0,
            "label": self.plan.label if self.plan else "",
            "profile": self.profile,
            "speed": self.speed,
            "trailing_pad": self.trailing_pad,
            "max_idle_gap": self.max_idle_gap,
            "preserve_wheel_distance": self.preserve_wheel_distance,
            "started_at": self.started_at,
            "events_sent": self.events_sent,
            "last_event_command": self.last_event_command,
            "last_event_time": self.last_event_time,
            "completed_at": self.completed_at,
            "stop_reason": self.stop_reason,
            "last_error": self.last_error,
            "last_info": self.last_info,
        }


def command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return str(command).split()


def scale_wheel_command(
    command: str,
    speed_scale: float,
    *,
    max_wheel_speed: float | None,
) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if not tokens or tokens[0].lower() not in {"wheel", "wheels", "speed"}:
        return command
    scaled = list(tokens)

    def scale_token(index: int) -> None:
        value = float(scaled[index])
        requested = value * float(speed_scale)
        if max_wheel_speed is not None and abs(requested) > float(max_wheel_speed):
            requested = float(max_wheel_speed) if requested > 0.0 else -float(max_wheel_speed)
        scaled[index] = f"{requested:.6g}"

    verb = scaled[0].lower()
    args = scaled[1:]
    try:
        if verb in {"wheels", "speed"} and len(args) == 2:
            scale_token(1)
            scale_token(2)
        elif verb == "wheel" and len(args) == 2 and _is_float(args[0]):
            scale_token(1)
            scale_token(2)
        elif verb == "wheel" and len(args) == 2 and args[0].lower() != "stop":
            scale_token(2)
    except (TypeError, ValueError):
        return command
    return " ".join(shlex.quote(token) for token in scaled)


def _is_float(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True
