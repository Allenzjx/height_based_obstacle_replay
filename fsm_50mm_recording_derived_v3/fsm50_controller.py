"""Event-gated runtime controller for the recording-derived 50 mm FSM.

The controller is intentionally Isaac-independent.  It owns lifecycle,
guard, dwell, retry, command-executor, telemetry, and fail-closed semantics;
an Isaac runner only has to provide validated :class:`FSM50Observation`
samples and a duck-typed robot adapter.

No transition is driven by a command endpoint or a recording duration.  The
only happy-path transition source is a live physical guard followed by the
state's minimum/stable/consecutive-sample contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .com_transfer_primitives import GuardDecision, Leg, LegIKCandidate
from .fsm50_executor import ExecutionResult, FSM50CommandExecutor
from .fsm50_guard_registry import FSM50GuardRegistry, GuardEvaluationContext
from .fsm50_observation import FSM50Observation
from .fsm50_state_model import FSM50State, FSM50StateTable, TransitionDwellTracker


class ControllerStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    SAFE_STOP = "SAFE_STOP"


_TRUSTED_RESTORE_METHODS = frozenset(
    {"TRUSTED_SIM_STATE_RESTORE", "VERIFIED_PREFIX_REPLAY"}
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _decision_mapping(decision: GuardDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "satisfied": bool(decision.satisfied),
        "abort": bool(decision.abort),
        "reason": str(decision.reason),
        "metrics": _plain(dict(decision.metrics)),
        "unmet": list(decision.unmet),
    }


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _leg_values(values: Any) -> list[str]:
    return [str(getattr(value, "value", value)) for value in (values or ())]


def _primitive_stage(state: FSM50State) -> dict[str, str]:
    """Expose the command-side primitive stage represented by a state.

    The actual actuator targets still come from the recording-derived command
    profile and are executed by :class:`FSM50CommandExecutor`.  This mapping is
    evidence, not a substitute for the live guard.
    """

    name = f"{state.state_id} {state.state_name}".upper()
    result = {"family": "NONE", "stage": "NONE"}
    if "REACTION_PRELOAD" in name or "LEVER_PRELOAD" in name:
        return {"family": "IMPULSE_REACTION_TRANSFER", "stage": "PRELOAD"}
    if "REACTION_PULSE" in name or "REACTION_TRANSFER" in name:
        return {"family": "IMPULSE_REACTION_TRANSFER", "stage": "PUSH"}
    if "REACTION_RELEASE" in name:
        return {"family": "IMPULSE_REACTION_TRANSFER", "stage": "RELEASE"}
    if "COAST" in name:
        return {"family": "IMPULSE_REACTION_TRANSFER", "stage": "COAST"}
    if "SETTLE_COM" in name:
        return {"family": "IMPULSE_REACTION_TRANSFER", "stage": "SETTLE"}
    if "VERIFY_" in name and "UNLOAD_READY" in name:
        return {"family": "IMPULSE_REACTION_TRANSFER", "stage": "VERIFY"}
    if "PREP_" in name and state.com_transfer_method.value == "ANCHORED_SUPPORT_ANGLE_TRANSFER":
        return {"family": "ANCHORED_SUPPORT_ANGLE_TRANSFER", "stage": "SUPPORT_ANCHOR"}
    if "SUPPORT_GEOMETRY" in name or "SUPPORT_ANGLE" in name:
        return {"family": "ANCHORED_SUPPORT_ANGLE_TRANSFER", "stage": "SUPPORT_ANGLE_RAMP"}
    if state.com_transfer_method.value == "ANCHORED_SUPPORT_ANGLE_TRANSFER":
        return {"family": "ANCHORED_SUPPORT_ANGLE_TRANSFER", "stage": "SUPPORT_HOLD"}
    return result


@dataclass(frozen=True)
class ControllerTickResult:
    status: ControllerStatus
    state_id: str
    previous_state_id: str
    time_s: float
    transitioned: bool
    transition_reason: str
    retry_index: int
    entry_guard: Mapping[str, Any] | None
    progress_guard: Mapping[str, Any] | None
    exit_guard: Mapping[str, Any] | None
    abort_guard: Mapping[str, Any] | None
    dwell: Mapping[str, Any] | None
    execution: Mapping[str, Any] | None
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "entry_guard",
            "progress_guard",
            "exit_guard",
            "abort_guard",
            "dwell",
            "execution",
            "evidence",
        ):
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))

    @property
    def succeeded(self) -> bool:
        return self.status == ControllerStatus.SUCCEEDED

    @property
    def fail_closed(self) -> bool:
        return self.status == ControllerStatus.SAFE_STOP

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "state_id": self.state_id,
            "previous_state_id": self.previous_state_id,
            "time_s": self.time_s,
            "transitioned": self.transitioned,
            "transition_reason": self.transition_reason,
            "retry_index": self.retry_index,
            "entry_guard": _plain(self.entry_guard),
            "progress_guard": _plain(self.progress_guard),
            "exit_guard": _plain(self.exit_guard),
            "abort_guard": _plain(self.abort_guard),
            "dwell": _plain(self.dwell),
            "execution": _plain(self.execution),
            "evidence": _plain(self.evidence),
        }


class FSM50Controller:
    """Run the ordered A0 -> F5 graph against live physics observations."""

    def __init__(
        self,
        adapter: Any,
        state_table: FSM50StateTable | str | Path,
        *,
        guard_registry: FSM50GuardRegistry | None = None,
        executor: FSM50CommandExecutor | None = None,
        start_state_id: str | None = None,
        restore_provenance: Mapping[str, Any] | None = None,
        telemetry_sink: Callable[[Mapping[str, Any]], None] | None = None,
        require_physically_verified: bool = False,
    ) -> None:
        self.adapter = adapter
        self.state_table = (
            FSM50StateTable.load(state_table)
            if isinstance(state_table, (str, Path))
            else state_table
        )
        if not isinstance(self.state_table, FSM50StateTable):
            raise TypeError("state_table must be FSM50StateTable or a path")
        self.registry = guard_registry or FSM50GuardRegistry()
        self.registry.validate_states(self.state_table.states)
        self.thresholds = dict(self.state_table.metadata.get("thresholds", {}) or {})
        actuators = dict(self.state_table.metadata.get("actuators", {}) or {})
        self.executor = executor or FSM50CommandExecutor(
            adapter,
            servo_rate_deg_s=float(actuators.get("servo_max_rate_deg_s", 150.0)),
            wheel_acceleration_rad_s2=float(
                actuators.get("wheel_max_acceleration_rad_s2", 4.0)
            ),
        )
        self.telemetry_sink = telemetry_sink
        self.timeline: list[dict[str, Any]] = []
        self._happy_path = tuple(
            state for state in self.state_table.states if state.state_id != "SAFE_STOP"
        )
        self._safe_state = self.state_table.get("SAFE_STOP")
        self._validate_graph(require_physically_verified=require_physically_verified)
        self._index = {
            state.state_id: index for index, state in enumerate(self._happy_path)
        }
        self.start_state_id = start_state_id or self._happy_path[0].state_id
        if self.start_state_id not in self._index:
            raise ValueError(f"start_state_id is not on the happy path: {self.start_state_id}")
        self.restore_provenance = self._validate_restore_provenance(
            self.start_state_id, restore_provenance
        )
        self.status = ControllerStatus.NOT_STARTED
        self.state: FSM50State | None = None
        self.context: GuardEvaluationContext | None = None
        self.dwell_tracker: TransitionDwellTracker | None = None
        self.entry_ready = False
        self.entry_decision: GuardDecision | None = None
        self.last_time_s: float | None = None
        self.retry_counts: dict[str, int] = {}
        self.safe_stop_reason = ""
        self.safe_stop_ack: dict[str, Any] = {}
        self.safe_stop_confirmed = False
        self._event_index = 0

    @property
    def current_state_id(self) -> str:
        return "" if self.state is None else self.state.state_id

    @property
    def completed_state_ids(self) -> tuple[str, ...]:
        return tuple(
            event["state_id"]
            for event in self.timeline
            if event.get("event") == "on_exit" and event.get("outcome") == "transition"
        )

    @property
    def happy_path_state_ids(self) -> tuple[str, ...]:
        return tuple(state.state_id for state in self._happy_path)

    def _validate_graph(self, *, require_physically_verified: bool) -> None:
        identifiers = [state.state_id for state in self._happy_path]
        if not identifiers or identifiers[0] != "A0_RESET_AND_SETTLE":
            raise ValueError("happy path must begin at A0_RESET_AND_SETTLE")
        if identifiers[-1] != "F5_SUCCESS":
            raise ValueError("happy path must end at F5_SUCCESS")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("happy-path state identifiers must be unique")
        if require_physically_verified:
            pending = [
                state.state_id for state in self._happy_path if state.pending_physical_replay
            ]
            if pending:
                raise ValueError(
                    "full FSM is blocked by unverified state provenance: "
                    + ", ".join(pending)
                )

    @staticmethod
    def _validate_restore_provenance(
        start_state_id: str, provenance: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        if start_state_id == "A0_RESET_AND_SETTLE":
            return _freeze(dict(provenance or {}))
        values = dict(provenance or {})
        method = str(values.get("method", "")).upper()
        missing: list[str] = []
        if method not in _TRUSTED_RESTORE_METHODS:
            missing.append("method=TRUSTED_SIM_STATE_RESTORE|VERIFIED_PREFIX_REPLAY")
        if values.get("validated") is not True:
            missing.append("validated=true")
        if str(values.get("state_id", "")) != start_state_id:
            missing.append("matching state_id")
        if not str(values.get("source_run_directory", "")):
            missing.append("source_run_directory")
        if not str(values.get("source_sha256", "")):
            missing.append("source_sha256")
        if missing:
            raise ValueError(
                "non-A0 start requires trusted restore/prefix provenance: "
                + ", ".join(missing)
            )
        return _freeze(values)

    def _emit(self, event: str, **payload: Any) -> dict[str, Any]:
        self._event_index += 1
        row = {
            "event_index": self._event_index,
            "event": str(event),
            **{str(key): _plain(value) for key, value in payload.items()},
        }
        self.timeline.append(row)
        if self.telemetry_sink is not None:
            self.telemetry_sink(_freeze(row))
        return row

    def _observation_evidence(self, observation: FSM50Observation) -> dict[str, Any]:
        target: dict[str, Any] | None = None
        if self.state is not None and self.state.target_com_leg is not None:
            try:
                direction = observation.target_direction_for(self.state.target_com_leg.value)
                target = {
                    "target_leg": direction.target_leg,
                    "target_contact_w": list(direction.target_contact_w),
                    "direction_w": list(direction.direction_w),
                    "direction_body": list(direction.direction_body),
                    "distance_m": direction.distance_m,
                    "source": direction.source,
                }
            except Exception as exc:
                target = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "control_ready": observation.control_ready,
            "missing_critical_fields": list(observation.missing_critical_fields),
            "expected_primary_diagonal": (
                "" if self.state is None else self.state.primary_diagonal.value
            ),
            "measured_primary_diagonal": _enum_text(observation.primary_diagonal),
            "support_legs": _leg_values(observation.support_legs),
            "light_support_legs": _leg_values(observation.light_support_legs),
            "swing_leg": (
                "" if self.state is None or self.state.swing_leg is None else self.state.swing_leg.value
            ),
            "active_leg": (
                "" if self.state is None or self.state.active_leg is None else self.state.active_leg.value
            ),
            "impulse_leg": (
                "" if self.state is None or self.state.impulse_leg is None else self.state.impulse_leg.value
            ),
            "target_com_leg": (
                "" if self.state is None or self.state.target_com_leg is None else self.state.target_com_leg.value
            ),
            "diagonal_load_share": dict(observation.diagonal_load_share),
            "support_polygon_valid": observation.support_polygon_valid,
            "support_polygon_margin_m": observation.support_polygon_margin_m,
            "two_leg_corridor": {
                "applicable": observation.two_leg_corridor.applicable,
                "valid": observation.two_leg_corridor.valid,
                "perpendicular_distance_m": observation.two_leg_corridor.perpendicular_distance_m,
                "segment_fraction": observation.two_leg_corridor.segment_fraction,
            },
            "live_com_target": target,
            "primitive": _primitive_stage(self.state) if self.state is not None else {},
        }

    def _new_context(
        self,
        state: FSM50State,
        observation: FSM50Observation,
        *,
        inherited: GuardEvaluationContext | None,
    ) -> GuardEvaluationContext:
        observation.require_control_ready()
        return GuardEvaluationContext.enter(
            state, observation, self.thresholds, inherited=inherited
        )

    def on_enter(
        self,
        state: FSM50State,
        observation: FSM50Observation,
        *,
        inherited: GuardEvaluationContext | None,
        transition_reason: str,
    ) -> GuardDecision:
        self.state = state
        self.context = self._new_context(state, observation, inherited=inherited)
        self.dwell_tracker = TransitionDwellTracker(state, float(observation.time_s))
        self.entry_decision = self.registry.evaluate(
            state.entry_guard.kind, observation, self.context
        )
        self.entry_ready = bool(
            self.entry_decision.satisfied and not self.entry_decision.abort
        )
        if self.entry_ready:
            self.executor.enter_state(state, time_s=float(observation.time_s))
        self._emit(
            "on_enter",
            state_id=state.state_id,
            time_s=observation.time_s,
            transition_reason=transition_reason,
            retry_index=self.retry_counts.get(state.state_id, 0),
            entry_guard=_decision_mapping(self.entry_decision),
            restore_provenance=_plain(self.restore_provenance),
            primitive=_primitive_stage(state),
        )
        return self.entry_decision

    def on_exit(
        self,
        observation: FSM50Observation,
        *,
        outcome: str,
        reason: str,
    ) -> None:
        if self.state is None:
            return
        self._emit(
            "on_exit",
            state_id=self.state.state_id,
            time_s=observation.time_s,
            outcome=outcome,
            reason=reason,
            retry_index=self.retry_counts.get(self.state.state_id, 0),
        )

    def start(self, observation: FSM50Observation) -> ControllerTickResult:
        if self.status != ControllerStatus.NOT_STARTED:
            raise RuntimeError("controller has already started")
        if not isinstance(observation, FSM50Observation):
            raise TypeError("start requires FSM50Observation")
        if observation.time_s is None or not math.isfinite(float(observation.time_s)):
            return self._enter_safe_stop(observation, "startup simulation time unavailable")
        self.last_time_s = float(observation.time_s)
        self.status = ControllerStatus.RUNNING
        state = self.state_table.get(self.start_state_id)
        try:
            decision = self.on_enter(
                state,
                observation,
                inherited=None,
                transition_reason="controller_start",
            )
        except Exception as exc:
            return self._enter_safe_stop(
                observation, f"state entry failed: {type(exc).__name__}: {exc}"
            )
        if decision.abort:
            return self._enter_safe_stop(
                observation, f"entry guard abort: {decision.reason}"
            )
        return self._result(
            observation,
            previous_state_id="",
            transitioned=True,
            transition_reason="controller_start",
            entry=decision,
        )

    def _next_state(self, state_id: str) -> FSM50State | None:
        index = self._index[state_id]
        return None if index + 1 >= len(self._happy_path) else self._happy_path[index + 1]

    def _retry_or_stop(
        self, observation: FSM50Observation, decision: GuardDecision
    ) -> ControllerTickResult:
        assert self.state is not None
        failed_state = self.state
        policy = failed_state.retry_policy
        count = self.retry_counts.get(failed_state.state_id, 0)
        if count < policy.maximum_retries:
            retry_state_id = policy.retry_state_id or failed_state.state_id
            if retry_state_id not in self._index:
                return self._enter_safe_stop(
                    observation,
                    f"retry target is not on happy path: {retry_state_id}",
                )
            self.retry_counts[failed_state.state_id] = count + 1
            inherited = self.context
            self.on_exit(observation, outcome="retry", reason=decision.reason)
            retry_state = self.state_table.get(retry_state_id)
            try:
                entry = self.on_enter(
                    retry_state,
                    observation,
                    inherited=inherited,
                    transition_reason=f"retry:{failed_state.state_id}",
                )
            except Exception as exc:
                return self._enter_safe_stop(
                    observation, f"retry entry failed: {type(exc).__name__}: {exc}"
                )
            if entry.abort:
                return self._enter_safe_stop(
                    observation, f"retry entry guard abort: {entry.reason}"
                )
            return self._result(
                observation,
                previous_state_id=failed_state.state_id,
                transitioned=True,
                transition_reason=f"retry {count + 1}/{policy.maximum_retries}",
                entry=entry,
                dwell=decision,
            )
        exhausted = (
            f"state timeout/retry exhausted at {failed_state.state_id}: "
            f"{decision.reason or 'physical exit event unavailable'}"
        )
        return self._enter_safe_stop(observation, exhausted)

    def update(
        self,
        observation: FSM50Observation,
        *,
        whole_body_corrections_deg: Mapping[Any, Mapping[str, float]] | None = None,
        ik_candidates: Mapping[Any, LegIKCandidate] | None = None,
        place_confirm_blend: float | None = None,
    ) -> ControllerTickResult:
        if not isinstance(observation, FSM50Observation):
            raise TypeError("update requires FSM50Observation")
        if self.status == ControllerStatus.NOT_STARTED:
            return self.start(observation)
        if self.status == ControllerStatus.SUCCEEDED:
            return self._result(
                observation,
                previous_state_id=self.current_state_id,
                transitioned=False,
                transition_reason="terminal success already latched",
            )
        if self.status == ControllerStatus.SAFE_STOP:
            return self._update_safe_stop(observation)
        if observation.time_s is None or not math.isfinite(float(observation.time_s)):
            return self._enter_safe_stop(observation, "simulation time unavailable")
        now = float(observation.time_s)
        if self.last_time_s is not None and now + 1.0e-12 < self.last_time_s:
            return self._enter_safe_stop(observation, "simulation time moved backwards")
        self.last_time_s = now
        assert self.state is not None and self.context is not None
        assert self.dwell_tracker is not None
        previous_state_id = self.state.state_id

        abort_decision = self.registry.evaluate(
            self.state.abort_guard.kind, observation, self.context
        )
        if abort_decision.abort:
            return self._enter_safe_stop(
                observation,
                f"abort guard {self.state.abort_guard.kind}: {abort_decision.reason}",
                abort=abort_decision,
            )

        if not self.entry_ready:
            self.entry_decision = self.registry.evaluate(
                self.state.entry_guard.kind, observation, self.context
            )
            if self.entry_decision.abort:
                return self._enter_safe_stop(
                    observation, f"entry guard abort: {self.entry_decision.reason}"
                )
            if not self.entry_decision.satisfied:
                return self._result(
                    observation,
                    previous_state_id=previous_state_id,
                    transitioned=False,
                    transition_reason="waiting for entry guard",
                    entry=self.entry_decision,
                    abort=abort_decision,
                )
            try:
                self.executor.enter_state(self.state, time_s=now)
            except Exception as exc:
                return self._enter_safe_stop(
                    observation, f"executor state entry failed: {type(exc).__name__}: {exc}"
                )
            self.entry_ready = True

        # A state may not stage another motion command after its maximum
        # simulation-time budget has already expired.  Retry/SAFE_STOP owns
        # this tick's only write instead.
        state_elapsed_s = max(0.0, now - self.dwell_tracker.entered_at_s)
        if state_elapsed_s > self.state.max_duration:
            timeout_decision = GuardDecision(
                False,
                True,
                "state maximum timeout exceeded",
                {"elapsed_s": state_elapsed_s},
            )
            return self._retry_or_stop(observation, timeout_decision)

        execution = self.executor.update(
            time_s=now,
            whole_body_corrections_deg=whole_body_corrections_deg,
            ik_candidates=ik_candidates,
            place_confirm_blend=place_confirm_blend,
        )
        if self.state.state_id == "F3_RECOVER_HOME_CONCURRENT":
            self.context.extra["atomic_home_ack"] = bool(
                execution.atomic_evidence.get("verified", False)
            )
        if not execution.ok or execution.fail_closed:
            return self._enter_safe_stop(
                observation,
                f"command executor fail-closed: {execution.error}",
                execution=execution,
            )

        progress_decision = self.registry.evaluate(
            self.state.progress_guard.kind, observation, self.context
        )
        exit_decision = self.registry.evaluate(
            self.state.exit_guard.kind, observation, self.context
        )
        if progress_decision.abort or exit_decision.abort:
            failed = progress_decision if progress_decision.abort else exit_decision
            return self._enter_safe_stop(
                observation, f"physical guard abort: {failed.reason}", execution=execution
            )
        dwell_decision = self.dwell_tracker.update(
            time_s=now, exit_condition=bool(exit_decision.satisfied)
        )
        if dwell_decision.abort:
            return self._retry_or_stop(observation, dwell_decision)

        self._emit(
            "update",
            state_id=self.state.state_id,
            time_s=now,
            retry_index=self.retry_counts.get(self.state.state_id, 0),
            entry_guard=_decision_mapping(self.entry_decision),
            abort_guard=_decision_mapping(abort_decision),
            progress_guard=_decision_mapping(progress_decision),
            exit_guard=_decision_mapping(exit_decision),
            dwell=_decision_mapping(dwell_decision),
            execution=execution.to_mapping(),
            evidence=self._observation_evidence(observation),
        )

        if not dwell_decision.satisfied:
            return self._result(
                observation,
                previous_state_id=previous_state_id,
                transitioned=False,
                transition_reason="waiting for physical event dwell",
                entry=self.entry_decision,
                progress=progress_decision,
                exit=exit_decision,
                abort=abort_decision,
                dwell=dwell_decision,
                execution=execution,
            )

        if self.state.state_id == "F5_SUCCESS":
            self.status = ControllerStatus.SUCCEEDED
            self.on_exit(
                observation,
                outcome="terminal_success",
                reason="strict terminal physical guard and dwell satisfied",
            )
            return self._result(
                observation,
                previous_state_id=previous_state_id,
                transitioned=False,
                transition_reason="strict terminal physical success",
                entry=self.entry_decision,
                progress=progress_decision,
                exit=exit_decision,
                abort=abort_decision,
                dwell=dwell_decision,
                execution=execution,
            )

        next_state = self._next_state(self.state.state_id)
        if next_state is None:
            return self._enter_safe_stop(
                observation, "happy path ended without F5 terminal success"
            )
        inherited = self.context
        self.on_exit(
            observation,
            outcome="transition",
            reason=f"{self.state.guard} physical guard and dwell satisfied",
        )
        try:
            entry = self.on_enter(
                next_state,
                observation,
                inherited=inherited,
                transition_reason=f"physical_guard:{previous_state_id}",
            )
        except Exception as exc:
            return self._enter_safe_stop(
                observation, f"next-state entry failed: {type(exc).__name__}: {exc}"
            )
        if entry.abort:
            return self._enter_safe_stop(
                observation, f"next-state entry guard abort: {entry.reason}"
            )
        return self._result(
            observation,
            previous_state_id=previous_state_id,
            transitioned=True,
            transition_reason=f"physical guard {inherited.state.guard} satisfied",
            entry=entry,
            progress=progress_decision,
            exit=exit_decision,
            abort=abort_decision,
            dwell=dwell_decision,
            execution=execution,
        )

    def _safe_stop_command(self) -> tuple[dict[str, Any], dict[str, Any]]:
        captured: dict[str, Any] = {}
        try:
            raw = self.adapter.capture_command_state()
            if isinstance(raw, Mapping):
                captured = dict(raw)
        except Exception as exc:
            captured = {"capture_error": f"{type(exc).__name__}: {exc}"}
        servos_raw = captured.get("servos", {})
        wheels_raw = captured.get("wheels", {})
        servos = (
            {str(name): float(value) for name, value in servos_raw.items()}
            if isinstance(servos_raw, Mapping)
            else {}
        )
        wheel_names = set(str(name) for name in dict(wheels_raw or {}))
        configured_wheels = dict(
            dict(self.state_table.metadata.get("actuators", {}) or {}).get(
                "wheel_joints", {}
            )
            or {}
        )
        wheel_names.update(str(name) for name in configured_wheels.values())
        wheels = {name: 0.0 for name in sorted(wheel_names)}
        payload = {
            "batch_id": f"fsm50-safe-stop-{self._event_index + 1}",
            "source": "fsm50_controller_safe_stop",
            "state_id": "SAFE_STOP",
            "atomic_concurrent": bool(servos and wheels),
            "servo_targets_deg": servos,
            "wheel_targets_rad_s": wheels,
            "emergency_override": True,
        }
        try:
            raw_ack = self.adapter.apply_motion_batch(payload)
            ack = dict(raw_ack) if isinstance(raw_ack, Mapping) else {
                "error": "apply_motion_batch returned a non-mapping acknowledgement"
            }
        except Exception as exc:
            ack = {"error": f"{type(exc).__name__}: {exc}"}
        requested_zero = bool(wheels) and all(value == 0.0 for value in wheels.values())
        applied_raw = ack.get("wheel_targets_applied")
        applied_zero = isinstance(applied_raw, Mapping) and set(applied_raw) == set(wheels) and all(
            math.isfinite(float(applied_raw[name])) and abs(float(applied_raw[name])) <= 1.0e-12
            for name in wheels
        )
        hold_ack = ack.get("servo_targets_applied")
        servo_hold = bool(servos) and isinstance(hold_ack, Mapping) and set(hold_ack) == set(servos) and all(
            abs(float(hold_ack[name]) - servos[name]) <= 1.0e-6 for name in servos
        )
        evidence = {
            "payload": payload,
            "ack": ack,
            "captured_command_state": captured,
            "zero_wheels_requested": requested_zero,
            "zero_wheels_acknowledged": applied_zero,
            "current_servo_hold_acknowledged": servo_hold,
            "applied": bool(not str(ack.get("error", "") or "") and applied_zero and servo_hold),
        }
        return ack, evidence

    def _enter_safe_stop(
        self,
        observation: FSM50Observation,
        reason: str,
        *,
        entry: GuardDecision | None = None,
        progress: GuardDecision | None = None,
        exit: GuardDecision | None = None,
        abort: GuardDecision | None = None,
        dwell: GuardDecision | None = None,
        execution: ExecutionResult | None = None,
    ) -> ControllerTickResult:
        previous_state_id = self.current_state_id
        if self.state is not None and self.state.state_id != "SAFE_STOP":
            self.on_exit(observation, outcome="safe_stop", reason=reason)
        inherited = self.context
        self.state = self._safe_state
        self.status = ControllerStatus.SAFE_STOP
        self.entry_ready = False
        self.safe_stop_reason = str(reason)
        if observation.control_ready:
            try:
                self.context = self._new_context(
                    self._safe_state, observation, inherited=inherited
                )
            except Exception:
                self.context = inherited
        else:
            self.context = inherited
        if self.context is not None:
            self.context.state = self._safe_state
        ack, command_evidence = self._safe_stop_command()
        self.safe_stop_ack = dict(ack)
        if self.context is not None:
            self.context.safe_stop_applied = bool(command_evidence["applied"])
        self._emit(
            "safe_stop",
            state_id="SAFE_STOP",
            previous_state_id=previous_state_id,
            time_s=observation.time_s,
            reason=reason,
            command_evidence=command_evidence,
            observation_evidence=self._observation_evidence(observation),
        )
        return self._result(
            observation,
            previous_state_id=previous_state_id,
            transitioned=previous_state_id != "SAFE_STOP",
            transition_reason=reason,
            entry=entry,
            progress=progress,
            exit=exit,
            abort=abort,
            dwell=dwell,
            execution=execution,
            extra_evidence={"safe_stop_command": command_evidence},
        )

    def _update_safe_stop(
        self, observation: FSM50Observation
    ) -> ControllerTickResult:
        previous_state_id = self.current_state_id
        extra: dict[str, Any] = {
            "safe_stop_reason": self.safe_stop_reason,
            "safe_stop_ack": self.safe_stop_ack,
        }
        decision: GuardDecision | None = None
        if self.context is None and observation.control_ready:
            try:
                self.context = self._new_context(
                    self._safe_state, observation, inherited=None
                )
            except Exception:
                self.context = None
        if self.context is not None:
            self.context.state = self._safe_state
            if not self.context.safe_stop_applied:
                ack, evidence = self._safe_stop_command()
                self.safe_stop_ack = dict(ack)
                self.context.safe_stop_applied = bool(evidence["applied"])
                extra["safe_stop_retry"] = evidence
            decision = self.registry.evaluate(
                self._safe_state.exit_guard.kind, observation, self.context
            )
            self.safe_stop_confirmed = bool(decision.satisfied and not decision.abort)
        self._emit(
            "safe_stop_update",
            state_id="SAFE_STOP",
            time_s=observation.time_s,
            guard=_decision_mapping(decision),
            confirmed=self.safe_stop_confirmed,
            evidence=self._observation_evidence(observation),
        )
        return self._result(
            observation,
            previous_state_id=previous_state_id,
            transitioned=False,
            transition_reason=self.safe_stop_reason,
            exit=decision,
            extra_evidence=extra,
        )

    def _result(
        self,
        observation: FSM50Observation,
        *,
        previous_state_id: str,
        transitioned: bool,
        transition_reason: str,
        entry: GuardDecision | None = None,
        progress: GuardDecision | None = None,
        exit: GuardDecision | None = None,
        abort: GuardDecision | None = None,
        dwell: GuardDecision | None = None,
        execution: ExecutionResult | None = None,
        extra_evidence: Mapping[str, Any] | None = None,
    ) -> ControllerTickResult:
        retry_state = previous_state_id or self.current_state_id
        evidence = self._observation_evidence(observation)
        evidence.update(dict(extra_evidence or {}))
        evidence.update(
            {
                "safe_stop_reason": self.safe_stop_reason,
                "safe_stop_confirmed": self.safe_stop_confirmed,
                "restore_provenance": _plain(self.restore_provenance),
                "completed_state_ids": list(self.completed_state_ids),
            }
        )
        return ControllerTickResult(
            status=self.status,
            state_id=self.current_state_id,
            previous_state_id=previous_state_id,
            time_s=(
                float(observation.time_s)
                if observation.time_s is not None and math.isfinite(float(observation.time_s))
                else float("nan")
            ),
            transitioned=bool(transitioned),
            transition_reason=str(transition_reason),
            retry_index=self.retry_counts.get(retry_state, 0),
            entry_guard=_decision_mapping(entry),
            progress_guard=_decision_mapping(progress),
            exit_guard=_decision_mapping(exit),
            abort_guard=_decision_mapping(abort),
            dwell=_decision_mapping(dwell),
            execution=None if execution is None else execution.to_mapping(),
            evidence=evidence,
        )


def validate_happy_path_reachability(
    state_table: FSM50StateTable | str | Path,
) -> tuple[str, ...]:
    """Validate and return the controller's deterministic happy-path order."""

    table = (
        FSM50StateTable.load(state_table)
        if isinstance(state_table, (str, Path))
        else state_table
    )
    registry = FSM50GuardRegistry()
    registry.validate_states(table.states)
    states = tuple(state for state in table.states if state.state_id != "SAFE_STOP")
    identifiers = tuple(state.state_id for state in states)
    if not identifiers or identifiers[0] != "A0_RESET_AND_SETTLE":
        raise ValueError("happy path must start at A0_RESET_AND_SETTLE")
    if identifiers[-1] != "F5_SUCCESS":
        raise ValueError("happy path must end at F5_SUCCESS")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("happy path contains duplicate states")
    return identifiers


FSM50RuntimeController = FSM50Controller


__all__ = [
    "ControllerStatus",
    "ControllerTickResult",
    "FSM50Controller",
    "FSM50RuntimeController",
    "validate_happy_path_reachability",
]
