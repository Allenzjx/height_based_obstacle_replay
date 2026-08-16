"""Small, recording-derived macro graph for the 50 mm obstacle.

This is intentionally separate from :mod:`fsm50_state_model`.  The latter is
the legacy 57-state semantic graph; it remains useful as a catalogue of old
failure modes, but it is not the control truth for Gate C.

The graph in this module owns *physical phases and transitions*.  Motion
timelines and recording provenance live in :mod:`fsm50_motion_profiles`, and
live feedback evaluation lives in :mod:`fsm50_macro_controller`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


LEGACY_57_STATE_CONFIG = "fsm50_config.yaml"
LEGACY_57_STATE_COUNT = 57
LEGACY_57_STATE_CONTROL_AUTHORITY = False

# This is a recording-derived reference only.  It seeds bounded S10 feedback;
# it is not itself an authoritative feedback policy and must never be reported
# as a newly consumed source action by that feedback controller.
FINAL_RECOVERY_REFERENCE_PROFILE_SEED: Mapping[str, Any] = {
    "source_version": "v010_20260806_220745_363972_manual",
    "strategy": "RECOVERY_PROFILE_1",
    "source_segment_indices": (140, 141),
    "authoritative_feedback_profile": "NOT_YET_PROVEN",
    "visual_target_only": {
        "source_version": "v008_20260806_211408_578700_manual",
        "source_segment_indices": (117,),
    },
    "fr_specific_alternate_only": {
        "source_version": "v003_20260805_224517_157723_manual",
        "strategy": "RECOVERY_PROFILE_2",
        "source_segment_indices": (104, 111),
    },
}

# Only FR-hip has a non-zero within-S10 increment in the four sealed successes
# (minimum 1.1 deg).  The other joints therefore have no recording-derived
# increment bound.  Use a smaller development probe for every axis and label
# it diagnostic rather than manufacturing profile truth.
FINAL_RECOVERY_FEEDBACK_LIMITS: Mapping[str, Any] = {
    "probe_kind": "CONSERVATIVE_DIAGNOSTIC_PROBE",
    "probe_delta_deg": 0.25,
    "increment_delta_deg": 0.25,
    "maximum_increments_per_leg": 8,
    "maximum_feedback_actions": 64,
    "joint_limit_margin_deg": 1.0,
    "contact_dwell_s": 0.25,
    "maximum_contact_dwell_wait_s": 0.75,
    "settle_dwell_s": 0.25,
    "maximum_settle_wait_s": 1.50,
    "maximum_n_plus_one_wait_steps": 1,
    "maximum_abs_joint_velocity_deg_s": 3.0,
    "maximum_abs_body_angular_velocity_rad_s": 0.15,
    "minimum_descent_m": 0.00025,
    "derivation_schema": "fsm50.s10_conservative_probe_derivation.v1",
    "derivation_sha256": "94d264434e81319c5266de0f0ba4a49001ce7864d9844f55a083837687bd1975",
    "recording_minimum_nonzero_increment_deg": {
        "front_right_hip": 1.0999999999999979,
    },
}


class MacroStateId(str, Enum):
    S0_INITIALIZE = "S0_INITIALIZE"
    S1_APPROACH_AND_PRE_FR_SHIFT = "S1_APPROACH_AND_PRE_FR_SHIFT"
    S2_FR_TRAVERSE = "S2_FR_TRAVERSE"
    S3_FL_TRAVERSE = "S3_FL_TRAVERSE"
    S4_FRONT_PAIR_ADVANCE = "S4_FRONT_PAIR_ADVANCE"
    S5_PRE_RR_COM_SHIFT = "S5_PRE_RR_COM_SHIFT"
    S6_RR_TRAVERSE = "S6_RR_TRAVERSE"
    S7_PRE_RL_SUPPORT_SETUP = "S7_PRE_RL_SUPPORT_SETUP"
    S8_RL_COM_SHIFT_AND_TRAVERSE = "S8_RL_COM_SHIFT_AND_TRAVERSE"
    S9_FINAL_ADVANCE = "S9_FINAL_ADVANCE"
    S10_POSTURE_RECOVERY = "S10_POSTURE_RECOVERY"
    SUCCESS = "SUCCESS"
    SAFE_STOP = "SAFE_STOP"


class MacroSubphase(str, Enum):
    PRELOAD = "PRELOAD"
    UNLOAD = "UNLOAD"
    LIFT = "LIFT"
    FACE_CLEAR = "FACE_CLEAR"
    TOP_PLACE = "TOP_PLACE"
    LOAD_CONFIRM = "LOAD_CONFIRM"
    ADVANCE = "ADVANCE"
    RECOVERY = "RECOVERY"
    FEEDBACK_RECOVERY = "FEEDBACK_RECOVERY"
    HOLD = "HOLD"
    RETRY = "RETRY"
    COMPLETE = "COMPLETE"
    SAFE_STOP = "SAFE_STOP"


class MacroGuardKind(str, Enum):
    INITIALIZED = "INITIALIZED"
    COM_SHIFT_OR_UNLOAD = "COM_SHIFT_OR_UNLOAD"
    LEG_TRAVERSED = "LEG_TRAVERSED"
    FRONT_PAIR_ADVANCED = "FRONT_PAIR_ADVANCED"
    SUPPORT_SETUP = "SUPPORT_SETUP"
    FINAL_ADVANCED = "FINAL_ADVANCED"
    POSTURE_RECOVERED = "POSTURE_RECOVERED"


class MacroRetryMode(str, Enum):
    HOLD_ONLY = "HOLD_ONLY"
    HOLD_TARGET_GUARD_RECHECK = "HOLD_TARGET_GUARD_RECHECK"


@dataclass(frozen=True)
class MacroGuardSpec:
    """A serializable physical-event guard.

    ``profile_must_complete`` makes the distinction from replay explicit:
    time is permitted to advance a recorded primitive, while these physical
    fields decide whether the macro state may transition.
    """

    kind: MacroGuardKind
    profile_must_complete: bool = True
    active_leg: str = ""
    target_com_leg: str = ""
    required_top_legs: tuple[str, ...] = ()
    required_support_legs: tuple[str, ...] = ()
    required_primary_diagonal: tuple[str, ...] = ()
    require_viable_support: bool = False
    require_support_wrench: bool = False
    release_physical_phase: str = ""
    require_airborne_before_crossing: bool = False
    require_body_crossed: bool = False
    minimum_com_displacement_m: float = 0.0
    minimum_body_progress_m: float = 0.0
    maximum_abs_roll_rad: float = math.radians(35.0)
    maximum_abs_pitch_rad: float = math.radians(35.0)

    def __post_init__(self) -> None:
        legs = {"FL", "FR", "RL", "RR"}
        for label, value in (
            ("active_leg", self.active_leg),
            ("target_com_leg", self.target_com_leg),
        ):
            if value and value not in legs:
                raise ValueError(f"{label} must name FL/FR/RL/RR")
        if not set(self.required_top_legs).issubset(legs):
            raise ValueError("required_top_legs contains an unknown leg")
        if not set(self.required_support_legs).issubset(legs):
            raise ValueError("required_support_legs contains an unknown leg")
        if self.required_primary_diagonal and (
            len(self.required_primary_diagonal) != 2
            or len(set(self.required_primary_diagonal)) != 2
            or not set(self.required_primary_diagonal).issubset(legs)
        ):
            raise ValueError("required_primary_diagonal must name two distinct legs")
        for label, value in (
            ("require_viable_support", self.require_viable_support),
            ("require_support_wrench", self.require_support_wrench),
        ):
            if type(value) is not bool:
                raise ValueError(f"{label} must be an exact bool")
        if not isinstance(self.release_physical_phase, str):
            raise ValueError("release_physical_phase must be a string")
        for name, value in (
            ("minimum_com_displacement_m", self.minimum_com_displacement_m),
            ("minimum_body_progress_m", self.minimum_body_progress_m),
            ("maximum_abs_roll_rad", self.maximum_abs_roll_rad),
            ("maximum_abs_pitch_rad", self.maximum_abs_pitch_rad),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class MacroRetryPolicy:
    maximum_retries: int = 0
    maximum_hold_s: float = 1.0
    runtime_retry_mode: MacroRetryMode = MacroRetryMode.HOLD_ONLY
    retry_requires_safe_attitude: bool = True
    guard_observation_window_basis: str = "STRUCTURAL_BOUNDED_HOLD"

    def __post_init__(self) -> None:
        if self.maximum_retries < 0:
            raise ValueError("maximum_retries must be non-negative")
        if not math.isfinite(float(self.maximum_hold_s)) or self.maximum_hold_s < 0.0:
            raise ValueError("maximum_hold_s must be finite and non-negative")
        if not self.guard_observation_window_basis.strip():
            raise ValueError("guard_observation_window_basis is required")
        if self.maximum_retries and self.runtime_retry_mode != MacroRetryMode.HOLD_TARGET_GUARD_RECHECK:
            raise ValueError(
                "runtime retries may only recheck the guard while holding the "
                "current target; profile rewind/hot switching is not permitted"
            )


@dataclass(frozen=True)
class MacroStateSpec:
    state_id: MacroStateId
    physical_purpose: str
    physical_phases: tuple[str, ...]
    active_leg: str = ""
    candidate_support_legs: tuple[str, ...] = ()
    completion_guard: MacroGuardSpec = field(
        default_factory=lambda: MacroGuardSpec(MacroGuardKind.INITIALIZED)
    )
    retry_policy: MacroRetryPolicy = field(default_factory=MacroRetryPolicy)
    next_state: MacroStateId = MacroStateId.SAFE_STOP
    profile_required: bool = True
    timeout_scale: float = 2.0
    minimum_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.physical_purpose.strip():
            raise ValueError("physical_purpose is required")
        if self.profile_required and not self.physical_phases:
            raise ValueError("profile-backed states require physical phases")
        if self.active_leg and self.active_leg not in {"FL", "FR", "RL", "RR"}:
            raise ValueError("active_leg must name FL/FR/RL/RR")
        if not set(self.candidate_support_legs).issubset({"FL", "FR", "RL", "RR"}):
            raise ValueError("candidate_support_legs contains an unknown leg")
        if not math.isfinite(float(self.timeout_scale)) or self.timeout_scale < 1.0:
            raise ValueError("timeout_scale must be finite and >= 1")
        if not math.isfinite(float(self.minimum_timeout_s)) or self.minimum_timeout_s <= 0.0:
            raise ValueError("minimum_timeout_s must be finite and positive")

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_id"] = self.state_id.value
        payload["next_state"] = self.next_state.value
        payload["completion_guard"]["kind"] = self.completion_guard.kind.value
        return payload


@dataclass(frozen=True)
class MacroFSMGraph:
    states: tuple[MacroStateSpec, ...]
    initial_state: MacroStateId = MacroStateId.S0_INITIALIZE
    success_state: MacroStateId = MacroStateId.SUCCESS
    safe_stop_state: MacroStateId = MacroStateId.SAFE_STOP
    graph_id: str = "fsm50-recording-derived-macro-v1"
    legacy_graph_role: str = "fsm50_config.yaml is LEGACY_SEMANTIC_GRAPH only"
    legacy_graph_state_count: int = LEGACY_57_STATE_COUNT
    legacy_graph_control_authority: bool = LEGACY_57_STATE_CONTROL_AUTHORITY

    def __post_init__(self) -> None:
        ids = tuple(state.state_id for state in self.states)
        if len(ids) != len(set(ids)):
            raise ValueError("macro state ids must be unique")
        active = tuple(
            state_id
            for state_id in ids
            if state_id not in {self.success_state, self.safe_stop_state}
        )
        if len(active) > 12:
            raise ValueError("Gate C macro graph may contain at most 12 active states")
        if self.legacy_graph_state_count != LEGACY_57_STATE_COUNT:
            raise ValueError("legacy graph state-count marker must remain 57")
        if self.legacy_graph_control_authority is not False:
            raise ValueError("the legacy 57-state graph cannot have Gate-C control authority")
        required = {self.initial_state, self.success_state, self.safe_stop_state}
        if not required.issubset(set(ids)):
            raise ValueError("graph is missing initial or terminal states")
        for state in self.states:
            if state.state_id not in {self.success_state, self.safe_stop_state}:
                if state.next_state not in set(ids):
                    raise ValueError(f"{state.state_id.value} has an unknown next_state")

    @property
    def active_state_count(self) -> int:
        return sum(
            state.state_id not in {self.success_state, self.safe_stop_state}
            for state in self.states
        )

    @property
    def active_state_ids(self) -> tuple[MacroStateId, ...]:
        return tuple(
            state.state_id
            for state in self.states
            if state.state_id not in {self.success_state, self.safe_stop_state}
        )

    def get(self, state_id: MacroStateId | str) -> MacroStateSpec:
        wanted = state_id if isinstance(state_id, MacroStateId) else MacroStateId(state_id)
        for state in self.states:
            if state.state_id == wanted:
                return state
        raise KeyError(wanted.value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.macro_graph.v1",
            "graph_id": self.graph_id,
            "initial_state": self.initial_state.value,
            "success_state": self.success_state.value,
            "safe_stop_state": self.safe_stop_state.value,
            "legacy_graph_role": self.legacy_graph_role,
            "legacy_graph_path": LEGACY_57_STATE_CONFIG,
            "legacy_graph_state_count": self.legacy_graph_state_count,
            "legacy_graph_control_authority": self.legacy_graph_control_authority,
            "states": [state.to_mapping() for state in self.states],
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_default_macro_graph() -> MacroFSMGraph:
    """Return the single graph shared by every successful recording profile."""

    common_attitude = {
        "maximum_abs_roll_rad": math.radians(35.0),
        "maximum_abs_pitch_rad": math.radians(35.0),
    }
    states = (
        MacroStateSpec(
            MacroStateId.S0_INITIALIZE,
            "Validate live finite state and establish episode/COM baselines.",
            ("INITIALIZE",),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.INITIALIZED,
                profile_must_complete=False,
                **common_attitude,
            ),
            next_state=MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            profile_required=False,
            minimum_timeout_s=1.0,
        ),
        MacroStateSpec(
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            "Approach and move body/COM toward RL so FR can unload.",
            ("INITIAL_APPROACH", "PRE_FR_COM_SHIFT"),
            active_leg="FR",
            candidate_support_legs=("FL", "RL", "RR"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.COM_SHIFT_OR_UNLOAD,
                active_leg="FR",
                target_com_leg="RL",
                require_viable_support=True,
                minimum_com_displacement_m=0.003,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_retries=1,
                maximum_hold_s=0.8,
                runtime_retry_mode=MacroRetryMode.HOLD_TARGET_GUARD_RECHECK,
                guard_observation_window_basis=(
                    "Gate-A feedback boundary: hold current target only while "
                    "re-observing FR unload; no action rewind"
                ),
            ),
            next_state=MacroStateId.S2_FR_TRAVERSE,
        ),
        MacroStateSpec(
            MacroStateId.S2_FR_TRAVERSE,
            "Unload, lift, clear the front face, and place FR on top.",
            ("FR_UNLOAD_AND_LIFT", "FR_FACE_CROSS", "FR_TOP_PLACE"),
            active_leg="FR",
            candidate_support_legs=("FL", "RL", "RR"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.LEG_TRAVERSED,
                active_leg="FR",
                required_top_legs=("FR",),
                require_airborne_before_crossing=True,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_hold_s=2.25,
                guard_observation_window_basis=(
                    "exact 120Hz-controller/held-15Hz-success shadow: v003 "
                    "post-profile FR TOP residual 2.008s plus sample/control "
                    "uncertainty, rounded to 0.25s"
                ),
            ),
            next_state=MacroStateId.S3_FL_TRAVERSE,
        ),
        MacroStateSpec(
            MacroStateId.S3_FL_TRAVERSE,
            "Unload, lift, clear the front face, and place FL on top.",
            ("FL_UNLOAD_AND_LIFT", "FL_FACE_CROSS", "FL_TOP_PLACE"),
            active_leg="FL",
            candidate_support_legs=("FR", "RL", "RR"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.LEG_TRAVERSED,
                active_leg="FL",
                required_top_legs=("FR", "FL"),
                require_airborne_before_crossing=True,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_hold_s=6.25,
                guard_observation_window_basis=(
                    "exact 120Hz-controller/held-15Hz four-success shadow: max "
                    "post-profile FL TOP residual 5.858s plus sample/control "
                    "uncertainty, rounded to 0.25s"
                ),
            ),
            next_state=MacroStateId.S4_FRONT_PAIR_ADVANCE,
        ),
        MacroStateSpec(
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            "Acknowledge the front-pair advance, action-backed only when the selected recording owns a distinct command window.",
            ("FRONT_PAIR_ADVANCE",),
            candidate_support_legs=("FR", "FL", "RL", "RR"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.FRONT_PAIR_ADVANCED,
                required_top_legs=("FR", "FL"),
                minimum_body_progress_m=0.03,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_hold_s=0.8,
                guard_observation_window_basis=(
                    "immediate-predecessor carried front-pair TOP/progress must "
                    "already exist; hold is bounded live re-observation only"
                ),
            ),
            next_state=MacroStateId.S5_PRE_RR_COM_SHIFT,
            profile_required=False,
        ),
        MacroStateSpec(
            MacroStateId.S5_PRE_RR_COM_SHIFT,
            "Move body/COM toward the FL support region so RR can unload.",
            ("PRE_RR_COM_SHIFT",),
            active_leg="RR",
            candidate_support_legs=("FR", "FL", "RL"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.COM_SHIFT_OR_UNLOAD,
                active_leg="RR",
                target_com_leg="FL",
                require_viable_support=True,
                minimum_com_displacement_m=0.003,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_retries=1,
                maximum_hold_s=0.8,
                runtime_retry_mode=MacroRetryMode.HOLD_TARGET_GUARD_RECHECK,
                guard_observation_window_basis=(
                    "Gate-A boundary audit: live RR AIR or immediate S4-to-S5 "
                    "inherited target-direction displacement; no action rewind"
                ),
            ),
            next_state=MacroStateId.S6_RR_TRAVERSE,
            profile_required=False,
        ),
        MacroStateSpec(
            MacroStateId.S6_RR_TRAVERSE,
            "Unload, lift, clear the front face, and place RR on top.",
            ("RR_UNLOAD_AND_LIFT", "RR_FACE_CROSS", "RR_TOP_PLACE"),
            active_leg="RR",
            candidate_support_legs=("FR", "FL", "RL"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.LEG_TRAVERSED,
                active_leg="RR",
                required_top_legs=("FR", "FL", "RR"),
                require_airborne_before_crossing=True,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_hold_s=3.75,
                guard_observation_window_basis=(
                    "exact 120Hz-controller/held-15Hz four-success shadow: max "
                    "post-profile RR TOP residual 3.55s plus sample/control "
                    "uncertainty, rounded to 0.25s"
                ),
            ),
            next_state=MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
        ),
        MacroStateSpec(
            MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
            "Create the FL/RR-biased support workspace used before RL lift.",
            ("PRE_RL_SUPPORT_SETUP",),
            active_leg="RL",
            candidate_support_legs=("FR", "FL", "RR"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.SUPPORT_SETUP,
                active_leg="RL",
                required_support_legs=("FL", "RR"),
                required_primary_diagonal=("FL", "RR"),
                require_viable_support=True,
                require_support_wrench=True,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_retries=1,
                maximum_hold_s=0.8,
                runtime_retry_mode=MacroRetryMode.HOLD_TARGET_GUARD_RECHECK,
                guard_observation_window_basis=(
                    "support geometry is live at the RR boundary except v009, "
                    "whose recorded causal prefix is profile-backed"
                ),
            ),
            next_state=MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
            # Optional by source: v009 owns its causal PRE_RL_COM_SHIFT prefix;
            # the other successes acknowledge live support without a profile.
            profile_required=False,
        ),
        MacroStateSpec(
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
            "Shift body/COM toward FR, then unload, lift, and place RL on top.",
            (
                "PRE_RL_COM_SHIFT",
                "RL_UNLOAD_AND_LIFT",
                "RL_FACE_CROSS",
                "RL_TOP_PLACE",
            ),
            active_leg="RL",
            candidate_support_legs=("FR", "FL", "RR"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.LEG_TRAVERSED,
                active_leg="RL",
                target_com_leg="FR",
                minimum_com_displacement_m=0.003,
                require_viable_support=True,
                require_support_wrench=True,
                release_physical_phase="RL_UNLOAD_AND_LIFT",
                # Traversal completion needs the active RL wheel on top.  A
                # previously crossed support wheel may be temporarily AIR;
                # final all-TOP is explicitly a secondary diagnostic.
                required_top_legs=("RL",),
                require_airborne_before_crossing=True,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_hold_s=7.25,
                guard_observation_window_basis=(
                    "exact 120Hz-controller/held-15Hz four-success shadow: max "
                    "post-profile RL TOP residual 6.92s plus sample/control "
                    "uncertainty, rounded to 0.25s"
                ),
            ),
            next_state=MacroStateId.S9_FINAL_ADVANCE,
        ),
        MacroStateSpec(
            MacroStateId.S9_FINAL_ADVANCE,
            "Advance the body beyond the front face and create recovery workspace.",
            ("FINAL_ADVANCE",),
            candidate_support_legs=("FR", "FL", "RR", "RL"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.FINAL_ADVANCED,
                require_body_crossed=True,
                minimum_body_progress_m=0.03,
                **common_attitude,
            ),
            retry_policy=MacroRetryPolicy(
                maximum_hold_s=1.0,
                guard_observation_window_basis=(
                    "immediate S8 boundary carries body/major-cross progress; "
                    "v010 additionally owns a distinct final-advance profile"
                ),
            ),
            next_state=MacroStateId.S10_POSTURE_RECOVERY,
            profile_required=False,
        ),
        MacroStateSpec(
            MacroStateId.S10_POSTURE_RECOVERY,
            "Apply the recording-derived recovery profile and finish recoverably.",
            ("FINAL_POSTURE_RECOVERY",),
            candidate_support_legs=("FR", "FL", "RR", "RL"),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.POSTURE_RECOVERED,
                required_top_legs=("FL", "FR", "RL", "RR"),
                require_viable_support=True,
                require_support_wrench=True,
                require_body_crossed=True,
                maximum_abs_roll_rad=math.radians(30.0),
                maximum_abs_pitch_rad=math.radians(30.0),
            ),
            retry_policy=MacroRetryPolicy(
                maximum_retries=1,
                maximum_hold_s=1.5,
                runtime_retry_mode=MacroRetryMode.HOLD_TARGET_GUARD_RECHECK,
                guard_observation_window_basis=(
                    "bounded final recoverability observation while holding the "
                    "recorded recovery target; no profile replay"
                ),
            ),
            next_state=MacroStateId.SUCCESS,
        ),
        MacroStateSpec(
            MacroStateId.SUCCESS,
            "Terminal task-success state.",
            (),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.POSTURE_RECOVERED, profile_must_complete=False
            ),
            next_state=MacroStateId.SUCCESS,
            profile_required=False,
        ),
        MacroStateSpec(
            MacroStateId.SAFE_STOP,
            "Terminal fail-closed state with all wheel targets zero.",
            (),
            completion_guard=MacroGuardSpec(
                MacroGuardKind.INITIALIZED, profile_must_complete=False
            ),
            next_state=MacroStateId.SAFE_STOP,
            profile_required=False,
        ),
    )
    return MacroFSMGraph(states=states)


PHYSICAL_PHASE_TO_MACRO_STATE: Mapping[str, MacroStateId] = {
    phase: state.state_id
    for state in build_default_macro_graph().states
    for phase in state.physical_phases
}
