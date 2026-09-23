"""Safety monitor for the LangGrasp pick-and-place stack (pure Python + numpy, no ROS, no torch).

The same object is used in-process by the pipeline and inside the ROS 2 safety node
(ros2_ws/src/langgrasp_ros/langgrasp_ros/safety_node.py). Every mechanism here maps to one hazard row of
docs/FMEA.md (H1..H8); the mapping is repeated next to each method so the code and the analysis stay in sync.

State machine (SafetyState):
    RUN            nominal, commands pass through (after limit / velocity clipping)
    REDUCED_SPEED  commands pass through with joint deltas scaled by ``reduced_speed_factor``
    HOLD           commands are replaced by "hold current pose"; recoverable when the reason clears
    ESTOP          latched; nothing moves until ``reset_estop()`` is called by a human
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Joint ranges of the SO-ARM100 Menagerie model (langgrasp/sim/assets/so_arm100/so_arm100.xml), radians.
DEFAULT_JOINT_LO = (-1.92, -3.32, -0.174, -1.66, -2.79)
DEFAULT_JOINT_HI = (1.92, 0.174, 3.14, 1.66, 2.79)


class SafetyState(Enum):
    RUN = "RUN"
    REDUCED_SPEED = "REDUCED_SPEED"
    HOLD = "HOLD"
    ESTOP = "ESTOP"


# diagnostic_msgs/DiagnosticStatus levels
LEVEL_OK = 0
LEVEL_WARN = 1
LEVEL_ERROR = 2
LEVEL_STALE = 3


@dataclass
class SafetyConfig:
    joint_lo: tuple = DEFAULT_JOINT_LO
    joint_hi: tuple = DEFAULT_JOINT_HI
    # rad/s per joint. Measured (results/safety_clip_audit.json, scripts/audit_safety_clips.py): the
    # in-process scripted executor commands up to 21.8 rad/s on the first waypoint of a move, so this
    # limit does bite. It is a limit, not a description of the executor: enforcing it changes the
    # trajectory (oracle picks 120/120 observing versus 73/120 enforcing), which is why the in-process
    # pipeline only reports what would be clipped and the ROS 2 safety node, whose executor commands
    # smaller per-tick deltas, applies it.
    max_joint_vel: float = 1.5
    # TCP geofence in world coordinates (metres). Covers workspace, tray and the observe pose with margin.
    geofence_lo: tuple = (-0.25, -0.34, 0.0)
    geofence_hi: tuple = (0.28, 0.02, 0.25)
    # Staleness timeouts (seconds) per heartbeat topic.
    staleness_s: dict = field(default_factory=lambda: {"camera": 0.5, "joint_states": 0.2, "command": 30.0})
    # Which stale topics latch an ESTOP (H2/H3/H5 proxies) versus only HOLD (a missing command is not dangerous).
    estop_on_stale: tuple = ("camera", "joint_states")
    grounding_conf_threshold: float = 0.35
    ambiguity_margin: float = 0.05  # top-1 minus top-2 grounding score below this => ambiguous
    policy_conf_threshold: float = 0.5
    reduced_speed_factor: float = 0.5
    require_human_confirm: bool = False
    control_dt: float = 0.1  # nominal period used when dt is not passed to check_joint_command


class SafetyMonitor:
    """Stateful safety monitor. All times are seconds on a monotonic clock supplied by the caller."""

    def __init__(self, config: SafetyConfig | None = None):
        self.cfg = config or SafetyConfig()
        self.lo = np.asarray(self.cfg.joint_lo, dtype=float)
        self.hi = np.asarray(self.cfg.joint_hi, dtype=float)
        self.geo_lo = np.asarray(self.cfg.geofence_lo, dtype=float)
        self.geo_hi = np.asarray(self.cfg.geofence_hi, dtype=float)
        self._last_seen: dict[str, float] = {}
        self._estop_reason: str | None = None
        self._hold_reasons: dict[str, str] = {}
        self._reduced_speed: bool = False
        self._reduced_reason: str = ""
        self._human_confirmed: bool = False
        self._last_reasons: list[str] = []
        self.n_clipped_limit = 0
        self.n_clipped_vel = 0

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> SafetyState:
        if self._estop_reason is not None:
            return SafetyState.ESTOP
        if self._hold_reasons:
            return SafetyState.HOLD
        if self._reduced_speed:
            return SafetyState.REDUCED_SPEED
        return SafetyState.RUN

    @property
    def reduced_speed(self) -> bool:
        return self._reduced_speed

    def set_reduced_speed(self, on: bool, reason: str = "") -> None:
        """H5 / H8 mitigation: switch to reduced-speed mode (for example while a human is near, or after a
        calibration-drift warning). Joint deltas are scaled by ``cfg.reduced_speed_factor`` in
        check_joint_command."""
        self._reduced_speed = bool(on)
        self._reduced_reason = reason if on else ""

    def hold(self, key: str, reason: str) -> None:
        """Enter HOLD for a named reason. Cleared with ``release(key)``."""
        self._hold_reasons[key] = reason

    def release(self, key: str) -> None:
        self._hold_reasons.pop(key, None)

    def estop(self, reason: str) -> None:
        """Latch the emergency stop. Stays latched until reset_estop() (human action)."""
        if self._estop_reason is None:
            self._estop_reason = reason

    def reset_estop(self) -> None:
        self._estop_reason = None
        self._hold_reasons.clear()

    @property
    def estop_reason(self) -> str | None:
        return self._estop_reason

    def confirm_human(self) -> None:
        """Operator confirmation for the current command (used when cfg.require_human_confirm is set)."""
        self._human_confirmed = True

    def clear_confirmation(self) -> None:
        self._human_confirmed = False

    # ------------------------------------------------------------------ H2 / H3 / H5: staleness
    def heartbeat(self, topic: str, t: float) -> None:
        self._last_seen[topic] = float(t)

    def check_staleness(self, t: float) -> list[str]:
        """Return the list of stale topics at time t and update the state.

        Hazard mapping: a stale camera stream covers H2 (occlusion / sensor dropout) and H3 (depth dropout);
        stale joint_states covers H4 (servo fault, the bus stops answering) and H8 (controller crash); a stale
        command only means the operator is silent, which is a HOLD rather than an ESTOP.
        """
        stale: list[str] = []
        for topic, timeout in self.cfg.staleness_s.items():
            last = self._last_seen.get(topic)
            if last is None or (t - last) > timeout:
                stale.append(topic)
        for topic in stale:
            age = (t - self._last_seen[topic]) if topic in self._last_seen else math.inf
            msg = f"{topic} stale ({age:.2f}s > {self.cfg.staleness_s[topic]:.2f}s)"
            if topic in self.cfg.estop_on_stale:
                self.estop(msg)
            else:
                self.hold(f"stale:{topic}", msg)
        for topic in self.cfg.staleness_s:
            if topic not in stale:
                self.release(f"stale:{topic}")
        return stale

    # ------------------------------------------------------------------ H4 / H8: joint limits and velocity
    def check_joint_command(self, q_target, q_now, dt: float | None = None) -> tuple[np.ndarray, list[str]]:
        """Clip a joint target to the joint limits and to the per-tick velocity limit.

        Hazard mapping: joint-limit clipping is the software half of H4 (servo overload at the mechanical stop);
        velocity clipping bounds kinetic energy for H5 (human in workspace) and H8 (collision with a fixture).
        In REDUCED_SPEED the allowed delta is additionally scaled by cfg.reduced_speed_factor. In HOLD or ESTOP
        the returned target is the current pose (no motion).
        """
        q_t = np.asarray(q_target, dtype=float).copy()
        q_n = np.asarray(q_now, dtype=float)
        dt = self.cfg.control_dt if dt is None else float(dt)
        reasons: list[str] = []
        if self.state in (SafetyState.HOLD, SafetyState.ESTOP):
            reasons.append(f"{self.state.value}: holding current pose")
            return q_n.copy(), reasons
        if not np.all(np.isfinite(q_t)):
            self.hold("nan_command", "non-finite joint target")
            reasons.append("non-finite joint target -> HOLD")
            return q_n.copy(), reasons
        clipped = np.clip(q_t, self.lo, self.hi)
        if np.any(np.abs(clipped - q_t) > 1e-9):
            bad = np.where(np.abs(clipped - q_t) > 1e-9)[0].tolist()
            reasons.append(f"joint limit clip on joints {bad}")
            self.n_clipped_limit += 1
        max_delta = self.cfg.max_joint_vel * dt
        if self._reduced_speed:
            max_delta *= self.cfg.reduced_speed_factor
        delta = clipped - q_n
        over = np.abs(delta) > max_delta + 1e-12
        if np.any(over):
            delta = np.clip(delta, -max_delta, max_delta)
            reasons.append(f"velocity clip on joints {np.where(over)[0].tolist()} (max {max_delta:.4f} rad/tick)")
            self.n_clipped_vel += 1
        return q_n + delta, reasons

    # ------------------------------------------------------------------ H8 / H5: geofence
    def check_tcp(self, tcp_pos) -> list[str]:
        """Geofence check of the tool centre point in world coordinates. A violation enters HOLD (recoverable
        once the TCP is back inside or an operator resets). Hazard mapping: H8 (fixture collision / calibration
        drift pushing the arm outside the taught volume) and H5 (keeps the arm out of the human's approach zone).
        """
        p = np.asarray(tcp_pos, dtype=float)
        reasons: list[str] = []
        if not np.all(np.isfinite(p)):
            reasons.append("tcp position non-finite")
        else:
            below = p < self.geo_lo
            above = p > self.geo_hi
            for i, ax in enumerate("xyz"):
                if below[i] or above[i]:
                    reasons.append(f"tcp {ax}={p[i]:.3f} outside geofence [{self.geo_lo[i]:.3f}, {self.geo_hi[i]:.3f}]")
        if reasons:
            self.hold("geofence", "; ".join(reasons))
        else:
            self.release("geofence")
        return reasons

    # ------------------------------------------------------------------ H1 / H6: grounding gate
    def gate_grounding(self, score, n_candidates: int) -> tuple[bool, str]:
        """Decide whether the language grounding result may be executed.

        ``score`` is the top-1 confidence, or a sequence of candidate scores (top-1 first, or unsorted).
        Hazard mapping: H1 (wrong-object grounding) via the confidence threshold and H6 (ambiguous language)
        via the top-2 margin and the human-confirmation option.
        """
        scores = np.atleast_1d(np.asarray(score, dtype=float))
        if scores.size == 0 or n_candidates <= 0:
            return False, "no candidates"
        s = np.sort(scores)[::-1]
        top = float(s[0])
        if not np.isfinite(top):
            return False, "non-finite grounding score"
        if top < self.cfg.grounding_conf_threshold:
            return False, f"grounding confidence {top:.3f} < {self.cfg.grounding_conf_threshold:.2f}"
        if s.size >= 2 and (top - float(s[1])) < self.cfg.ambiguity_margin:
            return False, f"ambiguous: top-2 scores {top:.3f} / {float(s[1]):.3f} within {self.cfg.ambiguity_margin:.2f}"
        if self.cfg.require_human_confirm and not self._human_confirmed:
            return False, "waiting for human confirmation"
        return True, "ok"

    # ------------------------------------------------------------------ H7: policy confidence
    def gate_policy(self, confidence: float) -> tuple[bool, str]:
        """H7 (policy out-of-distribution): below the threshold the policy may not act; the caller is expected
        to fall back to the scripted modular pipeline or to HOLD."""
        c = float(confidence)
        if not np.isfinite(c) or c < self.cfg.policy_conf_threshold:
            return False, f"policy confidence {c:.3f} < {self.cfg.policy_conf_threshold:.2f}"
        return True, "ok"

    # ------------------------------------------------------------------ reporting
    def diagnostics(self) -> dict:
        """diagnostic_msgs/DiagnosticStatus-like dict: name, level, message, values (list of {key, value})."""
        state = self.state
        level = {
            SafetyState.RUN: LEVEL_OK,
            SafetyState.REDUCED_SPEED: LEVEL_WARN,
            SafetyState.HOLD: LEVEL_WARN,
            SafetyState.ESTOP: LEVEL_ERROR,
        }[state]
        if state == SafetyState.ESTOP:
            message = self._estop_reason or "estop"
        elif state == SafetyState.HOLD:
            message = "; ".join(self._hold_reasons.values())
        elif state == SafetyState.REDUCED_SPEED:
            message = self._reduced_reason or "reduced speed"
        else:
            message = "ok"
        values = [
            {"key": "state", "value": state.value},
            {"key": "reduced_speed", "value": str(self._reduced_speed)},
            {"key": "n_clipped_limit", "value": str(self.n_clipped_limit)},
            {"key": "n_clipped_vel", "value": str(self.n_clipped_vel)},
        ]
        for topic in self.cfg.staleness_s:
            last = self._last_seen.get(topic)
            values.append({"key": f"last_{topic}", "value": "never" if last is None else f"{last:.3f}"})
        return {"name": "langgrasp/safety", "level": level, "message": message, "values": values}
