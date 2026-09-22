"""One test per hazard mechanism of docs/FMEA.md plus a real-sim integration test."""

import numpy as np
import pytest

from langgrasp.safety import SafetyConfig, SafetyMonitor, SafetyState


def make(**kw):
    return SafetyMonitor(SafetyConfig(**kw))


# H2 / H3: sensor dropout -> ESTOP
def test_stale_camera_topic_latches_estop():
    m = make()
    m.heartbeat("camera", 0.0)
    m.heartbeat("joint_states", 0.0)
    m.heartbeat("command", 0.0)
    assert m.check_staleness(0.1) == []
    assert m.state == SafetyState.RUN
    m.heartbeat("joint_states", 0.9)
    stale = m.check_staleness(1.0)
    assert "camera" in stale
    assert m.state == SafetyState.ESTOP
    assert "camera stale" in m.estop_reason


def test_stale_command_only_holds():
    m = make()
    t = 0.0
    for topic in ("camera", "joint_states", "command"):
        m.heartbeat(topic, t)
    m.heartbeat("camera", 31.0)
    m.heartbeat("joint_states", 31.0)
    assert m.check_staleness(31.0) == ["command"]
    assert m.state == SafetyState.HOLD
    m.heartbeat("command", 31.0)
    assert m.check_staleness(31.05) == []
    assert m.state == SafetyState.RUN


# H4: joint limits
def test_joint_limit_clip():
    m = make(max_joint_vel=100.0)
    q_now = np.zeros(5)
    q_target = np.array([5.0, -5.0, 0.0, 0.0, 0.0])
    q, reasons = m.check_joint_command(q_target, q_now, dt=0.1)
    assert np.isclose(q[0], m.lo[0] if m.lo[0] > -5 else 5) or np.isclose(q[0], m.hi[0])
    assert np.all(q >= m.lo - 1e-9) and np.all(q <= m.hi + 1e-9)
    assert any("joint limit" in r for r in reasons)
    assert m.n_clipped_limit == 1


# H5 / H8: velocity clip
def test_velocity_clip():
    m = make(max_joint_vel=1.0)
    q_now = np.zeros(5)
    q_target = np.array([0.5, -0.5, 0.05, 0.0, 0.0])
    q, reasons = m.check_joint_command(q_target, q_now, dt=0.1)
    np.testing.assert_allclose(q, [0.1, -0.1, 0.05, 0.0, 0.0])
    assert any("velocity clip" in r for r in reasons)
    q2, reasons2 = m.check_joint_command(np.array([0.05, 0, 0, 0, 0]), q_now, dt=0.1)
    assert reasons2 == [] and np.isclose(q2[0], 0.05)


# H8 / H5: geofence
def test_geofence_violation_holds_and_recovers():
    m = make()
    assert m.check_tcp([0.0, -0.2, 0.05]) == []
    assert m.state == SafetyState.RUN
    reasons = m.check_tcp([0.0, 0.3, 0.05])
    assert reasons and "outside geofence" in reasons[0]
    assert m.state == SafetyState.HOLD
    q, r = m.check_joint_command(np.ones(5) * 0.3, np.zeros(5))
    np.testing.assert_allclose(q, np.zeros(5))
    assert "HOLD" in r[0]
    assert m.check_tcp([0.0, -0.2, 0.05]) == []
    assert m.state == SafetyState.RUN


# H1: grounding confidence
def test_low_grounding_confidence_not_allowed():
    m = make()
    ok, reason = m.gate_grounding(0.2, n_candidates=3)
    assert not ok and "confidence" in reason
    ok, reason = m.gate_grounding([0.8, 0.3], n_candidates=2)
    assert ok and reason == "ok"
    assert m.gate_grounding([], n_candidates=0) == (False, "no candidates")


# H6: ambiguity
def test_ambiguity_flag():
    m = make()
    ok, reason = m.gate_grounding([0.61, 0.58, 0.1], n_candidates=3)
    assert not ok and "ambiguous" in reason
    ok, _ = m.gate_grounding([0.61, 0.50], n_candidates=2)
    assert ok


def test_require_human_confirm():
    m = make(require_human_confirm=True)
    ok, reason = m.gate_grounding(0.9, n_candidates=1)
    assert not ok and "confirmation" in reason
    m.confirm_human()
    assert m.gate_grounding(0.9, n_candidates=1)[0]


# H7: policy confidence
def test_policy_gate():
    m = make(policy_conf_threshold=0.5)
    assert not m.gate_policy(0.4)[0]
    assert m.gate_policy(0.6)[0]
    assert not m.gate_policy(float("nan"))[0]


# estop latch
def test_estop_latch_persists_until_reset():
    m = make()
    for topic in ("camera", "joint_states", "command"):
        m.heartbeat(topic, 0.0)
    m.estop("test")
    assert m.state == SafetyState.ESTOP
    # fresh heartbeats do not clear it
    for topic in ("camera", "joint_states", "command"):
        m.heartbeat(topic, 0.05)
    m.check_staleness(0.06)
    assert m.state == SafetyState.ESTOP
    q, r = m.check_joint_command(np.ones(5) * 0.1, np.zeros(5))
    np.testing.assert_allclose(q, np.zeros(5))
    m.reset_estop()
    assert m.state == SafetyState.RUN
    assert m.estop_reason is None


# reduced speed
def test_reduced_speed_scaling():
    m = make(max_joint_vel=1.0, reduced_speed_factor=0.5)
    q, _ = m.check_joint_command(np.array([1, 0, 0, 0, 0.0]), np.zeros(5), dt=0.1)
    assert np.isclose(q[0], 0.1)
    m.set_reduced_speed(True, "human near")
    assert m.state == SafetyState.REDUCED_SPEED
    q, reasons = m.check_joint_command(np.array([1, 0, 0, 0, 0.0]), np.zeros(5), dt=0.1)
    assert np.isclose(q[0], 0.05)
    assert any("velocity clip" in r for r in reasons)
    m.set_reduced_speed(False)
    assert m.state == SafetyState.RUN


def test_non_finite_command_holds():
    m = make()
    q, r = m.check_joint_command([np.nan, 0, 0, 0, 0], np.zeros(5))
    assert m.state == SafetyState.HOLD and np.all(np.isfinite(q))


# diagnostics
def test_diagnostics_format():
    m = make()
    d = m.diagnostics()
    assert set(d) == {"name", "level", "message", "values"}
    assert d["level"] == 0 and d["message"] == "ok"
    assert all(set(v) == {"key", "value"} and isinstance(v["value"], str) for v in d["values"])
    assert {v["key"] for v in d["values"]} >= {"state", "reduced_speed", "last_camera", "last_joint_states", "last_command"}
    m.estop("boom")
    d = m.diagnostics()
    assert d["level"] == 2 and d["message"] == "boom"
    assert dict((v["key"], v["value"]) for v in d["values"])["state"] == "ESTOP"


# integration with the real MuJoCo sim
@pytest.mark.parametrize("seeds", [(1000, 1001, 1002)])
def test_sim_pick_through_safety_monitor(seeds):
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from langgrasp.sim.controller import PickPlaceController
    from langgrasp.sim.env import LangGraspEnv
    from langgrasp.sim.scenarios import make_scenario
    from langgrasp.sim.scene import OBJECT_KINDS

    env = LangGraspEnv(seed=0, render=False)
    mon = SafetyMonitor(SafetyConfig(joint_lo=tuple(env.kin.lo), joint_hi=tuple(env.kin.hi)))
    orig_step = env.step
    log = {"ticks": 0, "reasons": 0}

    def guarded_step(q_arm=None, jaw=None):
        if q_arm is not None:
            mon.check_tcp(env.obs()["tcp_pos"])
            q_arm, reasons = mon.check_joint_command(q_arm, env.q_arm, dt=0.1)
            log["reasons"] += len(reasons)
        log["ticks"] += 1
        return orig_step(q_arm, jaw)

    env.step = guarded_step
    placed = 0
    for seed in seeds:
        sc = make_scenario(seed, "seen")
        env.reset(sc)
        p, psi = env.grasp_point(sc.target)
        r = PickPlaceController(env).run(p, psi, sc.target, width=OBJECT_KINDS[sc.target_obj.kind]["width"])
        placed += int(r.placed)
        assert mon.state in (SafetyState.RUN,), mon.diagnostics()
    assert placed >= 1, f"pick failed with the safety monitor in the loop for all seeds {seeds}"
    assert log["ticks"] > 0

    # a deliberately out-of-geofence target: the arm cannot reach +y at all, but moving toward it drives the TCP
    # toward the base (measured with velocity clipping: tcp y reaches -0.056), which a y<-0.10 fence must catch
    sc = make_scenario(seeds[0], "seen")
    env.reset(sc)
    mon2 = SafetyMonitor(SafetyConfig(joint_lo=tuple(env.kin.lo), joint_hi=tuple(env.kin.hi), geofence_hi=(0.28, -0.10, 0.20)))
    states = []
    tcps = []

    def guarded_step2(q_arm=None, jaw=None):
        if q_arm is not None:
            mon2.check_tcp(env.obs()["tcp_pos"])
            q_arm, _ = mon2.check_joint_command(q_arm, env.q_arm, dt=0.1)
        o = orig_step(q_arm, jaw)
        mon2.check_tcp(o["tcp_pos"])
        states.append(mon2.state)
        tcps.append(o["tcp_pos"].copy())
        return o

    env.step = guarded_step2
    ctl = PickPlaceController(env)
    ctl.psi = 0.0
    ctl.move_to(np.array([-0.1, 0.10, 0.05]), 1.2, "bad", strict=False)
    assert SafetyState.HOLD in states, np.array(tcps).round(3)
    assert mon2.state == SafetyState.HOLD
    # once in HOLD every further command is replaced by the current pose: the TCP must stop moving
    first_hold = states.index(SafetyState.HOLD)
    if first_hold + 2 < len(tcps):
        drift = np.linalg.norm(np.array(tcps[first_hold + 1 :]) - tcps[first_hold + 1], axis=1).max()
        assert drift < 0.01, f"arm kept moving after HOLD: drift {drift:.3f} m"
    env.close()
