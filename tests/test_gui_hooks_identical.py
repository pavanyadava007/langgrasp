"""The GUI instrumentation must not change what the pipeline decides.

The GUI adds optional hooks to ``ModularPipeline``. If they changed any output, every number in
``docs/RESULTS.md`` would become unreproducible from the GUI, so this file runs the same seeds twice, once
with the hooks off and once with a recording hook attached, and demands identical results.

Latency values are deliberately excluded from the comparison: instrumentation costs time (the queue puts and
the JSON conversion), which is why an instrumented run is never written into ``results/``. The set of timed
stages is compared, the durations are not.
"""

import numpy as np
import pytest

from langgrasp.gui.trace import STAGES, RecordingHooks
from langgrasp.perception.grounding import OracleGrounder
from langgrasp.policies.modular import ModularPipeline, PipelineConfig
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario

SEEDS = [(5000, "seen"), (5001, "seen"), (5002, "seen"), (5003, "seen"), (6000, "unseen"), (6001, "unseen"), (6002, "unseen"), (7000, "langvar"), (7001, "langvar"), (7002, "langvar")]


def run_seeds(hooks, seeds=SEEDS, grounding_threshold=0.35):
    """Run the pipeline over fixed scenes and return one comparable dict per scene."""
    env = LangGraspEnv(seed=0)
    try:
        pipe = ModularPipeline(
            env,
            OracleGrounder(env),
            None,
            PipelineConfig(depth_noise=True, grounding_threshold=grounding_threshold),
            seed=0,
            hooks=hooks,
        )
        out = []
        for seed, stratum in seeds:
            sc = make_scenario(seed, stratum)
            env.reset(sc)
            res = pipe.run_command(sc.command, sc.target)
            d = res.to_dict()
            d["latency_stages"] = sorted(d.pop("latency_ms"))
            out.append(d)
        return out
    finally:
        env.close()


@pytest.fixture(scope="module")
def paired_runs():
    off = run_seeds(None)
    hooks = RecordingHooks()
    on = run_seeds(hooks)
    return off, on, hooks


def test_results_are_identical_with_and_without_hooks(paired_runs):
    off, on, _ = paired_runs
    assert len(off) == len(SEEDS)
    for a, b in zip(off, on, strict=True):
        assert a == b, f"hooks changed the outcome of seed {a['command']!r}: {a} != {b}"


def test_something_actually_happened_in_those_runs(paired_runs):
    """A vacuous comparison would pass even if the pipeline aborted every scene."""
    off, _, _ = paired_runs
    placed = sum(1 for d in off if d["placed"])
    assert placed >= 8, f"only {placed}/10 scenes placed with the oracle grounder, the comparison is not meaningful"
    # the oracle grounder scores by kind word, so a synonym command ("driver") can still ground wrongly;
    # what matters here is that most scenes ran the whole pipeline rather than aborting early
    assert sum(1 for d in off if d["grounding_correct"]) >= 8
    assert all(d["mask_source"] == "oracle" for d in off)
    assert all(d["aborted"] is None for d in off)


def test_hooks_saw_every_stage(paired_runs):
    _, _, hooks = paired_runs
    stages = set(hooks.stages)
    expected = set(STAGES) - {"command"}  # the worker emits "command", the pipeline does not
    assert expected <= stages, f"missing stage events: {sorted(expected - stages)}"
    assert len(hooks.stages) >= 8 * len(SEEDS)


def test_stage_payloads_say_what_the_code_decided(paired_runs):
    _, _, hooks = paired_runs
    cap = hooks.payload("capture")
    assert cap["camera"] == "front" and cap["depth_noise"] is True and cap["fx_px"] > 0
    g = hooks.payload("grounding")
    assert g["candidates"] and {"box", "score", "label", "color_frac", "area_px"} == set(g["candidates"][0])
    sel = hooks.payload("select")
    assert sel["winner"] is not None and "n_candidates" in sel and sel["rule"]
    gate = hooks.payload("gate")
    # this fixture runs without a SafetyMonitor, so the gate is the pipeline's own threshold
    assert gate["decision"] == "allow" and gate["gate_input"].startswith("pipeline threshold")
    seg = hooks.payload("segment")
    assert seg["mask_source"] == "oracle" and seg["mask_px"] > 0
    fuse = hooks.payload("fuse")
    assert fuse["n_points_object"] >= 20 and fuse["n_points_backprojected"] >= fuse["n_points_object"]
    assert len(fuse["points_xyz"]) <= 400 and len(fuse["points_xyz"][0]) == 3
    ex = hooks.payload("execute")
    assert ex["steps"] > 0 and np.isfinite(ex["psi_used"]) and isinstance(ex["ik_ok"], bool)


def test_abort_path_is_identical_too():
    """A refusal is an outcome as well: the gate must refuse the same scenes with hooks on."""
    seeds = [(7000, "langvar"), (7001, "langvar")]
    off = run_seeds(None, seeds, grounding_threshold=0.9)
    hooks = RecordingHooks()
    on = run_seeds(hooks, seeds, grounding_threshold=0.9)
    assert off == on
    aborted = [d["aborted"] for d in off]
    assert any(a == "low_confidence" for a in aborted), f"expected a refusal at threshold 0.9, got {aborted}"
    assert hooks.payload("gate")["decision"] == "refuse"


def test_controller_factory_is_used_when_given():
    """The GUI builds the controller so it can attach record_fn; without a factory nothing changes."""
    from langgrasp.sim.controller import PickPlaceController

    env = LangGraspEnv(seed=0)
    try:
        calls = []

        def factory(e):
            calls.append(e)
            return PickPlaceController(e, record=True, record_fn=lambda _e, phase: calls.append(phase) or {})

        pipe = ModularPipeline(env, OracleGrounder(env), None, PipelineConfig(depth_noise=False), seed=0, controller_factory=factory)
        sc = make_scenario(5000, "seen")
        env.reset(sc)
        res = pipe.run_command(sc.command, sc.target)
        assert res.placed and calls and calls[0] is env
        assert "descend" in calls and "lift" in calls
    finally:
        env.close()


def test_gate_payload_with_the_safety_monitor_states_what_it_was_given():
    """FMEA H6 credits the monitor's top-2 ambiguity margin. On this code path the pipeline passes a single
    score, so the margin cannot fire, and the payload has to say so rather than implying coverage."""
    from langgrasp.safety import SafetyConfig, SafetyMonitor

    env = LangGraspEnv(seed=0)
    try:
        hooks = RecordingHooks()
        pipe = ModularPipeline(
            env,
            OracleGrounder(env),
            None,
            PipelineConfig(depth_noise=False),
            safety=SafetyMonitor(SafetyConfig(grounding_conf_threshold=0.30)),
            seed=0,
            hooks=hooks,
        )
        sc = make_scenario(5000, "seen")
        env.reset(sc)
        pipe.run_command(sc.command, sc.target)
        gate = hooks.payload("gate")
        assert gate["decision"] == "allow" and gate["threshold"] == 0.30
        assert "top-1 score only" in gate["gate_input"]
        assert gate["ambiguity_margin"] == 0.05
        assert "select_flagged_ambiguous" in gate
    finally:
        env.close()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="Grounding DINO needs a GPU; the oracle-grounder test above covers the plumbing")
def test_real_grounder_decisions_are_unaffected_by_the_hooks():
    """The same check on the real model path: Grounding DINO tiny, the colour check and a YOLO mask if built.

    Grounding DINO on CUDA is not bit-reproducible: two identical runs differ in the last few digits of the
    box coordinates and the score, because cuDNN is free to pick different kernels. So this test runs the
    pipeline three times (twice uninstrumented, once with hooks) and demands two things: every decision is
    identical, and the numeric wobble introduced by the hooks is no larger than the wobble the GPU produces
    on its own. Three seeds only, because this loads a 660 MB checkpoint and runs it once per command.
    """
    import os

    from langgrasp.perception.grounding import GroundingDINO

    seeds = [(5000, "seen"), (6000, "unseen"), (7000, "langvar")]
    seg_path = "checkpoints/yolo11n-seg-langgrasp.pt"
    decisions = ("grounding_correct", "grasped", "lifted", "placed", "aborted", "mask_source", "intent", "select_info", "command")
    env = LangGraspEnv(seed=0)
    try:
        grounder = GroundingDINO()
        grounder.warmup()
        segmenter = None
        if os.path.exists(seg_path):
            from langgrasp.perception.segmentation import Segmenter

            segmenter = Segmenter(seg_path, backend="pt")
            segmenter.warmup()

        def run(hooks):
            pipe = ModularPipeline(env, grounder, segmenter, PipelineConfig(depth_noise=True), seed=0, hooks=hooks)
            out = []
            for seed, stratum in seeds:
                sc = make_scenario(seed, stratum)
                env.reset(sc)
                d = pipe.run_command(sc.command, sc.target).to_dict()
                d.pop("latency_ms")
                out.append(d)
            return out

        base1, base2 = run(None), run(None)
        hooks = RecordingHooks()
        instr = run(hooks)

        def spread(a, b):
            """Largest absolute difference in the numbers the grounder produced."""
            box = max(abs(x - y) for da, db in zip(a, b, strict=True) for x, y in zip(da["box"], db["box"], strict=True))
            score = max(abs(da["score"] - db["score"]) for da, db in zip(a, b, strict=True))
            return box, score

        for i, (a, b, c) in enumerate(zip(base1, base2, instr, strict=True)):
            for key in decisions:
                assert a[key] == b[key] == c[key], f"seed {seeds[i]} disagreed on {key}: {a[key]} / {b[key]} / {c[key]}"
            for da, db in ((a, b), (a, c)):
                if da["grasp"] and db["grasp"]:
                    assert np.allclose(da["grasp"]["center"], db["grasp"]["center"], atol=1e-6)
                    assert abs(da["grasp"]["width"] - db["grasp"]["width"]) < 1e-6

        gpu_box, gpu_score = spread(base1, base2)
        hook_box, hook_score = spread(base1, instr)
        assert hook_box <= max(gpu_box, 1e-3) * 10, f"hooks moved the box by {hook_box} px, the GPU alone moves it by {gpu_box} px"
        assert hook_score <= max(gpu_score, 1e-4) * 10, f"hooks moved the score by {hook_score}, the GPU alone moves it by {gpu_score}"
        assert sum(1 for d in base1 if d["placed"]) >= 2, [d["aborted"] for d in base1]
        if segmenter is not None:
            assert any(d["mask_source"].startswith("yolo") for d in base1)
    finally:
        env.close()
