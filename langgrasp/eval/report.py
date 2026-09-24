"""Generate docs/RESULTS.md from results/*.json. Never hand-edit RESULTS.md: every number here is read from a
JSON written by a script that measured it. Missing files produce "not run" rows, never placeholders."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def _load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _ci(cell: dict | None) -> str:
    if not cell or cell.get("n", 0) == 0:
        return "not run"
    return f"{100 * cell['p']:.1f}% [{100 * cell['lo']:.0f}-{100 * cell['hi']:.0f}] (n={cell['n']})"


def _lat(s: dict | None, key: str = "median_ms") -> str:
    if not s or "median_ms" not in s or s.get("n", 1) == 0:
        return "-"
    return f"{s['median_ms']:.0f} / {s['p90_ms']:.0f} / {s['p99_ms']:.0f}"


def protocol_table(files: dict[str, str]) -> list[str]:
    rows = ["| Approach | Seen place | Unseen place | Lang-var place | Grounding acc. (all) | E2E latency med/p90/p99 ms | n |", "|---|---|---|---|---|---|---|"]
    for label, fname in files.items():
        d = _load(fname)
        if d is None:
            rows.append(f"| {label} | not run | not run | not run | not run | - | 0 |")
            continue
        s = d["summary"]["strata"]
        g = s.get("all", {}).get("grounding")
        lat = d["summary"].get("latency_ms", {}).get("end_to_end")
        rows.append(f"| {label} | {_ci(s.get('seen', {}).get('place'))} | {_ci(s.get('unseen', {}).get('place'))} | {_ci(s.get('langvar', {}).get('place'))} | {_ci(g) if g and g.get('n') else 'n/a'} | {_lat(lat)} | {d['summary']['n_trials']} |")
    return rows


def fixed_goal_table(files: dict[str, str]) -> list[str]:
    rows = ["| Approach | Grasp (lifted) | Place | Grounding acc. | Per-command latency med ms | n |", "|---|---|---|---|---|---|"]
    for label, fname in files.items():
        d = _load(fname)
        if d is None:
            rows.append(f"| {label} | not run | not run | - | - | 0 |")
            continue
        s = d["summary"]["strata"]["all"]
        g = s.get("grounding")
        lat = d["summary"].get("latency_ms", {})
        lat_cell = "-"
        for key, lname in (("end_to_end", "end_to_end"), ("policy_call", "policy_call"), ("episode", "episode, sim compute")):
            if key in lat and lat[key].get("n"):
                lat_cell = f"{lat[key]['median_ms']:.0f} ({lname})"
                break
        rows.append(f"| {label} | {_ci(s.get('grasp'))} | {_ci(s.get('place'))} | {_ci(g) if g and g.get('n') else 'n/a'} | {lat_cell} | {d['summary']['n_trials']} |")
    return rows


def breakdown_table(fname: str, key: str, title: str) -> list[str]:
    d = _load(fname)
    if d is None or not d["summary"].get(key):
        return []
    rows = [f"**{title}**", "", "| Group | Grasp (lifted) | Place | Grounding |", "|---|---|---|---|"]
    for k, v in d["summary"][key].items():
        rows.append(f"| {k} | {_ci(v.get('grasp'))} | {_ci(v.get('place'))} | {_ci(v.get('grounding')) if v.get('grounding') and v['grounding'].get('n') else 'n/a'} |")
    rows.append("")
    return rows


def latency_table(fname: str) -> list[str]:
    d = _load(fname)
    if d is None:
        return ["not run"]
    lat = d["summary"].get("latency_ms", {})
    rows = ["| Stage | median ms | p90 ms | p99 ms | n |", "|---|---|---|---|---|"]
    for k in ["parse", "capture", "grounding", "segmentation", "depth_fusion", "execute", "end_to_end"]:
        if k in lat and lat[k].get("n"):
            s = lat[k]
            rows.append(f"| {k} | {s['median_ms']:.1f} | {s['p90_ms']:.1f} | {s['p99_ms']:.1f} | {s['n']} |")
    return rows


def yolo_section() -> list[str]:
    out = []
    tr = _load("yolo_train.json")
    if tr:
        out += ["**YOLO11n-seg fine-tune (synthetic sim data)**", ""]
        out += [f"- train/val images: {tr.get('train_images')}/{tr.get('val_images')}, epochs {tr.get('epochs')}, {tr.get('minutes', 0):.0f} min on {tr.get('hardware')}"]
        out += [f"- box mAP50 {tr.get('box_map50')}, box mAP50-95 {tr.get('box_map50_95')}, mask mAP50 {tr.get('mask_map50')}, mask mAP50-95 {tr.get('mask_map50_95')}", ""]
    lt = _load("yolo_latency_l4.json")
    if lt:
        out += ["**YOLO11n-seg latency, batch 1, 640 input (" + str(lt.get("hardware", "L4")) + ")**", "", "| Backend | inference med/p90/p99 ms | end-to-end med/p90/p99 ms | box mAP50 | mask mAP50 |", "|---|---|---|---|---|"]
        mp = _load("yolo_export_map.json") or {}
        maps = mp.get("backends", mp) if isinstance(mp, dict) else {}
        for name, v in lt.get("backends", lt).items():
            if not isinstance(v, dict) or "inference_only" not in v:
                continue
            m = maps.get(name, {}) if isinstance(maps, dict) else {}
            fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
            out.append(f"| {name} | {_lat(v.get('inference_only'))} | {_lat(v.get('end_to_end'))} | {fmt(m.get('box_map50'))} | {fmt(m.get('mask_map50'))} |")
        out.append("")
    df = _load("depth_fusion_accuracy.json")
    if df:
        out += ["**Depth fusion / grasp-pose accuracy vs simulator ground truth**", "", "```", json.dumps({k: v for k, v in df.items() if k != "per_scene"}, indent=1)[:1500], "```", ""]
    return out


def pipeline_bench_table(fname: str) -> list[str]:
    d = _load(fname)
    if d is None:
        return ["not run"]
    rows = [f"Hardware: {d.get('hardware')}. {d.get('note', '')}", "", "| Segmenter backend | parse | capture (render) | grounding (GDINO tiny) | segmentation | depth fusion | perception total | executor (sim compute) |", "|---|---|---|---|---|---|---|---|"]
    for name, v in d.get("backends", {}).items():
        if "error" in v:
            rows.append(f"| {name} | error: {v['error'][:60]} | | | | | | |")
            continue
        cells = [_lat(v.get(k)) for k in ["parse", "capture", "grounding", "segmentation", "depth_fusion", "perception_total", "execute_sim"]]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("Cells are median / p90 / p99 in ms.")
    return rows


def gap_table(fname: str, task: str) -> list[str]:
    d = _load(fname)
    if d is None:
        return [f"**PPO {task}**: not run", ""]
    rows = [f"**PPO {task}: {d.get('label', '')}** (`results/{fname}`, {d.get('episodes_per_cell')} episodes per cell)", "", "| Checkpoint | Condition | Success [95% CI] |", "|---|---|---|"]
    for r in d.get("rows", []):
        lo, hi = r["ci95"]
        rows.append(f"| {r['checkpoint'].split('/')[-1]} | {r['condition']} | {r['k']}/{r['n']} = {100 * r['success']:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}] |")
    rows.append("")
    return rows


def _cell(r: dict | None) -> str:
    if r is None:
        return "not run"
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} = {100 * r['success']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _hold(r: dict | None) -> str:
    if r is None or "hold" not in r:
        return "-"
    return f"{100 * r['hold']:.1f}%"


def _first(curve: list[dict], key: str, level: float) -> dict | None:
    return next((p for p in curve if p[key] >= level), None)


def _steps_min(p: dict | None) -> str:
    return "not reached" if p is None else f"{p['steps']:,} ({p['minutes']:.1f} min)"


REACH_METHODS = [
    # label, gap json, row filter, training json
    ("PPO, no DR", "ppo_sim2sim_gap.json", lambda r: r.get("task") == "reach" and not r.get("train_dr"), "ppo_reach_nodr.json"),
    ("PPO, DR", "ppo_sim2sim_gap.json", lambda r: r.get("task") == "reach" and r.get("train_dr"), "ppo_reach_dr.json"),
    ("SAC (SB3), no DR", "sac_sim2sim_gap.json", lambda r: not r.get("train_dr"), "sac_reach_nodr.json"),
    ("SAC (SB3), DR", "sac_sim2sim_gap.json", lambda r: r.get("train_dr"), "sac_reach_dr.json"),
    ("OSC (Jacobian + mass matrix, no learning)", "osc_sim2sim_gap.json", lambda r: True, None),
    ("MPC (MPPI on the nominal MuJoCo model)", "mpc_sim2sim_gap.json", lambda r: True, None),
]
CONDS = ("nominal", "shifted", "shifted_latency2")


def reach_comparison() -> list[str]:
    """One table for all reach methods, read from the gap JSONs; missing files give 'not run' rows."""
    out = ["All cells: successes/episodes = rate [Wilson 95% CI], 200 deterministic episodes per cell, eval seed 12345, "
           "same start poses, targets and noise draws for every method. Sim-to-sim gap, no real robot. "
           "Hardware: NVIDIA L4 host / CPU, simulation. MPC nominal is privileged (exact model and state).", "",
           "| Method | Nominal | Shifted (1-tick) | Shifted + 2-tick latency | Source |", "|---|---|---|---|---|"]
    holds = ["| Method | Hold nominal | Hold shifted | Hold shifted + 2-tick |", "|---|---|---|---|"]
    for label, fname, keep, _ in REACH_METHODS:
        d = _load(fname)
        rows = {r["condition"]: r for r in d.get("rows", []) if keep(r)} if d else {}
        out.append(f"| {label} | " + " | ".join(_cell(rows.get(c)) for c in CONDS) + f" | `results/{fname}`{'' if d else ' (not run)'} |")
        holds.append(f"| {label} | " + " | ".join(_hold(rows.get(c)) for c in CONDS) + " |")
    out += ["", "Final-tick hold rate (true TCP within 1.5 cm at the last tick) in the same episodes:", ""] + holds + [""]
    # training budget and time to the PPO level
    out += ["**Training budget and time to the PPO success level (reach)**", "",
            "Train-env milestones: first block of 4096 env steps (one PPO iteration) whose finished episodes reach the "
            "success rate, stochastic policy, in the training env (DR on or off). Nominal-eval milestone: first periodic "
            "200-episode deterministic evaluation in the nominal sim (seed 999, every 20k steps) at 200/200; SAC only, "
            "PPO was evaluated only at the end.", "",
            "| Method | Env steps (total) | Gradient steps | Wall min | Device | Train-env >= 0.5 | >= 0.9 | >= 0.99 | Nominal eval 200/200 | Source |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for label, _, _, tname in REACH_METHODS:
        if tname is None:
            continue
        t = _load(tname)
        if t is None:
            out.append(f"| {label} | not run | | | | | | | | `results/{tname}` |")
            continue
        if "curve_detail" in t:  # PPO
            curve = t["curve_detail"]
            grads = t["iterations"] * t["config"]["epochs"] * -(-t["n_envs"] * t["n_steps_per_env"] // t["config"]["minibatch"])
            nominal = "-"
            dev = f"{t.get('device')} (update), CPU physics"
        else:
            curve = t["train_curve"]
            grads = t["gradient_steps"]
            m = t["milestones"].get("nominal_eval_success_ge_1.0")
            nominal = "not reached" if m is None else f"{m['steps']:,} ({m['minutes']:.1f} min)"
            dev = t.get("device", "-")
        ms = [_steps_min(_first(curve, "success_rate", lv)) for lv in (0.5, 0.9, 0.99)]
        out.append(f"| {label} | {t['total_steps']:,} | {grads:,} | {t['minutes']:.1f} | {dev} | " + " | ".join(ms) + f" | {nominal} | `results/{tname}` |")
    out.append("")
    # control compute
    out += ["**Per-step compute of the controllers (one act() call, control period 100 ms)**", "",
            "| Method | Single env, nominal: median / p90 / p99 ms | 50-env batch per call: median ms | Source |", "|---|---|---|---|"]
    for label, fname in (("OSC", "osc_sim2sim_gap.json"), ("MPC (MPPI)", "mpc_sim2sim_gap.json"), ("SAC policy (batch only)", "sac_sim2sim_gap.json")):
        d = _load(fname)
        if d is None:
            out.append(f"| {label} | not run | not run | `results/{fname}` |")
            continue
        st = d.get("single_env_timing", {}).get("per_tick_ms")
        single = f"{st['median']:.2f} / {st['p90']:.2f} / {st['p99']:.2f}" if st else "-"
        nom = next((r for r in d["rows"] if r["condition"] == "nominal"), None)
        batch = f"{nom['act_batch_ms']['median']:.1f}" if nom and "act_batch_ms" in nom else "-"
        out.append(f"| {label} | {single} | {batch} | `results/{fname}` |")
    mp = _load("mpc_sim2sim_gap.json")
    if mp:
        out += ["", f"MPC parameters: {json.dumps(mp.get('params'))}, {mp.get('physics_steps_per_tick'):,} physics steps rolled out per control tick per env, "
                f"{mp.get('rollout_threads')} rollout threads; parameters chosen by `scripts/tune_model_based.py` on the nominal sim, tuning seed 777 (`results/model_based_tuning.json`)."]
    oc = _load("osc_sim2sim_gap.json")
    if oc:
        out += [f"OSC parameters: {json.dumps(oc.get('params'))}, chosen the same way."]
    out.append("")
    return out


def shift_factor_table(fname: str = "reach_shift_factors.json") -> list[str]:
    d = _load(fname)
    if d is None:
        return ["**Single-factor shift diagnostic**: not run", ""]
    factors = list(d["factors"])
    methods = list(dict.fromkeys(r["method"] for r in d["rows"]))
    cell = {(r["method"], r["condition"]): r for r in d["rows"]}
    out = [f"**Single-factor shift diagnostic (reach)** (`results/{fname}`): each part of the shifted condition alone, "
           f"{d.get('episodes_per_cell')} episodes per cell, eval seed {d.get('eval_seed')}; success rate [Wilson 95% CI] / final-tick hold. "
           "Not used to tune anything. Sim-to-sim gap, no real robot.", "",
           "| Method | " + " | ".join(factors) + " |", "|---|" + "---|" * len(factors)]
    for m in methods:
        cells = []
        for f in factors:
            r = cell.get((m, f))
            cells.append("not run" if r is None else f"{100 * r['success']:.1f}% [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}] / {100 * r['hold']:.1f}%")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def generic_json_section(fname: str, title: str, keys: list[str] | None = None) -> list[str]:
    d = _load(fname)
    if d is None:
        return [f"**{title}**: not run", ""]
    if keys:
        d = {k: d[k] for k in keys if k in d}
    return [f"**{title}** (`results/{fname}`)", "", "```", json.dumps(d, indent=1)[:2500], "```", ""]


def build() -> str:
    lines = ["# Results", "", "Generated by `python -m langgrasp.eval.report` from `results/*.json`. Every number was measured on this machine: NVIDIA L4 GPU, x86 EC2 host, MuJoCo simulation. No Jetson and no real SO-101 arm were available; nothing below is a hardware measurement.", ""]
    lines += ["## 1. Evaluation protocol (stratified, Wilson 95% CI)", ""]
    lines += protocol_table({
        "Oracle executor (ground-truth grasp, no perception)": "oracle_protocol.json",
        "Modular: Grounding DINO + colour check + YOLO11-seg + depth fusion": "modular_protocol.json",
        "Modular, no colour check (ablation)": "modular_nocolor_protocol.json",
        "Modular, box-only mask (no YOLO, ablation)": "modular_noyolo_protocol.json",
    })
    lines.append("")
    lines += breakdown_table("modular_protocol.json", "lang_variants", "Modular pipeline by language variant")
    lines += breakdown_table("modular_protocol.json", "lighting", "Modular pipeline by lighting")
    lines += breakdown_table("modular_protocol.json", "kinds", "Modular pipeline by object kind")
    lines += breakdown_table("oracle_protocol.json", "kinds", "Oracle executor by object kind")
    d = _load("modular_protocol.json")
    if d and d["summary"].get("aborts"):
        lines += ["Aborts (modular): " + ", ".join(f"{k}: {v}" for k, v in d["summary"]["aborts"].items()), ""]
    lines += ["## 2. Fixed-goal comparison on identical scenes (red cube -> tray, 100 seeds)", ""]
    lines += fixed_goal_table({
        "Oracle executor (ground-truth grasp point)": "oracle_fixed_goal.json",
        "Modular pipeline": "modular_fixed_goal.json",
        "ACT attempt 1 (120 demos, 128 px, 20k steps)": "act_eval.json",
        "ACT attempt 2 (240 demos, 192 px, 25k steps)": "act_eval_192.json",
        "ACT attempt 3 (700 demos, 192 px cropped, jitter, 30k steps)": "act_eval_v3.json",
        "ACT attempt 3 resumed to 60k steps": "act_eval_v3_60k.json",
    })
    lines.append("")
    lines += ["## 3. Latency budget of the modular pipeline (per command, L4)", "", "Clean benchmark (`scripts/bench_pipeline.py`, no other GPU job running):", ""] + pipeline_bench_table("pipeline_latency_l4.json") + ["", "Stage timings recorded during the protocol runs above (these ran while ACT/YOLO training shared the GPU, so they are upper bounds; `execute` is simulation compute for the motion, not robot motion time):", ""] + latency_table("modular_protocol.json") + [""]
    lines += ["## 4. Perception stack", ""] + yolo_section()
    lines += ["## 5. ACT", ""] + generic_json_section("act_train.json", "ACT training", ["steps", "batch", "minutes", "final_loss", "config", "hardware"]) + generic_json_section("demos_act.json", "Demo collection")
    lines += ["## 6. Reinforcement learning (PPO, state-based, sim-to-sim gap; no real robot)", ""] + gap_table("ppo_sim2sim_gap.json", "reach") + gap_table("ppo_sim2sim_gap_lift.json", "lift")
    lines += ["### Reach: PPO vs SAC vs OSC vs MPC (sim-to-sim gap, no real robot)", ""] + reach_comparison() + shift_factor_table()
    for f in ["ppo_reach_dr.json", "ppo_reach_nodr.json", "ppo_lift_dr.json"]:
        lines += generic_json_section(f, f"PPO run {f}", ["task", "dr", "n_envs", "total_steps", "minutes", "final_success_nominal", "hardware"])
    lines += ["## 7. Speech, ROS 2, safety", ""] + generic_json_section("stt_latency_l4.json", "faster-whisper latency") + generic_json_section("ros2_smoke.json", "ROS 2 Humble pipeline smoke (Docker)")
    return "\n".join(lines) + "\n"


def main():
    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    txt = build()
    assert "\u2014" not in txt and "\u2013" not in txt
    out.write_text(txt)
    print(f"wrote {out} ({len(txt)} chars)")


if __name__ == "__main__":
    main()
