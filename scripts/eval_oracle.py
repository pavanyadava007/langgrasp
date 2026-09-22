"""Scripted oracle (ground-truth grasp points) through the shared evaluation protocol: the upper bound row."""

import argparse

import numpy as np

from langgrasp.eval.harness import run_protocol
from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scene import OBJECT_KINDS

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40, help="trials per stratum")
ap.add_argument("--out", default="results/oracle_protocol.json")
args = ap.parse_args()
env = LangGraspEnv(seed=0)


def run_one(sc):
    env.reset(sc)
    p, psi = env.grasp_point(sc.target)
    ctl = PickPlaceController(env)
    r = ctl.run(np.asarray(p), psi, sc.target, width=OBJECT_KINDS[sc.target_obj.kind]["width"])
    return {"grounding_correct": True, "grasped": r.grasped, "lifted": r.lifted, "placed": r.placed, "aborted": None, "latency_ms": {}, "extra": {"steps": r.steps, "rot_err": r.rot_err}}


res = run_protocol(run_one, {"seen": args.n, "unseen": args.n, "langvar": args.n}, "oracle_scripted", args.out, notes="ground-truth grasp point and jaw yaw from the simulator; no perception; upper bound for the executor")
for k, v in res["summary"]["strata"].items():
    print(k, "place", v["place"], "grasp", v["grasp"])
print("kinds", {k: (v["place"]["k"], v["place"]["n"]) for k, v in res["summary"]["kinds"].items()})
