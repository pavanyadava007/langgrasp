"""First autonomous pick with oracle grasp points: sanity check of scene, IK and controller."""

import sys
import time

from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario
from langgrasp.sim.scene import OBJECT_KINDS

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
env = LangGraspEnv(seed=0)
ok = 0
t0 = time.time()
for s in range(n):
    sc = make_scenario(1000 + s, "seen")
    env.reset(sc)
    p, psi = env.grasp_point(sc.target)
    ctl = PickPlaceController(env)
    r = ctl.run(p, psi, sc.target, width=OBJECT_KINDS[sc.target_obj.kind]["width"])
    ok += r.placed
    print(f"seed {sc.seed} {sc.target_obj.kind:12s} {sc.command:28s} grasped={r.grasped} lifted={r.lifted} placed={r.placed} ik_ok={r.ik_ok} maxerr={r.max_pos_err*1000:.1f}mm steps={r.steps}")
print(f"placed {ok}/{n} in {time.time()-t0:.1f}s")
if n >= 1:
    import imageio.v3 as iio
    env.reset(make_scenario(1000, "seen"))
    for cam in ["front", "side", "wrist"]:
        iio.imwrite(f"media/smoke_{cam}.png", env.render(cam))
    print("home tcp", env.kin.fk(env.q_arm)[0], "K", env.camera_intrinsics("front")[0, 0])
