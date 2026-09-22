# Reinforcement learning track: PPO on the simulated SO-101

Status: reach is the deliverable (works, with a domain-randomisation ablation and a sim-to-sim gap table).
Lift is the stretch task; see "Lift (stretch)" below for what happened within its time box.

Everything numeric on this page was produced by the commands listed under "Reproduce" on this host:
NVIDIA L4 (x86 EC2 host), 32 vCPU, shared with two other jobs during the runs. There is no real robot
in this project. The "gap" table below is a **sim-to-sim gap, no real robot**: a nominal simulator versus
a deliberately shifted copy of the same simulator.

## Files

| Path | What |
| --- | --- |
| `langgrasp/policies/rl/envs.py` | `VecArmEnv`: vectorised state-based MuJoCo env (reach, lift), `DomainRandomization` |
| `langgrasp/policies/rl/ppo.py` | dependency-free PPO (torch only), `train`, `evaluate`, `wilson`, checkpoint I/O |
| `scripts/train_ppo.py` | `--task reach|lift --dr/--no-dr --steps N --n-envs 64 --seed S --max-minutes M` |
| `scripts/eval_ppo.py` | 200-episode evaluation in nominal / shifted / shifted+2-tick sims, Wilson 95 % CIs |
| `tests/test_rl.py` | env shapes, determinism, reward sanity, PPO update, checkpoint roundtrip, tiny train loop (about 4 s) |
| `results/ppo_reach_nodr.json`, `results/ppo_reach_dr.json` | training curves and throughput |
| `results/ppo_sim2sim_gap.json` | the gap table, labelled "sim-to-sim gap, no real robot" |
| `checkpoints/ppo_reach_nodr.pt`, `checkpoints/ppo_reach_dr.pt` | policies + observation normaliser (gitignored) |

## Design

**Why state-based PPO, not pixels.** The perception stack of this project (open-vocabulary grounding,
YOLO11-seg, depth fusion) already turns language plus RGB-D into a 3-D target point. The RL policy
therefore only has to solve the control problem downstream of perception: joint angles in, joint targets
out, with the target point given. State-based PPO trains in minutes on the CPU-bound physics loop
(1.5 M steps in 7 minutes here) and needs no renderer; a pixel policy would multiply the cost by the
render time per frame and would relearn what the perception stack already does. The price is that the
policy inherits every perception error unfiltered, and it cannot exploit visual cues about contact.

**Why MuJoCo, not ManiSkill3 / SAPIEN.** The rest of the project is already a MuJoCo scene
(`langgrasp/sim`), so the RL env reuses the same arm XML (MuJoCo Menagerie SO-ARM100), the same
actuators (position servos, kp 50, force limit 3.5 N m) and the same TCP site and jaw pads that define
"grasped" for the scripted controller and the ACT policy. One physics model for every track means the
sim-to-sim comparison below shifts exactly the parameters the other tracks use. MuJoCo also runs
headless with plain CPU threads: `mj_step` releases the GIL, so 64 `MjData` stepped from a Python
thread pool reach several thousand env-steps per second on this box without a GPU renderer or Vulkan
(which is what a SAPIEN-based stack would want on a headless EC2 host).

**Environment.** `VecArmEnv(n_envs, task, seed, dr)` compiles the arm + table + one cube (`cube_0`)
from the project's scene XML with the other nine pool objects stripped (they are parked out of view in
the full scene anyway). This cuts `nq` from 76 to 13 and is what makes the loop fast. Each env has its
own `MjModel` copy (about 1 MB) so mass, friction and kp can be randomised per env, and its own
`MjData`. Control is 10 Hz (50 physics steps of 2 ms per tick, the same as `LangGraspEnv`).
Episodes start from the model's `home` keyframe plus uniform +-0.15 rad joint noise.

| | reach | lift |
| --- | --- | --- |
| observation | q(5), tcp(3), target(3), target - tcp(3) = 14 | q(5), tcp(3), cube(3), cube - tcp(3), jaw(1), jaw axis xy(2), cube x-axis xy(2) = 19 |
| action | [-1, 1]^5, 0.12 rad per tick added to the joint target | [-1, 1]^6, arm as reach + 0.5 rad per tick on the jaw target |
| reward | -dist + 2 [dist < 1.5 cm] | see `envs.py` docstring (approach, early-close penalty, at-point, grasp, height, lift bonus) |
| success | true dist < 1.5 cm at any tick ("reached"); final-tick version reported as "hold" | cube > 5 cm above rest while both pad groups touch it, at any tick |
| horizon | 40 ticks (4 s) | 60 ticks (6 s) |

The reported TCP in the observation is forward kinematics of the *reported* joint angles (noise and
calibration offset included), which is what a real controller with encoders only would see. Ground
truth is used only for the reward and the success test.

**PPO.** MLP actor and critic, 2 x 256 tanh, Gaussian policy with a state-independent log-std
(init -0.5), GAE lambda 0.95, gamma 0.99, clip 0.2, 4 epochs, minibatch 2048 of the 64 x 64 rollout,
Adam 3e-4, advantage normalisation, running observation normalisation saved in the checkpoint,
entropy 0, no value clipping, grad-norm clip 0.5, orthogonal init. Fixed-horizon episodes are treated as
terminal at the horizon (no bootstrap through the time limit), which is the simplest correct choice for a
fixed-length task. Seeds fix torch, numpy, the env RNG and the minibatch permutation, so a run is
reproducible bit-for-bit on the same host and thread count.

## Domain randomisation (Squint-style, physical + proprioceptive, no vision)

`DomainRandomization.train()`, sampled per episode per env unless stated:

| Knob | Value | How it is applied |
| --- | --- | --- |
| joint-position observation noise | sigma 5 deg, iid per tick | added to the reported q; the observed TCP is FK of the noisy q |
| action noise | sigma 0.1 on the [-1, 1] action | before integration into the joint target |
| action latency | 1 tick with probability 0.5 | per-env queue; the action applied is the one issued a tick earlier |
| cube mass (and inertia) | x U(0.7, 1.3) | `body_mass`, `body_inertia` on the env's own model copy, then `mj_setConst` |
| cube sliding friction | x U(0.7, 1.3) | `geom_friction[:, 0]` on the model copy |
| actuator kp | x U(0.8, 1.2) | `actuator_gainprm[:, 0]` and `actuator_biasprm[:, 1]` on the model copy (kv unchanged, so the damping ratio moves with kp) |
| gravity | unchanged | |

Per-env model copies were cheap enough (64 x about 1 MB, `copy.copy(MjModel)` in 5 ms), so kp is
randomised for real rather than approximated through action scaling. A constant calibration offset is
deliberately **not** part of training DR; it is the unmodelled part of the shifted evaluation.

## Measured throughput

| Setting | env-steps / s |
| --- | --- |
| full project scene (nq 76), 64 envs, serial `mj_step` | 318 |
| RL model (nq 13), 64 envs, serial | 1 271 |
| RL model, 64 envs, 8 threads, random actions (micro-benchmark) | 7 733 |
| reach training, env-only time, 8 threads, while two other jobs ran | 4 282 to 4 318 |
| reach training, end to end incl. FK observations, PPO update on the L4 | 3 534 to 3 562 |
| lift training, env-only, 6 threads (contacts are dearer), two runs concurrently | about 2 700 |

The gap between micro-benchmark and training throughput is the per-env Python work (FK for the
reported TCP, contact scan, reset) plus the load from the concurrent jobs; the PPO update itself is
negligible at this network size.

## Reach: training curves

Both runs: seed 0, 64 envs x 64 steps per iteration, 1 503 232 steps, 367 iterations, about 7 min each
on the L4 host (both ran concurrently). Success below is the fraction of episodes finished in that
iteration that reached the target at any tick, with the stochastic policy.

| | no DR | DR |
| --- | --- | --- |
| first iteration with success >= 0.5 | 110 592 steps | 90 112 steps |
| first iteration with success >= 0.9 | 167 936 | 167 936 |
| first iteration with success >= 0.99 | 217 088 | 229 376 |
| mean return at 1.5 M steps | 50.0 | 17.1 |
| train-env success, mean of last 5 iterations | 1.000 | 0.988 |
| train-env "hold" (within 1.5 cm at the last tick), last 5 iterations | 0.59 to 0.73 | 0.20 to 0.28 |

Full curves (steps, mean return, success rate per iteration) are in `results/ppo_reach_*.json`.
The DR run's lower return and hold rate are expected: 5 deg iid noise on five joints moves the
observed TCP by roughly 1 to 2 cm per tick, so a memoryless policy cannot sit still inside a
1.5 cm ball; it passes through the target instead.

## Sim-to-sim gap, no real robot (reach)

200 deterministic episodes per cell (mean action), eval seed 12345, Wilson 95 % CIs, from
`results/ppo_sim2sim_gap.json`. "Shifted" is the same MuJoCo model with: joint-position noise 5 deg,
fixed 1-tick action latency, cube mass x1.3, friction x0.7, kp x0.8 and a constant +1.5 deg offset on
every arm joint (true angle = commanded - offset, reported = true + offset). "Shifted + 2-tick" is the
same with a 2-tick latency. The offset and the 2-tick latency were never seen in training.

| Policy | Nominal | Shifted (1-tick) | Shifted + 2-tick latency |
| --- | --- | --- | --- |
| PPO reach, no DR | 1.000 [0.981, 1.000] (200/200) | 0.830 [0.772, 0.876] (166/200) | 0.400 [0.335, 0.469] (80/200) |
| PPO reach, DR | 1.000 [0.981, 1.000] (200/200) | 0.975 [0.943, 0.989] (195/200) | 0.545 [0.476, 0.613] (109/200) |

Final-tick "hold" rate in the same episodes: nominal 1.00 / 1.00; shifted 0.02 (no DR) / 0.16 (DR);
shifted + 2-tick 0.01 / 0.01. Reading: DR closes most of the 1-tick shifted gap for the lenient
any-tick metric (83.0 % to 97.5 %, non-overlapping CIs), helps but does not solve a latency the policy
never saw (40.0 % to 54.5 %), and the constant 1.5 deg calibration offset destroys precise
*holding* for both policies because the policy believes it is on target when its real TCP is about a
centimetre away. That last row is the argument for closing the loop on a measured TCP (vision) rather
than on encoders alone.

## Lift (stretch): measured, and a negative result

Two 40-minute time-boxed runs on the lift task (cube must be lifted above 5 cm while both finger pads
touch it), same PPO settings as reach, 64 envs, about 5.6-5.8 M env steps each (`results/ppo_lift_*.json`):

| Run | Train-env success at the end | Nominal eval (200 ep) | Shifted | Shifted + 2-tick latency |
|---|---|---|---|---|
| lift, no DR | 92.8% | 188/200 = 94.0% [89.8, 96.5] | 102/200 = 51.0% [44.1, 57.8] | 66/200 = 33.0% [26.9, 39.8] |
| lift, DR | 0.0% | 0/200 | 0/200 | 0/200 |

(`results/ppo_sim2sim_gap_lift.json`, Wilson 95% intervals, sim-to-sim gap, no real robot.)

Without domain randomisation PPO learns the lift in about 30 minutes and its return keeps rising at the
time box (0.39 at 1.9 M steps, 0.72 at 4.9 M, 0.94 at 5.8 M); its sim-to-sim gap is large (94% to 51%
under the dynamics shift, 33% with two ticks of latency), which is the point of the exercise. With the
Squint-style randomisation switched on, the same budget never produced a single lift: the return plateaus
around 12-14 from 1 M steps onward with 0% success, i.e. the agent settles for the approach reward and
never discovers the grasp-and-lift bonus under 5 deg joint noise and stochastic latency. This is a
documented negative result (the DR policy is not tuned; likely fixes are a curriculum from no-DR to DR,
a longer budget, or an off-policy method as in the Squint SO-101 paper where SAC transferred and PPO was
the weaker baseline). We did not spend more than the 40-minute box on it.

## Limitations, stated plainly

- No real robot. Every "transfer" number is nominal-sim versus shifted-sim of the same model.
  The shifted parameters are guesses at a plausible SO-101 mismatch, not measurements of one.
- PPO is known to transfer worse than SAC on the SO-101 in the Squint paper (arXiv:2602.21203); this
  track uses PPO because it is the simplest on-policy baseline to implement without dependencies, and
  the ablation here only shows that DR helps *this* PPO policy under *this* shift.
- Observations are state-based (joint encoders + a given 3-D target), not vision. The policy never
  sees the perception errors of the upstream stack, and the sim-to-sim shift does not model them.
- Success for reach is the lenient "within 1.5 cm at any tick"; the stricter final-tick hold rate is
  reported next to it and is much lower under shift.
- The horizon cut is treated as terminal (no time-limit bootstrap), which slightly biases the critic.
- Contacts in MuJoCo with the default solver are not a substitute for a measured gripper; lift results
  (if any) say nothing about real grasp robustness.
- The two reach runs share one seed; no seed sweep was run, so the curve milestones above are single
  samples.

## Reproduce

```bash
cd /home/ec2-user/workspace/langgrasp
MUJOCO_GL=egl .venv/bin/python scripts/train_ppo.py --task reach --no-dr --steps 1500000 --max-minutes 25 --seed 0
MUJOCO_GL=egl .venv/bin/python scripts/train_ppo.py --task reach --dr    --steps 1500000 --max-minutes 25 --seed 0
MUJOCO_GL=egl .venv/bin/python scripts/eval_ppo.py --checkpoints checkpoints/ppo_reach_nodr.pt checkpoints/ppo_reach_dr.pt
MUJOCO_GL=egl .venv/bin/python scripts/train_ppo.py --task lift --dr --steps 8000000 --max-minutes 40 --n-threads 6 --seed 0
.venv/bin/ruff check langgrasp tests scripts && MUJOCO_GL=egl .venv/bin/python -m pytest -q tests/test_rl.py
```
