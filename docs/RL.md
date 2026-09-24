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
| `langgrasp/policies/rl/sb3_env.py` | `ArmReachGymEnv` (gymnasium) and `VecArmSB3` (SB3 VecEnv) adapters over `VecArmEnv` |
| `langgrasp/policies/rl/sac.py`, `scripts/train_sac.py` | SAC (Stable Baselines 3 2.8.0) training with train-env and nominal-eval curves |
| `langgrasp/policies/rl/model_based.py` | `OSCController` (Jacobian + mass matrix) and `MPPIController` (MPPI on the nominal MuJoCo model) |
| `langgrasp/policies/rl/baseline_eval.py`, `scripts/eval_baselines.py` | the `eval_ppo.py` protocol for any controller; writes `results/{sac,osc,mpc}_sim2sim_gap.json` |
| `scripts/tune_model_based.py` | OSC / MPC parameter choice on the nominal sim, tuning seed 777; `results/model_based_tuning.json` |
| `scripts/ablate_shift_factors.py` | single-factor diagnostic of the shifted condition; `results/reach_shift_factors.json` |
| `tests/test_sb3_baselines.py` | wrapper shapes, determinism, reward parity with `VecArmEnv`, one episode per controller, tiny SAC loop |

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
| lift, DR from scratch | 0.0% | 0/200 | 0/200 | 0/200 |
| lift, DR curriculum (warm start from the no-DR policy, randomisation ramped 0 to 1 over the first 3 M of 6 M steps) | 75.6% | 182/200 = 91.0% [86.2, 94.2] | 92/200 = 46.0% [39.2, 52.9] | 43/200 = 21.5% [16.4, 27.7] |

(`results/ppo_sim2sim_gap_lift.json`, Wilson 95% intervals, sim-to-sim gap, no real robot. The RL env pins its own gripper opening (0.55 rad) so these checkpoints are unaffected by the executor's later retune; an earlier curriculum run made before that pin was discarded because it had trained in a different environment.)

Without domain randomisation PPO learns the lift in about 30 minutes and its return keeps rising at the
time box (0.39 at 1.9 M steps, 0.72 at 4.9 M, 0.94 at 5.8 M); its sim-to-sim gap is large (94% to 51%
under the dynamics shift, 33% with two ticks of latency), which is the point of the exercise.

With the Squint-style randomisation switched on from the start, the same budget never produced a single
lift: the return plateaus around 12-14 from 1 M steps onward with 0% success, i.e. the agent settles for
the approach reward and never discovers the grasp-and-lift bonus under 5 deg joint noise and stochastic
latency. That negative result stays in the table.

The curriculum fixes the discovery problem but not the gap. Initialising from the no-DR policy (67% in
the first iteration) and ramping the randomisation from zero to full strength over the first 3 M steps
(`scripts/train_ppo.py --task lift --dr --init-ckpt checkpoints/ppo_lift_nodr.pt --dr-ramp 0.5`, 45 minute
time box reached at 4.3 M steps, so only 1.3 M steps at full randomisation) ends at 76% in the fully
randomised training env, where training from scratch never left 0%. Evaluated deterministically it keeps
most of the nominal success (94% to 91%) but is not more robust than the no-DR policy under the dynamics
shift (51% to 46%) and is worse with two ticks of latency (33% to 21.5%); the intervals overlap for the
shifted row and not for the latency row. The honest summary for this budget: the curriculum makes
randomised training possible, and the randomisation set (one-tick latency at most, no calibration offset)
does not cover the shifts that hurt, so it buys no transfer robustness here. Next things to try, untested:
a longer run at full randomisation, adding the two-tick latency and a joint offset to the training set,
and an off-policy method (the Squint SO-101 paper transferred with SAC and found PPO the weaker baseline).

## Reach baselines: SAC, operational-space control, MPC (sim-to-sim gap, no real robot)

Three baselines on the same reach task, evaluated with exactly the protocol of `scripts/eval_ppo.py`: the
same three conditions, 200 deterministic episodes per cell, 50 envs, eval seed 12345, Wilson 95 % intervals.
No controller touches the env RNG, so every method sees the same start poses, targets and observation-noise
draws. Hardware: NVIDIA L4 host / CPU, simulation (all three baselines ran on the CPU; the GPU was busy with
another job). Every number below is in a JSON under `results/` written by a committed script; `docs/RESULTS.md`
renders the same table from those JSONs.

| Method | Nominal | Shifted (1-tick) | Shifted + 2-tick latency |
| --- | --- | --- | --- |
| PPO, no DR | 200/200 = 100.0% [98.1, 100.0] | 166/200 = 83.0% [77.2, 87.6] | 80/200 = 40.0% [33.5, 46.9] |
| PPO, DR | 200/200 = 100.0% [98.1, 100.0] | 195/200 = 97.5% [94.3, 98.9] | 109/200 = 54.5% [47.6, 61.3] |
| SAC (SB3), no DR | 200/200 = 100.0% [98.1, 100.0] | 175/200 = 87.5% [82.2, 91.4] | 95/200 = 47.5% [40.7, 54.4] |
| SAC (SB3), DR | 196/200 = 98.0% [95.0, 99.2] | 199/200 = 99.5% [97.2, 99.9] | 137/200 = 68.5% [61.8, 74.5] |
| OSC (Jacobian + mass matrix, no learning) | 200/200 = 100.0% [98.1, 100.0] | 180/200 = 90.0% [85.1, 93.4] | 77/200 = 38.5% [32.0, 45.4] |
| MPC (MPPI, nominal model; nominal row privileged) | 200/200 = 100.0% [98.1, 100.0] | 85/200 = 42.5% [35.9, 49.4] | 85/200 = 42.5% [35.9, 49.4] |

Sources: `results/ppo_sim2sim_gap.json`, `results/sac_sim2sim_gap.json`, `results/osc_sim2sim_gap.json`,
`results/mpc_sim2sim_gap.json`. Final-tick hold rates (nominal / shifted / shifted + 2-tick): SAC no DR
100 / 3.0 / 3.5 %, SAC DR 88.0 / 14.5 / 4.5 %, OSC 100 / 4.5 / 0.0 %, MPC 99.0 / 12.5 / 13.5 %.

**SAC (Stable Baselines 3 2.8.0).** `VecArmSB3` is an SB3 `VecEnv` directly over a 16-env `VecArmEnv` (same
14-D observation, [-1, 1]^5 action, reward, 40-tick horizon and any-tick success); `ArmReachGymEnv` is the
single-env gymnasium view used for the checks. The only deliberate difference to PPO is that the horizon is
a time-limit truncation (SAC bootstraps through it) instead of a terminal. The tests step both adapters and
`VecArmEnv` with the same seed and actions and require bit-identical observations and rewards, with and
without DR. SAC settings: SB3 defaults (2 x 256 ReLU, lr 3e-4, batch 256, buffer 1 M, tau 0.005, gamma 0.99,
automatic entropy), `learning_starts` 5 000, 8 gradient steps per vectorised step of 16 transitions (update
to data ratio 0.5), observation normalisation with `VecNormalize` (saved next to the checkpoint), seed 0,
40-minute time box, CPU (4 torch threads, 2 physics threads). The two runs ran concurrently with each other
and with the OSC / MPC tuning and evaluation, so their wall-clock is that of a loaded host.

| | PPO no DR | PPO DR | SAC no DR | SAC DR |
| --- | --- | --- | --- | --- |
| env steps in the run | 1 503 232 | 1 503 232 | 324 048 | 321 552 |
| gradient steps | 2 936 | 2 936 | 159 520 | 158 272 |
| wall-clock of the run | 7.0 min | 7.1 min | 40.0 min (time box) | 40.0 min (time box) |
| first 4096-step block with train-env success >= 0.9 | 167 936 steps, 1.1 min | 167 936, 1.1 min | 40 960, 4.5 min | 61 440, 7.9 min |
| first block >= 0.99 (the PPO level) | 217 088 steps, 1.4 min | 229 376, 1.4 min | 40 960, 4.5 min | 131 072, 19.0 min |
| first periodic nominal eval at 200/200 (every 20k steps, seed 999) | not measured during training | not measured | 40 000 steps, 4.4 min | never (best 196/200) |

(`results/ppo_reach_*.json`, `results/sac_reach_*.json`; a "block" is 4096 env steps, one PPO iteration, and
its success is that of the episodes finished in it with the stochastic policy in the training env.) Reading:
SAC needs about 5x fewer env steps than PPO to reach the PPO success level without DR (41k vs 217k) and
about 1.75x fewer with DR (131k vs 229k), but on this host it is slower in wall-clock: about 135 env steps/s
end to end on the CPU against about 3 500 for PPO, because every env step pays half a gradient step of three
MLPs. Under shift, SAC is at least as robust as PPO in every cell and the DR variant is the best learned policy
in the table (99.5 % shifted, 68.5 % with the 2-tick latency it never saw, against 97.5 % and 54.5 % for PPO DR;
the intervals do not overlap in the 2-tick row). This agrees in direction with the Squint SO-101 paper, which
found SAC the better transfer baseline; it is one seed per variant here.

What did not go well with SAC, stated plainly:
- SAC DR never reached 200/200 in the nominal sim during training (best periodic eval 196/200) and scores
  98.0 % nominal in the final evaluation, below its own 99.5 % in the shifted sim (the intervals overlap). Its
  nominal hold rate is 88 %, i.e. the policy trained under 5 deg observation noise keeps moving near the target.
- SAC no DR collapsed once during training and recovered: train-env success went 1.000 at 249 856 steps,
  0.625 at 258 048, 0.062 at 262 144, back to 1.000 at 274 432; the periodic nominal eval at 260 000 steps
  was 19/200. The final checkpoint (324 048 steps) is from after the recovery. No seed sweep was run, so it
  is unknown how often this happens.
- Installing SB3 required `stable-baselines3==2.8.0` (2.9 needs torch >= 2.8) and downgraded gymnasium from
  1.3.0 to 1.2.3 (SB3 2.8 requires gymnasium < 1.3). gymnasium is otherwise only used by lerobot 0.4.4, whose
  requirement (>= 1.1.1, < 2) is still met; torch stayed at 2.7.1.

**OSC.** `OSCController` computes, at the *reported* joint angles and with the nominal model, the translational
TCP Jacobian J (`mj_jacSite`) and the joint-space inertia M (`mj_fullM`), the task-space inertia
Lambda = (J M^-1 J^T + 1e-4 I)^-1 and the dynamically consistent inverse Jbar = M^-1 J^T Lambda, and commands
dq = Jbar k (target - tcp_reported) through the same action interface (action = dq / 0.12, clipped). This is
operational-space control in its velocity-level form for position servos: the torque loop of classical OSC is
the MuJoCo servo (kp 50), and because the env integrates actions into the servo target the loop has integral
action, which removes the gravity sag without a gravity term. No torque-level variant was built. Gains were
chosen on the nominal sim with tuning seed 777 only (`scripts/tune_model_based.py`: k in {0.5, 0.8, 1.0, 1.5},
null-space posture gain in {0, 0.1}; all but one setting scored 100/100 there; chosen k = 1.0, no null-space
term). Compute: 0.12 ms median per control tick for one env (p99 0.17 ms).

OSC is as robust as the learned policies to the 1-tick shift (90.0 %) and slightly worse with the 2-tick
latency (38.5 %). It needs no training at all, but it is also only possible because the task is pure
kinematics with a known model and a given target.

**MPC (MPPI).** `MPPIController` plans every 100 ms tick: 64 sampled action sequences of 4 ticks around the
shifted previous plan (sigma 0.5), each rolled out on a copy of the nominal model through all 50 physics
substeps per tick with `mujoco.rollout` (16 threads, 12 800 physics steps per tick per env), cost = the negative
env reward summed over the horizon plus a small action-rate term, exponentially weighted mean (temperature 0.02),
first action executed. The planning start state is the reported joint angles, the joint velocities of a shadow
copy of the nominal model driven by the controller's own commands, and the servo target it believes it has
set. Horizon and sigma were chosen on the nominal sim with seed 777 (grid 4 / 6 / 10 ticks x 0.3 / 0.5; the
shortest horizon scored best there, longer horizons held the target less well). Compute: 25.6 ms median per
tick for a single env (p90 35.2, p99 48.0 ms), within the 100 ms control period; the batched evaluation took
about 1.15 s per tick for 50 envs, 561 s for the three conditions.

The nominal MPC row is privileged: exact model and, without noise, exact state. In the shifted conditions it
still plans with the nominal model, and it fails: 42.5 % in both shifted rows, the worst method under shift.
The single-factor diagnostic below shows where: MPC is the only method that is fully robust to the 2-tick
latency alone (100 % with 1 or 2 ticks, presumably because it re-plans from the observed state every tick; PPO DR
also reaches 100 % with 1 tick) and it is unaffected by the dynamics shift,
but it drops to 68.5 % with the 5 deg joint noise alone and to 68.0 % with the 1.5 deg calibration offset alone,
where PPO, SAC no DR and OSC stay at 99.5 to 100 %. Two plausible causes, neither tested: the noisy reported
angles are written straight into the planning start state, so the shadow model's velocity estimate is driven
by the noise (no state filter); and the cost has a +2 bonus anywhere inside the 1.5 cm ball, so the planner has
little reason to centre the TCP, which leaves no margin for the centimetre-scale TCP error of the offset (OSC
drives the error to zero and keeps 97 % hold under the offset, MPC 6.5 %). A filtered state estimate and a
cost without the flat bonus are the obvious next steps; they were not tried, because they would be designed
after seeing the shifted results.

**Single-factor diagnostic** (`scripts/ablate_shift_factors.py`, `results/reach_shift_factors.json`; each part
of the shifted condition alone, same 200 episodes and seed; success, then final-tick hold):

| Method | noise 5 deg | latency 1 tick | latency 2 ticks | mass 1.3, friction 0.7, kp 0.8 | offset 1.5 deg |
| --- | --- | --- | --- | --- | --- |
| PPO, no DR | 100.0 / 29.0 | 71.5 / 4.0 | 53.5 / 1.5 | 100.0 / 100.0 | 100.0 / 99.5 |
| PPO, DR | 100.0 / 47.0 | 100.0 / 85.5 | 50.0 / 2.0 | 100.0 / 100.0 | 100.0 / 99.0 |
| SAC, no DR | 100.0 / 33.5 | 80.0 / 7.0 | 48.5 / 3.0 | 100.0 / 100.0 | 99.5 / 88.0 |
| SAC, DR | 100.0 / 43.5 | 95.5 / 62.5 | 61.5 / 4.0 | 99.0 / 90.0 | 86.5 / 64.0 |
| OSC | 100.0 / 28.0 | 73.0 / 5.5 | 45.0 / 3.5 | 100.0 / 100.0 | 100.0 / 97.0 |
| MPC | 68.5 / 24.5 | 100.0 / 97.0 | 100.0 / 97.0 | 100.0 / 100.0 | 68.0 / 6.5 |

(percent; Wilson intervals are in the JSON and in `docs/RESULTS.md`.) For the reactive methods (PPO, SAC,
OSC) latency is the factor that costs success, and training DR with a 1-tick latency is what buys the 1-tick
robustness; the dynamics shift is harmless for reach. The diagnostic also qualifies the reading of the PPO gap
table above: on its own the 1.5 deg offset leaves the PPO hold rate at 99 %, so the loss of holding in the
shifted row comes mostly from noise and latency, not from the offset alone. Only the SAC DR policy is hurt by
the offset alone (86.5 %).

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
MUJOCO_GL=egl .venv/bin/python scripts/train_sac.py --no-dr --max-minutes 40 --seed 0   # the two SAC runs ran concurrently
MUJOCO_GL=egl .venv/bin/python scripts/train_sac.py --dr    --max-minutes 40 --seed 0
MUJOCO_GL=egl .venv/bin/python scripts/tune_model_based.py                              # OSC / MPC parameters, nominal sim, seed 777
MUJOCO_GL=egl .venv/bin/python scripts/eval_baselines.py --methods sac osc mpc
MUJOCO_GL=egl .venv/bin/python scripts/ablate_shift_factors.py
.venv/bin/python -m langgrasp.eval.report                                               # regenerates docs/RESULTS.md
.venv/bin/ruff check langgrasp tests scripts && MUJOCO_GL=egl .venv/bin/python -m pytest -q tests/test_rl.py tests/test_sb3_baselines.py
```
