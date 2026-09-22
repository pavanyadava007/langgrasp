# ACT track: scripted demos, LeRobot dataset, training and closed-loop evaluation

ACT (Action Chunking with Transformers) trained with LeRobot 0.4.4 on demonstrations from the scripted
executor, then rolled out closed loop in the simulator on the same 100 scenes the modular pipeline and the
oracle executor are evaluated on. Everything is MuJoCo simulation on an NVIDIA L4 (x86 EC2 host).

## What is different from the plan, said plainly

- The demonstrations are produced by the scripted `PickPlaceController` with ground-truth grasp points,
  not by a human with a leader arm. They are clean, low-variance, and always follow the same phase
  structure (approach, hover, descend, close, lift, transport, lower, release, retreat).
- ACT has no language input. Its task is therefore the fixed goal "pick the red cube and place it in the
  tray" with 1-2 distractors of other kinds and colours; the modular pipeline and the oracle are evaluated
  on the identical fixed-goal scenes (`results/modular_fixed_goal.json`, `results/oracle_fixed_goal.json`)
  so the comparison is like for like.
- Images are 128x128 (front + wrist), state is the 6 joint positions, actions are the 6 joint targets
  the executor commanded at each 10 Hz tick.

## Files

| Path | Role |
|---|---|
| `langgrasp/policies/act/data.py` | dataset features, per-tick observation, episode recording, LeRobotDataset writer |
| `scripts/collect_demos.py` | collects N successful scripted episodes (failures are counted and discarded) |
| `scripts/train_act.py` | wraps `lerobot-train` with the ACT preset and writes `results/act_train.json` |
| `langgrasp/policies/act/policy.py` | `ACTRunner`: loads the checkpoint plus its pre/post processors, `act(front, wrist, state) -> 6-D target`, per-call latency |
| `scripts/eval_act.py` | closed-loop rollouts on the fixed-goal seeds 5000-5099 (horizon 90 ticks), oracle on the same scenes |
| `tests/test_act.py` | feature construction, tiny dataset round trip, runner preprocessing shapes |

## Demonstrations (`results/demos_act.json`)

- 120 successful episodes out of 122 attempts (expert success 98.4%; failed demo seeds [20110, 20112]), seeds 20000-20121, disjoint from the evaluation seeds.
- 7527 frames at 10 Hz; episode length 58-70 ticks (mean 62.7) including 5 hold ticks at the end so episodes finish at rest.
- Cameras ['front', 'wrist'] at 128x128, stored as video (libsvtav1), 18.6 MB on disk; collection took 2.79 minutes.
- Expert: scripted PickPlaceController with ground-truth grasp point (no human teleop).

## Training (`results/act_train.json`)

TRAIN_SECTION

## Closed-loop evaluation (`results/act_eval.json`, `results/oracle_fixed_goal.json`, `results/modular_fixed_goal.json`)

EVAL_SECTION

## Limitations

- Scripted demos: the policy imitates a deterministic expert, so its errors are mostly precision errors at
  the grasp, not strategy errors; a human-teleop dataset would be more varied and harder.
- Fixed goal only; no language conditioning (SmolVLA was the plan's language-conditioned policy and was
  cut first, as the plan's scope-cut order says).
- Simulation only; 128x128 renders without sensor noise on the images (the depth noise model applies to
  the modular pipeline only).
- One training seed, one checkpoint; no hyperparameter sweep.
