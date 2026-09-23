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

- LeRobot 0.4.4 ACT preset: ResNet-18 backbone (ImageNet init), dim_model 512, 8 heads, FFN 3200, 4 encoder / 1 decoder layers, VAE latent 32, KL weight 10, dropout 0.1, AdamW lr 1e-5 (backbone 1e-5), weight decay 1e-4, no scheduler.
- chunk_size 20, n_action_steps 10 (the plan's low-rate chunk advice for 10 Hz control), batch 32, 20000 steps = 85 epochs over the 120 demos, seed 1000.
- 28.1 minutes on the NVIDIA L4 (x86 EC2 host), shared with other jobs at 11.8 steps/s (379 samples/s), peak VRAM 1658 MB, data loading 7 ms of a 76 ms step (pyav video decoding, 8 workers).
- Loss (L1 + KL): step 100: 8.558, step 900: 1.319, step 4900: 0.148, step 9900: 0.072, step 14900: 0.052, step 19900: 0.047. The loss was still falling slowly at the end; no early stopping and no validation split were used (all 120 demos train the policy, the evaluation is closed loop on held-out seeds).

## Closed-loop evaluation (`results/act_eval.json`, `results/oracle_fixed_goal.json`, `results/modular_fixed_goal.json`)

100 fixed-goal scenes (seeds 5000-5099, red cube plus 1-2 distractors, random poses and lighting), closed loop at
10 Hz, horizon 90 ticks, success = cube released in the tray. Wilson 95% intervals.

| Approach | Grasp (lifted) | Place |
|---|---|---|
| Oracle executor (ground-truth grasp point) | 99/100 = 99% [95-100] | 100/100 = 100% [96-100] |
| Modular pipeline (Grounding DINO + YOLO11-seg + depth fusion + executor) | 90/100 = 90% [83-94] | 90/100 = 90% [83-94] |
| ACT, attempt 3 (700 demos, 192x192 cropped front view, start-pose jitter, 30k steps) | 48/100 = 48% [38-58] | 47/100 = 47% [38-57] |
| ACT, attempt 3 resumed to 60k steps | 51/100 = 51% [41-61] | 49/100 = 49% [39-59] |
| ACT, attempt 2 (240 demos, 192x192, 25k steps) | 54/100 = 54% [44-63] | 55/100 = 55% [45-64] |
| ACT, attempt 1 (120 demos, 128x128, 20k steps) | 6/100 = 6% [3-12] | 6/100 = 6% [3-12] |

ACT attempt 1 is a negative result and was diagnosed rather than hidden:

- Open loop on its own training frames the policy tracks the expert to a mean absolute error of 0.05 / 0.03 /
  0.03 / 0.01 / 0.05 rad on the five arm joints and 0.009 rad on the jaw (60 frames of episode 0), so the
  preprocessing and the LeRobot processor pipeline are consistent with training.
- Closed loop on training seeds it places 2 of 10. The trace on seed 20000 shows the whole approach and
  descent offset by about 3.4 cm in x (policy TCP -0.101 m vs expert -0.067 m) with the right timing and the
  right jaw commands; the fixed-jaw executor tolerates about 1 cm, so the grasp closes on nothing.
- A 3 degree base-yaw error is a 1.3 cm error at 0.25 m reach; the cube is about 5 pixels wide in the
  128x128 front image, so sub-centimetre localisation from that image is not available to the policy, and
  120 low-variance scripted demos give it little to interpolate between.
- Per-call policy latency on the L4: median 1.0 ms (9 of 10 ticks pop the action queue; a chunk prediction is about 10-40 ms).

**Attempt 3** applied the rest of the diagnosis: 700 demos (43917 frames) collected with the retuned executor
(4 cm pre-grasp opening; 700 of 700 attempts succeeded), a front-camera crop to the workspace so the 192x192 image spends its
pixels on the objects, and a uniform +-0.15 rad start-pose jitter so the approaches vary. Same ACT config, 30k steps
(22 epochs) in 63 minutes at 7.9 steps/s. Checkpoint selection on 30 held-out scenes: 10/30 at 10k,
13/30 at 20k, 15/30 at 30k steps, so the last checkpoint was used. On the 100 scenes it places 47/100
(47%, CI 38-57), not better than attempt 2 (55, CI 45-64) although the intervals overlap. With six times the
frames it saw only 22 epochs versus 53, the loss was still falling (0.046 at the end) and the selection curve was still
rising, so the result reads as under-training rather than as evidence against the extra data; resuming the same run to 60k steps
(`lerobot-train --resume`) gave 49/100 (CI 39-59), so attempt 3 plateaued below attempt 2 and the extra demos, the crop and the
jitter did not help at this budget; attempt 2 remains the reported ACT number. Note that attempts 1 and 2 were evaluated with the
executor's 5.5 cm opening and attempt 3 with 4 cm; the demos of each attempt match the executor used to evaluate it.

**Attempt 2** followed directly from that diagnosis: 240 demos (242 attempts, seeds 20000-20241) at 192x192, 15035 frames,
same ACT config, 25k steps (53 epochs) in 47 minutes on the otherwise idle L4 at 8.9 steps/s; loss step 100: 8.499, step 1000: 1.213, step 5000: 0.147, step 10000: 0.069, step 20000: 0.045, step 25000: 0.040.
Closed loop on the same 100 scenes it places 55/100 (55%, CI 45-64) with median policy-call latency 1.0 ms (p90 1 ms, one chunk prediction every 10 ticks).
The jump from 6% to 55% with doubled demos and 2.25x the pixels supports the localisation diagnosis; the gap to the modular pipeline (90%) and the oracle (100%) is the remaining precision at the grasp. Attempt 2 is the ACT number reported in `docs/RESULTS.md` and the README; attempt 1 stays in the table as the documented negative result. Untried next steps: 300+ demos, 224 px, a validation split for checkpoint selection, temporal ensembling, and a wider-clearance grasp in the demos so small offsets are tolerated.

## Limitations

- Scripted demos: the policy imitates a deterministic expert, so its errors are mostly precision errors at
  the grasp, not strategy errors; a human-teleop dataset would be more varied and harder.
- Fixed goal only; no language conditioning (SmolVLA was the plan's language-conditioned policy and was
  cut first, as the plan's scope-cut order says).
- Simulation only; 128x128 renders without sensor noise on the images (the depth noise model applies to
  the modular pipeline only).
- One training seed, one checkpoint; no hyperparameter sweep.
