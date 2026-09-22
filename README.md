# LangGrasp: language-guided pick-and-place with a measured edge stack (simulation)

"Pick the blue screwdriver" -> open-vocabulary grounding -> YOLO11-seg mask -> depth fusion and grasp pose
-> a 5-DOF SO-101 arm picks it and places it in a tray. This repository is the complete software stack of
the LangGrasp plan (modular pipeline, ACT imitation policy, PPO with domain randomisation, safety monitor
with an ISO 26262 / SOTIF style FMEA, ROS 2 node graph, TensorRT export and latency budgets), built and
evaluated end to end against a MuJoCo simulation of the SO-ARM100/SO-101 because no arm, RGB-D camera or
Jetson was available.

Everything numeric in this repo was measured on this machine (NVIDIA L4, x86 EC2 host) and is written by
scripts into `results/*.json`; `docs/RESULTS.md` is generated from those files and is the only place to
read numbers from. Nothing here is a hardware measurement. See `docs/HARDWARE_PLAN.md` for what changes
on the physical system.

## What is in the box

| Track | Implementation | Measured |
|---|---|---|
| Simulation + executor | MuJoCo scene with the Menagerie SO-ARM100 model, seeded scenarios in three strata (seen / unseen / language variation), damped-least-squares IK, scripted Cartesian pick-and-place with a fixed-jaw offset and tracking correction | oracle success by stratum and object kind, reachability map |
| Language | rule-based parser (action, colour, noun, spatial reference), faster-whisper speech-to-text with a typed fallback | STT latency per model size on L4 |
| Grounding | Grounding DINO tiny once per command, HSV colour verification, spatial resolver, generic-object fallback for unknown synonyms | grounding accuracy per language variant, with and without the colour check |
| Segmentation | YOLO11n-seg fine-tuned on rendered data (labels from the simulator's segmentation buffer), ONNX opset 17 static, TensorRT FP16 and INT8 engines | mAP per export, batch-1 latency per backend |
| Depth fusion | pinhole back-projection, table/outlier removal, PCA grasp axis, thickest-segment grasp point for elongated objects | grasp centre / yaw error vs ground truth, with a synthetic depth-noise model |
| ACT | scripted demos into a LeRobotDataset, LeRobot ACT training, closed-loop evaluation on the same seeds as the modular pipeline (fixed goal) | success with Wilson CI vs oracle, policy latency |
| RL | vectorised state-based MuJoCo env, minimal PPO, Squint-style domain randomisation, sim-to-sim gap table (no real robot) | success nominal vs shifted dynamics, DR ablation |
| Safety | SafetyMonitor: stale-topic watchdog, joint and velocity limits, TCP geofence, grounding confidence gate with ambiguity flag, e-stop latch, reduced-speed mode; FMEA H1-H8 | unit and in-sim integration tests |
| ROS 2 | Humble package with driver, command, grounding, grasp, policy, safety and latency-tracer nodes | pipeline smoke in a `ros:humble` container, per-stage latency |
| Evaluation | stratified protocol, Wilson 95% intervals, per-stage latency tracer, generated results page | `docs/RESULTS.md` |

## Quick start

See `docs/DEMO_GUIDE.md`. Short version:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/smoke_pick.py 20        # scripted picks, no perception
MUJOCO_GL=egl .venv/bin/python scripts/make_video.py --approach modular   # full pipeline demo video
make gate                                                       # ruff + pytest
make report                                                     # regenerate docs/RESULTS.md
```

## Results

The current numbers are in [docs/RESULTS.md](docs/RESULTS.md). Per-track write-ups: `docs/PERCEPTION.md`,
`docs/ACT.md`, `docs/RL.md`, `docs/FMEA.md`, `docs/ROS2.md`, `docs/DOCKER.md`.

## Honest scope

- Simulation only. The RGB-D camera is a rendered pinhole camera with a simple synthetic noise model; the
  arm is the Menagerie MJCF with position servos; demos are scripted, not teleoperated.
- Latency is measured on an NVIDIA L4 (about 30 TFLOPS FP16 class) and says nothing about the Jetson Orin
  Nano; TensorRT engines are device specific and must be rebuilt there.
- The "sim-to-real" gap of the brief is reported as a sim-to-sim gap (nominal versus shifted dynamics).
- SmolVLA, learned 6-DOF grasping and Isaac Lab / ManiSkill3 were not attempted (scope-cut order of the plan).

## Mapping to the five focus areas of the target internship

Deep learning (ACT, YOLO11-seg fine-tune), multimodal AI (language grounding with Grounding DINO and
speech input), robot manipulation (IK executor, grasp synthesis, ACT rollouts), vision engineering
(synthetic data, ONNX/TensorRT export, measured latency budgets, depth fusion) and reinforcement learning
(PPO with domain randomisation and a gap study). The safety analysis (`docs/FMEA.md`) is written the way an
automotive functional-safety engineer would write it for a benchtop cobot.

## License

MIT for this repository. The SO-ARM100 model files under `langgrasp/sim/assets/so_arm100` are from the
MuJoCo Menagerie (Apache-2.0, license file included). Grounding DINO, YOLO11 (AGPL-3.0 for Ultralytics
code and weights) and LeRobot keep their own licenses.

Author: Pavan Yadav Annappa.
