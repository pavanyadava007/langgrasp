# Demo guide

Everything runs headless on a Linux box with an NVIDIA GPU (tested: L4, driver 580, CUDA 12.6). No robot,
camera or Jetson is needed. Set `MUJOCO_GL=egl` for every command that touches the simulator.

## Setup

```bash
uv venv -p 3.10 .venv
uv pip install -p .venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -p .venv/bin/python -e ".[perception,learning,speech,dev]"
curl -L -o checkpoints/yolo11n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
```

The Grounding DINO tiny checkpoint (about 660 MB) is fetched from the Hugging Face Hub on first use.

## The 5-minute demo

1. Scripted executor sanity check (no perception): `make smoke` prints per-seed grasp/lift/place flags.
2. Full language pipeline on three scenes (seen, unseen, language variation) into a video with a
   latency overlay: `make video`, then open `media/demo.mp4`.
3. Interactive: `python scripts/demo_command.py --seed 5000 --command "pick the blue one"` runs one
   command and prints the decision path (query, candidates, colour fractions, grasp pose, timings).

## Reproduce the numbers

| What | Command | Writes |
|---|---|---|
| Oracle upper bound (120 trials) | `python scripts/eval_oracle.py --n 40` | results/oracle_protocol.json |
| Modular pipeline (120 trials + ablations) | `python scripts/eval_modular.py --n 40` | results/modular_*.json |
| Synthetic YOLO dataset | `python scripts/make_synth_dataset.py` | data/yolo_synth |
| YOLO11n-seg fine-tune | `python scripts/train_yolo.py` | checkpoints/yolo11n-seg-langgrasp.pt, results/yolo_train.json |
| Export + latency bench | `python scripts/export_yolo.py && python scripts/bench_yolo.py` | results/yolo_latency_l4.json |
| Scripted demos -> LeRobotDataset | `python scripts/collect_demos.py` | data/lerobot/langgrasp_pick |
| ACT training / eval | `python scripts/train_act.py`, `python scripts/eval_act.py` | results/act_*.json |
| PPO reach (DR and no DR) + gap table | `python scripts/train_ppo.py --task reach --dr`, `python scripts/eval_ppo.py` | results/ppo_*.json |
| Speech-to-text latency | `python -m langgrasp.language.stt --bench` | results/stt_latency_l4.json |
| ROS 2 pipeline in Docker | see docs/ROS2.md | results/ros2_smoke.json |
| Regenerate the results page | `make report` | docs/RESULTS.md |

`make gate` runs ruff and the full test suite (simulation tests included, under two minutes).

## Talking points for an interview

- Why the executor needs a fixed-jaw offset and an x/y integral correction (SO-101 has one moving
  finger; position servos sag about 1 cm under load at 10 Hz).
- Why grounding runs once per command and why the colour check exists (Grounding DINO tiny scores
  "blue screwdriver" on a yellow screwdriver almost as high as on the blue one; measured).
- What the sim-to-sim gap table can and cannot say about a real robot.
- The FMEA: which mechanisms are in software here, which need hardware telemetry.
