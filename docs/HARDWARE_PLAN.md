# Hardware plan and what this repository proves

The project brief targets a physical SO-101 arm, an Intel RealSense D435 and a Jetson Orin Nano Super 8GB.
None of these were available while this repository was built. This file maps each brief item to what is
implemented and measured here, and what remains hardware work.

| Brief item | Status here | On hardware |
|---|---|---|
| SO-101 assembly, LeRobot calibration, leader-arm teleop | Simulated arm (Menagerie SO-ARM100 MJCF, position servos kp=50); demos are scripted, not teleoperated | 3-4 h assembly, `lerobot-calibrate`, udev rules by USB path; record about 50 human demos |
| RealSense D435 RGB-D | MuJoCo rendered RGB + metric depth with a synthetic noise model (sigma 2 mm, 2% dropout); pinhole intrinsics from the camera fovy | `realsense_node`, aligned depth, real noise, IR washout on reflective objects (FMEA H3) |
| Jetson Orin Nano Super, JetPack 6.2, TensorRT 10.3 | TensorRT 10.16 engines built and benchmarked on an NVIDIA L4; numbers labelled L4 | Rebuild engines on the Jetson (engines are device specific), INT8 calibration on device, `nvpmodel` fixed power mode, report inference-only and end-to-end |
| Grounding DINO 1.5 Edge | The open Grounding DINO tiny checkpoint (transformers), once per command, measured latency on L4 | Edge checkpoint or MM-Grounding-DINO, TensorRT; fall back to YOLO-World/YOLOE if command latency > 300 ms |
| YOLO11-seg fine-tune, INT8 export | Fine-tuned on rendered data with labels from the simulator's segmentation buffer; FP16 and INT8 engines benchmarked | Label real images (SAM2-assisted), calibrate INT8 on the Jetson with real frames |
| Depth fusion + PCA grasp | Implemented and validated against simulator ground truth | Same code; add depth validity masks for real dropout |
| ACT via LeRobot | Trained on scripted sim demos, evaluated closed loop in sim | Train on human demos; deploy on Jetson (ACT fits in 8 GB) |
| SmolVLA (stretch) | Not attempted (scope-cut item 1 in the brief) | Async PolicyServer on the workstation, Jetson as RobotClient |
| PPO in ManiSkill3 with sim-to-real | PPO in a MuJoCo state-based env with domain randomisation; the gap is sim-to-sim (nominal vs shifted dynamics), clearly labelled | Real transfer needs the physical arm; the Squint paper reports PPO at about 62% real |
| ROS 2 Humble node graph | Package implemented; verified in a `ros:humble` Docker container as documented in ROS2.md | `dustynv/ros:humble` L4T container, pin the L4T tag, set RMW_IMPLEMENTATION and ROS_DOMAIN_ID |
| ISO 26262 + SOTIF FMEA | docs/FMEA.md with mechanisms implemented in `langgrasp/safety` and tested | Add servo temperature/current telemetry (H4), vision intrusion detection and a hardware e-stop (H5) |
| >= 100-trial evaluation with Wilson CIs | Run in simulation for the oracle, modular and ACT approaches | Repeat with the physical setup; keep the same seeds/commands for the language strata |

## Bill of materials and schedule

The BOM (about EUR 1050-1200 with a DIY SO-101 kit, a used D435 or an OAK-D Lite, and the Jetson at MSRP)
and the 10-week schedule are unchanged from the brief; the software in this repository covers the work of
weeks 4-10 except for anything that needs the physical devices. The scope-cut order also stands: SmolVLA,
learned 6-DOF grasp, INT8, unseen/language strata, RL sim-to-real.
