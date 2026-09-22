#!/bin/bash
# Build the ROS 2 Humble smoke image and run the full oracle-mode pipeline for RUN_SECONDS, writing results/ros2_smoke.json.
# Usage: docker/ros2_smoke.sh [seed] [run_seconds] [MUJOCO_GL]
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:-1001}"
RUN_SECONDS="${2:-30.0}"
GL="${3:-egl}"
docker build -t langgrasp-ros2:humble -f "$REPO/docker/Dockerfile.ros2" "$REPO/docker"
docker run --rm --name langgrasp_ros2_smoke -e MUJOCO_GL="$GL" -v "$REPO:/ws/langgrasp" langgrasp-ros2:humble \
  bash -c "ros2 launch langgrasp_ros langgrasp_sim.launch.py seed:=$SEED run_seconds:=$RUN_SECONDS out_json:=/ws/langgrasp/results/ros2_smoke.json; chown $(id -u):$(id -g) /ws/langgrasp/results/ros2_smoke.json"
