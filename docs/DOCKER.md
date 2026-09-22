# Docker images

| File | Base | Purpose | Status |
|---|---|---|---|
| `docker/Dockerfile.ros2` | `ros:humble` (Ubuntu 22.04, x86_64) | ROS 2 Humble + MuJoCo CPU + Mesa EGL for the `langgrasp_ros` smoke pipeline. No torch. | **Built and run here** (x86 EC2 host, no GPU passed to the container). See docs/ROS2.md. |
| `docker/Dockerfile.workstation` | `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` | Full stack: `pip install -e .[perception,learning,speech,dev]`. | **Not built**: the base pull plus the extras exceed the 3 GB disk budget of the shared host (19 GB free, 342 GB used). Dependency list mirrors `pyproject.toml`. |
| `docker/Dockerfile.jetson` | `dustynv/ros:humble-ros-base-l4t-r36.4.0` | ROS 2 Humble on JetPack 6 with `langgrasp_ros`, vision_msgs from source. | **Not built or tested, no Jetson available.** Pins and steps are a transcription of the x86 recipe. |

## Usage

ROS 2 smoke pipeline (what was verified):

```bash
docker/ros2_smoke.sh 1001 30.0 egl        # seed, run_seconds, MUJOCO_GL
# equivalent to:
docker build -t langgrasp-ros2:humble -f docker/Dockerfile.ros2 docker/
docker run --rm -e MUJOCO_GL=egl -v $PWD:/ws/langgrasp langgrasp-ros2:humble \
  bash -c "ros2 launch langgrasp_ros langgrasp_sim.launch.py seed:=1001 run_seconds:=30 out_json:=/ws/langgrasp/results/ros2_smoke.json"
```

The repo is mounted, not copied: the image carries only dependencies, `docker/ros2_entrypoint.sh` installs the
mounted package (`pip install --no-deps`), colcon-builds `ros2_ws/src` into `/ws/ros2_ws` inside the
container (the host tree stays clean) and sources it before running the command. Interactive shell:
`docker run --rm -it -v $PWD:/ws/langgrasp langgrasp-ros2:humble bash`.

Image size after build: see `docker images langgrasp-ros2` (ros:humble is about 0.7 GB, the Python deps add
under 1 GB). The container runs as root, so the results file is `chown`ed back to the host uid by
`ros2_smoke.sh`.

Workstation image (unverified recipe):

```bash
docker build -t langgrasp:workstation -f docker/Dockerfile.workstation .
docker run --rm --gpus all -it -v $PWD:/ws/langgrasp langgrasp:workstation python -c "import langgrasp"
```

Jetson image (unverified recipe): `docker build -t langgrasp:jetson -f docker/Dockerfile.jetson .` on the
device with the NVIDIA container runtime; run with `--runtime nvidia --network host`.

## ROS 2 environment variables

* `ROS_DOMAIN_ID` (0..101): all nodes of one pipeline must share it; pick a non-zero value on a shared LAN.
* `RMW_IMPLEMENTATION`: `rmw_fastrtps_cpp` (Humble default, used in the smoke run) or `rmw_cyclonedds_cpp`
  after installing `ros-humble-rmw-cyclonedds-cpp`. Cyclone is the usual choice on Jetson for lower CPU
  overhead with large image topics; not measured here.
* `--network host` is needed for DDS discovery across the container boundary; the smoke run keeps all nodes
  in one container and does not need it.
* `MUJOCO_GL`: `egl` worked in the container through Mesa llvmpipe (no GPU); `osmesa` is installed as the
  fallback and was not needed; on a GPU host with the NVIDIA runtime, EGL uses the device.

## Proxy note (this build host)

The Docker daemon on the build host injects the corporate proxy variables into containers, so apt and pip
work inside without extra flags. On a host without that, pass `--build-arg http_proxy=... --build-arg
https_proxy=...` or configure `~/.docker/config.json` proxies.
