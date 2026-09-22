# ROS 2 Humble package: `ros2_ws/src/langgrasp_ros`

Verified: built with colcon and run end to end inside `ros:humble` Docker on an x86 EC2 host, CPU only, no
torch, no GPU in the container. Not verified on a Jetson (none available) and not verified on a host ROS
install (ROS 2 is not installed on the host). Numbers below are from `results/ros2_smoke.json`, one 30 s run.

## Nodes and topics

| Node | Subscribes | Publishes | Notes |
|---|---|---|---|
| `sim_driver_node` | `/so101/joint_command_safe` (JointState, 5 arm + Jaw) | `/camera/color/image_raw` (rgb8), `/camera/aligned_depth_to_color/image_raw` (32FC1 m), `/camera/color/camera_info`, `/so101/joint_states`, `/so101/tcp_pose` (PointStamped), `/langgrasp/sim_status` (String JSON) | `LangGraspEnv` stepped at 10 Hz by a timer; images at 5 Hz, 240x320 by default, stamped at capture. Writes the scenario ground truth to `scene_json` for the oracle grounder. Logs `grasped / lifted / in_tray` of the target every second. |
| `command_node` | | `/langgrasp/command_text` (String) | From the `command` parameter, stdin (`stdin:=true`), or the scenario command in `scene_json`. Republished every 1 s so it doubles as the command heartbeat. |
| `grounding_node` | `/langgrasp/command_text`, `/camera/color/image_raw` | `/langgrasp/grounding` (String JSON: scores, target, psi, width), `/langgrasp/detections` (vision_msgs/Detection2DArray), `/langgrasp/target_point` (PointStamped, world) | `mode:=oracle` scores the objects of `scene_json` lexically (0.5 per matched colour / kind word) and projects their ground-truth positions into the image; no GPU. `mode:=gdino` imports `langgrasp.perception` if present and otherwise falls back to oracle with a warning; the detector call itself is not wired in this package (that API belongs to the perception track). |
| `grasp_node` | `/langgrasp/target_point`, `/langgrasp/grounding` | `/langgrasp/grasp_pose` (PoseStamped, top-down orientation with jaw yaw psi), `/langgrasp/grasp_meta` | Pure numpy, no kinematics model. |
| `policy_node` | `/langgrasp/grasp_pose`, `/langgrasp/grasp_meta`, `/so101/joint_states` | `/so101/joint_command` (JointState), `/langgrasp/policy_state` | Modular policy: `logic.PickExecutor` mirrors `PickPlaceController.run` (approach, hover, descend, close, lift, transport, lower, release, retreat, settle) with `ArmKinematics` on the SO-ARM100 model loaded from the package assets. One command per incoming joint_states message. |
| `safety_node` | `/so101/joint_states`, `/camera/color/image_raw`, `/langgrasp/command_text` (heartbeats), `/so101/tcp_pose`, `/langgrasp/grounding`, `/so101/joint_command`, `/langgrasp/estop_request` (Bool) | `/so101/joint_command_safe`, `/langgrasp/estop` (Bool), `/langgrasp/status` (DiagnosticArray) | Wraps `langgrasp.safety.SafetyMonitor` (docs/FMEA.md). The driver listens only to the `_safe` topic. Starts in HOLD until a grounding result passes the gate. |
| `latency_tracer` | every stage above plus `/langgrasp/sim_status`, `/langgrasp/policy_state`, `/langgrasp/status` | `/langgrasp/latency` (DiagnosticArray, 1 Hz) | Per-stage latency from the propagated capture stamps; writes `out_json` at shutdown. |

Launch: `ros2 launch langgrasp_ros langgrasp_sim.launch.py seed:=1001 stratum:=seen mode:=oracle run_seconds:=30 out_json:=results/ros2_smoke.json render:=true command:=""`.
The launch shuts everything down when the driver exits (`run_seconds`), which is when the tracer writes its file.

Message-free logic (image encoding, camera info, projection, oracle grounding, quaternions, the pick plan and
executor, the latency tracker) is in `langgrasp_ros/logic.py` and is tested on the host without rclpy in
`tests/test_ros2_msgs.py`, including a MuJoCo test that `PickExecutor` reproduces `PickPlaceController`.

## What was run and how

Recipe (`docker/ros2_smoke.sh`, image `docker/Dockerfile.ros2`):

1. `FROM ros:humble` (Ubuntu 22.04, Python 3.10), apt: `python3-pip python3-colcon-common-extensions
   ros-humble-vision-msgs ros-humble-diagnostic-msgs libegl1 libgl1-mesa-dri libgles2 libosmesa6`.
2. `pip3 install -U pip setuptools wheel` (the stock pip 22.0.2 built the repo as `UNKNOWN-0.0.0` because it
   does not read `[project]` metadata; this was the first failure), then `numpy<2.3 mujoco==3.8.1 scipy pyyaml
   opencv-python-headless imageio[ffmpeg] pillow`. No torch.
3. At container start (`docker/ros2_entrypoint.sh`): `pip3 install --no-deps /ws/langgrasp` from the mounted
   repo, copy `ros2_ws/src` to a scratch workspace, `colcon build --symlink-install`, source.
4. `MUJOCO_GL=egl ros2 launch langgrasp_ros langgrasp_sim.launch.py seed:=1001 run_seconds:=30`.

Rendering path that worked: **EGL through Mesa (llvmpipe) inside the container, no GPU passed through**.
The first rgb+depth frame took about 1.1 s (context creation), then a median of 47.2 ms per rgb+depth pair at
240x320 (`sim_status_last.render_rgbd_ms_median`, 150 frames). OSMesa was installed as the fallback and was
not needed. The driver's "joint states only" fallback (zero images) exists for hosts without any GL and was
not exercised.

Bugs found by the run, fixed: `policy_node` used `self.executor`, which shadows `rclpy.node.Node.executor`
(crash on the first joint_states callback); the grasp node received the target point before the psi/width
metadata (grasped with psi=0 and still placed, by luck), fixed by publishing the grounding JSON first.

## Measured (ROS 2 Humble in Docker on x86 EC2 host, CPU only, not Jetson)

Seed 1001, `pick the blue screwdriver`, oracle grounding, 30 s run, default QoS, rmw_fastrtps_cpp:

| Stage (arrival delta for the same capture stamp) | median ms | n |
|---|---|---|
| camera capture -> image arrival at tracer (includes the 47 ms software render, the stamp is taken before rendering) | 72.8 (p90 86.0) | 150 |
| camera -> detections | 1.51 | 1 |
| detections -> target_point | 0.26 | 1 |
| target_point -> grasp_pose | 0.24 | 1 |
| grasp_pose -> first joint_command (best_psi: five IK solves, plus the first waypoint IK) | 54.2 | 1 |
| joint_command -> joint_command_safe (safety node) | 1.04 (p90 1.26) | 65 |
| end to end, camera frame -> first safe joint command | 57.0 | 1 |

Outcome: `in_tray: true` for the target after 69 joint commands; `ik_ok: true`; policy `max_pos_err` 0.236 m
(intermediate waypoints near the raised observe pose are not strictly reachable, same behaviour as the
in-process controller, whose value is 0.064 m without the safety velocity clip; the clip slows tracking and
grows the transient error, the final waypoints converge). Safety: 51 velocity clips and 7 joint-limit clips
during the pick, 0 commands blocked, state RUN throughout, then ESTOP ("joint_states stale 0.22 s > 0.20 s")
when the driver shut down at `run_seconds`, which is the intended behaviour.

Caveats: one pick, one seed, n=1 for the single-shot stages; Python nodes with default single-threaded
executors; wall-clock stamps from the container clock; a run on the host GPU would change the render number
and nothing else materially. The grounding stage is the oracle, not a detector, so "camera -> detections" is
JSON lookup time, not perception time. Perception latency is measured by the perception track on the L4.

## RMW / domain

Default RMW on Humble is `rmw_fastrtps_cpp`; `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` works after
`apt-get install ros-humble-rmw-cyclonedds-cpp` (not tested here). Set `ROS_DOMAIN_ID` to isolate from other
ROS traffic on the same LAN; Docker's default bridge network does not forward DDS multicast, so to talk to
nodes outside the container run with `--network host` (the smoke run is self-contained and does not need it).
