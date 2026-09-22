"""sim_driver_node: wraps LangGraspEnv. Publishes camera + joint states, applies joint commands at 10 Hz.

Topics
  pub /camera/color/image_raw                 sensor_msgs/Image rgb8 (stamped at capture)
  pub /camera/aligned_depth_to_color/image_raw sensor_msgs/Image 32FC1 metres (same stamp)
  pub /camera/color/camera_info               sensor_msgs/CameraInfo
  pub /so101/joint_states                     sensor_msgs/JointState (5 arm + Jaw)
  pub /so101/tcp_pose                         geometry_msgs/PointStamped (world TCP, for the safety geofence)
  pub /langgrasp/sim_status                   std_msgs/String JSON {t, grasped, lifted, in_tray, target}
  sub /so101/joint_command_safe (param command_topic) sensor_msgs/JointState positions = 5 arm + jaw
Parameters: seed, stratum, scene_json, command_topic, image_height, image_width, camera_hz, render (bool),
  run_seconds (0 = forever).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

from langgrasp_ros.logic import (
    JOINT_NAMES,
    camera_info_fields,
    depth_fields,
    image_fields,
    scene_to_json,
    write_scene_json,
)


class SimDriverNode(Node):
    def __init__(self):
        super().__init__("sim_driver")
        self.declare_parameter("seed", 1001)
        self.declare_parameter("stratum", "seen")
        self.declare_parameter("scene_json", "/tmp/langgrasp_scene.json")
        self.declare_parameter("command_topic", "/so101/joint_command_safe")
        self.declare_parameter("image_height", 240)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("camera_hz", 5.0)
        self.declare_parameter("render", True)
        self.declare_parameter("run_seconds", 0.0)
        p = {k: self.get_parameter(k).value for k in ("seed", "stratum", "scene_json", "command_topic", "image_height", "image_width", "camera_hz", "render", "run_seconds")}
        self.p = p

        from langgrasp.sim.env import CONTROL_HZ, LangGraspEnv
        from langgrasp.sim.scenarios import make_scenario

        self.env = LangGraspEnv(seed=int(p["seed"]), render=bool(p["render"]))
        self.scenario = make_scenario(int(p["seed"]), str(p["stratum"]))
        self.env.reset(self.scenario)
        self.hw = (int(p["image_height"]), int(p["image_width"]))
        self.K = self.env.camera_intrinsics("front", size=self.hw)
        self.T = self.env.camera_extrinsics("front")
        grasp = {o.name: self.env.grasp_point(o.name) for o in self.scenario.objects}
        write_scene_json(scene_to_json(self.scenario, grasp, self.K, self.T, self.hw), str(p["scene_json"]))
        self.get_logger().info(f"scenario seed={self.scenario.seed} target={self.scenario.target} command='{self.scenario.command}' scene_json={p['scene_json']}")

        self.render_ok = bool(p["render"])
        self.render_mode = os.environ.get("MUJOCO_GL", "default")
        if self.render_ok:
            try:
                t0 = time.perf_counter()
                self.env.render("front", size=self.hw)
                self.env.render("front", size=self.hw, depth=True)
                self.get_logger().info(f"rendering works (MUJOCO_GL={self.render_mode}), first rgb+depth frame {1e3 * (time.perf_counter() - t0):.0f} ms")
            except Exception as e:  # noqa: BLE001
                self.render_ok = False
                self.get_logger().error(f"rendering unavailable (MUJOCO_GL={self.render_mode}): {type(e).__name__}: {e}; publishing joint states only")

        self.pub_rgb = self.create_publisher(Image, "/camera/color/image_raw", 2)
        self.pub_depth = self.create_publisher(Image, "/camera/aligned_depth_to_color/image_raw", 2)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 2)
        self.pub_js = self.create_publisher(JointState, "/so101/joint_states", 10)
        self.pub_tcp = self.create_publisher(PointStamped, "/so101/tcp_pose", 10)
        self.pub_status = self.create_publisher(String, "/langgrasp/sim_status", 2)
        self.sub_cmd = self.create_subscription(JointState, str(p["command_topic"]), self.on_command, 10)
        self.q_cmd = None
        self.jaw_cmd = None
        self.n_cmd = 0
        self.t_start = time.monotonic()
        self.control_hz = CONTROL_HZ
        self.camera_every = max(1, int(round(CONTROL_HZ / float(p["camera_hz"]))))
        self.tick = 0
        self.render_ms: list = []
        self.timer = self.create_timer(1.0 / CONTROL_HZ, self.on_tick)
        self.status_timer = self.create_timer(1.0, self.on_status)

    def on_command(self, msg: JointState):
        pos = list(msg.position)
        if len(pos) < 5:
            self.get_logger().warn(f"joint command with {len(pos)} positions ignored")
            return
        self.q_cmd = np.array(pos[:5], dtype=float)
        self.jaw_cmd = float(pos[5]) if len(pos) > 5 else None
        self.n_cmd += 1

    def on_tick(self):
        obs = self.env.step(self.q_cmd, self.jaw_cmd)
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = "so101_base"
        js.name = JOINT_NAMES
        js.position = [float(v) for v in obs["q_arm"]] + [float(obs["jaw"])]
        self.pub_js.publish(js)
        tcp = PointStamped()
        tcp.header.stamp = stamp
        tcp.header.frame_id = "world"
        tcp.point.x, tcp.point.y, tcp.point.z = (float(v) for v in obs["tcp_pos"])
        self.pub_tcp.publish(tcp)
        if self.tick % self.camera_every == 0:
            self.publish_camera()
        self.tick += 1
        run_s = float(self.p["run_seconds"])
        if run_s > 0 and time.monotonic() - self.t_start > run_s:
            self.on_status()
            self.get_logger().info("run_seconds elapsed, shutting down")
            raise SystemExit(0)

    def publish_camera(self):
        stamp = self.get_clock().now().to_msg()  # capture time
        if self.render_ok:
            t0 = time.perf_counter()
            rgb = self.env.render("front", size=self.hw)
            depth = self.env.render("front", size=self.hw, depth=True).astype(np.float32)
            self.render_ms.append(1e3 * (time.perf_counter() - t0))
        else:
            rgb = np.zeros((*self.hw, 3), dtype=np.uint8)
            depth = np.zeros(self.hw, dtype=np.float32)
        for msg_cls, fields, pub in ((Image, image_fields(rgb), self.pub_rgb), (Image, depth_fields(depth), self.pub_depth)):
            m = msg_cls()
            m.header.stamp = stamp
            m.header.frame_id = "camera_color_optical_frame"
            for k, v in fields.items():
                setattr(m, k, v)
            pub.publish(m)
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera_color_optical_frame"
        for k, v in camera_info_fields(self.K, *self.hw).items():
            setattr(info, k, v)
        self.pub_info.publish(info)

    def on_status(self):
        name = self.scenario.target
        st = {
            "t": self.env.t,
            "target": name,
            "grasped": bool(self.env.is_grasped(name)),
            "lifted": bool(self.env.is_lifted(name)),
            "in_tray": bool(self.env.in_tray(name)),
            "n_commands": self.n_cmd,
            "render": self.render_ok,
            "render_mode": self.render_mode,
            "image_hw": list(self.hw),
            "render_rgbd_ms_median": float(np.median(self.render_ms)) if self.render_ms else None,
            "n_frames": len(self.render_ms),
        }
        self.pub_status.publish(String(data=json.dumps(st)))
        self.get_logger().info(f"sim t={st['t']} target={name} grasped={st['grasped']} lifted={st['lifted']} in_tray={st['in_tray']} commands={self.n_cmd}")


def main(args=None):
    rclpy.init(args=args)
    node = SimDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
