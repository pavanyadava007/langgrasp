"""latency_tracer: per-stage and end-to-end latency from header stamps.

Every message in the chain camera -> detections -> target_point -> grasp_pose -> joint_command ->
joint_command_safe carries the capture stamp of the camera frame it was derived from; the tracer records the
arrival wall time per (stage, stamp) and reports per-stage deltas. Publishes /langgrasp/latency
(diagnostic_msgs/DiagnosticArray) at 1 Hz and writes `out_json` (results/ros2_smoke.json) at shutdown, including
the last /langgrasp/sim_status and /langgrasp/policy_state messages and the safety status.
"""

from __future__ import annotations

import json
import os
import platform
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

from langgrasp_ros.logic import STAGES, LatencyTracker, stamp_to_ns


class LatencyTracer(Node):
    def __init__(self):
        super().__init__("latency_tracer")
        self.declare_parameter("out_json", "results/ros2_smoke.json")
        self.declare_parameter("label", "ROS 2 Humble in Docker on x86 EC2 host, CPU only, not Jetson")
        self.tracker = LatencyTracker()
        self.sim_status = None
        self.policy_state = None
        self.policy_done = None
        self.safety = None
        self.t_start = time.time()
        topics = {
            "camera": (Image, "/camera/color/image_raw"),
            "detections": (Detection2DArray, "/langgrasp/detections"),
            "target_point": (PointStamped, "/langgrasp/target_point"),
            "grasp_pose": (PoseStamped, "/langgrasp/grasp_pose"),
            "joint_command": (JointState, "/so101/joint_command"),
            "joint_command_safe": (JointState, "/so101/joint_command_safe"),
        }
        assert list(topics) == STAGES
        for stage, (cls, topic) in topics.items():
            self.create_subscription(cls, topic, self._make_cb(stage), 10)
        self.create_subscription(String, "/langgrasp/sim_status", self.on_sim_status, 2)
        self.create_subscription(String, "/langgrasp/policy_state", self.on_policy_state, 2)
        self.create_subscription(DiagnosticArray, "/langgrasp/status", self.on_safety, 2)
        self.pub = self.create_publisher(DiagnosticArray, "/langgrasp/latency", 2)
        self.create_timer(1.0, self.on_timer)

    def _make_cb(self, stage):
        def cb(msg):
            now = self.get_clock().now().nanoseconds
            self.tracker.record(stage, stamp_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec), now)

        return cb

    def on_sim_status(self, msg: String):
        self.sim_status = json.loads(msg.data)

    def on_policy_state(self, msg: String):
        self.policy_state = msg.data
        if msg.data.startswith("{"):
            self.policy_done = json.loads(msg.data)

    def on_safety(self, msg: DiagnosticArray):
        if msg.status:
            s = msg.status[0]
            self.safety = {"level": int.from_bytes(s.level, "little"), "message": s.message, "values": {kv.key: kv.value for kv in s.values}}

    def on_timer(self):
        summ = self.tracker.summary()
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus(name="langgrasp/latency", level=b"\x00", message="per-stage latency (ms, median)")
        for k, v in summ["per_stage_ms"].items():
            st.values.append(KeyValue(key=k, value=f"{v.get('median_ms', float('nan')):.2f} (n={v['n']})"))
        e2e = summ["end_to_end_ms"]
        st.values.append(KeyValue(key="end_to_end", value=f"{e2e.get('median_ms', float('nan')):.2f} (n={e2e['n']})"))
        arr.status.append(st)
        self.pub.publish(arr)

    def write(self):
        out = str(self.get_parameter("out_json").value)
        summ = self.tracker.summary()
        report = {
            "hardware": str(self.get_parameter("label").value),
            "machine": platform.machine(),
            "rmw": os.environ.get("RMW_IMPLEMENTATION", "default (rmw_fastrtps_cpp on Humble)"),
            "run_seconds": round(time.time() - self.t_start, 1),
            "latency": summ,
            "sim_status_last": self.sim_status,
            "policy_state_last": self.policy_state,
            "policy_done": self.policy_done,
            "safety_last": self.safety,
            "note": "stamps are capture times from the sim driver clock; per-stage = arrival delta between consecutive stages for the same capture stamp; the policy stage includes the IK for the first waypoint",
        }
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        self.get_logger().info(f"wrote {out}: e2e median {summ['end_to_end_ms'].get('median_ms')} ms, sim_status {self.sim_status}")


def main(args=None):
    rclpy.init(args=args)
    node = LatencyTracer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
