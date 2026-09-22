"""safety_node: wraps langgrasp.safety.SafetyMonitor (docs/FMEA.md H1..H8).

sub /so101/joint_states, /camera/color/image_raw, /langgrasp/command_text   heartbeats (staleness -> HOLD/ESTOP)
sub /so101/tcp_pose            geofence (H8/H5)
sub /langgrasp/grounding       grounding confidence + ambiguity gate (H1/H6) -> HOLD until a confident result
sub /so101/joint_command       policy output; clipped to joint limits and velocity (H4/H5/H8)
sub /langgrasp/estop_request   std_msgs/Bool: true latches ESTOP, false resets it (operator action)
pub /so101/joint_command_safe  sensor_msgs/JointState (the only command the driver listens to)
pub /langgrasp/estop           std_msgs/Bool
pub /langgrasp/status          diagnostic_msgs/DiagnosticArray
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

from langgrasp.safety import SafetyConfig, SafetyMonitor, SafetyState
from langgrasp_ros.logic import JOINT_NAMES


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")
        self.declare_parameter("max_joint_vel", 1.5)
        self.declare_parameter("camera_timeout", 0.5)
        self.declare_parameter("joint_states_timeout", 0.2)
        self.declare_parameter("command_timeout", 30.0)
        self.declare_parameter("grounding_conf_threshold", 0.35)
        self.declare_parameter("require_human_confirm", False)
        self.declare_parameter("gate_grounding", True)
        cfg = SafetyConfig(
            max_joint_vel=float(self.get_parameter("max_joint_vel").value),
            staleness_s={
                "camera": float(self.get_parameter("camera_timeout").value),
                "joint_states": float(self.get_parameter("joint_states_timeout").value),
                "command": float(self.get_parameter("command_timeout").value),
            },
            grounding_conf_threshold=float(self.get_parameter("grounding_conf_threshold").value),
            require_human_confirm=bool(self.get_parameter("require_human_confirm").value),
        )
        self.mon = SafetyMonitor(cfg)
        self.gate = bool(self.get_parameter("gate_grounding").value)
        if self.gate:
            self.mon.hold("grounding", "no grounding result yet")
        self.q_now = None
        self.jaw_now = None
        self.last_state = None
        self.n_cmd = 0
        self.n_blocked = 0
        self.pub_safe = self.create_publisher(JointState, "/so101/joint_command_safe", 10)
        self.pub_estop = self.create_publisher(Bool, "/langgrasp/estop", 2)
        self.pub_status = self.create_publisher(DiagnosticArray, "/langgrasp/status", 2)
        self.create_subscription(JointState, "/so101/joint_states", self.on_joint_states, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 2)
        self.create_subscription(String, "/langgrasp/command_text", self.on_command_text, 1)
        self.create_subscription(PointStamped, "/so101/tcp_pose", self.on_tcp, 10)
        self.create_subscription(String, "/langgrasp/grounding", self.on_grounding, 2)
        self.create_subscription(JointState, "/so101/joint_command", self.on_joint_command, 10)
        self.create_subscription(Bool, "/langgrasp/estop_request", self.on_estop_request, 2)
        self.create_timer(0.05, self.on_watchdog)
        self.create_timer(0.2, self.on_status)
        self.t0 = self.now_s()
        # grace period: topics count as fresh until first seen, for the first 2 s after start
        for topic in cfg.staleness_s:
            self.mon.heartbeat(topic, self.t0 + 2.0)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_joint_states(self, msg: JointState):
        self.mon.heartbeat("joint_states", self.now_s())
        self.q_now = np.array(msg.position[:5], dtype=float)
        self.jaw_now = float(msg.position[5]) if len(msg.position) > 5 else None

    def on_image(self, msg: Image):
        self.mon.heartbeat("camera", self.now_s())

    def on_command_text(self, msg: String):
        self.mon.heartbeat("command", self.now_s())

    def on_tcp(self, msg: PointStamped):
        reasons = self.mon.check_tcp([msg.point.x, msg.point.y, msg.point.z])
        if reasons and self.mon.state == SafetyState.HOLD:
            self.get_logger().warn("geofence: " + "; ".join(reasons), throttle_duration_sec=1.0)

    def on_grounding(self, msg: String):
        if not self.gate:
            return
        d = json.loads(msg.data)
        ok, reason = self.mon.gate_grounding(d.get("scores", []), int(d.get("n_candidates", 0)))
        if ok:
            self.mon.release("grounding")
        else:
            self.mon.hold("grounding", f"grounding gate: {reason}")
            self.get_logger().warn(f"grounding gate blocked: {reason}", throttle_duration_sec=2.0)

    def on_estop_request(self, msg: Bool):
        if msg.data:
            self.mon.estop("operator estop request")
        else:
            self.mon.reset_estop()
            self.get_logger().info("estop reset by operator")

    def on_joint_command(self, msg: JointState):
        if self.q_now is None:
            return
        self.n_cmd += 1
        q_target = np.array(msg.position[:5], dtype=float)
        jaw = float(msg.position[5]) if len(msg.position) > 5 else self.jaw_now
        q_safe, reasons = self.mon.check_joint_command(q_target, self.q_now, dt=self.mon.cfg.control_dt)
        if self.mon.state in (SafetyState.HOLD, SafetyState.ESTOP):
            self.n_blocked += 1
            jaw = self.jaw_now if self.jaw_now is not None else jaw
        for r in reasons:
            self.get_logger().warn(r, throttle_duration_sec=1.0)
        out = JointState()
        out.header = msg.header
        out.name = JOINT_NAMES
        out.position = [float(v) for v in q_safe] + ([float(jaw)] if jaw is not None else [])
        self.pub_safe.publish(out)

    def on_watchdog(self):
        self.mon.check_staleness(self.now_s())
        state = self.mon.state
        if state != self.last_state:
            self.get_logger().info(f"safety state {self.last_state.value if self.last_state else None} -> {state.value}: {self.mon.diagnostics()['message']}")
            self.last_state = state
        self.pub_estop.publish(Bool(data=state == SafetyState.ESTOP))

    def on_status(self):
        d = self.mon.diagnostics()
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = d["name"]
        st.level = bytes([d["level"]])
        st.message = d["message"]
        st.hardware_id = "so101_sim"
        st.values = [KeyValue(key=v["key"], value=v["value"]) for v in d["values"]] + [KeyValue(key="n_commands", value=str(self.n_cmd)), KeyValue(key="n_blocked", value=str(self.n_blocked))]
        arr.status.append(st)
        self.pub_status.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
