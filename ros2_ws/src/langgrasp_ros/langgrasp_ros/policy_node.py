"""policy_node (modular): grasp pose -> Cartesian pick sequence -> joint commands.

Uses ArmKinematics (langgrasp.sim.kinematics) on the SO-ARM100 model from the package assets; it never steps a
simulation. One joint command is issued per incoming joint_states message (10 Hz), so the driver and the policy
run in lock-step. The first command of a sequence carries the grasp pose's header stamp (capture time) for the
latency tracer; later commands carry the joint_states stamp.
sub /langgrasp/grasp_pose geometry_msgs/PoseStamped, /langgrasp/grasp_meta std_msgs/String, /so101/joint_states
pub /so101/joint_command sensor_msgs/JointState (5 arm + Jaw), /langgrasp/policy_state std_msgs/String
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from langgrasp_ros.logic import JOINT_NAMES, PickExecutor, load_arm_kinematics, quaternion_to_psi


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("place", True)
        self.kin = load_arm_kinematics()
        self.pick: PickExecutor | None = None
        self.q_meas = None
        self.meta = {"width": 0.025}
        self.first_stamp = None
        self.state = "idle"
        self.pub = self.create_publisher(JointState, "/so101/joint_command", 10)
        self.pub_state = self.create_publisher(String, "/langgrasp/policy_state", 2)
        self.create_subscription(String, "/langgrasp/grasp_meta", self.on_meta, 2)
        self.create_subscription(PoseStamped, "/langgrasp/grasp_pose", self.on_pose, 2)
        self.create_subscription(JointState, "/so101/joint_states", self.on_joint_states, 10)
        self.get_logger().info("policy_node ready (modular Cartesian pick, ArmKinematics from package assets)")

    def on_meta(self, msg: String):
        self.meta = json.loads(msg.data)

    def on_pose(self, msg: PoseStamped):
        if self.pick is not None:
            return  # one pick per run
        o = msg.pose.orientation
        psi = quaternion_to_psi(o.x, o.y, o.z, o.w)
        xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.pick = PickExecutor(self.kin, xyz, psi, float(self.meta.get("width", 0.025)), place=bool(self.get_parameter("place").value))
        self.first_stamp = msg.header.stamp
        self.state = "executing"
        self.pub_state.publish(String(data=self.state))
        self.get_logger().info(f"starting pick at {np.round(xyz, 4).tolist()} psi={psi:.3f}")

    def on_joint_states(self, msg: JointState):
        self.q_meas = np.array(msg.position[:5], dtype=float)
        if self.pick is None or self.pick.done:
            return
        out = self.pick.tick(self.q_meas)
        if out is None:
            return
        q, jaw, phase = out
        cmd = JointState()
        cmd.header.stamp = self.first_stamp if self.first_stamp is not None else msg.header.stamp
        self.first_stamp = None
        cmd.header.frame_id = phase
        cmd.name = JOINT_NAMES
        cmd.position = [float(v) for v in q] + [float(jaw)]
        self.pub.publish(cmd)
        if phase != self.state:
            self.state = phase
            self.pub_state.publish(String(data=phase))
            self.get_logger().info(f"phase {phase} (tick {self.pick.ticks})")
        if self.pick.done:
            self.state = "done"
            self.pub_state.publish(String(data=json.dumps({"state": "done", "ticks": self.pick.ticks, "ik_ok": self.pick.ik_ok, "max_pos_err_m": self.pick.max_pos_err})))
            self.get_logger().info(f"pick sequence done: ticks={self.pick.ticks} ik_ok={self.pick.ik_ok} max_pos_err={self.pick.max_pos_err * 1e3:.1f} mm")


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
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
