"""grasp_node: target point -> grasp pose (top-down, jaw yaw psi). No kinematics model needed.

sub /langgrasp/target_point geometry_msgs/PointStamped
sub /langgrasp/grounding    std_msgs/String JSON (psi and width of the chosen object; oracle values)
pub /langgrasp/grasp_pose   geometry_msgs/PoseStamped (position = grasp point, orientation = top-down frame)
pub /langgrasp/grasp_meta   std_msgs/String JSON {psi, width, name}
"""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

from langgrasp_ros.logic import grasp_from_target


class GraspNode(Node):
    def __init__(self):
        super().__init__("grasp_node")
        self.meta = {"psi": 0.0, "width": 0.025, "name": ""}
        self.pub = self.create_publisher(PoseStamped, "/langgrasp/grasp_pose", 2)
        self.pub_meta = self.create_publisher(String, "/langgrasp/grasp_meta", 2)
        self.create_subscription(String, "/langgrasp/grounding", self.on_grounding, 2)
        self.create_subscription(PointStamped, "/langgrasp/target_point", self.on_point, 2)
        self.n = 0

    def on_grounding(self, msg: String):
        d = json.loads(msg.data)
        self.meta = {"psi": float(d.get("psi", 0.0)), "width": float(d.get("width", 0.025)), "name": d.get("target", "")}

    def on_point(self, msg: PointStamped):
        g = grasp_from_target({"grasp_xyz": [msg.point.x, msg.point.y, msg.point.z], **self.meta})
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = g["xyz"]
        qx, qy, qz, qw = g["quat_xyzw"]
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = qx, qy, qz, qw
        self.pub.publish(pose)
        self.pub_meta.publish(String(data=json.dumps({"psi": g["psi"], "width": g["width"], "name": g["name"]})))
        if self.n == 0:
            self.get_logger().info(f"grasp pose for {g['name']}: xyz={[round(v, 4) for v in g['xyz']]} psi={g['psi']:.3f}")
        self.n += 1


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
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
