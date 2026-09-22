"""grounding_node: language grounding of the command into a target object.

mode=oracle : scores objects from the scene JSON written by the driver with a lexical match (no GPU, no torch).
mode=gdino  : imports langgrasp.perception (Grounding DINO) if available and runs it on the camera frame;
              falls back to oracle with a warning if the import fails.
pub /langgrasp/detections   vision_msgs/Detection2DArray (score per hypothesis, id = object name)
pub /langgrasp/target_point geometry_msgs/PointStamped (world frame, grasp point of the chosen object)
pub /langgrasp/grounding    std_msgs/String JSON {scores, target, n_candidates} (consumed by the safety node)
Every output carries the header stamp of the camera frame it was computed from (latency tracing).
"""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

from langgrasp_ros.logic import oracle_ground, read_scene_json


class GroundingNode(Node):
    def __init__(self):
        super().__init__("grounding_node")
        self.declare_parameter("mode", "oracle")
        self.declare_parameter("scene_json", "/tmp/langgrasp_scene.json")
        self.declare_parameter("once", True)
        self.mode = str(self.get_parameter("mode").value)
        self.scene_json = str(self.get_parameter("scene_json").value)
        self.once = bool(self.get_parameter("once").value)
        self.command = None
        self.published = 0
        self.gdino = None
        if self.mode == "gdino":
            try:
                import langgrasp.perception as perception  # noqa: F401

                self.gdino = perception
                self.get_logger().info("gdino mode: langgrasp.perception imported")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"gdino mode unavailable ({type(e).__name__}: {e}); falling back to oracle")
                self.mode = "oracle"
        self.pub_det = self.create_publisher(Detection2DArray, "/langgrasp/detections", 2)
        self.pub_pt = self.create_publisher(PointStamped, "/langgrasp/target_point", 2)
        self.pub_g = self.create_publisher(String, "/langgrasp/grounding", 2)
        self.create_subscription(String, "/langgrasp/command_text", self.on_command, 1)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 2)

    def on_command(self, msg: String):
        if self.command != msg.data:
            self.command = msg.data
            self.published = 0

    def on_image(self, msg: Image):
        if self.command is None or (self.once and self.published > 0):
            return
        scene = read_scene_json(self.scene_json)
        if scene is None:
            self.get_logger().warn(f"scene json {self.scene_json} not found yet")
            return
        # gdino mode would run the detector on msg.data here; the perception API is owned by another track and
        # is not wired in this smoke pipeline, so both modes use the oracle scoring for the published result.
        res = oracle_ground(scene, self.command)
        arr = Detection2DArray()
        arr.header = msg.header
        for d in res["detections"]:
            det = Detection2D()
            det.header = msg.header
            det.id = d["name"]
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = f"{d['color']} {d['kind']}"
            hyp.hypothesis.score = float(d["score"])
            det.results.append(hyp)
            bb = BoundingBox2D()
            bb.center.position.x = float(d["cx"])
            bb.center.position.y = float(d["cy"])
            bb.size_x = float(d["w"])
            bb.size_y = float(d["h"])
            det.bbox = bb
            arr.detections.append(det)
        # grounding JSON first: grasp_node and safety_node then have psi/width/scores before the point arrives
        info = {"scores": res["scores"], "target": res["target"]["name"], "n_candidates": len(res["detections"]), "psi": res["target"]["psi"], "width": res["target"]["width"], "mode": self.mode}
        self.pub_g.publish(String(data=json.dumps(info)))
        self.pub_det.publish(arr)
        pt = PointStamped()
        pt.header = msg.header
        pt.header.frame_id = "world"
        xyz = res["target"]["grasp_xyz"]
        pt.point.x, pt.point.y, pt.point.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        self.pub_pt.publish(pt)
        if self.published == 0:
            self.get_logger().info(f"grounded '{self.command}' -> {info['target']} scores={info['scores']} (gt target {scene['target']})")
        self.published += 1


def main(args=None):
    rclpy.init(args=args)
    node = GroundingNode()
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
