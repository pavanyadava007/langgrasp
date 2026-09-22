"""command_node: publishes /langgrasp/command_text (std_msgs/String) from the `command` parameter, the scene
JSON written by the driver (parameter command="" means "use the scenario command"), or stdin once.
Republishes every `period` seconds so late subscribers and the safety heartbeat see it."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from langgrasp.language.stt import TextCommandSource, normalize_command
from langgrasp_ros.logic import read_scene_json


class CommandNode(Node):
    def __init__(self):
        super().__init__("command_node")
        self.declare_parameter("command", "")
        self.declare_parameter("scene_json", "/tmp/langgrasp_scene.json")
        self.declare_parameter("stdin", False)
        self.declare_parameter("period", 1.0)
        self.text = None
        cmd = str(self.get_parameter("command").value)
        if cmd:
            self.text = normalize_command(cmd)
        elif bool(self.get_parameter("stdin").value):
            self.text = TextCommandSource().next()
        self.pub = self.create_publisher(String, "/langgrasp/command_text", 1)
        self.timer = self.create_timer(float(self.get_parameter("period").value), self.on_timer)
        self.n = 0

    def on_timer(self):
        if self.text is None:
            scene = read_scene_json(str(self.get_parameter("scene_json").value))
            if scene is None:
                return
            self.text = normalize_command(scene["command"])
        self.pub.publish(String(data=self.text))
        if self.n == 0:
            self.get_logger().info(f"command: '{self.text}'")
        self.n += 1


def main(args=None):
    rclpy.init(args=args)
    node = CommandNode()
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
