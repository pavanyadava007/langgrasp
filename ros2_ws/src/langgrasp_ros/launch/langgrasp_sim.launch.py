"""ros2 launch langgrasp_ros langgrasp_sim.launch.py seed:=1001 mode:=oracle run_seconds:=25 out_json:=/ws/langgrasp/results/ros2_smoke.json"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    seed = LaunchConfiguration("seed")
    stratum = LaunchConfiguration("stratum")
    mode = LaunchConfiguration("mode")
    scene_json = LaunchConfiguration("scene_json")
    run_seconds = LaunchConfiguration("run_seconds")
    out_json = LaunchConfiguration("out_json")
    render = LaunchConfiguration("render")
    command = LaunchConfiguration("command")

    driver = Node(
        package="langgrasp_ros",
        executable="sim_driver_node",
        name="sim_driver",
        output="screen",
        parameters=[{"seed": seed, "stratum": stratum, "scene_json": scene_json, "run_seconds": run_seconds, "render": render}],
    )
    nodes = [
        driver,
        Node(package="langgrasp_ros", executable="command_node", name="command_node", output="screen", parameters=[{"scene_json": scene_json, "command": command}]),
        Node(package="langgrasp_ros", executable="grounding_node", name="grounding_node", output="screen", parameters=[{"mode": mode, "scene_json": scene_json}]),
        Node(package="langgrasp_ros", executable="grasp_node", name="grasp_node", output="screen"),
        Node(package="langgrasp_ros", executable="policy_node", name="policy_node", output="screen"),
        Node(package="langgrasp_ros", executable="safety_node", name="safety_node", output="screen"),
        Node(package="langgrasp_ros", executable="latency_tracer", name="latency_tracer", output="screen", parameters=[{"out_json": out_json}]),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="1001"),
            DeclareLaunchArgument("stratum", default_value="seen"),
            DeclareLaunchArgument("mode", default_value="oracle"),
            DeclareLaunchArgument("scene_json", default_value="/tmp/langgrasp_scene.json"),
            DeclareLaunchArgument("run_seconds", default_value="0.0"),
            DeclareLaunchArgument("out_json", default_value="results/ros2_smoke.json"),
            DeclareLaunchArgument("render", default_value="true"),
            DeclareLaunchArgument("command", default_value=""),
            *nodes,
            RegisterEventHandler(OnProcessExit(target_action=driver, on_exit=[Shutdown(reason="sim driver exited")])),
        ]
    )
