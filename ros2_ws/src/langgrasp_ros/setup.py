import os
from glob import glob

from setuptools import setup

package_name = "langgrasp_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pavan Yadav Annappa",
    maintainer_email="pavanyadava07@gmail.com",
    description="ROS 2 Humble nodes for LangGrasp (simulated SO-101 language-guided pick-and-place).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_driver_node = langgrasp_ros.sim_driver_node:main",
            "command_node = langgrasp_ros.command_node:main",
            "grounding_node = langgrasp_ros.grounding_node:main",
            "grasp_node = langgrasp_ros.grasp_node:main",
            "policy_node = langgrasp_ros.policy_node:main",
            "safety_node = langgrasp_ros.safety_node:main",
            "latency_tracer = langgrasp_ros.latency_tracer:main",
        ],
    },
)
