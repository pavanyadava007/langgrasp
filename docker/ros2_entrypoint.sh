#!/bin/bash
# Installs the mounted langgrasp repo (base deps only), builds langgrasp_ros in a scratch workspace and sources it.
set -e
source /opt/ros/humble/setup.bash
if [ -d /ws/langgrasp ]; then
  pip3 install --no-cache-dir --no-deps -q /ws/langgrasp
  rm -rf /ws/ros2_ws && mkdir -p /ws/ros2_ws && cp -r /ws/langgrasp/ros2_ws/src /ws/ros2_ws/src
  cd /ws/ros2_ws && colcon build --symlink-install > /ws/colcon_build.log 2>&1 || { cat /ws/colcon_build.log; exit 1; }
  source /ws/ros2_ws/install/setup.bash
fi
exec "$@"
