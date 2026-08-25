#!/bin/bash
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

# Kill rosbag first, gracefully where possible, so metadata is flushed.
pkill -INT -f "ros2 bag record" 2>/dev/null || true
sleep 3
pkill -9 -f "ros2 bag record" 2>/dev/null || true

# Kill ROS 2 launch and nodes.
pkill -9 -f "ros2 launch" 2>/dev/null || true
pkill -9 -f "ros2 run" 2>/dev/null || true
pkill -9 -f "_ros2_daemon" 2>/dev/null || true

# Kill baseline and mission nodes.
pkill -9 -f "vehicle_controller" 2>/dev/null || true
pkill -9 -f "vehicle_controller_openscvx" 2>/dev/null || true
pkill -9 -f "photographer_coordinator" 2>/dev/null || true
pkill -9 -f "poi_detection_node" 2>/dev/null || true
pkill -9 -f "cluster_manager" 2>/dev/null || true
pkill -9 -f "gcs" 2>/dev/null || true
pkill -9 -f "mission_controller" 2>/dev/null || true
pkill -9 -f "mission_timing" 2>/dev/null || true
pkill -9 -f "fleet_readiness_node" 2>/dev/null || true
pkill -9 -f "poi_scoring" 2>/dev/null || true
pkill -9 -f "position_logger" 2>/dev/null || true
pkill -9 -f "photo_capture_service" 2>/dev/null || true
pkill -9 -f "referee_node" 2>/dev/null || true
pkill -9 -f "spawner" 2>/dev/null || true
pkill -9 -f "mapping_node" 2>/dev/null || true

# Kill Gazebo.
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "ruby.*gz" 2>/dev/null || true
pkill -9 -f "gzserver" 2>/dev/null || true
pkill -9 -f "gzclient" 2>/dev/null || true

# Kill PX4 and middleware agents.
pkill -9 -f "px4" 2>/dev/null || true
pkill -9 -f "micrortps" 2>/dev/null || true
pkill -9 -f "micro_xrce" 2>/dev/null || true
pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true

echo "All pkill commands issued. Sleeping for 5 seconds..."
sleep 5
echo "Cleanup complete"

htop
