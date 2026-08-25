# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_PHOTOGRAPHERS = [
    {"instance": 2, "x": -5.0, "y": -25.0, "yaw": 0.0},
    {"instance": 4, "x": 5.0, "y": -25.0, "yaw": 0.0},
    {"instance": 5, "x": 0.0, "y": -30.0, "yaw": 0.0},
]


def load_spawn_profiles(world_key: str):
    try:
        mm_share = get_package_share_directory("mission_manager")
        path = os.path.join(mm_share, "config", "spawn_profiles.yaml")
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            content = f.read().replace("\t", "  ")
            profiles = yaml.safe_load(content) or {}
        profile = profiles.get(world_key) or profiles.get("default") or {}
        return profile.get("fleet", [])
    except Exception:
        return []


def launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration("world").perform(context)
    difficulty = LaunchConfiguration("difficulty").perform(context)
    world_model_source = LaunchConfiguration("world_model_source").perform(context)
    map_min_known_ratio = LaunchConfiguration("map_min_known_ratio").perform(context)
    map_min_stable_seconds = LaunchConfiguration("map_min_stable_seconds").perform(context)
    cluster_required_known_ratio = LaunchConfiguration("cluster_required_known_ratio").perform(context)
    require_photographer_startup_readiness = LaunchConfiguration("require_photographer_startup_readiness").perform(
        context
    ).strip().lower() in ["1", "true", "yes", "on"]
    require_explorer_startup_readiness = LaunchConfiguration("require_explorer_startup_readiness").perform(
        context
    ).strip().lower() in ["1", "true", "yes", "on"]
    photographer_startup_readiness_topic = LaunchConfiguration("photographer_startup_readiness_topic").perform(context)
    explorer_startup_readiness_topic = LaunchConfiguration("explorer_startup_readiness_topic").perform(context)

    # --- Controller Setup ---
    fleet = load_spawn_profiles(world)
    photographers = []
    for drone in fleet:
        if drone.get("role") == "photographer" and drone.get("instance") in [2, 4, 5]:
            photographers.append(
                {
                    "instance": drone["instance"],
                    "x": float(drone.get("x", 0.0)),
                    "y": float(drone.get("y", 0.0)),
                    "yaw": float(drone.get("yaw_deg", 0.0)),
                }
            )

    if not photographers:
        photographers = DEFAULT_PHOTOGRAPHERS.copy()
    photographers.sort(key=lambda p: p["instance"])
    photographer_names = [f"x500_gimbal_photographer_{p['instance']}" for p in photographers]

    nodes = []

    for index, p in enumerate(photographers):
        instance = p["instance"]
        controller_node = Node(
            package="caric2_baseline_2",
            executable="vehicle_controller_openscvx",
            name=f"photographer_controller_openscvx_px4_{instance}",
            namespace=f"px4_{instance}",
            output="screen",
            parameters=[
                {"drone_instance": instance},
                {"world": world},
                {"world_model_source": world_model_source},
                {"spawn_x": p["x"]},
                {"spawn_y": p["y"]},
                {"spawn_yaw": p["yaw"]},
                {"map_min_known_ratio": float(map_min_known_ratio)},
                {"map_min_stable_seconds": float(map_min_stable_seconds)},
                {"require_startup_readiness": require_photographer_startup_readiness},
                {"startup_readiness_topic": photographer_startup_readiness_topic},
            ],
        )
        nodes.append(controller_node)

    nodes.extend(
        [
            Node(
                package="caric2_baseline_1",
                executable="photographer_coordinator",
                name="photographer_coordinator",
                output="screen",
                parameters=[
                    {"num_photographers": len(photographers)},
                    {"assignment_timeout_seconds": 300.0},
                    {"world_model_source": world_model_source},
                    {"require_photographer_ready": True},
                    {"wait_for_all_photographers_ready_before_first_assignment": True},
                    {"photographer_names": photographer_names},
                    {"global_required_known_ratio": float(map_min_known_ratio)},
                    {"cluster_required_known_ratio": float(cluster_required_known_ratio)},
                    {"max_concurrent_trajectory_generations": 1},
                ],
            ),
            Node(
                package="caric2_baseline_1",
                executable="vehicle_controller_explorer",
                name="vehicle_controller_explorer",
                output="screen",
                parameters=[
                    {"world": world},
                    {"difficulty": difficulty},
                    {"require_startup_readiness": require_explorer_startup_readiness},
                    {"startup_readiness_topic": explorer_startup_readiness_topic},
                ],
            ),
            Node(
                package="caric2_baseline_1",
                executable="gcs",
                name="gcs",
                output="screen",
            ),
            Node(
                package="caric2_baseline_1",
                executable="poi_detection_node",
                name="poi_detection_node",
                output="screen",
                parameters=[{"world": world}, {"difficulty": "auto"}],
            ),
            Node(
                package="caric2_baseline_1",
                executable="cluster_manager",
                name="cluster_manager",
                output="screen",
                parameters=[
                    {"assignment_timeout": 300.0},
                    {"world_model_source": world_model_source},
                ],
            ),
        ]
    )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value="mbs", description="World key"),
            DeclareLaunchArgument("difficulty", default_value="auto", description="Difficulty or auto"),
            DeclareLaunchArgument("world_model_source", default_value="known", description="known or lidar_only"),
            DeclareLaunchArgument("require_photographer_startup_readiness", default_value="true"),
            DeclareLaunchArgument("require_explorer_startup_readiness", default_value="true"),
            DeclareLaunchArgument("photographer_startup_readiness_topic", default_value="/fleet_ready"),
            DeclareLaunchArgument("explorer_startup_readiness_topic", default_value="/fleet_ready"),
            DeclareLaunchArgument("cluster_required_known_ratio", default_value="0.4"),
            DeclareLaunchArgument("map_min_known_ratio", default_value="0.4"),
            DeclareLaunchArgument("map_min_stable_seconds", default_value="0.0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
