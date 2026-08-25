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


def generate_launch_description():
    world_arg = DeclareLaunchArgument("world", default_value="mbs", description="World key (for fleet, explorer paths)")
    require_photographer_ready_arg = DeclareLaunchArgument(
        "require_photographer_startup_readiness",
        default_value="true",
        description="Wait for the configured photographer readiness topic before photographer takeoff",
    )
    require_explorer_ready_arg = DeclareLaunchArgument(
        "require_explorer_startup_readiness",
        default_value="true",
        description="Wait for the configured explorer readiness topic before explorer takeoff",
    )
    photographer_ready_topic_arg = DeclareLaunchArgument(
        "photographer_startup_readiness_topic",
        default_value="/fleet_ready",
        description="Readiness Bool topic used by photographer controllers",
    )
    explorer_ready_topic_arg = DeclareLaunchArgument(
        "explorer_startup_readiness_topic",
        default_value="/fleet_ready",
        description="Readiness Bool topic used by explorer controllers",
    )

    def _load_fleet_from_yaml(profile_key: str):
        share_dir = get_package_share_directory("mission_manager")
        path = os.path.join(share_dir, "config", "spawn_profiles.yaml")
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get(profile_key) or {}).get("fleet") or []

    def _build_nodes(context):
        world = (LaunchConfiguration("world").perform(context) or "").strip() or "mbs"
        require_photographer_ready = LaunchConfiguration("require_photographer_startup_readiness").perform(
            context
        ).strip().lower() in ["1", "true", "yes", "on"]
        require_explorer_ready = LaunchConfiguration("require_explorer_startup_readiness").perform(
            context
        ).strip().lower() in ["1", "true", "yes", "on"]
        photographer_ready_topic = LaunchConfiguration("photographer_startup_readiness_topic").perform(context)
        explorer_ready_topic = LaunchConfiguration("explorer_startup_readiness_topic").perform(context)
        # Get mission settings from env (set by run.launch before baseline starts)
        world_model_source = os.environ.get("MISSION_WORLD_MODEL_SOURCE", "").strip() or "known"
        difficulty = os.environ.get("MISSION_DIFFICULTY", "").strip() or "easy"

        env_map_src = os.environ.get("MISSION_MAP_SOURCE", "").strip()
        if env_map_src and world_model_source == "known":
            world_model_source = env_map_src

        fleet = _load_fleet_from_yaml(world) or []

        def _is_photographer(d):
            role = str(d.get("role", "")).lower()
            model = str(d.get("model", "")).lower()
            return role == "photographer" or "photographer" in model

        photographer_fleet = [d for d in fleet if _is_photographer(d)]
        if photographer_fleet:
            photographer_names = [
                f"x500_gimbal_photographer_{int(d.get('instance', 0))}"
                for d in photographer_fleet
                if int(d.get("instance", 0)) > 0
            ]
        else:
            photographer_names = [f"x500_gimbal_photographer_{instance}" for instance in [2, 4, 5]]
        map_source = "live" if world_model_source == "lidar_only" else "file"

        nodes = [
            Node(
                package="caric2_baseline_1",
                executable="photographer_coordinator",
                name="photographer_coordinator",
                output="screen",
                parameters=[
                    {"num_photographers": len(photographer_fleet)},
                    {"assignment_timeout_seconds": 300.0},
                    {"world_model_source": world_model_source},
                    {"global_required_known_ratio": 0.35},
                    {"photographer_names": photographer_names},
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
                    {"require_startup_readiness": require_explorer_ready},
                    {"startup_readiness_topic": explorer_ready_topic},
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
                parameters=[{"world": world}],
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

        if not photographer_fleet:
            for instance in [2, 4, 5]:
                nodes.append(
                    Node(
                        package="caric2_baseline_1",
                        executable="vehicle_controller_astar",
                        name=f"photographer_controller_astar_px4_{instance}",
                        namespace=f"px4_{instance}",
                        output="screen",
                        parameters=[
                            {"drone_instance": instance},
                            {"world": world},
                            {"map_source": map_source},
                            {"poi_buffer_x": -8.0},
                            {"poi_buffer_y": 0.0},
                            {"require_startup_readiness": require_photographer_ready},
                            {"startup_readiness_topic": photographer_ready_topic},
                        ],
                    )
                )
        else:
            for d in photographer_fleet:
                instance = int(d.get("instance", 0))
                friendly = str(d.get("name", f"px4_{instance}"))
                nodes.append(
                    Node(
                        package="caric2_baseline_1",
                        executable="vehicle_controller_astar",
                        name=f"photographer_controller_astar_{friendly}",
                        namespace=f"px4_{instance}",
                        output="screen",
                        parameters=[
                            {"drone_instance": instance},
                            {"world": world},
                            {
                                "spawn_x": float(d.get("x", 0.0)),
                                "spawn_y": float(d.get("y", 0.0)),
                                "spawn_yaw": float(d.get("yaw_deg", 0.0)),
                            },
                            {"map_source": map_source},
                            {"poi_buffer_x": -8.0},
                            {"poi_buffer_y": 0.0},
                            {"require_startup_readiness": require_photographer_ready},
                            {"startup_readiness_topic": photographer_ready_topic},
                        ],
                    )
                )

        return nodes

    return LaunchDescription(
        [
            world_arg,
            require_photographer_ready_arg,
            require_explorer_ready_arg,
            photographer_ready_topic_arg,
            explorer_ready_topic_arg,
            OpaqueFunction(function=_build_nodes),
        ]
    )
