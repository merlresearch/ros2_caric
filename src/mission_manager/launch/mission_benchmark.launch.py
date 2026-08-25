# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    world = LaunchConfiguration("world", default="mbs")
    difficulty = LaunchConfiguration("difficulty", default="easy")
    run_style = LaunchConfiguration("run_style", default="full")
    time_limit = LaunchConfiguration("time_limit", default="600")
    world_model_source = LaunchConfiguration("world_model_source", default="known")
    planner_type = LaunchConfiguration("planner_type", default="openscvx")  # 'openscvx' or 'astar'
    save_map_snapshot = LaunchConfiguration("save_map_snapshot", default="false")
    snapshot_dir = LaunchConfiguration("snapshot_dir", default=os.path.expanduser("~/ros2_ws/src/mission_manager/maps"))
    fleet_readiness_require_heading = LaunchConfiguration("fleet_readiness_require_heading_good_for_control")

    def launch_setup(context, *args, **kwargs):
        world_key = world.perform(context)
        diff_key = difficulty.perform(context)
        run_style_val = run_style.perform(context)
        time_limit_val = int(time_limit.perform(context))
        world_model_source_val = world_model_source.perform(context)
        planner_type_val = planner_type.perform(context)

        spawner = Node(
            package="mission_manager",
            executable="spawner",
            name="spawner",
            parameters=[{"world_name": world_key}, {"difficulty": diff_key}],
            output="screen",
        )

        fleet_readiness = Node(
            package="mission_manager",
            executable="fleet_readiness_node",
            name="fleet_readiness_node",
            parameters=[
                {"world_name": world_key},
                {
                    "require_heading_good_for_control": fleet_readiness_require_heading.perform(context).strip().lower()
                    in ["1", "true", "yes", "on"]
                },
            ],
            output="screen",
        )

        referee = Node(
            package="mission_manager",
            executable="referee_node",
            name="referee_node",
            parameters=[
                {"los_distance_threshold": 100.0},
                {"world_name": world_key},
                {"difficulty": diff_key},
            ],
            output="screen",
        )

        scoring = Node(
            package="mission_manager",
            executable="poi_scoring_node",
            name="poi_scoring_node",
            parameters=[
                {"world_name": world_key},
                {"difficulty": diff_key},
                {"world_model_source": world_model_source_val},
                {"run_style": run_style_val},
                {"planner_type": planner_type_val},
                {"max_res_depth_m": 10.0},
            ],
            output="screen",
        )

        photosvc = Node(
            package="mission_manager",
            executable="photo_capture_service",
            name="photo_capture_service",
            parameters=[{"world_name": world_key}, {"difficulty": diff_key}],
            output="screen",
        )

        mission_ctl = Node(
            package="mission_manager",
            executable="mission_controller",
            name="mission_controller",
            parameters=[
                {"run_style": run_style_val},
                {"time_limit_sec": time_limit_val},
                {"world": world_key},
                {"difficulty": diff_key},
                {"world_model_source": world_model_source_val},
            ],
            output="screen",
            on_exit=Shutdown(reason="Mission complete"),
        )

        timing_logger = Node(
            package="mission_manager",
            executable="mission_timing_logger",
            name="mission_timing_logger",
            parameters=[
                {"world_name": world_key},
                {"difficulty": diff_key},
                {"run_style": run_style_val},
                {"time_limit_sec": time_limit_val},
                {"planner_type": planner_type_val},
            ],
            output="screen",
        )

        position_logger = Node(
            package="mission_manager",
            executable="position_logger",
            name="position_logger",
            parameters=[
                {"world_name": world_key},
                {"difficulty": diff_key},
                {"world_model_source": world_model_source_val},
                {"run_style": run_style_val},
                {"planner_type": planner_type_val},
            ],
            output="screen",
        )

        nodes = [
            spawner,
            fleet_readiness,
            photosvc,
            scoring,
            referee,
            mission_ctl,
            timing_logger,
            position_logger,
        ]

        if world_model_source_val == "lidar_only":
            save_snap_bool = save_map_snapshot.perform(context).strip().lower() in [
                "1",
                "true",
                "yes",
                "on",
            ]
            mapping = Node(
                package="mission_manager",
                executable="mapping_node",
                name="lidar_mapping_node",
                parameters=[
                    {"world_name": world_key},
                    {"difficulty": diff_key},
                    {"resolution_m": 0.3},
                    {"window_m": 120.0},
                    {"z_min": 0.5},
                    {"z_max": 80.0},
                    {"inflate_m": 0.2},
                    {"publish_rate_hz": 1.0},
                    {"min_known_ratio": 0.4},
                    {"stable_threshold": 0.02},
                    {"carve_free": True},
                    {"max_carve_hits": 10000},
                    {"stable_seconds": 0.0},
                    {"stop_when_stable": False},
                    {"self_filter_radius_m": 0.5},
                    {"save_snapshot": save_snap_bool},
                    {"snapshot_dir": snapshot_dir.perform(context)},
                ],
                output="screen",
            )
            nodes.append(mapping)

        return nodes

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value="mbs"),
            DeclareLaunchArgument("difficulty", default_value="easy"),
            DeclareLaunchArgument("run_style", default_value="full"),
            DeclareLaunchArgument("time_limit", default_value="600"),
            DeclareLaunchArgument("world_model_source", default_value="known"),
            DeclareLaunchArgument("planner_type", default_value="openscvx"),
            DeclareLaunchArgument("save_map_snapshot", default_value="false"),
            DeclareLaunchArgument("fleet_readiness_require_heading_good_for_control", default_value="false"),
            DeclareLaunchArgument(
                "snapshot_dir",
                default_value=os.path.expanduser("~/ros2_ws/src/mission_manager/maps"),
            ),
            SetEnvironmentVariable(name="MISSION_MAP_SOURCE", value=world_model_source),
            OpaqueFunction(function=launch_setup),
        ]
    )
