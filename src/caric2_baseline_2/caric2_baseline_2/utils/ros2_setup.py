# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

from ..assignment_core import handle_assignment_message

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx


def _load_mapping_cfg() -> dict:
    share_dir = get_package_share_directory("mission_manager")
    path = os.path.join(share_dir, "config", "mapping.yaml")
    if not os.path.exists(path):
        return {}
    content = open(path, "r").read().replace("\t", "  ")
    cfg = yaml.safe_load(content) or {}
    return cfg.get("defaults", {})


def declare_parameters(controller: VehicleControllerOpenSCvx) -> None:
    _map_cfg = _load_mapping_cfg()

    controller.declare_parameter("drone_instance", 2)
    controller.declare_parameter("spawn_x", 0.0)
    controller.declare_parameter("spawn_y", 0.0)
    controller.declare_parameter("spawn_yaw", 0.0)
    controller.declare_parameter("world", "mbs")
    controller.declare_parameter("world_model_source", "known")
    controller.declare_parameter("home_x", None)
    controller.declare_parameter("home_y", None)
    controller.declare_parameter("home_z", None)
    controller.declare_parameter("map_resolution", 0.3)
    controller.declare_parameter("map_z_min", _map_cfg.get("z_min", 0.5))
    controller.declare_parameter("map_z_max", _map_cfg.get("z_max", 80.0))
    controller.declare_parameter("map_min_known_ratio", _map_cfg.get("global_min_known_ratio", 0.75))
    controller.declare_parameter("map_min_stable_seconds", _map_cfg.get("stable_seconds", 3.0))
    controller.declare_parameter("trajectory_time_scale", 0.2)
    controller.declare_parameter("trajectory_completion_tolerance", 3.0)
    controller.declare_parameter("trajectory_completion_min_hold_seconds", 2.0)
    controller.declare_parameter("trajectory_completion_timeout_seconds", 45.0)
    controller.declare_parameter("require_startup_readiness", False)
    controller.declare_parameter("startup_readiness_topic", "/fleet_ready")


def read_parameters(controller: VehicleControllerOpenSCvx) -> None:
    controller.drone_instance = int(controller.get_parameter("drone_instance").value)
    controller.spawn_x = float(controller.get_parameter("spawn_x").value)
    controller.spawn_y = float(controller.get_parameter("spawn_y").value)
    controller.spawn_yaw_deg = float(controller.get_parameter("spawn_yaw").value)
    controller.world = controller.get_parameter("world").get_parameter_value().string_value or "mbs"
    controller.world_model_source = (
        controller.get_parameter("world_model_source").get_parameter_value().string_value or "known"
    )

    _hx = controller.get_parameter("home_x").value
    controller.home_x = float(_hx) if _hx is not None else None
    _hy = controller.get_parameter("home_y").value
    controller.home_y = float(_hy) if _hy is not None else None
    _hz = controller.get_parameter("home_z").value
    controller.home_z = float(_hz) if _hz is not None else None

    controller.map_resolution = float(controller.get_parameter("map_resolution").value)
    controller.map_z_min = float(controller.get_parameter("map_z_min").value)
    controller.map_z_max = float(controller.get_parameter("map_z_max").value)
    controller.map_min_known_ratio = float(controller.get_parameter("map_min_known_ratio").value)
    controller.map_min_stable_seconds = float(controller.get_parameter("map_min_stable_seconds").value)
    controller.time_scale = float(controller.get_parameter("trajectory_time_scale").value)
    controller.trajectory_completion_tolerance = float(
        controller.get_parameter("trajectory_completion_tolerance").value
    )
    controller.trajectory_completion_min_hold_seconds = float(
        controller.get_parameter("trajectory_completion_min_hold_seconds").value
    )
    controller.trajectory_completion_timeout_seconds = float(
        controller.get_parameter("trajectory_completion_timeout_seconds").value
    )
    controller.require_startup_readiness = bool(controller.get_parameter("require_startup_readiness").value)
    controller.startup_readiness_topic = (
        controller.get_parameter("startup_readiness_topic").get_parameter_value().string_value or "/fleet_ready"
    )
    controller.startup_ready = not controller.require_startup_readiness

    controller.topic_prefix = f"/px4_{controller.drone_instance}"


def create_qos_profile() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def setup_publishers(controller: VehicleControllerOpenSCvx, qos_profile: QoSProfile) -> None:
    controller.vehicle_command_publisher = controller.create_publisher(
        VehicleCommand, f"{controller.topic_prefix}/fmu/in/vehicle_command", qos_profile
    )
    controller.offboard_control_mode_publisher = controller.create_publisher(
        OffboardControlMode, f"{controller.topic_prefix}/fmu/in/offboard_control_mode", qos_profile
    )
    controller.trajectory_setpoint_publisher = controller.create_publisher(
        TrajectorySetpoint, f"{controller.topic_prefix}/fmu/in/trajectory_setpoint", qos_profile
    )
    controller.timing_publisher = controller.create_publisher(String, "/openscvx_mission_timing", qos_profile)

    reliable_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    controller.photographer_position_publisher = controller.create_publisher(
        String, "/photographer_positions", reliable_qos
    )
    controller.assignment_complete_publisher = controller.create_publisher(
        String, "/photographer_assignment_complete", reliable_qos
    )
    controller.photographer_timing_publisher = controller.create_publisher(
        String, "/photographer_timing_updates", reliable_qos
    )
    controller.ready_publisher = controller.create_publisher(String, "/photographer_ready", reliable_qos)
    controller.assignment_rejection_publisher = controller.create_publisher(
        String, "/photographer_assignment_rejected", reliable_qos
    )


def setup_subscribers(controller: VehicleControllerOpenSCvx, qos_profile: QoSProfile) -> None:
    controller.vehicle_local_position_subscriber = controller.create_subscription(
        VehicleLocalPosition,
        f"{controller.topic_prefix}/fmu/out/vehicle_local_position",
        controller.vehicle_local_position_callback,
        qos_profile,
    )
    controller.vehicle_status_subscriber = controller.create_subscription(
        VehicleStatus,
        f"{controller.topic_prefix}/fmu/out/vehicle_status_v1",
        controller.vehicle_status_callback,
        qos_profile,
    )

    latched_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    controller.assignment_subscriber = controller.create_subscription(
        String, "/photographer_assignments", lambda msg: handle_assignment_message(controller, msg), latched_qos
    )
    if controller.require_startup_readiness:
        controller.startup_readiness_subscriber = controller.create_subscription(
            Bool,
            controller.startup_readiness_topic,
            controller.startup_readiness_callback,
            latched_qos,
        )

    if controller.world_model_source == "lidar_only":
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        controller.map_occupancy_subscriber = controller.create_subscription(
            OccupancyGrid, "/mission/map/occupancy", controller.map_occupancy_callback, map_qos
        )
        controller.map_meta_subscriber = controller.create_subscription(
            String, "/mission/map/meta", controller.map_meta_callback, map_qos
        )
        controller.map_status_subscriber = controller.create_subscription(
            String, "/mission/map/status", controller.map_status_callback, map_qos
        )
