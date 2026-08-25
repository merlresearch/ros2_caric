# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Any

from px4_msgs.msg import OffboardControlMode, VehicleCommand  # type: ignore


def publish_vehicle_command(ctrl: Any, command: int, **params) -> None:
    msg = VehicleCommand()
    msg.command = int(command)
    msg.param1 = float(params.get("param1", 0.0))
    msg.param2 = float(params.get("param2", 0.0))
    msg.param3 = float(params.get("param3", 0.0))
    msg.param4 = float(params.get("param4", 0.0))
    msg.param5 = float(params.get("param5", 0.0))
    msg.param6 = float(params.get("param6", 0.0))
    msg.param7 = float(params.get("param7", 0.0))
    msg.target_system = int(ctrl.drone_instance) + 1
    msg.target_component = 1
    msg.source_system = 1
    msg.source_component = 1
    msg.from_external = True
    msg.timestamp = int(ctrl.get_clock().now().nanoseconds / 1000)
    ctrl.vehicle_command_publisher.publish(msg)


def arm(ctrl: Any) -> None:
    publish_vehicle_command(ctrl, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
    ctrl.get_logger().info("Arm command sent")


def disarm(ctrl: Any) -> None:
    publish_vehicle_command(ctrl, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
    ctrl.get_logger().info("Disarm command sent")


def engage_offboard_mode(ctrl: Any) -> None:
    publish_vehicle_command(ctrl, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
    ctrl.get_logger().info("Switching to offboard mode")


def land(ctrl: Any) -> None:
    publish_vehicle_command(ctrl, VehicleCommand.VEHICLE_CMD_NAV_LAND)
    ctrl.get_logger().info("Switching to land mode")


def publish_offboard_control_mode(ctrl: Any) -> None:
    msg = OffboardControlMode()
    msg.position = True
    msg.velocity = True
    msg.acceleration = True
    msg.attitude = False
    msg.body_rate = False
    msg.timestamp = int(ctrl.get_clock().now().nanoseconds / 1000)
    ctrl.offboard_control_mode_publisher.publish(msg)
