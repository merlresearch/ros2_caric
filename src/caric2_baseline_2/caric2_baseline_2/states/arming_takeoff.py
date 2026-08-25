# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

from px4_msgs.msg import VehicleStatus

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx

TAKEOFF_RETRY_LOG_SECONDS = 5.0


def handle_arming(ctrl: VehicleControllerOpenSCvx) -> None:
    if ctrl.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
        ctrl.publish_basic_trajectory_setpoint(0.0, 0.0, ctrl.takeoff_height)
        if ctrl.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if ctrl.offboard_setpoint_counter % 50 == 0:
                ctrl.get_logger().info("Vehicle armed - waiting for OFFBOARD before takeoff")
            ctrl.engage_offboard_mode()
            return

        ctrl.get_logger().info("Vehicle armed")
        ctrl.flight_state = "TAKEOFF"
        ctrl.takeoff_start_time = ctrl.get_clock().now().nanoseconds / 1e9
        ctrl.publish_timing_event("takeoff_start")
    else:
        ctrl.publish_basic_trajectory_setpoint(0.0, 0.0, ctrl.takeoff_height)
        now_s = ctrl.get_clock().now().nanoseconds / 1e9

        # Ensure we have received at least one valid position message from PX4 before arming
        if ctrl.vehicle_local_position.timestamp == 0:
            return

        if (now_s - ctrl.last_arm_cmd_time) > 2.0 and ctrl.arm_attempts < 50:
            ctrl.arm()
            ctrl.last_arm_cmd_time = now_s
            ctrl.arm_attempts += 1
            if ctrl.arm_attempts % 5 == 0:
                ctrl.get_logger().info(f"Arming attempt {ctrl.arm_attempts}/50...")
        if (
            ctrl.offboard_setpoint_counter % 50 == 0
            and ctrl.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
        ):
            ctrl.engage_offboard_mode()


def handle_takeoff(ctrl: VehicleControllerOpenSCvx) -> None:
    ctrl.publish_basic_trajectory_setpoint(0.0, 0.0, ctrl.takeoff_height)

    now_s = ctrl.get_clock().now().nanoseconds / 1e9

    if ctrl.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        if now_s - ctrl.last_arm_cmd_time > 1.0:
            ctrl.get_logger().warn("Takeoff watchdog: re-engaging OFFBOARD mode")
            ctrl.engage_offboard_mode()
            ctrl.last_arm_cmd_time = now_s
        return

    if ctrl.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
        if ctrl.vehicle_local_position.timestamp == 0:
            return
        if now_s - ctrl.last_arm_cmd_time > 2.0:
            ctrl.get_logger().warn("Takeoff watchdog: vehicle disarmed, resending ARM command")
            ctrl.arm()
            ctrl.last_arm_cmd_time = now_s
            ctrl.arm_attempts += 1
        return

    if (
        ctrl.takeoff_start_time is not None
        and now_s - ctrl.takeoff_start_time > TAKEOFF_RETRY_LOG_SECONDS
        and now_s - ctrl._last_takeoff_watchdog_log_time > TAKEOFF_RETRY_LOG_SECONDS
    ):
        ctrl.get_logger().warn(
            f"Takeoff watchdog: still climbing, local_z={ctrl.vehicle_local_position.z:.2f}, "
            f"target_z={ctrl.takeoff_height:.2f}"
        )
        ctrl._last_takeoff_watchdog_log_time = now_s

    if ctrl.vehicle_local_position.z < ctrl.takeoff_height + 0.5:
        ctrl.takeoff_complete_time = ctrl.get_clock().now().nanoseconds / 1e9
        ctrl.publish_timing_event("takeoff_complete")
        ctrl.get_logger().info("Takeoff complete - hovering")
        ctrl.flight_state = "HOVER_AFTER_TAKEOFF"
        ctrl.hover_start_time = ctrl.get_clock().now().nanoseconds / 1e9
        ctrl.offboard_setpoint_counter = 0


def handle_hover_after_takeoff(ctrl: VehicleControllerOpenSCvx) -> None:
    ctrl.publish_basic_trajectory_setpoint(0.0, 0.0, ctrl.takeoff_height)
    current_time = ctrl.get_clock().now().nanoseconds / 1e9
    hover_duration = current_time - ctrl.hover_start_time
    if hover_duration > 0.5:
        ctrl.get_logger().info("Hover complete - waiting for assignment")
        ctrl.flight_state = "WAITING_FOR_ASSIGNMENT"
        ctrl.publish_ready_for_assignment()
