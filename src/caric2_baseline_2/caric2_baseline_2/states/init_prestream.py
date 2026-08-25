# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

from px4_msgs.msg import VehicleStatus

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx

OFFBOARD_PRESTREAM_TICKS = 100
ARM_PRESTREAM_TICKS = 200


def handle_init(ctrl: VehicleControllerOpenSCvx) -> None:
    if not ctrl.startup_readiness_satisfied():
        ctrl.log_waiting_for_startup_readiness()
        return

    ctrl.get_logger().info("Entering PRESTREAM for OFFBOARD engagement")
    ctrl.mission_start_time = ctrl.get_clock().now().nanoseconds / 1e9
    ctrl.publish_timing_event("mission_start")
    ctrl.flight_state = "PRESTREAM"


def handle_prestream(ctrl: VehicleControllerOpenSCvx) -> None:
    ctrl.publish_offboard_control_mode()
    ctrl.publish_basic_trajectory_setpoint(0.0, 0.0, ctrl.takeoff_height)
    ctrl.prestream_ticks += 1

    if ctrl.prestream_ticks >= OFFBOARD_PRESTREAM_TICKS and (
        not ctrl.prestream_offboard_engaged
        or (ctrl.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD and ctrl.prestream_ticks % 50 == 0)
    ):
        ctrl.get_logger().info("Engaging OFFBOARD mode (after prestream)")
        ctrl.engage_offboard_mode()
        ctrl.prestream_offboard_engaged = True

    now_s = ctrl.get_clock().now().nanoseconds / 1e9
    if (not ctrl.prestream_arm_sent) and ctrl.prestream_ticks >= ARM_PRESTREAM_TICKS:
        # Check if EKF is publishing valid local position
        if ctrl.vehicle_local_position.timestamp == 0:
            if ctrl.prestream_ticks % 50 == 0:
                ctrl.get_logger().info("Waiting for valid local position from PX4...")
            return

        if ctrl.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if ctrl.prestream_ticks % 100 == 0:
                ctrl.get_logger().info("Waiting for OFFBOARD mode before arming...")
            if ctrl.prestream_ticks % 50 == 0:
                ctrl.engage_offboard_mode()
            return

        if (now_s - ctrl.last_arm_cmd_time) > 2.0:
            ctrl.get_logger().info("Sending ARM command (after prestream)")
            ctrl.arm()
            ctrl.last_arm_cmd_time = now_s
            ctrl.arm_attempts += 1
            ctrl.prestream_arm_sent = True
        ctrl.flight_state = "ARMING"
