# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from px4_msgs.msg import VehicleStatus

from ..assignment_core import publish_assignment_complete

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx


def handle_land(ctrl: VehicleControllerOpenSCvx) -> None:
    current_time_sec = ctrl.get_clock().now().nanoseconds / 1e9
    if current_time_sec - ctrl.landing_start_time >= 10.0:
        ctrl.land()
        ctrl.flight_state = "LANDED"
    else:
        pos = ctrl.vehicle_local_position
        ctrl.publish_basic_trajectory_setpoint(pos.x, pos.y, pos.z)


def handle_landed(ctrl: VehicleControllerOpenSCvx) -> None:
    if ctrl.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
        if ctrl.vehicle_local_position.z > -0.3:
            ctrl.landing_complete_time = ctrl.get_clock().now().nanoseconds / 1e9
            ctrl.publish_timing_event("landing_complete")
            ctrl.disarm()
            ctrl.flight_state = "DISARMED"


def handle_disarmed(ctrl: VehicleControllerOpenSCvx) -> None:
    trajectory_type = "optimal_openscvx" if ctrl.optimal_trajectory is not None else "basic_waypoint"
    publish_assignment_complete(ctrl, coverage_success=True)
    ctrl.get_logger().info(f"Mission complete! Used {trajectory_type} trajectory.")
    ctrl.save_sent_trajectory_artifacts(reason_suffix="mission_end")
    exit()


def _vehicle_speed(ctrl: VehicleControllerOpenSCvx) -> float:
    pos = ctrl.vehicle_local_position
    velocity = [
        float(getattr(pos, "vx", 0.0)),
        float(getattr(pos, "vy", 0.0)),
        float(getattr(pos, "vz", 0.0)),
    ]
    if not all(math.isfinite(v) for v in velocity):
        return float("inf")
    return math.sqrt(sum(v * v for v in velocity))


def handle_recover_after_failure(ctrl: VehicleControllerOpenSCvx) -> None:
    hold = getattr(ctrl, "recovery_hold_local_position", None)
    if hold is None:
        pos = ctrl.vehicle_local_position
        hold = [float(pos.x), float(pos.y), float(pos.z)]
        ctrl.recovery_hold_local_position = hold

    ctrl.publish_basic_trajectory_setpoint(hold[0], hold[1], hold[2])

    now_s = ctrl.get_clock().now().nanoseconds / 1e9
    speed = _vehicle_speed(ctrl)
    speed_threshold = float(getattr(ctrl, "assignment_recovery_speed_threshold", 0.75))
    stable_seconds = float(getattr(ctrl, "assignment_recovery_stable_seconds", 2.0))
    max_wait_seconds = float(getattr(ctrl, "assignment_recovery_max_wait_seconds", 20.0))

    if not getattr(ctrl, "_recovery_logged", False):
        reason = getattr(ctrl, "recovery_reason", "assignment_failure")
        ctrl.get_logger().info(f"Recovering after {reason}; waiting for speed <= {speed_threshold:.2f} m/s")
        ctrl._recovery_logged = True

    if speed <= speed_threshold:
        if getattr(ctrl, "recovery_stable_since", None) is None:
            ctrl.recovery_stable_since = now_s
        stable_elapsed = now_s - ctrl.recovery_stable_since
    else:
        ctrl.recovery_stable_since = None
        stable_elapsed = 0.0

    start_time = float(getattr(ctrl, "recovery_start_time", now_s))
    if now_s - float(getattr(ctrl, "_last_recovery_log_time", 0.0)) >= 2.0 and stable_elapsed < stable_seconds:
        ctrl.get_logger().info(
            f"Recovery hold: speed={speed:.2f} m/s, stable={stable_elapsed:.1f}/{stable_seconds:.1f}s"
        )
        ctrl._last_recovery_log_time = now_s

    if stable_elapsed >= stable_seconds or now_s - start_time >= max_wait_seconds:
        if stable_elapsed < stable_seconds:
            ctrl.get_logger().warn(f"Recovery timed out after {now_s - start_time:.1f}s with speed={speed:.2f} m/s")

        pos = ctrl.vehicle_local_position
        ctrl.waiting_hold_local_position = [float(pos.x), float(pos.y), float(pos.z)]
        ctrl.recovery_hold_local_position = None
        ctrl.recovery_stable_since = None
        ctrl.flight_state = "WAITING_FOR_ASSIGNMENT"
        ctrl.publish_ready_for_assignment()


def handle_waiting_for_assignment(ctrl: VehicleControllerOpenSCvx) -> None:
    hold = getattr(ctrl, "waiting_hold_local_position", None)
    if hold is None:
        hold = [0.0, 0.0, ctrl.takeoff_height]
    ctrl.publish_basic_trajectory_setpoint(hold[0], hold[1], hold[2])
    if ctrl.assignment_received:
        ctrl.get_logger().info("Assignment received - generating all phases")
        ctrl.flight_state = "GENERATE_ALL_PHASES"
