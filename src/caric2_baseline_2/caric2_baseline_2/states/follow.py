# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..assignment_core import (
    enter_assignment_recovery,
    publish_assignment_complete,
    publish_assignment_rejected,
)
from ..phases.common import local_to_world as common_local_to_world

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx

FOLLOW_TIME_SCALE = 0.2
POI_YAW_CAPTURE_RADIUS_M = 15.0
POI_YAW_ADVANCE_HYSTERESIS_M = 2.0


def _local_yaw_to_face(
    ctrl: VehicleControllerOpenSCvx,
    current_pos_enu: np.ndarray,
    target_pos_enu: np.ndarray,
) -> float:
    rel_e = float(target_pos_enu[0] - current_pos_enu[0])
    rel_n = float(target_pos_enu[1] - current_pos_enu[1])
    spawn_yaw_rad = math.radians(-float(getattr(ctrl, "spawn_yaw_deg", 0.0)))
    c, s = math.cos(spawn_yaw_rad), math.sin(spawn_yaw_rad)
    local_x = rel_n * c - rel_e * s
    local_y = rel_e * c + rel_n * s
    return math.atan2(local_y, local_x)


def _route_target_poi(ctrl: VehicleControllerOpenSCvx, current_pos_enu: np.ndarray):
    pois = getattr(ctrl, "phase2_ordered_pois_list_enu", None)
    if not pois:
        return None

    assignment_id = getattr(ctrl, "current_assignment_id", None)
    if getattr(ctrl, "_yaw_assignment_id", None) != assignment_id:
        ctrl._yaw_assignment_id = assignment_id
        ctrl._yaw_target_index = 0
        ctrl._yaw_target_min_dist = float("inf")

    ordered_idx = int(getattr(ctrl, "_yaw_target_index", 0))
    ordered_idx = max(0, min(ordered_idx, len(pois) - 1))

    while ordered_idx < len(pois) - 1:
        target = np.array(pois[ordered_idx], dtype=float)
        current_dist = float(np.linalg.norm(target - current_pos_enu))
        min_dist = min(float(getattr(ctrl, "_yaw_target_min_dist", float("inf"))), current_dist)
        ctrl._yaw_target_min_dist = min_dist

        has_reached_scoring_window = min_dist <= POI_YAW_CAPTURE_RADIUS_M
        has_passed_closest_approach = current_dist > min_dist + POI_YAW_ADVANCE_HYSTERESIS_M
        if not (has_reached_scoring_window and has_passed_closest_approach):
            break

        ordered_idx += 1
        ctrl._yaw_target_min_dist = float("inf")

    ctrl._yaw_target_index = ordered_idx
    ordered_target = np.array(pois[ordered_idx], dtype=float)
    if ordered_idx == len(pois) - 1:
        current_dist = float(np.linalg.norm(ordered_target - current_pos_enu))
        ctrl._yaw_target_min_dist = min(float(getattr(ctrl, "_yaw_target_min_dist", float("inf"))), current_dist)

    distances = [float(np.linalg.norm(np.array(poi, dtype=float) - current_pos_enu)) for poi in pois]
    nearest_idx = int(np.argmin(distances))
    if distances[nearest_idx] <= POI_YAW_CAPTURE_RADIUS_M:
        ctrl._yaw_current_target_index = nearest_idx
        return np.array(pois[nearest_idx], dtype=float)

    ctrl._yaw_current_target_index = ordered_idx
    return ordered_target


def handle_follow_trajectory(ctrl: VehicleControllerOpenSCvx) -> None:
    """Handle FOLLOW_TRAJECTORY state via direct time-based streaming."""
    if not ctrl.optimal_trajectory:
        ctrl.get_logger().error("No optimal trajectory - proceeding to LAND")
        ctrl.flight_state = "LAND"
        ctrl.landing_start_time = ctrl.get_clock().now().nanoseconds / 1e9
        ctrl.publish_timing_event("landing_start")
        return

    if not hasattr(ctrl, "follow_started_logged") or not ctrl.follow_started_logged:
        total_T = float(ctrl.optimal_trajectory["time"][-1])
        num_pts = len(ctrl.optimal_trajectory["position"])
        ctrl.get_logger().info(f"FOLLOW_TRAJECTORY start: {num_pts} points, T_end={total_T:.2f}s")
        ctrl.follow_started_logged = True

    trajectory_time = np.array(ctrl.optimal_trajectory["time"])
    trajectory_pos = np.array(ctrl.optimal_trajectory["position"])
    trajectory_vel = np.array(ctrl.optimal_trajectory["velocity"])
    trajectory_acc = np.array(ctrl.optimal_trajectory["acceleration"])
    trajectory_yaw = np.array(ctrl.optimal_trajectory["yaw"])
    trajectory_yawspeed = np.array(ctrl.optimal_trajectory["yawspeed"])
    n_points = len(trajectory_time)
    total_time = float(trajectory_time[-1])

    current_time = ctrl.get_clock().now().nanoseconds / 1e9
    elapsed_time = current_time - ctrl.trajectory_start_time
    current_pos_enu = np.array(
        common_local_to_world(
            float(ctrl.spawn_x),
            float(ctrl.spawn_y),
            float(ctrl.spawn_yaw_deg),
            float(ctrl.vehicle_local_position.x),
            float(ctrl.vehicle_local_position.y),
            float(ctrl.vehicle_local_position.z),
        )
    )

    # Heartbeat prevents assignment timeout during long trajectories
    heartbeat_interval = 15.0
    if not hasattr(ctrl, "last_heartbeat_time"):
        ctrl.last_heartbeat_time = current_time
    if current_time - ctrl.last_heartbeat_time >= heartbeat_interval:
        progress_pct = min(100.0, (elapsed_time / total_time) * 100.0) if total_time > 0 else 0.0
        ctrl.publish_timing_event(
            "trajectory_progress",
            {
                "elapsed_time": elapsed_time,
                "total_time": total_time,
                "progress_percent": progress_pct,
            },
        )
        ctrl.last_heartbeat_time = current_time

    if not hasattr(ctrl, "_cluster_complete_published"):
        ctrl._cluster_complete_published = False

    time_scale = float(getattr(ctrl, "time_scale", FOLLOW_TIME_SCALE))
    target_poi = _route_target_poi(ctrl, current_pos_enu)
    effective_elapsed = max(0.0, elapsed_time * time_scale)

    idx = np.searchsorted(trajectory_time, effective_elapsed, side="right")
    trajectory_index = min(idx, n_points - 1)

    if not ctrl._cluster_complete_published and trajectory_index >= ctrl.end_trajectory_index["phase2"]:
        ctrl.cluster_complete_wall_time = current_time
        ctrl.publish_timing_event("cluster_complete")
        ctrl._cluster_complete_published = True

    if effective_elapsed >= total_time:
        trajectory_index = n_points - 1
        pos_set = trajectory_pos[trajectory_index]
        vel_set = np.zeros(3)
        acc_set = np.zeros(3)
        yaw_set = float(trajectory_yaw[trajectory_index])
        yawspeed_set = 0.0

        if getattr(ctrl, "_trajectory_final_hold_start_time", None) is None:
            ctrl._trajectory_final_hold_start_time = current_time
            ctrl._trajectory_final_hold_logged = False

        hold_elapsed = current_time - ctrl._trajectory_final_hold_start_time
        final_error = float(np.linalg.norm(current_pos_enu - pos_set))
        completion_tolerance = float(getattr(ctrl, "trajectory_completion_tolerance", 3.0))
        min_hold = float(getattr(ctrl, "trajectory_completion_min_hold_seconds", 2.0))
        timeout = float(getattr(ctrl, "trajectory_completion_timeout_seconds", 45.0))
        close_enough = final_error <= completion_tolerance and hold_elapsed >= min_hold
        timed_out = hold_elapsed >= timeout

        if not close_enough and not timed_out and not getattr(ctrl, "_trajectory_final_hold_logged", False):
            ctrl.get_logger().info(
                "Trajectory time complete - holding final setpoint until vehicle catches up "
                f"(error={final_error:.2f}m, tolerance={completion_tolerance:.2f}m)"
            )
            ctrl._trajectory_final_hold_logged = True

        if close_enough:
            ctrl.save_sent_trajectory_artifacts(reason_suffix="traj_complete")
            ctrl.publish_timing_event("home_return")
            publish_assignment_complete(ctrl, coverage_success=True)
            ctrl.get_logger().info("Trajectory complete - waiting for next assignment")
            _reset_for_next_assignment(ctrl)
            return

        if timed_out:
            ctrl.get_logger().warn(
                "Trajectory completion hold timed out; rejecting assignment "
                f"(final error={final_error:.2f}m, tolerance={completion_tolerance:.2f}m)"
            )
            ctrl.save_sent_trajectory_artifacts(reason_suffix="traj_incomplete")
            publish_assignment_rejected(ctrl, "trajectory_completion_timeout")
            _reset_for_next_assignment(ctrl, publish_ready=False)
            enter_assignment_recovery(ctrl, "trajectory_completion_timeout")
            return
    else:
        ctrl._trajectory_final_hold_start_time = None
        ctrl._trajectory_final_hold_logged = False
        pos_set = trajectory_pos[trajectory_index]
        # vel scales by time_scale, acc by time_scale^2 for slowed playback
        vel_set = trajectory_vel[trajectory_index] * time_scale
        acc_set = trajectory_acc[trajectory_index] * (time_scale**2)
        yaw_set = float(trajectory_yaw[trajectory_index])
        yawspeed_set = float(trajectory_yawspeed[trajectory_index]) * time_scale

    # Yaw override: face the current route POI, else face velocity heading.
    yaw_override = None

    if target_poi is not None:
        yaw_override = _local_yaw_to_face(ctrl, current_pos_enu, target_poi)

    if yaw_override is None:
        speed_sq = vel_set[0] ** 2 + vel_set[1] ** 2
        if speed_sq > 0.25:
            velocity_target = current_pos_enu + np.array([float(vel_set[0]), float(vel_set[1]), 0.0])
            yaw_override = _local_yaw_to_face(ctrl, current_pos_enu, velocity_target)

    if yaw_override is not None:
        yaw_set = float(yaw_override)
        yawspeed_set = 0.0

    if ctrl.publish_setpoint_enu(pos_set, vel_set, acc_set, yaw_set, yawspeed_set):
        ctrl.actual_positions_enu.append(list(current_pos_enu))
    else:
        ctrl.get_logger().error("Failed to publish trajectory setpoint - proceeding to LAND")
        ctrl.flight_state = "LAND"
        ctrl.landing_start_time = ctrl.get_clock().now().nanoseconds / 1e9
        ctrl.publish_timing_event("landing_start")


def _reset_for_next_assignment(ctrl: VehicleControllerOpenSCvx, publish_ready: bool = True) -> None:
    ctrl.optimal_trajectory = None

    ctrl.phase2_started = False
    ctrl.phase2_completed = True
    ctrl.assignment_received = False
    ctrl.follow_started_logged = False
    ctrl._cluster_complete_published = False
    ctrl._yaw_assignment_id = None
    ctrl._yaw_target_index = 0
    ctrl._yaw_current_target_index = 0
    ctrl._yaw_target_min_dist = float("inf")
    ctrl._trajectory_final_hold_start_time = None
    ctrl._trajectory_final_hold_logged = False

    if publish_ready:
        ctrl.flight_state = "WAITING_FOR_ASSIGNMENT"
        ctrl.publish_ready_for_assignment()
