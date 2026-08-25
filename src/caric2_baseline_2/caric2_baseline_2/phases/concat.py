# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from .openscvx_utils import static_obstacle_avoidance_ctcs, is_inside_any_of_obstacles

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx


def concat_traj(parts: List[Dict]) -> Dict:
    """Concatenate multiple trajectory dicts into one continuous trajectory."""
    trajectory_keys = [
        "position",
        "velocity",
        "acceleration",
        "attitude",
        "angular_velocity",
        "yaw",
        "yawspeed",
        "time",
    ]
    out = {k: [] for k in trajectory_keys}
    out["poi_times"] = []
    out["pois_world"] = []
    out["part_end_times"] = []
    t_offset = 0.0
    for phase_index, traj in enumerate(parts):
        phase_start = t_offset
        if phase_index == 0:
            out["time"].extend(traj["time"])
        else:
            out["time"].extend([t + phase_start for t in traj["time"]])
        t_offset = phase_start + float(traj["time"][-1])
        out["part_end_times"].append(t_offset)
        for key in trajectory_keys:
            if key != "time":
                out[key].extend(traj[key])

        if "poi_times" in traj and "pois_world" in traj:
            out["pois_world"].extend(traj["pois_world"])
            out["poi_times"].extend([t_poi + phase_start for t_poi in traj["poi_times"]])
    out["success"] = True
    return out


def get_start_pos_vel_enu(vehicle_local_position, spawn_x, spawn_y, spawn_yaw_deg):
    """Get the starting position and velocity in ENU frame."""
    cur_pos_ned = np.array([vehicle_local_position.x, vehicle_local_position.y, vehicle_local_position.z], dtype=float)
    cur_vel_ned = np.array(
        [vehicle_local_position.vx, vehicle_local_position.vy, vehicle_local_position.vz], dtype=float
    )

    # NED -> world ENU conversion
    yaw_rad = math.radians(spawn_yaw_deg)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)

    v_n, v_e, v_d = cur_vel_ned[0], cur_vel_ned[1], cur_vel_ned[2]
    v_e_enu = v_n * s + v_e * c
    v_n_enu = v_n * c - v_e * s
    v_u_enu = -v_d
    start_vel_enu = np.array([v_e_enu, v_n_enu, v_u_enu], dtype=float)

    rel_n, rel_e = cur_pos_ned[0], cur_pos_ned[1]
    world_e = rel_n * s + rel_e * c
    world_n = rel_n * c - rel_e * s
    start_pos_enu = np.array([spawn_x + world_e, spawn_y + world_n, -cur_pos_ned[2]], dtype=float)
    return start_pos_enu, start_vel_enu


def generate_all_phases_upfront(controller: VehicleControllerOpenSCvx) -> bool:
    """Solve Phases 1, 2, 3 upfront and set the combined optimal_trajectory."""
    start_pos_enu, start_vel_enu = get_start_pos_vel_enu(
        controller.vehicle_local_position,
        controller.spawn_x,
        controller.spawn_y,
        controller.spawn_yaw_deg,
    )

    sanity_check_before_generating_all_phase_solution(controller, start_pos_enu)

    controller.get_logger().info("Attempting Phase 1")
    phase1 = controller.phase1.generate_enu(
        start_pos_enu,
        start_vel_enu,
        start_attitude=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        start_angular_velocity=np.zeros((3,), dtype=float),
        end_pos_enu=controller.phase2_ordered_pois_list_enu[0],
        buffered_end_pos_enu=controller.buffered_poi_list_enu[0],
    )

    remaining_buffered_pois_enu = controller.buffered_poi_list_enu[1:]
    remaining_original_pois_enu = controller.phase2_ordered_pois_list_enu[1:]
    if len(remaining_buffered_pois_enu) > 0:
        controller.get_logger().info("Attempting Phase 2")
        phase2 = controller.phase2.generate_enu(
            start_pos_enu=phase1["position"][-1],
            start_vel_enu=phase1["velocity"][-1],
            start_attitude=phase1["attitude"][-1],
            start_angular_velocity=phase1["angular_velocity"][-1],
            list_buffered_pois_enu=remaining_buffered_pois_enu,
            original_pois_enu=remaining_original_pois_enu,
        )
        phase3_start_traj = phase2
    else:
        phase2 = None
        phase3_start_traj = phase1

    phase3_start_pos = phase3_start_traj["position"][-1]
    phase3_start_vel = phase3_start_traj["velocity"][-1]
    phase3_start_attitude = phase3_start_traj["attitude"][-1]
    phase3_start_angular_velocity = phase3_start_traj["angular_velocity"][-1]

    controller.get_logger().info("Attempting Phase 3")
    phase3 = controller.phase3.generate_enu(
        start_pos_enu=phase3_start_pos,
        start_vel_enu=phase3_start_vel,
        start_attitude=phase3_start_attitude,
        start_angular_velocity=phase3_start_angular_velocity,
    )

    controller.end_trajectory_index["phase1"] = len(phase1["position"]) - 1
    if phase2 is not None:
        controller.optimal_trajectory = concat_traj([phase1, phase2, phase3])
        controller.end_trajectory_index["phase2"] = controller.end_trajectory_index["phase1"] + len(phase2["position"])
        controller.openscvx_planned_enu = controller.optimal_trajectory["position"]
        controller.end_trajectory_index["phase3"] = len(controller.openscvx_planned_enu) - 1
        return True
    else:
        controller.optimal_trajectory = concat_traj([phase1, phase3])
        controller.end_trajectory_index["phase2"] = controller.end_trajectory_index["phase1"]
        controller.openscvx_planned_enu = controller.optimal_trajectory["position"]
        controller.end_trajectory_index["phase3"] = len(controller.openscvx_planned_enu) - 1
        return True


def sanity_check_before_generating_all_phase_solution(
    controller: VehicleControllerOpenSCvx, start_pos_enu: np.ndarray
) -> None:
    if start_pos_enu[2] < controller.common_problem_config["ground_clearance"]:
        raise RuntimeError(
            "Start position does not meet ground clearance requirement! "
            f"{start_pos_enu[2]} is below the configured ground clearance "
            f"{controller.common_problem_config['ground_clearance']}."
        )

    comm_center_lhs = static_obstacle_avoidance_ctcs(
        centered_position=controller.common_problem_config["comm_center"] - start_pos_enu,
        boxes_for_obstacles=controller.obstacle_boxes,
        start_pos_enu=start_pos_enu,
        drone_radius=controller.common_problem_config["drone_radius"],
        lse_alpha=controller.common_problem_config["lse_alpha"],
        evaluate=True,
    )
    if is_inside_any_of_obstacles(comm_center_lhs):
        raise RuntimeError("Computed communication center is inside lse-inflated obstacle!")

    for buffered_poi in controller.buffered_poi_list_enu:
        buffered_poi_lhs = static_obstacle_avoidance_ctcs(
            centered_position=buffered_poi,
            boxes_for_obstacles=controller.obstacle_boxes,
            start_pos_enu=np.zeros((3,)),
            drone_radius=controller.common_problem_config["drone_radius"],
            lse_alpha=controller.common_problem_config["lse_alpha"],
            evaluate=True,
        )
        if np.any(is_inside_any_of_obstacles(buffered_poi_lhs)):
            raise RuntimeError("One of the buffered POIs is inside the lse-inflated obstacle!")
