# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

from ..assignment_core import enter_assignment_recovery, publish_assignment_rejected
from ..optimizer_worker_client import solve_with_optimizer_worker

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx


def _reset_after_generation_failure(ctrl: VehicleControllerOpenSCvx, reason: str) -> None:
    ctrl.optimal_trajectory = None
    ctrl.openscvx_planned_enu = []
    ctrl.assignment_received = False
    ctrl.phase2_started = False
    ctrl.phase2_completed = False
    ctrl.follow_started_logged = False
    ctrl._cluster_complete_published = False
    ctrl._yaw_assignment_id = None
    ctrl._yaw_target_index = 0
    ctrl._yaw_current_target_index = 0
    ctrl._yaw_target_min_dist = float("inf")
    ctrl._trajectory_final_hold_start_time = None
    ctrl._trajectory_final_hold_logged = False
    enter_assignment_recovery(ctrl, reason)


def handle_generate_all_phases(ctrl: VehicleControllerOpenSCvx) -> None:
    ctrl.generation_start_time = ctrl.get_clock().now().nanoseconds / 1e9
    ctrl.publish_timing_event("trajectory_generation_start")

    ctrl.get_logger().info("Generating OpenSCvx trajectory in the Python 3.12 optimizer worker.")
    try:
        success = solve_with_optimizer_worker(ctrl)
    except Exception as exc:
        cluster_id = getattr(ctrl, "current_cluster_id", None)
        ctrl.get_logger().error(f"OpenSCvx generation failed for cluster {cluster_id}: {exc}")
        publish_assignment_rejected(ctrl, "openscvx_generation_failed")
        _reset_after_generation_failure(ctrl, "openscvx_generation_failed")
        return

    if success:
        phase_times = getattr(ctrl, "openscvx_phase_times", {})

        total_pre = 0.0
        total_main = 0.0
        total_post = 0.0
        total_computation = 0.0

        for phase_name, timing_dict in phase_times.items():
            if isinstance(timing_dict, dict):
                total_pre += timing_dict.get("preprocessing_time", 0.0)
                total_main += timing_dict.get("main_solve_time", 0.0)
                total_post += timing_dict.get("postprocessing_time", 0.0)
                total_computation += timing_dict.get("total_time", 0.0)

        ctrl.publish_timing_event(
            "trajectory_generation_complete",
            {
                "openscvx_solve_time": total_computation,
                "openscvx_preprocessing_time": total_pre,
                "openscvx_main_solve_time": total_main,
                "openscvx_postprocessing_time": total_post,
            },
        )

        ctrl.publish_timing_event(
            "cluster_start",
            {
                "openscvx_solve_time": total_computation,
                "openscvx_preprocessing_time": total_pre,
                "openscvx_main_solve_time": total_main,
                "openscvx_postprocessing_time": total_post,
                "tsp_solve_time": getattr(ctrl, "last_tsp_solve_time", None),
                "num_pois": len(getattr(ctrl, "phase2_ordered_pois_list_enu", [])),
                "sequencing_method": "TSP_bruteforce",
            },
        )

        ctrl.get_logger().info("All phases generated - following combined trajectory")
        ctrl.flight_state = "FOLLOW_TRAJECTORY"
        ctrl.trajectory_start_time = ctrl.get_clock().now().nanoseconds / 1e9
        ctrl.publish_timing_event("trajectory_start")
    else:
        ctrl.get_logger().error("OpenSCvx generation returned failure")
        publish_assignment_rejected(ctrl, "openscvx_generation_failed")
        _reset_after_generation_failure(ctrl, "openscvx_generation_failed")
