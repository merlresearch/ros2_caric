# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import time
from itertools import permutations
from typing import Any, List

import numpy as np
from std_msgs.msg import String


def _publish_rejection(ctrl: Any, assignment_id: str, cluster_id: str, reason: str) -> None:
    if not hasattr(ctrl, "assignment_rejection_publisher"):
        return
    msg = String()
    msg.data = json.dumps(
        {
            "assignment_id": assignment_id,
            "cluster_id": cluster_id,
            "photographer_id": ctrl.photographer_id,
            "reason": reason,
        }
    )
    ctrl.assignment_rejection_publisher.publish(msg)


def publish_assignment_rejected(ctrl: Any, reason: str) -> bool:
    assignment_id = getattr(ctrl, "current_assignment_id", None)
    cluster_id = getattr(ctrl, "current_cluster_id", None)
    if not assignment_id or cluster_id is None:
        return False

    _publish_rejection(ctrl, assignment_id, cluster_id, reason)
    ctrl.get_logger().warn(f"Rejected cluster {cluster_id} assignment: {reason}")
    ctrl.current_assignment_id = None
    ctrl.current_cluster_id = None
    return True


def enter_assignment_recovery(ctrl: Any, reason: str) -> None:
    pos = ctrl.vehicle_local_position
    ctrl.recovery_hold_local_position = [float(pos.x), float(pos.y), float(pos.z)]
    ctrl.recovery_start_time = ctrl.get_clock().now().nanoseconds / 1e9
    ctrl.recovery_stable_since = None
    ctrl.recovery_reason = reason
    ctrl._recovery_logged = False
    ctrl._last_recovery_log_time = 0.0
    ctrl.flight_state = "RECOVER_AFTER_FAILURE"


def handle_assignment_message(ctrl: Any, msg) -> None:
    data = json.loads(msg.data)
    if data["photographer_id"] != ctrl.photographer_id:
        return

    if not hasattr(ctrl, "_last_rejection_reason"):
        ctrl._last_rejection_reason = None
        ctrl._last_rejection_cluster = None
        ctrl._rejection_count = 0

    if getattr(ctrl, "world_model_source", "known") == "lidar_only":
        if not getattr(ctrl, "map_ready", False):
            assignment_id = data.get("assignment_id")
            cluster_id = data.get("cluster_id")
            if ctrl._last_rejection_reason != "map_not_ready" or ctrl._last_rejection_cluster != cluster_id:
                ctrl.get_logger().info(f"Rejecting assignments: map_not_ready (cluster={cluster_id})")
                ctrl._last_rejection_reason = "map_not_ready"
                ctrl._last_rejection_cluster = cluster_id
                ctrl._rejection_count = 1
            else:
                ctrl._rejection_count += 1
            _publish_rejection(ctrl, assignment_id, cluster_id, "map_not_ready")
            return

    assignment_id = data.get("assignment_id", None)
    cluster_id = data.get("cluster_id", None)
    assignment_type = data.get("assignment_type", "cluster")

    if not hasattr(ctrl, "processed_cluster_ids"):
        ctrl.processed_cluster_ids = set()

    ctrl._last_rejection_reason = None
    ctrl._last_rejection_cluster = None

    if cluster_id is not None and cluster_id in getattr(ctrl, "processed_cluster_ids", set()):
        return

    if assignment_type == "cluster" and cluster_id is not None:
        current_aid = getattr(ctrl, "current_assignment_id", None)
        current_cid = getattr(ctrl, "current_cluster_id", None)

        if current_cid is not None and current_cid != cluster_id:
            ctrl.get_logger().info(f"Busy with {current_cid}, rejecting {cluster_id}")
            _publish_rejection(ctrl, assignment_id, cluster_id, "busy_with_other_cluster")
            return

        if (
            current_cid == cluster_id
            and current_aid is not None
            and assignment_id is not None
            and current_aid != assignment_id
        ):
            return

        ctrl.current_assignment_id = assignment_id
        ctrl.current_cluster_id = cluster_id

    if ctrl.flight_state not in (
        "HOVER_AFTER_TAKEOFF",
        "INIT",
        "WAITING_FOR_ASSIGNMENT",
    ):
        ctrl.get_logger().info(f"Assignment rejected - busy (state={ctrl.flight_state})")
        _publish_rejection(ctrl, assignment_id, cluster_id, f"busy_state_{ctrl.flight_state}")
        ctrl.current_assignment_id = None
        ctrl.current_cluster_id = None
        return

    if hasattr(ctrl, "publish_timing_event"):
        ctrl.publish_timing_event("assignment_start")

    ctrl.assignment_received = True
    pois = data["cluster_data"]["pois"]

    tsp_start = time.time()
    ctrl.phase2_ordered_pois_list_enu = brute_force_tsp_for_poi_enu(pois, start_position_enu=current_position_enu(ctrl))
    tsp_solve_time = time.time() - tsp_start
    ctrl.last_tsp_solve_time = tsp_solve_time

    ctrl.phase2_started = False
    ctrl.phase2_completed = False
    ctrl.get_logger().info(
        f"TSP order for {len(ctrl.phase2_ordered_pois_list_enu)} POIs (solve: {tsp_solve_time:.4f}s)"
    )

    ctrl.flight_state = "GENERATE_ALL_PHASES"


def current_position_enu(ctrl: Any) -> np.ndarray:
    yaw_rad = np.radians(ctrl.spawn_yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    rel_n = float(ctrl.vehicle_local_position.x)
    rel_e = float(ctrl.vehicle_local_position.y)
    world_e = rel_n * s + rel_e * c
    world_n = rel_n * c - rel_e * s
    return np.array(
        [
            float(ctrl.spawn_x) + world_e,
            float(ctrl.spawn_y) + world_n,
            -float(ctrl.vehicle_local_position.z),
        ],
        dtype=float,
    )


def brute_force_tsp_for_poi_enu(pois, start_position_enu=None):
    # Capped at 5 POIs: OpenSCvx Phase 2 doesn't converge beyond that
    n_pois = len(pois)
    if n_pois == 0:
        raise ValueError("No POIs provided for TSP ordering.")
    elif n_pois > 5:
        raise ValueError(
            f"OpenSCvx Phase 2 requires 5 or fewer POIs, got {n_pois}. "
            "Cluster manager should have enforced this limit."
        )

    best_order_poi_enu: List[np.ndarray] = []
    best_cost = float("inf")

    for perm in permutations(range(n_pois)):
        current_cost = 0.0
        current_position = np.array([pois[perm[0]]["position"][v] for v in ("x", "y", "z")], dtype=float)
        if start_position_enu is not None:
            current_cost += np.linalg.norm(current_position - np.array(start_position_enu, dtype=float))
        current_perm_position_list = [current_position]
        for idx in perm[1:]:
            p = pois[idx]
            target_position = np.array([p["position"][v] for v in ("x", "y", "z")], dtype=float)
            current_cost += np.linalg.norm(target_position - current_position)
            current_position = target_position
            current_perm_position_list.append(current_position)

        if current_cost < best_cost:
            best_cost = current_cost
            best_order_poi_enu = current_perm_position_list

    return best_order_poi_enu


def publish_assignment_complete(ctrl: Any, coverage_success: bool = True) -> None:
    assignment_id = getattr(ctrl, "current_assignment_id", None)
    cluster_id = getattr(ctrl, "current_cluster_id", None)
    if not assignment_id or cluster_id is None:
        return

    msg = String()
    msg.data = json.dumps(
        {
            "assignment_id": assignment_id,
            "cluster_id": cluster_id,
            "photographer_id": getattr(ctrl, "photographer_id", ""),
            "coverage_success": bool(coverage_success),
            "pois_completed": len(ctrl.phase2_ordered_pois_list_enu),
        }
    )
    ctrl.assignment_complete_publisher.publish(msg)
    ctrl.get_logger().info(f"Cluster {cluster_id} assignment complete")

    processed = getattr(ctrl, "processed_cluster_ids", set())
    processed.add(cluster_id)
    ctrl.processed_cluster_ids = processed
    ctrl.current_assignment_id = None
    ctrl.current_cluster_id = None
