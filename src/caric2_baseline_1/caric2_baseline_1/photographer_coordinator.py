#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os
import uuid
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

SPLIT_RETRY_REASONS = {
    "openscvx_generation_failed",
    "trajectory_completion_timeout",
}


def _load_mapping_cfg() -> dict:
    share_dir = get_package_share_directory("mission_manager")
    path = os.path.join(share_dir, "config", "mapping.yaml")
    if not os.path.exists(path):
        return {}
    content = open(path, "r").read().replace("\t", "  ")
    cfg = yaml.safe_load(content) or {}
    return cfg.get("defaults", {})


class PhotographerCoordinator(Node):
    def __init__(self):
        super().__init__("photographer_coordinator")

        _map_cfg = _load_mapping_cfg()

        self.declare_parameter("num_photographers", 3)
        self.declare_parameter("assignment_timeout_seconds", 180.0)
        self.declare_parameter("enable_partial_timing_logs", True)
        self.declare_parameter("cluster_required_known_ratio", _map_cfg.get("cluster_min_known_ratio", 0.5))
        self.declare_parameter("cluster_coverage_radius_m", _map_cfg.get("cluster_coverage_radius_m", 20.0))
        self.declare_parameter("enable_lidar_gating", True)
        self.declare_parameter("global_required_known_ratio", _map_cfg.get("global_min_known_ratio", 0.50))
        self.declare_parameter("world_model_source", "known")
        self.declare_parameter("require_photographer_ready", False)
        self.declare_parameter("wait_for_all_photographers_ready_before_first_assignment", False)
        self.declare_parameter("assignment_reject_retry_delay_seconds", 30.0)
        self.declare_parameter("ready_assignment_stale_grace_seconds", 5.0)
        self.declare_parameter("max_concurrent_trajectory_generations", 999)

        num_photographers = self.get_parameter("num_photographers").get_parameter_value().integer_value
        self.num_photographers = int(num_photographers)
        self.assignment_timeout = self.get_parameter("assignment_timeout_seconds").get_parameter_value().double_value
        self.enable_partial_timing_logs = (
            self.get_parameter("enable_partial_timing_logs").get_parameter_value().bool_value
        )
        self.required_known_ratio = float(self.get_parameter("cluster_required_known_ratio").value)
        self.cluster_radius_m = float(self.get_parameter("cluster_coverage_radius_m").value)
        self.enable_lidar_gating = bool(self.get_parameter("enable_lidar_gating").value)
        self.global_required_known_ratio = float(self.get_parameter("global_required_known_ratio").value)
        wms_param = self.get_parameter("world_model_source").get_parameter_value().string_value
        self.world_model_source = wms_param if wms_param else "known"
        self.require_photographer_ready = bool(self.get_parameter("require_photographer_ready").value)
        wait_all_ready_param = "wait_for_all_photographers_ready_before_first_assignment"
        self.wait_for_all_ready_before_first_assignment = bool(self.get_parameter(wait_all_ready_param).value)
        self.assignment_reject_retry_delay = float(self.get_parameter("assignment_reject_retry_delay_seconds").value)
        self.ready_assignment_stale_grace = float(self.get_parameter("ready_assignment_stale_grace_seconds").value)
        self.max_concurrent_trajectory_generations = int(
            self.get_parameter("max_concurrent_trajectory_generations").value
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.photographer_positions = {}
        self.photographer_los_status = {}
        self.photographer_alive_status = {}
        self.available_clusters = {}
        self.active_assignments = {}
        self.assigned_clusters = set()
        self.completed_clusters = set()
        self.ready_photographers = set()
        self.rejected_photographers_by_cluster = {}
        self.rejected_clusters_retry_at = {}
        self.offboard_setpoint_counter = 0

        self.mission_timings = {}
        self.stale_assignment_ids = set()
        self._ready_assignment_pending_logs = set()
        self._last_first_assignment_wait_log_time = 0.0
        self.first_assignment_attempted = False

        self.assignment_publisher = self.create_publisher(String, "/photographer_assignments", reliable_qos)
        self.timing_data_publisher = self.create_publisher(String, "/mission_timing_data", reliable_qos)

        self.create_subscription(String, "/cluster_information", self.cluster_callback, volatile_qos)
        self.create_subscription(String, "/photographer_positions", self.position_callback, reliable_qos)
        self.create_subscription(String, "/photographer_los_status", self.los_status_callback, reliable_qos)
        self.create_subscription(
            String,
            "/photographer_assignment_complete",
            self.assignment_complete_callback,
            volatile_qos,
        )
        self.create_subscription(
            String,
            "/photographer_timing_updates",
            self.timing_update_callback,
            reliable_qos,
        )
        self.create_subscription(String, "/mission/drone_status", self.drone_status_callback, reliable_qos)
        self.create_subscription(
            String,
            "/photographer_ready",
            self.photographer_ready_callback,
            volatile_qos,
        )
        self.create_subscription(
            String,
            "/photographer_assignment_rejected",
            self.assignment_rejected_callback,
            volatile_qos,
        )

        self.coordination_timer = self.create_timer(1.0, self.coordination_timer_callback)

        self.get_logger().info(
            f"Photographer Coordinator started | {num_photographers} photographers | timeout={self.assignment_timeout}s"
        )

        self.declare_parameter("photographer_names", [""])
        self.photographer_names = [
            name for name in self.get_parameter("photographer_names").get_parameter_value().string_array_value if name
        ]
        self.eligible_names = set()
        self.create_subscription(String, "/mission/eligible_drones", self._eligible_cb, reliable_qos)
        self.create_subscription(String, "/mission/status", self._mission_status_cb, reliable_qos)

        self.live_map = None
        self.live_meta = None
        self.create_subscription(OccupancyGrid, "/mission/map/occupancy", self._map_cb, reliable_qos)
        self.map_known_ratio = 0.0
        self.create_subscription(String, "/mission/map/status", self._map_status_cb, reliable_qos)

        self.friendly_to_gz: Dict[str, str] = {}
        mm_share = get_package_share_directory("mission_manager")
        profiles_path = os.path.join(mm_share, "config", "spawn_profiles.yaml")
        if os.path.exists(profiles_path):
            with open(profiles_path, "r") as f:
                profiles = yaml.safe_load(f) or {}
            default_profile = profiles.get("default", {}) or {}
            fleet = default_profile.get("fleet", []) or []
            for d in fleet:
                name = str(d.get("name", "")).strip()
                model = str(d.get("model", "")).strip()
                instance = int(d.get("instance", 0))
                if not name or not model or not instance:
                    continue
                gz_base = model[3:] if model.startswith("gz_") else model
                self.friendly_to_gz[name] = f"{gz_base}_{instance}"
            if self.friendly_to_gz:
                self.get_logger().info(f"Loaded friendly->gz mapping: {self.friendly_to_gz}")

    def republish_assignment(self, assignment):
        assignment["timestamp"] = self.get_clock().now().nanoseconds / 1e9
        assignment["last_update"] = assignment["timestamp"]
        assignment_msg = String()
        assignment_msg.data = json.dumps(assignment)
        self.assignment_publisher.publish(assignment_msg)

    def cluster_callback(self, msg):
        data = json.loads(msg.data)
        cluster_id = data.get("cluster_id", "")
        cluster_status = data.get("status", "unassigned")

        if cluster_status == "completed":
            self.get_logger().info(f"Cluster {cluster_id} completed. Removing from all lists.")
            self.available_clusters.pop(cluster_id, None)
            self.assigned_clusters.discard(cluster_id)
            self.rejected_photographers_by_cluster.pop(cluster_id, None)
            self.rejected_clusters_retry_at.pop(cluster_id, None)
            return

        if cluster_status == "timeout":
            self.assigned_clusters.discard(cluster_id)
            self.rejected_photographers_by_cluster.pop(cluster_id, None)
            self.rejected_clusters_retry_at.pop(cluster_id, None)
            for aid in [aid for aid, a in self.active_assignments.items() if a.get("cluster_id") == cluster_id]:
                self.mission_timings.pop(aid, None)
                self._ready_assignment_pending_logs.discard(aid)
                del self.active_assignments[aid]
            self.available_clusters[cluster_id] = data
            return

        if cluster_id in self.assigned_clusters:
            for assignment in self.active_assignments.values():
                if assignment.get("cluster_id") == cluster_id:
                    if len(assignment["cluster_data"]["pois"]) < len(data["pois"]):
                        self.get_logger().info(
                            f"Updating active assignment for cluster {cluster_id} with new POI data."
                        )
                        assignment["cluster_data"] = data
                        self.republish_assignment(assignment)
                    return

        if cluster_status != "unassigned":
            return
        if cluster_id in self.completed_clusters or cluster_id in self.assigned_clusters:
            return

        self.available_clusters[cluster_id] = data

    def position_callback(self, msg):
        data = json.loads(msg.data)
        photographer_id = data.get("photographer_id")
        position = data.get("position")
        if photographer_id and position:
            self.photographer_positions[photographer_id] = position

    def los_status_callback(self, msg):
        data = json.loads(msg.data)
        if "photographer_status" not in data:
            return
        for p_id, status in data["photographer_status"].items():
            has_los = status.get("has_los")
            if has_los is not None:
                if self.photographer_los_status.get(p_id) != has_los:
                    self.get_logger().info(f"LOS status for {p_id} changed to: {has_los}")
                self.photographer_los_status[p_id] = has_los

    def drone_status_callback(self, msg):
        data = json.loads(msg.data)
        for drone in data.get("drones", []):
            model = drone.get("model", "")
            alive = drone.get("alive", False)
            role = drone.get("role", "")
            if role == "photographer" and model:
                if self.photographer_alive_status.get(model) != alive:
                    self.get_logger().info(f"Drone {model} is now {'ALIVE' if alive else 'DEAD'}")
                self.photographer_alive_status[model] = alive

    def photographer_ready_callback(self, msg):
        data = json.loads(msg.data)
        photographer_id = data.get("photographer_id", "")
        if not photographer_id:
            return

        self.get_logger().info(f"Drone {photographer_id} ready - checking for pending assignments")
        self.ready_photographers.add(photographer_id)
        current_time = self.get_clock().now().nanoseconds / 1e9
        stale_assignments = [
            aid
            for aid, adata in list(self.active_assignments.items())
            if adata.get("photographer_id") == photographer_id
        ]
        for aid in stale_assignments:
            adata = self.active_assignments.get(aid, {})
            age = current_time - adata.get("last_update", adata.get("timestamp", current_time))
            if age < self.ready_assignment_stale_grace:
                if aid not in self._ready_assignment_pending_logs:
                    self.get_logger().info(
                        f"Ready from {photographer_id} arrived while assignment {aid} is still active; "
                        "waiting for completion/rejection before reassigning"
                    )
                    self._ready_assignment_pending_logs.add(aid)
                continue
            adata = self.active_assignments.pop(aid, {})
            cluster_id = adata.get("cluster_id")
            self._ready_assignment_pending_logs.discard(aid)
            if cluster_id:
                self.assigned_clusters.discard(cluster_id)
                if cluster_id not in self.available_clusters and cluster_id not in self.completed_clusters:
                    self.available_clusters[cluster_id] = adata.get("cluster_data", {})
                self.get_logger().info(
                    f"Cleared stale assignment {aid} for {photographer_id}, cluster {cluster_id} available again"
                )
        self.coordination_timer_callback()

    def assignment_rejected_callback(self, msg):
        data = json.loads(msg.data)
        assignment_id = data.get("assignment_id")
        cluster_id = data.get("cluster_id")
        reason = data.get("reason", "unknown")
        photographer_id = data.get("photographer_id")
        current_time = self.get_clock().now().nanoseconds / 1e9

        should_wait_for_split = False

        if assignment_id not in self.active_assignments:
            if cluster_id and photographer_id and cluster_id not in self.completed_clusters:
                self.rejected_photographers_by_cluster.setdefault(cluster_id, set()).add(photographer_id)
                self.rejected_clusters_retry_at[cluster_id] = current_time + self.assignment_reject_retry_delay
            return

        adata = self.active_assignments.pop(assignment_id)
        cluster_data = adata.get("cluster_data", {})
        should_wait_for_split = self._should_wait_for_split_retry(reason, cluster_data)
        photographer_id = photographer_id or adata.get("photographer_id")
        self.mission_timings.pop(assignment_id, None)
        self._ready_assignment_pending_logs.discard(assignment_id)

        if cluster_id:
            self.assigned_clusters.discard(cluster_id)
            self.available_clusters.pop(cluster_id, None)
            if photographer_id and not should_wait_for_split:
                self.rejected_photographers_by_cluster.setdefault(cluster_id, set()).add(photographer_id)
                self.rejected_clusters_retry_at[cluster_id] = current_time + self.assignment_reject_retry_delay
            if cluster_id not in self.completed_clusters and not should_wait_for_split:
                self.available_clusters[cluster_id] = cluster_data

        if should_wait_for_split:
            self.rejected_photographers_by_cluster.pop(cluster_id, None)
            self.rejected_clusters_retry_at.pop(cluster_id, None)
            self.get_logger().warn(
                f"Assignment {assignment_id} rejected by {photographer_id}: {reason}; "
                f"waiting for split retry clusters from cluster manager"
            )
        else:
            self.get_logger().warn(
                f"Assignment {assignment_id} rejected by {photographer_id}: {reason}; "
                f"cluster {cluster_id} available again"
            )
        self.coordination_timer_callback()

    def assignment_complete_callback(self, msg):
        data = json.loads(msg.data)
        assignment_id = data.get("assignment_id")
        cluster_id = data.get("cluster_id")
        photographer_id = data.get("photographer_id")

        if not cluster_id:
            return
        if cluster_id in self.completed_clusters:
            return

        self.completed_clusters.add(cluster_id)
        self.assigned_clusters.discard(cluster_id)
        self.available_clusters.pop(cluster_id, None)
        self.rejected_photographers_by_cluster.pop(cluster_id, None)
        self.rejected_clusters_retry_at.pop(cluster_id, None)
        self._ready_assignment_pending_logs.discard(assignment_id)

        self.get_logger().info(
            f"Cluster {cluster_id} completed by {photographer_id} (id: {assignment_id}). "
            f"Total completed: {len(self.completed_clusters)}"
        )

    def find_best_photographer_for_cluster(self, cluster_center, cluster_id=None, ignore_rejections=False):
        best_photographer = None
        min_distance = float("inf")
        eligible_now = self.eligible_names or set(self.photographer_names)
        rejected_for_cluster = self.rejected_photographers_by_cluster.get(cluster_id, set()) if cluster_id else set()
        for photographer_id, position in self.photographer_positions.items():
            if self.photographer_names and photographer_id not in self.photographer_names:
                continue
            if eligible_now and photographer_id not in eligible_now:
                continue
            if not ignore_rejections and photographer_id in rejected_for_cluster:
                continue
            if self.require_photographer_ready and photographer_id not in self.ready_photographers:
                continue
            if not self.photographer_los_status.get(photographer_id, False):
                continue
            if not self.photographer_alive_status.get(photographer_id, True):
                continue
            is_assigned = any(a.get("photographer_id") == photographer_id for a in self.active_assignments.values())
            if is_assigned:
                continue
            dx = position.get("x", 0.0) - cluster_center[0]
            dy = position.get("y", 0.0) - cluster_center[1]
            dz = position.get("z", 0.0) - cluster_center[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance < min_distance:
                min_distance = distance
                best_photographer = photographer_id

        return best_photographer

    def assign_cluster_to_photographer(self, cluster_data, photographer_id):
        assignment_id = str(uuid.uuid4())
        cluster_id = cluster_data.get("cluster_id", "")
        assignment_timestamp = self.get_clock().now().nanoseconds / 1e9

        if cluster_id in self.assigned_clusters:
            self.get_logger().warn(f"Cluster {cluster_id} already assigned. Skipping {photographer_id}")
            return False

        for a in self.active_assignments.values():
            if a.get("photographer_id") == photographer_id:
                self.get_logger().warn(f"Photographer {photographer_id} already has assignment. Skipping.")
                return False

        assignment = {
            "assignment_id": assignment_id,
            "photographer_id": photographer_id,
            "cluster_data": cluster_data,
            "cluster_id": cluster_id,
            "assignment_type": "cluster",
            "timestamp": assignment_timestamp,
            "last_update": assignment_timestamp,
        }

        self.active_assignments[assignment_id] = assignment
        self.assigned_clusters.add(cluster_id)
        self.ready_photographers.discard(photographer_id)
        self.first_assignment_attempted = True

        self.mission_timings[assignment_id] = {
            "photographer_id": photographer_id,
            "cluster_id": cluster_id,
            "assignment_start": assignment_timestamp,
            "cluster_start": None,
            "cluster_complete": None,
            "home_return": None,
            "metrics": {},
        }

        self.available_clusters.pop(cluster_id, None)

        assignment_msg = String()
        assignment_msg.data = json.dumps(assignment)
        self.assignment_publisher.publish(assignment_msg)
        self.get_logger().info(f"Assigned cluster {cluster_id} to {photographer_id} (id: {assignment_id})")
        return True

    def check_timeout_assignments(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        timed_out = [
            aid
            for aid, a in self.active_assignments.items()
            if current_time - a.get("last_update", a.get("timestamp", 0.0)) > self.assignment_timeout
        ]
        for assignment_id in timed_out:
            assignment = self.active_assignments.pop(assignment_id)
            cluster_data = assignment.get("cluster_data", {})
            cluster_id = assignment.get("cluster_id", "")
            photographer_id = assignment.get("photographer_id")

            self.get_logger().warn(f"Assignment {assignment_id} timed out for {photographer_id}")
            self.stale_assignment_ids.add(assignment_id)
            self.mission_timings.pop(assignment_id, None)
            self._ready_assignment_pending_logs.discard(assignment_id)
            self.assigned_clusters.discard(cluster_id)
            self.available_clusters[cluster_id] = cluster_data

    def _ready_photographers_for_first_assignment(self):
        if self.photographer_names:
            expected = set(self.photographer_names)
        else:
            expected = set()
            for idx in [2, 4, 5]:
                if len(expected) >= self.num_photographers:
                    break
                expected.add(f"x500_gimbal_photographer_{idx}")

        if not expected:
            return self.ready_photographers, self.num_photographers

        return self.ready_photographers.intersection(expected), len(expected)

    def _first_assignment_gate_open(self):
        if not self.wait_for_all_ready_before_first_assignment:
            return True
        if not self.require_photographer_ready:
            return True
        if self.first_assignment_attempted:
            return True
        if self.active_assignments or self.assigned_clusters or self.completed_clusters:
            return True
        if not self.available_clusters:
            return True

        ready, expected_count = self._ready_photographers_for_first_assignment()
        if expected_count <= 0 or len(ready) >= expected_count:
            return True

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self._last_first_assignment_wait_log_time >= 5.0:
            expected = set(self.photographer_names)
            if not expected:
                expected = {f"x500_gimbal_photographer_{idx}" for idx in [2, 4, 5][: self.num_photographers]}
            missing = sorted(expected - ready)
            self.get_logger().info(
                f"Waiting for all photographers to reach ready before first assignment: "
                f"{len(ready)}/{expected_count} ready; missing={missing}"
            )
            self._last_first_assignment_wait_log_time = current_time
        return False

    def _active_generation_count(self):
        count = 0
        for assignment_id in self.active_assignments:
            timing_data = self.mission_timings.get(assignment_id, {})
            if timing_data.get("cluster_start") is None:
                count += 1
        return count

    def _should_wait_for_split_retry(self, reason, cluster_data):
        return (
            self.world_model_source == "lidar_only"
            and reason in SPLIT_RETRY_REASONS
            and len(cluster_data.get("pois", [])) > 1
        )

    def coordination_timer_callback(self):
        self.check_timeout_assignments()

        if self.enable_lidar_gating and self.world_model_source == "lidar_only":
            if self.map_known_ratio < self.global_required_known_ratio:
                if self.offboard_setpoint_counter % 50 == 0:
                    self.get_logger().info(
                        f"Global gating active: known_ratio={self.map_known_ratio:.3f} "
                        f"< required {self.global_required_known_ratio:.3f}"
                    )
                self.offboard_setpoint_counter += 1
                return

        if not self._first_assignment_gate_open():
            self.offboard_setpoint_counter += 1
            return

        unassigned_clusters = {}
        available_clusters_items = list(self.available_clusters.items())
        assignments_made = 0

        for cluster_id, cluster_data in available_clusters_items:
            if cluster_id in self.assigned_clusters:
                continue

            if self.enable_lidar_gating and self.world_model_source == "lidar_only":
                center = cluster_data.get("center", [0.0, 0.0, 0.0])
                local_ratio = self._compute_local_coverage(center)
                if local_ratio is None or local_ratio < self.required_known_ratio:
                    unassigned_clusters[cluster_id] = cluster_data
                    if self.offboard_setpoint_counter % 50 == 0:
                        self.get_logger().info(
                            f"Gating {cluster_id}: coverage {local_ratio if local_ratio is not None else 'n/a'} "
                            f"< {self.required_known_ratio}"
                        )
                    continue

            if (
                self.max_concurrent_trajectory_generations > 0
                and self._active_generation_count() >= self.max_concurrent_trajectory_generations
            ):
                unassigned_clusters[cluster_id] = cluster_data
                if self.offboard_setpoint_counter % 50 == 0:
                    self.get_logger().info(
                        f"Waiting to assign {cluster_id}: "
                        f"{self._active_generation_count()}/"
                        f"{self.max_concurrent_trajectory_generations} "
                        "trajectory generations active"
                    )
                continue

            cluster_center = cluster_data.get("center", [0.0, 0.0, 0.0])
            best_photographer = self.find_best_photographer_for_cluster(cluster_center, cluster_id=cluster_id)
            if not best_photographer and self.rejected_photographers_by_cluster.get(cluster_id):
                current_time = self.get_clock().now().nanoseconds / 1e9
                retry_at = self.rejected_clusters_retry_at.get(cluster_id, 0.0)
                if current_time < retry_at:
                    unassigned_clusters[cluster_id] = cluster_data
                    if self.offboard_setpoint_counter % 50 == 0:
                        wait_s = retry_at - current_time
                        self.get_logger().warn(
                            f"Cluster {cluster_id} rejected by all currently eligible photographers; "
                            f"retrying in {wait_s:.1f}s"
                        )
                    continue

                rejected_photographers = self.rejected_photographers_by_cluster.pop(cluster_id, set())
                self.rejected_clusters_retry_at.pop(cluster_id, None)
                self.ready_photographers.update(rejected_photographers)
                best_photographer = self.find_best_photographer_for_cluster(cluster_center, cluster_id=cluster_id)
                if best_photographer:
                    self.get_logger().warn(f"Retry cooldown expired for cluster {cluster_id}; retrying assignment")

            if best_photographer:
                if self.assign_cluster_to_photographer(cluster_data, best_photographer):
                    assignments_made += 1
                else:
                    unassigned_clusters[cluster_id] = cluster_data
            else:
                unassigned_clusters[cluster_id] = cluster_data

        if assignments_made == 0 and available_clusters_items and self.offboard_setpoint_counter % 50 == 0:
            if not self.photographer_positions:
                self.get_logger().warn("No photographers have reported positions yet")
            elif not any(self.photographer_los_status.values()):
                self.get_logger().warn("No photographers have LOS to GCS")
            else:
                assigned_ids = [a.get("photographer_id") for a in self.active_assignments.values()]
                self.get_logger().warn(f"All photographers busy. Assigned: {assigned_ids}")

        self.available_clusters = unassigned_clusters
        self.offboard_setpoint_counter += 1

    def _mission_status_cb(self, msg: String):
        data = json.loads(msg.data)
        wms = data.get("world_model_source")
        if isinstance(wms, str):
            self.world_model_source = wms

    def _map_cb(self, msg):
        w = int(msg.info.width)
        h = int(msg.info.height)
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        self.live_map = np.array(msg.data, dtype=np.int16).reshape((h, w))
        self.live_meta = {
            "width": w,
            "height": h,
            "resolution": res,
            "origin_x": ox,
            "origin_y": oy,
        }

    def _map_status_cb(self, msg: String):
        data = json.loads(msg.data)
        kr = data.get("known_ratio")
        if isinstance(kr, (int, float)):
            self.map_known_ratio = float(kr)

    def _compute_local_coverage(self, center_xyz) -> Optional[float]:
        if self.live_map is None or self.live_meta is None:
            return None
        cx, cy = float(center_xyz[0]), float(center_xyz[1])
        res = self.live_meta["resolution"]
        ox = self.live_meta["origin_x"]
        oy = self.live_meta["origin_y"]
        w = self.live_meta["width"]
        h = self.live_meta["height"]
        radius_cells = max(1, int(round(self.cluster_radius_m / res)))
        cx_i = int((cx - ox) / res)
        cy_i = int((cy - oy) / res)
        x0 = max(0, cx_i - radius_cells)
        x1 = min(w, cx_i + radius_cells + 1)
        y0 = max(0, cy_i - radius_cells)
        y1 = min(h, cy_i + radius_cells + 1)
        sub = self.live_map[y0:y1, x0:x1]
        if sub.size == 0:
            return 0.0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx_i) * (xx - cx_i) + (yy - cy_i) * (yy - cy_i) <= radius_cells * radius_cells
        roi = sub[mask]
        if roi.size == 0:
            return 0.0
        known = int(np.count_nonzero(roi != -1))
        return (known / roi.size) if roi.size > 0 else 0.0

    def _eligible_cb(self, msg: String):
        data = json.loads(msg.data)
        names = data.get("eligible", []) or []
        mapped = []
        for n in names:
            n = str(n)
            mapped.append(self.friendly_to_gz.get(n, n))
        self.eligible_names = set(mapped)

    def _is_timing_complete(self, timing_data):
        required_events = [
            "assignment_start",
            "cluster_start",
            "cluster_complete",
            "home_return",
        ]
        return all(timing_data.get(event) is not None for event in required_events)

    def timing_update_callback(self, msg):
        data = json.loads(msg.data)
        assignment_id = data.get("assignment_id")
        photographer_id = data.get("photographer_id")
        timing_event = data.get("timing_event")
        timestamp = data.get("timestamp")

        if timing_event == "takeoff_complete" and photographer_id and timestamp:
            return

        if not all([assignment_id, photographer_id, timing_event, timestamp]):
            self.get_logger().warn(
                f"Invalid timing update: assignment_id={assignment_id}, "
                f"photographer_id={photographer_id}, timing_event={timing_event}, timestamp={timestamp}"
            )
            return

        if assignment_id in self.stale_assignment_ids:
            return
        if assignment_id not in self.mission_timings:
            self.get_logger().warn(f"Ignoring timing update for unknown assignment {assignment_id}")
            return

        timing_data = self.mission_timings[assignment_id]
        timing_data[timing_event] = timestamp

        metrics = data.get("metrics", {})
        if metrics:
            timing_data["metrics"].update(metrics)

        sequencing_method = data.get("sequencing_method")
        if sequencing_method:
            timing_data["sequencing_method"] = sequencing_method

        if assignment_id in self.active_assignments:
            self.active_assignments[assignment_id]["last_update"] = timestamp

        if timing_event == "home_return" and self._is_timing_complete(timing_data):
            self.publish_timing_data(assignment_id, timing_data, is_complete=True)
        elif self.enable_partial_timing_logs and timing_event in [
            "cluster_start",
            "cluster_complete",
            "home_return",
        ]:
            self.publish_timing_data(assignment_id, timing_data, is_complete=False, event_type=timing_event)
        else:
            self.get_logger().info(f"Timing cycle not yet complete for {assignment_id}")

    def publish_timing_data(self, assignment_id, timing_data, is_complete=False, event_type=None):
        assignment_start = timing_data.get("assignment_start")
        cluster_start = timing_data.get("cluster_start")
        cluster_complete = timing_data.get("cluster_complete")
        home_return = timing_data.get("home_return")

        assignment_timestamp = (
            datetime.fromtimestamp(assignment_start).strftime("%Y-%m-%d %H:%M:%S") if assignment_start else "unknown"
        )

        time_to_cluster = (cluster_start - assignment_start) if assignment_start and cluster_start else None
        time_within_cluster = (cluster_complete - cluster_start) if cluster_start and cluster_complete else None
        time_to_return_home = (home_return - cluster_complete) if cluster_complete and home_return else None
        total_time = (home_return - assignment_start) if assignment_start and home_return else None

        base_metrics = {
            "time_to_cluster": time_to_cluster,
            "time_within_cluster": time_within_cluster,
            "time_to_return_home": time_to_return_home,
            "total_time": total_time,
        }
        base_metrics.update(timing_data.get("metrics", {}))

        timing_msg = {
            "assignment_id": assignment_id,
            "photographer_id": timing_data["photographer_id"],
            "cluster_id": timing_data["cluster_id"],
            "timestamp": datetime.now().isoformat(),
            "assignment_timestamp": assignment_timestamp,
            "is_complete": is_complete,
            "current_event": event_type,
            "sequencing_method": timing_data.get("sequencing_method", ""),
            "metrics": base_metrics,
            "raw_timestamps": {
                "assignment_start": assignment_start,
                "cluster_start": cluster_start,
                "cluster_complete": cluster_complete,
                "home_return": home_return,
            },
        }

        ros_msg = String()
        ros_msg.data = json.dumps(timing_msg)
        self.timing_data_publisher.publish(ros_msg)

        if is_complete and assignment_id in self.mission_timings:
            del self.mission_timings[assignment_id]
            self.active_assignments.pop(assignment_id, None)
            self.get_logger().info(f"Cleaned up completed timing data for {assignment_id}")


def main(args=None):
    rclpy.init(args=args)
    coordinator = PhotographerCoordinator()
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            coordinator.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
