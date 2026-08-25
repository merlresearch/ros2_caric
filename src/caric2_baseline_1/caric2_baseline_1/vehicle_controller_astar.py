#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os
import time
from datetime import datetime

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from pathfinding.core.diagonal_movement import DiagonalMovement as PFDiag
from pathfinding.core.grid import Grid as PFGrid
from pathfinding.finder.a_star import AStarFinder as PFFinder
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from vehicle_controller_interfaces.srv import CapturePhoto

from .astar_path_planner import AStarPathPlanner


class VehicleControllerAStar(Node):
    def __init__(self):
        super().__init__("vehicle_controller_astar")

        self.declare_parameter("drone_instance", 2)
        self.declare_parameter("spawn_x", 0.0)
        self.declare_parameter("spawn_y", 0.0)
        self.declare_parameter("spawn_yaw", 0.0)
        self.declare_parameter("poi_buffer_x", 0.0)
        self.declare_parameter("poi_buffer_y", 0.0)
        self.declare_parameter("poi_buffer_z", 0.0)
        self.declare_parameter("poi_ordering_method", "tsp_networkx")
        self.declare_parameter("world", "mbs")
        self.declare_parameter("voxel_grid_path", "")
        self.declare_parameter("voxel_metadata_path", "")
        self.declare_parameter("waypoint_tolerance", 0.5)
        self.declare_parameter("use_astar_planning", True)
        self.declare_parameter("map_source", "live")
        self.declare_parameter("live_map_topic", "/mission/map/occupancy")
        self.declare_parameter("live_map_replan_hz", 1.0)
        self.declare_parameter("live_unknown_free_radius_m", 5.0)
        self.declare_parameter("live_unknown_free_enable_sec", 60.0)
        self.declare_parameter("nearest_walkable_max_radius_cells", 20)
        self.declare_parameter("live_planner_inflate_m", 1.0)
        self.declare_parameter("min_waypoint_spacing_m", 1.0)
        self.declare_parameter("live_allow_diagonal", True)
        self.declare_parameter("replan_min_period_sec", 3.0)
        self.declare_parameter("replan_changed_fraction_threshold", 0.02)
        self.declare_parameter("replan_lookahead_waypoints", 5)
        self.declare_parameter("planning_roi_pad_cells", 120)
        self.declare_parameter("unknown_free_global_enable_sec", 600.0)
        self.declare_parameter("unknown_free_global_known_ratio", 0.30)
        self.declare_parameter("require_startup_readiness", False)
        self.declare_parameter("startup_readiness_topic", "/fleet_ready")

        self.drone_instance = self.get_parameter("drone_instance").get_parameter_value().integer_value
        self.spawn_x = self.get_parameter("spawn_x").get_parameter_value().double_value
        self.spawn_y = self.get_parameter("spawn_y").get_parameter_value().double_value
        self.spawn_yaw_deg = self.get_parameter("spawn_yaw").get_parameter_value().double_value
        self.poi_buffer_x = self.get_parameter("poi_buffer_x").get_parameter_value().double_value
        self.poi_buffer_y = self.get_parameter("poi_buffer_y").get_parameter_value().double_value
        self.poi_buffer_z = self.get_parameter("poi_buffer_z").get_parameter_value().double_value
        self.poi_ordering_method = self.get_parameter("poi_ordering_method").get_parameter_value().string_value
        self.world = self.get_parameter("world").get_parameter_value().string_value
        self.voxel_grid_path = self.get_parameter("voxel_grid_path").get_parameter_value().string_value
        self.voxel_metadata_path = self.get_parameter("voxel_metadata_path").get_parameter_value().string_value
        self.waypoint_tolerance = self.get_parameter("waypoint_tolerance").get_parameter_value().double_value
        self.use_astar_planning = self.get_parameter("use_astar_planning").get_parameter_value().bool_value
        self.map_source = self.get_parameter("map_source").get_parameter_value().string_value

        map_source_env = os.environ.get("MISSION_MAP_SOURCE", "")
        if map_source_env:
            mapped = map_source_env.strip().lower()
            if mapped == "lidar_only":
                mapped = "live"
            elif mapped == "known":
                mapped = "file"
            self.map_source = mapped
            self.get_logger().info(f"Map source from env: {self.map_source}")

        self.live_map_topic = self.get_parameter("live_map_topic").get_parameter_value().string_value
        self.live_replan_hz = float(self.get_parameter("live_map_replan_hz").value)
        self.replan_min_period_sec = float(self.get_parameter("replan_min_period_sec").value)
        self.replan_changed_fraction_threshold = float(self.get_parameter("replan_changed_fraction_threshold").value)
        self.replan_lookahead_waypoints = int(self.get_parameter("replan_lookahead_waypoints").value)
        self.planning_roi_pad_cells = int(self.get_parameter("planning_roi_pad_cells").value)
        self.unknown_free_global_enable_sec = float(self.get_parameter("unknown_free_global_enable_sec").value)
        self.unknown_free_global_known_ratio = float(self.get_parameter("unknown_free_global_known_ratio").value)
        self.require_startup_readiness = bool(self.get_parameter("require_startup_readiness").value)
        self.startup_readiness_topic = (
            self.get_parameter("startup_readiness_topic").get_parameter_value().string_value or "/fleet_ready"
        )
        self.startup_ready = not self.require_startup_readiness
        self.unknown_free_radius_m = float(self.get_parameter("live_unknown_free_radius_m").value)
        self.unknown_free_enable_sec = float(self.get_parameter("live_unknown_free_enable_sec").value)
        self.nearest_walkable_max_radius = int(
            self.get_parameter("nearest_walkable_max_radius_cells").get_parameter_value().integer_value
        )
        self.live_inflate_m = float(self.get_parameter("live_planner_inflate_m").value)
        self.min_waypoint_spacing_m = float(self.get_parameter("min_waypoint_spacing_m").value)
        self.live_allow_diagonal = bool(self.get_parameter("live_allow_diagonal").get_parameter_value().bool_value)

        # Additional tolerance for detecting arrival at home (Euclidean radius)
        self.declare_parameter("home_arrival_radius", 1.0)
        home_arrival_radius_param = self.get_parameter("home_arrival_radius")
        self.home_arrival_radius = home_arrival_radius_param.get_parameter_value().double_value

        # Configure QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        transient_local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sys_id = self.drone_instance + 1
        self.topic_prefix = f"/px4_{self.drone_instance}"
        self.gazebo_model_name = f"x500_gimbal_photographer_{self.drone_instance}"

        self.path_planner = None
        self.live_map_meta = None
        self.live_map = None
        self._logged_live_map_ready = False
        self.map_known_ratio = None
        self.map_changed_fraction = None
        self.create_subscription(OccupancyGrid, self.live_map_topic, self._live_map_cb, transient_local_qos)
        replan_period = max(0.2, 1.0 / max(0.1, self.live_replan_hz))
        self.create_timer(replan_period, self._live_map_tick)
        if self.use_astar_planning:
            self._initialize_path_planner()

        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_arrival_time = None
        self.current_target = None
        self.node_start_time_sec = self.get_clock().now().nanoseconds / 1e9
        self.last_replan_time = 0.0
        self.local_offset_x = 0.0
        self.local_offset_y = 0.0
        self.local_offset_z = 0.0
        self.local_frame_calibrated = False
        self.transform_checked = False

        self.get_logger().info(
            f"Drone {self.drone_instance}: A*={'on' if self.use_astar_planning else 'off'}, "
            f"ordering={self.poi_ordering_method}, map_source={self.map_source}"
        )

        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, f"{self.topic_prefix}/fmu/in/vehicle_command", reliable_qos
        )
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, f"{self.topic_prefix}/fmu/in/offboard_control_mode", reliable_qos
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, f"{self.topic_prefix}/fmu/in/trajectory_setpoint", reliable_qos
        )

        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            f"{self.topic_prefix}/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            best_effort_qos,
        )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, f"{self.topic_prefix}/fmu/out/vehicle_status", self.vehicle_status_callback, best_effort_qos
        )
        self.vehicle_status_subscriber_v1 = self.create_subscription(
            VehicleStatus,
            f"{self.topic_prefix}/fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            best_effort_qos,
        )
        if self.require_startup_readiness:
            self.startup_readiness_subscriber = self.create_subscription(
                Bool,
                self.startup_readiness_topic,
                self.startup_readiness_callback,
                transient_local_qos,
            )

        self.position_publisher = self.create_publisher(String, "/photographer_positions", transient_local_qos)
        self.timing_publisher = self.create_publisher(String, "/mission_timing_data", transient_local_qos)
        self.timing_update_publisher = self.create_publisher(
            String, "/photographer_timing_updates", transient_local_qos
        )
        self._last_heartbeat_time = 0.0

        self.assignment_subscriber = self.create_subscription(
            String, "/photographer_assignments", self.assignment_callback, 10
        )
        self.assignment_complete_publisher = self.create_publisher(String, "/photographer_assignment_complete", 10)
        self.assignment_rejection_publisher = self.create_publisher(String, "/photographer_assignment_rejected", 10)
        self.los_status_subscriber = self.create_subscription(
            String, "/photographer_los_status", self.los_status_callback, 10
        )

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.data_received = False

        self.flight_state = "INIT"
        self.offboard_setpoint_counter = 0
        self.takeoff_height = -5.0
        self.home_position = [0.0, 0.0, 0.0]

        self.current_assignment = None
        self.current_assignment_id = None
        self.current_cluster_id = None
        self.processed_cluster_ids = set()
        self.rejected_assignment_ids = set()
        self.cluster_coverage_state = "MOVE_TO_CENTER"
        self.coverage_start_time = None
        self.coverage_position_index = 0

        self.cluster_poi_queue = []
        self.current_cluster_poi = None
        self.cluster_pois_completed = 0
        self.current_sequencing_method = "FIFO"
        self.last_tsp_solve_time = None

        self.has_los_to_gcs = False
        self.last_los_check_time = None
        self.mission_start_time = None
        self.takeoff_complete_time = None
        self.takeoff_complete_published = False

        self.mission_timings = {}
        self.current_assignment_astar_time = 0.0

        self.coverage_duration = 10.0
        self.coverage_height_variation = 3.0
        self.coverage_pattern_interval = 2.0

        self.photo_capture_client = self.create_client(CapturePhoto, "/capture_photo")
        self.photo_capture_attempted = False
        self.photo_capture_success = False
        # Non-blocking photo capture: never stall the control loop waiting on the service.
        self.declare_parameter("photo_capture_timeout_sec", 2.0)
        self.photo_capture_timeout_sec = float(self.get_parameter("photo_capture_timeout_sec").value)
        self._photo_future = None
        self._photo_request_time = None
        self.photo_duration = None

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.service_ready_logged = False
        self.los_check_client = self.create_client(SetBool, "/check_photographer_los")
        self._last_startup_readiness_log_time = 0.0

    def _initialize_path_planner(self):
        voxel_grid_path_eff = (self.voxel_grid_path or "").strip()
        voxel_meta_path_eff = (self.voxel_metadata_path or "").strip()
        if not voxel_grid_path_eff or not voxel_meta_path_eff:
            share_dir = get_package_share_directory("mission_manager")
            base_dir = os.path.join(share_dir, "models", self.world, "voxel_grid")
            if not voxel_grid_path_eff:
                voxel_grid_path_eff = os.path.join(base_dir, f"{self.world}_voxel_grid.npy")
            if not voxel_meta_path_eff:
                voxel_meta_path_eff = os.path.join(base_dir, f"{self.world}_voxel_grid_metadata.pkl")

        if not os.path.isabs(voxel_grid_path_eff):
            voxel_grid_path_eff = os.path.join(os.path.expanduser("~/ros2_ws"), voxel_grid_path_eff)
        if not os.path.isabs(voxel_meta_path_eff):
            voxel_meta_path_eff = os.path.join(os.path.expanduser("~/ros2_ws"), voxel_meta_path_eff)

        self.voxel_grid_path = voxel_grid_path_eff
        self.voxel_metadata_path = voxel_meta_path_eff

        self.get_logger().info(f"Loading voxel grid from: {self.voxel_grid_path}")
        self.path_planner = AStarPathPlanner(self.voxel_grid_path, self.voxel_metadata_path)
        self.get_logger().info("A* Path Planner initialized successfully")

    def plan_path_to_target(self, target_local):
        if self.map_source == "live" and self.live_map is not None and self.live_map_meta:
            return self._plan_path_live_2d(target_local)

        if not self.use_astar_planning or self.path_planner is None:
            current_pos = [self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z]
            return [current_pos, target_local]

        start_local = (self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z)
        self.get_logger().info(f"Planning A* path from {start_local} to {target_local}")

        t0 = time.time()
        waypoints = self.path_planner.plan_path_local(
            start_local, tuple(target_local), self.spawn_x, self.spawn_y, self.spawn_yaw_deg
        )
        self.current_assignment_astar_time += max(0.0, time.time() - t0)

        if not waypoints:
            self.get_logger().warn("A* planning failed")
            return []

        waypoints = self._decimate_waypoints(waypoints, target_local)
        self.get_logger().info(f"A* path found with {len(waypoints)} waypoints")
        return waypoints

    def _decimate_waypoints(self, waypoints, target_local):
        spacing2 = max(0.0, float(self.min_waypoint_spacing_m)) ** 2
        if spacing2 <= 0.0 or len(waypoints) <= 2:
            return waypoints
        decimated = []
        last_keep = None
        for wp in waypoints:
            lx, ly, lz = float(wp[0]), float(wp[1]), float(wp[2])
            if last_keep is None:
                decimated.append([lx, ly, lz])
                last_keep = (lx, ly, lz)
            else:
                dx, dy, dz = lx - last_keep[0], ly - last_keep[1], lz - last_keep[2]
                if (dx * dx + dy * dy + dz * dz) >= spacing2:
                    decimated.append([lx, ly, lz])
                    last_keep = (lx, ly, lz)
        tx_l, ty_l, tz_l = float(target_local[0]), float(target_local[1]), float(target_local[2])
        if not decimated:
            decimated = [[tx_l, ty_l, tz_l]]
        else:
            dx_t, dy_t, dz_t = decimated[-1][0] - tx_l, decimated[-1][1] - ty_l, decimated[-1][2] - tz_l
            if (dx_t * dx_t + dy_t * dy_t + dz_t * dz_t) > 0.04:
                decimated.append([tx_l, ty_l, tz_l])
        return decimated

    def _live_map_cb(self, msg: OccupancyGrid):
        w = int(msg.info.width)
        h = int(msg.info.height)
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        self.live_map = np.array(msg.data, dtype=np.int16).reshape((h, w))
        self.live_map_meta = {"width": w, "height": h, "resolution": res, "origin_x": ox, "origin_y": oy}

    def _live_map_tick(self):
        # Check if we need to replan based on live map updates
        if self.map_source != "live" or not self.current_target:
            return
        # Only replan if we have a valid live map
        if self.live_map is None or self.live_map_meta is None:
            if not self._logged_live_map_ready:
                self.get_logger().warn("Waiting for live map data...")
                self._logged_live_map_ready = True
            return
        # Throttle replanning
        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_replan_time) < max(0.5, self.replan_min_period_sec):
            return
        # Replan path using live map
        new_path = self._plan_path_live_2d(self.current_target)
        if new_path and len(new_path) > 1:
            # Preserve progress: resume near current pose with lookahead
            cx = float(self.vehicle_local_position.x)
            cy = float(self.vehicle_local_position.y)
            cz = float(self.vehicle_local_position.z)
            best_idx = 0
            best_d2 = float("inf")
            for idx, wp in enumerate(new_path):
                dx = wp[0] - cx
                dy = wp[1] - cy
                dz = wp[2] - cz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = idx
            start_idx = min(best_idx + max(0, self.replan_lookahead_waypoints), len(new_path) - 1)
            old_len = len(self.current_waypoints) if self.current_waypoints else 0
            self.current_waypoints = new_path
            self.current_waypoint_index = start_idx
            self.last_replan_time = now
            self.get_logger().info(f"Replanned path using live map: {old_len} -> {len(new_path)} waypoints")
        else:
            self.get_logger().warn("Failed to replan path with live map")

    def _plan_path_live_2d(self, target_local):
        t0 = time.time()
        if self.live_map is None or self.live_map_meta is None:
            return None
        grid = self.live_map
        meta = self.live_map_meta
        res = meta["resolution"]
        ox, oy = meta["origin_x"], meta["origin_y"]
        width, height = meta["width"], meta["height"]

        sx_w, sy_w, sz_w = self.local_to_world_coordinates(
            float(self.vehicle_local_position.x),
            float(self.vehicle_local_position.y),
            float(self.vehicle_local_position.z),
        )
        tx_w, ty_w, tz_w = self.local_to_world_coordinates(target_local[0], target_local[1], target_local[2])
        sx_i = max(0, min(width - 1, int((sx_w - ox) / res)))
        sy_i = max(0, min(height - 1, int((sy_w - oy) / res)))
        tx_i = max(0, min(width - 1, int((tx_w - ox) / res)))
        ty_i = max(0, min(height - 1, int((ty_w - oy) / res)))

        pad = max(10, self.planning_roi_pad_cells)
        roi_x0 = max(0, min(sx_i, tx_i) - pad)
        roi_x1 = min(width - 1, max(sx_i, tx_i) + pad)
        roi_y0 = max(0, min(sy_i, ty_i) - pad)
        roi_y1 = min(height - 1, max(sy_i, ty_i) + pad)
        roi_w = roi_x1 - roi_x0 + 1
        roi_h = roi_y1 - roi_y0 + 1

        now_sec = self.get_clock().now().nanoseconds / 1e9
        unknown_free_global = (now_sec - self.node_start_time_sec) < self.unknown_free_global_enable_sec or (
            self.map_known_ratio is not None and self.map_known_ratio < self.unknown_free_global_known_ratio
        )

        walk = np.ones((roi_h, roi_w), dtype=int)
        for y in range(roi_y0, roi_y1 + 1):
            for x in range(roi_x0, roi_x1 + 1):
                cell = grid[y, x]
                xi, yi = x - roi_x0, y - roi_y0
                if cell == 100:
                    walk[yi, xi] = 0
                elif cell == -1 and not unknown_free_global:
                    ds2 = ((x - sx_i) * res) ** 2 + ((y - sy_i) * res) ** 2
                    dt2 = ((x - tx_i) * res) ** 2 + ((y - ty_i) * res) ** 2
                    if not (ds2 < self.unknown_free_radius_m**2 or dt2 < self.unknown_free_radius_m**2):
                        walk[yi, xi] = 0

        g = PFGrid(matrix=walk.tolist())
        sx_l = max(0, min(g.width - 1, sx_i - roi_x0))
        sy_l = max(0, min(g.height - 1, sy_i - roi_y0))
        tx_l = max(0, min(g.width - 1, tx_i - roi_x0))
        ty_l = max(0, min(g.height - 1, ty_i - roi_y0))

        def _nearest_walkable(node_x, node_y):
            max_r = max(3, int(self.nearest_walkable_max_radius))
            if g.walkable(node_x, node_y):
                return g.node(node_x, node_y)
            for r in range(1, max_r + 1):
                x0 = max(0, node_x - r)
                x1 = min(g.width - 1, node_x + r)
                y0 = max(0, node_y - r)
                y1 = min(g.height - 1, node_y + r)
                for x in range(x0, x1 + 1):
                    if g.walkable(x, y0):
                        return g.node(x, y0)
                    if g.walkable(x, y1):
                        return g.node(x, y1)
                for y in range(y0 + 1, y1):
                    if g.walkable(x0, y):
                        return g.node(x0, y)
                    if g.walkable(x1, y):
                        return g.node(x1, y)
            return None

        start = _nearest_walkable(sx_l, sy_l)
        end = _nearest_walkable(tx_l, ty_l)
        if start is None or end is None:
            return None
        finder = PFFinder(diagonal_movement=(PFDiag.always if self.live_allow_diagonal else PFDiag.never))
        nodes, _ = finder.find_path(start, end, g)
        if not nodes:
            return None

        path_local = []
        last_keep = None
        spacing2 = self.min_waypoint_spacing_m**2
        for n in nodes:
            wx = ox + (roi_x0 + n.x) * res
            wy = oy + (roi_y0 + n.y) * res
            lx, ly, lz = self.world_to_local_coordinates(wx, wy, tz_w)
            if last_keep is None:
                path_local.append((lx, ly, lz))
                last_keep = (lx, ly, lz)
            else:
                dx, dy, dz = lx - last_keep[0], ly - last_keep[1], lz - last_keep[2]
                if (dx * dx + dy * dy + dz * dz) >= spacing2:
                    path_local.append((lx, ly, lz))
                    last_keep = (lx, ly, lz)

        ftx, fty, ftz = float(target_local[0]), float(target_local[1]), float(target_local[2])
        if not path_local:
            path_local.append((ftx, fty, ftz))
        else:
            dx_t, dy_t, dz_t = path_local[-1][0] - ftx, path_local[-1][1] - fty, path_local[-1][2] - ftz
            if (dx_t * dx_t + dy_t * dy_t + dz_t * dz_t) > 0.04:
                path_local.append((ftx, fty, ftz))

        self.current_assignment_astar_time += max(0.0, time.time() - t0)
        return path_local

    def start_navigation_to_target(self, target_local):
        self.current_target = [float(target_local[0]), float(target_local[1]), float(target_local[2])]

        if self.map_source == "live" and self.live_map is not None and self.live_map_meta is not None:
            self.get_logger().info("Planning path using live map")
            waypoints = self._plan_path_live_2d(target_local)
            if not waypoints:
                self.get_logger().warn("Live map planning failed, falling back to A* planner")
                waypoints = self.plan_path_to_target(target_local)
        else:
            waypoints = self.plan_path_to_target(target_local)

        if waypoints:
            if len(waypoints) > 1:
                first_wp = waypoints[0]
                cx, cy, cz = self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z
                dist = ((first_wp[0] - cx) ** 2 + (first_wp[1] - cy) ** 2 + (first_wp[2] - cz) ** 2) ** 0.5
                if dist < self.waypoint_tolerance * 0.5:
                    waypoints = waypoints[1:]

            self.current_waypoints = waypoints
            self.current_waypoint_index = 0
            self.waypoint_arrival_time = None
            self.get_logger().info(f"Starting waypoint navigation with {len(self.current_waypoints)} waypoints")
        else:
            self.get_logger().error("No valid waypoints generated")
            self.reject_current_assignment("planning_failed")
            self.current_waypoints = []

    def reject_current_assignment(self, reason):
        if not self.current_assignment_id:
            return
        self.rejected_assignment_ids.add(self.current_assignment_id)
        self.publish_assignment_rejection(self.current_assignment_id, self.current_cluster_id, reason)
        self.current_assignment = None
        self.current_assignment_id = None
        self.current_cluster_id = None
        self.current_cluster_poi = None
        self.cluster_poi_queue = []
        self.clear_waypoints()

    def publish_assignment_rejection(self, assignment_id, cluster_id, reason):
        msg = String()
        msg.data = json.dumps(
            {
                "assignment_id": assignment_id,
                "cluster_id": cluster_id,
                "photographer_id": self.gazebo_model_name,
                "reason": reason,
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
            }
        )
        self.assignment_rejection_publisher.publish(msg)

    def get_current_target_waypoint(self):
        if self.current_waypoints and 0 <= self.current_waypoint_index < len(self.current_waypoints):
            return self.current_waypoints[self.current_waypoint_index]
        return None

    def advance_to_next_waypoint(self):
        self.current_waypoint_index += 1
        self.waypoint_arrival_time = None
        if self.current_waypoint_index >= len(self.current_waypoints):
            self.get_logger().info("All waypoints completed!")
            return False
        wp = self.current_waypoints[self.current_waypoint_index]
        self.get_logger().info(
            f"Advancing to waypoint {self.current_waypoint_index + 1}/{len(self.current_waypoints)}: "
            f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})"
        )
        return True

    def is_at_waypoint(self, waypoint, tolerance=None):
        if tolerance is None:
            tolerance = self.waypoint_tolerance
        pos = self.vehicle_local_position
        return (
            abs(pos.x - waypoint[0]) < tolerance
            and abs(pos.y - waypoint[1]) < tolerance
            and abs(pos.z - waypoint[2]) < tolerance
        )

    def clear_waypoints(self):
        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_arrival_time = None
        self.current_target = None

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg
        if not self.local_frame_calibrated:
            self.local_offset_x = float(msg.x)
            self.local_offset_y = float(msg.y)
            self.local_offset_z = float(msg.z)
            self.local_frame_calibrated = True
            self.get_logger().info(
                f"Calibrated local offsets: "
                f"x0={self.local_offset_x:.2f}, "
                f"y0={self.local_offset_y:.2f}, "
                f"z0={self.local_offset_z:.2f}"
            )
            lx, ly, lz = float(msg.x), float(msg.y), float(msg.z)
            wx, wy, wz = self.local_to_world_coordinates(lx, ly, lz)
            rx, ry, rz = self.world_to_local_coordinates(wx, wy, wz)
            err = ((rx - lx) ** 2 + (ry - ly) ** 2 + (rz - lz) ** 2) ** 0.5
            sx_l, sy_l, sz_l = self.world_to_local_coordinates(self.spawn_x, self.spawn_y, 0.0)
            err0 = (
                (sx_l - self.local_offset_x) ** 2
                + (sy_l - self.local_offset_y) ** 2
                + (sz_l - self.local_offset_z) ** 2
            ) ** 0.5
            self.get_logger().info(f"Transform check: round-trip error={err:.3f}m, spawn offset error={err0:.3f}m")
            self.transform_checked = True
        if not self.data_received:
            self.data_received = True
            self.home_position = [0.0, 0.0, 0.0]
            self.get_logger().info(
                f"Vehicle data received. Home={self.home_position}, takeoff={self.takeoff_height}, "
                f"pos=[{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]"
            )

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def startup_readiness_callback(self, msg):
        was_ready = self.startup_ready
        self.startup_ready = bool(msg.data)
        if self.startup_ready and not was_ready:
            self.get_logger().info(f"Startup readiness satisfied via {self.startup_readiness_topic}")

    def startup_readiness_satisfied(self):
        return (not self.require_startup_readiness) or self.startup_ready

    def log_waiting_for_startup_readiness(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        if now_s - self._last_startup_readiness_log_time > 2.0:
            self.get_logger().info(f"Waiting for startup readiness on {self.startup_readiness_topic}")
            self._last_startup_readiness_log_time = now_s

    def assignment_callback(self, msg):
        data = json.loads(msg.data)
        if data.get("photographer_id", "") != self.gazebo_model_name:
            return

        assignment_type = data.get("assignment_type", "poi")

        if assignment_type == "cluster":
            assignment_id = data.get("assignment_id")
            cluster_data = data.get("cluster_data")
            cluster_id = data.get("cluster_id")

            if assignment_id in self.rejected_assignment_ids:
                return

            if (
                assignment_id is not None
                and cluster_data is not None
                and cluster_id is not None
                and cluster_id not in self.processed_cluster_ids
            ):
                if self.current_assignment_id == assignment_id:
                    return

                if self.flight_state not in {"INIT", "ARMING", "TAKEOFF", "WAITING_AT_HOME"}:
                    self.rejected_assignment_ids.add(assignment_id)
                    self.publish_assignment_rejection(assignment_id, cluster_id, f"busy_state_{self.flight_state}")
                    return

                if (
                    self.current_assignment_id is not None
                    and self.current_cluster_id is not None
                    and self.current_cluster_id != cluster_id
                ):
                    self.get_logger().warn(
                        f"Already busy with cluster {self.current_cluster_id}. Ignoring {cluster_id}."
                    )
                    return

                if self.current_cluster_id == cluster_id and self.current_assignment_id != assignment_id:
                    self.get_logger().warn(f"Duplicate assignment for cluster {cluster_id}. Ignoring.")
                    return

                self.current_assignment_astar_time = 0.0
                self.current_assignment = cluster_data
                self.current_assignment_id = assignment_id
                self.current_cluster_id = cluster_id
                self.publish_timing_event("assignment_start", assignment_id, cluster_id)

                cluster_pois = cluster_data.get("pois", [])
                if cluster_pois:
                    self.cluster_poi_queue = self.order_cluster_pois(cluster_pois)
                    self.current_cluster_poi = self.cluster_poi_queue.pop(0) if self.cluster_poi_queue else None
                    self.cluster_pois_completed = 0
                    self.clear_waypoints()
                    self.photo_capture_attempted = False
                    self.photo_capture_success = False
                    self.get_logger().info(
                        f"Cluster assignment {cluster_id}: {len(cluster_pois)} POIs via {self.poi_ordering_method}, "
                        f"starting with {self.current_cluster_poi['id']}"
                    )
            else:
                assignment_id = data.get("assignment_id")
                poi_data = data.get("poi_data")
                poi_id = data.get("poi_id")
                if assignment_id in self.rejected_assignment_ids:
                    return
                if (
                    assignment_id is not None
                    and poi_data is not None
                    and poi_id is not None
                    and poi_id not in self.processed_cluster_ids
                ):
                    if self.current_assignment_id == assignment_id:
                        return

                    if self.flight_state not in {"INIT", "ARMING", "TAKEOFF", "WAITING_AT_HOME"}:
                        self.rejected_assignment_ids.add(assignment_id)
                        self.publish_assignment_rejection(assignment_id, None, f"busy_state_{self.flight_state}")
                        return

                    if self.current_assignment_id is not None and self.current_assignment_id != assignment_id:
                        self.get_logger().warn(
                            f"Already busy with {self.current_assignment_id}. Ignoring {assignment_id}."
                        )
                        return

                    self.current_assignment_astar_time = 0.0
                    self.current_assignment = poi_data
                    self.current_assignment_id = assignment_id

                    poi_pos = poi_data.get("position", {})
                    lx, ly, lz = self.world_to_local_coordinates(
                        poi_pos.get("x", 0.0), poi_pos.get("y", 0.0), poi_pos.get("z", 0.0)
                    )
                    blx, bly, blz = self.apply_poi_buffer_local(lx, ly, lz)
                    self.current_assignment["local_target"] = [blx, bly, blz]

                    self.clear_waypoints()
                    self.photo_capture_attempted = False
                    self.photo_capture_success = False
                    self.get_logger().info(f"Received assignment for POI {poi_id} (assignment {assignment_id})")

    def los_status_callback(self, msg):
        data = json.loads(msg.data)
        our_status = data.get("photographer_status", {}).get(self.gazebo_model_name, {})
        previous_los = self.has_los_to_gcs
        self.has_los_to_gcs = our_status.get("has_los", False)
        self.last_los_check_time = self.get_clock().now()
        if previous_los != self.has_los_to_gcs:
            self.get_logger().info(
                f"LOS {'GAINED' if self.has_los_to_gcs else 'LOST'} "
                f"(distance: {our_status.get('distance_to_gcs', '?')}m)"
            )

    def arm_drone(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm command sent")

    def disarm_drone(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("Disarm command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def land_drone(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = self.sys_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def is_at_position(self, target_position, tolerance=1.0):
        pos = self.vehicle_local_position
        return (
            abs(pos.x - target_position[0]) < tolerance
            and abs(pos.y - target_position[1]) < tolerance
            and abs(pos.z - target_position[2]) < tolerance
        )

    def world_to_local_coordinates(self, world_x, world_y, world_z):
        spawn_yaw_rad = math.radians(-self.spawn_yaw_deg)
        rel_e = world_x - self.spawn_x
        rel_n = world_y - self.spawn_y
        cos_yaw = math.cos(spawn_yaw_rad)
        sin_yaw = math.sin(spawn_yaw_rad)
        local_x = rel_n * cos_yaw - rel_e * sin_yaw + self.local_offset_x
        local_y = rel_n * sin_yaw + rel_e * cos_yaw + self.local_offset_y
        local_z = -world_z + self.local_offset_z
        return local_x, local_y, local_z

    def local_to_world_coordinates(self, local_x, local_y, local_z):
        local_x = local_x - self.local_offset_x
        local_y = local_y - self.local_offset_y
        local_z = local_z - self.local_offset_z
        yaw_rad = math.radians(self.spawn_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        world_x = self.spawn_x + local_x * sin_yaw + local_y * cos_yaw
        world_y = self.spawn_y + local_x * cos_yaw - local_y * sin_yaw
        world_z = -local_z
        return world_x, world_y, world_z

    def apply_poi_buffer_local(self, local_x, local_y, local_z):
        buffered_x = local_x + self.poi_buffer_x
        buffered_y = local_y + self.poi_buffer_y
        buffered_z = local_z + self.poi_buffer_z
        return buffered_x, buffered_y, buffered_z

    def _enrich_poi_with_local_target(self, poi):
        poi_position = poi.get("position", {})
        original_x = poi_position.get("x", 0.0)
        original_y = poi_position.get("y", 0.0)
        original_z = poi_position.get("z", 0.0)

        local_x, local_y, local_z = self.world_to_local_coordinates(original_x, original_y, original_z)
        buffered_local_x, buffered_local_y, buffered_local_z = self.apply_poi_buffer_local(local_x, local_y, local_z)

        return {
            "id": poi.get("id", "unknown"),
            "position": poi_position,
            "local_target": [buffered_local_x, buffered_local_y, buffered_local_z],
        }

    def order_cluster_pois(self, cluster_pois):
        if not cluster_pois:
            return []

        # Enrich with local buffered targets up-front
        enriched = [self._enrich_poi_with_local_target(poi) for poi in cluster_pois]

        method = (self.poi_ordering_method or "fifo").lower()
        # Default sequencing method label for logs
        self.current_sequencing_method = (
            "FIFO" if method == "fifo" else ("TSP_networkX" if method == "tsp_networkx" else "TSP_greedy")
        )
        self.last_tsp_solve_time = None
        if method == "fifo":
            return enriched

        if method == "tsp_nearest":
            return self._order_pois_nearest(enriched)

        if method == "tsp_networkx":
            try:
                import networkx as nx
                import networkx.algorithms.approximation as nx_app
            except ImportError:
                self.get_logger().warn("networkx not installed; falling back to tsp_nearest")
                return self._order_pois_nearest(enriched)

            tsp_t0 = time.time()
            G = nx.Graph()
            num_pois = len(enriched)
            G.add_nodes_from(range(num_pois + 1))

            depot_x = float(self.vehicle_local_position.x)
            depot_y = float(self.vehicle_local_position.y)
            pos = {0: (depot_x, depot_y)}
            for i, poi in enumerate(enriched, start=1):
                pos[i] = (float(poi["local_target"][0]), float(poi["local_target"][1]))

            for i in range(num_pois + 1):
                for j in range(i + 1, num_pois + 1):
                    dist = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
                    G.add_edge(i, j, weight=dist)

            cycle = nx_app.christofides(G, weight="weight")

            ordered = []
            if 0 in cycle:
                first_zero_idx = cycle.index(0)
                k = (first_zero_idx + 1) % len(cycle)
                while cycle[k] != 0:
                    idx = cycle[k] - 1
                    if 0 <= idx < num_pois:
                        ordered.append(enriched[idx])
                    k = (k + 1) % len(cycle)
            else:
                for node_id in cycle:
                    idx = node_id - 1
                    if 0 <= idx < num_pois:
                        ordered.append(enriched[idx])

            if len(ordered) == num_pois:
                self.last_tsp_solve_time = max(0.0, time.time() - tsp_t0)
                return ordered
            self.get_logger().warn("tsp_networkx produced incomplete route; falling back to tsp_nearest")
            return self._order_pois_nearest(enriched)

        self.get_logger().warn(f"Unknown poi_ordering_method '{self.poi_ordering_method}', using FIFO")
        return enriched

    def _order_pois_nearest(self, enriched):
        unvisited = enriched.copy()
        ordered = []
        current_x = float(self.vehicle_local_position.x)
        current_y = float(self.vehicle_local_position.y)
        while unvisited:
            nearest_idx = min(
                range(len(unvisited)),
                key=lambda i: (unvisited[i]["local_target"][0] - current_x) ** 2
                + (unvisited[i]["local_target"][1] - current_y) ** 2,
            )
            next_poi = unvisited.pop(nearest_idx)
            ordered.append(next_poi)
            current_x, current_y = next_poi["local_target"][0], next_poi["local_target"][1]
        return ordered

    def complete_cluster_assignment(self):
        if self.current_assignment_id and self.current_cluster_id:
            self.processed_cluster_ids.add(self.current_cluster_id)

            complete_msg = String()
            complete_msg.data = json.dumps(
                {
                    "assignment_id": self.current_assignment_id,
                    "cluster_id": self.current_cluster_id,
                    "photographer_id": self.gazebo_model_name,
                    "coverage_success": True,
                    "pois_completed": self.cluster_pois_completed,
                }
            )
            self.assignment_complete_publisher.publish(complete_msg)

            self.get_logger().info(f"Cluster {self.current_cluster_id} completed ({self.cluster_pois_completed} POIs)")

            self.current_assignment = None
            self.current_assignment_id = None
            self.current_cluster_id = None
            self.cluster_coverage_state = "MOVE_TO_CENTER"
            self.coverage_start_time = None
            self.coverage_position_index = 0
            self.cluster_poi_queue = []
            self.current_cluster_poi = None
            self.cluster_pois_completed = 0
            # Reset event flags for the next assignment
            self.cluster_start_event_published = False
            self.cluster_complete_event_published = False
            self.clear_waypoints()

    def publish_position(self):
        pos = self.vehicle_local_position
        world_x, world_y, world_z = self.local_to_world_coordinates(pos.x, pos.y, pos.z)
        position_msg = String()
        position_msg.data = json.dumps(
            {
                "photographer_id": self.gazebo_model_name,
                "position": {"x": world_x, "y": world_y, "z": world_z},
                "timestamp": int(self.get_clock().now().nanoseconds / 1e9),
            }
        )
        self.position_publisher.publish(position_msg)

    def _current_poi_id(self):
        if self.current_cluster_poi:
            return self.current_cluster_poi["id"]
        if self.current_assignment:
            return self.current_assignment.get("id", "unknown")
        return "unknown"

    def start_photo_capture(self):
        """Fire off a photo capture request without blocking the control loop."""
        self._photo_future = None
        self._photo_request_time = self.get_clock().now().nanoseconds / 1e9
        if not self.photo_capture_client.service_is_ready():
            self.get_logger().warn("Photo capture service not ready; skipping capture")
            return

        request = CapturePhoto.Request()
        request.drone_id = self.drone_instance
        request.save_directory = ""
        poi_id = self._current_poi_id()
        request.filename_prefix = f"poi_{poi_id}"
        request.include_metadata = True

        self.get_logger().info(f"Capturing photo for POI {poi_id} (drone {self.drone_instance})")
        self._photo_future = self.photo_capture_client.call_async(request)

    def poll_photo_capture(self):
        """Resolve the in-flight capture (if any) without blocking. Returns True once settled."""
        now = self.get_clock().now().nanoseconds / 1e9

        if self._photo_future is not None and self._photo_future.done():
            response = self._photo_future.result()
            self._photo_future = None
            if response is not None and response.success:
                self.photo_capture_success = True
                self.photo_duration = 0.5
                self.get_logger().info(f"Photo captured: {response.image_path}")
            else:
                self.photo_capture_success = False
                self.photo_duration = 0.1
                msg = response.message if response is not None else "no response"
                self.get_logger().warn(f"Photo capture failed: {msg} - continuing")
            self.photo_start_time = self.get_clock().now()
            return True

        if self._photo_request_time is not None and (now - self._photo_request_time) > self.photo_capture_timeout_sec:
            if self._photo_future is not None:
                self._photo_future.cancel()
                self._photo_future = None
            self.photo_capture_success = False
            self.photo_duration = 0.1
            self.get_logger().warn(
                f"Photo capture did not respond within {self.photo_capture_timeout_sec:.1f}s - continuing"
            )
            self.photo_start_time = self.get_clock().now()
            return True

        return False

    def publish_timing_event(self, event_type, assignment_id, cluster_id):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if assignment_id not in self.mission_timings:
            self.mission_timings[assignment_id] = {
                "assignment_start": None,
                "cluster_start": None,
                "cluster_complete": None,
                "home_return": None,
            }
        timing_data = self.mission_timings[assignment_id]
        timing_data[event_type] = current_time

        assignment_start = timing_data.get("assignment_start")
        cluster_start = timing_data.get("cluster_start")
        cluster_complete = timing_data.get("cluster_complete")
        home_return = timing_data.get("home_return")

        time_to_cluster = (cluster_start - assignment_start) if assignment_start and cluster_start else None
        time_within_cluster = (cluster_complete - cluster_start) if cluster_start and cluster_complete else None

        time_to_return_home = None
        if event_type == "home_return" and cluster_complete:
            time_to_return_home = abs(current_time - cluster_complete)
        elif cluster_complete and home_return:
            time_to_return_home = abs(home_return - cluster_complete)

        if assignment_start and home_return:
            total_time = home_return - assignment_start
        elif assignment_start and event_type == "home_return":
            total_time = current_time - assignment_start
        else:
            total_time = None

        if event_type == "home_return" and not time_to_return_home:
            if total_time and time_to_cluster and time_within_cluster:
                time_to_return_home = max(0.0, total_time - (time_to_cluster + time_within_cluster))

        metrics = {
            "time_to_cluster": time_to_cluster,
            "time_within_cluster": time_within_cluster,
            "time_to_return_home": time_to_return_home,
            "total_time": total_time,
            "tsp_solve_time": self.last_tsp_solve_time,
            "astar_compute_time": self.current_assignment_astar_time,
        }

        timing_msg = String()
        timing_msg.data = json.dumps(
            {
                "photographer_id": self.gazebo_model_name,
                "assignment_id": assignment_id,
                "cluster_id": cluster_id,
                "timestamp": datetime.now().isoformat(),
                "is_complete": event_type == "home_return",
                "current_event": event_type,
                "metrics": metrics,
                "sequencing_method": self.current_sequencing_method,
                "raw_timestamps": {
                    "assignment_start": assignment_start,
                    "cluster_start": cluster_start,
                    "cluster_complete": cluster_complete,
                    "home_return": home_return,
                },
            }
        )
        self.timing_publisher.publish(timing_msg)

        # Also notify the coordinator on /photographer_timing_updates so it can advance
        # its own assignment lifecycle (and free this photographer on home_return)
        # instead of waiting for the assignment timeout.
        coordinator_update = String()
        coordinator_update.data = json.dumps(
            {
                "photographer_id": self.gazebo_model_name,
                "assignment_id": assignment_id,
                "cluster_id": cluster_id,
                "timing_event": event_type,
                "timestamp": current_time,
                "metrics": metrics,
                "sequencing_method": self.current_sequencing_method,
            }
        )
        self.timing_update_publisher.publish(coordinator_update)
        self.get_logger().info(f"Timing event: {event_type} for {cluster_id}")

        if event_type == "home_return":
            self.mission_timings.pop(assignment_id, None)

    def publish_heartbeat(self):
        if not (self.current_assignment_id and self.current_cluster_id):
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self._last_heartbeat_time < 5.0:
            return
        self._last_heartbeat_time = now_sec
        hb_msg = String()
        hb_msg.data = json.dumps(
            {
                "photographer_id": self.gazebo_model_name,
                "assignment_id": self.current_assignment_id,
                "cluster_id": self.current_cluster_id,
                "timing_event": "heartbeat",
                "timestamp": now_sec,
            }
        )
        self.timing_update_publisher.publish(hb_msg)

    def publish_takeoff_complete_event(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        event_msg = String()
        event_msg.data = json.dumps(
            {
                "photographer_id": self.gazebo_model_name,
                "timing_event": "takeoff_complete",
                "timestamp": now_sec,
            }
        )
        self.timing_update_publisher.publish(event_msg)
        self.get_logger().info(f"Published takeoff_complete event for {self.gazebo_model_name}")

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_position()
        self.publish_heartbeat()

        if not self.service_ready_logged and self.photo_capture_client.service_is_ready():
            self.get_logger().info("Photo capture service is available!")
            self.service_ready_logged = True
        elif not self.service_ready_logged and self.offboard_setpoint_counter % 100 == 0:
            self.get_logger().warn("Photo capture service not yet available...")

        if (
            self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            and self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and hasattr(self, "offboard_engaged")
            and self.offboard_setpoint_counter % 50 == 0
        ):
            self.engage_offboard_mode()

        if self.flight_state == "INIT":
            self.publish_trajectory_setpoint(
                self.home_position[0] if self.data_received else 0.0,
                self.home_position[1] if self.data_received else 0.0,
                self.takeoff_height,
            )

            if not self.data_received:
                if self.offboard_setpoint_counter % 50 == 0:
                    self.get_logger().warn("Waiting for vehicle data...")
                return

            if not self.startup_readiness_satisfied():
                self.log_waiting_for_startup_readiness()
                return

            self.get_logger().info("Initializing A* photographer drone...")
            if self.offboard_setpoint_counter == 10:
                self.arm_drone()
                self.flight_state = "ARMING"

        elif self.flight_state == "ARMING":
            self.publish_trajectory_setpoint(self.home_position[0], self.home_position[1], self.takeoff_height)

            if self.offboard_setpoint_counter % 10 == 0:
                arming_state = self.vehicle_status.arming_state
                nav_state = self.vehicle_status.nav_state
                self.get_logger().info(f"ARMING - arming_state: {arming_state}, nav_state: {nav_state}")

            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED and not hasattr(
                self, "offboard_engaged"
            ):
                self.get_logger().info("Vehicle armed - engaging offboard mode")
                self.engage_offboard_mode()
                self.offboard_engaged = True

            if (
                self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and hasattr(self, "offboard_engaged")
                and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):
                self.get_logger().info("Vehicle armed and offboard engaged")
                self.flight_state = "TAKEOFF"
            elif self.offboard_setpoint_counter > 300:
                self.get_logger().warn("Still not armed after 30 seconds - retrying...")
                self.arm_drone()
                self.offboard_setpoint_counter = 0

        elif self.flight_state == "TAKEOFF":
            self.publish_trajectory_setpoint(self.home_position[0], self.home_position[1], self.takeoff_height)

            if self.offboard_setpoint_counter % 10 == 0:
                current_z = self.vehicle_local_position.z
                self.get_logger().info(f"TAKEOFF - Current Z: {current_z:.2f}, Target: {self.takeoff_height:.2f}")

            if (
                self.vehicle_local_position.z < self.takeoff_height + 0.95
                and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):
                self.get_logger().info("Takeoff complete - waiting at home for assignments")
                self.flight_state = "WAITING_AT_HOME"
                self.mission_start_time = self.get_clock().now()

                # Publish takeoff_complete event to start mission timer
                if not self.takeoff_complete_published:
                    self.takeoff_complete_time = self.get_clock().now().nanoseconds / 1e9
                    self.publish_takeoff_complete_event()
                    self.takeoff_complete_published = True
            elif self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().warn("Vehicle disarmed during takeoff! Returning to ARMING...")
                self.offboard_engaged = False
                self.flight_state = "ARMING"
                self.offboard_setpoint_counter = 0

        elif self.flight_state == "WAITING_AT_HOME":
            # Stay at home position
            self.publish_trajectory_setpoint(self.home_position[0], self.home_position[1], self.takeoff_height)

            if self.offboard_setpoint_counter % 50 == 0:
                self.get_logger().info(f"WAITING - LOS: {self.has_los_to_gcs}")

            # Check for new assignment
            if self.current_assignment is not None:
                if self.current_cluster_id and self.current_cluster_poi:
                    # Handle cluster POI assignment
                    poi_id = self.current_cluster_poi["id"]

                    self.get_logger().info(f"Starting navigation to POI {poi_id}")
                    self.flight_state = "NAVIGATING_TO_POI"  # State transition first
                else:
                    # Handle legacy POI assignment
                    poi_id = self.current_assignment.get("id", "unknown")
                    self.get_logger().info(f"Starting navigation to POI {poi_id}")
                    self.flight_state = "NAVIGATING_TO_POI"  # State transition first

        elif self.flight_state == "NAVIGATING_TO_POI":
            # Start navigation logic if not already started
            if not self.current_waypoints:
                if self.current_cluster_poi:
                    target_local = self.current_cluster_poi["local_target"]
                else:
                    target_local = self.current_assignment["local_target"]
                self.start_navigation_to_target(target_local)

            current_waypoint = self.get_current_target_waypoint()

            if current_waypoint is None:
                self.get_logger().error("No current waypoint - returning home")
                self.flight_state = "RETURN_HOME"
                return

            # Navigate to current waypoint
            self.publish_trajectory_setpoint(current_waypoint[0], current_waypoint[1], current_waypoint[2])

            # Log progress periodically
            if self.offboard_setpoint_counter % 20 == 0:
                current_pos = self.vehicle_local_position
                self.get_logger().info(
                    f"NAVIGATING - Waypoint {self.current_waypoint_index + 1}/"
                    f"{len(self.current_waypoints)}: Current: ({current_pos.x:.1f}, "
                    f"{current_pos.y:.1f}, {current_pos.z:.1f}), Target: "
                    f"({current_waypoint[0]:.1f}, {current_waypoint[1]:.1f}, {current_waypoint[2]:.1f})"
                )

            # Check if we've reached the current waypoint
            if self.is_at_waypoint(current_waypoint):
                if self.waypoint_arrival_time is None:
                    self.waypoint_arrival_time = self.get_clock().now()
                    self.get_logger().info(f"Reached waypoint {self.current_waypoint_index + 1}")

                # Wait briefly at waypoint for stability
                elapsed = (self.get_clock().now() - self.waypoint_arrival_time).nanoseconds / 1e9
                if elapsed > 0.1:  # Wait 0.1 seconds at each waypoint
                    # Advance to next waypoint
                    if self.advance_to_next_waypoint():
                        # More waypoints available - continue navigation
                        pass
                    else:
                        # All waypoints completed - arrived at POI
                        self.get_logger().info("Navigation complete - arrived at POI!")
                        # Mark cluster_start at actual arrival to the first POI/cluster
                        if not getattr(self, "cluster_start_event_published", False):
                            if self.current_assignment_id and self.current_cluster_id:
                                self.publish_timing_event(
                                    "cluster_start", self.current_assignment_id, self.current_cluster_id
                                )
                            self.cluster_start_event_published = True
                        self.flight_state = "TAKING_PHOTO"
                        self.photo_start_time = self.get_clock().now()
                        # Do not reset cluster_start_event_published here;
                        # It will be cleared once the full cluster is completed

        elif self.flight_state == "TAKING_PHOTO":
            # Stay at POI position for photo
            current_waypoint = self.get_current_target_waypoint()
            if current_waypoint is None:
                # Use the last waypoint or fallback
                if self.current_waypoints:
                    current_waypoint = self.current_waypoints[-1]
                else:
                    # Fallback to current position
                    current_waypoint = [
                        self.vehicle_local_position.x,
                        self.vehicle_local_position.y,
                        self.vehicle_local_position.z,
                    ]

            self.publish_trajectory_setpoint(current_waypoint[0], current_waypoint[1], current_waypoint[2])

            # Take photo when first entering this state (non-blocking request)
            if not self.photo_capture_attempted:
                self.get_logger().info(f"Taking photo at POI {self._current_poi_id()}")
                self.start_photo_capture()
                self.photo_capture_attempted = True
                self.photo_duration = None

            # Wait (without blocking the control loop) for the capture to settle
            if self.photo_duration is None:
                self.poll_photo_capture()

            # Check if photo duration complete
            if (
                (self.photo_duration is not None)
                and (self.photo_start_time is not None)
                and ((self.get_clock().now() - self.photo_start_time).nanoseconds / 1e9 > self.photo_duration)
            ):

                if self.current_cluster_poi:
                    # Cluster POI handling
                    poi_id = self.current_cluster_poi["id"]
                    self.get_logger().info(f"Photo operation complete for POI {poi_id}")
                    self.cluster_pois_completed += 1

                    # Check if more POIs in cluster
                    if self.cluster_poi_queue:
                        # Move to next POI in cluster
                        self.current_cluster_poi = self.cluster_poi_queue.pop(0)
                        next_poi_id = self.current_cluster_poi["id"]

                        self.get_logger().info(f"Moving to next POI: {next_poi_id}")

                        # Reset navigation state for the new POI
                        self.clear_waypoints()

                        self.photo_capture_attempted = False
                        self.photo_capture_success = False
                        self.flight_state = "NAVIGATING_TO_POI"
                    else:
                        # All POIs in cluster completed
                        self.get_logger().info(f"All {self.cluster_pois_completed} POIs completed")
                        self.flight_state = "RETURN_HOME"
                else:
                    # Legacy POI handling
                    poi_id = self.current_assignment.get("id", "unknown")
                    self.get_logger().info(f"Photo complete for POI {poi_id} - returning home")
                    self.flight_state = "RETURN_HOME"

        elif self.flight_state == "RETURN_HOME":
            if not getattr(self, "cluster_complete_event_published", False):
                if self.current_assignment_id and self.current_cluster_id:
                    self.publish_timing_event("cluster_complete", self.current_assignment_id, self.current_cluster_id)
                    self.last_assignment_id_for_return = self.current_assignment_id
                    self.last_cluster_id_for_return = self.current_cluster_id
                    self.complete_cluster_assignment()
                self.cluster_complete_event_published = True

            home_target = [self.home_position[0], self.home_position[1], self.takeoff_height]

            if not self.current_waypoints:
                if self.use_astar_planning and self.path_planner:
                    self.start_navigation_to_target(home_target)
                else:
                    current_pos = [
                        self.vehicle_local_position.x,
                        self.vehicle_local_position.y,
                        self.vehicle_local_position.z,
                    ]
                    self.current_waypoints = [current_pos, home_target]
                    self.current_waypoint_index = 0
                    self.waypoint_arrival_time = None

            current_waypoint = self.get_current_target_waypoint()
            target_pos = current_waypoint if current_waypoint is not None else home_target
            self.publish_trajectory_setpoint(target_pos[0], target_pos[1], target_pos[2])

            arrived_home_axis = self.is_at_position(home_target, tolerance=self.waypoint_tolerance * 1.5)
            dx = self.vehicle_local_position.x - home_target[0]
            dy = self.vehicle_local_position.y - home_target[1]
            dz = self.vehicle_local_position.z - home_target[2]
            arrived_home = arrived_home_axis or ((dx * dx + dy * dy + dz * dz) ** 0.5 < self.home_arrival_radius)

            arrived_waypoint = current_waypoint and self.is_at_waypoint(current_waypoint)

            if arrived_home or (arrived_waypoint and self.current_waypoint_index >= len(self.current_waypoints) - 1):
                self.get_logger().info("Arrived at home position.")
                assignment_id_for_return = self.current_assignment_id or getattr(
                    self, "last_assignment_id_for_return", None
                )
                cluster_id_for_return = self.current_cluster_id or getattr(self, "last_cluster_id_for_return", None)

                if assignment_id_for_return and cluster_id_for_return:
                    self.publish_timing_event("home_return", assignment_id_for_return, cluster_id_for_return)
                    if hasattr(self, "last_assignment_id_for_return"):
                        del self.last_assignment_id_for_return
                        del self.last_cluster_id_for_return

                self.clear_waypoints()
                self.cluster_complete_event_published = False
                self.flight_state = "WAITING_AT_HOME"
                return

            if arrived_waypoint:
                if self.waypoint_arrival_time is None:
                    self.waypoint_arrival_time = self.get_clock().now()
                elapsed = (self.get_clock().now() - self.waypoint_arrival_time).nanoseconds / 1e9
                if elapsed > 0.5:
                    self.advance_to_next_waypoint()

        elif self.flight_state == "RETURN_TO_LAND":
            home_target = [self.home_position[0], self.home_position[1], self.takeoff_height]
            self.publish_trajectory_setpoint(self.home_position[0], self.home_position[1], self.takeoff_height)
            if self.is_at_position(home_target):
                self.get_logger().info("Returned home - landing")
                self.flight_state = "LAND"

        elif self.flight_state == "LAND":
            self.land_drone()
            self.flight_state = "LANDED"

        elif self.flight_state == "LANDED":
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                if self.vehicle_local_position.z > -0.3:
                    self.get_logger().info("Touchdown - disarming")
                    self.disarm_drone()
                    self.flight_state = "DISARMED"

        elif self.flight_state == "DISARMED":
            if self.offboard_setpoint_counter > 50:
                self.get_logger().info("Mission complete - exiting")
                exit()

        self.offboard_setpoint_counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = VehicleControllerAStar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
