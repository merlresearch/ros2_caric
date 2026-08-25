#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String

from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .assignment_core import brute_force_tsp_for_poi_enu, current_position_enu
from .config.phase_params import build_common_problem_config, declare_phase_parameters, read_phase_parameters
from .controller import logging as clog
from .controller import offboard as offb
from .controller import setpoints as sp
from .phases.common import get_comm_center_for_world, get_obstacles_for_world
from .phases.common import local_to_world as common_local_to_world
from .phases.common import world_to_local as common_world_to_local
from .phases.geometry_utils import grid_to_aggregated_boxes
from .states.dispatch import dispatch_state
from .utils.ros2_setup import (
    create_qos_profile,
    declare_parameters,
    read_parameters,
    setup_publishers,
    setup_subscribers,
)

HOME_FOLDER = os.path.expanduser("~")


class VehicleControllerOpenSCvx(Node):
    def __init__(self):
        super().__init__("vehicle_controller_openscvx")

        declare_parameters(self)
        read_parameters(self)
        declare_phase_parameters(self)
        read_phase_parameters(self)
        qos_profile = create_qos_profile()
        setup_publishers(self, qos_profile)
        setup_subscribers(self, qos_profile)

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.photographer_id = f"x500_gimbal_photographer_{self.drone_instance}"
        self.takeoff_height = -5.0
        self.offboard_setpoint_counter = 0

        self.takeoff_start_time = None
        self.takeoff_complete_time = None
        self.hover_start_time = None
        self.landing_start_time = None
        self.landing_complete_time = None
        self.mission_start_time = None

        self.target_world_position = [self.spawn_x, self.spawn_y, -self.takeoff_height]
        local_x, local_y, local_z = common_world_to_local(
            self.spawn_x,
            self.spawn_y,
            self.spawn_yaw_deg,
            self.target_world_position[0],
            self.target_world_position[1],
            self.target_world_position[2],
        )
        self.target_local_position = [local_x, local_y, local_z]

        self.optimal_trajectory = None
        self.trajectory_start_time = None
        self.end_trajectory_index = {"phase1": 0, "phase2": 0, "phase3": 0}

        if self.world_model_source == "known":
            self.obstacle_boxes = get_obstacles_for_world(self.world)
            self.map_ready = True
            self.get_logger().info(
                f"Known mode: loaded {len(self.obstacle_boxes)} obstacle boxes for world '{self.world}'"
            )
        else:
            self.obstacle_boxes = []
            self.map_ready = False
            self.get_logger().info("Lidar-only mode: waiting for mapping node to provide obstacles")

        self.dynamic_obstacles: List[Dict[str, Any]] = []
        self._obstacles_frozen: bool = False
        self.map_grid: Optional[np.ndarray] = None
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0
        self.map_width_cells: int = 0
        self.map_height_cells: int = 0

        self.common_problem_config = build_common_problem_config({"comm_center": get_comm_center_for_world(self.world)})

        self._phase2_ordered_pois_list_enu = []
        self._buffered_poi_list_enu = None
        self.assignment_received = False

        self.sent_setpoints_enu = []
        self.actual_positions_enu = []
        self.sent_traj_saved = False
        results_root = os.path.expanduser(
            os.environ.get(
                "MISSION_RESULTS_ROOT",
                f"{HOME_FOLDER:s}/ros2_ws/src/mission_manager/results",
            )
        )
        result_group = os.environ.get("MISSION_RESULTS_GROUP", "").strip()
        if result_group:
            self.trajectory_save_dir = os.path.join(results_root, "openscvx", result_group, "trajectory")
        else:
            self.trajectory_save_dir = os.path.join(results_root, "openscvx", "trajectory")

        self.phase2_started, self.phase2_completed = False, False

        self.flight_state = "INIT"
        self.prestream_ticks = 0
        self.prestream_offboard_engaged = False
        self.prestream_arm_sent = False
        self._last_hold_log_time = 0.0
        self._last_startup_readiness_log_time = 0.0
        self._last_takeoff_watchdog_log_time = 0.0
        self.last_arm_cmd_time = 0.0
        self.arm_attempts = 0

        self.timer = self.create_timer(0.01, self.timer_callback)
        self.get_logger().info("OpenSCvx Vehicle Controller Node Started")
        self.publish_position()

    @property
    def buffered_poi_list_enu(self):
        return np.array(self._buffered_poi_list_enu, dtype=float) if self._buffered_poi_list_enu is not None else []

    @property
    def phase2_ordered_pois_list_enu(self):
        return self._phase2_ordered_pois_list_enu

    @phase2_ordered_pois_list_enu.setter
    def phase2_ordered_pois_list_enu(self, value):
        self._phase2_ordered_pois_list_enu = value
        self._buffered_poi_list_enu = None  # Reset buffered POIs when ordered POIs are updated

    def publish_timing_event(self, event_type, additional_data=None):
        return clog.publish_timing_event(self, event_type, additional_data)

    def publish_ready_for_assignment(self):
        msg = String()
        msg.data = json.dumps(
            {"photographer_id": self.photographer_id, "timestamp": self.get_clock().now().nanoseconds / 1e9}
        )
        self.ready_publisher.publish(msg)
        self.get_logger().info(f"Published ready signal for {self.photographer_id}")

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

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

    def map_occupancy_callback(self, msg: OccupancyGrid) -> None:
        if self._obstacles_frozen or self.world_model_source != "lidar_only":
            return

        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_width_cells = msg.info.width
        self.map_height_cells = msg.info.height

        data = np.asarray(msg.data, dtype=np.int8)
        if data.size != self.map_width_cells * self.map_height_cells:
            self.get_logger().warn("Received OccupancyGrid with unexpected data size; ignoring.")
            return
        self.map_grid = data.reshape((self.map_height_cells, self.map_width_cells))

        self.dynamic_obstacles = grid_to_aggregated_boxes(
            grid=self.map_grid,
            origin_x=self.map_origin_x,
            origin_y=self.map_origin_y,
            resolution=self.map_resolution,
            z_min=self.map_z_min,
            z_max=self.map_z_max,
            treat_unknown_as_obstacle=False,
            min_box_area=3.0,
            max_aggregate_boxes=5,
            downsample_factor=2,
            use_dbscan=True,
            dbscan_eps=5.0,
            dbscan_min_samples=2,
        )

    def map_meta_callback(self, msg: String) -> None:
        if self.world_model_source != "lidar_only":
            return
        meta = json.loads(msg.data)
        if "z_min" in meta:
            self.map_z_min = float(meta["z_min"])
        if "z_max" in meta:
            self.map_z_max = float(meta["z_max"])
        if "resolution" in meta:
            self.map_resolution = float(meta["resolution"])

    def map_status_callback(self, msg: String) -> None:
        if self._obstacles_frozen or self.world_model_source != "lidar_only":
            return
        status = json.loads(msg.data)
        known_ratio = float(status.get("known_ratio", 0.5))
        stable_seconds = float(status.get("stable_seconds", 0.0))

        if not hasattr(self, "_map_status_log_count"):
            self._map_status_log_count = 0
        self._map_status_log_count += 1
        if self._map_status_log_count % 50 == 1:
            self.get_logger().info(
                f"Map status: known_ratio={known_ratio:.2f}, obstacles={len(self.dynamic_obstacles)}, "
                f"required={self.map_min_known_ratio:.2f}"
            )

        ready = (
            known_ratio >= self.map_min_known_ratio
            and stable_seconds >= self.map_min_stable_seconds
            and len(self.dynamic_obstacles) > 0
        )

        if ready and not self.map_ready:
            self._obstacles_frozen = True
            self.obstacle_boxes = self.dynamic_obstacles.copy()
            self.map_ready = True
            self.get_logger().info(
                f"Map ready! known_ratio={known_ratio:.2f}, stable={stable_seconds:.1f}s, "
                f"FROZEN {len(self.obstacle_boxes)} obstacle boxes."
            )
            if self.flight_state in ("WAITING_FOR_ASSIGNMENT", "HOVER_AFTER_TAKEOFF"):
                self.publish_ready_for_assignment()

    def _order_cluster_pois_tsp(self, pois_list):
        return brute_force_tsp_for_poi_enu(pois_list, start_position_enu=current_position_enu(self))

    def arm(self):
        offb.arm(self)

    def disarm(self):
        offb.disarm(self)

    def engage_offboard_mode(self):
        offb.engage_offboard_mode(self)

    def land(self):
        offb.land(self)

    def publish_vehicle_command(self, command, **params):
        offb.publish_vehicle_command(self, command, **params)

    def publish_offboard_control_mode(self):
        offb.publish_offboard_control_mode(self)

    def publish_setpoint_enu(self, pos_enu, vel_enu, acc_enu, yaw, yawspeed):
        return sp.publish_setpoint_enu(self, pos_enu, vel_enu, acc_enu, yaw, yawspeed)

    def publish_basic_trajectory_setpoint(self, x=0.0, y=0.0, z=0.0):
        return sp.publish_basic_trajectory_setpoint(self, x, y, z)

    def save_sent_trajectory_artifacts(self, reason_suffix: str = ""):
        return clog.save_sent_trajectory_artifacts(self, reason_suffix)

    def timer_callback(self):

        self.publish_offboard_control_mode()
        self.offboard_setpoint_counter += 1
        if self.offboard_setpoint_counter % 10 == 0:
            self.publish_position()

        if (
            self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            and self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and self.offboard_setpoint_counter % 50 == 0
        ):
            self.engage_offboard_mode()

        if self.offboard_setpoint_counter == 50:
            self.offboard_setpoint_counter = 0

        dispatch_state(self)

    def publish_position(self):
        pos = self.vehicle_local_position
        world_enu = common_local_to_world(
            float(self.spawn_x),
            float(self.spawn_y),
            float(self.spawn_yaw_deg),
            float(pos.x),
            float(pos.y),
            float(pos.z),
        )
        msg = String()
        msg.data = json.dumps(
            {
                "photographer_id": self.photographer_id,
                "position": {"x": world_enu[0], "y": world_enu[1], "z": world_enu[2]},
                "timestamp": int(self.get_clock().now().nanoseconds / 1e9),
            }
        )
        self.photographer_position_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    vehicle_controller = VehicleControllerOpenSCvx()
    try:
        rclpy.spin(vehicle_controller)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            vehicle_controller.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
