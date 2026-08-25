#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

EXPLORER_SCAN_YAW = 0.0


class VehicleControllerExplorer(Node):
    def __init__(self):
        super().__init__("vehicle_controller_explorer")

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.drone1_sys_id = 2
        self.drone1_topic_prefix = "/px4_1"
        self.drone2_sys_id = 4
        self.drone2_topic_prefix = "/px4_3"

        self.drone1_vehicle_command_publisher = self.create_publisher(
            VehicleCommand, f"{self.drone1_topic_prefix}/fmu/in/vehicle_command", reliable_qos
        )
        self.drone1_offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, f"{self.drone1_topic_prefix}/fmu/in/offboard_control_mode", reliable_qos
        )
        self.drone1_trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, f"{self.drone1_topic_prefix}/fmu/in/trajectory_setpoint", reliable_qos
        )
        self.drone2_vehicle_command_publisher = self.create_publisher(
            VehicleCommand, f"{self.drone2_topic_prefix}/fmu/in/vehicle_command", reliable_qos
        )
        self.drone2_offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, f"{self.drone2_topic_prefix}/fmu/in/offboard_control_mode", reliable_qos
        )
        self.drone2_trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, f"{self.drone2_topic_prefix}/fmu/in/trajectory_setpoint", reliable_qos
        )

        self.drone1_vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            f"{self.drone1_topic_prefix}/fmu/out/vehicle_local_position",
            self.drone1_vehicle_local_position_callback,
            best_effort_qos,
        )
        self.drone1_vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            f"{self.drone1_topic_prefix}/fmu/out/vehicle_status",
            self.drone1_vehicle_status_callback,
            best_effort_qos,
        )
        self.drone1_vehicle_status_subscriber_v1 = self.create_subscription(
            VehicleStatus,
            f"{self.drone1_topic_prefix}/fmu/out/vehicle_status_v1",
            self.drone1_vehicle_status_callback,
            best_effort_qos,
        )
        self.drone2_vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            f"{self.drone2_topic_prefix}/fmu/out/vehicle_local_position",
            self.drone2_vehicle_local_position_callback,
            best_effort_qos,
        )
        self.drone2_vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            f"{self.drone2_topic_prefix}/fmu/out/vehicle_status",
            self.drone2_vehicle_status_callback,
            best_effort_qos,
        )
        self.drone2_vehicle_status_subscriber_v1 = self.create_subscription(
            VehicleStatus,
            f"{self.drone2_topic_prefix}/fmu/out/vehicle_status_v1",
            self.drone2_vehicle_status_callback,
            best_effort_qos,
        )

        self.timing_publisher = self.create_publisher(String, "/explorer_timing_updates", reliable_qos)
        self.takeoff_complete_published = False
        self.get_logger().info("Starting explorer controller")
        self.drone1_data_received = False
        self.drone2_data_received = False

        self.declare_parameter("explorer_names", ["jurong", "raffles"])
        self.explorer_names = list(self.get_parameter("explorer_names").get_parameter_value().string_array_value)

        self.declare_parameter("world", "mbs")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("require_startup_readiness", False)
        self.declare_parameter("startup_readiness_topic", "/fleet_ready")
        self.world_key = self.get_parameter("world").get_parameter_value().string_value
        self.difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        self.require_startup_readiness = bool(self.get_parameter("require_startup_readiness").value)
        self.startup_readiness_topic = (
            self.get_parameter("startup_readiness_topic").get_parameter_value().string_value or "/fleet_ready"
        )
        self.startup_ready = not self.require_startup_readiness
        self._last_startup_readiness_log_time = 0.0
        if self.require_startup_readiness:
            self.startup_readiness_subscriber = self.create_subscription(
                Bool,
                self.startup_readiness_topic,
                self.startup_readiness_callback,
                latched_qos,
            )

        self.drone1_vehicle_local_position = VehicleLocalPosition()
        self.drone1_vehicle_status = VehicleStatus()
        self.drone2_vehicle_local_position = VehicleLocalPosition()
        self.drone2_vehicle_status = VehicleStatus()

        self.offboard_setpoint_counter = 0

        self.bbox_center = [41.5112126875, 0.2068166775, 36.59351567]
        self.bbox_size = [115.09100342, 25.26037979, 68.91789896]
        self.bbox_x_min = self.bbox_center[0] - self.bbox_size[0] / 2
        self.bbox_x_max = self.bbox_center[0] + self.bbox_size[0] / 2
        self.bbox_y_min = self.bbox_center[1] - self.bbox_size[1] / 2
        self.bbox_y_max = self.bbox_center[1] + self.bbox_size[1] / 2

        self.safety_gap = 10.0
        self.takeoff_hover_counter = 0
        self.takeoff_hover_time = 50
        self._load_explorer_patterns()

        self.drone1_current_waypoint_index = 0
        self.drone1_waypoints = []
        self.drone2_current_waypoint_index = 0
        self.drone2_waypoints = []

        self.generate_drone1_waypoints()
        self.generate_drone2_waypoints()
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.flight_state = "INIT"

    def _load_explorer_patterns(self):
        default_cfg = {
            "drone1": {"x_start": -10.0, "y_start": 0.0},
            "drone2": {"x_start": 0.0, "y_start": 0.0},
            "scan": {"x_decrement": 3.5, "y_increment": 10.0, "total_runs": 9},
            "altitudes": {"high_altitude": -65.0, "low_altitude": -4.0, "takeoff_height": -5.0},
        }
        cfg = default_cfg
        share_dir = get_package_share_directory("mission_manager")
        path = os.path.join(share_dir, "config", "explorer_patterns.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read().replace("\t", "  ")
            all_cfg = yaml.safe_load(content) or {}
            world_cfg = all_cfg.get(self.world_key)
            if not world_cfg and self.world_key == "mbs":
                world_cfg = all_cfg.get("default")
            world_cfg = world_cfg or {}
            cfg = {
                "drone1": {
                    **default_cfg["drone1"],
                    **(world_cfg.get("drone1") or {}),
                },
                "drone2": {
                    **default_cfg["drone2"],
                    **(world_cfg.get("drone2") or {}),
                },
                "scan": {
                    **default_cfg["scan"],
                    **(world_cfg.get("scan") or {}),
                },
                "altitudes": {
                    **default_cfg["altitudes"],
                    **(world_cfg.get("altitudes") or {}),
                },
            }
            diffs = world_cfg.get("difficulty_overrides") or {}
            if isinstance(diffs, dict) and self.difficulty in diffs:
                diff_cfg = diffs[self.difficulty] or {}
                cfg["scan"] = {**cfg["scan"], **(diff_cfg.get("scan") or {})}
                cfg["altitudes"] = {**cfg["altitudes"], **(diff_cfg.get("altitudes") or {})}

        self.drone1_flight_x_start = float(cfg["drone1"]["x_start"])
        self.drone1_flight_y_start = float(cfg["drone1"]["y_start"])
        self.drone2_flight_x_start = float(cfg["drone2"]["x_start"])
        self.drone2_flight_y_start = float(cfg["drone2"]["y_start"])
        self.x_decrement = float(cfg["scan"]["x_decrement"])
        self.y_increment = float(cfg["scan"]["y_increment"])
        self.total_runs = int(cfg["scan"]["total_runs"])
        self.high_altitude = float(cfg["altitudes"]["high_altitude"])
        self.low_altitude = float(cfg["altitudes"]["low_altitude"])
        self.takeoff_height = float(cfg["altitudes"]["takeoff_height"])

    def generate_drone1_waypoints(self):
        self.drone1_waypoints = [[self.drone1_flight_x_start, self.drone1_flight_y_start, self.takeoff_height]]
        for run in range(self.total_runs):
            x_pos = self.drone1_flight_x_start - (run * self.x_decrement)
            y_pos = self.drone1_flight_y_start + ((run + 1) * self.y_increment)
            altitude = self.high_altitude if (run % 2 == 0) else self.low_altitude
            self.drone1_waypoints.append([x_pos, y_pos, altitude])

    def generate_drone2_waypoints(self):
        self.drone2_waypoints = [[self.drone2_flight_x_start, self.drone2_flight_y_start, self.takeoff_height]]
        for run in range(self.total_runs):
            x_pos = self.drone2_flight_x_start + (run * self.x_decrement)
            y_pos = self.drone2_flight_y_start - ((run + 1) * self.y_increment)
            altitude = self.high_altitude if (run % 2 == 0) else self.low_altitude
            self.drone2_waypoints.append([x_pos, y_pos, altitude])

    def drone1_vehicle_local_position_callback(self, msg):
        self.drone1_vehicle_local_position = msg
        if not self.drone1_data_received:
            self.drone1_data_received = True
            self.get_logger().info("Drone 1 position data received")

    def drone1_vehicle_status_callback(self, msg):
        self.drone1_vehicle_status = msg

    def drone2_vehicle_local_position_callback(self, msg):
        self.drone2_vehicle_local_position = msg
        if not self.drone2_data_received:
            self.drone2_data_received = True
            self.get_logger().info("Drone 2 position data received")

    def drone2_vehicle_status_callback(self, msg):
        self.drone2_vehicle_status = msg

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

    def arm_drone(self, drone_id):
        if drone_id == 1:
            self.publish_vehicle_command_drone1(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            self.get_logger().info("Drone 1 arm command sent")
        elif drone_id == 2:
            self.publish_vehicle_command_drone2(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            self.get_logger().info("Drone 2 arm command sent")

    def disarm_drone(self, drone_id):
        if drone_id == 1:
            self.publish_vehicle_command_drone1(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
            self.get_logger().info("Drone 1 disarm command sent")
        elif drone_id == 2:
            self.publish_vehicle_command_drone2(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
            self.get_logger().info("Drone 2 disarm command sent")

    def engage_offboard_mode_drone(self, drone_id):
        if drone_id == 1:
            self.publish_vehicle_command_drone1(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.get_logger().info("Drone 1 switching to offboard mode")
        elif drone_id == 2:
            self.publish_vehicle_command_drone2(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.get_logger().info("Drone 2 switching to offboard mode")

    def land_drone(self, drone_id):
        if drone_id == 1:
            self.publish_vehicle_command_drone1(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("Drone 1 switching to land mode")
        elif drone_id == 2:
            self.publish_vehicle_command_drone2(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("Drone 2 switching to land mode")

    def _publish_vehicle_command(self, sys_id, publisher, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = sys_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        publisher.publish(msg)

    def publish_vehicle_command_drone1(self, command, **params):
        self._publish_vehicle_command(self.drone1_sys_id, self.drone1_vehicle_command_publisher, command, **params)

    def publish_vehicle_command_drone2(self, command, **params):
        self._publish_vehicle_command(self.drone2_sys_id, self.drone2_vehicle_command_publisher, command, **params)

    def _publish_offboard_control_mode(self, publisher):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        publisher.publish(msg)

    def publish_offboard_control_mode_drone1(self):
        self._publish_offboard_control_mode(self.drone1_offboard_control_mode_publisher)

    def publish_offboard_control_mode_drone2(self):
        self._publish_offboard_control_mode(self.drone2_offboard_control_mode_publisher)

    def _publish_trajectory_setpoint(self, publisher, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        publisher.publish(msg)

    def publish_trajectory_setpoint_drone1(self, x=0.0, y=0.0, z=0.0, yaw=EXPLORER_SCAN_YAW):
        self._publish_trajectory_setpoint(self.drone1_trajectory_setpoint_publisher, x, y, z, yaw)

    def publish_trajectory_setpoint_drone2(self, x=0.0, y=0.0, z=0.0, yaw=EXPLORER_SCAN_YAW):
        self._publish_trajectory_setpoint(self.drone2_trajectory_setpoint_publisher, x, y, z, yaw)

    def _is_at_waypoint(self, position, waypoint, tolerance=2.0):
        return (
            abs(position.x - waypoint[0]) < tolerance
            and abs(position.y - waypoint[1]) < tolerance
            and abs(position.z - waypoint[2]) < tolerance
        )

    def is_at_waypoint_drone1(self, waypoint, tolerance=2.0):
        return self._is_at_waypoint(self.drone1_vehicle_local_position, waypoint, tolerance)

    def is_at_waypoint_drone2(self, waypoint, tolerance=2.0):
        return self._is_at_waypoint(self.drone2_vehicle_local_position, waypoint, tolerance)

    def timer_callback(self):
        self.publish_offboard_control_mode_drone1()
        self.publish_offboard_control_mode_drone2()

        if self.flight_state == "INIT":
            self.publish_trajectory_setpoint_drone1(0.0, 0.0, self.takeoff_height)
            self.publish_trajectory_setpoint_drone2(
                self.drone2_flight_x_start, self.drone2_flight_y_start, self.takeoff_height
            )

            if not (self.drone1_data_received and self.drone2_data_received):
                if self.offboard_setpoint_counter % 150 == 0:
                    self.get_logger().warn(
                        f"Waiting for drone data - D1: {self.drone1_data_received} D2: {self.drone2_data_received}"
                    )
                return

            if not self.startup_readiness_satisfied():
                self.log_waiting_for_startup_readiness()
                return

            if self.offboard_setpoint_counter == 10:
                self.arm_drone(1)
                self.arm_drone(2)
                self.flight_state = "ARMING"

        elif self.flight_state == "ARMING":
            self.publish_trajectory_setpoint_drone1(0.0, 0.0, self.takeoff_height)
            self.publish_trajectory_setpoint_drone2(
                self.drone2_flight_x_start, self.drone2_flight_y_start, self.takeoff_height
            )

            d1_armed = self.drone1_vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            d2_armed = self.drone2_vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED

            if d1_armed and not hasattr(self, "drone1_offboard_engaged"):
                self.engage_offboard_mode_drone(1)
                self.drone1_offboard_engaged = True

            if d2_armed and not hasattr(self, "drone2_offboard_engaged"):
                self.engage_offboard_mode_drone(2)
                self.drone2_offboard_engaged = True

            if (
                d1_armed
                and d2_armed
                and hasattr(self, "drone1_offboard_engaged")
                and hasattr(self, "drone2_offboard_engaged")
            ):
                self.get_logger().info("Both vehicles armed and offboard engaged")
                self.flight_state = "TAKEOFF"

        elif self.flight_state == "TAKEOFF":
            self.publish_trajectory_setpoint_drone1(0.0, 0.0, self.takeoff_height)
            self.publish_trajectory_setpoint_drone2(
                self.drone2_flight_x_start, self.drone2_flight_y_start, self.takeoff_height
            )

            if (
                self.drone1_vehicle_local_position.z < self.takeoff_height + 0.95
                and self.drone2_vehicle_local_position.z < self.takeoff_height + 0.95
                and self.drone1_vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                and self.drone2_vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):

                self.takeoff_hover_counter += 1
                if self.takeoff_hover_counter >= self.takeoff_hover_time:
                    self.get_logger().info("Both drones takeoff complete, moving to start positions")
                    self.takeoff_hover_counter = 0
                    self.flight_state = "MOVE_TO_START"
                    if not self.takeoff_complete_published:
                        self.publish_timing_event("takeoff_complete")
                        self.takeoff_complete_published = True

        elif self.flight_state == "MOVE_TO_START":
            d1sx, d1sy = self.drone1_flight_x_start, self.drone1_flight_y_start
            d2sx, d2sy = self.drone2_flight_x_start, self.drone2_flight_y_start
            self.publish_trajectory_setpoint_drone1(d1sx, d1sy, self.takeoff_height)
            self.publish_trajectory_setpoint_drone2(d2sx, d2sy, self.takeoff_height)

            pos1 = self.drone1_vehicle_local_position
            pos2 = self.drone2_vehicle_local_position
            at_start_position = (
                abs(pos1.x - d1sx) < 0.8
                and abs(pos1.y - d1sy) < 0.8
                and abs(pos1.z - self.takeoff_height) < 0.8
                and abs(pos2.x - d2sx) < 0.8
                and abs(pos2.y - d2sy) < 0.8
                and abs(pos2.z - self.takeoff_height) < 0.8
            )

            if at_start_position:
                if not hasattr(self, "start_hover_counter"):
                    self.start_hover_counter = 0
                    self.start_hover_time = 3
                    self.get_logger().info("Both drones at start positions, stabilizing...")
                self.start_hover_counter += 1
                if self.start_hover_counter >= self.start_hover_time:
                    self.get_logger().info("Starting explorer mission")
                    self.flight_state = "ZIGZAG_MISSION"
                    self.drone1_current_waypoint_index = 0
                    self.drone2_current_waypoint_index = 0
            else:
                if hasattr(self, "start_hover_counter") and self.start_hover_counter > 0:
                    self.start_hover_counter = 0

        elif self.flight_state == "ZIGZAG_MISSION":
            if self.drone1_current_waypoint_index < len(self.drone1_waypoints):
                wp1 = self.drone1_waypoints[self.drone1_current_waypoint_index]
                self.publish_trajectory_setpoint_drone1(wp1[0], wp1[1], wp1[2])
                if self.is_at_waypoint_drone1(wp1):
                    self.drone1_current_waypoint_index += 1

            if self.drone2_current_waypoint_index < len(self.drone2_waypoints):
                wp2 = self.drone2_waypoints[self.drone2_current_waypoint_index]
                self.publish_trajectory_setpoint_drone2(wp2[0], wp2[1], wp2[2])
                if self.is_at_waypoint_drone2(wp2):
                    self.drone2_current_waypoint_index += 1

            if self.drone1_current_waypoint_index >= len(
                self.drone1_waypoints
            ) and self.drone2_current_waypoint_index >= len(self.drone2_waypoints):
                self.get_logger().info("Both drones completed all scanning runs! Returning home")
                self.flight_state = "RETURN"

        elif self.flight_state == "RETURN":
            self.publish_trajectory_setpoint_drone1(0.0, 0.0, self.takeoff_height)
            self.publish_trajectory_setpoint_drone2(
                self.drone2_flight_x_start, self.drone2_flight_y_start, self.takeoff_height
            )
            if self.offboard_setpoint_counter > 100:
                self.flight_state = "LAND"

        elif self.flight_state == "LAND":
            self.land_drone(1)
            self.land_drone(2)
            self.flight_state = "LANDED"

        elif self.flight_state == "LANDED":
            if (
                self.drone1_vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND
                and self.drone2_vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND
            ):
                if self.drone1_vehicle_local_position.z > -0.3 and self.drone2_vehicle_local_position.z > -0.3:
                    self.disarm_drone(1)
                    self.disarm_drone(2)
                    self.flight_state = "DISARMED"

        elif self.flight_state == "DISARMED":
            self.get_logger().info("Explorer mission complete!")
            exit()

        if self.offboard_setpoint_counter < 2000:
            self.offboard_setpoint_counter += 1

    def publish_timing_event(self, event_type: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "drone_id": "explorer_drones",
                "timing_event": event_type,
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
            }
        )
        self.timing_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleControllerExplorer()
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
