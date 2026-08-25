#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
"""Logs photographer positions from /photographer_positions to CSV for collision analysis."""

import csv
import json
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class PositionLogger(Node):
    def __init__(self):
        super().__init__("position_logger")

        self.declare_parameter("world_name", "mbs")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("world_model_source", "known")
        self.declare_parameter("run_style", "full")
        self.declare_parameter("planner_type", "openscvx")

        planner_type = self.get_parameter("planner_type").get_parameter_value().string_value or "openscvx"
        results_root = os.path.expanduser(
            os.environ.get(
                "MISSION_RESULTS_ROOT",
                os.path.expanduser("~/ros2_ws/src/mission_manager/results"),
            )
        )
        result_group = os.environ.get("MISSION_RESULTS_GROUP", "").strip()
        if result_group:
            default_log_dir = os.path.join(results_root, planner_type, result_group, "positions")
        else:
            default_log_dir = os.path.join(results_root, planner_type, "positions")
        self.declare_parameter("log_dir", default_log_dir)

        self.log_dir = self.get_parameter("log_dir").get_parameter_value().string_value
        self.world_name = self.get_parameter("world_name").get_parameter_value().string_value
        self.difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        self.world_model_source = self.get_parameter("world_model_source").get_parameter_value().string_value
        self.run_style = self.get_parameter("run_style").get_parameter_value().string_value

        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d_code = {"easy": "e", "medium": "m", "hard": "h"}.get(self.difficulty, "e")
        s_code = {"known": "k", "lidar_only": "l"}.get(self.world_model_source, "k")
        r_code = {"full": "f", "timed": "t"}.get(self.run_style, "f")
        config_str = f"{self.world_name}_{d_code}{s_code}{r_code}"
        self.csv_path = os.path.join(self.log_dir, f"drone_positions_{config_str}_{timestamp}.csv")

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "drone_id", "x", "y", "z"])

        position_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.position_sub = self.create_subscription(
            String, "/photographer_positions", self.position_callback, position_qos
        )

        self.drones_seen = set()
        self.position_count = 0

        self.get_logger().info(f"Position Logger started. Logging to: {self.csv_path}")

    def position_callback(self, msg):
        data = json.loads(msg.data)
        drone_id = data.get("photographer_id", "unknown")
        position = data.get("position", {})
        timestamp = data.get("timestamp", 0)
        x = position.get("x", 0.0)
        y = position.get("y", 0.0)
        z = position.get("z", 0.0)

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, drone_id, x, y, z])

        if drone_id not in self.drones_seen:
            self.drones_seen.add(drone_id)
            self.get_logger().info(f"New drone detected: {drone_id}")

        self.position_count += 1

    def destroy_node(self):
        print(
            f"Position Logger Summary: {self.position_count} positions, "
            f"{len(self.drones_seen)} drones ({sorted(self.drones_seen)}), file: {self.csv_path}"
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PositionLogger()
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
