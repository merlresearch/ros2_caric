#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class MissionController(Node):
    def __init__(self):
        super().__init__("mission_controller")
        self.declare_parameter("run_style", "full")  # 'full' or 'timed'
        self.declare_parameter("time_limit_sec", 600)
        self.declare_parameter("world_model_source", "known")  # 'known' or 'lidar_only'
        self.declare_parameter("world", "mbs")
        self.declare_parameter("difficulty", "easy")

        self.total_score = 0.0
        self.detected_count = 0
        self.total_poi_count = 0

        # Timer starts when first drone reaches takeoff_complete
        self.mission_timer_started = False
        self.start_time = None
        self.end_time = None
        self.mission_complete = False
        self.shutdown_countdown = None  # Countdown before printing summary

        self.drones_taken_off = set()
        self.photographers_used = set()
        self.num_clusters = 0

        self.drone_alive_status = {}  # {drone_model: bool}
        self.drone_seen_alive = set()  # drones that have reported alive=True at least once

        # Timing aggregates
        self.sum_openscvx_time = 0.0
        self.sum_time_to_cluster = 0.0
        self.sum_time_within_cluster = 0.0
        self.sum_time_to_return_home = 0.0
        self.completed_assignments = 0

        # Track raw timestamps for accurate duration calculation
        self.assignment_timestamps = []  # List of (assignment_start, home_return) tuples

        # Use TRANSIENT_LOCAL QoS for status publishers
        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.status_pub = self.create_publisher(String, "mission/status", transient_qos)
        self.score_sub = self.create_subscription(String, "total_score_status", self._score_cb, transient_qos)

        self.create_subscription(String, "/photographer_timing_updates", self._timing_event_cb, transient_qos)
        self.create_subscription(String, "/explorer_timing_updates", self._timing_event_cb, transient_qos)
        self.create_subscription(String, "/mission_timing_data", self._timing_data_cb, transient_qos)

        # Completion is an edge event for the current run. Do not latch it, or a
        # fresh controller can consume the previous run's completion immediately.
        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(String, "/mission_complete", self._mission_complete_cb, volatile_qos)
        self.create_subscription(String, "/mission/drone_status", self._drone_status_cb, transient_qos)

        self.timer = self.create_timer(1.0, self._tick)

        self.get_logger().info("Mission Controller started - waiting for first drone takeoff_complete")

    def _score_cb(self, msg: String):
        data = json.loads(msg.data)
        self.total_score = float(data.get("total_score", 0.0))
        self.detected_count = int(data.get("detected_count", 0))
        self.total_poi_count = int(data.get("total_poi_count", 0))

    def _timing_event_cb(self, msg: String):
        data = json.loads(msg.data)
        event_type = data.get("timing_event", "")
        drone_id = data.get("photographer_id", data.get("drone_id", "unknown"))

        if event_type == "takeoff_complete" and not self.mission_timer_started:
            self.mission_timer_started = True
            self.start_time = self.get_clock().now().nanoseconds / 1e9
            self.drones_taken_off.add(drone_id)
            self.get_logger().info(f"Mission timer STARTED on takeoff_complete from {drone_id}")
        elif event_type == "takeoff_complete":
            self.drones_taken_off.add(drone_id)

    def _timing_data_cb(self, msg: String):
        data = json.loads(msg.data)

        if not data.get("is_complete"):
            return

        photographer_id = data.get("photographer_id", "")
        if photographer_id:
            self.photographers_used.add(photographer_id)

        metrics = data.get("metrics", {}) or {}
        openscvx = metrics.get("openscvx_solve_time")
        to_cluster = metrics.get("time_to_cluster")
        within_cluster = metrics.get("time_within_cluster")
        to_return_home = metrics.get("time_to_return_home")

        if openscvx is not None:
            self.sum_openscvx_time += float(openscvx)
        if to_cluster is not None:
            self.sum_time_to_cluster += float(to_cluster)
        if within_cluster is not None:
            self.sum_time_within_cluster += float(within_cluster)
        if to_return_home is not None:
            self.sum_time_to_return_home += float(to_return_home)

        raw = data.get("raw_timestamps", {}) or {}
        assignment_start = raw.get("assignment_start")
        home_return = raw.get("home_return")
        if assignment_start is not None and home_return is not None:
            self.assignment_timestamps.append((float(assignment_start), float(home_return)))

        self.completed_assignments += 1

    def _drone_status_cb(self, msg: String):
        data = json.loads(msg.data)
        for drone in data.get("drones", []):
            model = drone.get("model", "") or ""
            role = drone.get("role", "") or ""
            # Some referees may omit 'alive' initially; treat missing as True
            alive = bool(drone.get("alive", True))

            if "photographer" not in role.lower() and "photographer" not in model.lower():
                continue

            previous_alive = self.drone_alive_status.get(model)
            self.drone_alive_status[model] = alive

            if alive:
                self.drone_seen_alive.add(model)
                continue

            if not self.mission_timer_started and previous_alive is not False:
                self.get_logger().warn(f"Photographer not alive before mission start: {model}")

    def _mission_complete_cb(self, msg: String):
        if self.mission_complete:
            return
        data = json.loads(msg.data)
        if data.get("event") == "mission_complete":
            if not self.mission_timer_started:
                self.get_logger().warn("Ignoring mission_complete before first takeoff_complete")
                return
            self.end_time = self.get_clock().now().nanoseconds / 1e9
            self.mission_complete = True
            self.num_clusters = data.get("total_clusters", 0)
            self.shutdown_countdown = 10  # gives other nodes time to finish printing
            elapsed = self.end_time - self.start_time if self.start_time else 0.0
            self.get_logger().info(f"MISSION COMPLETE! Elapsed: {elapsed:.2f}s - printing summary in 10s...")

    def _print_summary(self):
        elapsed = self.end_time - self.start_time if (self.start_time and self.end_time) else 0.0

        mission_duration_assignments = 0.0
        if self.assignment_timestamps:
            first_start = min(t[0] for t in self.assignment_timestamps)
            last_end = max(t[1] for t in self.assignment_timestamps)
            mission_duration_assignments = last_end - first_start

        world = self.get_parameter("world").get_parameter_value().string_value
        difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        run_style = self.get_parameter("run_style").get_parameter_value().string_value
        map_source = self.get_parameter("world_model_source").get_parameter_value().string_value

        drones_used = len(self.photographers_used) if self.photographers_used else len(self.drones_taken_off)

        avg_openscvx = self.sum_openscvx_time / self.completed_assignments if self.completed_assignments else 0.0
        avg_to_cluster = self.sum_time_to_cluster / self.completed_assignments if self.completed_assignments else 0.0
        avg_within_cluster = (
            self.sum_time_within_cluster / self.completed_assignments if self.completed_assignments else 0.0
        )
        avg_to_return_home = (
            self.sum_time_to_return_home / self.completed_assignments if self.completed_assignments else 0.0
        )

        w = 60
        sep = "═" * w

        lines = [
            "",
            "",
            f"╔{sep}╗",
            f'║{"MISSION SUMMARY":^{w}}║',
            f"╠{sep}╣",
            f'║{"Configuration":<{w}}║',
            f'╟{"─" * w}╢',
            f'║  {"World:":<22}{world:<{w-26}}║',
            f'║  {"Difficulty:":<22}{difficulty:<{w-26}}║',
            f'║  {"Run Style:":<22}{run_style:<{w-26}}║',
            f'║  {"Map Source:":<22}{map_source:<{w-26}}║',
            f"╠{sep}╣",
            f'║{"Results":<{w}}║',
            f'╟{"─" * w}╢',
            f'║  {"POIs Detected:":<22}{self.detected_count}/{self.total_poi_count:<{w-27}}║',
            f'║  {"Total Score:":<22}{self.total_score:<{w-26}.2f}║',
            f'║  {"Clusters:":<22}{self.num_clusters:<{w-26}}║',
            f'║  {"Assignments:":<22}{self.completed_assignments:<{w-26}}║',
            f'║  {"Drones Used:":<22}{drones_used:<{w-26}}║',
            f"╠{sep}╣",
            f'║{"Timing":<{w}}║',
            f'╟{"─" * w}╢',
            f'║  {"Mission Elapsed:":<22}{elapsed:<{w-27}.2f}s║',
            f'║  {"Assignments Duration:":<22}{mission_duration_assignments:<{w-27}.2f}s║',
            f'║  {"Total OpenSCvx:":<22}{self.sum_openscvx_time:<{w-27}.2f}s║',
            f'║  {"Avg OpenSCvx:":<22}{avg_openscvx:<{w-27}.2f}s║',
            f'║  {"Avg To Cluster:":<22}{avg_to_cluster:<{w-27}.2f}s║',
            f'║  {"Avg Within Cluster:":<22}{avg_within_cluster:<{w-27}.2f}s║',
            f'║  {"Avg Return Home:":<22}{avg_to_return_home:<{w-27}.2f}s║',
            f"╚{sep}╝",
            "",
        ]

        for line in lines:
            print(line)

    def _tick(self):

        if self.mission_timer_started and self.start_time:
            if self.mission_complete and self.end_time:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        else:
            elapsed = 0.0

        status = {
            "elapsed_sec": elapsed,
            "timer_started": self.mission_timer_started,
            "mission_complete": self.mission_complete,
            "run_style": self.get_parameter("run_style").get_parameter_value().string_value,
            "detected_count": self.detected_count,
            "total_poi_count": self.total_poi_count,
            "total_score": self.total_score,
            "world_model_source": self.get_parameter("world_model_source").get_parameter_value().string_value,
            "drones_taken_off": len(self.drones_taken_off),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

        run_style = self.get_parameter("run_style").value

        if run_style == "full" and self.mission_complete:
            if self.shutdown_countdown is not None:
                if self.shutdown_countdown > 0:
                    self.shutdown_countdown -= 1
                    return
                else:
                    self._print_summary()
                    raise SystemExit(0)

        elif run_style == "timed" and self.mission_timer_started:
            limit = int(self.get_parameter("time_limit_sec").value)
            if elapsed >= limit:
                self.end_time = self.get_clock().now().nanoseconds / 1e9
                self._print_summary()
                raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
