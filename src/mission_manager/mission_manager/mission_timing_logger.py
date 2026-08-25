#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import json
import os
import time
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class MissionTimingLogger(Node):
    """Aggregates per-assignment timing metrics from photographer controllers and records them into CSV."""

    def __init__(self) -> None:
        super().__init__("mission_timing_logger")

        self.declare_parameter("world_name", "")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("run_style", "full")
        self.declare_parameter("time_limit_sec", 600)
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
            default_output_dir = os.path.join(results_root, planner_type, result_group, "timing")
        else:
            default_output_dir = os.path.join(results_root, planner_type, "timing")
        self.declare_parameter("output_dir", default_output_dir)

        self.world_name = self.get_parameter("world_name").get_parameter_value().string_value
        self.difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        self.spawn_profile = self.world_name or "mbs"
        self.run_style = self.get_parameter("run_style").get_parameter_value().string_value
        self.time_limit_sec = int(self.get_parameter("time_limit_sec").get_parameter_value().integer_value or 0)
        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value

        self._start_wall_time = time.time()
        self.map_build_time: Optional[float] = None
        self.map_known_ratio_threshold = float(os.environ.get("GLOBAL_REQUIRED_KNOWN_RATIO", "0.35"))
        self.map_source = os.environ.get("MISSION_MAP_SOURCE", "").strip()

        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w = self.world_name or "mbs"
        d = self.difficulty[0] if self.difficulty else "e"
        s = self.map_source[0] if self.map_source else "k"
        r = self.run_style[0] if self.run_style else "f"
        config_prefix = f"{w}_{d}{s}{r}"
        self.csv_path = os.path.join(self.output_dir, f"mission_timing_{config_prefix}_{ts}.csv")
        self._csv_header_written = False

        self.num_assignments = 0
        self.sum_time_to_cluster = 0.0
        self.sum_time_within_cluster = 0.0
        self.sum_time_to_return_home = 0.0
        self.sum_total_time = 0.0

        self.mission_elapsed_sec = 0.0
        self.detected_count = 0
        self.total_poi_count = 0
        self.mission_complete = False
        self._final_summary_written = False

        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(String, "/mission_timing_data", self._timing_cb, transient_qos)
        self.create_subscription(String, "mission/status", self._status_cb, transient_qos)
        self.create_subscription(String, "/mission/map/status", self._map_status_cb, transient_qos)

        self.create_timer(10.0, self._flush_summary_periodic)

        self.get_logger().info(f"MissionTimingLogger writing to: {self.csv_path}")

    def _status_cb(self, msg: String) -> None:
        data = json.loads(msg.data)
        self.mission_elapsed_sec = float(data.get("elapsed_sec", self.mission_elapsed_sec))
        self.detected_count = int(data.get("detected_count", self.detected_count))
        self.total_poi_count = int(data.get("total_poi_count", self.total_poi_count))
        self.mission_complete = bool(data.get("mission_complete", self.mission_complete))
        if not self._final_summary_written and self.run_style == "full" and self.mission_complete:
            self._final_summary_written = True
            self._write_summary_row(final=True)

    def _timing_cb(self, msg: String) -> None:
        data = json.loads(msg.data)

        assignment_id = data.get("assignment_id", "")
        photographer_id = data.get("photographer_id", "")
        cluster_id = data.get("cluster_id", "")
        metrics = data.get("metrics", {}) or {}
        sequencing_method = str(data.get("sequencing_method", "")).strip()
        tsp_solve_time = self._to_float(metrics.get("tsp_solve_time"))
        astar_compute_time = self._to_float(metrics.get("astar_compute_time"))
        openscvx_solve_time = self._to_float(metrics.get("openscvx_solve_time"))
        openscvx_preprocessing_time = self._to_float(metrics.get("openscvx_preprocessing_time"))
        openscvx_main_solve_time = self._to_float(metrics.get("openscvx_main_solve_time"))
        openscvx_postprocessing_time = self._to_float(metrics.get("openscvx_postprocessing_time"))
        raw = data.get("raw_timestamps", {}) or {}
        is_complete = bool(data.get("is_complete", False))
        current_event = str(data.get("current_event", "")).strip()

        time_to_cluster = self._to_float(metrics.get("time_to_cluster"))
        time_within_cluster = self._to_float(metrics.get("time_within_cluster"))
        time_to_return_home = self._to_float(metrics.get("time_to_return_home"))
        total_time = self._to_float(metrics.get("total_time"))

        raw_asn = self._to_float(raw.get("assignment_start"))
        raw_start = self._to_float(raw.get("cluster_start"))
        raw_comp = self._to_float(raw.get("cluster_complete"))
        raw_home = self._to_float(raw.get("home_return"))

        def _safe_delta(a, b):
            if a is None or b is None:
                return None
            d = b - a
            return d if d >= 0 else None

        if time_to_cluster is None:
            time_to_cluster = _safe_delta(raw_asn, raw_start)
        if time_within_cluster is None:
            time_within_cluster = _safe_delta(raw_start, raw_comp)
        if time_to_return_home is None:
            time_to_return_home = _safe_delta(raw_comp, raw_home)
        if total_time is None:
            total_time = _safe_delta(raw_asn, raw_home)

        def _clamp_non_negative(label: str, v: Optional[float]) -> Optional[float]:
            if v is None:
                return None
            if v < 0:
                self.get_logger().warn(f"Negative duration for {label} on {assignment_id}; discarding from row")
                return None
            return v

        time_to_cluster = _clamp_non_negative("time_to_cluster", time_to_cluster)
        time_within_cluster = _clamp_non_negative("time_within_cluster", time_within_cluster)
        time_to_return_home = _clamp_non_negative("time_to_return_home", time_to_return_home)
        total_time = _clamp_non_negative("total_time", total_time)

        have_parts = time_to_cluster is not None and time_within_cluster is not None and time_to_return_home is not None
        if have_parts:
            total_from_parts = time_to_cluster + time_within_cluster + time_to_return_home
            if total_time is None or abs(total_time - total_from_parts) > 2.0:
                total_time = total_from_parts
        elif total_time is None and raw_asn is not None and raw_home is not None and raw_home >= raw_asn:
            total_time = raw_home - raw_asn

        if current_event == "cluster_complete":
            self._write_assignment_row(
                assignment_id=assignment_id,
                photographer_id=photographer_id,
                cluster_id=cluster_id,
                time_to_cluster=time_to_cluster,
                time_within_cluster=time_within_cluster,
                time_to_return_home=None,
                total_time=None,
                assignment_start=raw_asn,
                cluster_start=raw_start,
                cluster_complete=raw_comp,
                home_return=None,
                record_type="assignment_partial",
                sequencing_method=sequencing_method,
                tsp_solve_time=tsp_solve_time,
                astar_compute_time=astar_compute_time,
                openscvx_solve_time=openscvx_solve_time,
                openscvx_preprocessing_time=openscvx_preprocessing_time,
                openscvx_main_solve_time=openscvx_main_solve_time,
                openscvx_postprocessing_time=openscvx_postprocessing_time,
            )

        if current_event == "home_return" or is_complete:
            self._write_assignment_row(
                assignment_id=assignment_id,
                photographer_id=photographer_id,
                cluster_id=cluster_id,
                time_to_cluster=time_to_cluster,
                time_within_cluster=time_within_cluster,
                time_to_return_home=time_to_return_home,
                total_time=total_time,
                assignment_start=raw_asn,
                cluster_start=raw_start,
                cluster_complete=raw_comp,
                home_return=raw_home,
                record_type="assignment",
                sequencing_method=sequencing_method,
                tsp_solve_time=tsp_solve_time,
                astar_compute_time=astar_compute_time,
                openscvx_solve_time=openscvx_solve_time,
                openscvx_preprocessing_time=openscvx_preprocessing_time,
                openscvx_main_solve_time=openscvx_main_solve_time,
                openscvx_postprocessing_time=openscvx_postprocessing_time,
            )

        if is_complete and (
            time_to_cluster is not None
            and time_within_cluster is not None
            and time_to_return_home is not None
            and total_time is not None
        ):
            self.num_assignments += 1
            self.sum_time_to_cluster += time_to_cluster
            self.sum_time_within_cluster += time_within_cluster
            self.sum_time_to_return_home += time_to_return_home
            self.sum_total_time += total_time

    def _to_float(self, value) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    def _ensure_header(self, writer: csv.DictWriter) -> None:
        if not self._csv_header_written:
            writer.writeheader()
            self._csv_header_written = True

    def _write_assignment_row(self, **kwargs) -> None:
        row = {
            "record_type": kwargs.get("record_type", "assignment"),
            "timestamp": datetime.now().isoformat(),
            "world": self.world_name,
            "difficulty": self.difficulty,
            "spawn_profile": self.spawn_profile,
            "run_style": self.run_style,
            "map_source": self.map_source,
            "assignment_id": kwargs.get("assignment_id", ""),
            "photographer_id": kwargs.get("photographer_id", ""),
            "cluster_id": kwargs.get("cluster_id", ""),
            "time_to_cluster": kwargs.get("time_to_cluster"),
            "time_within_cluster": kwargs.get("time_within_cluster"),
            "time_to_return_home": kwargs.get("time_to_return_home"),
            "total_time": kwargs.get("total_time"),
            "assignment_start": kwargs.get("assignment_start"),
            "cluster_start": kwargs.get("cluster_start"),
            "cluster_complete": kwargs.get("cluster_complete"),
            "home_return": kwargs.get("home_return"),
            "sequencing_method": kwargs.get("sequencing_method", ""),
            "tsp_solve_time": kwargs.get("tsp_solve_time"),
            "astar_compute_time": kwargs.get("astar_compute_time"),
            "openscvx_solve_time": kwargs.get("openscvx_solve_time"),
            "openscvx_preprocessing_time": kwargs.get("openscvx_preprocessing_time"),
            "openscvx_main_solve_time": kwargs.get("openscvx_main_solve_time"),
            "openscvx_postprocessing_time": kwargs.get("openscvx_postprocessing_time"),
            "map_build_time": self.map_build_time,
            "mission_elapsed_sec": "",
            "num_assignments": "",
            "detected_count": "",
            "total_poi_count": "",
            "avg_time_to_cluster": "",
            "avg_time_within_cluster": "",
            "avg_time_to_return_home": "",
            "avg_total_time": "",
        }
        self._append_row(row)

    def _write_summary_row(self, final: bool = False) -> None:
        avg_to_cluster = (self.sum_time_to_cluster / self.num_assignments) if self.num_assignments else None
        avg_within = (self.sum_time_within_cluster / self.num_assignments) if self.num_assignments else None
        avg_return = (self.sum_time_to_return_home / self.num_assignments) if self.num_assignments else None
        avg_total = (self.sum_total_time / self.num_assignments) if self.num_assignments else None

        row = {
            "record_type": "summary",
            "timestamp": datetime.now().isoformat(),
            "world": self.world_name,
            "difficulty": self.difficulty,
            "spawn_profile": self.spawn_profile,
            "run_style": self.run_style,
            "map_source": self.map_source,
            "assignment_id": "",
            "photographer_id": "",
            "cluster_id": "",
            "time_to_cluster": "",
            "time_within_cluster": "",
            "time_to_return_home": "",
            "total_time": "",
            "assignment_start": "",
            "cluster_start": "",
            "cluster_complete": "",
            "home_return": "",
            "sequencing_method": "",
            "tsp_solve_time": "",
            "astar_compute_time": "",
            "openscvx_solve_time": "",
            "openscvx_preprocessing_time": "",
            "openscvx_main_solve_time": "",
            "openscvx_postprocessing_time": "",
            "map_build_time": self.map_build_time,
            "mission_elapsed_sec": self.mission_elapsed_sec,
            "num_assignments": self.num_assignments,
            "detected_count": self.detected_count,
            "total_poi_count": self.total_poi_count,
            "avg_time_to_cluster": avg_to_cluster,
            "avg_time_within_cluster": avg_within,
            "avg_time_to_return_home": avg_return,
            "avg_total_time": avg_total,
        }
        self._append_row(row)
        if final:
            print(f"Mission summary written: {self.csv_path}")

    def _append_row(self, row: dict) -> None:
        fieldnames = [
            "record_type",
            "timestamp",
            "world",
            "difficulty",
            "spawn_profile",
            "run_style",
            "map_source",
            "assignment_id",
            "photographer_id",
            "cluster_id",
            "time_to_cluster",
            "time_within_cluster",
            "time_to_return_home",
            "total_time",
            "assignment_start",
            "cluster_start",
            "cluster_complete",
            "home_return",
            "sequencing_method",
            "tsp_solve_time",
            "astar_compute_time",
            "openscvx_solve_time",
            "openscvx_preprocessing_time",
            "openscvx_main_solve_time",
            "openscvx_postprocessing_time",
            "map_build_time",
            "mission_elapsed_sec",
            "num_assignments",
            "detected_count",
            "total_poi_count",
            "avg_time_to_cluster",
            "avg_time_within_cluster",
            "avg_time_to_return_home",
            "avg_total_time",
        ]
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            self._ensure_header(writer)
            writer.writerow(row)

    def _flush_summary_periodic(self) -> None:
        self._write_summary_row(final=False)

    def _map_status_cb(self, msg: String) -> None:
        data = json.loads(msg.data)
        kr = float(data.get("known_ratio", 0.0))
        ts = time.time()
        if self.map_build_time is None and kr >= self.map_known_ratio_threshold:
            self.map_build_time = max(0.0, ts - self._start_wall_time)


def main(args=None):
    rclpy.init(args=args)
    node = MissionTimingLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._final_summary_written:
            node._write_summary_row(final=True)
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
