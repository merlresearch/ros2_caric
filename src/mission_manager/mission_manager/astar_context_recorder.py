#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
"""Record A* run context for later OpenSCvx replay/tuning."""

import json
import os
import pickle
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .world_config import (
    compute_gz_model_name,
    load_fleet,
    load_map_bounds,
    load_mapping_defaults,
    resolve_world_name,
)


def _json_loads(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except Exception:
        return {"_raw": raw}
    return data if isinstance(data, dict) else {"value": data}


def _stamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


class AstarContextRecorder(Node):
    """Collect the runtime context that defines an A* clustering/assignment scenario."""

    def __init__(self) -> None:
        super().__init__("astar_context_recorder")

        self.declare_parameter("world_name", "mbs")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("world_model_source", "known")
        self.declare_parameter("run_style", "full")
        self.declare_parameter("planner_type", "astar")
        self.declare_parameter("run_index", 1)
        self.declare_parameter("timeout_seconds", 900)
        self.declare_parameter("output_path", "")
        self.declare_parameter("record_position_samples", True)
        self.declare_parameter("position_sample_period_sec", 1.0)
        self.declare_parameter("flush_period_sec", 5.0)

        self.world_name = self.get_parameter("world_name").get_parameter_value().string_value or "mbs"
        self.difficulty = self.get_parameter("difficulty").get_parameter_value().string_value or "easy"
        self.world_model_source = self.get_parameter("world_model_source").get_parameter_value().string_value or "known"
        self.run_style = self.get_parameter("run_style").get_parameter_value().string_value or "full"
        self.planner_type = self.get_parameter("planner_type").get_parameter_value().string_value or "astar"
        self.run_index = int(self.get_parameter("run_index").value)
        self.timeout_seconds = int(self.get_parameter("timeout_seconds").value)
        self.record_position_samples = bool(self.get_parameter("record_position_samples").value)
        self.position_sample_period_sec = float(self.get_parameter("position_sample_period_sec").value)
        flush_period_sec = float(self.get_parameter("flush_period_sec").value)

        output_path = self.get_parameter("output_path").get_parameter_value().string_value
        if not output_path:
            results_root = os.path.expanduser(
                os.environ.get("MISSION_RESULTS_ROOT", "~/ros2_ws/src/mission_manager/results")
            )
            group = os.environ.get("MISSION_RESULTS_GROUP", "").strip()
            out_dir = os.path.join(results_root, self.planner_type, group, "context_pickles")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            d_code = self.difficulty[:1] or "e"
            s_code = self.world_model_source[:1] or "k"
            r_code = self.run_style[:1] or "f"
            output_path = os.path.join(
                out_dir,
                f"{self.world_name}_{d_code}{s_code}{r_code}_astar_context_run{self.run_index}_{ts}.pkl",
            )
        self.output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self._last_position_sample_time: Dict[str, float] = {}
        self._last_flush_error: Optional[str] = None

        self.record: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "created_at": _stamp_now(),
            "updated_at": None,
            "run": {
                "planner_type": self.planner_type,
                "world": self.world_name,
                "gazebo_world": resolve_world_name(self.world_name, self.difficulty),
                "difficulty": self.difficulty,
                "world_model_source": self.world_model_source,
                "run_style": self.run_style,
                "run_index": self.run_index,
                "timeout_seconds": self.timeout_seconds,
                "output_path": self.output_path,
            },
            "static_context": self._load_static_context(),
            "clusters": {},
            "cluster_messages": [],
            "cluster_status_messages": [],
            "assignments": [],
            "assignments_by_id": {},
            "assignment_completions": [],
            "assignment_rejections": [],
            "poi_detected_passed": [],
            "poi_all_detected": [],
            "photographer_positions": {
                "first": {},
                "latest": {},
                "samples": {},
                "position_sample_period_sec": self.position_sample_period_sec,
            },
            "mission_status": [],
            "score_status": [],
            "timing": {
                "mission_timing_data": [],
                "photographer_timing_updates": [],
                "cluster_timing_events": [],
            },
            "map": {
                "status": [],
                "meta": None,
                "latest_occupancy": None,
                "derived_obstacle_boxes": [],
            },
            "referee": {
                "drone_status": [],
                "drone_los_status": [],
                "photographer_los_status": [],
            },
            "counts": {},
        }

        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL,
            depth=100,
        )

        self.create_subscription(String, "/cluster_information", self._cluster_cb, volatile_qos)
        self.create_subscription(
            String, "/cluster_status", self._append_json_cb("cluster_status_messages"), volatile_qos
        )
        self.create_subscription(String, "/photographer_assignments", self._assignment_cb, transient_qos)
        self.create_subscription(
            String, "/photographer_assignment_complete", self._append_json_cb("assignment_completions"), volatile_qos
        )
        self.create_subscription(
            String, "/photographer_assignment_rejected", self._append_json_cb("assignment_rejections"), volatile_qos
        )
        self.create_subscription(
            String, "/poi_detected_passed", self._append_json_cb("poi_detected_passed"), volatile_qos
        )
        self.create_subscription(String, "/poi_all_detected", self._append_json_cb("poi_all_detected"), volatile_qos)
        self.create_subscription(String, "/photographer_positions", self._position_cb, transient_qos)
        self.create_subscription(String, "mission/status", self._append_json_cb("mission_status"), transient_qos)
        self.create_subscription(String, "/total_score_status", self._append_json_cb("score_status"), transient_qos)
        self.create_subscription(
            String, "/mission_timing_data", self._append_nested_json_cb("timing", "mission_timing_data"), transient_qos
        )
        self.create_subscription(
            String,
            "/photographer_timing_updates",
            self._append_nested_json_cb("timing", "photographer_timing_updates"),
            transient_qos,
        )
        self.create_subscription(
            String,
            "/cluster_timing_events",
            self._append_nested_json_cb("timing", "cluster_timing_events"),
            transient_qos,
        )
        self.create_subscription(
            String, "/mission/map/status", self._append_nested_json_cb("map", "status"), transient_qos
        )
        self.create_subscription(String, "/mission/map/meta", self._map_meta_cb, transient_qos)
        self.create_subscription(OccupancyGrid, "/mission/map/occupancy", self._map_occupancy_cb, transient_qos)
        self.create_subscription(
            String, "/mission/drone_status", self._append_nested_json_cb("referee", "drone_status"), transient_qos
        )
        self.create_subscription(
            String, "/drone_los_status", self._append_nested_json_cb("referee", "drone_los_status"), transient_qos
        )
        self.create_subscription(
            String,
            "/photographer_los_status",
            self._append_nested_json_cb("referee", "photographer_los_status"),
            transient_qos,
        )

        self.create_timer(max(1.0, flush_period_sec), self._write_pickle)
        self.get_logger().info(f"A* context recorder writing pickle to: {self.output_path}")

    def _load_static_context(self) -> Dict[str, Any]:
        fleet = load_fleet(self.world_name)
        mapping_defaults = load_mapping_defaults()
        map_bounds = load_map_bounds(self.world_name)
        photographers = [d for d in fleet if d.get("role") == "photographer" or "photographer" in d.get("model", "")]
        explorers = [d for d in fleet if d.get("role") == "explorer" or "explorer" in d.get("model", "")]

        for drone in fleet:
            drone["gz_model_name"] = compute_gz_model_name(drone["model"], drone["instance"])
            drone["start_position_enu"] = {
                "x": drone.get("x", 0.0),
                "y": drone.get("y", 0.0),
                "z": drone.get("z", 0.5),
                "yaw_deg": drone.get("yaw_deg", 0.0),
            }

        known_obstacles = []
        comm_center = [0.0, -25.0, 5.0]
        try:
            from caric2_baseline_2.phases.common import get_comm_center_for_world, get_obstacles_for_world

            known_obstacles = _to_builtin(get_obstacles_for_world(self.world_name))
            comm_center = _to_builtin(get_comm_center_for_world(self.world_name))
        except Exception as exc:
            self.get_logger().warn(f"Could not import OpenSCvx obstacle definitions: {exc}")

        los_boxes = self._load_los_boxes()

        return {
            "fleet": fleet,
            "photographers": photographers,
            "explorers": explorers,
            "map_bounds": map_bounds,
            "mapping_defaults": mapping_defaults,
            "known_obstacles_openscvx": known_obstacles,
            "los_obstacle_boxes": los_boxes,
            "communication": {
                "comm_center_enu": comm_center,
                "gcs_position_world": {"x": comm_center[0], "y": comm_center[1], "z": 0.0},
                "openscvx_comm_radius_m": 75.0,
                "referee_los_distance_threshold_m": 100.0,
            },
        }

    def _load_los_boxes(self) -> Dict[str, Any]:
        share_dir = get_package_share_directory("mission_manager")
        gazebo_world = resolve_world_name(self.world_name, self.difficulty)
        candidates = [
            os.path.join(share_dir, "models", gazebo_world, "bounding_boxes", "box_description.yaml"),
            os.path.join(share_dir, "models", self.world_name, "bounding_boxes", "box_description.yaml"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                return {"path": path, "boxes": yaml.safe_load(f) or {}}
        return {"path": None, "boxes": {}}

    def _append_json_cb(self, key: str):
        def cb(msg: String) -> None:
            self.record[key].append(_json_loads(msg.data))

        return cb

    def _append_nested_json_cb(self, section: str, key: str):
        def cb(msg: String) -> None:
            self.record[section][key].append(_json_loads(msg.data))

        return cb

    def _cluster_cb(self, msg: String) -> None:
        data = _json_loads(msg.data)
        cluster_id = data.get("cluster_id")
        if cluster_id:
            self.record["clusters"][cluster_id] = data
        self.record["cluster_messages"].append(data)

    def _assignment_cb(self, msg: String) -> None:
        data = _json_loads(msg.data)
        photographer_id = data.get("photographer_id")
        assignment_id = data.get("assignment_id")
        if photographer_id:
            data["photographer_position_at_assignment"] = self.record["photographer_positions"]["latest"].get(
                photographer_id
            )
        if assignment_id:
            self.record["assignments_by_id"][assignment_id] = data
        self.record["assignments"].append(data)

    def _position_cb(self, msg: String) -> None:
        data = _json_loads(msg.data)
        photographer_id = data.get("photographer_id", "unknown")
        timestamp = float(data.get("timestamp", self.get_clock().now().nanoseconds / 1e9) or 0.0)
        positions = self.record["photographer_positions"]
        if photographer_id not in positions["first"]:
            positions["first"][photographer_id] = data
        positions["latest"][photographer_id] = data

        if not self.record_position_samples:
            return
        last_sample = self._last_position_sample_time.get(photographer_id)
        if last_sample is not None and timestamp - last_sample < self.position_sample_period_sec:
            return
        positions["samples"].setdefault(photographer_id, []).append(data)
        self._last_position_sample_time[photographer_id] = timestamp

    def _map_meta_cb(self, msg: String) -> None:
        self.record["map"]["meta"] = _json_loads(msg.data)

    def _map_occupancy_cb(self, msg: OccupancyGrid) -> None:
        data = {
            "frame_id": msg.header.frame_id,
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "resolution": float(msg.info.resolution),
            "width": int(msg.info.width),
            "height": int(msg.info.height),
            "origin": {
                "x": float(msg.info.origin.position.x),
                "y": float(msg.info.origin.position.y),
                "z": float(msg.info.origin.position.z),
            },
            "data": list(msg.data),
        }
        self.record["map"]["latest_occupancy"] = data
        self.record["map"]["derived_obstacle_boxes"] = self._derive_obstacle_boxes(data)

    def _derive_obstacle_boxes(self, occupancy: Dict[str, Any]):
        try:
            from caric2_baseline_2.phases.openscvx_utils import grid_to_aggregated_boxes

            grid = np.asarray(occupancy["data"], dtype=np.int8).reshape((occupancy["height"], occupancy["width"]))
            meta = self.record["map"].get("meta") or {}
            defaults = self.record["static_context"].get("mapping_defaults", {})
            boxes = grid_to_aggregated_boxes(
                grid=grid,
                origin_x=float(occupancy["origin"]["x"]),
                origin_y=float(occupancy["origin"]["y"]),
                resolution=float(occupancy["resolution"]),
                z_min=float(meta.get("z_min", defaults.get("z_min", 0.5))),
                z_max=float(meta.get("z_max", defaults.get("z_max", 80.0))),
                treat_unknown_as_obstacle=False,
                min_box_area=3.0,
                max_aggregate_boxes=5,
                downsample_factor=2,
                use_dbscan=True,
                dbscan_eps=5.0,
                dbscan_min_samples=2,
            )
            return _to_builtin(boxes)
        except Exception as exc:
            if self._last_flush_error != str(exc):
                self.get_logger().warn(f"Could not derive OpenSCvx boxes from occupancy grid: {exc}")
                self._last_flush_error = str(exc)
            return []

    def _write_pickle(self) -> None:
        self.record["updated_at"] = _stamp_now()
        self.record["counts"] = {
            "clusters": len(self.record["clusters"]),
            "assignments": len(self.record["assignments"]),
            "assignment_completions": len(self.record["assignment_completions"]),
            "assignment_rejections": len(self.record["assignment_rejections"]),
            "poi_detected_passed": len(self.record["poi_detected_passed"]),
            "positioned_photographers": len(self.record["photographer_positions"]["latest"]),
            "map_status_messages": len(self.record["map"]["status"]),
        }
        tmp_path = f"{self.output_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.record, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.output_path)

    def destroy_node(self) -> None:
        self._write_pickle()
        self.get_logger().info(f"A* replay context pickle saved: {self.output_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AstarContextRecorder()
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
