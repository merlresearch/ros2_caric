#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import time
import uuid
import xml.etree.ElementTree as ET

import gz.transport13 as gz_transport
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gz.msgs10.logical_camera_image_pb2 import LogicalCameraImage
from gz.msgs10.pose_v_pb2 import Pose_V
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def _resolve_world_name(world_key: str, difficulty: str) -> str:
    diff_map = {"easy": 10, "medium": 50, "hard": 100}
    poi_count = diff_map.get(difficulty, 10)
    computed = f"{world_key}_{poi_count}_poi"
    share_dir = get_package_share_directory("mission_manager")
    worlds_yaml = os.path.join(share_dir, "config", "worlds.yaml")
    if not os.path.exists(worlds_yaml):
        return computed
    cfg = yaml.safe_load(open(worlds_yaml, "r").read().replace("\t", "  ")) or {}
    entry = (cfg.get("worlds") or {}).get(world_key)
    if isinstance(entry, dict):
        overrides = entry.get("sdf_overrides") or {}
        if difficulty in overrides:
            return os.path.splitext(overrides[difficulty])[0]
    return computed


def _load_fleet(profile_key: str = "default") -> list:
    share_dir = get_package_share_directory("mission_manager")
    path = os.path.join(share_dir, "config", "spawn_profiles.yaml")
    if not os.path.exists(path):
        return []
    cfg = yaml.safe_load(open(path, "r").read().replace("\t", "  ")) or {}
    profile = cfg.get(profile_key) or {}
    return profile.get("fleet") or []


def _compute_gz_model_name(model: str, instance: int) -> str:
    base = model[3:] if model.startswith("gz_") else model
    return f"{base}_{instance}"


def _difficulty_from_total_poi_count(total_count: int) -> str:
    if total_count >= 100:
        return "hard"
    if total_count >= 50:
        return "medium"
    return "easy"


class POIDetectionNode(Node):
    def __init__(self):
        super().__init__("poi_detection_node")

        self.declare_parameter("world_name", "")
        self.declare_parameter("world", "")
        self.declare_parameter("difficulty", "auto")

        world_name_param = self.get_parameter("world_name").get_parameter_value().string_value
        world_key_param = self.get_parameter("world").get_parameter_value().string_value
        difficulty_param = self.get_parameter("difficulty").get_parameter_value().string_value or "auto"
        spawn_profile = world_key_param or "mbs"

        env_world = os.environ.get("PX4_GZ_WORLD", "").strip()
        if difficulty_param == "auto":
            difficulty_param = self._infer_difficulty_from_mission_status() or os.environ.get(
                "MISSION_DIFFICULTY", "easy"
            )

        if world_name_param:
            self.world_name = world_name_param
        elif env_world:
            self.world_name = env_world
        elif world_key_param:
            self.world_name = _resolve_world_name(world_key_param, difficulty_param)
        else:
            self.world_name = _resolve_world_name("mbs", difficulty_param)
        self.get_logger().info(f"POI detection using world: {self.world_name}")

        self.poi_reference_list = self.load_poi_from_world()
        self.detected_poi_list = []
        self.run_id = str(uuid.uuid4())
        self.gz_node = gz_transport.Node()
        self.model_poses = {}

        self.detection_pub = self.create_publisher(String, "/poi_detections", 10)
        self.status_pub = self.create_publisher(String, "/poi_detection_status", 10)
        self.detected_list_pub = self.create_publisher(String, "/poi_detected_list", 10)
        self.all_detected_pub = self.create_publisher(String, "/poi_all_detected", 10)
        self.all_detected_published = False

        pose_topic = f"/world/{self.world_name}/pose/info"
        if not self.gz_node.subscribe(Pose_V, pose_topic, self.pose_callback):
            self.get_logger().error(f"Failed to subscribe to {pose_topic}")
        else:
            self.get_logger().info(f"Subscribed to pose topic: {pose_topic}")

        self.explorer_targets = []
        fleet = _load_fleet(spawn_profile)
        for d in fleet:
            if d.get("role") != "explorer":
                continue
            instance = int(d.get("instance", 0))
            model = str(d.get("model", ""))
            name = str(d.get("name", f"explorer_{instance}"))
            if not instance or not model:
                continue
            gz_name = _compute_gz_model_name(model, instance)
            self.explorer_targets.append((name, instance, gz_name))
        if not self.explorer_targets:
            self.get_logger().warn("No explorer targets found in fleet profile; node will idle.")

        for friendly_name, _instance, gz_name in self.explorer_targets:
            camera_topic = (
                f"/world/{self.world_name}/model/{gz_name}/" f"link/camera_link/sensor/logic_wide/logical_camera"
            )
            cb = self._make_logical_camera_callback(friendly_name)
            if not self.gz_node.subscribe(LogicalCameraImage, camera_topic, cb):
                self.get_logger().error(f"Failed to subscribe to {camera_topic}")
            else:
                self.get_logger().info(f"Subscribed to explorer {friendly_name}: {camera_topic}")

        self.create_timer(5.0, self.publish_updates)

    def _infer_difficulty_from_mission_status(self, timeout_sec: float = 3.0):
        inferred = {"difficulty": None}
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        def _status_cb(msg: String):
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                return

            difficulty = data.get("difficulty")
            if isinstance(difficulty, str) and difficulty:
                inferred["difficulty"] = difficulty
                return

            total_count = data.get("total_poi_count")
            if isinstance(total_count, int) and total_count > 0:
                inferred["difficulty"] = _difficulty_from_total_poi_count(total_count)

        subscription = self.create_subscription(String, "/mission/status", _status_cb, qos)
        deadline = time.monotonic() + timeout_sec
        while inferred["difficulty"] is None and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(subscription)

        if inferred["difficulty"]:
            self.get_logger().info(f"Inferred difficulty from /mission/status: {inferred['difficulty']}")
        else:
            self.get_logger().warn("Could not infer difficulty from /mission/status; falling back to easy")
        return inferred["difficulty"]

    def pose_callback(self, msg: Pose_V):
        for pose in msg.pose:
            self.model_poses[pose.name] = pose

    def load_poi_from_world(self):
        poi_list = []

        world_file = None
        share_dir = get_package_share_directory("mission_manager")
        candidate = os.path.join(share_dir, "worlds", f"{self.world_name}.sdf")
        if os.path.exists(candidate):
            world_file = candidate
        else:
            base_paths = [
                os.path.expanduser("~/ros2_ws/src/PX4-Autopilot/Tools/simulation/gz/worlds"),
                "src/PX4-Autopilot/Tools/simulation/gz/worlds",
                "../../PX4-Autopilot/Tools/simulation/gz/worlds",
                "../../../PX4-Autopilot/Tools/simulation/gz/worlds",
            ]
            for base_path in base_paths:
                path = os.path.join(base_path, f"{self.world_name}.sdf")
                if os.path.exists(path):
                    world_file = path
                    break

        if world_file is None:
            self.get_logger().error(f"Could not find world file for {self.world_name}.sdf in expected locations")
            return poi_list

        tree = ET.parse(world_file)
        root = tree.getroot()

        for model in root.findall(".//model"):
            name = model.get("name")
            if not name or not (name.startswith("poi_") or name.startswith("fp_poi_")):
                continue
            pose_elem = model.find("pose")
            if pose_elem is None:
                continue
            pose_data = pose_elem.text.strip().split()
            if len(pose_data) < 6:
                continue
            poi_list.append(
                {
                    "id": name,
                    "position": {
                        "x": float(pose_data[0]),
                        "y": float(pose_data[1]),
                        "z": float(pose_data[2]),
                    },
                    "orientation": {
                        "roll": float(pose_data[3]),
                        "pitch": float(pose_data[4]),
                        "yaw": float(pose_data[5]),
                    },
                }
            )

        return poi_list

    def _make_logical_camera_callback(self, drone_name: str):
        def _cb(msg: LogicalCameraImage):
            self.process_camera_data(msg, drone_name)

        return _cb

    def process_camera_data(self, msg: LogicalCameraImage, drone_id: str):
        if not msg.model:
            return

        new_detections = 0
        for model in msg.model:
            model_name = model.name
            if not (model_name.startswith("poi_") or model_name.startswith("fp_poi_")):
                continue
            if self.is_poi_already_detected(model_name):
                continue

            if model_name in self.model_poses:
                wp = self.model_poses[model_name]
                world_position = {"x": wp.position.x, "y": wp.position.y, "z": wp.position.z}
                world_orientation = {
                    "x": wp.orientation.x,
                    "y": wp.orientation.y,
                    "z": wp.orientation.z,
                    "w": wp.orientation.w,
                }
            else:
                p = model.pose
                world_position = {"x": p.position.x, "y": p.position.y, "z": p.position.z}
                world_orientation = {
                    "x": p.orientation.x,
                    "y": p.orientation.y,
                    "z": p.orientation.z,
                    "w": p.orientation.w,
                }
                self.get_logger().warn(f"World pose not available for {model_name}, using camera coordinates")

            detected_poi = {
                "id": model_name,
                "detected_by": drone_id,
                "detected_position": world_position,
                "detected_orientation": world_orientation,
                "detection_timestamp": self.get_clock().now().to_msg(),
            }
            self.detected_poi_list.append(detected_poi)
            new_detections += 1

            self.get_logger().info(
                f"NEW POI by {drone_id}: {model_name} "
                f"({world_position['x']:.2f}, {world_position['y']:.2f}, {world_position['z']:.2f}) "
                f"[{len(self.detected_poi_list)}/{len(self.poi_reference_list)}]"
            )
            self.publish_detection_event(detected_poi)

        if new_detections > 0:
            self.get_logger().info(f"{drone_id} camera: {len(msg.model)} models, {new_detections} new POIs")

    def is_poi_already_detected(self, poi_id):
        return any(poi["id"] == poi_id for poi in self.detected_poi_list)

    def publish_detection_event(self, detected_poi):
        msg = String()
        msg.data = json.dumps(
            {
                "event": "poi_detected",
                "run_id": self.run_id,
                "poi_id": detected_poi["id"],
                "detected_by": detected_poi["detected_by"],
                "position": detected_poi["detected_position"],
                "timestamp": detected_poi["detection_timestamp"].sec,
            }
        )
        self.detection_pub.publish(msg)

    def publish_status(self):
        detected_ids = {poi["id"] for poi in self.detected_poi_list}
        remaining = sum(1 for poi in self.poi_reference_list if poi["id"] not in detected_ids)
        total = len(self.poi_reference_list)
        pct = (len(self.detected_poi_list) / total * 100) if total else 0.0
        all_detected = remaining == 0 and total > 0

        msg = String()
        msg.data = json.dumps(
            {
                "run_id": self.run_id,
                "total_poi_count": total,
                "detected_count": len(self.detected_poi_list),
                "remaining_count": remaining,
                "completion_percentage": pct,
                "all_detected": all_detected,
            }
        )
        self.status_pub.publish(msg)

        if all_detected:
            if not self.all_detected_published:
                self.get_logger().info("ALL POIs DETECTED! Publishing /poi_all_detected signal.")
                self.all_detected_published = True
            self.publish_all_detected_signal()

    def publish_detected_list(self):
        detected_data = [
            {
                "id": poi["id"],
                "detected_by": poi["detected_by"],
                "position": poi["detected_position"],
                "orientation": poi["detected_orientation"],
                "timestamp": poi["detection_timestamp"].sec,
            }
            for poi in self.detected_poi_list
        ]
        msg = String()
        msg.data = json.dumps({"run_id": self.run_id, "detected_pois": detected_data, "count": len(detected_data)})
        self.detected_list_pub.publish(msg)

    def publish_all_detected_signal(self):
        msg = String()
        msg.data = json.dumps(
            {
                "event": "all_pois_detected",
                "run_id": self.run_id,
                "total_count": len(self.poi_reference_list),
                "detected_count": len(self.detected_poi_list),
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
                "poi_ids": [poi["id"] for poi in self.detected_poi_list],
            }
        )
        self.all_detected_pub.publish(msg)

    def publish_updates(self):
        self.publish_status()
        self.publish_detected_list()


def main(args=None):
    rclpy.init(args=args)
    node = POIDetectionNode()
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
