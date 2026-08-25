#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os

import gz.transport13 as gz_transport
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gz.msgs10.pose_v_pb2 import Pose_V
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from px4_msgs.msg import VehicleLandDetected, VehicleStatus

from .world_config import compute_gz_model_name, load_fleet, resolve_world_name


class RefereeNode(Node):
    def __init__(self):
        super().__init__("referee_node")

        self.declare_parameter("world_name", "")
        self.declare_parameter("difficulty", "easy")
        world_key = self.get_parameter("world_name").get_parameter_value().string_value
        difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        self.world_name = resolve_world_name(world_key or os.environ.get("PX4_GZ_WORLD", "mbs"), difficulty)

        self.declare_parameter("los_distance_threshold", 50.0)
        self.los_threshold = self.get_parameter("los_distance_threshold").get_parameter_value().double_value

        self.declare_parameter("los_use_aabb", True)
        self.declare_parameter("los_boxes_yaml", "")
        self.declare_parameter("los_eps", 1e-6)
        self.los_use_aabb = self.get_parameter("los_use_aabb").get_parameter_value().bool_value
        self.los_boxes_yaml = self.get_parameter("los_boxes_yaml").get_parameter_value().string_value
        self.los_eps = self.get_parameter("los_eps").get_parameter_value().double_value

        self.model_poses = {}
        self.gcs_position = None
        self.gcs_position_logged = False

        self._fleet = load_fleet("default")
        self.explorer_drones = []
        self.photographer_drones = []
        for d in self._fleet:
            gz_model = compute_gz_model_name(d["model"], d["instance"])
            if d.get("role") == "explorer" or "explorer" in d.get("model", ""):
                self.explorer_drones.append(gz_model)
            else:
                self.photographer_drones.append(gz_model)

        self.gz_node = gz_transport.Node()

        # Use TRANSIENT_LOCAL QoS for all status topics
        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.los_status_pub = self.create_publisher(String, "drone_los_status", transient_qos)
        self.photographer_los_pub = self.create_publisher(String, "photographer_los_status", transient_qos)
        self.drone_status_pub = self.create_publisher(String, "mission/drone_status", transient_qos)
        self.eligible_pub = self.create_publisher(String, "mission/eligible_drones", transient_qos)

        self.pose_topic = f"/world/{self.world_name}/pose/info"
        if not self.gz_node.subscribe(Pose_V, self.pose_topic, self.pose_callback):
            self.get_logger().error(f"Failed to subscribe to {self.pose_topic}")
        else:
            self.get_logger().info(f"Subscribed to pose topic: {self.pose_topic}")

        self.get_logger().info(f"Referee Node initialized with LOS threshold: {self.los_threshold}m")
        self.get_logger().info(f"Explorer drones: {self.explorer_drones}")
        self.get_logger().info(f"Photographer drones: {self.photographer_drones}")

        self._los_boxes = []  # list of dicts: { 'half': (hx,hy,hz), 'R': [[...]], 't': (tx,ty,tz) }
        if self.los_use_aabb:
            self._load_los_boxes()

        self.create_timer(1.0, self.publish_los_status)
        self.create_timer(1.0, self.publish_mission_status)

        self.heartbeat_timeout_s = 3.0
        self.px4_last_heartbeat = {}
        self.px4_failsafe = {}
        self.sticky_failsafe = {}
        self.land_detected = {}
        self.name_to_px4_ns = {}
        for d in self._fleet:
            self.name_to_px4_ns[d["name"]] = f"/px4_{d['instance']}"
        # Use best-effort QoS to match PX4 publishers
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for name, ns in self.name_to_px4_ns.items():
            # Subscribe to both v1 and non-v1 status topics to be robust across setups
            self.create_subscription(
                VehicleStatus,
                f"{ns}/fmu/out/vehicle_status_v1",
                lambda msg, dn=name: self._px4_status_cb(dn, msg),
                best_effort_qos,
            )
            self.create_subscription(
                VehicleStatus,
                f"{ns}/fmu/out/vehicle_status",
                lambda msg, dn=name: self._px4_status_cb(dn, msg),
                best_effort_qos,
            )
            self.create_subscription(
                VehicleLandDetected,
                f"{ns}/fmu/out/vehicle_land_detected",
                lambda msg, dn=name: self._px4_land_cb(dn, msg),
                best_effort_qos,
            )

        self.name_to_model = {d["name"]: compute_gz_model_name(d["model"], d["instance"]) for d in self._fleet}

    def pose_callback(self, msg: Pose_V):
        for pose in msg.pose:
            self.model_poses[pose.name] = pose
            if pose.name == "gcs":
                self.gcs_position = {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z}
                if not self.gcs_position_logged:
                    self.get_logger().info(f"GCS position detected: {self.gcs_position}")
                    self.gcs_position_logged = True

    def check_drone_los(self, drone_model_name):
        if self.gcs_position is None or drone_model_name not in self.model_poses:
            return False

        drone_pose = self.model_poses[drone_model_name]
        drone_position = {"x": drone_pose.position.x, "y": drone_pose.position.y, "z": drone_pose.position.z}

        if self.calculate_3d_distance(drone_position, self.gcs_position) > self.los_threshold:
            return False

        if not self.los_use_aabb or not self._los_boxes:
            return True

        p = (self.gcs_position["x"], self.gcs_position["y"], self.gcs_position["z"])
        q = (drone_position["x"], drone_position["y"], drone_position["z"])
        return not self._segment_blocked_by_any_box(p, q)

    def calculate_3d_distance(self, pos1, pos2):
        dx = pos1["x"] - pos2["x"]
        dy = pos1["y"] - pos2["y"]
        dz = pos1["z"] - pos2["z"]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def publish_los_status(self):
        los_status = {}

        for explorer in self.explorer_drones:
            los_status[explorer] = {"has_los": self.check_drone_los(explorer), "drone_type": "explorer"}
            if self.gcs_position is not None and explorer in self.model_poses:
                drone_pose = self.model_poses[explorer]
                los_status[explorer]["distance_to_gcs"] = self.calculate_3d_distance(
                    {"x": drone_pose.position.x, "y": drone_pose.position.y, "z": drone_pose.position.z},
                    self.gcs_position,
                )

        for photographer in self.photographer_drones:
            los_status[photographer] = {"has_los": self.check_drone_los(photographer), "drone_type": "photographer"}
            if self.gcs_position is not None and photographer in self.model_poses:
                drone_pose = self.model_poses[photographer]
                los_status[photographer]["distance_to_gcs"] = self.calculate_3d_distance(
                    {"x": drone_pose.position.x, "y": drone_pose.position.y, "z": drone_pose.position.z},
                    self.gcs_position,
                )

        msg = String()
        msg.data = json.dumps(
            {
                "timestamp": self.get_clock().now().to_msg().sec,
                "los_threshold": self.los_threshold,
                "drone_status": los_status,
            }
        )
        self.los_status_pub.publish(msg)

        # Also publish in photographer_los_status format for backward compatibility
        photographer_msg = String()
        photographer_msg.data = json.dumps(
            {
                "timestamp": self.get_clock().now().to_msg().sec,
                "photographer_status": {
                    drone: status for drone, status in los_status.items() if drone in self.photographer_drones
                },
            }
        )
        self.photographer_los_pub.publish(photographer_msg)

    # LOS OBB/AABB Box Utilities

    def _load_los_boxes(self):
        yaml_path = (self.los_boxes_yaml or "").strip()
        if not yaml_path:
            share_dir = get_package_share_directory("mission_manager")

            # Try exact world name first (e.g., mbs_10_poi), then base name (e.g., mbs)
            base_world = self.world_name.split("_")[0] if "_" in self.world_name else self.world_name
            candidate_paths = [
                os.path.join(share_dir, "models", self.world_name, "bounding_boxes", "box_description.yaml"),
            ]
            if base_world != self.world_name:
                candidate_paths.append(
                    os.path.join(share_dir, "models", base_world, "bounding_boxes", "box_description.yaml")
                )

            yaml_path = next((c for c in candidate_paths if os.path.exists(c)), None)
            if not yaml_path:
                for cand in candidate_paths:
                    self.get_logger().warn(f"LOS boxes YAML not found at: {cand}")
                self._los_boxes = []
                return

        if not os.path.isabs(yaml_path):
            yaml_path = os.path.abspath(yaml_path)

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        boxes = []
        for key, val in data.items():
            center = tuple(float(x) for x in (val.get("center") or []))
            size = tuple(float(x) for x in (val.get("size") or []))
            orient = list(val.get("orientation") or [])
            if len(center) != 3 or len(size) != 3 or len(orient) != 16:
                continue
            # Extract rotation R (3x3) and translation t from 4x4 row-major matrix
            R = [
                [orient[0], orient[1], orient[2]],
                [orient[4], orient[5], orient[6]],
                [orient[8], orient[9], orient[10]],
            ]
            t = (orient[3], orient[7], orient[11])
            half = (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)
            boxes.append({"half": half, "R": R, "t": t})

        self._los_boxes = boxes
        if self._los_boxes:
            self.get_logger().info(f"Loaded {len(self._los_boxes)} LOS boxes from: {yaml_path}")
        else:
            self.get_logger().warn(f"No valid LOS boxes parsed from: {yaml_path}")

    @staticmethod
    def _sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _mat3_transpose(R):
        return [
            [R[0][0], R[1][0], R[2][0]],
            [R[0][1], R[1][1], R[2][1]],
            [R[0][2], R[1][2], R[2][2]],
        ]

    @staticmethod
    def _mat3_vec(R, v):
        return (
            R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
        )

    def _world_to_box_local(self, x, box):
        """Transform a world point x=(x,y,z) into the local coordinates of the OBB box."""
        # For a rigid transform (R, t), inverse is (R^T, -R^T t)
        R = box["R"]
        t = box["t"]
        Rt = self._mat3_transpose(R)
        v = self._sub(x, t)
        return self._mat3_vec(Rt, v)

    def _point_in_aabb_local(self, xl, half, eps):
        return (
            -half[0] - eps <= xl[0] <= half[0] + eps
            and -half[1] - eps <= xl[1] <= half[1] + eps
            and -half[2] - eps <= xl[2] <= half[2] + eps
        )

    def _segment_hits_aabb_local(self, p, q, half, eps):
        """Segment–AABB slab test in local coords with bounds [-half,+half] in each axis."""
        # Early degenerate check
        if abs(q[0] - p[0]) <= eps and abs(q[1] - p[1]) <= eps and abs(q[2] - p[2]) <= eps:
            return self._point_in_aabb_local(p, half, eps)

        # Endpoints inside -> treat as hit
        if self._point_in_aabb_local(p, half, eps) or self._point_in_aabb_local(q, half, eps):
            return True

        t0, t1 = 0.0, 1.0
        for d in range(3):
            pd = p[d]
            qd = q[d]
            ld = -half[d]
            ud = half[d]
            denom = qd - pd
            if abs(denom) <= eps:
                # Parallel to this slab: must already be inside the slab
                if pd < ld - eps or pd > ud + eps:
                    return False
                continue
            t_enter = (ld - pd) / denom
            t_exit = (ud - pd) / denom
            if t_enter > t_exit:
                t_enter, t_exit = t_exit, t_enter
            if t_enter > t0:
                t0 = t_enter
            if t_exit < t1:
                t1 = t_exit
            if t0 - t1 > eps:
                return False
        return t0 <= t1 + eps

    def _segment_blocked_by_any_box(self, p_world, q_world):
        eps = float(self.los_eps)
        for box in self._los_boxes:
            p_local = self._world_to_box_local(p_world, box)
            q_local = self._world_to_box_local(q_world, box)
            if self._segment_hits_aabb_local(p_local, q_local, box["half"], eps):
                return True
        return False

    def _px4_status_cb(self, drone_name: str, msg: VehicleStatus):
        self.px4_last_heartbeat[drone_name] = self.get_clock().now().nanoseconds / 1e9
        # PX4 failsafe bit with stickiness (require several false to clear)
        current = bool(msg.failsafe)
        if current:
            # Latch failsafe permanently once seen
            self.sticky_failsafe[drone_name] = True
        self.px4_failsafe[drone_name] = current

    def _px4_land_cb(self, drone_name: str, msg: VehicleLandDetected):
        self.land_detected[drone_name] = msg

    def publish_mission_status(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        status = {"timestamp": now_s, "drones": [], "pairwise_distances": []}
        eligibles = []

        los_map = {model: self.check_drone_los(model) for model in self.name_to_model.values()}
        positions = {}

        for d in self._fleet:
            name = d["name"]
            role = d.get("role", "")
            model = self.name_to_model.get(name, "")
            hb = self.px4_last_heartbeat.get(name, 0.0)
            heartbeat_ok = (now_s - hb) <= self.heartbeat_timeout_s if hb else False
            failsafe = bool(self.sticky_failsafe.get(name, self.px4_failsafe.get(name, False)))
            collision_recent = False  # contacts disabled
            has_los = bool(los_map.get(model, False))

            pos_dict = None
            if model in self.model_poses:
                pose = self.model_poses[model]
                pos_dict = {"x": float(pose.position.x), "y": float(pose.position.y), "z": float(pose.position.z)}
                positions[name] = pos_dict

            alive = heartbeat_ok and not failsafe and not collision_recent
            if alive and has_los:
                eligibles.append(name)

            status["drones"].append(
                {
                    "name": name,
                    "role": role,
                    "model": model,
                    "alive": alive,
                    "has_los": has_los,
                    "in_failsafe": failsafe,
                    "collision_recent": collision_recent,
                    "last_px4_heartbeat_sec": hb,
                    "position": pos_dict,
                }
            )

        # Compute pairwise inter-agent distances (3D and horizontal)
        drone_names = sorted(positions.keys())
        for i in range(len(drone_names)):
            for j in range(i + 1, len(drone_names)):
                n1, n2 = drone_names[i], drone_names[j]
                p1, p2 = positions[n1], positions[n2]
                dx, dy, dz = p1["x"] - p2["x"], p1["y"] - p2["y"], p1["z"] - p2["z"]
                status["pairwise_distances"].append(
                    {
                        "drone_1": n1,
                        "drone_2": n2,
                        "distance_3d": math.sqrt(dx * dx + dy * dy + dz * dz),
                        "distance_xy": math.sqrt(dx * dx + dy * dy),
                    }
                )

        msg = String()
        msg.data = json.dumps(status)
        self.drone_status_pub.publish(msg)

        elig = String()
        elig.data = json.dumps({"timestamp": now_s, "eligible": eligibles})
        self.eligible_pub.publish(elig)


def main(args=None):
    rclpy.init(args=args)
    node = RefereeNode()

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
