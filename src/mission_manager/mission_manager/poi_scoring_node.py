#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import json
import os
import threading
import xml.etree.ElementTree as ET
from datetime import datetime

import gz.transport13 as gz_transport
import numpy as np
import rclpy
from gz.msgs10.logical_camera_image_pb2 import LogicalCameraImage
from gz.msgs10.pose_v_pb2 import Pose_V
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String

from vehicle_controller_interfaces.srv import CapturePhoto

from .world_config import resolve_world_name


class POIScoringNode(Node):
    def __init__(self):
        super().__init__("poi_scoring_node")
        self.declare_parameter("world_name", "")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("world_model_source", "known")
        self.declare_parameter("run_style", "full")
        self.declare_parameter("planner_type", "openscvx")  # 'openscvx' or 'astar'
        world_key = self.get_parameter("world_name").get_parameter_value().string_value
        self.difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        self.world_model_source = self.get_parameter("world_model_source").get_parameter_value().string_value
        self.run_style = self.get_parameter("run_style").get_parameter_value().string_value
        self.planner_type = self.get_parameter("planner_type").get_parameter_value().string_value or "openscvx"
        env_default = os.environ.get("PX4_GZ_WORLD", "mbs")
        self.world_name = resolve_world_name(world_key or env_default, self.difficulty)
        self.world_base = world_key or env_default

        self.callback_group = ReentrantCallbackGroup()  # important for the photo service client
        self.poi_reference_list = self.load_poi_from_world()

        self.model_poses = {}
        self.drone_poses = {}
        self.drone_velocities = {}

        self.poi_scores = {}
        self.setup_logging()

        self.photo_client = self.create_client(CapturePhoto, "capture_photo", callback_group=self.callback_group)
        self.photo_captured_scores = {}

        # Camera parameters from SDF (1280x720, hfov=80°, aspect=16:9, sensor_width=6.17mm)
        image_width = 1280
        image_height = 720
        self.fov_h = 80.0
        aspect_ratio = image_width / image_height
        fov_h_rad = np.radians(self.fov_h)
        self.fov_v = np.degrees(2.0 * np.arctan(np.tan(fov_h_rad / 2.0) / aspect_ratio))
        self.focal_length_pixels = (image_width / 2.0) / np.tan(fov_h_rad / 2.0)
        sensor_width_m = 6.17e-3
        self.pixel_size_m = sensor_width_m / image_width
        self.focal_length_m = self.focal_length_pixels * self.pixel_size_m

        self.declare_parameter("max_res_depth_m", 7.5)
        self.max_res_depth_m = float(self.get_parameter("max_res_depth_m").get_parameter_value().double_value)

        self.min_score_threshold = 0.2
        self.capture_interval = 0.1
        self.exposure_time = 0.001  # 1ms exposure time
        self.max_blur_pixels = 1.0

        self.gz_node = gz_transport.Node()

        self.score_pub = self.create_publisher(Float32MultiArray, "poi_scores", 10)
        # Use TRANSIENT_LOCAL QoS to match mission_controller subscriber
        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.total_score_pub = self.create_publisher(String, "total_score_status", transient_qos)

        self.pose_topic = f"/world/{self.world_name}/pose/info"
        if not self.gz_node.subscribe(Pose_V, self.pose_topic, self.pose_callback):
            self.get_logger().error(f"Failed to subscribe to {self.pose_topic}")
        else:
            self.get_logger().info(f"Subscribed to pose topic: {self.pose_topic}")

        self.camera_topics = {
            "Drone_2": (
                f"/world/{self.world_name}/model/"
                "x500_gimbal_photographer_2/link/camera_link_photographer/"
                "sensor/logic_wide/logical_camera"
            ),
            "Drone_4": (
                f"/world/{self.world_name}/model/"
                "x500_gimbal_photographer_4/link/camera_link_photographer/"
                "sensor/logic_wide/logical_camera"
            ),
            "Drone_5": (
                f"/world/{self.world_name}/model/"
                "x500_gimbal_photographer_5/link/camera_link_photographer/"
                "sensor/logic_wide/logical_camera"
            ),
        }

        for drone_id, topic in self.camera_topics.items():
            if not self.gz_node.subscribe(
                LogicalCameraImage, topic, lambda msg, did=drone_id: self.logical_camera_callback(msg, did)
            ):
                self.get_logger().error(f"Failed to subscribe to {topic}")
            else:
                self.get_logger().info(f"Subscribed to {drone_id}: {topic}")

        self.last_capture_time = {"Drone_2": 0.0, "Drone_4": 0.0, "Drone_5": 0.0}

        poi_count = len(self.poi_reference_list)
        self.get_logger().info(f"Loaded {poi_count} POI cubes from world file")
        poi_ids = [poi["id"] for poi in self.poi_reference_list]
        self.get_logger().info(f"POI Reference List: {poi_ids}")
        self.create_timer(2.0, self.publish_score_updates)

    def pose_callback(self, msg: Pose_V):
        for pose in msg.pose:
            self.model_poses[pose.name] = pose
            if pose.name in ["x500_gimbal_photographer_2", "x500_gimbal_photographer_4", "x500_gimbal_photographer_5"]:
                drone_id = f'Drone_{pose.name.split("_")[-1]}'
                current_time = self.get_clock().now().nanoseconds / 1e9
                if drone_id in self.drone_poses:
                    dt = current_time - self.drone_poses[drone_id]["timestamp"]
                    if dt > 0:
                        dx = pose.position.x - self.drone_poses[drone_id]["x"]
                        dy = pose.position.y - self.drone_poses[drone_id]["y"]
                        dz = pose.position.z - self.drone_poses[drone_id]["z"]
                        self.drone_velocities[drone_id] = {
                            "vx": dx / dt,
                            "vy": dy / dt,
                            "vz": dz / dt,
                            "timestamp": current_time,
                        }
                self.drone_poses[drone_id] = {
                    "x": pose.position.x,
                    "y": pose.position.y,
                    "z": pose.position.z,
                    "timestamp": current_time,
                }

    def load_poi_from_world(self):
        poi_list = []

        from ament_index_python.packages import get_package_share_directory

        share_dir = get_package_share_directory("mission_manager")
        candidate = os.path.join(share_dir, "worlds", f"{self.world_name}.sdf")
        world_file = candidate if os.path.exists(candidate) else None

        if world_file is None:
            for base_path in [
                os.path.expanduser("~/ros2_ws/src/PX4-Autopilot/Tools/simulation/gz/worlds"),
                "src/PX4-Autopilot/Tools/simulation/gz/worlds",
            ]:
                path = os.path.join(base_path, f"{self.world_name}.sdf")
                if os.path.exists(path):
                    world_file = path
                    break

        if world_file is None:
            self.get_logger().error(f"Could not find world file for {self.world_name}.sdf")
            return poi_list

        tree = ET.parse(world_file)
        root = tree.getroot()
        for model in root.findall(".//model"):
            name = model.get("name")
            if name and (name.startswith("poi_") or name.startswith("fp_poi_")):
                pose_elem = model.find("pose")
                if pose_elem is not None:
                    pose_data = pose_elem.text.strip().split()
                    if len(pose_data) >= 6:
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

    def logical_camera_callback(self, msg: LogicalCameraImage, drone_id: str):
        self.process_camera_data(msg, drone_id)

    def process_camera_data(self, msg: LogicalCameraImage, drone_id: str):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if drone_id not in self.last_capture_time:
            self.last_capture_time[drone_id] = 0.0
        if current_time - self.last_capture_time[drone_id] < self.capture_interval:
            return
        self.last_capture_time[drone_id] = current_time

        if not msg.model or drone_id not in self.drone_poses:
            return

        drone_pose = self.drone_poses[drone_id]
        drone_velocity = self.drone_velocities.get(drone_id, {"vx": 0.0, "vy": 0.0, "vz": 0.0})

        for model in msg.model:
            model_name = model.name
            if not model_name.startswith("poi_"):
                continue

            score_data = self.calculate_poi_score(model, drone_pose, drone_velocity, drone_id)
            score = score_data["score"]
            q_blur = score_data["q_blur"]
            q_res = score_data["q_res"]
            q_area = score_data["q_area"]

            is_new_poi = model_name not in self.poi_scores
            is_better_score = not is_new_poi and score > self.poi_scores[model_name]["score"]

            if is_new_poi and score <= 0.0:
                continue

            if is_new_poi or is_better_score:
                self.poi_scores[model_name] = {
                    "score": score,
                    "q_blur": q_blur,
                    "q_res": q_res,
                    "q_area": q_area,
                    "detected_by": drone_id,
                    "timestamp": current_time,
                    "position": {"x": model.pose.position.x, "y": model.pose.position.y, "z": model.pose.position.z},
                    "debug": score_data.get("debug"),
                }
                self.log_score_to_csv(model_name, score, q_blur, q_res, drone_id, model.pose)

                should_capture = score > self.min_score_threshold and (
                    model_name not in self.photo_captured_scores or score > self.photo_captured_scores[model_name]
                )
                if should_capture:
                    if model_name not in self.photo_captured_scores:
                        self.get_logger().info(f"First photo of {model_name} with score {score:.3f}")
                    self.capture_photo_for_poi(model_name, drone_id, score)
            else:
                self.get_logger().debug(
                    f"POI {model_name} score {score:.3f} not better than {self.poi_scores[model_name]['score']:.3f}"
                )

    def calculate_poi_score(self, model, drone_pose, drone_velocity, drone_id):
        if model.name in self.model_poses:
            world_pose = self.model_poses[model.name]
            poi_world_pos = np.array([world_pose.position.x, world_pose.position.y, world_pose.position.z])
        else:
            poi_world_pos = np.array([model.pose.position.x, model.pose.position.y, model.pose.position.z])

        drone_pos = np.array([drone_pose["x"], drone_pose["y"], drone_pose["z"]])
        distance = np.linalg.norm(poi_world_pos - drone_pos)

        if distance == 0:
            return {"score": 0.0, "q_blur": 0.0, "q_res": 0.0, "q_area": 0.0}

        q_blur, blur_dbg = self.blur_metric(poi_world_pos, drone_pos, drone_velocity, distance)
        depth = float(model.pose.position.x)  # logical camera frame: X is forward
        q_res, res_dbg = self.resolution_metric(depth)
        q_area, area_dbg = self.area_visibility_metric(distance, model)

        score = q_blur * q_res * q_area
        if score < 0.2:
            score = 0.0

        debug_inputs = {
            "distance": float(distance),
            "depth": float(abs(depth)),
            **(blur_dbg or {}),
            **(res_dbg or {}),
            "poi_rel_x": float(model.pose.position.y),
            "poi_rel_y": float(model.pose.position.z),
            **(area_dbg or {}),
        }
        return {"score": score, "q_blur": q_blur, "q_res": q_res, "q_area": q_area, "debug": debug_inputs}

    def blur_metric(self, poi_world_pos, drone_pos, drone_velocity, distance):
        ray = poi_world_pos - drone_pos
        d = np.linalg.norm(ray)
        if d == 0:
            return 1.0, {
                "drone_vx": float(drone_velocity.get("vx", 0.0)),
                "drone_vy": float(drone_velocity.get("vy", 0.0)),
                "drone_vz": float(drone_velocity.get("vz", 0.0)),
                "v_perp_mag": 0.0,
                "motion_shift_px": 0.0,
                "exposure_time_s": float(self.exposure_time),
                "max_blur_px": float(self.max_blur_pixels),
            }
        view_dir = ray / d
        v = np.array([drone_velocity["vx"], drone_velocity["vy"], drone_velocity["vz"]])
        v_perp = v - np.dot(v, view_dir) * view_dir
        v_perp_mag = np.linalg.norm(v_perp)
        gsd_m_per_px = (d * self.pixel_size_m) / self.focal_length_m
        shift = (v_perp_mag * self.exposure_time) / max(gsd_m_per_px, 1e-6)
        q_blur = max(0.0, 1.0 - shift / self.max_blur_pixels)
        debug = {
            "drone_vx": float(drone_velocity.get("vx", 0.0)),
            "drone_vy": float(drone_velocity.get("vy", 0.0)),
            "drone_vz": float(drone_velocity.get("vz", 0.0)),
            "v_perp_mag": float(v_perp_mag),
            "motion_shift_px": float(shift),
            "exposure_time_s": float(self.exposure_time),
            "max_blur_px": float(self.max_blur_pixels),
        }
        return q_blur, debug

    def resolution_metric(self, depth):
        """q_res = 1.0 at or closer than max_res_depth_m, decreases proportionally beyond that."""
        d = max(abs(float(depth)), 1e-6)
        gsd_mm_per_px = (d * self.pixel_size_m) / self.focal_length_m * 1000.0
        gsd_threshold_mm_per_px = (self.max_res_depth_m * self.pixel_size_m) / self.focal_length_m * 1000.0
        q_res = min(gsd_threshold_mm_per_px / max(gsd_mm_per_px, 1e-9), 1.0)
        debug = {
            "gsd_mm_per_px": float(gsd_mm_per_px),
            "target_depth_m": float(self.max_res_depth_m),
            "target_gsd_mm_per_px": float(gsd_threshold_mm_per_px),
        }
        return q_res, debug

    def area_visibility_metric(self, distance, model):
        """Compute visible fraction of POI in the camera frame (logical camera: X-forward, Y-horizontal, Z-vertical)."""
        poi_size = 0.5  # POIs are 0.5x0.5x0.5m cubes
        depth = max(abs(float(model.pose.position.x)), 1e-6)
        fov_width = 2.0 * depth * np.tan(np.radians(self.fov_h) / 2.0)
        fov_height = 2.0 * depth * np.tan(np.radians(self.fov_v) / 2.0)

        poi_rel_x = model.pose.position.y
        poi_rel_y = model.pose.position.z
        poi_half = poi_size / 2.0

        poi_left, poi_right = poi_rel_x - poi_half, poi_rel_x + poi_half
        poi_top, poi_bottom = poi_rel_y - poi_half, poi_rel_y + poi_half
        view_left, view_right = -fov_width / 2.0, fov_width / 2.0
        view_top, view_bottom = -fov_height / 2.0, fov_height / 2.0

        if poi_right <= view_left or poi_left >= view_right or poi_bottom <= view_top or poi_top >= view_bottom:
            return 0.0, {"fov_width_m": float(fov_width), "fov_height_m": float(fov_height), "visibility_ratio": 0.0}

        if poi_left >= view_left and poi_right <= view_right and poi_top >= view_top and poi_bottom <= view_bottom:
            visibility_ratio = 1.0
        else:
            visible_width = max(0.0, min(poi_right, view_right) - max(poi_left, view_left))
            visible_height = max(0.0, min(poi_bottom, view_bottom) - max(poi_top, view_top))
            visibility_ratio = (visible_width * visible_height) / (poi_size * poi_size)

        q_area = min(max(visibility_ratio, 0.0), 1.0)
        debug = {
            "fov_width_m": float(fov_width),
            "fov_height_m": float(fov_height),
            "visibility_ratio": float(visibility_ratio),
        }
        return q_area, debug

    def publish_score_updates(self):
        score_msg = Float32MultiArray()
        score_data = []
        total_score = 0.0

        for poi_id, score_data_dict in self.poi_scores.items():
            poi_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            if poi_id in self.model_poses:
                world_pose = self.model_poses[poi_id]
                poi_pos = {"x": world_pose.position.x, "y": world_pose.position.y, "z": world_pose.position.z}
            poi_id_num = int(poi_id.split("_")[-1])
            score_data.extend([float(poi_id_num), poi_pos["x"], poi_pos["y"], poi_pos["z"], score_data_dict["score"]])
            total_score += score_data_dict["score"]

        score_msg.data = score_data
        self.score_pub.publish(score_msg)

        detected_count = len(self.poi_scores)
        status_msg = String()
        status_msg.data = json.dumps(
            {
                "total_score": total_score,
                "detected_count": detected_count,
                "total_poi_count": len(self.poi_reference_list),
                "completion_percentage": (
                    (detected_count / len(self.poi_reference_list) * 100) if self.poi_reference_list else 0.0
                ),
            }
        )
        self.total_score_pub.publish(status_msg)

    def get_total_score(self):
        return sum(d["score"] for d in self.poi_scores.values())

    def get_detected_count(self):
        return len(self.poi_scores)

    def setup_logging(self):
        self._csv_lock = threading.Lock()
        results_root = os.path.expanduser(
            os.environ.get(
                "MISSION_RESULTS_ROOT",
                os.path.expanduser("~/ros2_ws/src/mission_manager/results"),
            )
        )
        result_group = os.environ.get("MISSION_RESULTS_GROUP", "").strip()
        if result_group:
            log_dir = os.path.join(results_root, self.planner_type, result_group, "scoring")
        else:
            log_dir = os.path.join(results_root, self.planner_type, "scoring")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d_code = {"easy": "e", "medium": "m", "hard": "h"}.get(self.difficulty, "e")
        s_code = {"known": "k", "lidar_only": "l"}.get(self.world_model_source, "k")
        r_code = {"full": "f", "timed": "t"}.get(self.run_style, "f")
        # Format: poi_scores_{world}_{difficulty}{source}{run_style}_{timestamp}.csv
        config_str = f"{self.world_base}_{d_code}{s_code}{r_code}"
        self.log_file_path = os.path.join(log_dir, f"poi_scores_{config_str}_{timestamp}.csv")
        with open(self.log_file_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "timestamp",
                    "poi_id",
                    "score",
                    "q_blur",
                    "q_res",
                    "q_area",
                    "detected_by",
                    "distance_m",
                    "depth_m",
                    "poi_x",
                    "poi_y",
                    "poi_z",
                    "drone_x",
                    "drone_y",
                    "drone_z",
                    "drone_vx",
                    "drone_vy",
                    "drone_vz",
                    "v_perp_mag",
                    "motion_shift_px",
                    "exposure_time_s",
                    "max_blur_px",
                    "gsd_mm_per_px",
                    "target_depth_m",
                    "target_gsd_mm_per_px",
                    "poi_rel_x",
                    "poi_rel_y",
                    "fov_width_m",
                    "fov_height_m",
                    "visibility_ratio",
                    "camera_pixel_size_m",
                    "camera_focal_length_m",
                    "fov_h_deg",
                    "fov_v_deg",
                    "poi_size_m",
                    "max_res_depth_m",
                    "image_path",
                ]
            )

    def log_score_to_csv(self, poi_id, score, q_blur, q_res, drone_id, poi_pose):
        """Keep only the highest-score row per POI — update existing row or append new one."""
        if not self.log_file_path:
            return

        with self._csv_lock:
            timestamp = datetime.now().isoformat()
            drone_pos = self.drone_poses.get(drone_id, {"x": 0.0, "y": 0.0, "z": 0.0})
            distance_calc = np.sqrt(
                (poi_pose.position.x - drone_pos["x"]) ** 2
                + (poi_pose.position.y - drone_pos["y"]) ** 2
                + (poi_pose.position.z - drone_pos["z"]) ** 2
            )
            image_path = ""
            if poi_id in self.photo_captured_scores:
                score_str = f"{self.photo_captured_scores[poi_id]:.3f}".replace(".", "_")
                q_area_val = self.poi_scores.get(poi_id, {}).get("q_area", 0.0)
                q_area_str = f"{q_area_val:.3f}".replace(".", "_")
                image_path = f"captured_images/{poi_id}_{drone_id}_score_{score_str}_area_{q_area_str}.jpg"

            q_area = self.poi_scores.get(poi_id, {}).get("q_area", 0.0)
            debug = self.poi_scores.get(poi_id, {}).get("debug", {})
            distance = float(debug.get("distance", distance_calc))
            depth = (
                float(abs(self.model_poses.get(poi_id, poi_pose).position.z))
                if poi_id in self.model_poses
                else float(abs(poi_pose.position.z))
            )

            new_row = [
                timestamp,
                poi_id,
                score,
                q_blur,
                q_res,
                q_area,
                drone_id,
                distance,
                debug.get("depth", depth),
                poi_pose.position.x,
                poi_pose.position.y,
                poi_pose.position.z,
                drone_pos["x"],
                drone_pos["y"],
                drone_pos["z"],
                debug.get("drone_vx", self.drone_velocities.get(drone_id, {}).get("vx", 0.0)),
                debug.get("drone_vy", self.drone_velocities.get(drone_id, {}).get("vy", 0.0)),
                debug.get("drone_vz", self.drone_velocities.get(drone_id, {}).get("vz", 0.0)),
                debug.get("v_perp_mag", 0.0),
                debug.get("motion_shift_px", 0.0),
                debug.get("exposure_time_s", self.exposure_time),
                debug.get("max_blur_px", self.max_blur_pixels),
                debug.get("gsd_mm_per_px", 0.0),
                debug.get("target_depth_m", self.max_res_depth_m),
                debug.get("target_gsd_mm_per_px", 0.0),
                debug.get("poi_rel_x", getattr(poi_pose.position, "x", 0.0)),
                debug.get("poi_rel_y", getattr(poi_pose.position, "y", 0.0)),
                debug.get("fov_width_m", 0.0),
                debug.get("fov_height_m", 0.0),
                debug.get("visibility_ratio", 0.0),
                self.pixel_size_m,
                self.focal_length_m,
                self.fov_h,
                self.fov_v,
                0.5,
                self.max_res_depth_m,
                image_path,
            ]

            with open(self.log_file_path, "r", newline="") as f:
                rows = list(csv.reader(f))

            # Find existing row index for this POI (search from end for efficiency)
            update_idx = next((i for i in range(len(rows) - 1, 0, -1) if len(rows[i]) > 1 and rows[i][1] == poi_id), -1)
            row_str = [str(x) if isinstance(x, (int, float)) else x for x in new_row]
            if update_idx > 0:
                rows[update_idx] = row_str
            else:
                rows.append(row_str)

            with open(self.log_file_path, "w", newline="") as f:
                csv.writer(f).writerows(rows)

    def capture_photo_for_poi(self, poi_id, drone_id, score):
        if not self.photo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Photo capture service not available")
            return

        drone_num = {"Drone_2": 2, "Drone_4": 4, "Drone_5": 5}.get(drone_id, 0)
        request = CapturePhoto.Request()
        request.drone_id = drone_num
        request.save_directory = os.path.expanduser("~/ros2_ws/src/mission_manager/captured_images")
        request.filename_prefix = f"{poi_id}_{drone_id}_score_{f'{score:.3f}'.replace('.', '_')}"
        request.include_metadata = True
        future = self.photo_client.call_async(request)
        future.add_done_callback(lambda f: self.photo_callback(f, poi_id, drone_id, score))

    def photo_callback(self, future, poi_id, drone_id, score):
        response = future.result()
        if response.success:
            self.photo_captured_scores[poi_id] = score
            if self.log_file_path and poi_id in self.poi_scores:
                self.update_csv_with_image_path(poi_id, response.image_path)
        else:
            self.get_logger().error(f"Photo capture failed for POI {poi_id}: {response.message}")

    def update_csv_with_image_path(self, poi_id, image_path):
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return

        with self._csv_lock:
            # Read with NUL character handling for robustness
            with open(self.log_file_path, "r", newline="", errors="replace") as f:
                rows = list(csv.reader(f.read().replace("\x00", "").splitlines()))

            if len(rows) < 2:
                return

            for i in range(len(rows) - 1, 0, -1):
                if len(rows[i]) >= 2 and rows[i][1] == poi_id:
                    rows[i][-1] = image_path
                    break

            with open(self.log_file_path, "w", newline="") as f:
                csv.writer(f).writerows(rows)


def main(args=None):
    rclpy.init(args=args)
    node = POIScoringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(
            f"Final Scoring Results: {node.get_detected_count()} POIs scored, Total Score: {node.get_total_score():.3f}"
        )
        if node.log_file_path:
            print(f"Score log saved to: {node.log_file_path}")
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
