#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from datetime import datetime
from threading import Lock

import cv2
import gz.msgs10.image_pb2 as gz_image_msg
import gz.transport13 as gz_transport
import numpy as np
import rclpy
from rclpy.node import Node

from vehicle_controller_interfaces.srv import CapturePhoto

from .world_config import compute_gz_model_name, load_fleet, resolve_world_name


class PhotoCaptureService(Node):
    """ROS2 service node for on-demand photo capture from Gazebo camera sensors."""

    def __init__(self):
        super().__init__("photo_capture_service")
        self.declare_parameter("world_name", "")
        self.declare_parameter("difficulty", "easy")
        _world_key = self.get_parameter("world_name").get_parameter_value().string_value
        _difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        env_default = os.environ.get("PX4_GZ_WORLD", "mbs")
        self.world_name = resolve_world_name(_world_key or env_default, _difficulty)

        self.declare_parameter("roles", ["photographer"])
        _spawn_profile = _world_key or "mbs"
        _roles = list(self.get_parameter("roles").get_parameter_value().string_array_value) or ["photographer"]
        self.drone_configs = {}
        fleet = load_fleet(_spawn_profile)
        for d in fleet:
            role = str(d.get("role", ""))
            if role not in _roles:
                continue
            instance = int(d.get("instance", 0))
            model = str(d.get("model", ""))
            if not instance or not model:
                continue
            gz_model = compute_gz_model_name(model, instance)
            self.drone_configs[instance] = {"model": gz_model, "has_camera": True}

        if not self.drone_configs:
            self.get_logger().warn(f"No matching drones in profile '{_spawn_profile}' for roles {_roles}.")

        self.latest_images = {}
        self.image_locks = {}
        self.gz_node = gz_transport.Node()

        self.default_save_dir = os.path.expanduser("~/ros2_ws/src/mission_manager/captured_images")
        os.makedirs(self.default_save_dir, exist_ok=True)

        self._setup_image_subscribers()

        self.photo_service = self.create_service(CapturePhoto, "capture_photo", self.capture_photo_callback)

        self.get_logger().info(
            f"Photo capture service started. Drones: {list(self.drone_configs.keys())} "
            f"| Save dir: {self.default_save_dir}"
        )

    def _setup_image_subscribers(self):
        for drone_id, config in self.drone_configs.items():
            if config["has_camera"]:
                self.latest_images[drone_id] = None
                self.image_locks[drone_id] = Lock()
                gz_topic = (
                    f'/world/{self.world_name}/model/{config["model"]}/'
                    f"link/camera_link_photographer/sensor/camera/image"
                )
                callback = self._create_image_callback(drone_id)
                success = self.gz_node.subscribe(gz_image_msg.Image, gz_topic, callback)
                if success:
                    self.get_logger().info(f"Subscribed to camera for drone {drone_id}: {gz_topic}")
                else:
                    self.get_logger().error(f"Failed to subscribe to camera for drone {drone_id}")

    def _create_image_callback(self, drone_id):
        def image_callback(msg):
            if not (hasattr(msg, "width") and hasattr(msg, "height") and hasattr(msg, "data")):
                return
            width = msg.width
            height = msg.height
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            if len(image_data) != width * height * 3:
                return
            image_array = image_data.reshape((height, width, 3))
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            with self.image_locks[drone_id]:
                self.latest_images[drone_id] = {
                    "image": image_bgr,
                    "timestamp": self.get_clock().now(),
                    "width": width,
                    "height": height,
                }

        return image_callback

    def capture_photo_callback(self, request, response):
        drone_id = request.drone_id
        if drone_id not in self.drone_configs:
            response.success = False
            response.message = f"Invalid drone ID: {drone_id}. Available: {list(self.drone_configs.keys())}"
            return response

        if not self.drone_configs[drone_id]["has_camera"]:
            response.success = False
            response.message = f"Drone {drone_id} does not have a camera"
            return response

        with self.image_locks[drone_id]:
            if self.latest_images[drone_id] is None:
                response.success = False
                response.message = f"No image available from drone {drone_id}. Is the simulation running?"
                return response
            image_data = self.latest_images[drone_id].copy()

        save_dir = request.save_directory if request.save_directory else self.default_save_dir
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = (
            f"{request.filename_prefix}_{timestamp_str}.jpg"
            if request.filename_prefix
            else f"drone_{drone_id}_{timestamp_str}.jpg"
        )
        image_path = os.path.join(save_dir, filename)

        if not cv2.imwrite(image_path, image_data["image"]):
            response.success = False
            response.message = f"Failed to save image to {image_path}"
            return response

        metadata_path = ""
        if request.include_metadata:
            metadata = {
                "drone_id": drone_id,
                "timestamp": timestamp.isoformat(),
                "image_path": image_path,
                "image_dimensions": {"width": image_data["width"], "height": image_data["height"]},
                "capture_time_ros": {
                    "sec": image_data["timestamp"].nanoseconds // 1000000000,
                    "nanosec": image_data["timestamp"].nanoseconds % 1000000000,
                },
            }
            metadata_filename = filename.replace(".jpg", "_metadata.json")
            metadata_path = os.path.join(save_dir, metadata_filename)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        response.success = True
        response.message = f"Photo captured successfully from drone {drone_id}"
        response.image_path = image_path
        response.metadata_path = metadata_path
        response.timestamp = self.get_clock().now().to_msg()
        return response


def main(args=None):
    rclpy.init(args=args)
    photo_service = PhotoCaptureService()
    try:
        rclpy.spin(photo_service)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            photo_service.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
