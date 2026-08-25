#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool


class GCSNode(Node):
    def __init__(self):
        super().__init__("gcs")

        self.explorer_drones = set()
        self.photographer_drones = set()
        self.model_by_name = {}
        self.name_by_model = {}

        self.processed_poi_ids = set()
        self.retained_pois_by_drone = {}
        self.published_poi_ids = set()
        self.passed_pois = []
        self.passed_poi_ids = set()
        self.poi_detection_run_id = None

        self.drone_los_status_by_model = {}
        self.drone_los_status_by_name = {}
        self.drone_alive_status_by_model = {}
        self.drone_alive_status_by_name = {}

        self.passed_poi_pub = self.create_publisher(String, "/poi_detected_passed", 10)

        self.poi_detection_sub = self.create_subscription(String, "/poi_detected_list", self.poi_detection_callback, 10)
        self.los_status_sub = self.create_subscription(String, "/drone_los_status", self.los_status_callback, 10)

        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.mission_status_sub = self.create_subscription(
            String, "/mission/drone_status", self.mission_status_callback, status_qos
        )

        self.los_check_service = self.create_service(
            SetBool, "/check_photographer_los", self.check_photographer_los_service
        )

        self.get_logger().info("GCS Node initialized")

        self.total_poi_received = 0
        self.total_poi_passed = 0
        self.total_poi_retained = 0

        self.create_timer(2.0, self.check_explorer_los_changes)
        self.create_timer(10.0, self.check_and_republish_pois)

    def los_status_callback(self, msg: String):
        status_data = json.loads(msg.data)
        drone_status = status_data.get("drone_status", {})
        for key, status in drone_status.items():
            has_los = status.get("has_los", False)
            self.drone_los_status_by_name[key] = has_los
            model = self.model_by_name.get(key)
            if model:
                self.drone_los_status_by_model[model] = has_los
            else:
                self.drone_los_status_by_model[key] = has_los

    def mission_status_callback(self, msg: String):
        data = json.loads(msg.data)
        drones = data.get("drones", []) or []
        new_explorers = set()
        new_photographers = set()
        for d in drones:
            name = d.get("name")
            model = d.get("model")
            role = d.get("role")
            alive = bool(d.get("alive", False))
            if name and model:
                self.model_by_name[name] = model
                self.name_by_model[model] = name
                self.drone_alive_status_by_name[name] = alive
                self.drone_alive_status_by_model[model] = alive
            if model and role:
                if role == "explorer":
                    new_explorers.add(model)
                elif role == "photographer":
                    new_photographers.add(model)
        self.explorer_drones = new_explorers
        self.photographer_drones = new_photographers
        for explorer in self.explorer_drones:
            if explorer not in self.retained_pois_by_drone:
                self.retained_pois_by_drone[explorer] = []

    def poi_detection_callback(self, msg: String):
        poi_data = json.loads(msg.data)
        if "detected_pois" not in poi_data:
            return
        run_id = poi_data.get("run_id")
        if not run_id:
            self.get_logger().warn("Ignoring POI detected list without run_id")
            return
        if self.poi_detection_run_id is None:
            self.poi_detection_run_id = run_id
        elif run_id != self.poi_detection_run_id:
            self.get_logger().warn("Ignoring POI detected list from a different run_id")
            return

        for poi in poi_data["detected_pois"]:
            poi_id = poi.get("id", "")
            if poi_id in self.processed_poi_ids:
                continue

            self.processed_poi_ids.add(poi_id)
            self.total_poi_received += 1

            detecting_drone = poi.get("detected_by", "")
            drone_model_name = self.get_drone_model_name(detecting_drone)

            if drone_model_name not in self.explorer_drones:
                self.get_logger().warn(f"POI {poi_id} detected by unknown explorer: {detecting_drone}")
                continue

            current_los = self.drone_los_status_by_model.get(drone_model_name, False)
            if current_los:
                self.send_poi_immediately(poi, detecting_drone)
            else:
                self.retain_poi_for_later(poi, drone_model_name, detecting_drone)

    def send_poi_immediately(self, poi_data, detecting_drone):
        poi_id = poi_data.get("id", "")
        if poi_id in self.published_poi_ids:
            return
        self.published_poi_ids.add(poi_id)
        self.publish_passed_poi(poi_data)
        self.total_poi_passed += 1
        self.get_logger().info(f"POI {poi_id} SENT IMMEDIATELY (detected by {detecting_drone} with LOS)")

    def retain_poi_for_later(self, poi_data, drone_model_name, detecting_drone):
        poi_id = poi_data.get("id", "")
        if poi_id in self.published_poi_ids:
            return
        self.retained_pois_by_drone.setdefault(drone_model_name, []).append(poi_data)
        self.total_poi_retained += 1
        self.get_logger().info(
            f"POI {poi_id} RETAINED (detected by {detecting_drone} - NO LOS, will send when LOS available)"
        )

    def check_explorer_los_changes(self):
        for explorer_drone in list(self.explorer_drones):
            current_los = self.drone_los_status_by_model.get(explorer_drone, False)
            previous_los = getattr(self, "_explorer_los_status", {}).get(explorer_drone, current_los)

            if not hasattr(self, "_explorer_los_status"):
                self._explorer_los_status = {}
            self._explorer_los_status[explorer_drone] = current_los

            if current_los and not previous_los:
                self.get_logger().info(f"Explorer {explorer_drone} GAINED LOS - sending retained POIs")
                self.send_retained_pois_for_explorer(explorer_drone)
            elif not current_los and previous_los:
                self.get_logger().info(f"Explorer {explorer_drone} LOST LOS")

    def send_retained_pois_for_explorer(self, explorer_drone):
        retained_pois = self.retained_pois_by_drone.get(explorer_drone, [])
        if not retained_pois:
            return

        sent_count = 0
        for poi_data in retained_pois:
            poi_id = poi_data.get("id", "")
            if poi_id in self.published_poi_ids:
                continue
            self.published_poi_ids.add(poi_id)
            self.publish_passed_poi(poi_data)
            self.total_poi_passed += 1
            sent_count += 1
            self.get_logger().info(f"POI {poi_id} SENT (retained from {explorer_drone})")

        self.get_logger().info(f"Sent {sent_count} retained POIs from {explorer_drone}")
        self.retained_pois_by_drone[explorer_drone] = []

    def check_photographer_los_service(self, request, response):
        photographer_with_los = self.get_photographers_with_los()
        if photographer_with_los:
            response.success = True
            response.message = f"Photographers with LOS: {photographer_with_los}"
        else:
            response.success = False
            response.message = "No photographers have LOS to GCS"
        return response

    def get_photographers_with_los(self):
        photographers_with_los = []
        for photographer_model in list(self.photographer_drones):
            has_los = self.drone_los_status_by_model.get(photographer_model, False)
            is_alive = self.drone_alive_status_by_model.get(photographer_model, True)
            if has_los and is_alive:
                photographers_with_los.append(photographer_model)
        return photographers_with_los

    def get_drone_model_name(self, detecting_drone):
        if detecting_drone in self.model_by_name:
            return self.model_by_name[detecting_drone]
        if detecting_drone in self.name_by_model:
            return detecting_drone
        return detecting_drone

    def publish_passed_poi(self, poi_data):
        poi_id = poi_data.get("id", "")
        if poi_id not in self.passed_poi_ids:
            self.passed_pois.append(poi_data)
            self.passed_poi_ids.add(poi_id)
            self.get_logger().info(f"Stored passed POI {poi_id} for re-publishing")

        passed_message = {"event": "poi_passed_los", "run_id": self.poi_detection_run_id, "poi_data": poi_data}
        msg = String()
        msg.data = json.dumps(passed_message)
        self.passed_poi_pub.publish(msg)

    def republish_available_pois(self):
        if not self.passed_pois:
            return
        for poi_data in self.passed_pois:
            passed_message = {"event": "poi_passed_los", "run_id": self.poi_detection_run_id, "poi_data": poi_data}
            msg = String()
            msg.data = json.dumps(passed_message)
            self.passed_poi_pub.publish(msg)

    def check_and_republish_pois(self):
        photographers_with_los = self.get_photographers_with_los()
        if photographers_with_los and self.passed_pois:
            self.republish_available_pois()


def main(args=None):
    rclpy.init(args=args)
    node = GCSNode()
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
