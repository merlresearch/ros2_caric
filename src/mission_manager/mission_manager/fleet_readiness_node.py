#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
"""Publish PX4 readiness gates for the benchmark fleet."""

import json
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .world_config import compute_gz_model_name, load_fleet


@dataclass
class DroneReadiness:
    instance: int
    name: str
    role: str
    model: str
    status: Optional[VehicleStatus] = None
    position: Optional[VehicleLocalPosition] = None
    status_received_at: float = 0.0
    position_received_at: float = 0.0
    ready: bool = False
    reason: str = "waiting_for_status"


class FleetReadinessNode(Node):
    def __init__(self):
        super().__init__("fleet_readiness_node")

        self.declare_parameter("world_name", "mbs")
        self.declare_parameter("status_publish_period_sec", 0.5)
        self.declare_parameter("stale_after_sec", 3.0)
        self.declare_parameter("require_preflight_checks", True)
        self.declare_parameter("require_heading_good_for_control", False)

        self.world_name = self.get_parameter("world_name").get_parameter_value().string_value or "mbs"
        self.stale_after_sec = float(self.get_parameter("stale_after_sec").value)
        self.require_preflight_checks = bool(self.get_parameter("require_preflight_checks").value)
        self.require_heading_good_for_control = bool(self.get_parameter("require_heading_good_for_control").value)

        self.drones: Dict[int, DroneReadiness] = {}
        for cfg in sorted(load_fleet(self.world_name), key=lambda item: int(item.get("instance", 999))):
            instance = int(cfg.get("instance", 0))
            if instance <= 0:
                continue
            role = str(cfg.get("role", "")).lower()
            model = str(cfg.get("model", ""))
            name = str(cfg.get("name") or compute_gz_model_name(model, instance))
            self.drones[instance] = DroneReadiness(
                instance=instance,
                name=name,
                role=role,
                model=model,
            )

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.fleet_ready_pub = self.create_publisher(Bool, "/fleet_ready", latched_qos)
        self.photographer_ready_pub = self.create_publisher(Bool, "/photographer_fleet_ready", latched_qos)
        self.explorer_ready_pub = self.create_publisher(Bool, "/explorer_fleet_ready", latched_qos)
        self.status_pub = self.create_publisher(String, "/fleet_readiness_status", latched_qos)

        for instance in self.drones:
            prefix = f"/px4_{instance}"
            self.create_subscription(
                VehicleStatus,
                f"{prefix}/fmu/out/vehicle_status_v1",
                lambda msg, idx=instance: self._status_cb(idx, msg),
                px4_qos,
            )
            self.create_subscription(
                VehicleStatus,
                f"{prefix}/fmu/out/vehicle_status",
                lambda msg, idx=instance: self._status_cb(idx, msg),
                px4_qos,
            )
            self.create_subscription(
                VehicleLocalPosition,
                f"{prefix}/fmu/out/vehicle_local_position",
                lambda msg, idx=instance: self._position_cb(idx, msg),
                px4_qos,
            )

        period = max(0.1, float(self.get_parameter("status_publish_period_sec").value))
        self._last_all_ready = None
        self._last_photographer_ready = None
        self._last_explorer_ready = None
        self.create_timer(period, self._publish_status)
        self.get_logger().info(f"Fleet readiness watching {len(self.drones)} drones for world '{self.world_name}'")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _status_cb(self, instance: int, msg: VehicleStatus) -> None:
        drone = self.drones.get(instance)
        if drone is None:
            return
        drone.status = msg
        drone.status_received_at = self._now_s()

    def _position_cb(self, instance: int, msg: VehicleLocalPosition) -> None:
        drone = self.drones.get(instance)
        if drone is None:
            return
        drone.position = msg
        drone.position_received_at = self._now_s()

    def _evaluate_drone(self, drone: DroneReadiness, now_s: float) -> bool:
        status = drone.status
        pos = drone.position

        if status is None or now_s - drone.status_received_at > self.stale_after_sec:
            drone.ready = False
            drone.reason = "waiting_for_status"
            return False

        if pos is None or now_s - drone.position_received_at > self.stale_after_sec:
            drone.ready = False
            drone.reason = "waiting_for_local_position"
            return False

        if status.timestamp == 0:
            drone.ready = False
            drone.reason = "status_timestamp_zero"
            return False

        if self.require_preflight_checks and not bool(status.pre_flight_checks_pass):
            drone.ready = False
            drone.reason = "preflight_checks_not_passed"
            return False

        if bool(status.failsafe):
            drone.ready = False
            drone.reason = "failsafe_active"
            return False

        if pos.timestamp == 0:
            drone.ready = False
            drone.reason = "position_timestamp_zero"
            return False

        if not (bool(pos.xy_valid) and bool(pos.z_valid)):
            drone.ready = False
            drone.reason = "local_position_not_valid"
            return False

        if self.require_heading_good_for_control and not bool(pos.heading_good_for_control):
            drone.ready = False
            drone.reason = "heading_not_good_for_control"
            return False

        drone.ready = True
        drone.reason = "ready"
        return True

    def _ready_for_role(self, role: Optional[str]) -> bool:
        selected = [
            drone for drone in self.drones.values() if role is None or drone.role == role or role in drone.model.lower()
        ]
        return bool(selected) and all(drone.ready for drone in selected)

    def _publish_bool(self, publisher, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        publisher.publish(msg)

    def _publish_status(self) -> None:
        self.require_preflight_checks = bool(self.get_parameter("require_preflight_checks").value)
        self.require_heading_good_for_control = bool(self.get_parameter("require_heading_good_for_control").value)
        now_s = self._now_s()
        for drone in self.drones.values():
            self._evaluate_drone(drone, now_s)

        all_ready = self._ready_for_role(None)
        photographer_ready = self._ready_for_role("photographer")
        explorer_ready = self._ready_for_role("explorer")

        self._publish_bool(self.fleet_ready_pub, all_ready)
        self._publish_bool(self.photographer_ready_pub, photographer_ready)
        self._publish_bool(self.explorer_ready_pub, explorer_ready)

        payload = {
            "fleet_ready": all_ready,
            "photographer_fleet_ready": photographer_ready,
            "explorer_fleet_ready": explorer_ready,
            "ready_count": sum(1 for drone in self.drones.values() if drone.ready),
            "total_count": len(self.drones),
            "drones": {
                drone.name: {
                    "instance": drone.instance,
                    "role": drone.role,
                    "model": drone.model,
                    "ready": drone.ready,
                    "reason": drone.reason,
                }
                for drone in self.drones.values()
            },
            "timestamp": now_s,
        }
        status_msg = String()
        status_msg.data = json.dumps(payload)
        self.status_pub.publish(status_msg)

        changed = (
            all_ready != self._last_all_ready
            or photographer_ready != self._last_photographer_ready
            or explorer_ready != self._last_explorer_ready
        )
        if changed:
            self.get_logger().info(
                "Readiness: "
                f"fleet={all_ready} photographers={photographer_ready} explorers={explorer_ready} "
                f"({payload['ready_count']}/{payload['total_count']} ready)"
            )
            self._last_all_ready = all_ready
            self._last_photographer_ready = photographer_ready
            self._last_explorer_ready = explorer_ready


def main(args=None):
    rclpy.init(args=args)
    node = FleetReadinessNode()
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
