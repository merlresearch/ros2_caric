#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os
import signal
import subprocess
import time
from contextlib import suppress

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from .world_config import load_fleet, resolve_world_name


class Spawner(Node):
    def __init__(self):
        super().__init__("spawner")
        self.declare_parameter("world_name", "mbs")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("resource_root", os.path.expanduser("~/ros2_ws/src/PX4-Autopilot/Tools/simulation/gz"))
        self.declare_parameter("agent_port", 8888)
        # Declare as string array to match launch-provided JSON strings
        self.declare_parameter("fleet", [""])

        self.processes = []
        self.agent_process = None

        # Resolve world now so links/env reflect difficulty
        _init_world_key = self.get_parameter("world_name").get_parameter_value().string_value
        _init_difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        self.world_name_resolved = resolve_world_name(
            _init_world_key or os.environ.get("PX4_GZ_WORLD", "mbs"), _init_difficulty
        )
        # spawn_profile is derived from world (they're always the same)
        self._spawn_profile = _init_world_key or "mbs"

        self._setup_resources()
        self._start_agent()
        self._spawn_fleet()

        self.get_logger().info("Spawner initialized")

    def _setup_resources(self):
        root = self.get_parameter("resource_root").get_parameter_value().string_value
        # Include mission_manager packaged assets first, then PX4 roots
        mission_share = get_package_share_directory("mission_manager")
        paths = [f"{mission_share}/models", f"{mission_share}/worlds"]
        paths += [f"{root}/models", f"{root}/worlds", f"{root}/models/caric2", f"{root}/worlds/caric2"]
        os.environ["GZ_SIM_RESOURCE_PATH"] = ":".join(paths)
        self.get_logger().info(f"GZ_SIM_RESOURCE_PATH={os.environ['GZ_SIM_RESOURCE_PATH']}")
        # Ensure expected PX4 default links resolve to mission assets if available
        self._ensure_px4_links(root, self.world_name_resolved, mission_share)

    def _make_symlink(self, link: str, preferred: str, fallback: str, label: str):
        target = preferred if os.path.exists(preferred) else fallback
        if not os.path.exists(link) and os.path.exists(target):
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(target, link)
            self.get_logger().info(f"Linked {label} {link} -> {target}")

    def _ensure_px4_links(self, gz_root: str, world_name: str, mission_share: str):
        worlds_root = os.path.join(gz_root, "worlds")
        models_root = os.path.join(gz_root, "models")

        self._make_symlink(
            link=os.path.join(worlds_root, f"{world_name}.sdf"),
            preferred=os.path.join(mission_share, "worlds", f"{world_name}.sdf"),
            fallback=os.path.join(worlds_root, "caric2", f"{world_name}.sdf"),
            label="world",
        )
        for model_name in ["x500_lidar_gimbal_explorer", "x500_gimbal_photographer"]:
            self._make_symlink(
                link=os.path.join(models_root, model_name),
                preferred=os.path.join(mission_share, "models", model_name),
                fallback=os.path.join(models_root, "caric2", model_name),
                label="model",
            )

    def _start_agent(self):
        port = int(self.get_parameter("agent_port").value)
        agent_build = os.path.expanduser("~/ros2_ws/src/Micro-XRCE-DDS-Agent/build")
        agent_bin = os.path.join(agent_build, "MicroXRCEAgent")
        if not os.path.exists(agent_bin):
            self.get_logger().warn("MicroXRCEAgent binary not found; skipping start")
            return
        self.agent_process = subprocess.Popen([agent_bin, "udp4", "-p", str(port)])
        self.get_logger().info(f"Started MicroXRCE-DDS-Agent on port {port}, pid={self.agent_process.pid}")
        time.sleep(3)

    def _spawn_fleet(self):
        fleet = load_fleet(self._spawn_profile)
        if not fleet:
            fleet_strs = self.get_parameter("fleet").get_parameter_value().string_array_value
            fleet = [json.loads(entry) for entry in fleet_strs if entry]
        # Use resolved world name
        os.environ["PX4_GZ_WORLD"] = self.world_name_resolved
        self.get_logger().info(f"Using world: {self.world_name_resolved}")

        # Sort fleet by instance to ensure main instance (usually 1) starts first
        fleet.sort(key=lambda x: int(x.get("instance", 999)))

        for i, cfg in enumerate(fleet):
            self._spawn_one(cfg)
            # Wait for Gazebo if this is the first instance
            if i == 0:
                self.get_logger().info("Waiting for Gazebo to initialize...")
                time.sleep(15)
            else:
                time.sleep(3)

    def _spawn_one(self, cfg):
        instance = int(cfg["instance"])
        autostart = int(cfg["autostart"])
        model = str(cfg["model"])
        x = float(cfg["x"])
        y = float(cfg["y"])
        z = float(cfg.get("z", 0.5))
        yaw_deg = float(cfg.get("yaw_deg", 0.0))
        yaw_rad = yaw_deg * math.pi / 180.0

        os.environ["PX4_SYS_AUTOSTART"] = str(autostart)
        os.environ["PX4_SIM_MODEL"] = model
        if instance > 1:
            os.environ["PX4_GZ_STANDALONE"] = "1"
        else:
            os.environ.pop("PX4_GZ_STANDALONE", None)
        os.environ["PX4_GZ_MODEL_POSE"] = f"{x},{y},{z},0,0,{yaw_rad}"

        px4_root = os.path.expanduser("~/ros2_ws/src/PX4-Autopilot")
        px4_bin = os.path.join(px4_root, "build/px4_sitl_default/bin/px4")
        name = str(cfg.get("name", f"drone_{instance}"))
        os.environ["ROS_NAMESPACE"] = name
        proc = subprocess.Popen([px4_bin, "-i", str(instance)], cwd=px4_root)
        self.processes.append(proc)
        self.get_logger().info(
            f"Spawned '{name}' (role={cfg.get('role', '')}) instance {instance} pid={proc.pid} model={model}"
        )

    def destroy(self):
        for p in self.processes:
            with suppress(Exception):
                p.send_signal(signal.SIGINT)
        for p in self.processes:
            with suppress(Exception):
                p.wait(timeout=2.0)
        if self.agent_process:
            with suppress(Exception):
                self.agent_process.send_signal(signal.SIGINT)
            with suppress(Exception):
                self.agent_process.wait(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = Spawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
