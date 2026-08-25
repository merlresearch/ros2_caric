#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import gz.transport13 as gz_transport
import numpy as np
import rclpy
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.msgs10.pose_v_pb2 import Pose_V
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


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


class LidarMappingNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_mapping_node")

        self.declare_parameter("world_name", "")
        self.declare_parameter("difficulty", "easy")
        self.declare_parameter("resolution_m", 0.3)
        self.declare_parameter("window_m", 150.0)
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("use_fixed_origin", True)
        self.declare_parameter("save_snapshot", True)
        self.declare_parameter("snapshot_dir", os.path.expanduser("~/ros2_ws/src/mission_manager/maps"))

        world_key = self.get_parameter("world_name").get_parameter_value().string_value
        difficulty = self.get_parameter("difficulty").get_parameter_value().string_value
        env_default = os.environ.get("PX4_GZ_WORLD", "mbs")
        self.world_name = resolve_world_name(world_key or env_default, difficulty)
        self.spawn_profile = world_key or env_default

        map_bounds = load_map_bounds(self.spawn_profile)

        cfg = load_mapping_defaults()
        default_min_known_ratio = cfg.get("min_known_ratio", cfg.get("global_min_known_ratio", 0.5))
        self.declare_parameter("z_min", cfg.get("z_min", 0.5))
        self.declare_parameter("z_max", cfg.get("z_max", 80.0))
        self.declare_parameter("inflate_m", cfg.get("inflate_m", 0.2))
        self.declare_parameter("min_known_ratio", default_min_known_ratio)
        self.declare_parameter("stable_threshold", cfg.get("stable_threshold", 0.02))
        self.declare_parameter("stable_seconds", cfg.get("stable_seconds", 2.0))
        self.declare_parameter("stop_when_stable", cfg.get("stop_when_stable", False))
        self.declare_parameter("carve_free", cfg.get("carve_free", True))
        self.declare_parameter("max_carve_hits", cfg.get("max_carve_hits", 10000))
        self.declare_parameter("self_filter_radius_m", cfg.get("self_filter_radius", 0.5))

        self.resolution = float(self.get_parameter("resolution_m").value)
        self.window = float(self.get_parameter("window_m").value)
        self.window_x = map_bounds["window_x"]
        self.window_y = map_bounds["window_y"]
        self.use_fixed_origin = bool(self.get_parameter("use_fixed_origin").value)
        self.fixed_origin_x = map_bounds["origin_x"]
        self.fixed_origin_y = map_bounds["origin_y"]
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)
        inflate_m = float(self.get_parameter("inflate_m").value)
        self.inflate_cells = max(0, int(round(inflate_m / self.resolution)))
        self.publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self.min_known_ratio = float(self.get_parameter("min_known_ratio").value)
        self.stable_threshold = float(self.get_parameter("stable_threshold").value)
        self.stable_seconds = float(self.get_parameter("stable_seconds").value)
        self.stop_when_stable = bool(self.get_parameter("stop_when_stable").value)
        self.carve_free = bool(self.get_parameter("carve_free").value)
        self.max_carve_hits = int(self.get_parameter("max_carve_hits").value)
        self.self_filter_radius = float(self.get_parameter("self_filter_radius_m").value)
        self.save_snapshot = bool(self.get_parameter("save_snapshot").value)
        self.snapshot_dir = self.get_parameter("snapshot_dir").get_parameter_value().string_value or os.path.expanduser(
            "~/ros2_ws/src/mission_manager/maps"
        )

        self.lidar_topics: List[str] = []
        fleet = load_fleet(self.spawn_profile)
        for d in fleet:
            role = str(d.get("role", ""))
            if role != "explorer":
                continue
            instance = int(d.get("instance", 0))
            model = str(d.get("model", ""))
            if not instance or not model:
                continue
            gz_model = compute_gz_model_name(model, instance)
            topic = f"/world/{self.world_name}/model/{gz_model}/" f"link/link_3d/sensor/lidar_3d_v2/scan/points"
            self.lidar_topics.append(topic)

        self.pose_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.cloud_world_points: List[np.ndarray] = []
        self.sensor_positions: Dict[str, Tuple[float, float]] = {}
        self._prev_grid: Optional[np.ndarray] = None
        self._stable_accum_s: float = 0.0
        self._stopped = False
        self._snapshot_saved = False
        self._seen_lidar_topics = set()

        self.gz_node = gz_transport.Node()
        pose_topic = f"/world/{self.world_name}/pose/info"
        if not self.gz_node.subscribe(Pose_V, pose_topic, self.pose_cb):
            self.get_logger().error(f"Failed to subscribe to {pose_topic}")
        else:
            self.get_logger().info(f"Subscribed to {pose_topic}")
        for t in self.lidar_topics:
            cb = self.pc_cb_factory(t)
            success = self.gz_node.subscribe(PointCloudPacked, t, cb)
            if not success:
                self.get_logger().error(f"Failed to subscribe to {t}")
            else:
                self.get_logger().info(f"Subscribed to {t}")

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.grid_pub = self.create_publisher(OccupancyGrid, "/mission/map/occupancy", latched_qos)
        self.meta_pub = self.create_publisher(String, "/mission/map/meta", latched_qos)
        self.status_pub = self.create_publisher(String, "/mission/map/status", latched_qos)

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_cb)
        if self.use_fixed_origin:
            grid_cells = int(math.ceil(self.window_x / self.resolution)) * int(
                math.ceil(self.window_y / self.resolution)
            )
            self.get_logger().info(
                f"Lidar mapping node started with FIXED origin: "
                f"({self.fixed_origin_x}, {self.fixed_origin_y}), "
                f"window=({self.window_x}m x {self.window_y}m), "
                f"resolution={self.resolution}m, "
                f"grid={grid_cells} cells"
            )
            self.get_logger().info(
                f"  Grid covers: X=[{self.fixed_origin_x}, {self.fixed_origin_x + self.window_x}], "
                f"Y=[{self.fixed_origin_y}, {self.fixed_origin_y + self.window_y}]"
            )
        else:
            self.get_logger().info(
                f"Lidar mapping node started with SENSOR-CENTERED origin, "
                f"window={self.window}m, resolution={self.resolution}m"
            )

    def pose_cb(self, msg: Pose_V) -> None:
        for p in msg.pose:
            name = p.name
            q = p.orientation
            R = quat_to_rot(q.x, q.y, q.z, q.w)
            t = np.array([p.position.x, p.position.y, p.position.z], dtype=np.float32)
            self.pose_map[name] = (R, t)

    def pc_cb_factory(self, topic: str):
        model: Optional[str] = None
        link: Optional[str] = None
        if "/model/" in topic:
            model = topic.split("/model/")[1].split("/")[0]
        if "/link/" in topic:
            link = topic.split("/link/")[1].split("/")[0]

        def cb(msg: PointCloudPacked):
            if topic not in self._seen_lidar_topics:
                self.get_logger().info(f"Point cloud callback triggered for {topic}")

            # Extract points (handle row padding)
            point_step = int(msg.point_step)
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            width = int(getattr(msg, "width", 0))
            height = int(getattr(msg, "height", 0))
            row_step = int(getattr(msg, "row_step", width * point_step))
            if height > 0 and width > 0 and row_step >= width * point_step:
                rows = []
                for r in range(height):
                    s = r * row_step
                    e = s + width * point_step
                    if e > buf.size:
                        break
                    rows.append(buf[s:e])
                if not rows:
                    return
                buf2 = np.concatenate(rows)
            else:
                buf2 = buf
            if buf2.size < point_step or (buf2.size % point_step) != 0:
                return
            arr = np.frombuffer(buf2, dtype=np.uint8).reshape((-1, point_step))
            x_off = y_off = z_off = None
            for f in msg.field:
                if f.name == "x":
                    x_off = int(f.offset)
                elif f.name == "y":
                    y_off = int(f.offset)
                elif f.name == "z":
                    z_off = int(f.offset)
            if x_off is None or y_off is None or z_off is None:
                return
            x = arr[:, x_off : x_off + 4].copy().view("<f4").reshape(-1)
            y = arr[:, y_off : y_off + 4].copy().view("<f4").reshape(-1)
            z = arr[:, z_off : z_off + 4].copy().view("<f4").reshape(-1)
            pts = np.stack([x, y, z], axis=1).astype(np.float32)
            if self.self_filter_radius > 0.0:
                r_xy = np.sqrt(np.square(pts[:, 0]) + np.square(pts[:, 1]))
                keep = r_xy >= self.self_filter_radius
                if not np.any(keep):
                    return
                pts = pts[keep]
            finite = np.isfinite(pts).all(axis=1)
            if not np.any(finite):
                return
            pts = pts[finite]

            R, t = self.lookup_pose_suffix(model, link)
            if R is None:
                if topic not in self._seen_lidar_topics:
                    self.get_logger().warn(
                        f"Pose lookup failed for topic {topic}. "
                        f"Model: {model}, Link: {link}. "
                        f"Available poses: {list(self.pose_map.keys())}"
                    )
                return
            pts_world = (pts @ R.T) + t

            mask = (pts_world[:, 2] >= self.z_min) & (pts_world[:, 2] <= self.z_max)
            sel = pts_world[mask]
            if sel.shape[0] == 0:
                sel = pts_world
            self.cloud_world_points.append(sel[:, :3])
            self.sensor_positions[topic] = (float(t[0]), float(t[1]))

            if topic not in self._seen_lidar_topics:
                self._seen_lidar_topics.add(topic)
                self.get_logger().info(
                    f"Received first LiDAR cloud on {topic} " f"(raw_pts={pts.shape[0]}, used_pts={sel.shape[0]})"
                )

        return cb

    def lookup_pose_suffix(
        self, model: Optional[str], link: Optional[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if model and link:
            target = f"{model}::{link}"
            for key, (R, t) in self.pose_map.items():
                if key.endswith(target):
                    return R, t
        if model:
            for key, (R, t) in self.pose_map.items():
                if key.endswith(model):
                    return R, t
        if self.pose_map:
            return next(iter(self.pose_map.values()))
        return None, None

    def timer_cb(self) -> None:
        if self._stopped:
            return
        if not self.cloud_world_points:
            if not self._seen_lidar_topics:
                self.get_logger().warn(
                    "No point cloud data received yet. "
                    f"Subscribed to {len(self.lidar_topics)} lidar topics: {self.lidar_topics}. "
                    f"Available poses: {list(self.pose_map.keys())}"
                )
            return

        pts_world = np.concatenate(self.cloud_world_points, axis=0)

        if self.use_fixed_origin:
            origin_x = self.fixed_origin_x
            origin_y = self.fixed_origin_y
            width_cells = int(math.ceil(self.window_x / self.resolution))
            height_cells = int(math.ceil(self.window_y / self.resolution))
        else:
            if self.sensor_positions:
                xs = [p[0] for p in self.sensor_positions.values()]
                ys = [p[1] for p in self.sensor_positions.values()]
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
            else:
                cx, cy = float(np.mean(pts_world[:, 0])), float(np.mean(pts_world[:, 1]))
            half = self.window / 2.0
            origin_x = cx - half
            origin_y = cy - half
            width_cells = int(math.ceil(self.window / self.resolution))
            height_cells = int(math.ceil(self.window / self.resolution))

        grid = self._build_grid(pts_world, origin_x, origin_y, width_cells, height_cells)

        if self.carve_free and self.sensor_positions:
            ix = np.floor((pts_world[:, 0] - origin_x) / self.resolution).astype(np.int32)
            iy = np.floor((pts_world[:, 1] - origin_y) / self.resolution).astype(np.int32)
            valid = (ix >= 0) & (ix < width_cells) & (iy >= 0) & (iy < height_cells)
            hits = list(set(zip(ix[valid].tolist(), iy[valid].tolist())))
            if len(hits) > self.max_carve_hits:
                hits = random.sample(hits, self.max_carve_hits)
            for _topic, (sx, sy) in self.sensor_positions.items():
                sx_i = int(math.floor((sx - origin_x) / self.resolution))
                sy_i = int(math.floor((sy - origin_y) / self.resolution))
                for hx, hy in hits:
                    for cx_i, cy_i in self._bresenham(sx_i, sy_i, hx, hy):
                        if 0 <= cx_i < width_cells and 0 <= cy_i < height_cells:
                            if grid[cy_i, cx_i] == -1:
                                grid[cy_i, cx_i] = 0

        # Morphological opening to remove speckle
        kernel = np.ones((3, 3), np.uint8)
        bin_occ = (grid == 100).astype(np.uint8)
        cleaned = cv2.morphologyEx(bin_occ, cv2.MORPH_OPEN, kernel)
        grid[(cleaned == 0) & (grid == 100)] = 0
        grid[cleaned == 1] = 100

        if self.inflate_cells > 0:
            occ = np.argwhere(grid == 100)
            for yy, xx in occ:
                y0 = max(0, yy - self.inflate_cells)
                y1 = min(height_cells, yy + self.inflate_cells + 1)
                x0 = max(0, xx - self.inflate_cells)
                x1 = min(width_cells, xx + self.inflate_cells + 1)
                rr = self.inflate_cells
                for ry in range(y0, y1):
                    for rx in range(x0, x1):
                        if (rx - xx) * (rx - xx) + (ry - yy) * (ry - yy) <= rr * rr:
                            if grid[ry, rx] != 100:
                                grid[ry, rx] = max(grid[ry, rx], 50)
            grid[grid == 50] = 100

        og = OccupancyGrid()
        og.header.stamp = self.get_clock().now().to_msg()
        og.header.frame_id = "world"
        og.info.resolution = float(self.resolution)
        og.info.width = int(width_cells)
        og.info.height = int(height_cells)
        og.info.origin.position.x = float(origin_x)
        og.info.origin.position.y = float(origin_y)
        og.info.origin.position.z = 0.0
        og.data = grid.astype(np.int8).flatten(order="C").tolist()
        self.grid_pub.publish(og)

        meta = {
            "world": self.world_name,
            "resolution": self.resolution,
            "resolution_m": self.resolution,
            "width_cells": width_cells,
            "height_cells": height_cells,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }
        m = String()
        m.data = json.dumps(meta)
        self.meta_pub.publish(m)

        known = int(np.count_nonzero(grid != -1))
        total = grid.size
        known_ratio = (known / total) if total > 0 else 0.0
        if self._prev_grid is not None and self._prev_grid.shape == grid.shape:
            changed = int(np.count_nonzero(grid != self._prev_grid))
            changed_frac = (changed / total) if total > 0 else 0.0
        else:
            changed_frac = 1.0
        self._prev_grid = grid.copy()

        status = {
            "known_ratio": known_ratio,
            "changed_fraction": changed_frac,
            "stable_seconds": self._stable_accum_s,
            "min_known_ratio": self.min_known_ratio,
            "stable_threshold": self.stable_threshold,
        }
        sm = String()
        sm.data = json.dumps(status)
        self.status_pub.publish(sm)

        if changed_frac <= self.stable_threshold:
            self._stable_accum_s += 1.0 / self.publish_rate_hz
        else:
            self._stable_accum_s = 0.0

        if (
            self.stop_when_stable
            and known_ratio >= self.min_known_ratio
            and self._stable_accum_s >= self.stable_seconds
        ):
            self.get_logger().info("Coverage/stability reached – stopping updates")
            self._stopped = True

        if (
            self.save_snapshot
            and not self._snapshot_saved
            and known_ratio >= self.min_known_ratio
            and self._stable_accum_s >= self.stable_seconds
        ):
            self._save_snapshot(grid, meta)
            self._snapshot_saved = True
            self.get_logger().info(f"Snapshot saved to {self.snapshot_dir}")

    def _build_grid(
        self,
        pts_world: np.ndarray,
        origin_x: float,
        origin_y: float,
        width_cells: int,
        height_cells: int,
    ) -> np.ndarray:
        grid = -1 * np.ones((height_cells, width_cells), dtype=np.int16)
        ix = np.floor((pts_world[:, 0] - origin_x) / self.resolution).astype(np.int32)
        iy = np.floor((pts_world[:, 1] - origin_y) / self.resolution).astype(np.int32)
        valid = (ix >= 0) & (ix < width_cells) & (iy >= 0) & (iy < height_cells)
        ix, iy = ix[valid], iy[valid]
        grid[iy, ix] = 100
        return grid

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
            yield x, y

    def _save_snapshot(self, grid: np.ndarray, meta: dict) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.snapshot_dir, f"live_map_{ts}")
        np.save(base + "_grid.npy", grid)
        with open(base + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        png = np.full_like(grid, 127, dtype=np.uint8)
        png[grid == 0] = 255
        png[grid == 100] = 0
        png = np.flipud(png)
        cv2.imwrite(base + ".png", png)


def main(args=None):
    rclpy.init(args=args)
    node = LidarMappingNode()
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
