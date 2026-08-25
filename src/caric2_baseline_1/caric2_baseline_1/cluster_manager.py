#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
"""Cluster Manager: Groups POIs into clusters for drone assignment using DBSCAN."""

import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sklearn.cluster import DBSCAN, KMeans
from std_msgs.msg import String

MAX_POIS_PER_CLUSTER = 5
FAILED_ASSIGNMENT_SPLIT_REASONS = {
    "openscvx_generation_failed",
    "trajectory_completion_timeout",
}


class ClusterManager(Node):
    def __init__(self):
        super().__init__("cluster_manager")

        self.declare_parameter("cluster_radius", 25.0)
        self.declare_parameter("clustering_interval", 5.0)
        self.declare_parameter("assignment_timeout", 180.0)
        self.declare_parameter("min_pois_before_clustering", 6)
        self.declare_parameter("world_model_source", "known")
        self.declare_parameter("split_failed_lidar_clusters", True)

        self.cluster_radius = self.get_parameter("cluster_radius").value
        self.clustering_interval = self.get_parameter("clustering_interval").value
        self.assignment_timeout = self.get_parameter("assignment_timeout").value
        self.min_pois_before_clustering = self.get_parameter("min_pois_before_clustering").value
        self.world_model_source = self.get_parameter("world_model_source").get_parameter_value().string_value
        self.split_failed_lidar_clusters = bool(self.get_parameter("split_failed_lidar_clusters").value)

        self.unassigned_pois = []
        self.active_clusters = {}
        self.assigned_cluster_ids = set()
        self.completed_clusters = {}
        self.processed_poi_ids = set()
        self.cluster_counter = 0
        self.last_clustering_time = self.get_clock().now()

        self.all_pois_detected = False
        self.total_expected_pois = 0
        self.mission_complete_published = False
        self.poi_detection_run_id = None

        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL,
            depth=100,
        )

        self.cluster_info_pub = self.create_publisher(String, "/cluster_information", volatile_qos)
        self.cluster_status_pub = self.create_publisher(String, "/cluster_status", volatile_qos)
        self.mission_complete_pub = self.create_publisher(String, "/mission_complete", volatile_qos)

        self.create_subscription(String, "/poi_detected_passed", self.poi_callback, volatile_qos)
        self.create_subscription(
            String,
            "/photographer_assignment_complete",
            self.assignment_complete_callback,
            volatile_qos,
        )
        self.create_subscription(
            String,
            "/photographer_assignment_rejected",
            self.assignment_rejected_callback,
            volatile_qos,
        )
        self.create_subscription(
            String,
            "/photographer_assignments",
            self.assignment_made_callback,
            volatile_qos,
        )
        self.create_subscription(String, "/poi_all_detected", self.all_pois_detected_callback, 10)

        self.create_timer(5.0, self.timer_callback)

        self.get_logger().info(
            f"ClusterManager | eps={self.cluster_radius}m, timeout={self.assignment_timeout}s, "
            f"max_pois={MAX_POIS_PER_CLUSTER}, wait_for={self.min_pois_before_clustering}"
        )

    def poi_callback(self, msg):
        data = json.loads(msg.data)
        if data.get("event") != "poi_passed_los":
            return
        run_id = data.get("run_id")
        if not run_id:
            self.get_logger().warn("Ignoring passed POI without run_id")
            return
        if self.poi_detection_run_id is None:
            self.poi_detection_run_id = run_id
        elif run_id != self.poi_detection_run_id:
            self.get_logger().warn("Ignoring passed POI from a different run_id")
            return
        poi_data = data.get("poi_data", {})
        poi_id = poi_data.get("id", "")
        if poi_id in self.processed_poi_ids or any(p.get("id") == poi_id for p in self.unassigned_pois):
            return
        self.unassigned_pois.append(poi_data)

    def assignment_made_callback(self, msg):
        data = json.loads(msg.data)
        cluster_id = data.get("cluster_id")
        if cluster_id and cluster_id in self.active_clusters:
            self.assigned_cluster_ids.add(cluster_id)
            self.active_clusters[cluster_id]["status"] = "assigned"
            self.active_clusters[cluster_id]["assigned_time"] = self.get_clock().now().nanoseconds / 1e9

    def assignment_complete_callback(self, msg):
        data = json.loads(msg.data)
        cluster_id = data.get("cluster_id")
        if cluster_id not in self.active_clusters:
            return
        cluster = self.active_clusters.pop(cluster_id)
        cluster["status"] = "completed"
        self.completed_clusters[cluster_id] = cluster
        self.assigned_cluster_ids.discard(cluster_id)
        self.publish_cluster_info(cluster)

    def assignment_rejected_callback(self, msg):
        data = json.loads(msg.data)
        cluster_id = data.get("cluster_id")
        reason = data.get("reason", "unknown")
        if cluster_id not in self.active_clusters:
            return

        cluster = self.active_clusters.pop(cluster_id)
        self.assigned_cluster_ids.discard(cluster_id)

        if self._should_split_failed_cluster(cluster, reason):
            children = self._split_failed_cluster(cluster, reason)
            self.get_logger().warn(
                f"Cluster {cluster_id} rejected ({reason}); split into {len(children)} retry clusters"
            )
            for child in children:
                self.active_clusters[child["id"]] = child
                self.publish_cluster_info(child)
            return

        cluster["status"] = "unassigned"
        cluster["timestamp"] = self.get_clock().now().nanoseconds / 1e9
        self.active_clusters[cluster_id] = cluster
        self.publish_cluster_info(cluster)
        self.get_logger().warn(f"Cluster {cluster_id} rejected ({reason}); marked unassigned for retry")

    def all_pois_detected_callback(self, msg):
        data = json.loads(msg.data)
        if data.get("event") == "all_pois_detected" and not self.all_pois_detected:
            run_id = data.get("run_id")
            if self.poi_detection_run_id and run_id and run_id != self.poi_detection_run_id:
                self.get_logger().warn("Ignoring all_pois_detected from a different run_id")
                return
            self.all_pois_detected = True
            self.total_expected_pois = data.get("total_count", 0)

    def timer_callback(self):
        self.check_assignment_timeouts()

        if self.unassigned_pois:
            self.try_add_pois_to_existing_clusters()
            elapsed = (self.get_clock().now() - self.last_clustering_time).nanoseconds / 1e9
            num_pois = len(self.unassigned_pois)
            should_cluster = (
                elapsed >= self.clustering_interval
                and num_pois > 0
                and (self.all_pois_detected or num_pois >= self.min_pois_before_clustering)
            )
            if should_cluster:
                if self.all_pois_detected:
                    self.get_logger().info(f"Clustering remaining {num_pois} POIs")
                self.perform_clustering()
                self.last_clustering_time = self.get_clock().now()

        self.check_mission_complete()
        self.publish_status()

    def perform_clustering(self):
        if not self.unassigned_pois:
            return

        points = np.array([[p["position"]["x"], p["position"]["y"], p["position"]["z"]] for p in self.unassigned_pois])

        labels = DBSCAN(eps=self.cluster_radius, min_samples=1).fit_predict(points)
        label_to_indices = {}
        for i, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(i)

        label_to_indices = self._split_large_clusters(label_to_indices, points)

        for label, indices in label_to_indices.items():
            cluster_pois = [self.unassigned_pois[i] for i in indices]
            center = np.mean(points[indices], axis=0).tolist()
            cluster_id = f"cluster_{self.cluster_counter}"
            self.cluster_counter += 1

            cluster = {
                "id": cluster_id,
                "pois": cluster_pois,
                "center": center,
                "status": "unassigned",
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
            }
            self.active_clusters[cluster_id] = cluster
            for poi in cluster_pois:
                self.processed_poi_ids.add(poi["id"])
            self.publish_cluster_info(cluster)
            self.get_logger().info(f"Created {cluster_id}: {len(cluster_pois)} POIs")

        self.unassigned_pois = []

    def _split_large_clusters(self, label_to_indices, points):
        result = {}
        next_label = max(label_to_indices.keys(), default=-1) + 1

        for label, indices in label_to_indices.items():
            if len(indices) <= MAX_POIS_PER_CLUSTER:
                result[label] = indices
            else:
                n_splits = max(2, (len(indices) + MAX_POIS_PER_CLUSTER - 1) // MAX_POIS_PER_CLUSTER)
                sub_labels = KMeans(n_clusters=n_splits, random_state=42, n_init=10).fit_predict(points[indices])

                sub_clusters = {}
                for i, sub in enumerate(sub_labels):
                    sub_clusters.setdefault(sub, []).append(indices[i])

                for sub, sub_indices in sub_clusters.items():
                    if len(sub_indices) <= MAX_POIS_PER_CLUSTER:
                        new_label = label if sub == 0 and label not in result else next_label
                        result[new_label] = sub_indices
                        if new_label != label:
                            next_label += 1
                    else:
                        sub_result = self._split_large_clusters({0: sub_indices}, points)
                        for sub_label, final_indices in sub_result.items():
                            result[next_label] = final_indices
                            next_label += 1
        return result

    def _should_split_failed_cluster(self, cluster, reason):
        return (
            self.split_failed_lidar_clusters
            and self.world_model_source == "lidar_only"
            and reason in FAILED_ASSIGNMENT_SPLIT_REASONS
            and len(cluster.get("pois", [])) > 1
        )

    def _split_failed_cluster(self, cluster, reason):
        pois = list(cluster.get("pois", []))
        points = np.array([[p["position"]["x"], p["position"]["y"], p["position"]["z"]] for p in pois])
        child_size = max(1, len(pois) // 2)
        n_splits = max(2, int(np.ceil(len(pois) / child_size)))
        n_splits = min(n_splits, len(pois))

        if n_splits == len(pois):
            labels = np.arange(len(pois))
        else:
            labels = KMeans(n_clusters=n_splits, random_state=42, n_init=10).fit_predict(points)

        split_depth = int(cluster.get("split_depth", 0)) + 1
        children = []
        for label in sorted(set(int(v) for v in labels)):
            indices = [i for i, v in enumerate(labels) if int(v) == label]
            child_pois = [pois[i] for i in indices]
            child_points = points[indices]
            child_id = f"{cluster['id']}_retry_{self.cluster_counter}"
            self.cluster_counter += 1
            children.append(
                {
                    "id": child_id,
                    "pois": child_pois,
                    "center": np.mean(child_points, axis=0).tolist(),
                    "status": "unassigned",
                    "timestamp": self.get_clock().now().nanoseconds / 1e9,
                    "parent_cluster_id": cluster["id"],
                    "split_depth": split_depth,
                    "retry_reason": reason,
                }
            )

        return children

    def try_add_pois_to_existing_clusters(self):
        remaining = []
        for poi in self.unassigned_pois:
            pos = poi["position"]
            poi_coords = np.array([pos["x"], pos["y"], pos["z"]])
            eligible_clusters = [
                c
                for c in self.active_clusters.values()
                if c["id"] not in self.assigned_cluster_ids and len(c["pois"]) < MAX_POIS_PER_CLUSTER
            ]

            if not eligible_clusters:
                remaining.append(poi)
                continue

            best_cluster, best_dist = None, float("inf")
            for cluster in eligible_clusters:
                dist = np.linalg.norm(poi_coords - np.array(cluster["center"]))
                if dist < best_dist:
                    best_dist, best_cluster = dist, cluster

            if best_cluster and best_dist <= self.cluster_radius:
                best_cluster["pois"].append(poi)
                positions = np.array(
                    [[p["position"]["x"], p["position"]["y"], p["position"]["z"]] for p in best_cluster["pois"]]
                )
                best_cluster["center"] = np.mean(positions, axis=0).tolist()
                self.processed_poi_ids.add(poi["id"])
                self.publish_cluster_info(best_cluster)
            else:
                remaining.append(poi)
        self.unassigned_pois = remaining

    def check_assignment_timeouts(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        for cluster_id, cluster in list(self.active_clusters.items()):
            if cluster["status"] == "assigned":
                age = current_time - cluster.get("assigned_time", cluster["timestamp"])
                if age > self.assignment_timeout:
                    cluster["status"] = "timeout"
                    self.publish_cluster_info(cluster)

    def check_mission_complete(self):
        if self.mission_complete_published:
            return

        all_pois_received = self.total_expected_pois > 0 and len(self.processed_poi_ids) >= self.total_expected_pois
        mission_done = (
            self.all_pois_detected
            and all_pois_received
            and len(self.unassigned_pois) == 0
            and len(self.active_clusters) == 0
            and len(self.completed_clusters) > 0
        )

        if mission_done:
            self.mission_complete_published = True
            total_pois = sum(len(c["pois"]) for c in self.completed_clusters.values())
            msg = String()
            msg.data = json.dumps(
                {
                    "event": "mission_complete",
                    "total_clusters": len(self.completed_clusters),
                    "total_pois": total_pois,
                    "timestamp": self.get_clock().now().nanoseconds / 1e9,
                }
            )
            self.mission_complete_pub.publish(msg)
            self.get_logger().info(f"MISSION COMPLETE! {len(self.completed_clusters)} clusters, {total_pois} POIs")

    def publish_cluster_info(self, cluster):
        msg = String()
        data = {
            "cluster_id": cluster["id"],
            "pois": cluster["pois"],
            "center": cluster["center"],
            "poi_count": len(cluster["pois"]),
            "status": cluster["status"],
            "timestamp": cluster["timestamp"],
        }
        for key in ("parent_cluster_id", "split_depth", "retry_reason"):
            if key in cluster:
                data[key] = cluster[key]
        msg.data = json.dumps(data)
        self.cluster_info_pub.publish(msg)

    def publish_status(self):
        msg = String()
        msg.data = json.dumps(
            {
                "unassigned_pois": len(self.unassigned_pois),
                "active_clusters": len(self.active_clusters),
                "completed_clusters": len(self.completed_clusters),
                "all_pois_detected": self.all_pois_detected,
                "mission_complete": self.mission_complete_published,
                "timestamp": self.get_clock().now().nanoseconds / 1e9,
            }
        )
        self.cluster_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterManager()
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
