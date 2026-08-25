# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


def grid_to_aggregated_boxes(
    grid: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    z_min: float,
    z_max: float,
    treat_unknown_as_obstacle: bool = False,
    min_box_area: float = 1.0,
    max_aggregate_boxes: int = 3,
    downsample_factor: int = 2,
    use_dbscan: bool = True,
    dbscan_eps: float = 5.0,
    dbscan_min_samples: int = 2,
) -> list[dict]:
    h, w = grid.shape
    effective_resolution = resolution

    if downsample_factor > 1:
        f = downsample_factor
        h_new = (h // f) * f
        w_new = (w // f) * f
        if h_new > 0 and w_new > 0:
            grid_crop = grid[:h_new, :w_new]
            if treat_unknown_as_obstacle:
                binary = (grid_crop != 0).astype(np.int8)
            else:
                binary = (grid_crop == 100).astype(np.int8)
            blocks = binary.reshape(h_new // f, f, w_new // f, f)
            downsampled = blocks.max(axis=(1, 3))
            grid = np.full_like(downsampled, fill_value=-1, dtype=np.int8)
            grid[downsampled == 0] = 0
            grid[downsampled == 1] = 100
            h, w = grid.shape
            effective_resolution = resolution * f

    if treat_unknown_as_obstacle:
        mask = (grid != 0).astype(np.int32)
    else:
        mask = (grid == 100).astype(np.int32)

    visited = np.zeros_like(mask, dtype=bool)
    raw_boxes: list[dict] = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                w_box = 1
                while (x + w_box < w) and mask[y, x + w_box] and not visited[y, x + w_box]:
                    w_box += 1
                h_box = 1
                while y + h_box < h:
                    row_ok = True
                    for k in range(w_box):
                        if not mask[y + h_box, x + k] or visited[y + h_box, x + k]:
                            row_ok = False
                            break
                    if row_ok:
                        h_box += 1
                    else:
                        break
                visited[y : y + h_box, x : x + w_box] = True

                box_min_x = origin_x + x * effective_resolution
                box_min_y = origin_y + y * effective_resolution
                box_w = w_box * effective_resolution
                box_h = h_box * effective_resolution
                cx = box_min_x + box_w / 2.0
                cy = box_min_y + box_h / 2.0
                half_x = box_w / 2.0
                half_y = box_h / 2.0
                half_z = (z_max - z_min) / 2.0

                raw_boxes.append(
                    {
                        "center": np.array([cx, cy, (z_min + z_max) / 2.0], dtype=float),
                        "half": np.array([half_x, half_y, half_z], dtype=float),
                        "yaw": 0.0,
                    }
                )

    if min_box_area > 0.0:
        raw_boxes = [b for b in raw_boxes if 4.0 * b["half"][0] * b["half"][1] >= min_box_area]

    if max_aggregate_boxes is None or max_aggregate_boxes <= 0:
        return raw_boxes
    if not raw_boxes:
        return []

    centers = np.array([b["center"][:2] for b in raw_boxes], dtype=float)
    n_boxes = centers.shape[0]

    if use_dbscan:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = dbscan.fit_predict(centers)
        unique_labels = set(labels)
        clusters: list[list[int]] = []
        for lbl in unique_labels:
            if lbl == -1:
                for i in range(n_boxes):
                    if labels[i] == -1:
                        clusters.append([i])
            else:
                cluster_members = [i for i in range(n_boxes) if labels[i] == lbl]
                if cluster_members:
                    clusters.append(cluster_members)
    else:
        k_clusters = min(max_aggregate_boxes, n_boxes)
        rng = np.random.default_rng(seed=42)
        idx_init = rng.choice(n_boxes, size=k_clusters, replace=False)
        mu = centers[idx_init].copy()
        labels_arr = np.zeros(n_boxes, dtype=int)
        for _ in range(15):
            for i in range(n_boxes):
                dists = np.sum((mu - centers[i]) ** 2, axis=1)
                labels_arr[i] = int(np.argmin(dists))
            for k in range(k_clusters):
                pts = centers[labels_arr == k]
                if pts.size > 0:
                    mu[k] = pts.mean(axis=0)
        clusters = [[] for _ in range(k_clusters)]
        for i in range(n_boxes):
            clusters[labels_arr[i]].append(i)
        clusters = [c for c in clusters if c]

    def cluster_area(idx_list: list[int]) -> float:
        return sum(4.0 * raw_boxes[i]["half"][0] * raw_boxes[i]["half"][1] for i in idx_list)

    clusters.sort(key=cluster_area, reverse=True)
    selected_clusters = clusters[:max_aggregate_boxes]

    aggregated: list[dict] = []
    for idxs in selected_clusters:
        if not idxs:
            continue
        pts: list[np.ndarray] = []
        for i in idxs:
            c = raw_boxes[i]["center"]
            hh = raw_boxes[i]["half"]
            pts.append(np.array([c[0] - hh[0], c[1] - hh[1]], dtype=float))
            pts.append(np.array([c[0] - hh[0], c[1] + hh[1]], dtype=float))
            pts.append(np.array([c[0] + hh[0], c[1] - hh[1]], dtype=float))
            pts.append(np.array([c[0] + hh[0], c[1] + hh[1]], dtype=float))
        agg_box = _fit_oriented_box_mvee(pts, z_min, z_max)
        aggregated.append(agg_box)

    return _remove_nested_boxes(aggregated)


def _fit_oriented_box_mvee(
    pts: list[np.ndarray],
    z_min: float,
    z_max: float,
) -> dict:
    p_array = np.asarray(pts, dtype=float)
    if p_array.shape[0] < 3:
        return _axis_aligned_box(p_array, z_min, z_max)

    hull = ConvexHull(p_array)
    hull_pts = p_array[hull.vertices]

    best_area = float("inf")
    best_box = None
    n_hull = len(hull_pts)

    for i in range(n_hull):
        p1 = hull_pts[i]
        p2 = hull_pts[(i + 1) % n_hull]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-9:
            continue
        angle = math.atan2(edge[1], edge[0])
        c, s = math.cos(-angle), math.sin(-angle)
        rotation = np.array([[c, -s], [s, c]])
        rotated = (rotation @ hull_pts.T).T
        min_x_r, max_x_r = rotated[:, 0].min(), rotated[:, 0].max()
        min_y_r, max_y_r = rotated[:, 1].min(), rotated[:, 1].max()
        width, height = max_x_r - min_x_r, max_y_r - min_y_r
        area = width * height
        if area < best_area:
            best_area = area
            cx_r = 0.5 * (min_x_r + max_x_r)
            cy_r = 0.5 * (min_y_r + max_y_r)
            c_back, s_back = math.cos(angle), math.sin(angle)
            rotation_back = np.array([[c_back, -s_back], [s_back, c_back]])
            center_world = rotation_back @ np.array([cx_r, cy_r])
            best_box = {
                "center": np.array([center_world[0], center_world[1], 0.5 * (z_min + z_max)], dtype=float),
                "half": np.array(
                    [max(0.5 * width, 0.1), max(0.5 * height, 0.1), 0.5 * (z_max - z_min)],
                    dtype=float,
                ),
                "yaw": float(angle),
            }

    return best_box if best_box is not None else _axis_aligned_box(p_array, z_min, z_max)


def _axis_aligned_box(points: np.ndarray, z_min: float, z_max: float) -> dict:
    xs, ys = points[:, 0], points[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    width, height = max(max_x - min_x, 0.1), max(max_y - min_y, 0.1)
    return {
        "center": np.array([cx, cy, 0.5 * (z_min + z_max)], dtype=float),
        "half": np.array([0.5 * width, 0.5 * height, 0.5 * (z_max - z_min)], dtype=float),
        "yaw": 0.0,
    }


def _remove_nested_boxes(boxes: list[dict]) -> list[dict]:
    if len(boxes) <= 1:
        return boxes

    def box_area(box):
        return 4.0 * box["half"][0] * box["half"][1]

    sorted_boxes = sorted(boxes, key=box_area, reverse=True)
    kept = []
    for box in sorted_boxes:
        should_remove = False
        for larger in kept:
            if _box_redundant(box, larger):
                should_remove = True
                break
        if not should_remove:
            kept.append(box)
    return kept


def _box_redundant(smaller: dict, larger: dict) -> bool:
    smaller_c = smaller["center"][:2]
    smaller_h = smaller["half"][:2]
    larger_c = larger["center"][:2]
    larger_h = larger["half"][:2]
    larger_yaw = larger.get("yaw", 0.0)
    smaller_yaw = smaller.get("yaw", 0.0)

    cos_o, sin_o = math.cos(-larger_yaw), math.sin(-larger_yaw)
    dx = smaller_c[0] - larger_c[0]
    dy = smaller_c[1] - larger_c[1]
    lx = dx * cos_o - dy * sin_o
    ly = dx * sin_o + dy * cos_o

    margin = 1.0
    if abs(lx) <= larger_h[0] + margin and abs(ly) <= larger_h[1] + margin:
        return True

    cos_i, sin_i = math.cos(smaller_yaw), math.sin(smaller_yaw)
    corners_inside = 0
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            corner_lx = sx * smaller_h[0]
            corner_ly = sy * smaller_h[1]
            wx = smaller_c[0] + corner_lx * cos_i - corner_ly * sin_i
            wy = smaller_c[1] + corner_lx * sin_i + corner_ly * cos_i
            ddx = wx - larger_c[0]
            ddy = wy - larger_c[1]
            llx = ddx * cos_o - ddy * sin_o
            lly = ddx * sin_o + ddy * cos_o
            if abs(llx) <= larger_h[0] + margin and abs(lly) <= larger_h[1] + margin:
                corners_inside += 1
    return corners_inside >= 3
