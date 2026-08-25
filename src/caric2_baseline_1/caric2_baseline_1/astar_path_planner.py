#!/usr/bin/env python3
# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
"""A* path planner using a pre-loaded 3D voxel grid for obstacle avoidance."""

import math
import pickle
from typing import List, Optional, Tuple

import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import binary_dilation


class AStarPathPlanner:
    def __init__(self, voxel_grid_path: str, voxel_metadata_path: str):
        self.voxel_grid_path = voxel_grid_path
        self.voxel_metadata_path = voxel_metadata_path

        self._load_voxel_data()

        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        self.min_clearance = 3
        self.default_flight_height = 10

        self._slice_cache = {}
        self._precompute_clearance_grid()

    def _load_voxel_data(self):
        self.voxel_grid = np.load(self.voxel_grid_path)
        with open(self.voxel_metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.resolution = self.metadata["resolution"]
        self.origin = self.metadata["origin"]
        self.shape = self.metadata["shape"]
        self.free_value = self.metadata["free_value"]
        self.occupied_value = self.metadata["occupied_value"]

    def _precompute_clearance_grid(self):
        obstacle_mask = self.voxel_grid != self.free_value
        struct_size = 2 * self.min_clearance + 1
        structure = np.ones((struct_size, struct_size, struct_size), dtype=bool)
        dilated_obstacles = binary_dilation(obstacle_mask, structure=structure)
        self._clearance_grid = ~dilated_obstacles

    def world_to_voxel(self, world_x: float, world_y: float, world_z: float) -> Tuple[int, int, int]:
        rel_x = world_x - self.origin[0]
        rel_y = world_y - self.origin[1]
        rel_z = world_z - self.origin[2]
        voxel_x = int(rel_x / self.resolution)
        voxel_y = int(rel_y / self.resolution)
        voxel_z = int(rel_z / self.resolution)
        return voxel_x, voxel_y, voxel_z

    def voxel_to_world(self, voxel_x: int, voxel_y: int, voxel_z: int) -> Tuple[float, float, float]:
        world_x = self.origin[0] + voxel_x * self.resolution
        world_y = self.origin[1] + voxel_y * self.resolution
        world_z = self.origin[2] + voxel_z * self.resolution
        return world_x, world_y, world_z

    def local_to_world(
        self, local_x: float, local_y: float, local_z: float, spawn_x: float, spawn_y: float, spawn_yaw_deg: float
    ) -> Tuple[float, float, float]:
        spawn_yaw_rad = math.radians(spawn_yaw_deg)
        cos_yaw = math.cos(spawn_yaw_rad)
        sin_yaw = math.sin(spawn_yaw_rad)
        rel_n = local_x * cos_yaw + local_y * sin_yaw
        rel_e = -local_x * sin_yaw + local_y * cos_yaw
        world_x = spawn_x + rel_e
        world_y = spawn_y + rel_n
        world_z = -local_z
        return world_x, world_y, world_z

    def world_to_local(
        self, world_x: float, world_y: float, world_z: float, spawn_x: float, spawn_y: float, spawn_yaw_deg: float
    ) -> Tuple[float, float, float]:
        spawn_yaw_rad = math.radians(-spawn_yaw_deg)
        rel_e = world_x - spawn_x
        rel_n = world_y - spawn_y
        cos_yaw = math.cos(spawn_yaw_rad)
        sin_yaw = math.sin(spawn_yaw_rad)
        local_x = rel_n * cos_yaw - rel_e * sin_yaw
        local_y = rel_n * sin_yaw + rel_e * cos_yaw
        local_z = -world_z
        return local_x, local_y, local_z

    def is_valid_voxel(self, voxel_x: int, voxel_y: int, voxel_z: int) -> bool:
        return 0 <= voxel_x < self.shape[0] and 0 <= voxel_y < self.shape[1] and 0 <= voxel_z < self.shape[2]

    def create_2d_slice(self, flight_height_world: float) -> np.ndarray:
        _, _, voxel_z = self.world_to_voxel(0, 0, flight_height_world)
        voxel_z = max(0, min(voxel_z, self.shape[2] - 1))

        if voxel_z in self._slice_cache:
            return self._slice_cache[voxel_z]

        pathfinding_grid = self._clearance_grid[:, :, voxel_z].astype(int)
        self._slice_cache[voxel_z] = pathfinding_grid
        return pathfinding_grid

    def plan_path_local(
        self,
        start_local: Tuple[float, float, float],
        end_local: Tuple[float, float, float],
        spawn_x: float,
        spawn_y: float,
        spawn_yaw_deg: float,
    ) -> Optional[List[Tuple[float, float, float]]]:
        start_world = self.local_to_world(
            start_local[0], start_local[1], start_local[2], spawn_x, spawn_y, spawn_yaw_deg
        )
        end_world = self.local_to_world(end_local[0], end_local[1], end_local[2], spawn_x, spawn_y, spawn_yaw_deg)
        world_path = self.plan_path_world(start_world, end_world)
        if world_path is None:
            return None
        local_path = []
        for world_point in world_path:
            local_point = self.world_to_local(
                world_point[0], world_point[1], world_point[2], spawn_x, spawn_y, spawn_yaw_deg
            )
            local_path.append(local_point)
        return local_path

    def plan_path_world(
        self, start_world: Tuple[float, float, float], end_world: Tuple[float, float, float]
    ) -> Optional[List[Tuple[float, float, float]]]:
        start_vx, start_vy, start_vz = self.world_to_voxel(start_world[0], start_world[1], start_world[2])
        end_vx, end_vy, end_vz = self.world_to_voxel(end_world[0], end_world[1], end_world[2])

        if not self.is_valid_voxel(start_vx, start_vy, start_vz) or not self.is_valid_voxel(end_vx, end_vy, end_vz):
            return None

        avg_world_z = (start_world[2] + end_world[2]) / 2.0
        pathfinding_matrix_2d = self.create_2d_slice(avg_world_z)
        grid = Grid(matrix=pathfinding_matrix_2d.T.tolist())

        start_node = grid.node(start_vx, start_vy)
        end_node = grid.node(end_vx, end_vy)

        if not grid.walkable(start_node.x, start_node.y):
            return None
        if not grid.walkable(end_node.x, end_node.y):
            return None

        path_nodes, _ = self.finder.find_path(start_node, end_node, grid)

        if not path_nodes:
            return None

        world_path = []
        num_nodes = len(path_nodes)
        for i, node in enumerate(path_nodes):
            ratio = i / (num_nodes - 1) if num_nodes > 1 else 0.0
            interp_vz = int(start_vz + (end_vz - start_vz) * ratio)
            interp_vz = max(0, min(interp_vz, self.shape[2] - 1))
            world_x, world_y, world_z = self.voxel_to_world(node.x, node.y, interp_vz)
            world_path.append((world_x, world_y, world_z))
        return world_path
