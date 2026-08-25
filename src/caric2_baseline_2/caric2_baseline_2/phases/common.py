# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def local_to_world(
    spawn_x: float,
    spawn_y: float,
    spawn_yaw_deg: float,
    local_x: float,
    local_y: float,
    local_z: float,
) -> Tuple[float, float, float]:
    yaw_rad = math.radians(spawn_yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    rel_n = float(local_x)
    rel_e = float(local_y)
    world_e = rel_n * sin_yaw + rel_e * cos_yaw
    world_n = rel_n * cos_yaw - rel_e * sin_yaw
    world_x = float(spawn_x) + world_e
    world_y = float(spawn_y) + world_n
    world_z = -float(local_z)
    return (world_x, world_y, world_z)


def world_to_local(
    spawn_x: float,
    spawn_y: float,
    spawn_yaw_deg: float,
    world_x: float,
    world_y: float,
    world_z: float,
) -> Tuple[float, float, float]:
    spawn_yaw_rad = math.radians(-spawn_yaw_deg)
    rel_e = float(world_x) - float(spawn_x)
    rel_n = float(world_y) - float(spawn_y)
    cos_yaw = math.cos(spawn_yaw_rad)
    sin_yaw = math.sin(spawn_yaw_rad)
    local_x = rel_n * cos_yaw - rel_e * sin_yaw
    local_y = rel_e * cos_yaw + rel_n * sin_yaw
    local_z = -float(world_z)
    return (local_x, local_y, local_z)


MBS_BOXES: List[Dict] = [
    {
        "name": "box1_deck",
        "center": np.array([37.95942455, 0.07421307, 55.82170486], dtype=float),
        "half": np.array([53.934815295, 10.19887834, 4.451805725], dtype=float),
        "yaw": -0.4281,
    },
    {
        "name": "box2_towerA",
        "center": np.array([13.15921211, 12.35749626, 25.68494957], dtype=float),
        "half": np.array([10.130066875, 5.792826535, 25.68494957], dtype=float),
        "yaw": -0.2589,
    },
    {
        "name": "box3_towerB",
        "center": np.array([43.73695755, 3.39359474, 25.68494957], dtype=float),
        "half": np.array([9.20425415, 7.324481965, 25.68494957], dtype=float),
        "yaw": -0.4549,
    },
    {
        "name": "box4_towerC",
        "center": np.array([70.28045654, -12.58303833, 25.68494957], dtype=float),
        "half": np.array([9.52007103, 5.327802655, 25.68494957], dtype=float),
        "yaw": -0.5935,
    },
]

# Wire spans are solid boxes covering all wire levels to prevent gap-finding
POWERLINE_BOXES: List[Dict] = [
    {
        "name": "tower_0",
        "center": np.array([0.0, 0.0, 18.5], dtype=float),
        "half": np.array([6.0, 8.0, 18.5], dtype=float),
        "yaw": 0.0,
    },
    {
        "name": "tower_1",
        "center": np.array([50.0, 0.0, 18.5], dtype=float),
        "half": np.array([6.0, 8.0, 18.5], dtype=float),
        "yaw": 0.0,
    },
    {
        "name": "tower_2",
        "center": np.array([100.0, 0.0, 18.5], dtype=float),
        "half": np.array([6.0, 8.0, 18.5], dtype=float),
        "yaw": 0.0,
    },
    {
        "name": "wire_span0",
        "center": np.array([25.0, 0.0, 22.0], dtype=float),
        "half": np.array([22.0, 8.0, 9.0], dtype=float),
        "yaw": 0.0,
    },
    {
        "name": "wire_span1",
        "center": np.array([75.0, 0.0, 22.0], dtype=float),
        "half": np.array([22.0, 8.0, 9.0], dtype=float),
        "yaw": 0.0,
    },
]

WORLD_OBSTACLES: Dict[str, List[Dict]] = {
    "mbs": MBS_BOXES,
    "powerline": POWERLINE_BOXES,
}

WORLD_COMM_CENTERS: Dict[str, np.ndarray] = {
    "mbs": np.array([0.0, -25.0, 5.0], dtype=float),
    "powerline": np.array([50.0, -20.0, 5.0], dtype=float),
}


def get_obstacles_for_world(world: str) -> List[Dict]:
    """Return obstacle boxes for the given world. Defaults to empty list if unknown."""
    return WORLD_OBSTACLES.get(world.lower(), [])


def get_comm_center_for_world(world: str) -> np.ndarray:
    """Return comm_center (GCS location with hover height) for the given world."""
    return WORLD_COMM_CENTERS.get(world.lower(), np.array([0.0, -25.0, 5.0], dtype=float))
