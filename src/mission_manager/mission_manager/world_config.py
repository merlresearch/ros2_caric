# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import yaml
from ament_index_python.packages import get_package_share_directory


def resolve_world_name(world_key: str, difficulty: str) -> str:
    """Return Gazebo world name (SDF basename without extension) given a base key and difficulty.

    Mapping: <world_key>_<N>_poi where N in {10, 50, 100} for {easy, medium, hard}.
    """
    diff_map = {"easy": 10, "medium": 50, "hard": 100}
    poi_count = diff_map.get(difficulty, 10)
    return f"{world_key}_{poi_count}_poi"


def load_fleet(profile_key: str = "default") -> list:
    """Load fleet configuration from spawn_profiles.yaml.

    Returns a list of dicts: {name, role, instance, autostart, model, x, y, yaw_deg}
    """
    share_dir = get_package_share_directory("mission_manager")
    path = os.path.join(share_dir, "config", "spawn_profiles.yaml")
    if not os.path.exists(path):
        return []
    content = open(path, "r").read().replace("\t", "  ")
    cfg = yaml.safe_load(content) or {}
    profile = cfg.get(profile_key) or {}
    fleet = profile.get("fleet") or []
    normalized = []
    for item in fleet:
        normalized.append(
            {
                "name": str(item.get("name", "drone")),
                "role": str(item.get("role", "explorer")),
                "instance": int(item["instance"]),
                "autostart": int(item["autostart"]),
                "model": str(item["model"]),
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "z": float(item.get("z", 0.5)),
                "yaw_deg": float(item.get("yaw_deg", 0.0)),
            }
        )
    return normalized


def load_map_bounds(profile_key: str = "default") -> dict:
    """Load map bounds from spawn_profiles.yaml for the given world.

    Returns a dict with keys: origin_x, origin_y, window_x, window_y
    Falls back to 'default' profile if world-specific bounds not found.
    """
    defaults = {
        "origin_x": -35.0,
        "origin_y": -30.0,
        "window_x": 130.0,
        "window_y": 60.0,
    }
    share_dir = get_package_share_directory("mission_manager")
    path = os.path.join(share_dir, "config", "spawn_profiles.yaml")
    if not os.path.exists(path):
        return defaults
    content = open(path, "r").read().replace("\t", "  ")
    cfg = yaml.safe_load(content) or {}
    profile = cfg.get(profile_key) or cfg.get("default") or {}
    bounds = profile.get("map_bounds") or {}
    return {
        "origin_x": float(bounds.get("origin_x", defaults["origin_x"])),
        "origin_y": float(bounds.get("origin_y", defaults["origin_y"])),
        "window_x": float(bounds.get("window_x", defaults["window_x"])),
        "window_y": float(bounds.get("window_y", defaults["window_y"])),
    }


def load_mapping_defaults() -> dict:
    """Load mapping defaults from config/mapping.yaml (returns {} on error)."""
    share_dir = get_package_share_directory("mission_manager")
    path = os.path.join(share_dir, "config", "mapping.yaml")
    if not os.path.exists(path):
        return {}
    content = open(path, "r").read().replace("\t", "  ")
    cfg = yaml.safe_load(content) or {}
    return cfg.get("defaults") or {}


def compute_gz_model_name(model: str, instance: int) -> str:
    """Map PX4 gz model to Gazebo model name used on topics/pose.

    Examples:
    - gz_x500_gimbal_photographer + 2 -> x500_gimbal_photographer_2
    - gz_x500_lidar_gimbal_explorer + 3 -> x500_lidar_gimbal_explorer_3
    """
    base = model
    if model.startswith("gz_"):
        base = model[3:]
    return f"{base}_{instance}"
