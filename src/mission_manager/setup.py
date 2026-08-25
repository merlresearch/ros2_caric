# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
import os

from setuptools import find_packages, setup

package_name = "mission_manager"


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


def collect_data_files(src_root: str, install_subdir: str):
    """Collect data_files preserving subdirectory structure under install_subdir.

    Returns list of tuples: [(dest_dir, [file1, file2, ...]), ...]
    """
    entries = {}
    if not os.path.isdir(src_root):
        return []
    for root, _dirs, files in os.walk(src_root):
        if not files:
            continue
        rel_dir = os.path.relpath(root, src_root)
        if rel_dir == ".":
            dest = os.path.join("share", package_name, install_subdir)
        else:
            dest = os.path.join("share", package_name, install_subdir, rel_dir)
        entries.setdefault(dest, [])
        for f in files:
            entries[dest].append(os.path.join(root, f))
    return [(dest, files) for dest, files in entries.items()]


launch_files = package_files("launch")
config_files = package_files("config")
models_data_files = collect_data_files("models", "models")
worlds_data_files = collect_data_files("worlds", "worlds")

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", launch_files),
        ("share/" + package_name + "/config", config_files),
    ]
    + models_data_files
    + worlds_data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Abraham P. Vinod",
    maintainer_email="vinod@merl.com",
    description="Mission benchmarking for CARIC 2",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            # direct entry points
            "referee_node = mission_manager.referee_node:main",
            "poi_scoring_node = mission_manager.poi_scoring_node:main",
            "photo_capture_service = mission_manager.photo_capture_service:main",
            "mission_timing_logger = mission_manager.mission_timing_logger:main",
            "mission_controller = mission_manager.mission_controller:main",
            "spawner = mission_manager.spawner:main",
            "fleet_readiness_node = mission_manager.fleet_readiness_node:main",
            "mapping_node = mission_manager.mapping_node:main",
            "position_logger = mission_manager.position_logger:main",
            "astar_context_recorder = mission_manager.astar_context_recorder:main",
        ],
    },
)
