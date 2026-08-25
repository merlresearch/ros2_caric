# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
import os

from setuptools import find_packages, setup

package_name = "caric2_baseline_1"

# Ensure resource marker exists (required by ROS2 ament index)
resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource")
os.makedirs(resource_dir, exist_ok=True)
resource_file = os.path.join(resource_dir, package_name)
if not os.path.exists(resource_file):
    with open(resource_file, "w") as f:
        f.write(package_name + "\n")


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


launch_files = package_files("launch")

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", launch_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Abraham P. Vinod",
    maintainer_email="vinod@merl.com",
    description="CARIC2 Baseline 1 (A*)",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "vehicle_controller_astar = caric2_baseline_1.vehicle_controller_astar:main",
            "vehicle_controller_explorer = caric2_baseline_1.vehicle_controller_explorer:main",
            "photographer_coordinator = caric2_baseline_1.photographer_coordinator:main",
            "gcs = caric2_baseline_1.gcs_node:main",
            "poi_detection_node = caric2_baseline_1.poi_detection_node:main",
            "cluster_manager = caric2_baseline_1.cluster_manager:main",
        ],
    },
)
