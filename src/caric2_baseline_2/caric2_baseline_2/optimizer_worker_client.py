# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
"""Python 3.10-safe client for the required OpenSCvx Python 3.12 worker."""
from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _default_worker_python() -> str:
    workspace_env = os.environ.get("ROS2_CARIC_WS")
    if workspace_env:
        return str(Path(workspace_env).expanduser() / ".venv-py312-openscvx" / "bin" / "python")

    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for path in (start, *start.parents):
            candidate = path / ".venv-py312-openscvx" / "bin" / "python"
            if candidate.exists():
                return str(candidate)
    return str(Path.home() / "ros2_ws" / ".venv-py312-openscvx" / "bin" / "python")


def _worker_cache_root(python_executable: str) -> Path:
    python_path = Path(python_executable).absolute()
    if ".venv-py312-openscvx" in python_path.parts:
        return python_path.parents[2]
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for path in (start, *start.parents):
            if (path / ".venv-py312-openscvx").exists():
                return path
    return Path.cwd().resolve()


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _build_request(controller) -> Dict[str, Any]:
    vehicle_local_position = controller.vehicle_local_position
    return {
        "vehicle_local_position_ned": {
            "x": vehicle_local_position.x,
            "y": vehicle_local_position.y,
            "z": vehicle_local_position.z,
            "vx": vehicle_local_position.vx,
            "vy": vehicle_local_position.vy,
            "vz": vehicle_local_position.vz,
        },
        "common_problem_config": _to_builtin(controller.common_problem_config),
        "spawn_x": controller.spawn_x,
        "spawn_y": controller.spawn_y,
        "spawn_yaw_deg": controller.spawn_yaw_deg,
        "phase2_ordered_pois_list_enu": _to_builtin(controller.phase2_ordered_pois_list_enu),
        "obstacle_boxes": _to_builtin(controller.obstacle_boxes),
    }


def solve_with_optimizer_worker(controller) -> bool:
    python_executable = _default_worker_python()
    if not os.path.exists(python_executable):
        raise RuntimeError(
            "OpenSCvx optimizer worker Python was not found: "
            f"{python_executable}. Follow the baseline 2 Python 3.12 worker setup in README.md."
        )

    request = _build_request(controller)
    with tempfile.TemporaryDirectory(prefix="openscvx_worker_") as tmpdir:
        request_path = os.path.join(tmpdir, "request.pkl")
        response_path = os.path.join(tmpdir, "response.pkl")
        with open(request_path, "wb") as f:
            pickle.dump(request, f)

        env = os.environ.copy()
        workspace = _worker_cache_root(python_executable)
        env.setdefault("XDG_CACHE_HOME", os.path.join(str(workspace), ".cache-py312"))
        env.setdefault("MPLCONFIGDIR", os.path.join(str(workspace), ".mplconfig-py312"))

        cmd = [
            python_executable,
            "-m",
            "caric2_baseline_2.optimizer_worker",
            "--request",
            request_path,
            "--response",
            response_path,
        ]
        completed = subprocess.run(cmd, env=env)

        if not os.path.exists(response_path):
            raise RuntimeError(
                "OpenSCvx optimizer worker did not produce a response. " f"returncode={completed.returncode}"
            )

        with open(response_path, "rb") as f:
            response = pickle.load(f)

    if "optimal_trajectory" in response:
        controller.optimal_trajectory = response["optimal_trajectory"]
    if "openscvx_planned_enu" in response:
        controller.openscvx_planned_enu = response["openscvx_planned_enu"]
    if "buffered_poi_list_enu" in response:
        controller._buffered_poi_list_enu = response.get("buffered_poi_list_enu")
    if "end_trajectory_index" in response:
        controller.end_trajectory_index = response["end_trajectory_index"]
    if "openscvx_phase_times" in response:
        controller.openscvx_phase_times = response["openscvx_phase_times"]

    if not response.get("success"):
        raise RuntimeError(
            "OpenSCvx optimizer worker failed: "
            f"{response.get('error_type')}: {response.get('error')}\n{response.get('traceback', '')}"
        )

    return True
