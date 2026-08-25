# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import pickle
import traceback
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from .config.phase_params import build_common_problem_config
from .phases.concat import generate_all_phases_upfront
from .phases.openscvx_utils import buffer_poi
from .phases.phase1 import Phase1
from .phases.phase2 import Phase2
from .phases.phase3 import Phase3


class WorkerLogger:
    def info(self, msg):
        print(f"[INFO] {msg}", flush=True)

    def warn(self, msg):
        print(f"[WARN] {msg}", flush=True)

    def error(self, msg):
        print(f"[ERROR] {msg}", flush=True)


class WorkerController:
    def __init__(self, request: Dict[str, Any]):
        self._logger = WorkerLogger()
        self.vehicle_local_position = SimpleNamespace(**request["vehicle_local_position_ned"])
        self.obstacle_boxes = _restore_obstacle_boxes(request["obstacle_boxes"])
        self.common_problem_config = build_common_problem_config(request["common_problem_config"])
        self.spawn_x = request["spawn_x"]
        self.spawn_y = request["spawn_y"]
        self.spawn_yaw_deg = request["spawn_yaw_deg"]
        self._phase2_ordered_pois_list_enu = np.array(request["phase2_ordered_pois_list_enu"], dtype=float)
        self._buffered_poi_list_enu = None

        self.openscvx_phase_times = {}
        self.end_trajectory_index = {"phase1": 0, "phase2": 0, "phase3": 0}
        self.optimal_trajectory = None
        self.openscvx_planned_enu = None

        self.phase1 = Phase1(self)
        self.phase2 = Phase2(self)
        self.phase3 = Phase3(self)

    @property
    def phase2_ordered_pois_list_enu(self):
        return self._phase2_ordered_pois_list_enu

    @property
    def buffered_poi_list_enu(self):
        if len(self.phase2_ordered_pois_list_enu) == 0:
            return []
        if self._buffered_poi_list_enu is None:
            self._buffered_poi_list_enu = buffer_poi(
                pois_enu=self.phase2_ordered_pois_list_enu,
                boxes_for_obstacles=self.obstacle_boxes,
                drone_radius=self.common_problem_config["drone_radius"],
                lse_alpha=self.common_problem_config["lse_alpha"],
                use_fixed_dimension_buffering=self.common_problem_config["use_fixed_dimension_buffering"],
                safety_margin=self.common_problem_config["safety_margin"],
            )
        return self._buffered_poi_list_enu

    def get_logger(self):
        return self._logger


def _restore_obstacle_boxes(obstacle_boxes):
    restored = []
    for box in obstacle_boxes:
        box_restored = dict(box)
        for key in ("center", "half"):
            if key in box_restored:
                box_restored[key] = np.array(box_restored[key], dtype=float)
        restored.append(box_restored)
    return restored


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


def _partial_response(controller: WorkerController, success: bool) -> Dict[str, Any]:
    response = {
        "success": success,
        "optimal_trajectory": _to_builtin(controller.optimal_trajectory),
        "openscvx_planned_enu": _to_builtin(controller.openscvx_planned_enu),
        "end_trajectory_index": _to_builtin(controller.end_trajectory_index),
        "openscvx_phase_times": _to_builtin(controller.openscvx_phase_times),
    }
    try:
        response["buffered_poi_list_enu"] = _to_builtin(controller.buffered_poi_list_enu)
    except BaseException as exc:
        response["buffered_poi_error"] = f"{type(exc).__name__}: {exc}"
    return response


def solve_request(request: Dict[str, Any]) -> Dict[str, Any]:
    controller = WorkerController(request)
    try:
        generate_all_phases_upfront(controller)
    except BaseException as exc:
        response = _partial_response(controller, success=False)
        response.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return response
    return _partial_response(controller, success=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Pickle file containing the worker request.")
    parser.add_argument("--response", required=True, help="Pickle file to write the worker response.")
    args = parser.parse_args(argv)

    with open(args.request, "rb") as f:
        request = pickle.load(f)

    try:
        response = solve_request(request)
    except BaseException as exc:
        response = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    with open(args.response, "wb") as f:
        pickle.dump(response, f)

    return 0 if response["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
