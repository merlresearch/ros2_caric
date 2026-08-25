# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np


COMMON_PROBLEM_CONFIG = {
    "m": 2.08,
    "negative_g": -9.81,
    "J_b": np.array([1.0, 1.0, 1.0]),
    "max_xy_position": 100.0,
    "max_z_position": 100.0,
    "photographer_comm_radius": 10.0,
    "max_vel_hor": 50.0,
    "max_vel_ver": 50.0,
    "poi_vel_tol": 0.10,
    "poi_angular_vel_tol": 0.10,
    "max_angular_vel": 10.0,
    "drone_radius": 0.30,
    "ground_clearance": 2.5,
    "lse_alpha": 20.0,
    "safety_margin": 0.5,
    "time_parameter_multiplier_guess": 20.0,
    "use_fixed_dimension_buffering": 1,
}
COMMON_PROBLEM_CONFIG["time_parameter_magnitude_max_vel"] = 0.2 * np.linalg.norm(
    [
        COMMON_PROBLEM_CONFIG["max_vel_hor"],
        COMMON_PROBLEM_CONFIG["max_vel_hor"],
        COMMON_PROBLEM_CONFIG["max_vel_ver"],
    ]
)


def build_common_problem_config(overrides=None):
    overrides = overrides or {}
    config = {
        key: value.copy() if isinstance(value, np.ndarray) else value for key, value in COMMON_PROBLEM_CONFIG.items()
    }
    config.update(overrides)
    for key in ("J_b", "comm_center"):
        if key in config:
            config[key] = np.array(config[key], dtype=float)
    if "time_parameter_magnitude_max_vel" not in overrides:
        config["time_parameter_magnitude_max_vel"] = 0.2 * np.linalg.norm(
            [
                config["max_vel_hor"],
                config["max_vel_hor"],
                config["max_vel_ver"],
            ]
        )
    return config


def declare_phase_parameters(controller) -> None:
    controller.declare_parameter("vel_ff_scale", 0.8)
    controller.declare_parameter("acc_ff_scale", 0.3)
    controller.declare_parameter("acc_hor_max", 7.0)
    controller.declare_parameter("acc_up_max", 5.0)
    controller.declare_parameter("acc_down_max", 3.5)


def read_phase_parameters(controller) -> None:
    controller.vel_ff_scale = float(controller.get_parameter("vel_ff_scale").value)
    controller.acc_ff_scale = float(controller.get_parameter("acc_ff_scale").value)
    controller.acc_hor_max = float(controller.get_parameter("acc_hor_max").value)
    controller.acc_up_max = float(controller.get_parameter("acc_up_max").value)
    controller.acc_down_max = float(controller.get_parameter("acc_down_max").value)
