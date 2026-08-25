# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Any, Dict, List, Union

import jax.numpy as jnp
import numpy as np
import openscvx as ox


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return q / q_norm


def drone_states_with_few_undefined(
    N: int,
    max_xy_position,
    max_z_position,
    max_vel_hor,
    max_vel_ver,
    max_angular_vel,
):
    centered_position = ox.State("position", shape=(3,))
    centered_position.max = np.array([max_xy_position, max_xy_position, max_z_position])
    centered_position.min = np.array([-max_xy_position, -max_xy_position, -max_z_position])
    centered_position.initial = np.array([0, 0, 0], dtype=float)
    centered_position.final = [ox.Free(0), ox.Free(0), ox.Free(0)]
    centered_position.guess = np.zeros((N, 3), dtype=float)

    velocity = ox.State("velocity", shape=(3,))
    velocity.max = np.array([max_vel_hor, max_vel_hor, max_vel_ver])
    velocity.min = np.array([-max_vel_hor, -max_vel_hor, -max_vel_ver])
    velocity.initial = np.zeros((3,))
    velocity.final = [("free", 0), ("free", 0), ("free", 0)]
    velocity.guess = np.zeros((N, 3), dtype=float)

    attitude = ox.State("attitude", shape=(4,))
    attitude.max = np.array([1, 1, 1, 1])
    attitude.min = np.array([-1, -1, -1, -1])
    attitude.scaling_min = np.array([-2.0, -2.0, -2.0, -2.0])
    attitude.scaling_max = np.array([2.0, 2.0, 2.0, 2.0])
    attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
    attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
    attitude.guess = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (N, 1))

    angular_velocity = ox.State("angular_velocity", shape=(3,))
    angular_velocity.max = max_angular_vel * np.ones((3,))
    angular_velocity.min = -max_angular_vel * np.ones((3,))
    angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
    angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]
    angular_velocity.guess = np.zeros((N, 3), dtype=float)

    return centered_position, velocity, attitude, angular_velocity


def drone_controls(
    N: int,
    m: float,
    negative_g: float,
):
    thrust_force = ox.Control("thrust_force", shape=(3,))
    thrust_force.max = 5 * np.array([0, 0, 1.8 * m * (-negative_g)])  # ~1.8x weight, tuned for X500
    thrust_force.min = np.array([0, 0, 0])
    initial_control = np.array([0.0, 0.0, m * (-negative_g)])
    thrust_force.guess = np.repeat(np.expand_dims(initial_control, axis=0), N, axis=0)

    torque = ox.Control("torque", shape=(3,))
    torque.max = np.array([18.665, 18.665, 0.55562])
    torque.min = np.array([-18.665, -18.665, -0.55562])
    torque.scaling_min = np.array([-18.665, -18.665, -2.0])
    torque.scaling_max = np.array([18.665, 18.665, 2.0])
    torque.guess = np.zeros((N, 3))

    return thrust_force, torque


def drone_dynamics(
    states: List[ox.State],
    controls: List[ox.Control],
    m: float,
    negative_g: float,
    J_b: np.ndarray,
):
    _, velocity, attitude, angular_velocity = states
    thrust_force, torque = controls

    q_norm = ox.linalg.Norm(attitude)
    attitude_normalized = attitude / q_norm

    J_b_inv = 1.0 / J_b
    J_b_diag = ox.linalg.Diag(J_b)

    return {
        "position": velocity,
        "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
        + np.array([0, 0, negative_g], dtype=np.float64),
        "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
        "angular_velocity": ox.linalg.Diag(J_b_inv)
        @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    }


def static_obstacle_avoidance_ctcs(
    centered_position: ox.State,
    boxes_for_obstacles: List[Dict[str, Any]],
    start_pos_enu: List[np.ndarray],
    drone_radius: float,
    lse_alpha: float,
    evaluate: bool,
) -> Union[List[ox.CTC], List[float]]:
    constraints = []
    for b in boxes_for_obstacles:
        d = centered_position - (b["center"] - start_pos_enu)
        if evaluate:
            lhs_constraint_evaluation = _static_obstacle_lhs_numeric(
                d=d,
                half=b["half"],
                yaw=b["yaw"],
                drone_radius=drone_radius,
                lse_alpha=lse_alpha,
            )
            constraints.extend(np.atleast_1d(lhs_constraint_evaluation).tolist())
        else:
            lhs_constraint = _static_obstacle_lhs_symbolic(
                d=d,
                half=b["half"],
                yaw=b["yaw"],
                drone_radius=drone_radius,
                lse_alpha=lse_alpha,
            )
            constraints.append(ox.ctcs((lhs_constraint >= 1.0), idx=1))
    return constraints


def _static_obstacle_lhs_numeric(
    d: np.ndarray,
    half: np.ndarray,
    yaw: float,
    drone_radius: float,
    lse_alpha: float,
):
    inflated_half = np.asarray(half, dtype=float) + drone_radius
    rotated_d = rotation_matrix_z(-yaw) @ d
    lse_terms = np.atleast_2d(lse_alpha * np.abs(rotated_d) / inflated_half)
    max_lse_terms = np.max(lse_terms, axis=1, keepdims=True)
    smooth_max = (np.squeeze(max_lse_terms) + np.log(np.sum(np.exp(lse_terms - max_lse_terms), axis=1))) / lse_alpha
    return smooth_max - (np.log(3) / lse_alpha)


def _static_obstacle_lhs_symbolic(
    d,
    half: np.ndarray,
    yaw: float,
    drone_radius: float,
    lse_alpha: float,
):
    inflated_half = jnp.array(half, dtype=jnp.float32) + drone_radius
    rotated_d = rotation_matrix_z(-yaw) @ d
    lse_terms = [
        lse_alpha * ox.symbolic.expr.Abs(rotated_d_coord) / inflated_half_coord
        for rotated_d_coord, inflated_half_coord in zip(rotated_d, inflated_half)
    ]
    smooth_max = ox.symbolic.expr.LogSumExp(*lse_terms) / lse_alpha
    return smooth_max - (np.log(3) / lse_alpha)


def guess_drone_position_trajectory_phase13(
    end_point_enu: np.ndarray, N: int, position_min: np.ndarray, position_max: np.ndarray
):
    if np.any(position_min > position_max):
        raise ValueError("position_min cannot be greater than position_max")
    if np.any(end_point_enu <= position_min) or np.any(end_point_enu >= position_max):
        raise ValueError("All centered_pois_enu are out of bounds defined by position_min and position_max")
    ideal_waypoints_including_start = np.vstack((np.zeros((3,)), end_point_enu))
    t_nodes = np.linspace(0, 1, N)
    interpolated_waypoints = np.empty((N, 3))
    for i in range(3):
        interpolated_waypoints[:, i] = np.interp(t_nodes, [0, 1], ideal_waypoints_including_start[:, i])
    return interpolated_waypoints


def compute_drone_position_trajectory_from_solution(
    results: ox.OptimizationResults,
    start_pos_enu: np.ndarray,
    m: float,
    negative_g: float,
) -> Dict:
    traj = {
        k: []
        for k in [
            "position",
            "velocity",
            "acceleration",
            "attitude",
            "angular_velocity",
            "yaw",
            "yawspeed",
            "time",
        ]
    }
    traj["success"] = True

    Nres = results.trajectory["position"].shape[0]
    traj["yaw"] = [0.0] * Nres
    traj["yawspeed"] = [0.0] * Nres
    traj["position"] = (results.trajectory["position"] + start_pos_enu).tolist()
    traj["velocity"] = np.asarray(results.trajectory["velocity"], dtype=np.float32).tolist()
    traj["attitude"] = np.asarray(results.trajectory["attitude"], dtype=np.float32).tolist()
    traj["angular_velocity"] = np.asarray(results.trajectory["angular_velocity"], dtype=np.float32).tolist()
    traj["time"] = np.squeeze(np.asarray(results.trajectory["time"], dtype=np.float32)).tolist()

    attitude_as_array = np.asarray(results.trajectory["attitude"], dtype=np.float32)
    force_as_array = np.asarray(results.trajectory["thrust_force"], dtype=np.float32)
    accel = []
    for attitude_normal, force in zip(attitude_as_array, force_as_array):
        qcdm_matrix = quat_to_dcm(attitude_normal)
        acceleration = (1.0 / m) * qcdm_matrix @ force + np.array([0, 0, negative_g])
        accel.append(acceleration.tolist())
    traj["acceleration"] = accel
    return traj


def is_inside_any_of_obstacles(lhs):
    lhs_array = np.array(lhs)
    if lhs_array.ndim != 1:
        raise ValueError("lhs must be a 1D array or list")
    return (np.array(lhs) <= 1.0).any()


def rotation_matrix_z(yaw: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]], dtype=np.float32)


def buffer_poi(
    pois_enu: List[np.ndarray],
    boxes_for_obstacles: List[Dict[str, Any]],
    drone_radius: float,
    lse_alpha: float,
    use_fixed_dimension_buffering: int | None,
    safety_margin: float,
):
    list_buffered_pois = []
    for poi in pois_enu:
        buffered_poi = poi.copy()
        for b in boxes_for_obstacles:
            d = poi - b["center"]
            h_j = jnp.array(b["half"], dtype=jnp.float32)
            inflated_h_j = h_j + drone_radius

            rotated_d = rotation_matrix_z(-b["yaw"]) @ d

            rotated_d_is_in_box_b = (np.abs(rotated_d / inflated_h_j) < 1.0).all()
            if rotated_d_is_in_box_b:
                if use_fixed_dimension_buffering is not None:
                    dimension_to_buffer = use_fixed_dimension_buffering
                else:
                    rotated_xy = rotated_d[:2]
                    inflated_xy = inflated_h_j[:2]
                    distances_to_boundary_xy = inflated_xy - np.abs(rotated_xy)
                    dimension_to_buffer = int(np.argmin(distances_to_boundary_xy))

                rotated_buffered_d = rotated_d.copy()
                rotated_buffered_d[dimension_to_buffer] = (
                    np.sign(rotated_d[dimension_to_buffer]) * inflated_h_j[dimension_to_buffer]
                )

                buffered_poi = rotation_matrix_z(b["yaw"]) @ rotated_buffered_d + b["center"]
                lhs_static_obstacle = static_obstacle_avoidance_ctcs(
                    centered_position=buffered_poi,
                    boxes_for_obstacles=boxes_for_obstacles,
                    start_pos_enu=np.zeros((3,)),
                    drone_radius=drone_radius,
                    lse_alpha=lse_alpha,
                    evaluate=True,
                )
                if is_inside_any_of_obstacles(lhs_static_obstacle):
                    print("Buffered POI is still inside obstacle, applying iterative correction!")
                    correction_norm_ub = 1.0
                    correction_vec_norm = np.linalg.norm(buffered_poi - poi)
                    if correction_vec_norm < 1e-6:
                        correction_direction = np.random.randn(3)
                        correction_direction /= np.linalg.norm(correction_direction)
                    else:
                        correction_direction = (buffered_poi - poi) / correction_vec_norm
                    while is_inside_any_of_obstacles(lhs_static_obstacle):
                        correction_norm_lb = correction_norm_ub
                        correction_norm_ub *= 2.0
                        temp_buffered_poi = buffered_poi + (correction_norm_ub * correction_direction)
                        lhs_static_obstacle = static_obstacle_avoidance_ctcs(
                            centered_position=temp_buffered_poi,
                            boxes_for_obstacles=boxes_for_obstacles,
                            start_pos_enu=np.zeros((3,)),
                            drone_radius=drone_radius,
                            lse_alpha=lse_alpha,
                            evaluate=True,
                        )
                    while correction_norm_ub >= correction_norm_lb + 0.01:
                        correction_norm_mid = 0.5 * (correction_norm_lb + correction_norm_ub)
                        temp_buffered_poi = buffered_poi + (correction_norm_mid * correction_direction)
                        lhs_static_obstacle = static_obstacle_avoidance_ctcs(
                            centered_position=temp_buffered_poi,
                            boxes_for_obstacles=boxes_for_obstacles,
                            start_pos_enu=np.zeros((3,)),
                            drone_radius=drone_radius,
                            lse_alpha=lse_alpha,
                            evaluate=True,
                        )
                        if is_inside_any_of_obstacles(lhs_static_obstacle):
                            correction_norm_lb = correction_norm_mid
                        else:
                            correction_norm_ub = correction_norm_mid
                    print(f"Final correction norm: {correction_norm_ub + safety_margin}")
                    buffered_poi = buffered_poi + ((correction_norm_ub + safety_margin) * correction_direction)
                break
        list_buffered_pois.append(buffered_poi)
    return np.array(list_buffered_pois)
