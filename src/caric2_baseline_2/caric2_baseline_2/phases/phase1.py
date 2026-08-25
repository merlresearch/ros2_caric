# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Dict, List

import numpy as np
import numpy.linalg as la
import openscvx as ox

from .base import PhaseBase
from .openscvx_utils import guess_drone_position_trajectory_phase13, normalize_quaternion


class Phase1(PhaseBase):

    def __init__(self, controller):
        super().__init__(controller)
        self.phase = "phase1"
        self.problem_list, self.states, self.controls, self.time = self.setup_openscvx_problem()

    def setup_openscvx_problem(self):
        start_pos_enu = ox.Parameter("start_pos_enu", shape=(3,), value=np.array([0.0, 0.0, 0.0]))
        initial_attitude = ox.Parameter("initial_attitude", shape=(4,), value=np.array([1.0, 0.0, 0.0, 0.0]))
        initial_angular_velocity = ox.Parameter("initial_angular_velocity", shape=(3,), value=np.zeros((3,)))
        centered_buffered_end_pose_enu = ox.Parameter("centered_buffered_end_pose_enu", shape=(3,), value=np.ones((3,)))
        centered_true_end_pose_enu = ox.Parameter("centered_true_end_pose_enu", shape=(3,), value=np.ones((3,)))

        states, controls, dynamics, time, constraints = self.setup_common_openscvx_problem(start_pos_enu)

        centered_position, velocity, attitude, angular_velocity = states
        constraints.append((attitude == initial_attitude).convex().at([0]))
        constraints.append((angular_velocity == initial_angular_velocity).convex().at([0]))

        constraints.append(
            (time.at(self.openscvx_number_of_nodes - 2) + 1.0 <= time.at(self.openscvx_number_of_nodes - 1)).convex()
        )
        for t in [self.openscvx_number_of_nodes - 2, self.openscvx_number_of_nodes - 1]:
            constraints.append((centered_position == centered_buffered_end_pose_enu).convex().at([t]))
            constraints.append(
                (ox.linalg.Norm(velocity, ord=2) <= self.controller.common_problem_config["poi_vel_tol"])
                .convex()
                .at([t])
            )
            constraints.append(
                (
                    ox.linalg.Norm(angular_velocity, ord=2)
                    <= self.controller.common_problem_config["poi_angular_vel_tol"]
                )
                .convex()
                .at([t])
            )

        p_s_s = self.R_sb @ ox.spatial.QDCM(attitude).T @ (centered_true_end_pose_enu - centered_position)
        constraints.append(
            ox.ctcs(ox.linalg.Norm(self.A_cone @ p_s_s, ord=self.norm_type) - (self.c.T @ p_s_s) <= 0.0).over(
                (self.openscvx_number_of_nodes - 2, self.openscvx_number_of_nodes - 1)
            )
        )

        problem = self.setup_common_problem_initialize(
            dynamics=dynamics,
            states=states,
            controls=controls,
            time=time,
            constraints=constraints,
        )

        self.controller.get_logger().info("Phase 1 problem setup complete.")
        return problem, states, controls, time

    def generate_enu(
        self,
        start_pos_enu,
        start_vel_enu,
        start_attitude,
        start_angular_velocity,
        end_pos_enu,
        buffered_end_pos_enu,
    ) -> Dict[str, List]:
        start_attitude = normalize_quaternion(start_attitude)
        start_angular_velocity = np.asarray(start_angular_velocity, dtype=float)

        centered_true_end_pos_enu_value = np.array(end_pos_enu - start_pos_enu, dtype=float)
        centered_buffered_end_pos_enu_value = np.array(buffered_end_pos_enu - start_pos_enu, dtype=float)
        centered_position, velocity, attitude, angular_velocity = self.states
        centered_position.final = centered_buffered_end_pos_enu_value
        velocity.initial = start_vel_enu
        attitude.initial = start_attitude
        angular_velocity.initial = start_angular_velocity

        for autotuner_index in range(self.n_autotuners):
            self.problem_list[autotuner_index].parameters["start_pos_enu"] = np.array(start_pos_enu, dtype=float)
            self.problem_list[autotuner_index].parameters[
                "centered_true_end_pose_enu"
            ] = centered_true_end_pos_enu_value
            self.problem_list[autotuner_index].parameters[
                "centered_buffered_end_pose_enu"
            ] = centered_buffered_end_pos_enu_value
            self.problem_list[autotuner_index].parameters["initial_attitude"] = start_attitude
            self.problem_list[autotuner_index].parameters["initial_angular_velocity"] = start_angular_velocity

        self.centered_position_guess = guess_drone_position_trajectory_phase13(
            end_point_enu=centered_buffered_end_pos_enu_value,
            N=self.openscvx_number_of_nodes,
            position_min=centered_position.min,
            position_max=centered_position.max,
        )
        centered_position.guess = self.centered_position_guess

        b = self.R_sb @ np.array([0, 1, 0])
        a = np.squeeze(centered_true_end_pos_enu_value - centered_buffered_end_pos_enu_value)
        q_xyz = np.cross(b, a)
        q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
        q_no_norm = np.hstack((q_w, q_xyz))
        q = normalize_quaternion(q_no_norm)
        attitude.guess = np.tile(start_attitude, (self.openscvx_number_of_nodes, 1))
        attitude.guess[self.openscvx_number_of_nodes - 1] = q
        angular_velocity.guess = np.tile(start_angular_velocity, (self.openscvx_number_of_nodes, 1))

        self.update_time_guess(list_of_waypoints=[start_pos_enu, end_pos_enu])
        return self.generate_trajectory_from_openscvx_solve(start_pos_enu)
