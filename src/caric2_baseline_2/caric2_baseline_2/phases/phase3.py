# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Dict, List

import numpy as np
import openscvx as ox

from .base import PhaseBase
from .openscvx_utils import guess_drone_position_trajectory_phase13, normalize_quaternion


class Phase3(PhaseBase):
    def __init__(self, controller):
        super().__init__(controller)
        self.phase = "phase3"
        self.problem_list, self.states, self.controls, self.time = self.setup_openscvx_problem()

    def setup_openscvx_problem(self):
        start_pos_enu = ox.Parameter("start_pos_enu", shape=(3,), value=np.array([0.0, 0.0, 0.0]))
        initial_attitude = ox.Parameter("initial_attitude", shape=(4,), value=np.array([1.0, 0.0, 0.0, 0.0]))
        initial_angular_velocity = ox.Parameter("initial_angular_velocity", shape=(3,), value=np.zeros((3,)))

        states, controls, dynamics, time, constraints = self.setup_common_openscvx_problem(start_pos_enu)

        centered_position, velocity, attitude, angular_velocity = states
        constraints.append((attitude == initial_attitude).convex().at([0]))
        constraints.append((angular_velocity == initial_angular_velocity).convex().at([0]))

        centered_comm_center = self.controller.common_problem_config["comm_center"] - start_pos_enu
        constraints.append(
            (
                ox.linalg.Norm(centered_position - centered_comm_center, ord=2)
                <= self.controller.common_problem_config["photographer_comm_radius"]
            )
            .convex()
            .at([self.openscvx_number_of_nodes - 1])
        )
        constraints.append(
            (ox.linalg.Norm(velocity, ord=2) <= self.controller.common_problem_config["poi_vel_tol"])
            .convex()
            .at([self.openscvx_number_of_nodes - 1])
        )

        problem = self.setup_common_problem_initialize(
            dynamics=dynamics,
            states=states,
            controls=controls,
            time=time,
            constraints=constraints,
        )

        self.controller.get_logger().info("Phase 3 problem setup complete.")
        return problem, states, controls, time

    def generate_enu(
        self,
        start_pos_enu,
        start_vel_enu,
        start_attitude,
        start_angular_velocity,
    ) -> Dict[str, List]:

        centered_position, velocity, attitude, angular_velocity = self.states
        velocity.initial = start_vel_enu
        start_attitude = normalize_quaternion(start_attitude)
        start_angular_velocity = np.asarray(start_angular_velocity, dtype=float)
        attitude.initial = start_attitude
        attitude.guess = np.tile(start_attitude, (self.openscvx_number_of_nodes, 1))
        angular_velocity.initial = start_angular_velocity
        angular_velocity.guess = np.tile(start_angular_velocity, (self.openscvx_number_of_nodes, 1))

        for autotuner_index in range(self.n_autotuners):
            self.problem_list[autotuner_index].parameters["start_pos_enu"] = np.array(start_pos_enu, dtype=float)
            self.problem_list[autotuner_index].parameters["initial_attitude"] = start_attitude
            self.problem_list[autotuner_index].parameters["initial_angular_velocity"] = start_angular_velocity

        centered_comm_center = np.array(
            self.controller.common_problem_config["comm_center"] - start_pos_enu, dtype=float
        )
        self.centered_position_guess = guess_drone_position_trajectory_phase13(
            end_point_enu=centered_comm_center,
            N=self.openscvx_number_of_nodes,
            position_min=centered_position.min,
            position_max=centered_position.max,
        )
        centered_position.guess = self.centered_position_guess

        distance_to_comm_center = np.linalg.norm(centered_comm_center)
        if distance_to_comm_center > self.controller.common_problem_config["photographer_comm_radius"]:
            end_pos_enu_on_boundary_of_comm_ball = (
                start_pos_enu
                + (
                    (distance_to_comm_center - self.controller.common_problem_config["photographer_comm_radius"])
                    / distance_to_comm_center
                )
                * centered_comm_center
            )
            self.update_time_guess(list_of_waypoints=[start_pos_enu, end_pos_enu_on_boundary_of_comm_ball])
            return self.generate_trajectory_from_openscvx_solve(start_pos_enu)
        else:
            raise RuntimeError(
                "Drone is already very close to the comm center. Phase 3 should not be called in this case. "
                f"Current distance to comm center: {distance_to_comm_center} < photographer_comm_radius: "
                f"{self.controller.common_problem_config['photographer_comm_radius']}"
            )
