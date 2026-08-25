# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import io
import contextlib as ctx
from collections import deque

import numpy as np
import openscvx as ox
from openscvx import Problem as TrajOptProblem

from .openscvx_utils import (
    compute_drone_position_trajectory_from_solution,
    drone_controls,
    drone_dynamics,
    drone_states_with_few_undefined,
    static_obstacle_avoidance_ctcs,
)

OPENSCVX_DIAGNOSTIC_MODE = False  # Set to True to save diagnostic data when OpenSCVX fails to converge
OPENSCVX_RAISE_ERROR_ON_FAILURE = True  # Set to True to raise an error when OpenSCVX fails to converge


class PhaseBase:
    def __init__(self, controller: Any) -> None:
        self.controller = controller
        if not hasattr(self.controller, "openscvx_phase_times"):
            self.controller.openscvx_phase_times = {}
        self.phase = ""
        self.openscvx_number_of_nodes = 10  # Phase 2 overwrites it based on n_poi
        self.centered_position_guess = None

        self.autotuner_list = [
            {"autotuner": ox.AdaptiveProximalWeight()},
            {"autotuner": ox.AugmentedLagrangian()},
        ]
        self.n_autotuners = len(self.autotuner_list)

        alpha_x = 1.396263  # 80 deg horizontal FOV
        aspect_ratio = 1280 / 720
        alpha_y = alpha_x / aspect_ratio
        self.A_cone = np.diag(
            [
                1 / np.tan(alpha_x),
                1 / np.tan(alpha_y),
                0,
            ]
        )  # Conic Matrix in Sensor Frame
        self.c = np.array([0, 0, 1])
        self.norm_type = "inf"
        # Align camera axis (x) with drone forward axis (x)
        self.R_sb = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def generate(self, *args, **kwargs) -> Dict[str, List]:
        raise NotImplementedError

    def setup_common_openscvx_problem(
        self, start_pos_enu: ox.Parameter
    ) -> Tuple[List[ox.State], List[ox.Control], ox.Dynamics, ox.Time, List[ox.Constraint]]:
        centered_position, velocity, attitude, angular_velocity = drone_states_with_few_undefined(
            N=self.openscvx_number_of_nodes,
            max_xy_position=self.controller.common_problem_config["max_xy_position"],
            max_z_position=self.controller.common_problem_config["max_z_position"],
            max_vel_hor=self.controller.common_problem_config["max_vel_hor"],
            max_vel_ver=self.controller.common_problem_config["max_vel_ver"],
            max_angular_vel=self.controller.common_problem_config["max_angular_vel"],
        )
        self.centered_position_guess = centered_position.guess
        states = [centered_position, velocity, attitude, angular_velocity]

        thrust_force, torque = drone_controls(
            N=self.openscvx_number_of_nodes,
            m=self.controller.common_problem_config["m"],
            negative_g=self.controller.common_problem_config["negative_g"],
        )
        controls = [thrust_force, torque]

        dynamics = drone_dynamics(
            states=states,
            controls=controls,
            m=self.controller.common_problem_config["m"],
            negative_g=self.controller.common_problem_config["negative_g"],
            J_b=self.controller.common_problem_config["J_b"],
        )
        time = ox.Time(initial=0.0, min=0.0, max=1e4, final=("minimize", 1.0))  # see self.update_time_guess

        constraints = []
        constraints_static_obstacle_avoidance_ctcs = static_obstacle_avoidance_ctcs(
            centered_position=centered_position,
            boxes_for_obstacles=self.controller.obstacle_boxes,
            start_pos_enu=start_pos_enu,
            drone_radius=self.controller.common_problem_config["drone_radius"],
            lse_alpha=self.controller.common_problem_config["lse_alpha"],
            evaluate=False,
        )
        constraints.extend(constraints_static_obstacle_avoidance_ctcs)
        for state in states:
            constraints.append(ox.ctcs(state <= state.max, idx=0))
            constraints.append(ox.ctcs(state >= state.min, idx=0))
        constraints.append(
            ox.ctcs(
                centered_position[2] + start_pos_enu[2] >= self.controller.common_problem_config["ground_clearance"],
                idx=0,
            )
        )

        return states, controls, dynamics, time, constraints

    def setup_common_problem_initialize(
        self,
        dynamics: ox.Dynamics,
        states: List[ox.State],
        controls: List[ox.Control],
        time: ox.Time,
        constraints: List[ox.Constraint],
    ):
        problem_list = []
        for algorithm in self.autotuner_list:
            problem = TrajOptProblem(
                dynamics=dynamics,
                states=states,
                controls=controls,
                time=time,
                constraints=constraints,
                N=self.openscvx_number_of_nodes,
                algorithm=algorithm,
            )

            problem.settings.prp.dt = 0.1
            problem.settings.sim.save_compiled = True
            problem.initialize()
            problem_list.append(problem)
        return problem_list

    def generate_trajectory_from_openscvx_solve(self, start_pos_enu):
        timing_init, timing_solve, timing_post = 0, 0, 0
        for ep_vc in [1e-4, 1e-3]:
            for autotuner_index in range(self.n_autotuners):
                type_of_autotuner = type(self.autotuner_list[autotuner_index]["autotuner"])
                self.controller.get_logger().info(
                    f"Trying with {str(type_of_autotuner):s} and ep_vc={ep_vc:.1e} for phase {self.phase}."
                )
                self.problem_list[autotuner_index].algorithm.ep_vc = ep_vc
                self.problem_list[autotuner_index].reset()
                buf = io.StringIO()
                with ctx.redirect_stdout(buf), ctx.redirect_stderr(buf):
                    self.problem_list[autotuner_index].solve()
                results = self.problem_list[autotuner_index].post_process()
                timing_init += self.problem_list[autotuner_index].timing_init
                timing_solve += self.problem_list[autotuner_index].timing_solve
                timing_post += self.problem_list[autotuner_index].timing_post
                if results.converged:
                    self.controller.get_logger().info(
                        f"CONVERGED on Phase {self.phase:s} | {str(type_of_autotuner):s}, ep_vc={ep_vc:.1e}."
                    )
                    break
                else:
                    self.controller.get_logger().info(
                        f"DID NOT CONVERGE on Phase {self.phase:s} | {str(type_of_autotuner):s}, ep_vc={ep_vc:.1e}."
                    )
                    last_20 = deque(buf.getvalue().splitlines(), maxlen=20)
                    self.controller.get_logger().info("Last 20 lines of OpenSCVX output:")
                    for line in last_20:
                        self.controller.get_logger().info(line)
            if results.converged:
                break

        if not results.converged:
            if OPENSCVX_DIAGNOSTIC_MODE:
                import pickle
                import time

                data = {
                    "vehicle_local_position_ned": {
                        "x": self.controller.vehicle_local_position.x,
                        "y": self.controller.vehicle_local_position.y,
                        "z": self.controller.vehicle_local_position.z,
                        "vx": self.controller.vehicle_local_position.vx,
                        "vy": self.controller.vehicle_local_position.vy,
                        "vz": self.controller.vehicle_local_position.vz,
                    },
                    "common_problem_config": self.controller.common_problem_config,
                    "spawn_x": self.controller.spawn_x,
                    "spawn_y": self.controller.spawn_y,
                    "spawn_yaw_deg_rad": self.controller.spawn_yaw_deg,
                    "pois": self.controller.phase2_ordered_pois_list_enu,
                    "obstacle_boxes": self.controller.obstacle_boxes,
                    "failure_title": f"{self.phase}",
                }
                datetime_str = time.strftime("%Y%m%d_%H%M%S")
                save_pickle_path = f"{datetime_str:s}_openscvx_failure_{self.phase}.pkl"
                self.controller.get_logger().info(
                    f"OpenSCVX failed to converge in {self.phase}. Saving diagnostic data to {save_pickle_path:s}."
                )
                with open(save_pickle_path, "wb") as f:
                    pickle.dump(data, f)
                try:
                    import os
                    from openscvx.plotting import plot_states

                    os.makedirs("figures", exist_ok=True)
                    plot_states(results, style="publication", pdf_path=f"figures/{self.phase:s}_states.pdf")
                except Exception as exc:
                    self.controller.get_logger().warn(f"OpenSCVX diagnostic plotting failed for {self.phase}: {exc}")
            elif OPENSCVX_RAISE_ERROR_ON_FAILURE:
                raise RuntimeError(f"OpenSCVX failed to converge in {self.phase}.")
            else:
                self.controller.get_logger().info(
                    f"OpenSCVX failed to converge in {self.phase}, moving on with last solution with algorithm "
                    f"{str(type(self.autotuner_list[autotuner_index]['autotuner'])):s} and ep_vc {ep_vc:.1e}."
                )

        self.controller.openscvx_phase_times[self.phase] = {
            "preprocessing_time": timing_init,
            "main_solve_time": timing_solve,
            "postprocessing_time": timing_post,
            "total_time": timing_init + timing_solve + timing_post,
        }

        return compute_drone_position_trajectory_from_solution(
            results=results,
            start_pos_enu=start_pos_enu,
            m=self.controller.common_problem_config["m"],
            negative_g=self.controller.common_problem_config["negative_g"],
        )

    def update_time_guess(self, list_of_waypoints) -> None:
        if self.phase == "phase2":
            raise RuntimeError("Phase 2 time guess was not properly loaded!")
        total_distance = 0.0
        if len(list_of_waypoints) != 2:
            raise ValueError("List of waypoints must be two points to compute travel distance.")
        total_distance = np.linalg.norm(list_of_waypoints[1] - list_of_waypoints[0])
        total_travel_time = total_distance / self.controller.common_problem_config["time_parameter_magnitude_max_vel"]
        time_guess = total_travel_time * self.controller.common_problem_config["time_parameter_multiplier_guess"]
        self.time.final = ("minimize", time_guess)
        print(f"Updated time guess for phase {self.phase} = {time_guess:.2f} s. N_waypoints: {len(list_of_waypoints)}")
