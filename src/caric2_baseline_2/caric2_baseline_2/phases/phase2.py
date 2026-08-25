# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import PhaseBase
from .phase1 import Phase1
from .concat import concat_traj


class Phase2(PhaseBase):
    def __init__(self, controller):
        super().__init__(controller)
        self.phase = "phase2"
        self._leg = Phase1(controller)

    def generate_enu(
        self,
        start_pos_enu,
        start_vel_enu,
        start_attitude,
        start_angular_velocity,
        original_pois_enu,
        list_buffered_pois_enu,
    ) -> Dict[str, List]:
        num_pois = len(list_buffered_pois_enu)
        if num_pois <= 0:
            raise RuntimeError("Phase 2 generate_enu called with zero POIs.")

        self._leg.phase = "phase1"

        cur_pos = np.array(start_pos_enu, dtype=float)
        cur_vel = np.array(start_vel_enu, dtype=float)
        cur_attitude = np.array(start_attitude, dtype=float)
        cur_angular_velocity = np.array(start_angular_velocity, dtype=float)
        leg_trajectories = []
        guess_segments = []
        for poi, buffered_poi in zip(original_pois_enu, list_buffered_pois_enu):
            try:
                leg_traj = self._leg.generate_enu(
                    cur_pos,
                    cur_vel,
                    cur_attitude,
                    cur_angular_velocity,
                    np.array(poi, dtype=float),
                    np.array(buffered_poi, dtype=float),
                )
                leg_trajectories.append(leg_traj)
                guess_segments.append(self._leg.centered_position_guess + cur_pos)
                cur_pos = np.array(leg_traj["position"][-1], dtype=float)
                cur_vel = np.array(leg_traj["velocity"][-1], dtype=float)
                cur_attitude = np.array(leg_traj["attitude"][-1], dtype=float)
                cur_angular_velocity = np.array(leg_traj["angular_velocity"][-1], dtype=float)
            except Exception as e:
                raise RuntimeError(f"Phase 2 failed to generate trajectory for POI {poi}") from e

        self.centered_position_guess = np.vstack(guess_segments) - np.array(start_pos_enu, dtype=float)
        return concat_traj(leg_trajectories)
