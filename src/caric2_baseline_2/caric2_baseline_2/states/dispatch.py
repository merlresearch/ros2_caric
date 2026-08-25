# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

from . import arming_takeoff as st_arm
from . import follow as st_follow
from . import generate as st_gen
from . import init_prestream as st_init
from . import waypoint_land as st_wpt

if TYPE_CHECKING:
    from ..vehicle_controller_openscvx import VehicleControllerOpenSCvx


def dispatch_state(ctrl: VehicleControllerOpenSCvx) -> None:
    state = ctrl.flight_state

    if state == "INIT":
        st_init.handle_init(ctrl)
    elif state == "PRESTREAM":
        st_init.handle_prestream(ctrl)
    elif state == "ARMING":
        st_arm.handle_arming(ctrl)
    elif state == "TAKEOFF":
        st_arm.handle_takeoff(ctrl)
    elif state == "HOVER_AFTER_TAKEOFF":
        st_arm.handle_hover_after_takeoff(ctrl)
    elif state == "GENERATE_ALL_PHASES":
        st_gen.handle_generate_all_phases(ctrl)
    elif state == "FOLLOW_TRAJECTORY":
        st_follow.handle_follow_trajectory(ctrl)
    elif state == "RECOVER_AFTER_FAILURE":
        st_wpt.handle_recover_after_failure(ctrl)
    elif state == "LAND":
        st_wpt.handle_land(ctrl)
    elif state == "LANDED":
        st_wpt.handle_landed(ctrl)
    elif state == "DISARMED":
        st_wpt.handle_disarmed(ctrl)
    elif state == "WAITING_FOR_ASSIGNMENT":
        st_wpt.handle_waiting_for_assignment(ctrl)
