# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import Any

import numpy as np

from px4_msgs.msg import TrajectorySetpoint  # type: ignore

from ..phases.common import local_to_world as common_local_to_world
from ..phases.common import world_to_local as common_world_to_local


def publish_setpoint_enu(ctrl: Any, pos_enu, vel_enu, acc_enu, yaw: float, yawspeed: float) -> bool:
    msg = TrajectorySetpoint()
    lx, ly, lz = common_world_to_local(
        float(ctrl.spawn_x),
        float(ctrl.spawn_y),
        float(ctrl.spawn_yaw_deg),
        float(pos_enu[0]),
        float(pos_enu[1]),
        float(pos_enu[2]),
    )
    msg.position = [lx, ly, lz]

    if np.any(np.isnan(vel_enu)):
        msg.velocity = [float("nan"), float("nan"), float("nan")]
    else:
        # ENU velocity -> local NED
        e_dot = float(vel_enu[0])
        n_dot = float(vel_enu[1])
        u_dot = float(vel_enu[2])
        yaw_neg = math.radians(-float(ctrl.spawn_yaw_deg))
        c, s = math.cos(yaw_neg), math.sin(yaw_neg)
        vx_local = n_dot * c - e_dot * s
        vy_local = e_dot * c + n_dot * s
        vz_local = -u_dot
        vel_ff_scale = float(getattr(ctrl, "vel_ff_scale", 1.0))
        vel_scaled = np.array([vx_local, vy_local, vz_local], dtype=float) * vel_ff_scale
        msg.velocity = [float(vel_scaled[0]), float(vel_scaled[1]), float(vel_scaled[2])]

    if np.any(np.isnan(acc_enu)):
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
    else:
        # ENU acceleration -> local NED with clamping
        ae = float(acc_enu[0])
        an = float(acc_enu[1])
        au = float(acc_enu[2])
        yaw_neg = math.radians(-float(ctrl.spawn_yaw_deg))
        c, s = math.cos(yaw_neg), math.sin(yaw_neg)
        ax_local = an * c - ae * s
        ay_local = ae * c + an * s
        az_local = -au
        acc_ff_scale = float(getattr(ctrl, "acc_ff_scale", 1.0))
        acc_scaled = np.array([ax_local, ay_local, az_local], dtype=float) * acc_ff_scale
        acc_hor = np.linalg.norm(acc_scaled[:2])
        acc_hor_max = float(getattr(ctrl, "acc_hor_max", 7.0))
        if acc_hor > acc_hor_max and acc_hor > 1e-6:
            acc_scaled[:2] = acc_scaled[:2] * (acc_hor_max / acc_hor)
        acc_up = acc_scaled[2]
        acc_up_max = float(getattr(ctrl, "acc_up_max", 5.0))
        acc_down_max = float(getattr(ctrl, "acc_down_max", 3.5))
        if acc_up > acc_up_max:
            acc_scaled[2] = acc_up_max
        if acc_up < -acc_down_max:
            acc_scaled[2] = -acc_down_max
        msg.acceleration = [float(acc_scaled[0]), float(acc_scaled[1]), float(acc_scaled[2])]

    msg.yaw = float(yaw)
    msg.yawspeed = float(yawspeed)
    msg.jerk = [float("nan"), float("nan"), float("nan")]
    msg.timestamp = int(ctrl.get_clock().now().nanoseconds / 1000)
    ctrl.trajectory_setpoint_publisher.publish(msg)
    ctrl.sent_setpoints_enu.append([float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])])
    return True


def publish_basic_trajectory_setpoint(
    ctrl: Any, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = None
) -> None:
    msg = TrajectorySetpoint()
    msg.position = [x, y, z]
    msg.velocity = [float("nan"), float("nan"), float("nan")]
    msg.acceleration = [float("nan"), float("nan"), float("nan")]
    msg.yaw = 0.0 if yaw is None else float(yaw)
    msg.yawspeed = 0.0
    msg.jerk = [float("nan"), float("nan"), float("nan")]
    msg.timestamp = int(ctrl.get_clock().now().nanoseconds / 1000)
    ctrl.trajectory_setpoint_publisher.publish(msg)
    world_enu = common_local_to_world(
        float(ctrl.spawn_x),
        float(ctrl.spawn_y),
        float(ctrl.spawn_yaw_deg),
        float(x),
        float(y),
        float(z),
    )
    ctrl.sent_setpoints_enu.append([float(world_enu[0]), float(world_enu[1]), float(world_enu[2])])
