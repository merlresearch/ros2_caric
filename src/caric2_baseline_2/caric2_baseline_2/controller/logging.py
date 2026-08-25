# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from std_msgs.msg import String  # noqa: E402

CLUSTER_EVENTS = frozenset(
    [
        "takeoff_complete",
        "assignment_start",
        "cluster_start",
        "cluster_complete",
        "home_return",
    ]
)


def publish_timing_event(ctrl: Any, event_type: str, additional_data: Optional[Dict] = None) -> None:
    current_time = ctrl.get_clock().now().nanoseconds / 1e9

    if event_type in CLUSTER_EVENTS:
        timing_data = {
            "photographer_id": getattr(ctrl, "photographer_id", "unknown"),
            "assignment_id": getattr(ctrl, "current_assignment_id", None),
            "cluster_id": getattr(ctrl, "current_cluster_id", None),
            "timing_event": event_type,
            "timestamp": current_time,
        }
        if additional_data:
            timing_data["metrics"] = additional_data

        msg = String()
        msg.data = json.dumps(timing_data)
        ctrl.photographer_timing_publisher.publish(msg)
    else:
        timing_data = {
            "vehicle_id": getattr(ctrl, "photographer_id", "openscvx_drone"),
            "event_type": event_type,
            "timestamp": current_time,
            "iso_timestamp": datetime.now().isoformat(),
        }
        if additional_data:
            timing_data.update(additional_data)

        msg = String()
        msg.data = json.dumps(timing_data)
        ctrl.timing_publisher.publish(msg)


def save_sent_trajectory_artifacts(ctrl: Any, reason_suffix: str = "") -> None:
    if getattr(ctrl, "sent_traj_saved", False):
        return
    if not getattr(ctrl, "sent_setpoints_enu", None):
        return

    os.makedirs(ctrl.trajectory_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    world = getattr(ctrl, "world", "mbs")
    drone_id = getattr(ctrl, "drone_instance", 0)
    base = f"{timestamp}{('_' + reason_suffix) if reason_suffix else ''}_traj_{world}_drone{drone_id}"

    positions_py = np.array(ctrl.sent_setpoints_enu, dtype=float).tolist()
    actual_positions_py = np.array(getattr(ctrl, "actual_positions_enu", []), dtype=float).tolist()

    planned_nodes_py = []
    if hasattr(ctrl, "optimal_trajectory") and ctrl.optimal_trajectory and ctrl.optimal_trajectory.get("position"):
        planned_nodes_py = np.array(ctrl.optimal_trajectory["position"], dtype=float).tolist()

    payload = {
        "frame": "ENU",
        "positions_enu": positions_py,
        "actual_positions_enu": actual_positions_py,
        "openscvx_planned_enu": planned_nodes_py,
        "metadata": {
            "planned_waypoints": len(positions_py),
            "actual_samples": len(actual_positions_py),
            "openscvx_nodes": len(planned_nodes_py),
        },
    }
    json_path = os.path.join(ctrl.trajectory_save_dir, base + ".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    sent_arr = np.array(ctrl.sent_setpoints_enu, dtype=float) if ctrl.sent_setpoints_enu else np.empty((0, 3))
    actual_arr = np.array(actual_positions_py, dtype=float) if actual_positions_py else np.empty((0, 3))
    planned_arr = np.array(planned_nodes_py, dtype=float) if planned_nodes_py else np.empty((0, 3))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if planned_arr.size > 0:
        ax.plot(
            planned_arr[:, 0], planned_arr[:, 1], planned_arr[:, 2], "g-", lw=2.5, alpha=0.7, label="OpenSCVX planned"
        )
    if sent_arr.size > 0:
        ax.plot(sent_arr[:, 0], sent_arr[:, 1], sent_arr[:, 2], "b-", lw=2.0, alpha=0.7, label="Sent setpoints")
    if actual_arr.size > 0:
        ax.plot(
            actual_arr[:, 0],
            actual_arr[:, 1],
            actual_arr[:, 2],
            color="orange",
            linestyle="--",
            lw=2.0,
            alpha=0.9,
            label="Actual path",
        )

    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.set_zlabel("U (m)")
    ax.set_title("Trajectory summary (ENU): planned vs. sent vs. actual")
    ax.legend(loc="upper left")

    png_path = os.path.join(ctrl.trajectory_save_dir, base + ".png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ctrl.sent_traj_saved = True
