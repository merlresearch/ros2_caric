<!--
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: BSD-3-Clause
-->
<div align="center">
  <h1>Coordinated Aerial Inspection of Infrastructure with Heterogeneous Drones</h1>
  <div>
    <a href='https://samuelfolorunsho.com/' target='_blank'>Samuel Folorunsho<sup>1,3</sup></a>;
    <a href='https://haynec.github.io/' target='_blank'>Chris Hayner<sup>2,3</sup></a>;
    <a href='https://www.merl.com/people/dicairano' target='_blank'>Stefano Di Cairano<sup>3</sup></a>;
    <a href='https://www.merl.com/people/vinod' target='_blank'>Abraham P. Vinod<sup>3</sup></a>
  </div>
  <sup>1</sup>University of Illinois Urbana-Champaign
  <sup>2</sup>University of Washington
  <sup>3</sup>MERL
  <br>
  <br>
  <div>
    <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
  </div>
  <br>
</div>

`ros2_caric` is a ROS 2 implementation for infrastructure inspection using a team of heterogeneous drones. It provides the code used in the work: *S. Folorunsho, C. Hayner, S. Di Cairano, A. P. Vinod, "Coordinated aerial inspection of infrastructure with heterogeneous drones", IFAC World Congress, 2026*.

We present a coordinated aerial inspection framework in which heterogeneous drones collaborate to map infrastructure, detect points of interest, and capture high-quality inspection imagery. Building on the ROS 1 Cooperative Aerial Robots Inspection Challenge (CARIC), this release provides a ROS 2/PX4 implementation and a hierarchical coordination framework that combines multi-agent task assignment with continuous-time, sequential-convex trajectory optimization for perception-aware photographer motion under dynamics, obstacle, and communication constraints. Across two inspection scenarios, the proposed method achieves comparable inspection quality to an A-star baseline while substantially reducing scoring mission time and balancing photographer utilization.

## System configuration
We developed and tested `ros2_caric` on Ubuntu 22.04 with ROS 2 (Humble). We also strongly recommend using a GPU. We have tested `ros2_caric` on:
- a standard desktop with NVIDIA GeForce RTX 4090, and
- a gaming laptop with NVIDIA GeForce RTX 4050.

`ros2_caric` uses Gazebo Harmonic LTS instead of the traditional Gazebo Classic. Please see [https://gazebosim.org/docs/harmonic/gazebo_classic_migration/](https://gazebosim.org/docs/harmonic/gazebo_classic_migration/) on how to migrate to this newer version of Gazebo.

## Installation instructions

1. `git clone --recursive https://github.com/merlresearch/ros2_caric.git ~/ros2_ws`
    - This step may take a few minutes.
1. Fetch ROS 1 CARIC assets for the MBS world from [ntu-aris/caric_mission](https://github.com/ntu-aris/caric_mission).
    1. `cd ~/ros2_ws`.
    1. Run `./utils/fetch_ROS1_caric_assets.sh`.
1. Add the custom PX4 airframes
    1. Copy files `cp ~/ros2_ws/src/px4_files/ROMFS/2210* ~/ros2_ws/src/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/`.
    2. Inside `~/ros2_ws/src/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt`, update the CMakeLists.txt entries at the end of the file (look for # [22000...)
        ```
	    # [22000, 22999] Reserve for custom models
	    22100_gz_x500_gimbal_photographer
	    22101_gz_x500_lidar_gimbal_explorer
        ```
1. Make px4 (See https://docs.px4.io/main/en/ros2/user_guide#install-px4 for more details)
    1. `bash ~/ros2_ws/src/PX4-Autopilot/Tools/setup/ubuntu.sh`
        1. If it fails due to keyring, edit line number 206 with `sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg --no-check-certificate`
    1. `cd ~/ros2_ws/src/PX4-Autopilot/`
    1. `make px4_sitl`
1. Set up Micro XRCE https://docs.px4.io/main/en/ros2/user_guide#setup-micro-xrce-dds-agent-client
    1. `cd ../Micro-XRCE-DDS-Agent` or `cd ~/ros2_ws/src/Micro-XRCE-DDS-Agent`
    1. `mkdir build`
    1. `cd build`
    1. `cmake ..`
    1. To avoid build failure at this particular commit, replace `2.12.x` in line 99 of `~/ros2_ws/src/Micro-XRCE-DDS-Agent/CMakeLists.txt` with `v2.12.1`. (prepend with `v` and replace `x` with `1`)
        - See https://github.com/PX4/PX4-Autopilot/issues/24477#issuecomment-2710838732 for more details.
    1. `make`
        - This step may take a few minutes.
    1. `sudo make install`
    1. `sudo ldconfig /usr/local/lib/`
1. It is strongly recommended to have a more recent version of `setuptools` to avoid `--editable` errors. We tested it with: `pip install setuptools==58.2.0`
1. For the baseline 1 implementation, we need to install the following modules:
    1. `pip install pathfinding scikit-learn kneed networkx numpy==1.26.4`
    	1. `pathfinding`: For A-star. Tested on 1.0.17.
    	1. `scikit-learn`: For clustering. Tested on 1.7.1.
    	1. `kneed`: For k-means clustering. Tested on 0.8.5.
    	1. `networkx`: TSP and path planning graph generation. Tested on 3.4.2.
        1. `numpy`: To play nicely with the ROS 2 version, numpy version must be less than 2. Tested on 1.26.4.
1. For the baseline 2 implementation (Proposed approach in IFAC WC paper using OpenSCvx), we use a separate Python 3.12
worker process, allowing us to use more recent JAX releases and the latest version of OpenSCvx.
    1. Set the setup environment variables. Add these to your shell startup file if you use baseline 2 often.
        ```
        export ROS2_CARIC_WS=~/ros2_ws
        export UV_CACHE_DIR=$ROS2_CARIC_WS/.uv-cache
        ```
    1. Install `uv` locally in the workspace.
        ```
        curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
        UV_UNMANAGED_INSTALL=$ROS2_CARIC_WS/.tools/uv sh /tmp/uv-install.sh
        ```
    1. Create the Python 3.12 OpenSCvx worker environment.
        ```
        cd $ROS2_CARIC_WS
        .tools/uv/uv python install 3.12
        .tools/uv/uv venv --python 3.12 .venv-py312-openscvx
        .tools/uv/uv pip install --python .venv-py312-openscvx -e OpenSCvx -e src/caric2_baseline_2 scikit-learn
        ```
        Baseline 2 uses `$ROS2_CARIC_WS/.venv-py312-openscvx/bin/python` for the OpenSCvx worker. Tested on `cvxpy`
        1.9.2, `jaxlib` 0.11.0, and `openscvx` 0.5.3.dev46.
1. Colcon build
    1. `cd ~/ros2_ws`
    1. `colcon build --packages-skip caric2_baseline_1 caric2_baseline_2 mission_manager vehicle_controller_interfaces`
        - This step may take a few minutes.
    1. `colcon build --symlink-install --packages-select caric2_baseline_1 caric2_baseline_2 mission_manager vehicle_controller_interfaces`

## Running the simulation

*Note*: At each terminal, run `source ~/ros2_ws/install/setup.bash` first.

1. Terminal 1: `ros2 launch mission_manager mission_benchmark.launch.py`.
    1. Optional parameters:
        - world: mbs/powerline
        - difficulty: easy/medium
        - run_style: full/timed
        - time_limit: in seconds (only valid if run_style is timed)
        - world_model_source: known/lidar_only
        - save_map_snapshot: False/True
    1. Example full command: `ros2 launch mission_manager mission_benchmark.launch.py world:=powerline difficulty:=easy run_style:=full  world_model_source:=known save_map_snapshot:=False`
2. Wait until gz shows all 5 drones in the "Entity Tree" (on the bottom right).
    1. (Optional) Adding camera views:
        1. You may add camera views by clicking on the three vertical dots on the top right, and then choosing/searching "Image Display".
        1. Select the drone information to get the camera.
3. Terminal 2: Run either one of the baselines provided with this package (assuming).
    1. *Baseline 1: A-star*: `ros2 launch caric2_baseline_1 caric2_baseline_1_launch.launch.py world:=powerline world_model_source:=known`
    1. *Baseline 2: Proposed approach in IFAC WC paper using OpenSCvx*: `ros2 launch caric2_baseline_2 caric2_baseline_2_launch.launch.py world:=powerline world_model_source:=known`
    1. *Note*: The parameters used must match those provided to mission_manager in Terminal 1.
4. You can view the score across the point of interest by viewing the following topics:
`ros2 topic echo /total_score_status`. If the `run_style` is not timed, you will have to terminate the scripts.
    1. *Recommended step:* When terminating a simulation, please run `~/ros2_ws/utils/cleanup.sh` to kill all processes.

## Notes

`ros2_caric` provides a cohesive package for coordinating heterogeneous drones in infrastructure-inspection scenarios
with dynamics, perception, communication, and safety constraints. To make the codebase easier to release, run, and
maintain within the available time, we made several simplifying assumptions compared to the discussion in the
accompanying paper. These limitations are documented here for transparency, and may be addressed in future releases.

- The original implementation was developed using OpenSCvx v0.2, whereas the
current version of `ros2_caric` uses OpenSCvx v0.5. During this migration and open-source release process, some functionality from the original development version was simplified or disabled.

  - The released version has its OpenSCvx planner run in a separate Python environment using Python 3.12. This allows the ROS 2 stack to
  remain compatible with its native Python version, Python 3.10, while enabling the optimizer to use a newer
  Python/OpenSCvx setup.

  - The Phase 2 implementation in `caric2_baseline_2` has been simplified as a sequence of Phase 1 solves, with
  perception constraints imposed at the terminal node of each segment.

- The coordinator does not currently reassign drones to revisit a point of interest when the score remains low.
Supporting this behavior would require additional bookkeeping and is left for future improvement.

- As mentioned in the paper, obstacle avoidance currently accounts only for agent-environment collisions. Agent-agent collision avoidance is not
modeled. This did not generally cause issues in our simulations because of the environment scale and the typical spacing
between vehicles.


## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for our policy on contributions.

## License

This code is released under `BSD-3-Clause` license, as found in the [LICENSE.md](./LICENSE.md) file.  The PX4 Gazebo
model assets include work credited upstream to PX4 Autopilot for Drones, Rudis Laboratories, Benjamin Perseghetti,
Stefano Colli, Auterion, and NXP-branded HoverGames model textures/meshes. The corresponding copyright notices are
recorded in [`.reuse/dep5`](./.reuse/dep5), and the BSD-3-Clause license text is included in
[LICENSES/BSD-3-Clause.md](./LICENSES/BSD-3-Clause.md).


## Acknowledgements

`ros2_caric` provides a native ROS 2 implementation inspired by the Cooperative Aerial Robots Inspection Challenge (CARIC) benchmark (https://ntu-aris.github.io/caric/).
The original CARIC benchmark, which has served as an inspiration for this implementation, was implemented in ROS 1 using C++/Python.
`ros2_caric` was written from scratch in Python.
We provide `fetch_ROS1_caric_assets.sh` to fetch the relevant GPLv2 assets for simulation purposes from [https://github.com/ntu-aris/caric_mission](https://github.com/ntu-aris/caric_mission).

Beyond native ROS 2 support, `ros2_caric` provides several new features/extensions:
- Adds a new world model (Powerline),
- Integrates with the PX4-Autopilot stack,
- Extends the scoring calculation to include area-based metrics,
- Removes the dependency on external point-cloud libraries, and
- Provides rendering for photo-capture functionality.

## Citing

If you use this software in your research, please cite it using the metadata from [CITATION.cff](https://github.com/merlresearch/ros2_caric/tree/main/CITATION.cff) or the following bibtex entry.

```
@inproceedings{folorunsho2026coordinated,
    title={{Coordinated aerial inspection of infrastructure with heterogeneous drones}},
  author={Samuel Folorunsho, Christopher Hayner, Stefano Di Cairano, Abraham P. Vinod},
  booktitle={IFAC World Congress},
  year={2026},
  note = {(accepted)}
}
```

## Contact

This repository was primarily developed by Samuel Folorunsho and Abraham P. Vinod.

For questions or bugs, contact Abraham P. Vinod (Email: [vinod@merl.com](mailto:vinod@merl.com), [abraham.p.vinod@ieee.org](mailto:abraham.p.vinod@ieee.org)).
