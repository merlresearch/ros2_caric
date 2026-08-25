#!/bin/bash
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

# SPDX-License-Identifier: BSD-3-Clause

# Set target directories
YAML_TARGET="src/mission_manager/models/mbs/bounding_boxes/box_description.yaml"
STL_TARGET="src/mission_manager/models/mbs/mbs.stl"

# Create directories if they don't exist
mkdir -p "$(dirname "$YAML_TARGET")"
mkdir -p "$(dirname "$STL_TARGET")"

# Download files
curl -L -o "$YAML_TARGET" "https://raw.githubusercontent.com/ntu-aris/caric_mission/master/models/mbs/bounding_boxes/box_description.yaml"
curl -L -o "$STL_TARGET" "https://raw.githubusercontent.com/ntu-aris/caric_mission/master/models/mbs/mbs.stl"

echo "Files downloaded successfully."
