#!/usr/bin/env bash
# Phase 1b setup — Jetson Orin Nano (Ubuntu 24.04 / JetPack 7.2)
# Run once. Installs ROS 2 Jazzy + pipeline packages, creates the project
# .venv, installs Python deps, and wires up .bashrc.
#
# The colcon workspace IS the project directory:
#   ~/Documents/Projects/Pathfinder/
#
# PASS/FAIL gate: run the verify lines printed at the end.
set -e

PROJECT_DIR="$HOME/Documents/Projects/Pathfinder"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== Phase 1b setup: ROS 2 Jazzy + Pathfinder .venv ==="
echo "    Project dir: $PROJECT_DIR"

# ── 1. ROS 2 Jazzy apt repo ──────────────────────────────────────────────────
echo "[1/6] Adding ROS 2 apt repo..."
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# ── 2. ROS 2 + pipeline packages ─────────────────────────────────────────────
echo "[2/6] Installing ROS 2 Jazzy + pipeline packages..."
sudo apt install -y \
    ros-jazzy-desktop \
    ros-jazzy-rtabmap-ros \
    ros-jazzy-v4l2-camera \
    ros-jazzy-cv-bridge \
    ros-jazzy-camera-calibration \
    python3-colcon-common-extensions \
    python3-rosdep

# ── 3. rosdep ────────────────────────────────────────────────────────────────
echo "[3/6] Initialising rosdep..."
sudo rosdep init 2>/dev/null || echo "  (rosdep already initialised, skipping)"
rosdep update

# ── 4. Project .venv ─────────────────────────────────────────────────────────
# --system-site-packages makes rclpy, sensor_msgs, cv_bridge, tensorrt (all in
# system Python) visible inside the venv without reinstalling them.
echo "[4/6] Creating .venv at $VENV_DIR ..."
python3 -m venv --system-site-packages "$VENV_DIR"

# ── 5. Python deps ───────────────────────────────────────────────────────────
# Install requirements.txt into the project .venv.
# Key package: cuda-python (provides cuda.bindings.runtime for TensorRTBackend).
echo "[5/6] Installing Python deps from requirements.txt ..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# ── 6. .bashrc additions ─────────────────────────────────────────────────────
echo "[6/6] Updating ~/.bashrc..."

add_if_missing() {
    grep -qxF "$1" ~/.bashrc || echo "$1" >> ~/.bashrc
}

add_if_missing "source /opt/ros/jazzy/setup.bash"
add_if_missing "source $VENV_DIR/bin/activate"
# ament_python hardwires /usr/bin/python3 into entry point scripts regardless
# of venv state. Adding the .venv site-packages to PYTHONPATH is the standard
# fix so the ROS node finds the packages installed in the project .venv.
add_if_missing "export PYTHONPATH=$VENV_DIR/lib/python3.12/site-packages:\$PYTHONPATH"
# Workspace install overlay — valid after first colcon build.
add_if_missing "source $PROJECT_DIR/install/setup.bash 2>/dev/null || true"

echo ""
echo "=== Done. Open a NEW terminal, then verify: ==="
echo ""
echo "  1. ROS 2:"
echo "     ros2 --help > /dev/null && echo 'ROS 2 OK'"
echo ""
echo "  2. cuda.bindings (TensorRTBackend needs this):"
echo "     python3 -c \"from cuda.bindings import runtime; print('cuda.bindings OK')\""
echo ""
echo "  3. rtabmap_ros installed:"
echo "     ros2 pkg list | grep rtabmap"
echo ""
echo "NEXT: build the metric TRT engine:"
echo "  cd ~/Depth-Anything-V2"
echo "  .venv/bin/python $PROJECT_DIR/scripts/phase1b_export_metric.py"
echo "  # then run the trtexec command it prints"
