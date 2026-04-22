#!/usr/bin/env bash
set -euo pipefail

echo "====================================================="
echo "   Robot Dev Environment Setup (WSL2 SAFE VERSION)"
echo "====================================================="

# Determine the real user and home even if run with sudo
if [[ -n "${SUDO_USER-}" && "$SUDO_USER" != "root" ]]; then
  REAL_USER="$SUDO_USER"
  REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
else
  REAL_USER="$(id -un)"
  REAL_HOME="$HOME"
fi

WORKSPACE="$REAL_HOME/robot_ws"
ROS_DISTRO="jazzy"
ROS_DOMAIN_ID_VALUE=11
RMW_IMPLEMENTATION_VALUE="rmw_cyclonedds_cpp"

echo "[info] Using user: $REAL_USER"
echo "[info] Using home: $REAL_HOME"
echo "[info] Workspace:  $WORKSPACE"

echo "[1/8] Updating APT..."
apt update && apt upgrade -y

echo "[2/8] Installing dependencies safe for WSL2..."
apt install -y \
  build-essential cmake git wget curl unzip \
  python3 python3-pip python3-colcon-common-extensions \
  python3-rosdep python3-argcomplete \
  libopencv-dev \
  libeigen3-dev \
  libboost-all-dev \
  libyaml-cpp-dev \
  libgoogle-glog-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
  v4l-utils

echo "[3/8] Ensure workspace folders exist..."
mkdir -p "$WORKSPACE/src"
chown -R "$REAL_USER":"$REAL_USER" "$WORKSPACE"

echo "[4/8] Initialize rosdep (non-fatal if sources empty)..."
# rosdep caches under /root if run with sudo; prefer running as REAL_USER
sudo -u "$REAL_USER" bash -lc "rosdep update || true"
sudo -u "$REAL_USER" bash -lc "rosdep install --from-paths '$WORKSPACE/src' --ignore-src -y || true"

echo "[5/8] Source ROS and build the workspace..."
if [ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
  echo "[error] /opt/ros/$ROS_DISTRO/setup.bash not found. Please install ROS 2 $ROS_DISTRO in WSL2 first."
  echo "        See: https://docs.ros.org/en/$ROS_DISTRO/Installation.html"
  exit 1
fi

sudo -u "$REAL_USER" bash -lc "source /opt/ros/$ROS_DISTRO/setup.bash && cd '$WORKSPACE' && colcon build --symlink-install"

echo "[6/8] Persist environment for the real user (~/.bashrc)..."
BRC="$REAL_HOME/.bashrc"
if ! grep -q "Robot Workspace ROS settings (WSL2)" "$BRC" 2>/dev/null; then
  {
    echo ""
    echo "# --- Robot Workspace ROS settings (WSL2) ---"
    echo "export ROS_PYTHON_VERSION=3"
    echo "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE"
    echo "export RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION_VALUE"
    echo "source /opt/ros/$ROS_DISTRO/setup.bash"
    echo "source \"\$HOME/robot_ws/install/setup.bash\""
  } >> "$BRC"
  chown "$REAL_USER":"$REAL_USER" "$BRC"
fi

echo "[7/8] ldconfig (best-effort)..."
ldconfig || true

echo "[8/8] Done!"
echo "====================================================="
echo " WSL2 Robot Dev environment ready for $REAL_USER."
echo " Open a new WSL terminal or run:  source \"$REAL_HOME/.bashrc\""
echo " Workspace: $WORKSPACE"
echo "====================================================="
