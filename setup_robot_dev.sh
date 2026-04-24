
#!/usr/bin/env bash
set -e

echo "====================================================="
echo "   Robot Dev Environment Setup for Raspberry Pi 5"
echo "====================================================="

# --- CONFIG VARIABLES ---
WORKSPACE="/opt/robot_ws"
ROS_DISTRO="jazzy"
ROS_DOMAIN_ID_VALUE=11
RMW_IMPLEMENTATION_VALUE="rmw_cyclonedds_cpp"

echo "[1/11] Updating APT..."
apt update && apt upgrade -y

echo "[2/11] Installing system and ROS dependencies..."
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
    libusb-1.0-0-dev \
    libudev-dev udev \
    serial-tools \
    v4l-utils \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp

echo "[3/11] Sourcing ROS2..."
source /opt/ros/$ROS_DISTRO/setup.bash

echo "[4/11] Running rosdep..."
rosdep update
rosdep install --from-paths $WORKSPACE/src --ignore-src -y

echo "[5/11] Adding users to groups..."
for USERNAME in $(ls /home)
do
    usermod -aG robotdev $USERNAME || true
    usermod -aG dialout $USERNAME || true
done

echo "[6/11] Building ORB-SLAM3..."
cd $WORKSPACE/ORB_SLAM3

if [ ! -f "Vocabulary/ORBvoc.txt" ]; then
    echo "Downloading ORB vocabulary..."
    wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz
    tar -xzf ORBvoc.txt.tar.gz -C Vocabulary/
fi

chmod +x build.sh
./build.sh

echo "[7/11] Building ROS2 workspace..."
cd $WORKSPACE
colcon build --symlink-install

echo "[8/11] Configuring global ROS2 environment variables..."
for HOME_DIR in /home/*; do
    BRC="$HOME_DIR/.bashrc"

    if ! grep -q "ROS_DOMAIN_ID" "$BRC"; then
        {
            echo ""
            echo "# --- Robot Workspace ROS settings ---"
            echo "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE"
            echo "export RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION_VALUE"
            echo "source /opt/ros/$ROS_DISTRO/setup.bash"
            echo "source /opt/robot_ws/install/setup.bash"
        } >> "$BRC"
    fi
done

echo "[9/11] Configuring UART (disabling serial console)..."
sed -i 's/console=serial0,115200//g' /boot/firmware/cmdline.txt
systemctl stop serial-getty@ttyAMA0.service
systemctl disable serial-getty@ttyAMA0.service

echo "[10/11] Syncing shared libraries..."
ldconfig

echo "[11/11] Cleanup and finish..."
echo "====================================================="
echo " Robot Dev setup complete!"
echo " Reboot recommended: sudo reboot"
echo "====================================================="
