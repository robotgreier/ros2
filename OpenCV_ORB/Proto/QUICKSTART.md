# Quick Start Guide - Vision System Testing

## Immediate Testing (No ROS2 Required)

### Step 1: Install Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv v4l-utils
pip3 install --break-system-packages numpy opencv-contrib-python
```

### Step 2: Test Camera

```bash
# Check if camera is detected
v4l2-ctl --list-devices

# You should see something like:
# C922 Pro Stream Webcam (usb-xxxx):
#     /dev/video0
#     /dev/video1

# Test camera with simple script
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

### Step 3: Run Standalone Vision Test

```bash
# Make script executable
chmod +x test_vision_standalone.py

# Run the test
python3 test_vision_standalone.py
```

**Expected behavior:**
- Window opens showing camera feed
- Green circles show ORB feature points
- 8x8 grid overlay with feature counts
- FPS counter in top-left
- If you show an ArUco marker, it will be detected and labeled

**Controls:**
- `q` - Quit
- `g` - Toggle grid display
- `f` - Toggle feature display
- `a` - Toggle ArUco display
- `s` - Save screenshot

### Step 4: Generate ArUco Markers for Testing

```bash
# Generate all project markers
python3 generate_aruco_markers.py --mode project

# This creates folder: aruco_markers/
# Print the files and attach to objects

# Generate a single test marker
python3 generate_aruco_markers.py --mode single --id 0
```

## Testing Grid Feature Encoding

```bash
# Test the encoding utilities
python3 feature_encoding.py
```

This will show examples of different encoding schemes for your FPGA SNN.

## Testing ArUco Detection

```bash
# Run ArUco detector test
python3 aruco_detector.py
```

Show printed ArUco markers to the camera - you'll see:
- Marker ID
- Object type
- Distance estimate
- 3D pose axes

## ROS2 Integration (After Basic Tests Pass)

### Step 1: Create ROS2 Package

```bash
cd ~/taskbot_ws/src
ros2 pkg create --build-type ament_python taskbot_vision \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge

cd taskbot_vision
```

### Step 2: Copy Files to Package

```bash
# Copy Python modules
cp vision_node.py taskbot_vision/
cp feature_encoding.py taskbot_vision/
cp aruco_detector.py taskbot_vision/
cp fpga_interface_node.py taskbot_vision/

# Copy launch file
mkdir -p launch
cp vision_launch.py launch/
```

### Step 3: Update setup.py

Edit `setup.py` and add to `entry_points`:

```python
entry_points={
    'console_scripts': [
        'vision_node = taskbot_vision.vision_node:main',
        'fpga_interface_node = taskbot_vision.fpga_interface_node:main',
    ],
},
```

Also add to `data_files`:

```python
(os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
```

### Step 4: Build Package

```bash
cd ~/taskbot_ws
colcon build --packages-select taskbot_vision
source install/setup.bash
```

### Step 5: Run ROS2 Nodes

```bash
# Terminal 1: Vision node
ros2 run taskbot_vision vision_node

# Terminal 2: Check topics
ros2 topic list
ros2 topic echo /vision/grid_features

# Terminal 3: FPGA interface (without real FPGA)
ros2 run taskbot_vision fpga_interface_node --ros-args \
    -p enable_fpga:=false \
    -p motor_simulation:=true
```

## Troubleshooting

### Camera not working
```bash
# Check permissions
ls -la /dev/video0
sudo usermod -a -G video $USER
# Log out and back in

# Test different camera index
python3 test_vision_standalone.py  # Edit camera_index if needed
```

### Low FPS / Performance Issues
```bash
# Reduce grid size
# Edit test_vision_standalone.py:
# self.grid_rows = 4
# self.grid_cols = 4

# Reduce ORB features
# self.orb_max_features = 200
```

### Import Errors
```bash
# Verify OpenCV installation
python3 -c "import cv2; print(cv2.__version__)"

# Should print version 4.x.x

# If missing cv2.aruco:
pip3 install --break-system-packages opencv-contrib-python
```

## Expected Performance

On Raspberry Pi 5:
- **FPS**: 25-30 fps at 640x480
- **Features**: 200-500 ORB features per frame
- **Grid**: 8x8 grid (64 cells)
- **Latency**: ~30-40ms per frame

## Next Steps

1. ✅ Verify standalone vision test works
2. ✅ Print and test ArUco markers
3. ✅ Integrate with ROS2
4. ⬜ Implement FPGA serial communication
5. ⬜ Train/configure SNN on FPGA
6. ⬜ Test complete navigation pipeline
7. ⬜ Add ultrasonic distance sensor integration
8. ⬜ Implement state machine for pick-and-place

## File Overview

```
vision_node.py              - Main ROS2 vision processing node
feature_encoding.py         - SNN encoding utilities
aruco_detector.py          - Object detection and classification
fpga_interface_node.py     - FPGA communication interface
test_vision_standalone.py  - Standalone test (no ROS2)
generate_aruco_markers.py  - Marker generation utility
vision_launch.py           - ROS2 launch configuration
README.md                  - Complete documentation
QUICKSTART.md             - This file
```

## Support

If you encounter issues:

1. Check camera is working: `v4l2-ctl --list-devices`
2. Verify OpenCV: `python3 -c "import cv2; print(cv2.__version__)"`
3. Check permissions: `groups` (should include 'video')
4. Test standalone first before ROS2 integration
5. Review logs for specific error messages

Good luck with your bachelor project! 🤖
