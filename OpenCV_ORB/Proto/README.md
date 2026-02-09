# Energy-Aware Autonomous Taskbot - Vision System

OpenCV-based vision system for ROS2 Jazzy with neuromorphic FPGA navigation accelerator.

## System Overview

### Architecture
```
C922 Camera → Vision Node → Grid Features → FPGA SNN → Motor Commands
                         ↘ ArUco Detection → Object Tracking
```

### Components

1. **vision_node.py** - Main ROS2 node
   - Camera capture (640x360 @ 30fps)
   - ORB feature detection
   - Grid-based feature encoding
   - ArUco marker detection

2. **feature_encoding.py** - Feature encoding utilities
   - Rate-based encoding for SNNs
   - Population encoding
   - Directional encoding
   - Navigation-specific encoding

3. **aruco_detector.py** - Object detection and classification
   - ArUco marker detection with pose estimation
   - Object type mapping
   - Distance estimation
   - Navigation target selection

4. **fpga_interface_node.py** - FPGA communication
   - Grid feature encoding for FPGA
   - Serial communication with FPGA
   - Motor command generation
   - Fallback navigation

## Installation

### Prerequisites

```bash
# System dependencies (Ubuntu 24.04)
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-opencv \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    v4l-utils

# Python dependencies
pip3 install --break-system-packages \
    numpy \
    opencv-contrib-python \
    pyserial
```

### ROS2 Package Setup

```bash
# Create workspace
mkdir -p ~/taskbot_ws/src
cd ~/taskbot_ws/src

# Create package
ros2 pkg create --build-type ament_python taskbot_vision \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge

# Copy files to package
cd taskbot_vision/taskbot_vision
cp /path/to/vision_node.py .
cp /path/to/feature_encoding.py .
cp /path/to/aruco_detector.py .
cp /path/to/fpga_interface_node.py .

# Copy launch file
cd ../launch
cp /path/to/vision_launch.py .

# Build
cd ~/taskbot_ws
colcon build --packages-select taskbot_vision
source install/setup.bash
```

### Camera Configuration

```bash
# Verify camera is detected
v4l2-ctl --list-devices

# Test camera
v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=360,pixelformat=MJPG
ffplay /dev/video0

# Grant permissions (if needed)
sudo usermod -a -G video $USER
```

## Configuration

### Grid Configuration

Edit parameters in launch file or command line:

```python
# Grid size (8x8 recommended for real-time performance)
grid_rows: 8
grid_cols: 8

# ORB features
orb_features: 500

# Camera settings
frame_width: 640
frame_height: 360
fps: 30
```

### ArUco Marker Setup

Configure object types in `aruco_detector.py`:

```python
self.object_database = {
    0: {'type': 'box_small', 'destination': 'zone_a', 'color': (255, 0, 0)},
    1: {'type': 'box_medium', 'destination': 'zone_a', 'color': (255, 0, 0)},
    2: {'type': 'cylinder', 'destination': 'zone_b', 'color': (0, 255, 0)},
    # ... add your objects
    10: {'type': 'zone_a', 'destination': None, 'color': (255, 255, 0)},
    11: {'type': 'zone_b', 'destination': None, 'color': (255, 0, 255)},
}
```

### FPGA Communication

Configure serial port in FPGA interface:

```python
fpga_port: '/dev/ttyUSB0'
fpga_baudrate: 115200
enable_fpga: true
```

## Usage

### Basic Launch

```bash
# Launch vision system
ros2 launch taskbot_vision vision_launch.py

# With debug visualization
ros2 launch taskbot_vision vision_launch.py publish_debug_image:=true

# Custom grid size
ros2 launch taskbot_vision vision_launch.py grid_rows:=16 grid_cols:=16
```

### Launch Both Vision and FPGA Interface

```bash
# Terminal 1: Vision node
ros2 run taskbot_vision vision_node

# Terminal 2: FPGA interface
ros2 run taskbot_vision fpga_interface_node --ros-args \
    -p fpga_port:=/dev/ttyUSB0 \
    -p enable_fpga:=true

# Terminal 3: Monitor topics
ros2 topic echo /vision/grid_features
ros2 topic echo /vision/aruco_detections
ros2 topic echo /cmd_vel
```

### View Debug Visualization

```bash
# Install image viewer (on Raspberry Pi with display)
sudo apt install ros-jazzy-rqt-image-view

# View debug image
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

## ROS2 Topics

### Published Topics

- `/vision/grid_features` (Int32MultiArray)
  - Format: `[rows, cols, feature_count_cell_0, ..., feature_count_cell_N]`
  - 8x8 grid = 2 + 64 = 66 integers
  
- `/vision/aruco_detections` (Float32MultiArray)
  - Format: `[num_markers, id1, x1, y1, area1, angle1, id2, ...]`
  - Each marker: 5 floats (id, x, y, area, angle)

- `/vision/debug_image` (Image) - Optional visualization
  - BGR8 encoding
  - Shows grid, features, ArUco markers

- `/cmd_vel` (Twist) - Motor commands
  - linear.x: forward velocity (m/s)
  - angular.z: rotation velocity (rad/s)

## Testing

### Test Vision Node Only

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float32MultiArray

class VisionTester(Node):
    def __init__(self):
        super().__init__('vision_tester')
        
        self.grid_sub = self.create_subscription(
            Int32MultiArray,
            'vision/grid_features',
            self.grid_callback,
            10
        )
        
        self.aruco_sub = self.create_subscription(
            Float32MultiArray,
            'vision/aruco_detections',
            self.aruco_callback,
            10
        )
    
    def grid_callback(self, msg):
        if len(msg.data) >= 2:
            rows, cols = msg.data[0], msg.data[1]
            features = msg.data[2:]
            total_features = sum(features)
            print(f'Grid {rows}x{cols}: {total_features} total features')
    
    def aruco_callback(self, msg):
        if len(msg.data) >= 1:
            num_markers = int(msg.data[0])
            print(f'Detected {num_markers} ArUco markers')

def main():
    rclpy.init()
    node = VisionTester()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

### Test Camera Directly

```python
#!/usr/bin/env python3
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Camera test - Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Feature Encoding for SNN

### Example: Get Encoded Features

```python
from feature_encoding import GridFeatureEncoder
import numpy as np

# Initialize encoder
encoder = GridFeatureEncoder(grid_rows=8, grid_cols=8)

# Example grid from vision node
grid = np.random.randint(0, 20, size=(8, 8))

# Rate-based encoding (simplest for LIF neurons)
rate_encoded = encoder.encode_rate_based(grid, normalize=True, max_features=50)
print(f"Rate encoded shape: {rate_encoded.shape}")  # (64,)

# Directional encoding (useful for navigation)
dir_encoded = encoder.encode_directional(grid, normalize=True)
print(f"Directional features: {dir_encoded}")

# Full navigation encoding
nav_encoded = encoder.encode_for_navigation(grid, target_position=(320, 180))
print(f"Navigation encoding keys: {nav_encoded.keys()}")
```

### FPGA Protocol

Data format sent to FPGA:

```
Byte 0: 0xAA (start marker)
Byte 1: NUM_FEATURES & 0xFF (low byte)
Byte 2: (NUM_FEATURES >> 8) & 0xFF (high byte)
Byte 3-N: Feature values (uint8, 0-255)
```

Expected response from FPGA:

```
Byte 0-1: Linear velocity (int16, -32767 to 32767)
Byte 2-3: Angular velocity (int16, -32767 to 32767)
```

## Camera Calibration

For accurate distance estimation, calibrate your C922:

```python
from aruco_detector import CameraCalibrator
import cv2

# Initialize calibrator
calibrator = CameraCalibrator(
    checkerboard_size=(9, 6),  # Internal corners
    square_size_mm=25.0
)

# Capture calibration images
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Capture 15-20 images of checkerboard from different angles")
print("Press SPACE to capture, 'q' to finish")

while True:
    ret, frame = cap.read()
    cv2.imshow('Calibration', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if calibrator.add_calibration_image(frame):
            print(f"Captured! Total: {len(calibrator.objpoints)}")
        else:
            print("Checkerboard not found")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Perform calibration
if len(calibrator.objpoints) >= 10:
    camera_matrix, dist_coeffs, error = calibrator.calibrate((640, 360))
    print(f"Calibration complete! Error: {error:.4f}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")
    
    # Save for later use
    np.savez('camera_calibration.npz', 
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
```

## Performance Optimization

### Grid Size vs Performance

| Grid Size | Features/Cell | Processing Time | Navigation Quality |
|-----------|---------------|-----------------|-------------------|
| 4x4       | 0-100        | ~10ms          | Coarse            |
| 8x8       | 0-50         | ~15ms          | Good              |
| 16x16     | 0-20         | ~25ms          | Fine              |

For real-time on Raspberry Pi 5, **8x8 is recommended**.

### Optimization Tips

1. **Reduce ORB features**: 300-500 is sufficient
2. **Lower FPS if needed**: 20fps still provides good navigation
3. **Disable debug image**: Saves CPU for processing
4. **Use JPEG compression**: If transmitting over network

## Troubleshooting

### Camera not detected
```bash
# Check camera
ls -la /dev/video*
v4l2-ctl --list-devices

# Try different index
ros2 run taskbot_vision vision_node --ros-args -p camera_index:=1
```

### Low frame rate
```bash
# Check CPU usage
htop

# Reduce grid size or ORB features
ros2 run taskbot_vision vision_node --ros-args \
    -p grid_rows:=4 -p grid_cols:=4 -p orb_features:=200
```

### FPGA communication errors
```bash
# Check serial port permissions
ls -la /dev/ttyUSB*
sudo usermod -a -G dialout $USER

# Test serial connection
screen /dev/ttyUSB0 115200

# Enable simulation mode
ros2 run taskbot_vision fpga_interface_node --ros-args \
    -p enable_fpga:=false -p motor_simulation:=true
```

### ArUco markers not detected
- Ensure good lighting
- Print markers at sufficient size (>50mm)
- Check marker dictionary matches (DICT_4X4_50)
- Try increasing resolution temporarily to verify markers

## Integration with FPGA SNN

### LIF Neuron Input Mapping

For an 8x8 grid (64 cells), you'll need:
- **Input layer**: 64 LIF neurons (one per grid cell)
- **Encoding**: Feature count → spike rate
- **Additional inputs**: 4 directional bias neurons (optional)

### Example SNN Architecture

```
Input Layer (64 neurons) → Hidden Layer (32 neurons) → Output Layer (2 neurons)
                                                        ├─ Linear velocity
                                                        └─ Angular velocity
```

### Spike Rate Encoding

```python
# In FPGA firmware, convert feature count to spike probability
spike_probability = min(feature_count / 50.0, 1.0)
spike = (random() < spike_probability) ? 1 : 0
```

## Project Structure

```
taskbot_vision/
├── taskbot_vision/
│   ├── __init__.py
│   ├── vision_node.py           # Main vision processing
│   ├── feature_encoding.py      # SNN encoding utilities
│   ├── aruco_detector.py        # Object detection
│   └── fpga_interface_node.py   # FPGA communication
├── launch/
│   └── vision_launch.py         # ROS2 launch file
├── config/
│   └── params.yaml              # Configuration parameters
├── test/
│   └── test_vision.py           # Unit tests
├── package.xml
├── setup.py
└── README.md
```

## License

This code is provided for educational purposes for your bachelor project.

## Authors

Bachelor Project Team - Energy-Aware Autonomous Taskbot

## References

- ROS2 Jazzy Documentation: https://docs.ros.org/en/jazzy/
- OpenCV Python: https://docs.opencv.org/4.x/
- ArUco Markers: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- ORB Features: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
