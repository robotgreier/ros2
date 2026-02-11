# Occupancy Grid System Documentation

## Overview

The occupancy grid system converts ORB features and ArUco markers into a 2D grid with 3-bit encoding for navigation.

## 3-Bit Encoding Scheme

| Binary | Decimal | Meaning | Color | Description |
|--------|---------|---------|-------|-------------|
| 000 | 0 | Unknown | Dark Grey | No features detected / unexplored |
| 001 | 1 | Free Space | Light Grey/White | Navigable space without targets |
| 011 | 3 | Pickup Target | Green | Object to pick up (ArUco) |
| 111 | 7 | Drop-off Zone | Blue | Destination zone (ArUco) |

## Architecture

```
Vision Node → Grid Features + ArUco Detections
                        ↓
            Occupancy Grid Node
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   Custom Grid    ROS OccupancyGrid  Visualization
  (3-bit UInt8)     (for rviz)        (Image)
```

## Components

### 1. occupancy_grid_generator.py
Core logic for generating occupancy grids.

**Key Features:**
- Feature-based free space detection
- ArUco marker integration
- Configurable thresholds
- Visualization generation
- Encoding for FPGA transmission

**Main Class:** `OccupancyGridGenerator`

### 2. occupancy_grid_node.py
ROS2 node that subscribes to vision data and publishes grids.

**Subscribed Topics:**
- `/vision/grid_features` (Int32MultiArray)
- `/vision/aruco_detections` (Float32MultiArray)

**Published Topics:**
- `/occupancy_grid/custom` (UInt8MultiArray) - 3-bit encoded grid
- `/occupancy_grid/ros` (OccupancyGrid) - Standard ROS format for rviz
- `/occupancy_grid/visualization` (Image) - Visual representation

## Installation

### Add to ROS2 Package

```bash
cd ~/robot_ws/src/taskbot_vision/taskbot_vision

# Copy files
cp occupancy_grid_generator.py .
cp occupancy_grid_node.py .
```

### Update setup.py

Add to entry_points:

```python
entry_points={
    'console_scripts': [
        'vision_node = taskbot_vision.vision_node:main',
        'fpga_interface_node = taskbot_vision.fpga_interface_node:main',
        'mock_camera_node = taskbot_vision.mock_camera_node:main',
        'occupancy_grid_node = taskbot_vision.occupancy_grid_node:main',  # ADD THIS
    ],
},
```

### Rebuild

```bash
cd ~/robot_ws
colcon build --packages-select taskbot_vision
source install/setup.bash
```

## Usage

### Basic Usage

```bash
# Terminal 1: Vision node (or mock camera + vision)
ros2 run taskbot_vision vision_node --ros-args \
    -p use_camera_topic:=true

# Terminal 2: Occupancy grid node
ros2 run taskbot_vision occupancy_grid_node

# Terminal 3: View visualization
ros2 run rqt_image_view rqt_image_view /occupancy_grid/visualization
```

### With Mock Camera

```bash
# Terminal 1: Mock camera
ros2 run taskbot_vision mock_camera_node

# Terminal 2: Vision node
ros2 run taskbot_vision vision_node --ros-args -p use_camera_topic:=true

# Terminal 3: Occupancy grid
ros2 run taskbot_vision occupancy_grid_node

# Terminal 4: View
ros2 topic echo /occupancy_grid/custom --once
```

### Parameters

```bash
ros2 run taskbot_vision occupancy_grid_node --ros-args \
    -p grid_rows:=8 \
    -p grid_cols:=8 \
    -p frame_width:=640 \
    -p frame_height:=360 \
    -p publish_visualization:=true \
    -p publish_ros_occupancy:=true
```

## Message Formats

### Custom Occupancy Grid (UInt8MultiArray)

```
data: [rows, cols, cell_0_0, cell_0_1, ..., cell_7_7]
```

Example for 8x8 grid:
```
data: [8, 8, 0, 1, 1, 3, 1, 1, 1, 0, ...]
                ^  ^  ^  ^  (occupancy values)
```

### Decoding Example

```python
import rclpy
from std_msgs.msg import UInt8MultiArray
import numpy as np

def occupancy_callback(msg):
    rows = msg.data[0]
    cols = msg.data[1]
    
    # Extract grid
    occupancy_flat = np.array(msg.data[2:], dtype=np.uint8)
    occupancy_grid = occupancy_flat.reshape((rows, cols))
    
    # Decode
    unknown_cells = (occupancy_grid == 0).sum()
    free_cells = (occupancy_grid == 1).sum()
    pickup_cells = (occupancy_grid == 3).sum()
    dropoff_cells = (occupancy_grid == 7).sum()
    
    print(f"Unknown: {unknown_cells}, Free: {free_cells}")
    print(f"Pickup: {pickup_cells}, Dropoff: {dropoff_cells}")
```

## Integration with FPGA

### Send to FPGA

The occupancy grid can be efficiently sent to your FPGA for navigation decisions:

```python
from occupancy_grid_generator import OccupancyGridGenerator

generator = OccupancyGridGenerator()

# Get occupancy grid (from ROS callback)
occupancy = ...  # 8x8 numpy array

# Encode for transmission
encoded_bytes = generator.encode_for_transmission(occupancy)

# Send to FPGA via serial
fpga_serial.write(encoded_bytes)  # 64 bytes for 8x8 grid
```

### FPGA Decoding

On the FPGA side, each cell uses 3 bits:

```c
// Pseudo-code for FPGA
for (int i = 0; i < 64; i++) {
    uint8_t cell_value = occupancy_data[i];
    
    // Decode 3-bit value
    bool is_unknown = (cell_value == 0);   // 000
    bool is_free = (cell_value == 1);      // 001
    bool is_pickup = (cell_value == 3);    // 011
    bool is_dropoff = (cell_value == 7);   // 111
    
    // Use for navigation decision
    if (is_pickup) {
        // Navigate toward pickup target
    }
}
```

## Tuning Parameters

### Feature Thresholds

In `occupancy_grid_generator.py`:

```python
self.min_features_free = 3   # Min features to consider "detected"
self.max_features_free = 25  # Max features for "free space"
```

**Adjust based on your environment:**
- **Sparse features**: Lower thresholds (min=2, max=15)
- **Dense features**: Higher thresholds (min=5, max=30)
- **Test**: Run and visualize to tune

### Grid Resolution

Trade-off between detail and computation:

```bash
# Fine detail (slower)
-p grid_rows:=16 -p grid_cols:=16

# Standard (recommended)
-p grid_rows:=8 -p grid_cols:=8

# Coarse (faster)
-p grid_rows:=4 -p grid_cols:=4
```

## Visualization

The visualization shows:
- Color-coded cells based on occupancy
- 3-bit binary values in each cell
- Legend explaining encoding
- Grid lines for cell boundaries

**Colors:**
- **Dark Grey** (000): Unknown
- **Light Grey** (001): Free space
- **Green** (011): Pickup target
- **Blue** (111): Drop-off zone

## Using with rviz2

The node publishes standard ROS `OccupancyGrid` messages:

```bash
# Launch rviz2
ros2 run rviz2 rviz2

# Add display
# - Add → By topic → /occupancy_grid/ros → OccupancyGrid
# - Set Fixed Frame: camera_frame
```

## Navigation Integration

### Find Targets

```python
from occupancy_grid_generator import OccupancyGridGenerator

generator = OccupancyGridGenerator()

# Get pickup targets
pickup_cells = generator.get_target_cells(occupancy_grid, 'pickup')
print(f"Pickup targets at: {pickup_cells}")

# Get drop-off zones
dropoff_cells = generator.get_target_cells(occupancy_grid, 'dropoff')
print(f"Drop-off zones at: {dropoff_cells}")

# Find nearest target from current position
current_pos = (4, 4)  # Grid cell (row, col)
nearest = generator.find_nearest_target(occupancy_grid, current_pos, 'pickup')
print(f"Navigate to: {nearest}")
```

### Path Planning

The occupancy grid provides:
1. **Free space** for path planning
2. **Target locations** for goal setting
3. **Unknown areas** to avoid or explore

Use with path planning algorithms:
- A* search
- Dijkstra
- Potential fields
- RRT

## Example: Complete System Test

```bash
# 1. Start mock camera
ros2 run taskbot_vision mock_camera_node &

# 2. Start vision
ros2 run taskbot_vision vision_node --ros-args -p use_camera_topic:=true &

# 3. Start occupancy grid
ros2 run taskbot_vision occupancy_grid_node &

# 4. Monitor
ros2 topic hz /occupancy_grid/custom
ros2 topic echo /occupancy_grid/custom --once

# 5. Visualize
ros2 run rqt_image_view rqt_image_view /occupancy_grid/visualization

# When done
killall ros2
```

## Troubleshooting

### No occupancy grid published

**Check:**
```bash
# Are features being published?
ros2 topic hz /vision/grid_features

# Are ArUco markers detected?
ros2 topic echo /vision/aruco_detections --once

# Is occupancy node running?
ros2 node list | grep occupancy
```

### All cells show as unknown (000)

**Cause:** Not enough ORB features detected

**Fix:** 
- Adjust thresholds in `occupancy_grid_generator.py`
- Improve lighting
- Add texture to environment
- Lower `min_features_free` threshold

### Markers not showing in grid

**Check:**
- ArUco markers are detected: `ros2 topic echo /vision/aruco_detections`
- Marker IDs match `object_database` in `occupancy_grid_node.py`
- Markers are within camera frame

## Performance

**Expected performance on Raspberry Pi 5:**
- Grid generation: < 5ms
- Visualization: < 10ms
- Total latency: < 20ms
- Rate: 30 Hz (matches camera fps)

**Optimization tips:**
- Disable visualization if not needed
- Use smaller grid (4x4) for faster processing
- Disable ROS occupancy grid if only using custom format

## Summary

✅ **3-bit encoding** for efficient representation  
✅ **Feature-based** free space detection  
✅ **ArUco integration** for targets and zones  
✅ **ROS2 native** with standard messages  
✅ **FPGA ready** with simple byte encoding  
✅ **Visualization** for debugging  

The occupancy grid system provides a compact, efficient representation of the environment for your neuromorphic navigation system! 🚀
