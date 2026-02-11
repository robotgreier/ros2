#!/usr/bin/env python3
"""
Occupancy Grid Node
Subscribes to vision data and publishes occupancy grid with 3-bit encoding
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float32MultiArray, UInt8MultiArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import sys
import os

# Import the occupancy grid generator
sys.path.append(os.path.dirname(__file__))
from occupancy_grid_generator import OccupancyGridGenerator


class OccupancyGridNode(Node):
    """
    ROS2 node that generates and publishes occupancy grids from vision data
    """
    
    def __init__(self):
        super().__init__('occupancy_grid_node')
        
        # Parameters
        self.declare_parameter('grid_rows', 8)
        self.declare_parameter('grid_cols', 8)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 360)
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('publish_ros_occupancy', True)  # Standard ROS OccupancyGrid
        
        self.grid_rows = self.get_parameter('grid_rows').value
        self.grid_cols = self.get_parameter('grid_cols').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.publish_viz = self.get_parameter('publish_visualization').value
        self.publish_ros_occ = self.get_parameter('publish_ros_occupancy').value
        
        # Initialize occupancy grid generator
        self.generator = OccupancyGridGenerator(
            self.grid_rows,
            self.grid_cols,
            self.frame_width,
            self.frame_height
        )
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.grid_features_sub = self.create_subscription(
            Int32MultiArray,
            'vision/grid_features',
            self.grid_features_callback,
            10
        )
        
        self.aruco_sub = self.create_subscription(
            Float32MultiArray,
            'vision/aruco_detections',
            self.aruco_callback,
            10
        )
        
        # Publishers
        # Custom 3-bit occupancy grid
        self.occupancy_pub = self.create_publisher(
            UInt8MultiArray,
            'occupancy_grid/custom',
            10
        )
        
        # Standard ROS OccupancyGrid (optional, for rviz)
        if self.publish_ros_occ:
            self.ros_occupancy_pub = self.create_publisher(
                OccupancyGrid,
                'occupancy_grid/ros',
                10
            )
        
        # Visualization
        if self.publish_viz:
            self.viz_pub = self.create_publisher(
                Image,
                'occupancy_grid/visualization',
                10
            )
        
        # State
        self.current_feature_grid = None
        self.current_aruco_detections = None
        
        self.get_logger().info(f'Occupancy Grid Node started: {self.grid_rows}x{self.grid_cols}')
        self.get_logger().info('3-bit encoding: 000=Unknown, 001=Free, 011=Pickup, 111=Dropoff')
    
    def grid_features_callback(self, msg):
        """Process grid features from vision node"""
        if len(msg.data) < 2:
            return
        
        rows = msg.data[0]
        cols = msg.data[1]
        
        if rows != self.grid_rows or cols != self.grid_cols:
            self.get_logger().warn(f'Grid size mismatch: expected {self.grid_rows}x{self.grid_cols}, got {rows}x{cols}')
            return
        
        # Extract feature grid
        grid_size = rows * cols
        if len(msg.data) != 2 + grid_size:
            return
        
        grid_flat = np.array(msg.data[2:], dtype=np.int32)
        self.current_feature_grid = grid_flat.reshape((rows, cols))
        
        # Generate and publish occupancy grid
        self.generate_and_publish()
    
    def aruco_callback(self, msg):
        """Process ArUco detections from vision node"""
        if len(msg.data) < 1:
            self.current_aruco_detections = None
            return
        
        num_markers = int(msg.data[0])
        if num_markers == 0:
            self.current_aruco_detections = None
            return
        
        # Parse detections
        detections = []
        for i in range(num_markers):
            base_idx = 1 + i * 5
            if base_idx + 5 <= len(msg.data):
                marker_id = int(msg.data[base_idx])
                center_x = msg.data[base_idx + 1]
                center_y = msg.data[base_idx + 2]
                
                # Determine object type based on marker ID
                # This should match your aruco_detector object_database
                object_info = self.get_object_info(marker_id)
                
                detection = {
                    'id': marker_id,
                    'center': (center_x, center_y),
                    'object_info': object_info
                }
                detections.append(detection)
        
        self.current_aruco_detections = detections
        
        # Generate and publish occupancy grid
        self.generate_and_publish()
    
    def get_object_info(self, marker_id):
        """Get object information based on marker ID"""
        # This should match the object_database in aruco_detector.py
        object_database = {
            0: {'type': 'box_small', 'destination': 'zone_a'},
            1: {'type': 'box_medium', 'destination': 'zone_a'},
            2: {'type': 'cylinder', 'destination': 'zone_b'},
            3: {'type': 'sphere', 'destination': 'zone_b'},
            4: {'type': 'tool', 'destination': 'zone_c'},
            5: {'type': 'tool', 'destination': 'zone_c'},
            10: {'type': 'zone_a', 'destination': None},
            11: {'type': 'zone_b', 'destination': None},
            12: {'type': 'zone_c', 'destination': None},
        }
        
        return object_database.get(marker_id, {
            'type': 'unknown',
            'destination': None
        })
    
    def generate_and_publish(self):
        """Generate occupancy grid and publish"""
        if self.current_feature_grid is None:
            return
        
        # Generate occupancy grid
        occupancy = self.generator.generate_occupancy_grid(
            self.current_feature_grid,
            self.current_aruco_detections
        )
        
        # Publish custom 3-bit occupancy grid
        self.publish_custom_occupancy(occupancy)
        
        # Publish standard ROS occupancy grid (for rviz)
        if self.publish_ros_occ:
            self.publish_ros_occupancy(occupancy)
        
        # Publish visualization
        if self.publish_viz:
            self.publish_visualization(occupancy)
    
    def publish_custom_occupancy(self, occupancy):
        """Publish custom 3-bit occupancy grid"""
        msg = UInt8MultiArray()
        
        # Flatten occupancy grid
        flat = occupancy.flatten()
        
        # Message format: [rows, cols, occupancy_data...]
        msg.data = [self.grid_rows, self.grid_cols] + flat.tolist()
        
        self.occupancy_pub.publish(msg)
    
    def publish_ros_occupancy(self, occupancy):
        """Publish standard ROS OccupancyGrid message for rviz"""
        msg = OccupancyGrid()
        
        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        # Grid info
        msg.info.resolution = 0.1  # 10cm per cell (adjust as needed)
        msg.info.width = self.grid_cols
        msg.info.height = self.grid_rows
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        
        # Convert 3-bit to standard occupancy values (0-100)
        # 0 = free, 100 = occupied, -1 = unknown
        ros_data = np.full(occupancy.shape, -1, dtype=np.int8)
        
        ros_data[occupancy == self.generator.UNKNOWN] = -1  # Unknown
        ros_data[occupancy == self.generator.FREE] = 0      # Free
        ros_data[occupancy == self.generator.PICKUP] = 50   # Pickup (moderate)
        ros_data[occupancy == self.generator.DROPOFF] = 75  # Dropoff
        
        # Flatten and convert
        msg.data = ros_data.flatten().tolist()
        
        self.ros_occupancy_pub.publish(msg)
    
    def publish_visualization(self, occupancy):
        """Publish visualization image"""
        viz = self.generator.visualize_occupancy_grid(occupancy, cell_size=60)
        
        msg = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        self.viz_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = OccupancyGridNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()