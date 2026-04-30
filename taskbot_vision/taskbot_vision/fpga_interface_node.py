#!/usr/bin/env python3
"""
FPGA Interface Node
Receives grid features from vision node, sends to FPGA SNN, receives motor commands
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import serial
import struct


class FPGAInterfaceNode(Node):
    """
    Interface between ROS2 vision system and FPGA neuromorphic accelerator
    """
    
    def __init__(self):
        super().__init__('fpga_interface_node')
        
        # Declare parameters
        self.declare_parameter('fpga_port', '/dev/ttyUSB0')
        self.declare_parameter('fpga_baudrate', 115200)
        self.declare_parameter('enable_fpga', True)
        self.declare_parameter('motor_simulation', False)
        
        # Get parameters
        self.fpga_port = self.get_parameter('fpga_port').value
        self.fpga_baudrate = self.get_parameter('fpga_baudrate').value
        self.enable_fpga = self.get_parameter('enable_fpga').value
        self.motor_simulation = self.get_parameter('motor_simulation').value
        
        # Initialize FPGA connection
        self.fpga_serial = None
        if self.enable_fpga:
            try:
                self.fpga_serial = serial.Serial(
                    self.fpga_port,
                    self.fpga_baudrate,
                    timeout=0.1
                )
                self.get_logger().info(f'FPGA connected on {self.fpga_port}')
            except Exception as e:
                self.get_logger().error(f'Failed to connect to FPGA: {e}')
                self.enable_fpga = False
        
        # Subscribers
        self.grid_sub = self.create_subscription(
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
        self.motor_cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        # State
        self.grid_rows = 8
        self.grid_cols = 8
        self.current_grid = None
        self.current_aruco = None
        self.target_object = None
        
        # Motor control limits
        self.max_linear_vel = 0.5  # m/s
        self.max_angular_vel = 1.0  # rad/s
        
        self.get_logger().info('FPGA Interface Node started')
    
    def grid_features_callback(self, msg):
        """Process grid features from vision node"""
        # Parse message: [rows, cols, feature_counts...]
        if len(msg.data) < 2:
            return
        
        self.grid_rows = msg.data[0]
        self.grid_cols = msg.data[1]
        
        grid_size = self.grid_rows * self.grid_cols
        if len(msg.data) != 2 + grid_size:
            self.get_logger().warn(f'Invalid grid data size: {len(msg.data)}')
            return
        
        # Extract grid
        grid_flat = np.array(msg.data[2:], dtype=np.int32)
        self.current_grid = grid_flat.reshape((self.grid_rows, self.grid_cols))
        
        # Process with FPGA
        self.process_navigation()
    
    def aruco_callback(self, msg):
        """Process ArUco detections from vision node"""
        if len(msg.data) < 1:
            return
        
        num_markers = int(msg.data[0])
        if num_markers == 0:
            self.current_aruco = None
            return
        
        # Parse detections: [num_markers, id1, x1, y1, area1, angle1, ...]
        detections = []
        for i in range(num_markers):
            base_idx = 1 + i * 5
            if base_idx + 5 <= len(msg.data):
                detection = {
                    'id': int(msg.data[base_idx]),
                    'x': msg.data[base_idx + 1],
                    'y': msg.data[base_idx + 2],
                    'area': msg.data[base_idx + 3],
                    'angle': msg.data[base_idx + 4]
                }
                detections.append(detection)
        
        self.current_aruco = detections
        
        # Update navigation target
        self.update_target()
    
    def update_target(self):
        """Update navigation target based on ArUco detections"""
        if not self.current_aruco:
            self.target_object = None
            return
        
        # Simple strategy: target the closest object (largest area)
        # In practice, you'd filter by object type and task state
        target = max(self.current_aruco, key=lambda x: x['area'])
        self.target_object = target
        
        self.get_logger().info(f'Target updated: ID {target["id"]} at ({target["x"]:.0f}, {target["y"]:.0f})')
    
    def process_navigation(self):
        """
        Main navigation processing:
        1. Encode grid features
        2. Send to FPGA SNN
        3. Receive motor commands
        4. Publish motor commands
        """
        if self.current_grid is None:
            return
        
        # Encode features for FPGA
        encoded_data = self.encode_for_fpga(self.current_grid)
        
        # Send to FPGA and get response
        if self.enable_fpga and self.fpga_serial:
            motor_commands = self.fpga_process(encoded_data)
        else:
            # Fallback: simple reactive navigation
            motor_commands = self.simple_navigation(self.current_grid)
        
        # Publish motor commands
        self.publish_motor_commands(motor_commands)
    
    def encode_for_fpga(self, grid):
        """
        Encode grid features for FPGA transmission
        Returns byte array ready for serial transmission
        """
        # Flatten and normalize grid
        grid_flat = grid.flatten().astype(np.float32)
        grid_normalized = grid_flat / (np.max(grid_flat) + 1e-6)
        
        # Add target bias if we have a target
        target_bias = np.zeros_like(grid_flat, dtype=np.float32)
        if self.target_object:
            # Calculate which grid cell contains the target
            cell_width = 640 // self.grid_cols
            cell_height = 360 // self.grid_rows
            
            target_col = int(self.target_object['x'] / cell_width)
            target_row = int(self.target_object['y'] / cell_height)
            target_col = min(target_col, self.grid_cols - 1)
            target_row = min(target_row, self.grid_rows - 1)
            
            # Create gradient toward target
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    idx = row * self.grid_cols + col
                    dist = np.sqrt((row - target_row)**2 + (col - target_col)**2)
                    max_dist = np.sqrt(self.grid_rows**2 + self.grid_cols**2)
                    target_bias[idx] = 1.0 - (dist / max_dist)
        
        # Combine features and target bias
        combined = np.concatenate([grid_normalized, target_bias])
        
        # Convert to bytes (header + data)
        # Header: [START_BYTE, NUM_FEATURES_LSB, NUM_FEATURES_MSB]
        num_features = len(combined)
        header = struct.pack('BBB', 0xAA, num_features & 0xFF, (num_features >> 8) & 0xFF)
        
        # Data: each feature as uint8 (0-255)
        data = (combined * 255).astype(np.uint8).tobytes()
        
        return header + data
    
    def fpga_process(self, encoded_data):
        """
        Send data to FPGA and receive motor commands
        
        Args:
            encoded_data: Byte array to send to FPGA
        
        Returns:
            (linear_velocity, angular_velocity) tuple
        """
        try:
            # Send data
            self.fpga_serial.write(encoded_data)
            
            # Wait for response (4 bytes: linear_vel, angular_vel as int16)
            response = self.fpga_serial.read(4)
            
            if len(response) == 4:
                linear_raw, angular_raw = struct.unpack('hh', response)
                
                # Convert from int16 range to actual velocities
                linear_vel = (linear_raw / 32767.0) * self.max_linear_vel
                angular_vel = (angular_raw / 32767.0) * self.max_angular_vel
                
                return (linear_vel, angular_vel)
            else:
                self.get_logger().warn('Incomplete FPGA response')
                return (0.0, 0.0)
                
        except Exception as e:
            self.get_logger().error(f'FPGA communication error: {e}')
            return (0.0, 0.0)
    
    def simple_navigation(self, grid):
        """
        Simple reactive navigation (fallback when FPGA not available)
        This is a basic obstacle avoidance algorithm
        
        Args:
            grid: Feature grid
        
        Returns:
            (linear_velocity, angular_velocity) tuple
        """
        # Split grid into left, center, right regions
        left_region = grid[:, :self.grid_cols//3]
        center_region = grid[:, self.grid_cols//3:2*self.grid_cols//3]
        right_region = grid[:, 2*self.grid_cols//3:]
        
        # Sum features in each region
        left_features = np.sum(left_region)
        center_features = np.sum(center_region)
        right_features = np.sum(right_region)
        
        # If we have a target, navigate toward it
        if self.target_object:
            frame_center_x = 320
            target_x = self.target_object['x']
            
            # Calculate error from center
            error = target_x - frame_center_x
            
            # Proportional control for turning
            angular_vel = -error / 320.0 * self.max_angular_vel
            
            # Move forward if target is roughly centered and not too many obstacles
            if abs(error) < 80 and center_features < 50:
                linear_vel = 0.3
            else:
                linear_vel = 0.1
        else:
            # No target: simple obstacle avoidance
            # Turn away from obstacles
            if center_features > 20:  # Obstacle ahead
                if left_features < right_features:
                    angular_vel = self.max_angular_vel * 0.5  # Turn left
                else:
                    angular_vel = -self.max_angular_vel * 0.5  # Turn right
                linear_vel = 0.0
            else:
                # Path is clear
                linear_vel = 0.2
                angular_vel = 0.0
        
        return (linear_vel, angular_vel)
    
    def publish_motor_commands(self, commands):
        """
        Publish motor commands as Twist message
        
        Args:
            commands: (linear_velocity, angular_velocity) tuple
        """
        linear_vel, angular_vel = commands
        
        # Create Twist message
        cmd = Twist()
        cmd.linear.x = float(linear_vel)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(angular_vel)
        
        self.motor_cmd_pub.publish(cmd)
        
        if self.motor_simulation:
            self.get_logger().info(f'Motor CMD: linear={linear_vel:.2f}, angular={angular_vel:.2f}')
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.fpga_serial:
            self.fpga_serial.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = FPGAInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
