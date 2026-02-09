#!/usr/bin/env python3
"""
Mock Camera Node for Testing Without Hardware
Generates synthetic frames with features and ArUco markers
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class MockCameraNode(Node):
    def __init__(self):
        super().__init__('mock_camera_node')
        
        # Parameters
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 360)
        self.declare_parameter('fps', 30)
        self.declare_parameter('add_features', True)
        self.declare_parameter('add_aruco', True)
        
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.fps = self.get_parameter('fps').value
        self.add_features = self.get_parameter('add_features').value
        self.add_aruco = self.get_parameter('add_aruco').value
        
        # Publisher
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Timer for publishing
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.publish_frame)
        
        # Frame counter for animation
        self.frame_count = 0
        
        # ArUco dictionary
        if self.add_aruco:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        self.get_logger().info(f'Mock camera started: {self.frame_width}x{self.frame_height} @ {self.fps} FPS')
        self.get_logger().info(f'Features: {self.add_features}, ArUco: {self.add_aruco}')
    
    def generate_synthetic_frame(self):
        """Generate a synthetic frame with features and markers"""
        # Create base frame with gradient
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(self.frame_height):
            intensity = int(50 + (y / self.frame_height) * 100)
            frame[y, :] = [intensity, intensity, intensity]
        
        # Add some moving geometric shapes (to create features)
        if self.add_features:
            # Moving circle
            t = self.frame_count * 0.05
            cx = int(self.frame_width / 2 + 150 * np.cos(t))
            cy = int(self.frame_height / 2 + 100 * np.sin(t))
            cv2.circle(frame, (cx, cy), 40, (200, 200, 200), 2)
            
            # Random rectangles
            for i in range(5):
                x = int((self.frame_width / 6) * (i + 0.5))
                y = int(self.frame_height / 2 + 50 * np.sin(t + i))
                cv2.rectangle(frame, (x-30, y-30), (x+30, y+30), (180, 180, 180), 2)
            
            # Add some random noise points (features)
            np.random.seed(self.frame_count % 100)
            num_points = 50
            for _ in range(num_points):
                px = np.random.randint(0, self.frame_width)
                py = np.random.randint(0, self.frame_height)
                size = np.random.randint(2, 8)
                cv2.circle(frame, (px, py), size, (255, 255, 255), -1)
        
        # Add ArUco markers
        if self.add_aruco:
            # Generate a few ArUco markers and place them in the frame
            markers_to_add = [
                (0, 100, 100, 60),   # (id, x, y, size)
                (1, 500, 100, 60),
                (10, 320, 250, 80),  # Zone marker (larger)
            ]
            
            for marker_id, x, y, size in markers_to_add:
                # Generate marker - compatible with older OpenCV versions
                try:
                    # Try new API first (OpenCV 4.7+)
                    marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size)
                except AttributeError:
                    # Fall back to old API (OpenCV 4.6 and earlier)
                    marker_img = cv2.aruco.drawMarker(self.aruco_dict, marker_id, size)
                
                # Convert to BGR
                marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
                
                # Place in frame
                x_start = max(0, x - size // 2)
                y_start = max(0, y - size // 2)
                x_end = min(self.frame_width, x + size // 2)
                y_end = min(self.frame_height, y + size // 2)
                
                if x_end > x_start and y_end > y_start:
                    marker_h = y_end - y_start
                    marker_w = x_end - x_start
                    frame[y_start:y_end, x_start:x_end] = cv2.resize(marker_bgr, (marker_w, marker_h))
        
        # Add frame counter
        cv2.putText(frame, f"Mock Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def publish_frame(self):
        """Generate and publish a frame"""
        frame = self.generate_synthetic_frame()
        
        # Convert to ROS message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        # Publish
        self.image_pub.publish(msg)
        
        # Increment counter
        self.frame_count += 1


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MockCameraNode()
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