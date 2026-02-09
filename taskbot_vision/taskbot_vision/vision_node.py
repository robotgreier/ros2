#!/usr/bin/env python3
"""
Vision Node for Energy-Aware Autonomous Taskbot
Handles camera capture, ORB feature detection with grid, and ArUco marker detection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Declare parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('use_camera_topic', False)  # NEW: Use ROS2 topic instead
        self.declare_parameter('camera_topic', 'camera/image_raw')  # NEW: Topic name
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 360)
        self.declare_parameter('fps', 30)
        self.declare_parameter('grid_rows', 8)
        self.declare_parameter('grid_cols', 8)
        self.declare_parameter('orb_features', 500)
        self.declare_parameter('publish_debug_image', False)
        
        # Get parameters
        self.camera_index = self.get_parameter('camera_index').value
        self.use_camera_topic = self.get_parameter('use_camera_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.fps = self.get_parameter('fps').value
        self.grid_rows = self.get_parameter('grid_rows').value
        self.grid_cols = self.get_parameter('grid_cols').value
        self.orb_max_features = self.get_parameter('orb_features').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        
        # Initialize camera or topic subscription
        self.cap = None
        self.camera_sub = None
        self.latest_frame = None
        
        if self.use_camera_topic:
            # Subscribe to camera topic instead of direct camera access
            self.get_logger().info(f'Using camera topic: {self.camera_topic}')
            self.camera_sub = self.create_subscription(
                Image,
                self.camera_topic,
                self.camera_callback,
                10
            )
        else:
            # Initialize camera directly
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                self.get_logger().error('Failed to open camera')
                raise RuntimeError('Camera initialization failed')
            
            self.get_logger().info(f'Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps} FPS')
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.orb_max_features)
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Check OpenCV version and use appropriate API
        try:
            # New API (OpenCV 4.7+)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.aruco_api_new = True
            self.get_logger().info('Using new ArUco API (OpenCV 4.7+)')
        except (AttributeError, TypeError):
            # Old API (OpenCV 4.6 and earlier)
            # For old API, we don't need detector object, just dict and params
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except:
                self.aruco_params = None
            self.aruco_detector = None
            self.aruco_api_new = False
            self.get_logger().info('Using old ArUco API (OpenCV 4.6 and earlier)')
        
        # CV Bridge for ROS image conversion
        self.bridge = CvBridge()
        
        # Publishers
        self.grid_feature_pub = self.create_publisher(
            Int32MultiArray, 
            'vision/grid_features', 
            10
        )
        self.aruco_pose_pub = self.create_publisher(
            Float32MultiArray,
            'vision/aruco_detections',
            10
        )
        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(
                Image,
                'vision/debug_image',
                10
            )
        
        # Timer for processing loop
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.process_frame)
        
        # Grid calculation
        self.cell_height = self.frame_height // self.grid_rows
        self.cell_width = self.frame_width // self.grid_cols
        
        self.get_logger().info(f'Grid: {self.grid_rows}x{self.grid_cols}, Cell size: {self.cell_width}x{self.cell_height}')
        self.get_logger().info('Vision node started')
    
    def camera_callback(self, msg):
        """Callback for camera topic subscription"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
    
    def process_frame(self):
        """Main processing loop - captures and processes each frame"""
        # Get frame from camera or topic
        if self.use_camera_topic:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
        else:
            ret, frame = self.cap.read()
            
            if not ret:
                self.get_logger().warn('Failed to capture frame')
                return
        
        # Convert to grayscale for ORB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Process grid features
        grid_features = self.compute_grid_features(keypoints)
        self.publish_grid_features(grid_features)
        
        # Detect ArUco markers
        aruco_data = self.detect_aruco_markers(frame)
        if aruco_data is not None:
            self.publish_aruco_detections(aruco_data)
        
        # Publish debug image if enabled
        if self.publish_debug:
            debug_img = self.create_debug_image(frame, keypoints, grid_features, aruco_data)
            self.publish_debug_image(debug_img)
    
    def compute_grid_features(self, keypoints):
        """
        Divide frame into grid and count features in each cell
        Returns: 2D numpy array (grid_rows x grid_cols) with feature counts
        """
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Calculate grid cell indices
            col = min(x // self.cell_width, self.grid_cols - 1)
            row = min(y // self.cell_height, self.grid_rows - 1)
            
            grid[row, col] += 1
        
        return grid
    
    def encode_grid_features(self, grid):
        """
        Encode grid features for FPGA/SNN processing
        Returns encoded data optimized for neuromorphic processing
        
        This creates a compact representation:
        - Feature count per cell
        - Normalized spatial weighting (cells closer to center get different weights)
        - Can be extended with directional bias
        """
        # Flatten grid for transmission
        flat_grid = grid.flatten()
        
        # Optional: Add spatial encoding (position weight)
        # Center cells might be more important for navigation
        center_row, center_col = self.grid_rows // 2, self.grid_cols // 2
        
        spatial_weights = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Distance from center (normalized)
                dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
                max_dist = np.sqrt(center_row**2 + center_col**2)
                weight = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
                spatial_weights.append(weight)
        
        spatial_weights = np.array(spatial_weights, dtype=np.float32)
        
        return flat_grid, spatial_weights
    
    def publish_grid_features(self, grid):
        """Publish grid feature data to ROS2 topic"""
        msg = Int32MultiArray()
        
        # Flatten grid and add metadata
        flat_grid, spatial_weights = self.encode_grid_features(grid)
        
        # Message layout: [rows, cols, feature_counts...]
        msg.data = [self.grid_rows, self.grid_cols] + flat_grid.tolist()
        
        self.grid_feature_pub.publish(msg)
    
    def detect_aruco_markers(self, frame):
        """
        Detect ArUco markers in frame
        Returns list of detected markers with [id, center_x, center_y, area]
        """
        # Use appropriate API based on OpenCV version
        if self.aruco_api_new:
            # New API (OpenCV 4.7+)
            corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        else:
            # Old API (OpenCV 4.6 and earlier)
            if self.aruco_params is not None:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame, 
                    self.aruco_dict, 
                    parameters=self.aruco_params
                )
            else:
                # Very old API - no parameters
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame, 
                    self.aruco_dict
                )
        
        if ids is None:
            return None
        
        detections = []
        for i, corner in enumerate(corners):
            # Get marker ID
            marker_id = ids[i][0]
            
            # Calculate center point
            corner_points = corner[0]
            center_x = np.mean(corner_points[:, 0])
            center_y = np.mean(corner_points[:, 1])
            
            # Calculate approximate area (for distance estimation)
            area = cv2.contourArea(corner_points)
            
            # Calculate orientation (angle of marker)
            # Vector from center to first corner
            vec_x = corner_points[0][0] - center_x
            vec_y = corner_points[0][1] - center_y
            angle = np.arctan2(vec_y, vec_x)
            
            detections.append([marker_id, center_x, center_y, area, angle])
        
        return detections
    
    def publish_aruco_detections(self, detections):
        """Publish ArUco marker detections"""
        msg = Float32MultiArray()
        
        # Flatten detections: [num_markers, id1, x1, y1, area1, angle1, id2, ...]
        flat_data = [len(detections)]
        for det in detections:
            flat_data.extend(det)
        
        msg.data = flat_data
        self.aruco_pose_pub.publish(msg)
    
    def create_debug_image(self, frame, keypoints, grid_features, aruco_data):
        """Create visualization image with all detections"""
        debug_img = frame.copy()
        
        # Draw grid
        for i in range(1, self.grid_rows):
            y = i * self.cell_height
            cv2.line(debug_img, (0, y), (self.frame_width, y), (100, 100, 100), 1)
        for j in range(1, self.grid_cols):
            x = j * self.cell_width
            cv2.line(debug_img, (x, 0), (x, self.frame_height), (100, 100, 100), 1)
        
        # Draw feature counts in each cell
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                count = grid_features[row, col]
                if count > 0:
                    x = col * self.cell_width + self.cell_width // 2
                    y = row * self.cell_height + self.cell_height // 2
                    
                    # Color intensity based on feature count
                    intensity = int(min(255, count * 30))
                    cv2.circle(debug_img, (x, y), 15, (0, intensity, 0), -1)
                    cv2.putText(debug_img, str(count), (x-10, y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw ORB keypoints
        cv2.drawKeypoints(debug_img, keypoints, debug_img, 
                         color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Draw ArUco markers
        if aruco_data is not None:
            for det in aruco_data:
                marker_id, cx, cy, area, angle = det
                cx, cy = int(cx), int(cy)
                
                # Draw marker center and ID
                cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(debug_img, f'ID:{int(marker_id)}', (cx+10, cy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw orientation arrow
                arrow_len = 30
                end_x = int(cx + arrow_len * np.cos(angle))
                end_y = int(cy + arrow_len * np.sin(angle))
                cv2.arrowedLine(debug_img, (cx, cy), (end_x, end_y), (255, 0, 0), 2)
        
        return debug_img
    
    def publish_debug_image(self, image):
        """Publish debug image to ROS2 topic"""
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.debug_image_pub.publish(msg)
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VisionNode()
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