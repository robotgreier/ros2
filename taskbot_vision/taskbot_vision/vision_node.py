#!/usr/bin/env python3
"""
Vision Node for Energy-Aware Autonomous Taskbot
Handles camera capture, ORB feature detection with grid, and ArUco marker detection
"""

import rclpy                                                    # ROS2 Python client library
from rclpy.node import Node                                     # Base class for ROS2 nodes
from sensor_msgs.msg import Image                               # ROS2 message type for images
from std_msgs.msg import Int32MultiArray, Float32MultiArray     # ROS2 message types for feature and marker data
from geometry_msgs.msg import PoseStamped                       # ROS2 message type for pose data   
from cv_bridge import CvBridge                                  # OpenCV to ROS image conversion
import cv2                                                      # OpenCV library for computer vision
import numpy as np                                              # NumPy for numerical operations

# VisionNode class definition
class VisionNode(Node):
    def __init__(self):
        # Initialize the ROS2 node with the name 'vision_node'
        super().__init__('vision_node')
        
        # Declare default parameters if the launch file does not provide them
        # These parameters can be overridden by the launch file or command line
        self.declare_parameter('camera_index', 0)                   # Camera index for direct camera access (if not using topic subscription)
        self.declare_parameter('use_camera_topic', False)           # Option to subscribe to camera topic (True) instead of direct camera access
        self.declare_parameter('camera_topic', 'camera/image_raw')  # Topic name for camera subscription if use_camera_topic is True
        self.declare_parameter('frame_width', 640)                  # Frame width for camera capture
        self.declare_parameter('frame_height', 480)                 # Frame height for camera capture
        self.declare_parameter('fps', 30)                           # Frames per second for processing loop
        self.declare_parameter('grid_rows', 8)                      # Number of rows in the grid for feature counting
        self.declare_parameter('grid_cols', 8)                      # Number of columns in the grid for feature counting
        self.declare_parameter('orb_features', 500)                 # Maximum number of ORB features to detect
        self.declare_parameter('publish_debug_image', False)        # Option to publish debug image with detections visualized
        
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
        
        # Initialize input: direct camera or ROS topic subscription
        self.cap = None                                 # OpenCV VideoCapture object
        self.camera_sub = None                          # ROS2 subscription for camera topic
        self.latest_frame = None                        # Latest frame from camera topic
        
        # If use_camera_topic is True, subscribe to the topic instead of direct camera access
        if self.use_camera_topic:
            self.get_logger().info(f'Using camera topic: {self.camera_topic}')
            self.camera_sub = self.create_subscription(
                Image,
                self.camera_topic,
                self.camera_callback,
                10
            )
        else:
            # Initialize camera directly
            self.cap = cv2.VideoCapture(self.camera_index)              # Open camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)    # Set frame width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)  # Set frame height
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)                    # Set frames per second
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                self.get_logger().error('Failed to open camera')
                raise RuntimeError('Camera initialization failed')
            
            # Log camera initialization details and settings
            self.get_logger().info(f'Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps} FPS')
        
        # Initialize ORB detector for feature detection
        self.orb = cv2.ORB_create(nfeatures=self.orb_max_features)
        
        # Initialize ArUco detector for marker detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Check OpenCV version and use appropriate API
        try:
            # New API (OpenCV 4.7+)
            self.aruco_params = cv2.aruco.DetectorParameters()                                  # Create parameters object
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)   # Create detector object
            self.aruco_api_new = True                                                           # Flag to indicate new API usage
            self.get_logger().info('Using new ArUco API (OpenCV 4.7+)')                         # Log API version being used
        except (AttributeError, TypeError):
            # Old API (OpenCV 4.6 and earlier)
            # For old API, we don't need detector object, just dict and params
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()   # Create parameters for old API
            except:
                self.aruco_params = None                                    # Old API might not have parameters creation function
            self.aruco_detector = None                                      # Old API does not use a separate detector object
            self.aruco_api_new = False                                      # Flag to indicate old API usage    
            self.get_logger().info('Using old ArUco API (OpenCV 4.6 and earlier)')
        
        # CV Bridge for ROS image conversion between OpenCV and ROS Image messages
        self.bridge = CvBridge()
        
        # Publishers for grid features and ArUco detections
        self.grid_feature_pub = self.create_publisher(
            # publish grid features as a flat array with metadata (rows, cols, feature counts) per grid cell
            Int32MultiArray, # Int32MultiArray is used for grid features since they are counts (integers)
            'vision/grid_features', 
            10
        )
        self.aruco_pose_pub = self.create_publisher(
            # publish ArUco detections as a flat array: [num_markers, id1, x1, y1, area1, angle1, id2, ...]
            Float32MultiArray, # Float32MultiArray is used for ArUco detections since they include continuous values (positions, area, angle)
            'vision/aruco_detections',
            10
        )
        if self.publish_debug:
            # Publisher for debug image with visualizations of detections
            self.debug_image_pub = self.create_publisher(
                Image, # Publish debug image as ROS Image message
                'vision/debug_image',
                10
            )
        
        # Timer for processing loop
        timer_period = 1.0 / self.fps                                       # Set timer period based on desired frames per second
        self.timer = self.create_timer(timer_period, self.process_frame)    # Create timer that calls process_frame at the specified rate
        
        # Grid cell calculation size based on frame dimensions and grid configuration
        self.cell_height = self.frame_height // self.grid_rows  # Calculate cell height based on frame height and number of grid rows
        self.cell_width = self.frame_width // self.grid_cols    # Calculate cell width based on frame width and number of grid columns
        
        # Log initialization details and settings for grid and feature detection
        self.get_logger().info(f'Grid: {self.grid_rows}x{self.grid_cols}, Cell size: {self.cell_width}x{self.cell_height}')
        self.get_logger().info('Vision node started')
    
    # Callback function for camera topic subscription - updates latest frame for processing
    def camera_callback(self, msg):
        """Callback for camera topic subscription"""
        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        except Exception as e:
            # Log error if image conversion fails
            self.get_logger().error(f'Failed to convert image: {e}')
    
    def process_frame(self):
        """Main processing loop - captures and processes each frame"""
        # Get frame from camera or topic
        if self.use_camera_topic:
            if self.latest_frame is None:       # If no frame has been received yet, skip processing
                return
            frame = self.latest_frame.copy()    # Use a copy of the latest frame to avoid modifying the original data from the topic
        else:
            ret, frame = self.cap.read()        # Capture frame from camera
            if not ret:                         # If frame capture fails, log a warning and skip processing
                self.get_logger().warn('Failed to capture frame')
                return
        
        # Convert to grayscale for ORB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB features in the grayscale image (keypoints and descriptors)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Compute grid features by counting keypoints in each cell of the defined grid
        grid_features = self.compute_grid_features(keypoints)
        self.publish_grid_features(grid_features)
        
        # Detect ArUco markers on the original color frame to get more accurate detections (color can help with marker detection)
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
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32) # Initialize grid with zeros to count features in each cell
        
        for kp in keypoints:                                                # Iterate through each detected keypoint
            x, y = int(kp.pt[0]), int(kp.pt[1])                             # Get keypoint coordinates (x, y) in pixel space
            
            col = min(x // self.cell_width, self.grid_cols - 1)             # Calculate column index based on x coordinate and cell width
            row = min(y // self.cell_height, self.grid_rows - 1)            # Calculate row index based on y coordinate and cell height
            
            grid[row, col] += 1                                             # Increment feature count for the corresponding grid cell 
        
        return grid                                                         # Return the grid with feature counts for each cell
    
    def encode_grid_features(self, grid):
        """
        Encode grid features for FPGA/SNN processing
        Returns encoded data optimized for neuromorphic processing
        
        This creates a compact representation:
        - Feature count per cell
        - Normalized spatial weighting (cells closer to center get different weights)
        - Can be extended with directional bias
        """
        
        flat_grid = grid.flatten() # Flatten the 2D grid into a 1D array for easier processing and transmission to FPGA/SNN
        
        # Optional: Add spatial encoding (position weight) by calculating distance from center of the grid and applying a weighting function
        # Center cells might be more important for navigation and obstacle avoidance, so we can give them higher weights
        center_row, center_col = self.grid_rows // 2, self.grid_cols // 2 # Calculate center cell indices for spatial weighting

       # Calculate spatial weights based on distance from center (normalized to [0, 1]) 
        spatial_weights = []                                                    # Initialize list to hold spatial weights for each cell
        for row in range(self.grid_rows):                                       # Iterate through each row of the grid
            for col in range(self.grid_cols):                                   # Iterate through each column of the grid
                dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)   # Calculate distance from the center cell using Euclidean distance formula
                max_dist = np.sqrt(center_row**2 + center_col**2)               # Calculate maximum possible distance from the center (corner cell) for normalization
                weight = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0       # Normalize distance to [0, 1] and invert it so that closer cells have higher weights (weight is 1 at center and decreases towards edges)
                spatial_weights.append(weight)                                  # Append the calculated weight for the current cell to the spatial_weights list
        
        spatial_weights = np.array(spatial_weights, dtype=np.float32)           # Convert spatial weights list to a NumPy array for easier handling and potential use in FPGA/SNN processing
        
        return flat_grid, spatial_weights                                       # Return the flattened grid and corresponding spatial weights for each cell, which can be used for further processing or transmission to FPGA/SNN
    
    def publish_grid_features(self, grid):
        """Publish grid feature data to ROS2 topic"""
        msg = Int32MultiArray()         # Create a new Int32MultiArray message to hold the grid feature data
        
        flat_grid, spatial_weights = self.encode_grid_features(grid) # Encode the grid features and get the flattened grid and spatial weights for each cell
        
        msg.data = [self.grid_rows, self.grid_cols] + flat_grid.tolist() # message format: [num_rows, num_cols, feature_count_cell1, feature_count_cell2, ...]
        
        self.grid_feature_pub.publish(msg)              # Publish the grid feature data to the 'vision/grid_features' topic for use by other nodes (e.g., FPGA/SNN processing, navigation, etc.)    
    
    def detect_aruco_markers(self, frame):
        """
        Detect ArUco markers in frame
        Returns list of detected markers with [id, center_x, center_y, area]
        """
        # Use appropriate API based on OpenCV version
        if self.aruco_api_new:
            # New API (OpenCV 4.7+)
            corners, ids, rejected = self.aruco_detector.detectMarkers(frame) # Detect ArUco markers and returns corners, ids, and rejected candidates
        else:
            # Old API (OpenCV 4.6 and earlier)
            if self.aruco_params is not None:                                 # If parameters are available for old API, use them
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame, 
                    self.aruco_dict, 
                    parameters=self.aruco_params
                )
            else:
                # Very old API - no parameters
                corners, ids, rejected = cv2.aruco.detectMarkers(             # Detect ArUco markers using the old API without parameters
                    frame, 
                    self.aruco_dict
                )
        
        if ids is None: # If no markers are detected, return None to indicate no detections
            return None
        
        detections = []                        # Initialize list to hold detected marker information (id, center_x, center_y, area, angle)
        for i, corner in enumerate(corners):   # Iterate through each detected marker's corners to extract information about the marker
            marker_id = ids[i][0]              # Get the marker ID from the ids array (ids is a 2D array, so we take the first element of the first dimension)
            
            # Calculate center point of the marker by averaging the corner points (corners is a list of arrays, where each array contains the 4 corner points of the detected marker)
            corner_points = corner[0]
            center_x = np.mean(corner_points[:, 0])
            center_y = np.mean(corner_points[:, 1])
            
            # Calculate approximate area (for distance estimation) by using the contour area of the marker corners (how large is the marker in the frame, helps estimate distance to the marker)
            area = cv2.contourArea(corner_points)   
            
            # Calculate orientation (angle of marker) 
            # Vector from center to first corner
            vec_x = corner_points[0][0] - center_x  # Calculate the vector x-axis from center to corner point  
            vec_y = corner_points[0][1] - center_y  # Calculate the vector y-axis from center to corner point
            angle = np.arctan2(vec_y, vec_x)        # Calculate the angle of the marker in radians using arctangent of the vector components (gives the orientation of the marker relative to the camera)
            
            detections.append([marker_id, center_x, center_y, area, angle]) # Append the detected marker information (id, center coordinates, area, and angle) to the detections list        
        return detections   # Return the list of detected markers with their information for further processing or publishing to ROS2 topics
    
    def publish_aruco_detections(self, detections):
        """Publish ArUco marker detections"""
        msg = Float32MultiArray()
        
        # Flatten detections: [num_markers, id1, x1, y1, area1, angle1, id2, ...] in a single array for easier processing by downstream nodes (e.g., FPGA/SNN)
        flat_data = [len(detections)]
        for det in detections:
            flat_data.extend(det)
        
        msg.data = flat_data
        self.aruco_pose_pub.publish(msg)    # Publish the ArUco marker detection data to the 'vision/aruco_detections' topic for use by other nodes (e.g., navigation, obstacle avoidance, etc.)
    
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
        
        # Draw ORB keypoints with rich keypoint visualization (size and orientation) for better debugging and understanding of the detected features
        cv2.drawKeypoints(debug_img, keypoints, debug_img, 
                         color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
        
        # Draw ArUco markers with ID, center, and orientation for better visualization of detected markers and their poses in the frame
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

def main(args=None):            # Main function to initialize and run the VisionNode
    rclpy.init(args=args)       # Initialize the ROS2 Python client library
    
    try:
        node = VisionNode()     # Create an instance of the VisionNode class, which sets up the ROS2 node, parameters, publishers, and processing loop
        rclpy.spin(node)        # Keep the node running and processing callbacks until it is shut down (Ctrl+C or programmed)
    except KeyboardInterrupt:   # Handle graceful shutdown on Ctrl+C
        pass
    except Exception as e:      # Error handling for any exceptions during node initialization or execution
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()    # Shutdown the ROS2 client library to clean up resources and allow for a graceful exit


if __name__ == '__main__':      # Entry point for the script, calls the main function to start the VisionNode
    main()