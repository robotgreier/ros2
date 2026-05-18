#!/usr/bin/env python3
"""
ArUco Marker Detection and Object Classification
Handles detection, pose estimation, and object type classification for the taskbot
"""

import cv2
import numpy as np


class ArucoObjectDetector:
    """
    Detects ArUco markers and classifies objects for pick-and-place tasks
    """
    
    def __init__(self, 
                 marker_dict=cv2.aruco.DICT_4X4_50,
                 marker_size_mm=50.0,
                 camera_matrix=None,
                 dist_coeffs=None):
        """
        Args:
            marker_dict: ArUco dictionary to use
            marker_size_mm: Real-world size of markers in millimeters
            camera_matrix: Camera calibration matrix (3x3)
            dist_coeffs: Camera distortion coefficients
        """
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
        
        # Check OpenCV version and use appropriate API
        try:
            # New API (OpenCV 4.7+)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.aruco_api_new = True
        except (AttributeError, TypeError):
            # Old API (OpenCV 4.6 and earlier)
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except:
                self.aruco_params = None
            self.detector = None
            self.aruco_api_new = False
        
        # Marker properties
        self.marker_size_mm = marker_size_mm
        self.marker_size_m = marker_size_mm / 1000.0  # Convert to meters
        
        # Camera calibration (if not provided, use defaults for C922)
        # Note: You should calibrate your specific camera for best results
        if camera_matrix is None:
            # Default approximation for C922 at 640x360
            fx = 600.0  # Focal length x
            fy = 600.0  # Focal length y
            cx = 320.0  # Principal point x
            cy = 180.0  # Principal point y
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5, dtype=np.float32)
        else:
            self.dist_coeffs = dist_coeffs
        
        # Object type mapping (marker ID to object type and destination)
        self.object_database = {
            # Format: marker_id: {'type': 'object_name', 'destination': zone_id}
            0: {'type': 'box_small', 'destination': 'zone_a', 'color': (255, 0, 0)},
            1: {'type': 'box_medium', 'destination': 'zone_a', 'color': (255, 0, 0)},
            2: {'type': 'cylinder', 'destination': 'zone_b', 'color': (0, 255, 0)},
            3: {'type': 'sphere', 'destination': 'zone_b', 'color': (0, 255, 0)},
            4: {'type': 'tool', 'destination': 'zone_c', 'color': (0, 0, 255)},
            5: {'type': 'tool', 'destination': 'zone_c', 'color': (0, 0, 255)},
            # Destination markers (zones)
            10: {'type': 'zone_a', 'destination': None, 'color': (255, 255, 0)},
            11: {'type': 'zone_b', 'destination': None, 'color': (255, 0, 255)},
            12: {'type': 'zone_c', 'destination': None, 'color': (0, 255, 255)},
        }
    
    def detect_markers(self, frame):
        """
        Detect all ArUco markers in frame
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of detections, each containing:
            {
                'id': marker_id,
                'corners': corner_points,
                'center': (x, y),
                'area': pixel_area,
                'angle': rotation_angle,
                'tvec': translation_vector,
                'rvec': rotation_vector,
                'distance': estimated_distance,
                'object_info': object_metadata
            }
        """
        # Detect markers using appropriate API
        if self.aruco_api_new:
            # New API (OpenCV 4.7+)
            corners, ids, rejected = self.detector.detectMarkers(frame)
        else:
            # Old API (OpenCV 4.6 and earlier)
            if self.aruco_params is not None:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame,
                    self.aruco_dict,
                    parameters=self.aruco_params
                )
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame,
                    self.aruco_dict
                )
        
        if ids is None:
            return []
        
        # Estimate pose for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 
            self.marker_size_m,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        detections = []
        for i in range(len(ids)):
            marker_id = ids[i][0]
            corner_points = corners[i][0]
            
            # Calculate center
            center_x = np.mean(corner_points[:, 0])
            center_y = np.mean(corner_points[:, 1])
            
            # Calculate area
            area = cv2.contourArea(corner_points)
            
            # Calculate rotation angle (in image plane)
            vec_x = corner_points[0][0] - center_x
            vec_y = corner_points[0][1] - center_y
            angle = np.arctan2(vec_y, vec_x)
            
            # Get pose vectors
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            
            # Calculate distance (from translation vector)
            distance = np.linalg.norm(tvec)
            
            # Get object information
            object_info = self.object_database.get(marker_id, {
                'type': 'unknown',
                'destination': None,
                'color': (128, 128, 128)
            })
            
            detection = {
                'id': marker_id,
                'corners': corner_points,
                'center': (center_x, center_y),
                'area': area,
                'angle': angle,
                'rvec': rvec,
                'tvec': tvec,
                'distance': distance,
                'object_info': object_info
            }
            
            detections.append(detection)
        
        return detections
    
    def get_objects_to_collect(self, detections):
        """
        Filter detections to get only objects that need to be collected
        
        Args:
            detections: List of all detections
        
        Returns:
            List of objects that should be picked up
        """
        objects = [d for d in detections if d['object_info']['destination'] is not None]
        return objects
    
    def get_destination_zones(self, detections):
        """
        Filter detections to get destination zone markers
        
        Args:
            detections: List of all detections
        
        Returns:
            Dictionary mapping zone names to their detections
        """
        zones = {}
        for d in detections:
            obj_type = d['object_info']['type']
            if obj_type.startswith('zone_'):
                zones[obj_type] = d
        return zones
    
    def find_nearest_object(self, detections, object_type=None):
        """
        Find the nearest object to pick up
        
        Args:
            detections: List of all detections
            object_type: Optional filter by object type
        
        Returns:
            Detection of nearest object, or None
        """
        objects = self.get_objects_to_collect(detections)
        
        if object_type:
            objects = [o for o in objects if o['object_info']['type'] == object_type]
        
        if not objects:
            return None
        
        # Find nearest by distance
        nearest = min(objects, key=lambda x: x['distance'])
        return nearest
    
    def get_navigation_target(self, detections, current_task='collect'):
        """
        Determine navigation target based on current task
        
        Args:
            detections: List of all detections
            current_task: 'collect' or 'deliver'
        
        Returns:
            Target detection and task type, or (None, None)
        """
        if current_task == 'collect':
            # Find nearest object to collect
            target = self.find_nearest_object(detections)
            return target, 'approach_object'
        
        elif current_task == 'deliver':
            # Requires knowing what object is being carried; not yet implemented
            return None, None
        
        return None, None
    
    def draw_detections(self, frame, detections, draw_axes=True, draw_info=True):
        """
        Draw detected markers and information on frame
        
        Args:
            frame: Input/output image
            detections: List of detections
            draw_axes: Draw 3D axes on markers
            draw_info: Draw text information
        
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        for det in detections:
            marker_id = det['id']
            corners = det['corners']
            center = det['center']
            obj_info = det['object_info']
            distance = det['distance']
            
            # Draw marker boundary
            corners_int = corners.astype(int)
            cv2.polylines(output, [corners_int], True, obj_info['color'], 2)
            
            # Draw center
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(output, center_int, 4, obj_info['color'], -1)
            
            # Draw 3D axes if requested
            if draw_axes:
                cv2.drawFrameAxes(
                    output,
                    self.camera_matrix,
                    self.dist_coeffs,
                    det['rvec'],
                    det['tvec'],
                    self.marker_size_m * 0.5
                )
            
            # Draw information text
            if draw_info:
                text_lines = [
                    f"ID: {marker_id}",
                    f"Type: {obj_info['type']}",
                    f"Dist: {distance*1000:.0f}mm"
                ]
                
                if obj_info['destination']:
                    text_lines.append(f"-> {obj_info['destination']}")
                
                y_offset = center_int[1] - 60
                for line in text_lines:
                    cv2.putText(output, line, (center_int[0] + 15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_info['color'], 2)
                    y_offset += 20
        
        return output
    
    def estimate_distance_from_area(self, area, reference_area=10000, reference_distance=500):
        """
        Estimate distance based on marker area (fallback if pose estimation fails)
        
        Args:
            area: Marker area in pixels
            reference_area: Area of marker at reference distance
            reference_distance: Reference distance in mm
        
        Returns:
            Estimated distance in mm
        """
        if area <= 0:
            return float('inf')
        
        # Inverse square relationship
        distance = reference_distance * np.sqrt(reference_area / area)
        return distance


class CameraCalibrator:
    """
    Helper class for camera calibration using checkerboard pattern
    Run this once to get accurate camera parameters for your C922
    """
    
    def __init__(self, checkerboard_size=(9, 6), square_size_mm=25.0):
        """
        Args:
            checkerboard_size: Internal corners (cols, rows)
            square_size_mm: Size of each square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
    
    def add_calibration_image(self, frame):
        """
        Add a calibration image for camera calibration
        
        Args:
            frame: Image containing checkerboard pattern
        
        Returns:
            True if pattern found, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_refined)
            
            return True
        
        return False
    
    def calibrate(self, frame_size=(640, 360)):
        """
        Perform camera calibration
        
        Args:
            frame_size: Image size (width, height)
        
        Returns:
            (camera_matrix, dist_coeffs, calibration_error)
        """
        if len(self.objpoints) < 10:
            raise ValueError("Need at least 10 calibration images")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            frame_size,
            None,
            None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i],
                                              camera_matrix, dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        return camera_matrix, dist_coeffs, mean_error


# Example usage
if __name__ == '__main__':
    # Initialize detector
    detector = ArucoObjectDetector()
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    print("ArUco Object Detector Test")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        detections = detector.detect_markers(frame)
        
        # Draw detections
        output = detector.draw_detections(frame, detections)
        
        # Get objects to collect
        objects = detector.get_objects_to_collect(detections)
        if objects:
            print(f"Found {len(objects)} objects to collect")
            nearest = detector.find_nearest_object(detections)
            if nearest:
                print(f"  Nearest: {nearest['object_info']['type']} "
                      f"at {nearest['distance']*1000:.0f}mm")
        
        cv2.imshow('ArUco Detection', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()