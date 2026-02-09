#!/usr/bin/env python3
"""
Standalone test script for vision system
Tests camera, ORB features, grid division, and ArUco detection
Run this first to verify OpenCV is working before using ROS2
"""

import cv2
import numpy as np
import time
from collections import deque


class VisionTester:
    def __init__(self):
        print("=" * 60)
        print("Vision System Standalone Test")
        print("=" * 60)
        
        # Configuration
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 360
        self.fps = 30
        self.grid_rows = 8
        self.grid_cols = 8
        self.orb_max_features = 500
        
        # Initialize camera
        print("\n1. Initializing camera...")
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        
        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Update dimensions to actual values
        self.frame_width = int(actual_width)
        self.frame_height = int(actual_height)
        
        print(f"   ✓ Camera opened successfully")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS: {int(actual_fps)}")
        
        # Initialize ORB
        print("\n2. Initializing ORB detector...")
        self.orb = cv2.ORB_create(nfeatures=self.orb_max_features)
        print(f"   ✓ ORB detector created (max features: {self.orb_max_features})")
        
        # Initialize ArUco
        print("\n3. Initializing ArUco detector...")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        print(f"   ✓ ArUco detector created (DICT_4X4_50)")
        
        # Grid parameters - calculate using actual resolution
        self.cell_width = self.frame_width // self.grid_cols
        self.cell_height = self.frame_height // self.grid_rows
        print(f"\n4. Grid configuration: {self.grid_rows}x{self.grid_cols}")
        print(f"   Cell size: {self.cell_width}x{self.cell_height} pixels")
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.feature_history = deque(maxlen=30)
        
        print("\n" + "=" * 60)
        print("Test ready!")
        print("=" * 60)
        print("\nControls:")
        print("  'q' or 'Q' or ESC - Quit")
        print("  'g' or 'G' - Toggle grid display")
        print("  'f' or 'F' - Toggle feature display")
        print("  'a' or 'A' - Toggle ArUco display")
        print("  's' or 'S' - Save screenshot")
        print("  Close window (X button) - Also quits")
        print("\n⚠️  IMPORTANT: Click on the OpenCV window to ensure it has focus!")
        print("    If keys don't work, try clicking the window first.\n")
        print()
        
        # Display flags
        self.show_grid = True
        self.show_features = True
        self.show_aruco = True
        
    def compute_grid_features(self, keypoints):
        """Compute feature count per grid cell"""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            col = min(x // self.cell_width, self.grid_cols - 1)
            row = min(y // self.cell_height, self.grid_rows - 1)
            grid[row, col] += 1
        
        return grid
    
    def detect_aruco(self, frame):
        """Detect ArUco markers"""
        corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        return corners, ids
    
    def draw_visualization(self, frame, keypoints, grid, aruco_corners, aruco_ids):
        """Draw all visualizations on frame"""
        vis = frame.copy()
        
        # Draw grid
        if self.show_grid:
            for i in range(1, self.grid_rows):
                y = i * self.cell_height
                cv2.line(vis, (0, y), (self.frame_width, y), (100, 100, 100), 1)
            for j in range(1, self.grid_cols):
                x = j * self.cell_width
                cv2.line(vis, (x, 0), (x, self.frame_height), (100, 100, 100), 1)
            
            # Draw feature counts
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    count = grid[row, col]
                    if count > 0:
                        x = col * self.cell_width + self.cell_width // 2
                        y = row * self.cell_height + self.cell_height // 2
                        
                        # Color based on count
                        intensity = int(min(255, count * 30))
                        cv2.circle(vis, (x, y), 12, (0, intensity, 0), -1)
                        cv2.putText(vis, str(count), (x-8, y+5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw ORB features
        if self.show_features:
            cv2.drawKeypoints(vis, keypoints, vis,
                            color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Draw ArUco markers
        if self.show_aruco and aruco_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, aruco_corners, aruco_ids)
            
            # Draw additional info
            for i, corner in enumerate(aruco_corners):
                marker_id = aruco_ids[i][0]
                center = corner[0].mean(axis=0)
                
                cv2.putText(vis, f"ID: {marker_id}",
                           (int(center[0]) + 15, int(center[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis
    
    def draw_stats(self, frame, num_features, num_aruco, fps):
        """Draw statistics overlay"""
        # Background for stats
        cv2.rectangle(frame, (5, 5), (250, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, 110), (255, 255, 255), 2)
        
        # Stats text
        stats = [
            f"FPS: {fps:.1f}",
            f"Features: {num_features}",
            f"ArUco: {num_aruco}",
            f"Grid: {self.grid_rows}x{self.grid_cols}"
        ]
        
        y = 25
        for stat in stats:
            cv2.putText(frame, stat, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 22
        
        # Status indicators
        status_y = self.frame_height - 10
        if self.show_grid:
            cv2.putText(frame, "[G]rid", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if self.show_features:
            cv2.putText(frame, "[F]eatures", (70, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if self.show_aruco:
            cv2.putText(frame, "[A]rUco", (160, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def run(self):
        """Main test loop"""
        frame_count = 0
        
        # Create window and try to give it focus
        cv2.namedWindow('Vision Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vision Test', 800, 600)
        
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame!")
                break
            
            # Convert to grayscale for ORB
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB features
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            # Compute grid features
            grid = self.compute_grid_features(keypoints)
            
            # Detect ArUco markers
            aruco_corners, aruco_ids = self.detect_aruco(frame)
            
            # Draw visualization
            vis_frame = self.draw_visualization(frame, keypoints, grid, 
                                                aruco_corners, aruco_ids)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)
            
            # Track features
            self.feature_history.append(len(keypoints))
            
            # Draw stats
            num_aruco = len(aruco_ids) if aruco_ids is not None else 0
            self.draw_stats(vis_frame, len(keypoints), num_aruco, avg_fps)
            
            # Display
            cv2.imshow('Vision Test', vis_frame)
            
            # Print periodic stats
            frame_count += 1
            if frame_count % 30 == 0:
                avg_features = np.mean(self.feature_history)
                print(f"Frame {frame_count:4d} | "
                      f"FPS: {avg_fps:5.1f} | "
                      f"Features: {avg_features:5.1f} | "
                      f"ArUco: {num_aruco}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Multiple ways to quit
            if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
                print("\nQuitting...")
                break
            elif key == ord('g') or key == ord('G'):
                self.show_grid = not self.show_grid
                print(f"Grid display: {'ON' if self.show_grid else 'OFF'}")
            elif key == ord('f') or key == ord('F'):
                self.show_features = not self.show_features
                print(f"Feature display: {'ON' if self.show_features else 'OFF'}")
            elif key == ord('a') or key == ord('A'):
                self.show_aruco = not self.show_aruco
                print(f"ArUco display: {'ON' if self.show_aruco else 'OFF'}")
            elif key == ord('s') or key == ord('S'):
                filename = f"screenshot_{frame_count}.png"
                cv2.imwrite(filename, vis_frame)
                print(f"Saved {filename}")
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty('Vision Test', cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed, quitting...")
                    break
            except:
                # Window doesn't exist anymore
                print("\nWindow closed, quitting...")
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}")
        print(f"Average features detected: {np.mean(self.feature_history):.2f}")
        print(f"Min features: {np.min(self.feature_history):.0f}")
        print(f"Max features: {np.max(self.feature_history):.0f}")
        print("=" * 60)


def main():
    try:
        tester = VisionTester()
        tester.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
