#!/usr/bin/env python3
"""
Occupancy Grid Generator
Creates 2D occupancy grid from ORB features and ArUco markers with 3-bit encoding

Encoding:
- 000 (0): Not detected / Unknown (dark grey)
- 001 (1): Free space without targets (light grey/white)
- 011 (3): Target object - pickup (green)
- 111 (7): Target object - drop-off zone (blue)
"""

import numpy as np
import cv2


class OccupancyGridGenerator:
    """
    Generates 2D occupancy grid from vision data
    """
    
    def __init__(self, grid_rows=8, grid_cols=8, frame_width=640, frame_height=480):
        """
        Args:
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
            frame_width: Camera frame width
            frame_height: Camera frame height
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.cell_width = frame_width // grid_cols
        self.cell_height = frame_height // grid_rows
        
        # Occupancy values (3-bit encoding)
        self.UNKNOWN = 0      # 000 - Not detected
        self.FREE = 1         # 001 - Free space
        self.PICKUP = 3       # 011 - Target to pickup
        self.DROPOFF = 7      # 111 - Drop-off zone
        
        # Feature thresholds for free space detection
        self.min_features_free = 3    # Minimum features to consider "detected"
        self.max_features_free = 25   # Maximum features to consider "free" (not obstacle)
    
    def generate_occupancy_grid(self, feature_grid, aruco_detections=None):
        """
        Generate occupancy grid from features and ArUco markers
        
        Args:
            feature_grid: 2D numpy array (grid_rows x grid_cols) with feature counts
            aruco_detections: List of ArUco detections with object info
                             Each detection: {'id', 'center', 'object_info', ...}
        
        Returns:
            2D numpy array (grid_rows x grid_cols) with 3-bit occupancy values
        """
        # Initialize grid as unknown
        occupancy = np.full((self.grid_rows, self.grid_cols), self.UNKNOWN, dtype=np.uint8)
        
        # Step 1: Mark cells based on feature density
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                feature_count = feature_grid[row, col]
                
                if feature_count >= self.min_features_free and feature_count <= self.max_features_free:
                    # Detected and likely free space
                    occupancy[row, col] = self.FREE
                elif feature_count > self.max_features_free:
                    # Too many features - might be obstacle or dense area
                    # Keep as UNKNOWN for now
                    occupancy[row, col] = self.UNKNOWN
                # else: feature_count < min_features_free -> stays UNKNOWN
        
        # Step 2: Mark ArUco marker cells
        if aruco_detections:
            for detection in aruco_detections:
                center_x, center_y = detection['center']
                obj_info = detection['object_info']
                
                # Calculate grid cell for marker
                col = int(center_x / self.cell_width)
                row = int(center_y / self.cell_height)
                
                # Ensure within bounds
                col = min(max(col, 0), self.grid_cols - 1)
                row = min(max(row, 0), self.grid_rows - 1)
                
                # Determine occupancy type based on object
                if obj_info['destination'] is not None:
                    # Object that needs to be picked up
                    occupancy[row, col] = self.PICKUP
                elif obj_info['type'].startswith('zone_'):
                    # Drop-off zone
                    occupancy[row, col] = self.DROPOFF
        
        return occupancy
    
    def occupancy_to_costmap(self, occupancy_grid, unknown_cost=50):
        """
        Convert occupancy grid to navigation costmap
        
        Args:
            occupancy_grid: 2D array with 3-bit occupancy values
            unknown_cost: Cost for unknown cells (0-100)
        
        Returns:
            2D array with costs (0-100) for navigation
        """
        costmap = np.zeros_like(occupancy_grid, dtype=np.uint8)
        
        costmap[occupancy_grid == self.UNKNOWN] = unknown_cost  # Unknown - moderate cost
        costmap[occupancy_grid == self.FREE] = 0                # Free - no cost
        costmap[occupancy_grid == self.PICKUP] = 0              # Target - attractive (low cost)
        costmap[occupancy_grid == self.DROPOFF] = 0             # Zone - attractive (low cost)
        
        return costmap
    
    def visualize_occupancy_grid(self, occupancy_grid, cell_size=50):
        """
        Create visualization of occupancy grid
        
        Args:
            occupancy_grid: 2D array with occupancy values
            cell_size: Size of each cell in pixels for visualization
        
        Returns:
            BGR image for display
        """
        # Create image
        img_height = self.grid_rows * cell_size
        img_width = self.grid_cols * cell_size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Color mapping (BGR format)
        colors = {
            self.UNKNOWN: (64, 64, 64),      # Dark grey
            self.FREE: (240, 240, 240),      # Light grey/white
            self.PICKUP: (0, 255, 0),        # Green
            self.DROPOFF: (255, 0, 0)        # Blue
        }
        
        # Fill cells
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                occupancy = occupancy_grid[row, col]
                color = colors.get(occupancy, (128, 128, 128))  # Default grey
                
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Draw grid lines
        for i in range(self.grid_rows + 1):
            y = i * cell_size
            cv2.line(img, (0, y), (img_width, y), (0, 0, 0), 1)
        for j in range(self.grid_cols + 1):
            x = j * cell_size
            cv2.line(img, (x, 0), (x, img_height), (0, 0, 0), 1)
        
        # Add labels
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                occupancy = occupancy_grid[row, col]
                
                x = col * cell_size + cell_size // 2 - 15
                y = row * cell_size + cell_size // 2 + 5
                
                # Show binary representation
                binary = f"{occupancy:03b}"
                cv2.putText(img, binary, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add legend
        legend_height = 120
        legend = np.ones((legend_height, img_width, 3), dtype=np.uint8) * 255
        
        legend_items = [
            (self.UNKNOWN, "000: Unknown", (64, 64, 64)),
            (self.FREE, "001: Free Space", (240, 240, 240)),
            (self.PICKUP, "011: Pickup Target", (0, 255, 0)),
            (self.DROPOFF, "111: Drop-off Zone", (255, 0, 0))
        ]
        
        y_offset = 20
        for _, label, color in legend_items:
            # Color box
            cv2.rectangle(legend, (10, y_offset), (30, y_offset + 15), color, -1)
            cv2.rectangle(legend, (10, y_offset), (30, y_offset + 15), (0, 0, 0), 1)
            
            # Label
            cv2.putText(legend, label, (40, y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 25
        
        # Combine grid and legend
        result = np.vstack([img, legend])
        
        return result
    
    def get_navigable_cells(self, occupancy_grid):
        """
        Get list of navigable (free) cells
        
        Args:
            occupancy_grid: 2D array with occupancy values
        
        Returns:
            List of (row, col) tuples for navigable cells
        """
        navigable = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if occupancy_grid[row, col] == self.FREE:
                    navigable.append((row, col))
        return navigable
    
    def get_target_cells(self, occupancy_grid, target_type='pickup'):
        """
        Get cells containing targets
        
        Args:
            occupancy_grid: 2D array with occupancy values
            target_type: 'pickup' or 'dropoff'
        
        Returns:
            List of (row, col) tuples for target cells
        """
        if target_type == 'pickup':
            target_value = self.PICKUP
        elif target_type == 'dropoff':
            target_value = self.DROPOFF
        else:
            return []
        
        targets = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if occupancy_grid[row, col] == target_value:
                    targets.append((row, col))
        return targets
    
    def find_nearest_target(self, occupancy_grid, current_pos, target_type='pickup'):
        """
        Find nearest target cell from current position
        
        Args:
            occupancy_grid: 2D array with occupancy values
            current_pos: (row, col) tuple of current position
            target_type: 'pickup' or 'dropoff'
        
        Returns:
            (row, col) tuple of nearest target, or None
        """
        targets = self.get_target_cells(occupancy_grid, target_type)
        
        if not targets:
            return None
        
        current_row, current_col = current_pos
        
        # Find nearest using Manhattan distance
        min_dist = float('inf')
        nearest = None
        
        for target_row, target_col in targets:
            dist = abs(target_row - current_row) + abs(target_col - current_col)
            if dist < min_dist:
                min_dist = dist
                nearest = (target_row, target_col)
        
        return nearest
    
    def encode_for_transmission(self, occupancy_grid):
        """
        Encode occupancy grid for efficient transmission to FPGA
        
        Args:
            occupancy_grid: 2D array with occupancy values
        
        Returns:
            Packed byte array
        """
        # Flatten grid
        flat = occupancy_grid.flatten()
        
        # Pack 3-bit values efficiently (not fully optimized, but clear)
        # For 64 cells (8x8), we need 64 * 3 = 192 bits = 24 bytes
        
        return flat.astype(np.uint8).tobytes()
    
    def decode_from_transmission(self, data):
        """
        Decode occupancy grid from transmitted data
        
        Args:
            data: Byte array
        
        Returns:
            2D occupancy grid
        """
        flat = np.frombuffer(data, dtype=np.uint8)
        grid = flat.reshape((self.grid_rows, self.grid_cols))
        return grid


# Example usage and testing
if __name__ == '__main__':
    import time
    
    print("Occupancy Grid Generator Test")
    print("=" * 60)
    
    # Create generator
    generator = OccupancyGridGenerator(grid_rows=8, grid_cols=8)
    
    # Simulate feature grid
    feature_grid = np.array([
        [0,  2,  5,  8, 10, 12,  8,  3],
        [1,  5, 10, 15, 18, 15, 10,  5],
        [2,  8, 12, 20, 22, 18, 12,  6],
        [0,  5, 10, 15, 18, 15, 10,  4],
        [1,  4,  8, 12, 15, 12,  8,  3],
        [0,  3,  6, 10, 12, 10,  6,  2],
        [1,  2,  4,  6,  8,  6,  4,  1],
        [0,  1,  2,  3,  4,  3,  2,  0]
    ])
    
    # Simulate ArUco detections
    aruco_detections = [
        {
            'id': 0,
            'center': (100, 100),  # Grid cell (2, 1)
            'object_info': {
                'type': 'box_small',
                'destination': 'zone_a'
            }
        },
        {
            'id': 1,
            'center': (500, 100),  # Grid cell (2, 6)
            'object_info': {
                'type': 'cylinder',
                'destination': 'zone_b'
            }
        },
        {
            'id': 10,
            'center': (320, 270),  # Grid cell (6, 4) - Drop-off zone
            'object_info': {
                'type': 'zone_a',
                'destination': None
            }
        }
    ]
    
    # Generate occupancy grid
    print("\nGenerating occupancy grid...")
    occupancy = generator.generate_occupancy_grid(feature_grid, aruco_detections)
    
    print("\nOccupancy Grid (3-bit values):")
    print(occupancy)
    
    print("\nOccupancy Grid (binary):")
    for row in occupancy:
        binary_row = ' '.join([f"{val:03b}" for val in row])
        print(binary_row)
    
    # Get navigable cells
    navigable = generator.get_navigable_cells(occupancy)
    print(f"\nNavigable cells: {len(navigable)}")
    
    # Get targets
    pickup_targets = generator.get_target_cells(occupancy, 'pickup')
    dropoff_targets = generator.get_target_cells(occupancy, 'dropoff')
    print(f"Pickup targets: {pickup_targets}")
    print(f"Drop-off zones: {dropoff_targets}")
    
    # Find nearest target from position (0, 0)
    nearest = generator.find_nearest_target(occupancy, (0, 0), 'pickup')
    print(f"\nNearest pickup target from (0,0): {nearest}")
    
    # Visualize
    print("\nGenerating visualization...")
    vis = generator.visualize_occupancy_grid(occupancy, cell_size=60)
    
    # Display
    cv2.imshow('Occupancy Grid', vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Test encoding/decoding
    print("\nTesting transmission encoding...")
    encoded = generator.encode_for_transmission(occupancy)
    print(f"Encoded size: {len(encoded)} bytes")
    
    decoded = generator.decode_from_transmission(encoded)
    print(f"Decoding successful: {np.array_equal(occupancy, decoded)}")
    
    print("\n" + "=" * 60)
    print("Test complete!")