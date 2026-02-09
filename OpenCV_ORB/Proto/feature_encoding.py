#!/usr/bin/env python3
"""
Feature Encoding Utilities for SNN/FPGA Processing
Provides different encoding schemes for converting visual features to spike patterns
"""

import numpy as np


class GridFeatureEncoder:
    """
    Encodes grid-based visual features for neuromorphic processing
    Supports multiple encoding schemes optimized for LIF SNNs
    """
    
    def __init__(self, grid_rows=8, grid_cols=8, frame_width=640, frame_height=360):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.cell_width = frame_width // grid_cols
        self.cell_height = frame_height // grid_rows
        
        # Precompute spatial encodings
        self._compute_spatial_maps()
    
    def _compute_spatial_maps(self):
        """Precompute spatial relationship maps for efficient encoding"""
        # Center bias map (for navigation - center is important)
        self.center_bias = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        center_row, center_col = self.grid_rows // 2, self.grid_cols // 2
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
                max_dist = np.sqrt(center_row**2 + center_col**2)
                self.center_bias[row, col] = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        
        # Directional maps (left, right, top, bottom regions)
        self.direction_maps = {
            'left': np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32),
            'right': np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32),
            'top': np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32),
            'bottom': np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        }
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Left to right gradient
                self.direction_maps['left'][row, col] = 1.0 - (col / (self.grid_cols - 1))
                self.direction_maps['right'][row, col] = col / (self.grid_cols - 1)
                
                # Top to bottom gradient
                self.direction_maps['top'][row, col] = 1.0 - (row / (self.grid_rows - 1))
                self.direction_maps['bottom'][row, col] = row / (self.grid_rows - 1)
    
    def encode_rate_based(self, feature_grid, normalize=True, max_features=50):
        """
        Rate-based encoding: feature count maps to spike rate
        
        Args:
            feature_grid: 2D array of feature counts per cell
            normalize: Normalize to [0, 1] range
            max_features: Expected maximum features per cell (for normalization)
        
        Returns:
            Flattened array suitable for rate-based SNN input
        """
        grid = feature_grid.astype(np.float32)
        
        if normalize:
            grid = np.clip(grid / max_features, 0.0, 1.0)
        
        return grid.flatten()
    
    def encode_population(self, feature_grid, num_neurons_per_cell=4):
        """
        Population encoding: each cell maps to multiple neurons
        Useful for encoding both feature density and spatial information
        
        Args:
            feature_grid: 2D array of feature counts per cell
            num_neurons_per_cell: Number of neurons to encode each cell
        
        Returns:
            Flattened array with population-coded features
        """
        total_neurons = self.grid_rows * self.grid_cols * num_neurons_per_cell
        encoded = np.zeros(total_neurons, dtype=np.float32)
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                base_idx = (row * self.grid_cols + col) * num_neurons_per_cell
                count = feature_grid[row, col]
                
                # Encode count across population
                if count > 0:
                    # Different neurons respond to different count ranges
                    thresholds = [1, 5, 10, 20]  # Example for 4 neurons
                    for i, thresh in enumerate(thresholds[:num_neurons_per_cell]):
                        encoded[base_idx + i] = 1.0 if count >= thresh else count / thresh
        
        return encoded
    
    def encode_directional(self, feature_grid, normalize=True):
        """
        Directional encoding: provides separate channels for each direction
        Useful for navigation - tells the SNN where features are concentrated
        
        Args:
            feature_grid: 2D array of feature counts per cell
            normalize: Normalize feature counts
        
        Returns:
            Dictionary with directional feature sums
        """
        grid = feature_grid.astype(np.float32)
        if normalize:
            grid = grid / (np.max(grid) + 1e-6)
        
        # Weight features by their directional position
        directional_features = {}
        for direction, weight_map in self.direction_maps.items():
            directional_features[direction] = np.sum(grid * weight_map)
        
        # Add center-weighted total
        directional_features['center'] = np.sum(grid * self.center_bias)
        
        return directional_features
    
    def encode_for_navigation(self, feature_grid, target_position=None):
        """
        Navigation-specific encoding combining multiple strategies
        
        Args:
            feature_grid: 2D array of feature counts per cell
            target_position: Optional (x, y) target in frame coordinates
        
        Returns:
            Comprehensive encoding for navigation decision-making
        """
        # Basic rate encoding
        rate_encoded = self.encode_rate_based(feature_grid)
        
        # Directional features
        directional = self.encode_directional(feature_grid)
        
        # Center-weighted features (for obstacle avoidance)
        center_features = feature_grid * self.center_bias
        
        # Target-biased encoding if target is provided
        target_bias = np.zeros_like(feature_grid, dtype=np.float32)
        if target_position is not None:
            tx, ty = target_position
            target_col = min(int(tx / self.cell_width), self.grid_cols - 1)
            target_row = min(int(ty / self.cell_height), self.grid_rows - 1)
            
            # Create gradient toward target
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    dist = np.sqrt((row - target_row)**2 + (col - target_col)**2)
                    max_dist = np.sqrt(self.grid_rows**2 + self.grid_cols**2)
                    target_bias[row, col] = 1.0 - (dist / max_dist)
        
        return {
            'rate_encoded': rate_encoded,
            'directional': directional,
            'center_weighted': center_features.flatten(),
            'target_bias': target_bias.flatten(),
            'raw_grid': feature_grid.flatten()
        }
    
    def create_snn_input_vector(self, feature_grid, encoding_type='rate', **kwargs):
        """
        Create input vector for FPGA SNN based on encoding type
        
        Args:
            feature_grid: 2D array of feature counts
            encoding_type: 'rate', 'population', 'directional', or 'navigation'
            **kwargs: Additional parameters for specific encodings
        
        Returns:
            1D numpy array ready for FPGA transmission
        """
        if encoding_type == 'rate':
            return self.encode_rate_based(feature_grid, **kwargs)
        elif encoding_type == 'population':
            return self.encode_population(feature_grid, **kwargs)
        elif encoding_type == 'directional':
            dir_features = self.encode_directional(feature_grid, **kwargs)
            # Flatten directional features to array
            return np.array(list(dir_features.values()), dtype=np.float32)
        elif encoding_type == 'navigation':
            nav_features = self.encode_for_navigation(feature_grid, **kwargs)
            # Concatenate all navigation features
            return np.concatenate([
                nav_features['rate_encoded'],
                np.array(list(nav_features['directional'].values())),
                nav_features['center_weighted']
            ])
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")


class ObstacleDetector:
    """
    Detects potential obstacles based on feature distribution
    High feature density in close proximity suggests obstacles
    """
    
    def __init__(self, grid_rows=8, grid_cols=8):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
    def detect_obstacles(self, feature_grid, threshold=10):
        """
        Identify cells that likely contain obstacles
        
        Args:
            feature_grid: 2D array of feature counts
            threshold: Minimum features to consider as obstacle
        
        Returns:
            Binary obstacle map (1 = obstacle, 0 = clear)
        """
        obstacle_map = (feature_grid >= threshold).astype(np.int32)
        return obstacle_map
    
    def detect_clear_path(self, feature_grid, low_threshold=5, high_threshold=15):
        """
        Detect regions with moderate features (navigable space)
        Too few = featureless (might be void/edge), too many = obstacle
        
        Args:
            feature_grid: 2D array of feature counts
            low_threshold: Minimum features for navigable space
            high_threshold: Maximum features before considering obstacle
        
        Returns:
            Binary navigable map (1 = navigable, 0 = avoid)
        """
        navigable = ((feature_grid >= low_threshold) & 
                     (feature_grid <= high_threshold)).astype(np.int32)
        return navigable
    
    def get_navigation_direction(self, feature_grid):
        """
        Suggest navigation direction based on feature distribution
        
        Returns:
            Direction string ('left', 'right', 'forward', 'backward')
        """
        # Split into vertical thirds
        left_third = feature_grid[:, :self.grid_cols//3]
        center_third = feature_grid[:, self.grid_cols//3:2*self.grid_cols//3]
        right_third = feature_grid[:, 2*self.grid_cols//3:]
        
        # Count features in each region
        left_features = np.sum(left_third)
        center_features = np.sum(center_third)
        right_features = np.sum(right_third)
        
        # Lower features = more free space (in general)
        if left_features < min(center_features, right_features):
            return 'left'
        elif right_features < min(center_features, left_features):
            return 'right'
        elif center_features < min(left_features, right_features):
            return 'forward'
        else:
            return 'backward'


def normalize_grid(feature_grid, method='minmax'):
    """
    Normalize feature grid for consistent SNN input
    
    Args:
        feature_grid: 2D array of feature counts
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized grid
    """
    if method == 'minmax':
        min_val = np.min(feature_grid)
        max_val = np.max(feature_grid)
        if max_val > min_val:
            return (feature_grid - min_val) / (max_val - min_val)
        else:
            return feature_grid
    elif method == 'zscore':
        mean = np.mean(feature_grid)
        std = np.std(feature_grid)
        if std > 0:
            return (feature_grid - mean) / std
        else:
            return feature_grid - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Example usage
if __name__ == '__main__':
    # Test the encoding
    encoder = GridFeatureEncoder(grid_rows=8, grid_cols=8)
    
    # Simulate a feature grid
    test_grid = np.random.randint(0, 30, size=(8, 8))
    
    print("Test Grid:")
    print(test_grid)
    print("\nRate-based encoding:")
    print(encoder.encode_rate_based(test_grid))
    
    print("\nDirectional encoding:")
    print(encoder.encode_directional(test_grid))
    
    print("\nNavigation encoding with target at (320, 180):")
    nav_encoding = encoder.encode_for_navigation(test_grid, target_position=(320, 180))
    for key, value in nav_encoding.items():
        print(f"{key}: {value.shape}")
