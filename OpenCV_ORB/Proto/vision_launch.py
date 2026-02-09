#!/usr/bin/env python3
"""
ROS2 Launch file for Vision System
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for vision system"""
    
    # Declare launch arguments
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Camera device index'
    )
    
    frame_width_arg = DeclareLaunchArgument(
        'frame_width',
        default_value='640',
        description='Frame width'
    )
    
    frame_height_arg = DeclareLaunchArgument(
        'frame_height',
        default_value='360',
        description='Frame height'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Camera FPS'
    )
    
    grid_rows_arg = DeclareLaunchArgument(
        'grid_rows',
        default_value='8',
        description='Number of grid rows'
    )
    
    grid_cols_arg = DeclareLaunchArgument(
        'grid_cols',
        default_value='8',
        description='Number of grid columns'
    )
    
    orb_features_arg = DeclareLaunchArgument(
        'orb_features',
        default_value='500',
        description='Maximum ORB features to detect'
    )
    
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='false',
        description='Publish debug visualization images'
    )
    
    # Vision node
    vision_node = Node(
        package='taskbot_vision',  # Replace with your package name
        executable='vision_node',
        name='vision_node',
        output='screen',
        parameters=[{
            'camera_index': LaunchConfiguration('camera_index'),
            'frame_width': LaunchConfiguration('frame_width'),
            'frame_height': LaunchConfiguration('frame_height'),
            'fps': LaunchConfiguration('fps'),
            'grid_rows': LaunchConfiguration('grid_rows'),
            'grid_cols': LaunchConfiguration('grid_cols'),
            'orb_features': LaunchConfiguration('orb_features'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
        }]
    )
    
    return LaunchDescription([
        camera_index_arg,
        frame_width_arg,
        frame_height_arg,
        fps_arg,
        grid_rows_arg,
        grid_cols_arg,
        orb_features_arg,
        publish_debug_arg,
        vision_node,
    ])
