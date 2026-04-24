#!/usr/bin/env python3
"""
ROS2 Launch file for monocular ORB-SLAM vision node. 
This launch file starts the vision node with configurable parameters for camera settings, ORB feature detection, and debug visualization.

Launch commands in terminal:
$ ros2 launch taskbot_vision vision_launch.py

To customize parameters, use:
$ ros2 launch taskbot_vision vision_launch.py camera_index:=1 frame_width:=800 frame_height:=600 fps:=60 grid_rows:=10 grid_cols:=10 orb_features:=1000 publish_debug_image:=true
"""

from launch import LaunchDescription                    # To wrap the launch file
from launch_ros.actions import Node                     # To launch a ROS2 node with specified parameters and configurations
from launch.actions import DeclareLaunchArgument        # To make the launch files configurable with command-line arguments
from launch.substitutions import LaunchConfiguration    # To be able to use the launch arguments in node parameters


def generate_launch_description():
    """
    Function needed to generate launch description for the vision node.
    Declare launch arguments below.
    """

    camera_index_arg = DeclareLaunchArgument(
        # To specify which camera device to use (e.g., 0 for default webcam, 1 for external camera)
        'camera_index',
        default_value='0',
        description='Camera device index'
    )
    
    frame_width_arg = DeclareLaunchArgument(
        # To set the width of the camera frames.
        'frame_width',
        default_value='640',
        description='Frame width'
    )
    
    frame_height_arg = DeclareLaunchArgument(
        # To set the height of the camera frames.
        'frame_height',
        default_value='480',
        description='Frame height'
    )
    
    fps_arg = DeclareLaunchArgument(
        # To set the frames per second for the camera capture.
        'fps',
        default_value='30',
        description='Camera FPS'
    )
    
    grid_rows_arg = DeclareLaunchArgument(
        # To set number of grid rows for ORB feature detection grid.
        'grid_rows',
        default_value='8',
        description='Number of grid rows'
    )
    
    grid_cols_arg = DeclareLaunchArgument(
        # To set number of grid columns for ORB feature detection grid.
        'grid_cols',
        default_value='8',
        description='Number of grid columns'
    )
    
    orb_features_arg = DeclareLaunchArgument(
        # To set maximum number of ORB features to detect.
        'orb_features',
        default_value='500',
        description='Maximum ORB features to detect'
    )
    
    publish_debug_arg = DeclareLaunchArgument(
        # To enable or disable publishing of debug visualization images.
        'publish_debug_image',
        default_value='false',
        description='Publish debug visualization images'
    )
    
    # Vision node
    vision_node = Node(
        # Launch the vision node with the specified parameters
        package='taskbot_vision',
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
        # Declare all launch arguments and the vision node
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
