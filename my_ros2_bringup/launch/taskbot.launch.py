from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    
    # Path to the camera config file inside the ROS2 package
    camera_config = PathJoinSubstitution([
        FindPackageShare("robot_camera_config"),
        "config",
        "c922.yaml"
    ])


    return LaunchDescription([

        DeclareLaunchArgument('motor_control', default_value='true'),

        # Camera node with link to .yaml config file for camera parameters
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="c922_camera",
            namespace="camera",

            parameters=[camera_config],

            # Remapping so output becomes /camera/image_raw
            remappings=[
                ("image_raw", "image_raw")
            ]
        ),
        
        
        # Static TF base_link → camera_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf',
            arguments=[
                "--x", "0.10",
                "--y", "0.0",
                "--z", "0.09",
                "--qx", "0.0",
                "--qy", "0.0",
                "--qz", "0.0",
                "--qw", "1.0",
                "--frame-id", "base_link",
                "--child-frame-id", "camera_link"
            ],
            output='screen'
        ),

        # Proximety_adapter_node
        Node(
            package='distance_sensor',
            executable='proximity_adapter_node',
            name='proximity_adapter_node',
            output='screen'
        ),    
        
        # Distance sensor node
        Node(
            package='distance_sensor',
            executable='distance_sensor_node',
            name='distance_sensor_node',
            output='screen'
        ),        

        # Motor control node
        Node(
            package='motor_control',
            executable='motor_control_node',
            name='motor_control_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('motor_control'))
        ),

        # Gripper node
        Node(
            package='motor_control',
            executable='gripper_node',
            name='gripper_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('motor_control'))
        ),
        
        # Grab node
        # Node(
        #    package='grab_node',
        #    executable='grab_node',
        #    name='grab_node',
        #    output='screen'
        #),
        
        # Command arbiter node
        Node(
            package='cmd_arbiter',
            executable='cmd_arbiter',
            name='cmd_arbiter',
            output='screen'
        ),

        # Encoding node
        Node(
            package='encoding_node',
            executable='encoding_node',
            name='encoding_node',
            output='screen',
            parameters=[{
            # "proximity_topic": "/ultrasonic/front/scan",
            "output_topic": "/snn/input",
            "proximity_bin_edges": [0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
            "aruco_n_bins": 3,
        }],

        # OpenCD keypoint grid node
        ),
        Node(
            package='opencv_nodes',
            executable='img_kp_grid',
            name='img_kp_grid',
            output='screen',
            parameters=[{
            "response_threshold": 0.0,
            "use_clahe": False,
        }],

        # OpenCV image recognition node
        ),
        Node(
            package='opencv_nodes',
            executable='img_recog',
            name='img_recog',
            output='screen'
        ),

        # Emergency stop node based on distance sensor
        Node(
            package='proximity_stop',
            executable='proximity_stop_node',
            name='proximity_stop',
            output='screen'
        ),

        # Python SNN node
        Node(
            package='python_snn_node',
            executable='snn_node',
            name='python_snn_node',
            output='screen',
            parameters=[
                os.path.join(get_package_share_directory('python_snn_node'), 'config', 'params.yaml'),
                {
                    'log_enable': True,
                    'log_mode': 'A',
                }
            ]
        ),

        # Task manager node to coordinate
        Node(
            package='task_manager',
            executable='task_manager',
            name='task_manager',
            output='screen'
        ),

    ])