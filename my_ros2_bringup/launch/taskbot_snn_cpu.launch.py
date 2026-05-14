from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():

    # Parse params.yaml with PyYAML (supports anchors/aliases) and extract
    # per-node dicts so rcl never sees raw YAML anchors.
    _params_file = os.path.join(
        get_package_share_directory('my_ros2_bringup'), 'config', 'params.yaml')
    with open(_params_file) as f:
        _all = yaml.safe_load(f)

    def p(node_name):
        return _all.get(node_name, {}).get('ros__parameters', {})
    
    # Path to the camera config file inside the ROS2 package
    camera_config = PathJoinSubstitution([
        FindPackageShare("robot_camera_config"),
        "config",
        "c922.yaml"
    ])

    return LaunchDescription([

        DeclareLaunchArgument('motor_control', default_value='true'),

        # Camera node with link to .yaml config file for camera parameters
#        Node(
#            package="v4l2_camera",
#            executable="v4l2_camera_node",
#            name="c922_camera",
#            namespace="camera",
#            parameters=[
#                camera_config,
#                {
#                "camera_name": "c922",
#                "camera_info_url": "file:///opt/robot_ws/install/robot_camera_config/share/robot_camera_config/config/c922_camera_info.yaml",
#                }],
#            remappings=[("image_raw", "image_raw")]
#        ),

        # Static TF base_link → camera_link
#        Node(
#            package='tf2_ros',
#            executable='static_transform_publisher',
#            name='camera_tf',
#            arguments=[
#                "--x", "0.10", "--y", "0.0", "--z", "0.09",
#                "--qx", "0.0", "--qy", "0.0", "--qz", "0.0", "--qw", "1.0",
#                "--frame-id", "base_link", "--child-frame-id", "camera_link"
#            ],
#            output='screen'
#        ),

        # Proximity adapter node
#        Node(
#            package='distance_sensor',
#            executable='proximity_adapter_node',
#            name='proximity_adapter_node',
#            output='screen'
#        ),

        # Distance sensor node
#        Node(
#            package='distance_sensor',
#            executable='distance_sensor_node',
#            name='distance_sensor_node',
#            output='screen'
#        ),
    
        # Motor control node
#        Node(
#            package='motor_control',
#            executable='motor_control_node',
#            name='motor_control_node',
#            output='screen',
#            parameters=[p('motor_control_node')],
#            condition=IfCondition(LaunchConfiguration('motor_control'))
#        ),

        # Gripper node
#        Node(
#            package='motor_control',
#            executable='gripper_node',
#            name='gripper_node',
#            output='screen',
#            condition=IfCondition(LaunchConfiguration('motor_control'))
#        ),

        # Grab node
#        Node(
#           package='grab_node',
#           executable='grab_node',
#           name='grab_node',
#           parameters=[p('grab_node')],
#           output='screen'
#        ),

        # Grab node / prox_node
#        Node(
#           package='grab_node',
#           executable='prox_node',
#           name='prox_node',
#           parameters=[p('prox_node')],
#           output='screen'
#        ),

        # Command arbiter node
#        Node(
#            package='cmd_arbiter',
#            executable='cmd_arbiter',
#            name='cmd_arbiter',
#            output='screen',
#            parameters=[p('cmd_arbiter')],
#        ),

        # Encoding node
#        Node(
#            package='encoding_node',
#            executable='encoding_node',
#            name='encoding_node',
#            output='screen',
#            parameters=[p('encoding_node')],
#        ),

        # OpenCV keypoint grid node
#        Node(
#            package='opencv_nodes',
#            executable='img_kp_grid',
#            name='img_kp_grid',
#            output='screen',
#            parameters=[p('img_kp_grid')],
#        ),

        # OpenCV image recognition node
#        Node(
#            package='opencv_nodes',
#            executable='img_recog',
#            name='img_recog',
#            output='screen',
#            parameters=[p('img_recog')],
#        ),

        # Emergency stop node based on distance sensor
#        Node(
#            package='proximity_stop',
#            executable='proximity_stop_node',
#            name='proximity_stop',
#            output='screen',
#            parameters=[p('proximity_stop')],
#        ),

        # Python SNN node
        Node(
            package='python_snn_node',
            executable='snn_node',
            name='python_snn_node',
            output='screen',
            parameters=[p('python_snn_node')],
        ),

        # Task manager node to coordinate
#        Node(
#            package='task_manager',
#            executable='task_manager',
#            name='task_manager',
#            output='screen'
#        ),

        # Power monitor node
        Node(
            package='power_monitor',
            executable='system_power_node',
            name='system_power_node',
            output='screen'
        ),

        Node(
            package='power_monitor',
            executable='fpga_power_node',
            name='fpga_power_node',
            output='screen'
        ),

        Node(
            package='power_monitor',
            executable='snn_logger_node',
            name='snn_logger_node',
            output='screen'
        ),

#        Node(
#            package='dopamine_reward_node',
#            executable='dopamine_reward_node',
#            name='dopamine_reward_node',
#            output='screen'
#        ),

        # Spike train publisher (synthetic, deterministic SNN workload)
        Node(
            package='encoding_node',
            executable='spike_train_publisher',
            name='spike_train_publisher',
            output='screen',
            parameters=[p('spike_train_publisher')],
        ),

    ])
