from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():

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

        # Camera node with link to .yaml config file for camera parameters
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="c922_camera",
            namespace="camera",
            parameters=[camera_config],
            remappings=[("image_raw", "image_raw")]
        ),

        # Static TF base_link → camera_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf',
            arguments=[
                "--x", "0.10", "--y", "0.0", "--z", "0.09",
                "--qx", "0.0", "--qy", "0.0", "--qz", "0.0", "--qw", "1.0",
                "--frame-id", "base_link", "--child-frame-id", "camera_link"
            ],
            output='screen'
        ),

        # Proximity adapter node
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
            parameters=[p('motor_control_node')],
        ),

        # Gripper node
        Node(
            package='motor_control',
            executable='gripper_node',
            name='gripper_node',
            output='screen'
        ),

        # Grab node
        # Node(
        #    package='grab_node',
        #    executable='grab_node',
        #    name='grab_node',
        #    parameters=[p('grab_node')],
        #    output='screen'
        #),

        # Command arbiter node
        # Node(
        #     package='cmd_arbiter',
        #     executable='cmd_arbiter',
        #     name='cmd_arbiter',
        #     parameters=[p('cmd_arbiter')],
        #     output='screen'
        # ),

        # Encoding node
        Node(
            package='encoding_node',
            executable='encoding_node',
            name='encoding_node',
            output='screen',
            parameters=[p('encoding_node')],
        ),

        # OpenCV keypoint grid node
        Node(
            package='opencv_nodes',
            executable='img_kp_grid',
            name='img_kp_grid',
            output='screen',
            parameters=[p('img_kp_grid')],
        ),

        # OpenCV image recognition node
        Node(
            package='opencv_nodes',
            executable='img_recog',
            name='img_recog',
            output='screen',
            parameters=[p('img_recog')],
        ),

        # Emergency stop node based on distance sensor
        # Node(
        #    package='proximity_stop',
        #    executable='proximity_stop_node',
        #    name='proximity_stop',
        #    parameters=[p('proximity_stop')],
        #    output='screen'
        #),

        # Python SNN node
        # Node(
        #     package='python_snn_node',
        #     executable='snn_node',
        #     name='python_snn_node',
        #     parameters=[p('python_snn_node')],
        #     output='screen'
        # ),

        # Task manager node to coordinate
        Node(
            package='task_manager',
            executable='task_manager',
            name='task_manager',
            output='screen'
        ),

    ])
