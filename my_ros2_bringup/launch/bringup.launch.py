from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    shared_params = os.path.join(
        get_package_share_directory('my_ros2_bringup'), 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='grab_node',
            executable='grab_node',
            name='grab_node',
            output='screen',
            parameters=[shared_params],
        ),
        Node(
            package='cmd_arbiter',
            executable='cmd_arbiter',
            name='cmd_arbiter',
            output='screen',
            parameters=[shared_params],
        ),

        # Proximity adapter node
        Node(
            package='distance_sensor',
            executable='proximity_adapter_node',
            name='proximity_adapter_node',
            output='screen'
        ),

        Node(
            package='encoding_node',
            executable='encoding_node',
            name='encoding_node',
            output='screen',
            parameters=[shared_params],
        ),
        Node(
            package='opencv_nodes',
            executable='img_kp_grid',
            name='img_kp_grid',
            output='screen',
            parameters=[shared_params],
        ),
        Node(
            package='opencv_nodes',
            executable='img_recog',
            name='img_recog',
            output='screen',
            parameters=[shared_params],
        ),
        Node(
            package='proximity_stop',
            executable='proximity_stop_node',
            name='proximity_stop',
            output='screen',
            parameters=[shared_params],
        ),
        # Node(
        #     package='python_snn_node',
        #     executable='snn_node',
        #     name='python_snn_node',
        #     output='screen'
        # ),
        Node(
            package='task_manager',
            executable='task_manager',
            name='task_manager',
            output='screen'
        ),
    ])