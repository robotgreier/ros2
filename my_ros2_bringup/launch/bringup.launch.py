from launch import LaunchDescription
from launch_ros.actions import Node
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

    return LaunchDescription([
        Node(
            package='grab_node',
            executable='grab_node',
            name='grab_node',
            output='screen',
            parameters=[p('grab_node')],
        ),
        Node(
            package='cmd_arbiter',
            executable='cmd_arbiter',
            name='cmd_arbiter',
            output='screen',
            parameters=[p('cmd_arbiter')],
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
            parameters=[p('encoding_node')],
        ),
        Node(
            package='opencv_nodes',
            executable='img_kp_grid',
            name='img_kp_grid',
            output='screen',
            parameters=[p('img_kp_grid')],
        ),
        Node(
            package='opencv_nodes',
            executable='img_recog',
            name='img_recog',
            output='screen',
            parameters=[p('img_recog')],
        ),
        Node(
            package='proximity_stop',
            executable='proximity_stop_node',
            name='proximity_stop',
            output='screen',
            parameters=[p('proximity_stop')],
        ),
        # Node(
        #     package='python_snn_node',
        #     executable='snn_node',
        #     name='python_snn_node',
        #     parameters=[p('python_snn_node')],
        #     output='screen'
        # ),
        Node(
            package='task_manager',
            executable='task_manager',
            name='task_manager',
            output='screen'
        ),
    ])
