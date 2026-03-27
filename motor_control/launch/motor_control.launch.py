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
            package='motor_control',
            executable='motor_control_node',
            name='motor_control_node',
            parameters=[p('motor_control_node')],
            output='screen'
        )
    ])
