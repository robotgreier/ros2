# launch/snn.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('python_snn_node')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='python_snn_node',
            executable='python_snn_node',
            name='python_snn_node',
            output='screen',
            parameters=[params_file],   # <-- peker til installert kopi
        )
    ])
