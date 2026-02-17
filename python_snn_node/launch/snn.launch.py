from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='python_snn_node',
            executable='python_snn_node',
            name='python_snn_node',
            output='screen',
            parameters=[os.path.join(
                os.getenv('COLCON_CURRENT_PREFIX', ''),
                'share', 'python_snn_node', 'config', 'params.yaml'
            )]
        )
    ])
