from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
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
            executable='power_logger',
            name='power_logger',
            output='screen'
        ),
    ])
