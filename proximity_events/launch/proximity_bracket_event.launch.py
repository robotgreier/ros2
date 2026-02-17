from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input_topic', default_value='/ultrasonic/front/scan'),
        DeclareLaunchArgument('output_topic', default_value='/proximity/event'),
        DeclareLaunchArgument('publish_on_change_only', default_value='false'),
        # YAML list syntax for ROS parameters:
        DeclareLaunchArgument('bin_edges', default_value='[0.2, 0.5, 1.0]'),

        Node(
            package='proximity_events',
            executable='proximity_bracket_event',
            name='proximity_bracket_event',
            output='screen',
            parameters=[{
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'publish_on_change_only': LaunchConfiguration('publish_on_change_only'),
                'bin_edges': LaunchConfiguration('bin_edges'),
            }],
        ),
    ])
