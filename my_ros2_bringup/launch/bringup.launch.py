from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='grab_node',
            executable='grab_node',
            name='grab_node',
            output='screen'
        ),
        Node(
            package='cmd_arbiter',
            executable='cmd_arbiter',
            name='cmd_arbiter',
            output='screen'
        ),
        Node(
            package='encoding_node',
            executable='encoding_node',
            name='encoding_node',
            output='screen',
            parameters=[{
            "proximity_topic": "/ultrasonic/front/scan",
            "output_topic": "/snn/input",
            "proximity_bin_edges": [0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        }],
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
        ),
        Node(
            package='opencv_nodes',
            executable='img_recog',
            name='img_recog',
            output='screen'
        ),
        Node(
            package='proximity_stop',
            executable='proximity_stop_node',
            name='proximity_stop',
            output='screen'
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