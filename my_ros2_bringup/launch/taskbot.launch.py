from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    
    # Path to the camera config file inside the ROS2 package
    camera_config = PathJoinSubstitution([
        FindPackageShare("robot_camera_config"),
        "config",
        "c922.yaml"
    ])


    return LaunchDescription([
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="c922_camera",
            namespace="camera",

            parameters=[camera_config],

            # Remapping so output becomes /camera/image_raw
            remappings=[
                ("image_raw", "image_raw")
            ]
        ),
        Node(
            package='distance_sensor',
            executable='distance_sensor_node',
            name='distance_sensor_node',
            output='screen'
        ),        
        Node(
            package='motor_control_node',
            executable='motor_control_node',
            name='motor_control',
            parameters=[
                # Robot geometry and kinematics, needs to be updated for our robot:
                {'wheel_separation': 0.15},
                {'wheel_radius': 0.027},
                {'max_wheel_linear_speed': 0.8},
                # DRI0054 defaults:
                {'i2c_address': 0x60},     # per DFRobot docs
                {'left_motor_id': 1},
                {'right_motor_id': 2},
                {'invert_left': False},
                {'invert_right': True},    # often one side needs inversion
                {'cmd_vel_timeout': 0.5},
                {'slew_rate': 6.0},        # throttle units/sec
                {'stop_mode': 'brake'}     # or 'coast'
            ],
            output='screen'
        ),
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
        Node(
            package='python_snn_node',
            executable='snn_node',
            name='python_snn_node',
            output='screen'
        ),
        Node(
            package='task_manager',
            executable='task_manager',
            name='task_manager',
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf',
            arguments=["0.10", "0.0", "0.09", "0", "0", "0", "base_link", "camera_link"],
            output='screen'
        )

    ])