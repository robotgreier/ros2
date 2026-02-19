from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='motor_control_node',
            executable='motor_control_node',
            name='motor_control',
            parameters=[
                # Robot geometry and kinematics, needs to be updated for our robot:
                {'wheel_separation': 0.32},
                {'wheel_radius': 0.05},
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
        )
    ])
