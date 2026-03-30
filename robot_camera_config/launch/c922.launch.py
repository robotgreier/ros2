from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
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
            output="screen",
            parameters=[{
                "camera_info_url": "file:///opt/robot_ws/install/robot_camera_config/share/robot_camera_config/config/c922_camera_info.yaml",
            }],
            remappings=[
                ("image_raw", "image_raw")
            ]
        )
    ])