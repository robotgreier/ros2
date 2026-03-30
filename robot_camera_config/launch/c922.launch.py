from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    config_file = PathJoinSubstitution([
        FindPackageShare("robot_camera_config"),
        "config",
        "c922_camera_info.yaml"
    ])

    return LaunchDescription([
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="camera",
            namespace="",
            output="screen",

            parameters=[{
                "camera_info_url": config_file,
            }],

            # Remapping so output becomes /camera/image_raw
            remappings=[
                ("image_raw", "image_raw")
            ]
        )
    ])
