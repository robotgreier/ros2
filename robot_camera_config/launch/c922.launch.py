from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    config = PathJoinSubstitution([
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

            parameters=[config],

            # Remapping so output becomes /camera/image_raw
            remappings=[
                ("image_raw", "image_raw")
            ]
        )
    ])
