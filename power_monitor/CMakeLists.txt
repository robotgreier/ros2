cmake_minimum_required(VERSION 3.8)
project(power_monitor)

# ROS2 message generation
find_package(rosidl_default_generators REQUIRED)

# Generate the message
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PowerReading.msg"
)

# Make this package export the message runtime dependency
ament_export_dependencies(rosidl_default_runtime)

# Standard ament footer
ament_package()
