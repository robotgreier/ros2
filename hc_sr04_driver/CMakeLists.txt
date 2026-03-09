cmake_minimum_required(VERSION 3.8)
project(hc_sr04_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(hc_sr04_laserscan_node src/hc_sr04_laserscan_node.cpp)
ament_target_dependencies(hc_sr04_laserscan_node rclcpp sensor_msgs)

# Link pigpio
target_link_libraries(hc_sr04_laserscan_node pigpio pthread)

install(TARGETS
  hc_sr04_laserscan_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()