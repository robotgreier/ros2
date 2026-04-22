cmake_minimum_required(VERSION 3.8)
project(power_monitor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  power_monitor/system_power_node.py
  power_monitor/fpga_power_node.py
  power_monitor/power_logger_node.py
  power_monitor/ina219.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()