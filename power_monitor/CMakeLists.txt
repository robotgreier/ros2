cmake_minimum_required(VERSION 3.8)
project(power_monitor)

# --- Required for python packages ---
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# --- Message generation ---
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PowerReading.msg"
)

ament_export_dependencies(rosidl_default_runtime)

# --- Install python package ---
ament_python_install_package(${PROJECT_NAME})

# --- Install your executables ---
install(PROGRAMS
  power_monitor/system_power_node.py
  power_monitor/fpga_power_node.py
  power_monitor/power_logger_node.py
  power_monitor/ina219.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- Testing boilerplate required by rosidl ---
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()