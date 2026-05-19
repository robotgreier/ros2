# ROS 2 repository
# SPIKE: Energy Efficient Autonomous Taskbot — SNN Driven Navigation Based on FAST Keypoints, Distance Sensing and Object Recognition
>
>## Launch files -> my_ros2_bringup -> launch  
-Terminal command: ros2 launch my_ros2_bringup launchfilename.launch.py  
-Parameter tuning: my_ros2_bringup -> config -> params.yaml  
>
>## System overview  
-Raspberry Pi 4  
--Ubuntu 24.04 (server)  
--ROS 2 Jazzy Jalisco (base)  
>
-Arty A7 FPGA   
--UART communication Pi-FPGA: uart -> uart_node  
--FPGA actuator commands: fpga_action_decoder -> fpga_action_decoder_node  
>
-DRI0054 Raspberry Pi Motor Driver HAT  
--Emakefun MotorDriver files: motor_control -> motor_control  
> 
-FIT0579 metal gear DC motor x2  
--Differential drive: motor_control -> motor_control_node  
>
-SG90 servo motor  
--Gripper: motor_control -> gripper_node  
>
-C922 USB camera   
--calibration file: robot_camera_config -> config -> c922.yaml  
--FAST keypoints, SNN input: opencv_nodes -> img_kp_grid  
--ArUco object detection, SNN input: opencv_nodes -> img_recog  
> 
-HC-SR04 Ultrasonic distance sensor  
--Emergency stop: proximity_stop -> proximity_stop_node  
--Distance measurements, SNN input: distance_sensor -> distance_sensor_node  
>
-APDS-9930 proximity sensor  
--Sense objects to grip: grab_node -> prox_node  
>
-SEN0291 wattmeters x2  
--INA219 communication: power_monitor -> ina219  
--System measurements: power_monitor -> system_power_node  
--FPGA measurements: power_monitor -> fpga_power_node  
--Logging of measurements: power_monitor -> power_logger_node  
>  
-Python LIF SNN
--Software emulation: python_snn_node -> snn_node
>
-LIF SNN reward system
--Dopamine logic tuning: dopamine_reward_node -> dopamine_logic
--Dopamine reward: dopamine_reward_node -> dopamine_reward_node
