# SPIKE – ROS 2 Workspace
 
ROS 2 workspace for **SPIKE**, an autonomous taskbot that compares the same reward-modulated spiking neural network (SNN) running as Python software on a Raspberry Pi 4 and as hardware on an Arty A7-100T FPGA.
 
The workspace handles sensing, spike encoding, reward computation, task state, motor control, and power logging. The SNN runs either as the Python node `python_snn_node` or on the FPGA through `uart_node`, selected by launch file.
 
## Relation to the paper
 
| Paper section | Packages |
|---|---|
| §2.1 Task and Platform | `task_manager`, `motor_control`, `grab_node`, `robot_camera_config` |
| §2.2 From Sensing to Action | `opencv_nodes` (FAST, ArUco), `distance_sensor`, `encoding_node`, `python_snn_node`, `dopamine_reward_node` |
| §2.3 One Network, Two Hardware Substrates | `python_snn_node` (software), `uart` + `fpga_action_decoder` (FPGA) |
| §2.4 Separating Computation from Communication | `power_monitor` (two INA219-based wattmeters, power logging) |
 
## System overview
 
| Hardware | Package → node |
|---|---|
| Raspberry Pi 4 (Ubuntu 24.04 Server, ROS 2 Jazzy) | – |
| Arty A7-100T FPGA | `uart` → `uart_node`, `fpga_action_decoder` → `fpga_action_decoder_node` |
| Logitech C922 camera | `robot_camera_config` (calibration), `opencv_nodes` → `img_kp_grid` (FAST), `img_recog` (ArUco) |
| HC-SR04 ultrasonic sensor | `distance_sensor` → `distance_sensor_node`, `proximity_stop` → `proximity_stop_node` |
| APDS-9930 proximity sensor | `grab_node` → `prox_node` |
| DRI0054 motor driver, 2 × FIT0579 motors | `motor_control` → `motor_control_node` |
| SG90 servo (gripper) | `motor_control` → `gripper_node` |
| 2 × SEN0291 wattmeters | `power_monitor` → `system_power_node`, `fpga_power_node`, `power_logger` |
| Python LIF SNN | `python_snn_node` → `snn_node` |
| Reward system | `dopamine_reward_node` → `dopamine_reward_node` |
 
## Installation
 
On a Raspberry Pi 4 with Ubuntu 24.04 and ROS 2 Jazzy:
 
```bash
git clone https://github.com/robotgreier/ros2.git ~/robot_ws/src
cd ~/robot_ws
rosdep install --from-paths src --ignore-src -y
colcon build
source install/setup.bash
```
 
`setup_robot_dev.sh` automates the system setup (dependencies, rosdep, Cyclone DDS). Review the configuration variables at the top of the script before running it.
 
## Running
 
All parameters are set in `my_ros2_bringup/config/params.yaml`.
 
| Configuration | Command |
|---|---|
| Software SNN (Python on the Raspberry Pi) | `ros2 launch my_ros2_bringup taskbot.launch.py` |
| Hardware SNN (FPGA over UART) | `ros2 launch my_ros2_bringup taskbot_fpga.launch.py` |
| Software SNN with rosbag recording | `ros2 launch my_ros2_bringup taskbot_ros2bag.launch.py` |
| Manual control | `ros2 launch my_ros2_bringup taskbot_manual.launch.py` |
 
For the FPGA configuration, program the Arty A7 with a bitstream from [fpga_project](https://github.com/robotgreier/fpga_project) first.
 
## Related repositories
 
- [python_neural_network](https://github.com/robotgreier/python_neural_network) – Python SNN and hyperparameter tuning
- [fpga_project](https://github.com/robotgreier/fpga_project) – SystemVerilog SNN for the Arty A7-100T
## Citation
 
E. Holm, H. Torkildsen, G. J. Arason, J. K. Dall, B. Nossum, and A. Muhtaroglu, "R-STDP-Based Biomimetic Robot Navigation: A System-Level Energy and Latency Analysis," submitted to *Biomimetics* (MDPI), 2026.
 
## License
 
Apache License 2.0 – see [LICENSE](LICENSE).
 
