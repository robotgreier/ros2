from setuptools import setup
import os
from glob import glob

# Name of python package, matches the folder name in src/ros2/taskbot_vision, package.xml and resource file.
package_name = 'taskbot_vision'

# Setup function to define the package and its contents
setup(
    name=package_name,  # Name of the package, used for installation and referencing in ROS2
    version='0.0.1',    # Version of the package, can be updated as needed for releases
    
    packages=[package_name], #Packages to install

    # Data files to include in the package, such as resource files, package.xml, and launch files
    data_files=[
        ('share/ament_index/resource_index/packages',   # To register the package with ament index, allowing it to be found by ROS2 tools
            ['resource/' + package_name]),              # Resource file that contains the package name, used for ament index registration
        ('share/' + package_name, ['package.xml']),     # Package manifest file that describes the package and its dependencies, required for ROS2 packages
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')), # To include all launch files in the launch directory
    ],

    install_requires=['setuptools'], 
    zip_safe=True, 
    maintainer='Henrik Torkildsen',
    maintainer_email='henrik.torkildsen@gmail.com',
    description='Vision system for energy-aware autonomous taskbot',
    license='Apache License 2.0',
    tests_require=['pytest'], 
    
    # Define console scripts for the nodes in the package, allowing them to be run directly from the command line after installation
    entry_points={
        'console_scripts': [
            # vision_node is the main node for processing camera input and performing ORB-SLAM
            'vision_node = taskbot_vision.vision_node:main',

            # fpga_interface_node handles communication with FPGA for offloading computations
            'fpga_interface_node = taskbot_vision.fpga_interface_node:main',

            # mock_camera_node simulates a camera for testing purposes
            'mock_camera_node = taskbot_vision.mock_camera_node:main',

            # occupancy_grid_node generates and publises an occupancy grid from the detected features and the aruco markers
            'occupancy_grid_node = taskbot_vision.occupancy_grid_node:main',
        ],
    },
)