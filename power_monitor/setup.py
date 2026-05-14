from setuptools import find_packages, setup

package_name = 'power_monitor'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/power_nodes.launch.py']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henrik',
    maintainer_email='henrik.torkildsen@gmail.com',
    description='ROS2 nodes for reading INA219 (DFRobot SEN0291) power sensors and logging/visualization.',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'system_power_node = power_monitor.system_power_node:main',
            'fpga_power_node = power_monitor.fpga_power_node:main',
            'power_logger = power_monitor.power_logger_node:main',
            'snn_logger_node = power_monitor.snn_logger_node:main',
        ],
    },
)
