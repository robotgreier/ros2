from setuptools import find_packages, setup

package_name = 'motor_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/motor_control.launch.py']),
    ],
    install_requires=['setuptools', 'smbus2'],
    zip_safe=True,
    maintainer='henrik',
    maintainer_email='henrik.torkildsen@gmail.com',
    description='ROS2 node to drive DFRobot DRI0054 Motor HAT via I2C from /cmd_vel',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_control_node = motor_control.motor_control_node:main',
        ],
    },
)
