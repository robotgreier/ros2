from setuptools import setup
import os
from glob import glob

package_name = 'taskbot_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Team',
    maintainer_email='your.email@example.com',
    description='Vision system for energy-aware autonomous taskbot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = taskbot_vision.vision_node:main',
            'fpga_interface_node = taskbot_vision.fpga_interface_node:main',
            'mock_camera_node = taskbot_vision.mock_camera_node:main',
        ],
    },
)