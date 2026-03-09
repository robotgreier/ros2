from setuptools import find_packages, setup

package_name = 'python_snn_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/python_snn_node/config', [
        'config/params.yaml',
        'config/weights.mem',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gudmundur',
    maintainer_email='gudmundur@example.com',
    description='Python SNN ROS2 node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This registers: ros2 run python_snn_node snn_node
            'snn_node = python_snn_node.snn_node:main',
        ],
    },
)