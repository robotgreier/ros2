from setuptools import find_packages, setup

package_name = 'distance_sensor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henrik',
    maintainer_email='henrik.torkildsen@gmail.com',
    description='Ultrasonic sensor node for HC-SR04 distance measurement',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'distance_sensor_node = distance_sensor.distance_sensor_node:main',
        ],
    },
)
