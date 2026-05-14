from setuptools import find_packages, setup

package_name = 'encoding_node'

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
    description='Encoder node subscribes to sensor data and publish input_snn',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'encoding_node = encoding_node.encoding_node:main',
            'spike_train_publisher = encoding_node.spike_train_publisher:main',
        ],
    },
)
