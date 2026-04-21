from setuptools import setup

package_name = 'dopamine_reward_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email',
    description='Dopamine reward node for SNN/FPGA system',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dopamine_reward_node = dopamine_reward_node.dopamine_reward_node:main',
        ],
    },
)