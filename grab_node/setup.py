from setuptools import find_packages, setup

package_name = 'grab_node'

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
    maintainer='gudmundur',
    maintainer_email='guara1010@oslomet.no',
    description='Package to handle gripper operations',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
           'grab_node = grab_node.grab_node:main',
           'prox_node = grab_node.prox_node:main',
        ],
    },
)
