from setuptools import find_packages, setup

package_name = 'opencv_nodes'

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
    maintainer_email='gudmundur@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'image_preprocess = opencv_nodes.image_preprocess:main',
        'keypoint_grid = opencv_nodes.keypoint_grid:main',
    ],

    },
)
