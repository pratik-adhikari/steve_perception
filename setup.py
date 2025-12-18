from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'steve_perception'

setup(
    # Package metadata
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # Install data files for ROS 2
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.ini')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    # Maintainer information
    maintainer='pratik',
    maintainer_email='pratik.adhikari@smail.inf.h-brs.de',
    description='Robot Agnostic perception frontend (steve)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    # Executable entry points
    entry_points={
        'console_scripts': [
            'perception_node = steve_perception.perception_node:main',
        ],
    },
)
