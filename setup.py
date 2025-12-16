from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'steve_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        
        (os.path.join('share', package_name, 'config'),
         # Install both yaml and ini (RTAB-Map INI files are required at runtime).
         glob('config/*.yaml') + glob('config/*.ini')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pratik',
    maintainer_email='pratik.adhikari@smail.inf.h-brs.de',
    description='Robot Agnostic perception frontend (steve)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = steve_perception.perception_node:main',
        ],
    },
)
