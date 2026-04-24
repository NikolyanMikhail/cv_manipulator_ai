import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cv_manipulator_task'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nikol',
    maintainer_email='nikol@todo.todo',
    description='CV Manipulator Task',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_brain_node = cv_manipulator_task.ai_brain_node:main',
            'camera_reader = cv_manipulator_task.camera_reader:main',
            'data_collector = cv_manipulator_task.data_collector:main',
        ],
    },
)
