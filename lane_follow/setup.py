from setuptools import setup

package_name = 'lane_follow'

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
    maintainer='Kedar Prasad Karpe, Griffon McMahon, Jiatong Sun',
    maintainer_email='karpenet@seas.upenn.edu, gmcmahon@seas.upenn.edu, jtsun@seas.upenn.edu',
    description='f1tenth lane_follow',
    license='UPenn',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follow_node = lane_follow.lane_follow_node:main',
        ],
    },
)
