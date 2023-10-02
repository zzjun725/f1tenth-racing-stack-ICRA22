from setuptools import setup

package_name = 'opponent_predictor'

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
    description='f1tenth opponent predictor lab',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'opponent_predictor_node = opponent_predictor.opponent_predictor_node:main',
        ],
    },
)
