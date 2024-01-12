from setuptools import setup

package_name = 'pure_pursuit'

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
    maintainer='Zhijun Zhuang, Jiatong Sun',
    maintainer_email='zhijunz@seas.upenn.edu, jtsun@seas.upenn.edu',
    description='f1tenth pure pursuit package',
    license='UPenn',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = pure_pursuit.pure_pursuit_node:main',
        ],
    },
)
