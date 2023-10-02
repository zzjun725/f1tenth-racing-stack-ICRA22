#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


def generate_launch_description():
    ld = LaunchDescription()
    print(os.getcwd())
    config = os.path.join(
        'src',
        'dummy_car',
        'config',
        'params.yaml'
    )
    config_dict = yaml.safe_load(open(config, 'r'))
    dummy_car_node = Node(
        package="dummy_car",
        executable="dummy_car_node.py",
        name="dummy_car_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            # YAML Params
            config_dict,

            # RVIZ Params
            {"visualize": False},

            # Pure Pursuit Params
            {"lookahead_distance": 0.5},
            {"lookahead_attenuation": 0.6},
            {"lookahead_idx": 16},
            {"lookbehind_idx": 0},

            # PID Control Params
            {"kp": 0.5},
            {"ki": 0.0},
            {"kd": 0.0},
            {"max_control": MAX_STEER},
            {"steer_alpha": 1.0},

            # Car Params
            {"overwrite_speed": True},
            {"speed": 0.0},

            # interp
            {'minL': 0.5},
            {'maxL': 1.5},
            {'minP': 0.5},
            {'maxP': 0.7},
            {'interpScale': 20},
            {'Pscale': 7.0},
            {'Lscale': 7.0},
            {'D': 5.0},
            {'vel_scale': 0.8},

            {'avoid_dist': 0.1}  # 0.4
        ]
    )
    ld.add_action(dummy_car_node)

    return ld
