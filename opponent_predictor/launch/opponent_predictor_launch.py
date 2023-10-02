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
        'opponent_predictor',
        'config',
        'params.yaml'
    )
    config_dict = yaml.safe_load(open(config, 'r'))

    map_name = config_dict['map_name']
    map_cfg = os.path.join(
        'src',
        'opponent_predictor',
        'maps',
        map_name + ".yaml"
    )
    map_cfg_dict = yaml.safe_load(open(map_cfg, 'r'))

    opponent_predictor_node = Node(
        package="opponent_predictor",
        executable="opponent_predictor_node",
        name="opponent_predictor_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            # YAML Params
            config_dict,
            map_cfg_dict,

            # RVIZ Params
            {"debug_img": False},
            {"visualize": True},
            {"visualize_grid": True},
            {"visualize_obstacle": False},
            {"visualize_opp_pose": True},
            {"visualize_opp_bbox": True},

            # Grid Params
            {"grid_xmin": 0.0},
            {"grid_xmax": 5.0},
            {"grid_ymin": -2.5},
            {"grid_ymax": 2.5},
            {"grid_resolution": 0.04},
            {"plot_resolution": 0.02},
            {"grid_safe_dist": 0.1},

            # Wall Params
            {"wall_safe_dist": 0.2},

            # Obstacle Params
            {"cluster_dist_tol": WIDTH + 2 * WHEEL_LENGTH},
            {"cluster_size_tol": 15},
            {'avoid_dist': 6.0} , # 0.4
        ]
    )
    ld.add_action(opponent_predictor_node)

    return ld
