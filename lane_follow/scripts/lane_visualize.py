#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Int16

"""
Constant Definition
"""
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


class LaneVisualize(Node):
    """
    Class for lane visualization
    """

    def __init__(self):
        super().__init__("lane_visualize_node")

        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")
        self.declare_parameter("traj_file")

        # Global Map Params
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value

        # Optimal Trajectory
        self.traj_file = self.get_parameter("traj_file").get_parameter_value().string_value
        traj_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, self.traj_file + ".csv")
        traj_data = np.loadtxt(traj_csv_loc, delimiter=';', skiprows=0)
        self.num_traj_pts = len(traj_data)
        self.traj_x = traj_data[:, 1]
        self.traj_y = traj_data[:, 2]
        self.traj_pos = np.vstack((self.traj_x, self.traj_y)).T
        self.traj_yaw = traj_data[:, 3]
        self.traj_v = traj_data[:, 5]
        self.v_min = np.min(self.traj_v)
        self.v_max = np.max(self.traj_v)

        # Lanes Waypoints
        self.num_lanes = self.get_parameter("num_lanes").get_parameter_value().integer_value
        self.lane_files = self.get_parameter("lane_files").get_parameter_value().string_array_value

        self.num_lane_pts = []
        self.lane_x = []
        self.lane_y = []
        self.lane_pos = []

        assert len(self.lane_files) == self.num_lanes
        for i in range(self.num_lanes):
            lane_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
            self.lane_pos = np.vstack((self.lane_x[-1], self.lane_y[-1])).T

        # Topics & Subs, Pubs
        self.timer = self.create_timer(1.0, self.timer_callback)

        traj_topic = "/global_path/optimal_trajectory"
        self.traj_pub_ = self.create_publisher(Marker, traj_topic, 10)

        lane_topic = []
        self.lane_pub_ = []
        for i in range(self.num_lanes):
            lane_topic.append("/global_path/lane_" + str(i))
            self.lane_pub_.append(self.create_publisher(Marker, lane_topic[i], 10))

    def timer_callback(self):
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        if visualize:
            self.visualize_global_path()
            self.visualize_lanes()

    def visualize_global_path(self):
        # Publish trajectory waypoints
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "global_planner"
        marker.type = 4
        marker.action = 0
        marker.points = []
        marker.colors = []

        for i in range(self.num_traj_pts + 1):
            this_point = Point()
            this_point.x = self.traj_x[i % self.num_traj_pts]
            this_point.y = self.traj_y[i % self.num_traj_pts]
            marker.points.append(this_point)

            this_color = ColorRGBA()
            speed_ratio = (self.traj_v[i % self.num_traj_pts] - self.v_min) / (self.v_max - self.v_min)
            this_color.a = 1.0
            this_color.r = (1 - speed_ratio)
            this_color.g = speed_ratio
            marker.colors.append(this_color)

        this_scale = 0.1
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.traj_pub_.publish(marker)

    def visualize_lanes(self):
        # Publish lane waypoints
        for lane_idx in range(self.num_lanes):
            num_pts = self.num_lane_pts[lane_idx]
            target_x = self.lane_x[lane_idx]
            target_y = self.lane_y[lane_idx]

            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = 0
            marker.ns = "global_planner"
            marker.type = 4
            marker.action = 0
            marker.points = []
            marker.colors = []
            for i in range(num_pts + 1):
                this_point = Point()
                this_point.x = target_x[i % num_pts]
                this_point.y = target_y[i % num_pts]
                marker.points.append(this_point)

                this_color = ColorRGBA()
                this_color.a = 1.0
                this_color.r = 0.5
                this_color.g = 0.5
                this_color.b = 0.5
                marker.colors.append(this_color)

            this_scale = 0.1
            marker.scale.x = this_scale
            marker.scale.y = this_scale
            marker.scale.z = this_scale

            marker.pose.orientation.w = 1.0

            self.lane_pub_[lane_idx].publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("Lane Visualize Initialized")
    lane_visualize_node = LaneVisualize()
    rclpy.spin(lane_visualize_node)

    lane_visualize_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
