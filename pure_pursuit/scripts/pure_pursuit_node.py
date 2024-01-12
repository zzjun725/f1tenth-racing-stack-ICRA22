#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union
import scipy.spatial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from collections import deque

"""
Constant Definition
"""
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length)

class PurePursuit(Node):
    """
    Class for lane follow
    """

    def __init__(self):
        super().__init__("pure_pursuit_node")

        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("lane_occupied_dist")
        self.declare_parameter("obs_activate_dist")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")
        self.declare_parameter("traj_file")

        self.declare_parameter("lookahead_distance")
        self.declare_parameter("lookahead_attenuation")
        self.declare_parameter("lookahead_idx")
        self.declare_parameter("lookbehind_idx")

        self.declare_parameter("kp_steer")
        self.declare_parameter("ki_steer")
        self.declare_parameter("kd_steer")
        self.declare_parameter("max_steer")
        self.declare_parameter("alpha_steer")

        self.declare_parameter("kp_pos")
        self.declare_parameter("ki_pos")
        self.declare_parameter("kd_pos")

        self.declare_parameter("follow_speed")
        self.declare_parameter("lane_dist_thresh")

        # interp
        self.declare_parameter('minL')
        self.declare_parameter('maxL')
        self.declare_parameter('minP')
        self.declare_parameter('maxP')
        self.declare_parameter('interpScale')
        self.declare_parameter('Pscale')
        self.declare_parameter('Lscale')
        self.declare_parameter('D')
        self.declare_parameter('vel_scale')

        self.declare_parameter('minL_corner')
        self.declare_parameter('maxL_corner')
        self.declare_parameter('minP_corner')
        self.declare_parameter('maxP_corner')
        self.declare_parameter('Pscale_corner')
        self.declare_parameter('Lscale_corner')

        self.declare_parameter('avoid_v_diff')
        self.declare_parameter('avoid_L_scale')
        self.declare_parameter('pred_v_buffer')
        self.declare_parameter('avoid_buffer')
        self.declare_parameter('avoid_span')

        # PID Control Params
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0
        self.prev_steer = 0.0
        self.prev_ditem = 0.0

        # Global Map Params
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value
        print(self.map_name)
        # Lanes Waypoints
        self.num_lanes = self.get_parameter("num_lanes").get_parameter_value().integer_value
        self.lane_files = self.get_parameter("lane_files").get_parameter_value().string_array_value

        self.num_lane_pts = []
        self.lane_x = []
        self.lane_y = []
        self.lane_v = []
        self.lane_pos = []

        assert len(self.lane_files) == self.num_lanes
        for i in range(self.num_lanes):
            lane_csv_loc = os.path.join("src", "pure_pursuit", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
            self.lane_v.append(lane_data[:, 2])
            self.lane_pos.append(np.vstack((self.lane_x[-1], self.lane_y[-1]), ).T)
        # In pure pursuit mode, we always use the last lane as the target lane
        self.traj_x = self.lane_x[-1][:]
        self.traj_y = self.lane_y[-1][:]
        self.traj_v = self.lane_v[-1][:]
        self.traj_pos = self.lane_pos[-1][:]
        print(f'length of last lane{len(self.lane_pos[-1])}')
        print(f'max v{np.max(self.lane_v[-1])}')
        # In pure pursuit mode, we do not overtake or car-follow
        self.overtake_wpIdx = set()
        self.slow_wpIdx = set()
        self.corner_wpIdx = set()

        # Car Status Variables
        self.lane_idx = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if self.real_test else "/ego_racecar/odom"
        drive_topic = "/drive"
        waypoint_topic = "/waypoint"

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)

        print('node_init_files')

    def odom_callback(self, odom_msg: Odometry):
        self.curr_vel = odom_msg.twist.twist.linear.x

    def find_wp_target(self, L, traj_distances, curr_pos, curr_idx, lane_idx=None):
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        segment_end = curr_idx
        while traj_distances[segment_end] <= L:
            segment_end = (segment_end + 1) % self.num_traj_pts
        segment_begin = safe_changeIdx(self.num_traj_pts, segment_end, -1)
        x_array = np.linspace(self.traj_x[segment_begin], self.traj_x[segment_end], interpScale)
        y_array = np.linspace(self.traj_y[segment_begin], self.traj_y[segment_end], interpScale)
        v_array = np.linspace(self.traj_v[segment_begin], self.traj_v[segment_end], interpScale)
        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-curr_pos, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        target_global = np.array([x_array[i_interp], y_array[i_interp]])
        target_v = v_array[i_interp]
        L = np.linalg.norm(curr_pos - target_global)
        target_point = np.array([x_array[i_interp], y_array[i_interp]])
        return target_point, target_v

    def find_interp_point(self, L, begin, target):
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        x_array = np.linspace(begin[0], target[0], interpScale)
        y_array = np.linspace(begin[1], target[1], interpScale)
        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-target, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        interp_point = np.array([x_array[i_interp], y_array[i_interp]])
        return interp_point


    def pose_callback(self, pose_msg: Union[PoseStamped, Odometry]):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args:
            pose_msg (PoseStamped / Odometry): incoming message from subscribed topic
        Returns:

        """


        # print('callback')
        cur_speed = self.curr_vel
        # In pure pursuit, only one lane is used as the optimal
        self.last_lane = -1

        #### Read pose data ####
        if self.real_test:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.pose.orientation

        curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                              1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))
        #### Read pose data ####

        #### use optimal traj to get curr_pose_idx ####
        curr_pos_idx = np.argmin(np.linalg.norm(self.lane_pos[-1][:, :2] - curr_pos, axis=1))

        #### interp for finding target ####
        # print(f'current lane{self.last_lane}')
        # traj_distances = np.linalg.norm(self.lane_pos[self.last_lane][:, :2] - curr_pos, axis=1)
        curr_lane_nearest_idx = np.argmin(np.linalg.norm(self.lane_pos[self.last_lane][:, :2] - curr_pos, axis=1))
        traj_distances = np.linalg.norm(self.lane_pos[self.last_lane][:, :2] - self.lane_pos[self.last_lane][curr_lane_nearest_idx, :2], axis=1)
        segment_end = np.argmin(traj_distances)
        num_lane_pts = len(self.lane_pos[self.last_lane])
        if curr_pos_idx in self.corner_wpIdx:
            L = self.get_L_w_speed(cur_speed, corner=True)
        else:
            L = self.get_L_w_speed(cur_speed)
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        while traj_distances[segment_end] <= L:
            segment_end = (segment_end + 1) % num_lane_pts
        segment_begin = (segment_end - 1 + num_lane_pts) % num_lane_pts
        x_array = np.linspace(self.lane_x[self.last_lane][segment_begin], self.lane_x[self.last_lane][segment_end], interpScale)
        y_array = np.linspace(self.lane_y[self.last_lane][segment_begin], self.lane_y[self.last_lane][segment_end], interpScale)
        v_array = np.linspace(self.lane_v[self.last_lane][segment_begin], self.lane_v[self.last_lane][segment_end], interpScale)
        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-curr_pos, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        target_v = v_array[i_interp]
        self.target_point = np.array([x_array[i_interp], y_array[i_interp]])
        #### interp for finding target ####

        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])

        # Get desired speed and steering angle
        vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value
        speed = target_v * vel_scale
        L = np.linalg.norm(curr_pos - self.target_point)
        gamma = 2 / L ** 2
        error = gamma * target_y
        # steer = self.get_steer(error)
        if curr_pos_idx in self.corner_wpIdx:
            steer = self.get_steer_w_speed(cur_speed, error, corner=True)
            speed = speed * 1.0
        else:
            steer = self.get_steer_w_speed(cur_speed, error)

        # steer = self.get_steer_w_speed(cur_speed, error)
        # Publish drive message
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)

        # print("cur_L: %.2f cur_vel: %.2f\t cmd_vel: %.2f\t P: %.2f" % (L, self.curr_vel,
        #                                                        speed,
        #                                                        np.rad2deg(steer)))

        # Visualize waypoints
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        if visualize:
            self.visualize_target()

        return None


    def visualize_target(self):
        # Publish target waypoint
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "target_waypoint"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.target_point[0]
        marker.pose.position.y = self.target_point[1]

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        marker.lifetime.nanosec = int(1e8)

        self.waypoint_pub_.publish(marker)

    def get_L_w_speed(self, speed, corner=False):
        if corner:
            maxL = self.get_parameter('maxL_corner').get_parameter_value().double_value
            minL = self.get_parameter('minL_corner').get_parameter_value().double_value
            Lscale = self.get_parameter('Lscale_corner').get_parameter_value().double_value
        else:
            maxL = self.get_parameter('maxL').get_parameter_value().double_value
            minL = self.get_parameter('minL').get_parameter_value().double_value
            Lscale = self.get_parameter('Lscale').get_parameter_value().double_value
        interp_L_scale = (maxL-minL) / Lscale

        return interp_L_scale * speed + minL

    def get_steer_w_speed(self, speed, error, corner=False):
        if corner:
            maxP = self.get_parameter('maxP_corner').get_parameter_value().double_value
            minP = self.get_parameter('minP_corner').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale_corner').get_parameter_value().double_value
        else:
            maxP = self.get_parameter('maxP').get_parameter_value().double_value
            minP = self.get_parameter('minP').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale').get_parameter_value().double_value

        interp_P_scale = (maxP-minP) / Pscale
        cur_P = maxP - speed * interp_P_scale
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        kd = self.get_parameter('D').get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        # print(f'd_error: {d_error}')
        if not self.real_test:
            if d_error == 0:
                d_error = self.prev_ditem
            else:
                self.prev_ditem = d_error
                self.prev_steer_error = error
        else:
            self.prev_ditem = d_error
            self.prev_steer_error = error
        if corner:
            steer = cur_P * error
        else:
            steer = cur_P * error + kd * d_error
        # print(f'cur_p_item:{cur_P * error},  cur_d_item:{kd * d_error}')
        new_steer = np.clip(steer, -max_control, max_control)
        return new_steer


def main(args=None):
    rclpy.init(args=args)
    print("Pure Pursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
