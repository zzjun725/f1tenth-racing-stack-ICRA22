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

class LaneFollow(Node):
    """
    Class for lane follow
    """

    def __init__(self):
        super().__init__("lane_follow_node")

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
        # Optimal Trajectory
        self.traj_file = self.get_parameter("traj_file").get_parameter_value().string_value
        traj_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, self.traj_file + ".csv")
        print(f'read optimal raceline from {traj_csv_loc}')
        traj_data = np.loadtxt(traj_csv_loc, delimiter=';', skiprows=0)
        self.num_traj_pts = len(traj_data)
        self.traj_x = traj_data[:, 1]
        self.traj_y = traj_data[:, 2]
        self.traj_pos = np.vstack((self.traj_x, self.traj_y)).T
        self.traj_yaw = traj_data[:, 3]
        self.traj_v = traj_data[:, 5]
        print(f'length of traj_v{len(self.traj_v)}')
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
            lane_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
            self.lane_v.append(lane_data[:, 2])
            self.lane_pos.append(np.vstack((self.lane_x[-1], self.lane_y[-1]), ).T)
        print(f'length of last lane{len(self.lane_pos[-1])}')
        print(f'max v{np.max(self.lane_v[-1])}')
        overtaking_idx_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, 'overtaking_wp_idx.npy')
        data = np.load(overtaking_idx_csv_loc, mmap_mode = 'r')
        # self.overtake_wpIdx = set(list(data))
        self.overtake_wpIdx = set(range(190, 240))
        print(self.overtake_wpIdx)

        slow_idx_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, 'slowdown_wp_idx.npy')
        data2 = np.load(slow_idx_csv_loc, mmap_mode='r')
        self.slow_wpIdx = set(list(data2))
        print(self.slow_wpIdx)

        self.corner_wpIdx = set(list(range(0, 80)) + list(range(280, 305)))

        # Car Status Variables
        self.lane_idx = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None

        # Obstacle Variables
        self.obstacles = None
        self.opponent = np.array([np.inf, np.inf])
        self.lane_free = [True] * self.num_lanes
        self.declare_parameter('avoid_dist')
        self.opponent_v = 0.0
        self.opponent_last = np.array([0.0, 0.0])
        self.opponent_timestamp = 0.0
        self.pred_v_buffer = self.get_parameter('pred_v_buffer').get_parameter_value().integer_value
        self.pred_v_counter = 0
        self.avoid_buffer = self.get_parameter('avoid_buffer').get_parameter_value().integer_value
        self.avoid_counter = 0
        self.detect_oppo = False
        self.avoid_L_scale = self.get_parameter('avoid_L_scale').get_parameter_value().double_value
        self.last_lane = -1

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if self.real_test else "/ego_racecar/odom"
        obstacle_topic = "/opp_predict/bbox"
        opponent_topic = "/opp_predict/state"
        drive_topic = "/drive"
        waypoint_topic = "/waypoint"

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.obstacle_sub_ = self.create_subscription(PoseArray, obstacle_topic, self.obstacle_callback, 1)
        self.opponent_sub_ = self.create_subscription(PoseStamped, opponent_topic, self.opponent_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)

        print('node_init_files')

    def odom_callback(self, odom_msg: Odometry):
        self.curr_vel = odom_msg.twist.twist.linear.x

    def obstacle_callback(self, obstacle_msg: PoseArray):
        obstacle_list = []
        for obstacle in obstacle_msg.poses:
            x = obstacle.position.x
            y = obstacle.position.y
            obstacle_list.append([x, y])
        self.obstacles = np.array(obstacle_list) if obstacle_list else None

        if self.obstacles is None:
            self.lane_free = np.array([True] * self.num_lanes)
            return

        lane_occupied_dist = self.get_parameter("lane_occupied_dist").get_parameter_value().double_value
        for i in range(self.num_lanes):
            d = scipy.spatial.distance.cdist(self.lane_pos[i], self.obstacles)
            self.lane_free[i] = (np.min(d) > lane_occupied_dist)
        # print(f'lane_free_situation {self.lane_free}')


    def opponent_callback(self, opponent_msg: PoseStamped):
        opponent_x = opponent_msg.pose.position.x
        opponent_y = opponent_msg.pose.position.y
        self.opponent = np.array([opponent_x, opponent_y])
        # print(self.opponent)

        ## velocity
        if not np.any(np.isinf(self.opponent)):
            # print(self.detect_oppo)
            if self.detect_oppo:
                oppoent_dist_diff = np.linalg.norm(self.opponent - self.opponent_last)
                # self.opponent_v = 0.0
                # if oppoent_dist_diff != 0:
                if self.pred_v_counter == 7:
                    self.pred_v_counter = 0
                    cur_time = opponent_msg.header.stamp.nanosec/1e9 + opponent_msg.header.stamp.sec
                    time_interval = cur_time - self.opponent_timestamp
                    self.opponent_timestamp = cur_time
                    opponent_v = oppoent_dist_diff / max(time_interval, 0.005)
                    self.opponent_last = self.opponent.copy()
                    self.opponent_v = opponent_v
                    # print(f'cur distance diff {oppoent_dist_diff}')
                    print(f'cur oppoent v {self.opponent_v}')
                else:
                    self.pred_v_counter += 1
            else:
                self.detect_oppo = True
                self.opponent_last = self.opponent.copy()
        else:
            self.detect_oppo = False

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

    def avoid_static(self):
        pass

    def avoid_dynamic(self):
        pass

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
        print(f'cur_speed{cur_speed}')
        # print(self.lane_free)

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
        print(f'curr pos idx{curr_pos_idx}')
        # curr_speed_refer_idx = np.argmin(np.linalg.norm())
        # slow_factor = 1.0
        if curr_pos_idx in self.slow_wpIdx:
            slow_factor = 0.7375
        else:
            slow_factor = 1.0

        #### switch back to optimal raceline ####
        if self.lane_free[-1]:
            self.avoid_counter = min(self.avoid_buffer, self.avoid_counter + 1)
        else:
            self.avoid_counter = 0
        if self.avoid_counter == self.avoid_buffer and self.lane_free[-1]:
            self.last_lane = -1
        if curr_pos_idx not in self.overtake_wpIdx:
            self.last_lane = -1
        #### switch back to optimal raceline ####

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
        target_global = np.array([x_array[i_interp], y_array[i_interp]])
        if self.last_lane == -1:
            target_v = v_array[i_interp]
        else:
            target_v = self.lane_v[-1][curr_pos_idx] * 0.6
            print(f'target_V {target_v}')
        self.target_point = np.array([x_array[i_interp], y_array[i_interp]])
        #### interp for finding target ####

        # Choose new target point from the closest lane if obstacle exists
        avoid_dist = self.get_parameter('avoid_dist').get_parameter_value().double_value
        avoid_dist = avoid_dist * max(cur_speed, 3.0)
        avoid_v_diff = self.get_parameter('avoid_v_diff').get_parameter_value().double_value
        avoid_span = self.get_parameter('avoid_span').get_parameter_value().double_value

        # print(self.opponent)
        if not np.any(np.isinf(self.opponent)):
            # print('detect obs')
            # if np.any(np.isinf(self.opponent)):
            #     self.opponent = self.opponent_last.copy()
            cur_obs_dist = np.linalg.norm(self.opponent - curr_pos)
            v_diff = self.curr_vel - self.opponent_v
            print(f'opponent_v: {self.opponent_v}')
            print(f'cur_obs_distance: {cur_obs_dist}')
            print(f'last_lane" {self.last_lane}')
            print(f'lane free {self.lane_free}')
            # print(f'cur_avoid_dist: {avoid_dist}')
            # print(f'cur_obs: {self.opponent}')
            # if cur_obs_dist <= avoid_dist and (v_diff > avoid_v_diff or v_diff < -3):
        ################## ONLY LANE SWITCHING #########################
            if not np.any(self.lane_free):
                target_v = max(self.opponent_v * 0.8, 0.0)
                # print('avoid when not free')
                # lane_targets = []
                # lane_targets_distance_from_obs = []
                # for i in range(self.num_lanes):
                #     dist_to_lane = np.linalg.norm(self.lane_pos[i][:, :2] - self.target_point, axis=1)
                #     min_idx = np.argmin(dist_to_lane)
                #     lane_targets.append(self.lane_pos[i][min_idx].copy())
                #     lane_targets_distance_from_obs.append(np.linalg.norm(self.lane_pos[i][min_idx]-self.opponent))
                # safest_lane_idx = np.argmax(np.array(lane_targets_distance_from_obs))
                # self.target_point = lane_targets[safest_lane_idx]
                # self.last_lane = safest_lane_idx
                # # pass
            else:
                if self.detect_oppo and cur_obs_dist <= avoid_dist and abs(v_diff) >= avoid_v_diff and not self.lane_free[self.last_lane]:
                    print('obs_detected')
                    #### overtaking, only use lane 0 ####
                    if curr_pos_idx in self.overtake_wpIdx and (self.lane_free[0] or self.lane_free[-1]):
                        print('overtake')
                        target_lane_idx = -((-self.last_lane + 1) % 2)
                        self.last_lane = target_lane_idx
                        dist_to_lane = np.linalg.norm(self.lane_pos[target_lane_idx][:, :2] - self.target_point, axis=1)  # (n, 2)  (1, 2)
                        min_idx = np.argmin(dist_to_lane)
                        lane_target = self.lane_pos[target_lane_idx][min_idx]
                        self.target_point = lane_target
                    else:
                        ##### if obstacle is not moving, force to switch lane ######
                        if self.opponent_v < 1.0:
                            # target_v = 0.0
                            target_v = max(self.opponent_v*0.8, 0.0)
                            # for i in range(self.num_lanes):
                            #     if self.lane_free[i]:
                            #         target_lane_idx = i
                            #         self.last_lane = target_lane_idx
                            #         dist_to_lane = np.linalg.norm(self.lane_pos[target_lane_idx][:, :2] - self.target_point, axis=1)
                            #         min_idx = np.argmin(dist_to_lane)
                            #         lane_target = self.lane_pos[target_lane_idx][min_idx]
                            #         self.target_point = lane_target
                            #         target_v = target_v * 0.8
                            #         print('avoid')
                            #         break
                        else:
                            ##### if obstacle is moving, follow it ######
                            target_v = max(self.opponent_v*0.8, 0.0)
                # else:
                #     target_v = max(self.opponent_v * 0.9, 0.0)


        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])

        # Get desired speed and steering angle
        # speed = self.traj_v[self.curr_idx % self.num_traj_pts]

        # interp for speed
        # if self.last_lane == 1:
        #     target_v = 2.0
        vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value
        # if target_v > 9.0 :
        #     target_v = 9.0
        speed = target_v * vel_scale * slow_factor
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

    def get_lookahead_dist(self, curr_idx):
        """
        This method should calculate the lookahead distance based on past and future waypoints

        Args:
            curr_idx (ndarray[int]): closest waypoint index
        Returns:
            lookahead_dist (float): lookahead distance

        """
        L = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        lookahead_idx = self.get_parameter("lookahead_idx").get_parameter_value().integer_value
        lookbehind_idx = self.get_parameter("lookbehind_idx").get_parameter_value().integer_value
        slope = self.get_parameter("lookahead_attenuation").get_parameter_value().double_value

        yaw_before = self.traj_yaw[(curr_idx - lookbehind_idx) % self.num_traj_pts]
        yaw_after = self.traj_yaw[(curr_idx + lookahead_idx) % self.num_traj_pts]
        yaw_diff = abs(yaw_after - yaw_before)
        if yaw_diff > np.pi:
            yaw_diff = yaw_diff - 2 * np.pi
        if yaw_diff < -np.pi:
            yaw_diff = yaw_diff + 2 * np.pi
        yaw_diff = abs(yaw_diff)
        if yaw_diff > np.pi / 2:
            yaw_diff = np.pi / 2
        L = max(0.5, L * (np.pi / 2 - yaw_diff * slope) / (np.pi / 2))

        return L

    def get_steer(self, error):
        """ Get desired steering angle by PID
        """
        kp = self.get_parameter("kp_steer").get_parameter_value().double_value
        ki = self.get_parameter("ki_steer").get_parameter_value().double_value
        kd = self.get_parameter("kd_steer").get_parameter_value().double_value
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        alpha = self.get_parameter("alpha_steer").get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        self.prev_steer_error = error
        self.steer_integral += error
        steer = kp * error + ki * self.steer_integral + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        new_steer = alpha * new_steer + (1 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer

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
    print("Lane Follow Initialized")
    lane_follow_node = LaneFollow()
    rclpy.spin(lane_follow_node)

    lane_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
