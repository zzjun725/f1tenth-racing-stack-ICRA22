#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

"""
Constant Definition
"""
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)

def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length)

class DummyCar(Node):
    """
    Class for dummy car
    """

    def __init__(self):
        super().__init__('dummy_car_node')

        # ROS Params
        self.declare_parameter('dummy_car_file')

        self.declare_parameter('visualize')

        self.declare_parameter('real_test')
        self.declare_parameter('map_name')

        self.declare_parameter('lookahead_distance')
        self.declare_parameter('lookahead_attenuation')
        self.declare_parameter('lookahead_idx')
        self.declare_parameter('lookbehind_idx')

        self.declare_parameter('kp')
        self.declare_parameter('ki')
        self.declare_parameter('kd')
        self.declare_parameter("max_control")
        self.declare_parameter("steer_alpha")

        self.declare_parameter("overwrite_speed")
        self.declare_parameter("speed")

        # PID Control Params
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0

        # Global Map Params
        map_name = self.get_parameter("map_name").get_parameter_value().string_value
        dummy_car_file = self.get_parameter("dummy_car_file").get_parameter_value().string_value
        csv_loc = os.path.join('src', 'dummy_car', 'csv', map_name, dummy_car_file + '.csv')

        waypoints = np.loadtxt(csv_loc, delimiter=';', skiprows=0)
        self.num_pts = len(waypoints)
        self.target_x = waypoints[:, 1]
        self.target_y = waypoints[:, 2]
        self.target_pos = np.vstack((self.target_x, self.target_y)).T
        self.target_v = waypoints[:, 3]
        self.target_yaw = waypoints[:, 5]
        self.v_max = np.max(self.target_v)
        self.v_min = np.min(self.target_v)

        self.num_traj_pts = len(waypoints)
        self.traj_x = waypoints[:, 1]
        self.traj_y = waypoints[:, 2]
        self.traj_pos = np.vstack((self.traj_x, self.traj_y)).T
        self.traj_yaw = waypoints[:, 3]
        self.traj_v = waypoints[:, 5]

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

        # Car Status Params
        self.lane_idx = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None
        self.prev_steer_error = 0.0

        # Topics & Subs, Pubs
        pose_topic = "/opp_racecar/odom"
        odom_topic = "/opp_racecar/odom"
        drive_topic = "/opp_drive"
        waypoint_topic = "/opp_waypoint"
        path_topic = "/opp_global_path"

        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.path_pub_ = self.create_publisher(Marker, path_topic, 10)

    def odom_callback(self, odom_msg: Odometry):
        self.curr_vel = odom_msg.twist.twist.linear.x

    def pose_callback(self, pose_msg: Union[PoseStamped, Odometry]):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args:
            pose_msg (PoseStamped / Odometry): incoming message from subscribed topic
        Returns:

        """
        # print('call back')
        # Read pose data
        cur_speed = self.curr_vel
        curr_x = pose_msg.pose.pose.position.x
        curr_y = pose_msg.pose.pose.position.y
        curr_pos = np.array([curr_x, curr_y])
        curr_quat = pose_msg.pose.pose.orientation

        curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                              1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        # Find index of the current point
        distances = np.linalg.norm(self.target_pos - curr_pos, axis=1)
        self.curr_idx = np.argmin(distances)

        # Get lookahead distance
        # L = self.get_lookahead_dist(self.curr_idx)
        #
        # # Binary search goal waypoint to track
        # self.goal_idx = self.curr_idx
        # while distances[self.goal_idx] <= L:
        #     self.goal_idx = (self.goal_idx + 1) % self.num_pts
        #
        # left = self.target_pos[(self.goal_idx - 1) % self.num_pts, :]
        # right = self.target_pos[self.goal_idx % self.num_pts, :]
        #
        # while True:
        #     mid = (left + right) / 2
        #     dist = np.linalg.norm(mid - curr_pos)
        #     if abs(dist - L) < 1e-2:
        #         self.target_point = mid
        #         break
        #     elif dist > L:
        #         right = mid
        #     else:
        #         left = mid

        #### interp for finding target ####
        L = self.get_L_w_speed(cur_speed)
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        segment_end = self.curr_idx
        while distances[segment_end] <= L:
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
        self.target_point = np.array([x_array[i_interp], y_array[i_interp]])
        # print(self.target_point)
        #### interp for finding target ####

        # Transform goal point to vehicle frame of reference
        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])

        # Get desired speed and steering angle
        # speed = self.target_v[self.curr_idx % self.num_pts]
        # gamma = 2 / L ** 2
        # error = gamma * target_y
        # steer = self.get_steer(error)

        # interp for speed
        vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value
        speed = target_v * vel_scale

        gamma = 2 / L ** 2
        error = gamma * target_y
        # steer = self.get_steer(error)
        steer = self.get_steer_w_speed(cur_speed, error)

        overwrite_speed = self.get_parameter('overwrite_speed').get_parameter_value().bool_value
        if overwrite_speed:
            speed = float(self.get_parameter('speed').get_parameter_value().double_value)

        # Publish drive message
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)
        # print('drive')

        # Visualize waypoints
        visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        if visualize:
            self.visualize_waypoints()

        return None

    def get_lookahead_dist(self, curr_idx):
        """
        This method should calculate the lookahead distance based on past and future waypoints

        Args:
            curr_idx (ndarray[int]): closest waypoint index
        Returns:
            lookahead_dist (float): lookahead distance

        """
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_idx = self.get_parameter('lookahead_idx').get_parameter_value().integer_value
        lookbehind_idx = self.get_parameter('lookbehind_idx').get_parameter_value().integer_value
        slope = self.get_parameter('lookahead_attenuation').get_parameter_value().double_value

        yaw_before = self.target_yaw[(curr_idx - lookbehind_idx) % self.num_pts]
        yaw_after = self.target_yaw[(curr_idx + lookahead_idx) % self.num_pts]
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
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        max_control = self.get_parameter('max_control').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error
        steer = kp * error + ki * self.integral + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        new_steer = alpha * new_steer + (1 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer

    def visualize_waypoints(self):
        # Publish all waypoints
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = 0
        marker.ns = 'global_planner'
        marker.type = 4
        marker.action = 0
        marker.points = []
        marker.colors = []
        for i in range(self.num_pts + 1):
            this_point = Point()
            this_point.x = self.target_x[i % self.num_pts]
            this_point.y = self.target_y[i % self.num_pts]
            marker.points.append(this_point)

            this_color = ColorRGBA()
            speed_ratio = (self.target_v[i % self.num_pts] - self.v_min) / (self.v_max - self.v_min)
            this_color.a = 1.0
            this_color.r = (1 - speed_ratio)
            this_color.g = speed_ratio
            marker.colors.append(this_color)

        this_scale = 0.1
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.path_pub_.publish(marker)

        # Publish target waypoint
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = 0
        marker.ns = 'opponent_waypoint_target'
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.target_point[0]
        marker.pose.position.y = self.target_point[1]

        speed_ratio = (self.target_v[self.goal_idx] - self.v_min) / (self.v_max - self.v_min)
        marker.color.a = 1.0
        marker.color.r = (1 - speed_ratio)
        marker.color.g = speed_ratio

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.waypoint_pub_.publish(marker)

    def get_L_w_speed(self, speed):
        maxL = self.get_parameter('maxL').get_parameter_value().double_value
        minL = self.get_parameter('minL').get_parameter_value().double_value
        Lscale = self.get_parameter('Lscale').get_parameter_value().double_value
        interp_L_scale = (maxL-minL) / Lscale

        return interp_L_scale * speed + minL

    def get_steer_w_speed(self, speed, error):
        maxP = self.get_parameter('maxP').get_parameter_value().double_value
        minP = self.get_parameter('minP').get_parameter_value().double_value
        Pscale = self.get_parameter('Pscale').get_parameter_value().double_value
        interp_P_scale = (maxP-minP) / Pscale
        cur_P = maxP - speed * interp_P_scale
        max_control = self.get_parameter("max_control").get_parameter_value().double_value
        kd = self.get_parameter('D').get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        self.prev_steer_error = error
        steer = cur_P * error + kd * d_error
        # print(f'cur_p_item:{cur_P * error},  cur_d_item:{kd * d_error}')
        new_steer = np.clip(steer, -max_control, max_control)
        return new_steer



def main(args=None):
    rclpy.init(args=args)
    print("Dummy Car Initialized")
    dummy_car_node = DummyCar()
    rclpy.spin(dummy_car_node)

    dummy_car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()