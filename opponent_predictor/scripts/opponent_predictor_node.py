#!/usr/bin/env python3
import numpy as np
import math
from PIL import Image
import os
import yaml
import cv2
from time import time
from scipy.spatial import distance

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int16


class OpponentPredictor(Node):
    """
    Implement opponent predictor on the car
    """

    def __init__(self):
        super().__init__("opponent_predictor_node")

        # ROS Params
        self.declare_parameter("debug_img")
        self.declare_parameter("visualize")
        self.declare_parameter("visualize_grid")
        self.declare_parameter("visualize_obstacle")
        self.declare_parameter("visualize_opp")
        self.declare_parameter("visualize_opp_pose")
        self.declare_parameter("visualize_opp_bbox")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("map_img_ext")

        self.declare_parameter("track_file")
        self.declare_parameter("inner_bound")
        self.declare_parameter("outer_bound")

        self.declare_parameter("grid_xmin")
        self.declare_parameter("grid_xmax")
        self.declare_parameter("grid_ymin")
        self.declare_parameter("grid_ymax")
        self.declare_parameter("grid_resolution")
        self.declare_parameter("plot_resolution")
        self.declare_parameter("grid_safe_dist")
        self.declare_parameter("goal_safe_dist")

        self.declare_parameter("cluster_dist_tol")
        self.declare_parameter("cluster_size_tol")
        self.declare_parameter("avoid_dist")

        # Global Map Variables
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        map_name = self.get_parameter("map_name").get_parameter_value().string_value
        map_img_ext = self.get_parameter("map_img_ext").get_parameter_value().string_value

        self.map, self.map_metadata = self.read_map(map_name, map_img_ext)

        track_file = self.get_parameter("track_file").get_parameter_value().string_value
        inner_bound = self.get_parameter("inner_bound").get_parameter_value().string_value
        outer_bound = self.get_parameter("outer_bound").get_parameter_value().string_value

        track_file = os.path.join("src", "opponent_predictor", "csv", map_name, track_file + ".npy")
        inner_bound = os.path.join("src", "opponent_predictor", "csv", map_name, inner_bound + ".npy")
        outer_bound = os.path.join("src", "opponent_predictor", "csv", map_name, outer_bound + ".npy")

        self.track = np.load(track_file)
        self.inner_bound = np.load(inner_bound)
        self.outer_bound = np.load(outer_bound)

        # Local Map Variables
        self.grid = None
        self.obstacle = None  # subtract wall from grid
        self.opponent = None  # occupancy grid of opponent car

        # Ego Car State Variables
        self.ego_global_pos = None
        self.ego_global_yaw = None

        # Opponent Car State Variables
        self.opp_global_pos = None
        self.opp_global_yaw = None

        # Other Variables
        self.frame_cnt = 0

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        scan_topic = "/scan"

        grid_topic = "/grid"
        obstacle_topic = "/obstacle"
        opp_state_topic = "/opp_predict/state"
        opp_bbox_topic = "/opp_predict/bbox"
        opp_viz_pose_topic = "/opp_predict/viz/pose"
        opp_viz_bbox_topic = "/opp_predict/viz/bbox"
        fps_topic = "/fps"

        self.timer = self.create_timer(1.0, self.timer_callback)

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)

        self.scan_sub_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 1)

        self.grid_pub_ = self.create_publisher(MarkerArray, grid_topic, 10)
        self.obstacle_pub_ = self.create_publisher(MarkerArray, obstacle_topic, 10)
        self.fps_pub_ = self.create_publisher(Int16, fps_topic, 10)
        # self.opp_state_pub_ = self.create_publisher(Pose, opp_state_topic, 10)
        self.opp_state_pub_ = self.create_publisher(PoseStamped, opp_state_topic, 10)
        self.opp_bbox_pub_ = self.create_publisher(PoseArray, opp_bbox_topic, 10)
        self.opp_viz_pose_pub_ = self.create_publisher(Marker, opp_viz_pose_topic, 10)
        self.opp_viz_bbox_pub_ = self.create_publisher(MarkerArray, opp_viz_bbox_topic, 10)

        # timestamped
        self.scan_timestamped_nanosec = None
        self.scan_timestamped_sec = None

    def timer_callback(self):
        fps = Int16()
        fps.data = self.frame_cnt
        self.frame_cnt = 0
        self.fps_pub_.publish(fps)
        self.get_logger().info("fps: %d" % fps.data)

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args:
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        ranges = np.array(scan_msg.ranges)
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)

        xmin = self.get_parameter("grid_xmin").get_parameter_value().double_value
        xmax = self.get_parameter("grid_xmax").get_parameter_value().double_value
        ymin = self.get_parameter("grid_ymin").get_parameter_value().double_value
        ymax = self.get_parameter("grid_ymax").get_parameter_value().double_value
        resolution = self.get_parameter("grid_resolution").get_parameter_value().double_value
        grid_safe_dist = self.get_parameter("grid_safe_dist").get_parameter_value().double_value

        nx = int((xmax - xmin) / resolution) + 1
        ny = int((ymax - ymin) / resolution) + 1

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        y, x = np.meshgrid(y, x)
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        ray_idx = ((phi - scan_msg.angle_min) / scan_msg.angle_increment).astype(int)
        obs_rho = ranges[ray_idx]

        self.grid = np.where(np.abs(rho - obs_rho) < grid_safe_dist, 1.0, 0.0)  # 1: occupied  0: free

        # cut angle
        # self.grid = np.where(-0.8 < phi, self.grid, 0.0)
        # self.grid = np.where(phi < 0.8, self.grid, 0.0)

        self.grid = np.dstack((self.grid, x, y))  # (h, w, 3)

        # current time
        # cur_time = scan_msg.header.stamp.nanosec/1e9 + scan_msg.header.stamp.sec
        self.scan_timestamped_nanosec = scan_msg.header.stamp.nanosec
        self.scan_timestamped_sec = scan_msg.header.stamp.sec

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter"s inferred pose

        Args:
            pose_msg (PoseStamped or Odometry): incoming message from subscribed topic
        Returns:

        """
        # Read current state
        if self.real_test:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_quat = pose_msg.pose.pose.orientation

        self.ego_global_pos = np.array([curr_x, curr_y])
        self.ego_global_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                                         1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        # Find opponent
        self.get_opponent()

        # Visualize waypoint
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        visualize_grid = self.get_parameter("visualize_grid").get_parameter_value().bool_value
        visualize_obstacle = self.get_parameter("visualize_obstacle").get_parameter_value().bool_value
        visualize_opp_pose = self.get_parameter("visualize_opp_pose").get_parameter_value().bool_value
        visualize_opp_bbox = self.get_parameter("visualize_opp_bbox").get_parameter_value().bool_value
        if visualize:
            if visualize_grid:
                self.visualize_occupancy_grid()
            if visualize_obstacle:
                self.visualize_obstacle()
            if visualize_opp_pose:
                self.visualize_opponent_pose()
            if visualize_opp_bbox:
                self.visualize_opponent_bbox()

        # Increase frame count
        self.frame_cnt += 1

        return None

    @staticmethod
    def read_map(map_name, map_img_ext):
        map_img_path = os.path.splitext(map_name)[0] + map_img_ext
        map_img_path = os.path.join("src", "opponent_predictor", "maps", map_img_path)

        map_cfg_path = os.path.splitext(map_name)[0] + ".yaml"
        map_cfg_path = os.path.join("src", "opponent_predictor", "maps", map_cfg_path)

        map_img = Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)
        map_img = np.asarray(map_img).astype(np.float64)
        map_img[map_img <= 128.] = 0.
        map_img[map_img > 128.] = 255.

        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # load map yaml
        map_metadata = yaml.safe_load(open(map_cfg_path, 'r'))
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']
        origin_x = origin[0]
        origin_y = origin[1]

        image_xs, image_ys = np.meshgrid(np.arange(map_width), np.arange(map_height))
        map_xs = image_xs * map_resolution + origin_x
        map_ys = image_ys * map_resolution + origin_y

        map_vs = np.where(map_img > 0, 0.0, 1.0)  # 1: occupied  0: free

        return np.dstack((map_vs, map_xs, map_ys)), (map_height, map_width, map_resolution, origin_x, origin_y)

    @staticmethod
    def transform_coords_to_img(path, height, s, tx, ty):
        new_path_x = (path[:, 0] - tx) / s
        new_path_y = height - (path[:, 1] - ty) / s
        return np.vstack((new_path_x, new_path_y)).T.astype(np.int32)

    def get_opponent(self):
        if self.grid is None:
            return
        # self.visualize_occupancy_grid()
        # Calculate wall points
        map_v = self.map[:, :, 0].flatten()
        map_x = self.map[:, :, 1].flatten()
        map_y = self.map[:, :, 2].flatten()

        map_x = map_x[map_v > 0]
        map_y = map_y[map_v > 0]
        map_point = np.vstack((map_x, map_y)).T

        # Calculate grid points in map frame
        R = np.array([[np.cos(self.ego_global_yaw), -np.sin(self.ego_global_yaw)],
                      [np.sin(self.ego_global_yaw), np.cos(self.ego_global_yaw)]])

        grid_v = self.grid[:, :, 0].flatten()
        grid_x = self.grid[:, :, 1].flatten()
        grid_y = self.grid[:, :, 2].flatten()

        grid_x = grid_x[grid_v > 0]
        grid_y = grid_y[grid_v > 0]
        grid_x, grid_y = R @ np.vstack((grid_x, grid_y)) + self.ego_global_pos.reshape(-1, 1)
        grid_point = np.vstack((grid_x, grid_y)).T

        grid_point_on_img = self.transform_coords_to_img(grid_point,
                                                         self.map_metadata[0],
                                                         self.map_metadata[2],
                                                         self.map_metadata[3],
                                                         self.map_metadata[4])

        # Find obstacle that is on the track
        obstacle_idx = []
        for idx in range(len(grid_point_on_img)):
            x = int(grid_point_on_img[idx, 0])
            y = int(grid_point_on_img[idx, 1])
            inside_outer_bound = cv2.pointPolygonTest(self.outer_bound, (x, y), False)
            if inside_outer_bound != 1:
                continue
            outside_inner_bound = cv2.pointPolygonTest(self.inner_bound, (x, y), False)
            if outside_inner_bound != -1:
                continue
            obstacle_idx.append(idx)

        # Enable debug_img to visualize
        debug_img = self.get_parameter("debug_img").get_parameter_value().bool_value
        if debug_img:
            dummy_img = np.ones((self.map_metadata[0], self.map_metadata[1], 3))
            for i in range(len(self.outer_bound)):
                cv2.line(dummy_img, self.outer_bound[i - 1], self.outer_bound[i], (0, 0, 255), 1)
            for i in range(len(self.inner_bound)):
                cv2.line(dummy_img, self.inner_bound[i - 1], self.inner_bound[i], (0, 0, 255), 1)
            for idx in range(len(grid_point_on_img)):
                color = (0, 255, 0) if idx in obstacle_idx else (255, 0, 0)
                cv2.circle(dummy_img, grid_point_on_img[idx], 1, color, 1)
            self.show_result([dummy_img], title="debug")
            exit(0)

        # If no opponent is found, return
        if len(obstacle_idx) == 0:
            self.opp_global_pos = None
            self.opponent = None
            opp_state = PoseStamped()
            opp_state.pose.position.x = np.inf
            opp_state.pose.position.y = np.inf
            opp_state.header.stamp.nanosec = self.scan_timestamped_nanosec
            opp_state.header.stamp.sec = self.scan_timestamped_sec
            self.opp_state_pub_.publish(opp_state)

            opp_bbox = PoseArray()
            self.opp_bbox_pub_.publish(opp_bbox)
            return
        self.obstacle = grid_point[obstacle_idx]
        # print(f'obstacle_grid_num: {len(obstacle_idx)}')

        # Find the largest cluster as opponent car
        cluster_dist_tol = self.get_parameter("cluster_dist_tol").get_parameter_value().double_value
        cluster_size_tol = self.get_parameter("cluster_size_tol").get_parameter_value().double_value
        clusters = self.cluster(self.obstacle, cluster_dist_tol)
        sizes = [len(cluster) for cluster in clusters]
        # print(f'cluster num: {len(clusters)}')
        if max(sizes) < cluster_size_tol:
            self.opp_global_pos = None
            self.opponent = None
            opp_state = PoseStamped()
            opp_state.position.x = np.inf
            opp_state.position.y = np.inf
            opp_state.header.stamp.nanosec = self.scan_timestamped_nanosec
            opp_state.header.stamp.sec = self.scan_timestamped_sec
            self.opp_state_pub_.publish(opp_state)
            return
        opponent_idx = clusters[np.argmax(sizes)]
        self.opponent = self.obstacle[opponent_idx]
        print(f'opponent_grid_num: {len(opponent_idx)}')

        # Use the point closest to ego car as opponent position
        opponent_dist = np.linalg.norm(self.opponent - self.ego_global_pos, axis=1)

        ### ignore far obstacle ###
        # avoid_dist = self.get_parameter('avoid_dist').get_parameter_value().double_value
        # if opponent_dist.any() > avoid_dist:
        #     return
        ### ignore far obstacle ###

        # self.opp_global_pos = self.opponent[np.argmin(opponent_dist)]

        #### global pos calculate ###
        self.opp_global_pos = np.mean(self.opponent, axis=0).flatten()
        # print(self.opponent)

        # Publish opponent state
        # opp_state = Pose()
        # opp_state.position.x = self.opp_global_pos[0]
        # opp_state.position.y = self.opp_global_pos[1]
        # self.opp_state_pub_.publish(opp_state)
        opp_state = PoseStamped()
        opp_state.pose.position.x = self.opp_global_pos[0]
        opp_state.pose.position.y = self.opp_global_pos[1]
        opp_state.header.stamp.nanosec = self.scan_timestamped_nanosec
        opp_state.header.stamp.sec = self.scan_timestamped_sec
        # print(f'cur_time: {opp_state.header.stamp.nanosec/1e9 + opp_state.header.stamp.sec}')
        self.opp_state_pub_.publish(opp_state)

        opp_bbox = PoseArray()
        for pt in self.opponent:
            pose = Pose()
            pose.position.x = pt[0]
            pose.position.y = pt[1]
            opp_bbox.header.frame_id = "/map"
            opp_bbox.poses.append(pose)
        self.opp_bbox_pub_.publish(opp_bbox)

    def cluster(self, points, tol):
        n = len(points)
        parents = [i for i in range(n)]
        count = n

        c = distance.cdist(points, points)
        for i in range(n):
            for j in range(i):
                dist = c[i, j]
                if dist > tol:
                    continue
                if self.union(parents, i, j):
                    count -= 1

        clusters = {}
        for i in range(n):
            root = parents[i]
            if root not in clusters.keys():
                clusters[root] = [root]
            else:
                clusters[root].append(i)

        return list(clusters.values())

    def union(self, parents, i, j):
        root_i = self.find(parents, i)
        root_j = self.find(parents, j)

        if root_i != root_j:
            parents[root_i] = root_j
            return True

        return False

    def find(self, parents, i):
        if parents[i] == i:
            return i
        parents[i] = self.find(parents, parents[i])
        return parents[i]

    def visualize_occupancy_grid(self):
        if self.grid is None:
            return

        grid_resolution = self.get_parameter("grid_resolution").get_parameter_value().double_value
        plot_resolution = self.get_parameter("plot_resolution").get_parameter_value().double_value
        down_sample = max(1, int(plot_resolution / grid_resolution))

        grid = self.grid.copy()
        grid = grid[::down_sample, ::down_sample, :]  # down sample for faster plotting

        grid_v = grid[:, :, 0].flatten()
        grid_x = grid[:, :, 1].flatten()
        grid_y = grid[:, :, 2].flatten()

        # Transform occupancy grid into map frame
        pos = np.vstack((grid_x.flatten(), grid_y.flatten()))
        R = np.array([[np.cos(self.ego_global_yaw), -np.sin(self.ego_global_yaw)],
                      [np.sin(self.ego_global_yaw), np.cos(self.ego_global_yaw)]])
        grid_x, grid_y = R @ pos + self.ego_global_pos.reshape(-1, 1)

        # Publish occupancy grid
        marker_arr = MarkerArray()
        # print(f'occu grid num {len(np.nonzero(grid_v)[0])}')
        for i in range(len(grid_v)):
            if grid_v[i] == 0:
                continue

            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = i
            marker.ns = "occupancy_grid_%u" % i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = grid_x[i]
            marker.pose.position.y = grid_y[i]

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.lifetime.nanosec = int(1e8)

            marker_arr.markers.append(marker)

        self.grid_pub_.publish(marker_arr)

    def visualize_obstacle(self):
        if self.obstacle is None:
            return

        marker_arr = MarkerArray()

        for i, pt in enumerate(self.obstacle):
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = i
            marker.ns = "obstacle_%u" % i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = pt[0]
            marker.pose.position.y = pt[1]

            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.lifetime.nanosec = int(1e8)

            marker_arr.markers.append(marker)

        self.obstacle_pub_.publish(marker_arr)

    def visualize_opponent_pose(self):
        if self.opp_global_pos is None:
            return

        # Publish opponent position
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "opponent_pose"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = float(self.opp_global_pos[0])
        marker.pose.position.y = float(self.opp_global_pos[1])

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        marker.lifetime.nanosec = int(1e8)

        self.opp_viz_pose_pub_.publish(marker)

    def visualize_opponent_bbox(self):
        if self.opponent is None:
            return

        marker_arr = MarkerArray()

        for i, pt in enumerate(self.opponent):
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = i
            marker.ns = "opponent_%u" % i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = pt[0]
            marker.pose.position.y = pt[1]

            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.lifetime.nanosec = int(1e8)

            marker_arr.markers.append(marker)

        self.opp_viz_bbox_pub_.publish(marker_arr)

    @staticmethod
    def show_result(imgs, title):
        if not imgs:
            return False
        height, width = imgs[0].shape[:2]
        w_show = 800
        scale_percent = float(w_show / width)
        h_show = int(scale_percent * height)
        dim = (w_show, h_show)
        img_resizes = []
        for img in imgs:
            img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img_resizes.append(img_resize)
        img_show = cv2.hconcat(img_resizes)
        cv2.imshow(title, img_show)

        print("Press Q to abort / other keys to proceed")
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            return False
        else:
            cv2.destroyAllWindows()
            return True


def main(args=None):
    rclpy.init(args=args)
    opponent_predictor_node = OpponentPredictor()
    rclpy.spin(opponent_predictor_node)

    opponent_predictor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
