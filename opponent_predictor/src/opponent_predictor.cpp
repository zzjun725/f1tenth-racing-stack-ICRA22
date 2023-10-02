#include <fstream>
#include <cmath>

#include "opponent_predictor.h"

OpponentPredictor::OpponentPredictor() : Node("opponent_predictor_node") {
    // ROS Params
    this->declare_parameter("debug_img");
    this->declare_parameter("visualize");
    this->declare_parameter("visualize_grid");
    this->declare_parameter("visualize_obstacle");
    this->declare_parameter("visualize_opp");
    this->declare_parameter("visualize_opp_pose");
    this->declare_parameter("visualize_opp_bbox");

    this->declare_parameter("real_test");
    this->declare_parameter("map_name");
    this->declare_parameter("map_img_ext");
    this->declare_parameter("resolution");
    this->declare_parameter("origin");

    this->declare_parameter("track_file");
    this->declare_parameter("inner_bound");
    this->declare_parameter("outer_bound");

    this->declare_parameter("grid_xmin");
    this->declare_parameter("grid_xmax");
    this->declare_parameter("grid_ymin");
    this->declare_parameter("grid_ymax");
    this->declare_parameter("grid_resolution");
    this->declare_parameter("plot_resolution");
    this->declare_parameter("grid_safe_dist");
    this->declare_parameter("goal_safe_dist");

    this->declare_parameter("cluster_dist_tol");
    this->declare_parameter("cluster_size_tol");
    this->declare_parameter("avoid_dist");

    // General Variables
    real_test = this->get_parameter("real_test").as_bool();

    if (real_test) {
        pose_topic = "/pf/pose/odom";
    } else {
        pose_topic = "/ego_racecar/odom";
    }

    // Global Map Variables
    string map_name = this->get_parameter("map_name").as_string();
    string map_img_ext = this->get_parameter("map_img_ext").as_string();
    read_map(map_name, map_img_ext);

    string inner_bound_path = this->get_parameter("inner_bound").as_string();
    inner_bound_path = "src/opponent_predictor/csv/" + map_name + "/" + inner_bound_path + ".csv";
    inner_bound = read_bound(inner_bound_path);

    string outer_bound_path = this->get_parameter("outer_bound").as_string();
    outer_bound_path = "src/opponent_predictor/csv/" + map_name + "/" + outer_bound_path + ".csv";
    outer_bound = read_bound(outer_bound_path);

    cout << inner_bound.size() << '\t' << outer_bound.size() << endl;

    // Timers
    timer_ = this->create_wall_timer(1s, std::bind(&OpponentPredictor::timer_callback, this));

    // Subscribers
    pose_sub_ = this->create_subscription<Odometry>(pose_topic, 1, std::bind(&OpponentPredictor::pose_callback, this,
                                                                             std::placeholders::_1));
    scan_sub_ = this->create_subscription<LaserScan>(scan_topic, 1, std::bind(&OpponentPredictor::scan_callback, this,
                                                                              std::placeholders::_1));

    // Publishers
    fps_pub_ = this->create_publisher<Int16>(fps_topic, 10);

    opp_state_pub_ = this->create_publisher<PoseStamped>(opp_state_topic, 10);
    opp_bbox_pub_ = this->create_publisher<PoseArray>(opp_bbox_topic, 10);

    grid_pub_ = this->create_publisher<MarkerArray>(grid_topic, 10);
    obstacle_pub_ = this->create_publisher<MarkerArray>(obstacle_topic, 10);
    opp_viz_pose_pub_ = this->create_publisher<Marker>(opp_viz_pose_topic, 10);
    opp_viz_bbox_pub_ = this->create_publisher<MarkerArray>(opp_viz_bbox_topic, 10);
}

void OpponentPredictor::timer_callback() {
    Int16 fps;
    fps.data = frame_cnt;
    frame_cnt = 0;
    fps_pub_->publish(fps);
    RCLCPP_INFO(this->get_logger(), "fps: %d", fps.data);
}

void OpponentPredictor::scan_callback(const LaserScan::ConstSharedPtr scan_msg) {
    vector<float> ranges(scan_msg->ranges);
    for (auto &range: ranges) {
        if (range < scan_msg->range_min) {
            range = scan_msg->range_min;
        } else if (range > scan_msg->range_max) {
            range = scan_msg->range_max;
        }
    }

    double xmin = this->get_parameter("grid_xmin").as_double();
    double xmax = this->get_parameter("grid_xmax").as_double();
    double ymin = this->get_parameter("grid_ymin").as_double();
    double ymax = this->get_parameter("grid_ymax").as_double();
    double resolution = this->get_parameter("grid_resolution").as_double();
    double grid_safe_dist = this->get_parameter("grid_safe_dist").as_double();

    int nx = int((xmax - xmin) / resolution) + 1;
    int ny = int((ymax - ymin) / resolution) + 1;

    double x_resolution = (xmax - xmin) / (nx - 1);
    double y_resolution = (ymax - ymin) / (ny - 1);

    // Discretize x and y
    vector<double> xs(nx), ys(ny);
    vector<double>::iterator ptr;
    double val;
    for (ptr = xs.begin(), val = xmin; ptr != xs.end(); ++ptr) {
        *ptr = val;
        val += x_resolution;
    }
    for (ptr = ys.begin(), val = ymin; ptr != ys.end(); ++ptr) {
        *ptr = val;
        val += y_resolution;
    }

    if (grid.empty()) {
        vector<vector<double>> grid_v(nx, vector<double>(ny, -1e8));
        vector<vector<double>> grid_x(nx, vector<double>(ny, -1e8));
        vector<vector<double>> grid_y(nx, vector<double>(ny, -1e8));

        grid.push_back(grid_v);
        grid.push_back(grid_x);
        grid.push_back(grid_y);
    }

    for (int i = 0; i < nx; ++i) {
        double x = xs[i];
        for (int j = 0; j < ny; ++j) {
            double y = ys[j];
            double rho = sqrt(x * x + y * y);
            double phi = atan2(y, x);
            int ray_idx = int((phi - scan_msg->angle_min) / scan_msg->angle_increment);

            grid[0][i][j] = (abs(rho - ranges[ray_idx]) < grid_safe_dist);
            grid[1][i][j] = x;
            grid[2][i][j] = y;
        }
    }

    // Record current time stamp
    scan_sec = scan_msg->header.stamp.sec;
    scan_nanosec = scan_msg->header.stamp.nanosec;
}

void OpponentPredictor::pose_callback(const Odometry::ConstSharedPtr pose_msg) {
    double curr_x = pose_msg->pose.pose.position.x;
    double curr_y = pose_msg->pose.pose.position.y;
    double quat_x = pose_msg->pose.pose.orientation.x;
    double quat_y = pose_msg->pose.pose.orientation.y;
    double quat_z = pose_msg->pose.pose.orientation.z;
    double quat_w = pose_msg->pose.pose.orientation.w;

    ego_global_pos = {curr_x, curr_y};
    ego_global_yaw = atan2(2 * (quat_w * quat_z + quat_x * quat_y),
                           1 - 2 * (quat_y * quat_y + quat_z * quat_z));

    // Find opponent
    get_opponent();

    // Visualization
    bool visualize = this->get_parameter("visualize").as_bool();
    bool visualize_grid = this->get_parameter("visualize_grid").as_bool();
    bool visualize_obs = this->get_parameter("visualize_obstacle").as_bool();
    bool visualize_opp_pose = this->get_parameter("visualize_opp_pose").as_bool();
    bool visualize_opp_bbox = this->get_parameter("visualize_opp_bbox").as_bool();

    if (visualize) {
        if (visualize_grid) {
            visualize_occupancy_grid();
        }
        if (visualize_obs) {
            visualize_obstacle();
        }
        if (visualize_opp_pose) {
            visualize_opponent_pose();
        }
        if (visualize_opp_bbox) {
            visualize_opponent_bbox();
        }
    }

    // Increase frame count
    frame_cnt++;
}

void OpponentPredictor::read_map(const string &map_name, const string &map_img_ext) {
    string map_img_path = "src/opponent_predictor/maps/" + map_name + map_img_ext;
    cv::Mat map_img = cv::imread(map_img_path);
    img_h = map_img.rows;
    img_w = map_img.cols;

    map_resolution = this->get_parameter("resolution").as_double();
    origin_x = this->get_parameter("origin").as_double_array()[0];
    origin_y = this->get_parameter("origin").as_double_array()[1];
}

vector<cv::Point> OpponentPredictor::read_bound(const string &file) {
    fstream fin;
    fin.open(file, ios::in);
    string line, word;

    vector<cv::Point> bound;

    while (getline(fin, line)) {
        stringstream s(line);
        vector<string> row;
        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        bound.emplace_back(stoi(row[0]), stoi(row[1]));
    }

    return bound;
}

vector<cv::Point2f> OpponentPredictor::transform_coords_to_img(const vector<vector<double>> &path) const {
    vector<cv::Point2f> new_path;
    for (const auto &pt: path) {
        double x = (pt[0] - origin_x) / map_resolution;
        double y = img_h - (pt[1] - origin_y) / map_resolution;
        new_path.emplace_back(x, y);
    }
    return new_path;
}

void OpponentPredictor::get_opponent() {
    // If laser scan not available yet, return
    if (grid.empty()) {
        return;
    }

    // Clear containers
    opp_global_pos.clear();
    obstacle.clear();
    opponent.clear();

    // Calculate grid points in map frame
    vector<vector<double>> grid_point;

    int nx = (int) grid[0].size();
    int ny = (int) grid[0][0].size();
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (grid[0][i][j] == 0.0) continue;
            double local_x = grid[1][i][j];
            double local_y = grid[2][i][j];

            double global_x = cos(ego_global_yaw) * local_x - sin(ego_global_yaw) * local_y + ego_global_pos[0];
            double global_y = sin(ego_global_yaw) * local_x + cos(ego_global_yaw) * local_y + ego_global_pos[1];

            grid_point.push_back({global_x, global_y});
        }
    }

    vector<cv::Point2f> grid_point_on_img = transform_coords_to_img(grid_point);

    // Find obstacle that is on the track
    vector<int> obstacle_idx;
    for (int i = 0; i < (int) grid_point_on_img.size(); ++i) {
        cv::Point2f curr_point = grid_point_on_img[i];
        double inside_outer_bound = pointPolygonTest(outer_bound, curr_point, false);
        if (inside_outer_bound != 1.0) {
            continue;
        }
        double outside_inner_bound = pointPolygonTest(inner_bound, curr_point, false);
        if (outside_inner_bound != -1.0) {
            continue;
        }
        obstacle_idx.push_back(i);
    }

    // Enable debug_img to visualize
    bool debug_img = this->get_parameter("debug_img").as_bool();
    if (debug_img) {
        cv::Mat dummy_img(img_w, img_h, CV_8UC3, cv::Scalar(255, 255, 255));
        draw_bound(dummy_img, outer_bound);
        draw_bound(dummy_img, inner_bound);
        for (int i = 0; i < (int) grid_point_on_img.size(); ++i) {
            auto itr = std::find(obstacle_idx.begin(), obstacle_idx.end(), i);
            if (itr != obstacle_idx.cend()) {
                cv::circle(dummy_img, grid_point_on_img[i], 1, cv::Scalar(0, 255, 0), 1);
            } else {
                cv::circle(dummy_img, grid_point_on_img[i], 1, cv::Scalar(255, 0, 0), 1);
            }
        }
        show_result({dummy_img}, "debug");
        exit(EXIT_FAILURE);
    }

    // If no opponent is found, return
    if (obstacle_idx.empty()) {
        PoseStamped opp_state;
        opp_state.pose.position.x = INFINITY;
        opp_state.pose.position.y = INFINITY;
        opp_state.header.stamp.sec = scan_sec;
        opp_state.header.stamp.nanosec = scan_nanosec;
        opp_state_pub_->publish(opp_state);

        PoseArray opp_bbox;
        opp_bbox.header.frame_id = "/map";
        opp_bbox.header.stamp.sec = scan_sec;
        opp_bbox.header.stamp.nanosec = scan_nanosec;
        for (const auto &pt: opponent) {
            Pose pose;
            pose.position.x = pt[0];
            pose.position.y = pt[1];
            opp_bbox.poses.push_back(pose);
        }
        opp_bbox_pub_->publish(opp_bbox);
        return;
    }

    // Find the largest cluster and large enough as opponent car
    for (const auto i: obstacle_idx) {
        obstacle.push_back(grid_point[i]);
    }

    double cluster_dist_tol = this->get_parameter("cluster_dist_tol").as_double();
    int cluster_size_tol = (int) this->get_parameter("cluster_size_tol").as_int();
    vector<vector<int>> clusters = cluster(obstacle, cluster_dist_tol);
    int max_cluster_size = 0;
    int max_cluster_idx = 0;
    for (int i = 0; i < (int) clusters.size(); ++i) {
        if ((int) clusters[i].size() > max_cluster_size) {
            max_cluster_size = (int) clusters[i].size();
            max_cluster_idx = i;
        }
    }

    if (max_cluster_size < cluster_size_tol) {
        PoseStamped opp_state;
        opp_state.pose.position.x = INFINITY;
        opp_state.pose.position.y = INFINITY;
        opp_state.header.stamp.sec = scan_sec;
        opp_state.header.stamp.nanosec = scan_nanosec;
        opp_state_pub_->publish(opp_state);

        return;
    }

    vector<int> opponent_idx = clusters[max_cluster_idx];
    for (const auto i: opponent_idx) {
        opponent.push_back(obstacle[i]);
    }

    // Use average pose for estimation
    double mean_x = 0.0, mean_y = 0.0;
    for (const auto &pt: opponent) {
        mean_x += pt[0];
        mean_y += pt[1];
    }
    mean_x /= double(opponent.size());
    mean_y /= double(opponent.size());
    cout << "opponent_size" << ": " << opponent.size() << endl;

    opp_global_pos = {mean_x, mean_y};

    // Publish result
    PoseStamped opp_state;
    opp_state.pose.position.x = opp_global_pos[0];
    opp_state.pose.position.y = opp_global_pos[1];
    opp_state.header.stamp.sec = scan_sec;
    opp_state.header.stamp.nanosec = scan_nanosec;
    opp_state_pub_->publish(opp_state);
    // cout << opp_global_pos[0] << ' ' << opp_global_pos[1] << endl;

    PoseArray opp_bbox;
    opp_bbox.header.frame_id = "/map";
    opp_bbox.header.stamp.sec = scan_sec;
    opp_bbox.header.stamp.nanosec = scan_nanosec;
    for (const auto &pt: opponent) {
        Pose pose;
        pose.position.x = pt[0];
        pose.position.y = pt[1];
        opp_bbox.poses.push_back(pose);
    }
    opp_bbox_pub_->publish(opp_bbox);
}
