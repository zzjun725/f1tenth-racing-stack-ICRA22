#ifndef OPPONENT_PREDICTOR_NODE_H
#define OPPONENT_PREDICTOR_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int16.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

typedef std_msgs::msg::Int16 Int16;
typedef sensor_msgs::msg::LaserScan LaserScan;
typedef nav_msgs::msg::Odometry Odometry;
typedef geometry_msgs::msg::Pose Pose;
typedef geometry_msgs::msg::PoseStamped PoseStamped;
typedef geometry_msgs::msg::PoseArray PoseArray;
typedef visualization_msgs::msg::Marker Marker;
typedef visualization_msgs::msg::MarkerArray MarkerArray;

using namespace std;

class OpponentPredictor : public rclcpp::Node {
public:
    OpponentPredictor();

private:
    // General Variables
    bool real_test = true;

    // Global Map Variables
    int img_h = 0, img_w = 0;
    double map_resolution = 1.0;
    double origin_x = 0.0;
    double origin_y = 0.0;

    vector<cv::Point> inner_bound, outer_bound;

    // Local Map Variables
    vector<vector<vector<double>>> grid;
    vector<vector<double>> obstacle;
    vector<vector<double>> opponent;

    // Car State Variables
    vector<double> ego_global_pos;
    double ego_global_yaw = 0.0;

    vector<double> opp_global_pos;
    double opp_global_yaw = 0.0;

    // Time Variables
    short frame_cnt = 0;
    int scan_sec = 0;
    uint scan_nanosec = 0;

    // Topics
    string pose_topic;
    string scan_topic = "/scan";

    string fps_topic = "/fps";

    string opp_state_topic = "/opp_predict/state";
    string opp_bbox_topic = "/opp_predict/bbox";

    string grid_topic = "/grid";
    string obstacle_topic = "/obstacle";
    string opp_viz_pose_topic = "/opp_predict/viz/pose";
    string opp_viz_bbox_topic = "/opp_predict/viz/bbox";

    // Timers
    rclcpp::TimerBase::SharedPtr timer_;

    // Subscribers
    rclcpp::Subscription<Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<LaserScan>::SharedPtr scan_sub_;

    // Publishers
    rclcpp::Publisher<Int16>::SharedPtr fps_pub_;

    rclcpp::Publisher<PoseStamped>::SharedPtr opp_state_pub_;
    rclcpp::Publisher<PoseArray>::SharedPtr opp_bbox_pub_;

    rclcpp::Publisher<MarkerArray>::SharedPtr grid_pub_;
    rclcpp::Publisher<MarkerArray>::SharedPtr obstacle_pub_;
    rclcpp::Publisher<Marker>::SharedPtr opp_viz_pose_pub_;
    rclcpp::Publisher<MarkerArray>::SharedPtr opp_viz_bbox_pub_;

    // Member Functions
    void timer_callback();

    void scan_callback(const LaserScan::ConstSharedPtr scan_msg);

    void pose_callback(const Odometry::ConstSharedPtr pose_msg);

    void read_map(const string &map_name, const string &map_img_ext);

    static vector<cv::Point> read_bound(const string& file);

    vector<cv::Point2f> transform_coords_to_img(const vector<vector<double>>& path) const;

    void get_opponent();

    vector<vector<int>> cluster(const vector<vector<double>> &points, double tol);

    void connect(vector<int>& parents, int i, int j);

    int find(vector<int>& parents, int i);

    void visualize_occupancy_grid();

    void visualize_obstacle();

    void visualize_opponent_pose();

    void visualize_opponent_bbox();

    static void draw_bound(cv::Mat &image, const vector<cv::Point> &bound);

    static void show_result(const vector<cv::Mat>& images, const string& title);
};


#endif //OPPONENT_PREDICTOR_NODE_H
