#include <opencv2/highgui.hpp>

#include "opponent_predictor.h"

void OpponentPredictor::visualize_occupancy_grid() {
    if (grid.empty()) {
        return;
    }

    double grid_resolution = this->get_parameter("grid_resolution").as_double();
    double plot_resolution = this->get_parameter("plot_resolution").as_double();
    int down_sample = max(1, int(plot_resolution / grid_resolution));

    int nx = (int) grid[0].size();
    int ny = (int) grid[0][0].size();

    MarkerArray marker_arr;

    int id = 0;

    for (int i = 0; i < nx; ++i) {
        if (i % down_sample) continue;
        for (int j = 0; j < ny; ++j) {
            if (j % down_sample) continue;
            if (grid[0][i][j] == 0.0) continue;

            // Transform to map frame
            // Rotation
            double x = grid[1][i][j] * cos(ego_global_yaw) - grid[2][i][j] * sin(ego_global_yaw);
            double y = grid[1][i][j] * sin(ego_global_yaw) + grid[2][i][j] * cos(ego_global_yaw);

            // Translation
            x += ego_global_pos[0];
            y += ego_global_pos[1];

            // Add marker
            Marker marker;
            marker.header.frame_id = "/map";
            marker.id = i;
            marker.ns = "occupancy_grid_" + to_string(id++);
            marker.type = Marker::CUBE;
            marker.action = Marker::ADD;

            marker.pose.position.x = x;
            marker.pose.position.y = y;

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;

            marker.lifetime.nanosec = int(1e8);

            marker_arr.markers.push_back(marker);
        }
    }

    grid_pub_->publish(marker_arr);
}

void OpponentPredictor::visualize_obstacle() {
    if (obstacle.empty()) {
        return;
    }

    MarkerArray marker_arr;
    for (int i = 0; i < (int) obstacle.size(); ++i) {
        Marker marker;
        marker.header.frame_id = "/map";
        marker.id = i;
        marker.ns = "obstacle_" + to_string(i);
        marker.type = Marker::CUBE;
        marker.action = Marker::ADD;

        marker.pose.position.x = obstacle[i][0];
        marker.pose.position.y = obstacle[i][1];

        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        marker.color.a = 1.0;

        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;

        marker.lifetime.nanosec = int(1e8);

        marker_arr.markers.push_back(marker);
    }

    obstacle_pub_->publish(marker_arr);
}

void OpponentPredictor::visualize_opponent_pose() {
    if (opp_global_pos.empty()) {
        return;
    }

    Marker marker;
    marker.header.frame_id = "/map";
    marker.id = 0;
    marker.ns = "opponent_pose";
    marker.type = Marker::CUBE;
    marker.action = Marker::ADD;

    marker.pose.position.x = opp_global_pos[0];
    marker.pose.position.y = opp_global_pos[1];

    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;

    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;

    marker.lifetime.nanosec = int(1e8);

    opp_viz_pose_pub_->publish(marker);
}

void OpponentPredictor::visualize_opponent_bbox() {
    if (opponent.empty()) {
        return;
    }

    MarkerArray marker_arr;
    for (int i = 0; i < (int) opponent.size(); ++i) {
        Marker marker;
        marker.header.frame_id = "/map";
        marker.id = i;
        marker.ns = "opponent_" + to_string(i);
        marker.type = Marker::CUBE;
        marker.action = Marker::ADD;

        marker.pose.position.x = opponent[i][0];
        marker.pose.position.y = opponent[i][1];

        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        marker.color.a = 1.0;

        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;

        marker.lifetime.nanosec = int(1e8);

        marker_arr.markers.push_back(marker);
    }

    opp_viz_bbox_pub_->publish(marker_arr);
}

void OpponentPredictor::draw_bound(cv::Mat &image, const vector<cv::Point> &bound) {
    int n_pts = (int) bound.size();
    for (int i = 0; i < n_pts - 1; ++i) {
        cv::line(image, bound[i], bound[i + 1], cv::Scalar(0, 0, 255), 1);
    }
    cv::line(image, bound[n_pts - 1], bound[0], cv::Scalar(0, 0, 255), 1);
}

void OpponentPredictor::show_result(const vector<cv::Mat> &images, const string &title) {
    if (images.empty()) {
        return;
    }

    int height = images[0].rows;
    int width = images[0].cols;
    int w_show = 800;
    double scale_percent = double(w_show) / double(width);
    int h_show = int(scale_percent * height);

    vector<cv::Mat> images_resize;
    for (const auto &image: images) {
        cv::Mat image_resize;
        cv::resize(image, image_resize, cv::Size(w_show, h_show));
        images_resize.push_back(image_resize);
    }

    cv::Mat image_show;
    cv::hconcat(images_resize, image_show);
    cv::imshow(title, image_show);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
