#include <rclcpp/rclcpp.hpp>
#include "opponent_predictor.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpponentPredictor>());
    rclcpp::shutdown();
    return 0;
}