echo "alias underlay='source /opt/ros/foxy/setup.bash'" >> /root/.bashrc
echo "alias overlay='source /f1tenth_ws/install/setup.bash'" >> /root/.bashrc
echo "alias sim='ros2 launch f1tenth_stack bringup_launch.py'" >> /root/.bashrc
echo "alias car5_ros='underlay && overlay && export ROS_DOMAIN_ID=47'" >> /root/.bashrc
echo "alias pf_sim='ros2 launch particle_filter localize_launch.py'" >> /root/.bashrc
echo "alias slam_sim='ros2 launch slam_toolbox online_async_launch.py params_file:=/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml'" >> /root/.bashrc
echo "alias nc_listen='nc -l 9899 | tar xvf - '" >> /root/.bashrc
echo "alias lane_follow='ros2 launch lane_follow lane_follow_launch.py'" >> /root/.bashrc
echo "alias oppo_pred='ros2 launch opponent_predictor opponent_predictor_launch.py'" >> /root/.bashrc
cp ./src/scripts/.tmux.conf /root/.tmux.conf
