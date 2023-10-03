This is the codebase we released for ICRA'22 F1tenth Competition. Check an exciting video in which we competed with the ETH Zurich team!

<video id="video" controls="" preload="none" poster="http://zzjun725.github.io/files/projects/icra.jpg">
      <source id="mp4" src="http://zzjun725.github.io/files/projects/icracut.mp4" type="video/mp4">
      </video>

![icra1](http://zzjun725.github.io/files/projects/icra_poster1.png)

![icra1](http://zzjun725.github.io/files/projects/icra_poster2.png) 

**Instructions**

1. Put maps(.png, .pgm, etc) and its configuration(.yaml) under `maps`
2. Edit `config/params.yaml` to include correct map name
3. Run `./scripts/populate.sh` to populate data files into subdirectories
4. Run `python3 trajectory_generator/lane_generator.py` to generate track data from image
5. Run `python3 trajectory_generator/main_globaltraj.py` to generate race line file
6. Run `./scripts/populate.sh` again to populate latest data files
7. Bring up simulation RVIZ(see [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros))
8. Run `ros2 launch opponent_predictor opponent_predictor_launch.py`
9. Run `ros2 launch lane_follow lane_follow_launch.py`
10. Run `ros2 launch dummy_car dummy_car_launch.py` (simulation only)



**ThirdParty Library**

raceline optimization: TUM [global raceline optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) 



**Attendees for ICRA'22 F1tenth Competition**

[Zhijun Zhuang](https://www.linkedin.com/in/zhijun-zhuang-01a140205/), [Jiatong Sun](https://www.linkedin.com/in/jiatong-sun/), [Pankti Hitesh Parekh](https://www.linkedin.com/in/panktiparekh/)

