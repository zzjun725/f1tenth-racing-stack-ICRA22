This is the codebase we released for our ICRA'22 F1tenth Competition. Check an exciting [video](http://zzjun725.github.io/files/projects/icracut.mp4) in which we competed with the ETH Zurich team! 

# Overview

We are using pure-pursuit controller and a lane switcher for overtaking/obstacle avoidance.

![icra1](http://zzjun725.github.io/files/projects/icra_poster1.png)

![icra1](http://zzjun725.github.io/files/projects/icra_poster2.png) 

# Installation

### Download the package

To use the stack in your workspace, you need to put them inside the `src` folder so the structure of your workspace should be like this:

```
├── build
├── install
├── log
└── src
    ├── config
    ├── csv
    ├── dummy_car
    ├── trajectory_generator
    ├── lane_follow
    ├── maps
    ├── opponent_predictor
    ├── scripts
    └── <YOUR OTHER PACKAGES>
```



### Create virtual environment and install the dependencies

Note: This package use the TUM [global raceline optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) which requires specific version of numpy and sklearn packages which can be incompatible with other application. So, it is better to use `Anaconda3`/`venv` to create a virtual environment specific for using the raceline optimization. Here, we use the `venv` package in python.

- Go inside the `src` folder: `cd ./src`

- Install venv:  `sudo apt install python3.8-venv`

- Create virtual environment, exclude it from the colcon build and source the virtual environment: `python3 -m venv ./venv && touch ./venv/COLCON_IGNORE && source ./venv/bin/activate  ` 

- Install the dependencies: `pip install -r trajectory_generator/requirements.txt`



### Process the map and generate the optimal raceline 

Before you execute the following steps, make sure you are inside the `./src` folder and the virtual environment is activated with: `source ./venv/bin/activate`.

- Put maps(.png, .pgm, etc) and its configuration(.yaml) under `maps`
- Edit `config/params.yaml` to include correct map name
- Run `./scripts/populate.sh` to populate data files into subdirectories
- Run `python3 trajectory_generator/lane_generator.py` to generate track data from image(Click the pop-up image, Press any keys( `Enter`, for example) to continue).
- Run `python3 trajectory_generator/main_globaltraj.py` to generate race line file.
- Run `python3 trajectory_generator/raceline_scripts.py ` to visualize the curvature of the generated optimal raceline.
- Run `./scripts/populate.sh` again to populate generated raceline into subdirectories



### Use the generated raceline for lane_follow(single agent)

You should do `colcon build` and launch the node from the workspace folder `./` as normal, not inside the `src`.

- Bring up simulation RVIZ(see [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros)): `ros2 launch f1tenth_gym_ros gym_bridge_launch.py`

- Run `ros2 launch lane_follow lane_follow_launch.py`

You can load the config file `./config/race.rviz` for rviz2. The global raceline is published under topic`/global_path/optimal_trajectory`, and the target of pure pursuit is published under topic `/waypoint`.



**ThirdParty Library**

raceline optimization: TUM [global raceline optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) 



**Attendees for ICRA'22 F1tenth Competition**

[Zhijun Zhuang](https://www.linkedin.com/in/zhijun-zhuang-01a140205/), [Jiatong Sun](https://www.linkedin.com/in/jiatong-sun/), [Pankti Hitesh Parekh](https://www.linkedin.com/in/panktiparekh/)

