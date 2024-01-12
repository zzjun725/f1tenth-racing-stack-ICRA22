- Sim / Real

In `./config/params.yaml`: set `real_test` to False if used in simulation, to True if used in real car.

- Use your own map

In `./config/params.yaml`: set `map_name`  and `map_img_ext` as your desired map name and extension.

Then, put the map file under `./maps` folder, put the `lane_optimal.csv` file under `./csv/<YOUR_MAP_NAME>`. 

- Adjust the parameters

All the parameters related to the controller(liked the Kp, Kd, the look ahead distance) can be adjust accordingly in `./launch/pure_pursuit_launch.py`

Then, do `colcon build` to update the change. 
