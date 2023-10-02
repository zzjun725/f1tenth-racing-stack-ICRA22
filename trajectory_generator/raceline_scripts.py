import os
import csv
from turtle import ycor
from unicodedata import name
import fire
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import numpy.linalg as LA
# import seaborn as sns
import cv2
import yaml

module = os.path.dirname(os.path.abspath(__file__))
config_file = module + "/config/params.yaml"
with open(config_file, 'r') as stream:
    parsed_yaml = yaml.safe_load(stream)
input_map = parsed_yaml["map_name"]
input_map_ext = parsed_yaml["map_img_ext"]

cur_map_path = os.path.join(module, 'maps', input_map + input_map_ext)
src_map_path = os.path.join(os.path.split(module)[0], 'maps', input_map + input_map_ext)

cur_csv_dir = os.path.join(module, 'outputs', input_map)
src_csv_dir = os.path.join(os.path.split(module)[0], 'csv', input_map)
globalWpMinInterval = 0.1


def scale_map(mapname=None, scale_percent=200, backup_exist=True):
    if not mapname:
        mapname = input_map
    map_path = os.path.join(module, 'maps', input_map + input_map_ext)
    back_map_path = os.path.join(module, 'maps', input_map + '_backup' + input_map_ext)
    if not backup_exist:
        os.system(f'cp {map_path} {back_map_path}')
    # raw_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    raw_map = cv2.imread(back_map_path, cv2.IMREAD_UNCHANGED)
    # scale_percent = 130  # percent of original size
    width = int(raw_map.shape[1] * scale_percent / 100)
    height = int(raw_map.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(raw_map, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(map_path, resized)
    print(f'scale map {input_map} at {scale_percent / 100}')


def smooth_waypoints(filename='race_v1.csv', gap_thres=0.4, interpolate_times=2):
    if filename:
        output_name = filename.split('.')[0] + '_smooth' + '.csv'
        with open(os.path.join(cur_csv_dir, filename)) as f:
            waypoints = csv.reader(f)
            new_waypoints = []
            # import ipdb; ipdb.set_trace()
            for i, row in enumerate(waypoints):
                # row = row.split(',')
                if i == 1:
                    first_point = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                    x_old, y_old = float(row[0]), float(row[1])
                    r_old, l_old = float(row[2]), float(row[3])
                    new_waypoints.append(np.array([x_old, y_old, r_old, l_old]))
                    continue
                if i > 1:
                    x, y = float(row[0]), float(row[1])
                    r, l = float(row[2]), float(row[3])
                    if abs(x - x_old) + abs(y - y_old) > globalWpMinInterval:
                        if np.linalg.norm(np.array([x, y]) - np.array([x_old, y_old])) > gap_thres:
                            interp_points = [np.array([x_old, y_old, r_old, l_old]), np.array([x, y, r, l])]
                            for _ in range(interpolate_times):
                                i = 0
                                n = len(interp_points)
                                while i < n:
                                    interp_points.insert(i + 1, (interp_points[i] + interp_points[i + 1]) / 2)
                                    i += 2
                            new_waypoints.extend(interp_points[1:])
                            # import ipdb; ipdb.set_trace()
                        else:
                            new_waypoints.append(np.array([x, y, r, l]))
                        x_old, y_old, r_old, l_old = x, y, r, l
            ## check the end point with the first
            # print(f'origin wp number{i-1}')
            # import ipdb; ipdb.set_trace()
            if np.linalg.norm(np.array([x_old, y_old]) - np.array(first_point)[:2]) > gap_thres:
                interp_points = [np.array([x_old, y_old, r_old, l_old]), np.array([first_point])]
                for _ in range(interpolate_times):
                    i = 0
                    n = len(interp_points)
                    while i < n:
                        interp_points.insert(i + 1, (interp_points[i] + interp_points[i + 1]).squeeze() / 2)
                        i += 2
                new_waypoints.extend(interp_points[1:-1])
            # import ipdb; ipdb.set_trace()
            print(f'new wp number{len(new_waypoints)}')
        with open(os.path.join(cur_csv_dir, output_name), 'w') as f:
            # f.write(f'# x_m,y_m,w_tr_right_m,w_tr_left_m\n')
            for waypoint in new_waypoints:
                x, y, r, l = waypoint
                f.write('%f, %f, %f, %f\n' % (x, y, r, l))
            f.close


def get_smooth_lane(shrink_dist=0.2, option='center', original_wp='centerline.csv', backup_exist=True):
    wp_x = []
    wp_y = []
    wp_r = []
    wp_l = []

    original_wp_f = os.path.join(cur_csv_dir, original_wp)
    backup_wp_f = os.path.join(cur_csv_dir, 'centerline_ori.csv')
    if not backup_exist:
        os.system(f'cp {original_wp_f} {backup_wp_f}')
    # new_wp_f = open(os.path.join(cur_csv_dir, option+'_lane.csv'))
    new_wp_f = open(original_wp_f, 'w')
    if option == 'center' or option == 'optimal':
        l_shrink = shrink_dist
        r_shrink = shrink_dist
    elif option == 'left':
        l_shrink = shrink_dist
        r_shrink = 0.0
    else:
        l_shrink = 0.0
        r_shrink = shrink_dist

    with open(backup_wp_f) as f:
        print(f'load {backup_wp_f}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            try:
                x = float(row[0])
            except:
                new_wp_f.write(row[0] + '\n')
                # print(row)
                continue
            x_, y_ = float(row[0]), float(row[1])
            r_, l_ = float(row[2]), float(row[3])
            # if abs(x_ - last_x) + abs(y_ - last_y) > globalWpMinInterval:
            wp_x.append(x_)
            wp_y.append(y_)
            wp_r.append(r_)
            wp_l.append(l_)
            # last_x = x_
            # last_y = y_
            new_wp_f.write('%4f, %4f, %4f, %4f\n' % (x_, y_, r_ - r_shrink, l_ - l_shrink))
    new_wp_f.close()
    globaltraj_path = os.path.join(module, 'main_globaltraj.py')
    os.system(f'python3 {globaltraj_path}')
    # import time
    # time.sleep(6)
    try:
        output_optimal_raceline_path = os.path.join(cur_csv_dir, 'traj_race_cl.csv')
        new_lane_path = os.path.join(cur_csv_dir, 'lane_' + option + '.csv')
        lane_f = open(new_lane_path, 'w')
        lane_f.write('# x/m \n')
        with open(output_optimal_raceline_path) as f:
            print(f'load {output_optimal_raceline_path}')
            waypoints_file = csv.reader(f)
            for i, row in enumerate(waypoints_file):
                # import ipdb;
                # ipdb.set_trace()
                if i > 2:
                    row = row[0].split(';')
                    x_ = float(row[1])
                    y_ = float(row[2])
                    v_ = float(row[5])
                    lane_f.write('%4f, %4f, %4f\n' % (x_, y_, v_))
        lane_f.close()
        # os.system(f'cp {output_optimal_raceline_path} {new_lane_path}')
        print('succeed generate smooth_new_lane')
    except:
        print('fail generate raceline')


def draw_lanes(lane_l='lane_left.csv', lane_r='lane_right.csv', lane_center='lane_center.csv'):
    xs, ys = [], []
    for filename in (lane_l, lane_r, lane_center):
        x = []
        y = []
        with open(os.path.join(cur_csv_dir, filename)) as f:
            print(f'load {filename}')
            waypoints_file = csv.reader(f)
            for i, row in enumerate(waypoints_file):
                try:
                    _ = float(row[0])
                except:
                    continue
                x.append(float(row[0]))
                y.append(float(row[1]))
        xs.append(deepcopy(x))
        ys.append(deepcopy(y))
    for x, y, draw_opt, label in zip(xs, ys, ('-bo', '-ro', '-ko'), ('left', 'right', 'center')):
        plt.plot(x, y, draw_opt, markersize=0.5, label=label)
    plt.axis('equal')
    plt.legend()
    plt.show()


def draw_optimalwp(filename='traj_race_cl.csv', ref_filename='centerline.csv'):
    x = []
    y = []

    with open(os.path.join(cur_csv_dir, filename)) as f:
        print(f'load {filename}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            # import ipdb;
            # ipdb.set_trace()
            if i > 2:
                row = row[0].split(';')
                x.append(float(row[1]))
                y.append(float(row[2]))
    print(f'length of optimal{len(x)}')
    plt.plot(x, y, '-bo', markersize=0.5, label='optimal raceline')

    ref_x = []
    ref_y = []

    with open(os.path.join(cur_csv_dir, ref_filename)) as f:
        print(f'load {ref_filename}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            try:
                x = float(row[0])
            except:
                continue
            ref_x.append(float(row[0]))
            ref_y.append(float(row[1]))

    plt.axis('equal')
    plt.plot(ref_x, ref_y, '-ro', markersize=0.1, label='centerline')
    plt.legend()
    plt.show()


def PJcurvature(x, y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    refer to https://github.com/Pjer-zhang/PJCurvature for detail
    """
    t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
    t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)
    return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2. + b[1] ** 2.)


def read_wp(wpfile='./wp/wp.csv', removelap=True, interp=False):
    wp_x = []
    wp_y = []
    last_x = 0
    last_y = 0
    with open(os.path.join(cur_csv_dir, wpfile)) as f:
        print(f'load {wpfile}')
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            try:
                x = float(row[0])
            except:
                continue
            x_, y_ = float(row[0]), float(row[1])
            if abs(x_ - last_x) + abs(y_ - last_y) > globalWpMinInterval:
                wp_x.append(x_)
                wp_y.append(y_)
                last_x = x_
                last_y = y_

    if removelap:
        wp_x, wp_y = remove_overlap_wp(wp_x, wp_y)
    if interp:
        start_x, start_y = wp_x[0], wp_y[0]
        end_x, end_y = wp_x[-1], wp_y[-1]
        interp_x, interp_y = (start_x + end_x) / 2, (start_y + end_y) / 2

        wp_x.insert(0, interp_x)
        wp_y.insert(0, interp_y)
        wp_x.append(interp_x)
        wp_y.append(interp_y)
    print(f'number of {wpfile} after remove overlap: {len(wp_x)}')
    return wp_x, wp_y


def remove_overlap_wp(x, y):
    start_p = np.array([x[0], y[0]])
    for i in range(len(x))[10:]:
        p = np.array([x[i], y[i]])
        if LA.norm(p - start_p) < 0.05:
            return x[:i], y[:i]
    return x, y


def draw_wp(wpfile='centerline.csv', removelap=True):
    x, y = read_wp(wpfile, removelap)
    plt.plot(x, y, 'o', markersize=0.5)
    # sns.scatterplot(x=x, y=y, hue=len(x) - np.arange(len(x)))
    plt.axis('equal')
    plt.show()


def visualize_curvature_for_wp(wpfile='lane_optimal.csv', ka_thres=0.05):
    wp_x, wp_y = read_wp(wpfile, removelap=False)
    kappa = []
    no = []
    po = []
    ka = []
    for idx in range(len(wp_y))[1:-2]:
        x = wp_x[idx - 1:idx + 2]
        y = wp_y[idx - 1:idx + 2]
        kappa, norm = PJcurvature(x, y)
        ka.append(kappa)
        no.append(norm)
        po.append([x[1], y[1]])

    po = np.array([[wp_x[0], wp_y[1]]]+po+[[wp_x[-2], wp_y[-2]]]+[[wp_x[-1], wp_y[-1]]])
    no = np.array([no[0]]+no+[no[-2]]+[no[-1]])
    ka = np.array([ka[0]]+ka+[ka[-2]]+[ka[-1]])

    # overtaking
    i = 0
    n_ka = len(ka)
    segments = []
    while i < n_ka:
        if abs(ka[i]) < ka_thres:
            begin = i
            while i < n_ka and abs(ka[i]) < ka_thres:
                i += 1
            if i-begin > 20:
                segments.append((begin, i-1))
        else:
            i += 1
    overtaking_idx = []
    for seg in segments:
        b, e = seg[0], seg[1]
        i = b
        while i <= e:
            overtaking_idx.append(i)
            i += 1
    print(overtaking_idx)
    idx_path = os.path.join(cur_csv_dir, 'overtaking_wp_idx')
    np.save(idx_path, np.array(overtaking_idx))

    fig = plt.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(2, 1, 1)
    for i in overtaking_idx:
        plt.scatter(po[i, 0], po[i, 1], c='r')
    plt.plot(po[:, 0], po[:, 1])
    plt.quiver(po[:, 0], po[:, 1], ka * no[:, 0], ka * no[:, 1])
    plt.axis('equal')

    ax = fig.add_subplot(2, 1, 2)
    plt.plot(ka, '-bo', markersize=0.1)
    plt.show()


def update_map():
    print(f'copy{cur_map_path} to {src_map_path}')
    # os.system(f'cp -R {cur_csv_dir} {src_csv_dir}')
    os.system(f'cp {cur_map_path} {src_map_path}')

if __name__ == '__main__':
    fire.Fire({
        'draw_wp': draw_wp,
        'draw_optimal': draw_optimalwp,
        'smooth_wp': smooth_waypoints,
        'vis_curv': visualize_curvature_for_wp,
        'update_map': update_map,
        'smooth_lane': get_smooth_lane,
        'scale_map': scale_map,
        'draw_lanes': draw_lanes
    })
