"""

Probabilistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)

"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# parameter
N_SAMPLE = 100  # number of sample_points，随机采样点个数
N_KNN = 10  # number of edge from one sampled point 一个随机采样点连接边个数
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length  最大边长距离

show_animation = True
show_roadmap = True


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(sx, sy, gx, gy, ox, oy, rr):
    obstacle_kd_tree = cKDTree(np.vstack((ox, oy)).T)
    # vstack((ox, oy))将ox,oy数组打包为堆叠矩阵（二维数组），.T将矩阵转置
    # cKDTree() 用C写的快速查找KDTree类，执行速度更快

    sample_x, sample_y = sample_points(sx, sy, gx, gy,
                                       rr, ox, oy, obstacle_kd_tree)  # 获取随机采样点
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree)

    rx, ry = dijkstra_planning(
        sx, sy, gx, gy, road_map, sample_x, sample_y)  # 作dijkstra路径规划

    return rx, ry


# 判断连接边是否与障碍物干涉
def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    rr: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = cKDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)  # 查询最近邻点，k为返回的最近邻点数。
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            # 判断连接边是否与障碍物干涉
            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)  # 记录循环过程中该随机采样点的邻接节点
    if show_roadmap:
        plot_road_map(road_map, sample_x, sample_y)  # 绘制邻接边
        plt.plot(sample_x, sample_y, ".b")

    return road_map


# 作dijkstra路径规划
def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    gx: goal x position [m]
    gy: goal y position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    rr: robot radius [m]
    road_map: ??? [m] 邻接表
    sample_x: ??? [m] 随机采样点表x
    sample_y: ??? [m] 随机采样点表y

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)  # dijkstra算法
        # c_id = min(open_set, key=lambda o: open_set[o].cost + calc_heuristic(goal_node, open_set[o]))  # A*算法
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


# A*算法代价函数
def calc_heuristic(n1, n2):
    w = 1.0  # weight of heuristic
    d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
    return d


# 绘制邻接边
def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-c")


# 获取随机采样点
def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree):
    # rr为robot_size
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    while len(sample_x) < N_SAMPLE:
        tx = (random.random() * (max_x - min_x)) + min_x
        ty = (random.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])  # 返回最近邻点的距离dist和在数组中的顺序index

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    # 随机采样点中增加起点和终点
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m] 起点
    gx = 50.0  # [m]
    gy = 50.0  # [m] 终点
    robot_size = 1.0  # [m]

    ox = []
    oy = []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "xr")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size)

    assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.plot(sx, sy, "xr")
        plt.plot(gx, gy, "xr")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()