import copy
import math
import random
import time

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np

show_animation = True


class RRT:

    def __init__(self, obstacleList, randArea,
                 expandDis=2.0, goalSampleRate=10, maxIter=200):

        self.start = None
        self.goal = None
        self.min_rand = randArea[0]  # 采样范围最小值
        self.max_rand = randArea[1]  # 采样范围最大值
        self.expand_dis = expandDis  # 采样步长设置为2
        self.goal_sample_rate = goalSampleRate  # 目标采样率设置为10%，即有10%的概率以终点设置为目标采样点
        self.max_iter = maxIter  # 设置最大的迭代数量
        self.obstacle_list = obstacleList  # 设置障碍物
        self.node_list = None  # 存储RRT树，树上的节点

    def rrt_planning(self, start, goal, animation=True):
        start_time = time.time()  # 进行计时，为最后统一时间
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]  # 把开始节点加入作为树的根节点
        path = None

        for i in range(self.max_iter):
            rnd = self.sample()  # 选取采样点Xrand
            n_ind = self.get_nearest_list_index(self.node_list, rnd)  # 找到离采样点最近的树节点
            nearestNode = self.node_list[n_ind]  # 得到最近树节点Xnear

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)  # 树生长的方向
            newNode = self.get_new_node(theta, n_ind, nearestNode)  # 得到Xnew节点

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x,
                                                       nearestNode.y)  # 检测新节点Xnew到Xnear是否有障碍物
            if noCollision:
                self.node_list.append(newNode)  # 新节点Xnew加入树里面
                if animation:
                    self.draw_graph(newNode, path)

                # 判断是否到终点附近
                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1
                        path = self.get_final_course(lastIndex)  # 找所求路径
                        pathLen = self.get_path_len(path)  # 求路径长度
                        print("current path length: {}, It costs {} s".format(pathLen, time.time() - start_time))

                        if animation:
                            self.draw_graph(newNode, path)
                        return path

    def rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None
        lastPathLength = float('inf')  # 当前找到的路径的最短长度

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
            # 前面几步与RRT算法一样
            if noCollision:
                nearInds = self.find_near_nodes(newNode)  # 找到Xnew圆域范围内的候选节点
                newNode = self.choose_parent(newNode, nearInds)  # 重新选择Xnew的父节点

                self.node_list.append(newNode)  # 将新节点加入树
                self.rewire(newNode, nearInds)  # 重新选择Xnew圆域内节点的父节点

                if animation:
                    self.draw_graph(newNode, path)

                # 判断是否接近终点
                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1

                        tempPath = self.get_final_course(lastIndex)
                        tempPathLen = self.get_path_len(tempPath)
                        # 不断更新最端路径
                        if lastPathLength > tempPathLen:
                            path = tempPath
                            lastPathLength = tempPathLen
                            print(
                                "current path length: {}, It costs {} s".format(tempPathLen, time.time() - start_time))

        return path

    def informed_rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        # max length we expect to find in our 'informed' sample space,
        # starts as infinite
        cBest = float('inf')
        path = None

        # 椭圆的初始化相关
        # Computing the sampling space,计算由起点和终点确定的椭球相关参数
        cMin = math.sqrt(pow(self.start.x - self.goal.x, 2)
                         + pow(self.start.y - self.goal.y, 2))
        xCenter = np.array([[(self.start.x + self.goal.x) / 2.0],
                            [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / cMin],  # cos(theta)
                       [(self.goal.y - self.start.y) / cMin], [0]])  # sin(theta)

        e_theta = math.atan2(a1[1], a1[0])

        # 论文方法求旋转矩阵（2选1）
        # first column of identity matrix transposed
        # id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        # M = a1 @ id1_t
        # U, S, Vh = np.linalg.svd(M, True, True)
        # C = np.dot(np.dot(U, np.diag(
        #     [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])),
        #            Vh)

        # 直接用二维平面上的公式（2选1）
        C = np.array([[math.cos(e_theta), -math.sin(e_theta), 0],
                      [math.sin(e_theta), math.cos(e_theta), 0],
                      [0, 0, 1]])

        for i in range(self.max_iter):
            # Sample space is defined by cBest
            # cMin is the minimum distance between the start point and the goal
            # xCenter is the midpoint between the start and the goal
            # cBest changes when a new path is found

            rnd = self.informed_sample(cBest, cMin, xCenter, C)
            # informed_RRT*与RRT*区别在于采样。以下代码与RRT*一样
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
            if noCollision:
                nearInds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearInds)

                self.node_list.append(newNode)
                self.rewire(newNode, nearInds)

                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1
                        tempPath = self.get_final_course(lastIndex)
                        tempPathLen = self.get_path_len(tempPath)
                        if tempPathLen < cBest:
                            path = tempPath
                            cBest = tempPathLen  # 当前路径长度作为CBest,找到路径更新椭球cBest从而影响采样程序
                            print(
                                "current path length: {}, It costs {} s".format(tempPathLen, time.time() - start_time))
            # informed_RRT*画图程序
            if animation:
                self.draw_graph_informed_RRTStar(xCenter=xCenter,
                                                 cBest=cBest, cMin=cMin,
                                                 e_theta=e_theta, rnd=rnd, path=path)

        return path

    # 进行采样
    def sample(self):
        # 随机采样率小于所设置的目标采样率则直接返回终点，大于则随机取点
        # 有随机采样到终点使得路径导向终点
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand)]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]
        return rnd

    # 重新选择Xnew的父节点
    def choose_parent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            # 检测圆域内节点是否与障碍物产生碰撞
            if self.check_collision(self.node_list[i], theta, d):
                dList.append(self.node_list[i].cost + d)  # 不产生碰撞则记录经过圆域内该节点到Xnew的距离
            else:
                dList.append(float('inf'))  # 产生碰撞则记录距离为无穷大

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]  # 记录最小距离的下标

        if minCost == float('inf'):
            print("min cost is inf")
            return newNode

        newNode.cost = minCost  # 更新最短距离
        newNode.parent = minInd  # 更新父节点

        return newNode

    # 找到Xnew圆域范围内的节点
    def find_near_nodes(self, newNode):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))  # log()默认以e为底，越后面节点数越多，半径越小
        d_list = [(node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2
                  for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    # 椭圆内采样
    def informed_sample(self, cMax, cMin, xCenter, C):
        if cMax < float('inf'):
            r = [cMax / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
            L = np.diag(r)  # 转为对角矩阵
            xBall = self.sample_unit_ball()  # 在单位圆里面采样
            rnd = np.dot(np.dot(C, L), xBall) + xCenter  # 从椭圆坐标系转换到世界坐标系
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sample()  # 没找到路径时正常采样

        return rnd

    @staticmethod
    # 在单位圆里面采样
    def sample_unit_ball():
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    @staticmethod
    # 求路径长度
    def get_path_len(path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y) ** 2)

        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    @staticmethod
    # 找到离采样点最近的树节点，返回树中节点离采样点最近的树节点下标
    def get_nearest_list_index(nodes, rnd):
        dList = [(node.x - rnd[0]) ** 2
                 + (node.y - rnd[1]) ** 2 for node in nodes]
        minIndex = dList.index(min(dList))
        return minIndex

    # 得到Xnew节点
    def get_new_node(self, theta, n_ind, nearestNode):
        newNode = copy.deepcopy(nearestNode)  # 深拷贝，2个变量地址不同不相互影响

        newNode.x += self.expand_dis * math.cos(theta)
        newNode.y += self.expand_dis * math.sin(theta)

        newNode.cost += self.expand_dis
        newNode.parent = n_ind
        return newNode

    # 判断是否接近终点
    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    # 重新选择Xnew圆域内节点的父节点
    def rewire(self, newNode, nearInds):
        n_node = len(self.node_list)
        for i in nearInds:
            nearNode = self.node_list[i]

            d = math.sqrt((nearNode.x - newNode.x) ** 2
                          + (nearNode.y - newNode.y) ** 2)

            s_cost = newNode.cost + d

            if nearNode.cost > s_cost:
                theta = math.atan2(newNode.y - nearNode.y,
                                   newNode.x - nearNode.x)
                if self.check_collision(nearNode, theta, d):
                    nearNode.parent = n_node - 1  # 更新该节点父节点为Xnew
                    nearNode.cost = s_cost

    @staticmethod
    # 计算点到线的距离
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return (p - v).dot(p - v)  # v == w case
        l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2))  # 如果p投影不在vm线段内，则取0或1为了下一步取到v或者w为端点
        projection = v + t * (w - v)  # Projection falls on the segment
        return (p - projection).dot(p - projection)

    # 检测新节点Xnew到Xnear是否有障碍物
    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox, oy, size) in self.obstacle_list:
            # 通过圆心到直线的距离判断是否发生碰撞
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]),  # np.array([])构建一维数组
                np.array([x2, y2]),
                np.array([ox, oy]))
            if dd <= size ** 2:
                return False  # collision
        return True

    # 检测圆域内节点是否与障碍物产生碰撞
    def check_collision(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)
        end_x = tmpNode.x + math.cos(theta) * d
        end_y = tmpNode.y + math.sin(theta) * d
        return self.check_segment_collision(tmpNode.x, tmpNode.y, end_x, end_y)

    # 找到所求路径
    def get_final_course(self, lastIndex):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph_informed_RRTStar(self, xCenter=None, cBest=None, cMin=None, e_theta=None, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if cBest != float('inf'):
                self.plot_ellipse(xCenter, cBest, cMin, e_theta)  # 画椭圆图

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([-2, 18, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    # 画椭圆图
    def plot_ellipse(xCenter, cBest, cMin, e_theta):  # pragma: no cover

        a = math.sqrt(cBest ** 2 - cMin ** 2) / 2.0
        b = cBest / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = xCenter[0]
        cy = xCenter[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)    # 0到2*pi按0.1等分
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        # 欧拉角的输入顺序与 参数 ‘zyx’ 相匹配，第一个角度是绕 z 轴旋转的，第二个角度是绕 y 轴旋转的，第三个角度是绕 x 轴旋转的
        # as_matrix()[0:2, 0:2]取矩阵的前2*2矩阵
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")

    # 更新路径和节点图
    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacle_list:
            # self.plot_circle(ox, oy, size)
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        plt.axis([-2, 18, -2, 15])
        plt.grid(True)
        plt.pause(0.01)


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start rrt planning")

    # create obstacles
    obstacleList = [
        (3, 3, 1.5),
        (12, 2, 3),
        (3, 9, 2),
        (9, 11, 2),
    ]
    # obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
    #                 (9, 5, 2), (8, 10, 1)]

    # Set params
    rrt = RRT(randArea=[-2, 18], obstacleList=obstacleList, maxIter=200)  # 采样范围，障碍物，最多迭代次数
    # path = rrt.rrt_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    # path = rrt.rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    path = rrt.informed_rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    print("Done!!")

    if show_animation and path:
        plt.show()


if __name__ == '__main__':
    main()
