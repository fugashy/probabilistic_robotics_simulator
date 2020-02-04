# -*- coding: utf-8 -*-
from sys import path
path.append('../')
from math import cos, sin, pi, sqrt

import numpy as np
from scipy.stats import multivariate_normal

import robots

class Particle():
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        # 重み
        # これと尤度をかける
        self.weight = weight

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        # nn, no, on, ooの順にドローされる
        v_nn, v_no, v_on, v_oo = noise_rate_pdf.rvs()

        noised_nu = \
            nu + v_nn * sqrt(abs(nu) / time) + v_no * sqrt(abs(omega) / time)
        noised_omega = \
            omega + v_on * sqrt(abs(nu) / time) + v_oo * sqrt(abs(omega) / time)

        self.pose = robots.IdealRobot.state_transition(
            noised_nu, noised_omega, time, self.pose)

    def observation_update(self, observation):
        print(observation)


class Mcl():
    def __init__(self, init_pose, num, motion_noise_stds={'nn': 0.19, 'no': 0.001, 'on': 0.13, 'oo': 0.2}):
        u"""パーティクルを使ってロボットの確からしい位置を推定するクラス

        Args:
            init_pose(np.array): 初期位置
            num(int): パーティクルの数
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
        """
        # 初期の重みは等価値
        # 総和が1になればスケールは変わらない
        initial_weight = 1. / weight
        self.particles = \
            [
                Particle(init_pose, initial_weight)
                for i in range(num)
            ]

        u"""
        4Dのガウス分布オブジェクト
        vnn     0     0     0
          0   vno     0     0
          0     0   von     0
          0     0     0   voo
        """
        v = motion_noise_stds
        c = np.diag([v['nn']**2, v['no']**2, v['on']**2, v['oo']**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

    def motion_update(self, nu, omega, time):
        u"""パーティクルを動かす"""
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)


    def draw(self, ax, elems):
        u"""パーティクルを矢印として表示

        Args:
            ax(matplotlib.axes,_subplots.AxesSubplot): サブプロットオブジェクト
            elems([matplotlib.XXX]): 描画可能なオブジェクト(Text, PathCollectionなど)
        """
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [cos(p.pose[2]) for p in self.particles]
        vys = [sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, \
                angles='xy', scale_units='xy', scale=1.5, color='blue', alpha=0.5))

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation)
