# -*- coding: utf-8 -*-
from sys import path
path.append('../')
from math import cos, sin, pi

import numpy as np
from scipy.stats import multivariate_normal

from src import particles


class Mcl():
    def __init__(self, init_pose, num, motion_noise_stds):
        u"""パーティクルを使ってロボットの確からしい位置を推定するクラス

        Args:
            init_pose(np.array): 初期位置
            num(int): パーティクルの数
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
        """
        self.particles = \
            [
                particles.Particle(init_pose)
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
        print(self.motion_noise_rate_pdf.cov)

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
