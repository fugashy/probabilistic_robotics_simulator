# -*- coding: utf-8 -*-
from sys import path
path.append('../')
from math import cos, sin, pi

from src import particles


class Mcl():
    def __init__(self, init_pose, num):
        self.particles = \
            [
                particles.Particle(init_pose)
                for i in range(num)
            ]

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
