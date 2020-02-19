# -*- coding: utf-8 -*-
u"""ランドマーク"""

import numpy as np

import utilities


class Point2DLandmark:
    u"""2Dの点ランドマーク"""

    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = np.nan

    def draw(self, ax, elems):
        u"""点ランドマークを散布図として描画する

        参照で渡されている描画要素elemsを追加するのが目的

        Args:
            ax(matplotlib.axes,_subplots.AxesSubplot): サブプロットオブジェクト
            elems([matplotlib.XXX]): 描画可能なオブジェクト(Text, PathCollectionなど)
        """
        c = ax.scatter(
            self.pos[0], self.pos[1],
            s=100, marker='*', label='landmark', color='orange')
        elems.append(c)
        elems.append(
            ax.text(
                self.pos[0], self.pos[1],
                'id:' + str(self.id), fontsize=10))


class Point2DLandmarkEstimated(Point2DLandmark):
    def __init__(self):
        super().__init__(0., 0.)
        self.cov = np.array([[1., 0.], [0., 2.]])

    def draw(self, ax, elems):
        if self.cov is None:
            return

        c = ax.scatter(
            self.pos[0], self.pos[1], s=100, marker='*', label='landmarks', color='blue')
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], 'id:' + str(self.id), fontsize=10))

        e = utilities.sigma_ellipse(self.pos, self.cov, 3)
        elems.append(ax.add_patch(e))
