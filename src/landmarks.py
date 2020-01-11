# -*- coding: utf-8 -*-
u"""ランドマーク"""

import numpy as np


class Landmark:
    u"""2Dの点ランドマーク"""

    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = np.nan

    def draw(self, ax, elems):
        u"""点ランドマークを散布図として描画する

        参照で渡されている描画要素elemsを追加するのが目的

        Args:
            ax():
            elems():
        """
        print('type of ax = {}'.format(type(ax)))
        print('type of elems = {}'.format(type(elems)))
        c = ax.scatter(
            self.pos[0], self.pos[1],
            s=100, marker='*', label='landmark', color='orange')
        elems.append(c)
        elems.append(
            ax.text(
                self.pos[0], self.pos[1],
                'id:' + str(self.id), fontsize=10))
