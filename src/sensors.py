# -*- coding: utf-8 -*-
u"""センサー"""

from math import atan2, cos, sin
import numpy as np


class IdealCamera():
    u"""理想的な観測をするカメラ

    Landmarkを観測し，距離と角度を求める
    """

    def __init__(
            self,
            env_map,
            distance_range=(0.5, 6.0),
            direction_range=(-np.pi / 3., np.pi / 3.)):
        u"""マップの登録, 有効範囲の設定"""
        self.map = env_map
        self.distance_range = distance_range
        self.direction_range = direction_range
        # 最後に計測したときの結果
        self.lastdata = []

    def data(self, cam_pose):
        u"""与えられた姿勢からのランドマークの観測結果を返す

        Args:
            cam_pose(np.array): 観測位置姿勢

        Returns:
            ((np.array, int)): 観測結果とID
        """
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            if self._visible(p):
                observed.append((p, lm.id))

        self.lastdata = observed

        return self.lastdata

    def draw(self, ax, elems, cam_pose):
        x, y, theta = cam_pose

        for lm in self.lastdata:
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * cos(direction + theta)
            ly = y + distance * sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')

    @classmethod
    def observation_function(cls, cam_pos, obj_pos):
        u"""観測方程式

        Args:
            cam_pos(np.array): 観測位置姿勢(x, y, theta)
            obj_pos(np.array): 観測対象位置姿勢(x, y)

        Returns:
            (np.array): 観測結果[距離, 角度]
        """
        # 距離を求める
        # 観測対象は点なので角度は使わない
        diff = obj_pos - cam_pos[0: 2]

        # 角度を求める
        # 正規化もする
        phi = atan2(diff[1], diff[0]) - cam_pos[2]
        while phi >= np.pi:
            phi -= 2. * np.pi
        while phi < np.pi:
            phi += 2. * np.pi

        # *diffでhypotの各引数にx, yとして展開されて渡される
        return np.array([np.hypot(*diff), normalized_phi]).T

    def _visible(self, polar_pos):
        u"""画角に入っているかどうか，距離が有効かどうか

        Args:
            polar_pos(np.array): 対象との距離・角度
        """
        if polar_pos is None:
            return False

        in_valid_distance = \
            self.distance_range[0] <= polar_pos[0] <= self.distance_range[1]
        in_valid_direction = \
            self.direction_range[0] <= polar_pos[1] <= self.direction_range[1]

        return in_valid_distance and in_valid_direction
