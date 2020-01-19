# -*- coding: utf-8 -*-
u"""センサー"""

from math import atan2, cos, sin, pi
import numpy as np
from scipy.stats import norm, uniform


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
        # http://hima-tubusi.blogspot.com/2016/12/blog-post_12.html
        normalized_phi = atan2(sin(phi), cos(phi))

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


class Camera(IdealCamera):
    def __init__(self,
            env_map,
            distance_range=(0.5, 6.0), direction_range=(-pi / 3., pi / 3),
            distance_noise_rate=0.1, direction_noise_rate=pi/90.):
        super().__init__(env_map, distance_range, direction_range)

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise_rate = direction_noise_rate

    def _noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0] * self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise_rate)
        return np.array([ell, phi]).T

    def data(self, cam_pose):
        u"""与えられた姿勢からのランドマークの観測結果を返す

        Args:
            cam_pose(np.array): 観測位置姿勢

        Returns:
            ((np.array, int)): 観測結果とID
        """
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self._visible(z):
                z = self._noise(z)
                observed.append((z, lm.id))

        self.lastdata = observed

        return self.lastdata
