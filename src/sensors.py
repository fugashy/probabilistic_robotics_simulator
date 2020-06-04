# -*- coding: utf-8 -*-
from abc import abstractmethod
from math import atan2, cos, sin, pi
import numpy as np
from scipy.stats import norm, uniform

class Sensor():
    u"""センサー"""

    @abstractmethod
    def data(self, sensor_pose):
        u"""与えられた姿勢からのランドマークの観測結果を返す

        Args:
            cam_pose(np.array): 観測位置姿勢

        Returns:
            ((np.array, int)): 観測結果とID
        """
        raise NotImplementedError('Abstract method of Sensor')


class IdealCamera(Sensor):
    u"""理想的な観測をするカメラ

    Landmarkを観測し，距離と角度を求める
    """

    def __init__(
            self,
            env_map,
            distance_range=(0.5, 6.0),
            direction_range=(-np.pi / 3., np.pi / 3.)):
        u"""マップの登録, 有効範囲の設定"""
        super().__init__()
        self.map = env_map
        self.distance_range = distance_range
        self.direction_range = direction_range
        # 最後に計測したときの結果
        self.lastdata = []

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks():
            p = self.observation_function(cam_pose, lm.pos)
            if self._visible(p):
                observed.append((p, lm.id))

        self.lastdata = observed

        return self.lastdata

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
            distance_noise_rate=0.05, direction_noise=pi/180.,
            distance_bias_rate_stddev=0.05, direction_bias_stddev=pi/180.,
            phantom_prob=0.0, phantom_range_x=(-5., 5.), phantom_range_y=(-5., 5.),
            oversight_prob=0.1,
            occulusion_prob=0.0):
        super().__init__(env_map, distance_range, direction_range)

        # 観測結果に直接与えるノイズ係数
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

        # 観測結果に与えるバイアス
        self.distance_bias_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale=direction_bias_stddev)

        # ファントムのシミュレーション
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(
            loc=(rx[0], ry[0]), scale=(rx[1] - rx[0], ry[1] - ry[0]))
        self.phantom_prob = phantom_prob

        # 見落とし
        self.oversight_prob = oversight_prob

        # オクルージョン
        # 一定確率でセンサ値を大きくすることにする
        self.occulusion_prob = occulusion_prob


    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks():
            z = self.observation_function(cam_pose, lm.pos)
            z = self._phantom(cam_pose, z)
            z = self._oversight(z)
            if self._visible(z):
                z = self._noise(z)
                z = self._bias(z)
                observed.append((z, lm.id))

        self.lastdata = observed

        return self.lastdata

    def _noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0] * self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T

    def _bias(self, relpos):
        return relpos + np.array(
            [relpos[0] * self.distance_bias_std, self.direction_bias]).T

    def _phantom(self, cam_pose, relpos):
        # 0 ~ 1の間の一様分布からドローしてしきい値処理
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos

    def _oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos

    def _occulusion(self, relpos):
        u"""乱数がしきい値を下回ったとき現在のセンサー値に雑音を足す"""
        if uniform.rvs() < self.occulusion_prob:
            ell = relpos[0] + uniform.rvs() * (self.distance_range[1] - relpos[0])
            return np.array([ell, relpos[1]]).T
        else:
            return relpos
