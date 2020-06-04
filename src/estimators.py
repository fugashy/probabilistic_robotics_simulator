# -*- coding: utf-8 -*-
from abc import abstractmethod
import copy
from math import cos, sin, pi, sqrt

import numpy as np
from scipy.stats import multivariate_normal

import particles
import robots
import sensors
import utilities

# 原典のパラメータ
_MOTION_NOISE_STDDEV = {
    'nn': 0.19, 'no': 1e-5,
    'on': 0.13, 'oo': 0.20
}
# 自分で推定したパラメータ
#   _MOTION_NOISE_STDDEV = {
#       'nn': 0.201, 'no': 1e-5,
#       'on': 0.127, 'oo': 0.227
#   }

class Estimator():
    u"""位置推定インターフェースクラス"""
    def __init__(self):
        self._pose = np.array([0., 0., 0.])

    @abstractmethod
    def motion_update(self, nu, omega, time):
        u"""位置更新

        Args:
            nu(float): 速度
            omega(float): 角速度
            time(float): 前回から経過した時間(sec)
        """
        raise NotImplementedError('This is an abstractmethod')

    @abstractmethod
    def observation_update(self, observation):
        u"""観測結果から位置を更新する
        Args:
            observation((np.array, int)): 観測距離と角度，ID
        """
        raise NotImplementedError('This is an abstractmethod')

    def pose(self):
        u"""推定器としてアウトプットする位置姿勢"""
        return self._pose


class Mcl(Estimator):
    def __init__(
            self,
            map_, init_pose, particles_,
            motion_noise_stds=_MOTION_NOISE_STDDEV,
            distance_dev_rate=0.076, direction_dev=0.026):
        u"""パーティクルを使ってロボットの確からしい位置を推定するクラス

        Args:
            envmap(maps.Map): 環境
            init_pose(np.array): 初期位置
            particles_(Particle): パーティクル
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
            distance_dev_rate(float): 観測した距離に比例するばらつき
            direction_dev(float): 観測した角度のばらつき
        """
        super().__init__()
        if type(particles_) is not list:
            raise TypeError('particles_ is not list')
        if not isinstance(particles_[0], particles.Particle):
            raise TypeError(
                '{} is not a child of Particle'.format(type(particles_)))

        self._map = map_
        self._particles = particles_

        u"""
        4Dのガウス分布オブジェクト
        vnn     0     0     0
          0   vno     0     0
          0     0   von     0
          0     0     0   voo
        """
        v = motion_noise_stds
        c = np.diag([v['nn']**2, v['no']**2, v['on']**2, v['oo']**2])
        self._motion_noise_rate_pdf = multivariate_normal(cov=c)

        self._distance_dev_rate = distance_dev_rate
        self._direction_dev = direction_dev

        # 尤度が最大のパーティクルを代表値とする
        # 初期では適当に先頭
        self._ml = self._particles[0]
        self._pose = self._ml.pose

    def motion_update(self, nu, omega, time):
        u"""パーティクルを動かす"""
        for p in self._particles:
            p.motion_update(nu, omega, time, self._motion_noise_rate_pdf)

    def observation_update(self, observation):
        for p in self._particles:
            p.observation_update(
                observation,
                self._map,
                self._distance_dev_rate,
                self._direction_dev)
        # リサンプリング後は重みが均一になるので，観測後に最大尤度の姿勢は保持する
        self._set_ml()
        self._resampling()

    def _resampling(self):
        u"""系統サンプリングを行う"""
        # 重みを累積ベクトル
        # 値の大きさは気にせず，順番に足し合わせる
        ws = np.cumsum([e.weight for e in self._particles])

        if ws[-1] < 1e-100:
            ws = [e + 1e-100 for e in ws]

        step = ws[-1] / len(self._particles)

        r = np.random.uniform(0.0, step)

        cur_pos = 0

        ps = []

        while len(ps) < len(self._particles):
            if r < ws[cur_pos]:
                ps.append(self._particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self._particles = [copy.deepcopy(e) for e in ps]
        for p in self._particles:
            p.weight = 1.0 / len(self._particles)

    def _set_ml(self):
        u"""最大尤度のパーティクルを推定機としての出力する姿勢とする"""
        i = np.argmax([p.weight for p in self._particles])
        self._ml = self._particles[i]
        self._pose = self._ml.pose


class FastSlam(Mcl):
    def __init__(self,
            init_pose, particles_,
            motion_noise_stds=_MOTION_NOISE_STDDEV,
            distance_dev_rate=0.14, direction_dev=0.05):
        u"""パーティクルを使ってロボットの確からしい位置を推定するクラス

        Args:
            init_pose(np.array): 初期位置
            particles_(Particle): パーティクル
            landmarks_(Point2DLandmark): ランドマーク
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
            distance_dev_rate(float): 観測した距離に比例するばらつき
            direction_dev(float): 観測した角度のばらつき
        """
        super().__init__(
            None, init_pose, particles_, motion_noise_stds,
            distance_dev_rate, direction_dev)

        self._particles = particles_
        self._ml = self._particles[0]

    def observation_update(self, observation):
        for p in self._particles:
            p.observation_update(
                observation, self._distance_dev_rate, self._direction_dev)
        self._set_ml()
        self._resampling()


class ExtendedKalmanFilter(Estimator):
    u"""拡張カルマンフィルタを用いて自己位置を更新するクラス"""

    def __init__(
            self,
            map_, init_pose,
            motion_noise_stds=_MOTION_NOISE_STDDEV,
            distance_dev_rate=0.14, direction_dev=0.05):
        u"""初期設定

        Args:
            map_(maps.Map): 環境地図
            init_pose(np.array): 初期位置
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
            distance_dev_rate(float): 観測した距離に比例するばらつき
            direction_dev(float): 観測した角度のばらつき
        """
        self._map = map_
        # 信念分布を表す正規分布
        self._belief = multivariate_normal(
            mean=np.array([0., 0., 0.]), cov=np.diag([1e-10, 1e-10, 1e-10]))
        self._motion_noise_stds = motion_noise_stds
        # [x, y, theta]の3次元
        self._pose = self._belief.mean

        self._distance_dev_rate = distance_dev_rate
        self._direction_dev = direction_dev

    def motion_update(self, nu, omega, time):
        # 0割回避
        if abs(omega) < 1e-5:
            omega = 1e-5

        theta = self._belief.mean[2]

        M = utilities.matM(nu, omega, time, self._motion_noise_stds)
        A = utilities.matA(nu, omega, time, theta)
        F = utilities.matF(nu, omega, time, theta)

        self._belief.cov = F @ self._belief.cov @ F.T + A @ M @ A.T
        self._belief.mean = robots.IdealRobot.state_transition(nu, omega, time, self._belief.mean)
        self._pose = self._belief.mean

    def observation_update(self, observation):
        for d in observation:
            z = d[0]
            obs_id = d[1]

            H = utilities.matH(self._belief.mean, self._map.landmarks()[obs_id].pos)
            estimated_z = sensors.IdealCamera.observation_function(
                self._belief.mean, self._map.landmarks()[obs_id].pos)
            Q = utilities.matQ(estimated_z[0] * self._distance_dev_rate, self._direction_dev)
            K = self._belief.cov @ H.T @ np.linalg.inv(Q + H @ self._belief.cov @ H.T)

            self._belief.mean += K @ (z - estimated_z)
            self._belief.cov = (np.eye(3) - K @ H) @ self._belief.cov
            self._pose = self._belief.mean
