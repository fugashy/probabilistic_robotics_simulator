# -*- coding: utf-8 -*-
import copy
from math import cos, sin, pi, sqrt

import numpy as np
from scipy.stats import multivariate_normal

import particles
import robots
import sensors
import utilities

# 原典のパラメータ
#   _MOTION_NOISE_STDDEV = {
#       'nn': 0.19, 'no': 1e-5,
#       'on': 0.13, 'oo': 0.20
#   }
# 自分で推定したパラメータ
_MOTION_NOISE_STDDEV = {
    'nn': 0.201, 'no': 1e-5,
    'on': 0.127, 'oo': 0.227
}


class Mcl():
    def __init__(
            self,
            map_, init_pose, num,
            motion_noise_stds=_MOTION_NOISE_STDDEV,
            distance_dev_rate=0.14, direction_dev=0.05):
        u"""パーティクルを使ってロボットの確からしい位置を推定するクラス

        Args:
            envmap(maps.Map): 環境
            init_pose(np.array): 初期位置
            num(int): パーティクルの数
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
            distance_dev_rate(float): 観測した距離に比例するばらつき
            direction_dev(float): 観測した角度のばらつき
        """
        self.map = map_
        # 初期の重みは等価値
        # 総和が1になればスケールは変わらない
        initial_weight = 1. / num
        self.particles = \
            [
                particles.Particle(init_pose, initial_weight)
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

        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        # 尤度が最大のパーティクルを代表値とする
        # 初期では適当に先頭
        self.ml = self.particles[0]
        self.pose = self.ml.pose

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
        # パーティクルの重みに比例した長さの矢印を描くため
        vxs = \
            [
                cos(p.pose[2]) * p.weight * len(self.particles)
                for p in self.particles
            ]
        vys = \
            [
                sin(p.pose[2]) * p.weight * len(self.particles)
                for p in self.particles
            ]
        elems.append(ax.quiver(xs, ys, vxs, vys,
            angles='xy', scale_units='xy', scale=1.5, color='blue', alpha=0.5))

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(
                observation,
                self.map,
                self.distance_dev_rate,
                self.direction_dev)
        # リサンプリング後は重みが均一になるので，観測後に最大尤度の姿勢は保持する
        self.set_ml()
        self.resampling()

    def resampling(self):
        u"""系統サンプリングを行う"""
        # 重みを累積ベクトル
        # 値の大きさは気にせず，順番に足し合わせる
        ws = np.cumsum([e.weight for e in self.particles])

        if ws[-1] < 1e-100:
            ws = [e + 1e-100 for e in ws]

        step = ws[-1] / len(self.particles)

        r = np.random.uniform(0.0, step)

        cur_pos = 0

        ps = []

        while len(ps) < len(self.particles):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0 / len(self.particles)

    def set_ml(self):
        u"""最大尤度のパーティクルを推定機としての出力する姿勢とする"""
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose


class FastSlam(Mcl):
    def __init__(self,
            map_, init_pose, particle_num, landmark_num,
            motion_noise_stds=_MOTION_NOISE_STDDEV,
            distance_dev_rate=0.14, direction_dev=0.05):
        u"""パーティクルを使ってロボットの確からしい位置を推定するクラス

        Args:
            envmap(maps.Map): 環境
            init_pose(np.array): 初期位置
            particle_num(int): パーティクルの数
            landmark_num(int): ランドマークの数
            motion_noise_stds(dict): 並進速度，回転速度2x2=4Dの標準偏差
            distance_dev_rate(float): 観測した距離に比例するばらつき
            direction_dev(float): 観測した角度のばらつき
        """
        super().__init__(
            map_, init_pose, particle_num, motion_noise_stds, distance_dev_rate, direction_dev)

        self.particles = \
            [
                particles.MapParticle(init_pose, 1. / particle_num, landmark_num)
                for i in range(particle_num)
            ]
        self.ml = self.particles[0]

    def draw(self, ax, elems):
        super().draw(ax, elems)
        self.ml.map.draw(ax, elems)


class ExtendedKalmanFilter():
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
        self.map = map_
        # 信念分布を表す正規分布
        self.belief = multivariate_normal(
            mean=np.array([0., 0., 0.]), cov=np.diag([1e-10, 1e-10, 1e-10]))
        self.motion_noise_stds = motion_noise_stds
        # [x, y, theta]の3次元
        self.pose = self.belief.mean

        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

    def motion_update(self, nu, omega, time):
        u"""位置更新

        Args:
            nu(float): 速度
            omega(float): 角速度
            time(float): 前回から経過した時間(sec)
        """
        # 0割回避
        if abs(omega) < 1e-5:
            omega = 1e-5

        theta = self.belief.mean[2]

        M = utilities.matM(nu, omega, time, self.motion_noise_stds)
        A = utilities.matA(nu, omega, time, theta)
        F = utilities.matF(nu, omega, time, theta)

        self.belief.cov = F @ self.belief.cov @ F.T + A @ M @ A.T
        self.belief.mean = robots.IdealRobot.state_transition(nu, omega, time, self.belief.mean)
        self.pose = self.belief.mean


    def observation_update(self, observation):
        u"""観測結果から位置を更新する
        Args:
            observation((np.array, int)): 観測距離と角度，ID
        """
        for d in observation:
            z = d[0]
            obs_id = d[1]

            H = utilities.matH(self.belief.mean, self.map.landmarks[obs_id].pos)
            estimated_z = sensors.IdealCamera.observation_function(
                self.belief.mean, self.map.landmarks[obs_id].pos)
            Q = utilities.matQ(estimated_z[0] * self.distance_dev_rate, self.direction_dev)
            K = self.belief.cov @ H.T @ np.linalg.inv(Q + H @ self.belief.cov @ H.T)

            self.belief.mean += K @ (z - estimated_z)
            self.belief.cov = (np.eye(3) - K @ H) @ self.belief.cov
            self.pose = self.belief.mean

    def draw(self, ax, elems):
        u"""自己位置を誤差楕円で描画

        Args:
            ax(matplotlib.axes,_subplots.AxesSubplot): サブプロットオブジェクト
            elems([matplotlib.XXX]): 描画可能なオブジェクト(Text, PathCollectionなど)
        """
        # xyの3シグマ楕円
        # スライシング[0:2]でxyのみ抽出
        e = utilities.sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        # θ方向の誤差3シグマ
        x, y, c = self.belief.mean
        sigma3 = sqrt(self.belief.cov[2, 2]) * 3.
        xs = [x + cos(c - sigma3), x, x + cos(c + sigma3)]
        ys = [y + sin(c - sigma3), y, y + sin(c + sigma3)]
        elems += ax.plot(xs, ys, color='blue', alpha=0.5)
