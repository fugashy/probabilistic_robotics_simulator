# -*- coding: utf-8 -*-
import copy
from sys import path
path.append('../')
from math import atan2, cos, sin, pi, sqrt

from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import multivariate_normal

import robots
import sensors


def sigma_ellipse(p, cov, n):
    u"""共分散行列の楕円描画オブジェクトを返す

    Args:
        p(list): 描画基準位置
        cov(np.ndarray): 共分散行列
        n(float): スケール?

    Returns:
        matplotlib.patches.Ellipse: 誤差楕円
    """
    eigen_values, eigen_vector = np.linalg.eig(cov)
    angle = atan2(eigen_vector[:, 0][1], eigen_vector[:, 0][0]) / pi * 180.
    return Ellipse(
        p, width=2*n*sqrt(eigen_values[0]), height=2*n*sqrt(eigen_values[1]),
        angle=angle, fill=False, color='blue', alpha=0.5)

def matM(nu, omega, time, stds):
    u"""nu, omega空間の共分散行列を計算する

    弓状になったりする分布をガウス分布で近似したもの

    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        time(float): 前回からの経過時間[sec]
        stds(dict): 並進速度，回転速度2x2=4Dの標準偏差

    Returns:
        共分散行列M_{t}
    """
    return np.diag(
        [
            stds['nn']**2. * abs(nu) / time + stds['no']**2. * abs(omega) / time,
            stds['on']**2. * abs(nu) / time + stds['oo']**2. * abs(omega) / time
        ])

def matA(nu, omega, time, theta):
    u"""テイラー展開したときの1次の項に登場する係数行列を計算する
    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        time(float): 前回からの経過時間[sec]
        theta(float): 角度

    Returns:
        係数行列A_{t}
    """
    st = sin(theta)
    ct = cos(theta)

    stw = sin(theta + omega * time)
    ctw = cos(theta + omega * time)

    return np.array(
        [
            [(stw - st) / omega, -nu / (omega**2) * (stw - st) + nu / omega * time * ctw],
            [(-ctw + ct) / omega, -nu / (omega**2) * (-ctw + ct) + nu / omega * time * stw],
            [0., time]
        ])

def matF(nu, omega, time, theta):
    u"""状態方程式fを\mu_{t-1}周りでx_{t-1}で偏微分したときのヤコビ行列を計算する
    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        time(float): 前回からの経過時間[sec]
        theta(float): 角度

    Returns:
        ヤコビ行列F_{t}
    """
    F = np.diag([1., 1., 1.])

    F[0, 2] = nu / omega * (cos(theta + omega * time) - cos(theta))
    F[1, 2] = nu / omega * (sin(theta + omega * time) - sin(theta))

    return F

def matH(pose, landmark_pos):
    mx, my = landmark_pos
    mux, muy, mut = pose

    q = (mux - mx)**2 + (muy - my)**2

    return np.array(
        [
            [(mux - mx) / np.sqrt(q), (muy - my) / np.sqrt(q), 0.0],
            [(my - muy) / q, (mux - mx) / q, -1.0]
        ])

def matQ(distance_dev, direction_dev):
    return np.diag(np.array([distance_dev**2, direction_dev**2]))


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

    def observation_update(
            self, observation, map_, distance_dev_rate, direction_dev):
        u"""観測結果からパーティクルの位置を更新する
        Args:
            observation((np.array, int)): 観測距離と角度，ID
            map_(maps.Map): 環境
            distance_dev_rate(float): 観測した距離に比例するばらつき
            direction_dev(float): 観測した角度のばらつき
        """
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            u"""パーティクルの位置と地図からランドマークの距離と方角を得る(理論値)
                IDがわかるからやれる芸当
            """
            # ランドマークの位置
            pos_on_map = map_.landmarks[obs_id].pos
            # パーティクルから観測した場合の位置
            particle_suggest_pos = sensors.IdealCamera.observation_function(
                self.pose, pos_on_map)

            u"""尤度の計算"""
            # 距離に比例したばらつき
            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            # 共分散行列
            cov = np.diag([distance_dev**2, direction_dev**2])
            # 重みを更新
            # N(パーティクルから観測したときの理想の値, 分散)に従うと仮定して
            # そこからどのくらい外れているのか，もしくはあっているのかを得る
            self.weight *= multivariate_normal(
                mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


class Mcl():
    def __init__(
            self,
            map_, init_pose, num,
            motion_noise_stds={'nn': 0.19, 'no': 0.001, 'on': 0.13, 'oo': 0.2},
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


class ExtendedKalmanFilter():
    u"""拡張カルマンフィルタを用いて自己位置を更新するクラス"""

    def __init__(
            self,
            map_, init_pose,
            motion_noise_stds={'nn': 0.19, 'no': 0.001, 'on': 0.13, 'oo': 0.2},
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

        M = matM(nu, omega, time, self.motion_noise_stds)
        A = matA(nu, omega, time, theta)
        F = matF(nu, omega, time, theta)

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

            H = matH(self.belief.mean, self.map.landmarks[obs_id].pos)
            estimated_z = sensors.IdealCamera.observation_function(
                self.belief.mean, self.map.landmarks[obs_id].pos)
            Q = matQ(estimated_z[0] * self.distance_dev_rate, self.direction_dev)
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
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        # θ方向の誤差3シグマ
        x, y, c = self.belief.mean
        sigma3 = sqrt(self.belief.cov[2, 2]) * 3.
        xs = [x + cos(c - sigma3), x, x + cos(c + sigma3)]
        ys = [y + sin(c - sigma3), y, y + sin(c + sigma3)]
        elems += ax.plot(xs, ys, color='blue', alpha=0.5)
