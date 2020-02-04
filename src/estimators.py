# -*- coding: utf-8 -*-
import copy
from sys import path
path.append('../')
from math import cos, sin, pi, sqrt

import numpy as np
from scipy.stats import multivariate_normal

import robots
import sensors

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
