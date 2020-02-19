# -*- coding: utf-8 -*-
from math import sqrt

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
        if observation is None:
            return

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
