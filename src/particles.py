# -*- coding: utf-8 -*-
from abc import abstractmethod
from math import sqrt

import numpy as np
from scipy.stats import multivariate_normal

import landmarks
import maps
import robots
import sensors
import utilities

# TODO(fugashy) observation_updateがインターフェースが違うので，いつか治す

class Particle():
    def __init__(self, init_pose=np.array([0., 0., 0.]), weight=np.nan):
        self.pose = init_pose
        # 重み
        # これと尤度をかける
        self.weight = weight

    @abstractmethod
    def motion_update(self, nu, omega, time, noise_rate_pdf):
        raise NotImplementedError('')


class SimpleParticle(Particle):
    u"""教科書どおりのパーティクル"""

    def __init__(self, init_pose=np.array([0., 0., 0.]), weight=np.nan):
        super().__init__(init_pose, weight)

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
            pos_on_map = map_.landmarks()[obs_id].pos
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


class MapParticle(SimpleParticle):
    def __init__(self, init_pose, weight, map_):
        super().__init__(init_pose, weight)
        self.map = map_

    def observation_update(
            self, observation, distance_dev_rate, direction_dev):
        for d in observation:
            z = d[0]
            landmark = self.map.landmarks()[d[1]]

            if landmark.cov is None:
                self._init_landmark_estimation(
                    landmark, z, distance_dev_rate, direction_dev)
            else:
                self._observation_update_landmark(
                    landmark, z, distance_dev_rate, direction_dev)

    def _init_landmark_estimation(
            self, landmark, z, distance_dev_rate, direction_dev):
        landmark.pos = z[0] * np.array([
            np.cos(self.pose[2] + z[1]),
            np.sin(self.pose[2] + z[1])]).T + self.pose[0:2]

        H = utilities.matH(self.pose, landmark.pos)[0:2, 0:2]
        Q = utilities.matQ(distance_dev_rate * z[0], direction_dev)

        landmark.cov = np.linalg.inv(H.T @ np.linalg.inv(Q) @ H)

    def _observation_update_landmark(
            self, landmark, z, distance_dev_rate, direction_dev):
        # ランドマークの推定位置から予想される計測値
        estimated_z = sensors.IdealCamera.observation_function(
            self.pose, landmark.pos)

        if estimated_z[0] < 0.01:
            return

        H = -utilities.matH(self.pose, landmark.pos)[0:2, 0:2]
        Q = utilities.matQ(distance_dev_rate * estimated_z[0], direction_dev)
        K = landmark.cov @ H.T @ np.linalg.inv(Q + H @ landmark.cov @ H.T)

        # 重みの更新
        Q_z = H @ landmark.cov @ H.T + Q
        self.weight *= multivariate_normal(mean=estimated_z, cov=Q_z).pdf(z)

        landmark.pos = K @ (z - estimated_z) + landmark.pos
        landmark.cov = (np.eye(2) - K @ H) @ landmark.cov
