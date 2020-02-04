# -*- coding: utf-8 -*-
import math
u"""エージェントクラス

考える主体のこと

ロボティクスや人工知能の分野ではそう呼ぶ
"""

class Agent():
    u"""考える主体

    観測を受け取り、制御指令を変えす
    """

    def __init__(self, nu, omega):
        u"""速度・角速度(制御指令)の設定

        Args:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
        """
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        u"""意思決定

        観測があったらそれを考慮するように実装される
        今は単に，最初に与えられた制御指令をそのまま出力する

        Args:
            observation((np.array, int)): 観測した位置とID

        Returns:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
        """
        return self.nu, self.omega


class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = '({:.2f}, {:.2f}, {})'.format(x, y, int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))

    def decision(self, observation=None):
        self.estimator.motion_update(
            self.prev_nu, self.prev_omega, self.time_interval)

        self.prev_nu, self.prev_omega = self.nu, self.omega

        self.estimator.observation_update(observation)

        return self.nu, self.omega
