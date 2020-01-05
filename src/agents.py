# -*- coding: utf-8 -*-
u"""エージェントクラス

考える主体のこと

ロボティクスや人工知能の分野ではそう呼ぶ
"""

class Agent():
    u"""考える主体

    観測を受け取り、制御指令を変えす
    """

    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega
