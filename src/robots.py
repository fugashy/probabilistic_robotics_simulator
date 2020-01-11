# -*- coding: utf-8 -*-
u"""さまざまな2D移動ロボットを表現するクラス

ノイズ無し・ありなど
"""
from math import sin, cos, fabs
import matplotlib.patches as patches
import numpy as np

class IdealRobot():
    u"""理想的な2D移動をするロボット"""

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        u"""状態遷移関数

        角速度omegaが0のときとそうでない場合で変わる

        Args:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
            time(float): 時間[s]
            pose(np.array): 遷移元の状態

        Returns:
            更新された状態(np.array)
        """
        theta_0 = pose[2]

        translate_state_omega_is_almost_zero = lambda nu, theta_0, omega, time: \
            np.array([
                nu * cos(theta_0),
                nu * sin(theta_0),
                omega]) * time

        translate_state_omega_is_not_almost_zero = lambda nu, theta_0, omega, time: \
            np.array([
                nu / omega * (sin(theta_0 + omega * time) - sin(theta_0)),
                nu / omega * (-cos(theta_0 + omega * time) + cos(theta_0)),
                omega * time])

        if fabs(omega) < 1e-10:
            return pose + translate_state_omega_is_almost_zero(
                nu, theta_0, omega, time)
        else:
            return pose + translate_state_omega_is_not_almost_zero(
                nu, theta_0, omega, time)

    def __init__(self, pose, agent=None, sensor=None, color='black'):
        u"""初期位置・色の設定

        Args:
            pose(np.array): [x(m), y(m), yaw(rad)]
            agent(Agent): エージェント
            sensor(IdealCamera): 観測者
            color(string): red, green, blue, black, etc...
        """
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.sensor = sensor
        # 軌跡を描画するため
        self.poses = [pose]

    def draw(self, ax, elems):
        u"""描画

        Args:
            ax(matplotlib.axes,_subplots.AxesSubplot): サブプロットオブジェクト
            elems([matplotlib.XXX]): 描画可能なオブジェクト(Text, PathCollectionなど)

        Returns:
            なし
        """
        x, y, theta = self.pose

        # ロボットの向きを示す線分
        xn = x + self.r * cos(theta)
        yn = y + self.r * sin(theta)
        # sx.plotがlistを返すので+=とする not append
        elems += ax.plot([x, xn], [y, yn], color=self.color)

        c = patches.Circle(
            xy=(x, y), radius=self.r, fill=False, color=self.color)

        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot(
            [e[0] for e in self.poses],
            [e[1] for e in self.poses],
            linewidth=0.5, color=self.color)

        if self.sensor is not None and len(self.poses) > 1:
            # センサ値を得たときの時刻が-2要素にある...らしい
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent is not None and hasattr(self.agent, 'draw'):
            self.agent.draw(ax, elems)

    def one_step(self, time_interval):
        u"""1コマすすめる

        Agentが存在する場合は、制御指令を得て位置を更新する
        """
        if self.agent is None:
            return

        obs = None
        if self.sensor is not None:
            obs = self.sensor.data(self.pose)

        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
