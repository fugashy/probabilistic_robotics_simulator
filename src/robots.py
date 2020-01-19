# -*- coding: utf-8 -*-
u"""さまざまな2D移動ロボットを表現するクラス

ノイズ無し・ありなど
"""
from math import sin, cos, fabs, pi
import matplotlib.patches as patches
import numpy as np
from scipy.stats import expon, norm, uniform

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


class Robot(IdealRobot):
    u"""いろんなノイズも含まれた，リアルに近めなロボット

    移動に対する雑音などを含む
    """

    def __init__(self, pose, agent=None, sensor=None,
            color='black',
            noise_per_meter=5., noise_std=pi / 60.,
            bias_rate_stds=(0.1, 0.1),
            expected_stuck_time=1e100, expected_escape_time=1e-100,
            kidnap_range_x=(-5., -5.), kidnap_range_y=(-5., 5.)):
        u"""初期位置・色・ノイズなどの設定

        Args:
            pose(np.array): [x(m), y(m), yaw(rad)]
            agent(Agent): エージェント
            sensor(IdealCamera): 観測者
            color(string): red, green, blue, black, etc...
            noise_per_meter(float): 1mあたりになにか踏んでしまう回数
            noise_std(float): 踏んだときに姿勢に与えるノイズの標準偏差
            bias_rate_stds((float, float)): 速度指令・角速度指令に与えるバイアスの標準偏差
            expected_stuck_time(float): スタックが起きるまでの時間の期待値
            expected_escape_time(float): スタックから抜け出すまでの時間の期待値
            expected_kidnap_time(float): 誘拐が発生するまでの時間の期待値
            kidnap_range_x((float, float)): 誘拐後のロボット位置x範囲
            kidnap_range_y((float, float)): 誘拐後のロボット位置y範囲
        """
        super().__init__(pose, agent, sensor, color)

        # 移動に対して発生するノイズ
        # 移動すればするほどノイズが大きくなっていく，指数分布に従う確率密度関数
        # exponは指数分布の機能を提供するメソッド
        # 1e-100で0割対策
        self.noise_pdf = expon(scale=1./(1e-100 + noise_per_meter))
        # rvsはドローするための関数(Random variates 確率変数)
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)

        # 指令加えるバイアス
        # N(0, std)に従う分布からドローしておく
        # locは期待値
        # 1.0を設定することで，平均値を引いたときは1.0となる
        # こういう設定にしておくことでrateになる
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        # スタック(なにかに引っかかって動かなくなる現象)をシミュレートするための変数
        # 発生するまでの時間，抜け出すまでの時間はともに指数分布の確率変数
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self._stucking = False

        # キッドナップをシミュレートするための変数
        # 発生するまでの時間は指数分布の確率変数
        # 融解後の位置姿勢は指定されたレンジの一様分布の確率変数
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(
            loc=(rx[0], ry[0], 0.), scale=(rx[1]-rx[0], ry[1]-ry[0], 2.*pi))

    def one_step(self, time_interval):
        u"""一コマすすめる

        Args:
            time_interval(float): シミュレート時間間隔[s]
        """
        if self.agent is None:
            return

        obs = None
        if self.sensor is not None:
            obs = self.sensor.data(self.pose)

        nu, omega = self.agent.decision(obs)
        nu, omega = self._bias(nu, omega)
        nu, omega = self._stuck(nu, omega, time_interval)

        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self._noise(self.pose, nu, omega, time_interval)
        self.pose = self._kidnap(pose, time_interval)

    def _noise(self, pose, nu, omega, time_interval):
        u"""ノイズを付加すべき距離を移動したら，確率密度関数に従ったノイズを与える

        Args:
            pose(np.array): x, y, yaw
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
            time_interval(float): シミュレート時間間隔[s]
        """
        # ドローしておいた指数分布から得られるノイズ源接触までの距離を
        self.distance_until_noise -= \
            abs(nu) * time_interval + self.r * abs(omega) * time_interval
        if self.distance_until_noise <= 0.:
            # 再度ドロー
            self.distance_until_noise += self.noise_pdf.rvs()
            # ノイズを姿勢に与える
            pose[2] += self.theta_noise.rvs()

        return pose

    def _bias(self, nu, omega):
        u"""制御指令にバイアスを与える

        事前に計算してある比率をかけ合わせることで一定のバイアスをかける

        Args:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]

        Returns:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
        """
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega

    def _stuck(self, nu, omega, time_interval):
        u"""スタック状態の切り替え

        Args:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
            time_interval(float): シミュレート時間間隔[s]

        Returns:
            nu(float): 速度[m/s]
            omega(float): 角速度[rad/s]
        """
        if self._stucking:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.:
                self.time_until_escape += self.escape_pdf.rvs()
                self._stucking = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self._stucking = True

        # pythonでは数値演算にbooleanを使うと1 or 0になる
        return nu * (not self._stucking), omega * (not self._stucking)

    def _kidnap(self, pose, time_interval):
        u"""キッドナップを発生させる

        Args:
            pose(np.array): x, y, yaw

        Returns:
            誘拐後の位置姿勢
            何もなければそのまま
            pose(np.array): x, y, yaw
        """
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
