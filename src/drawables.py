# -*- coding: utf-8 -*-
from abc import abstractmethod
from math import cos, pi, sin
from matplotlib import (
    pyplot as plt,
    animation as anm,
    patches,
    use as enable_plt_option
)
# notebook上でアニメーションできるようになるらしい
enable_plt_option('nbagg')

import numpy as np

from agents import (
    SimpleAgent,
    EstimationAgent,
)
from estimators import (
    Mcl,
    FastSlam,
    ExtendedKalmanFilter,
)
from landmarks import (
    Point2DLandmark,
    Point2DLandmarkEstimated,
)
from maps import (
    Map,
)
from robots import (
    IdealRobot,
    RealRobot,
)
from sensors import (
    IdealCamera,
    Camera,
)

class Simulator(object):
    def __init__(self, time_span, time_interval, debuggable=False):
        u"""主にシミュレート時間の設定

        Args:
            time_span(float): 何秒シミュレートするのか
            time_interval(float): 何秒間隔でシミュレートするか
            debuggable(bool): アニメートさせるかどうか
        """
        self.time_span = time_span
        self.time_interval = time_interval
        # 様々なオブジェクトのいれもの
        self.objects = []
        # notebook側でエラーがわかりにくいので、必要に応じてメッセージを出すためのフラグ
        self.debuggable = debuggable
        self.ani = None

    def append(self, obj):
        u"""描画可能なオブジェクトを追加する"""
        # Drawableの派生クラスを渡しているはずなのにだめなのかわからない
        if not isinstance(obj, Drawable):
            raise TypeError('{} is not drawable object'.format(type(obj)))
        self.objects.append(obj)

    def draw(self):
        u"""持っているオブジェクトを描画する

        8x8インチの描画エリア
        サブプロット準備
        アスペクト比をそろえる
        5x5mのスケール
        ラベルの表示
        """
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle('World')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)

        elems = []

        if self.debuggable:
            for i in range(int(self.time_span/self.time_interval)):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig, self.one_step, fargs=(elems, ax),
                frames=int(self.time_span / self.time_interval),
                interval=int(self.time_interval * 1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        u"""1コマすすめる"""
        while elems:
            elems.pop().remove()

        elems.append(ax.text(
            -4.4, 4.5,
            't= %.2f[s]' % (self.time_interval * i),
            fontsize=10))

        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, 'one_step'):
                obj.one_step(self.time_interval)


class Drawable():
    u"""描画可能オブジェクト"""

    @abstractmethod
    def draw(self, ax, elems, **fkwds):
        u"""渡されたAxesを使って描画処理をし，elemsに追加する

        Args:
            elems([matplotlib.XXX]): 描画加工なオブジェクト(Text, PathCollectionなど)
            ax(AxesSubplot):
            fkwds(dict): なにか追加で入れたいものがあれば

        Returns:
            なし
        """
        raise NotImplementedError('This is an abstractmethod of Drawable.')


class DrawableSimpleAgent(SimpleAgent, Drawable):
    def __init__(self, nu, omega):
        SimpleAgent.__init__(self, nu, omega)

    def draw(self, ax, elems, **fkwds):
        pass

class DrawableEstimationAgent(EstimationAgent, Drawable):
    def __init__(self, time_interval, nu, omega, estimator):
        if not isinstance(estimator, Drawable):
            raise TypeError('Input estimator is not drawable')
        EstimationAgent.__init__(self, time_interval, nu, omega, estimator)

    def draw(self, ax, elems, **fkwds):
        self._estimator.draw(ax, elems, **fkwds)
        x, y, t = self._estimator.pose()
        s = '({:.2f}, {:.2f}, {})'.format(x, y, int(t*180/pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))


class DrawableMcl(Mcl, Drawable):
    def __init__(
            self, map_, init_pose, particles_,
            motion_noise_stds={
                'nn': 0.19, 'no': 1e-5,
                'on': 0.13, 'oo': 0.20},
            distance_dev_rate=0.14, direction_dev=0.05):
        Mcl.__init__(
                self, map_, init_pose, particles_, motion_noise_stds,
                distance_dev_rate, direction_dev)

    def draw(self, ax, elems, **fkwds):
        u"""パーティクルを矢印として表示"""
        xs = [p.pose[0] for p in self._particles]
        ys = [p.pose[1] for p in self._particles]
        # パーティクルの重みに比例した長さの矢印を描くため
        vxs = \
            [
                cos(p.pose[2]) * p.weight * len(self._particles)
                for p in self._particles
            ]
        vys = \
            [
                sin(p.pose[2]) * p.weight * len(self._particles)
                for p in self._particles
            ]
        elems.append(ax.quiver(xs, ys, vxs, vys,
            angles='xy', scale_units='xy', scale=1.5, color='blue', alpha=0.5))


class DrawableFastSlam(FastSlam, Drawable):
    def __init__(
            self, init_pose, particle_num, landmark_num,
            motion_noise_stds={
                'nn': 0.19, 'no': 1e-5,
                'on': 0.13, 'oo': 0.20},
            distance_dev_rate=0.14, direction_dev=0.05):
        FastSlam.__init__(
                self, map_, init_pose, landmark_num, motion_noise_stds,
                distance_dev_rate, direction_dev)

    def draw(self, ax, elems, **fkwds):
        xs = [p.pose[0] for p in self._particles]
        ys = [p.pose[1] for p in self._particles]
        # パーティクルの重みに比例した長さの矢印を描くため
        vxs = \
            [
                cos(p.pose[2]) * p.weight * len(self._particles)
                for p in self._particles
            ]
        vys = \
            [
                sin(p.pose[2]) * p.weight * len(self._particles)
                for p in self._particles
            ]
        elems.append(ax.quiver(xs, ys, vxs, vys,
            angles='xy', scale_units='xy', scale=1.5, color='blue', alpha=0.5))

        self._ml.map.draw(ax, elems)


class DrawableExtendedKalmanFilter(ExtendedKalmanFilter, Drawable):
    def __init__(
            self,
            map_, init_pose,
            motion_noise_stds={
                'nn': 0.19, 'no': 1e-5,
                'on': 0.13, 'oo': 0.20},
            distance_dev_rate=0.14, direction_dev=0.05):
        ExtendedKalmanFilter.__init__(
            map_, init_pose,
            motion_noise_stds, distance_dev_rate, direction_dev)

    def draw(self, ax, elems, **fkwds):
        u"""自己位置を誤差楕円で描画"""
        # xyの3シグマ楕円
        # スライシング[0:2]でxyのみ抽出
        e = utilities.sigma_ellipse(self._belief.mean[0:2], self._belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        # θ方向の誤差3シグマ
        x, y, c = self._belief.mean
        sigma3 = sqrt(self._belief.cov[2, 2]) * 3.
        xs = [x + cos(c - sigma3), x, x + cos(c + sigma3)]
        ys = [y + sin(c - sigma3), y, y + sin(c + sigma3)]
        elems += ax.plot(xs, ys, color='blue', alpha=0.5)


class DrawablePoint2DLandmark(Point2DLandmark, Drawable):
    def __init__(self, x, y):
        Point2DLandmark.__init__(self, x, y)

    def draw(self, ax, elems, **fkwds):
        u"""点ランドマークを散布図として描画する"""
        c = ax.scatter(
            self.pos[0], self.pos[1],
            s=100, marker='*', label='landmark', color='orange')
        elems.append(c)
        elems.append(
            ax.text(
                self.pos[0], self.pos[1],
                'id:' + str(self.id), fontsize=10))


class Point2DLandmarkEstimated(Point2DLandmarkEstimated, Drawable):
    def __init__(self):
        Point2DLandmarkEstimated.__init__()

    def draw(self, ax, elems, **fkwds):
        if self._cov is None:
            return

        c = ax.scatter(
            self.pos[0], self.pos[1],
            s=100, marker='*', label='landmarks', color='blue')
        elems.append(c)
        elems.append(
            ax.text(
                self.pos[0], self.pos[1],
                'id:' + str(self.id), fontsize=10))

        e = utilities.sigma_ellipse(self.pos, self._cov, 3)
        elems.append(ax.add_patch(e))


class DrawableMap(Map, Drawable):
    def __init__(self):
        Map.__init__(self);

    def append_landmark(self, landmark):
        if not isinstance(landmark, Drawable):
            raise TypeError('Drawable landmark is required')
        super().append_landmark(landmark)

    def draw(self, ax, elems, **fkwds):
        for lm in self._landmarks:
            lm.draw(ax, elems)


class DrawableIdealRobot(IdealRobot, Drawable):
    def __init__(self, pose, agent=None, sensor=None, color='black'):
        if agent is not None and not isinstance(agent, Drawable):
            raise TypeError('agent is not a child of Drawable')
        if sensor is not None and not isinstance(sensor, Drawable):
            raise TypeError('sensor is not a child of Drawable')

        IdealRobot.__init__(self, pose, agent, sensor, color)

    def draw(self, ax, elems, **fkwds):
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


class DrawableRealRobot(RealRobot, Drawable):
    def __init__(self, pose, agent=None, sensor=None, color='black',
            noise_per_meter=5., noise_std=pi / 60.,
            bias_rate_stds=(0.1, 0.1),
            expected_stuck_time=1e100, expected_escape_time=1e-100,
            expected_kidnap_time=1e100, kidnap_range_x=(-5., -5.), kidnap_range_y=(-5., 5.)):
        if agent is not None and not isinstance(agent, Drawable):
            raise TypeError('agent is not a child of Drawable')
        if sensor is not None and not isinstance(sensor, Drawable):
            raise TypeError('sensor is not a child of Drawable')
        super().__init__(pose, agent, sensor,
            color, noise_per_meter, noise_std,
            bias_rate_stds,
            expected_stuck_time, expected_escape_time,
            expected_kidnap_time, kidnap_range_x, kidnap_range_y)

    def draw(self, ax, elems, **fkwds):
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
            self.sensor.draw(ax, elems, cam_pose=self.poses[-2])
        if self.agent is not None and hasattr(self.agent, 'draw'):
            self.agent.draw(ax, elems)


class DrawableIdealCamera(IdealCamera, Drawable):
    def __init__(self,
            env_map,
            distance_range=(0.5, 6.0),
            direction_range=(-np.pi / 3., np.pi / 3.)):
        if not isinstance(env_map, Drawable):
            raise TypeError('agent is not a child of Drawable')
        IdealCamera.__init__(self, env_map, distance_range, direction_range)

    def draw(self, ax, elems, **fkwds):
        x, y, theta = fkwds['cam_pose']

        for lm in self.lastdata:
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * cos(direction + theta)
            ly = y + distance * sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')


class DrawableCamera(Camera, Drawable):
    def __init__(self,
            env_map,
            distance_range=(0.5, 6.0), direction_range=(-pi / 3., pi / 3),
            distance_noise_rate=0.1, direction_noise=pi/90.,
            distance_bias_rate_stddev=0.1, direction_bias_stddev=pi/90.,
            phantom_prob=0.0, phantom_range_x=(-5., 5.), phantom_range_y=(-5., 5.),
            oversight_prob=0.1,
            occulusion_prob=0.0):
        if not isinstance(env_map, Drawable):
            raise TypeError('agent is not a child of Drawable')
        Camera.__init__(
            self, env_map,
            distance_range, direction_range,
            distance_noise_rate, direction_noise,
            distance_bias_rate_stddev, direction_bias_stddev,
            phantom_prob, phantom_range_x, phantom_range_y,
            oversight_prob,
            occulusion_prob)

    def draw(self, ax, elems, **fkwds):
        x, y, theta = fkwds['cam_pose']

        for lm in self.lastdata:
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * cos(direction + theta)
            ly = y + distance * sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')
